"""Launch decision-bench (quick/full/edge) on LLaDA + a LoRA adapter, on SageMaker GPU.

Packages dbench_entry.py + jul-bis lib/jul + the decision_bench package + a suite JSONL, mounts the
adapter, and writes predictions.jsonl + report.json (decision-bench format).

Usage:
    python deployment/sagemaker-eval/launch_dbench.py \\
        --model llada-8b-instruct --readout auto --suite quick --adapter s3://.../model.tar.gz --spot

Environment variables (required):
    JUL_AWS_ACCOUNT      - AWS account ID (e.g. 123456789012)
    JUL_SAGEMAKER_ROLE   - SageMaker execution role ARN
    JUL_SAGEMAKER_BUCKET - S3 bucket for SageMaker artifacts
    JUL_DBENCH           - path to the decision-bench repo
"""

from __future__ import annotations

import argparse
import io
import os
import tarfile
import time

import boto3

REGION = os.environ.get("AWS_DEFAULT_REGION", "eu-west-1")
ACCOUNT = os.environ.get("JUL_AWS_ACCOUNT", "")
ROLE = os.environ.get("JUL_SAGEMAKER_ROLE", "")
BUCKET = os.environ.get("JUL_SAGEMAKER_BUCKET", "")
DLC = ("763104351884.dkr.ecr.eu-west-1.amazonaws.com/"
       "pytorch-training:2.8.0-gpu-py312-cu129-ubuntu22.04-sagemaker")
DEFAULT_MOE_PIN = "4.53.3"

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
DBENCH = os.environ.get("JUL_DBENCH", os.path.expanduser("~/dev/decision-bench"))
DBENCH_PKG = os.path.join(DBENCH, "decision_bench")
SUITES = {"quick": "data/quick-v1.jsonl", "full": "data/bench-v1.jsonl", "edge": "data/edge-v1.jsonl"}


def make_source_tar(pin: str = "") -> bytes:
    reqs = (f"transformers=={pin}\n".encode() if pin else b"") + b"accelerate\npeft\nsafetensors\npyyaml\n"
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(os.path.join(HERE, "dbench_entry.py"), arcname="dbench_entry.py")
        tar.add(os.path.join(REPO, "scripts", "llada_head.py"), arcname="llada_head.py")
        tar.add(os.path.join(REPO, "lib", "jul"), arcname="lib/jul")
        tar.add(DBENCH_PKG, arcname="dbench_src/decision_bench")
        info = tarfile.TarInfo("requirements.txt"); info.size = len(reqs)
        tar.addfile(info, io.BytesIO(reqs))
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="llada-8b-instruct")
    ap.add_argument("--suite", choices=["quick", "full", "edge"], default="quick")
    ap.add_argument("--readout", default="auto", choices=["anchor", "multitoken", "auto", "auto-ce"])
    ap.add_argument("--adapter", default="", help="S3 URI of a LoRA adapter model.tar.gz")
    ap.add_argument("--mask-id", default="126336")
    ap.add_argument("--instance", default="ml.g5.2xlarge")
    ap.add_argument("--region", default=REGION, help="override region (e.g. us-east-1 for capacity)")
    ap.add_argument("--image", default=None, help="DLC image URI (else derived from region)")
    ap.add_argument("--max-run", type=int, default=3600 * 5)
    ap.add_argument("--spot", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    region = args.region
    bucket = BUCKET or os.environ.get("JUL_SAGEMAKER_BUCKET", "")
    # DLC tag suffix differs by region mirror (us-east-1 carries a -v1 suffix on this tag).
    dlc_tag = ("2.8.0-gpu-py312-cu129-ubuntu22.04-sagemaker-v1" if region == "us-east-1"
               else "2.8.0-gpu-py312-cu129-ubuntu22.04-sagemaker")
    image = args.image or f"763104351884.dkr.ecr.{region}.amazonaws.com/pytorch-training:{dlc_tag}"

    sess = boto3.Session(region_name=region)
    s3 = sess.client("s3"); sm = sess.client("sagemaker")
    BUCKET_L = bucket
    job = f"jul-llada-dbench-{args.suite}-{int(time.time())}"
    prefix = f"jul-dbench/{job}"
    pin = "4.57.1" if "llada2" in args.model.lower() else (DEFAULT_MOE_PIN if "moe" in args.model.lower() else "")

    print(f"PLAN: job={job} model={args.model} suite={args.suite} readout={args.readout} pin={pin or 'none'}")
    if args.dry_run:
        return 0

    s3.upload_file(os.path.join(DBENCH, SUITES[args.suite]), BUCKET_L, f"{prefix}/input/bench/suite.jsonl")
    s3.put_object(Bucket=BUCKET_L, Key=f"{prefix}/code/source.tar.gz", Body=make_source_tar(pin))

    channels = [{"ChannelName": "bench",
                 "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix",
                     "S3Uri": f"s3://{BUCKET_L}/{prefix}/input/bench/",
                     "S3DataDistributionType": "FullyReplicated"}}}]
    if args.adapter:
        channels.append({"ChannelName": "adapter",
                         "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix",
                             "S3Uri": args.adapter.rsplit("/", 1)[0] + "/",
                             "S3DataDistributionType": "FullyReplicated"}}})

    hyper = {"sagemaker_program": "dbench_entry.py",
             "sagemaker_submit_directory": f"s3://{BUCKET_L}/{prefix}/code/source.tar.gz"}
    stopping = {"MaxRuntimeInSeconds": args.max_run}
    extra = {}
    if args.spot:
        extra["EnableManagedSpotTraining"] = True
        stopping["MaxWaitTimeInSeconds"] = args.max_run + 3600

    sm.create_training_job(
        TrainingJobName=job,
        AlgorithmSpecification={"TrainingImage": image, "TrainingInputMode": "File"},
        RoleArn=ROLE, InputDataConfig=channels,
        OutputDataConfig={"S3OutputPath": f"s3://{BUCKET_L}/{prefix}/output/"},
        ResourceConfig={"InstanceType": args.instance, "InstanceCount": 1, "VolumeSizeInGB": 100},
        StoppingCondition=stopping, HyperParameters=hyper,
        Environment={"JUL_MODEL": args.model, "JUL_DTYPE": "bfloat16",
                     "JUL_LLADA_MASK_ID": str(args.mask_id), "JUL_LLADA_READOUT": args.readout,
                     "JUL_LLADA_N_MASK": "1", "JUL_LLADA_N_STEPS": "1",
                     **({"JUL_LLADA_DEVICE_MAP": "auto"} if "llada2" in args.model.lower() else {})},
        **extra,
    )
    print(f"\nlaunched: {job}" + (" (spot)" if args.spot else ""))
    print(f"result: s3://{BUCKET_L}/{prefix}/output/{job}/output/model.tar.gz (predictions.jsonl + report.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
