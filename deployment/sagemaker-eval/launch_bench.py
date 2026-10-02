"""Launch the public Jev benchmark (BTZSC pilot v1) on LLaDA + a LoRA adapter, on SageMaker GPU.

Packages bench_entry.py + jul-bis lib/jul + the jev_benchmarks package + the BTZSC manifest, mounts
the adapter, and creates a GPU training job that writes jev-bench.json.

Usage:
    python deployment/sagemaker-eval/launch_bench.py \\
        --model llada-8b-instruct \\
        --adapter s3://.../jul-train/<ce-job>/output/<ce-job>/output/model.tar.gz --spot

Environment variables (required):
    JUL_AWS_ACCOUNT      - AWS account ID (e.g. 123456789012)
    JUL_SAGEMAKER_ROLE   - SageMaker execution role ARN
    JUL_SAGEMAKER_BUCKET - S3 bucket for SageMaker artifacts
    JUL_WORK             - path to the jul working project (for jev_benchmarks)
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
# the jev_benchmarks package and BTZSC manifest live in the working project
JUL_WORK = os.environ.get("JUL_WORK", os.path.expanduser("~/dev/jul"))
JEV_SRC = os.path.join(JUL_WORK, "external", "jev-benchmarks", "src", "jev_benchmarks")
MANIFEST = os.path.join(JUL_WORK, "external", "jev-benchmarks", "results", "runs",
                        "btzsc-pilot-v1", "manifest.jsonl")


def make_source_tar(pin: str = "") -> bytes:
    reqs = (f"transformers=={pin}\n".encode() if pin else b"") + b"accelerate\npeft\nsafetensors\nnumpy\n"
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(os.path.join(HERE, "bench_entry.py"), arcname="bench_entry.py")
        tar.add(os.path.join(REPO, "lib", "jul"), arcname="lib/jul")
        tar.add(JEV_SRC, arcname="jev_src/jev_benchmarks")
        info = tarfile.TarInfo("requirements.txt"); info.size = len(reqs)
        tar.addfile(info, io.BytesIO(reqs))
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="llada-8b-instruct")
    ap.add_argument("--adapter", default="", help="S3 URI of a LoRA adapter model.tar.gz")
    ap.add_argument("--mask-id", default="126336")
    ap.add_argument("--readout", default="anchor", choices=["anchor", "multitoken", "auto"],
                    help="anchor: one token per option (saturates on many classes); "
                         "multitoken: length-normalized sequence-likelihood of each label text; "
                         "auto: route by option count (anchor few / multitoken many)")
    ap.add_argument("--instance", default="ml.g5.2xlarge")
    ap.add_argument("--image", default=DLC)
    ap.add_argument("--max-run", type=int, default=3600 * 3)
    ap.add_argument("--spot", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    sess = boto3.Session(region_name=REGION)
    s3 = sess.client("s3"); sm = sess.client("sagemaker")
    job = f"jul-llada-bench-{int(time.time())}"
    prefix = f"jul-bench/{job}"
    pin = DEFAULT_MOE_PIN if "moe" in args.model.lower() else ""

    print(f"PLAN: job={job} model={args.model} adapter={'yes' if args.adapter else 'no'} pin={pin or 'none'}")
    if args.dry_run:
        return 0

    s3.upload_file(MANIFEST, BUCKET, f"{prefix}/input/bench/manifest.jsonl")
    s3.put_object(Bucket=BUCKET, Key=f"{prefix}/code/source.tar.gz", Body=make_source_tar(pin))

    channels = [{"ChannelName": "bench",
                 "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix",
                     "S3Uri": f"s3://{BUCKET}/{prefix}/input/bench/",
                     "S3DataDistributionType": "FullyReplicated"}}}]
    if args.adapter:
        channels.append({"ChannelName": "adapter",
                         "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix",
                             "S3Uri": args.adapter.rsplit("/", 1)[0] + "/",
                             "S3DataDistributionType": "FullyReplicated"}}})

    hyper = {"sagemaker_program": "bench_entry.py",
             "sagemaker_submit_directory": f"s3://{BUCKET}/{prefix}/code/source.tar.gz"}
    stopping = {"MaxRuntimeInSeconds": args.max_run}
    extra = {}
    if args.spot:
        extra["EnableManagedSpotTraining"] = True
        stopping["MaxWaitTimeInSeconds"] = args.max_run + 3600

    sm.create_training_job(
        TrainingJobName=job,
        AlgorithmSpecification={"TrainingImage": args.image, "TrainingInputMode": "File"},
        RoleArn=ROLE, InputDataConfig=channels,
        OutputDataConfig={"S3OutputPath": f"s3://{BUCKET}/{prefix}/output/"},
        ResourceConfig={"InstanceType": args.instance, "InstanceCount": 1, "VolumeSizeInGB": 100},
        StoppingCondition=stopping, HyperParameters=hyper,
        Environment={"JUL_MODEL": args.model, "JUL_DTYPE": "bfloat16",
                     "JUL_LLADA_MASK_ID": str(args.mask_id),
                     "JUL_LLADA_READOUT": args.readout,
                     "JUL_LLADA_N_MASK": "1", "JUL_LLADA_N_STEPS": "1"},  # no tricks: best regime for a trained model
        **extra,
    )
    print(f"\nlaunched: {job}" + (" (spot)" if args.spot else ""))
    print(f"result: s3://{BUCKET}/{prefix}/output/{job}/output/model.tar.gz (jev-bench.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
