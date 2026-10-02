"""Launch the LLaDA zero-shot [MASK] eval on SageMaker (GPU), via boto3.

Packages the whole jul-bis library (lib/jul, with the 'llada' backend and mask reader), the eval
script and the entry point into a source tar, uploads the sealed holdout as an input channel, and
creates a SageMaker training job on a PyTorch GPU DLC. The job writes llada-eval.json to
$SM_MODEL_DIR, which SageMaker tars to S3 (output path printed at the end).

This is a *feasibility gate*, not training: on a GPU the 439-record holdout runs in minutes, where a
Mac's MPS took ~150 s per forward.

Usage:
    python deployment/sagemaker-eval/launch_eval.py \\
        --holdout data/eval-holdout-v2/holdout.jsonl \\
        --model llada-8b-instruct --instance ml.g5.2xlarge --spot
    # quick signal first:
    python deployment/sagemaker-eval/launch_eval.py --limit 24 --spot

Environment variables (required):
    JUL_AWS_ACCOUNT      - AWS account ID (e.g. 123456789012)
    JUL_SAGEMAKER_ROLE   - SageMaker execution role ARN
    JUL_SAGEMAKER_BUCKET - S3 bucket for SageMaker artifacts
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

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))

# LLaDA's remote code predates transformers 5.x; the backend ships an in-process compat shim, but for
# a job that must just load and run, we pin the transformers the authors targeted (dense LLaDA-8B:
# 4.46.3, LLaDA-MoE: 4.53.2 -> 4.53.x satisfies both) so the remote RoPE init runs natively instead of
# through the shim. transformers has no hard upper torch pin, so 4.53.x is fine on the DLC's torch 2.8
# / py3.12. RISK: if a future DLC bumps a transitive dep past 4.53's range, relax this pin (the shim
# still repairs an unpinned load). peft/accelerate are needed for adapter loading (Option 2).
DEFAULT_MOE_PIN = "4.53.3"  # only the MoE remote code needs the pin; dense 8B uses the DLC's 5.x + shim


def make_source_tar(pin: str = "") -> bytes:
    """entry point + eval script + the whole jul package (lib/jul) + requirements."""
    reqs = (f"transformers=={pin}\n".encode() if pin else b"") + b"accelerate\npeft\nsafetensors\n"
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(os.path.join(HERE, "eval_entry.py"), arcname="eval_entry.py")
        tar.add(os.path.join(REPO, "scripts", "eval_llada_zeroshot.py"), arcname="eval_llada_zeroshot.py")
        # ship the library so the 'llada' backend, mask reader and presets are importable
        tar.add(os.path.join(REPO, "lib", "jul"), arcname="lib/jul")
        info = tarfile.TarInfo("requirements.txt"); info.size = len(reqs)
        tar.addfile(info, io.BytesIO(reqs))
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--holdout", default="data/eval-holdout-v2/holdout.jsonl")
    ap.add_argument("--model", default="llada-8b-instruct", help="llada-8b-instruct | illada-8b-instruct")
    ap.add_argument("--instance", default="ml.g5.2xlarge", help="GPU instance (g5.2xlarge = 1x A10G 24GB)")
    ap.add_argument("--image", default=DLC)
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N records")
    ap.add_argument("--mask-id", default="126336", help="JUL_LLADA_MASK_ID (LLaDA 126336, iLLaDA 5)")
    ap.add_argument("--n-mask", type=int, default=1, help="K consecutive [MASK]s, averaged (variance reduction)")
    ap.add_argument("--n-steps", type=int, default=1, help="iterative demasking passes (diffusion refine)")
    ap.add_argument("--adapter", default="", help="S3 URI of a LoRA adapter model.tar.gz (Option 2 gate)")
    ap.add_argument("--max-run", type=int, default=3600 * 4)
    ap.add_argument("--spot", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    sess = boto3.Session(region_name=REGION)
    s3 = sess.client("s3")
    sm = sess.client("sagemaker")
    job = f"jul-llada-eval-{int(time.time())}"
    prefix = f"jul-eval/{job}"

    plan = {"job": job, "instance": args.instance, "image": args.image, "model": args.model,
            "limit": args.limit, "role": ROLE, "bucket": BUCKET,
            "holdout_s3": f"s3://{BUCKET}/{prefix}/input/holdout/",
            "output_s3": f"s3://{BUCKET}/{prefix}/output/"}
    print("PLAN:")
    for k, v in plan.items():
        print(f"  {k}: {v}")
    if args.dry_run:
        print("\n--dry-run: nothing launched.")
        return 0

    s3.upload_file(args.holdout, BUCKET, f"{prefix}/input/holdout/holdout.jsonl")
    pin = DEFAULT_MOE_PIN if "moe" in args.model.lower() else ""
    s3.put_object(Bucket=BUCKET, Key=f"{prefix}/code/source.tar.gz", Body=make_source_tar(pin))

    channels = [{"ChannelName": "holdout",
                 "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix",
                     "S3Uri": f"s3://{BUCKET}/{prefix}/input/holdout/",
                     "S3DataDistributionType": "FullyReplicated"}}}]
    if args.adapter:
        # the adapter model.tar.gz lives at an S3 URI; SageMaker downloads the prefix to the channel dir
        adapter_prefix = args.adapter.rsplit("/", 1)[0] + "/"
        channels.append({"ChannelName": "adapter",
                         "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix",
                             "S3Uri": adapter_prefix, "S3DataDistributionType": "FullyReplicated"}}})
    hyper = {"sagemaker_program": "eval_entry.py",
             "sagemaker_submit_directory": f"s3://{BUCKET}/{prefix}/code/source.tar.gz",
             "model": args.model}
    if args.limit:
        hyper["limit"] = str(args.limit)

    stopping = {"MaxRuntimeInSeconds": args.max_run}
    extra = {}
    if args.spot:
        extra["EnableManagedSpotTraining"] = True
        stopping["MaxWaitTimeInSeconds"] = args.max_run + 3600

    sm.create_training_job(
        TrainingJobName=job,
        AlgorithmSpecification={"TrainingImage": args.image, "TrainingInputMode": "File"},
        RoleArn=ROLE,
        InputDataConfig=channels,
        OutputDataConfig={"S3OutputPath": f"s3://{BUCKET}/{prefix}/output/"},
        ResourceConfig={"InstanceType": args.instance, "InstanceCount": 1, "VolumeSizeInGB": 100},
        StoppingCondition=stopping,
        HyperParameters=hyper,
        Environment={"JUL_MODEL": args.model, "JUL_DTYPE": "bfloat16",
                     "JUL_LLADA_MASK_ID": str(args.mask_id),
                     "JUL_LLADA_N_MASK": str(args.n_mask), "JUL_LLADA_N_STEPS": str(args.n_steps)},
        **extra,
    )
    print(f"\nlaunched: {job}" + (" (spot)" if args.spot else ""))
    print(f"poll: aws sagemaker describe-training-job --training-job-name {job} "
          f"--query TrainingJobStatus --region {REGION}")
    print(f"result: s3://{BUCKET}/{prefix}/output/model.tar.gz  (contains llada-eval.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
