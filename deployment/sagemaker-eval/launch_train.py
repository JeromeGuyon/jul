"""Launch LLaDA / LLaDA-MoE [MASK] fine-tuning on SageMaker (Option 2), via boto3.

Packages jul-bis's lib/jul (backend + mask + shim), llada_train.py and train_entry.py, uploads the
soft-labelled MixExample corpus, and creates a GPU training job. Writes a LoRA adapter to S3.

Usage:
    python deployment/sagemaker-eval/launch_train.py \\
        --train data/mix/train.soft.jsonl \\
        --base inclusionAI/LLaDA-MoE-7B-A1B-Instruct --stage a --instance ml.g5.2xlarge --spot

The defaults follow the GIFT paper (arXiv 2509.20863) LoRA SFT recipe: loss=gift, LoRA r=128,
alpha=256, lr=1e-5, epochs=3 (the paper trains up to 20). Use --loss ce for the baseline or
--loss diffusion for the plain LLaDA 1/t objective.

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
# LLaDA / LLaDA-MoE remote code was written for the transformers 4.4x-4.5x rope API. Under the DLC's
# transformers 5.x it breaks in the RoPE init (ROPE_INIT_FUNCTIONS lost the "default" key -> KeyError
# 'default', then 'factor'). The backend ships a shim (_patch_remote_code_for_transformers5) that
# repairs this in-process, but the *robust* fix for a training run is to install the transformers the
# authors targeted: the MoE config declares 4.53.2, the dense LLaDA-8B declares 4.46.3, so 4.53.x
# satisfies both remote codes without any shimming. transformers has no hard upper torch pin, so
# 4.53.x is compatible with the DLC's torch 2.8 / py3.12. RISK: 4.53 is older than the DLC's stack;
# if a future DLC bumps a transitive dep (tokenizers, accelerate) past 4.53's range, relax this pin or
# fall back to the in-process shim (which is also kept, so an unpinned run still loads).
# The pin is CONDITIONAL: the dense LLaDA-8B loads fine on the DLC's native transformers 5.x via the
# shim (proven), so it uses no pin; only the MoE remote code needs 4.53.3.
DEFAULT_MOE_PIN = "4.53.3"

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))


def make_source_tar(pin: str = "") -> bytes:
    reqs = (f"transformers=={pin}\n".encode() if pin else b"") + b"accelerate\npeft\nsafetensors\n"
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(os.path.join(HERE, "train_entry.py"), arcname="train_entry.py")
        tar.add(os.path.join(REPO, "scripts", "llada_train.py"), arcname="llada_train.py")
        tar.add(os.path.join(REPO, "scripts", "moe_lora.py"), arcname="moe_lora.py")
        tar.add(os.path.join(REPO, "lib", "jul"), arcname="lib/jul")
        info = tarfile.TarInfo("requirements.txt"); info.size = len(reqs)
        tar.addfile(info, io.BytesIO(reqs))
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", default="data/mix/train.soft.jsonl")
    ap.add_argument("--base", default="inclusionAI/LLaDA-MoE-7B-A1B-Instruct")
    ap.add_argument("--stage", default="a", choices=["a", "b"])
    # GIFT (arXiv 2509.20863) recommended LoRA SFT config: loss=gift, r=128, alpha=256, lr=1e-5, epochs>1.
    ap.add_argument("--loss", default="gift", choices=["ce", "diffusion", "gift", "ordinal"],
                    help="Stage-A loss: ce (baseline), diffusion (LLaDA 1/t), gift (entropy, default)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--lora-r", type=int, default=128)
    ap.add_argument("--lora-alpha", type=int, default=None, help="default 2*r (safe); never a fixed 256 with small r")
    ap.add_argument("--moe-lora-mode", default="routing", choices=["routing", "all", "attention"],
                    help="routing-guided MoE LoRA (default): hot experts + gate + shared + attention")
    ap.add_argument("--moe-hot-frac", type=float, default=0.25,
                    help="fraction of hottest experts per layer to adapt (routing mode)")
    ap.add_argument("--mask-id", default="", help="JUL_LLADA_MASK_ID override (else config/tokenizer)")
    ap.add_argument("--head", type=int, default=0, help="1 = learned read head on [MASK] hidden state")
    ap.add_argument("--head-proj", type=int, default=256, help="projection dim of the learned head")
    ap.add_argument("--instance", default="ml.g5.2xlarge")
    ap.add_argument("--image", default=DLC)
    ap.add_argument("--max-run", type=int, default=3600 * 6)
    ap.add_argument("--transformers-pin", default=None,
                    help="pin transformers in the DLC. Default: 4.53.3 for a MoE base (its remote "
                         "code needs it), 4.57.1 for iLLaDA (its saved version), none for dense "
                         "LLaDA-8B (loads on the DLC's 5.x via the shim).")
    ap.add_argument("--spot", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.lora_alpha is None:
        args.lora_alpha = 2 * args.lora_r
    # auto-select the transformers pin from the base unless the user forces one
    if args.transformers_pin is None:
        base_l = args.base.lower()
        args.transformers_pin = ("4.57.1" if "illada" in base_l
                                 else DEFAULT_MOE_PIN if "moe" in base_l else "")

    sess = boto3.Session(region_name=REGION)
    s3 = sess.client("s3"); sm = sess.client("sagemaker")
    job = f"jul-llada-train-{args.stage}-{int(time.time())}"
    prefix = f"jul-train/{job}"

    print(f"PLAN: job={job} base={args.base} stage={args.stage} loss={args.loss} "
          f"lora_r={args.lora_r} alpha={args.lora_alpha} lr={args.lr} epochs={args.epochs} "
          f"moe_lora={args.moe_lora_mode} hot_frac={args.moe_hot_frac} instance={args.instance}")
    if args.dry_run:
        print("--dry-run"); return 0

    s3.upload_file(args.train, BUCKET, f"{prefix}/input/train/train.jsonl")
    s3.put_object(Bucket=BUCKET, Key=f"{prefix}/code/source.tar.gz", Body=make_source_tar(args.transformers_pin))

    channels = [{"ChannelName": "train",
                 "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix",
                     "S3Uri": f"s3://{BUCKET}/{prefix}/input/train/",
                     "S3DataDistributionType": "FullyReplicated"}}}]
    hyper = {"sagemaker_program": "train_entry.py",
             "sagemaker_submit_directory": f"s3://{BUCKET}/{prefix}/code/source.tar.gz",
             "base": args.base, "stage": args.stage, "loss": args.loss, "epochs": str(args.epochs),
             "lr": str(args.lr), "lora-r": str(args.lora_r), "lora-alpha": str(args.lora_alpha),
             "moe-lora-mode": args.moe_lora_mode, "moe-hot-frac": str(args.moe_hot_frac),
             "head": str(args.head), "head-proj": str(args.head_proj)}
    env = {"LLADA_BASE": args.base, "JUL_DTYPE": "bfloat16"}
    if args.mask_id:
        env["JUL_LLADA_MASK_ID"] = str(args.mask_id)

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
        Environment=env,
        **extra,
    )
    print(f"\nlaunched: {job}" + (" (spot)" if args.spot else ""))
    print(f"poll: aws sagemaker describe-training-job --training-job-name {job} "
          f"--query TrainingJobStatus --region {REGION}")
    print(f"adapter: s3://{BUCKET}/{prefix}/output/{job}/output/model.tar.gz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
