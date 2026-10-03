"""SageMaker entry point: LLaDA zero-shot [MASK] eval on the sealed holdout (GPU).

Runs inside an AWS Deep Learning Container (PyTorch GPU). It ships the whole jul-bis package
(lib/jul, including the 'llada' backend and the mask reader) in the source dir, adds it to sys.path,
loads LLaDA on the GPU, and runs scripts/eval_llada_zeroshot.evaluate on the holdout mounted at
$SM_CHANNEL_HOLDOUT. The JSON report is written to $SM_MODEL_DIR so SageMaker tars it to S3.

Nothing is trained here: this is the Option-1 feasibility gate, on a GPU where ~120-200 ms/forward
(vs ~150 s on a Mac's MPS) makes the 439-record holdout tractable in minutes.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time


def _find_holdout(channel: str) -> str:
    if os.path.isfile(channel):
        return channel
    files = glob.glob(os.path.join(channel, "*.jsonl"))
    if not files:
        raise SystemExit(f"no .jsonl in holdout channel {channel}")
    return files[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("JUL_MODEL", "llada-8b-instruct"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--holdout", default=os.environ.get("SM_CHANNEL_HOLDOUT", "/opt/ml/input/data/holdout"))
    ap.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    # the packaged jul-bis library and scripts live next to this entry point
    sys.path.insert(0, os.path.join(here, "lib"))
    sys.path.insert(0, here)  # eval_llada_zeroshot.py is shipped alongside

    # optional LoRA adapter (Option 2): a channel holding model.tar.gz (adapter + tokenizer)
    adapter_dir = os.environ.get("SM_CHANNEL_ADAPTER", "")
    if adapter_dir and os.path.isdir(adapter_dir):
        import tarfile
        tars = glob.glob(os.path.join(adapter_dir, "*.tar.gz"))
        dest = "/opt/ml/adapter"
        os.makedirs(dest, exist_ok=True)
        if tars:
            with tarfile.open(tars[0]) as t:
                t.extractall(dest)
        cand = os.path.join(dest, "lora_adapter")
        os.environ["JUL_LLADA_ADAPTER"] = cand if os.path.isdir(cand) else dest
        print(f"[eval-llada] adapter -> {os.environ['JUL_LLADA_ADAPTER']}", flush=True)

    import torch
    print(f"[eval-llada] torch={torch.__version__} cuda={torch.cuda.is_available()} "
          f"device={'cuda' if torch.cuda.is_available() else 'cpu'}", flush=True)

    holdout = _find_holdout(args.holdout)
    print(f"[eval-llada] model={args.model} holdout={holdout} limit={args.limit}", flush=True)

    from eval_llada_zeroshot import evaluate, print_report

    t = time.time()
    rep = evaluate(args.model, holdout, backend="llada", limit=args.limit)
    rep["seconds"] = round(time.time() - t, 1)
    print_report(rep)

    os.makedirs(args.model_dir, exist_ok=True)
    out = os.path.join(args.model_dir, "llada-eval.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print(f"[eval-llada] report -> {out} in {rep['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
