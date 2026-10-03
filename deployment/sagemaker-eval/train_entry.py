"""SageMaker training entry: fine-tune LLaDA / LLaDA-MoE at the [MASK] readout (Option 2).

Trains the same [MASK] readout that inference uses (jul.mask.MaskReader), on the MixExample corpus
(state/instructions/options/gold/soft). LoRA on attention proj keeps the diffusion backbone intact
(full-FT of a big model collapses, as measured on Harrier). Two stages:

  Stage A: cross-entropy on the gold option, or KL against the teacher soft distribution when present.
  Stage B: RLCD (Brier + asymmetric confident-error penalty) — calibration-first. Enabled with
           --stage b (usually after Stage A, from its adapter).

The readout, prompt and anchors come from jul.mask (shipped in source_dir), so training and inference
read the state identically. Writes the LoRA adapter + tokenizer to $SM_MODEL_DIR.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time


def _load_rows(path: str):
    if os.path.isdir(path):
        files = [f for f in os.listdir(path) if f.endswith(".jsonl")]
        path = os.path.join(path, files[0])
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def _train_listwise_head(args, model, tok, reader, to_q, device, PeftModel):
    """Train the listwise choice head alone, on top of a FROZEN adapter.

    One change under test: a wemm-style listwise reading of choice (query = [MASK] hidden, keys = each
    option's closing-separator hidden). The CE iLLaDA-v7 adapter is loaded is_trainable=False, so the
    backbone and its LoRA get no gradient; only the two projections choice_q / choice_k are trained,
    on CHOICE examples only, with cross-entropy on the gold option. Config is frozen by the caller
    (proj=256, lr=1e-3, epochs=2, AdamW). Writes read_head.pt + read_head.json + train_config.json.
    """
    import json as _json
    import random as _random
    import time as _time

    import torch
    from torch.optim import AdamW
    from jul import llada_head as LH
    from jul.types import options_of

    import glob as _glob
    import tarfile as _tarfile

    def _has_adapter(d):
        return bool(d) and os.path.isfile(os.path.join(d, "adapter_config.json"))

    # Resolve args.adapter to a directory that actually holds adapter_config.json. SageMaker mounts the
    # adapter channel as a directory containing model.tar.gz (training channels are NOT auto-extracted),
    # so the raw channel dir has no adapter_config.json: extract the tar and point at lora_adapter/.
    ch = args.adapter or os.environ.get("SM_CHANNEL_ADAPTER", "")
    if not _has_adapter(args.adapter):
        tars = _glob.glob(os.path.join(ch, "*.tar.gz")) if ch and os.path.isdir(ch) else []
        if tars:
            dest = "/opt/ml/adapter"; os.makedirs(dest, exist_ok=True)
            with _tarfile.open(tars[0]) as t:
                t.extractall(dest)
            cand = os.path.join(dest, "lora_adapter")
            args.adapter = cand if _has_adapter(cand) else dest
            print(f"[llada-train] extracted frozen adapter -> {args.adapter}", flush=True)
    if not _has_adapter(args.adapter):
        raise SystemExit(f"--head-listwise needs a frozen --adapter with adapter_config.json "
                         f"(got {args.adapter!r}, channel {ch!r})")
    # load the adapter FROZEN: is_trainable=False -> no LoRA parameter should carry a gradient.
    model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    model.eval()                                   # frozen backbone + frozen LoRA
    lora_params = [(n, p) for n, p in model.named_parameters() if "lora_" in n.lower()]
    lora_grad = [n for n, p in lora_params if p.requires_grad]
    print(f"[llada-train] frozen adapter from {args.adapter}: "
          f"lora_tensors={len(lora_params)} lora_requires_grad={len(lora_grad)}", flush=True)
    if lora_grad:
        raise SystemExit(f"adapter not frozen: {len(lora_grad)} LoRA params require grad, e.g. {lora_grad[:3]}")

    # build CHOICE examples only, with the shared listwise prompt builder
    from jul.mask import _render
    rows = _load_rows(args.train)
    examples = []                                  # (tokens, mask_pos, ends, gold_index)
    skipped_types = {"noul": 0, "score": 0}
    for r in rows:
        if r["type"] != "choice":
            skipped_types[r["type"]] = skipped_types.get(r["type"], 0) + 1
            continue
        q = to_q(r)
        opts = options_of(q)
        keys = [o.key for o in opts]
        if r["gold"] not in keys:
            continue
        gi = keys.index(r["gold"])
        state = r["state"] if isinstance(r["state"], str) else _render(r["state"])
        tokens, mask_pos, ends = reader._listwise_choice_tokens(r["instructions"], opts, state)
        if len(tokens) <= reader.spec.max_state_tokens + 256 and len(ends) == len(opts):
            examples.append((tokens, mask_pos, ends, gi))
    print(f"[llada-train] listwise CHOICE examples={len(examples)} "
          f"(skipped noul={skipped_types.get('noul',0)} score={skipped_types.get('score',0)}) "
          f"mask_id={reader.mask_id}", flush=True)
    if reader.mask_id != 5 and "illada" in args.base.lower():
        raise SystemExit(f"expected mask_id 5 for iLLaDA, got {reader.mask_id}")

    # the head: only choice_q / choice_k are trained; everything else is frozen
    in_emb = model.get_input_embeddings()
    hidden = model.config.hidden_size if hasattr(model.config, "hidden_size") else in_emb.embedding_dim
    head = LH.ReadHead(hidden=hidden, emb=in_emb, proj=args.head_proj)
    head = head.to(device=device, dtype=next(model.parameters()).dtype)
    for n, p in head.named_parameters():
        p.requires_grad = n.startswith("choice_q.") or n.startswith("choice_k.")
    head.train()
    trainable = [(n, p) for n, p in head.named_parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for _, p in trainable)
    # the only trainable tensors in the whole run must be the two projections' weight+bias
    expected = {"choice_q.weight", "choice_q.bias", "choice_k.weight", "choice_k.bias"}
    got = {n for n, _ in trainable}
    print(f"[llada-train] trainable head params: tensors={sorted(got)} total={n_trainable}", flush=True)
    if got != expected:
        raise SystemExit(f"trainable set must be exactly the two projections, got {sorted(got)}")
    model_grad = [n for n, p in model.named_parameters() if p.requires_grad]
    if model_grad:
        raise SystemExit(f"backbone must be frozen, but {len(model_grad)} params require grad")

    opt = AdamW([p for _, p in trainable], lr=args.lr)
    rng = _random.Random(args.seed)
    for epoch in range(args.epochs):
        order = list(range(len(examples)))
        rng.shuffle(order)
        total, n, t0, window = 0.0, 0, _time.time(), 0.0
        for n_i, idx in enumerate(order, 1):
            tokens, mask_pos, ends, gold = examples[idx]
            ids = torch.tensor([tokens], dtype=torch.long, device=device)
            with torch.no_grad():
                out = model(input_ids=ids, output_hidden_states=True)
                hs = out.hidden_states[-1][0]              # (T, hidden) — backbone frozen
            h_mask = hs[mask_pos].detach()
            h_ends = hs[ends].detach()
            loss = LH.listwise_choice_loss(head, h_mask, h_ends, gold, device)
            opt.zero_grad(); loss.backward(); opt.step()
            lv = float(loss.detach()); total += lv; n += 1; window += lv
            if n % 500 == 0:
                print(f"[llada-train] epoch {epoch+1} step {n} slice_loss={window/500:.4f} "
                      f"cum_loss={total/n:.4f}", flush=True)
                window = 0.0
        print(f"[llada-train] epoch {epoch+1}/{args.epochs} listwise-choice "
              f"mean_loss={total/max(1,n):.4f} ({_time.time()-t0:.0f}s)", flush=True)

    os.makedirs(args.model_dir, exist_ok=True)
    # re-save the frozen adapter so the eval tar carries both the adapter and the head
    model.save_pretrained(os.path.join(args.model_dir, "lora_adapter"))
    torch.save(head.state_dict(), os.path.join(args.model_dir, "read_head.pt"))
    _json.dump({"hidden": head.hidden, "proj": head.proj, "max_levels": head.score.out_features,
                "reading": "listwise"},
               open(os.path.join(args.model_dir, "read_head.json"), "w"))
    tok.save_pretrained(args.model_dir)
    _json.dump({"base": args.base, "stage": "head-listwise", "loss": "ce",
                "head": "listwise-choice", "frozen_adapter": args.adapter,
                "proj": args.head_proj, "lr": args.lr, "epochs": args.epochs,
                "examples": len(examples), "mask_id": reader.mask_id,
                "trainable_tensors": sorted(got), "trainable_params": n_trainable},
               open(os.path.join(args.model_dir, "train_config.json"), "w"))
    print(f"[llada-train] saved listwise head + frozen adapter -> {args.model_dir}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.environ.get("LLADA_BASE", "inclusionAI/LLaDA-MoE-7B-A1B-Instruct"))
    ap.add_argument("--stage", default="a", choices=["a", "b"])
    # GIFT (arXiv 2509.20863) recommended LoRA SFT config: r=128, alpha=256, lr=1e-5, several epochs.
    ap.add_argument("--loss", default="gift", choices=["ce", "diffusion", "gift", "ordinal"],
                    help="Stage-A reconstruction loss: ce (baseline), diffusion (LLaDA 1/t), "
                         "gift (entropy-importance 1/t_i, default and recommended by GIFT)")
    ap.add_argument("--epochs", type=int, default=3, help="GIFT trains several epochs (paper: up to 20), not 1")
    ap.add_argument("--seed", type=int, default=0, help="shuffle seed for epoch ordering (reproducible)")
    ap.add_argument("--head", type=int, default=0,
                    help="1 = train a learned per-type read head on the [MASK] hidden state (DiffEmbed) "
                         "instead of reading vocabulary logits")
    ap.add_argument("--head-listwise", type=int, default=0,
                    help="1 = train ONLY the listwise choice head (wemm port: query=[MASK] hidden, "
                         "keys=each option's closing-separator hidden) on CHOICE examples only, on top "
                         "of a FROZEN adapter (--adapter loaded is_trainable=False). The backbone LoRA "
                         "gets no gradient; only choice_q/choice_k are trained.")
    ap.add_argument("--head-proj", type=int, default=256, help="projection dim of the learned head")
    ap.add_argument("--lr", type=float, default=1e-5, help="GIFT LoRA learning rate")
    ap.add_argument("--lora-r", type=int, default=128, help="GIFT LoRA rank")
    ap.add_argument("--lora-alpha", type=int, default=None,
                    help="LoRA alpha; default 2*r (scaling 2x). A fixed 256 with r=16 gives a 16x "
                         "scaling that collapses the readout — always keep alpha ~ 2*r.")
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--lora-targets", default="",
                    help="comma-separated module suffixes, overriding --moe-lora-mode (e.g. q_proj,k_proj,v_proj)")
    # Routing-guided MoE LoRA (MoE-Sieve, arXiv 2603.24044): attention-only LoRA ignores the experts.
    ap.add_argument("--moe-lora-mode", default=os.environ.get("JUL_MOE_LORA_MODE", "routing"),
                    choices=["routing", "all", "attention"],
                    help="routing: LoRA on hot experts + gate + shared + attention (default); "
                         "all: every expert + gate + shared + attention; "
                         "attention: q/k/v/o only (MoE-blind, legacy)")
    ap.add_argument("--moe-hot-frac", type=float,
                    default=float(os.environ.get("JUL_MOE_HOT_FRAC", "0.25")),
                    help="fraction of hottest experts per layer to adapt in routing mode")
    ap.add_argument("--moe-calib", type=int, default=int(os.environ.get("JUL_MOE_CALIB", "32")),
                    help="number of built examples used to profile routing (routing mode)")
    ap.add_argument("--alpha", type=float, default=3.0, help="RLCD asymmetric penalty weight (stage b)")
    ap.add_argument("--max-state", type=int, default=512)
    ap.add_argument("--train", default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    ap.add_argument("--adapter", default=os.environ.get("SM_CHANNEL_ADAPTER", ""))
    ap.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    args = ap.parse_args()
    if args.lora_alpha is None:
        args.lora_alpha = 2 * args.lora_r   # scaling 2x, the safe default for any rank

    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(here, "lib"))
    sys.path.insert(0, here)

    import torch
    from torch.optim import AdamW
    from transformers import AutoModel, AutoTokenizer

    # backend shim (transformers 5.x) + the mask reader / trainer
    from jul.backends.llada import _patch_remote_code_for_transformers5, _guess_mask_id
    from jul.mask import MaskReader, MaskSpec
    from jul.types import Choice, Noul, Score, NoulCriteria, options_of
    import llada_train as LT

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[llada-train] stage={args.stage} loss={args.loss} base={args.base} device={device} "
          f"lora_r={args.lora_r} alpha={args.lora_alpha} lr={args.lr} epochs={args.epochs}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    # transformers 4.53 (the LLaDA-MoE pin) expects torch_dtype=; 5.x renamed it dtype=. Try both.
    with _patch_remote_code_for_transformers5(args.base):
        try:
            model = AutoModel.from_pretrained(args.base, trust_remote_code=True, torch_dtype=dtype)
        except TypeError:
            model = AutoModel.from_pretrained(args.base, trust_remote_code=True, dtype=dtype)
    try:
        _ = model.config.use_cache
    except AttributeError:
        model.config.use_cache = False
    model.to(device)

    from peft import LoraConfig, get_peft_model, PeftModel
    import moe_lora as MLORA

    # a MaskReader over this tokenizer, to build training examples exactly like inference
    class _BB:
        name = args.base
        def __init__(s):
            s.tokenizer = tok
            s.mask_id = _guess_mask_id(args.base, tok, model.config)
    spec = MaskSpec.default()
    spec = spec.__class__(**{**spec.__dict__, "max_state_tokens": args.max_state})
    reader = MaskReader(_BB(), spec)

    def to_q(r):
        if r["type"] == "choice":
            return Choice(instructions=r["instructions"], criteria=dict(r["options"]))
        if r["type"] == "noul":
            o = r["options"]
            return Noul(instructions=r["instructions"],
                        criteria=NoulCriteria(true=o.get("true", "Yes."), false=o.get("false", "No.")))
        levels = [r["options"][k] for k in sorted(r["options"], key=lambda x: int(x))]
        return Score(instructions=r["instructions"], criteria=levels)

    # ---- listwise choice head, trained alone on a FROZEN adapter (the diagnostic of this run) -------
    if args.head_listwise:
        _train_listwise_head(args, model, tok, reader, to_q, device, PeftModel)
        return

    rows = _load_rows(args.train)
    examples = []
    for r in rows:
        q = to_q(r)
        opts = options_of(q)
        keys = [o.key for o in opts]
        if r["gold"] not in keys:
            continue
        gi = keys.index(r["gold"])
        soft = None
        if isinstance(r.get("soft"), dict):
            soft = [float(r["soft"].get(k, 0.0)) for k in keys]
        ex = LT.build_example(reader, r["type"], r["instructions"], opts, r["state"], gi, soft)
        if len(ex.tokens) <= reader.spec.max_state_tokens + 64:
            examples.append(ex)
    print(f"[llada-train] {len(examples)}/{len(rows)} examples built", flush=True)

    # LoRA target modules (create fresh, or resume from a Stage-A adapter for Stage B).
    # For a MoE base, target modules follow --moe-lora-mode: routing-guided by default, so LoRA
    # lands on the hot experts + router gate + shared experts + attention instead of attention alone
    # (attention-only LoRA never touches the experts, where a sparse MoE does its work).
    if args.adapter and os.path.isdir(args.adapter):
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=True)
        print(f"[llada-train] resumed adapter from {args.adapter}", flush=True)
        moe_info = {"mode": "resumed"}
        target_modules = None
    else:
        # profile routing on a handful of built examples (one free forward each), before wrapping.
        if args.lora_targets:
            target_modules = [t.strip() for t in args.lora_targets.split(",") if t.strip()]
            moe_info = {"mode": "explicit", "targets": target_modules}
        else:
            calib = [{"input_ids": ex.tokens} for ex in examples[: max(1, args.moe_calib)]]
            target_modules, moe_info = MLORA.build_target_modules(
                model, mode=args.moe_lora_mode, hot_frac=args.moe_hot_frac,
                calib_batches=calib, device=device)
        print(f"[llada-train] moe-lora mode={args.moe_lora_mode} "
              f"experts_selected={moe_info.get('experts_selected')}/{moe_info.get('experts_total')} "
              f"gates={moe_info.get('num_gates')} shared={moe_info.get('num_shared')} "
              f"target_modules={len(target_modules)}", flush=True)
        lconf = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                           target_modules=target_modules, bias="none")
        model = get_peft_model(model, lconf)
    model.train()

    # Optional learned read head on the [MASK] hidden state (DiffEmbed-style), trained jointly with LoRA.
    head = None
    if args.head:
        from jul import llada_head as LH
        in_emb = model.get_input_embeddings()
        hidden = model.config.hidden_size if hasattr(model.config, "hidden_size") else in_emb.embedding_dim
        head = LH.ReadHead(hidden=hidden, emb=in_emb, proj=args.head_proj)
        head = head.to(device=device, dtype=next(model.parameters()).dtype)
        head.train()
        print(f"[llada-train] learned head on: hidden={hidden} proj={args.head_proj}", flush=True)

    params = [p for p in model.parameters() if p.requires_grad]
    if head is not None:
        params += [p for p in head.parameters() if p.requires_grad]
    opt = AdamW(params, lr=args.lr)
    rng = random.Random(args.seed)
    for epoch in range(args.epochs):
        order = list(range(len(examples)))
        rng.shuffle(order)  # shuffle each epoch so families are interleaved, not trained in blocks
        total, n, t0 = 0.0, 0, time.time()
        window = 0.0  # per-slice (last 500) loss, so a diverging slice is visible, not hidden by the running mean
        for n_i, idx in enumerate(order, 1):
            ex = examples[idx]
            if head is not None:
                # read the [MASK] hidden state and train the per-type head (not the vocab logits).
                ids = torch.tensor([ex.tokens], dtype=torch.long, device=device)
                out = model(input_ids=ids, output_hidden_states=True)
                h = out.hidden_states[-1][0, ex.mask_pos]
                loss = LH.head_loss(head, h, ex, device)
            else:
                # Stage A: ce | diffusion (LLaDA 1/t) | gift (entropy 1/t_i); t sampled ~U(0,1] per step.
                loss = LT.train_step(model, ex, args.stage, device, alpha=args.alpha, loss=args.loss)
            opt.zero_grad(); loss.backward(); opt.step()
            lv = float(loss.detach())
            total += lv; n += 1; window += lv
            if n % 500 == 0:
                print(f"[llada-train] epoch {epoch+1} step {n} "
                      f"slice_loss={window/500:.4f} cum_loss={total/n:.4f}", flush=True)
                window = 0.0
        print(f"[llada-train] epoch {epoch+1}/{args.epochs} loss={args.loss} "
              f"mean_loss={total/max(1,n):.4f} ({time.time()-t0:.0f}s)", flush=True)

    os.makedirs(args.model_dir, exist_ok=True)
    model.save_pretrained(os.path.join(args.model_dir, "lora_adapter"))
    if head is not None:
        torch.save(head.state_dict(), os.path.join(args.model_dir, "read_head.pt"))
        json.dump({"hidden": head.hidden, "proj": head.proj, "max_levels": head.score.out_features},
                  open(os.path.join(args.model_dir, "read_head.json"), "w"))
        print("[llada-train] saved learned read head", flush=True)
    tok.save_pretrained(args.model_dir)
    json.dump({"base": args.base, "stage": args.stage, "loss": args.loss, "lora_r": args.lora_r,
               "lora_alpha": args.lora_alpha, "lr": args.lr, "epochs": args.epochs,
               "moe_lora_mode": args.moe_lora_mode, "moe_hot_frac": args.moe_hot_frac,
               "lora_targets": target_modules,
               "moe_experts_selected": moe_info.get("experts_selected"),
               "moe_experts_total": moe_info.get("experts_total"),
               "examples": len(examples), "mask_id": reader.mask_id},
              open(os.path.join(args.model_dir, "train_config.json"), "w"))
    print(f"[llada-train] saved adapter -> {args.model_dir}/lora_adapter", flush=True)


if __name__ == "__main__":
    main()
