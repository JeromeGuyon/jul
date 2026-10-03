# LLaDA in JuL — Option 1 (read) and Option 2 (train)

This fork adds a **masked-diffusion** backbone (LLaDA / iLLaDA) to JuL and reads it at a single
`[MASK]`, the native readout of a diffusion LM and the closest open analogue to Jev's
non-autoregressive "parallel sampler". Two options, sharing one readout.

## Why LLaDA

Jev's org forked `ML-GSAI/LLaDA` (a diffusion LM), not a causal decoder. The three public Jev
reimplementations JuL tracks (Kev = causal decoder + pointer head, Laya = encoder + `[MASK]`, JuL =
vectors + pointer) all use **causal or encoder** backbones. A masked-diffusion backbone is the
missing fourth branch, and the only one that has the `[MASK]` readout **natively** (it is its
pre-training objective) while keeping a generative LM's world knowledge and a **bidirectional**
view of the state. This is the single controlled variable to test the standing open question in the
working project's `DECISION_TREE.md`: *does the backbone family, not just the data, move zero-shot
generalization?* (today: legal leave-one-family-out ≈ 0.525, i.e. chance).

## Option 1 — read a frozen LLaDA at `[MASK]` (zero-shot)

New code, all behind `method="mask"` / `backend="llada"`, no change to the moteur or the public API:

| File | Role |
|------|------|
| `lib/jul/backends/llada.py` | `LLaDABackbone`: loads LLaDA via `AutoModel(trust_remote_code)`, resolves the mask id (env `JUL_LLADA_MASK_ID` > tokenizer mask token > family default: LLaDA 126336 / iLLaDA 5), exposes `mask_logits(tokens, mask_positions)` — one forward pass, logits at the mask. |
| `lib/jul/mask.py` | `MaskSpec` + `MaskReader.logits(...)` — same signature as `PointerReader.logits`. Builds `state\n<head>[MASK]`, reads the option-anchor logits at the mask, remaps noul `[false,true]` to jul's option order, divides by a temperature. |
| `lib/jul/engine.py` | `Engine.mask` created when `preset.method == "mask"`. |
| `lib/jul/client.py` | `system_one` routes a mask preset through `MaskReader`, same calibration path as the pointer. |
| `lib/jul/presets.py` | `llada_repo` field; `mask_preset(...)` factory; built-in presets `llada-8b-instruct`, `illada-8b-instruct`. |
| `scripts/eval_llada_zeroshot.py` | Accuracy + ECE per family + leave-one-family-out on the sealed holdout, same metrics as the working project's `eval_holdout.py`. |

Run (needs the LLaDA weights and, realistically, a GPU):

```bash
JUL_LLADA_MASK_ID=126336 python scripts/eval_llada_zeroshot.py \
    --model llada-8b-instruct \
    --holdout data/eval-holdout-v2/holdout.jsonl \
    --json runs/llada-8b-instruct-eval-v2.json
```

Compare its `in_mix` ECE and `leave_one_family_out` accuracy head-to-head against Harrier-0.6B LoRA
(0.847 / 0.525). If the diffusion backbone is better calibrated or generalizes further **with no
training**, Option 2 is worth the GPU.

## Option 2 — train LLaDA at `[MASK]` (the same readout, optimized)

`scripts/llada_train.py` keeps the exact Option-1 readout and trains it. Two stages, mirroring the
project's Harrier pipeline:

- **Stage A — distill / cross-entropy.** `build_example(...)` reuses `MaskReader`'s own prompt and
  anchors, so training and inference read the state identically. Loss = cross-entropy on the gold
  option, or KL against a teacher distribution when `soft` labels are present. **LoRA** on q/k/v/o
  (full fine-tuning of an 8B collapses — measured on Harrier: LoRA 0.847 vs full-FT 0.302). No
  pointer head: the `[MASK]` distribution *is* the output.
- **Stage B — RLCD.** A strictly **proper scoring rule** (Brier) plus an **asymmetric penalty on
  confident errors** (`rlcd_loss`): being wrong at 99% costs far more than at 55%. This is the
  calibration-first objective Jev's RLCD is described to optimize; a proper scoring rule is uniquely
  minimized by reporting one's true probabilities. Optionally wrapped in a GRPO-style policy
  gradient.

The maths (`cross_entropy`, `kl_distillation`, `brier`, `rlcd_loss`) and `train_step` run on CPU
with a tiny random masked-LM (`toy_model_and_tokenizer`), covered by `tests/test_llada_train.py`.

### Wiring into the existing SageMaker pipeline (no GPU run here)

The working project trains on SageMaker via `deployment/sagemaker-training/{train_entry.py,launch.py}`
with a base repo and a JSONL corpus (`data/mix/harrier-train.jsonl`, records already carry
`state`, `type`, `instructions`, `options`, `gold`, optional `soft`). To target LLaDA:

1. **Base repo.** Pass `--base GSAI-ML/LLaDA-8B-Instruct` (or iLLaDA) instead of the Harrier repo.
   Load with `AutoModel(trust_remote_code=True)` (LLaDA ships remote code), not
   `AutoModelForCausalLM`.
2. **Readout.** Replace the pointer-head forward with `llada_train.option_logits_from_model` — no
   head to add, no delimiter tokens to inject, no embedding resize. Each record becomes a
   `MaskExample` via `build_example`.
3. **Loss.** Stage A: `train_step(..., stage="a")` (KL when `soft` present, else CE). Gate on the
   sealed holdout with `eval_llada_zeroshot.py`. Stage B: `train_step(..., stage="b")` only if the
   RLCD calibration beats temperature-scaled Stage A on out-of-distribution families.
4. **LoRA.** For the dense 8B, q/k/v/o_proj. For **LLaDA-MoE**, attention-only LoRA never touches
   the experts (where a sparse MoE does its work): use the routing-guided targeting in
   `scripts/moe_lora.py` (`--moe-lora-mode routing`, default). It profiles the router on a few calib
   examples, keeps the hottest `--moe-hot-frac` (0.25) experts per layer, and puts LoRA on those hot
   experts + the router gate + shared experts (none in this checkpoint) + attention. `max-len` ≥ 1024.
5. **Instance.** g5/g4dn spot as in the working project; a diffusion 8B has the same memory profile
   as a dense 8B for a single forward pass (no KV-cache growth — attention is bidirectional).
6. **Cost control.** LLaDA-MoE-7B-A1B (≈1B active) is the "intelligence per dollar" variant to try
   if the dense 8B is too slow; it reuses the same `[MASK]` readout.

Nothing above requires a GPU to *wire*; the calibration maths and the readout are verified on CPU.
A real run is a `launch.py` invocation with the base repo swapped and the loss module pointed at
`llada_train`.

## Running it for real — findings (2026-09-29, this machine)

Verified end-to-end on an Apple Silicon Mac (MPS, no CUDA), transformers 5.17, torch 2.14:

- **It loads and reads.** LLaDA-8B loads on MPS (~190 s) and the `[MASK]` readout returns finite
  logits over the full 126 464 vocab. On a real ticket ("charged twice"), the client returns
  `team -> billing` at **0.977** and `is_bug -> 0.124` — a correct, confident, well-shaped decision.
  The mask mechanism is validated on real weights, not just stubs.
- **transformers 5.x compatibility shim (required).** LLaDA's published remote code predates
  transformers 5.x and breaks in several places; `backends/llada.py` patches them minimally, without
  downgrading transformers (the rest of JuL stays on 5.x):
  1. `LLaDAModelLM.all_tied_weights_keys` missing -> set to `{}`.
  2. `tie_weights()` called with `missing_keys=/recompute_mapping=` -> wrapped to ignore new kwargs.
  3. `LLaDAConfig.use_cache` missing -> set to `False` (bidirectional, no KV cache).
  4. **RoPE init (LLaDA-MoE).** transformers 5.x removed the `"default"` entry from
     `ROPE_INIT_FUNCTIONS` (plain-RoPE moved into the `RotaryEmbedding` class). The MoE remote code
     still does `ROPE_INIT_FUNCTIONS["default"]` -> `KeyError: 'default'`, and some 5.x paths then
     index `rope_parameters["factor"]` -> `KeyError: 'factor'`. The shim registers a self-contained
     `"default"` init that computes `inv_freq` from `rope_theta`/`head_dim`/`partial_rotary_factor`
     alone (no `factor`, scaling 1.0), matching the `(config, device, seq_len=...)` signature. It is
     registered only when missing, so it is a no-op on transformers 4.x and never perturbs the dense
     8B path. Verified: the MoE `LLaDAMoERotaryEmbedding` initialises and returns correct
     `(cos, sin)` of head_dim 128 under transformers 5.17.
- **Preferred fix for a SageMaker run: pin transformers.** The `launch_train.py`/`launch_eval.py`
  DLC `requirements.txt` pins `transformers==4.53.3` — the version the authors targeted (LLaDA-MoE
  config declares 4.53.2, dense LLaDA-8B declares 4.46.3, so 4.53.x satisfies both remote codes
  natively, no shimming). transformers has no hard upper torch pin, so 4.53.x is compatible with the
  DLC's torch 2.8 / py3.12. This is more robust than fighting the moving 5.x RoPE API in a training
  run. RISK: 4.53 is older than the DLC stack; if a future DLC bumps a transitive dep past 4.53's
  range, relax the pin — the in-process shim above is kept, so an unpinned load still works.
- **Performance wall.** ~120-200 s **per question** on MPS float32 (the full-vocab forward dominates;
  longer prompts are slower). The 439-record holdout with several questions each is ~13+ hours —
  run it detached (`nohup ... &`), not interactively. A CUDA GPU is the right place for the full gate.
- **No MLX backend for LLaDA.** JuL's fast Apple-Silicon path (`backends/mlx.py`, via `mlx_lm`) is
  written for *causal* models and there is no published MLX conversion of LLaDA's masked-diffusion
  architecture. MLX would be the deployment path for a *validated* model, not for this feasibility
  gate; torch+MPS is enough to decide. A future `backends/mlx_llada.py` is possible once an MLX port
  of LLaDA exists.

## MLX backend for LLaDA2-MoE (added 2026-09-29, this machine)

The "future `backends/mlx_llada.py`" above now exists. `mlx-community` published a 4-bit MLX
conversion of the LLaDA2-MoE diffusion model, so the fast Apple-Silicon path is available for a
LLaDA-family model read at `[MASK]`. Verified end to end on this Mac (mlx 0.32, mlx_lm 0.31.1):

- **`lib/jul/backends/mlx_llada.py`** — `MLXLLaDABackbone` (`backend="mlx_llada"`), the MLX twin of
  the torch `llada` backend: same `mask_logits(tokens, mask_positions)` contract read by
  `decision.MaskReader`, but the forward runs through `mlx_lm` on the Metal GPU. Loads the tokenizer
  with `trust_remote_code=True` so the load is non-interactive. Mask id resolves to `<|mask|>`
  (156895) for LLaDA2.
- **`lib/jul/vendor/llada2_moe.py`** — the `llada2_moe` architecture for `mlx_lm`. mlx_lm resolves a
  model by importing `mlx_lm.models.<model_type>`, and stock mlx_lm does not ship `llada2_moe`, so
  the backend registers this vendored module under that import path at load time (a no-op if a newer
  mlx_lm provides it). It is the Bailing/Ling MoE stack (fused `query_key_value`, per-head query/key
  RMSNorm, partial RoPE `rotary_dim=64`, grouped-sigmoid routing with an `expert_bias` and
  `routed_scaling_factor=2.5`, `first_k_dense_replace=1`, one shared expert) with the diffusion
  change: **attention is bidirectional (no causal mask)**.
- **Preset** `llada2-mini-4bit` -> `mlx-community/LLaDA2.0-mini-preview-4bit` in `MODELS`.

Findings on this machine:
- **It loads and reads.** A cloze "The capital of France is `<mask>`." scores ` Paris` first
  (then Berlin, Madrid, Tokyo) — the readout is correct on real 4-bit weights.
- **A full decision works.** `MaskReader` + `Choice` routes the "charged twice" ticket to
  `billing` at ~0.75 (technical 0.008, sales 0.243).
- **Footprint.** ~8-9 GB resident (16B total / ~1B active, 4-bit), forward in a few seconds per
  question on an M-series GPU — no CUDA, no 120-200 s/question wall the dense torch/MPS path hit.
- **Tests.** `tests/test_mlx_llada.py` (marked `slow`, `JUL_SLOW=1`) locks both the cloze readout
  and the ticket decision. The full suite stays green (`python -m pytest tests`).
- **Not trainable this way.** This is a 4-bit inference build; fine-tuning still starts from the
  BF16 `inclusionAI/LLaDA2.0-mini-preview` under dFactory/torch, not from this MLX checkpoint.

## System One readout optimizations on the MLX backend (added 2026-09-30)

Two levers were implemented and benchmarked on the Jev BTZSC pilot (300 examples, same manifest and
metric code as `scripts/bench_jul.py`). The reference is the single-anchor readout ("baseline").

**Anchor-only projection** (`backends/mlx_llada.py::option_logits`). At a [MASK] the full 157k-way
`lm_head` projection is replaced by a projection onto just the option anchor rows. Numerically
identical to `mask_logits(...)[:, anchors]` (verified max abs diff 0.0), so **zero quality change**,
and the saving grows with the number of options. On the holdout-439 (220 examples, 11 families) the
decisions were identical and p50 latency dropped 60% (69 -> 43 ms), up to 2.8x on many-option
families. `MaskReader._read_option_logits` uses it automatically when the backbone exposes it.

**Multi-token (sequence-likelihood) readout** (`MaskSpec.readout="multitoken"`, env
`JUL_LLADA_READOUT=multitoken`). Instead of one anchor token per option, it scores the full label
text: strip the token-level common prefix shared by all option texts, place K=max-distinctive-suffix
[MASK]s, run one bidirectional forward, and score each option as the length-normalized mean
log-softmax of its own suffix tokens (`mean` aggregation; `sum` collapses short labels). This is the
native parallel diffusion read and unblocks many-class tasks the anchor readout saturates on.

Jev pilot, 100 rows/dataset:

| Config | AG News acc/ECE | Banking77 acc/ECE | Emotion acc/ECE | Mean acc | Mean ECE | p50 |
|---|---|---|---|---|---|---|
| baseline (anchor) | 0.84 / 0.058 | 0.02 / 0.006 | 0.43 / 0.378 | 0.430 | 0.147 | 50 ms |
| anchor-only | 0.84 / 0.058 | 0.02 / 0.006 | 0.43 / 0.378 | 0.430 | 0.147 | 53 ms |
| **multitoken** | 0.87 / 0.088 | **0.19** / 0.152 | 0.43 / **0.164** | **0.497** | **0.135** | **35 ms** |

Findings:
- **Multi-token unblocks the many-class collapse.** Banking77 0.02 -> 0.19 (~10x), AG News +0.03,
  Emotion accuracy flat but ECE 0.378 -> 0.164. Mean accuracy 0.430 -> 0.497, mean ECE 0.147 -> 0.135.
- **It is also faster**, not slower: one K-mask forward beats the baseline's per-option projection.
  Banking77 p50 260 -> 35 ms (÷7) on the integration run.
- **Iterative demasking does not help the multi-token readout.** On Banking77: n_steps=1 -> 0.19
  (30 ms), n_steps=2 -> 0.20 (59 ms, +0.01 within noise, 2x latency), n_steps=3 -> 0.18 (regresses).
  Keep n_steps=1. The ~0.19 Banking77 ceiling is a model/label-similarity limit, not a readout one.
- **Verdict:** adopt anchor-only always (free) and multitoken for choice questions with many/verbose
  options. Rejected levers (measured): early-skip (no safe latency win — the decision only stabilizes
  in the last layers), n_mask/n_steps tricks (cost latency, no quality gain on these tasks),
  multi-question one-pass (2.2x faster but cross-talk drops decision agreement to 0.84), LLaDA2.2
  swap (no MLX-loadable 2.2-mini; Levenshtein editing is out of scope for a single-pass readout).

Tests: `tests/test_mlx_llada.py` covers the anchor cloze/ticket reads and the multi-token topic read
(marked `slow`, `JUL_SLOW=1`). Bench: `scripts/bench_llada_mlx_optims.py`
(`--configs baseline,anchor-only,multitoken`), per-exploration scripts in `scripts/explore_*.py`,
reports under `runs/explore-*/` and `runs/jev-bench-llada-mlx/`.

## SageMaker GPU eval — REAL RESULT (2026-09-29, your AWS profile, eu-west-1)

Ran the Option-1 gate on a real GPU (g5.2xlarge, A10G 24GB, managed spot) via
`deployment/sagemaker-eval/{launch_eval.py,eval_entry.py}`. The job ships jul-bis's `lib/jul`
(llada backend + compat shim), loads LLaDA-8B on the GPU, runs the sealed holdout, writes the JSON.

- **Smoke (24 nli):** acc 0.750, ECE 0.103 — 155 s total, 363 s billable.
- **Full holdout (439, 11 families):** in-mix **acc 0.594 / ECE 0.172**; true zero-shot
  **legal 0.500 / ECE 0.406** — 189 s eval, 384 s billable.

Per family (LLaDA-8B `[MASK]`, zero-shot, no training): strong support 1.000 (ECE 0.003),
theme 0.925, nli 0.744 (ECE 0.055), moderation 0.725; weak finance 0.200, emotion 0.350,
tool_routing 0.425 (ECE 0.534, overconfident), sentiment 0.475, intent 0.500.

**Gate verdict.** "Diffusion generalizes better in zero-shot" is **not** supported: LLaDA zero-shot
legal (0.500) ~= Harrier-0.6B LoRA legal (0.525), both near chance. Changing the backbone family
does not by itself unlock zero-shot generalization — the project's thesis "the gain is in the data,
not the architecture" (DECISION_TREE D0) holds. LLaDA is strong zero-shot where lexical meaning
suffices and collapses/overconfident on reasoning tasks — the profile of Harrier head-only (0.576)
before LoRA, so Option 2 (train LLaDA at [MASK]) *could* close the gap, at 8B vs 0.6B cost.
Reports: `runs/llada-eval.json` (24), `runs/llada-eval-full439.json` (439).

## Full exploration result (2026-09-29) — tricks + Option 2 training

All on GPU (g5.2xlarge spot, A10G), sealed holdout 439, your AWS profile.

| Config | in-mix acc | in-mix ECE | legal (zero-shot) | p50 latency |
|---|---|---|---|---|
| LLaDA-8B baseline (1 mask, 1 step) | 0.594 | 0.172 | 0.500 | 49 ms |
| LLaDA-8B tricks (n_mask=3, n_steps=2) | 0.574 | 0.108 | **0.625** | 98 ms |
| LLaDA-8B + Stage A LoRA + tricks | 0.591 | 0.112 | 0.500 | 107 ms |
| **Harrier-0.6b LoRA (reference)** | **0.847** | 0.078 | 0.525 | ~13 ms |

Findings:
- **Inference tricks help, for free.** The diffusion multi-position sampler (n_mask=3 + n_steps=2)
  lifts true zero-shot legal 0.500 -> 0.625 (above Harrier-LoRA 0.525) and ECE 0.406 -> 0.085, with
  no training. This is the architecture-specific lever Harrier (causal) does not have. (legal n=40:
  the accuracy delta needs a larger holdout to confirm; the ECE gain is robust.)
- **Stage A LoRA did NOT close the accuracy gap** (0.574 -> 0.591) and *regressed* zero-shot legal
  (0.625 -> 0.500) and support calibration (ECE 0.003 -> 0.370): the adapter overfit in-mix families.
  Likely causes: 1 epoch / LR 1e-4 / r=16 unoptimized on an 8B; the [MASK] readout is less directly
  steered by q/k/v/o LoRA than Harrier's dedicated pointer head; the Sonnet soft labels were shaped
  for Harrier's delimiter format, not LLaDA's [MASK] prompt (mild train/inference mismatch).
- **LLaDA-MoE-7B-A1B could not be loaded** under transformers 5.x: its remote code
  (`modeling_lladamoe`) breaks in RoPE init (`ROPE_INIT_FUNCTIONS['default']`, then `'factor'`).
  Two shim attempts failed on the same subsystem, so the MoE (the intelligence-per-dollar / latency
  bet) needs a transformers ~4.46 venv, not more shims. Dense 8B (proven) was used for Option 2.

**Overall verdict.** On in-mix accuracy an 8B diffusion model (~0.59, with or without this LoRA)
stays far below a well-trained 0.6B causal model (Harrier 0.847) on the project's own data —
"data > architecture" (DECISION_TREE D0) holds again. The only lever that moved true zero-shot was a
diffusion-native *inference* trick, not training. Latency ~50-107 ms is viable but 4-8x Harrier.
Training runs: Stage A `jul-llada-train-a-1790711478` (LoRA, loss 0.30, 798 s billable).
Reports: `runs/llada-eval-{nmask3,nsteps4,nmask3-nsteps2,stageA-tricks}.json`.

## Status

- Option 1: implemented, unit-tested (`tests/test_mask.py`), eval script parses the real sealed
  holdout. A real accuracy/ECE number needs the 8B weights + GPU.
- Option 2: losses + example builder + `train_step` implemented and CPU-smoke-tested
  (`tests/test_llada_train.py`); SageMaker wiring documented above, no GPU run performed.
- Full suite: `python -m pytest tests` — green, no regression on the existing JuL behavior.
