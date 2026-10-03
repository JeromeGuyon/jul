# Handoff — LLaDA-based typed decider (Jev-like) on decision-bench

Pick-up note for a fresh session. Everything below is measured, not assumed. Numbers are the honest
UNSEEN (zero-shot) figures with Wilson/bootstrap intervals, not best-of-on-test.

## 0. One-paragraph summary

We are building a typed decision reader (Choice / Noul / Score) on top of **LLaDA-8B-Instruct** (a masked
**diffusion** LM, read at a `[MASK]` position, not autoregressive), in `/Users/jerome/dev/jul-bis`
(fork of `usejul/jul`). We measure it on **decision-bench** (`/Users/jerome/dev/decision-bench`) against
Jev (0.873) and **wemm-4b v2.1 (0.829 UNSEEN, 0.849 ALL)**. Current honest ceiling of our LLaDA-8B
decider is **~0.713 UNSEEN**. Two leverage tests were run and both came back negative (details below):
a learned read head, and a bigger backbone (LLaDA2.0-mini). The real remaining edge of LLaDA is **flat
~50 ms latency on 1 GPU**, which the MoE backbone destroys.

## 1. Methodology rules (do not break these — they were hard-won)

- Never pick a config on the test set. Freeze routing/config BEFORE measuring. Report with Wilson CI.
- One change at a time between two runs.
- Train format == inference format (shared `build_example` / readout).
- Verify for free before paying a GPU job (render diffs, distributions, splits, `build_example` on the
  whole corpus). We lost 2 GPU jobs to a `dict + str` we could have caught locally.
- Contamination: separate SEEN sources (seen in training) vs UNSEEN. The honest number is UNSEEN.
- Do not oversell. `≥3 pts on ALL` is the significance bar; `±2 pts` is noise at n≈1600.
- `wemm v2.1` training corpus is **open-weights, closed-data** — irrecoverable. The strict "same corpus"
  test is impossible; stop chasing it.

## 2. Where the code is (git)

- Repo: `/Users/jerome/dev/jul-bis`, branch `experimental/llada-learned-head-20261002`.
- Remotes: `fork` = `https://github.com/JeromeGuyon/jul.git` (push here; you have write access),
  `origin` = `usejul/jul` (NO write access — never push here).
- Committed tip: `8c7c28f` (= `f2034b7` base + Hermes PR#1: noul fix + 4 structural fixes).
- **UNCOMMITTED (important), must be committed next:**
  - `lib/jul/backends/llada.py` — the 6 llada2_moe backend fixes (see §6). Modified, not committed.
  - `scripts/paired_bootstrap.py` — new, untracked.
- Push target when ready: `git push fork experimental/llada-learned-head-20261002`.

## 3. The comparison (iso: same items, same splitter `scripts/score_split.py`)

UNSEEN, with `JUL_SEEN_EXTRA=dbpedia,mnli,banking77,agnews,trec,imdb,sst5,boolq,amazon,yelp`
(the decision-v7 training sources, so UNSEEN is honest):

| system (UNSEEN)            | ALL   | choice | noul  | score | predictions file        |
|----------------------------|-------|--------|-------|-------|--------------------------|
| LLaDA-8B CE (orig corpus)  | 0.701 | 0.782  | 0.752 | 0.395 | `runs/ft/predictions.jsonl` |
| LLaDA-8B readout-logits v7 | 0.713 | 0.797  | 0.745 | 0.437 | `runs/v7/predictions.jsonl` |
| LLaDA-8B learned head v7   | 0.708 | 0.770  | 0.737 | 0.493 | `runs/head2/predictions.jsonl` |
| **wemm-4b v2.1 (target)**  | 0.829 | 0.872  | 0.859 | 0.661 | in decision-bench results |

Gap to wemm = **−11.6 pts UNSEEN**, widest on **score (−22 pts)** then noul (−11) then choice (−7.5).

## 4. Result A — learned read head (NEGATIVE, verified)

Idea (Hermes): wemm/Kev/Mapika read with a learned head/pointer/projection on a representation, not raw
vocab logits. LLaDA is bidirectional → `[MASK]` hidden state is a good representation (DiffEmbed
2505.15045). So put a per-type head on the hidden state.

- Implemented: `lib/jul/llada_head.py` `ReadHead` (choice = bilinear pointer; noul = 2-logit linear;
  score = ordinal head). Backend `mask_hidden()` returns last-layer hidden at `[MASK]`. Training via
  `train_entry.py --head 1`; inference via env `JUL_LLADA_HEAD` → `MaskReader._head_scores`.
- Trained LLaDA-8B dense `--head 1` on decision-v7 (r16/alpha32/lr5e-5/2ep), loss 0.199.
- **Paired bootstrap** (`scripts/paired_bootstrap.py`, official `decision_bench.judge`, same items):
  - all:    Δ −0.005  [−0.026, +0.016]  P(Δ>0)=0.31  → null
  - choice: Δ −0.027  [−0.054, +0.000]  P=0.02       → **real small DROP** (key = letter A/B/C embedding,
    carries no option content; head learns position, not option)
  - noul:   Δ −0.008  → null
  - score:  Δ +0.056  [−0.024, +0.133]  P=0.91, n=286 → **NOT significant** (CI crosses 0)
- Confound: with `--head`, the LoRA is trained on the head loss, not CE — so even the score bump mixes
  two variables (reading + adapter). Verdict: **no significant gain, degrades choice.**
- Bug found+fixed on the way (Hermes root cause): `options_of(Noul)=[true,false]`, inference wrongly
  remapped to `[false,true]` → every noul answer inverted (noul 0.263 = 1−0.737). Fixed: read in jul
  order. Regression test in `tests/test_mask.py` (RED P(true)=0.0009 → GREEN). The FIRST head eval
  (job 1790927713) is INVALID because of this; the valid one is job 1790929814 → `runs/head2`.

## 5. Result B — bigger backbone LLaDA2.0-mini (NOT VIABLE, stop criterion hit)

- LLaDA2.0-mini = 16B MoE (256 experts, 8 active, ~1.4B active/token), Apache-2.0, `inclusionAI`.
  Backbone quality > LLaDA-8B (MMLU 80.5, avg 71.7 ≈ Qwen3-8B). Weights 32.5 GB BF16 (7 shards).
- **Latency (the stop criterion), job 1790941091 us-east-1, g5.12xlarge 4×A10G, device_map auto, BF16:**
  **p50 = 458 ms, p95 = 637 ms, mean 472 ms** (296 quick items, 0 err) → `runs/llada2lat/`.
  vs LLaDA-8B dense **p50 = 51 ms on 1×A10G**.
- Verdict: ~9× slower AND 4 GPUs instead of 1 → the flat-50 ms edge disappears, cost/decision explodes.
  wemm-4b does 0.829 with a 4B on 1 GPU. **LLaDA2.0-mini is not a viable backbone here.** Do not spend
  more optimizing heads on this MoE line.
- NB: 458 ms is the naive transformers MoE path (experts in a Python loop). InclusionAI's dInfer/SGLang
  engine would be faster, but our `[MASK]` decision readout does not go through their engine.

## 6. Backend fixes for llada2_moe (uncommitted, in lib/jul/backends/llada.py)

Six remote-code requirements, each surfaced by a GPU run; all fixed and scoped so the dense 8B is NOT
regressed (the dense repo maps BOTH `AutoModel` and `AutoModelForCausalLM` to the same `LLaDAModelLM`):
1. transformers pin **4.57.1** for llada2 (the repo's saved version; `TransformersKwargs` import).
2. `JUL_LLADA_DEVICE_MAP=auto` to shard 32.5 GB across GPUs (OOM on 1×24 GB otherwise).
3. `attention_mask` required (dense tolerated None).
4. 4D block mask shape `(1, 1, L, L)`.
5. mask dtype = model's **float** (not long/bool) — SDPA `_unmask_unattended` wants float.
6. load with **`AutoModelForCausalLM`** (not `AutoModel`): base model gives truncated 2048 output →
   `IndexError 6177/2048`; the LM class gives full vocab 157184 logits.
Helper added: `LLaDABackbone._attn_mask(ids)` returns the 4D float mask only for `model_type` containing
`llada2`, else None. Loader tries `AutoModelForCausalLM` then falls back to `AutoModel`.

## 7. Infra / AWS (profile ark-sb, account 356912105607)

- SageMaker launchers now read env vars (sanitized): `JUL_AWS_ACCOUNT`, `JUL_SAGEMAKER_ROLE`,
  `JUL_SAGEMAKER_BUCKET`, `JUL_DBENCH`, `AWS_DEFAULT_REGION`. Export them before launching.
- Multi-region: `launch_dbench.py` takes `--region` and `--image`. us-east-1 bucket created:
  `amazon-sagemaker-356912105607-us-east-1-f3a64759569c`. us-east-1 DLC that works:
  `763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.9.0-gpu-py312-cu130-ubuntu22.04-sagemaker`
  (the `-v1` tag gives `manifest unknown`). eu-west-1 DLC: `...:2.8.0-gpu-py312-cu129-...-sagemaker`.
- GPU capacity: g5 very tight in eu-west-1 after ~09:00 CET. g5.12xlarge available in us-east-1.
  Quota spot g5 = 1 (sequential); on-demand lets you parallelize.
- Corpus decision-v7 (public Kev suite): `/Users/jerome/dev/jul/data/mix/decision-v7.clean.jsonl`
  (15576 items, shuffled, decontaminated vs bench-v1: 0 text overlap; sources dbpedia+mnli shared →
  handled via UNSEEN). Built by `scripts/convert_decision_v7.py` from HF `jaredpalmer/kev-suites@cc4bac8`
  path `v7/decision-v7/train.jsonl`.
- Key adapters on S3 (bucket `amazon-sagemaker-356912105607-eu-west-1-f3a64759569c`):
  - CE baseline (sane, r16/alpha32): `jul-train/jul-llada-train-a-1790711478/...`
  - readout-logits v7: `.../jul-llada-train-a-1790890108/...`
  - learned head v7: `.../jul-llada-train-a-1790922224/...` (contains read_head.pt + read_head.json)

## 8. How to reproduce a run

Train (dense, decision-v7, CE readout):
```
export JUL_AWS_ACCOUNT=356912105607 JUL_SAGEMAKER_ROLE=arn:aws:iam::356912105607:role/service-role/AmazonSageMaker-ExecutionRole-20230621T162682
export JUL_SAGEMAKER_BUCKET=amazon-sagemaker-356912105607-eu-west-1-f3a64759569c AWS_DEFAULT_REGION=eu-west-1 JUL_DBENCH=/Users/jerome/dev/decision-bench
AWS_PROFILE=ark-sb python deployment/sagemaker-eval/launch_train.py \
  --train /Users/jerome/dev/jul/data/mix/decision-v7.clean.jsonl \
  --base GSAI-ML/LLaDA-8B-Instruct --stage a --loss ce \
  --lora-r 16 --epochs 2 --lr 5e-5 --instance ml.g5.2xlarge
```
Eval full + split:
```
AWS_PROFILE=ark-sb python deployment/sagemaker-eval/launch_dbench.py \
  --model llada-8b-instruct --suite full --readout auto-ce \
  --adapter s3://.../model.tar.gz --instance ml.g5.2xlarge
# then: JUL_SEEN_EXTRA='dbpedia,mnli,banking77,agnews,trec,imdb,sst5,boolq,amazon,yelp' \
#       python scripts/score_split.py decision-bench/data/bench-v1.jsonl runs/<x>/predictions.jsonl
```
Readouts: `JUL_LLADA_READOUT` ∈ {anchor, multitoken, auto, auto-ce}. `auto-ce` is best for a CE adapter
(choice→anchor, noul→yes/no multi-anchor, score→sequence-likelihood). Learned head: set `JUL_LLADA_HEAD`
to a `read_head.pt` (eval harness auto-exports it when the adapter tar contains one).

## 9. Recommended next steps (in order)

1. **Commit the uncommitted work** (`lib/jul/backends/llada.py` + `scripts/paired_bootstrap.py`) to the
   experimental branch and push to `fork`. It is the only copy of the 6 llada2_moe fixes.
2. **Check iLLaDA (arXiv 2606.25331)** — a *dense* 8B LLaDA retrained from scratch. VERIFY FIRST (free):
   are the weights public on HF? what are its numbers? If public, it is the best "same-architecture,
   better backbone" candidate WITHOUT the MoE latency trap. Preset slot exists: `illada-8b-instruct`
   (`GSAI-ML/iLLaDA-8B-Instruct`, mask_id 5) in `lib/jul/backbone.py` — confirm the real repo id.
   **CHECKED 2026-10-02:** weights public, Apache-2.0, not gated, repo id confirmed (sha `5769f04`).
   Dense 7.62B, GQA 8 KV heads, vocab 155136, tied embeddings, BF16 ~16.5 GB (fits 1×A10G).
   Instruct: MMLU 71.6 (vs 65.5 LLaDA), MMLU-Pro 52.3 (vs 37.0), MMLU-Redux 76.4 (vs 68.9).
   `<[MASK]>` = id 5; `config.mask_token_id` absent, tokenizer declares no mask/unk token. Remote code
   `ILLaDAForCausalLM` (both auto classes), attention_mask optional, saved with transformers 4.57.1.
   Silent bugs found and fixed before any GPU job: `_guess_mask_id` returned 3 (UNK) for iLLaDA, and
   `launch_dbench --mask-id` defaulted to 126336 (a valid token in iLLaDA's vocab) → now 5. Both
   launchers pin 4.57.1 for iLLaDA (tied lm_head under unpinned 5.x is a risk).
   **LATENCY MEASURED** (job 1790961340, eu-west-1, 1×A10G, quick 296, no adapter, readout auto,
   0 errors) → `runs/illadalat/`: **p50 51.3 ms, p95 71.0 ms** ≈ dense. Stop criterion NOT hit.
   Iso zero-shot vs LLaDA-8B base (`runs/db`, same items/readout, paired bootstrap 10k):
   all 0.517→0.659 Δ+0.142 [+0.071,+0.213]; noul Δ+0.242 [+0.109,+0.375]; choice Δ+0.055
   [−0.023,+0.133] n.s.; score Δ+0.100 [−0.075,+0.275] n.s. (n=40). Untrained backbones only — says
   nothing yet about the adapted UNSEEN ceiling.
   **Next: CE adapter on decision-v7, same recipe as `runs/v7`, then full eval + UNSEEN split.**
   Iso trap: LLaDA has no `o_proj` (its output proj is `attn_out`), so the default attention suffixes
   adapt q/k/v on LLaDA but q/k/v/o on iLLaDA. Pass `--lora-targets q_proj,k_proj,v_proj`.
   **ADAPTED RESULT** (train job 1791043685, ce/r16/a32/5e-5/2ep, q/k/v, mask_id 5; eval job
   1791050981, full, auto-ce, 0 err) → `runs/illada-v7/`. UNSEEN n=1591: **ALL 0.733 [0.711,0.754]**,
   choice 0.806, noul 0.762, score 0.493. Paired vs `runs/v7`: all Δ+0.020 [−0.001,+0.041] P=0.97;
   choice +0.008 n.s.; noul +0.017 n.s.; score +0.056 [−0.010,+0.119] n.s. Latency with adapter p50
   54.5 / p95 86.3 ms, 1×A10G. **Verdict: below the 3-pt bar, not significant.** The +14-pt zero-shot
   lead shrinks to +2 after the same adaptation → backbone knowledge is not the bottleneck at this
   recipe; the gap to wemm (−9.6 UNSEEN, −16.8 on score) lies in corpus and/or readout.
   Repro: `python scripts/paired_bootstrap.py --baseline runs/v7/predictions.jsonl
   --candidate runs/illada-v7/predictions.jsonl --baseline-label v7 --candidate-label illada`.
3. **Listwise choice head** (dense, cheap, same forward): key = hidden state at EACH option's marker
   position (sees the option text), not the letter embedding. This is the only honest test of "does the
   reading block choice?" — the bilinear-on-letter head failed exactly there (choice −2.7). If this does
   not recover 0.797→higher, listwise probably won't either, and reading is not the bottleneck.
4. If staying on readout-logits: the plateau is ~0.713 UNSEEN; more data at the same scale did not move
   ALL (decision-v7 vs CE = +1.2, noise). The honest story is: LLaDA-8B dense ≈ 0.713, flat 50 ms/1 GPU;
   wemm 0.829 but 4B needs its closed corpus. LLaDA's edge is latency, not quality.

## 10. Open review points NOT yet done (from Hermes)

- Split the branch into reviewable PRs.
- Env vars still read inside the library in places (should be injected).
- `auto-ce` behavior documentation.
- `runs/` artifacts not committed to decision-bench results in native format (real run.json, pinned HF
  revision) — if you commit LLaDA runs to decision-bench, use the bench's own runner, not reconstructed
  JSON.
- Document honest numbers in `docs/llada-decision-bench-recipe.md` (UNSEEN 0.713, not the old best-of
  0.753) and the alpha=2r bug (an earlier bug where lora_alpha=256 with r=16 gave 16× scaling and
  collapsed the readout — already fixed, default is now `2*r`).
