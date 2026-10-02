# Using LLaDA2 (masked diffusion) as an advanced classifier — findings & literature

This note consolidates the empirical work done on the MLX LLaDA2-MoE backend for System One typed
decisions, and steps back to place it against the published literature on diffusion LMs as
classifiers. All numbers are from `runs/` in this repo; all claims about external work cite the
paper.

## 1. What we built and measured

The backend (`lib/jul/backends/mlx_llada.py`, arch `lib/jul/vendor/llada2_moe.py`) reads a decision
at a `[MASK]` position of `mlx-community/LLaDA2.0-mini-preview-4bit`, bidirectionally, in one forward
pass. On top of it we implemented and benchmarked several readouts and optimizations on the Jev BTZSC
pilot (AG News 4 labels, Banking77 72, Emotion 6) and the 439-example holdout.

### Readouts and levers, with the measured verdict

| Lever | Where | Effect (measured) | Verdict |
|---|---|---|---|
| **anchor-only projection** | `mlx_llada.py::option_logits`, `mask.py` | identical quality (diff 0.0), latency −60% (holdout), ÷2 on Banking77 | **adopt (free)** |
| **multi-token readout** | `mask.py` `readout="multitoken"` | Banking77 0.02→0.19, mean acc 0.430→0.497, ECE 0.147→0.135, faster | **adopt for many-class** |
| **8-bit vs 4-bit** | model choice | Banking77 0.19→0.34 (same method) | **adopt if RAM allows** |
| **per-class bias** (fit on val) | `pisteC` / `validate_banking_combo` | Banking77 0.34→0.62 | **adopt (needs ~200 labels)** |
| **two-stage temperature** | `recalibrate_banking_combo` | keeps acc 0.62, ECE 0.35→0.125 | **adopt for calibration** |
| n_mask / n_steps tricks | `mask.py` | +0 quality, ×2–×3 latency on these tasks | reject (default) |
| early-skip (logit-lens) | (explored) | quality collapse or no latency win | reject |
| multi-question one-pass | (explored) | ×2.2 faster but cross-talk drops agreement to 0.84 | reject (needs segmented attention) |
| hybrid e5 → LLaDA re-rank | `hybrid_e5_rerank` | re-rank *hurts* vs e5 alone (0.60→0.39) | reject the re-rank |

### The Banking77 trajectory (the headline)

Read at a single anchor token, a 72-way task collapses to **0.02**. Stacking the diffusion-native
and calibration levers, on the full test with a strict no-leak protocol (fit on val 200, eval on test
100, confirmed on train 1000):

```
anchor mono-token           0.02
+ multi-token readout       0.19
+ 8-bit (same model)        0.34
+ per-class bias (val fit)  0.62   ← competitive with e5-small-v2 (0.60)
+ two-stage temperature     0.62 / ECE 0.125   ← accuracy kept, calibration recovered
```

Final pipeline: `p = softmax( (multitoken_scores / T1 + bias) / T2 )`, T1/bias/T2 all fit on val.

## 2. The two mechanisms we found (and why they generalize)

1. **The readout channel, not the model, was the first ceiling.** A single-anchor comparison
   saturates on many-class tasks. Scoring the *full label text* over a block of masks
   (sequence-likelihood) is the diffusion-native fix and unblocks it. This is a *method* limit.
2. **A per-label score-scale bias, correctable and transferable.** Some label suffixes score
   systematically high/low regardless of the input. Because both splits are class-balanced, fitting a
   per-class additive bias on val is *not* learning a frequency prior — it removes an input-independent
   readout bias, which is why it transfers val→test cleanly (0.585→0.540, train 0.578). It lifts
   accuracy but inflates confidence, so a second temperature is needed to recalibrate.

## 3. Step back — what the literature says (and how it lines up)

The key reference is **"Discrete Diffusion Language Models Are Training-Free Multi-Label Classifiers"**
(Pawan Kumar, arXiv:2608.14649, SIAM SDM 2026), which uses LLaDA-8B and Dream-7B *as-is* as
classifiers. It independently confirms almost every empirical finding here, and adds theory:

- **Slot-position asymmetry.** Packing all labels/answers into one all-masked prompt makes the
  alphabetically-first slot collapse (99.4% positive on GoEmotions, 100% on Reuters), dragging macro-F1
  to ~0. This is *exactly* the many-slot failure mode; their fix — **per-label entailment scoring**,
  querying every label at the same syntactic position with a `yes/no` verbalizer at one `[MASK]` — is
  permutation-invariant and recovers Reuters macro 10.9→38.2. Our multi-token readout is a variant of
  the same principle (query the label text at a fixed position rather than a shared answer suffix).
- **Calibration is essential and is done exactly as we did it.** They state raw diffusion log-odds are
  poorly calibrated (ECE 0.4–0.7) and fit **temperature + threshold on a 200-example validation slice**
  — the same budget and the same tool we used. Their Appendix 18 flags **per-label / Platt-style fits**
  as headroom (+2–3 micro-F1), which is precisely the per-class bias that gave us +0.28.
- **Instruct ≫ Base**, and the gain is a **per-label bias correction**: their Reuters positive-rate
  plot (Appendix 21) shows Instruct reduces per-label rate error 10.45%→5.54%. That is the same
  "per-label score-scale bias" our additive bias corrects at inference time, without a better
  checkpoint.
- **They tested LLaDA2.0-mini (16B MoE) too** (Appendix 27): int4, per-label, Reuters 56.8/31.2 →
  67.1/38.4 with a "main topic" prompt. So our model is literally the one they flag as future work,
  and prompt/template choice is a documented lever we have not yet swept.
- **Retrieval caps, not helps, prompt-based scoring.** Their Theorem 4.4 gives a hard recall/F1
  ceiling when a retriever shortlists labels: EURLEX at k=32 caps micro-F1 at ~47%. This matches our
  hybrid-e5 result from the other direction: the embedding does the useful work, and the diffusion
  re-rank inside the shortlist adds nothing (they use the retriever only to *fit in the prompt budget*,
  not to improve accuracy).
- **JSR / iterative refinement hurts** — their Section 7 documents monotone degradation from iterative
  local updates, the same reason our n_steps demasking did not help.

Two more relevant lines from the wider literature:

- **Surface-form competition & Domain-Conditional PMI** (Holtzman et al., EMNLP 2021,
  arXiv:2104.08315): LM label scoring is biased by each label string's a-priori likelihood; subtract
  the score under a neutral context. This is exactly our piste B (PMI): it improved macro-F1
  (0.114→0.178) by de-biasing rare intents. The per-class additive bias is a learned, stronger version
  of the same correction.
- **MDMs as language understanders** (Nie et al., "Scaling up Masked Diffusion Models on Text",
  arXiv:2410.18514): a 1.1B MDM beats a same-data 1.1B AR model on 4/8 zero-shot understanding
  benchmarks, and proposes unsupervised classifier-free guidance — evidence that masked diffusion
  carries genuine discriminative signal, consistent with our results.

## 4. Consolidated recommendation

For **advanced classification with LLaDA2**, the recipe that the literature and our measurements agree
on:

1. **Per-label / per-answer scoring at a fixed position** (never a long all-masked multi-slot suffix).
   For binary/entailment: `yes/no` verbalizer. For single-label choice: sequence-likelihood of each
   label text (our multi-token readout). Both avoid slot-position asymmetry.
2. **Prefer an Instruct checkpoint and the best quantization you can fit** (8-bit > 4-bit here).
3. **Calibrate on a small labelled slice**: temperature for calibration; a per-class additive bias for
   accuracy on many-class tasks; then a second temperature to recalibrate. Threshold/verbalizer/prompt
   template are all selected on the same 200-example val slice.
4. **Sweep the prompt/question template** — documented as a large lever (Reuters 60.6→80.5 micro).
   Not yet done here; likely the biggest untapped gain.
5. **Do not iterate (JSR/n_steps) and do not use diffusion to re-rank an embedding shortlist** — both
   are measured negatives.

## 5. Honest limits

- Banking77 tops out ~0.62 here vs 0.88 for a contrastive embedding (`wemm-4b`) zero-shot: for pure
  retrieval-style many-class tagging, a trained embedding is still structurally stronger. The diffusion
  readout's edge is a single model that also does entailment, scoring and yes/no with calibrated
  probabilities, and reads the text and the label *together* (an embedding cannot).
- The per-class bias and the two temperatures need ~200 labelled val examples: this is light
  calibration, not zero-shot, and not model training.
- Test slices are 100–439 rows (wide CI); the 1000-row train read is the tie-breaker.
- Prompt-template sensitivity means every number is conditional on the template; a val-selected sweep
  is mandatory before trusting a cell.
- Banking77/Emotion are in MTEB, so embedding baselines may have seen them; AG News is the clean
  comparison (there the diffusion multi-token readout is 0.87).

## 6. Pointers

- Code: `lib/jul/backends/mlx_llada.py`, `lib/jul/mask.py` (`readout="multitoken"`),
  `scripts/explore_1_multitoken.py`, `scripts/piste{A,B,C,D}_*.py`, `scripts/hybrid_e5_rerank.py`,
  `scripts/validate_banking_combo.py`, `scripts/recalibrate_banking_combo.py`.
- Reports: `runs/jev-bench-llada-mlx/`, `runs/banking-{A,B,C,D,hybrid,combo,recalib}/`.
- Key paper: arXiv:2608.14649 (dLLM-SetScore, SIAM SDM 2026). Supporting: arXiv:2104.08315 (DCPMI),
  arXiv:2410.18514 (MDM scaling / understanding), arXiv:2502.09992 (LLaDA), arXiv:2512.15745 (LLaDA2.0).
</content>
</invoke>
