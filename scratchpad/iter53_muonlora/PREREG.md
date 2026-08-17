# Iter 53 pre-registration — written 2026-08-17 22:2x, while the run was at step ~1150 of 10,935

Recorded BEFORE any eval number exists, so it is a prediction rather than a story fitted to the
result. Tool: `spectra.py` (CPU, seconds), run on the step-matched pair
`i53_ws_1000.pth` vs `i45_ws_1000.pth` — same recipe, same seed, augmentation off, differing in
`RWKV_MUON_INCLUDE_LORA=1` and nothing else. (Annotating a running experiment in a separate file
rather than in its runner, per the Ops rule that cost iters 43 and 46.)

## 1. The lever is strongly ENGAGED — this is not another inert flag

⚠ First version of this table lumped `lora_*` and `*scale*` together, because that is the Muon
EXCLUSION rule. But the flag moves only the `lora`-named tensors — `*scale*` stays on AdamW in
**both** runs, so it is a free **internal negative control**, measured on tensors of the same kind
and shape rather than on 80x80 matrices. Splitting them raised the lever's signal from +32.65% to
+48.26% and supplied the floor to read it against:

| group | n | median Δ stable rank | median rel. change | median Δ‖W‖_F |
|---|---|---|---|---|
| **`lora_*` (THE LEVER)** | 94 | **+0.6425** | **+48.26%** (max +151.6%) | +22.85% |
| `*scale*` (INERT control) | 26 | −0.0507 | −2.49% | +10.28% |
| everything else (on Muon in both) | 69 | −0.2182 | −1.70% | −0.37% |

Muon orthogonalizes the update, so flatter LoRA spectra is exactly its predicted signature, at a
**20:1 ratio over the floor**. Contrast iters 48 and 50, where the diagnostic showed a mechanism
that was *learned but negligible*: here the intervention is large. Whatever the eval says, it is
not saying "the flag did nothing."

★ **And the control corrects a reading I had already written down.** I flagged the lever's
`||W||_F` growth (+22.85% median, 220% max) as worth watching. But the INERT tensors — whose
optimizer treatment is identical in both runs — also grew **+10.28%**, because changing how the
LoRA matrices train changes the gradients reaching everything else. So most of the norm movement is
**indirect coupling, not the lever**, and only the stable-rank change survives attribution. A
control group is what separates "my intervention did this" from "the model moved."

## 2. The stated premise is directionally true but ~9× weaker than claimed — and the obvious fix inverts it

The proposal says the LoRA matrices are "the most anisotropic matrices in the model." Stable rank is
**not comparable across shapes** — these are rank-4/rank-2 bottlenecks next to 80×80 matrices — and
the two obvious normalizations disagree with each other:

| comparison | LoRA | Muon-managed | reads as |
|---|---|---|---|
| raw stable rank | 2.01 | 17.94 | LoRA 9× more anisotropic |
| ÷ min(shape) | 0.52 | 0.23 | **inverted** — LoRA 2× *flatter* |
| **÷ same-shape Gaussian** | **0.695** | **0.846** | LoRA 18% more concentrated |

Only the third is meaningful: a random (4,80) Gaussian already sits near 0.67 of its maximum stable
rank while a random 80×80 sits near 0.25, so dividing by `min(shape)` over-corrects by exactly the
amount that flips the sign. Computed empirically per shape rather than quoted from Marchenko–Pastur.

**Carry this:** *a shape-dependent statistic needs a shape-matched random reference, not a
shape-normalized ratio.* Same family as the median-vs-max lesson — a statistic that looks comparable
across objects usually is not.

## 3. The prediction

Given (1) a large intervention and (2) a real but modest 18% headroom, this is a genuine test rather
than a foregone null. Ranked by what I expect:

1. **Null or small harm** — most likely. The proposal's own pre-registered counter-hypothesis now
   has a measured mechanism behind it: a rank-4 bottleneck exists *to* concentrate, it is already at
   0.695 of random spread, and the flag pushes it further toward flat (+32.65%). Flattening the
   thing whose job is to be low-rank is the failure mode.
2. **Real gain** — if Muon's regularizer effect scales with how much of the model it covers, adding
   4.9% of params should buy a fraction of the +0.0019 iter 29 measured. That fraction is plausibly
   under the 0.0001 bar.

**If it is a null, the family verdict is "optimizer coverage, not optimizer choice, is exhausted"**
and the LoRA weight-decay entry (plan rank 8) should be read as the gentler retry of the *same*
question — not as an independent lever.

## 4. What would change my mind before the verdict

Re-run `spectra.py 10000` when the WS-final checkpoint lands. If the +32.65% at step 1000 has
*decayed* by step 10,000 — the way Muon's train-loss edge decays over training — then the flag is a
transient of early optimization and the eval should be read as measuring almost nothing.
