# Iter 53 pre-registration — written 2026-08-17 22:2x, while the run was at step ~1150 of 10,935

Recorded BEFORE any eval number exists, so it is a prediction rather than a story fitted to the
result. Tool: `spectra.py` (CPU, seconds), run on the step-matched pair
`i53_ws_1000.pth` vs `i45_ws_1000.pth` — same recipe, same seed, augmentation off, differing in
`RWKV_MUON_INCLUDE_LORA=1` and nothing else. (Annotating a running experiment in a separate file
rather than in its runner, per the Ops rule that cost iters 43 and 46.)

## 1. The lever is strongly ENGAGED — this is not another inert flag

| group | median Δ stable rank | median relative change |
|---|---|---|
| **LoRA / scale (the lever)** | **+0.4697** | **+32.65%** (max +151.6%) |
| everything else (control) | −0.2182 | −1.70% |

Muon orthogonalizes the update, so flatter LoRA spectra is exactly its predicted signature, and the
control group barely moves. Contrast iters 48 and 50, where the diagnostic showed the mechanism was
*learned but negligible*: here the intervention is large. Whatever the eval says, it is not saying
"the flag did nothing."

⚠ Also moved: `||W||_F` on the LoRA group is **+11.17% median, 220% max**. Both groups sit at
`wd=0` (confirmed in the run's own banner: `27,520 LoRA in a wd=0 group`), so this is Muon's update
geometry, not a decay difference. A 220% norm growth on some tensor is worth looking at if the run
degrades — reported as a MAX because a median cannot see a blow-up (iter 51's lesson).

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
