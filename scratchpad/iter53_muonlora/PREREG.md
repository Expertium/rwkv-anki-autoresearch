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

## 2. ⚠ CORRECTED — the premise HOLDS. My first correction of it was itself wrong.

**What this section said first, and why it was wrong.** I reported the LoRA matrices as only 18%
more concentrated than the Muon-managed ones (0.695 vs 0.846) and called the proposal's premise
"~9x weaker than claimed." That measurement was taken on the **CANDIDATE** checkpoint — which by
step 1000 had already had the flag raise its LoRA stable rank by 48%. **I measured the premise on
the treated model.** `spectra.py` now reads the CONTROL for Q1.

Measured correctly, on the champion:

| checkpoint | LoRA | Muon-managed | LoRA as a fraction |
|---|---|---|---|
| champion @ step 1000 | **0.5082** | 0.8524 | 60% |
| champion @ step 10935 | **0.5197** | 0.8146 | 64% |

(stable rank ÷ E[same-shape Gaussian]; 1.0 = as spread as random init.) **The champion's LoRA
matrices really are substantially more anisotropic** — and stably so across training. The premise is
sound.

**What survives from the original section, and it is the reusable part:** raw stable rank
(2.01 vs 17.94) is not comparable across shapes and overstates this ~9x; dividing by `min(shape)`
(0.52 vs 0.23) *inverts* the sign, because a random (4,80) Gaussian already sits near 0.67 of its
maximum while a random 80x80 sits near 0.25. Only the shape-matched random reference is meaningful.

**Two lessons, and the second is the one I keep re-learning:**
1. A shape-dependent statistic needs a shape-matched RANDOM reference, not a shape-normalized ratio.
2. **A premise must be measured on the UNTREATED model.** Reaching for the checkpoint that happened
   to be loaded is how a treatment effect gets read as a baseline property — the same error family
   as iter 47's step-50-vs-final comparison, committed while writing up a tool built to avoid it.

## 3b. PERSISTENCE CHECK at step 2000 — answered in the negative, and it raises a new worry

Section 4 said the prediction should be discounted if the step-1000 engagement had *decayed* by
later checkpoints, i.e. if it were a transient of early optimization. Re-run at the step-2000
matched pair:

| group | step 1000 | step 2000 |
|---|---|---|
| `lora_*` median rel. Δ stable rank | +48.26% | **+50.88%** |
| `*scale*` (inert) | −2.49% | −1.08% |
| `lora_*` median Δ‖W‖_F | +22.85% | **+36.43%** (max 336%) |
| `*scale*` Δ‖W‖_F | +10.28% | **+3.76%** |

The spectral effect **persists and grows slightly** — not a transient, so the escape route is
closed and the eval will be measuring something real.

★ **But the norm columns invert their story between the two checkpoints, and that is the new
worry.** At step 1000 the lever's ‖W‖_F growth was mostly indirect coupling (control +10.28% against
lever +22.85%). By step 2000 the control's drift has *fallen* to +3.76% while the lever's has *risen*
to +36.43% — so the norm growth is increasingly the lever's own, and it is accelerating rather than
settling. **Mechanism:** Muon takes a fixed-norm step along an orthogonalized direction, while Adam's
step adapts to gradient scale; with `wd=0` on this group there is nothing pulling the norm back. Over
10,935 steps that is a plausible route to harm, and it is *specific to moving params onto Muon
without giving them decay*.

**→ CHECK AT WS-FINAL (`spectra.py 10000`): does ‖W‖_F growth saturate or keep climbing?** If it is
still climbing at step 10,000, then plan rank 8 (weight decay on the LoRA group) stops being "the
gentler retry of the same question" and becomes **the fix for this run's failure mode** — i.e. the
right follow-up is Muon *plus* decay, not decay instead of Muon.

## 3c. RESOLVED AT WS-FINAL (step 10,935) — the norm growth does NOT saturate

| step | `lora_*` Δ stable rank | `lora_*` Δ‖W‖_F | `*scale*` (inert) Δ‖W‖_F |
|---|---|---|---|
| 1,000 | +48.26% | +22.85% | +10.28% |
| 2,000 | +50.88% | +36.43% | +3.76% |
| **10,935** | **+66.60%** (max +205.6%) | **+70.56%** (max 372%) | **+2.04%** |

Both columns answered:

* **Spectral engagement persists and grows** (+48 → +51 → +67%) while the inert control settles at
  +2.7%. Section 4's escape route — "discount this if the effect decayed" — is closed.
* **★ The norm growth does NOT saturate.** It roughly doubles between step 2,000 and WS-final while
  the inert control's indirect drift falls to +2.04%, so it is unambiguously the lever's own. Muon
  takes a fixed-norm step along an orthogonalized direction where Adam's adapts to gradient scale,
  and this group has `wd=0`, so nothing pulls it back.

**Per the pre-registration, this promotes plan rank 8.** Weight decay on the LoRA group is no longer
"a knob nobody chose" — it is **the fix for this run's specific failure mode, if it fails**, and the
right form is Muon *plus* decay rather than decay instead of Muon.

**And the trajectory sharpens the counter-hypothesis into something measurable.** The lever does not
merely close the gap to the Muon-managed group — it **overshoots**: LoRA goes from the champion's
0.52 to the candidate's **0.828**, past the 0.81 where the rest of the model sits. A rank-4
bottleneck exists *to* concentrate, and this flattens it past everything around it.

## 3d. THE DEPLOYED (decay-final) PAIR — and what stops the norm growth

`spectra.py 10935 d` compares `i53_d_10935` against `i45_d_10935`, i.e. the models the verdict
actually scores. Decay is half of all training here (`decay_ratio = 1.0`), so it had as much room to
move the weights as WS did.

| | WS-final | **decay-final (deployed)** |
|---|---|---|
| `lora_*` Δ stable rank | +66.60% | **+66.61%** |
| `lora_*` Δ‖W‖_F | +70.56% | **+62.40%** |
| `*scale*` (inert) Δ‖W‖_F | +2.04% | +2.16% |

**The spectral effect is unchanged through decay** (+66.60 → +66.61%, which is as stable as this
measurement gets). So the deployed model carries the full intervention; nothing washed out.

**★ But the norm growth came DOWN slightly, and the reason matters more than the number.** Decay
anneals the LR to zero, and Muon's step is *fixed-norm times LR* — so the growth stops because the
schedule ran out, **not because anything pulls the weights back**. There is no restoring force in
this configuration: `wd = 0` on the group, and Muon's update carries no scale feedback.

**→ CONSEQUENCE FOR THE 10x ENDGAME, which is the part worth carrying.** At ~12.5 epochs instead of
2, the same mechanism runs ~6x longer before the schedule stops it. A +62% norm growth at this budget
is not obviously harmful; the same process unchecked over the endgame's run is a different
proposition, and it would be discovered *there*, at ~4 days of GPU. **If iter 53 is adopted, the
endgame must either carry weight decay on the LoRA group or re-measure this at the longer budget.**
That is a cheap thing to know now and an expensive thing to find out later.

## 4. What would change my mind before the verdict

Re-run `spectra.py 10000` when the WS-final checkpoint lands. If the +32.65% at step 1000 has
*decayed* by step 10,000 — the way Muon's train-loss edge decays over training — then the flag is a
transient of early optimization and the eval should be read as measuring almost nothing.
