# Scheduling-monotonicity plan (Andrew 2026-07-16)

**Problem (Andrew's screenshot):** Anki button intervals from RWKV-Curve at fixed desired
retention can invert — e.g. Again 17d > Hard 1d. Nothing in training constrains the four
counterfactual post-rating curves: at each history position only the ACTUAL rating is ever
trained, so the other three one-step-advanced states are off-distribution extrapolation.
Andrew's directive: the monotonicity constraint should live IN THE MODEL, not (only) as a
post-hoc clamp in Anki — a clamp makes "intervals the model wants" ≠ "intervals users get".

**Key reduction:** if the four post-rating curves are ordered POINTWISE
(P_Again(t) ≤ P_Hard(t) ≤ P_Good(t) ≤ P_Easy(t) ∀t), then intervals are ordered at EVERY
desired retention simultaneously (t*(r) = sup{t: P_r(t) ≥ DR} is monotone in set inclusion).
So the trainable/structural surrogate = pointwise curve ordering, not per-DR interval order.

Two independent monotonicity axes:
1. **Rating axis** (the screenshot bug): order across the 4 counterfactual curves.
2. **Time axis:** each individual curve should be non-increasing in t — the mixture's basis
   curves are currently unconstrained, so a single curve can be non-monotone → multiple
   DR crossings, ill-defined intervals.

## Staged plan (recommended)

### Stage 0 — AUDIT (do first, cheap, no training)
Offline script: sample eval users (RNN mode), at every review position compute the 4
counterfactual intervals at DR ∈ {0.80, 0.90, 0.95} (same imputed duration/elapsed for all
four ratings — mirroring what Anki's button computation must do, since duration is unknown
at display time). Measure: violation rate, severity (ratio of inverted intervals), which
pairs (expect Again-vs-Hard to dominate — post-lapse curves are where the data is
strangest), and time-axis non-monotonicity rate of raw curves. This sizes the problem and
is the baseline every fix is measured against. (CPU RNN on a few dozen users is enough;
run when the GPU/CPU are free.)

### Stage 1 — time-axis monotonicity by construction (a normal track-1 iter)
Reparameterize the 128 basis curves as cumulative sums of −softplus increments from a free
level (each basis pointwise decreasing in t ⇒ any softmax mixture decreasing). Near-zero
accuracy risk expected; standard gate. Also enables sorting bases by pointwise strength,
which any later structural rating-ordering trick would build on.

**RESOLVED BY REMOVAL (Andrew 2026-07-16 late):** code audit showed the fixed 0.9^(t/s_i)
bases are ALREADY monotone (softmax mixture of decreasing exponentials); the only
non-monotone piece was the learned free residual added via piecewise-LINEAR interp over the
64/128 log-spaced time points. First fix attempt = cummin projection of that residual
(RWKV_MONO_CURVES, built + smoked, never trained). Superseded the same evening by Andrew's
directive: **disable the piecewise correction entirely, both tracks**
(RWKV_NO_AHEAD_RESIDUAL=1 → curve = pure mixture, monotone in t by construction; the
raw-mixture BCE term AHEAD_RAW_SCALE=0.5 already supervises the mixture directly). Iter 22
(redefined, `scratchpad/iter22_nores`) measures the accuracy cost; verdict is Andrew's call.
Time-axis monotonicity needs no reparametrization anymore.

### Stage 2 — counterfactual button-consistency training (the main fix, soft)
Add a training loss term mirroring the deploy computation exactly:
- At each training SEGMENT END, take the full chained-model state — **the stateful WKV
  kernel (built + parity-verified, shelved 2026-07; see [[stateful-bptt-shelved]]) returns
  exactly these boundary states**, the one missing ingredient in the parallel form.
- Append a synthetic next review with each of the 4 ratings (identical imputed
  duration/elapsed across the four — the deploy contract), advance ONE RNN step each,
  read the 4 curves.
- Hinge penalty on pointwise order violations:
  L_mono = λ · Σ_{r<r'} Σ_{128 t-points} relu(P_r(t) − P_{r'}(t) + margin).
- Coverage: thousands of segment ends per batch; cost ≈ 4 one-token steps + 4 head reads
  per point — a few % of step time. Soft ⇒ spends accuracy budget only where data pulls
  against ordering; measured with the standard research gate.
- Elapsed-feature imputation for the synthetic review: reuse the segment's last-review
  elapsed, or sample log-uniform; decide at implementation via the Stage-0 audit's
  violation geography.

### Stage 3 — exact guarantee at inference: isotonic projection AS PART OF THE MODEL
Isotonic-project the 4 curves pointwise (or the 4 intervals) at inference — in the Rust
engine AND our eval harness, so the projected curve IS the model's defined output (not a
scheduler-side hack; kills the wants-vs-gets mismatch by definition). After Stage 2 the
projection should be a rare no-op; Stage 0/2 metrics quantify how rare.

### Rejected-for-now alternative — hard architectural ordering
Route the rating's effect on the IMMEDIATE curve through an ordered scalar bottleneck
(cumsum-softplus rating offsets shifting softmax mass along strength-sorted bases ⇒ exact
stochastic-dominance ordering, zero runtime cost). Downside: severs the rich path from the
just-completed review (the most informative one) to the next curve — the deep state update
would only benefit LATER predictions. Revisit only if Stage 2+3 measurably fail or cost
too much logloss.

## Deploy-contract notes
- Button computation in Anki: append hypothetical review NOW with rating r; duration is
  unknown at display time ⇒ impute ONE value shared by all four buttons (running-mean
  duration or a constant). Iter 18 proved duration is real signal — the imputation choice
  must be part of the frozen deploy contract and mirrored in Stage-2 training.
- Post-lapse UX: Anki may show relearning steps for Again instead of the model interval —
  scheduler-side; unaffected by this plan.

**Status:** recorded 2026-07-16 (during track-2 A2). Stage 1 (time axis) RESOLVED BY
REMOVAL the same evening — Andrew directed the piecewise-linear correction disabled in
both tracks (RWKV_NO_AHEAD_RESIDUAL=1); iter 22 (redefined) measures the cost. Remaining
queue: Stage 0 audit when compute frees up, Stage 2 (rating axis, needs stateful-kernel
wiring), Stage 3 at Rust-deploy time.
