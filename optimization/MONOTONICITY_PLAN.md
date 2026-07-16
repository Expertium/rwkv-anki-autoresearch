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

### Stage 2 — LEARNABLE ISOTONIC RECTIFIER in the model (Andrew 2026-07-16 late — the
### main fix; supersedes the hinge-loss draft, which is demoted to optional regularizer)
Put the order-enforcing operation INSIDE the model and train through it, so monotonicity
holds by construction at deploy AND the model learns to live with the constraint at
minimal logloss (no train≠deploy mismatch; this also absorbs the old Stage 3 — the deploy
projection IS this same operator).

**Operator (Andrew's design):** PAVA-style pooling where the pooled value is a LEARNABLE
GENERALIZED POWER MEAN, with 3 pair-specific powers — p_AH (Again–Hard), p_HG
(Hard–Good), p_GE (Good–Easy), each p ∈ [−2, 2]:
- At a decision point, evaluate the 4 counterfactual curves at the queried t →
  (P_Again, P_Hard, P_Good, P_Easy), each in (0,1).
- Left-to-right scan; a violating adjacent pair is POOLED — both members take
  M_p(a,b) = ((aᵖ+bᵖ)/2)^(1/p) with that junction's power. Pooling-to-tie is what
  guarantees order (a power mean lies strictly between min and max, so adjusting one side
  alone enforces nothing). Cascade rule for multi-block merges: size-weighted power mean
  using the junction's power; block members all take the merged value.
- p semantics: p→−2 biases the pooled value toward the LOWER curve, p=1 = arithmetic
  (classic PAVA), p→+2 toward the HIGHER — the model learns, per button pair, which side
  to trust when they conflict. 3 scalar params total.
- Gradients: when a violation pools, the BCE gradient of the labeled (actual-rating)
  curve flows into BOTH pooled curves and through the synthetic one-step advances into
  the shared trunk — the model is actively taught to separate curves where data demands
  it and to accept cheap ties where it doesn't; the p_j learn the least-damaging pooling.
- Numerics/parametrization: p_j = 2·tanh(θ_j), init θ = atanh(0.5) → p=1 (exact PAVA at
  start); unified stable form via exp((1/p)·log-mean-exp(p·log x)) with a geometric-mean
  switch near |p| < 1e-3. Fresh-init behavior is benign (all 4 curves near-identical →
  pooling ties near-equal values ≈ identity).
- **Coverage/integration — REDESIGNED 2026-07-16 late (major simplification): IN-SEQUENCE
  PROBE ROWS, no stateful kernel needed.** Verified at the kernel level (rwkv_ops
  reference + CUDA): a SKIP row's output IS its own one-step advance read off the
  updated state — only the state COMMIT is masked (`torch.where(skip, old, next)`). So a
  counterfactual button probe = a skip row with real review-i features, grade one-hot
  swapped to rating r and duration set to the imputed constant, inserted immediately
  BEFORE real row i (reads the post-(i−1) state, advances, output readable, state
  discarded — invisible to every later row). This is EXACTLY the existing imm query-row
  machinery (data_processing.add_queries) with different feature masking — probes carry
  outcome features (swapped) and query-flag 0, queries carry masked outcomes and flag 1.
  Outcome-dependent dims = add_queries' reject list: grade one-hot (4), duration (1),
  card state (1, = col 22, already zeroed by champion recipe). Token-shift invisibility
  via time_shift_selects (the query rows already solve this). Insertion happens at
  PREPARE-BATCH time (on the fly from stored rows — LMDBs untouched, invariant intact).
  Coverage is a DENSITY KNOB (seeded subset of ahead-labeled rows, e.g. 5–10%): cost =
  +3–4 inserted rows per probed row (~+15–40% tokens at 5–10%; full coverage would be
  ~4×, too dear for training but fine for eval/audit). Pool the 4 probe curve values at
  the row's label-elapsed t; λ-weighted BCE on the pooled ACTUAL-rating probe vs the
  row's existing ahead label; counterfactual probes get gradient only through pooling.
  The 4th (pressed) probe uses the imputed duration like the others (mirrors the deploy
  button exactly); the real row's normal ahead loss (real duration, no pooling) is
  untouched. Benchmark eval stays on the committed (real-row) curves — the rectifier
  shapes training and defines deploy BUTTON output; the Stage-0 audit measures ordering.
- **Duration imputation — DECIDED (2026-07-16, Andrew delegated):** ONE value shared by
  all 4 probes — this is causally correct, not just a convention: the duration is the
  time spent on the card BEFORE the press, so it cannot depend on which button gets
  pressed. The value = a GLOBAL CONSTANT (train-set median duration), frozen into the
  deploy contract; simplest possible, zero deploy state, and since it's shared its effect
  on the ORDERING is second-order. Only duration is imputed — the probe row's elapsed and
  all history features are real at both train and deploy. The probe advance for the
  PRESSED rating also uses the imputed duration (it mirrors the deploy button probe); the
  persistent state advance keeps the real duration (both at train and deploy) — the probe
  is a throwaway. The λ-weighted probe BCE (pressed probe vs the row's ahead label) is
  the training signal through the rectifier; the main losses stay untouched. Upgrade path
  if Stage-0 audit / iter-23 shows sensitivity: per-user EMA of durations (one scalar
  carried beside the state; fixed decay in the contract). Build-time checklist: enumerate
  ALL outcome-dependent dims of the 92 (INPUT_FEATURES.md) — rating one-hot (9:13),
  duration, any derived columns — and swap/impute them consistently in probe rows.
- Optional per-pair margin ε if strict (no-tie) button ordering is wanted.
- **Iter-24 extension (Andrew 2026-07-16): weight the pooling mean by the p-head's
  predicted button-press probabilities** (the Instant-mode output at the same decision
  point — available identically at train and deploy): weighted power mean
  M_p(a,b; w) = ((w_a·aᵖ + w_b·bᵖ)/(w_a+w_b))^(1/p), block weight = sum of member
  weights. Per-instance trust weighting — the likely button's curve barely moves, the
  off-distribution counterfactuals absorb the correction; composes with the learned
  powers. Queue order fixed: iter 22 (no-residual) → iter 23 (learnable PAVA, unweighted)
  → iter 24 (+ probability weighting).
- Fallback/regularizer: the original hinge penalty
  L_mono = λ · Σ_{r<r'} Σ_t relu(P_r(t) − P_{r'}(t) + margin) can be added on top purely
  to REDUCE tie frequency, not to enforce (the rectifier already guarantees order).

### Stage 3 — exact guarantee at inference: isotonic projection AS PART OF THE MODEL
**Absorbed into Stage 2 (2026-07-16):** the learnable rectifier IS the inference-time
projection — same operator, same learned powers, implemented in the Rust engine AND our
eval harness, so the rectified curve is the model's defined output (kills the
wants-vs-gets mismatch by definition). After Stage-2 training the pooling should be a
rare near-no-op; Stage 0 metrics quantify how rare. Only remaining stage-3 work = the
Rust port of the operator (a dozen lines) + parity vectors.

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
both tracks (RWKV_NO_AHEAD_RESIDUAL=1); iter 22 (redefined) measures the cost. Stage 2
UPGRADED the same evening to Andrew's learnable power-mean rectifier (in-model, absorbs
stage 3's projection). Remaining queue: Stage 0 audit when compute frees up → Stage-2
rectifier iter (main build = stateful-kernel wiring for segment-boundary states) → Rust
port of the operator at deploy time.
