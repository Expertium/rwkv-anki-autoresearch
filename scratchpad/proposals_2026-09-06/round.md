# Proposal round 2026-09-06 -- done IN THE MAIN CONVERSATION (Andrew: "don't use subagents")

Three passes with the protocol's three priors (literature / domain / reject-log steelman), written by
one author after the 2026-09-05 agents were killed by the usage limit three times. Their finished
screens are folded in (numbers in `research_5k_verbose.md` iter 65).

Reference = **realcyc** (gen-5, KD-off): **ahead 0.298083 / imm 0.263592**, n=2,499. Stop criterion
ahead <= 0.2950 (0.0031 away, BINDING) and imm <= 0.2640 (already met, 0.0004 headroom). Accept =
raw >= 0.0001 both modes at p<1e-4 (curve-side: ahead >= 0.0001, imm not significantly worse). Noise
floor +/-7.5e-5. Running now: hord (invented, curve-side), then muonscale (adopted). Next slot after
muonscale = INVENTED.

## The finding that reframes the round: the model is FIT-LIMITED at this budget

iter 65 (SAM) found a 42% flatter minimum and lost in both modes -- with the TRAIN loss rising
alongside the held-out loss. Add the LAWA screen (the WS-checkpoint average is worse than the WS-final
on every window; consecutive displacements are orthogonal), the tuner's dropout x0.5, wd 0.2's loss,
lorawd's tie, and the picture is consistent: at 2 epochs of WSD there is **no generalisation gap for a
regulariser to close**; every lever that trades fit for anything else loses. Muon's edge (iters 29/53)
is fit per step through spectral coverage, not flatness.

Consequences for what can still pay in phase 4:
1. **Fit per step** -- optimizer coverage (closes with muonscale), batch/steps (a budget lever, phase 5).
2. **Where the fit is spent** -- objective alignment with the SCORED distribution (proposal B1, new).
3. **New information** -- the features phase did this; nothing further without a rebuild.
4. **The metric's own structure** -- rectification (hord running), directed trades (Andrew).
Regularisers, label tricks that re-shape the curve logit, routing between heads, extra scopes and
capacity are all closed on measurement.

## Screens run this round (CPU, zero GPU) -- what they killed and what they opened

| screen | result | verdict |
|---|---|---|
| PCGrad premise: trunk cos(g_ahead, g_imm), 6 chunks | +0.046..+0.340, 0 chunks with conflict | **DEAD** (PCGrad is inert without conflict) |
| LAWA: avg of last 2/3/4/6 WS ckpts vs WS-final, 4 chunks | median total +0.0011/+0.0021/+0.0009/+0.0028 WORSE; cos(Δ9→10, Δ10→11) = −0.08 | **DEAD** (kill line: not better by 0.002) |
| cold-grade probe: does the trunk carry the k+1 grade beyond R? | CE(grade\|R) 0.776 vs CE(grade\|R,x) 0.913 | **DEAD** for aux grade heads; explains iter 64 |
| t-bucket logit recalibration, fit on half the train users | in-sample prize +0.00046; held-out **−0.00059** | **DEAD** (fold-unstable, like rank 10) |
| ahead loss by history position (TimeSeriesSplit(5) never scores the first sixth) | first sixth: by-user BCE **0.189**, fail 7.0%; rest: **0.279**, fail 13-15% | **OPENS B1** -- 17% of the labelled rows train on a distribution the metric never scores |
| curve-parameter dump (w,S,d on real rows) | one dominant power law (w 0.87, S ~133 d, d ~0.26) + a 12% slow tail + a 0.2% fast component | the family is not the long-t limit; no lever |
| curve head's own Dropout(0.1) | train-mode cost +0.0013 ahead (user 107) | measured; not a lever |
| chunk-reset cost (continuous vs fresh state at row 16,384; users 102/106) | chunk-mean ahead cost **+0.00014 (102) / +0.00486 (106)**; first 256 rows +0.028/+0.024, recovered by ~1-2k rows on 102; on 106 a PERSISTENT +0.0064 in rows 8k-16k (cards whose earlier reviews sit in the previous chunk restart cold) | **A1 alive** (the recoverable early part), **A2 = the persistent part, Andrew's call** |

## A. Literature prior (adopted)

**A1. Learned initial state per stream ("state tuning", BlinkDL/RWKV-LM; RWKV-infctx-trainer).**
Family: pipeline / state. Training resets every stream to ZERO at each 16,384-row chunk; deploy runs
continuously from one cold start per user. A learned initial WKV state + shift per (stream, layer)
replaces the zeros (5 streams x layers x (H·K·K + shifts) ~ 6.5k params; per-card/per-note STATE
SIZE unchanged, the init is shared). Deploy: the engine starts from the learned vectors instead of
zeros -- one load-time change, no parity risk beyond a fresh trace. **Binding iff the reset-cost
screen shows a cold-start cost above ~0.0005 ahead over a chunk** that recovers within a few
thousand rows (the part a learned init can buy; chunk carry buys the rest). Expected ahead
+0.0000..+0.0003 if binding. 1 run. Gate: both modes.
**A2. Chunk-continuous training (state carry across a user's chunks; TBPTT as in RWKV-infctx-trainer).**
Same screen, the other half: if the cost does NOT recover within the chunk, only carrying the state
recovers it. Multi-day (the shelved STATEFUL_BPTT plan: the batching is by chunk size, so a user's
chunks are not adjacent in the group order) -- **Andrew's call** on the days; not this round's slot.
**A3. Multi-task uncertainty weighting (Kendall, Gal & Cipolla 2018, CVPR).** Learns the ahead/imm
loss weights from homoscedastic uncertainty. It is the principled form of the directed imm-for-ahead
trade (rank 4): imm has exactly 0.0004 of headroom above its stop criterion and ahead is 0.0031
short, so a learned weighting that moves imm by up to 0.0004 while buying ahead is the only trade
with a defined budget. **Directed -- Andrew's call**; if approved, run the FIXED version (imm-scale
0.5, rank 4) first -- one variable, no new dynamics.
**A4. Nesterov momentum warm-up on Muon (modded-nanogpt records; momentum 0.85→0.95 over the first
~300-500 steps).** Fit-per-step, training-only, zero deploy debt. But it is HP-adjacent and the
champion HPs were confirmed against 19 alternatives; expected inside the floor at this budget. **Phase
5**, with QAT on, not a phase-4 slot.
**A5. Data echoing (Choi et al. 2019).** Repeats each fetched batch k times = more optimizer steps
per epoch. In a fit-limited, dispatch-bound step it is a budget lever in disguise; the endgame IS the
budget lever. Note only.

## B. Domain prior (invented)

**B1. Train on what is scored: weight the ahead AND imm losses by `label_is_equalize` (rank 1 of
this round).** The benchmark scores each user's TimeSeriesSplit(5) test folds, i.e. never the first
sixth of a history, and (after the delta_t > 0 rule) not every later row either. Training weights
every labelled row equally. Measured on realcyc's 218,841 train-user rows: the never-scored first
sixth has by-user ahead BCE **0.189 vs 0.279** for the rest, failure rate 7% vs 13-15% -- a different,
easier distribution that takes ~17% of the fit. The flag is IN the train db (`global_labels` col 5:
1-88% of a chunk's rows, near 0 for a user's first chunk), and the repo already uses exactly this
restriction for the PAVA probes (`probe_equalize_only=True`, prepare_batch.py:795), so the change is
a weight on two masks in `_get_loss`: unscored rows x alpha, scored rows x 1. **alpha = 0.25 first**
(alpha 0 makes a user's first chunk a zero-gradient sequence = wasted compute, and the early rows may
carry state-dynamics supervision the later rows need). Train-only, zero deploy debt, ~10 lines,
default alpha 1.0 = byte-identical. Gate: BOTH modes (both metrics score the same rows).
Expected ahead **+0.0000..+0.0003**, imm same sign. Pre-registered abort: either mode worse by
> 0.0002 => the unscored rows are load-bearing supervision; close. Retry: none (a milder alpha only
interpolates). ⚠ Not the iter-37 mechanism: that re-weighted USERS to match the by-user mean; this
selects ROWS to match the scored set, and iter 37's refutation (worse in every size quartile) does
not bear on it.
**B2. Probe the first review** (rank 12; contract change) -- **Andrew's call**, unchanged.
**B3. (rejected while writing) Learning-step rows as a separate objective.** 25% of labelled rows
are same-day (t < 1 h) and carry 26.5% of the ahead loss; but the model is calibrated there
(mean p 0.874 vs success 0.867) and the benchmark scores them, so excluding or down-weighting them
would fight the metric. Dead by arithmetic.

## C. Reject-log steelman

**C1. imm-scale 0.5 (rank 4)** -- steelman of the pbin dose-response: the imm/ahead trade is linear
through zero, and the stop criterion now gives imm a DEFINED budget (0.0004). Expected ahead
+0.00015..+0.00045 for imm −0.0002..−0.0005; the risk is overshooting imm's 0.2640 line. **Directed
-- Andrew.** Screen that supports it: the trunk gradients of the two objectives are nearly
orthogonal (cos +0.05..+0.34), so re-weighting moves the fit rather than being absorbed.
**C2. SAM, again, on the ENDGAME's checkpoints.** iter 65's mechanism is a missing gap, not a
wrong dose; the 10x run is the first place a train-vs-held-out gap can exist. Re-run
`sam_probe.py` + the train/val gap on that checkpoint before deciding; not a phase-4 slot.
**C3. MAX 65536 → 32768 at fixed epochs** -- steelman of iter 34: the measured batch effect
(110000→65536 cost 0.0003, recovered by LR tuning) predicts ~+0.0003 for halving again, at ~1.7x
wall-clock per run (dispatch-bound: 2x steps ≈ 2x time) and it re-bases the cost of every later run.
A **phase-5** decision (it is a budget/HP lever), flagged here because it is one of the few
fit-per-step levers left.
**C4. hord dose follow-ups** -- pending hord (2026-09-06 ~15:00).
**C5. ordcut with a decoupled scale** -- registered and demoted (iter 64); nothing new to add.

## Ranking (expected ahead gain per GPU hour, phase-4-eligible only)

| rank | lever | provenance | gate | expected ahead | cost | state |
|---|---|---|---|---|---|---|
| 1 | **B1 scored-set loss weighting, alpha 0.25** | invented | both | +0.0000..+0.0003 | 1 run | screen DONE (distribution gap measured); **BUILD NOW = the invented slot after muonscale** |
| 2 | A1 learned initial state | adopted (RWKV state tuning) | both | +0.0000..+0.0003 | 1 run + trace | **screen DONE, alive**: the cold start costs +0.024..+0.028 ahead on a chunk's first 256 rows and ~+0.01 over the first 1-2k; the ADOPTED slot after eqw (needs the stateful kernel path + a deploy init: ~1 day of build) |
| 3 | C1 / A3 imm-for-ahead trade | invented / adopted | DIRECTED | +0.00015..+0.00045 | 1 run | **Andrew** |
| 4 | A2 chunk-continuous | adopted (infctx) | both | 0..+0.0003 | multi-day | **Andrew**; screen RUNNING |
| 5 | B2 first-review probes | invented | contract | −0.0001..+0.0002 | re-score + 1 run | **Andrew** |
| -- | A4, A5, C2, C3 | | | | | phase 5/6 notes, not slots |

## Addendum: the chunk-reset screen, and what it says about the METRIC

Training AND eval reset every stream at each 16,384-row chunk boundary; deploy never does. Measured on
realcyc with the deploy RNN (rows 16,384-32,768 of two train users, continuous state vs a fresh
process): the reset costs **+0.00014 (user 102) / +0.00486 (user 106)** ahead over the chunk. The
first 256 rows pay +0.024..+0.028 and the first 1-2k rows ~+0.01 (the part a LEARNED INITIAL STATE
can buy -- A1, now alive), but user 106 also carries a PERSISTENT +0.0064 across rows 8k-16k: cards
whose earlier reviews sit in the previous chunk restart as if new, and only state CARRY (A2) fixes
that. Two consequences for Andrew: (1) A2 (chunk-continuous training) has real evidence now, and its
value on the metric could be ~0.001+ ahead on multi-chunk users -- but only if EVAL carries state
too, which changes how the metric is computed relative to srs-benchmark's chunked RWKV numbers; that
is a methodology decision, not a model lever; (2) A1 is the cheap half and needs no such decision.

### A1 build scoping (read before building; ~1-2 days, not the "~1 day" in the table)

- The CUDA stateful kernel (`rwkv7_wkv_forward_stateful_*`, `rwkv_ops.py:117`) TAKES `state0_BHKK` but
  its backward returns **no gradient for state0** (truncated BPTT by design). A LEARNED init needs
  dL/dstate0 = the backward's carried state-gradient at t=0, which the sequential backward already
  propagates internally -- exposing it is a kernel change (one extra output) + a parity test against
  `reference_rwkv7_stateful` (which is differentiable and can produce the reference gradient).
- The stateful path is used today only for `T > state_clamp_window` streams under the clamp
  (`rwkv_model.py:1095`); every other stream runs `RWKV7_WKV.apply` (the time-parallel path, which
  cannot take an initial state). A learned init therefore moves ALL streams onto the sequential
  kernel: re-measure steps/s first (the 2026-07 profile says the recurrence dominates either way).
- Time-shift states (att + ffn shift, (C,) each) get a learned vector per (stream, layer): trivial.
- Deploy: `rwkv_rnn_model.init_state()` (line 51) starts from the learned tensors; Rust reads two
  extra tensors per layer; a fresh parity trace. Per-card/per-note STATE SIZE unchanged (shared init).
- Params: 13 layer-steps x (5·16·16 + 2·80) ≈ 18.7k (+3.3%).

## The arithmetic Andrew should see (a direction question, not a request)

realcyc ahead 0.2981. The 10x budget is projected at −0.0042 (the 2026-08-11 calibration) → 0.2939,
which would meet 0.2950 -- but QAT costs **+0.0023 ahead** with learnable catalogs → **0.2962**,
0.0012 above the stop criterion. Phase 4's accepted iterations have averaged ~+0.0002 ahead each
(iters 45, 53) and the last five slots were 0/5. So, as things stand, **the criterion is met by the
endgame only if the QAT tax shrinks by ~0.0012 or phase 4 finds ~6 more accepted-size gains**. The
QAT tax is the single largest controllable term left (5x an accepted iteration), and phase 5 is the
first phase that trains with QAT on. That is the fork: keep spending phase-4 slots at ~0/5, or move
to phase 5 early and attack the tax. Both are defensible; it is Andrew's call, and B1 runs either way
because it costs one slot and is the last in-phase lever with a measured, un-run mechanism.
