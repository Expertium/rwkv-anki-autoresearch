# Side experiments (recorded separately from the research loop)

Meme/curiosity runs that are NOT candidates for any champion and do not enter
`research_log.jsonl` or the acceptance-gate tables. Full pipelines, honest evals.

## SE-1 — "Blind RWKV" vs FSRS-7 (2026-07-19, Andrew's directive)

**Question:** train the d=32 model *without* interval-length features and *without*
grades — the two signals every classical SRS algorithm relies on — and see whether a
crippled RWKV can still beat FSRS-7.

**Setup:** `RWKV_ZERO_FEATURES=0,1,2,3,4,5,6,7,9,10,11,12,22` — all six elapsed/interval
features (dims 0–7) + the grade one-hot (9–12) + card state (22, recipe-standard).
Duration, counts, day-cycles, IDs, and everything else kept. Standard 64-basis curve
head, iter-23-era recipe on the current pipeline (1 ep WS + 0.25 ep decay, seed 1234,
MAX=110000, train 1–5000, eval 5001–10000). Forced deviations: vprune OFF (the champion
val ref would false-kill a deliberately crippled model), PAVA/probes OFF (grade probes
are meaningless with grades zeroed), state clamp ON (τ=300 — full-n insurance).
Run dir `scratchpad/meme_blind/` (memebd_1638.pth kept); results
`result/RWKV[-P]-meme_blind.jsonl`, n=5000, 0 NaN-skips. WS 91m, decay 23m, eval 92m.

**Results (users 5001–10000, by-user mean LogLoss, paired on all 5000):**

| Model | LogLoss | vs FSRS-7 | per-user wins vs FSRS-7 |
|---|---|---|---|
| FSRS-7 (`sched_penalties-short-secs-recency`) | **0.317933** | — | — |
| Blind RWKV, ahead mode | 0.351922 | +0.033989 worse | wins 376/5000 (7.5%) |
| Blind RWKV, imm mode | 0.341322 | +0.023389 worse | wins 1,251/5000 (25.0%) |
| (Full RWKV champion iter 25, ahead, for scale) | 0.304427 | −0.013506 better | — |

Wilcoxon (FSRS better): p ≈ 0 in both modes.

**Verdict: no — a blind RWKV cannot beat FSRS-7.** Interval + grade information is worth
~0.048 of ahead LogLoss to RWKV (0.3044 → 0.3519), ~3.5× the full model's entire margin
over FSRS-7 (~0.0135). "Everything else" (duration, activity counts, day-cycles, identity
structure, within-day phase) recovers a surprisingly respectable absolute level — 0.352
ahead / 0.341 imm is far closer to FSRS-7 than to a constant predictor — but it cannot
substitute for the canonical SRS signals.

**Interpretation caveats:** (1) day-resolution intervals remain *partially*
reconstructible from the cycle features (rows 22–28 share a per-batch phase, so day gaps
between a card's appearances are recoverable in principle) and rows 12/13 count activity
since the card's last review — so this measures "no explicit interval/grade signal," not
"no temporal information"; the harsher variant (also zeroing dims 16–17-adjacent cycle
context) would score worse. (2) Grades are truly gone; duration is the only correlate.
(3) The blind model's imm mode beating its own ahead mode by 0.011 (vs ~0.031 for the
full model) shows the ahead task suffers more from blindness — predicting *decay over an
unknown interval* is exactly where the interval features were load-bearing.

## SE-2 -- GRU / LSTM stream baselines: is RWKV-7 needed? (2026-07-23..24, Andrew's directive)

**Question:** replace ONLY the per-stream RWKV-7 stacks with classic GRU/LSTM stacks at
~equal parameters (~1.5M, the track-2 champion scale) -- same 5-stream hierarchy and
depths (card2/deck4/note1/preset3/user3), same 92-dim input FC, same instant/curve heads,
same pipeline/budget/seed -- and measure whether RWKV-7's complexity earns its keep.

**Implementation:** `rwkv/model/rnn_baseline.py` (RWKV_BASELINE_CELL=gru|lstm): per-layer
cuDNN cells, torch-RNG inter-layer dropout, skip-semantics matched to the WKV kernel via
compact-run-scatter (smoke-verified bit-close vs a stepwise reference incl. interior
skips), windowed h-carry for >65k-token users, fp32 weights behind bf16 boundary casts,
(layer,window) gradient checkpointing. Deviations from the RWKV recipe (forced):
RWKV_DETERMINISTIC=0 (cuDNN RNN backward nondet), vprune OFF (cross-arch val ref),
no token-shift input mix (that is RWKV machinery -- classic cells read x_t only).

**GRU result (h=128, 1,556,496 params, val half 5001-7500, n=2500, 0 nanskips):**

| model | ahead | imm | vs A13 (1.469M RWKV) |
|---|---|---|---|
| GRU streams | 0.415110 | 0.415352 | +0.116 / +0.148 WORSE (p=1.0 both) |
| A13 RWKV | 0.298837 | 0.267805 | -- |
| FSRS-7 (ref) | ~0.3179 | -- | GRU loses to FSRS-7 by ~0.10 |
| SE-1 blind RWKV (ref) | 0.3519 | 0.3413 | GRU (with ALL features) loses to BLIND RWKV |

Val trajectory plateaued at ~0.385/0.385 by mid-WS (RWKV: 0.325/0.306); the 0.25-ep decay
barely moved it (0.3854 -> eval 0.415 on the val half). **Striking secondary observation:
GRU ahead == imm to 3 decimals -- the GRU cannot exploit the immediate-prediction
conditioning at all, while RWKV's imm advantage is ~0.031.**

**Training speed (the other half of the question):** on the real group-size mix the GRU
trained ~2x FASTER wall-clock (WS 2.5 h at ~2.5 steps/s vs RWKV d=128's ~4.7 h at ~1.3);
on max-size 32k-token groups it is ~3x SLOWER (0.35 vs 1.15 steps/s) -- classic RNNs pay
sequentially for T, RWKV's chunk-parallel kernel is ~flat. CPU/deploy inference was not
measured (moot given the accuracy).

**Caveats:** HPs (peak_lr 1e-3, wd, clip) are RWKV-tuned, 1-epoch budget, no
GRU-specific tuning -- but the gap (~0.12-0.15) is ~100x the phase's typical effect
sizes and far beyond tuning slack. **Verdict so far: RWKV-7's complexity is decisively
needed -- the recurrence itself (matrix-valued state + decay/gating machinery), not just
the training pipeline, carries the accuracy.** LSTM (h=104, 1,521,360 params) running.

**⚠ v1 RESULTS ABOVE = IMPLEMENTATION BUG (diagnosed 2026-07-24 ~14:00, Andrew's
suspicion confirmed):** the pipeline's skip rows are QUERY rows (one per non-first
review, outcome zeroed, elapsed/interval features KEPT, carrying the labels); the WKV
kernel reads them as x_t-conditioned queries of the un-advanced state, but RNNStream v1
returned the bare predecessor state -- every labeled prediction was made WITHOUT the
elapsed interval. Hence ahead==imm and worse-than-blind-RWKV. **v2 fix: per-layer
UNCOMMITTED one-step probe Cell(x_query, h_prev)** (one extra T=1 cuDNN call per layer,
sync-free; LSTM probes use c=0, a documented fresh-cell caveat since cuDNN hides
per-step c). Smoke-verified vs a corrected stepwise reference. v1 numbers kept above as
the bug record; v2 results replace them as the honest baseline comparison. (The LSTM v1
run was killed mid-WS at the diagnosis -- its WS val plateau matched the GRU's ~0.385.)

**⚠ v2 ALSO KILLED (2026-07-24 ~19:00, step ~7.5k of 22.3k, Andrew's "you sure there
are no other bugs?" audit):** mid-WS val showed ahead==imm AGAIN (~0.385/0.385 at step
7000 where A14 shows a consistent ~0.02 imm advantage), so an end-to-end sensitivity
check was built (`scratchpad/baseline_gru/probe_sensitivity_check.py`, CPU, on the live
step-7000 ckpt): zeroing ALL 92 feature dims of every query row changed the imm
predictions by EXACTLY 0.0 -- the trained model was still interval-blind despite the
mechanically-correct v2 probe (module-level and smoke tests pass; a v1-semantics control
in the same script shows the test discriminates). Stage trace (`probe_diag2.py`): the
query perturbation enters at 10.9 (features2card), exits the card stream at 0.53, then
ATTENUATES ~3-10x PER LAYER through the chain (deck 1.2e-3, note 3.4e-4, preset 1.9e-5,
user 1.1e-6, heads ~0). Root cause: v2 stacked bare cells with NO residual connections;
gates trained for long retention (z->1) suppress one-step probe inputs, and 13
non-residual layers multiply the suppression to nothing. RWKV is immune BY STRUCTURE:
each layer is a pre-norm residual block (x = x + att(ln(x))), so query features ride the
residual stream to the heads at full strength -- an unanticipated, real answer to "what
does RWKV's structure buy": **the residual skeleton is what makes one-step query
conditioning survive depth; a readout that multiplies by r(x_t) can't be gated shut.**
**v3 (relaunched 2026-07-24 ~19:20): pre-norm per-layer residuals x = x + proj(Cell(LN(x)))**
-- the standard attention-vs-RNN ablation skeleton. GRU h=128 -> 1,559,824 params; LSTM
h=104->92 (pays for per-layer projs) -> 1,488,688. LN weights auto-land in the no-decay
optimizer group (dim-based rule). Smoke ALL PASS (stepwise-ref exact <=5e-7, both cells,
windowed + CUDA mega-user + cast paths). Post-run gate: re-run probe_sensitivity_check
on the trained v3 ckpt -- imm must respond to query features before the result counts.
