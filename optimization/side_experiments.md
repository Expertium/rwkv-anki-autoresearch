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

### ★ HONEST GRU RESULT (v3, 2026-07-25 04:06) -- RWKV-7 WINS, BUT ONLY BY ~0.002/~0.003

**GRU v3 (h=128, 1,559,824 params, per-layer pre-norm residuals, verified query-probe
sensitivity), val half 5001-7500, n=2500, 0 nanskips:**

| model | params | ahead | imm | vs A13 RWKV |
|---|---|---|---|---|
| **A13 RWKV-7 (d=128)** | 1,468,724 | **0.298837** | **0.267805** | -- |
| **GRU v3 streams** | 1,559,824 (+6.2%) | 0.300778 | 0.270525 | **+0.001941 / +0.002720 worse** (p=1.0 both) |
| **LSTM v3 streams** | 1,488,688 (+1.4%) | 0.301103 | 0.270973 | **+0.002266 / +0.003169 worse** (p=1.0 both) |
| A14 RWKV-7 (current champ) | 1,380,660 | 0.298798 | 0.267746 | better AND 7-11% smaller than either cell |
| FSRS-7 (full-range ref) | 21 | 0.317933 | -- | all three nets beat it by >0.016 |

**Both classic cells agree** (LSTM v3 done 2026-07-25 10:16, n=2500, 0 nanskips): ordering is
**RWKV-7 > GRU > LSTM**, with the two classic cells only 0.0003/0.0004 apart and both landing
~0.002-0.003 behind RWKV-7. Two independent cell families reproducing the same deficit makes
this a property of the recurrence CLASS, not a quirk of one cell. The LSTM is also the
closest parameter match (+1.4% vs A13) and still loses by slightly more than the GRU.
Training speed went the other way: LSTM 1.74 steps/s > RWKV 1.24 > GRU 1.18 (the LSTM runs
h=92 vs the GRU's h=128), so RWKV-7's win is on accuracy-per-parameter, not raw step rate.
⚠ LSTM caveat: cuDNN hides per-step cell state, so its query probes start from c=0
("fresh-cell readout") -- a small handicap unique to the LSTM; the GRU has no such caveat and
is the cleaner of the two comparisons.

**Interpretation (the answer to Andrew's question).** RWKV-7's recurrence IS worth
something real, but it is worth **~0.002 ahead / ~0.003 imm at matched parameters** -- not
the 0.116/0.148 the v1 bug implied. Context: that margin is ~4-9x the phase's acceptance bar
and roughly the size of ALL recent accepted architecture wins COMBINED (iter 23 PAVA
+0.0003/+0.0001, iter 26 GRU-head N=3 +0.0005/+0.0001, iter 29 Muon +0.0001/+0.0005) -- so
it is not a rounding error, and RWKV-7 also costs FEWER params (A14 beats the GRU while
being 11.5% smaller) and trains slightly faster (1.24 vs 1.18 steps/s, and RWKV pays a
determinism tax the GRU cannot).
**But the bigger share of the leaderboard margin is NOT the recurrence:** a plain GRU with
the same 92-dim features, the same instant/curve heads and the same pipeline already beats
FSRS-7 by ~0.017 (caveat: the FSRS number is full-range 5001-10000, ours is the val half).
So of the ~0.019 total margin over FSRS-7, the shared features/heads/training carry ~0.017
and RWKV-7's specific recurrence adds the last ~0.002.
**Deploy nuance worth remembering:** the GRU's per-entity state is a VECTOR (h floats per
layer) while RWKV-7's is MATRIX-valued (H*K*K per layer, ~32x larger at d=128) -- so if a
classic cell's accuracy deficit were ever closed, it would be the cheaper thing to ship.
The quantization endgame (9 B/card) was built for the RWKV state, so this is a note for a
future deploy-side comparison, not a live proposal.

**⚠ The three v1/v2 tables below are BUG RECORDS, not results** (v1: no query probe at all;
v2: probe correct but no residuals, so it was suppressed 3-10x per layer and imm stayed
blind). Kept because both failure modes are instructive.

**GRU v1 result (BUGGED -- h=128, 1,556,496 params, val half 5001-7500, n=2500, 0 nanskips):**

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

**Caveats (still apply to v3):** HPs (peak_lr 1e-3, wd 0.01, clip 0.25) are RWKV-tuned,
1-epoch budget, no cell-specific tuning, and the GRU gets no token-shift input mix (that is
RWKV machinery). A tuned GRU could plausibly close part of a 0.002-0.003 gap -- so read the
v3 result as "RWKV-7 wins by a small but real margin under RWKV's own recipe", not as a
proof of a floor. ⚠ The v1 sentence that once stood here ("decisively needed... ~0.12-0.15
gap") was an artifact of the interval-blind bug -- deleted, see the v3 section above.

**SE-2 CLOSED 2026-07-25 10:17.** Andrew's question -- "is the complexity RWKV brings to the
table needed?" -- answered: **yes, but it buys ~0.002 ahead / ~0.003 imm at matched
parameters, roughly a tenth of the ~0.019 total margin over FSRS-7.** The other ~0.017 comes
from the shared 92-dim features, the instant/curve heads and the training pipeline, which a
plain GRU or LSTM inherits unchanged. Since RWKV-7 also reaches that accuracy with FEWER
parameters (A14: 1.38M vs 1.49-1.56M) it stays the right choice for the deploy target, and
no track-1/track-2 plan changes. Three engineering lessons banked: (1) query rows are the
pipeline's prediction rows -- any new stream type must condition on them (v1 bug);
(2) residual blocks are what carry that conditioning through depth -- bare stacked cells
attenuate it 3-10x per layer (v2 bug); (3) memory-hungry variants need the eval's
empty_cache interval lowered and mega-batch cuDNN calls chunked (the two eval failures).

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
**v3 EVAL WEDGE + FIX (2026-07-25 00:30):** the v3 eval froze on its 11th user
(~11 GB VRAM reserved, 27 GB host working set, 0% GPU util) — WDDM oversubscription
spilling GPU memory to host RAM, NOT the A9-style fetch race, and NOT co-tenancy (the
FSRS benchmark held only ~0.5 GB; killing our eval left 580 MiB total). Cause: the eval
loop called `torch.cuda.empty_cache()` only every 20 users, and the GRU streams fragment
the caching allocator far faster than a bf16 RWKV eval (fp32 stream weights, per-layer
probe tensors, per-user shape changes). Fix: `RWKV_EVAL_EMPTY_CACHE_EVERY` env in
get_result.py (default 20 = the historical constant → RWKV runs byte-identical), set to 1
for the retry plus `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. The retry resumes
from the 10 scored users. Lesson: memory-hungry ARCH VARIANTS need the eval cache interval
lowered; a frozen process at high VRAM + huge host working set = WDDM paging, and the
tell-tale is 0% GPU util with the allocator near the 12 GB ceiling.

**v3 (relaunched 2026-07-24 ~19:20): pre-norm per-layer residuals x = x + proj(Cell(LN(x)))**
-- the standard attention-vs-RNN ablation skeleton. GRU h=128 -> 1,559,824 params; LSTM
h=104->92 (pays for per-layer projs) -> 1,488,688. LN weights auto-land in the no-decay
optimizer group (dim-based rule). Smoke ALL PASS (stepwise-ref exact <=5e-7, both cells,
windowed + CUDA mega-user + cast paths). Post-run gate: re-run probe_sensitivity_check
on the trained v3 ckpt -- imm must respond to query features before the result counts.
