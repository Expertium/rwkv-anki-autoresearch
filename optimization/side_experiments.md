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
