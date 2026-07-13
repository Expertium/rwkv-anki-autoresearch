# RWKV 5k phase

Train **1–5000**, eval **5001–10000** (held-out half); budget **1 WS epoch** + tuned-ratio decay (2→1 on 2026-07-09 via the iter-2 budget A/B — the 2nd epoch adds nothing). Detail & reasoning → [research_5k_notes.md](research_5k_notes.md).

`iter` = experiment number (chronological). `status` = **accepted** (new champion) or **rejected** —
the **current champion = the highest-iter accepted row**. `p-value` = paired per-user one-sided
Wilcoxon (candidate vs the iter named in parentheses, same 5000 eval users;
`optimization/paired_pvalue.py`), shown `ahead / imm`. **Accept gate (Andrew 2026-07-08): BOTH
modes need p < 0.0001** in addition to the ≥0.0003-both-modes improvement, and **params ≤ 225,000**
(the phase's hard cap; current champion sits at 193,724). `provenance` is binary (Andrew
2026-07-13): **invented** = self-generated (by Claude or Andrew, no external source); **adopted** =
backed by an external source — a paper / GitHub link (e.g. shrink-perturb = Ash & Adams 2020) or a
pre-existing artifact (the upstream d=128 model). `summary` ≤ 20 words (Andrew 2026-07-13) —
full per-iteration notes live in [research_5k_verbose.md](research_5k_verbose.md) (AI-only) and
`research_log.jsonl`.

| iter | trained on | ahead | imm | logloss | status | p-value | params | provenance | summary |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 101–4999 | 0.2964 | 0.2649 | exact | — (target) | — (reference) | 2,762,884 | adopted | Old d=128 leaderboard model, unquantized — the fp target to beat on 5001–10000. |
| 1 | 1–5000 | 0.3066 | 0.2783 | exact | **accepted** | 1.0 / 1.0 (vs iter 0) | 193,724 | invented | champ5k_r1 = first 5k champion (H=2/K=16, q72u quant-aware, 2ep budget). Superseded by iter 2. |
| 2 | 1–5000 | 0.3066 | 0.2779 | exact | **accepted** | 0.31 / 6.1e-62 (vs iter 1) | 193,724 | invented | **champ5k_b1 = CURRENT CHAMPION**: iter 1 at half budget (1ep WS + 0.25ep decay) — 2nd epoch adds nothing. |
| 3 | 1–5000 | 0.3072 | 0.2786 | exact | rejected | 1.0 / 1.0 (vs iter 2) | 193,724 | invented | champ5k_t1 = tuner winner (wd 0.2, dropout 0.5); its 200-user subset win inverted at n=5000. HP tuning closed. |
| 4 | 1–5000 | 0.3069 | 0.2781 | exact | rejected | 1.0 / 1.0 (vs iter 2) | 193,460 | invented | Ladder deck rung: deck H=1 (state 1.89x free) — no effect; deck not state-limited. |
| 5 | 1–5000 | 0.3068 | 0.2783 | exact | rejected | 1.0 / 1.0 (vs iter 2) | 193,526 | invented | Ladder preset rung: preset H=1 — no effect. Ops: parallel eval wedged; sequential-shard rule introduced. |
| 6 | 1–5000 | 0.3063 | 0.2776 | exact | rejected | 1.3e-20 / 1.5e-29 (vs iter 2) | 193,526 | invented | Ladder user rung: user H=1 — first real signal, but imm +0.000258 missed the 0.0003 bar. |
| 7 | 1–5000 | 0.3069 | 0.2773 | exact | rejected | 1.0 / 7.8e-143 (vs iter 2) | 203,928 | invented | User H=1 + 4th layer: mode trade — imm +0.0006 better, ahead −0.0003 worse. |
| 8 | 1–5000 | 0.3067 | 0.2780 | exact | rejected | 0.88 / 1.0 (vs iter 2) | 193,526 | invented | Seed-pair test of iter 6 (seed 4321): NULL — iter 6 was seed luck. Ladder closed, 0/5 rungs. |
| 9 | 1–5000 | 0.3074 | 0.2789 | exact | rejected | 1.0 / 1.0 (vs iter 2) | 193,724 | adopted | Shrink-perturb init (Ash & Adams 2020): worse both modes — early val lead washed out. Init family closed. |
| 10 | 1–5000 | 0.3069 | 0.2782 | exact | rejected | 1.0 / 1.0 (vs iter 2) | 193,724 | invented | Warmup KD from d=128 teacher: worse both modes, same arc as iter 9. Early-intervention family closed. |
