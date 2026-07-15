# RWKV 5k phase

Train **1–5000**, eval **5001–10000** (held-out half); budget **1 WS epoch** + tuned-ratio decay (2→1 on 2026-07-09 via the iter-2 budget A/B — the 2nd epoch adds nothing). Detail & reasoning → [research_5k_notes.md](research_5k_notes.md).

**Two research tracks (Andrew 2026-07-14), separate tables below:** **Track 1** = improve the
small (d=32) model — the table it always was. **Track 2** = ablate the old d=128 model downward;
acceptance = `50,000·(LL_after − LL_before)/(params_before − params_after) ≤ 0.0001` **in BOTH
modes** (≤ 0.0001 logloss degradation per 50k params removed; params must strictly decrease;
"before" = the current track-2 champion). Alternate ~12 h blocks between tracks (~5 track-1 iters
vs 1 track-2 iter per block).

**QAT PARKED (Andrew 2026-07-14, from iter 14 on):** rows ≤ 13 record QUANT-AWARE logloss (q72u);
later rows are PLAIN bf16 — screening is plain-vs-plain in both tracks, and ONE quant-aware run of
the final champion happens when research closes. Plain and QAT-era logloss are NOT comparable
(the plain re-baseline row quantifies the gap). `champion_5k.json` stays = the QAT deploy-truth
champion; the plain screening champion lives in `champion_5k_plain.json` (vprune ref for plain
candidates).

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
| 9 | 1–5000 | 0.3074 | 0.2789 | exact | rejected | 1.0 / 1.0 (vs iter 2) | 193,724 | adopted | Shrink-perturb init (Ash & Adams 2020): worse both modes — early val lead washed out. Init family 0/1, deprioritized. |
| 10 | 1–5000 | 0.3069 | 0.2782 | exact | rejected | 1.0 / 1.0 (vs iter 2) | 193,724 | invented | Warmup KD from d=128 teacher: worse both modes, same arc as iter 9. Early-intervention family 0/2, deprioritized. |
| 11 | 1–5000 | 0.3075 | 0.2788 | exact | rejected | 1.0 / 1.0 (vs iter 2) | 193,852 | invented | Additive grade embedding (4×32 bypass around the input MLP): worse both modes (~0.0009) — the bypass distorts the shared trunk. |
| 12 | 1–5000 | 0.3069 | 0.2781 | exact | rejected | 1.0 / 1.0 (vs iter 2) | 210,236 | invented | SRS-head resolution 64→128 at 5k data: no effect (both ~−0.00025, noise-band) — heads not resolution-limited. |
| 13 | 1–5000 | 0.3068 | 0.2782 | exact | rejected | 1.0 / 1.0 (vs iter 2) | 208,060 | invented | Channel mixer 1.0→1.5: no effect (both ~−0.0002) — capacity-at-5k family 0/2. Last QAT-era iteration. |
| 14 | 1–5000 | 0.3037 | 0.2734 | exact | **accepted** | 0.0 / 0.0 (vs iter 2, info) | 193,724 | invented | **champ5k_plain = PLAIN re-baseline (QAT parked)**: new screening champion; the QAT tax = +0.0029/+0.0044. |

## Track 2 — ablate the old d=128 model

Start = the upstream d=128 arch retrained through the CURRENT track-1 pipeline (plain, 1 ep WS +
0.25 ep decay, **MAX=32768** — the track-2 standard; 66000 thrashes 12 GB at d=128; the upstream
.pth got 12 epochs and is not budget-comparable). `ratio` = `50,000·ΔLL/Δparams` per mode;
**accept iff BOTH ≤ 0.0001**. Current track-2 champion = the highest-A accepted row.
⚠ n=4993: the 1-ep d=128 anchor NaNs on 7 mega-chunk eval users (≥500k-token segments; recorded
in `result/RWKV-track2_a0.nanskip.jsonl`) — all track-2 comparisons run on the finite-user
intersection. Anchor context (intersection-paired): vs upstream 12-ep +0.0037/+0.0044 worse
(the 1-ep budget tax at d=128); vs champ5k_plain (193,724 params) −0.0036/−0.0042 better
(what 2.57M extra params buy at matched budget).

| iter | ahead | imm | status | params | Δparams | ratio a/i (per 50k) | provenance | summary |
|---|---|---|---|---|---|---|---|---|
| A0 | 0.2999 | 0.2690 | anchor | 2,762,884 | — | — (baseline) | adopted | d=128 arch retrained with our 1-ep plain recipe — the track-2 "before" anchor (n=4993, 7 NaN-skips). |
