# RWKV 5k phase

Train **1–5000**, eval **5001–10000** (held-out half); budget **1 WS epoch** + tuned-ratio decay (2→1 on 2026-07-09 via the iter-2 budget A/B — the 2nd epoch adds nothing). Detail & reasoning → [research_5k_notes.md](research_5k_notes.md).

**⚠ VAL/TEST SPLIT (Andrew 2026-07-21, both tracks, from iter 29 / post-A8 onward):** the eval
half is split into **val = users 5001–7500** (every candidate evals ONLY this half; all
accept/reject verdicts + p-gates run on it, n=2500) and **test = 7501–10000** (touched ONLY at
each track's close for the final honest numbers — never for decisions). Candidates gate vs the
champion's existing jsonls via `paired_pvalue --intersect` (pairs restrict to val users
automatically). Rows ≤ iter 28 / ≤ A8 were full-range n=5000; the `eval users` column (track 1)
and n notes mark the switch. Training-time validation (5001–5010) and the tuner range
(5001–6000) already sit inside val — unchanged. Final test numbers get their own small table at
track close.

**Two research tracks (Andrew 2026-07-14), separate tables below:** **Track 1** = improve the
small (d=32) model — the table it always was. **Track 2** = ablate the old d=128 model downward;
acceptance = `100,000·(LL_after − LL_before)/(params_before − params_after) ≤ 0.0001` **in BOTH
modes** (≤ 0.0001 logloss degradation per **100k** params removed — tightened from per-50k by
Andrew 2026-07-15 after A0 landed; params must strictly decrease; "before" = the current track-2
champion). Alternate ~12 h blocks between tracks (~5 track-1 iters vs 1 track-2 iter per block).
Sizing recommendation (Andrew 2026-07-16, soft): **aim for ≥5% param reduction per iteration,
ideally more** — bundle small cuts rather than spending a ~12 h run on <5%.

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
modes need p < 0.0001** in addition to the magnitude bar, and **params ≤ 225,000**.
**Magnitude bar LOOSENED (Andrew 2026-07-19, first applied to iter 26): each mode's improvement
ROUNDED TO 4 DECIMALS must be ≥ 0.0001 (raw ≥ 0.00005) in both modes — was ≥ 0.0003 (iters ≤ 25)**
(the phase's hard cap; current champion sits at 193,724). `provenance` is binary (Andrew
2026-07-13): **invented** = self-generated (by Claude or Andrew, no external source); **adopted** =
backed by an external source — a paper / GitHub link (e.g. shrink-perturb = Ash & Adams 2020) or a
pre-existing artifact (the upstream d=128 model). `summary` ≤ 20 words (Andrew 2026-07-13) —
full per-iteration notes live in [research_5k_verbose.md](research_5k_verbose.md) (AI-only) and
`research_log.jsonl`.

> **★ NUMBERS ASCEND DOWN THIS TABLE, AND ARE ASSIGNED AT VERDICT TIME (Andrew, 2026-08-18).**
> A number is given when the result is recorded, not when the run is queued, so `iter N` means
> *the Nth result* and the table reads as a history of what was known when. Sort by number; there
> are no gaps and no out-of-sequence rows.
>
> **QAT#2 (`qtaxg_i45kd`) was renumbered 56 -> 54** on 2026-08-18 to remove the last gap. That was
> free precisely because its number lived in NO path -- its directory is `scratchpad/qat_tax/` and
> its checkpoints are `qtaxg_i45kd_*`. Every other run has its number baked into a directory and a
> checkpoint prefix, which is why history is otherwise not renumbered.
>
> **NUMBERS ARE ASSIGNED AT VERDICT, IN COMPLETION ORDER -- so this table is BOTH ascending and
> chronological, with no exceptions.** Reserving a number in advance is what broke that, so it is
> no longer done.
>
> ⚠ **52 IS PERMANENTLY VACANT, and it is the one scar of the transition.** It was pre-assigned
> to the `kdalpha` run under the OLD queue-time convention. That run then finished third, after
> QAT#2 (54) and `muonlora` (53), so under the new rule it takes **55**. Giving it 52 anyway would
> have put a lower number below higher ones -- the exact disorder the rule exists to remove. No
> future gap can appear, because numbers are no longer reserved ahead of a verdict.
>
> **⚠⚠ THREE DIRECTORY SLUGS LIE, and the resolved numbers are below (updated 2026-08-19 as each
> verdict landed).** This block previously PREDICTED the remaining numbers, and its predictions were
> wrong -- it had `cmixpow` at 55 in one paragraph and 56 in a table, and `decayshape` at 58. Both
> were assigned on the assumption that the runs would finish in queue order. They did not:
> `decayshape` finished FIRST of the remainder and took **56**, so `cmixpow` took **57**.
> **That is the rule working, not failing** -- and it is also why predicting a number is now
> forbidden. A prediction table is a reservation wearing a different hat.
>
> | run (directory slug) | number | slug agrees? |
> |---|---|---|
> | `iter57_decayshape` | **56** (verdict 04:40, rejected) | **no** |
> | `iter54_cmixpow` | **57** (verdict 10:50, rejected) | **no** |
> | `kdalpha025` | assigned at ITS verdict | n/a -- **no number in its path, by design** |
> | `iter55_rgate` | assigned at ITS verdict | **no** -- 55 is kdalpha |
>
> Trust `exp` in `research_log.jsonl`; the digits in a directory name are not the number. This is
> precisely why the rule forbids putting a number in a run directory or checkpoint prefix --
> `kdalpha025` is the first run named for its lever alone, and it is the only one here that cannot
> acquire a lying slug.

| iter | trained on | ahead | imm | vs old (a / i) | logloss | status | p-value | params | NaN users | provenance | summary |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 101–4999 | 0.2964 | 0.2649 | — (ref) | exact | — (target) | — (reference) | 2,762,884 | 0 | adopted | Old d=128 leaderboard model, unquantized — the fp target to beat on 5001–10000. |
| 0ᵛ | 101–4999 | 0.2946ᵛ | 0.2636ᵛ | — (ref) | exact | — (target) | — (reference) | 2,762,884 | 0 | adopted | Same model restricted to the VAL half (5001–7500, n=2500) — directly diffable against every ᵛ row. Re-verified 2026-08-05: fresh eval reproduces the 2026-07-03 numbers to 1e-6/user (as intended: no rectifier, piecewise residual ON). |
| 1 | 1–5000 | 0.3066 | 0.2783 | +0.0120 / +0.0147 | exact | **accepted** | 1.0 / 1.0 (vs iter 0) | 193,724 | 0 | invented | champ5k_r1 = first 5k champion (H=2/K=16, q72u quant-aware, 2ep budget). Superseded by iter 2. |
| 2 | 1–5000 | 0.3066 | 0.2779 | +0.0120 / +0.0143 | exact | **accepted** | 0.31 / 6.1e-62 (vs iter 1) | 193,724 | 0 | invented | **champ5k_b1 = CURRENT CHAMPION**: iter 1 at half budget (1ep WS + 0.25ep decay) — 2nd epoch adds nothing. |
| 3 | 1–5000 | 0.3072 | 0.2786 | +0.0126 / +0.0150 | exact | rejected | 1.0 / 1.0 (vs iter 2) | 193,724 | 0 | invented | champ5k_t1 = tuner winner (wd 0.2, dropout 0.5); its 200-user subset win inverted at n=5000. HP tuning closed. |
| 4 | 1–5000 | 0.3069 | 0.2781 | +0.0123 / +0.0145 | exact | rejected | 1.0 / 1.0 (vs iter 2) | 193,460 | 0 | invented | Ladder deck rung: deck H=1 (state 1.89x free) — no effect; deck not state-limited. |
| 5 | 1–5000 | 0.3068 | 0.2783 | +0.0122 / +0.0147 | exact | rejected | 1.0 / 1.0 (vs iter 2) | 193,526 | 0 | invented | Ladder preset rung: preset H=1 — no effect. Ops: parallel eval wedged; sequential-shard rule introduced. |
| 6 | 1–5000 | 0.3063 | 0.2776 | +0.0117 / +0.0140 | exact | rejected | 1.3e-20 / 1.5e-29 (vs iter 2) | 193,526 | 0 | invented | Ladder user rung: user H=1 — first real signal, but imm +0.000258 missed the 0.0003 bar. |
| 7 | 1–5000 | 0.3069 | 0.2773 | +0.0123 / +0.0137 | exact | rejected | 1.0 / 7.8e-143 (vs iter 2) | 203,928 | 0 | invented | User H=1 + 4th layer: mode trade — imm +0.0006 better, ahead −0.0003 worse. |
| 8 | 1–5000 | 0.3067 | 0.2780 | +0.0121 / +0.0144 | exact | rejected | 0.88 / 1.0 (vs iter 2) | 193,526 | 0 | invented | Seed-pair test of iter 6 (seed 4321): NULL — iter 6 was seed luck. Ladder closed, 0/5 rungs. |
| 9 | 1–5000 | 0.3074 | 0.2789 | +0.0128 / +0.0153 | exact | rejected | 1.0 / 1.0 (vs iter 2) | 193,724 | 0 | adopted | Shrink-perturb init (Ash & Adams 2020): worse both modes — early val lead washed out. Init family 0/1, deprioritized. |
| 10 | 1–5000 | 0.3069 | 0.2782 | +0.0123 / +0.0146 | exact | rejected | 1.0 / 1.0 (vs iter 2) | 193,724 | 0 | invented | Warmup KD from d=128 teacher: worse both modes, same arc as iter 9. Early-intervention family 0/2, deprioritized. |
| 11 | 1–5000 | 0.3075 | 0.2788 | +0.0129 / +0.0152 | exact | rejected | 1.0 / 1.0 (vs iter 2) | 193,852 | 0 | invented | Additive grade embedding (4×32 bypass around the input MLP): worse both modes (~0.0009) — the bypass distorts the shared trunk. |
| 12 | 1–5000 | 0.3069 | 0.2781 | +0.0123 / +0.0145 | exact | rejected | 1.0 / 1.0 (vs iter 2) | 210,236 | 0 | invented | SRS-head resolution 64→128 at 5k data: no effect (both ~−0.00025, noise-band) — heads not resolution-limited. |
| 13 | 1–5000 | 0.3068 | 0.2782 | +0.0122 / +0.0146 | exact | rejected | 1.0 / 1.0 (vs iter 2) | 208,060 | 0 | invented | Channel mixer 1.0→1.5: no effect (both ~−0.0002) — capacity-at-5k family 0/2. Last QAT-era iteration. |
| 14 | 1–5000 | 0.3037 | 0.2734 | +0.0091 / +0.0098 | exact | **accepted** | 0.0 / 0.0 (vs iter 2, info) | 193,724 | 0 | invented | **champ5k_plain = PLAIN re-baseline (QAT parked)**: new screening champion; the QAT tax = +0.0029/+0.0044. |
| 15 | 1–5000 | 0.3037 | 0.2732 | +0.0091 / +0.0096 | exact | **accepted** | 1.5e-08 / 1.6e-42 (vs iter 14, better) | 193,724 | 0 | adopted | **Drop review-state feature (RWKV_ZERO_FEATURES=22, Andrew's directive)**: not worse, slightly better both modes — new champion; deploy no longer needs Anki review state. |
| 16 | 1–5000 | 0.3037 | 0.2734 | +0.0091 / +0.0098 | exact | rejected | 0.97 / 1.0 (vs iter 15) | 194,780 | 0 | invented | Prehead output gate (zero-init identity): no effect both modes — the shared readout is not gating-limited. Readout family 0/1. |
| 17 | 1–5000 | 0.3039 | 0.2728 | +0.0093 / +0.0092 | exact | rejected | 1.0 / 1.7e-173 (vs iter 15) | 193,724 | 0 | invented | Binary-recall loss term (pbin 0.5): MODE TRADE — imm +0.0004 better, ahead −0.0002 worse. First real plain-era effect; scale-0.25 variant queued. |
| 18 | 1–5000 | 0.3055 | 0.2756 | +0.0109 / +0.0120 | exact | rejected | 1.0 / 1.0 (vs iter 15) | 193,724 | 0 | adopted | Drop review-duration feature (directed, gate ≤0.0003 both): +0.0018/+0.0024 worse — duration is real signal; deploy keeps it. |
| 19 | 1–5000 | 0.3038 | 0.2730 | +0.0092 / +0.0094 | exact | rejected | 1.0 / 1.6e-70 (vs iter 15) | 193,724 | 1 | invented | pbin at 0.25: same mode trade at half amplitude — dose-response linear, no scale can pass both modes. Lever exhausted. |
| 20 | 1–5000 | 0.3035 | 0.2731 | +0.0089 / +0.0095 | exact | rejected | 2.0e-10 / 2.0e-25 (vs iter 15, better) | 194,620 | 0 | invented | Cross-head readout mix (per-channel scalar): BOTH modes better (+0.00018/+0.00011), p-gate passes, magnitudes miss the bar. K×K variant queued. |
| 21 | 1–5000 | 0.3045 | 0.2732 | +0.0099 / +0.0096 | exact | rejected | 1.0 / 0.03 (vs iter 15) | 208,060 | 0 | invented | Cross-head mix v2 (full K×K): ahead −0.0009 worse, imm tied — 16× capacity erased v1's gain. Channel not capacity-limited; no-wd v1 variant queued. |
| 22 | 1–5000 | 0.3045 | 0.2735 | +0.0099 / +0.0099 | exact | **accepted** (re-baseline) | 1.0 / 1.0 (vs iter 15, worse) | 193,724 | 0 | invented | No-residual cost accepted (Andrew): ahead +0.0008/imm +0.0003 = price of monotone-in-t. NEW track-1 reference. |
| 23 | 1–5000 | 0.3042 | 0.2734 | +0.0096 / +0.0098 | exact | **accepted** (directed) | 1.3e-33 / 8.1e-15 (vs iter 22, better) | 193,727 | 0 | invented | Learnable-PAVA rectifier: both modes better (+0.00028/+0.00012); Andrew-accepted for the ordered-buttons constraint itself, not logloss. NEW champion. |
| 24 | 1–5000 | 0.3042 | 0.2734 | +0.0096 / +0.0098 | exact | rejected | 0.54 / 0.03 (vs iter 23) | 193,727 | 0 | invented | p-head pooling weights: null vs iter 23 (+0.00004/+0.000002) — uniform suffices. Bonus: PAVA effect reproduced vs iter 22 (+0.00031/+0.00012). |
| 25 | 1–5000 | 0.3044 | 0.2734 | +0.0098 / +0.0098 | exact | **accepted** (size exception) | 1.0 / 0.38 (vs iter 23) | 171,066 | 0 | adopted | GRU power-curve head at d=32: parity inside the budget at −11.7% params — Andrew-accepted as a size win. NEW champion; both tracks now share the GRU head. |
| 26 | 1–5000 | 0.3039 | 0.2734 | +0.0093 / +0.0098 | exact | **accepted** (new gate) | 4.4e-42 / 4.8e-09 (vs iter 25, better) | 171,453 | 0 | adopted | GRU N=3: ahead +0.00049 = LARGEST ahead gain of the phase; imm +0.00009 → 0.0001 under the NEW rounded-4dp ≥0.0001 gate. NEW champion. |
| 27 | 1–5000 | 0.3044 | 0.2735 | +0.0098 / +0.0099 | exact | rejected | 1.0 / 1.0 (vs iter 26) | 171,840 | 0 | adopted | GRU N=4: ahead −0.0004 / imm −0.0002 worse than N=3 — the N-sweep peaks at 3, closed. Val-parity again lost eval. |
| 28 | 1–5000 | 0.3041 | 0.2735 | +0.0095 / +0.0099 | exact | rejected | 1.0 / 1.0 (vs iter 26) | 172,349 | 0 | invented | xhead v1 re-bench: iter 20's gain did NOT transfer to the GRU-N=3 recipe (both modes worse). v3 rationale inverted → family closed. |
| — | 101–4999 | **0.2946ᵛ** | **0.2636ᵛ** | — (ref) | exact | — (target) | — (reference) | 2,762,884 | 0 | adopted | **The old d=128 model on the VAL half** (= row 0ᵛ, repeated here as the progress yardstick for the ᵛ rows below). Unrectified, residual ON. |
| 29 | 1–5000 | 0.3020ᵛ | 0.2714ᵛ | +0.0074 / +0.0078 | exact | **accepted** | 2.5e-06 / 6.5e-71 (vs iter 26, better) | 171,453 | 0 | adopted | Hybrid Muon+AdamW: ahead +0.00014, imm +0.00049 (largest imm gain of the phase) — first optimizer-family win. ᵛ = VAL half 5001–7500 (first val-split row; absolute values not comparable to rows ≤28). NEW champion. |
| 30 | 1–5000 | 0.3024ᵛ | 0.2713ᵛ | +0.0078 / +0.0077 | exact | rejected | 1.0 / 4.2e-11 (vs iter 29) | 171,453 | 0 | adopted | Cautious wd on the Muon groups: pure trade — imm +0.00014 better, ahead −0.00038 worse. Optimizer family 1/2; iter 29 stands. |
| 31 | 1–5000 | 0.2989ᵛ | 0.2676ᵛ | +0.0043 / +0.0040 | exact | **accepted** | 6.0e-26 / 1.5e-209 (vs A18) | 558,212 | 0 | invented | Graft track-1's three wins (PAVA + GRU N=3 + Muon) onto the A18 trunk: ahead +0.00039, imm +0.00075, both clear. First merged-lineage iter. Bundle — does not attribute. |
| 32 | 1–5000 | 0.2983ᵛ | 0.2672ᵛ | +0.0037 / +0.0036 | exact | **accepted** | 2.3e-66 / 3.1e-143 (vs iter 31) | 558,212 | 0 | adopted | Full-run distillation from the d=128 teacher: ahead +0.00058, imm +0.00043. Closes 13% of the ahead gap / 11% of imm to the teacher. KD costs ~9% wall-clock. CHAMPION — rectified 0.3003/0.2673, the baseline iter 33 was gated against. |
| 33 | 1–5000 | 0.3031ᵛʳ | 0.2681ᵛʳ | +0.0085 / +0.0045 | exact | rejected | 1.0 / 1.0 (vs iter 32 RECT) | 558,212 | 0 | invented | Withhold the current row's duration from its own ahead prediction (deploy-contract). Both modes worse. ⚠ 3 changes bundled — cannot attribute. |
| 34 | 1–5000 | 0.2990ᵛʳ | 0.2662ᵛʳ | +0.0044 / +0.0026 | exact | **accepted** | 1.8e-152 / ~0 (vs iter 32 RECT) | 558,212 | 0 | invented | MAX=65536 tuner: Muon LR ÷8 (+0.00183) + decay_ratio 1.0 (+0.00145) + dropout ×0.5. ahead +0.00130, imm +0.00104 — and 1.68× faster training. NEW champion. |
| 35 | 1–5000 | 0.2988ᵛʳ | 0.2659ᵛʳ | +0.0042 / +0.0023 | exact | **accepted** | 5.9e-11 / 7.9e-71 (vs iter 34) | 558,212 | 0 | adopted | Seed pair at 4321: KD confirmed at 2nd seed (+0.00016/+0.00025 within-seed); tuned+KD beats iter 34; iter-34 recipe seed-robust to ~2e-5. NEW champion. |
| 36 | 1–5000 | 0.2983ᵛʳ | 0.2660ᵛʳ | +0.0037 / +0.0024 | exact | **accepted** (directed) | 5.1e-67 / 1.0 (vs iter 35) | 558,212 | 0 | invented | PAVA λ→0.2: ahead +0.00048 for imm −0.00008 (5.9:1). Gate failed on imm; Andrew took the trade. NEW champion; deploy λ is now 0.2. |
| 37 | 1–5000 | 0.2986ᵛʳ | 0.2661ᵛʳ | +0.0040 / +0.0025 | exact | rejected | 1.0 / 1.0 (vs iter 36) | 558,212 | 0 | invented | By-user loss weighting (1/Nᵤ, 4308× imbalance): worse in EVERY size quartile incl. the small users it targeted — mechanism refuted, not underdosed. Objective-alignment 0/1. |
| 38 | 1–5000 | 0.2982ᵛʳ | 0.2660ᵛʳ | +0.0036 / +0.0024 | exact | rejected | 3.9e-06 / 8.6e-07 (vs iter 36, better) | 558,212 | 0 | adopted | KD α 0.5→0.75: BOTH modes better, p-gate passes — but imm +0.000048 misses the ≥0.00005 bar by 2e-6. Nearest miss of the phase; α=0.9 running per conduct rule 2. |
| 39 | 1–5000 | 0.2982ᵛʳ | 0.2659ᵛʳ | +0.0036 / +0.0023 | exact | **accepted** | 2.2e-10 / 7.8e-37 (vs iter 36) | 558,212 | 0 | adopted | KD α→0.9: ahead +0.000158, imm +0.000153 — clean full-gate pass; dose curve monotone up, imm accelerating. Moots iter 38's near-miss (0.9 dominates 0.75). NEW champion; recipe α is now 0.9. |
| 40 | 1–5000 | 0.2982ᵛʳ | 0.2659ᵛʳ | +0.0036 / +0.0023 | exact | rejected | 0.71 / 1.0 (vs iter 39) | 558,212 | 0 | adopted | KD α→1.0 (pure-teacher WS): ahead flat, imm −0.000067 worse — peak BRACKETED at ~0.9, lever closed. The 10% hard labels in WS still carry signal. |
| 41 | 1–5000 | 0.2979ᵛʳ | 0.2655ᵛʳ | +0.0033 / +0.0019 | exact | **accepted** | 5.1e-24 / 7.5e-95 (vs iter 39) | 558,212 | 0 | invented | Interleave (round-robin layers across scopes) + fine-to-coarse order bundle: ahead +0.00029, imm +0.00040 — the largest both-modes architectural gain of the phase. Topology family opens 1/1. NEW champion. |
| 42 | 1–5000 | 0.2984ᵛʳ | 0.2661ᵛʳ | +0.0038 / +0.0025 | exact | rejected | 1.0 / 1.0 (vs iter 41) | 558,212 | 0 | invented | De-bundle control: fine-to-coarse ORDER alone, sequential. Worse than iter 41 (−0.00049/−0.00061) AND than iter 39's old order (−0.00020/−0.00022) — the order is a small NEGATIVE; INTERLEAVING carries the whole iter-41 gain. Rust needs the interleave port. |
| 43 | 1–5000 | 0.2980ᵛʳ | 0.2655ᵛʳ | +0.0034 / +0.0019 | exact | rejected (tie) | 0.42 / 0.098 (vs iter 41) | 558,212 | 0 | invented | The 2×2's 4th cell: interleave at the ORIGINAL order. Statistical TIE with the champion in both modes — under interleaving, within-round order stops mattering. Order lever CLOSED; the reorder could be dropped for Rust-port simplicity at zero measured cost. |
| 44 | 1–5000 | 0.2979ᵛʳ | 0.2655ᵛʳ | +0.0033 / +0.0019 | exact | rejected (tie) | 0.93 / 1.0e-4 (vs iter 41) | 558,212 | 0 | invented | Endpoint-anchored layer PLACEMENT (RWKV_ILV_SPREAD) so shallow streams also run LATE. Statistical tie; prediction of a gain was WRONG. Third indistinguishable schedule (spread ≤7.5e-5) → rearrangement sub-family EXHAUSTED, and that spread calibrated the accept bar. |
| 45 | 1–5000 | 0.2977ᵛʳ | 0.2654ᵛʳ | +0.0031 / +0.0018 | exact | **accepted** | 3.9e-47 / 1.4e-82 (vs iter 41) | 558,212 | 0 | invented | KD kept through the DECAY phase (alpha 0.9 WS → 0.5 decay), zero code. ahead +0.000192, imm +0.000104. WS trace IDENTICAL to iter 41 for all 10,935 steps, so the gain is decay-only. Distillation 4/4. NEW champion. |
| 46 | 1–5000 | 0.2977ᵛʳ | 0.2654ᵛʳ | +0.0031 / +0.0018 | exact | rejected (tie) | 0.996 / 0.014 (vs iter 45) | 558,212 | 0 | invented | Privileged self-distillation imm→ahead (beta 0.7, teacher detached). Null: both modes inside the noise floor. The teacher shares the trunk and forward pass, so it is a re-expression of what the student already computes — the 0.032 gap is not transferable by soft targets. |
| 47 | 1–5000 | 0.3000ᵠ | 0.2690ᵠ | +0.0054 / +0.0055 | exact | rejected | 1.0 / 1.0 (vs cblearn) | 558,212 | 0 | invented | Rank-1-friendly regulariser (RWKV_QAT_RANK1_REG=0.05) on the WKV state. **ᵠ = QUANT-AWARE eval basis — NOT comparable to the plain rows above; compared only against its twin qtaxd_cblearn.** ahead −0.000035 (inside noise), imm −0.000180 (a small real regression). Cut the exact rank-1 truncation error 43% card / 75% note and the deployed logloss did not improve → the reconstruction ladder's ranking of rank-1 as the largest term does not survive as a logloss ranking. Andrew's objection confirmed; the CONTROL drifts −24.4% note toward rank-1 unaided. |
| 48 | 1–5000 | 0.2977ᵛʳ | 0.2654ᵛʳ | +0.0031 / +0.0018 | exact | rejected (tie) | 0.19 / 0.37 (vs iter 45) | 558,216 | 0 | invented | Retrievability-coupled rating head: curve logit R(t) added to the 4 rating logits (zero-init). Exact tie (+9e-6/+1.3e-5). The coupling WAS learned and sign-correct on Again but tiny — the trunk already carries retrievability. With iter 46, closes the ahead-vs-imm-gap family: routing is not the deficiency. |
| 49 | 1–5000 | 0.2976ᵛʳ | 0.2653ᵛʳ | +0.0031 / +0.0018 | exact | rejected | 0.11 / 5.3e-16 (vs iter 45) | 584,282 | 0 | invented | Restore the user/preset LAYER-0 channel mixers (+26,070 params, +4.7%). Both modes improve but both miss the raw ≥0.0001 bar: ahead +0.000067 at p=0.11 is inside the ±7.5e-5 noise floor (a coin flip), imm +0.000087 is real by rank but sub-threshold. Capacity at the general streams' ENTRY layer is not the bottleneck — capacity-at-5k goes 0/3. |
| 50 | 1–5000 | 0.2977ᵛʳ | 0.2654ᵛʳ | +0.0031 / +0.0018 | exact | rejected (tie) | 0.52 / 0.86 (vs iter 45) | 558,292 | 0 | **Andrew** | THE DECK TREE (L=2): the deck stream runs a second time over reviews grouped by the deck's PARENT, same module object + an 80-float level embedding. Exact tie (+7e-6 / −2.4e-5). The embedding WAS learned (L2=1.77, ~2× a features2card row) — the model used the parent level and gained nothing. The 5-stream hierarchy already brackets that scope: deck below, preset/user above. |
| 51 | 1–5000 | — | — | — | n/a | **failed (NaN)** | n/a | 558,212 | n/a (3,684 skipped batches) | invented | Polar-Express Newton-Schulz schedule for Muon. Died hollow at step 411: production `a+b+c`=0.7010 makes p(1)=0.70 a CONTRACTION, and a thin rank-1 momentum matrix sits at σ_max≈1.0012 in bf16. Accuracy means p(1)→1, which diverges there. Closed on mechanism; flag raises at import. |
| **53** | 1–5000 | **0.2975**ᵛʳ | **0.2652**ᵛʳ | +0.0029 / +0.0016 | exact | **ACCEPTED — NEW CHAMPION** | 3.5e-08 / 2.7e-54 (vs iter 45) | 558,212 | 0 | invented | Muon on the 27,520 LoRA params (4.9%), own wd=0 group. Regularizer signature: no train-loss edge on ahead, real held-out gain. |
| 54 | 1-5000 | 0.2999ᵠ | 0.2689ᵠ | +0.0053 / +0.0054 | exact | rejected (tie) | 6.2e-04 / 1.0 (vs cblearn) | 558,212 | 0 | invented | QAT#2 - KD teacher swapped from the d=128 dump to our own plain iter-45 champion, quant-aware decay. **ᵠ = QUANT-AWARE basis; comparable only to its twin qtaxd_cblearn.** Exact tie (+8.4e-5 / -7.0e-5, both inside the +/-7.5e-5 floor). Predicted that morning by a minutes-of-CPU screen: the two teachers agree at r=0.9460 because iter 45 IS the d=128 teacher's own student. The QAT tax does not live in the teacher. |
| 55 | 1–5000 | 0.2977ᵛʳ | 0.2655ᵛʳ | +0.0031 / +0.0019 | exact | rejected | 1.0 / 1.0 (vs iter 45) | 558,212 | 0 | invented | KD alpha_decay 0.5→0.9. Regresses. alpha=0.9 wins in WS but loses in decay — confirms the pre-registered KD-calibration cost. |
| 56 | 1–5000 | 0.2976ᵛʳ | 0.2653ᵛʳ | +0.0030 / +0.0017 | exact | rejected | 0.985 / 1.0 (vs iter 53) | 558,212 | 0 | invented | Linear LR decay shape. Real but sub-bar vs iter 45; loses to iter 53. Even perfect stacking fails. |
| 57 | 1-5000 | 0.2977 | 0.2654 | +0.0031 / +0.0018 | exact | rejected (tie) | 0.99 / 1.0 (vs iter 53) | 558,225 | 0 | invented | Learnable channel-mixer exponent. Exact tie vs iter 45. All 4 live exponents moved 2.0->1.26-1.86 and bought nothing. |
| 58 | 1-5000 | 0.2978 | 0.2654 | +0.0032 / +0.0018 | exact | rejected | 1.0 / 1.0 (vs iter 53) | 558,212 | 0 | invented | KD alpha_decay 0.5->0.25. Loses, as 0.9 did. 0.5 is an interior optimum; lever CLOSED. |
| 59 | 1-5000 | 0.2977 | 0.2654 | +0.0031 / +0.0019 | exact | rejected (tie) | 1.0 / 1.0 (vs iter 53) | 558,536 | 0 | invented | FSRS retrievability gate on the delta rule. Exact tie. Gain learned NEGATIVE on L0 -- opposite the FSRS sign. |
| 60 | 1-5000 | 0.3002 | 0.2686 | +0.0056 / +0.0050 | exact | rejected (ratio) | 1.0 / 1.0 (vs iter 53) | 84,007 | 0 | Andrew | Hybrid arm A: pure shrink to d=32, 84k params. Ratio 5.7x/7.1x over bar. Parameter-efficiency curve has a knee below 558k. |
| 61 | 1-5000 | 0.3004 | 0.2688 | +0.0058 / +0.0052 | exact | rejected (ratio) | 1.0 / 1.0 (vs iter 53) | 100,263 | 0 | Andrew | Hybrid arm B: feature MLP doubled at fixed depths. WORSE than arm A in both modes -- widening the feature pathway does not pay. |
| featA2 | 1-5000 | 0.2982ᵛʳ | 0.2656ᵛʳ | +0.0036 / +0.0020 | exact | control (not a candidate) | 4.6e-13 / 2.3e-27 (vs featA, BETTER) | 558,212 | 0 | Andrew | Features-A/B control re-run on the id-fixed published dbs. Same recipe, seed and KD-off env as featA; only the dbs differ. **⚠ RETRACTED 2026-09-01 — featA's db was built 2026-07-03 and featA2's 2026-08-21, straddling the 08-19 sentinel-cumsum fix, so +0.000148 / +0.000169 is Bug A PLUS a global input change, NOT the id fix. Do not price the rebuild on it; the id fixes' accuracy value is UNMEASURED (they stay justified on correctness).** Originally read as — larger than iters 39, 45 or 53 individually. The champion lineage trained on the unfixed db, so that gain is unclaimed. KD-OFF, so NOT comparable to iter 53. |
| **e2sc** | 1-5000 | **0.2979**ᵉ | **0.2657**ᵉ | n/a (basis) | exact | **BASELINE — the new reference** | 1.0 / 1.0 (vs iter 53) | 558,212 | 0 | Andrew | **ᵉ = END-TO-START interval basis (Andrew 2026-08-30: "e2s should be used both in train AND eval. That should be the new default for all future runs"). NOT comparable to any row above, all of which are end-to-END.** iter 53's recipe, byte-identical except three db paths; KD teacher dump regenerated on the e2s batch stream (the old dump's `labels_sum` checksum cannot see an input-side change). Closes a train/deploy divergence: a live Anki scheduler computes `now() − last_review_time`, which is end-to-start and structurally cannot be otherwise. ⚠ The −0.000366 / −0.000484 vs iter 53 BUNDLES three changes — the interval plus the Bug A and Bug C fixes, both of which should HELP — so the interval's own cost is LARGER. **The `fixc` arm has now isolated it: +0.000225 / +0.000400** (see the next row). Size gate 0/2500. |
| **fixc** | 1-5000 | 0.2977ᵛʳ | 0.2653ᵛʳ | +0.0031 / +0.0017 | exact | control (interval isolation) | 1.0 / 1.0 (vs iter 53) | 558,212 | 0 | Andrew | **THE END-TO-END CONTROL FOR e2sc — same id-fixed rebuild, end-to-start switched off, everything else byte-identical. => END-TO-START COSTS +0.000225 ahead / +0.000400 imm (p=7.4e-31 / 1.1e-116, n=2500).** That is the SIZE OF THE CORRECTION, not a rejection: deploy computes `now() − last_review_time` whatever we train on, so every end-to-end row above — iter 53 included — is optimistic by about this much as a DEPLOY estimate. All three predictions in `e2s/PREREG.md` confirmed: both modes in the 0.0001–0.0005 band; imm degrades MORE than ahead (1.78×, the one expected to be wrong — and the signature of a leak about the CURRENT review, which is what imm predicts); concentrated in same-day rows (top vs bottom quartile 6.6× on ahead). Single-variable pair ASSERTED in phase 0b: identical entry counts, all five id streams byte-identical, card_features differing on 8.420% of entries — that third condition is the anti-false-green check. ⚠ fixc is +0.000141 / +0.000084 WORSE than iter 53, but that is a **CROSS-GENERATION artifact, resolved 2026-09-01**: iter 53's db was built 2026-07-03 and fixc's 2026-08-31, straddling the 08-19 sentinel-cumsum fix (a GLOBAL input change). Confirmed by concentration — the gap is NOT where the id fixes act (Spearman rho -0.019; imm runs the wrong way; the 621 users with <0.5% NaN-note reviews show it anyway). |
| **featB** | 1-5000 | 0.2979ᵛʳᶠ | **0.2632**ᵛʳᶠ | n/a (own lineage) | **22.2% differ** | **treatment — the features WIN** | 1.6e-17 / 6.6e-302 (vs featA2) | 565,252 | 0 | Andrew | **ᶠ = the 23 REAL-TIMESTAMP FEATURES, `-id` gen-3 dbs, input 92→114. vs featA2: +0.000303 ahead / +0.002371 imm.** The imm gain is ~8x the ahead gain and ~13x iter 53's — the largest single move of the phase. Features-only (adding back the e2s penalty featB pays in its own bundle): ~+0.00057 / +0.00273. **Pre-registered before any number existed:** P3 confirmed (554/2500 users differ in `size`, predicted 20-40%) and the size-matched subset (n=1946) AGREES IN SIGN (+0.000347/+0.002330), so the dataset swap is not doing the work; P1 direction right but the ahead gain fell BELOW the predicted band; **P2 REFUTED and that is the finding — the gain is NOT concentrated in same-day users** (top/bottom 1.36x/1.46x vs a 2x bar, rho +0.012/+0.172), so it is not the fine-grained clock columns but the always-defined ones (age, tenure, creation batch). ⚠ NOT a champion candidate: gate #1 fails by construction and gen 3 still carries Bug C — adoption re-bases on gen 4. |
| **gen4base** | 1-5000 | 0.2981ᵍ⁴ | **0.2635**ᵍ⁴ | +0.0035 / −0.0000 | n/a (new lineage) | **BASELINE — the gen-4 features lineage** | n/a | 565,252 | 0 | Andrew | **ᵍ⁴ = featB's exact recipe (KD off) on the GENERATION-4 `-id` dbs: Bug C fixed AND the equalize set re-selected with `delta_t > 0` on end-to-start gaps (`label_filter_db_id_e2s`). n=2,499 — user 6701 EXCLUDED (four identical OOMs at 36.09 GiB, the WDDM ceiling). The reference for realcyc / lorawd / the LOO drops; size baseline snapshotted (`size_baseline_id_e2s.json`, 126,657,015 scored reviews). Informational vs featB, NOT a gate (two-variable bundle, 2,156/2,499 users differ in size): −0.000191 / −0.000306 raw; size-identical n=343 +0.000130 / −0.000155. The nominal loss is the direction the label filter predicts (it drops ~0.19% of rows that are 1.46× easier), so Bug C's own value stays unmeasured. vs the old d=128 model on the VAL half: imm sits AT its number (−0.000013) with a 4.95× smaller trunk and KD off; ahead is 0.0031 from the 0.2950 stop criterion and is the binding mode. |

**`vs old (a / i)` = how far this row still is from the OLD d=128 model** (Andrew's ask, 2026-08-12). `row - baseline` for ahead / imm, so **positive = still worse, negative = we have beaten it**. Baseline = `pretrain/RWKV_trained_on_101_4999.pth` unquantized, restricted to the **VAL half 5001-7500** (the only set candidates are scored on) = **0.294612 / 0.263561**; its full-range 5001-10000 numbers (0.296385 / 0.264905) are a different user set and are not used here, which is why the reference row itself reads `-- (ref)`.

⚠ **The ahead side of this column is PESSIMISTIC.** The baseline predates the rectified gate, while rows from iter 33 on are RECTIFIED (the deploy-honest metric, `vr`). Rectification costs `ahead` ~+0.0019..0.0036 depending on the model -- a price the baseline never paid -- so the real ahead gap is smaller than shown by roughly that much. `imm` is closer to like-for-like (the rectifier does not touch the rating head), up to ~0.0003 of probe-insertion noise. **This column is a progress indicator, never a gate** -- the gate is always vs the current champion. Regenerate with `python optimization/vs_old_column.py` after adding a row.

## Track 2 — ablate the old d=128 model

Start = the upstream d=128 arch retrained through the CURRENT track-1 pipeline (plain, 1 ep WS +
0.25 ep decay, **MAX=32768** — the track-2 standard; 66000 thrashes 12 GB at d=128; the upstream
.pth got 12 epochs and is not budget-comparable). `ratio` = `100,000·ΔLL/Δparams` per mode,
ΔLL = candidate − champion, so **NEGATIVE ratio = candidate BETTER** (summaries quote raw
deltas in the opposite, improvement-positive convention); **accept iff BOTH ≤ 0.0001** (Andrew 2026-07-15, tightened from per-50k: the A0-vs-champ5k_plain
collapse itself costs 0.000074/0.000086 per 50k — the old bar would accept ablations no better
than the collapse average; the per-100k bar demands ~1.5–1.7× better). Current track-2 champion
= the highest-A accepted row.
⚠ n=4993: the 1-ep d=128 anchor NaNs on 7 mega-chunk eval users (≥500k-token segments; recorded
in `result/RWKV-track2_a0.nanskip.jsonl`) — all track-2 comparisons run on the finite-user
intersection. Anchor context (intersection-paired): vs upstream 12-ep +0.0037/+0.0044 worse
(the 1-ep budget tax at d=128); vs champ5k_plain (193,724 params) −0.0036/−0.0042 better
(what 2.57M extra params buy at matched budget).
**Since A4 (2026-07-18) the reference = the no-residual re-anchor (0 NaN-skips): future
comparisons pair on full n=5000, and every track-2 run is no-residual (mandatory recipe).**

| iter | ahead | imm | status | params | Δparams | ratio ahead/imm (per 100k) | NaN users | provenance | summary |
|---|---|---|---|---|---|---|---|---|---|
| A0 | 0.2999 | 0.2690 | anchor | 2,762,884 | — | — (baseline) | 7 | adopted | d=128 arch retrained with our 1-ep plain recipe — the track-2 "before" anchor (n=4993, 7 NaN-skips). |
| A1 | 0.2998 | 0.2691 | **accepted** | 2,320,516 | −442,368 | −0.00002 / +0.00001 | 0 | invented | All channel mixers → 1.0: ahead better, imm +0.00004 — ~50× inside the gate. Zero NaN-skips (A0: 7). New track-2 champion. |
| A2 | 0.3002 | 0.2693 | rejected | 2,204,412 | -116,104 | +0.000155 / +0.000017 | 0 | invented | Deck 4L->3L: ahead +0.00018 worse = 1.55x the per-100k bar (imm passes). Deck depth is load-bearing for ahead. |
| A3 | 0.3000 | 0.2684 | rejected (unstable; re-gate vs A4 PASSED) | 2,126,224 | -194,292 | +0.000228 / -0.000054 | 129 | Andrew | GRU curve head: imm BETTER p=2e-21 (first t2 accuracy win); ahead confounded by residual removal; 129 NaN users. |
| A4 | 0.3005 | 0.2693 | **accepted** (re-baseline) | 2,320,516 | 0 | — (re-anchor) | 0 | directed | No-residual re-anchor of A1: ahead +0.0005 = d=128 residual price, imm improves. New reference; A3 re-gate PASSES but unstable. |
| A5 | 0.3005 | 0.2691 | **accepted** | 2,115,359 | −205,157 | +0.000014 / −0.000066 | 0 | invented | GRU head + L0-v_lora strip + state clamp: imm +0.00014 BETTER (p=4e-38), ahead noise; clamp → 0 NaN-skips; WS 1.67× faster. New champion. |
| A6 | 0.3004 | 0.2692 | **accepted** | 1,949,624 | −165,735 | −0.000062 / +0.000066 | 0 | invented | Strip 5 bottom-saliency channel mixers (RWKV_STRIP_CMIX): ahead +0.0001 BETTER, imm price 1.5× inside the bar. New champion, −16% vs A4. |
| A7 | 0.3004 | 0.2690 | **accepted** | 1,767,226 | −182,398 | −0.000035 / −0.000148 | 0 | invented | user 4L→3L + note.L1/deck.L2 mixer strips: BETTER both modes (imm +0.00027, p=9e-118!) — user depth was pure fat. New champion, −26% vs A4. |
| A8 | 0.3004 | 0.2690 | **accepted** | 1,617,975 | −149,251 | +0.000010 / +0.000027 | 0 | invented | card 3L→2L + card.L1 mixer strip: ~zero accuracy cost (10×/3.7× inside the bar), −41% vs 2.76M; per-card state −1/3. Training-val NaN transients (contained, eval clean) = A9 watch item. New champion. |
| A9 | 0.2986ᵛ | 0.2676ᵛ | **accepted** | 1,468,724 | −149,251 | −0.000066 / −0.000007 | 0 | invented | note 2L→1L (halves per-note deploy state) + user.L0/preset.L0 mixer strips: BETTER both modes, −46.8% vs 2.76M; cleanest run of the chain (zero NaN activity). ᵛ = VAL half (first val-split track-2 row). New champion. |
| A10 | 0.2989ᵛ | 0.2679ᵛ | rejected | 1,319,473 | −149,251 | +0.000196 / +0.000176 | 0 | invented | user 3L→2L + note.L0/deck.L3 mixer strips: BOTH ratios over the bar (1.96×/1.76×) — the chain's first floor after 5 accepts. Suspect = note.L0 (bare time-mixer note stream); A11 de-bundles. |
| A11 | 0.2989ᵛ | 0.2677ᵛ | rejected | 1,352,620 | −116,104 | +0.000251 / +0.000073 | 0 | invented | A10 minus note.L0 strip: ahead damage IDENTICAL to A10 → user depth floors at 3L (owns ahead); note.L0's mixer was the imm poison (~+0.00018). Depth floors mapped; preset 3L→2L = last untried → A12. |
| A12 | 0.2987ᵛ | 0.2677ᵛ | rejected | 1,385,767 | −82,957 | +0.000090 / +0.000123 | 0 | invented | preset 3L→2L: imm ratio 1.23× the bar (ahead 0.90× passes). Preset depth floors at 3L — ALL depth floors now mapped (card2/deck4/note1/preset3/user3); track 2 goes structural. |
| A13 | 0.2988ᵛ | 0.2678ᵛ | **accepted** (directed re-anchor) | 1,468,724 | 0 | n/a (re-baseline) | 0 | Andrew | State-feature removal (ZERO_FEATURES=22) re-anchor: costs +0.00021/+0.00019 at d=128 (opposite sign vs d=32!) — price recorded, directive stands. New track-2 reference. |
| A14 | 0.2988ᵛ | 0.2677ᵛ | **accepted** | 1,380,660 | −88,064 | −0.000044 / −0.000067 | 0 | invented | LoRA dims halved (all streams): BETTER both modes — the ranks were oversized. First structural cut; −50.03% vs 2.76M (halfway mark crossed). New champion. |
| A15 | 0.2990ᵛ | 0.2681ᵛ | **accepted** | 808,762 | −571,898 | +0.000041 / +0.000064 | 0 | Andrew (delegated) | **THE WIDTH CUT: d_model 128→96 (N_HEADS 4→3, K=32 kept).** Largest single cut of the track: −41.4% params, per-card state −25%. Ratios 41%/64% of the bar. Training-val tracked A14 exactly all run ⇒ the d=128 trunk was over-wide. 3.41× below 2.76M. New champion. |
| A16 | 0.2999ᵛ | 0.2688ᵛ | rejected | 388,032 | −420,730 | +0.000198 / +0.000171 | 0 | invented | **d_model 96→64 (N_HEADS 2): THE WIDTH FLOOR.** Both modes ~1.7–2.0× the bar — per-param cost quadrupled in one rung vs A15. Would have been 7.11× below 2.76M. A15 stands. |
| A17 | 0.2993ᵛ | 0.2683ᵛ | rejected (by 26 millionths) | 584,766 | −223,996 | +0.000112 / +0.000083 | 0 | invented | **d_model 96→80 (5 heads × K=16).** imm PASSES (83% of bar), ahead 112% — a 0.000026 raw miss, ~15× inside cross-seed spread ⇒ noise-limited, not a floor. Per-card state −56% (K=16). Retry = A18 (same width + LoRA 8→4 buys the allowance). |
| A18 | 0.2993ᵛ | 0.2684ᵛ | **accepted** (directed) | 557,246 | −251,516 | +0.000108 / +0.000111 | 0 | invented | **A17's width + LoRA 8→4 = 4.95× below 2.76M. VERDICT CHANGED BY ANDREW 2026-07-26** — the ≥5× goal outranks a marginal-rate gate missed by ~10%; cumulative cost vs A0 is only +0.00096/+0.00053. New track-2 champion; per-card state 2,880 floats. Findings stand: two draws at d=80 ~110% ⇒ width floor reached, and the LoRA halving is no longer free at this width (was positive at d=128) ⇒ the trunk is capacity-limited. |

### Track-2 compression curve — the WHOLE lineage on ONE scale (val half, recomputed 2026-07-25)

Andrew asked whether the stored result files let us re-score old runs on the 2.5k val half.
They do: `result/RWKV-<tag>.jsonl` holds one record per user and the benchmark metric is the
unweighted mean of per-user LogLoss, so restricting the average to users 5001–7500 gives
exactly what a val-half eval would have produced — no GPU, no re-run. Tool:
`python optimization/val_half_recompute.py [tags] [--lo N --hi N]` (defaults: all track-2
tags, users 5001–7500). **This removes the ᵛ-marker caveat retroactively — the pre-split
rows (A0–A8) and the post-split rows are now directly comparable.**

| run | params | vs 2.76M | ahead | imm | Δahead vs A0 | Δimm vs A0 |
|---|---|---|---|---|---|---|
| A0 (1-ep d=128 retrain) | 2,762,884 | 1.00× | 0.298342 | 0.267858 | — | — |
| A1 (mixers→1.0) | 2,320,516 | 1.19× | 0.298252 | 0.267927 | −0.000090 | +0.000069 |
| A2 (deck 4L→3L, rej) | 2,204,412 | 1.25× | 0.298435 | 0.267936 | +0.000093 | +0.000078 |
| A3 (GRU head, unstable) | 2,126,224 | 1.30× | 0.298184 | 0.266818 | −0.000158 | −0.001040 |
| A4 (no-residual re-anchor) | — | — | 0.298798 | 0.267867 | +0.000456 | +0.000009 |
| A5 | 2,115,359 | 1.30× | 0.298813 | 0.267722 | +0.000471 | −0.000136 |
| A6 | 1,949,624 | 1.42× | 0.298758 | 0.267837 | +0.000416 | −0.000021 |
| A7 (user 4L→3L) | 1,767,226 | 1.56× | 0.298689 | 0.267576 | +0.000347 | −0.000282 |
| A8 (card 3L→2L) | 1,617,975 | 1.71× | 0.298723 | 0.267625 | +0.000381 | −0.000233 |
| A9 (note 2L→1L) | 1,468,724 | 1.88× | 0.298625 | 0.267615 | +0.000283 | −0.000243 |
| A10 (rej) | 1,319,473 | 2.09× | 0.298918 | 0.267877 | +0.000576 | +0.000019 |
| A11 (rej) | 1,352,620 | 2.04× | 0.298916 | 0.267700 | +0.000574 | −0.000158 |
| A12 (rej) | 1,385,767 | 1.99× | 0.298699 | 0.267717 | +0.000357 | −0.000141 |
| A13 (state-feature re-anchor) | 1,468,724 | 1.88× | 0.298837 | 0.267805 | +0.000495 | −0.000053 |
| A14 (LoRA halving) | 1,380,660 | 2.00× | 0.298798 | 0.267746 | +0.000456 | −0.000112 |
| **A15 (d_model 128→96)** | **808,762** | **3.41×** | 0.299031 | 0.268111 | **+0.000689** | **+0.000253** |

**Headline: the entire journey from the 2.76M-param A0 to the 808k-param A15 — a 3.41×
reduction — cost +0.000689 ahead and +0.000253 imm.** For scale, that total is about the
size of ONE accepted track-1 accuracy iteration, and it is ~1/3 of what the GRU baseline
gave up (SE-2: +0.0019/+0.0027) for a model that is *larger*. Two structure notes fall out:
(a) imm was essentially FREE until A15 (every accepted rung sat at or below A0 on imm; the
width cut is the first to spend it), and (b) the rejected rungs A10–A12 are visible as the
depth floors — they cost more ahead than their neighbours while saving less.
Caveats: A3's row is n=2436 (its 129 NaN skips) and A4 predates the params column in the
log; A0's n=2498 (7 nanskips, 2 in the val half).
