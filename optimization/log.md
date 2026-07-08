# Optimization log (steps 4–5–7)

Regenerated from `log.jsonl` (do not edit by hand). `comment` is in the jsonl only.
Gates: LL not worse than iter0 by >+0.0015 (both modes); state ≤ iter0; size identical.
Gates are vs ITER0 (a floor), NOT the champion — passing all gates does NOT mean accepted.
status: accepted = kept (adopted as a champion or a valid alternative); rejected = not kept
(failed a gate, OR passed the iter0 floor but unreliable/regressed — e.g. iter11).


## Baseline to beat (100/100 research phase, eval users 101-200, --short --secs)

The research workbench is train users 1-100 / eval 101-200 (Andrew 2026-06-29). The headline
baseline is the ORIGINAL d=128 arch (2.76M params) trained FROM SCRATCH on the SAME 1-100 -- this
isolates architecture from training-data quantity (the published d=128 row trained on 5000 users,
so its far-lower numbers are partly just 50x more data). by-user-mean LogLoss; `imm`=RWKV-P
(immediate), `ahead`=RWKV (forgetting-curve). GOAL: a SMALLER model whose imm/ahead matches or
beats the `old d=128 (trained 1-100)` row.
★ RULE (Andrew 2026-06-29): a CHAMPION's comparison logloss must be the DEPLOYED model -- with
quantization AND low-rank state enabled (measured via the Rust engine on 101-200), NOT fp32 --
since that is what ships. The d=128 baseline stays fp32 (it's the accuracy target, not deployable).
Champion rows showing fp32 are PLACEHOLDERS until their deployed (quant+low-rank) number is measured.

| model | params | d | trained on | chunk | MAX | ahead LL | imm LL | note |
|---|---|---|---|---|---|---|---|---|
| * old d=128 (trained 1-100) -- BASELINE TO BEAT | 2,762,884 | 128 | 1-100 | sc8k 8192 | 18000 | 0.3203 | 0.2819 | old ARCH at EQUAL data (1-100, aug-off, WS 6ep). THE target: a <=225k model must match/beat this. 14x-smaller d=32 champion costs ~0.008 imm vs this |
| old d=128 (published, trained 5000-10000) | 2,762,884 | 128 | 5000-10000 | n/a | n/a | 0.2989 | 0.2646 | published leaderboard ceiling; 50x more training data than 1-100 -> ~0.017 imm of the gap is DATA, not arch |
| champion sc8k (d=32, 1-100) UNTUNED [fp32, aug-off] | 192,800 | 32 | 1-100 | sc8k 8192 | 66000 | 0.3242 | 0.2926 | pre-tuning baseline anchor (peak_lr 7e-4 / clip 0.5 / epochs 6). variance 0.000000 (aug-off+determ). Behind d=128 by +0.0107 imm / +0.0039 ahead -- which HP tuning then closed |
| TUNED champion WS-15 (no decay) [fp32] | 192,800 | 32 | 1-100 | sc8k 8192 | 66000 | 0.3163 | 0.2820 | HP-tuned WS only: peak_lr 1e-3 / warmup 200 / wd 0.01 / clip 0.25 / epochs 15. ckpt hp_epochs_15_2400.pth. Superseded by +decay below |
| ** CHAMPION WS-15 + 4ep decay (d=32, 1-100) [fp32] | 192,800 | 32 | 1-100 | sc8k 8192 | 66000 | 0.3148 | 0.2802 | WS-15 + 4-epoch cosine decay (LR 1e-3->0). vs WS-15: ahead -0.0014 / imm -0.0018 (decay accepted). * BEATS the d=128 baseline on BOTH modes (ahead by 0.0055, imm by 0.0017) at 14x fewer params, pure training (HP tune + WSD), zero arch change. ckpt scratchpad/tuner/decay15/decay15_640.pth |
| ** DEPLOYED champion BOTH-low-rank int4 (est. 100u) | 192,800 | 32 | 1-100 | sc8k 8192 | 66000 | 0.3140 | 0.2806 | DEPLOY = card AND note BOTH rank-2 low-rank, int4 factors + int4 shifts (Andrew 2026-06-30: logloss with quant+low-rank on BOTH). Sizes: card 96 B (0.094 KiB) + note 288 B (0.28 KiB). PTQ penalty essentially FREE (16-user: imm +0.0004 / ahead -0.0008) -> deployed ~= fp32 champion. vs d=128 baseline: BEATS both (imm by 0.0013, ahead by 0.0063). ESTIMATE = exact 100u fp32 + 16-user penalty (scratchpad/run_deploy_bothlr.sh). int2 FACTORS rejected: +0.0068 imm AND panics on 12/17 users (too coarse for rank-2 factors). Note: per-step note SVD fails on power-user 187 (1/17) -> deploy needs a fallback there |
| *** CHAMPION 1500u 1ep WS + decay (d=32) [fp32] | 192,800 | 32 | 1000-2499 (1500u) | sc8k 8192 | 66000 | 0.3097 | 0.2764 | NEW CHAMPION 2026-06-30: DATA VARIETY beats repetition. 1 epoch WS on ~1500 users (3351 steps) + 0.27-epoch decay (904 steps). vs prev 100u champion (WS-15+4ep): +0.0051 ahead / +0.0038 imm. vs d=128 baseline: +0.0106 ahead / +0.0056 imm. SAME arch/params/state -- only train DATA changed -> model is DATA-limited, scale toward 5k. ckpt scratchpad/exp_1500/t1500d_904.pth, weights reference/champ_1500d.safetensors. Deployed (quant+low-rank) number PENDING re-measure (~+0.0005 penalty est) |


## Research-phase experiments (100/100 workbench)

Train users 1-100 / eval 101-200 (sc8k 8192-chunk, MAX=66000, augmentation OFF, deterministic).
ACCEPT iff BOTH ahead AND imm improve by >=0.0003 vs the CURRENT champion (params<=225k; card+note
per-entity state unchanged; eval review-count identical). The champion is monotonic. d_ahead/d_imm
are vs the champion at the time (positive = better). Verbose notes live in `research_log.md`.

| exp | change | params | ahead LL | imm LL | Δahead vs champ | Δimm vs champ | status | note |
|---|---|---|---|---|---|---|---|---|
| champion | WS-15 + 4ep cosine decay (curves/points 64), d=32 [1,4,3,3,3] | 192,800 | 0.3148 | 0.2802 | — | — | CHAMPION | HP-tuned + decay; beats d=128 baseline on both; deployed (int4 both-low-rank) targets met |
| exp1 | num_curves/num_points 64->128 | 209,312 | 0.3152 | 0.2802 | -0.0004 | -0.0000 | REJECT | +16.5k params for ZERO gain (ahead worse, imm tied); head resolution is not the lever |
| exp2 | channel_mixer_factor 1.0->1.5 | 207,136 | 0.3153 | 0.2804 | -0.0005 | -0.0002 | REJECT | +14k FFN-capacity HURTS both -> d=32 model is DATA-limited at 100u, not capacity-limited |
| decay8 | WS-15 + 8-epoch decay (vs 4ep) | 192,800 | 0.3152 | 0.2801 | -0.0003 | +0.0001 | REJECT | longer decay: ahead slightly worse, imm +0.00013 (< gate); 4-epoch decay is fine |
| ep18 | WS-18 + 4ep decay (vs WS-15) | 192,800 | 0.3149 | 0.2798 | -0.0001 | +0.0004 | REJECT | epochs SATURATE at 15 on 100 users (ahead worse, imm better) -> motivates the more-DATA experiment |
| t1500ws | 1 epoch WS on 1500 users (1000-2499), WS-only (no decay) | 192,800 | 0.3128 | 0.2788 | +0.0020 | +0.0014 | ACCEPT | DATA VARIETY beats repetition: beats the WS-15+4ep-decay champion on BOTH modes, WS-ONLY. Decay phase next. Confirms data-limited -> supports the 5k direction |
| t1500d | t1500ws + 0.27-epoch cosine decay (1500 users; WS:decay ratio = champion's 15:4) | 192,800 | 0.3097 | 0.2764 | +0.0051 | +0.0038 | NEW CHAMPION | data-variety recipe (1 epoch on ~1500 users + decay) BEATS the 100u/15ep champion by +0.0051 ahead / +0.0038 imm -- the +0.0038 imm exceeds the entire iter0->old-champion loop. fp32; deployed (quant+low-rank) number pending. Confirms data-limited -> scale toward 5k (needs the smaller/faster model) |
| h2k16 | H=2/K=16 (n_heads=2, head_dim=16) via the new K<32 CUDA kernel; 1500u champion recipe (1ep WS 1000-2499 + 0.27ep decay) | 193,724 | 0.3097 | 0.2766 | -0.0000 | -0.0002 | NEW CHAMPION (size/speed) | SIZE/SPEED win (efficiency-budget gate, NOT the +0.0003 monotonic gate). Per-card WKV state HALVED 1088->576 floats; WS train 1.16x faster (1.182 vs 1.020 steps/s); accuracy PARITY (both modes within 0.0002, far inside +0.0015 budget). Still beats d=128 baseline +0.0106 ahead/+0.0053 imm. K<32 kernel parity-verified (test_k16_wkv.py). Next: re-tune HPs (smaller model + larger data). Rust deployed rev/s pending K<32 engine port. ckpt scratchpad/exp_h2k16/h2k16d_904.pth -> reference/champ_h2k16.safetensors |
| champ5k_r1 | FIRST 5k champion run: train 1-5000 (2ep WS = 13108 steps + 0.5ep decay = 3277), quant-aware q72u with per-run LEARNABLE codebooks (cb exports at both seams), champion HPs (peak_lr 1e-3, warmup 200, wd 0.01, clip 0.25), MAX=110000, eval 5001-10000 | 193,724 | 0.3066 | 0.2783 | -0.0102 | -0.0134 | 5k CHAMPION (phase starting point) | PROMOTED to champion_5k.json (ckpt champ5kd_3277.pth + its learned cbs; WS step trace = the Wilcoxon prune ref). vs d=128 fp target on the same 5000 users: -0.010187 ahead / -0.013418 imm (paired one-sided Wilcoxon p=1.0 both -- behind the target, as expected for the 15x-smaller quant-aware model; the phase's job is closing this). Wall-clock: WS 5h00m (~1.36 s/step avg), decay 72 min, eval 2x-sharded 66 min (+14 min resume) -> ~7.0h clean pipeline. Two latent bugs hit+fixed: LEARN=1 optim resume param-group mismatch (f71f43b), per-user lmdb env leak in get_benchmark_info killed eval shard 0 at user 2007 (7d095e3; n=5000 finish gate caught it). Tuner baseline (5001-5200 subset): ahead 0.294204 imm 0.270881. Next: hp_tuner_5k loop. |
| champ5k_b1 | BUDGET A/B: champion recipe at HALF budget -- WS 1 ep (6554 steps) + 0.25 ep decay (1638), everything else identical to champ5k_r1 (champion HPs, q72u learnable cbs) | 193,724 | 0.3066 | 0.2779 | -0.0001 | +0.0004 | NEW CHAMPION (size/speed: budget halved) | vs champ5k_r1 paired on the same 5000 users: ahead -0.000058 (p=0.31, indistinguishable), imm +0.000430 BETTER (p=6.1e-62). The 2nd WS epoch (same 5000 users reshuffled) adds NOTHING -- consistent with the data-variety-beats-repetition lesson. ADOPTED (Andrew's rule): WS 1 ep + ratio-0.25 decay is now the budget for ALL 5k runs (tuner trials AND research runs); champion runs drop ~7h -> ~3.5h. SIZE/SPEED-class accept (efficiency budget, p-gate exempt; imm improvement is a bonus). Promoted: ckpt champ5kb1d_1638.pth + its learned cbs; 6554-step WS trace = the new prune ref. Wall-clock: WS 2h27m, decay 37m, eval 89m (2-sharded). |


## Iteration table (steps 4-5-7)

| # | status | timestamp | ahead LL | imm LL | params | state KiB | throughput (rev/s) | wilcoxon p | size✓ | LL✓ | state✓ | summary |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | baseline | 2026-06-27T22:21:25 | 0.3740 | 0.3195 | 2,762,884 | 51.0 | 181.8 | n/a | baseline | baseline | baseline | Frozen baseline: current arch d_model=128, 2.76M params, train 1-100 eval 101-200 |
| 1 | rejected | 2026-06-27T22:51:13 | 0.3533 | 0.3212 | 804,036 | 25.5 | n/a | n/a | PASS | FAIL | PASS | iter1: halve d_model 128->64 (N_HEADS 4->2), 3.44x fewer params, half state |
| 2 | rejected | 2026-06-27T23:12:36 | 0.3629 | 0.3269 | 804,036 | 25.5 | n/a | n/a | PASS | FAIL | PASS | iter2: d_model=64 + IMMEDIATE_SCALE=2.0 to recover imm via ahead-slack |
| 3 | accepted | 2026-06-27T23:34:45 | 0.3586 | 0.3184 | 804,036 | 25.5 | 199.3 | 9.77e-04 | PASS | PASS | PASS | iter3 CHAMPION: d_model=64 + WSD decay phase; both modes beat iter0, 3.44x fewer params |
| 4 | accepted | 2026-06-28T01:27:50 | 0.3596 | 0.3166 | 693,444 | 25.5 | 216.4 | 9.77e-04 | PASS | PASS | PASS | iter4: channel_mixer_factor->1.0 all streams; 693k params (3.98x vs iter0), faster |
| 5 | accepted | 2026-06-28T01:51:42 | 0.3592 | 0.3124 | 644,292 | 25.5 | 216.4 | 0.0029 | PASS | PASS | PASS | iter5: halve rank-16 LoRAs (decay/a/gate 16->8); 644k params (4.29x vs iter0) |
| 6 | accepted | 2026-06-28T02:12:05 | 0.3564 | 0.3139 | 583,996 | 25.5 | 244.3 | 9.77e-04 | PASS | PASS | PASS | iter6: trim deck/user 4->3 layers ([3,3,2,3,3]); 584k params (4.73x vs iter0) |
| 7 | rejected | 2026-06-28T02:26:56 | 0.3573 | 0.3221 | 187,808 | 12.75 | n/a | n/a | PASS | FAIL | PASS | iter7: d_model 64->32 (N_HEADS 1); 188k params (14.7x) but imm gate FAIL +0.0026 |
| 8 | rejected | 2026-06-28T02:32:02 | 0.3551 | 0.3220 | 187,808 | 12.75 | n/a | n/a | PASS | FAIL | PASS | iter8: d=32 WS-only (no decay) -- imm still FAIL +0.0026; decay was not the cause |
| 9 | rejected | 2026-06-28T02:45:53 | 0.3574 | 0.3321 | 205,540 | 12.75 | n/a | n/a | PASS | FAIL | PASS | iter9: d=32 + restore [3,4,2,3,4] -- adding layers at d=32 HURT imm badly (+0.0127), rejected |
| 10 | rejected | 2026-06-28T02:58:50 | 0.3615 | 0.3327 | 173,472 | 12.75 | n/a | n/a | PASS | FAIL | PASS | iter10: d=32 + LoRA 8->4 -- HURT both modes (imm +0.0132), rejected; cutting LoRA does not transfer to d=32 |
| 11 | rejected | 2026-06-28T03:16:23 | 0.3567 | 0.3195 | 209,312 | 12.75 | 254.7 | 9.77e-04 | PASS | PASS | PASS | iter11 d=32 LoRA16, 209k (13.2x): imm 0.319494 passed by LUCK; exact re-run iter12 FAILED 0.326500 -> REJECTED as unreliable |
| 12 | rejected | 2026-06-28T03:28:40 | 0.3548 | 0.3265 | 209,312 | 12.75 | n/a | n/a | PASS | FAIL | PASS | iter12: re-run of iter11 imm 0.3265 (vs 0.3195) -- d=32 variance ~0.007 >> headroom; iter6 is reliable champion |
| 13 | rejected | 2026-06-28T03:34:05 | 0.3881 | 0.3873 | 209,312 | 12.75 | n/a | n/a | PASS | FAIL | PASS | iter13: naive soup of iter11+iter12 FAILS (incoherent, 0.388/0.387) -- basin problem; iter6 stays reliable champion |
| 14 | rejected | 2026-06-28T04:01:07 | 0.3563 | 0.3321 | 209,312 | 12.75 | n/a | n/a | PASS | FAIL | PASS | iter14: SWA of iter11 WS ckpts -- imm 0.332 WORSE; iter11's 0.319 was a lucky high-LR oscillation low |
| 15 | rejected | 2026-06-28T04:04:38 | 0.3580 | 0.3378 | 209,312 | 12.75 | n/a | n/a | PASS | FAIL | PASS | iter15: SWA of iter12 WS ckpts -- confirms SWA-WS fails for d=32; iter6 is the reliable champion |
| 16 | accepted | 2026-06-28T04:27:24 | 0.3491 | 0.3043 | 555,324 | 25.5 | 247.1 | 0.9860 | PASS | PASS | PASS | iter16 CHAMPION: d=64 + LoRA 8->4; 555k (4.98x); big imm+ahead gain vs iter6 |
| 17 | rejected | 2026-06-28T04:45:10 | 0.3546 | 0.3121 | 540,988 | 25.5 | 245.4 | n/a | PASS | PASS | PASS | iter17: d=64 LoRA 2 underfits, regresses vs iter16; LoRA 4 is the d=64 floor, NOT adopted |
| 18 | accepted | 2026-06-28T05:06:38 | 0.3541 | 0.3132 | 527,224 | 17.0 | 267.8 | 9.77e-04 | PASS | PASS | PASS | iter18: card 3->2 LEAN alt 527k/17KiB/268rev-s but imm +0.009 vs iter16; iter16 stays champion |
| 19 | rejected | 2026-06-28T08:23:59 | 0.3590 | 0.3432 | 209,312 | 12.75 | n/a | n/a | PASS | FAIL | PASS | iter19 d=32 LoRA16 peak LR 3.5e-4 -- HALVED LR underfits, imm 0.343 FAIL |
| 20 | accepted | 2026-06-28T09:00:12 | 0.3515 | 0.3158 | 209,312 | 12.75 | 244.4 | 9.77e-04 | PASS | PASS | PASS | iter20 d=32 LoRA16 +4-epoch decay (iter11 seed): imm 0.3158 PASS, 13.2x reliable |
| 21 | accepted | 2026-06-28T09:00:15 | 0.3486 | 0.3151 | 209,312 | 12.75 | 244.4 | 9.77e-04 | PASS | PASS | PASS | iter21 d=32 +4-epoch decay (iter12 seed): imm 0.3151, confirms variance fix |
| 22 | accepted | 2026-06-28T09:00:17 | 0.3482 | 0.3152 | 209,312 | 12.75 | 244.4 | 9.77e-04 | PASS | PASS | PASS | iter22 d=32 +4-epoch decay (FRESH seed): imm 0.3152, 3rd point confirms reliability |
| 23 | accepted | 2026-06-28T09:34:58 | 0.3439 | 0.3011 | 555,324 | 25.5 | 247.2 | 0.3760 | PASS | PASS | PASS | iter23 d=64 iter16-arch + 4-epoch decay: imm 0.301092 (-0.0032), NEW accuracy champion |
| 24 | accepted | 2026-06-28T09:43:30 | 0.3425 | 0.2998 | 555,324 | 25.5 | 247.2 | 0.3760 | PASS | PASS | PASS | iter24 d=64 6-epoch decay: imm 0.299844 (-0.0012 vs 4ep), NEW accuracy champion |
| 25 | accepted | 2026-06-28T09:56:50 | 0.3417 | 0.2989 | 555,324 | 25.5 | 247.2 | 0.3760 | PASS | PASS | PASS | iter25 d=64 8-epoch decay: imm 0.298889 (-0.00096 vs 6ep), best accuracy; decay sweep diminishing |
| 26 | rejected | 2026-06-28T10:11:36 | 0.3501 | 0.3159 | 209,312 | 12.75 | n/a | n/a | PASS | PASS | PASS | iter26 d=32 8-epoch decay: imm 0.315859, does NOT beat 4-epoch (iter21 0.315078); longer decay is arch-dependent |
| 27 | rejected | 2026-06-28T11:10:34 | 0.3486 | 0.3151 | 209,312 | 12.75 | 284.9 | n/a | PASS | PASS | PASS | iter27 int8 weight PTQ REJECTED: B=1 = fp32 but ~3x slower under multi-stream load; file size not a priority, keep fp32 weights |
| 28 | rejected | 2026-06-28T11:10:37 | 0.3487 | 0.3156 | 209,312 | 12.75 | 283.9 | n/a | PASS | PASS | PASS | iter28 int4 weight PTQ REJECTED: B=1 = fp32 but ~3x slower under multi-stream load; keep fp32 weights |
| 29 | accepted | 2026-06-28T13:11:01 | 0.3472 | 0.3130 | 192,800 | 12.75 | 282.2 | 0.5000 | PASS | PASS | PASS | SRS heads 128->64 (num_curves/num_points): -16512 params (192800, 14.3x), accuracy improves, speed-neutral |
| 30 | rejected | 2026-06-28T13:28:56 | 0.3512 | 0.3206 | 182,398 | 8.5 | n/a | n/a | PASS | PASS | PASS | card stream 3->2: state 8.5 KiB (-33%), 182398 params; passes gate but imm budget nearly spent |
| 31 | accepted | 2026-06-28T13:54:29 | 0.3508 | 0.3154 | 192,800 | 8.5 | 282.6 | 0.7270 | PASS | PASS | PASS | REBALANCE card 3->2 + note 2->3: state 12.75->8.5 KiB (-33%), 192800 params (==iter29) |
| 32 | rejected | 2026-06-28T14:31:49 | 0.3516 | 0.3165 | 192,800 | 8.5 | n/a | n/a | PASS | PASS | PASS | rebalance variant card 3->2 + user 3->4: 8.5 KiB but WORSE than iter31 note-grow, dominated |
| 33 | rejected | 2026-06-28T14:48:51 | 0.3672 | 0.3721 | 169,888 | 8.5 | n/a | n/a | PASS | FAIL | PASS | FC/head inner width 4->2 (head_fc_mult): 169888 params (16.3x) but imm REGRESSES badly |
| 34 | rejected | 2026-06-28T15:09:46 | 0.3581 | 0.3286 | 184,672 | 8.5 | n/a | n/a | PASS | FAIL | PASS | surgical input-FC width 4->2 (features_fc_mult, heads kept 4): 184672 params but imm +0.009 FAIL |
| 35 | accepted | 2026-06-28T15:36:16 | 0.3519 | 0.3165 | 192,800 | 4.25 | 281.4 | 2.41e-04 | PASS | PASS | PASS | card 2->1 + note 3->4 ([1,3,4,3,3]): per-card state 8.5->4.25 KiB (-50%), 192800 params |
| 36 | accepted | 2026-06-28T16:02:03 | 0.3480 | 0.3139 | 192,800 | 4.25 | 285.6 | 4.25e-04 | PASS | PASS | PASS | card 1 + deck 3->4 ([1,4,3,3,3]): 4.25 KiB, deck-grow DOMINATES iter35 note-grow, deploy-optimal |
| 38 | rejected | 2026-06-28T16:53:54 | 0.3498 | 0.3157 | 192,800 | 4.25 | n/a | n/a | PASS | PASS | PASS | note 3to2 + deck 4to5 rebalance, param-neutral, note state 12.75to8.5 KiB |

## State quantization (deploy-time PTQ on the iter36 champion)

Per-stream round-trip of the recurrent WKV state through int8/int4/int2 at inference
(weights stay fp32). Deltas are by-user-mean vs the **fp32 Rust baseline** on the 17
smallest of users 101-200 (full RNN export of the larger users is infeasible). Gate is
vs iter0 floor (+0.0015 budget, ceilings imm 0.320975 / ahead 0.375546) -- all PASS the
floor, but `verdict` flags how much of the deploy budget each burns. RULE: quant
aggressiveness is proportional to 1/recurrence-length (card int4 ok, note wants int8,
deck/preset/user stay fp32). KiB = quantized per-card / per-note state size.

| config | card KiB | note KiB | imm Δ | ahead Δ | gate | verdict |
|---|---|---|---|---|---|---|
| card int8 | 1.06 | 12.75 | +0.0000 | +0.0000 | PASS | valid |
| card int4 | 0.53 | 12.75 | +0.0004 | +0.0004 | PASS | valid |
| card int2 | 0.27 | 12.75 | +0.0012 | +0.0005 | PASS | valid-extreme |
| card+note int8 | 1.06 | 3.19 | +0.0001 | +0.0002 | PASS | valid |
| card int4 + note int8 | 0.53 | 3.19 | +0.0005 | +0.0006 | PASS | RECOMMENDED-DEPLOY |
| card int2 + note int8 | 0.27 | 3.19 | +0.0013 | +0.0007 | PASS | valid-extreme |
| card int4 + note int4 | 0.53 | 1.59 | +0.0036 | +0.0054 | PASS-iter0-only | REJECTED |


## Quant-aware training (QAT) experiments

State-QAT: the card/note WKV state is round-tripped through int-N every step during training
(STE gradient), so weights adapt to the deploy-time quant. BOTH modes recorded: imm (RWKV-P)
and ahead = the forgetting-curve mode (RWKV). Numbers are by-user-mean on the **17-user gate**
(rust deploy-quant vs rust fp32), NOT the 100-user kernel eval. TWO SEPARATE costs:
- `quant cost` = deploy(quant) - same-QAT-model fp32 = the cost QAT REMOVES (near 0 = QAT works).
- `fp32 ft-regress` = QAT-model fp32 - champion fp32 = an fp32 regression from the (short) fine-
  tune, NOT a quant effect. Decay-only QAT leaves this positive; full-WS QAT / deck-preset grow
  aim to drive it to ~0. NOTE: for the SAME aggressive config, QAT beats PTQ (PTQ card int2+note
  int4 ~+0.0044 FAILS; QAT total +0.0025 PASSES) -- the ft-regress is the only thing left to kill.
Gate vs iter0 (imm ceiling 0.320975, ahead 0.375546).

| # | params | training mode | deploy config | deploy imm | deploy ahead(fc) | quant cost imm | quant cost ahead | fp32 ft-regress imm | fp32 ft-regress ahead | gate | state |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 39 | 192,800 | decay-only QAT (124-step decay from iter36 WS-final) | card int2 + note int4 | 0.2985 | 0.3276 | +0.0000 | -0.0001 | +0.0025 | +0.0011 | PASS | 0.27+1.59 KiB (card+note) |
| 40 | 192,800 | full WS+QAT FROM SCRATCH (558 WS + 124 decay) -- REJECTED | card int2 + note int4 | 0.3103 | 0.3387 | +0.0045 | +0.0053 | +0.0098 | +0.0067 | PASS(worse) | 0.27+1.59 KiB (card+note) |
| 41 | 265,614 | moderate grow [1,8,3,6,3] (deck 4->8, preset 3->6), fresh WS + decay-QAT | card int2 + note int4 | 0.3016 | 0.3327 | +0.0006 | +0.0014 | +0.0049 | +0.0046 | PASS(worse) | 0.27+1.59 KiB (card+note); +72,814 params in deck/preset (state-free on card) |
| 42 | 411,242 | aggressive grow [1,16,3,12,3] (deck 4->16, preset 3->12), fresh WS + decay-QAT | card int2 + note int4 | 0.3584 | 0.3771 | +0.0115 | +0.0184 | +0.0508 | +0.0320 | FAIL | 0.27+1.59 KiB (card+note); +218,442 params over champ in deck/preset |
| 43 | 192,800 | decay-only QAT (124-step decay from iter36 WS-final) -- NOTE INT2 target | card int2 + note int2 | 0.2995 | 0.3286 | +0.0035 | +0.0024 | -0.0001 | -0.0005 | PASS | 0.27+0.80 KiB (card+note); note int2 = the >=2x note target MET |
| 44 | 192,800 | LONGER 8-epoch decay-QAT from iter36 WS-final -- card int2 + note int2 | card int2 + note int2 | 0.2954 | 0.3233 | +0.0030 | +0.0014 | -0.0036 | -0.0047 | PASS | 0.27+0.80 KiB (card+note); note int2 = >=2x target |
| 45 | 192,800 | 16-epoch decay-QAT from iter36 WS-final -- card int2 + note int2 (QAT plateau probe) | card int2 + note int2 | 0.2926 | 0.3246 | +0.0047 | +0.0055 | -0.0082 | -0.0075 | PASS | 0.27+0.80 KiB (card+note); note int2 = >=2x target |
| 46 | 192,800 | iter45 weights + LOW-RANK card deploy (rank-2, int4 factors + int4 shifts) + note int2 -- PTQ, no retrain | card rank-2 int4 (lowrank) + note int2; shifts quantized (RWKV_QUANT_SHIFTS) | 0.2915 | 0.3236 | +0.0037 | +0.0045 | -0.0082 | -0.0075 | PASS | card 0.094 KiB (96 B: rank-2 int4 WKV 64 B + int4 shifts 32 B) + note 0.80 KiB |
| 46 | 192,800 | FIRST real low-rank QAT (card rank2-int4 lowrank QAT + note int2 QAT), 8-epoch decay -- REJECTED | card rank2:int4 lowrank + note int2 | 0.3036 | 0.3316 | +0.0103 | +0.0072 | -0.0028 | -0.0023 | PASS(worse) | card 0.094 KiB (rank2-int4) + note 0.80 KiB |
| 45 | 192,800 | both-low-rank PTQ (card+note rank2-int4 + int4 shifts) on iter45 -- FULL 17-user gate incl 187 via fast Gram+eigen SVD | RWKV_STATE_LOWRANK_SCOPE=card:2:int4,note:2:int4 RWKV_QUANT_SHIFTS=1 on reference/rwkv_iter45.safetensors | 0.2888 | 0.3201 | +0.0010 | +0.0010 | -0.0082 | -0.0075 | PASS (imm 0.288831<=0.320975, ahead 0.320098<=0.375546); beats champ_fp32 -0.007233 imm / -0.006533 ahead | card 96 B (rank-2 int4 WKV + int4 shifts) + note ~288 B (rank-2 int4) |

