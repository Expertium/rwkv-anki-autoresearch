# Optimization log (steps 4–5–7)

Regenerated from `log.jsonl` (do not edit by hand). `comment` is in the jsonl only.
Gates: LL not worse than iter0 by >+0.0015 (both modes); state ≤ iter0; size identical.
Gates are vs ITER0 (a floor), NOT the champion — passing all gates does NOT mean accepted.
status: accepted = kept (adopted as a champion or a valid alternative); rejected = not kept
(failed a gate, OR passed the iter0 floor but unreliable/regressed — e.g. iter11).

| # | status | timestamp | ahead LL | imm LL | params | state KiB | throughput (rev/s) | wilcoxon p | size✓ | LL✓ | state✓ | summary |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | baseline | 2026-06-27T22:21:25 | 0.374046 | 0.319475 | 2,762,884 | 51.0 | 181.8 | n/a | baseline | baseline | baseline | Frozen baseline: current arch d_model=128, 2.76M params, train 1-100 eval 101-200 |
| 1 | rejected | 2026-06-27T22:51:13 | 0.353276 | 0.321249 | 804,036 | 25.5 | n/a | n/a | PASS | FAIL | PASS | iter1: halve d_model 128->64 (N_HEADS 4->2), 3.44x fewer params, half state |
| 2 | rejected | 2026-06-27T23:12:36 | 0.362885 | 0.326886 | 804,036 | 25.5 | n/a | n/a | PASS | FAIL | PASS | iter2: d_model=64 + IMMEDIATE_SCALE=2.0 to recover imm via ahead-slack |
| 3 | accepted | 2026-06-27T23:34:45 | 0.358576 | 0.318373 | 804,036 | 25.5 | 199.3 | 9.77e-04 | PASS | PASS | PASS | iter3 CHAMPION: d_model=64 + WSD decay phase; both modes beat iter0, 3.44x fewer params |
| 4 | accepted | 2026-06-28T01:27:50 | 0.359626 | 0.316557 | 693,444 | 25.5 | 216.4 | 9.77e-04 | PASS | PASS | PASS | iter4: channel_mixer_factor->1.0 all streams; 693k params (3.98x vs iter0), faster |
| 5 | accepted | 2026-06-28T01:51:42 | 0.359155 | 0.312356 | 644,292 | 25.5 | 216.4 | 0.0029 | PASS | PASS | PASS | iter5: halve rank-16 LoRAs (decay/a/gate 16->8); 644k params (4.29x vs iter0) |
| 6 | accepted | 2026-06-28T02:12:05 | 0.356386 | 0.313864 | 583,996 | 25.5 | 244.3 | 9.77e-04 | PASS | PASS | PASS | iter6: trim deck/user 4->3 layers ([3,3,2,3,3]); 584k params (4.73x vs iter0) |
| 7 | rejected | 2026-06-28T02:26:56 | 0.357251 | 0.322097 | 187,808 | 12.75 | n/a | n/a | PASS | FAIL | PASS | iter7: d_model 64->32 (N_HEADS 1); 188k params (14.7x) but imm gate FAIL +0.0026 |
| 8 | rejected | 2026-06-28T02:32:02 | 0.355052 | 0.322027 | 187,808 | 12.75 | n/a | n/a | PASS | FAIL | PASS | iter8: d=32 WS-only (no decay) -- imm still FAIL +0.0026; decay was not the cause |
| 9 | rejected | 2026-06-28T02:45:53 | 0.357412 | 0.332125 | 205,540 | 12.75 | n/a | n/a | PASS | FAIL | PASS | iter9: d=32 + restore [3,4,2,3,4] -- adding layers at d=32 HURT imm badly (+0.0127), rejected |
| 10 | rejected | 2026-06-28T02:58:50 | 0.361493 | 0.332713 | 173,472 | 12.75 | n/a | n/a | PASS | FAIL | PASS | iter10: d=32 + LoRA 8->4 -- HURT both modes (imm +0.0132), rejected; cutting LoRA does not transfer to d=32 |
| 11 | rejected | 2026-06-28T03:16:23 | 0.356688 | 0.319494 | 209,312 | 12.75 | 254.7 | 9.77e-04 | PASS | PASS | PASS | iter11 d=32 LoRA16, 209k (13.2x): imm 0.319494 passed by LUCK; exact re-run iter12 FAILED 0.326500 -> REJECTED as unreliable |
| 12 | rejected | 2026-06-28T03:28:40 | 0.354833 | 0.326500 | 209,312 | 12.75 | n/a | n/a | PASS | FAIL | PASS | iter12: re-run of iter11 imm 0.3265 (vs 0.3195) -- d=32 variance ~0.007 >> headroom; iter6 is reliable champion |
| 13 | rejected | 2026-06-28T03:34:05 | 0.388074 | 0.387334 | 209,312 | 12.75 | n/a | n/a | PASS | FAIL | PASS | iter13: naive soup of iter11+iter12 FAILS (incoherent, 0.388/0.387) -- basin problem; iter6 stays reliable champion |
| 14 | rejected | 2026-06-28T04:01:07 | 0.356341 | 0.332069 | 209,312 | 12.75 | n/a | n/a | PASS | FAIL | PASS | iter14: SWA of iter11 WS ckpts -- imm 0.332 WORSE; iter11's 0.319 was a lucky high-LR oscillation low |
| 15 | rejected | 2026-06-28T04:04:38 | 0.358003 | 0.337775 | 209,312 | 12.75 | n/a | n/a | PASS | FAIL | PASS | iter15: SWA of iter12 WS ckpts -- confirms SWA-WS fails for d=32; iter6 is the reliable champion |
| 16 | accepted | 2026-06-28T04:27:24 | 0.349075 | 0.304314 | 555,324 | 25.5 | 247.1 | 0.9860 | PASS | PASS | PASS | iter16 CHAMPION: d=64 + LoRA 8->4; 555k (4.98x); big imm+ahead gain vs iter6 |
| 17 | rejected | 2026-06-28T04:45:10 | 0.354634 | 0.312111 | 540,988 | 25.5 | 245.4 | n/a | PASS | PASS | PASS | iter17: d=64 LoRA 2 underfits, regresses vs iter16; LoRA 4 is the d=64 floor, NOT adopted |
| 18 | accepted | 2026-06-28T05:06:38 | 0.354123 | 0.313217 | 527,224 | 17.0 | 267.8 | 9.77e-04 | PASS | PASS | PASS | iter18: card 3->2 LEAN alt 527k/17KiB/268rev-s but imm +0.009 vs iter16; iter16 stays champion |
| 19 | rejected | 2026-06-28T08:23:59 | 0.359029 | 0.343160 | 209,312 | 12.75 | n/a | n/a | PASS | FAIL | PASS | iter19 d=32 LoRA16 peak LR 3.5e-4 -- HALVED LR underfits, imm 0.343 FAIL |
| 20 | accepted | 2026-06-28T09:00:12 | 0.351458 | 0.315773 | 209,312 | 12.75 | 244.4 | 9.77e-04 | PASS | PASS | PASS | iter20 d=32 LoRA16 +4-epoch decay (iter11 seed): imm 0.3158 PASS, 13.2x reliable |
| 21 | accepted | 2026-06-28T09:00:15 | 0.348625 | 0.315078 | 209,312 | 12.75 | 244.4 | 9.77e-04 | PASS | PASS | PASS | iter21 d=32 +4-epoch decay (iter12 seed): imm 0.3151, confirms variance fix |
| 22 | accepted | 2026-06-28T09:00:17 | 0.348199 | 0.315237 | 209,312 | 12.75 | 244.4 | 9.77e-04 | PASS | PASS | PASS | iter22 d=32 +4-epoch decay (FRESH seed): imm 0.3152, 3rd point confirms reliability |
| 23 | accepted | 2026-06-28T09:34:58 | 0.343852 | 0.301092 | 555,324 | 25.5 | 247.2 | 0.3760 | PASS | PASS | PASS | iter23 d=64 iter16-arch + 4-epoch decay: imm 0.301092 (-0.0032), NEW accuracy champion |
| 24 | accepted | 2026-06-28T09:43:30 | 0.342512 | 0.299844 | 555,324 | 25.5 | 247.2 | 0.3760 | PASS | PASS | PASS | iter24 d=64 6-epoch decay: imm 0.299844 (-0.0012 vs 4ep), NEW accuracy champion |
| 25 | accepted | 2026-06-28T09:56:50 | 0.341697 | 0.298889 | 555,324 | 25.5 | 247.2 | 0.3760 | PASS | PASS | PASS | iter25 d=64 8-epoch decay: imm 0.298889 (-0.00096 vs 6ep), best accuracy; decay sweep diminishing |
| 26 | rejected | 2026-06-28T10:11:36 | 0.350099 | 0.315859 | 209,312 | 12.75 | n/a | n/a | PASS | PASS | PASS | iter26 d=32 8-epoch decay: imm 0.315859, does NOT beat 4-epoch (iter21 0.315078); longer decay is arch-dependent |
| 27 | rejected | 2026-06-28T11:10:34 | 0.348630 | 0.315073 | 209,312 | 12.75 | 284.9 | n/a | PASS | PASS | PASS | iter27 int8 weight PTQ REJECTED: B=1 = fp32 but ~3x slower under multi-stream load; file size not a priority, keep fp32 weights |
| 28 | rejected | 2026-06-28T11:10:37 | 0.348748 | 0.315556 | 209,312 | 12.75 | 283.9 | n/a | PASS | PASS | PASS | iter28 int4 weight PTQ REJECTED: B=1 = fp32 but ~3x slower under multi-stream load; keep fp32 weights |
| 29 | accepted | 2026-06-28T13:11:01 | 0.347166 | 0.312980 | 192,800 | 12.75 | 282.2 | 0.5000 | PASS | PASS | PASS | SRS heads 128->64 (num_curves/num_points): -16512 params (192800, 14.3x), accuracy improves, speed-neutral |
| 30 | rejected | 2026-06-28T13:28:56 | 0.351230 | 0.320608 | 182,398 | 8.5 | n/a | n/a | PASS | PASS | PASS | card stream 3->2: state 8.5 KiB (-33%), 182398 params; passes gate but imm budget nearly spent |
| 31 | accepted | 2026-06-28T13:54:29 | 0.350770 | 0.315438 | 192,800 | 8.5 | 282.6 | 0.7270 | PASS | PASS | PASS | REBALANCE card 3->2 + note 2->3: state 12.75->8.5 KiB (-33%), 192800 params (==iter29) |
| 32 | rejected | 2026-06-28T14:31:49 | 0.351563 | 0.316532 | 192,800 | 8.5 | n/a | n/a | PASS | PASS | PASS | rebalance variant card 3->2 + user 3->4: 8.5 KiB but WORSE than iter31 note-grow, dominated |
| 33 | rejected | 2026-06-28T14:48:51 | 0.367233 | 0.372061 | 169,888 | 8.5 | n/a | n/a | PASS | FAIL | PASS | FC/head inner width 4->2 (head_fc_mult): 169888 params (16.3x) but imm REGRESSES badly |
| 34 | rejected | 2026-06-28T15:09:46 | 0.358057 | 0.328615 | 184,672 | 8.5 | n/a | n/a | PASS | FAIL | PASS | surgical input-FC width 4->2 (features_fc_mult, heads kept 4): 184672 params but imm +0.009 FAIL |
| 35 | accepted | 2026-06-28T15:36:16 | 0.351903 | 0.316508 | 192,800 | 4.25 | 281.4 | 2.41e-04 | PASS | PASS | PASS | card 2->1 + note 3->4 ([1,3,4,3,3]): per-card state 8.5->4.25 KiB (-50%), 192800 params |
| 36 | accepted | 2026-06-28T16:02:03 | 0.347959 | 0.313864 | 192,800 | 4.25 | 285.6 | 4.25e-04 | PASS | PASS | PASS | card 1 + deck 3->4 ([1,4,3,3,3]): 4.25 KiB, deck-grow DOMINATES iter35 note-grow, deploy-optimal |
| 38 | rejected | 2026-06-28T16:53:54 | 0.349766 | 0.315707 | 192,800 | 4.25 | n/a | n/a | PASS | PASS | PASS | note 3to2 + deck 4to5 rebalance, param-neutral, note state 12.75to8.5 KiB |

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
| card int8 | 1.06 | 12.75 | +0.000002 | +0.000002 | PASS | valid |
| card int4 | 0.53 | 12.75 | +0.000355 | +0.000351 | PASS | valid |
| card int2 | 0.27 | 12.75 | +0.001249 | +0.000493 | PASS | valid-extreme |
| card+note int8 | 1.06 | 3.19 | +0.000118 | +0.000217 | PASS | valid |
| card int4 + note int8 | 0.53 | 3.19 | +0.000470 | +0.000577 | PASS | RECOMMENDED-DEPLOY |
| card int2 + note int8 | 0.27 | 3.19 | +0.001319 | +0.000669 | PASS | valid-extreme |
| card int4 + note int4 | 0.53 | 1.59 | +0.003569 | +0.005360 | PASS-iter0-only | REJECTED |


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
| 39 | 192,800 | decay-only QAT (124-step decay from iter36 WS-final) | card int2 + note int4 | 0.298538 | 0.327633 | +0.000018 | -0.000127 | +0.002456 | +0.001129 | PASS | 0.27+1.59 KiB (card+note) |
| 40 | 192,800 | full WS+QAT FROM SCRATCH (558 WS + 124 decay) -- REJECTED | card int2 + note int4 | 0.310306 | 0.338650 | +0.004456 | +0.005340 | +0.009787 | +0.006680 | PASS(worse) | 0.27+1.59 KiB (card+note) |
| 41 | 265,614 | moderate grow [1,8,3,6,3] (deck 4->8, preset 3->6), fresh WS + decay-QAT | card int2 + note int4 | 0.301564 | 0.332704 | +0.000634 | +0.001446 | +0.004866 | +0.004627 | PASS(worse) | 0.27+1.59 KiB (card+note); +72,814 params in deck/preset (state-free on card) |
| 42 | 411,242 | aggressive grow [1,16,3,12,3] (deck 4->16, preset 3->12), fresh WS + decay-QAT | card int2 + note int4 | 0.358399 | 0.377111 | +0.011489 | +0.018432 | +0.050846 | +0.032048 | FAIL | 0.27+1.59 KiB (card+note); +218,442 params over champ in deck/preset |
| 43 | 192,800 | decay-only QAT (124-step decay from iter36 WS-final) -- NOTE INT2 target | card int2 + note int2 | 0.299469 | 0.328570 | +0.003525 | +0.002431 | -0.000120 | -0.000492 | PASS | 0.27+0.80 KiB (card+note); note int2 = the >=2x note target MET |
| 44 | 192,800 | LONGER 8-epoch decay-QAT from iter36 WS-final -- card int2 + note int2 | card int2 + note int2 | 0.295436 | 0.323291 | +0.002983 | +0.001372 | -0.003610 | -0.004712 | PASS | 0.27+0.80 KiB (card+note); note int2 = >=2x target |
| 45 | 192,800 | 16-epoch decay-QAT from iter36 WS-final -- card int2 + note int2 (QAT plateau probe) | card int2 + note int2 | 0.292560 | 0.324638 | +0.004742 | +0.005548 | -0.008245 | -0.007541 | PASS | 0.27+0.80 KiB (card+note); note int2 = >=2x target |
| 46 | 192,800 | iter45 weights + LOW-RANK card deploy (rank-2, int4 factors + int4 shifts) + note int2 -- PTQ, no retrain | card rank-2 int4 (lowrank) + note int2; shifts quantized (RWKV_QUANT_SHIFTS) | 0.291471 | 0.323603 | +0.003653 | +0.004514 | -0.008245 | -0.007541 | PASS | card 0.094 KiB (96 B: rank-2 int4 WKV 64 B + int4 shifts 32 B) + note 0.80 KiB |
| 46 | 192,800 | FIRST real low-rank QAT (card rank2-int4 lowrank QAT + note int2 QAT), 8-epoch decay -- REJECTED | card rank2:int4 lowrank + note int2 | 0.303617 | 0.331601 | +0.010320 | +0.007245 | -0.002766 | -0.002274 | PASS(worse) | card 0.094 KiB (rank2-int4) + note 0.80 KiB |
| 45 | 192,800 | both-low-rank PTQ (card+note rank2-int4 + int4 shifts) on iter45 -- FULL 17-user gate incl 187 via fast Gram+eigen SVD | RWKV_STATE_LOWRANK_SCOPE=card:2:int4,note:2:int4 RWKV_QUANT_SHIFTS=1 on reference/rwkv_iter45.safetensors | 0.288831 | 0.320098 | +0.001013 | +0.001008 | -0.008245 | -0.007541 | PASS (imm 0.288831<=0.320975, ahead 0.320098<=0.375546); beats champ_fp32 -0.007233 imm / -0.006533 ahead | card 96 B (rank-2 int4 WKV + int4 shifts) + note ~288 B (rank-2 int4) |

