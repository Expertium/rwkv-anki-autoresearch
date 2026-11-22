# Card/note state quantization - results

Started 2026-11-19. Andrew's task: quantize the **card** and **note** recurrent states only, to at
most **half of naive fp32->int4** bits, with **both heads within +0.0005** of the current best and
**at most 25% slower** (Rust engine) than unquantized.

## The anchor and the gate

**Anchor = the same run without quantization (Andrew, 2026-11-20 22:22).** Every result on this page
is judged against *"whatever run is exactly identical EXCEPT for quantization"*: the 124-step decay
trained from `pretrain/rwkv/opt_iter36/rwkv_iter36_558.pth` with quantization OFF - tag
`qnj_it36ctl`, **0.333802 ahead / 0.306211 imm** on the seeded 750-user roster. Points measured at a
different step budget use their own identical unquantized twin (e.g. `qnj_it36ctl496` for the
496-step point), never a borrowed budget.

**Row 0b: the 496-step fp32 control (scored 2026-11-20 22:46).** The same data, recipe and start
checkpoint with `RWKV_MAX_STEPS=496` instead of 124 - tag `qnj_it36ctl496` - scores **0.321659
ahead / 0.294614 imm**, i.e. 0.012143/0.011598 BETTER than row 0 (ub95 +0.012911/+0.012336,
p 2.0e-108/1.1e-122). The decay length, not the quantizer, is what moves the loss at this scale,
so row 8 is quoted against this twin and not against row 0.

Two earlier framings are **withdrawn**: the comparison against the `iter36` champion
(`rwkv_iter36_124.pth`, 0.342813/0.314272) mixes in a different training run and different data, so
its 0.0085 "win" was not a quantization result and must not be quoted; and the comparison against the
QAT model's own fp32 weights measures precision degradation alone, not the cost of the recipe.

**Priority order for further work (Andrew, 2026-11-20): (1) smaller state size, (2) lower log loss
against the identical unquantized run, (3) speed** - a 10% slowdown is acceptable.

## Fixed basis

| item | value |
|---|---|
| anchor checkpoint (row 0) | `scratchpad/quant_plan/qat_it36ctl/ctl_124.pth` - the 124-step fp32 decay from `rwkv_iter36_558.pth` (iter36 WS-final) |
| training data | `F:/rwkv_lmdb/train_db_5k_h1_adaptive92` restricted to users 1-100, `MAX_TRAIN_GLOBAL_LEN=66000` (160 groups/epoch, 0 chunks dropped, 9.66M rows). The original users-1-100 databases are gone from this machine |
| recipe | `scratchpad/quant_plan/qat_decay.py`: `RWKV_MAX_STEPS=124` (the LR anneals over exactly those 124 steps), `EPOCHS=2`, `PEAK_LR=0.001`, WD 0.01, TRAIN_MODE D, `RWKV_NO_JIT=1`, no JIT |
| roster | `scratchpad/quant_plan/roster750_seed20261119.json` - 750 random VAL users from 5001-7500, seed 20261119 |
| eval | `optimization/eval_sharded.py` + `rwkv/get_result_config_base5k.toml`, canonical merged files `result/adw_<tag>.jsonl` (ahead) and `result/adwP_<tag>.jsonl` (imm) |
| simulated deploy | the WKV state is round-tripped to the quantized format at EVERY step of the forward pass (Python fake-quant, the mirror of the Rust engine's per-step compression) |
| comparison | `scratchpad/quant_plan/compare_tags.py` - per-user paired mean delta, one-sided 95% upper bound, one-sided Wilcoxon p for "worse" |
| model geometry | 192,800 params, 92 inputs, H=1 K=32, card 1 layer, note 3 layers; per layer the state is 1024 WKV floats + 2 x 32 token-shift floats = 1088 floats |

## The bars

| constraint | card | note |
|---|---|---|
| 2: half of naive int4 (`int4` = plain fp32->int4 of the whole state) | state 1088 floats -> int4 4352 bits -> **bar 2176 bits = 272 B** | state 3264 floats -> int4 13056 bits -> **bar 6528 bits = 816 B** |
| 3: log loss, judged against the identical unquantized run | both heads <= +**0.0005** | same |
| 6: speed | Rust rev/s >= **0.75x** unquantized | same |

Bit counts are the per-step persisted payload (WKV matrix + 2 token-shift vectors), excluding the
per-step scale, which every scheme needs.

## Results (against the identical unquantized run)

`d` columns are paired per-user means against **row 0** - the identical fp32 decay with quantization
OFF (row 8 against its own twin, row 0b; rows 11 and 15-18 against their own same-length twins). **That
is the gate as Andrew defined it (2026-11-21): fp32 + no QAT + no quantization versus QAT + quantized.**
A quantizer-trained checkpoint evaluated with the quantizer OFF (the "twin") is an ATTRIBUTION tool
only; it is never the comparison number.
A row passes only if BOTH heads are <= +0.0005. Speed is per config (it does not depend on the
checkpoint) and comes from the deploy engine table below.

| Iter | Method | card bits (B) | note bits (B) | speed: Rust rev/s | speed: x unquant | ahead LogLoss | imm LogLoss | d ahead | d imm | verdict | corruption |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0 | **ANCHOR**: the identical 124-step decay, NO quantization (fp32 state) | 34,816 (4,352) | 104,448 (13,056) | 2,863 | 1.00x | 0.333802 | 0.306211 | - | - | reference, n=750, 0 OOM, 0 NaN | **PASS** (150 p(recall) tensors, 0 non-finite; no quantized stream in this config, so only p(recall) is observable - the attribution control) |
| 0b | **ANCHOR for row 8**: the identical 496-step decay, NO quantization (fp32 state), tag `qnj_it36ctl496` | 34,816 (4,352) | 104,448 (13,056) | 2,863 | 1.00x | 0.321659 | 0.294614 | - | - | reference for row 8 only; it is 0.012143/0.011598 BETTER than row 0, so its d columns are not comparable to rows 1-7 | - |
| 1 | PTQ, no training: card full-matrix int2 + note rank-1 int8 applied to the anchor checkpoint | 2,176 (272) | 3,072 (384) | 2,586 | 0.903x | 0.335225 | 0.308888 | +0.001424 | +0.002676 | **FAIL**, 2.8x/5.4x over, ub95 +0.001586/+0.002928 | - |
| 2 | QAT both streams full-matrix int2 (the naive at-the-bar scheme) | 2,176 (272) | 6,528 (816) | 2,628 | 0.918x | 0.337559 | 0.311924 | +0.003757 | +0.005712 | **FAIL**, 7.5x/11.4x over, ub95 +0.004026/+0.006157 | - |
| 3 | QAT card full-matrix int2 + note rank-1 int8 (Andrew's first pick) | 2,176 (272) | 3,072 (384) | 2,586 | 0.903x | 0.335038 | 0.307253 | +0.001236 | +0.001042 | **FAIL**, 2.5x/2.1x over, ub95 +0.001375/+0.001205 | - |
| 4 | **QAT rank-1 int8 factors + int8 shifts, card and note - BEST SO FAR** | **1,024 (128)** | 3,072 (384) | 2,586 | 0.903x | **0.334305** | **0.307221** | **+0.000503** | **+0.001009** | **FAIL by a hair**: ahead is 0.0000035 over the ceiling (ub95 +0.000607), imm 2x over (ub95 +0.001122) | - |
| 5 | QAT rank-1 int4 factors + int4 shifts, card and note (HALF of row 4's bits) | 512 (64) | 1,536 (192) | 2,586 | 0.903x | 0.334852 | 0.307894 | +0.001050 | +0.001682 | **FAIL**, 2.1x/3.4x over, ub95 +0.001164/+0.001841, p 2e-63/3.1e-75 | - |
| 6 | QAT rank-1 int4 factors + int2 shifts | 384 (48) | 1,152 (144) | 2,586 | 0.903x | 0.335247 | 0.308491 | +0.001445 | +0.002280 | **FAIL**, 2.9x/4.6x over, ub95 +0.001592/+0.002464, p 3.0e-66/1.9e-88 | - |
| 7 | QAT rank-1 int2 factors + int2 shifts (QUARTER of row 4) | 256 (32) | 768 (96) | 2,586 | 0.903x | 0.337406 | 0.311143 | +0.003604 | +0.004932 | **FAIL**, 7.2x/9.9x over, ub95 +0.003871/+0.005296, p 9.7e-93/1.8e-93 | - |
| 8 | QAT rank-1 int8 at iter45's 496-step decay length (axis 2, same bits as row 4) | 1,024 (128) | 3,072 (384) | 2,586 | 0.903x | 0.322791 | 0.296287 | +0.001132 | +0.001674 | **FAIL**, 2.3x/3.3x over, but **d is against its own 496-step twin (row 0b)**, ub95 +0.001254/+0.001796 | - |
| 9 | **QAT rank-1 int8 factors + int4 shifts: the bits belong in the FACTORS - BEST on axis 1** | **768 (96)** | **2,304 (288)** | 2,586 | 0.903x | **0.334318** | **0.307241** | **+0.000516** | **+0.001030** | **FAIL by a hair**, but 25% fewer bits than row 4 at the SAME tax (ub95 +0.000620/+0.001144, p 8.4e-31/8.9e-54) | - |
| 10 | QAT rank-1 int4 factors + int8 shifts: the same 768/2,304 bits moved to the SHIFTS | 768 (96) | 2,304 (288) | 2,586 | 0.903x | 0.334847 | 0.307873 | +0.001045 | +0.001661 | **FAIL**, 2.1x/3.3x over - so the factors need int8 and the shifts do not | - |
| 11 | QAT rank-1 int8 at a 992-step decay (8x the champion's 124 steps, relaxed rule) | 1,024 (128) | 3,072 (384) | 2,586 | 0.903x | 0.318088 | 0.291307 | +0.000874 | +0.001675 | **FAIL**, vs its own 992-step twin `qnj_it36ctl992` | - |
| 12 | QAT rank-1 int4 at a 992-step decay | 512 (64) | 1,536 (192) | 2,586 | 0.903x | *not run* | *not run* | - | - | chain 4's own guard: the 496-step `l1i4` tax (+0.001581) was worse than the 124-step one (+0.001050) | - |
| 13 | QAT rank-1 int4 at a 496-step decay (same length as row 8, half the bits) | 512 (64) | 1,536 (192) | 2,586 | 0.903x | 0.323240 | 0.296663 | +0.001581 | +0.002050 | **FAIL**, vs the 496-step twin (row 0b) | - |
| 14 | QAT rank-1 int4 factors + int2 shifts at a 496-step decay | 384 (48) | 1,152 (144) | 2,586 | 0.903x | 0.323603 | 0.297312 | +0.001944 | +0.002698 | **FAIL**, vs the 496-step twin, 3.9x/5.4x over | - |
| 15 | **QAT card rank-2 int8 + note rank-1 int8 - THE FIRST PASSING RECIPE** | **1,536 (192)** | **3,072 (384)** | **2,324** | **0.812x** | **0.333848** | **0.306518** | **+0.000046** | **+0.000307** | **PASS**: ub95 +0.000112/+0.000369, both heads inside the ceiling, and the deploy engine runs it at 0.812x (>= 0.75x) | **PASS** (158,986 state calls, 150 p(recall) tensors, 0 non-finite, 4.7 min on the 50-user roster) |
| 16 | **QAT card rank-3 int8 + note rank-1 int8 - the lowest tax of any row** | **2,048 (256)** | **3,072 (384)** | *see note* | *see note* | **0.333808** | **0.306388** | **+0.000006** | **+0.000176** | **PASS**: ub95 +0.000070/+0.000234; the ahead head is at zero and the immediate head is 3x inside the ceiling | **PASS** (158,986 state calls, 150 p(recall) tensors, 0 non-finite, 4.8 min on the 50-user roster) |
| 17 | **QAT card rank-2 int8 factors + int4 shifts, note rank-1 int8 + int4 shifts - BEST on axis 1** | **1,280 (160)** | **2,304 (288)** | **2,290** | **0.800x** | **0.333853** | **0.306539** | **+0.000052** | **+0.000327** | **PASS**: ub95 +0.000118/+0.000390, the row-15 tax for 22% fewer bits (3,584 vs 4,608) | **PASS** (158,986 state calls, 150 p(recall) tensors, 0 non-finite, 4.8 min on the 50-user roster) |
| 18 | QAT card rank-3 int8 + int4 shifts, note rank-1 int8 + int4 shifts | 1,792 (224) | 2,304 (288) | 2,290 | 0.800x | 0.333821 | 0.306422 | +0.000019 | +0.000210 | **PASS**: ub95 +0.000082/+0.000268, 256 fewer card bits than row 16 at the same int4 shifts (vs row 0) | **PASS** (158,986 state calls, 150 p(recall) tensors, 0 non-finite, 4.7 min on the 50-user roster) |
| 19 | QAT card at row 4's exact config (rank-1 int8 + int8 shifts), note rank-1 int4 factors + int4 shifts - the note-pool cut | 1,024 (128) | 1,536 (192) | 2,586 | 0.903x | 0.334524 | 0.307559 | +0.000722 | +0.001348 | **FAIL**, 1.4x/2.7x over, ub95 +0.000821/+0.001465: cutting the note from int8 to int4 costs about +0.00022 ahead / +0.00034 imm on top of row 4 (difference of deltas, not a paired test) | - |
| 20 | QAT card at row 4's exact config (rank-1 int8 + int8 shifts), note rank-1 int4 factors + **int2** shifts - the note-pool cut, second step | 1,024 (128) | 1,152 (144) | 2,586 | 0.903x | 0.334648 | 0.307968 | +0.000846 | +0.001756 | **FAIL**, 1.7x/3.5x over, ub95 +0.000955/+0.001892, p 2.7e-51/8.9e-87: the note int2 shifts cost another +0.000124 ahead / +0.000408 imm over row 19 for only 384 bits saved | - |
| 21 | QAT card rank-2 int8 + int4 shifts, note rank-1 int4 factors + int4 shifts - the combined cut | 1,280 (160) | 1,536 (192) | 2,290 | 0.800x | 0.334130 | 0.306850 | +0.000328 | +0.000638 | **FAIL by 0.00014**: ahead is 0.65x of the ceiling (ub95 +0.000397), imm is 1.3x over (ub95 +0.000711); the note cut in l2i8s4n4 costs about +0.000276 ahead / +0.000311 imm on top of row 17 at 768 fewer note bits | - |
| 22 | QAT card rank-3 int8 + int4 shifts (row 18's card), note rank-1 int4 factors + int4 shifts - the note cut on the low-tax card | 1,792 (224) | 1,536 (192) | 2,290 | 0.800x | 0.334099 | 0.306737 | +0.000297 | +0.000525 | **FAIL by 0.000025**: the immediate head is 1.05x over the ceiling (ub95 +0.000594) while ahead sits at 0.6x of it (ub95 +0.000366), so the int4 note is the blocker; against row 18 the 768-bit note cut costs about +0.000278 ahead / +0.000315 imm (difference of deltas, not a paired test) | - |
| 23 | QAT card **rank-2 int4** factors + int4 shifts, note rank-1 int8 + int4 shifts - rank 2 at the cheap precision, the SAME 768/2,304 bits as row 9 | 768 (96) | 2,304 (288) | 2,290 | 0.800x | 0.334068 | 0.306792 | +0.000266 | +0.000580 | **FAIL**, imm is 1.16x over the ceiling (ub95 +0.000337/+0.000661, p 6.9e-16/1.5e-38): at EQUAL bits rank-2 int4 beats rank-1 int8 (row 9) on both heads by about 0.00025/0.00045, so extra RANK is worth more than extra factor precision - but the imm head still needs the 2,304-bit note | - |
| 24 | QAT card **rank-4 int4** factors + int4 shifts, note rank-1 int8 + int4 shifts - the int4 rank ladder, first step (`l4i4s4`) | 1,280 (160) | 2,304 (288) | *see note* | *see note* | 0.334019 | 0.306674 | +0.000218 | +0.000463 | **FAIL**: imm is 1.08x over the ceiling (ub95 +0.000287/+0.000538); at the SAME 1,280 card bits it is +0.000166/+0.000136 worse than row 17 (rank-2 int8), so int4 factors give back what the extra rank buys (difference of deltas, not a paired test) | - |
| 25 | QAT card **rank-7 int4** factors + int4 shifts, note rank-1 int8 + int4 shifts - the int4 rank ladder, second step (`l7i4s4`) | 2,048 (256) | 2,304 (288) | *see note* | *see note* | 0.334011 | 0.306678 | +0.000209 | +0.000466 | **FAIL**: imm is 1.08x over the ceiling (ub95 +0.000279/+0.000542); at the SAME 2,048 card bits it is +0.000203/+0.000290 worse than row 16 (rank-3 int8), so past rank 2 the FACTOR PRECISION is the binding constraint, not rank (difference of deltas, not a paired test) | - |
| 26 | QAT card rank-2 int8 + int4 shifts (row 17's card), note **rank-2 int4** factors + int4 shifts - the note's own rank-vs-precision swap at row 17's exact 1,280/2,304 bits (`n2i4s4`) | 1,280 (160) | 2,304 (288) | *see note* | *see note* | 0.333982 | 0.306703 | +0.000180 | +0.000492 | **FAIL by a hair**: both point estimates are inside the ceiling but the immediate head's ub95 +0.000543 is 1.09x over the +0.0005 bound (ahead ub95 +0.000227); at the SAME 3,584 bits as row 17 it is +0.000128 ahead / +0.000165 imm worse (difference of deltas, not a paired test), the OPPOSITE of the card, where row 23 showed rank-2 int4 beating rank-1 int8 - so on the 3-layer note FACTOR PRECISION beats RANK | - |
| 27 | QAT card rank-2 int8 + note rank-1 int8 at a 496-step decay - row 15's recipe at row 8's length (`l2i8`) | 1,536 (192) | 3,072 (384) | 2,324 | 0.812x | 0.322374 | 0.295406 | +0.000715 | +0.000793 | **FAIL**, against its own 496-step twin `qnj_it36ctl496` (row 0b): ahead is 1.43x and imm 1.59x over the ceiling, ub95 +0.000802/+0.000870, p 8.3e-50/1.7e-63 - the longer decay does not remove the rank-1 note's tax | - |
| 28 | QAT card rank-2 int8 + note rank-1 int8 at a 992-step decay - row 15's recipe at row 11's length (`l2i8`) | 1,536 (192) | 3,072 (384) | 2,324 | 0.812x | 0.317768 | 0.290460 | +0.000554 | +0.000829 | **FAIL**, against its own 992-step twin `qnj_it36ctl992` (row 11's control): ahead is 1.1x and imm 1.66x over the ceiling, ub95 +0.000660/+0.000916, p 8.7e-30/3.7e-67 | - |
| 29 | QAT card rank-2 int8 + int4 shifts (row 17's card), note **rank-2 int8** + int4 shifts - the note's own rank ladder, first step (`l2i8n2i8`) | 1,280 (160) | 3,840 (480) | 2,429 | 0.803x | 0.333687 | 0.306314 | -0.000115 | +0.000103 | **PASS on the loss gate**: ahead is BETTER than the anchor by 0.000115 (ub95 -0.000084, p 1.0000) and imm sits at 0.21x of the ceiling (ub95 +0.000127, p 3.3e-15); the engine's rank-2 NOTE path reads 0.600x on the REFERENCE solver but **0.803x with the fast rank-r solver** (`RWKV_LOWRANK_FAST=1`, states/s 0.688x, median per-user 0.810x, users 107/136/156, `rust_speed_fastrank_{on,off}.json`), so it clears the 0.75x floor; the fast solver's own deploy cost is imm +0.000051 / ahead +0.000000 on the same traces (| **PASS** (158,986 state calls, 150 p(recall) tensors, 0 non-finite, 5.2 min on the 50-user roster) |
| 30 | QAT card **rank-3 int8** + int4 shifts (row 18's card), note **rank-2 int8** + int4 shifts - the note's own rank ladder, second step (`l3i8n2i8`) | 1,792 (224) | 3,840 (480) | 2,393 | 0.791x | 0.333658 | 0.306198 | -0.000144 | -0.000014 | **PASS on BOTH heads, and the lowest-tax row measured so far**: ahead is BETTER than the anchor by 0.000144 (ub95 -0.000121, p_worse 1.0000) and the immediate head's point estimate is 0.000014 BELOW the anchor (ub95 +0.000002 - the 95% bound sits on the anchor, p_worse 0.851, i.e. no evidence of worsening), so this is the first row whose imm head is not measurably worse than the unquantized-equivalent run at all; it spends 448 more card bits than row 29 (1,792 vs 1,280) for 0.000029/0.000117 better on the two heads, and its rank-2 NOTE path clears the speed floor at 0.791x with the fast rank-r solver (`rust_speed_fastrank_on.json`; | **PASS** (158,986 state calls, 150 p(recall) tensors, 0 non-finite, 5.3 min on the 50-user roster) |
| 31 | QAT card rank-2 int8 + int4 shifts (row 17's card), note **rank-2 int8 + int8 shifts** - is the note's SHIFT precision free at rank 2? (`l2i8n2i8s8`) | 1,280 (160) | 4,608 (576) | n/a | n/a | 0.333683 | 0.306295 | -0.000119 | +0.000084 | **PASS on both heads, and the lowest imm tax of the note-rank-2 family**: ahead is BETTER than the anchor by 0.000119 (ub95 -0.000088, p_worse 1.0000) and imm is 0.000084 above it (ub95 +0.000108, p_worse 7.2e-11) - but it spends 768 MORE note bits than row 29 (4,608 vs 3,840) to buy 0.000019 of imm, so an int8 note shift level is not free. **Deployable on the speed axis since 2026-11-22**: `RWKV_STATE_SHIFT_LEVEL` is now a PER-STREAM map (`CompressCfg::shift_level` in `rust/rwkv-infer/src/model.rs`, GPU_RUN_BOARD Amendment 23), so `card:int4,note:int8` is expressible; when this row was recorded the option took ONE bare level and a per-stream string was silently dropped, after which the shifts followed the WKV FACTOR level. What is still missing is its deploy SPEED: the rank-2 note path needs `RWKV_LOWRANK_FAST=1`, which reads 0.803x on the neighbouring `l2i8n2i8` config, and at 5,888 bits the row is a loss instrument rather than an axis-1 candidate | **PASS** (158,986 state calls, 150 p(recall) tensors, 0 non-finite, 5.0 min on the 50-user roster) |
| 32 | QAT card rank-2 int8 + int4 shifts (row 17's card), note **rank-3 int8** + int4 shifts - the note's rank ladder, last rung (`l2i8n3i8`) | 1,280 (160) | 5,376 (672) | n/a | n/a | 0.333817 | 0.306334 | +0.000015 | +0.000122 | **PASS on both heads, and it CLOSES the note's rank ladder**: against row 29 (`l2i8n2i8`, the same card) rank 3 is WORSE on BOTH heads (-0.000115 ahead / +0.000103 imm at rank 2) while spending 1,536 MORE note bits (5,376 vs 3,840), and against row 30 (rank 2 on a rank-3 card) it is worse still. The note's residual tax is therefore NOT rank-limited - extra note rank cannot buy it back - so the zero-bit levers (E1 tail penalty, chains 15/16) are the right next move rather than more note rank. Speed n/a: a rank-3 note needs `RWKV_LOWRANK_FAST=1` (0.791x at rank 2 on a rank-3 card, row 30), | **PASS** (158,986 state calls, 150 p(recall) tensors, 0 non-finite, 5.1 min on the 50-user roster) |
| 33 | QAT card **rank-3 int4** factors + int4 shifts, note rank-1 int8 + int4 shifts (1,024/2,304 = 3,328 bits) - the int4-rank ladder's last rung with room to move (`l3i4s4`, `it36equalbits`, landed 17:16) | 1,024 (128) | 2,304 (288) | 2,290 | 0.800x | 0.334010 | 0.306683 | +0.000208 | +0.000472 | **FAIL**: the immediate head is 1.10x over the ceiling (ub95 +0.000549, p 4.5e-28) while ahead sits at 0.56x of it (ub95 +0.000278), so the int4-rank curve plateaus about 0.00047 imm short and RANK cannot buy back int4 factor precision; it does beat the rank-2 int4 point (row 23, +0.000266/+0.000580 at 768 card bits) by about 0.00006/0.00011 for 256 extra card bits, so 1,024 is the cheapest card of the int4 ladder. Speed cell = the measured rank-3 card path (`l3i8s4`, 2,290 rev/s = 0.800x, as row 18): the factor width does not change the card's verify path | - |
| 34 | QAT card rank-2 int8 + **int2** shifts, note rank-1 int8 + int4 shifts (1,152/2,304 = 3,456 bits) - the card-shift cut of row 15 (`l2i8s2`, `it36equalbits`, landed 17:47) | 1,152 (144) | 2,304 (288) | 2,358 | 0.809x | 0.334140 | 0.306659 | +0.000338 | +0.000448 | **FAIL**: the immediate head is 1.06x over the ceiling (ub95 +0.000532, p 1.6e-26) while ahead is inside at 0.86x of it (ub95 +0.000432), so cutting the card's shifts int8 -> int2 costs about +0.00029 ahead / +0.00014 imm on top of row 15 (difference of deltas, not a paired test) and about +0.00029/+0.00012 on top of row 17 for only 128 fewer bits - the case where rank beats precision again, since row 23's int4-factor card at 768 bits is better on both heads. Speed cell = the measured rank-2 card path (`l2i8`, 2,357.8 rev/s = 0.809x, `rust_speed_idle_l2i8.json`); this row's per-stream shift levels are not expressible in the engine (row 31's `RWKV_STATE_SHIFT_LEVEL` finding), so a deploy run would take one bare level | - |
| 35 | QAT card rank-2 int8 + int4 shifts (row 17's card), note rank-1 int8 + **int3** shifts (1,280/2,112 = 3,392 bits) - the note's SHIFT vectors a half-step below the working int4 (`n1i8s3`, `it36equalbits`, landed 18:17) | 1,280 (160) | 2,112 (264) | 2,358 | 0.809x | 0.333865 | 0.306575 | +0.000063 | +0.000364 | **PASS**: both heads inside the +0.0005 ceiling with their upper bounds (ahead 0.26x of it, ub95 +0.000129; imm 0.73x, ub95 +0.000426, p 7.4e-29, so the imm rise is REAL but small). The note's third shift bit is nearly free: about +0.000011 ahead / +0.000037 imm on top of row 17's int4 note shifts (difference of deltas, not a paired test) for 192 fewer note bits, i.e. far cheaper per bit than the card-shift cut of row 34 - and the note's int2 step (row 20, +0.000408 imm) is where the note's shift precision actually breaks | `pending` |
| 36 | QAT card **rank-3 int4** factors + int4 shifts (row 33's card), note rank-1 **int4** factors + int4 shifts - the int4 note on top of the int4 card, 1,024/1,536 = 2,560 bits (`l3i4s4n4`, `it36shrink`, landed 20:04:49; speed cell = row 33's measured card path, because the factor and shift widths do not change the verify path) | 1,024 (128) | 1,536 (192) | 2,290 | 0.800x | 0.334353 | 0.307104 | +0.000551 | +0.000892 | **FAIL**: the immediate head is 1.78x over the ceiling (ub95 +0.000997, p 4.3e-64) and ahead 1.10x over it (ub95 +0.000634, p 1.4e-34); the int4 note costs about +0.00034 ahead / +0.00042 imm on top of row 33's int8 note at the SAME 1,024 card bits (difference of deltas, not a paired test), so the int4-rank card cannot carry the note-pool cut either - at 2,560 bits it lands 0.00089 imm short, i.e. 1,024 bits below row 17 buys nothing | - |
| 37 | QAT card rank-2 int8 + int4 shifts (row 17's card), note **rank-2 int4** factors + **int3** shifts - the same 2,112 note bits as row 35 spent on RANK instead of factor precision, 1,280/2,112 = 3,392 bits (`n2i4s3`, `it36equalbits`, landed 19:09:39; speed cell = the fast-solver rank-2 note path of row 29, and the per-stream int3 note shift level alone is still not expressible in the engine - row 35's `RWKV_STATE_SHIFT_LEVEL` finding) | 1,280 (160) | 2,112 (264) | 2,429 | 0.803x | 0.333999 | 0.306721 | +0.000197 | +0.000509 | **FAIL by 0.000009**: the immediate head's point estimate is 1.02x over the ceiling and its ub95 +0.000561 is 1.12x over, while ahead sits at 0.39x of it (ub95 +0.000242); at the SAME 3,392 bits as row 35 the rank-2 int4 note is about +0.000134 ahead / +0.000145 imm worse (difference of deltas, not a paired test), so on the 3-layer note FACTOR PRECISION beats RANK a second time (row 26 showed it first) and the note's int3 shift rung does not rescue rank 2 | - |
| 38 | QAT card rank-2 int8 + int4 shifts (row 17's card), note **rank-1 int3** factors + int4 shifts - the note's int3 FACTOR rung, the last precision step above int2, 1,280/1,344 = 2,624 bits (`l2i8s4n3`, `it36shrink`, landed 20:35:29; speed cell = row 17's measured `l2i8s4` card path, because the note stays rank 1 and its factor width does not change the verify path, and both streams use int4 shifts, so row 35's per-stream shift-level limitation does not apply here) | 1,280 (160) | 1,344 (168) | 2,290 | 0.800x | 0.334383 | 0.307240 | +0.000581 | +0.001028 | **FAIL**: the immediate head is 2.06x over the ceiling on its point estimate and 2.26x on its ub95 (+0.001129, p 8.4e-75), ahead 1.16x / 1.34x (+0.000671, p 2.6e-36); against row 17 (the identical card, note rank-1 int8 + int4 shifts at 2,304 bits) the 960-bit note factor cut costs about +0.00053 ahead / +0.00070 imm (difference of deltas, not a paired test), whereas row 35's int3 SHIFT rung on the same int8 note cost only +0.000011/+0.000037 - so on the note the FACTOR width is the binding constraint and int3 is one step too far | - |
| 39 | QAT card rank-2 int8 + **int2** shifts, note rank-1 **int4** factors + int4 shifts (1,152/1,536 = 2,688 bits) - the card's int2 shift rung combined with the int4 note (`l2i8s2n4`, `it36shrink`, landed 21:06:17; speed cell = the measured rank-2 card path `l2i8`, 2,358 rev/s = 0.809x, as row 34 - the shift width does not change the verify path) | 1,152 (144) | 1,536 (192) | 2,358 | 0.809x | 0.334426 | 0.307009 | +0.000625 | +0.000798 | **FAIL on BOTH heads**: the immediate head is 1.60x over the ceiling (ub95 +0.000893, p 3.5e-57) and ahead 1.25x over it (ub95 +0.000724, p 2.1e-31), so the int4 note and the int2 card shifts add up - neither cut is affordable: against row 21 (the SAME int4 note, card at int4 shifts and 128 more card bits) the card's int4 -> int2 shift cut costs about +0.00030 ahead / +0.00016 imm, the same order as the +0.00029 ahead the int8 -> int2 card shift cut cost on row 34, so the card's shifts break at int2 whatever the note does | - |
| 40 | QAT card **rank-3 int8** + int4 shifts (row 18's card), note rank-1 **int4** factors + **int2** shifts (1,792/1,152 = 2,944 bits) - the note's int2 SHIFT rung on top of the int4 note (`l3i8s4n4s2`, `it36shrink`, landed 21:37:36; speed cell = row 18's measured card path `l3i8s4`, 2,290 rev/s = 0.800x, because the rank-3 card sets the number and the note's shift width does not change its rank-1 verify path) | 1,792 (224) | 1,152 (144) | 2,290 | 0.800x | 0.334141 | 0.307093 | +0.000340 | +0.000882 | **FAIL**: the immediate head is 1.93x over the ceiling (ub95 +0.000967, p 3.8e-75) and ahead 0.84x of it (ub95 +0.000421, p 9.3e-19); against row 22 (the SAME card, the int4 note with int4 shifts, 1,536 note bits) the note's int4 -> int2 shift cut costs about +0.00004 ahead / +0.00036 imm for 384 fewer note bits (difference of deltas, not a paired test), so the note's shifts are already at their int4 floor once the factors are int4 - the same conclusion row 20 reached on the rank-1 int8 note | - |
| 41 | QAT card **rank-3 int4** factors + int4 shifts, note rank-1 int8 + **int3** shifts - the cheap card under the cheap note, 1,024/2,112 = 3,136 bits (`l3i4s4n8s3`, `it36combo`, landed 22:11; speed cell = row 33's measured card path `l3i4s4`, because the card is identical and the note's shift width does not change its rank-1 verify path) | 1,024 (128) | 2,112 (264) | 2,290 | 0.800x | 0.334034 | 0.306727 | +0.000232 | +0.000516 | **FAIL by 0.000016 on the immediate head**: ahead sits at 0.46x of the ceiling (ub95 +0.000301, p 4.8e-15) but the imm point estimate is +0.000516 with ub95 +0.000589 (p 1.6e-37). It is the smallest total that has come close - 256 BELOW row 35, the smallest passing row - and the two cuts compose SUBLINEARLY: the card alone costs +0.000208/+0.000472 (row 33, the same card with an int8 note) and the note's int3 shift rung alone +0.000011/+0.000037 (row 35, the same rung on row 17's card), yet together they cost +0.000232/+0.000516. The 1,024 card bits are the expensive half though: the int6 rank-2 card reaches the SAME 1,024 bits for +0.000052/+0.000115 on 11 long histories (`l2i6c`), which is what chain 21 (`l2i6s4n1i8s3`, this exact 3,136 bits) tests | - |

Rows 5-7, 9-11, 13-14, 19 and 20 all use the rank-1 path, whose engine speed was measured once (`l1i4` and
`l1i8` both 2,586 rev/s = 0.903x); the factor/shift bit widths do not change that path, so their speed
cells are that measurement rather than a per-candidate run. Row 21 uses row 17's exact card config (`l2i8s4`, 2,290 rev/s = 0.800x); only its note factor width differs (int4 vs int8), which does not change the path. Row 22 uses row 18's card config (`l3i8s4`, 2,290 rev/s = 0.800x); its note is the faster rank-1 path, so the card sets the number. Rows 24-25 (`l4i4s4`, `l7i4s4`) have NO deploy-engine run: the closest measured rank>=2 card paths are 2,324 rev/s = 0.812x (rank-2 int8, row 15) and 2,290 rev/s = 0.800x (rank-2/3 int8 with int4 shifts, rows 17-18), all above the 0.75x floor, but a `card:4:int4` / `card:7:int4` path has never been measured on the engine, and both rows fail the loss gate, so it was not measured. Row 26 (`n2i4s4`) reuses row 17's exact card config (`l2i8s4`, 2,290 rev/s = 0.800x) but moves the rank 2 to the NOTE. The engine PARSES and RUNS that path: a rank-2 note measures 1,743.0 rev/s = 0.600x and rank-3 1,686.8 = 0.581x (`rust_speed_probe_note2.json`, user 107, 1 rep, while a decay trained; `l2i8s4` read 0.783x in that same probe, so the idle figure is about 0.62x), both FAILING the 0.75x floor, because `compress_wkv_state` uses the rank-1 power iteration at rank 1 and a full KxK Gram eigendecomposition at rank >= 2 - so a rank >= 2 note needed a FAST rank-r engine path, not a parser change - **it has one since 2026-11-22 (E2, `RWKV_LOWRANK_FAST=1`); see the fast-solver section below**. Row 26 fails the loss gate as well, so the engine was not measured for it. Row 29 (`l2i8n2i8`) is the first row with a rank-2 NOTE that passes the loss gate, and with the fast solver it clears the speed floor as well, so it is a deploy candidate once its corruption check passes.

**The rank-2 speed is fine on the CARD; the earlier 0.629x was a both-streams measurement.**
`l2i4` (rank-2 on card AND note, all four note layers) is 0.629x rev/s / 0.448x states/s, which is what
made rank >= 2 look undeployable. Row 15 puts rank 2 on the CARD only - 1 layer, and the cheap stream -
and the deploy engine measures it at **2,323.6 rev/s = 0.812x** (states/s 0.660x, median per-user 0.803x),
i.e. ABOVE the 0.75x floor. It was measured while a QAT decay was training, and the idle-machine re-measurement settles it: `scratchpad/quant_plan/rust_speed_idle_l2i8.json` (2026-11-22 11:47, `it36l2i8_124.safetensors`, users 107/136/156, 2 reps, GPU idle) reads **2,357.8 rev/s = 0.809x** for the rank-2 card path (states/s 0.658x, median per-user 0.795x) against plain 2,915.1 rev/s, so the under-load figure was accurate to about 1% rather than 3% pessimistic.
(`l1i8` reads 0.902x on that idle run, matching the 0.903x it read in the other idle measurement). Every config of that run passes the floor - plain 1.000x, `l1i8` 0.902x, `l2i8` 0.809x, `l2i8s4` 0.809x, `l3i8s4` 0.809x - and the three rank>=2 card configs are indistinguishable from each other, because neither the factor width nor the shift width changes the card's verify path: only the rank does. The constraint-6 verdicts of rows 15-18 stand unchanged. Row 17 (`l2i8s4`) 0.800x and row 18 (`l3i8s4`) 0.800x were measured in the same run. No engine
change is needed ON THE CARD: the deploy engine already parses `card:2:int8`, and the torch eval path mirrors its
rounding (the 3-user deploy grid ranks rank-2 below rank-1 on the immediate head, as the 750-user eval does).

**The rank>=2 NOTE speed blocker is GONE: the fast rank-r solver (E2, 2026-11-22).**
`compress_wkv_state` solved rank >= 2 with a KxK Gram matrix plus a full symmetric eigendecomposition per
layer per step (~35 us of a ~43 us compress call on a 32x32 state, measured by the `compress_bench` test
in `model.rs`), while rank 1 used a power iteration - which is why a rank-2 NOTE read 0.618x rev/s and
rank 1 read 0.902x. `RWKV_LOWRANK_FAST=1` replaces the eigensolve with a warm-started block subspace
iteration on fixed-size arrays (never forming the Gram) plus a Rayleigh-Ritz step on the (R+1)x(R+1)
projection; the tracked block is the R wanted directions plus ONE extra direction, refreshed from the
strongest column of A on every call, so a NEW dominant direction can still enter. Same session, same
weights (`it36l2i8n2i8_124.safetensors`), same users (107/136/156), 2 reps: `l2i8n2i8` **2,429.2 rev/s =
0.803x** (reference solver 1,801.5 = 0.618x) and `l3i8n2i8` **2,393.0 = 0.791x** (reference 1,781.4 =
0.611x), against plain 3,025.1 / 2,915.1 rev/s; the rank-2 CARD configs improve too, 2,357.8 -> 2,544.9
rev/s (0.809x -> 0.841x). Files: `rust_speed_fastrank_on.json`, `rust_speed_fastrank_off.json`.

**The fast solver's own cost is measured, not assumed.** `RWKV_LOWRANK_FAST_CHECK=1` runs BOTH solvers,
records the largest relative Frobenius difference between the two rank-r reconstructions, and still
WRITES the reference factors - so a checked run reproduces the reference run bit for bit: its
`rust_pred_*.json` dumps are byte-identical to the no-flag run on all three configs
(`rust_fastrank_check.json`). The difference itself is concentrated: of 64,132 rank>=2 calls on
`l2i8n2i8`, 61,339 (95.6%) are under 1e-3 relative, 256 under 1e-2, 415 under 1e-1, 57 above 1e-1 and
NONE above 1.0, while 2,065 calls (3.2%) fall back to the reference because the state is degenerate
(`mgs_into` rejects a rank-deficient block), which is cheap on such a state. The largest single-review
p(recall) change is 8.1e-2 (p99 7.1e-3, p999 4.3e-2 over 16,033 immediate-head predictions), so the
MAX is a few outlier reviews rather than a systematic shift. The deploy-relevant number is the scored log
loss on those same traces: `RWKV_LOWRANK_FAST=1` costs **imm +0.000051 / ahead +0.000000** on `l2i8n2i8`
(`l3i8n2i8` +0.000225 / -0.000012; `l2i8s4` -0.000031 / -0.000004), the same order as the quantization
tax it sits next to. It is a DEPLOY-side solver choice only - the 750-user gate numbers come from the
torch eval path, which uses the reference truncation - and `bits_for` ignores the flag, so no bit budget
moves. A finiteness run with the flag on still PASSES (`rust_finite_fastrank.json`: 0 non-finite of
65,671,168 card/note state observations and 29,347 p(recall) values, state_maxabs 0.62). `RWKV_LOWRANK_FAST_ITERS` is the subspace-iteration count and the default 2 was CHOSEN, not assumed: `4` was measured on the same config and is WORSE (scored imm tax +0.000221 vs +0.000051, mean per-review |dp| 4.99e-4 vs 4.74e-4, and a worse relative-Frobenius histogram - 173 calls above 1e-1 vs 57), because a longer iteration drives the block further towards the top-3 invariant subspace of the PREVIOUS state, which is a worse warm start for the current one (`rust_fastrank_iters2.json`). Python<->Rust
PARITY runs must NOT set the flag: the fast factors are not bit-identical to the reference by construction.

**The `corruption` column (added 2026-11-21, Andrew's gate).** A row is deployable only if the quantized recipe produces NO non-finite value anywhere: if even ONE card/note state value or even ONE p(recall) is non-finite, on the users with the longest review histories and with those histories NOT truncated, the recipe FAILS. `pending` = the `it36corrupt` chain had not reached that row yet (it checked the four passing rows 15-18 plus the quantization-OFF fp32 control of row 0, so the check also answers whether a blow-up needs quantization at all); `-` = not checked, because the row fails the log-loss gate and is not a deploy candidate. The verdicts land in `scratchpad/quant_plan/corruption/corruption_corrupt_<tag>.json` and are filled in here. **All five checks finished on 2026-11-22 11:41 and all five PASS**: rows 15, 16, 17 and 18 (158,986 raw card/note state observations each, 0 non-finite, plus 150 p(recall) tensors each, 0 non-finite, on 50 of 50 users in 4.7-4.8 min each) and the fp32 control of row 0 (150 p(recall) tensors, 0 non-finite; its config has no quantized stream, so only p(recall) is observable there - it answers whether a blow-up would need quantization at all, and it does not blow up either). `pending` means the chain has not reached the row yet; row 35 (`n1i8s3`) is the only `pending` cell now, because `it36corrupt2` reaches it first (its earlier points `l2i4s4`/`l3i4s4`/`l2i8s2` are gate-skipped - rows 23, 33 and 34 fail the loss gate).

**The card's rank ladder (rows 15-16) confirms the mechanism and shows the knee.** Against row 0:

| card | row | card bits | d ahead | d imm |
|---|---|---:|---:|---:|
| rank-1 int8 | 4 | 1,024 | +0.000503 | +0.001009 |
| rank-2 int8 | 15 | 1,536 | +0.000046 | +0.000307 |
| rank-3 int8 | 16 | 2,048 | +0.000006 | +0.000176 |

So the immediate head is bought with rank and the curve flattens fast: the first extra rank removes
70% of the tax, the second removes another 43% of what is left. Rank-3 is the axis-2 choice (lowest
tax, +0.000006/+0.000176, so it leaves ~+0.0003 of the imm budget to spend on a smaller NOTE pool),
while rank-2 is the axis-1 choice (1,536 card bits, 512 fewer). Rows 17-18 (int4 shifts) and the
`it36notecut`/`it36rankprec`/`it36shrink` chains then search the (rank, precision, note-level) space
for the smallest state that still passes.

**Status: rows 15-18 PASS all six constraints; rows 19-22, the note-pool cuts, all FAIL; rows 33-41 FAIL too; row 35 (`n1i8s3`) PASSES** (row 41 `l3i4s4n8s3` +0.000232/+0.000516 - the cheap card under the cheap note reaches the smallest total yet at 3,136 bits and misses the imm ceiling by 0.000016, landed 22:08; the held-out fixed-basis screen then rejects the codec NOTE at equal bits, +0.002035/+0.002756 on 7 users) (row 40 `l3i8s4n4s2` +0.000340/+0.000882 - the note's int2 shift rung on the int4 note lands 1.93x over the imm ceiling at 2,944 bits; row 39 `l2i8s2n4` +0.000625/+0.000798 - the int2 card shifts plus the int4 note fail both heads at 2,688 bits, landed 21:06 and 21:37) (row 38 `l2i8s4n3` +0.000581/+0.001028 - the note's factor width cannot drop int8 -> int3: on row 17's own card 960 fewer note bits cost about 0.00053/0.00070, landed 20:35; row 36 `l3i4s4n4` +0.000551/+0.000892 - the int4 note on the int4-rank card lands 0.00089 imm short, so the int4-rank card cannot carry the note-pool cut, landed 20:04; row 37 `n2i4s3` +0.000197/+0.000509 - rank 2 on the note at row 35's own 2,112 bits is 0.000009 over the ceiling, so factor precision beats rank on the note a second time, landed 19:09, both recorded 2026-11-22 20:05) (row 35 `n1i8s3`, card 1,280 / note 2,112 = 3,392 bits, +0.000063/+0.000364, ub95 +0.000129/+0.000426 - the note's shifts can drop int4 -> int3 for 192 fewer bits at a near-zero cost, landed 18:17; row 33 `l3i4s4` +0.000208/+0.000472 - the int4-rank ladder plateaus about 0.00047 imm short of the ceiling; row 34 `l2i8s2` +0.000338/+0.000448 - cutting the card's shifts to int2 is NOT free, 2026-11-22 18:05) (row 18 is the lowest-tax point of the ladder, 1,792 card / 2,304 note bits = 4,096, +0.000019/+0.000210, ub95 +0.000082/+0.000268 at 0.800x; row 17 is the new axis-1 choice, 1,280 card / 2,304 note bits = 3,584,
the row-15 tax for 22% fewer bits at 0.800x; row 15 the same recipe with int8 shifts, 4,608 total; row 16 the axis-2 choice at 2,048 with a near-zero tax). Card 1,280-1,536 bits (2.83x
smaller than naive int4, bar 2,176) and note 2,304-3,072 bits (2.13x-2.83x smaller, bar 6,528); both heads inside the
+0.0005 ceiling with their upper bounds too (row 17 +0.000052 ub95 +0.000118 ahead, +0.000327 ub95 +0.000390
imm; row 15 +0.000046/ub95 +0.000112 and +0.000307/ub95 +0.000369); the 750-user roster; and the deploy engine at 0.800x-0.812x rev/s (>= 0.75x; re-measured on an idle machine at **0.809x** for the rank-2 card path, `rust_speed_idle_l2i8.json`, so the under-load cells are accurate to about 1%). It buys the immediate head
with RANK on the card, exactly what the `cn8` verdict predicted (see "the tax is RANK TRUNCATION"
below). Row 9 stays the best axis-1 rank-1 point (768/2,304 bits at row 4's tax) for reference. Row 22 (`l3i8s4n4`, the note cut on row 18's card) is 0.000025 from passing on the imm head, so an int4 note is not enough: 2,304 note bits (rows 17-18) remain the cheapest passing note.  **Row 23 (`l2i4s4`) FAILS, and it settles one axis-1 question**: putting rank 2 on the card and the card factors at int4 reaches the SAME 768 card / 2,304 note bits as row 9 and scores 0.334068/0.306792 = **+0.000266 ahead / +0.000580 imm** (ub95 +0.000337/+0.000661) - better than row 9's +0.000516/+0.001030 on both heads at identical bits, so at a fixed bit budget rank beats factor precision on the card; the imm head is still 1.16x over the ceiling, so an int8-quality note (2,304 bits) remains part of the cheapest passing set. The `corruption` column is complete: **all four passing rows plus the fp32 control PASS the finiteness check** (finished 2026-11-22 11:41) - each of rows 15-18 produced 158,986 raw state observations and 150 p(recall) tensors on the 50 longest-history users with no non-finite value anywhere, and the quantization-OFF control of row 0 produced 150 clean p(recall) tensors. **Rows 24-25 (`l4i4s4`, `l7i4s4`, the int4-rank ladder from `it36rankprec`) FAIL on the imm head**: +0.000463/+0.000466 (ub95 +0.000538/+0.000542), with ahead inside at +0.000218/+0.000209. Against the int8 rows at the SAME card-bit count (row 17 at 1,280, row 16 at 2,048) they are about +0.00014 to +0.00029 worse on both heads (difference of deltas, not a paired test), so past rank 2 the card's FACTOR PRECISION is the binding constraint. The int4-rank curve plateaus and does not reach the ceiling - rank 2 imm +0.000580 (row 23), rank 4 +0.000463, rank 7 +0.000466 - so the queued `l3i4s4` (rank-3 int4, 1,024 bits, in `it36equalbits`) is the last int4-rank point with room to move, and it has to close about +0.00025 of imm tax in one rank step. **Row 26 (`n2i4s4`) FAILS by a hair**: at row 17's exact 1,280/2,304 bits it moves the rank 2 from the card to the NOTE (rank-2 int4 factors) and scores +0.000180/+0.000492 (ub95 +0.000227/+0.000543), so the imm head's upper bound is 1.09x over the ceiling - the swap is the REVERSE of the card's, i.e. on the 3-layer note FACTOR PRECISION beats RANK, and the int8 note of rows 15-18 stays the cheapest passing note. **Row 27 (`l2i8` at 496 steps) FAILS**: the passing card-rank-2 recipe of row 15 does not carry to a longer decay - 0.322374/0.295406 = +0.000715/+0.000793 against its own 496-step twin `qnj_it36ctl496` (row 0b), ub95 +0.000802/+0.000870 - so the decay-length axis stays closed at 124 steps even with the card-rank recipe. **Row 28 (`l2i8` at 992 steps) FAILS too**: the same card-rank-2 recipe is +0.000554/+0.000829 (ub95 +0.000660/+0.000916) against its own 992-step twin `qnj_it36ctl992`, so the length axis stays closed with rank on the card as well as at rank 1.  **Row 29 (`l2i8n2i8`, the note's rank ladder) PASSES the loss gate and reverses the note-pool result**: with row 17's card held, the note goes to rank-2 int8 + int4 shifts (3,840 bits) and scores 0.333687/0.306314 = **-0.000115/+0.000103** (ub95 -0.000084/+0.000127) - the ahead head is BETTER than the anchor and the imm head is 0.21x of the ceiling, where the int4 note cuts of rows 21-22 sat at 1.05-1.3x over it. So RANK removes the tax on the note as well as on the card, which is what the `cn8` verdict said the tax was. The blocker had moved to the ENGINE: a rank-2 note path reads 0.600x on the reference solver (0.75x floor), so the engine needed a FAST rank-r path, not a parser change - `RWKV_LOWRANK_FAST=1` (E2, 2026-11-22) supplies it and measures 0.803x for this row (see the fast-solver section above), so row 29 is a deploy candidate once its corruption check passes (corruption: pending, `it36corrupt2`).

**Row 30 (`l3i8n2i8`) lands and takes over as the lowest-tax row: the note's rank ladder pays for itself in loss at 5,632 bits.** With row 18's card held (rank-3 int8, 1,792 bits), the note goes to rank-2 int8 + int4 shifts (3,840 bits) and scores 0.333658/0.306198 = **-0.000144 ahead / -0.000014 imm** (ub95 -0.000121/+0.000002, p_worse 1.0000/0.851) against `qnj_it36ctl`. Both point estimates are at or below the anchor and the immediate head's 95% bound sits exactly on it, so this is the first row in the ladder whose immediate head carries no measurable tax at all. Against row 29 (same note width, card rank 2 at 1,280 bits) it is 0.000029 ahead / 0.000117 imm BETTER for 448 more card bits, so the card's rank-3 step buys real immediate-head loss. **But rank 2 does not make the note SMALLER - it makes it less taxed**: at 3,840 note bits it costs 1,536 more bits than the int8 rank-1 note of rows 17-18 (2,304) that already passes, and the equal-bit rank-vs-precision swap on the note was already settled the other way by row 26 (`n2i4s4`, rank-2 int4 at row 17's exact 2,304 note bits) which failed by a hair. So on axis 1 (smallest state) row 29 stays the preference at 5,120 bits and this row is a LOSS instrument: it removes the note's tax, which is the prerequisite for shrinking the note later. Speed 0.791x with the fast rank-r solver (0.611x without, `rust_speed_fastrank_{on,off}.json`), above the 0.75x floor; corruption pending (`it36corrupt2` already carries this tag).

**Row 31 (`l2i8n2i8s8`, chain 14 point 3) passes both heads, and settles that the note's SHIFT precision is not free.** Card rank-2 int8 + int4 shifts (row 17's card) with the note at rank-2 int8 + int8 shifts: 1,280/4,608 = 5,888 bits, ahead 0.333683 (delta -0.000119, ub95 -0.000088) and imm 0.306295 (delta +0.000084, ub95 +0.000108). It is the best imm tax of the note-rank-2 family (row 29 +0.000103, row 30 -0.000014 at rank 3 on the card), but 768 extra note bits buy only 0.000019 of imm. **The row is loss-only until the engine's shift-level parser gains per-stream support**: `RWKV_STATE_SHIFT_LEVEL` accepts one bare `intN`, and a per-stream string is silently ignored (the shifts then follow the WKV factor level). That silent fallback also applies to any earlier recorded row whose config used a per-stream shift string; `c2_nl1i8` is unaffected, because its factor levels (card int2, note int8) are exactly the shift levels it intended.

**The shipped row-17 weights run 16% faster with one deploy flag and no accuracy change, and the cost of cutting factor precision is history-dependent.** Two deploy-side probes on the three LONGEST corrupt50 histories (users 7250 / 5052 / 5939, 28-29k reviews each) settle both. (1) `l2i8s4`, `l2i8s4f` (the E2 fast rank-r solver, `RWKV_LOWRANK_FAST=1`) and `l2i8s4f1` (the same plus `RWKV_LOWRANK_FAST_ITERS=1`) agree to <=3e-6 on BOTH heads - tax -0.003711/-0.003620, -0.003714/-0.003623, -0.003714/-0.003621 - so the approximation does not drift as history grows, and with 0.800x -> 0.927x rev/s (median per-user 0.921x) the same weights, bits and log loss run 16% faster for free. (2) The exponent rungs `l2i6s4` (1,024/1,920 = 2,944 bits) and `l2i5s4` (896/1,728 = 2,624) cost FAR more on long histories than the 2-user short-trace screen suggested: per-user paired imm deltas are +0.000067/+0.000226/+0.000742 (mean +0.000345, std 0.00035) for int6 and +0.000629/+0.001147/+0.002877 (mean +0.001551, std 0.00118) for int5, i.e. the same sign on every user and growing with history length, against +0.000003/+0.000017 on the short traces. Asymmetric factor budgets (`RWKV_LOWRANK_VLEVEL`, V coarsened while U stays int8) are NOT a lever: at the same 640-bit saving V int4 costs +0.001015 (std 0.00118) versus int6's +0.000345, and the VLEVEL spread is ~15x wider, with V int3's negative mean driven by ONE user (5052 reads -0.006228 against +0.001639/+0.003570 elsewhere) - so coarsening V does not add a smooth error, it destabilizes the trajectory.

**Row 32 (`l2i8n3i8`, chain 14 point 4) passes both heads and closes the note's rank ladder at rank 2.** Card rank-2 int8 + int4 shifts (row 17's card) with the note at rank-3 int8 + int4 shifts: 1,280/5,376 = 6,656 bits, ahead 0.333817 (delta +0.000015, ub95 +0.000033, p_worse 0.215) and imm 0.306334 (delta +0.000122, ub95 +0.000142, p_worse 2.0e-30). The decisive comparison is row 29 `l2i8n2i8`, the same card with the note at rank 2 (3,840 note bits): it reads -0.000115 ahead / +0.000103 imm, i.e. BETTER on both heads for 1,536 fewer note bits. Note rank 3 also loses to rank 2 on a rank-3 card (row 30 -0.000144/-0.000014). So on the note, rank 2 dominates rank 3, and the note's remaining rank-1 tax is not a rank-truncation problem that more rank can pay off. That is exactly the premise E1 rests on: measure the tail energy the truncation discards and penalize it, at zero bits and zero deploy cost.

**The factor-precision axis is nearly exhausted: over 21 long-history users an int6 card costs ~2x what a 3-user probe said, for 640 bits.** Chain 17 asks whether the card's factors can drop from int8 to int6 (2,944 total bits, 640 below row 17) or int5 (2,624, 960 below), which had never been TRAINABLE before 2026-11-22 (`rwkv/architecture.py` `_QMAX` lacked int6/int5). Before spending a 750-user QAT run on it, the deploy engine priced the change on the longest histories available as per-user PAIRED deltas against `l2i8s4`, with 2,000 bootstrap resamples (`rust_wide_screen_fast_wide3_int6long_it36int6wide_b4.json`, instrument `corruption/wide_screen.py`, 21 users = the 3 longest plus the 18 the `trace_corrupt50` export had finished). **`l2i6s4` costs +0.000442 ahead [+0.000232, +0.000701] and +0.000726 imm [+0.000420, +0.001100], better on 2/21 users on both heads. `l2i5s4` costs +0.001458 ahead [+0.000826, +0.002251] and +0.002189 imm [+0.001389, +0.003065], better on 1/21.** Both intervals are far clear of zero and the sign is consistent on 19-20 of 21 users, so this is not an outlier effect. Two readings matter. (1) The 3-user read of the same pair (+0.000345/+0.001551 imm) understated int6 by 2.1x, the same trap Amendment 17 recorded - the 3-user mean is one hard user plus two others. (2) The PTQ tax is not the QAT tax: on this sample QAT recovers roughly half of a PTQ tax, but row 17's own QAT imm is already +0.000327 (ub95 +0.000390) against a +0.0005 ceiling, so int6 would have to recover almost the entire +0.0007 PTQ gap to land under it. Chain 17 still runs - a roster of 750 is cheaper than a wrong assumption, and the measured answer is the deliverable - but the expectation is now on record: **int6 is at best a coin flip and int5 is expected to fail.** CORRECTION: the screen's last config (`l2i4s4n4`, the 2,304-bit bottom rung) was NOT measured - that name exists in `quant_sweep`'s candidates but not in `rust_grid`'s CONFIGS, so the batch ended with a KeyError after the four configs above were complete. With int5 at +0.0022 imm, the bottom rung has no plausible path, so it was not re-run.

**The deploy engine's own finiteness guard clears 40 long histories - 7.2 billion state values, zero non-finite.** Andrew's blowup gate was already clean in PYTHON (rows 15-18 and 29-32). The deploy engine has its OWN guard (`RWKV_FINITE_CHECK`, `--strict`, exit 3 on any non-finite value) and it had only ever seen the 3-user parity trace, so the shipped guard had never been exercised at the history lengths the gate is about. The `trace_corrupt50` export reached the driver's minimum of 40 complete traces, and `corruption/corrupt50_finite_run.py --tag corrupt50` ran over all of them: **rc=0**, i.e. the engine itself found nothing. The `l2i8s4` (row 17) config observed 7,053,440 state compressions and 7,222,722,560 state values, 451,420,160 shift-vector values and 3,459,629 p(recall) values, on the RAW and the QUANTIZED side of every compression, with ZERO bad values of any kind (largest |state| 0.553, largest |shift vector| 5.380); `plain` observed 3,459,629 p(recall) values, zero bad, and no state values at all because nothing is quantized there. On histories of up to 95,597 reviews this is roughly 45,000x the Python gate's 158,986 state calls on the same roster, so the two implementations now agree at the lengths that matter. The last 10 traces were still exporting; re-run with `--tag corrupt50_rest` when they land.



**The error-feedback screen closes: EF is real but it cannot shrink the state.** `RWKV_LOWRANK_EF` is implemented in the deploy engine and had never been measured anywhere in this lineage. It is now screened on 21 of the longest corrupt50 histories, as per-user PAIRED deltas against `l2i8s4` inside each batch with 2,000 bootstrap resamples (`rust_wide_screen_ef_b3_ef_b2_ef_b1_fast_wide3_ef3.json`, instrument `corruption/wide_screen.py`). Three readings. (1) **The mechanism works**: at row 17's own config plus EF (`l2i8s4ef`, 1,536/3,072 = 4,608 bits) both heads improve with intervals clear of zero, -0.001167 ahead [-0.002081, -0.000339] and -0.002453 imm [-0.004826, -0.000539], better on 10/15 and 8/15 users. But the error buffer is real state, +1,024 bits, so that is axis 2 - and axis 2 is saturated (rows 29-31 sit at or below their own fp32 twins). (2) **EF cannot pay for itself on axis 1**: the best layout is `l2i4s4n4ef` (both streams int4 + int4 buffer) at 1,024/2,304 = 3,328 bits, 256 BELOW row 17, and it reads STATISTICALLY EQUAL to row 17 on both heads (+0.000049 ahead [-0.000737, +0.000875], -0.000376 imm [-0.001852, +0.000911], better on 10/21 and 6/21). A 3-user read of the same config had shown -0.001310 imm and looked like a 256-bit win; 21 users cut it to -0.000376 and an interval over zero. So EF at int4 factors buys a real but small -7% state at equal accuracy, no more. (3) **Coarsening the buffer removes the benefit**, because the buffer IS the mechanism: `l2i4s4n4ef2` (buffer at int2, 896/1,920 = 2,816 bits, exactly row 21's size) is clearly worse, +0.003439 ahead [+0.000587, +0.007513] and +0.002696 imm [+0.001449, +0.004236], better on 1/6 and 0/6. EF is therefore closed as an axis-1 lever. It also cannot be gated today - `RWKV_LOWRANK_EF` exists only in the Rust engine while the gate is an eval-path measurement - so it needs no decision from Andrew unless a future recipe wants its -0.0025 imm at more bits.

**The int6 tax is mostly the NOTE's: the card-only int6 rung is nearly free.** The 21-user read above priced int6 as a whole (`l2i6s4`, both streams int6, +0.000442 ahead / +0.000726 imm). This screen splits it on 11 long-history users by quantizing ONE stream at int6 and leaving the other at row 17's own level, as per-user PAIRED deltas against `l2i8s4` inside one batch (`rust_wide_screen_it36int6split.json`, instrument `corruption/wide_screen.py`, 2,000 bootstrap resamples; absolutes in `rust_grid_it36int6split.json`). Two readings. (1) **The CARD's int6 is nearly free**: `l2i6c` (card rank-2 int6, note rank-1 int8 + int4 shifts = 1,024/2,304 = 3,328 bits, 256 BELOW row 17) costs +0.000052 [-0.000007, +0.000109] better 2/11 ahead and +0.000115 [+0.000018, +0.000218] better 2/11 imm, so its ahead interval CONTAINS zero and its imm interval only just excludes it - an order of magnitude inside row 17's own +0.000052/+0.000327, for 256 fewer card bits. (2) **The NOTE's int6 carries the rest**: `l2i6n` (card rank-2 int8, note rank-1 int6 = 1,280/1,920 = 3,200 bits, 384 below row 17) costs +0.000268 [+0.000079, +0.000601] better 0/11 ahead and +0.000307 [+0.000124, +0.000558] better 1/11 imm, worse on both heads than the card-only rung for only 128 fewer bits. The two parts sum to about +0.000321 ahead / +0.000422 imm, the same order as the +0.000442/+0.000726 that `l2i6s4` costs on the 21-user set (a decomposition, not an exact sum - the two screens share only 3 users and the note's rank differs). The practical consequence is that the CARD-only int6 rung is the one worth a 750-user QAT point, and chain 18 (`it36int6card`) trains it (`l2i6c`, then its note-only twin `l2i6n` for attribution), both 124 steps against `qnj_it36ctl`. Deploy-side deltas only, so they rank the pair; the 750-user eval decides. Absolutes on the 11 users: plain 0.372026/0.360820, `l2i8s4` 0.368347/0.358356, `l2i6c` 0.368399/0.358471, `l2i6n` 0.368615/0.358663.

**The fixed basis (approach 1) does not transfer across users: it is a CARD loss and a NOTE candidate.** The rank-r scheme stores U_r and V_r, i.e. it re-describes a BASIS at every step. Approach 1 instead stores a FIXED one: with P, Q fixed K x m matrices the payload is the m x m coefficient block C = P^T W Q, so a layer costs m*m*qbits + 64*shift_bits and the deploy work is two small matmuls - cheaper than the Gram/power-iteration compress, so axis 3 gains as well. The basis is a MODEL-side codec object (like a DCT basis): no discrete entries, no nearest-neighbour lookup, no per-user vectors - not a codebook. The measurement that decides it is `corruption/fit_fixedbasis.py` on 1,347 raw pre-truncation states per (stream, layer) from 3 users (7213/7044/6960, every=1, row 17's checkpoint, `state_subspace_l2i8s4all.npz`), fitting P from the top eigenvectors of E[W W^T] and Q from those of E[W^T W]. Scored on the SAME states the curve looks like a win - 0.9935-0.9995 at m=4-8 - but held out per user it collapses:

| key | n | in-sample m=4 | m=8 | m=16 | LOO m=4 | m=8 | m=16 |
|---|---:|---:|---:|---:|---:|---:|---:|
| card L0 | 1347 | 0.9938 | 0.9997 | 1.0000 | 0.7181 | 0.9195 | 0.9596 |
| note L0 | 1347 | 0.9935 | 0.9995 | 1.0000 | 0.8559 | 0.9205 | 0.9604 |
| note L1 | 1347 | 0.9856 | 0.9995 | 1.0000 | 0.7510 | 0.8939 | 0.9659 |
| note L2 | 1347 | 0.9848 | 0.9993 | 1.0000 | 0.7816 | 0.8757 | 0.9647 |
| note ALL (3 layers) | 4041 | 0.8636 | 0.9679 | 0.9992 | 0.5671 | 0.7616 | 0.9220 |

Two consequences. (1) **The CARD is the wrong place for a codec.** Its shipped rank-2 truncation is EXACT (0.9998 of the energy kept) for 1,280 bits, while the best fixed basis needs m=16 - 2,304 bits, 1,024 more than the whole card bar's worth of rank - to reach 0.9596 held out. There is nothing left for a codec to buy there. (2) **The NOTE is the interesting side.** Its shipped rank-1 truncation keeps only 0.904-0.951 of the energy yet PASSES the gate, and a fixed basis holds out 0.9205-0.8757 at m=8 for the SAME 768 bits per layer (m=6: 0.8179-0.8713), so a codec note is worth a real 750-user measurement. The two candidates therefore hold the card at row 17's rank 2 and switch ONLY the note: `l2i8fb8n` (card rank-2 int8 + note fixed basis m=8 int8 = 1,280/2,304 = 3,584 bits, row 17's EXACT size - a pure codec A/B against the shipped rank-1 note) and `l2i8fb6n` (1,280/1,632 = 2,912 bits, 672 BELOW row 17 - the size claim). Both are implemented end to end: the Python side is `fake_fixedbasis_state` in `rwkv/model/rwkv_ops.py` (STE W -> P(quant(P^T W Q))Q^T on the coefficient block, with a per-stream `RWKV_FIXEDBASIS_SCOPE`; 10/10 checks in `corruption/test_fixedbasis.py`) and the deploy side is `fixedbasis_roundtrip` in `rust/rwkv-infer/src/model.rs` plus the branch in `fast.rs::compress_stream_state` (4 unit tests; the whole suite is 15 passed / 1 ignored), driven by `RWKV_STATE_FIXEDBASIS_SCOPE` + `RWKV_FIXEDBASIS_PATH` reading `corruption/fixedbasis_l2i8s4all.txt` (48,601 bytes, 4 entries, one fitted file serves every m because only the first m columns are read). The engine runs the path on real corrupt50 traces (`rust_grid_it36fbsmoke.json`, users 7213/7250, `deploy_iter17_l2i8s4.safetensors`): plain 0.369532/0.363273, `l2i8s4` 0.368217/0.361383, `fb8i8` 0.369105/0.362425, `l2i8fb8n` 0.368729/0.363213. Those are 2-user numbers, so they RANK only (every quantized config reads about 0.0036 BETTER than plain on these long traces, the trace-path offset): at the same 3,584 bits the codec note is +0.000511 ahead / +0.001830 imm against the shipped rank-1 note, i.e. a wash on ahead and slightly better on imm. Chain 19 (`it36fixedbasis`) trains both points (124 steps against `qnj_it36ctl`) and then runs its OWN corruption phase for whichever passes the gate, because it is LAST in run order and `it36corrupt2` cannot pick the rows up.

**Row 41 (`l3i4s4n8s3`) misses the immediate-head ceiling by 0.000016, so 3,136 bits is NOT yet a passing size - but the two cuts compose sublinearly, which is the useful part.** The cheap card (rank-3 int4 + int4 shifts, 1,024 bits) under the cheap note (rank-1 int8 + int3 shifts, 2,112 bits) lands 0.334034/0.306727 = +0.000232/+0.000516 against `qnj_it36ctl` (ub95 +0.000301/+0.000589, p 4.8e-15/1.6e-37): the ahead head is comfortably inside the ceiling at 0.46x of it and the immediate head is 1.03x over on its point estimate. Against the sum of the parts, +0.000219 ahead / +0.000509 imm, the combination is BETTER than additive on both heads - the note's int3 shift rung is nearly free when the card is already coarse. The cheap half is the NOTE, not the card: rank-3 int4 factors are the expensive way to buy the card's last 256 bits (row 33 prices that card at +0.000472 imm), while the int6 rank-2 card reaches the same 1,024 bits for +0.000115 imm on 11 long histories (`l2i6c`, Amendment 22). That is exactly the swap chain 21 (`it36int6combo`, point `l2i6s4n1i8s3`, 1,024 + 2,112 = 3,136 bits, the same total) makes, and it is pre-registered in `quant_sweep.CANDIDATES`, `rust_grid.CONFIGS` and `check_corruption.CONFIGS`. `rust_grid_it36int6shift.json` prices it on the same long histories first.

**The held-out fixed-basis screen rejects the codec NOTE: 4-6x the tax at equal bits, on 0-1 of 7 users.** `fixedbasis_l2i8s4all.npz` was fitted on three users (7213/7044/6960), so a deploy-side screen over 7 OTHER long histories - all held out from the fit - is the fair test of whether the LOO capture table actually costs loss. The engine ran `plain`, `l2i8s4`, `l2i8fb8n` and `l2i8fb6n` on row 17's deploy weights, and `corruption/wide_screen.py` computed per-user PAIRED deltas against `l2i8s4` with 2,000 bootstrap resamples (`rust_wide_screen_it36fbwide_b1.json`; absolutes in `rust_grid_it36fbwide_b1.json`):

| config | card/note bits | total | d ahead (95% CI) | d imm (95% CI) |
|---|---:|---:|---|---|
| `l2i8fb8n` codec note m=8 (row 17's exact size) | 1,280/2,304 | 3,584 | +0.002035 [+0.001001, +0.003044] better 1/7 | +0.002756 [+0.001430, +0.004158] better 0/7 |
| `l2i8fb6n` codec note m=6 (480 below row 35) | 1,280/1,632 | 2,912 | +0.001455 [+0.000604, +0.002400] better 1/7 | +0.002154 [+0.001058, +0.003456] better 0/7 |

Both are 4-6x the +0.0005 ceiling and the sign is consistent - the codec note is BETTER on one user out of 7 on ahead and on NO user on the immediate head - so this is not an outlier effect. It is also the most expensive thing measured on the note so far: the note's int6 factor cut costs +0.000268/+0.000307 and the card's int6 +0.000052/+0.000115 on comparable long-history screens, i.e. a codec buys 672 bits for roughly 7-9x the price of a factor-width cut. That is consistent with the LOO table: the shipped rank-1 truncation keeps 0.904-0.951 of each state's energy while the codec holds out only 0.8757-0.9205 at m=8, and the energy the codec drops is spread across directions the model did not choose. THE ONE REGIME THIS SCREEN CANNOT SEE is QAT: the weights here were trained for the rank-r path, so the codec's cost includes the mismatch between the basis and how the state is actually used, and training can push state energy into a fixed subspace. Chain 19 (`it36fixedbasis`) is therefore still the fair test - `l2i8fb8n` at equal bits measures how much QAT recovers, and `l2i8fb6n` is the size claim at 2,912 bits - but the expectation is now recorded: on this evidence the codec note FAILS unless QAT recovers most of a +0.0020/+0.0028 gap.


**The per-stream shift level makes row 35 deployable, and it unblocks rows 31, 34 and 37 on the speed axis.** `RWKV_STATE_SHIFT_LEVEL` used to take ONE bare level, so a per-stream string such as `card:int4,note:int3` was silently dropped and the shifts fell back to the WKV factor level - which is why rows 31, 34, 35 and 37 were loss-only. It is now a per-stream map (`CompressCfg::shift_level: HashMap<usize, f64>` in `rust/rwkv-infer/src/model.rs`; a bare level still fills all five streams), and row 35 (`n1i8s3`, 1,280/2,112 = 3,392 bits - the SMALLEST passing row) measures `rust_speed.py` on users 7213/7250, one rep, run while the GPU held a roster sweep (`rust_speed_underload_n1i8s3.json`): plain 2732.5 rev/s, `l2i8s4` 2203.6 (0.806x rev, median per-user 0.809), `n1i8s3` 2155.5 (0.789x rev, 0.654x states, median per-user 0.800 over users 0.800/0.782) - both above the 0.75x floor, so row 35 is now DEPLOYABLE. The absolutes are under load; the ratios are what matter.




**Row 19 (`nr4`, the note-pool cut) FAILS**: card held at row 4's config, note dropped to rank-1 int4
factors + int4 shifts (1,024/1,536 bits) scores 0.334524/0.307559 = **+0.000722/+0.001348**, so the
note cannot pay the rank-1 int4 price the card's ladder just removed from it - the imm head goes
0.0003 further over the ceiling than row 4 instead of closing. The passing note is the int8 one of
rows 15-18 (2,304 bits with int4 shifts). The chain's next point (`nr4s2`, note 1,152 bits) and the
combined `l2i8s4n4`/`l3i8s4n4` points stay pre-registered; read them against this row.

**Rows 20-21 land, and the combined cut is 0.00014 from passing.** `nr4s2` (row 20) drops the note shifts to int2 and takes the tax UP, not down: +0.000846/+0.001756, i.e. +0.000124/+0.000408 worse than row 19 for 384 bits saved, so the note needs int4 shifts and row 19 is the floor of the rank-1 int4 note pool (1,536 bits). `l2i8s4n4` (row 21) is the informative one: it keeps the note at row 19 int4 config and puts rank 2 back on the card, recovering +0.000394/+0.000710 relative to row 19 to land at +0.000328/+0.000638 - the ahead head is now 0.65x of the ceiling and the imm head 1.3x over, the closest any note cut has come. Against row 17 (same 1,280 card bits, note 2,304) the 768-bit note cut costs about +0.000276/+0.000311, so the note is worth roughly +0.0003 of imm tax per 768 bits, and row 17 note at 2,304 bits remains the cheapest note that passes. `l3i8s4n4` (rank-3 card at the same int4 note, 1,792/1,536 bits) is scoring now (started 11:58) and is the last pre-registered point of this chain.

**The bits-vs-tax curve (rows 5-7) is the axis-1 answer, and it is not free.** Walking down from row 4:

| card/note bits | row | d ahead | d imm |
|---|---|---|---|
| 1,024 / 3,072 | 4 | +0.000503 | +0.001009 |
| 512 / 1,536 | 5 | +0.001050 | +0.001682 |
| 384 / 1,152 | 6 | +0.001445 | +0.002280 |
| 256 / 768 | 7 | +0.003604 | +0.004932 |

The curve is convex: the first halving doubles the tax, and the last step blows it up (row 7 is
7.2x/9.9x over the ceiling), so int2 factors are past the knee. Both streams are still far under their
constraint-2 bars at rows 5-6 (card 512 or 384 vs 2,176; note 1,536 or 1,152 vs 6,528), so axis 1 is a
trade-off curve rather than a hard limit.

**Rows 9-10 answer the bit-split question, and the answer is counter-intuitive: the bits belong in the
FACTORS, not the shifts.** Both rows spend 768/2,304 bits. Row 9 gives the factors int8 and the shifts
int4 (+0.000516/+0.001030) and lands within +0.00002 of row 4, which spends 1,024/3,072 - i.e. the
shift vectors can drop from int8 to int4 for FREE, saving 25% of the bits. Row 10 does the opposite
(int4 factors, int8 shifts) and is as bad as row 5 (+0.001045/+0.001661), even though it has 50% MORE
bits. The factors are rank-1's real content; the shifts ride along.

**The tax is RANK TRUNCATION, not precision (settled 2026-11-21 04:33 by the `cn8` diagnostic).**

`cn8` is full-matrix int8 on both streams - no rank truncation at all - and it is 4x over both bars
(8,704 / 26,112 bits), so it is a DIAGNOSTIC, never a candidate. Its tax against row 0 is
**+0.000033 / +0.000022**: an 8-bit state is essentially lossless when it is not truncated. The rank-1
family's +0.000503/+0.001009 is therefore the rank-1 TRUNCATION, not the 8-bit factor precision, and no
finer level can buy it back. The fix has to add RANK.

Which stream should get the rank? The per-stream attribution pair ran on row 4's OWN checkpoint (its own
quantizer, so there is no quantizer mismatch), each quantizing one stream and leaving the other in fp32:

| run | card bits | note bits | ahead | imm | d ahead | d imm |
|---|---:|---:|---:|---:|---:|---:|
| row 4 `l1i8` (both streams rank-1 int8) | 1,024 | 3,072 | 0.334305 | 0.307221 | +0.000503 | +0.001009 |
| `l1i8c`, CARD only | 1,024 | fp32 | 0.335084 | 0.307353 | +0.001282 | +0.001142 |
| `l1i8n`, NOTE only | fp32 | 3,072 | 0.334175 | 0.306470 | +0.000373 | +0.000258 |

The note stream at rank-1 int8 costs +0.000373/+0.000258 on its own - INSIDE the ceiling on both heads.
The card stream carries the immediate-head tax by itself (+0.001142, which is even more than the two
streams cost together, because each single-stream run has the other stream mismatched against its QAT
training). So the rank must go to the CARD, and the card's 2,176-bit bar has room: 1,024 bits at rank-1
int8 + int8 shifts, 1,536 at rank-2, 2,048 at rank-3 (128 bits under the bar), 2,176 exactly at rank-4
int8 + int2 shifts. Rows 15-18 measure that (running, 2026-11-21 07:45).

Second consequence: the note pool (3,072 of row 4's 4,096 bits) is where the BITS are while the card is
where the TAX is, so a note cut is worth measuring as well - rank-1 int4 (1,536 bits) or rank-1 int4
factors + int2 shifts (1,152). `it36notecut` runs that (`nr4`, `nr4s2`) and then the combined
`l2i8s4n4` / `l3i8s4n4` points, gated on the rank ladder actually helping.

**The decay-length axis is CLOSED (2026-11-21 06:12), and it is closed in the awkward direction.**
Three lengths of the same `l1i8` recipe now have their own same-length fp32 twin:

| decay | quantized | fp32 twin | d ahead | d imm |
|---|---|---|---|---|
| 124 (row 4) | 0.334305 / 0.307221 | 0.333802 / 0.306211 | +0.000503 | +0.001009 |
| 496 (row 8) | 0.322791 / 0.296287 | 0.321659 / 0.294614 | +0.001132 | +0.001674 |
| 992 (row 11) | 0.318088 / 0.291307 | 0.317213 / 0.289632 | +0.000874 | +0.001675 |

The tax does not fall at any length, and it roughly doubles past 124 steps on the immediate head. Andrew
relaxed the step budget (2026-11-21 01:00: "instead of using 124 steps, you can use more steps ... if it
ultimately ends up turning a failing recipe into a winning one, it's fine"), and this is the answer: the
longer decays do not turn a failing recipe into a winning one. What they DO give is a much better model
in ABSOLUTE terms - the 992-step quantized run is 0.0162/0.0159 BETTER than row 4 on both heads, because
the fp32 twin itself improves by 0.016589/0.016579 between 124 and 992 steps. So for the rest of this arm the
length is held at the champion's 124 steps (comparable, and the cheapest), and `l1i4` at 992 was correctly
skipped by chain 4's own guard (the 496-step `l1i4` tax was already worse than the 124-step one).

### fp32 twins (reference rows, needed to split "training drift" from "quantizer cost")

**Caveat (2026-11-21 01:15): this split is a diagnostic, not a clean decomposition.** A twin is a
checkpoint that was TRAINED with the quantizer in the loop and then evaluated with it removed, so it is a
mismatched model, not a second fp32 model. The measured signs show that: the quantized eval is BETTER
than its own twin on the ahead head (row 3 -0.001062, row 4 -0.001144) and near-neutral on the immediate
head (row 3 -0.000560, row 4 +0.000192). So QAT does not merely add state error; it changes the operating
point, and removing the quantizer afterwards moves the model away from where it was trained. Read the
table as "how sensitive is this checkpoint to the presence of the quantizer", and use the TOTAL tax
against row 0 (or row 0b at 496 steps) as the gate number.

Each twin is the SAME QAT-trained checkpoint evaluated with quantization OFF, so the pair isolates the
quantizer. These are not candidates.

| twin of | fp32 twin tag | ahead | imm | d ahead vs row 0 | d imm vs row 0 |
|---|---|---|---:|---:|---:|---:|
| row 3 | `qnj_it36qat` | 0.336100 | 0.307813 | +0.002298 (ub95 +0.002489) | +0.001602 (ub95 +0.001793) |
| row 4 | `qnj_it36qatl1i8` | 0.335448 | 0.307029 | +0.001647 (ub95 +0.001782) | +0.000817 (ub95 +0.000923) |

### Decomposition (the tax of each step, 750 users, paired)

| pair | d ahead | d imm | reading |
|---|---:|---:|---|
| row 1 - row 0 (**PTQ COST**) | +0.001424 | +0.002676 | quantizing the fp32 model at the end, no adaptation |
| row 3's twin - row 0 (**MODEL DRIFT**, card int2 + note rank-1 int8) | +0.002298 | +0.001602 | what QAT training costs when the result is evaluated in fp32 |
| row 3 - its twin (**PRECISION DEGRADATION**) | -0.001063 | -0.000560 | the quantizer's own cost on that QAT model: the quantized eval is BETTER than fp32 |
| row 3 - row 0 (**FULL TAX**) | +0.001236 | +0.001042 | the gate number for that recipe |
| row 4's twin - row 0 (**MODEL DRIFT**, rank-1 int8) | +0.001647 | +0.000817 | smaller drift than row 3: a rank-1 quantizer needs less adaptation |
| row 4 - its twin (**PRECISION DEGRADATION**) | -0.001144 | +0.000192 | better than fp32 on ahead; +0.000192 on imm - that head alone is inside the ceiling |
| **row 4 - row 0 (FULL TAX)** | **+0.000503** | **+0.001009** | **the gate number for the best recipe** |
| row 2 - its twin (`qnj_it36qat`) | +0.001459 | +0.004111 | QAT does not rescue full-matrix int2 (p 8.0e-21/1.9e-83) |

Reading: QAT mostly removes the **immediate-head** part of the PTQ cost. For the best recipe the
ahead head ends at the ceiling and the immediate head at 2x; the immediate head is therefore what the
next experiments must attack (longer QAT decay, different bit allocation).

## Appendix A - PTQ measured on the `iter36` champion (NOT gate-comparable)

These four measurements are valid, but the checkpoint is a **different training run** (iter36's own
decay used data that no longer exists), so they are not "identical except quantization" and must not
be compared with the rows above. They are kept because they show how the untrained schemes rank on a
second, independent checkpoint.

Checkpoint `pretrain/rwkv/opt_iter36/rwkv_iter36_124.pth`, unquantized = 0.342813 ahead / 0.314272
imm on the same roster.

| scheme | card bits | note bits | ahead | imm | d ahead | d imm |
|---|---:|---:|---:|---:|---:|---:|
| full-matrix int2, both streams | 2,176 | 6,528 | 0.346921 | 0.319816 | +0.004108 | +0.005544 |
| rank-1 int4, both streams | 512 | 1,536 | 0.344532 | 0.315691 | +0.001719 | +0.001419 |
| rank-1 int8, both streams | 1,024 | 3,072 | 0.344697 | 0.315316 | +0.001884 | +0.001044 |
| card int2 + note rank-1 int8 | 2,176 | 3,072 | 0.344899 | 0.316326 | +0.002085 | +0.002054 |

The ranking matches the anchor-line table: the rank-1 family spends its bits far better than the
full-matrix family, and no untrained scheme comes close to the +0.0005 ceiling on either checkpoint.

## Attribution runs (diagnostics, NOT candidates)

These quantize ONE stream and leave the other in fp32, so they violate constraint 1 and cannot be
deployed. They were measured on a third checkpoint (the adaptive-WD incumbent), so their deltas are
only meaningful *within* this table.

| Run | Method | card bits (B) | note bits (B) | ahead LogLoss | imm LogLoss | d ahead | d imm |
|---:|---|---:|---:|---:|---:|---:|---:|
| d1 | rank-1 int4 applied to the CARD stream only, note fp32 | 512 (64) | 104,448 (13,056) | 0.300980 | 0.272852 | +0.000371 | +0.000843 |
| d2 | rank-1 int4 applied to the NOTE stream only, card fp32 | 34,816 (4,352) | 1,536 (192) | 0.301105 | 0.272518 | +0.000496 | +0.000508 |

Reading: the two streams are close to additive (0.000943/0.001397 when both are quantized). Card
hurts the immediate head most; note hurts the ahead head slightly more.

**The same attribution on row 4's own checkpoint, QAT-trained for BOTH streams (2026-11-21 03:32),
i.e. with the row 0 gate and no quantizer mismatch on the quantized stream:**

| Run | Method | card bits (B) | note bits (B) | ahead LogLoss | imm LogLoss | d ahead | d imm |
|---|---|---:|---:|---:|---:|---:|---:|
| d3 | `l1i8c`: rank-1 int8 on the CARD stream only, note fp32 | 1,024 (128) | 104,448 (13,056) | 0.335084 | 0.307353 | +0.001282 | +0.001142 |
| d4 | `l1i8n`: rank-1 int8 on the NOTE stream only, card fp32 | 34,816 (4,352) | 3,072 (384) | 0.334175 | 0.306470 | +0.000373 | +0.000258 |

This pair is the one that decides where the rank goes (see "the tax is RANK TRUNCATION" above):
the note is inside the ceiling on both heads by itself and the card owns the immediate-head tax.

**Mechanism diagnostic (2026-11-21 04:33):** `cn8`, full-matrix int8 on both streams with NO rank
truncation, 8,704/26,112 bits, 4x over both bars - DIAGNOSTIC ONLY, never a candidate. It scores
0.333834/0.306234 = **+0.000033/+0.000022** against row 0, so 8-bit precision of the FULL state is
free and the rank-1 tax is truncation.

## Rust deploy grid (the deploy engine itself, 3-user ranking tool)

Parity traces users 107/136/156 (~16k reviews), same exported weights. The noise floor of a 3-user
mean is about +/-0.0005, so read this as a family ranking, not as exact taxes.

| config | Method | card bits (B) | note bits (B) | ahead LogLoss | imm LogLoss | tax ahead | tax imm |
|---|---|---:|---:|---:|---:|---:|---:|
| plain | unquantized | 34,816 (4,352) | 104,448 (13,056) | 0.246986 | 0.206220 | - | - |
| cn2 | full-matrix int2 WKV + int2 shifts, card+note | 2,176 (272) | 6,528 (816) | 0.250103 | 0.210630 | +0.003117 | +0.004410 |
| cn3 | full-matrix int3 (OVER bar) | 3,264 (408) | 9,792 (1,224) | 0.249255 | 0.206730 | +0.002269 | +0.000510 |
| l1i2 | rank-1 int2 factors + int2 shifts | 256 (32) | 768 (96) | 0.249345 | 0.211450 | +0.002360 | +0.005230 |
| l1i3 | rank-1 int3 factors + int3 shifts | 384 (48) | 1,152 (144) | 0.249284 | 0.207378 | +0.002299 | +0.001158 |
| l1i4 | rank-1 int4 factors + int4 shifts | 512 (64) | 1,536 (192) | 0.246886 | 0.206681 | -0.000099 | +0.000461 |
| l1i5 | rank-1 int5 factors + int5 shifts | 640 (80) | 1,920 (240) | 0.246170 | 0.206828 | -0.000816 | +0.000608 |
| l1i6 | rank-1 int6 factors + int6 shifts | 768 (96) | 2,304 (288) | 0.246848 | 0.207017 | -0.000138 | +0.000797 |
| l1i8 | rank-1 int8 factors + int8 shifts | 1,024 (128) | 3,072 (384) | 0.246636 | 0.206723 | -0.000350 | +0.000503 |
| l1i4als | rank-1 int4 + ALS(3) factor fitting | 512 (64) | 1,536 (192) | 0.246626 | 0.206707 | -0.000360 | +0.000487 |
| l1i4h | rank-1 int4 + Hadamard rotation | 512 (64) | 1,536 (192) | 0.247783 | 0.207778 | +0.000798 | +0.001558 |
| l1i4p | rank-1 int4 + per-column scales | 512 (64) | 1,536 (192) | 0.246886 | 0.206681 | -0.000099 | +0.000461 |
| l1i2hp | rank-1 int2 + Hadamard + per-column | 256 (32) | 768 (96) | 0.251239 | 0.211099 | +0.004253 | +0.004879 |
| l1i3hp | rank-1 int3 + Hadamard + per-column | 384 (48) | 1,152 (144) | 0.249557 | 0.207596 | +0.002571 | +0.001376 |

Readings:
1. **The full-matrix int-N family is too lossy at the bar**: int2 (exactly AT the bit bar) costs
   +0.0031/+0.0044 and int3 (over the bar) still costs +0.0023 - 5-9x the +0.0005 ceiling.
2. **The rank-1 family is ~equal to unquantized within the noise floor** and is 2-4x UNDER the bit bar.
3. **Factor precision above int4 does not help** (int5/int6/int8 sit where int4 sits): at rank 1 the
   residual error is the rank truncation itself. Extra bits above int4 are wasted.
4. Hadamard pre-rotation HURTS this trunk; per-column scales are neutral; ALS factor fitting is
   neutral-to-slightly-better. All are free in bits.
5. **rank-2 fails the speed constraint** and produced a wild +0.088/+0.109 tax on the power users.

### The QAT checkpoints on the same 3-user sample (tax vs each checkpoint's own fp32 run)

| checkpoint | config | tax ahead | tax imm |
|---|---|---:|---:|
| QAT `c2_nl1i8` | card int2 + note rank-1 int8 | -0.003549 | +0.001802 |
| QAT `c2_nl1i8` | both streams int2 (`cn2`) | -0.000593 | +0.004837 |
| QAT `c2_nl1i8` | rank-1 int8 (`l1i8`) | -0.003877 | +0.002399 |
| fp32 control `ctl_124` | card int2 + note rank-1 int8 | -0.001266 | +0.003673 |
| fp32 control `ctl_124` | both streams int2 (`cn2`) | +0.001138 | +0.005466 |
| fp32 control `ctl_124` | rank-1 int8 (`l1i8`) | -0.001171 | +0.005041 |

QAT roughly halves the immediate-head tax on this sample. 3 users can only rank; the 750-user rows
decide.

### Speed on the deploy engine (constraint 6)

`rust_speed.py`, 2 runs per config (best kept), on the engine's own `user N: M reviews in T s (R
rev/s)` line; `states/s` is the DEPLOY_SPEED_LOG metric. The `c2_nl1i8`/`l1i8`/`cn2`/`l1i4` rows are
the IDLE-machine run `rust_speed_it36qatl1i8.json` (2026-11-20 10:06); the others were taken while a
roster sweep held the GPU, so their ratios are trustworthy and their absolutes are not.

| config | Method | rev/s (abs) | x unquant (rev/s) | states/s (abs) | x unquant (states/s) | median per-user x | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| plain | unquantized | 2,863 | 1.000x | 6,062 | 1.000x | 1.000 | reference |
| l1i8 | rank-1 int8 factors + shifts, card+note | 2,586 | 0.903x | 4,888 | 0.806x | 0.898 | PASS |
| c2_nl1i8 | card int2 + note rank-1 int8 | 2,586 | 0.903x | 4,873 | 0.804x | 0.894 | PASS |
| l1i4 | rank-1 int4 factors + shifts, card+note | 2,586 | 0.903x | 4,862 | 0.802x | 0.894 | PASS |
| cn2 | full-matrix int2 WKV + int2 shifts, card+note | 2,628 | 0.918x | 4,930 | 0.813x | 0.903 | PASS |
| l1i4c | rank-1 int4, card only | 2,813 | 0.982x | 5,606 | 0.943x | 0.963 | PASS |
| l1i4n | rank-1 int4, note only | 2,718 | 0.949x | 5,127 | 0.862x | 0.923 | PASS |
| l2i4 | rank-2 int4 factors + shifts | 1,802 | 0.629x | 2,667 | 0.448x | 0.621 | **FAIL** |

Readings:
1. Every rank-1 or full-matrix int-N config is **inside** the 25%-slower budget (2-10% slower); the
   retrievability pass `R/s` stays ~5,970-6,000 for every config, so the cost is quantization work.
2. **Rank 2 is the only family that fails** (0.63x rev/s, 0.45x states/s): no fused fast path, so the
   per-step reconstruction dominates. Rank 2 is disqualified regardless of its loss.
3. Cheaper scopes are faster: card-only 0.98x, note-only 0.95x, both 0.90x.
4. Rank-1 int2/int3 are the same structure as int4/int8, so their speed is expected to be ~0.90x too;
   they will be measured on the idle machine if they become candidates.

## Iteration log

- **Row 0, anchor `qnj_it36ctl`**: the 124-step decay replayed from `rwkv_iter36_558.pth` on users
  1-100 of the 5k db, quantization OFF. 10.2 min of training, final in-run validation 0.3743/0.3625.
  0.333802 ahead / 0.306211 imm on the roster. Every delta on this page is against this run.
- **Row 1, PTQ `qc2_nl1i8_it36ctlq`**: card full-matrix int2 + note rank-1 int8 (2,176/3,072 bits)
  applied to the row-0 checkpoint with no training. 0.335225/0.308888, +0.001424/+0.002676. FAIL.
- **Row 2, QAT `qcn2_it36qatq`**: both streams full-matrix int2, 124 steps. 0.337559/0.311924,
  +0.003757/+0.005712. FAIL - QAT does not rescue the naive at-the-bar scheme.
- **Row 3, QAT `qc2_nl1i8_it36qatq`**: card int2 + note rank-1 int8, 124 steps, 3.8 min, banners
  `[QAT] card_id=1.0`, `[QAT-LOWRANK] note_id=rank1/fq127.0`, `[QAT-SHIFT] card_id=1.0, note_id=127.0`.
  0.335038/0.307253, +0.001236/+0.001042. FAIL.
- **Row 4, QAT `ql1i8_it36qatl1i8q`**: rank-1 int8 on both streams (1,024/3,072 bits), 124 steps,
  4.0 min, banners `[QAT-LOWRANK] card_id=rank1/fq127.0, note_id=rank1/fq127.0`. 0.334305/0.307221,
  **+0.000503/+0.001009** - the best point, one hair over the ceiling on ahead and 2x over on imm.
  Quantizer mismatch check: the same checkpoint evaluated with the `c2_nl1i8` quantizer gives
  0.335534/0.308766 (+0.001732/+0.002554), so the deploy must use the training-time quantizer.
- **Rows 5-8, `it36qatsmall` (finished 2026-11-21 00:51:55, rc=0)**: job 0 scored the 496-step fp32
  control as `qnj_it36ctl496` (0.321659/0.294614 - row 0b, 0.012143/0.011598 BETTER than row 0); then
  the 124-step bit ladder `l1i4` (512/1,536 bits) +0.001050/+0.001682, `l1i4s2` (384/1,152)
  +0.001445/+0.002280, `l1i2` (256/768) +0.003604/+0.004932; then `l1i8` at 496 steps, which is
  +0.001132/+0.001674 against its own 496-step twin and so refutes the longer-decay route. Each point
  was a 0.04-epoch smoke, one QAT decay, and one quantized eval; every run printed its
  `[QAT-LOWRANK]`/`[QAT-SHIFT]` banners, saw 160 groups, and reserved 5.6 GB at peak.
- **Rows 9-10 plus the 496-step ladder, `it36qatsmall2` (finished 2026-11-21 03:08:54, rc=0)**:
  `l1i4` at 496 steps = +0.001581/+0.002050 against `qnj_it36ctl496` (worse than `l1i8`@496, as
  expected at half the bits); the 768/2,304-bit split pair `l1i8s4` = +0.000516/+0.001030 (row 9,
  the new axis-1 best - the int8 factors keep the tax while the int4 shifts save 25% of the bits)
  and `l1i4s8` = +0.001045/+0.001661 (row 10); then `l1i4s2` at 496 steps = +0.001944/+0.002698.
  Its own guard correctly skipped `l1i2`@496.
- **`it36qatsmall3` (finished 2026-11-21 04:33:14, rc=0)**: the per-stream attribution pair on row
  4's own checkpoint (`l1i8c` +0.001282/+0.001142, `l1i8n` +0.000373/+0.000258) and the `cn8`
  mechanism diagnostic with its own 124-step QAT decay (+0.000033/+0.000022 - the decisive result).
- **Rows 11-12, `it36qatsmall4` (finished 2026-11-21 06:12:50, rc=0)**: the 992-step fp32 twin
  `qnj_it36ctl992` = 0.317213/0.289632, i.e. 0.016589/0.016579 better than row 0 (that gain is the
  extra training alone) and 0.025600/0.024640 better than the iter36 champion `qnj_it36_124`; then
  `l1i8` at 992 = +0.000874/+0.001675 against that twin, and `l1i4`@992 skipped by the guard.
- **Rows 15-18, `it36rank` (launched 2026-11-21 07:45, finished 2026-11-21 10:05:24, rc=0)**: the card rank ladder - `l2i8`
  (card rank-2 int8, 1,536/3,072), `l3i8` (card rank-3 int8, 2,048/3,072), `l2i8s4` (1,280/2,304)
  and `l3i8s4` (1,792/2,304). This is the direct test of the `cn8` verdict: if the rank-1 tax is
  truncation, buying rank inside the card's bar is what removes it.
- **`it36notecut` (queued behind `it36rank` at 2026-11-21 07:48)**: the note pool cut - `nr4` (note
  rank-1 int4 factors + int4 shifts, 1,024/1,536), `nr4s2` (1,024/1,152), then, only if the best
  card-rank imm delta is under +0.00070, the combined `l2i8s4n4` (1,280/1,536) and `l3i8s4n4`
  (1,792/1,536).
- **`it36qatsmall3` (queued behind it)**: the per-stream attribution pair `l1i8c` (card rank-1 int8
  only) and `l1i8n` (note rank-1 int8 only) on row 4's own checkpoint, which splits row 4's ahead/imm
  tax into a card part and a note part. Attribution rows only - one stream stays fp32, so they cannot
  meet constraint 2. It then trains the mechanism diagnostic `cn8` (full-matrix int8, 8,704/26,112 bits,
  4x over the bars, DIAGNOSTIC ONLY) with its own 124-step QAT decay.
- **Rows 11-12, `it36qatsmall4` (queued 2026-11-21 01:05)**: the relaxed step-budget point. It trains
  the 992-step fp32 twin (`qnj_it36ctl992`, `--no-quant`, scored against `qnj_it36_124` for context
  only), then `l1i8` at 992 steps against that twin, then `l1i4` at 992 steps but only if the 496-step
  `l1i4` ahead tax beat the 124-step one. 992 steps is 8x the champion's 124 and 2x the last measured
  point, so it is the last sensible length on this curve.
- **Appendix A, `it36ptq`**: the four untrained schemes re-scored on the iter36 champion (a different
  run, kept for ranking only).
- **Preliminary PTQ sweep on the adaptive-WD incumbent (`sweep_1119pm.log`)**: the first pass, which
  established that no untrained scheme passes and that the rank-1 family ranks best. Superseded by
  the tables above.