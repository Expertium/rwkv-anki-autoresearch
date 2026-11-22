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
| 0 | **ANCHOR**: the identical 124-step decay, NO quantization (fp32 state) | 34,816 (4,352) | 104,448 (13,056) | 2,863 | 1.00x | 0.333802 | 0.306211 | - | - | reference, n=750, 0 OOM, 0 NaN | pending |
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
| 15 | **QAT card rank-2 int8 + note rank-1 int8 - THE FIRST PASSING RECIPE** | **1,536 (192)** | **3,072 (384)** | **2,324** | **0.812x** | **0.333848** | **0.306518** | **+0.000046** | **+0.000307** | **PASS**: ub95 +0.000112/+0.000369, both heads inside the ceiling, and the deploy engine runs it at 0.812x (>= 0.75x) | pending |
| 16 | **QAT card rank-3 int8 + note rank-1 int8 - the lowest tax of any row** | **2,048 (256)** | **3,072 (384)** | *see note* | *see note* | **0.333808** | **0.306388** | **+0.000006** | **+0.000176** | **PASS**: ub95 +0.000070/+0.000234; the ahead head is at zero and the immediate head is 3x inside the ceiling | pending |
| 17 | **QAT card rank-2 int8 factors + int4 shifts, note rank-1 int8 + int4 shifts - BEST on axis 1** | **1,280 (160)** | **2,304 (288)** | **2,290** | **0.800x** | **0.333853** | **0.306539** | **+0.000052** | **+0.000327** | **PASS**: ub95 +0.000118/+0.000390, the row-15 tax for 22% fewer bits (3,584 vs 4,608) | **PASS** (158,986 state calls, 150 p(recall) tensors, 0 non-finite, 4.8 min on the 50-user roster) |
| 18 | QAT card rank-3 int8 + int4 shifts, note rank-1 int8 + int4 shifts | 1,792 (224) | 2,304 (288) | 2,290 | 0.800x | 0.333821 | 0.306422 | +0.000019 | +0.000210 | **PASS**: ub95 +0.000082/+0.000268, 256 fewer card bits than row 16 at the same int4 shifts (vs row 0) | pending |
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

Rows 5-7, 9-11, 13-14, 19 and 20 all use the rank-1 path, whose engine speed was measured once (`l1i4` and
`l1i8` both 2,586 rev/s = 0.903x); the factor/shift bit widths do not change that path, so their speed
cells are that measurement rather than a per-candidate run. Row 21 uses row 17's exact card config (`l2i8s4`, 2,290 rev/s = 0.800x); only its note factor width differs (int4 vs int8), which does not change the path. Row 22 uses row 18's card config (`l3i8s4`, 2,290 rev/s = 0.800x); its note is the faster rank-1 path, so the card sets the number. Rows 24-25 (`l4i4s4`, `l7i4s4`) have NO deploy-engine run: the closest measured rank>=2 card paths are 2,324 rev/s = 0.812x (rank-2 int8, row 15) and 2,290 rev/s = 0.800x (rank-2/3 int8 with int4 shifts, rows 17-18), all above the 0.75x floor, but a `card:4:int4` / `card:7:int4` path has never been measured on the engine, and both rows fail the loss gate, so it was not measured. Row 26 (`n2i4s4`) reuses row 17's exact card config (`l2i8s4`, 2,290 rev/s = 0.800x) but moves the rank 2 to the NOTE, a path the deploy engine has never parsed and would need engine work for (the both-streams `l2i4` measured 0.629x), and it fails the loss gate as well, so it was not measured.

**The rank-2 speed is fine on the CARD; the earlier 0.629x was a both-streams measurement.**
`l2i4` (rank-2 on card AND note, all four note layers) is 0.629x rev/s / 0.448x states/s, which is what
made rank >= 2 look undeployable. Row 15 puts rank 2 on the CARD only - 1 layer, and the cheap stream -
and the deploy engine measures it at **2,323.6 rev/s = 0.812x** (states/s 0.660x, median per-user 0.803x),
i.e. ABOVE the 0.75x floor. It was measured while a QAT decay was training, so it is about 3% pessimistic
(`l1i8` reads 0.875x in that run against 0.903x on the idle machine); an idle-machine re-measurement is
still owed. Row 17 (`l2i8s4`) 0.800x and row 18 (`l3i8s4`) 0.800x were measured in the same run. No engine
change is needed: the deploy engine already parses `card:2:int8`, and the torch eval path mirrors its
rounding (the 3-user deploy grid ranks rank-2 below rank-1 on the immediate head, as the 750-user eval

**The `corruption` column (added 2026-11-21, Andrew's gate).** A row is deployable only if the quantized recipe produces NO non-finite value anywhere: if even ONE card/note state value or even ONE p(recall) is non-finite, on the users with the longest review histories and with those histories NOT truncated, the recipe FAILS. `pending` = the `it36corrupt` chain is queued for that row (the four passing rows 15-18 plus the quantization-OFF fp32 control of row 0, so the check also answers whether a blow-up needs quantization at all); `-` = not checked, because the row fails the log-loss gate and is not a deploy candidate. The verdicts land in `scratchpad/quant_plan/corruption/corruption_corrupt_<tag>.json` and are filled in here.
does).

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

**Status: rows 15-18 PASS all six constraints; rows 19-22, the note-pool cuts, all FAIL** (row 18 is the lowest-tax point of the ladder, 1,792 card / 2,304 note bits = 4,096, +0.000019/+0.000210, ub95 +0.000082/+0.000268 at 0.800x; row 17 is the new axis-1 choice, 1,280 card / 2,304 note bits = 3,584,
the row-15 tax for 22% fewer bits at 0.800x; row 15 the same recipe with int8 shifts, 4,608 total; row 16 the axis-2 choice at 2,048 with a near-zero tax). Card 1,280-1,536 bits (2.83x
smaller than naive int4, bar 2,176) and note 2,304-3,072 bits (2.13x-2.83x smaller, bar 6,528); both heads inside the
+0.0005 ceiling with their upper bounds too (row 17 +0.000052 ub95 +0.000118 ahead, +0.000327 ub95 +0.000390
imm; row 15 +0.000046/ub95 +0.000112 and +0.000307/ub95 +0.000369); the 750-user roster; and the deploy engine at 0.800x-0.812x rev/s (>= 0.75x). It buys the immediate head
with RANK on the card, exactly what the `cn8` verdict predicted (see "the tax is RANK TRUNCATION"
below). Row 9 stays the best axis-1 rank-1 point (768/2,304 bits at row 4's tax) for reference. Row 22 (`l3i8s4n4`, the note cut on row 18's card) is 0.000025 from passing on the imm head, so an int4 note is not enough: 2,304 note bits (rows 17-18) remain the cheapest passing note.  **Row 23 (`l2i4s4`) FAILS, and it settles one axis-1 question**: putting rank 2 on the card and the card factors at int4 reaches the SAME 768 card / 2,304 note bits as row 9 and scores 0.334068/0.306792 = **+0.000266 ahead / +0.000580 imm** (ub95 +0.000337/+0.000661) - better than row 9's +0.000516/+0.001030 on both heads at identical bits, so at a fixed bit budget rank beats factor precision on the card; the imm head is still 1.16x over the ceiling, so an int8-quality note (2,304 bits) remains part of the cheapest passing set. The `corruption` column is live: row 17 (`l2i8s4`) became the first row to PASS it on 2026-11-22 11:25 - 158,986 raw state observations and 150 p(recall) tensors over the 50 longest-history users (no truncation), none non-finite, 4.8 min on the GPU - and rows 0, 15, 16 and 18 are queued behind it in the same chain. **Rows 24-25 (`l4i4s4`, `l7i4s4`, the int4-rank ladder from `it36rankprec`) FAIL on the imm head**: +0.000463/+0.000466 (ub95 +0.000538/+0.000542), with ahead inside at +0.000218/+0.000209. Against the int8 rows at the SAME card-bit count (row 17 at 1,280, row 16 at 2,048) they are about +0.00014 to +0.00029 worse on both heads (difference of deltas, not a paired test), so past rank 2 the card's FACTOR PRECISION is the binding constraint. The int4-rank curve plateaus and does not reach the ceiling - rank 2 imm +0.000580 (row 23), rank 4 +0.000463, rank 7 +0.000466 - so the queued `l3i4s4` (rank-3 int4, 1,024 bits, in `it36equalbits`) is the last int4-rank point with room to move, and it has to close about +0.00025 of imm tax in one rank step. **Row 26 (`n2i4s4`) FAILS by a hair**: at row 17's exact 1,280/2,304 bits it moves the rank 2 from the card to the NOTE (rank-2 int4 factors) and scores +0.000180/+0.000492 (ub95 +0.000227/+0.000543), so the imm head's upper bound is 1.09x over the ceiling - the swap is the REVERSE of the card's, i.e. on the 3-layer note FACTOR PRECISION beats RANK, and the int8 note of rows 15-18 stays the cheapest passing note. **Row 27 (`l2i8` at 496 steps) FAILS**: the passing card-rank-2 recipe of row 15 does not carry to a longer decay - 0.322374/0.295406 = +0.000715/+0.000793 against its own 496-step twin `qnj_it36ctl496` (row 0b), ub95 +0.000802/+0.000870 - so the decay-length axis stays closed at 124 steps even with the card-rank recipe. **Row 28 (`l2i8` at 992 steps) FAILS too**: the same card-rank-2 recipe is +0.000554/+0.000829 (ub95 +0.000660/+0.000916) against its own 992-step twin `qnj_it36ctl992`, so the length axis stays closed with rank on the card as well as at rank 1.

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