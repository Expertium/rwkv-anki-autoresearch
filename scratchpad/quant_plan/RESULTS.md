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

`d` columns are paired per-user means against **row 0** (row 8 against its own twin, **row 0b**).
A row passes only if BOTH heads are <= +0.0005. Speed is per config (it does not depend on the
checkpoint) and comes from the deploy engine table below.

| Iter | Method | card bits (B) | note bits (B) | speed: Rust rev/s | speed: x unquant | ahead LogLoss | imm LogLoss | d ahead | d imm | verdict |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | **ANCHOR**: the identical 124-step decay, NO quantization (fp32 state) | 34,816 (4,352) | 104,448 (13,056) | 2,863 | 1.00x | 0.333802 | 0.306211 | - | - | reference, n=750, 0 OOM, 0 NaN |
| 0b | **ANCHOR for row 8**: the identical 496-step decay, NO quantization (fp32 state), tag `qnj_it36ctl496` | 34,816 (4,352) | 104,448 (13,056) | 2,863 | 1.00x | 0.321659 | 0.294614 | - | - | reference for row 8 only; it is 0.012143/0.011598 BETTER than row 0, so its d columns are not comparable to rows 1-7 |
| 1 | PTQ, no training: card full-matrix int2 + note rank-1 int8 applied to the anchor checkpoint | 2,176 (272) | 3,072 (384) | 2,586 | 0.903x | 0.335225 | 0.308888 | +0.001424 | +0.002676 | **FAIL**, 2.8x/5.4x over, ub95 +0.001586/+0.002928 |
| 2 | QAT both streams full-matrix int2 (the naive at-the-bar scheme) | 2,176 (272) | 6,528 (816) | 2,628 | 0.918x | 0.337559 | 0.311924 | +0.003757 | +0.005712 | **FAIL**, 7.5x/11.4x over, ub95 +0.004026/+0.006157 |
| 3 | QAT card full-matrix int2 + note rank-1 int8 (Andrew's first pick) | 2,176 (272) | 3,072 (384) | 2,586 | 0.903x | 0.335038 | 0.307253 | +0.001236 | +0.001042 | **FAIL**, 2.5x/2.1x over, ub95 +0.001375/+0.001205 |
| 4 | **QAT rank-1 int8 factors + int8 shifts, card and note - BEST SO FAR** | **1,024 (128)** | 3,072 (384) | 2,586 | 0.903x | **0.334305** | **0.307221** | **+0.000503** | **+0.001009** | **FAIL by a hair**: ahead is 0.0000035 over the ceiling (ub95 +0.000607), imm 2x over (ub95 +0.001122) |
| 5 | QAT rank-1 int4 factors + int4 shifts, card and note (HALF of row 4) | 512 (64) | 1,536 (192) | 2,586 | 0.903x | *running* | *running* | *running* | *running* | *running* - `it36qatsmall` |
| 6 | QAT rank-1 int4 factors + int2 shifts | 384 (48) | 1,152 (144) | *running* | *running* | *running* | *running* | *running* | *running* | *running* - `it36qatsmall` |
| 7 | QAT rank-1 int2 factors + int2 shifts (QUARTER of row 4) | 256 (32) | 768 (96) | *running* | *running* | *running* | *running* | *running* | *running* | *running* - `it36qatsmall` |
| 8 | QAT rank-1 int8 at iter45's 496-step decay length (axis 2, same bits as row 4) | 1,024 (128) | 3,072 (384) | 2,586 | 0.903x | *running* | *running* | *running* | *running* | *running* - `it36qatsmall`; **d against row 0b**, not row 0 |

**Status: nothing passes the +0.0005 bar yet.** Row 4 is the best point: ahead is essentially at the
ceiling, the immediate head is twice over. Rows 3 and 1 are 2-3x over on both heads, and row 2 shows
that QAT does not rescue a naive full-matrix quantizer. Rows 5-7 walk the bit ladder down (the axis
Andrew ranks first); row 8 tests a longer QAT decay at row 4's bits.

**Row 0b reorders the priorities.** Lengthening the decay from 124 to 496 steps at identical bits and
identical quantization moves the loss about 25x more than the whole quantization tax does
(0.0121/0.0116 vs 0.0005/0.0010). The remaining axis-2 work is therefore not "find a cleverer
quantizer" but "give the decay enough steps to absorb the quantizer"; rows 5-7 stay valuable because
smaller state is Andrew's first priority.

### fp32 twins (reference rows, needed to split "training drift" from "quantizer cost")

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
- **Rows 5-8, `it36qatsmall` (running)**: the bit ladder `l1i4` (512/1,536), `l1i4s2` (384/1,152),
  `l1i2` (256/768), each 124-step QAT + one quantized eval against row 0; then `l1i8` at 496 steps
  against its own 496-step fp32 twin (`ctl_496.pth`, evaluated as `qnj_it36ctl496`).
- **Appendix A, `it36ptq`**: the four untrained schemes re-scored on the iter36 champion (a different
  run, kept for ranking only).
- **Preliminary PTQ sweep on the adaptive-WD incumbent (`sweep_1119pm.log`)**: the first pass, which
  established that no untrained scheme passes and that the rank-1 family ranks best. Superseded by
  the tables above.