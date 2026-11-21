# Card/note state quantization (opened 2026-11-19)

Andrew's new direction after TA3 was closed as a negative result. This file holds the pre-registered
setup, the bit arithmetic, the roster, the harness notes and the live status.

## The requirement (Andrew, 2026-11-19)

1. Quantize **card** and **note** states only. Leave deck, preset and global (user) states alone.
2. The quantized card/note states must be **at least 2x smaller in bits than int4**. He clarified
   that "int4" means *literally* fp32 -> int4, the simplest per-element quantization, with **no
   rank-2 factorisation and no other structure**:
   `state_size(our_quant(state)) <= 0.5 * state_size(int4(state))` for card and for note.
3. Log loss on **both** heads must not be higher than **current best + 0.0005**.
4. PTQ or QAT are both allowed: quantize with no extra training, or take the best run's checkpoint
   and redo **only the one decay epoch** as QAT. Do **not** redo all 10 epochs.
   * **(2026-11-21 01:00) The step budget is RELAXED.** Andrew: "instead of using 124 steps, you can
     use more steps. It will make results less apples-to-apples comparable, but if it ultimately ends
     up turning a failing recipe into a winning one, it's fine". So a QAT point may use any decay
     length. The rule that survives is that a point is only comparable to an fp32 twin trained at the
     SAME length on the same data, and that the row states its length. See RESULTS.md rows 11-12
     (`it36qatsmall4`, 992 steps).
5. Test on a **random 750-user subset** of the 2.5k eval set (VAL 5001-7500).
6. **(2026-11-19, later)** *Inference speed must be no more than 25% slower than the unquantized
   model*: if unquantized predicts R users/s, quantized must be >= `0.75*R` (ideally the loss is
   negligible). Measured as eval wall-clock per user, and confirmed on the deploy path (Rust CPU
   engine) for any candidate that also passes the loss bar. Record the ratio with the result.

## Bit arithmetic on the trunk that is actually in this checkout

`pretrain/rwkv/opt_qat46/rwkv_iter46_248.pth`: 444 tensors, 192,800 params, 92 inputs.
Stream stacks (`rwkv_modules.<i>`): 0 = **card** (1 block), 1 = deck (4), 2 = **note** (3),
3 = preset (3), 4 = user (3). Every `time_mixer.W_*` is (32, 32) and every 1-D vector is length 32,
i.e. **H=1, K=32, C=32**, so per layer the recurrent state is

    WKV matrix 1 x 32 x 32 = 1024 floats + 2 token-shift vectors x 32 = 64 floats = 1088 floats = 34,816 bits fp32

Naive int4 of that state (one scale per tensor, everything kept):

| stream | layers | state floats | int4 bits | int4 bytes | **bar (half)** |
|---|---:|---:|---:|---:|---:|
| card | 1 | 1,088 | 4,352 | 544 B | **2,176 bits = 272 B** |
| note | 3 | 3,264 | 13,056 | 1,632 B | **6,528 bits = 816 B** |

The bar is exactly **2 bits per state element**.

**Consequence to keep in mind.** The current deploy recipe is far *below* this ceiling: on the d=64
trunk (`s64hqr4b`) it is 403 card bits / 214 note bits, and on this H=1 trunk the same recipe would be
about half of the WKV codes. So constraint 2 is a ceiling with roughly 10x headroom, not a
compression challenge. The binding constraint is **3**: at up to 2 bits per element, get the
quantization penalty down to **<= +0.0005**. The known tax of the deployed 4-stage b10 recipe is
+0.0018 ahead / +0.0028 imm (d=64 trunk, vs its plain twin `s64hc10`). So the direction is
"spend the available bits to make quantization nearly free", not "go smaller".

## Evaluation roster (constraint 5)

`scratchpad/quant_plan/roster750_seed20261119.json` — 750 users drawn uniformly at random with
`random.Random(20261119)` from the 2,496 eligible one-batch VAL users in
`F:/rwkv_lmdb/test_db_5k_adaptive92` (the eligibility rule is the project's existing one: exactly one
stored batch, which also drops the four multi-batch users 5626 / 6630 / 6701 / 6810).
Overlap with the old *deterministic* screen roster is 213/750 (28.4%), as expected for a random draw.

Builder: `scratchpad/quant_plan/make_roster.py` (read-only on the LMDB).

## Harness

`scratchpad/quant_plan/eval_baseline.py` reuses the adaptive-WD chain's own `eval_users` helper, so
the path is identical to every screen this lineage has ever run:
`optimization/eval_sharded.py` with `rwkv/get_result_config_base5k.toml`, one shard,
`solo-threshold 0`, outputs `result/adw_<tag>.jsonl` (ahead) and `result/adwP_<tag>.jsonl` (imm).

**Gotcha (cost 1 failed run, 2026-11-19).** The `env` handed to `eval_users` must start from
`os.environ.copy()`. Passing a bare dict strips `SystemRoot`/`PATH` from the shard child and it dies
with `OSError: [WinError 10106] The requested service provider could not be loaded or initialized`
while importing `asyncio._overlapped`. `eval_env()` now copies the environment and only overrides the
four feature-layout flags (`RWKV_ID_FEATURES`, `RWKV_REAL_CYCLES`, `RWKV_FINAL_LAYOUT`,
`RWKV_E2S_PUBLISHED` = 0) plus `RWKV_VRAM_CAP_GB=11`.

`--limit N` runs a short sanity eval on the first N roster users (~1 min). Use it before a full run;
`eval_sharded` refuses to overwrite an existing tag, so always use a fresh tag.

## Live status

* 2026-11-21 08:40 - **ROW 15 PASSES ALL SIX CONSTRAINTS - the first passing recipe.** `l2i8` = QAT card
  **rank-2** int8 factors + int8 shifts, note rank-1 int8 (card 1,536 bits / 192 B, note 3,072 bits /
  384 B - 2.83x and 2.13x smaller than naive int4). Loss on the 750-user roster: 0.333848/0.306518 =
  **+0.000046 ahead (ub95 +0.000112) and +0.000307 imm (ub95 +0.000369)** against row 0 - both heads
  inside the +0.0005 ceiling, upper bounds included. Deploy engine speed: 2,323.6 rev/s = **0.812x**
  (states/s 0.660x, median per-user 0.803x) >= the 0.75x floor, measured while a QAT decay was training
  (~3% pessimistic: `l1i8` read 0.875x in that run against 0.903x idle). The rank-2 drop is the mechanism
  confirmed: rank-1 imm +0.001009 -> rank-2 +0.000307 (3.3x better). The old "rank >= 2 is undeployable"
  conclusion came from `l2i4`, which put rank 2 on the NOTE as well (4 layers); card-only rank-2 is 0.812x
  and needs NO engine change - the engine already parses `card:2:int8`. Artifacts: export
  `rust_artifacts/it36l2i8_124.safetensors`, speed `rust_speed_rank2card.json`, 3-user deploy grid
  `rust_grid_it36l2i8.json` (rank-2 ranks below rank-1 on the immediate head there too). Owed: the
  idle-machine speed re-measurement, and Andrew's decision on whether this replaces the deployed
  403/214-bit recipe. Rows 16-18 (rank-3, int4 shifts) are still running; `it36notecut` is queued.

* 2026-11-21 08:00 - **The mechanism is SETTLED and the frontier moved: `cn8` says the tax is RANK
  TRUNCATION, and row 9 is a 25%-smaller state at the same tax.** Four chains finished overnight
  (`it36qatsmall2` 03:08:54, `it36qatsmall3` 04:33:14, `it36qatsmall4` 06:12:50, all rc=0).
  (1) `cn8` (full-matrix int8, NO rank truncation, 8,704/26,112 bits, 4x over the bars, DIAGNOSTIC
  ONLY) = **+0.000033/+0.000022** against row 0: an 8-bit state is essentially lossless when it is not
  truncated, so the rank-1 tax is TRUNCATION and a finer level cannot buy it back.
  (2) The per-stream attribution on row 4's own checkpoint: `l1i8c` (card only) +0.001282/+0.001142 vs
  `l1i8n` (note only) +0.000373/+0.000258 - the NOTE stream is inside the ceiling on both heads by
  itself, the CARD stream owns the immediate-head tax.
  (3) Row 9 `l1i8s4` (int8 factors + int4 shifts, 768/2,304 bits) = +0.000516/+0.001030, within +0.00002
  of row 4 while spending 25% FEWER bits: the shift vectors can drop int8 -> int4 for free, and the bits
  belong in the FACTORS (row 10, the reverse split, is as bad as row 5 with 50% more bits).
  (4) The decay-length axis is CLOSED: `l1i8`'s tax is +0.000503/+0.001009 at 124 steps,
  +0.001132/+0.001674 at 496, +0.000874/+0.001675 at 992 - it never falls, so a longer decay does not
  turn the failing recipe into a winning one. It DOES give a much better model in ABSOLUTE terms: the
  992-step fp32 twin is 0.317213/0.289632 (0.016589/0.016579 better than row 0) and its quantized copy
  is 0.0162/0.0159 better than row 4. The absolute and relative readings now disagree - record both.
  Running: `it36rank` (card rank-2/3 inside the 2,176-bit bar, rows 15-18, launched 07:45) and
  `it36notecut` behind it. **Blocker: rank >= 2 has no fused deploy path, so the engine measures
  0.629x rev/s / 0.448x states/s - it cannot be deployed until a fused rank-r readout exists, which
  needs Andrew's approval.**

* 2026-11-21 07:45 - **Gate clarified by Andrew (binding): "fp32, no QAT, without quantization vs QAT,
  with quantization".** A QAT-trained checkpoint evaluated with quantization OFF (the "twin") is an
  attribution tool only, never the comparison number; equal decay steps on both sides are ideal, a
  mismatched number is acceptable but not ideal.

* 2026-11-21 01:05 - **Andrew relaxed the QAT step budget, and chain 4 (`it36qatsmall4`) is launched to
  exploit it.** "instead of using 124 steps, you can use more steps ... if it ultimately ends up turning
  a failing recipe into a winning one, it's fine". Chain 4 waits behind chain 3 and then: (1) trains and
  evals the **992-step fp32 twin** (`it36ctl992`, `--no-quant`, tag `qnj_it36ctl992`, scored against
  `qnj_it36_124` for context only); (2) `l1i8` at 992 steps (row 4's exact bits) against that twin;
  (3) `l1i4` at 992 steps ONLY if the 496-step `l1i4` ahead tax beat the 124-step one. 992 = 8x the
  champion's 124 and 2x the last measured point, so it is the last sensible length on this curve. The
  twin is mandatory: longer training helps fp32 too (row 0b is 0.0121/0.0116 better than row 0), so a
  point without its own same-length twin cannot separate "more training" from "quantization". Recorded
  in RESULTS.md (rows 11-12 + the relaxation paragraph), NOTES requirement 4 and board amendment 9.
  Launched detached 01:05:50, lock `it36qatsmall4_chain.lock` taken, first log line
  `=== it36qatsmall4 chain start ===`.

* 2026-11-21 01:00 - **Chain 3 relaunched with the mechanism diagnostic `cn8` added** (full-matrix int8
  both streams, 8,704/26,112 bits, 4x over the bars - DIAGNOSTIC ONLY, never a candidate). Row 4's
  residual immediate-head error (+0.001009, 2x over the ceiling) is either the rank-1 STRUCTURE (the WKV
  matrix is only approximately rank 1) or the 8-bit FACTOR PRECISION; `cn8` has no truncation at all, so
  its tax separates the two and decides whether a finer level (int16 factors/shifts, or unquantized
  factors) could help. It needs its own 124-step QAT decay because evaluating a checkpoint with a
  quantizer it was not trained for costs ~+0.0012/+0.0015 and would swamp the signal. Chain 3 order:
  `l1i8c`, `l1i8n` (the per-stream attribution pair on row 4's checkpoint), then `cn8`. Added to
  `quant_sweep.CANDIDATES` by `_add_cn8_diag.py`; board amendment 8b records the relaunch.

* 2026-11-21 00:55 - **Row 8 REFUTES the longer-decay hypothesis.** `l1i8` at 496 steps (row 4's exact
  bits) = 0.322791/0.296287 against its own 496-step fp32 twin (row 0b, 0.321659/0.294614) =
  **+0.001132 ahead / +0.001674 imm** (ub95 +0.001254/+0.001796, p 8.7e-51/6.1e-88). The tax MORE than
  doubles versus the 124-step pair (+0.000503/+0.001009), even though the absolute quantized model is
  0.0115 ahead of row 4. The fp32 twin simply improves faster than a fixed-bit quantizer can follow, so
  compliance has to be won at the champion's own decay length and the useful axis-2 lever is the bit
  ALLOCATION, not more training. `it36qatsmall_chain` ENDED at 00:51:55 with all four of its points
  scored (rows 5-8), and `it36qatsmall2_chain` picked up the GPU.

* 2026-11-21 00:14 - **Row 7 (`l1i2`, 256/768 bits, int2 factors + int2 shifts) = +0.003604 ahead /
  +0.004932 imm** (0.337406/0.311143, ub95 +0.003871/+0.005296, p 9.7e-93/1.8e-93) = **7.2x/9.9x over**
  the ceiling. That closes the bit ladder and the curve is convex: 1,024/3,072 -> +0.000503/+0.001009,
  512/1,536 -> +0.001050/+0.001682, 384/1,152 -> +0.001445/+0.002280, 256/768 -> +0.003604/+0.004932.
  So int2 factors are past the knee and rows 5-6 are the frontier of the axis-1 trade-off. The run was
  clean (`rank1/fq1.0` int2 factors + shift qmax 1.0, 160 groups, in-run validations 0.3869/0.3790 then
  0.3816/0.3721, peak reserved 5.604 GB). `it36qatsmall` then started its last point, `l1i8` at 496 steps
  (smoke rc=0 at 00:12:32); the second chain's `l1i2`@496 guard (+0.0025) will therefore skip that
  point, which is what it was written for.

* 2026-11-21 00:00 - **Row 6 (`l1i4s2`, 384/1,152 bits) = +0.001445 ahead / +0.002280 imm** against the
  124-step control (0.335247/0.308491, ub95 +0.001592/+0.002464, p 3.0e-66/1.9e-88), i.e. 2.9x/4.6x over
  the ceiling. Taking the token shifts from int4 to int2 costs +0.000395/+0.000598 on its own, so the
  SHIFT vectors are the expensive side of the rank-1 split. Both streams stay far under the bars (384 vs
  2,176 and 1,152 vs 6,528). The run was clean: `rank1/fq7.0` factors + shift qmax 1.0 (int2), 160
  groups, in-run validations 0.3842/0.3766 then 0.3768/0.3683, peak reserved 5.604 GB.

* 2026-11-21 00:00 - **Two 768/2,304-bit attribution candidates added** to `quant_sweep.CANDIDATES`:
  `l1i8s4` (int8 factors + int4 shifts) and `l1i4s8` (int4 factors + int8 shifts) - the same total size
  with the bits moved between the two sides, so the pair separates factor error from shift error (roster
  sweep rows 9-10). Added by `_add_mixed_candidates.py`; `bits_for` confirms 768/2,304 for both.

* 2026-11-21 00:00 - **Second screening chain launched** (`it36qatsmall2_chain.py`, detached 23:47, lock
  `it36qatsmall2_chain.lock`). It waits for `it36qatsmall_chain`'s end marker with zero matching
  processes, then runs: `l1i4` at 496 steps vs `qnj_it36ctl496` (can the longer decay absorb row 5's
  tax?), `l1i8s4`, `l1i4s8`, `l1i4s2` at 496 steps, and `l1i2` at 496 steps only if the 124-step `l1i2`
  point is within +0.0025. Hazard fixed on launch: the venv `python.exe` is a shim that spawns the real
  interpreter, so process counting cannot identify a single instance (it reported 2 for one chain) - the
  chain therefore does NOT count itself in the busy pattern and uses the lock file as the only
  double-launch guard. Delete the lock by hand after a machine outage.

* 2026-11-20 23:15 - **Row 5 (`l1i4`, 512/1,536 bits) scored: +0.001050 ahead / +0.001682 imm** against
  the 124-step control (0.334852/0.307894, ub95 +0.001164/+0.001841, p 2e-63/3.1e-75) = **2.1x/3.4x over**
  the +0.0005 ceiling. Halving row 4's bits roughly doubles the tax, so the bits-vs-tax curve is steep at
  the bottom. The run itself was clean (2 saves at 50/124; in-run validations 0.3850/0.3768 then
  0.3773/0.3681; 160 groups; `[QAT-LOWRANK] rank1/fq7.0` + `QAT-SHIFT 7.0`, i.e. int4 confirmed;
  peak_reserved 5.604 GB). `it36qatsmall` then moved on to `l1i4s2` (384/1,152 bits; smoke rc=0 at
  23:14:57), then `l1i2` (256/768), then `l1i8` at 496 steps.

* 2026-11-20 22:52 - **PUSHED TO GITHUB.** Commit `69121550` ("Document the card/note state-quantization
  arm, gated on the identical unquantized twin") is on `origin/main` (`git ls-remote` confirms it): 421
  files, 21,889 insertions - the arm's runners, `RESULTS.md`/`NOTES.md`, the board and research-log
  updates, the ram-guard/detach repairs, the adaptive_wd/ta3 notes and runners, and new `.gitignore`
  rules for the regenerable per-user prediction dumps (`scratchpad/quant_plan/preds_*/`,
  `rust_grid_preds/`, `rust_speed_preds/`) plus the stray root-level files. The 2,158 stale scratchpad
  paths that are deleted locally were deliberately NOT committed (they are still on the remote), and
  `model_weights.txt` (a derived 4 MB dump from `rewrite_pth_text_helper.py`) is now ignored, not tracked.

* 2026-11-20 22:46 - **Row 0b scored: the 496-step fp32 control (`qnj_it36ctl496`) = 0.321659 ahead /
  0.294614 imm**, i.e. **0.012143/0.011598 BETTER than row 0** (ub95 +0.012911/+0.012336, p 2.0e-108/
  1.1e-122) and 0.021154/0.019659 better than the iter36 champion. The decay length moves the loss about
  25x more than the whole quantization tax does (0.0121/0.0116 vs 0.0005/0.0010), so row 8 (`l1i8` at 496
  steps) is quoted against THIS twin, never against row 0. `it36qatsmall_chain` job 0 finished rc=0 in
  20.4 min (22:45:52); the chain is now on the `l1i4` 124-step QAT (its smoke finished rc=0 in 1.2 min at
  22:47:03).

* 2026-11-20 22:25 - **Andrew REDEFINED THE GATE and re-ranked the axes.** A result is judged against
  "whatever run that is exactly identical EXCEPT for quantization" - the fp32 control with the same
  recipe, data and step budget (`qnj_it36ctl` for 124-step points, `qnj_it36ctl496` for the 496-step
  point). The anchor-champion comparison (+0.0085/+0.0071) is context only and must NOT be quoted as
  the quantization cost; the believable number for `l1i8` is **+0.000503 ahead / +0.001009 imm**.
  Priority for further work: **(1) smaller state (highest), (2) lower log loss against that identical
  unquantized run, (3) speed** (10% slower is acceptable). Consequence: the arm now walks DOWN the bit
  ladder instead of up.
* 2026-11-20 22:25 - **`it36qatsmall_chain.py` launched** (detached, log `it36qatsmall_chain.log`):
  job 0 scores the 496-step fp32 control `qat_it36ctl496/ctl_496.pth` (trained 10:21 today, never
  evaluated because the machine went down at ~10:28); then, all at 124 steps and all against
  `qnj_it36ctl`: `l1i4` (512/1,536 bits), `l1i4s2` (384/1,152 bits), `l1i2` (256/768 bits); then
  `l1i8` at 496 steps against the 496-step twin. Each point = 0.04-epoch smoke + QAT + one quantized
  eval (~20 min), so the result is a bits-vs-tax curve. Added `l1i3`/`l1i2` to
  `quant_sweep.CANDIDATES` (they were only in the Rust grid before).
* 2026-11-20 22:24 - **The 10:28 outage:** the previous chains died mid-run. `ctl_496.pth` completed
  cleanly (10:21), but `qat496` (candidate `c2_nl1i8`, 496 steps) was killed around global step 250;
  its `qat_it36qat496/` folder holds only `qat_50.pth` and must NOT be scored or resumed. Nothing else
  was lost - all four PTQ rows and rows 8/10 were already complete.

* 2026-11-20 10:15 - **Constraint 6 PASSES for both recommended configs + the queued diagnostic chain
  started.** `rust_speed.py` on the IDLE machine (weights `rust_artifacts/it36qatl1i8_124.safetensors`,
  users 107/136/156, 2 reps best kept, `rust_speed_it36qatl1i8.json`): plain 2,863 rev/s / 6,061.7
  states/s = 1.000x; `c2_nl1i8` 2,586 / 0.903x (0.804x states/s, median per-user 0.894) PASS; `cn2`
  2,628.4 / 0.918x (0.813x, 0.903) PASS; `l1i8` 2,586 / 0.903x (0.806x, 0.898) PASS; `l1i4` 2,586 /
  0.903x (0.802x, 0.894) PASS. R/s is ~5,970 in every config, so the extra time is quantization work.
  Quantizer MISMATCH matters: the `l1i8`-trained checkpoint evaluated with the `c2_nl1i8` quantizer
  gives 0.335534/0.308766 = +0.001229/+0.001545 worse than with its own quantizer -> deploy with the
  training-time quantizer.
  Then `it36qat496` started on the freed GPU at 10:07 (smoke rc=0, then the 496-step fp32 control).
  `it36qatl1i4` waits behind it. NOTE both waiting chains were patched at 10:02: their "GPU is free"
  test now also matches the other chain's script, `rust_speed.py`/`rust_grid.py` and `rwkv-infer`
  (the engine is not a python process), because the l1i8 chain's export + speed steps come AFTER its
  last result file and the old test would have let training start on top of the speed benchmark.
* 2026-11-20 09:56 - **BETTER RECIPE FOUND: rank-1 int8 on BOTH streams under QAT beats the first pick
  at half the card bits.** `ql1i8_it36qatl1i8q` (QAT candidate `l1i8`, 1,024/3,072 bits, evaluated WITH
  quantization): **0.334305 ahead / 0.307221 imm** = -0.008508/-0.007052 vs the anchor champion
  (ub95 -0.007324/-0.006205) -- the best row so far, better than row 8 (`c2_nl1i8`, 2,176/3,072 bits)
  by 0.000733 ahead / 0.000032 imm. Its fp32 twin `qnj_it36qatl1i8` is 0.335448/0.307029
  (-0.007365/-0.007244), so:
  * precision degradation (deploy - twin) = **-0.001144 ahead / +0.000192 imm** -> the immediate head's
    quantizer cost alone is INSIDE the +0.0005 ceiling, the ahead head is better than the twin.
  * model drift (twin - control) = +0.001647/+0.000817, SMALLER than the int2-card candidate's
    +0.002298/+0.001602 (less adaptation needed for a rank-1 quantizer).
  * full tax (deploy - the fp32 control row 5) = **+0.000503 ahead / +0.001009 imm** (ub95
    +0.000607/+0.001122): the ahead tax sits exactly at the ceiling, the imm tax is still 2x over.
  Chain `it36qatl1i8` then exported the deploy artifact and is running `rust_speed.py` for the
  constraint-6 numbers. Queued behind it: `it36qat496` (496-step c2_nl1i8 + twin) and `it36qatl1i4`
  (rank-1 int4, 512/1,536 bits, under QAT).
* 2026-11-20 09:21 - **PTQ re-base on the anchor COMPLETE (`it36ptq`, rc=0, all four schemes FAIL).**
  Base = `rwkv_iter36_124.pth` (0.342813/0.314272), 750 users, paired vs `qnj_it36_124`:
  | row | scheme | bits card/note | ahead | imm | d ahead | d imm |
  |---|---|---:|---:|---:|---:|---:|
  | 1 | cn2 full-matrix int2 both streams | 2,176/6,528 | 0.346921 | 0.319816 | +0.004108 | +0.005544 |
  | 2 | l1i4 rank-1 int4 | 512/1,536 | 0.344532 | 0.315691 | +0.001719 | +0.001419 |
  | 3 | l1i8 rank-1 int8 | 1,024/3,072 | 0.344697 | 0.315316 | +0.001884 | +0.001044 |
  | 4 | c2_nl1i8 card int2 + note rank-1 int8 | 2,176/3,072 | 0.344899 | 0.316326 | +0.002085 | +0.002054 |
  Every untrained scheme is 2.1x-11x over the +0.0005 ceiling, and the rank-1 family again beats the
  full-matrix mix at the same base. So on this anchor PTQ alone is closed; QAT is required.
  Summary JSON: `quant_sweep_it36ptq.json`.
* 2026-11-20 09:17 - **`it36qatl1i8_chain` started on the freed GPU** (the ptq chain ended rc=0):
  smoke rc=0 in 0.8 min, then the 124-step QAT run for candidate `l1i8` (rank-1 int8, 1,024/3,072
  bits) from `rwkv_iter36_558.pth`. `it36qat496_chain` and `it36qatl1i4_chain` are queued behind it.
* 2026-11-20 08:20 - **The full tax decomposition is measured (750 users, paired).** Reference for
  the tax is the fp32 control (row 5 of `RESULTS.md`, 0.333802/0.306211 - exactly the same 124
  training steps WITHOUT quantization):
  | pair | d ahead | d imm |
  |---|---:|---:|
  | PTQ cost (PTQ on the control, `qc2_nl1i8_it36ctlq` 0.335225/0.308888) | +0.001424 | +0.002676 |
  | model drift (QAT fp32 twin `qnj_it36qat`) | +0.002298 | +0.001602 |
  | precision degradation (QAT+quant `qc2_nl1i8_it36qatq` - its own twin) | -0.001063 | -0.000560 |
  | **full tax** (QAT+quant - the fp32 control) | **+0.001236** | **+0.001042** |
  | recovered by QAT (PTQ cost - precision degradation) | +0.002487 | +0.003236 |
  **So there are two readings of constraint 3 and both must be stated:** vs the ANCHOR champion the
  deploy row is -0.007776/-0.007019 (PASS, 15x margin); vs its own budget-matched fp32 twin it costs
  +0.001236/+0.001042 (ub95 +0.001375/+0.001205) = 2.4x/2.1x OVER the +0.0005 ceiling. QAT mainly
  fixes the IMMEDIATE head (its PTQ cost +0.002676 -> +0.001042 of full tax). Also: PTQ alone on the
  control is +0.001424/+0.002676, so QAT buys -0.000188 ahead / -0.001634 imm at equal bits.
* 2026-11-20 08:25 - **Queued `it36qat496_chain.py`** (detached): the same candidate `c2_nl1i8` at
  iter45's 496-step decay length PLUS its own 496-step fp32 twin, then fp32 + quantized evals of
  both. Purpose: test whether the residual full tax shrinks with QAT length (history qat_log #39-#45
  suggests yes). 496 steps is 4x iter36's own decay -> a scaling DIAGNOSTIC, not the faithful replay.
  It waits for `adw_qc2_nl1i8_it36qatl1i8q.jsonl` (the end of the l1i8 chain), so the GPU stays single-job.
* 2026-11-20 07:49 - **FIRST COMPLIANT RESULT: the QAT redeploy of iter36 passes all three log-loss
  bars with margin.** `qc2_nl1i8_it36qatq` = the QAT checkpoint `qat_it36qat/qat_124.pth` (card
  full-matrix int2 + note rank-1 int8, 2,176/3,072 bits) evaluated WITH quantization on the seeded
  750-user roster: **ahead 0.335038 / imm 0.307253** (14.6 min, 51.3 users/min, n=750/750, 0 OOM).
  * vs the ANCHOR baseline `qnj_it36_124` (0.342813/0.314272, unquantized fp32): **-0.007776 ahead /
    -0.007019 imm** -- far inside the +0.0005 bar, so constraints 2 and 3 hold (2,176 bits is exactly
    the bar, 3,072 is half of the note bar).
  * vs its own fp32 twin `qnj_it36qat` (0.336100/0.307813): -0.001063 ahead / -0.000560 imm, i.e. on
    the QAT-trained model the QUANTIZED deploy is BETTER than the fp32 twin -- the QAT tax is gone.
  * attribution of the -0.0078: the fp32 CONTROL shows the data substitution alone is worth
    -0.009011/-0.008061 (its own decay beats iter36's), QAT training costs +0.002298/+0.001602
    against that control, and the quantizer costs -0.001063/-0.000560 on the QAT model. So the
    honest reading is "the redo's decay (different data) + QAT + quantization nets -0.0078", not
    "quantization alone bought 0.0078".
  Constraint 6 (speed) is still to be measured on an idle machine.
* 2026-11-20 08:04 - **QAT does NOT dissolve the naive at-the-bar scheme.** Same QAT checkpoint,
  same 750 users, quantized deploy `qcn2_it36qatq` (both streams full-matrix int2 + int2 shifts,
  2,176/6,528 bits = exactly the bar): ahead 0.337559 / imm 0.311924 = **+0.001459 ahead / +0.004111
  imm vs the fp32 twin** (ub95 +0.001708/+0.004455, p 8.0e-21/1.9e-83) -> FAIL. So the 124-step QAT
  does not rescue the full-matrix int-N scheme; the note stream must go rank-1, exactly as Andrew's
  pick (`c2_nl1i8`) does. Bit ALLOCATION matters under QAT, not just the bit count.
* 2026-11-20 07:43 - **Second QAT screening slot queued: candidate `l1i8`** (rank-1 int8 factors +
  int8 shifts on card AND note, 1,024/3,072 bits = 2x under the bar). Chain
  `scratchpad/quant_plan/it36qatl1i8_chain.py` (detached, log `it36qatl1i8_chain.log`): it waits for
  the PTQ re-base (`result/adw_qc2_nl1i8_it36ptq.jsonl`, i.e. the ptq chain's last job) AND an idle
  sweep, then does 0.04-epoch smoke -> 124-step QAT from `rwkv_iter36_558.pth` (same
  `--train-users 1 100 --max-train-len 66000`) -> unquantized eval vs `qnj_it36_124` -> quantized
  eval (`l1i8,c2_nl1i8`) vs that twin -> export of the deploy artifact -> `rust_speed.py --reps 2` on
  the idle machine for `c2_nl1i8,cn2,l1i8,l1i4`. Rationale: the PTQ sweep says the rank-1 family
  spends bits ~10x better than the full-matrix family at the bar, so the second screening slot goes
  to the best untrained scheme rather than to another full-matrix variant. ~4 min of QAT on this bed.
* 2026-11-20 07:43 - **The 3,000/3,000-bit experiment is VETOED** (Andrew, 2026-11-19: "At that bit
  budget codebooks become too big + inference speed drops"). The run board's "Current constraints"
  line still lists it as approved; do NOT launch it. The live ceiling is constraint 2: card <= 2,176
  bits, note <= 6,528 bits (half of naive fp32->int4).
* 2026-11-20 07:45 - **Anchor redo TRAINED, deploy tax being measured.** Anchor = iter36, and the
  pair is `rwkv_iter36_558.pth` (pre-decay, WS-final) -> 124-step decay -> `rwkv_iter36_124.pth`
  (post-decay, the baseline). All three runs used `--epochs 2 --max-steps 124 --train-users 1 100
  --max-train-len 66000` (160 groups/epoch, 0 skipped) from `rwkv_iter36_558.pth`:
  * smoke `qat_it36qatsmoke` (0.04 ep = 6 steps): rc=0, `MODEL SAVED.`, banners
    `[QAT] card_id=1.0`, `[QAT-LOWRANK] note_id=rank1/fq127.0`, `[QAT-SHIFT] card_id=1.0, note_id=127.0`.
  * control `qat_it36ctl` (`--no-quant`, fp32, 124 steps): 10.2 min, final in-run validation
    0.3743 ahead / 0.3625 imm.
  * QAT `qat_it36qat` (candidate `c2_nl1i8` = card full-matrix int2 + note rank-1 int8,
    2,176/3,072 bits): 3.8 min, final in-run validation 0.3770 / 0.3653.
  On the 750-user roster, unquantized (`qnj_*`), paired per user:
  | tag | ahead | imm | d ahead | d imm |
  |---|---:|---:|---:|---:|
  | `it36_124` ANCHOR (post-decay champion) | 0.342813 | 0.314272 | - | - |
  | `it36_558` (WS-final) | 0.351102 | 0.322708 | +0.008289 | +0.008436 |
  | `it36ctl` (replay, fp32) | 0.333802 | 0.306211 | -0.009011 | -0.008061 |
  | `it36qat` (replay, QAT) | 0.336100 | 0.307813 | -0.006713 | -0.006459 |
  So the substituted data (5k db, users 1-100, full coverage) gives a BETTER decay than iter36's own
  (-0.009), and QAT costs **+0.002298 ahead / +0.001602 imm on the fp32 model** versus its
  budget-matched fp32 control. Old-regime diagnostics vs the same anchor: `it45_496`
  0.336628/0.306384, `it46_248` 0.342243/0.313402, `it44_248` 0.340167/0.311412 - all ~0.036-0.042
  behind the adaptive incumbent (0.300609/0.272009) on this roster, so this lineage is a mechanism
  bed, not a deploy candidate.
* 2026-11-20 07:45 - deploy tax (the quantized evals of `qat_124` and `ctl_124`) RUNNING:
  `it36qat_eval2.py` -> `qc2_nl1i8_it36qatq`, `qcn2_it36qatq`, `qc2_nl1i8_it36ctlq`; afterwards
  `it36ptq_chain.py` re-bases the PTQ rows on `rwkv_iter36_124.pth` (`l1i4,cn2,l1i8,c2_nl1i8`, tag
  `it36ptq`).
* 2026-11-20 07:40 - Rust engine, 3-user parity traces (ranking tool only): tax vs each checkpoint's
  own unquantized run, QAT checkpoint `c2_nl1i8` -0.003549 ahead / **+0.001802 imm**, `cn2`
  -0.000593/+0.004837, `l1i8` -0.003877/+0.002399; the fp32 control checkpoint: `c2_nl1i8`
  -0.001266/+0.003673, `cn2` +0.001138/+0.005466, `l1i8` -0.001171/+0.005041. QAT roughly halves the
  imm tax on this sample. JSON: `rust_grid_it36qat.json`, `rust_grid_it36ctl.json`.
* 2026-11-19 21:45 - **STOPPED** the first full QAT epoch `qat_l1i8e1` (launched 21:35 from
  `adaptive_173416.pth`): it started from the POST-decay checkpoint, which conflicts with the anchor
  rule. Its partial artifacts stay in `qat_l1i8e1/` and must not be resumed or scored.
* 2026-11-19 21:35 - **QAT full decay epoch RUNNING** (detached,
  `scratchpad/quant_plan/qat_l1i8e1/`, launcher log `qat_l1i8e1_launcher.log`). 43,354 steps,
  ~12 h at the measured ~1.0 steps/s. Same env as the smoke.
* 2026-11-19 21:24 - **QAT smoke PASSED** (started 20:47): `qat_decay.py --candidate l1i8
  --epochs 0.04 --tag l1i8smoke` reached step 1,734, printed `MODEL SAVED.`, returned 0, had **0
  skipped batches**, wrote model+optimizer pairs every 50 steps (final `qat_1734.pth`) and validated
  0.3493/0.3348 on the 2 internal users. Quantization was ACTIVE
  (`[QAT-LOWRANK] card_id=rank1/fq127.0, note_id=rank1/fq127.0`, `[QAT-SHIFT] card_id=127.0,
  note_id=127.0`) - this is the fixed replacement for the adaptive-WD QAT stage that never quantized
  anything. Keep the smoke folder separate from the full run so it cannot clobber it.
* 2026-11-19 20:47 - **QAT smoke running** (`qat_decay.py --candidate l1i8 --epochs 0.04 --tag
  l1i8smoke`, detached, log `scratchpad/quant_plan/qat_l1i8smoke/train.log`). Quantization confirmed
  ACTIVE: `[QAT-LOWRANK] set: card_id=rank1/fq127.0, note_id=rank1/fq127.0` and
  `[QAT-SHIFT] state_shift_qmax set: card_id=127.0, note_id=127.0`. Checkpoint loaded from
  `adaptive_173416.pth`, internal validation users `[5002, 5004]` (both outside the 750 roster).
  Run it detached; ~1,750 steps at the observed ~0.6 steps/s.
* 2026-11-19 20:35 - **PTQ sweep CLOSED, nothing passes.** `quant_sweep_1119pm.json` +
  `sweep_1119pm.log`. Iter 1 `l1i4` (rank-1 int4 factors+shifts, card+note, 512/1,536 bits)
  +0.000943/+0.001397 FAIL; iter 2 `cn2` (full-matrix int2, exactly at the 2,176/6,528 bit bar)
  +0.009785/+0.011579 FAIL (20x over); iter 3 `l1i8` (rank-1 int8, 1,024/3,072 bits)
  +0.000794/+0.001132 FAIL but the best untrained scheme. Attribution (one stream only, invalid as a
  candidate): card-only `l1i4c` +0.000371/+0.000843, note-only `l1i4n` +0.000496/+0.000508 - the
  streams are nearly additive, the immediate head is hurt most by the CARD stream. Full tables in
  `RESULTS.md`.
* 2026-11-19 20:20 - **speed measured for every config** with the new `rust_speed.py` (2 runs per
  config, deploy engine, parity traces 107/136/156): plain 2,863 rev/s / 5,947 states/s = 1.00x;
  rank-1 int4 0.903x rev/s (0.813x states/s); rank-1 int8 0.903x (0.804x); full-matrix int2 0.918x
  (0.812x); card-only 0.982x; note-only 0.949x; rank-2 int4 0.629x rev/s (**0.448x states/s - FAILS
  constraint 6**). So only rank 2 is disqualified on speed; the `R/s` retrievability pass stays at
  ~5,850-5,900 for every config.
* 2026-11-19 15:57 — **baseline eval running**: incumbent plain checkpoint
  `scratchpad/adaptive_wd/adaptive_fullrun_92_20261105/attempt_04_epoch_04_wd_0.01/adaptive_173416.pth`
  (4 epochs from QAT46, best plain model of this lineage; old deterministic-roster screen
  0.304392 ahead / 0.272142 imm) on the new random roster, tag `qbl_incumbent_a04`.
  Log `scratchpad/quant_plan/eval_incumbent_a04.out`, shard log `scratchpad/eval_shards/shard_s0.log`,
  summary `scratchpad/quant_plan/eval_summary_qbl_incumbent_a04.json`. ETA ~1 h.

## Constraint 6 measured on the Rust engine (the deploy path)

Harness (built 2026-11-19):
* `scratchpad/quant_plan/export_weights_only.py <pth> <dir> <name.safetensors>` — reuses
  `export_rnn_trace.py`'s own `export_weights()`, so the stream order / interleave / state-clamp
  recipe facts land in the safetensors metadata exactly as a normal export writes them. Weights:
  `scratchpad/quant_plan/rust_artifacts/adaptive173416.safetensors`.
* `rust/rwkv-infer/target/release/rwkv-infer.exe <users...>` with `RWKV_WEIGHTS`, `RWKV_TRACE_DIR`
  (`reference` = users 107/136/156, `reference_iter45` = users 101-104, both 92-wide), and the
  deploy quant env (`RWKV_STATE_QUANT_SCOPE` for full-matrix int-N, `RWKV_STATE_LOWRANK_SCOPE` for
  rank-r, `RWKV_QUANT_SHIFTS=1` + `RWKV_STATE_SHIFT_LEVEL` for the shift vectors). It prints
  `R/s (query pass, retrievability)` and `states/s (state pass, review_features)` per user — the
  `states/s` figure is the DEPLOY_SPEED_LOG metric, and rev/s is the per-review total.
* `scratchpad/quant_plan/rust_score.py --trace-dir <dir> --users ... --preds name=dir` — scores the
  written `rust_pred_*.json` against the trace's own equalize set and labels (same metric as
  `verify_rust.py`), so the deploy quantization tax is measurable per config.
* **cmd gotcha:** write env vars as `set "NAME=value"` — the bare `set NAME=value & cmd` form leaves a
  trailing space in the value and the engine then dies with `os error 3`.

Measured (concurrent with the roster eval, so treat the rates as indicative; a clean repeated bench
is done for the final candidate):

| config | card bits | note bits | states/s | ratio | rev/s | ratio |
|---|---:|---:|---:|---:|---:|---:|
| unquantized | 34,816 | 104,448 | 6,020 | 1.000 | 2,900 | 1.000 |
| full-matrix int2, shifts int2 (card+note) | 2,176 | 6,528 | 4,893 | 0.813 | 2,618 | 0.903 |
| rank-1 int4, shifts int4 (card+note) | 512 | 1,536 | 4,830 | 0.802 | 2,649 | 0.913 |
| rank-2 int4, shifts int4 (card+note) | 768 | 2,304 | 2,660 | **0.442** | — | — |

**rank-2 or deeper low-rank FAILS constraint 6 in the engine as written** (0.44x): only rank-1 has a
fused fast path; higher ranks run a per-step Gram+eigen decomposition. So the live families are
full-matrix int-N and rank-1 int-N (plus the PQ catalog variants of rank-1).

Deploy tax on the parity traces (`rust_score.py`, same traces, same weights):

| config | users | ahead tax | imm tax |
|---|---|---:|---:|
| full-matrix int2 card+note, int2 shifts | 107/136/156 | +0.003117 | +0.004410 |
| rank-1 int4 card+note, int4 shifts | 107/136/156 | -0.000099 | +0.000461 |
| rank-1 int4 card+note, int4 shifts | 101-104 (power users) | -0.000078 | +0.001038 |
| rank-2 int4 card+note, int4 shifts | 101-104 | +0.088324 | +0.109058 |

Read: the simplest scheme (full-matrix int2, exactly AT the bit bar) is 6-9x over the loss bar; rank-1
int4 is at or inside the bar on ahead and slightly over on imm for the power users. The rank-2 result
is a red flag for that path's stability (or a bug) and is only worth investigating if rank-1 fails.
Results table for Andrew: `scratchpad/quant_plan/RESULTS.md`.

## Anchor investigation (2026-11-19 22:00)

Andrew's rule for the anchor: the run must have a checkpoint BEFORE its one decay epoch (so the
decay can be redone with QAT) AND after it (for the baseline score); filter out runs without a
pre-decay checkpoint, then take the one with the lowest log loss on both heads. All of the following
checkpoints were verified to share the arm's geometry (444 tensors, 192,800 params, 92 inputs), so
they are all scoreable on the same roster:

| checkpoint | what it is | pre-decay checkpoint |
|---|---|---|
| `pretrain/rwkv/opt_iter36/rwkv_iter36_558.pth` | iter36 **WS-final** = the PRE-decay checkpoint every old-regime decay run started from | (is the pre) |
| `pretrain/rwkv/opt_iter36/rwkv_iter36_124.pth` | iter36 **after its decay** = the `[1,4,3,3,3]` champion / `champ_fp32` | `rwkv_iter36_558.pth` |
| `pretrain/rwkv/opt_qat44/rwkv_iter44_248.pth` | 248-step decay from iter36 WS-final (qat_log #44) | `rwkv_iter36_558.pth` |
| `pretrain/rwkv/opt_qat45/rwkv_iter45_496.pth` | 496-step decay from iter36 WS-final (qat_log #45) | `rwkv_iter36_558.pth` |
| `pretrain/rwkv/opt_qat46/rwkv_iter46_248.pth` | last old-regime decay run; the adaptive run's start | `rwkv_iter36_558.pth` |
| `adaptive_fullrun_92_20261105/attempt_01..11/adaptive_*.pth` | adaptive-WD run, 1 epoch = 43,354 steps each | previous attempt's final |

### CORRECTION 2026-11-19 22:20 -- the 558/124 pair was the OTHER WAY ROUND

The table above is right; the earlier version of this section (and the first `score_anchors.py` job
list) had `rwkv_iter36_558.pth` as the post-decay champion. It is not. Evidence:

* `optimization/qat_log.jsonl` #40 calls the iter36 lineage "**558 WS + 124 decay**".
* Decay runs RESTART their step counter: iter39/43 = `124` = a "124-step decay" from iter36 WS-final,
  iter44 = `248`, iter45 = `496`, iter46 = `248`. A continued counter would have given 682/372/620.
* File mtimes in `opt_iter36` come in two phases: `200,300,400,500,558` written 15:41-15:45 (the WS),
  then `50,100,124` written 15:46-15:47 (the decay, which overwrote the WS's own 50/100 saves).
* The optimizer files differ between the phases (1,894,918 B at 200-558 vs 1,895,302 B at 124), so
  they are two runs, and 124 is the LATER one.
* `reference/rwkv_iter39_124.safetensors` is named after its own decay's step count, 124.

So: **start the QAT redo from `rwkv_iter36_558.pth`; the baseline to beat is `rwkv_iter36_124.pth`.**

### The decay is 124 steps, and the "n-epoch" labels imply 31 groups/epoch

`qat_log` #44/#45 call their runs "8-epoch"/"16-epoch" and their step counts are 248/496, i.e. exactly
8x/16x **31**. Every old-regime step count in this lineage is a multiple of 31, so the data config
those runs used had **31 groups per epoch** -- and iter36's own decay was **4 of them = 124 steps**
(NOT the 2 epochs originally assumed). The old dbs are gone (see below), so 31 is not reproducible
today; the replay therefore fixes the STEP COUNT (124) instead, which is what the decay length
actually was.

### iter44/iter45 are NOT QAT checkpoints -- the cell-3 caveat is void

`HISTORY.md` (line ~256): "iter43/44/45 were NOT real QAT. The restored champion `architecture.py`
(arch_iter36) was MISSING the `[QAT]` scope parser that arch_iter41/42 have, so `state_qmax` stayed
inf -> fake-quant never ran (the iter44/45 logs have ZERO `[QAT]` lines). They were plain LONGER-DECAY
fine-tunes + PTQ int2 at the gate." So their unquantized scores are VALID fp32 numbers, and they are
legitimate anchor candidates (lowest loss = 16-epoch iter45). The parsers were restored afterwards.

### The old 1-100 training data is GONE; the replay uses the 5k db restricted to users 1-100

`train_db` / `train_db_sc8k` (users 1-100, the dbs iter36 and every old-regime decay ran on) exist
nowhere on C: or F: -- only the 5k dbs are left (`F:/rwkv_lmdb/train_db_5k_h1_adaptive92` and friends),
and `label_filter_db` in the repo root is the only survivor of the old pipeline. So the replay trains
on the 5k db with `TRAIN_USERS 1 100`. `probe_groups.py` (read-only, exact replica of
`get_groups`) says that range is 684 chunks / 9,664,102 rows, and at `MAX_TRAIN_GLOBAL_LEN=66000`
that is **160 groups/epoch, 0 chunks dropped** (the old db gave 170, so the two are close). The
adaptive arm already warm-started from an old-regime checkpoint (`opt_qat46/rwkv_iter46_248`) onto
this 5k db, so the feature layout is compatible; the `--no-quant` control run is what proves the
substitution still reproduces iter36's own loss level.

**Key fact that disqualifies the adaptive attempts as "decay epochs":** `attempt_04/train.log` prints
`[decay-shape] LR decay shape = constant (default is 1-sin(pi x/2), mass 0.3634)` and
`[decay-resume] LR schedule continues at step 130062 of 173416 (factor 1.000000)`. So attempts 1-4
are consecutive one-epoch segments of ONE flat-LR 4-epoch schedule (EPOCHS=4, PEAK_LR=0.001), not
standalone decay epochs, and none of them anneals the LR to zero. The controller's own
`final_qat/train.toml` (EPOCHS=1, TRAIN_MODE=D, PEAK_LR=0.001, loads attempt_04/adaptive_173416) is
the only fully-annealed "one decay epoch" of that arm, and its "after" checkpoint never existed
because the stage was blocked. Its "before" checkpoint is the incumbent itself.

**iter44/iter45 are valid anchors (earlier caveat withdrawn):** an earlier note here said they were
trained with int2 fake-quant ACTIVE and that scoring them unquantized would therefore be the invalid
"cell 3" comparison. That is wrong. `HISTORY.md` (line ~256) says iter43/44/45 ran with
`state_qmax = inf` because the restored `arch_iter36` architecture was missing the `[QAT]` scope
parser, so they are plain fp32 LONGER-DECAY fine-tunes (both logs have zero `[QAT]` lines). Their
unquantized scores are therefore valid, and they are legitimate anchor candidates. Their own
fp32-base numbers on the old 17-user gate are 0.287818 imm / 0.319089 ahead (iter45) and
0.296064/0.326631 (iter36 champ_fp32).

**Relaunched 2026-11-19 22:23 (corrected order):** `score_anchors.py` now scores `it36_124` FIRST
(the anchor baseline itself), then `it36_558` (the pre-decay checkpoint), then the `iter45/46/44`
diagnostics; finished jobs are skipped, so re-running it is safe. It also scores only the AFTER
checkpoints, so after the first job every delta is relative to the anchor rather than to
`adaptive_173416`.

**Chain 2026-11-19 22:23 -** `it36qat_chain.py` (detached, `it36qat_chain.log`) waits for the anchor
baseline eval, then runs: the 0.04-epoch smoke, the `--no-quant` fp32 control (124 steps), the QAT run
(candidate `c2_nl1i8` = card full-matrix int2 + note rank-1 int8, 124 steps), an unquantized eval of
each new checkpoint against `qnj_it36_124`, and finally the remaining anchor diagnostics.

**Stopped 2026-11-19 21:45:** the first full QAT run (`qat_l1i8e1`, started 21:35 from
adaptive_173416) was killed after ~10 minutes because it started from the POST-decay checkpoint of
its epoch. Its partial artifacts stay in `scratchpad/quant_plan/qat_l1i8e1/` but must not be resumed
or evaluated as a result. The validated smoke `qat_l1i8smoke` (step 1,734, `MODEL SAVED.`, 0 skipped
batches, quant banners active) still stands as the code-path check.

`qat_decay.py` now accepts `--start-folder/--start-name/--start-optim/--peak-lr/--dry-run`, so the
epoch can be redone from ANY checkpoint; the optimizer is auto-matched by step number and every path
is validated before launch.

## Open question for Andrew

Which checkpoint is "the best run" whose decay epoch we may redo, and what number is "current best"
for the +0.0005 bar? The deployed lineage (`s64hc10` / `s64hqr4b`, d=64, 10 WS + 1 decay, full-VAL
0.299369 / 0.267496) has **no checkpoint in this checkout** — only the 92-input d=32 lineage is
present (`pretrain/rwkv/opt_qat46/rwkv_iter46_248.pth` and the adaptive attempts). A search of
`F:\box_archive` (depth 3) found no `.pth`.

## Advisor consult (2026-11-19)

`gpt-6.1-sol` (via the advisor extension; first attempt was truncated, second succeeded). Its plan,
which this arm follows:

1. Read the geometry off the checkpoint, do not guess it (done: H=1, K=32, card 1 layer, note 3).
2. Draw a seeded random 750-user subset and **re-measure the baseline on it** — required, because
   the bar is an absolute +0.0005.
3. Do a cheap **PTQ attribution first** (card-only vs note-only) to find which stream carries the
   loss, then test PTQ variants, and spend GPU QAT-decay time only on the best PTQ variant.
4. Ranked methods: (a) re-budget the existing recipe (spend the freed bits on finer factor coding /
   residual stages / coarser-to-finer shift coding); (b) port the locked joint-uv pair-catalog recipe
   (72 b/card on the d=32 geometry, penalty +0.0011/+0.0002); (c) sensitivity-based bit allocation
   between card and note.
5. Treat +0.0005 as a **non-inferiority** test: report the paired mean delta and its CI per head.
   Measured seed noise is up to ~0.0005 on the immediate head, so a thin pass needs a same-recipe
   rerun at a second seed before it counts.
