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

* 2026-11-22 20:35 - **The note's int3 FACTOR rung is one step too far: `l2i8s4n3` lands as RESULTS row 38 and FAILS.** Card held at row 17's exact config (rank-2 int8 + int4 shifts, 1,280 bits) and the note's factors cut int8 -> **int3** (2,304 -> 1,344 bits, 2,624 total): 0.334383/0.307240 = **+0.000581/+0.001028** against `qnj_it36ctl` (ub95 +0.000671/+0.001129, p 2.6e-36/8.4e-75), i.e. ahead 1.34x and imm 2.26x over the ceiling. On row 17's own card the 960-bit cut costs about +0.00053 ahead / +0.00070 imm (difference of deltas, not a paired test), while row 35's int3 SHIFT rung on the int8 note cost only +0.000011/+0.000037 - so the note's FACTOR width, not its shift width, is the binding constraint. Corruption cell `-` (fails the loss gate, so it is never checked); recorded by the 21:05 autorun. `it36shrink` moved on to `l2i8s2n4` (card int2 shifts + int4 note) at 20:35.
* 2026-11-22 20:35 - **The int6 tax is mostly the NOTE's; the card-only int6 rung (`l2i6c`, 3,328 bits = 256 BELOW row 17) is nearly free.** 11 long-history users, per-user paired vs `l2i8s4`: `l2i6c` +0.000052 ahead [-0.000007, +0.000109] / +0.000115 imm [+0.000018, +0.000218], while the note-only twin `l2i6n` (3,200 bits) costs +0.000268/+0.000307. So the 21-user int6 price (+0.000442/+0.000726) is mostly the note's, and the card rung is the one to train at 750 users: chain 18 `it36int6card` (`l2i6c` then `l2i6n`, 124 steps vs `qnj_it36ctl`) is queued behind `it36int6`. `rust_wide_screen_it36int6split.json`.
* 2026-11-22 20:10 - **The deploy engine's finiteness guard now covers 40 long histories: 7,222,722,560 state values, ZERO non-finite.** `corruption/corrupt50_finite_run.py --tag corrupt50` (40 complete `trace_corrupt50` traces, `--strict`, so rc=0 is the engine's own statement): `l2i8s4` observed 7,053,440 compressions, 7,222,722,560 state values, 451,420,160 shift-vector values and 3,459,629 p(recall) values with zero bad on both the raw and the quantized side; `plain` 3,459,629 p(recall) values, zero bad. That closes the coverage gap that existed because the guard had only seen the 3-user parity trace - the Python gate (158,986 state calls on the same 50-user roster) and the shipped engine now agree. 10 traces were still exporting; re-run with `--tag corrupt50_rest`.
* 2026-11-22 20:05 - **Both points that landed since the last check-in FAIL, and each closes an axis question.** RESULTS row 36 (`l3i4s4n4`, card rank-3 int4 + int4 shifts, note rank-1 int4 + int4 shifts, 1,024 card / 1,536 note = 2,560 bits, `it36shrink`, landed 20:04:49) is the int4 note on top of the int4-rank card: 0.334353/0.307104 = **+0.000551 ahead / +0.000892 imm** vs `qnj_it36ctl` (ub95 +0.000634/+0.000997, p 1.4e-34/4.3e-64), so ahead is 1.10x and imm 1.78x over the ceiling, and the int4 note costs about +0.00034/+0.00042 on top of row 33 at the SAME 1,024 card bits (difference of deltas, not paired) - the int4-rank card cannot carry the note-pool cut, and 1,024 bits below row 17 buys nothing. RESULTS row 37 (`n2i4s3`, card held at row 17's 1,280 bits, note rank-2 int4 + int3 shifts, 2,112 note bits = the same 3,392 total as row 35, `it36equalbits`, landed 19:09:39) **FAILS by 0.000009**: 0.333999/0.306721 = +0.000197 ahead / +0.000509 imm (ub95 +0.000242/+0.000561, p 3.9e-15/1.4e-73), the imm point estimate and bound both just over the ceiling, so spending row 35's 2,112 note bits on RANK instead of factor precision costs about +0.000134/+0.000145 - factor precision beats rank on the 3-layer note a second time (row 26 first) and the note's int3 shift rung does not rescue rank 2. Both rows take `-` in the corruption column (not deploy candidates, never checked). Sequence at 20:05: `it36shrink` is on its next point (`l2i8s4n3` smoke started 20:04:49), `it36int6`/`it36e1`/`it36combo`/`it36corrupt2`/`it36e1shrink` wait; no chain needed a restart and `QUEUE.md` section 5 is empty. Recorded by the 20:05 hourly supervisor.

* 2026-11-22 19:45 - **int6/int5 priced on 21 long-history users: the factor-precision axis is nearly exhausted.** Paired deltas vs `l2i8s4`, bootstrap 95%: `l2i6s4` (2,944 bits, 640 below row 17) +0.000442 ahead [+0.000232,+0.000701] / +0.000726 imm [+0.000420,+0.001100], better on only 2/21 - and `l2i5s4` (2,624) +0.001458 / +0.002189, better on 1/21. Sign-consistent on 19-20/21, so the earlier 3-user read (+0.000345 imm) understated int6 by 2.1x. QAT recovers about half a PTQ tax on this sample, and row 17's own QAT imm is +0.000327 (ub95 +0.000390) against the +0.0005 ceiling, so chain 17's int6 point needs nearly the whole gap back: **int6 is at best a coin flip, int5 is expected to fail.** The screen's bottom rung (`l2i4s4n4`) was not measured (that name is in `quant_sweep`, not in `rust_grid`'s CONFIGS) and was not re-run - with int5 at +0.0022 imm it has no plausible path.
* 2026-11-22 19:40 - **The corruption column closes for rows 29-32: all four PASS on the 50 longest histories, histories not truncated.** `check_corruption.py` on `corrupt50.json` (2,349,827 reviews / 4,609,289 section rows): `l2i8n2i8` (row 29, 5.2 min), `l3i8n2i8` (row 30, 5.3 min), `l2i8n2i8s8` (row 31, 5 min) and `l2i8n3i8` (row 32, 5 min) each scored 50/50 users with 158,986 state calls and 150 p(recall) tensors - the SAME observation count as row 17, which is the control that the probe was really exercised - and ZERO non-finite values on both heads, with no NaN-guard skips. So the whole note rank ladder is clean at the history lengths that matter, and rows 29/30/32 are deployable on this gate. NEW TOOL: `corruption/gpu_hold.py` - a CPU-only, self-expiring queue hold that lets a non-chain GPU job (here the four checks) run without racing a chain handoff; it needs no GPU and cannot overlap GPU work. Still open: the Rust engine's own `--strict` finiteness guard has only seen the 3-user parity trace, and the 50-trace export is at 39/50 for `corrupt50_finite_run.py`.
* 2026-11-22 19:05 - **The note's shift precision has one free step: `n1i8s3` lands as RESULTS row 35 and PASSES at 3,392 total bits.** Card held at row 17's rank-2 int8 + int4 shifts (1,280 bits) and the note taken from int4 to **int3** shifts (2,304 -> 2,112 bits): 0.333865/0.306575 = **+0.000063 ahead / +0.000364 imm** vs `qnj_it36ctl` (ub95 +0.000129/+0.000426), so ahead sits at 0.26x of the ceiling and imm at 0.73x - a pass, and the 192-bit saving costs only about +0.000011/+0.000037 over row 17 (difference of deltas, not paired). The note's shift ladder therefore has a usable int3 rung, unlike the CARD's (row 34, int2, imm 1.06x over) and unlike the note's int2 (row 20, +0.000408 imm on the note). It cost 22.6 min of eval; speed cell = the measured rank-2 card path (2,358 rev/s = 0.809x, `rust_speed_idle_l2i8.json`) because the int3 level does not change the verify path. Corruption: `pending` - `it36corrupt2` has not scored this tag yet. Recorded by the 19:05 hourly supervisor; `it36equalbits` was running `n2i4s3` (same 2,112 note bits with rank-2 int4 factors) when this landed.

* 2026-11-22 18:30 - **The error-feedback screen is closed: EF is real but cannot shrink the state.** On 21 of the longest corrupt50 histories (per-user PAIRED deltas vs `l2i8s4`, bootstrap intervals): (a) `l2i8s4ef` (row 17 + EF, 1,536/3,072 = 4,608 bits) improves BOTH heads with intervals clear of zero, -0.001167 ahead / -0.002453 imm - so the mechanism works - but the buffer is real state and costs +1,024 bits, which is axis 2 and axis 2 is already saturated (rows 29-31). (b) `l2i4s4n4ef` (1,024/2,304 = 3,328, i.e. 256 BELOW row 17) is STATISTICALLY EQUAL to row 17 (+0.000049 ahead / -0.000376 imm, intervals over zero), so EF at int4 factors buys exactly a -7% state at equal accuracy and no more - the earlier 3-user read of -0.001310 imm shrank to -0.000376 at 21 users, which is the trap the new wide screen exists to catch. (c) `l2i4s4n4ef2` (buffer at int2, 896/1,920 = 2,816, exactly row 21's size) is clearly WORSE, +0.003439 ahead / +0.002696 imm with intervals well clear of zero and better on 1/6 and 0/6 users - the buffer IS the mechanism, so coarsening it removes the benefit. **EF is closed as an axis-1 lever** and needs no decision from Andrew unless a future recipe wants the -0.0025 imm at more bits; note it also cannot be gated today (Rust-only flag, eval-path gate). New instrument: `corruption/wide_screen.py` aggregates `rust_grid` batch tags into paired deltas with sign counts and intervals.
* 2026-11-22 18:05 - **Both points that landed since the last check-in FAIL, and each one closes an axis question.** RESULTS row 33 (`l3i4s4`, card rank-3 int4 + int4 shifts, 1,024 card / 2,304 note bits, `it36equalbits`, landed 17:16) is the int4-rank ladder's last rung with room to move: 0.334010/0.306683 = **+0.000208 ahead / +0.000472 imm** vs `qnj_it36ctl` (ub95 +0.000278/+0.000549), so ahead sits at 0.56x of the ceiling and imm is 1.10x over it - the int4 curve plateaus about 0.00047 imm short at any rank, i.e. RANK cannot buy back int4 factor precision on the card (it is still the cheapest card of that ladder: 0.00006/0.00011 better than row 23's rank-2 int4 for 256 extra bits). RESULTS row 34 (`l2i8s2`, card rank-2 int8 + **int2** shifts, 1,152 card / 2,304 note bits, landed 17:47) is the card-shift cut of row 15: **+0.000338/+0.000448** (ub95 +0.000432/+0.000532), ahead inside at 0.86x and imm 1.06x over, so cutting the card's shifts int8 -> int2 costs about +0.00029 ahead / +0.00014 imm for 384 bits, where int8 -> int4 (row 15 -> row 17) cost +0.000006/+0.000020 for 256 bits - **int4, not int2, is the shift floor on the CARD**, which keeps row 17 (1,280/2,304 = 3,584 bits, 0.800x) as the smallest passing card-shift configuration. Both rows take `-` in the corruption column (they fail the log-loss gate, so they are not deploy candidates and are never checked). `it36equalbits` is the active chain and has moved on to `n1i8s3@124` (eval started 17:55); `it36shrink`, `it36combo`, `it36corrupt2` and the interactive session's `it36e1`/`it36e1shrink`/`it36int6` are waiting for a free GPU.

* 2026-11-22 17:55 - **Error feedback (`RWKV_LOWRANK_EF`) is implemented in the deploy engine and has NO record anywhere in this lineage** (RESULTS.md, NOTES.md, the board and every other .md/.json under `optimization/` and `scratchpad/quant_plan` return nothing for LOWRANK_EF / error-feedback / noise-shaping / EF_RANK / EF_LEVEL / EF_PQ). The engine documents it as cancelling the compounding DC bias of factor quantization, which is exactly the shape of today's other finding (the factor-precision cost grows with history), so it was screened. Result: **inconclusive, and recordable only as such.** On the 3 longest users a rank-1 int4 error buffer looks good - `l2i8s4ef` 1,536/3,072 = 4,608 bits reads imm -0.002225 vs row 17, and `l2i4s4n4ef` (both streams int4 + EF) 1,024/2,304 = 3,328 bits reads -0.001310, i.e. 256 bits BELOW row 17 and still better. But the mean is dominated by user 5052 (imm 0.533-0.545 versus 0.170/0.265 for the others), which swings ~-0.006 under several unrelated configs (VLEVEL int3 too); the per-user deltas are +0.000187/-0.005988/-0.000874 for `l2i8s4ef`. So it is a one-user effect. **Blocker:** `RWKV_LOWRANK_EF` exists ONLY in the Rust engine, and the 750-user gate is an eval-path measurement whose Python quantization has no EF - so an EF recipe cannot be scored by the current gate at all. Turning EF into a candidate needs a decision from Andrew (implement EF in the eval path, or accept a Rust-path gate); it is NOT a change to make quietly because it touches how results are accepted. Next step needing no decision: widen the screen over the `trace_corrupt50` set (26/50 at 17:45) - the exporter is CPU-only and matches no BUSY_PAT, so this costs no GPU time.

* 2026-11-22 17:50 - **Method lesson about every deploy-side probe in this project.** All of them run on 2-3 users, and the per-user spread is enormous: on the same three long traces the imm baseline ranges 0.170 / 0.265 / 0.544. A mean over three such users is a mean over one easy, one medium and one hard trace, so any config that happens to suit the hard user looks like a win (and the hard user is the one whose behaviour is most sensitive to state precision). Absolute taxes on long traces are also unusable, because every quantized config reads ~0.0036 BETTER than `plain` (the documented trace-path offset). What survives this is the PAIRED per-user delta with a consistent sign: that is why `int6` (+0.000067/+0.000226/+0.000742, all positive) is credible and EF / VLEVEL are not. Treat any single-sign-free deploy-side delta below ~0.002 as unmeasured.
* 2026-11-22 17:50 - **Three long-history probe findings, all on the row-17 deploy weights over the 3 longest corrupt50 traces (users 7250/5052/5939, 28-29k reviews each).** (1) AXIS 3: `l2i8s4`, `l2i8s4f` and `l2i8s4f1` agree to <=3e-6 on both heads, so the fast rank-r solver with `RWKV_LOWRANK_FAST_ITERS=1` does not drift with history - with 0.800x -> 0.927x rev/s (median per-user 0.921x) the same weights, bits and log loss run 16% faster for free. (2) AXIS 1: the cost of cutting factor precision GROWS with history length. Per-user paired imm deltas vs `l2i8s4` are `l2i6s4` +0.000067/+0.000226/+0.000742 (mean +0.000345, std 0.00035) and `l2i5s4` +0.000629/+0.001147/+0.002877 (mean +0.001551, std 0.00118) - same sign on every user - against +0.000003/+0.000017 on the 2-user short-trace screen, so that screen understated int6 ~100x. **Chain 17 expectation on record: `l2i6s4` is a coin flip on imm (row 17's ub95 is already +0.000390 against the +0.0005 ceiling) and `l2i5s4` / `l2i4s4n4` are expected to FAIL.** (3) AXIS 1 NEGATIVE: `RWKV_LOWRANK_VLEVEL` is dropped - at the same 640-bit saving V int4 costs +0.001015 (std 0.00118) vs int6's +0.000345, and the VLEVEL spread is ~15x wider with V int3's negative mean driven by one user (5052 -0.006228 vs +0.001639/+0.003570), i.e. coarsening V destabilizes the trajectory rather than adding a smooth error. Method: per-user PAIRED deltas are what make these readable, because on long traces every quantized config reads ~0.0036 better than `plain` (the known trace-path offset), so absolute taxes cannot rank configs.

* 2026-11-22 17:45 - **`it36equalbits` is the active chain** (it took the GPU at 16:46:55 and has since run `l3i4s4` and started `l2i8s2` at 17:18:25). `it36shrink` is still waiting for a free GPU handoff, which is by design: both it and `it36equalbits` wait only on an idle GPU (`it36shrink` on the ended `it36noterank2`, `it36equalbits` on the ended `it36corrupt`) - the race resolves to whoever polls first, and `it36combo`/`it36corrupt2`/`it36e1`/`it36e1shrink`/`it36int6` queue behind them in order. **Operational rule learned the hard way:** a CPU-only job that matches a chain's BUSY_PAT (including `rust_grid.py`, `rwkv-infer.exe`, and even a `DEVICE=cpu` `rwkv.train_rwkv`) strands the queue across a handoff, because `wait_for_prev` returns only on `ended AND busy == 0`. Two of my own jobs did exactly that at 16:42 and idled the GPU; both were killed. Such jobs must fit inside a point's own window (smoke 1-3 min + decay 7-10 min + eval 20-56 min), which is how the three probes above were run.
* 2026-11-22 16:25 - **Chain 14 finished; its last point lands as RESULTS row 32 (`l2i8n3i8`, card rank-2 int8 + int4 shifts, note rank-3 int8 + int4 shifts, 1,280/5,376 = 6,656 bits): PASS on both heads, 0.333817/0.306334 = +0.000015 ahead / +0.000122 imm vs `qnj_it36ctl` (ub95 +0.000033/+0.000142).** It CLOSES the note's rank ladder: rank 3 is worse than rank 2 on BOTH heads (row 29 on the same card reads -0.000115 ahead / +0.000103 imm) for 1,536 MORE note bits, and worse than rank 2 on a rank-3 card too (row 30). The note's residual tax is not rank-limited, which is the premise the E1 tail-energy penalty rests on - chains 15/16 stay the right next move. `it36shrink` then took the GPU at 16:46 (candidate `l3i4s4`) after the queue's first free handoff; `it36equalbits` and `it36shrink` both waited only on an idle GPU, which is by design (`it36equalbits` waits on the old `it36corrupt` log, already ended).

* 2026-11-22 16:45 - **`int6` and `int5` are now trainable, and the factor-precision ladder is queued as chain 17 (`it36int6`).** `rwkv/architecture.py:151` `_QMAX` held only int8/int4/int3/int2/fp32, so `card:2:int6` raised `KeyError` in `_apply_qat_scopes`; the Rust deploy engine has always spoken both levels (`parse_level`, `model.rs:2104-2105`) and both quantization paths are generic in qmax, so the keys were added (additive only - no existing recipe names them), together with `quant_sweep._QBITS`/`_QMAX` and three candidates at row 17's EXACT layout (card rank 2, note rank 1, int4 shifts): `l2i6s4` 1,024/1,920 = 2,944 bits (640 below row 17), `l2i5s4` 896/1,728 = 2,624 (960 below, and 192 below row 21), `l2i4s4n4` 768/1,536 = 2,304 (1,280 below; the only BOTH-int4 combination - row 21 was note-int4 only and row 23 card-int4 only). A 2-user Rust PTQ probe on the row-17 checkpoint put `l2i6s4` at +0.000003 ahead / +0.000002 imm against `l2i8s4` for 640 fewer bits and `l2i5s4` at +0.000017/-0.000073, but the probe CANNOT settle the gate: calibrated against rows 21 and 23 it reproduces the note-int8->int4 cost at only 54% and gets the card-int8->int4 sign WRONG, so every point is a full 750-user QAT run. The int6 path was pre-flighted in the real trainer on CPU while the GPU was busy: the `[QAT-LOWRANK]` banner reported `card_id=rank2/fq31.0, note_id=rank1/fq31.0` and the first step trained finite (0.837127 / 0.3954 / 0.442); that pre-flight was then stopped, because any process matching the chains' BUSY_PAT strands the queue's handoffs.
* 2026-11-22 15:35 - **Row 31 is loss-only, E1 (chain 15) waits behind `it36corrupt2`, and the iter-17 bundle is zipped and re-verified from a fresh extract.** `l2i8n2i8s8` cannot be expressed by the deploy engine: `RWKV_STATE_SHIFT_LEVEL` takes ONE bare level and `parse_level` silently drops `card:int4,note:int8`, so that string behaves exactly like `int8` (user 107: max abs diff 0.0 vs `int8`, 0.087 vs `int4`). The `l2i8n3i8` smoke wedged at 14:57 (0.00 s CPU for 20 min, GPU idle, no disk events) and the lib's 20-min monitor killed and retried it - passed in 1.6 min; the four 14:51 `l2i8n2i8s8` eval processes are still alive at 0 CPU, children that outlived an `rc=0` parent. E1's tail measurement landed: the card at rank 2 keeps 0.99381 of the state energy, the note at rank 1 keeps 0.96943 (tails 0.62% / 3.06%). `scratchpad/row17_l2i8s4_bundle.zip` (12.6 MB, 180 entries) was rebuilt after running `verify.cmd` from a fresh extract exposed two defects (it named a trace the bundle does not ship, and its by-design-failing step 4c was fatal); the corrected run reproduces every documented number. The Rust finiteness check is being extended to the real 50-user roster (10/50 traces exported).
* 2026-11-22 15:25 - **Chain 14 point 3 lands as RESULTS row 31 (`l2i8n2i8s8`, card rank-2 int8 + int4 shifts, note rank-2 int8 + int8 shifts, 1,280/4,608 = 5,888 bits): PASS on both heads** - 0.333683/0.306295 = -0.000119 ahead / +0.000084 imm vs `qnj_it36ctl` (ub95 -0.000088/+0.000108, p_worse 1.0000/7.2e-11). The best imm tax of the note-rank-2 family, but it spends 768 more note bits than row 29 for 0.000019 of imm, so int8 note shifts are not free. **Loss-only: `RWKV_STATE_SHIFT_LEVEL` takes ONE bare level** and `card:int4,note:int8` is silently ignored (verified: predictions identical to `int8`, 0.087 away from `int4`), so a per-stream shift level needs a parser change before this row can be quoted against constraint 6. Chain 14 moved to its last point (`l2i8n3i8`) at 14:56:47.

* 2026-11-22 15:30 - **Three findings from building the iter-17 bundle, recorded because they change how earlier numbers must be read.** (1) `reference/trace_user_*` is a June-2026 export whose FEATURE recipe differs from the current one (same review indices, feature values up to 3.0 apart), so scoring current weights on it is out of distribution: the deploy tax measured there read +0.0027 imm / -0.0030 ahead, while the same measurement on a fresh current-lineage export reads +0.000316 imm / +0.000335 ahead, in line with row 17's own 750-user gate (+0.000327 imm). `rust_score.json`, `rust_speed_idle.json` and `rust_finite_probe.json` stay valid as INTERNAL comparisons (Rust vs Rust on one trace set) but not as levels. (2) The Python RNN inference path (`rwkv_rnn_model` -> `rwkv_ops.single_timestep`) does NOT apply the state quantization; only the training/eval path (`quant_aware_rwkv7`) does. A trace's embedded `py_pred_*` is therefore an UNQUANTIZED reference: the engine's `plain` config matches it to <=8.5e-07 per review and 0.000000 log loss on 3,465 / 3,865 / 12,915 / 23,455-review histories, and a quantized Python<->Rust comparison must come from the eval path. (3) The AHEAD head is not comparable between the eval path (LMDB) and the trace path: they differ by about 0.005 log loss on a long user WITH and WITHOUT quantization (the unquantized control shows the same gap), i.e. it is an input difference in the exporter's frame replica, not a quantization difference. The IMM head agrees to 8e-05 on the long user. Compare ahead numbers only within one path.
* 2026-11-22 14:05 - **Chain 14 point 2 lands as RESULTS row 30 (`l3i8n2i8`, card rank-3 int8 + note rank-2 int8 + int4 shifts, 1,792/3,840 = 5,632 bits): PASS on BOTH heads, and the lowest tax measured so far** - 0.333658/0.306198 = **-0.000144 ahead / -0.000014 imm** vs `qnj_it36ctl` (ub95 -0.000121/+0.000002, p_worse 1.0000/0.851). The immediate head's 95% bound sits exactly ON the anchor, so for the first time the imm head carries no measurable tax at all. It beats row 29 by 0.000029/0.000117 for 448 more card bits, but rank 2 makes the note LESS TAXED, not SMALLER (3,840 bits vs the 2,304-bit int8 rank-1 note of rows 17-18 that already passes), so row 29 remains the axis-1 preference at 5,120 bits and this row is the loss instrument. Speed 0.791x with `RWKV_LOWRANK_FAST=1` (0.611x without). Corruption: pending (`it36corrupt2`). Also repaired the stale RESULTS.md sentence that still said a rank>=2 note is not deployable. `RESULTS.md` row 30, `quant_sweep_it36l3i8n2i8.json`, `rust_speed_fastrank_on.json`.

* 2026-11-22 14:15 - **The rank>=2 NOTE speed blocker is gone: E2 (the fast rank-r solver) is implemented, proven and measured, and row 29 is now a loss-gate AND speed pass.** `RWKV_LOWRANK_FAST=1` replaces the KxK Gram + full symmetric eigensolve in `compress_wkv_state` with a warm-started block subspace iteration on fixed-size arrays + Rayleigh-Ritz on the (R+1)x(R+1) projection; the tracked block is the R wanted directions plus one extra, refreshed from the strongest column of A on every call, so a new dominant direction can still enter, and it travels with the per-entity state (`FastLayerState::basis`). Same session, `it36l2i8n2i8_124.safetensors`, users 107/136/156, 2 reps: `l2i8n2i8` **2,429.2 rev/s = 0.803x** (reference solver 1,801.5 = 0.618x) and `l3i8n2i8` **2,393.0 = 0.791x** (reference 1,781.4 = 0.611x), against plain 3,025.1 / 2,915.1 rev/s; the rank-2 CARD configs improve too (2,357.8 -> 2,544.9 = 0.841x). **Proof it is safe to deploy:** `RWKV_LOWRANK_FAST_CHECK=1` runs BOTH solvers, records the largest relative Frobenius difference and still WRITES the reference factors, so a checked run is byte-identical to the no-flag run on all three configs; on `l2i8n2i8` 95.6% of 64,132 rank>=2 calls are under 1e-3 relative, 57 above 1e-1, none above 1.0, and 3.2% fall back to the reference (degenerate states, cheap there); the scored deploy cost is imm +0.000051 / ahead +0.000000, the same order as the quantization tax; and a finiteness run with the flag ON passes (`rust_finite_fastrank.json`: 0 non-finite of 65,671,168 state values, state_maxabs 0.62). `RWKV_LOWRANK_FAST_ITERS` stays at its default 2 - 4 was measured WORSE on the deploy metric (scored imm +0.000221 vs +0.000051, worse relative-Frobenius histogram, `rust_fastrank_iters2.json`). The flag is OFF by default, is recorded in `rust_grid.CONFIGS` for the three rank>=2-note configs (it does NOT change `bits_for`), and must stay UNSET for any Python<->Rust parity run. No 0.04-epoch smoke applies: this is an engine solver change with no training path. **Files:** `rust_fastrank_check.py` + `rust_fastrank_check.json`, `rust_speed_fastrank_{on,off}.json`, `rust_finite_fastrank.json`, `_analyze_fastrank_drift.py`, the `compress_bench` test module in `model.rs` (the eigensolve is ~35 us of a ~43 us rank>=2 compress call). RESULTS.md row 29 + a new fast-solver section; GPU_RUN_BOARD amendment 13. `cargo build --release` clean, `cargo test --release` 11/11 (the bench is the 12th, ignored). **Also repaired:** the RESULTS.md paragraph after the card-rank-ladder table was truncated mid-sentence ("...as the 750-user eval") with its tail "does)." orphaned after the corruption paragraph - the two halves are rejoined; no text was invented.

* 2026-11-22 13:05 - **Row 29 (`l2i8n2i8`) PASSES the loss gate, and it is the first rank-2 NOTE point.** `it36noterank2` finished its first point at 12:58 (42.0 min eval): card held at row 17's `l2i8s4` config, note moved to rank-2 int8 + int4 shifts, 1,280 card / 3,840 note bits = 5,120. Result 0.333687/0.306314 = **-0.000115 ahead / +0.000103 imm**, ub95 -0.000084/+0.000127 against `qnj_it36ctl` - ahead is BETTER than the anchor and imm is 0.21x of the ceiling. That reverses the note-pool conclusion of rows 21-22: on the note, RANK removes the tax that int4 factors could not (they were 1.05-1.3x over), exactly as `cn8` predicted, since the tax is truncation. **The blocker is now the ENGINE, not the loss**: a rank-2 note path reads 1,743.0 rev/s = 0.600x (`rust_speed_probe_note2.json`), under the 0.75x constraint, so row 29 is not deployable until the engine has a fast rank-r note path. Corruption is `pending` - `it36corrupt2` (16 gate-guarded points, four of them this chain's) runs last and is still waiting. The chain is now on its second point, `l3i8n2i8@124` (started 12:59), which is card rank-3 + the same rank-2 note.
* 2026-11-22 12:55 - **The Rust finiteness check is finished, proven, and measured to be free when off.** `rust/rwkv-infer/src/finite.rs` (new, `RWKV_FINITE_CHECK` off by default / `=1` counts / `=strict` exits 3) counts non-finite values on the DEPLOY path, where `san()` otherwise hides a blown-up p(recall) as 0.5 and `compress_wkv_state` skips degenerate components; it only observes, so a checked run is bit-identical. Three fixes the advisor demanded, and one bug found by making them. **(1) OFF IS FREE, measured against the pre-change build.** `measure_offpath_oldbuild.py` read HEAD's `main.rs`/`model.rs`, built them into a throw-away target dir, and ran both binaries back to back on all 18 reference users, 3 reps each, check OFF: plain 2,903.7 vs 2,838.9 rev/s (ratio **1.023**), `l2i8s4` 2,287.5 vs 2,275.2 (**1.005**), and the 36 prediction dumps are byte-identical. So the published constraint-6 numbers (0.809x etc.) stay valid with no re-measurement. **(2) ON costs about 1%**: `measure_finite_offpath.py`, 18 users, 5 interleaved rounds, median ON/OFF plain **0.985**, `l2i8s4` **0.991** - the state scan itself - and the dumps are byte-identical there too. **(3) `maxabs` split into `state_maxabs` / `vec_maxabs`**, because the old mixed number was the SHIFT vectors hiding the state: the engine now reads state 0.533-0.647 vs shifts 5.17 on the probe users, matching the Python probe (state 0.35, shifts 5.23). `cargo test` 11/11 pass; the strict path was re-proven after the change (`RWKV_FINITE_SELFTEST=1` -> `planted=1`, engine exits 3, driver reports `FAIL ... offending keys: ['planted']`). **Bug found by the split:** `rust_finite.py`'s TOTAL regex hard-coded `p_bad=.. planted=..`, so inserting the two maxima between them silently stopped the match and turned a planted STRICT failure into `INCOMPLETE`; the driver now parses the TOTAL line as generic `key=value` pairs. **Also fixed `rust_grid.bits_for`**: a BARE `RWKV_STATE_SHIFT_LEVEL` (what every deploy config uses, since the engine's `parse_level` takes ONE global level) was read as note-only, so the card fell back to its WKV width and `l2i8s4` was reported as 1,536 card bits instead of the true 1,280. It now agrees with `quant_sweep.bits_for` on all six shared names (`l2i8` 1,536/3,072, `l2i8s4` 1,280/2,304, `l3i8s4` 1,792/2,304, `l2i8n2i8` 1,280/3,840, `l3i8n2i8` 1,792/3,840, `l2i8n3i8` 1,280/5,376). No recorded artifact changes: a recomputation pass over every `rust_grid_*.json` shows the fix explains NONE of their `bits` fields (`rust_grid_small.json`'s `l1i4` note reads 6,912, which this bug cannot produce - it is drift from an older config definition - so it was left alone on purpose), and RESULTS.md already quoted `quant_sweep`'s correct numbers. The docs that said a rank >= 2 note "needs engine work" now say what was measured: it RUNS at 0.600x (rank 2) / 0.581x (rank 3) and needs a FAST rank-r path, not a parser change.

* 2026-11-22 11:50 - **The owed idle-machine deploy measurement is done, and the rank-2 card path really is 0.81x.** `rust_speed_idle_l2i8.json` (weights `rust_artifacts/it36l2i8_124.safetensors`, users 107/136/156, 2 reps, GPU idle, `plain` baseline 2,915.1 rev/s): `l1i8` 2,628.4 rev/s = **0.902x**, `l2i8` 2,357.8 = **0.809x** (states/s 0.658x, median per-user 0.795x), `l2i8s4` 2,357.8 = 0.809x, `l3i8s4` 2,357.8 = 0.809x - all above the 0.75x floor. So the 0.812x/0.800x cells of rows 15-18, measured while a decay was training, were accurate to about 1% and were NOT 3% pessimistic as RESULTS.md feared; the RESULTS.md speed paragraph and Status line now say this, and `QUEUE.md` section 3 is closed. Two details worth keeping: the three rank>=2 card configs are indistinguishable in speed (neither the factor width nor the shift width changes the card's verify path, only the rank does), and `states/s` is 0.658x while `rev/s` is 0.809x - constraint 6 is written on rev/s, which is the one that matters.
* 2026-11-22 11:45 - **The corruption check is FINISHED and every point PASSES: rows 15-18 and the fp32 control are all clean on the 50 longest-history users.** `it36corrupt` logged `=== it36corrupt chain end ===` at 11:41:38 after five points, 4.7-4.8 min each (the fp32 control only 1.8 min, because it has no quantizer to run). Verdict table, from `corruption/corruption_corrupt_<tag>.json`:

  | point | row | state calls | state non-finite | p(recall) tensors | p non-finite | scored | mean LogLoss ahead / imm on this roster |
  |---|---|---|---|---|---|---|---|
  | `l2i8` | 15 | 158,986 | 0 | 150 | 0 | 50/50 | 0.382082 / 0.366948 |
  | `l3i8` | 16 | 158,986 | 0 | 150 | 0 | 50/50 | 0.382168 / 0.366710 |
  | `l2i8s4` | 17 | 158,986 | 0 | 150 | 0 | 50/50 | 0.382094 / 0.366979 |
  | `l3i8s4` | 18 | 158,986 | 0 | 150 | 0 | 50/50 | 0.382198 / 0.366806 |
  | `ctl_fp32` | 0 | 0 (no quantizer) | - | 150 | 0 | 50/50 | 0.382646 / 0.366875 |

  Two things to read out of that. (1) The state count is identical for all four quantized rows (158,986) because the roster, the layer count and the sequence plan are the same - rank and bit width do not change how many state values exist, only their representation, so the four rows are directly comparable. (2) On these 50 long-history users quantization is **not** worse than fp32 on the ahead head (0.38208-0.38220 vs the control's 0.38265) and about equal on the immediate head (0.36671-0.36698 vs 0.36687), i.e. the blow-up risk Andrew worried about does not appear even as a loss regression at the extreme of the history-length distribution. That is a 50-user roster, not the 750-user gate roster, so it is not an acceptance result - it is the finiteness gate plus a sanity band. Live cells: RESULTS.md rows 0 and 15-18 now read `PASS`, no cell is `pending`. The GPU went idle at 11:41:38 and no chain is queued (all other chains are PAUSED by Andrew's instruction), so the next step is the owed idle deploy measurement of `QUEUE.md` section 3, which is the only item still authorized.
* 2026-11-22 11:30 - **Andrew's stop instruction is in force: the corruption check is the only thing running.** At 11:21 he said "Let's stop further runs for now and ONLY do corruption checks for (as of right now) passed runs", so at 11:24 I killed `it36shrink`, `it36equalbits`, `it36combo` and `it36corrupt2` and deleted their lock files (their logs have no end marker now). `QUEUE.md` sections 2 and 5 mark those four as PAUSED and forbid a restart - that guard matters, because the supervisor's restart rule only needs `GPU IDLE` plus a 45-min-stale log, which those logs will look like as soon as `it36corrupt` ends. **The rows that pass as of now are exactly 15 (`l2i8`), 16 (`l3i8`), 17 (`l2i8s4`) and 18 (`l3i8s4`)**: every other sweep FAILS, including the overnight ones (row 24 `l4i4s4` +0.000218/+0.000463, row 25 `l7i4s4` +0.000209/+0.000466, row 26 `n2i4s4` +0.000180/+0.000492) and both longer decays (row 27 +0.000715/+0.000793 at 496 steps, row 28 +0.000554/+0.000829 at 992). So the corruption chain's five points are those four rows plus the quantization-OFF control `ctl_fp32`. **The 02:33-11:15 stall was hardware, not the code.** `_diag_stall.py` showed `rwkv.train_rwkv` (pid 29396) and its fetch worker (26952) at 0.000 s user and 0.000 s system CPU over a 6 s window while the log said `status: running`, and `_diag_disk.py` found the System-log events `[disk] 153` (I/O retried at LBA 0x131488190) and `[disk] 51` (paging error on `\DR1`) at 02:33:45-46 with F: = Disk 1 (2.49 TB free): the F: LMDB read wedged the training loop for 8.7 h. Two fixes: `it36ladder_lib.run` is now a watchdog (20 min with neither log growth nor tree CPU -> `taskkill /F /T` the tree, one retry, `STALL_RC=-9`), and `status_report.py` prints a `CPU over 5 s:` line per workload process, so a wedge reads `WEDGE? live workload processes burned no CPU` instead of `GPU BUSY` (a healthy 4-process tree reads about 14 s of CPU per 5 s of wall time, ~2.8 cores; "all sampled processes exited during the sample" means a chain changed point between the two samples, which is normal).
* 2026-11-22 11:25 - **Row 17 (`l2i8s4`) is the first row to PASS the corruption check.** `it36corrupt` scored it to `corruption/corruption_corrupt_l2i8s4.json`: rc=0, 50 of 50 users scored, **158,986 raw card/note state observations and 150 p(recall) tensors, 0 non-finite, 0 NaN skips**, 4.8 min on the 50-user longest-history roster (no truncation), on the Python QAT path (`RWKV_QAT_FUSED=0`, the bit-parity twin of the fused CUDA kernels) so the state is visible before `_sanitize_state` caps it. The `corruption` cell of RESULTS.md row 17 is filled by `corruption/record_corruption.py`, the only writer of that column; `pending` in a cell means the chain has not reached that row yet. Points 2-5 (`l2i8` row 15, `l3i8` row 16, `l3i8s4` row 18, `ctl_fp32` attribution control) follow, about 5-10 min each; a corruption FAIL is a stop-the-line event (QUEUE.md section 7) and would need a state-growth fix plus the advisor consult Andrew authorized.
* 2026-11-22 03:05 - **Row 28 `l2i8` at 992 steps FAILS, so the length axis stays closed with rank too.** The card-rank-2 recipe of row 15 (1,536/3,072 bits) trained for 992 steps instead of 124 scores 0.317768/0.290460 = **+0.000554 ahead / +0.000829 imm** against its own 992-step twin `qnj_it36ctl992` (row 11's control, 0.317213/0.289632), ub95 +0.000660/+0.000916, p 8.7e-30/3.7e-67: ahead is 1.1x and imm 1.66x over the +0.0005 ceiling, and both heads are worse than the same recipe at 124 steps (+0.000046/+0.000307, row 15) and at 496 steps (+0.000715/+0.000793, row 27). The speed cell is row 15's measured rank-2 card path (2,324 rev/s = 0.812x), because the factor width does not change that path. Recorded as RESULTS.md row 28, corruption `-` (not a deploy candidate), stamped in `autorun/RECORDED.txt`. Chain state at this check-in: GPU BUSY - `it36longdecay` ended cleanly at 02:29:54 and `it36shrink` started its `l3i4s4n4` QAT decay at 02:32:04, with 4 workload processes live; `it36corrupt`, `it36equalbits`, `it36combo` and `it36corrupt2` wait behind it. No corruption verdict exists yet (only the CPU smoke files), so the `pending` cells of rows 0 and 15-18 are unchanged.
* 2026-11-22 02:05 - **Row 27 `l2i8` at 496 steps FAILS: the passing card-rank recipe does not survive a longer decay.** The row-15 recipe (card rank-2 int8 + note rank-1 int8, 1,536/3,072 bits) trained for 496 steps instead of 124 scores 0.322374/0.295406 = **+0.000715 ahead / +0.000793 imm** against its own 496-step twin `qnj_it36ctl496` (row 0b), ub95 +0.000802/+0.000870, p 8.3e-50/1.7e-63 - ahead is 1.43x and imm 1.59x over the +0.0005 ceiling, and both heads are WORSE than the same recipe at 124 steps (+0.000046/+0.000307, row 15). So the rank ladder's gain is a 124-step result: the decay-length axis stays closed at the champion's 124 steps, which is what the rank-1 rows 8/11 already showed, now confirmed with the recipe that actually passes. Recorded as RESULTS.md row 27, corruption `-` (not a deploy candidate), stamped in `autorun/RECORDED.txt`. Chain state at this check-in: `it36longdecay` went straight on to `l2i8@992` (sweep started 02:02:01, GPU BUSY), and `it36shrink`, `it36corrupt`, `it36equalbits`, `it36combo`, `it36corrupt2` wait behind it in that order. No corruption verdict exists yet (only the CPU smoke files), so the `pending` cells of rows 0 and 15-18 are unchanged.
* 2026-11-22 01:05 - **Row 26 `n2i4s4` FAILS by a hair, and the note answers the card's rank-vs-precision question the OPPOSITE way.** Card held at row 17's exact config (rank-2 int8 + int4 shifts, 1,280 bits), note moved from rank-1 int8 to **rank-2 int4** factors with int4 shifts - the same 2,304 note bits, so the row is row 17 with rank and precision swapped on the 3-layer note. It scored 0.333982/0.306703 = **+0.000180 ahead / +0.000492 imm** against `qnj_it36ctl` (ub95 +0.000227/+0.000543): both point estimates sit inside the +0.0005 ceiling, but the imm head's upper bound is 1.09x over it, so it FAILS. Against row 17 (identical card, identical bits, +0.000052/+0.000327) the swap costs about +0.000128 ahead / +0.000165 imm (difference of deltas, not a paired test) - the reverse of row 23, where rank-2 int4 beat rank-1 int8 on the 1-layer CARD by about +0.00025/+0.00045. So at a fixed bit budget **rank buys the card and precision buys the note**, which fits the whole frontier: every passing note is the int8 one, and row 26 is the first and only rank-on-note point run so far, so treat it as a single point rather than a ladder. Recorded as RESULTS.md row 26, corruption `-` (not a deploy candidate), stamped in `autorun/RECORDED.txt`. Speed cells are `*see note*`: the card is row 17's measured 2,290 rev/s = 0.800x, but a `note:2:int4` path has never been measured on the engine and would need engine work, and the row fails the loss gate. Chain state at this check-in: `it36rankprec` ended cleanly (00:52:16) and the GPU is BUSY on `it36longdecay` (the 496-step `l2i8` decay, started 00:53:42, 11 min in at 01:05); `it36shrink`, `it36corrupt`, `it36equalbits`, `it36combo`, `it36corrupt2` wait behind it in that order. No corruption verdict exists yet (only the three CPU smoke files), so the `pending` cells of rows 0 and 15-18 are unchanged.

* 2026-11-22 00:05 - **Rows 24-25 land from `it36rankprec`, and the int4-rank ladder FAILS.** `l4i4s4` (card rank-4 int4 factors + int4 shifts, note at row 17's rank-1 int8 + int4 shifts, 1,280/2,304 bits) scored 0.334019/0.306674 = **+0.000218 ahead / +0.000463 imm** (ub95 +0.000287/+0.000538) and `l7i4s4` (rank-7 int4, 2,048/2,304) 0.334011/0.306678 = **+0.000209/+0.000466** (ub95 +0.000279/+0.000542) against `qnj_it36ctl`: the ahead head is inside the ceiling on both, the imm head is 1.08x over on both. At a FIXED card-bit count the int8 cards beat them on both heads by about +0.00014 (row 17 vs row 24) and +0.00029 (row 16 vs row 25, difference of deltas, not a paired test), so past rank 2 FACTOR PRECISION is the binding constraint; the int4-rank imm curve is rank 2 +0.000580 (row 23), rank 4 +0.000463, rank 7 +0.000466, i.e. it buys about 70% of the gap once and then plateaus, so `l3i4s4` in `it36equalbits` is the last int4-rank point with room (it must close about +0.00025 of imm tax in one rank step). Recorded as RESULTS.md rows 24-25, corruption `-` because neither is a deploy candidate, and stamped in `autorun/RECORDED.txt`. No deploy-engine run exists for a `card:4:int4`/`card:7:int4` path, so their speed cells are `*see note*` (rank>=2 card paths measured so far: 0.812x at rank-2 int8, 0.800x at rank-2/3 int8 with int4 shifts, both above the 0.75x floor). Chain state at this check-in: `it36rankprec` went straight on to `n2i4s4` (QAT decay started 00:01:30, GPU busy), and `it36longdecay`, `it36shrink`, `it36corrupt`, `it36equalbits`, `it36combo`, `it36corrupt2` wait behind it in that order.

* 2026-11-21 23:45 - **The night queue is closed and the corruption column now fills itself.** Two chains were added behind the running ones, so the GPU has work through the night and the morning. (1) **`it36combo` (chain 13, 3 points)** crosses the two cut axes that until now were always measured with the other stream held at row 17's level: `l3i4s4n8s3` (1,024/2,112 = 3,136), `l3i4s4n3` (1,024/1,344 = 2,368) and `l2i4s4n8s3` (768/2,112 = 2,880) - the cheap card UNDER the cheap note, which is the only way the total beats 3,584; a failure says the two cuts do not compose, which is the answer the frontier needs. (2) **`it36corrupt2` (chain 12, 8 points)** exists so that every row chains 11/13 create gets its corruption verdict - and only if it is a deploy candidate: each point reads its own sweep verdict first and SKIPS when `ub95 > +0.0005` on either head (row 23 already fails, so it is skipped) or when no verdict exists yet (a missing verdict is never a pass). It runs after chain 13, so the sweeps it guards already exist. (3) **`corruption/record_corruption.py` is now the only writer of the corruption CELLS**: it reads `corruption_corrupt_<tag>.json`, finds the tag's row (built-in map for rows 0/15-18/23, otherwise the unique row naming the tag), and replaces the last cell - refusing any row that is not 12 cells wide. `QUEUE.md` section 4 gained step 4 telling the hourly supervisor to run it before it pushes RESULTS.md. (4) `check_corruption.py`'s CONFIGS gained the 8 later tags. Verified before launch: `record_corruption.py` filled a synthetic verdict into row 17 and the cell was restored byte-for-byte (RESULTS.md has no diff), and the gate guard returns False for `it36l2i4s4`, True for rows 17/18 and None for the not-yet-run chain 11 stamps. Queue order now: `it36rankprec` (running) -> `it36longdecay` -> `it36shrink` -> `it36corrupt` (10) -> `it36equalbits` (11) -> `it36combo` (13) -> `it36corrupt2` (12): 8 heavy decay points plus at most 8 cheap corruption points.

* 2026-11-21 23:25 - **Chain 11 `it36equalbits` is queued behind `it36corrupt`, so the GPU does not idle when the current queue drains.** The chain takes the equal-bits question rows 9/23 settled on the card (rank beats precision at a fixed bit count) to the three places the running ladder leaves open, four 124-step decay points all scored against row 0: **`l3i4s4`** (card rank-3 int4 + int4 shifts, note at row 17's rank-1 int8 + int4 shifts, 1,024/2,304 = **3,328**) - the cheapest card that can pass, since `l2i4s4` was only +0.000580 imm at 768 bits and one more rank has room to close the 0.00008 gap; **`l2i8s2`** (card rank-2 int8 + **int2** shifts, 1,152/2,304 = 3,456) - the card's own shift vectors one level below the free int4 (rows 9/10); and the equal-2,112-bit note pair **`n1i8s3`** (rank-1 int8 + int3 shifts) and **`n2i4s3`** (rank-2 int4 + int3 shifts), 1,280/2,112 = 3,392 each - int3 is the half-step between the working int4 shifts and the int2 shifts that cost row 20 +0.000408 imm, and the pair asks the note-side rank-vs-precision question directly. All four are inside the bit bars (card 1,024-1,280 of 2,176, note 2,112-2,304 of 6,528) and need no engine work beyond what row 15 already proved (`card:2:int8` parses; rank 2 on the card measured 0.812x). Because it is queued behind `it36corrupt`, its `BUSY_PAT` adds `probe_get_result|check_corruption` - the exact trap `QUEUE.md` warns about - and `it36equalbits` is now in `QUEUE.md`'s LIVE list so the hourly supervisor may restart it. Also corrected a stale `AGENTS.md` claim in the same pass: **rank >= 2 is deployable on the card** (0.812x), only a rank >= 2 note would need engine work; the 0.629x figure was a both-streams measurement.

* 2026-11-21 23:05 - **Row 23 `l2i4s4` FAILS, and RESULTS.md now carries Andrew's `corruption` column.** `l2i4s4` holds the card at rank 2 with **int4 factors** (the cheap precision) + int4 shifts and the note at row 9's rank-1 int8 + int4 shifts: 768 card / 2,304 note bits - the SAME bit budget as row 9, where the card is rank-1 int8. It scored 0.334068/0.306792 = **+0.000266 ahead / +0.000580 imm** against `qnj_it36ctl` (ub95 +0.000337/+0.000661, p 6.9e-16/1.5e-38): ahead is 0.53x of the ceiling, **imm is 1.16x over it**. The informative part is the head-to-head at identical bits: rank-2 int4 beats rank-1 int8 by about **+0.00025 ahead / +0.00045 imm**, so at a fixed card budget extra RANK buys more than extra factor precision - but it does not rescue the imm head, which still needs the int8-quality 2,304-bit note. Recorded as RESULTS.md row 23 and stamped in `autorun/RECORDED.txt`. **New column**: `corruption`, requested by Andrew at 22:3x - `pending` for row 0 (the fp32 control) and the four passing rows 15-18, which the queued `it36corrupt` chain will score, and `-` for every row that fails the log-loss gate (not a deploy candidate, so it is never checked). The column legend lives in RESULTS.md, and `autorun/QUEUE.md` section 4 now tells the hourly supervisor that the main table has 12 columns and to write `pending` rather than a guess.

* 2026-11-21 22:55 - **The 10-hour three-chain deadlock is fixed, and the corruption check is queued behind the running queue.** (1) Fix: `it36rankprec`, `it36longdecay` and `it36shrink` each listed the other two chain SCRIPTS in `BUSY_PAT`, so each saw its siblings' waiting processes forever (`matching processes=4`) and none could take the GPU - idle 12:25-22:23, about 10 h. `_fix_chain_busypat2.py` rewrote all three to real GPU work only (`qat_decay.py|quant_sweep.py|rwkv.train_rwkv|eval_sharded.py|rust_speed.py|rust_grid.py|rwkv-infer`); the waiters were killed, the stale locks deleted, and the three chains relaunched 22:21:59-22:22:00. `it36rankprec` ran the `l2i4s4` smoke, its 124-step decay (6.1 min, rc=0) and started the quantized eval at 22:29:18; `it36longdecay` and `it36shrink` wait behind it correctly. (2) New chain `it36corrupt` (launched 22:50, waits for `it36shrink`): Andrew's corruption gate - score the 50 users with the longest average review histories, histories NOT truncated, and fail if even ONE card/note state value or even ONE p(recall) is non-finite. The eval's own guards cannot answer it (`_get_loss` returns None only on NaN, never on +inf, and the QAT state path caps non-finite states before quantizing, `_sanitize_state` at `rwkv_ops.py:313`), so `corruption/finite_probe.py` monkey-patches the eval path (no repo source edited) and watches the RAW state, the shift vectors and every `p_curve`/`p_imm`/`p_imm_all`, at `RWKV_QAT_FUSED=0` (the documented bit-parity Python twin of the fused kernels) - one process, no sharding, on `corruption/corrupt50.json` (50 users, 4,609,289 section rows; the 62 users over the 200,000-row VRAM cap are recorded in `excluded_too_big` and backfilled). Points in value order: `l2i8s4` (row 17), `l2i8` (15), `l3i8` (16), `l3i8s4` (18), then the quantization-OFF control (`qat_it36ctl/ctl_124.pth`). CPU smoke (user 7325, full history, `l2i8`): 828 state calls / 189,923,328 state values, maxabs 0.345, 0 non-finite, 3 p(recall) tensors clean -> PASS. Two guards came with it: `status_report.py`'s WORKLOAD pattern now includes `probe_get_result|check_corruption`, and the chain logs a heartbeat every 5 minutes while a point runs (a point can take an hour, and a silent `run()` would look like a dead chain to the hourly supervisor). Verdicts land in `corruption/corruption_corrupt_<tag>.json` and become a corruption column in RESULTS.md.

* 2026-11-21 13:05 - **Row 22 `l3i8s4n4` FAILS by 0.000025: the int4 note blocks the imm head on every card.** Card held at row 18's config (rank-3 int8 + int4 shifts, 1,792 bits), note cut to rank-1 int4 factors + int4 shifts (1,536 bits). It scored 0.334099/0.306737 = **+0.000297 ahead / +0.000525 imm** against `qnj_it36ctl` (ub95 +0.000366/+0.000594, p 1.6e-17/1.2e-51): ahead is 0.6x inside the ceiling, imm is 1.05x over it. Against row 18 the 768-bit note cut costs about +0.000278/+0.000315 (difference of deltas, not a paired test), matching the row-21 estimate, so a note below 2,304 bits is not reachable at this rank ladder. Recorded as RESULTS.md row 22 and stamped in `autorun/RECORDED.txt`. **Supervisor note: `it36rankprec`, `it36longdecay` and `it36shrink` are in a circular wait** - each chain's busy pattern lists the other two, so each logs `matching processes=4` and none can start (GPU idle since 12:25). Their waiter gives up after 900 min, at about 00:15 on 2026-11-22. No restart is authorized (their logs are fresh, and a restart would recreate the same wait), so this needs Andrew.

* 2026-11-21 12:05 - **Rows 20-21 FAIL: the note pool cannot go below int4, and the combined cut is 0.00014 from passing.** `nr4s2` (note rank-1 int4 factors + int2 shifts, 1,024/1,152 bits) scored 0.334648/0.307968 = **+0.000846/+0.001756** (ub95 +0.000955/+0.001892, p 2.7e-51/8.9e-87), i.e. +0.000124/+0.000408 WORSE than row 19 for 384 bits saved, so int2 shifts are harmful on the note (row 20). `l2i8s4n4` (card rank-2 int8 + int4 shifts, note rank-1 int4 + int4 shifts, 1,280/1,536 bits = 2,816 total) scored 0.334130/0.306850 = **+0.000328/+0.000638** - the ahead head 0.65x INSIDE the ceiling and the imm head 1.3x over, the closest any note cut has come (row 21); against row 17 the 768-bit note cut costs about +0.000276/+0.000311, so row 17 note (2,304 bits) is still the cheapest note that passes. Recorded as RESULTS.md rows 20-21 and stamped in `autorun/RECORDED.txt`. `it36notecut` is scoring its last pre-registered point `l3i8s4n4` (1,792/1,536); eval started 11:58 and was running at this check-in.

* 2026-11-21 11:05 - **Row 19 `nr4` FAILS: the note pool cannot drop to int4 while the card keeps its
  rank gain.** The note cut (note rank-1 int4 factors + int4 shifts, 1,536 bits; card held at row 4's
  exact rank-1 int8 config) scored 0.334524/0.307559 = **+0.000722 ahead / +0.001348 imm** against
  `qnj_it36ctl` (ub95 +0.000821/+0.001465), about +0.0002/+0.0003 worse than row 4 - so the cheap note
  gives back the card's immediate-head gain instead of adding to it. Recorded as RESULTS.md row 19 and
  stamped in `autorun/RECORDED.txt` (sweep `quant_sweep_it36nr4.json`). `it36notecut` is still running
  and was scoring `nr4s2` (note rank-1 int4 + int2 shifts, 1,152 bits) at 10:48; the combined
  `l2i8s4n4`/`l3i8s4n4` points remain gated on its own rule.

* 2026-11-21 10:05 - **Row 18 `l3i8s4` lands and PASSES** (`it36rank`, rc=0, 26.9 min sweep):
  card rank-3 int8 + int4 shifts, note rank-1 int8 + int4 shifts, 1,792/2,304 bits = 4,096 total,
  scored 0.333821/0.306422 = +0.000019 ahead / +0.000210 imm against `qnj_it36ctl` (ub95
  +0.000082/+0.000268), deploy engine 2,290 rev/s = 0.800x. It is the lowest-tax point of the card
  rank ladder (row 16's tax at 256 fewer card bits, because the int4 shifts are free), so the ladder
  now has two passing cost points: row 17 (1,280 card, 3,584 total, +0.000052/+0.000327) and row 18
  (1,792 card, 4,096 total, +0.000019/+0.000210). `it36rank` ended 10:05:24; `it36notecut` takes the
  lock next.
* 2026-11-21 09:50 - **An hourly supervisor is now running (Andrew's request).** Windows Task Scheduler
  task `RWKV-Quant-Autorun` (every hour at :05, `MultipleInstances=IgnoreNew`, 30-min limit, runs only
  while Andrew is logged on) calls `scratchpad/quant_plan/autorun/run_autorun.cmd`, which takes a
  single-instance lock and runs the goose recipe `quant_hourly.yaml` headless. Its contract is
  `autorun/QUEUE.md`; `status_report.py` is its deterministic state dump; `AUTORUN_LOG.md` holds one
  block per run; `RECORDED.txt` lists the sweep stamps already written into RESULTS.md. It may only
  restart a LIVE chain whose log stalled >45 min on an idle GPU, run the one-time idle deploy
  measurement (`rust_speed_idle_l2i8.json`), and record landed results. It never starts a new research
  direction, never writes to git, and never runs a second GPU workload. Verified end to end twice: the
  first run recorded row 17, the second (via `Start-ScheduledTask`) logged "nothing to do, GPU BUSY"
  and exited 0. Before editing RESULTS.md / NOTES.md / the run board, check
  `autorun_lock.py status` and re-read the file immediately before the change.

* 2026-11-21 09:38 - **Row 17 (`l2i8s4`) PASSES: the int4 shifts are free at rank 2 too, so the axis-1
  state drops to 3,584 bits.** QAT card rank-2 int8 factors + int4 shifts, note rank-1 int8 + int4
  shifts: card 1,280 bits (160 B, 1.70x under the 2,176 bar), note 2,304 bits (288 B, 2.83x under the
  6,528 bar). Loss 0.333853/0.306539 = **+0.000052 ahead (ub95 +0.000118) / +0.000327 imm (ub95
  +0.000390)** against row 0, i.e. row 15's tax for 1,024 fewer bits (22% less state). Deploy engine
  2,290 rev/s = **0.800x** (states/s 0.656x) against row 15's 0.812x; both above the 0.75x floor and both
  measured while a decay was training (~3% pessimistic). Recorded as RESULTS.md row 17, stamped in
  `autorun/RECORDED.txt` (sweep `quant_sweep_it36l2i8s4.json`). `it36rank` is now training `l3i8s4`
  (row 18), started 09:31. Still owed: the idle-machine speed re-measurement of rows 15/17.

* 2026-11-21 09:20 - **Row 16 (card rank-3 int8) is the lowest-tax row, and the rank ladder shows the
  knee.** `l3i8` (card rank-3 int8 + int8 shifts, note rank-1 int8; 2,048/3,072 bits) =
  0.333808/0.306388 = **+0.000006 ahead / +0.000176 imm** against row 0 (ub95 +0.000070/+0.000234). The
  immediate-head tax by rank: rank-1 +0.001009, rank-2 +0.000307 (row 15), rank-3 +0.000176 - the first
  extra rank removes 70%, the second another 43% of the rest. So rank-3 is the axis-2 choice (near-zero
  tax, and it leaves ~+0.0003 of the immediate-head budget to spend on a smaller NOTE pool) while rank-2
  is the axis-1 choice (512 fewer card bits). **Three new chains launched 09:15, queued behind the two
  running ones: `it36rankprec` (equal-bits rank-vs-precision probes: card rank-2 int4 at 768 bits vs row
  9's rank-1 int8, rank-4 int4 at 1,280 vs row 17's rank-2 int8, rank-7 int4 at 2,048 vs row 16's rank-3
  int8, and note rank-2 int4 at 2,304 vs row 17's note rank-1 int8), `it36longdecay` (rank-2 at 496 and
  992 steps against their own same-length twins - the absolute-loss axis, worth ~0.012-0.017 absolute),
  and `it36shrink` (four combos under row 15's 4,608 bits: 2,560 / 2,624 / 2,688 / 2,944).** Total queue
  ~13 points, roughly 8 h of GPU work.

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
