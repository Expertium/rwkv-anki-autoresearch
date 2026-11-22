"""Record the three long-history probe findings (Amendment 16 + NOTES bullets + a RESULTS paragraph).

All numbers come from runs made 2026-11-22 17:00-17:45 on the row-17 deploy weights
(`deploy_iter17_l2i8s4.safetensors`) over the THREE LONGEST-history corrupt50 traces (users 7250,
5052, 5939; 28-29k reviews each, 80-84 MB):

  rust_grid_fast_wide3.json      plain, l2i8s4, l2i8s4f, l2i8s4f1     -> axis 3
  rust_grid_vlevel3.json         VLEVEL int4/int3/int2                 -> axis 1 negative
  rust_grid_int6long.json        l2i6s4, l2i5s4                        -> axis 1 scaling
  rust_grid_long_paired.json     per-user paired deltas vs l2i8s4      -> the noise picture

Both docs mix CRLF and LF, so anchors are matched as RAW BYTES and asserted unique.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "scratchpad" / "quant_plan" / "RESULTS.md"
NOTES = ROOT / "scratchpad" / "quant_plan" / "NOTES.md"
BOARD = ROOT / "optimization" / "GPU_RUN_BOARD.md"

PARA = (
    "**The shipped row-17 weights run 16% faster with one deploy flag and no accuracy change, and "
    "the cost of cutting factor precision is history-dependent.** Two deploy-side probes on the three "
    "LONGEST corrupt50 histories (users 7250 / 5052 / 5939, 28-29k reviews each) settle both. "
    "(1) `l2i8s4`, `l2i8s4f` (the E2 fast rank-r solver, `RWKV_LOWRANK_FAST=1`) and `l2i8s4f1` (the "
    "same plus `RWKV_LOWRANK_FAST_ITERS=1`) agree to <=3e-6 on BOTH heads - tax -0.003711/-0.003620, "
    "-0.003714/-0.003623, -0.003714/-0.003621 - so the approximation does not drift as history grows, "
    "and with 0.800x -> 0.927x rev/s (median per-user 0.921x) the same weights, bits and log loss run "
    "16% faster for free. (2) The exponent rungs `l2i6s4` (1,024/1,920 = 2,944 bits) and `l2i5s4` "
    "(896/1,728 = 2,624) cost FAR more on long histories than the 2-user short-trace screen suggested: "
    "per-user paired imm deltas are +0.000067/+0.000226/+0.000742 (mean +0.000345, std 0.00035) for "
    "int6 and +0.000629/+0.001147/+0.002877 (mean +0.001551, std 0.00118) for int5, i.e. the same sign "
    "on every user and growing with history length, against +0.000003/+0.000017 on the short traces. "
    "Asymmetric factor budgets (`RWKV_LOWRANK_VLEVEL`, V coarsened while U stays int8) are NOT a lever: "
    "at the same 640-bit saving V int4 costs +0.001015 (std 0.00118) versus int6's +0.000345, and the "
    "VLEVEL spread is ~15x wider, with V int3's negative mean driven by ONE user (5052 reads -0.006228 "
    "against +0.001639/+0.003570 elsewhere) - so coarsening V does not add a smooth error, it "
    "destabilizes the trajectory."
)

BOARD_AMENDMENT = """
## Amendment 16 (2026-11-22 17:50): the fast solver is free on long histories; factor-precision cost scales with history; VLEVEL is dropped

**Axis 3 - the shipped row-17 weights get 16% faster for free.** On the three LONGEST corrupt50
histories (users 7250 / 5052 / 5939, 28-29k reviews each, 80-84 MB traces) `l2i8s4`, `l2i8s4f`
(`RWKV_LOWRANK_FAST=1`) and `l2i8s4f1` (plus `RWKV_LOWRANK_FAST_ITERS=1`) agree to **<=3e-6 on both
heads** (tax -0.003711/-0.003620, -0.003714/-0.003623, -0.003714/-0.003621). Together with the speed
measurements (0.800x -> 0.844-0.889x -> 0.927x rev/s, median per-user 0.921x) this makes ITERS=1 the
deploy setting for row 17: same weights, same bits, same log loss, 16% faster, no retraining. The flag
must stay OFF for any Python<->Rust parity run, because the fast factors are an approximation rather
than a bit-identical truncation. Evidence: `rust_grid_fast_wide3.json`, `rust_speed_fastrow17.json`,
`rust_speed_fastiters.json`.

**Axis 1 - the cost of cutting factor precision GROWS with history length, so the 2-user screen
understated it ~100x.** Per-user paired imm deltas against `l2i8s4` (`rust_grid_long_paired.json`):

| config | 7250 | 5052 | 5939 | mean | std |
| --- | --- | --- | --- | --- | --- |
| `l2i6s4` (1,024/1,920 = 2,944 bits) | +0.000067 | +0.000226 | +0.000742 | +0.000345 | 0.00035 |
| `l2i5s4` (896/1,728 = 2,624 bits) | +0.000629 | +0.001147 | +0.002877 | +0.001551 | 0.00118 |
| `l2i8s4v4` (V int4, 2,944 bits) | +0.000653 | +0.000057 | +0.002334 | +0.001015 | 0.00118 |
| `l2i8s4v3` (V int3, 2,784 bits) | +0.001639 | -0.006228 | +0.003570 | -0.000340 | 0.00519 |
| `l2i8s4v2` (V int2, 2,624 bits) | +0.003583 | -0.006194 | +0.005652 | +0.001013 | 0.00633 |

Both exponent rungs are the SAME SIGN on every user and grow with length (the longest user, 5939, pays
the most). **Expectation now on record before chain 17 runs:** `l2i6s4` is a coin flip - row 17's own
imm is +0.000327 with ub95 +0.000390 against a +0.0005 ceiling, so an added ~+0.0001-0.0003 leaves
little room - and `l2i5s4` plus the bottom rung `l2i4s4n4` are EXPECTED TO FAIL on imm. They still run,
because the gate is the 750-user roster and a roster average sits below the long-user figure but above
zero, and the measured answer is the deliverable. If int6 lands, it is a 640-bit (18%) cut.

**Axis 1 negative - `RWKV_LOWRANK_VLEVEL` is dropped.** At the SAME 640-bit saving, coarsening V while U
stays int8 costs +0.001015 (std 0.00118) against int6's +0.000345 (std 0.00035): three times larger and
three times noisier, so the bits are better spent symmetrically. Worse, the VLEVEL spread is ~15x wider
than the symmetric rungs' and V int3's negative mean is driven by ONE user (5052 reads -0.006228 while
the other two read +0.001639/+0.003570), which means coarsening V does not add a smooth error - it
destabilizes the state trajectory. No trainer support and no chain are added for it; the engine knob
stays available for future work.

**Method note.** The per-user paired difference is what makes these readable: the grid JSONs store only
per-config means, and on these long traces every quantized config reads BETTER than `plain` by ~0.0036
(the trace-path offset already documented), so absolute taxes cannot rank configs. Differencing the same
users pairwise cancels that offset and leaves the std above, which is what separates
`int6` (sign-consistent, std 0.00035) from VLEVEL (sign-flipping, std 0.005-0.006).
"""

BULLET = (
    "* 2026-11-22 17:50 - **Three long-history probe findings, all on the row-17 deploy weights over "
    "the 3 longest corrupt50 traces (users 7250/5052/5939, 28-29k reviews each).** (1) AXIS 3: "
    "`l2i8s4`, `l2i8s4f` and `l2i8s4f1` agree to <=3e-6 on both heads, so the fast rank-r solver with "
    "`RWKV_LOWRANK_FAST_ITERS=1` does not drift with history - with 0.800x -> 0.927x rev/s (median "
    "per-user 0.921x) the same weights, bits and log loss run 16% faster for free. (2) AXIS 1: the cost "
    "of cutting factor precision GROWS with history length. Per-user paired imm deltas vs `l2i8s4` are "
    "`l2i6s4` +0.000067/+0.000226/+0.000742 (mean +0.000345, std 0.00035) and `l2i5s4` "
    "+0.000629/+0.001147/+0.002877 (mean +0.001551, std 0.00118) - same sign on every user - against "
    "+0.000003/+0.000017 on the 2-user short-trace screen, so that screen understated int6 ~100x. "
    "**Chain 17 expectation on record: `l2i6s4` is a coin flip on imm (row 17's ub95 is already "
    "+0.000390 against the +0.0005 ceiling) and `l2i5s4` / `l2i4s4n4` are expected to FAIL.** "
    "(3) AXIS 1 NEGATIVE: `RWKV_LOWRANK_VLEVEL` is dropped - at the same 640-bit saving V int4 costs "
    "+0.001015 (std 0.00118) vs int6's +0.000345, and the VLEVEL spread is ~15x wider with V int3's "
    "negative mean driven by one user (5052 -0.006228 vs +0.001639/+0.003570), i.e. coarsening V "
    "destabilizes the trajectory rather than adding a smooth error. Method: per-user PAIRED deltas are "
    "what make these readable, because on long traces every quantized config reads ~0.0036 better than "
    "`plain` (the known trace-path offset), so absolute taxes cannot rank configs.\n"
    "\n"
    "* 2026-11-22 17:45 - **`it36equalbits` is the active chain** (it took the GPU at 16:46:55 and has "
    "since run `l3i4s4` and started `l2i8s2` at 17:18:25). `it36shrink` is still waiting for a free GPU "
    "handoff, which is by design: both it and `it36equalbits` wait only on an idle GPU (`it36shrink` on "
    "the ended `it36noterank2`, `it36equalbits` on the ended `it36corrupt`) - the race resolves to "
    "whoever polls first, and `it36combo`/`it36corrupt2`/`it36e1`/`it36e1shrink`/`it36int6` queue behind "
    "them in order. **Operational rule learned the hard way:** a CPU-only job that matches a chain's "
    "BUSY_PAT (including `rust_grid.py`, `rwkv-infer.exe`, and even a `DEVICE=cpu` `rwkv.train_rwkv`) "
    "strands the queue across a handoff, because `wait_for_prev` returns only on `ended AND busy == 0`. "
    "Two of my own jobs did exactly that at 16:42 and idled the GPU; both were killed. Such jobs must "
    "fit inside a point's own window (smoke 1-3 min + decay 7-10 min + eval 20-56 min), which is how "
    "the three probes above were run.\n"
)


def sub(raw: bytes, old: bytes, new: bytes, label: str) -> bytes:
    n = raw.count(old)
    if n != 1:
        raise SystemExit(f"ABORT [{label}]: anchor found {n} times, expected 1")
    print(f"  {label}: ok")
    return raw.replace(old, new, 1)


def main() -> int:
    res = RESULTS.read_bytes()
    anchor = b"**Row 32 (`l2i8n3i8`, chain 14 point 4)"
    res = sub(res, anchor, PARA.encode("utf-8") + b"\n\n" + anchor, "RESULTS paragraph")
    RESULTS.write_bytes(res)

    notes = NOTES.read_bytes()
    notes = sub(notes, b"## Live status\n\n", b"## Live status\n\n" + BULLET.encode("utf-8"),
                "NOTES bullets")
    NOTES.write_bytes(notes)

    board = BOARD.read_bytes()
    if b"## Amendment 16" in board:
        print("  BOARD: Amendment 16 already present")
    else:
        if not board.endswith(b"\n"):
            board += b"\n"
        BOARD.write_bytes(board + BOARD_AMENDMENT.encode("utf-8"))
        print("  BOARD: Amendment 16 appended")
    return 0


if __name__ == "__main__":
    sys.exit(main())
