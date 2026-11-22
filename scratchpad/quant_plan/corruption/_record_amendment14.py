"""Record Amendment 14 in `optimization/GPU_RUN_BOARD.md` and one bullet in `NOTES.md`.

Both files mix CRLF and LF (the BOARD tail is LF, the NOTES `## Live status` heading is LF), so both
writes are RAW BYTES: the board gets an append, NOTES gets an insertion right after its
`## Live status\\n\\n` anchor. Uniqueness is asserted so a re-run cannot duplicate the text.

Amendment 14 covers, all measured 2026-11-22 15:0x-15:3x:
  * RESULTS row 31 (`l2i8n2i8s8`) is LOSS-ONLY - the deploy engine has ONE global shift level;
  * the 14:57 wedge and the self-heal that caught it, plus the four orphaned eval children;
  * chain 15 `it36e1` (E1) launched and waiting behind `it36corrupt2`;
  * the row-17 bundle zipped and re-verified from a fresh extract alone;
  * the Rust finiteness check being pointed at the real 50-user roster.

Usage: .venv/Scripts/python.exe scratchpad/quant_plan/corruption/_record_amendment14.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BOARD = ROOT / "optimization" / "GPU_RUN_BOARD.md"
NOTES = ROOT / "scratchpad" / "quant_plan" / "NOTES.md"

AMENDMENT = """
## Amendment 14 (2026-11-22 15:35): row 31 is loss-only, E1 is queued and waiting, the iter-17 bundle is zipped and re-verified

**RESULTS row 31 (`l2i8n2i8s8`, card rank-2 int8 + int4 shifts, note rank-2 int8 + int8 shifts,
1,280/4,608 = 5,888 bits) is LOSS-ONLY.** ahead 0.333683 (delta -0.0001185, ub95 -8.7842e-05,
p 0.99999999995) and imm 0.306295 (delta +0.0000836, ub95 +0.0001077, p 7.1967e-11) both PASS the
log-loss gate, but the DEPLOY ENGINE cannot express the recipe: `RWKV_STATE_SHIFT_LEVEL` takes ONE bare
level, and a per-stream string such as `card:int4,note:int8` is silently dropped by `parse_level`
(`rust/rwkv-infer/src/model.rs` matches only `intN`), after which the shifts fall back to the WKV FACTOR
level. Verified by three engine runs on user 107: `card:int4,note:int8` gave predictions IDENTICAL to
`int8` (max abs diff 0.0) and 0.087 away from `int4`. 768 extra note bits over row 29 buy 0.000019 of
imm, so int8 note shifts are not free either.

**The 20-minute wedge monitor worked.** The `l2i8n3i8` 0.04-epoch smoke wedged at 14:57:17: all four
workload processes sat at 0.00 s CPU for 20 min, GPU 1,701 MiB at 2%, and the System log had no disk
events in the preceding 2 h. At 15:17:18 `it36ladder_lib` killed the tree (`rc=-9`) and the retry passed
in 1.6 min, so the decay started 15:18:52. Second wedge of the day (the first is the 02:33 F: I/O error)
and the first one the monitor caught by itself. Four processes from the 14:51 `l2i8n2i8s8` eval are STILL
alive at 0 CPU and 4-55 MB (pids 22560, 33820, 34852, 33020): the shard parents returned rc=0 while their
children lingered. They did not block the retry, so they are left alone as debris.

**Chain 15 `it36e1` (E1) is launched and WAITING behind `it36corrupt2`** (lock taken 15:22:07;
`WAIT_MAX_MIN` raised to 2,400 min inside its own script, because the queue ahead is 11 sweep points plus
a 16-point corruption chain and the shared 900-min default could expire). E1 is the advisor's
first-ranked lever: `lambda * mean over streams of the truncated tail energy` added to the TRAINING loss -
zero state bits, zero deploy cost. Measured tails of row 17's own states
(`corruption/tail_energy_l2i8s4.json`, 5,388 raw states, users 7213/7044/6960): the card at rank 2 keeps
0.99381 of the state energy (tail 0.62%), the note at rank 1 keeps 0.96943 (tail 3.06%). Three points, all
scored against `qnj_it36ctl` exactly like row 17: `l2i8s4pf` (the fused-off control, which separates
"fused off" from "penalty on"), `l2i8s4e1a` (lambda 1.0 = ~2.4% of the ~1.02 loss) and `l2i8s4e1b`
(lambda 10.0 = ~25%). E1 must run on the Python per-step QAT loop (`RWKV_QAT_FUSED=0`, the bit-parity
twin of the fused kernels, because the fused rank-1 kernel truncates inside the kernel), which is slower
per step, so the three points cost about 4 h. Decision rule recorded in the chain docstring: if neither
lambda moves the tax, E1 is closed; if `e1b` moves it without wrecking the loss, lambda is refined.

**The iter-17 bundle is zipped and re-verified from the zip alone.** `scratchpad/row17_l2i8s4_bundle.zip`
(12,610,091 bytes, 180 entries) and the folder hold 177 files / 44.04 MB; `MANIFEST.md` verifies 177/177
SHA-256 against the folder with no extra files. `verify.cmd` was extracted into the repo root and run: the
`cargo build --release` finished in 1m18s, the deploy tax printed +0.000335 / +0.000316 (identical to every
earlier measurement), the strict finiteness check passed on 43,474,944 state values, 4a/4b printed
`PARITY: PASS` to at most 8.5e-07 per review on 3,465/3,865/12,915/23,455-review histories, 4c printed the
expected FAIL (the trace's own Python reference is UNQUANTIZED), and the speed step measured 0.804x rev/s
(median per-user 0.801x). Two defects in `verify.cmd` were found by that run and fixed: it named user 156,
whose trace the bundle does not ship (the current label DB puts review 1142 of that user in the equalize
set and the exporter aborts on it), and step 4c - which fails by design - was wired as fatal.

**The Rust finiteness check is being pointed at the real 50-user roster.** `corruption/check_corruption.py`
already covers this in Python (row 17: 50/50 users, 158,986 state calls, 0 non-finite, 285.8 s), but the
DEPLOY engine's own check had only ever run on the 3-user parity trace. `corruption/export_corrupt50_traces.py`
is exporting the 50 longest-history users (5 shards x 10 users; 10 of 50 done at 15:30) so that
`rust_finite.py --strict` can run on the same users the Python gate used, at the same untruncated history
lengths.
"""

BULLET = """* 2026-11-22 15:35 - **Row 31 is loss-only, E1 (chain 15) waits behind `it36corrupt2`, and the iter-17 bundle is zipped and re-verified from a fresh extract.** `l2i8n2i8s8` cannot be expressed by the deploy engine: `RWKV_STATE_SHIFT_LEVEL` takes ONE bare level and `parse_level` silently drops `card:int4,note:int8`, so that string behaves exactly like `int8` (user 107: max abs diff 0.0 vs `int8`, 0.087 vs `int4`). The `l2i8n3i8` smoke wedged at 14:57 (0.00 s CPU for 20 min, GPU idle, no disk events) and the lib's 20-min monitor killed and retried it - passed in 1.6 min; the four 14:51 `l2i8n2i8s8` eval processes are still alive at 0 CPU, children that outlived an `rc=0` parent. E1's tail measurement landed: the card at rank 2 keeps 0.99381 of the state energy, the note at rank 1 keeps 0.96943 (tails 0.62% / 3.06%). `scratchpad/row17_l2i8s4_bundle.zip` (12.6 MB, 180 entries) was rebuilt after running `verify.cmd` from a fresh extract exposed two defects (it named a trace the bundle does not ship, and its by-design-failing step 4c was fatal); the corrected run reproduces every documented number. The Rust finiteness check is being extended to the real 50-user roster (10/50 traces exported).
"""


def main() -> int:
    board = BOARD.read_bytes()
    if b"## Amendment 14 (" in board:
        print("ABORT: Amendment 14 is already in the board")
        return 1
    BOARD.write_bytes(board + AMENDMENT.encode("utf-8"))
    print(f"appended Amendment 14 to {BOARD} (+{len(AMENDMENT)} chars)")

    notes = NOTES.read_bytes()
    anchor = b"## Live status\n\n"
    if notes.count(anchor) != 1:
        print(f"ABORT: NOTES.md anchor found {notes.count(anchor)} times, expected 1")
        return 1
    if b"Row 31 is loss-only, E1 (chain 15) waits" in notes:
        print("ABORT: the NOTES bullet is already present")
        return 1
    NOTES.write_bytes(notes.replace(anchor, anchor + BULLET.encode("utf-8"), 1))
    print(f"inserted the Live-status bullet in {NOTES} (+{len(BULLET)} chars)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
