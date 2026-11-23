"""Record the corruption PASSes of rows 48 and 49 (NOTES bullet + BOARD Amendment 36).

The two cells themselves were written by `corruption/record_corruption.py` (the only writer of those
cells); this script records the NARRATIVE: a new live-status bullet that supersedes the two bullets whose
tail still said the cell "stays pending", and a board amendment with the observed counts.

Line endings: NOTES.md is CRLF and GPU_RUN_BOARD.md is LF in this checkout; both are detected, the edit is
done in LF and written back, with a byte round-trip check.
"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]           # scratchpad/_record... -> repo root
RUN = ROOT / "scratchpad" / "quant_plan"
NOTES = RUN / "NOTES.md"
BOARD = ROOT / "optimization" / "GPU_RUN_BOARD.md"

NOTES_ANCHOR = "* 2026-11-23 03:10 - **Row 49 (`l2i6n`, 1,280/1,920 = 3,200 bits)"
NOTES_BULLET = (
    "* 2026-11-23 03:20 - **The corruption column closes for rows 48 and 49: both PASS on the 50 "
    "longest histories (this supersedes the `pending` clause at the tail of the two bullets below).** "
    "`corruption/_watch_and_check_chain18.py` - the queue-hold watcher built for a chain that has NO "
    "corruption phase of its own - ran `check_corruption.py` for `l2i6c` at 03:08:32 and for `l2i6n` at "
    "03:13:14, each rc=0 in 4.6 min: **50 of 50 users scored, 0 skips, 158,986 raw card/note state "
    "observations per config (70.5 billion state ELEMENT values, counted on the RAW and the QUANTIZED "
    "side), 4.41 billion shift-vector values and 150 p(recall) tensors, with ZERO non-finite values of "
    "any kind** (largest |state| 0.598 for `l2i6c` and 0.693 for `l2i6n`, largest |shift vector| 5.531, "
    "largest p 0.991, largest loss scalar 11.93). Histories were NOT truncated and no user was skipped, "
    "so 3,328 and 3,200 bits are fully deployable sizes: the two rows that split the int6 tax are clean "
    "on Andrew's blowup gate as well as on the loss. Filling the cells needed two new `ROWS` entries in "
    "`corruption/record_corruption.py`, because rows 41 and 45 pre-register `l2i6c` and `l2i6n` by name "
    "in their own cells, so the \"exactly one row-line mentions the tag\" fallback resolved neither; a "
    "tag listed in `ROWS` now resolves by RESULTS row number instead. Chain 19 (`it36fixedbasis`) took "
    "the GPU at 03:19:05, one minute after the watcher released its hold.\n"
)

BOARD_AMEND = """
## Amendment 36 (2026-11-23 03:20): the chain-18 finiteness checks close their cells - rows 48 and 49 PASS

`it36int6card` has no corruption phase of its own, so `corruption/_watch_and_check_chain18.py` held the
queue from 02:08:47 with a CPU-only `gpu_hold.py --tag check_corruption` and ran both checks back to back
in the first free window after the chain's end marker (03:07:47):

    config  row  check start  driver      state observations  p(recall) tensors  non-finite  users   max |state|
    l2i6c    48  03:08:32     rc=0 4.6 min            158,986                150           0  50/50   0.598
    l2i6n    49  03:13:14     rc=0 4.6 min            158,986                150           0  50/50   0.693

Each run also observes 4.41 billion shift-vector values and ~70.5 billion state element values on the RAW
and the QUANTIZED side of every compression, with zero non-finite values of any kind (largest |shift
vector| 5.531, largest p 0.991, largest loss scalar 11.93). Histories were NOT truncated and no user was
skipped. So both mixed-int6 rows are deployable on Andrew's blowup gate and the whole 3,200-/3,328-bit
band is clean. `record_corruption.py` needed two new `ROWS` entries for these tags: rows 41 and 45 name
both tags in their own cells, so the "exactly one row-line mentions it" fallback could not resolve them.
Chain 19 (`it36fixedbasis`) took the GPU at 03:19:05, about a minute after the watcher released the hold.
"""


def edit(path: Path, fn):
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    crlf = text.count("\r\n") > text.count("\n") - text.count("\r\n")
    lf = text.replace("\r\n", "\n")
    new = fn(lf)
    if new is None:
        return False
    out = new.replace("\n", "\r\n") if crlf else new
    path.write_bytes(out.encode("utf-8"))
    if path.read_bytes().decode("utf-8") != out:
        print(f"FAIL: {path.name} round trip differs")
        return False
    print(f"{path.name}: written ({len(raw)} -> {len(out.encode('utf-8'))} bytes, "
          f"{'CRLF' if crlf else 'LF'})")
    return True


def do_notes(lf: str):
    if "The corruption column closes for rows 48 and 49" in lf:
        print("FAIL: NOTES bullet already present")
        return None
    if lf.count(NOTES_ANCHOR) != 1:
        print(f"FAIL: NOTES anchor x{lf.count(NOTES_ANCHOR)}")
        return None
    return lf.replace(NOTES_ANCHOR, NOTES_BULLET + NOTES_ANCHOR, 1)


def do_board(lf: str):
    if "Amendment 36" in lf:
        print("FAIL: Amendment 36 already present")
        return None
    if "Amendment 35" not in lf:
        print("FAIL: Amendment 35 missing")
        return None
    return lf.rstrip("\n") + "\n" + BOARD_AMEND


def main() -> int:
    ok = edit(NOTES, do_notes)
    ok = edit(BOARD, do_board) and ok
    print("corruption PASS recorded" if ok else "SOMETHING FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
