"""One-shot (2026-11-21 23:45): the NOTES.md Live-status bullet + the GPU_RUN_BOARD amendment for the
night queue (chains 12/13, the corruption column filler). Refuses to run twice."""
import pathlib

HERE = pathlib.Path(__file__).resolve().parents[1]      # .../scratchpad/quant_plan
NOTES = HERE / "NOTES.md"
BOARD_PATH = HERE.parents[1] / "optimization" / "GPU_RUN_BOARD.md"

BULLET = """* 2026-11-21 23:45 - **The night queue is closed and the corruption column now fills itself.** Two chains were added behind the running ones, so the GPU has work through the night and the morning. (1) **`it36combo` (chain 13, 3 points)** crosses the two cut axes that until now were always measured with the other stream held at row 17's level: `l3i4s4n8s3` (1,024/2,112 = 3,136), `l3i4s4n3` (1,024/1,344 = 2,368) and `l2i4s4n8s3` (768/2,112 = 2,880) - the cheap card UNDER the cheap note, which is the only way the total beats 3,584; a failure says the two cuts do not compose, which is the answer the frontier needs. (2) **`it36corrupt2` (chain 12, 8 points)** exists so that every row chains 11/13 create gets its corruption verdict - and only if it is a deploy candidate: each point reads its own sweep verdict first and SKIPS when `ub95 > +0.0005` on either head (row 23 already fails, so it is skipped) or when no verdict exists yet (a missing verdict is never a pass). It runs after chain 13, so the sweeps it guards already exist. (3) **`corruption/record_corruption.py` is now the only writer of the corruption CELLS**: it reads `corruption_corrupt_<tag>.json`, finds the tag's row (built-in map for rows 0/15-18/23, otherwise the unique row naming the tag), and replaces the last cell - refusing any row that is not 12 cells wide. `QUEUE.md` section 4 gained step 4 telling the hourly supervisor to run it before it pushes RESULTS.md. (4) `check_corruption.py`'s CONFIGS gained the 8 later tags. Verified before launch: `record_corruption.py` filled a synthetic verdict into row 17 and the cell was restored byte-for-byte (RESULTS.md has no diff), and the gate guard returns False for `it36l2i4s4`, True for rows 17/18 and None for the not-yet-run chain 11 stamps. Queue order now: `it36rankprec` (running) -> `it36longdecay` -> `it36shrink` -> `it36corrupt` (10) -> `it36equalbits` (11) -> `it36combo` (13) -> `it36corrupt2` (12): 8 heavy decay points plus at most 8 cheap corruption points.
"""

AMENDMENT = """Row 11 amendment 15 (2026-11-21 23:45, Moscow). **The night queue is closed; the corruption column fills itself.**

- **Row 23 recorded (`l2i4s4`, FAIL).** Card rank-2 **int4** + int4 shifts with row 9's note: +0.000266 ahead / +0.000580 imm, at the SAME 768/2,304 bits as row 9 (rank-1 int8). At equal bits rank-2 int4 beats rank-1 int8 by about +0.00025/+0.00045, so at a fixed budget RANK beats factor precision on the card; the imm head is still 1.16x over the ceiling.
- **New column `corruption`** (Andrew's gate) in `scratchpad/quant_plan/RESULTS.md`: `pending` for row 0 and rows 15-18 (queued in `it36corrupt`), `-` where the row fails the log-loss gate and so is not a deploy candidate.
- **Chain 11 `it36equalbits`** (launched 23:01, waits for `it36corrupt`): `l3i4s4` 1,024/2,304, `l2i8s2` 1,152/2,304, `n1i8s3` and `n2i4s3` 1,280/2,112. Its BUSY_PAT carries `probe_get_result|check_corruption`, the trap section 2 of the supervisor's `QUEUE.md` warns about.
- **Chain 13 `it36combo`** (launched 23:08, waits for `it36equalbits`): the cheap card under the cheap note - `l3i4s4n8s3` 3,136, `l3i4s4n3` 2,368, `l2i4s4n8s3` 2,880 total bits. All inside the bars (card 768-1,024 of 2,176; note 1,344-2,112 of 6,528). Same note as chain 11: a card rank >= 2 needs no engine work on the card stream.
- **Chain 12 `it36corrupt2`** (launched 23:08, waits for `it36combo`): the corruption verdict for every later row, gate-guarded (skips a row whose own sweep fails the log-loss gate, and skips when the verdict is missing - never a pass). `corruption/check_corruption.py` CONFIGS gained the eight later tags.
- **`corruption/record_corruption.py`** is the only writer of the corruption cells, and `QUEUE.md` section 4 step 4 orders the hourly supervisor to run it before it pushes RESULTS.md. Verified in a dry run plus one real write (then reverted: RESULTS.md is byte-identical to the pushed state).
- Order: `it36rankprec` (running, `l4i4s4` eval since 23:02) -> `it36longdecay` -> `it36shrink` -> `it36corrupt` -> `it36equalbits` -> `it36combo` -> `it36corrupt2`. Host RAM 43 GB free, RAM guard heartbeating, one process per chain, no stale locks.
"""


def insert_bullet():
    with NOTES.open(encoding="utf-8", newline="") as fh:
        text = fh.read()
    if "The night queue is closed" in text:
        print("bullet already present")
        return
    for nl in ("\n", "\r\n"):
        anchor = f"## Live status{nl}{nl}"
        if anchor in text:
            text = text.replace(anchor, anchor + BULLET.replace("\n", nl), 1)
            break
    else:
        raise SystemExit("Live status anchor not found")
    with NOTES.open("w", encoding="utf-8", newline="") as fh:
        fh.write(text)
    print("bullet inserted into", NOTES)


def append_board():
    with BOARD_PATH.open(encoding="utf-8", newline="") as fh:
        text = fh.read()
    if "amendment 15 (2026-11-21 23:45" in text:
        print("board amendment already present")
        return
    if not text.endswith("\n"):
        text += "\n"
    text += "\n" + AMENDMENT
    with BOARD_PATH.open("w", encoding="utf-8", newline="") as fh:
        fh.write(text)
    print("amendment appended to", BOARD_PATH)


if __name__ == "__main__":
    insert_bullet()
    append_board()
