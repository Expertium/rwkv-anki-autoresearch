"""Chain 11: the equal-bits follow-ups - rank vs precision on the two streams the ladder has not cut.

Rows 9 and 23 answered "rank or precision?" on the CARD at a fixed bit count (`l2i4s4`, rank-2 int4,
beat row 9's rank-1 int8 at the same 768 bits by about +0.00025 ahead / +0.00045 imm). These four
points carry the same question to the places the earlier chains left open, all scored against row 0
(the identical unquantized 124-step decay) with the note's 2,304 bits or a half-step below them:

  l3i4s4   card rank-3 int4 + int4 shifts, note rank-1 int8 + int4   1,024 / 2,304 = 3,328 total
           the cheapest CARD that can pass: 256 bits below row 17's 1,280, and `l2i4s4` was only
           +0.000580 imm at 768 card bits, so the imm head has one rank of room to close the gap.
  l2i8s2   card rank-2 int8 + int2 shifts, note rank-1 int8 + int4   1,152 / 2,304 = 3,456 total
           the card's own SHIFT vectors at int2 (the note's 2,304-bit level is held): rows 9/10 showed
           the card shifts can drop to int4 for free, this asks whether they can go one level lower.
  n1i8s3   card rank-2 int8 + int4 shifts, note rank-1 int8 + int3   1,280 / 2,112 = 3,392 total
           the note's SHIFT vectors at int3 - a half-step between the working int4 and the int2 of row
           20, which cost +0.000408 imm on the note (rows 19-20).
  n2i4s3   card rank-2 int8 + int4 shifts, note rank-2 int4 + int3   1,280 / 2,112 = 3,392 total
           the SAME 2,112 note bits as n1i8s3 with rank-2 int4 instead of rank-1 int8 factors, so the
           pair is the note-side version of the rows 9/23 comparison.

Nothing here needs engine work that the deploy engine does not already do: the card stays at rank <= 3
(the engine parses `card:2:int8`, and rank 2 on the CARD alone measured 0.812x), and the note stays
rank-1 int8 / rank-2 int4. This chain waits for `it36corrupt`, so its BUSY_PAT must ALSO match the
corruption driver (`probe_get_result.py` / `check_corruption.py`), or it would start on top of it.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import it36ladder_lib as lib  # noqa: E402

BUSY_PAT = ("qat_decay.py|quant_sweep.py|rwkv.train_rwkv|eval_sharded.py|rust_speed.py|rust_grid.py|"
            "rwkv-infer|probe_get_result|check_corruption")

POINTS = [
    ("l3i4s4", "it36l3i4s4", 124, "qnj_it36ctl"),
    ("l2i8s2", "it36l2i8s2", 124, "qnj_it36ctl"),
    ("n1i8s3", "it36n1i8s3", 124, "qnj_it36ctl"),
    ("n2i4s3", "it36n2i4s3", 124, "qnj_it36ctl"),
]

if __name__ == "__main__":
    sys.exit(lib.chain_main("it36equalbits", "it36corrupt_chain.log",
                            "=== it36corrupt chain end ===", POINTS, BUSY_PAT))
