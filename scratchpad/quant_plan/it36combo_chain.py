"""Chain 13: the two axes crossed - the cheap CARD under the cheap NOTE (2026-11-21 23:35).

Every row so far holds one stream at row 17's level and moves the other. Chain 11 cuts the card
(rank-3 int4 = 1,024 bits, rank-2 int4 = 768) and the shrink chain cuts the note (rank-1 int3 factors
= 1,344, int3 shifts = 2,112), but never together, so the totals stay at 3,584. These three points
put the cheap card UNDER the cheap note, which is the only way the total reaches 2,368-3,136:

  l3i4s4n8s3  card rank-3 int4 + int4 shifts, note rank-1 int8 + int3 shifts    1,024 / 2,112 = 3,136
  l3i4s4n3    card rank-3 int4 + int4 shifts, note rank-1 int3 factors + int4    1,024 / 1,344 = 2,368
  l2i4s4n8s3  card rank-2 int4 + int4 shifts, note rank-1 int8 + int3 shifts      768 / 2,112 = 2,880

They are informative whatever chain 11 returns: a failure says the two cuts do not compose, which the
frontier needs to know, and each is scored against row 0 (the identical unquantized 124-step decay)
like every other row. All three stay inside the bit bars (card 768-1,024 of 2,176; note 1,344-2,112 of
6,528) and need no engine work beyond what row 15 already proved.

The chain waits for `it36equalbits`; BUSY_PAT also carries the corruption driver, because a chain that
runs after a corruption chain must see that work as GPU work (QUEUE.md section 2's warning).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import it36ladder_lib as lib  # noqa: E402

BUSY_PAT = ("qat_decay.py|quant_sweep.py|rwkv.train_rwkv|eval_sharded.py|rust_speed.py|rust_grid.py|"
            "rwkv-infer|probe_get_result|check_corruption")

POINTS = [
    ("l3i4s4n8s3", "it36l3i4s4n8s3", 124, "qnj_it36ctl"),
    ("l3i4s4n3", "it36l3i4s4n3", 124, "qnj_it36ctl"),
    ("l2i4s4n8s3", "it36l2i4s4n8s3", 124, "qnj_it36ctl"),
]

if __name__ == "__main__":
    sys.exit(lib.chain_main("it36combo", "it36equalbits_chain.log",
                            "=== it36equalbits chain end ===", POINTS, BUSY_PAT))
