"""Append GPU_RUN_BOARD.md amendment 17 (2026-11-22 11:45): the corruption chain finished with five
PASS verdicts. Appends with LF endings, like the last line of the file."""
from pathlib import Path

BOARD = Path(__file__).resolve().parents[3] / "optimization" / "GPU_RUN_BOARD.md"

TEXT = """
Row 11 amendment 17 (2026-11-22 11:45, Moscow). **The corruption check is DONE: all five points PASS.**

`it36corrupt` logged `=== it36corrupt chain end ===` at 11:41:38.

| point | RESULTS row | state calls | state non-finite | p(recall) tensors | p non-finite | users | min | mean LogLoss ahead / imm |
|---|---|---|---|---|---|---|---|---|
| `l2i8` | 15 | 158,986 | 0 | 150 | 0 | 50/50 | 4.7 | 0.382082 / 0.366948 |
| `l3i8` | 16 | 158,986 | 0 | 150 | 0 | 50/50 | 4.8 | 0.382168 / 0.366710 |
| `l2i8s4` | 17 | 158,986 | 0 | 150 | 0 | 50/50 | 4.8 | 0.382094 / 0.366979 |
| `l3i8s4` | 18 | 158,986 | 0 | 150 | 0 | 50/50 | 4.7 | 0.382198 / 0.366806 |
| `ctl_fp32` | 0 | 0 (no quantizer) | - | 150 | 0 | 50/50 | 1.8 | 0.382646 / 0.366875 |

- Every verdict is `corruption/corruption_corrupt_<tag>.json`; the RESULTS.md cells were written by
  `corruption/record_corruption.py`. No cell is `pending` any more.
- The state count is the same for all four quantized rows because the roster, the layer count and the
  sequence plan are the same: rank and bit width change the representation, not the number of state
  values, so the four rows are directly comparable.
- On this roster quantization is not worse than fp32 on the ahead head (0.38208-0.38220 vs 0.38265) and
  about equal on the immediate head. A 50-user roster is not the 750-user gate roster, so this is the
  finiteness gate plus a sanity band, not an acceptance result.
- The chain ran with `RWKV_QAT_FUSED=0` (the Python twin of the fused CUDA kernels) because the raw state
  is only visible there, and with `probe_get_result.py` monkey-patching the eval path, so no repo source
  was edited for the measurement.
- **Next: the GPU is idle and no chain is queued** (every other chain is PAUSED by Andrew's 11:21
  instruction), so the only authorized work left is the owed idle deploy measurement of `QUEUE.md`
  section 3 (`rust_speed.py --tag idle_l2i8`), which fixes the ~3% pessimistic 0.812x by measuring the
  rank-2 card path on an idle machine.
"""

with BOARD.open("a", encoding="utf-8", newline="") as fh:
    fh.write(TEXT.lstrip("\n"))
print("appended", len(TEXT), "chars")
