"""Register three SHIFT-width rungs below row 45 (2,944 bits) in rust_grid.CONFIGS, for a CPU screen.

WHY. Row 45's card is rank-2 int6 + int4 shifts (1,024) and its note is rank-1 int6 + int4 shifts
(1,920). The one knob measured to be nearly free is the SHIFT width: row 35's note int4 -> int3 cut
cost +0.000011/+0.000037 for 192 bits, while every FACTOR cut below int6 costs 6x more per bit
(`it36wide6`). So the cheapest untested rungs are shift cuts:

  l2i6s3n1i6s4   960 + 1,920 = 2,880   card int4 -> int3 shifts only (64 bits off row 45)
  l2i6s3n1i6s3   960 + 1,728 = 2,688   both streams' shifts int4 -> int3 (256 bits off row 45)
  l2i6s2n1i6s3   896 + 1,728 = 2,624   card int4 -> int2 as well (row 34's int2 case, priced for scale)

`RWKV_STATE_SHIFT_LEVEL` is a per-stream map in the deploy engine (`CompressCfg::shift_level`,
GPU_RUN_BOARD Amendment 23), and the all-int3 layouts are expressible as one bare level too.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "scratchpad" / "quant_plan" / "rust_grid.py"

ANCHOR = '''    "l2i6s4n1i4s3": {"RWKV_STATE_LOWRANK_SCOPE": "card:2:int6,note:1:int4",
                     "RWKV_QUANT_SHIFTS": "1",
                     "RWKV_STATE_SHIFT_LEVEL": "card:int4,note:int3"},
'''

BLOCK = '''    # THE SHIFT-WIDTH RUNGS BELOW ROW 45 (2026-11-23). Row 45 is card rank-2 int6 + int4 shifts
    # (1,024) and note rank-1 int6 + int4 shifts (1,920). Every FACTOR cut below int6 costs 6x more
    # per bit, but the SHIFT cut is nearly free where it has been measured (row 35: the note's int4 ->
    # int3 cost +0.000011/+0.000037 for 192 bits). These price the card's shift ladder and the
    # both-streams cut.
    "l2i6s3n1i6s4": {"RWKV_STATE_LOWRANK_SCOPE": "card:2:int6,note:1:int6",
                     "RWKV_QUANT_SHIFTS": "1",
                     "RWKV_STATE_SHIFT_LEVEL": "card:int3,note:int4"},
    "l2i6s3n1i6s3": {"RWKV_STATE_LOWRANK_SCOPE": "card:2:int6,note:1:int6",
                     "RWKV_QUANT_SHIFTS": "1",
                     "RWKV_STATE_SHIFT_LEVEL": "card:int3,note:int3"},
    "l2i6s2n1i6s3": {"RWKV_STATE_LOWRANK_SCOPE": "card:2:int6,note:1:int6",
                     "RWKV_QUANT_SHIFTS": "1",
                     "RWKV_STATE_SHIFT_LEVEL": "card:int2,note:int3"},
'''


def main() -> int:
    raw = P.read_bytes().decode("utf-8")
    lf = raw.replace("\r\n", "\n")
    if lf.count(ANCHOR) != 1:
        print(f"FAIL: anchor found {lf.count(ANCHOR)} times")
        return 1
    out_lf = lf.replace(ANCHOR, ANCHOR + BLOCK)
    for name in ("l2i6s3n1i6s4", "l2i6s3n1i6s3", "l2i6s2n1i6s3"):
        if out_lf.count(name) != 1:
            print(f"FAIL: {name} appears {out_lf.count(name)} times")
            return 1
    out = out_lf.replace("\n", "\r\n")
    P.write_bytes(out.encode("utf-8"))
    if P.read_bytes().decode("utf-8") != out:
        print("FAIL: round-trip mismatch")
        return 1
    print("registered l2i6s3n1i6s4, l2i6s3n1i6s3, l2i6s2n1i6s3")
    return 0


if __name__ == "__main__":
    sys.exit(main())