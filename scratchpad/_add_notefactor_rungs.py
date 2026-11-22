"""Register two note-FACTOR rungs in rust_grid.CONFIGS (a CPU-only deploy screen, no GPU).

WHY. The 7-user wide screen (`it36wide5`) priced the sub-3,136 corner and left two facts: the note's
int8 -> int6 FACTOR cut costs only +0.000170/+0.000216 for 384 note bits, and the note's int2 SHIFT
rung costs +0.0012-0.0014 imm at either factor width (a dead end). So the cheapest bits on offer are
the note's FACTOR width, not its shift width:

  l2i6s4n1i5s3  1,024 + 1,536 = 2,560   note rank-1 int5 factors + int3 shifts
  l2i6s4n1i4s3  1,024 + 1,344 = 2,368   note rank-1 int4 factors + int3 shifts

`l2i6s4n1i5s3` sits at the SAME 2,560 bits as `l2i6s4n1i6s2` (already priced at
+0.000104/+0.001642), but buys them with the note's factor width instead of its shift width, so it
should fix the immediate head that the int2-shift rung wrecked. `l2i6s4n1i4s3` probes where the note's
factor cliff is, between row 41's int8 (+0.000219 imm) and row 42's int3 (+0.001343 imm).

`bits_for` must agree with quant_sweep.bits_for: note rank-1 intN with int3 shifts, 3 layers =
3*(2*1*32*N + 64*3).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "scratchpad" / "quant_plan" / "rust_grid.py"

ANCHOR = '''    "l2i6s4n1i6s2": {"RWKV_STATE_LOWRANK_SCOPE": "card:2:int6,note:1:int6",
                     "RWKV_QUANT_SHIFTS": "1",
                     "RWKV_STATE_SHIFT_LEVEL": "card:int4,note:int2"},
'''

BLOCK = '''    # THE NOTE-FACTOR RUNGS UNDER THE int6 CARD (2026-11-23). `it36wide5` showed the note's
    # int8 -> int6 FACTOR cut is cheap (+0.000170/+0.000216 for 384 bits) while its int2 SHIFT rung is
    # a dead end (+0.0012-0.0014 imm). These buy bits with the note's FACTOR width instead.
    "l2i6s4n1i5s3": {"RWKV_STATE_LOWRANK_SCOPE": "card:2:int6,note:1:int5",
                     "RWKV_QUANT_SHIFTS": "1",
                     "RWKV_STATE_SHIFT_LEVEL": "card:int4,note:int3"},
    "l2i6s4n1i4s3": {"RWKV_STATE_LOWRANK_SCOPE": "card:2:int6,note:1:int4",
                     "RWKV_QUANT_SHIFTS": "1",
                     "RWKV_STATE_SHIFT_LEVEL": "card:int4,note:int3"},
'''


def main() -> int:
    raw = P.read_bytes().decode("utf-8")
    lf = raw.replace("\r\n", "\n")
    if lf.count(ANCHOR) != 1:
        print(f"FAIL: anchor found {lf.count(ANCHOR)} times")
        return 1
    out_lf = lf.replace(ANCHOR, ANCHOR + BLOCK)
    for name in ("l2i6s4n1i5s3", "l2i6s4n1i4s3"):
        got = out_lf.count(name)
        if got < 1:
            print(f"FAIL: {name} missing after the insert")
            return 1
        print(f"  {name}: {got} occurrence(s)")
    out = out_lf.replace("\n", "\r\n")
    P.write_bytes(out.encode("utf-8"))
    if P.read_bytes().decode("utf-8") != out:
        print("FAIL: round-trip mismatch")
        return 1
    print("inserted l2i6s4n1i5s3 and l2i6s4n1i4s3")
    return 0


if __name__ == "__main__":
    sys.exit(main())
