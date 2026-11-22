"""Add the E1-on-shrink candidates (chain 16) to `quant_sweep.CANDIDATES`.

Chain 16 is GATED on chain 15 (`it36e1`) actually reducing the tax, so these four points only run
when the tail-energy penalty has proved it does something. They are the two near-miss shrink layouts
with each of the two lambdas:

  row 21's layout  `l2i8s4n4`   card rank-2 int8 + int4 shifts, note rank-1 int4 + int4
                    1,280/1,536 = 2,816 bits, FAILED by 0.00014 imm (+0.000328/+0.000638)
  row 22's layout  `l3i8s4n4`   card rank-3 int8 + int4 shifts, note rank-1 int4 + int4
                    1,792/1,536 = 3,328 bits, FAILED by 0.000025 imm (+0.000297/+0.000525)

Both are the smallest layouts the ladder has measured, and both are inside reach of a penalty that
removes part of the note's rank-1 truncation tax (the note's rank-1 int8 tax alone is +0.000258 imm).
Row 17 is 3,584 bits, so row 21 saves 768 and row 22 saves 256.

Raw-bytes insertion (quant_sweep.py mixes CRLF and LF; the anchor is CRLF). The script then imports
the module and prints the bit accounting, which proves the insertion is valid.

Usage: .venv/Scripts/python.exe scratchpad/quant_plan/corruption/_add_e1shrink_candidates.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
QS = HERE.parent / "quant_sweep.py"

ANCHOR = (
    b'    "l2i8s4e1b": ({"RWKV_QAT_LOWRANK_SCOPE": "card:2:int8,note:1:int8",\r\n'
    b'                   "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int4",\r\n'
    b'                   "RWKV_QAT_FUSED": "0", "RWKV_QAT_TAIL_PENALTY": "10.0"},\r\n'
    b'                  "row 17 + tail-energy penalty lambda=10.0 (~25% of the loss) (1,280/2,304)", None),\r\n'
)

BLOCK = [
    "    # ---- E1 applied to the two NEAR-MISS shrink layouts (chain 16, gated on chain 15 result) ----",
    "    # Row 21's layout (`l2i8s4n4`: card rank-2 int8 + int4 shifts, note rank-1 int4 + int4, 2,816",
    "    # bits) failed the log-loss gate by 0.00014 on imm (+0.000328/+0.000638). Row 22's layout",
    "    # (`l3i8s4n4`: card rank-3 int8 + the same int4 note, 3,328 bits) failed by only 0.000025",
    "    # (+0.000297/+0.000525). Row 17 is 3,584 bits, so these save 768 and 256 bits.",
    "    # Both failures are the note's rank-1 int4 tax, and the note's rank-1 int8 tax alone is",
    "    # +0.000258 imm, so a penalty that removes part of the note's truncation tax could flip them.",
    '    "l2i8s4n4e1a": ({"RWKV_QAT_LOWRANK_SCOPE": "card:2:int8,note:1:int4",',
    '                     "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int4",',
    '                     "RWKV_QAT_FUSED": "0", "RWKV_QAT_TAIL_PENALTY": "1.0"},',
    '                    "row 21\'s layout + tail penalty lambda=1.0 (1,280/1,536 = 2,816)", None),',
    '    "l2i8s4n4e1b": ({"RWKV_QAT_LOWRANK_SCOPE": "card:2:int8,note:1:int4",',
    '                     "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int4",',
    '                     "RWKV_QAT_FUSED": "0", "RWKV_QAT_TAIL_PENALTY": "10.0"},',
    '                    "row 21\'s layout + tail penalty lambda=10.0 (1,280/1,536 = 2,816)", None),',
    '    "l3i8s4n4e1a": ({"RWKV_QAT_LOWRANK_SCOPE": "card:3:int8,note:1:int4",',
    '                     "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int4",',
    '                     "RWKV_QAT_FUSED": "0", "RWKV_QAT_TAIL_PENALTY": "1.0"},',
    '                    "row 22\'s layout + tail penalty lambda=1.0 (1,792/1,536 = 3,328)", None),',
    '    "l3i8s4n4e1b": ({"RWKV_QAT_LOWRANK_SCOPE": "card:3:int8,note:1:int4",',
    '                     "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int4",',
    '                     "RWKV_QAT_FUSED": "0", "RWKV_QAT_TAIL_PENALTY": "10.0"},',
    '                    "row 22\'s layout + tail penalty lambda=10.0 (1,792/1,536 = 3,328)", None),',
]
NEW = "\r\n".join(BLOCK).encode("utf-8") + b"\r\n"


def main() -> int:
    raw = QS.read_bytes()
    hits = raw.count(ANCHOR)
    if hits != 1:
        print(f"ABORT: anchor found {hits} times, expected exactly 1")
        return 1
    if b"l2i8s4n4e1a" in raw:
        print("ABORT: the chain-16 candidates are already present")
        return 1
    QS.write_bytes(raw.replace(ANCHOR, ANCHOR + NEW))
    print(f"patched {QS} (+{len(NEW)} bytes)")

    sys.path.insert(0, str(HERE.parent))
    import quant_sweep as qs  # noqa: E402

    for name in ("l2i8s4n4e1a", "l2i8s4n4e1b", "l3i8s4n4e1a", "l3i8s4n4e1b"):
        env, _comment, override = qs.CANDIDATES[name]
        print(f"{name:14s} bits={qs.bits_for(env, override)}  "
              f"lambda={env.get('RWKV_QAT_TAIL_PENALTY')}  fused={env.get('RWKV_QAT_FUSED')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())