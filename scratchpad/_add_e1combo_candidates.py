"""Register the E1 x frontier combination candidates: `l2i6s4e1c/d/e`.

Row 45 (`l2i6s4`, 1,024/1,920 = 2,944 bits) PASSES the log-loss gate, but its immediate head sits at
0.89x of the +0.0005 ceiling (ub95 +0.000446), so the next 128-bit cut has no room. If the E1 tail
penalty lowers the deploy tax at all, it is worth spending on that layout - which is what these three
candidates are: row 45's exact scope on the fused-off Python QAT loop, plus the penalty at stride 8.

The three lambdas match chain 23's points one for one, so chain 24 can pick the lambda whose deploy
deltas actually beat the penalty-free control and run the matching combination.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "scratchpad" / "quant_plan" / "quant_sweep.py"

ANCHOR = '''    "l2i8s4e1e": ({"RWKV_QAT_LOWRANK_SCOPE": "card:2:int8,note:1:int8",
                   "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int4",
                   "RWKV_QAT_FUSED": "0", "RWKV_QAT_TAIL_PENALTY": "10.0",
                   "RWKV_QAT_TAIL_STRIDE": "8"},
                  "row 17 + tail-energy penalty lambda=10.0, stride 8 (~25% of the loss) (1,280/2,304)", None),
'''

BLOCK = '''    # ---- E1 x THE FRONTIER LAYOUT (chain 24, gated on chain 23's own deltas) ------------------------
    # These are row 45's scope exactly (card rank-2 int6 + int4 shifts, note rank-1 int6 + int4 shifts,
    # 1,024/1,920 = 2,944 bits) with the E1 penalty on. Row 45 PASSES but its immediate head is at
    # 0.89x of the ceiling, so any tax the penalty removes is directly spendable there.
    "l2i6s4e1c": ({"RWKV_QAT_LOWRANK_SCOPE": "card:2:int6,note:1:int6",
                   "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int4",
                   "RWKV_QAT_FUSED": "0", "RWKV_QAT_TAIL_PENALTY": "1.0",
                   "RWKV_QAT_TAIL_STRIDE": "8"},
                  "row 45 (2,944 bits) + tail-energy penalty lambda=1.0, stride 8", None),
    "l2i6s4e1d": ({"RWKV_QAT_LOWRANK_SCOPE": "card:2:int6,note:1:int6",
                   "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int4",
                   "RWKV_QAT_FUSED": "0", "RWKV_QAT_TAIL_PENALTY": "3.0",
                   "RWKV_QAT_TAIL_STRIDE": "8"},
                  "row 45 (2,944 bits) + tail-energy penalty lambda=3.0, stride 8", None),
    "l2i6s4e1e": ({"RWKV_QAT_LOWRANK_SCOPE": "card:2:int6,note:1:int6",
                   "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int4",
                   "RWKV_QAT_FUSED": "0", "RWKV_QAT_TAIL_PENALTY": "10.0",
                   "RWKV_QAT_TAIL_STRIDE": "8"},
                  "row 45 (2,944 bits) + tail-energy penalty lambda=10.0, stride 8", None),
'''


def main() -> int:
    raw = P.read_bytes().decode("utf-8")
    lf = raw.replace("\r\n", "\n")
    if lf.count(ANCHOR) != 1:
        print(f"FAIL: anchor found {lf.count(ANCHOR)} times")
        return 1
    out_lf = lf.replace(ANCHOR, ANCHOR + BLOCK)
    for name in ("l2i6s4e1c", "l2i6s4e1d", "l2i6s4e1e"):
        if out_lf.count(name) != 1:
            print(f"FAIL: {name} appears {out_lf.count(name)} times")
            return 1
    out = out_lf.replace("\n", "\r\n")
    P.write_bytes(out.encode("utf-8"))
    if P.read_bytes().decode("utf-8") != out:
        print("FAIL: round-trip mismatch")
        return 1
    print("registered l2i6s4e1c, l2i6s4e1d, l2i6s4e1e")
    return 0


if __name__ == "__main__":
    sys.exit(main())
