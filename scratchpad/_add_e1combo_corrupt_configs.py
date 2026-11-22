"""Register the E1 x frontier combination rows in check_corruption.CONFIGS (chain 24).

Eval env = `l2i6s4`'s own deploy contract: the tail penalty is a TRAINING-ONLY lever, inert at eval
(an eval pass carries an empty stream tag), and `run_one` forces `RWKV_QAT_FUSED=0` itself so the
state is observable on the Python per-step loop.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "scratchpad" / "quant_plan" / "corruption" / "check_corruption.py"

ANCHOR = '''    "l2i4s4n4": (RUN / "qat_it36l2i4s4n4" / "qat_124.pth", "l2i4s4n4",
                 "chain 17 point 3 (int4 card factors + int4 note factors, 2,304 bits)"),
'''

BLOCK = '''    # CHAIN 24: the E1 tail penalty on row 45's frontier layout (2,944 bits). Registered now because
    # chain 24 picks the lambda whose deploy deltas beat the penalty-free control and runs the
    # matching combination; a passing combination then needs its own finiteness check like any row.
    "l2i6s4e1c": (RUN / "qat_it36l2i6s4e1c" / "qat_124.pth", "l2i6s4",
                  "chain 24 (E1 lambda=1.0 on row 45's 2,944-bit layout)"),
    "l2i6s4e1d": (RUN / "qat_it36l2i6s4e1d" / "qat_124.pth", "l2i6s4",
                  "chain 24 (E1 lambda=3.0 on row 45's 2,944-bit layout)"),
    "l2i6s4e1e": (RUN / "qat_it36l2i6s4e1e" / "qat_124.pth", "l2i6s4",
                  "chain 24 (E1 lambda=10.0 on row 45's 2,944-bit layout)"),
'''


def main() -> int:
    raw = P.read_bytes().decode("utf-8")
    lf = raw.replace("\r\n", "\n")
    if lf.count(ANCHOR) != 1:
        print(f"FAIL: anchor found {lf.count(ANCHOR)} times")
        return 1
    out_lf = lf.replace(ANCHOR, ANCHOR + BLOCK)
    for name in ("l2i6s4e1c", "l2i6s4e1d", "l2i6s4e1e"):
        if out_lf.count(name) != 2:
            print(f"FAIL: {name} appears {out_lf.count(name)} times, expected 2")
            return 1
    out = out_lf.replace("\n", "\r\n")
    P.write_bytes(out.encode("utf-8"))
    if P.read_bytes().decode("utf-8") != out:
        print("FAIL: round-trip mismatch")
        return 1
    print("registered the three chain-24 corruption configs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
