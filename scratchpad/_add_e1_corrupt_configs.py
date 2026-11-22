"""Register the three E1 corruption configs in check_corruption.CONFIGS.

The eval env is taken from CANDIDATES[cand]; the E1 points use `l2i8s4pf`'s env, because the tail
penalty is a TRAINING-ONLY lever: it is not part of the deploy contract and it is inert at eval (an
eval pass carries an empty stream tag, so nothing accumulates). `run_one` forces RWKV_QAT_FUSED=0
itself, so the Python per-step loop - the only path where the state is visible - is used either way.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "scratchpad" / "quant_plan" / "corruption" / "check_corruption.py"

ANCHOR = '''    "l2i6s4n1i6s3": (RUN / "qat_it36l2i6s4n1i6s3" / "qat_124.pth", "l2i6s4n1i6s3",
                     "chain 22 point 1 (int6 card + note int6 factors + note int3 shifts, 2,752 bits)"),
'''

BLOCK = '''    # E1 TAIL-ENERGY PENALTY, chain 23 (2026-11-23). Row 17's layout trained on the fused-off Python
    # QAT loop with `lambda * mean tail fraction` added to the loss. The deploy env is unchanged
    # (`l2i8s4pf`'s), so the check measures the same state path as row 44's layout; what changes is
    # what the training objective pushed the state spectrum toward.
    "l2i8s4e1c": (RUN / "qat_it36l2i8s4e1c" / "qat_124.pth", "l2i8s4pf",
                  "chain 23 point 1 (E1 tail penalty lambda=1.0, stride 8, row 17's layout)"),
    "l2i8s4e1d": (RUN / "qat_it36l2i8s4e1d" / "qat_124.pth", "l2i8s4pf",
                  "chain 23 point 2 (E1 tail penalty lambda=3.0, stride 8, row 17's layout)"),
    "l2i8s4e1e": (RUN / "qat_it36l2i8s4e1e" / "qat_124.pth", "l2i8s4pf",
                  "chain 23 point 3 (E1 tail penalty lambda=10.0, stride 8, row 17's layout)"),
'''


def main() -> int:
    raw = P.read_bytes().decode("utf-8")
    lf = raw.replace("\r\n", "\n")
    print("bare LF:", lf.count("\n") - lf.count("\r\n"), "| CRLF lines:", raw.count("\r\n"))
    if lf.count(ANCHOR) != 1:
        print(f"FAIL: anchor found {lf.count(ANCHOR)} times")
        return 1
    out_lf = lf.replace(ANCHOR, ANCHOR + BLOCK)
    if not all(out_lf.count(f"l2i8s4e1{n}") == 2 for n in "cde"):
        print("FAIL: bad insert", [out_lf.count(f"l2i8s4e1{n}") for n in "cde"])
        return 1
    out = out_lf.replace("\n", "\r\n")
    P.write_bytes(out.encode("utf-8"))
    chk = P.read_bytes().decode("utf-8")
    if chk != out:
        print("FAIL: round-trip mismatch")
        return 1
    print("inserted the three E1 corruption configs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
