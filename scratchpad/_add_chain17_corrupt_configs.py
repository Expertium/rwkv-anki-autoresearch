"""Register chain 17's three points as corruption configs (l2i6s4 first: it just PASSED).

`l2i6s4` is RESULTS row 45 and the new axis-1 frontier, so Andrew's standing order applies
immediately: a corruption check follows a passing row as soon as the GPU is free. The driver's eval
env comes from CANDIDATES[cand], and the candidate's own env is exactly the deploy env (the
`RWKV_QAT_FUSED=0` twin is forced by `run_one`).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "scratchpad" / "quant_plan" / "corruption" / "check_corruption.py"

ANCHOR = '''    "l2i6s4n1i6s3": (RUN / "qat_it36l2i6s4n1i6s3" / "qat_124.pth", "l2i6s4n1i6s3",
                     "chain 22 point 1 (int6 card + note int6 factors + note int3 shifts, 2,752 bits)"),
'''

BLOCK = '''    # CHAIN 17 (it36int6, 2026-11-23): the int6 factor ladder on row 17's layout. `l2i6s4` PASSED
    # the log-loss gate at 2,944 bits (+0.000125/+0.000382, ub95 +0.000189/+0.000446) - the smallest
    # passing row so far, 448 bits below row 35 - so its finiteness check runs first.
    "l2i6s4": (RUN / "qat_it36l2i6s4" / "qat_124.pth", "l2i6s4",
               "chain 17 point 1 (int6 factors on both streams, 2,944 bits) - RESULTS row 45"),
    "l2i5s4": (RUN / "qat_it36l2i5s4" / "qat_124.pth", "l2i5s4",
               "chain 17 point 2 (int5 card factors + int6 note factors, 2,624 bits)"),
    "l2i4s4n4": (RUN / "qat_it36l2i4s4n4" / "qat_124.pth", "l2i4s4n4",
                 "chain 17 point 3 (int4 card factors + int4 note factors, 2,304 bits)"),
'''


def main() -> int:
    raw = P.read_bytes().decode("utf-8")
    lf = raw.replace("\r\n", "\n")
    if lf.count(ANCHOR) != 1:
        print(f"FAIL: anchor found {lf.count(ANCHOR)} times")
        return 1
    out_lf = lf.replace(ANCHOR, ANCHOR + BLOCK)
    for name in ("l2i6s4\"", "l2i5s4\"", "l2i4s4n4\""):
        n = out_lf.count(name)
        if n < 2:
            print(f"FAIL: {name} appears {n} times, expected >= 2")
            return 1
    out = out_lf.replace("\n", "\r\n")
    P.write_bytes(out.encode("utf-8"))
    if P.read_bytes().decode("utf-8") != out:
        print("FAIL: round-trip mismatch")
        return 1
    print("registered l2i6s4, l2i5s4, l2i4s4n4")
    return 0


if __name__ == "__main__":
    sys.exit(main())
