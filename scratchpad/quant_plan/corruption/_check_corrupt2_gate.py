import pathlib
import sys

RUN = pathlib.Path(r"C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\quant_plan")
sys.path.insert(0, str(RUN))
import it36corrupt2_chain as c

expect = {
    "it36l2i4s4": False,   # row 23: imm ub95 +0.000661 -> not a deploy candidate
    "it36l3i8s4": True,    # row 18: passes
    "it36l2i8s4": True,    # row 17: passes
    "it36l3i4s4": None,    # chain 11, not run yet
    "it36n1i8s3": None,    # chain 11, not run yet
}
bad = 0
for stamp, want in expect.items():
    got = c.gate_passed(stamp)
    flag = "ok" if got is want else "MISMATCH"
    if got is not want:
        bad += 1
    print(f"{stamp:14s} got={got!s:5s} want={want!s:5s} {flag}")
print("gate-guard mismatches:", bad)
print("points:", len(c.POINTS), [p[0] for p in c.POINTS])
