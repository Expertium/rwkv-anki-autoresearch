"""Resume-skip smoke comparator: the resumed run's group-key sequence must equal the
uninterrupted run's sequence from the resume point on (group order is python-random
driven, independent of torch RNG, so keys must match EXACTLY; losses may differ
slightly -- the dropout stream restarts at the resume point).

Usage: python compare_keys.py <a_log> <b_log> <resume_completed_steps>
"""
import re
import sys

KEY = re.compile(r"^\[\(")


def keys(path):
    return [l.rstrip("\n") for l in open(path, encoding="utf-8", errors="replace")
            if KEY.match(l)]


a_log, b_log, skip = sys.argv[1], sys.argv[2], int(sys.argv[3])
a, b = keys(a_log), keys(b_log)
expect = a[skip:]
print(f"A: {len(a)} group steps; B: {len(b)}; expecting B == A[{skip}:] ({len(expect)})")
assert len(b) == len(expect), f"LENGTH MISMATCH: B has {len(b)}, expected {len(expect)}"
for i, (x, y) in enumerate(zip(expect, b)):
    assert x == y, (f"KEY MISMATCH at resumed step {skip + i + 1}:\n  A: {x}\n  B: {y}")
print("KEYS_MATCH_EXACT")
