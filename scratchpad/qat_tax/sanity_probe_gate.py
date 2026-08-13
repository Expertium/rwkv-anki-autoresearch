"""Validity gate for a QAT eval probe: is the number SANE, before the 7-h full eval?

Motivated by cell 3 (2026-08-13), which spent ~3 h of GPU producing a number that could not be
used — a 10-user probe would have exposed it in minutes. This gate encodes "cheap probe first"
for VALIDITY: compare the probe tags against iter 45's plain results on the intersection and fail
unless the cost sits in a plausible band.

Band: [-0.005, +0.02] per mode. A working QAT eval on this trunk costs a few thousandths (the
measured tax is +0.004/+0.006; an improvement could be slightly negative); a broken one is either
~0 (quantization silently off — the architecture.py bug) or huge (wrong/mismatched codebook, the
0.40/0.55 cell-3 class). Both failure modes land far outside the band.

Usage: python sanity_probe_gate.py <ahead_tag> <imm_tag>   (result/RWKV-<tag>.jsonl etc.)
Exit 0 = sane, 46 = out of band / missing.
"""
import json
import sys

BAND = (-0.005, 0.02)


def load(p):
    d = {}
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                r = json.loads(line)
                d[r["user"]] = r["metrics"]["LogLoss"]
    return d


def main():
    tag = sys.argv[1]
    ok = True
    for mode, cand, base in (("ahead", f"result/RWKV-{tag}.jsonl", "result/RWKV-iter45_kddecay.jsonl"),
                             ("imm", f"result/RWKV-P-{tag}.jsonl", "result/RWKV-P-iter45_kddecay.jsonl")):
        try:
            c, b = load(cand), load(base)
        except OSError as e:
            print(f"[probe-gate] {mode}: MISSING {e}")
            ok = False
            continue
        ks = sorted(set(c) & set(b))
        if not ks:
            print(f"[probe-gate] {mode}: no common users")
            ok = False
            continue
        cost = sum(c[k] for k in ks) / len(ks) - sum(b[k] for k in ks) / len(ks)
        inband = BAND[0] <= cost <= BAND[1]
        print(f"[probe-gate] {mode}: cost {cost:+.6f} on n={len(ks)} -> "
              f"{'SANE' if inband else 'OUT OF BAND ' + str(BAND)}")
        ok = ok and inband
    return 0 if ok else 46


if __name__ == "__main__":
    sys.exit(main())
