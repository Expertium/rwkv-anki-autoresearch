"""Summarize the round-robin speed arms: median steps/s per arm, plus the noise floor.

Reads scratchpad/profile_prep/combo_arms.log, which interleaves
    === round R arm NAME ...
    BENCH_RESULT ... steps_per_sec=X ...

Reports the median per arm and, crucially, the SPREAD WITHIN each arm -- the QAT-JIT test
found identical-flag runs varying by 5.3%, so an arm-to-arm difference smaller than the
within-arm spread is not a result.
"""
import os
import re
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "combo_arms.log")

ARM_RE = re.compile(r"=== round (\d+) arm (\w+)")
SPS_RE = re.compile(r"steps_per_sec=([0-9.]+)")
BAN_RE = re.compile(r"\[(compile|muon)\]")


def main():
    arms, banners, cur = {}, {}, None
    for line in open(LOG, errors="replace"):
        m = ARM_RE.search(line)
        if m:
            cur = m.group(2)
            continue
        if cur and BAN_RE.search(line):
            banners.setdefault(cur, set()).add(line.strip()[:60])
        m = SPS_RE.search(line)
        if m and cur:
            arms.setdefault(cur, []).append(float(m.group(1)))

    if not arms:
        print("no BENCH_RESULT rows found")
        return

    print(f"{'arm':10s} {'n':>2s} {'median':>8s} {'min':>8s} {'max':>8s} {'spread':>8s}  flag proof")
    med = {}
    for a, v in arms.items():
        med[a] = st.median(v)
        spread = (max(v) - min(v)) / max(st.median(v), 1e-9) * 100
        proof = "; ".join(sorted(banners.get(a, []))) or "-"
        print(f"{a:10s} {len(v):2d} {med[a]:8.4f} {min(v):8.4f} {max(v):8.4f} {spread:7.1f}%  {proof}")

    # widest within-arm spread = the noise floor any claim must clear
    floor = max((max(v) - min(v)) / max(st.median(v), 1e-9) for v in arms.values() if len(v) > 1)
    print(f"\nnoise floor (widest within-arm spread): {floor*100:.1f}%")

    if "base" in med:
        print("\nvs base:")
        for a in med:
            if a == "base":
                continue
            r = med[a] / med["base"]
            verdict = "REAL" if abs(r - 1) > floor else "inside noise -- NOT established"
            print(f"  {a:10s} {r:6.3f}x   {verdict}")
    if "nojit" in med and "compile" in med:
        r = med["compile"] / med["nojit"]
        verdict = "REAL" if abs(r - 1) > floor else "inside noise -- NOT established"
        print(f"\ncompile vs its own required baseline (nojit): {r:.3f}x   {verdict}")


if __name__ == "__main__":
    main()
