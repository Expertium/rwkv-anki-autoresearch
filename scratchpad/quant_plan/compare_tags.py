"""Paired comparison of eval tags on their common users: means, deltas, one-sided bounds.

Reads `result/adw_<tag>.jsonl` (ahead) and `result/adwP_<tag>.jsonl` (imm), pairs users, and reports
per head: mean, delta vs the FIRST tag, a one-sided 95% upper bound on the delta (the quantity
constraint 3 bounds at +0.0005), and a one-sided Wilcoxon p for "worse than the first tag".

Usage:
  .venv/Scripts/python.exe scratchpad/quant_plan/compare_tags.py qbl_incumbent_a04 qnj_1119pm ql1i4_1119pm
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scratchpad" / "quant_plan"))
import quant_sweep as qs  # noqa: E402


def main():
    tags = sys.argv[1:]
    if len(tags) < 2:
        raise SystemExit("need at least two tags")
    pairs = {}
    for t in tags:
        p = qs.load_pair(t)
        if p is None:
            raise SystemExit(f"no merged results for tag {t}")
        pairs[t] = p
    print(f"{'tag':22s} {'n':>4s} {'ahead':>10s} {'imm':>10s} {'d ahead':>10s} {'ub95':>10s} "
          f"{'p':>9s} {'d imm':>10s} {'ub95':>10s} {'p':>9s}")
    base = pairs[tags[0]]
    for t in tags:
        a = qs.paired(base, pairs[t], 0)
        i = qs.paired(base, pairs[t], 1)
        print(f"{t:22s} {a['n']:4d} {a['mean_cur']:10.6f} {i['mean_cur']:10.6f} "
              f"{a['delta']:+10.6f} {a['ub95']:+10.6f} {a['p_worse']:9.3g} "
              f"{i['delta']:+10.6f} {i['ub95']:+10.6f} {i['p_worse']:9.3g}")


if __name__ == "__main__":
    main()
