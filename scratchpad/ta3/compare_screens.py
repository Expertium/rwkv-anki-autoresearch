"""Paired screen comparison for the TA3 arm against its control (or any two screen tags).

Both arms are evaluated by `optimization/eval_sharded.py` on the same fixed 750-user roster, and
the merged per-user outputs land in `result/adw_<tag>.jsonl` (ahead mode) and
`result/adwP_<tag>.jsonl` (immediate mode). This script pairs by user, reports the per-mode mean
delta, the one-sided Wilcoxon p-value for "treatment is better" (lower LogLoss), and applies the
project gate: BOTH modes must improve by >= 0.0001 with p < 1e-4.

Usage:
  .venv/Scripts/python.exe scratchpad/ta3/compare_screens.py \
      --ta3 ta3_fullrun_20261115_XXXXXXXX_e04_YYYYYY \
      --plain adaptive_fullrun_92_20261105_90fecbaa_a04

With no --ta3/--plain it just lists the candidate tags it can see.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "result"
GATE_DELTA = 0.0001
GATE_P = 1e-4


def load(tag):
    out = []
    for prefix in ("adw", "adwP"):
        path = RESULT / f"{prefix}_{tag}.jsonl"
        if not path.exists():
            raise SystemExit(f"missing {path}")
        rows = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows[int(row["user"])] = float(row["metrics"]["LogLoss"])
        out.append(rows)
    return out  # [ahead, imm]


def wilcoxon(diffs):
    try:
        from scipy.stats import wilcoxon
        stat, p = wilcoxon(diffs, alternative="greater")
        return float(p)
    except Exception:
        # Fallback: sign test, exact by binomial.
        import math
        pos = sum(1 for d in diffs if d > 0)
        n = sum(1 for d in diffs if d != 0)
        if n == 0:
            return 1.0
        return sum(math.comb(n, k) for k in range(pos, n + 1)) / (2 ** n)


def compare(ta3_tag, plain_tag):
    ta, tp = load(ta3_tag), load(plain_tag)
    print(f"TA3   (treatment): {ta3_tag}")
    print(f"PLAIN (control)  : {plain_tag}")
    for mode, (a, b) in zip(("ahead", "imm"), zip(ta, tp)):
        users = sorted(set(a) & set(b))
        only_a, only_b = sorted(set(a) - set(b)), sorted(set(b) - set(a))
        if not users:
            raise SystemExit(f"{mode}: no shared users")
        diffs = [b[u] - a[u] for u in users]  # positive = TA3 lower loss = better
        mean_a = sum(a[u] for u in users) / len(users)
        mean_b = sum(b[u] for u in users) / len(users)
        mean_d = sum(diffs) / len(diffs)
        med = sorted(diffs)[len(diffs) // 2]
        p = wilcoxon(diffs)
        n_better = sum(1 for d in diffs if d > 0)
        n_worse = sum(1 for d in diffs if d < 0)
        verdict = "PASS" if (mean_d >= GATE_DELTA and p < GATE_P) else "fail"
        print(f"  {mode:5s}: n={len(users)} ta3={mean_a:.6f} plain={mean_b:.6f} "
              f"delta=+{mean_d:.6f} median=+{med:.6f} better/worse={n_better}/{n_worse} "
              f"p={p:.3e} -> gate {verdict}")
        if only_a or only_b:
            print(f"         roster mismatch: only_ta3={only_a[:10]} only_plain={only_b[:10]}")
    print(f"  gate: both modes >= {GATE_DELTA} and p < {GATE_P}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ta3", default="")
    ap.add_argument("--plain", default="")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if not args.ta3 or not args.plain:
        tags = sorted(p.stem.replace("adw_", "", 1) for p in RESULT.glob("adw_*.jsonl"))
        for t in tags:
            print(t)
        raise SystemExit(0)
    compare(args.ta3, args.plain)
