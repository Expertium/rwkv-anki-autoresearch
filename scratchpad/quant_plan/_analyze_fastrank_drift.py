"""How large is the FAST-path prediction drift in the DEPLOY-relevant sense?

`rust_fastrank_check.py` reports the MAX per-review change in p(recall), which says nothing about the
average cost. The gate is a MEAN log loss, so this reads the dumps that run already produced and
reports the distribution: mean |dp|, the |dp|/p scale that drives the log-loss delta, and how many
reviews are affected at all. Reference = the same config with no fast-path flag; fast = RWKV_LOWRANK_FAST=1.

Usage:
  .venv/Scripts/python.exe scratchpad/quant_plan/_analyze_fastrank_drift.py [--tag check] [--dir <live dir>]
"""
import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="check")
    ap.add_argument("--dir", default=None)
    args = ap.parse_args()
    live = Path(args.dir) if args.dir else (
        ROOT / "scratchpad" / "quant_plan" / f"rust_fastrank_preds_live_{args.tag}")

    print(f"reading {live}")
    print(f"{'config':10s} {'key':11s} {'n':>7s} {'mean|dp|':>9s} {'p99':>8s} {'p999':>8s} "
          f"{'max':>8s} {'>1e-3':>7s} {'>1e-2':>7s} {'>5e-2':>7s} {'mean |dp|/p':>11s}")
    for run_dir in sorted(live.glob("*_fast")):
        cfg = run_dir.name.split("_it")[0]
        ref_dir = live / run_dir.name.replace("_fast", "_ref")
        if not ref_dir.exists():
            print(f"{cfg}: no reference run next to {run_dir.name}")
            continue
        for key in ("pred_imm", "pred_ahead"):
            diffs, rels = [], []
            for p in sorted(ref_dir.glob("rust_pred_*.json")):
                a = json.loads(p.read_text(encoding="utf-8"))
                b = json.loads((run_dir / p.name).read_text(encoding="utf-8"))
                va, vb = a.get(key) or [], b.get(key) or []
                for x, y in zip(va, vb):
                    if x is None or y is None:
                        continue
                    d = abs(float(x) - float(y))
                    diffs.append(d)
                    if 0.02 < float(x) < 0.98:
                        rels.append(d / max(float(x), 1e-6))
            if not diffs:
                continue
            diffs.sort()
            n = len(diffs)

            def q(f):
                return diffs[min(n - 1, int(f * n))]
            print(f"{cfg:10s} {key:11s} {n:7d} {statistics.mean(diffs):9.2e} {q(0.99):8.2e} "
                  f"{q(0.999):8.2e} {diffs[-1]:8.2e} {sum(d > 1e-3 for d in diffs):7d} "
                  f"{sum(d > 1e-2 for d in diffs):7d} {sum(d > 5e-2 for d in diffs):7d} "
                  f"{statistics.mean(rels) if rels else float('nan'):11.2e}")


if __name__ == "__main__":
    main()
