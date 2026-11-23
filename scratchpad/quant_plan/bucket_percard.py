"""Per-CARD |delta p| readout from a bucket_tax per-review dump (the other team's item 3).

`bucket_tax.py` writes `bucket_rows_<tag>_<config>.npz` (one row per review: user, card_id, the two
heads' plain and quantized p, the label). Its own report gives the per-REVIEW |dp| percentiles and the
share of CARDS with a review past a threshold. The other team asked for the per-CARD distribution
(median, P99, max) plus those shares, so this tool aggregates the same dump per (user, card_id).

A card id is unique only WITHIN a user, so the group key must be (user, card_id).

Usage:
  .venv/Scripts/python.exe scratchpad/quant_plan/bucket_percard.py ^
      --sets "iter55_quant=bucket_rows_row55_l2i8s4.npz,row17_quant=bucket_rows_deploy17_l2i8s4.npz,..." ^
      --out scratchpad/quant_plan/bucket_percard_row55.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def head_stats(df, col):
    gmax = df.groupby(["user", "card_id"])[col].max().dropna()
    gmean = df.groupby(["user", "card_id"])[col].mean().dropna()
    rev = df[col].dropna()
    if not len(gmax) or not len(rev):
        return {"n_cards": 0, "n_reviews": 0}
    q = lambda s, p: float(s.quantile(p))
    return {
        "n_cards": int(len(gmax)),
        "n_reviews": int(len(rev)),
        "card_max": {"p50": float(gmax.median()), "p90": q(gmax, 0.90), "p99": q(gmax, 0.99),
                     "max": float(gmax.max()),
                     "share_gt_0.05": float((gmax > 0.05).mean()),
                     "share_gt_0.10": float((gmax > 0.10).mean())},
        "card_mean": {"p50": float(gmean.median()), "p90": q(gmean, 0.90), "p99": q(gmean, 0.99),
                      "max": float(gmean.max())},
        "review": {"p50": float(rev.median()), "p90": q(rev, 0.90), "p99": q(rev, 0.99),
                   "max": float(rev.max()),
                   "share_gt_0.05": float((rev > 0.05).mean()),
                   "share_gt_0.10": float((rev > 0.10).mean())},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", required=True, help="label=npz_path,label=npz_path")
    ap.add_argument("--preds-root", default=str(ROOT / "scratchpad" / "quant_plan"))
    ap.add_argument("--out", default=str(ROOT / "scratchpad" / "quant_plan" / "bucket_percard.json"))
    a = ap.parse_args()

    root = Path(a.preds_root)
    report = {"kind": "per-card |delta p| from bucket_tax dumps", "sets": {}}
    for spec in (s.strip() for s in a.sets.split(",") if s.strip()):
        label, _, rel = spec.partition("=")
        p = root / rel
        z = np.load(p, allow_pickle=False)
        df = pd.DataFrame({k: z[k] for k in
                           ("user", "card_id", "p_imm_p", "p_imm_q", "p_ahead_p", "p_ahead_q")})
        z.close()
        df["dimm"] = (df["p_imm_q"] - df["p_imm_p"]).abs()
        df["dahead"] = (df["p_ahead_q"] - df["p_ahead_p"]).abs()
        report["sets"][label] = {"npz": rel, "imm": head_stats(df, "dimm"),
                                 "ahead": head_stats(df, "dahead")}
        r = report["sets"][label]
        print(f"{label:12s} imm cards={r['imm']['n_cards']:,} "
              f"card_max p50 {r['imm']['card_max']['p50']:.5f} p99 {r['imm']['card_max']['p99']:.5f} "
              f">0.10 {100 * r['imm']['card_max']['share_gt_0.10']:.2f}%   "
              f"ahead >0.10 {100 * r['ahead']['card_max']['share_gt_0.10']:.2f}%", flush=True)

    out = Path(a.out)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
