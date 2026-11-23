"""The answer-count mix of a user roster: bucket shares and the <=k-earlier-answer shares.

No model is involved - this reads the review logs only. It exists to put our per-bucket tax next to
the other team's (2026-11-23) numbers on the same footing: their P10/P25/P50/P75/P90/P95/P99 answer
counts and their "share of predicted reviews whose card had <=k earlier answers" (15/28/48/69/85% for
k=1/2/4/8/16).

Usage:
  .venv/Scripts/python.exe scratchpad/quant_plan/bucket_mix.py --users-file scratchpad/quant_plan/roster750_seed20261119.json
  .venv/Scripts/python.exe scratchpad/quant_plan/bucket_mix.py --users 5025,7044 --out scratchpad/quant_plan/bucket_mix_corrupt50.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(r"C:\Users\Andrew\anki-revlogs-10k")
EDGES = [(1, 2), (3, 4), (5, 8), (9, 15), (16, 31), (32, 127), (128, 1 << 30)]


def ordinals(user):
    df = pd.read_parquet(DATA / "revlogs" / f"user_id={user}", columns=["card_id"])
    return (df.groupby("card_id").cumcount() + 1).to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", default="")
    ap.add_argument("--users-file", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.users_file:
        users = json.loads(Path(args.users_file).read_text(encoding="utf-8"))
        users = [int(u) for u in (users.get("users") if isinstance(users, dict) else users)]
        label = Path(args.users_file).stem
    else:
        users = [int(u) for u in args.users.split(",")]
        label = "roster"

    ord_all = np.concatenate([ordinals(u) for u in users])
    cards = []
    for u in users:
        df = pd.read_parquet(DATA / "revlogs" / f"user_id={u}", columns=["card_id"])
        cards.append(df.groupby("card_id").size().to_numpy())
    card_lens = np.concatenate(cards)

    report = {"label": label, "n_users": len(users), "n_reviews": int(ord_all.size),
              "n_cards": int(card_lens.size)}
    print(f"{label}: {len(users)} users, {ord_all.size} answers, {card_lens.size} cards")
    qs = [10, 25, 50, 75, 90, 95, 99]
    print("answers per card P" + "/P".join(str(q) for q in qs) + ": " +
          "/".join(str(int(np.percentile(card_lens, q))) for q in qs))
    report["card_len_percentiles"] = {f"P{q}": int(np.percentile(card_lens, q)) for q in qs}
    print(f"{'bucket':>8s} {'reviews':>10s} {'share':>7s}")
    report["buckets"] = {}
    for lo, hi in EDGES:
        n = int(((ord_all >= lo) & (ord_all <= hi)).sum())
        name = f"{lo}-{hi}" if hi < (1 << 30) else "128+"
        report["buckets"][name] = {"reviews": n, "share": n / ord_all.size}
        print(f"{name:>8s} {n:10d} {n / ord_all.size:7.3f}")
    print(f"{'<=k':>4s} {'share of answers':>17s}")
    report["le_k"] = {}
    for k in (0, 1, 2, 4, 8, 15, 16, 31):
        s = float((ord_all <= k + 1).mean())
        report["le_k"][str(k)] = s
        print(f"{k:>4d} {s:17.3f}")
    out = Path(args.out) if args.out else ROOT / "scratchpad" / "quant_plan" / f"bucket_mix_{label}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
