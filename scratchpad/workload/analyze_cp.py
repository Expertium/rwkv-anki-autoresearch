"""v2 report: workload ratio per DR level, FSRS-7 re-optimized at every checkpoint.

Every table is broken out BY DR LEVEL and never collapsed into a range (Andrew 2026-08-22).

TWO CARD SETS, because the choice turned out to move the answer by a factor of ~2 and
neither option is free:

  alive  (PRIMARY) -- cards that are actually reviewed again after the checkpoint. Uses
          hindsight to SELECT the population, but not to make either algorithm's decision,
          and it is identical for both arms. A card the user never touches again generates
          no workload in reality under any scheduler, so counting it is a phantom.
  active (sensitivity) -- every card whose last review at or before D left it in the review
          queue. Strictly past-only, which is the cleaner causal story, but at user 5100
          day 300 only 22 of 187 such cards are ever reviewed again, and the two arms
          disagree more on the abandoned ones: F/R@90 is 0.441 on `active` against 0.814 on
          `alive` for that user. So the phantom cards are NOT neutral and both numbers get
          reported.

Per-user statistic is POOLED across that user's checkpoints (sum of FSRS workload / sum of
RWKV workload): a per-checkpoint ratio has an unstable denominator, and in v1 one user's
mean-of-ratios reached 386 against a pooled 1.52.

Usage: .venv/Scripts/python.exe scratchpad/workload/analyze_cp.py [--json-out f.json]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from combine import DR_LEVELS  # noqa: E402
from analyze import sign_test_p  # noqa: E402

CP = Path(__file__).resolve().parent / "cp"
PCT = [round(dr * 100) for dr in DR_LEVELS]


def load():
    frames = [pd.read_parquet(f) for f in sorted(CP.glob("cp_u*.parquet"))]
    frames = [f for f in frames if len(f)]
    return pd.concat(frames, ignore_index=True) if frames else None


def per_user_ratio(d, fcol, rcol):
    g = d.groupby("user")[[fcol, rcol]].sum()
    return (g[fcol] / g[rcol]).to_numpy()


def table(d, fpre, rpre, title):
    print("")
    print(title)
    print("   %-6s %9s %9s %11s %9s %9s %6s"
          % ("DR", "median", "geomean", "p25..p75", "frac>1", "p", "n"))
    print("   " + "-" * 66)
    out = []
    for p in PCT:
        r = per_user_ratio(d, "%s_%d" % (fpre, p), "%s_%d" % (rpre, p))
        r = r[np.isfinite(r) & (r > 0)]
        if len(r) < 3:
            continue
        rec = {"dr": p, "median": float(np.median(r)),
               "geomean": float(np.exp(np.log(r).mean())),
               "p25": float(np.percentile(r, 25)), "p75": float(np.percentile(r, 75)),
               "frac_gt1": float((r > 1).mean()), "p": sign_test_p(r), "n": int(len(r))}
        out.append(rec)
        print("   %-6s %9.3f %9.3f %5.2f..%-5.2f %9.2f %9.4f %6d"
              % ("%d%%" % p, rec["median"], rec["geomean"], rec["p25"], rec["p75"],
                 rec["frac_gt1"], rec["p"], rec["n"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    d = load()
    if d is None:
        print("no checkpoint outputs in %s" % CP)
        return
    d = d.copy()
    users = sorted(d["user"].unique())
    print("=" * 78)
    print("WORKLOAD REPLAY v2 -- FSRS-7 RE-OPTIMIZED AT EVERY CHECKPOINT (--recency)")
    print("%d users, %d checkpoints, %d on default parameters (too little history to fit)"
          % (len(users), len(d), int(d["used_default"].sum())))
    print("median active cards/checkpoint %d, of which alive %d (%.0f%%)"
          % (d["n_active"].median(), d["n_alive"].median(),
             100 * d["n_alive"].sum() / max(d["n_active"].sum(), 1)))
    print("=" * 78)

    alive = table(d, "wfa", "wra",
                  "1. WORKLOAD RATIO BY DESIRED RETENTION -- ALIVE cards (PRIMARY)")
    print("   ratio > 1 = FSRS-7 costs MORE reviews/day, i.e. RWKV-Curve is more efficient.")

    act = table(d, "wf", "wr",
                "2. SENSITIVITY -- every active card, including ones never reviewed again")

    print("")
    print("3. ABSOLUTE ANCHOR -- workload vs the load the user ACTUALLY carried")
    print("   Alive cards only, since a card with no next review has no actual interval to")
    print("   compare against. Median over users of pooled W_model / pooled W_actual.")
    print("   %-6s %14s %14s" % ("DR", "FSRS / actual", "RWKV / actual"))
    print("   " + "-" * 38)
    anchor = []
    g = d.groupby("user")[["w_actual"] + ["wfa_%d" % p for p in PCT]
                          + ["wra_%d" % p for p in PCT]].sum()
    g = g[g["w_actual"] > 0]
    for p in PCT:
        f = float(np.median(g["wfa_%d" % p] / g["w_actual"]))
        r = float(np.median(g["wra_%d" % p] / g["w_actual"]))
        anchor.append({"dr": p, "fsrs_over_actual": f, "rwkv_over_actual": r})
        print("   %-6s %14.2f %14.2f" % ("%d%%" % p, f, r))
    print("   1.0 = the model asks for exactly the review load the user really carried.")

    print("")
    print("4. RATIO BY HOW MUCH HISTORY THE OPTIMIZER HAD (alive cards, pooled)")
    d["bucket"] = pd.cut(d["n_train"], [0, 400, 2000, 8000, 25000, 10 ** 9],
                         labels=["<400 (default)", "400-2k", "2k-8k", "8k-25k", ">25k"])
    print("   %-16s %8s " % ("train rows", "n_ckpt")
          + " ".join("%7s" % ("%d%%" % p) for p in PCT))
    print("   " + "-" * 74)
    buckets = []
    for b in d["bucket"].cat.categories:
        sub = d[d["bucket"] == b]
        if len(sub) < 10:
            continue
        cells = [sub["wfa_%d" % p].sum() / sub["wra_%d" % p].sum() for p in PCT]
        buckets.append({"bucket": str(b), "n": int(len(sub)),
                        "ratio": {str(p): float(c) for p, c in zip(PCT, cells)}})
        print("   %-16s %8d " % (b, len(sub)) + " ".join("%7.3f" % c for c in cells))

    ab = []
    if "wv1a_90" in d.columns:
        print("")
        print("5. WHAT THE RE-OPTIMIZATION CHANGED (alive cards)")
        print("   Single-variable A/B: same users, days, cards and mask. Only the parameter")
        print("   vector differs -- prefix-refit vs the stored final one, which has already")
        print("   seen ~80%% of the history.")
        print("   %-6s %13s %13s %10s %9s" % ("DR", "final w", "refit w", "change", "p"))
        print("   " + "-" * 56)
        for p in PCT:
            gg = d.groupby("user")[["wfa_%d" % p, "wv1a_%d" % p, "wra_%d" % p]].sum()
            r_new = (gg["wfa_%d" % p] / gg["wra_%d" % p]).to_numpy()
            r_old = (gg["wv1a_%d" % p] / gg["wra_%d" % p]).to_numpy()
            ok = np.isfinite(r_new) & np.isfinite(r_old) & (r_old > 0)
            pv = sign_test_p(r_new[ok] / r_old[ok])
            ab.append({"dr": p, "median_final": float(np.median(r_old[ok])),
                       "median_refit": float(np.median(r_new[ok])), "p": pv})
            print("   %-6s %13.3f %13.3f %+10.3f %9.4f"
                  % ("%d%%" % p, np.median(r_old[ok]), np.median(r_new[ok]),
                     np.median(r_new[ok]) - np.median(r_old[ok]), pv))
        print("   refit > final means the clairvoyant parameters HAD been flattering FSRS-7.")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            "n_users": len(users), "n_checkpoints": int(len(d)),
            "n_default": int(d["used_default"].sum()),
            "alive": alive, "active": act, "anchor": anchor,
            "by_train_bucket": buckets, "refit_vs_final": ab,
        }, indent=1), encoding="utf-8")
        print("")
        print("wrote %s" % args.json_out)


if __name__ == "__main__":
    main()
