"""The interval experiment's verdict, written BEFORE the arms reported.

Pair: `fixc` (end-to-END control) versus `e2sc` (end-to-START treatment). Same code, same id
fixes, same recipe, same seed, same KD schedule, same eval range; the dbs differ only in the
interval columns (verified: `verify_e2s_columns.py` showed exactly columns 2-7 moved and 0 of
415,668 changed entries moved upward).

It exists now, unrun, on purpose. An analysis written after seeing the numbers can always be
made to fit them; `PREREG.md` records three predictions and this file is the fixed test of each.

  1. e2s is WORSE on both modes, by +0.0001 to +0.0005.
     Rationale: end-to-start removes `duration(k)`, which is unavailable at prediction time.
     FSRS-7 lost +0.000111 at matched size for the same reason.

  2. ★ imm degrades MORE than ahead. `imm` predicts the rating of THE CURRENT review -- the very
     review whose duration leaks -- so for it the leak is a direct signal about the target.
     `ahead` predicts a future review, where duration(k) reaches the target only through the
     curve's sampling point. If AHEAD moves more, the mechanism is wrong and the effect is about
     curve placement rather than leakage.

  3. The effect is concentrated in SAME-DAY reviews. The definitions are numerically
     near-identical on longer intervals (duration is a median 0.001% of the gap, and 0.00% of
     long rows move by 10% or more). Anything showing up on long-interval users is the REFIT,
     not the interval.

⚠ THE GATE DOES NOT APPLY, and that is the point of writing it down first. The research gate
accepts only changes that improve BOTH modes. This arm is not trying to be better, it is trying to
be HONEST: a live Anki scheduler computes `now() - last_review_time`, which is end-to-start and
structurally cannot be anything else. A small regression here is the EXPECTED result and is the
size of the correction, not a reason to reject. Under the standard gate the arm that matches
deploy would be rejected for matching deploy.

Usage:
  interval_verdict.py --shares            (slow, CPU: cache per-user same-day share; run when
                                           the GPU is NOT mid-run -- it reads 2,500 parquet files)
  interval_verdict.py                     (fast: the verdict, using the cached shares if present)
"""

import glob
import json
import os
import statistics as st
import sys

CACHE = "scratchpad/features_ab/sameday_share.json"
ROOT = r"C:/Users/Andrew/anki-revlogs-10k/revlogs"
CTRL, TREAT = "fixc", "e2sc"
LO, HI = 5001, 7500


def build_shares():
    """Per-user fraction of reviews whose END-TO-END gap is under a day. Cached to JSON."""
    import numpy as np
    import pandas as pd

    out = {}
    for u in range(LO, HI + 1):
        hits = glob.glob(os.path.join(ROOT, "user_id=%d" % u, "*.parquet"))
        if not hits:
            continue
        try:
            df = pd.read_parquet(hits[0], columns=["elapsed_seconds"])
        except Exception:                                   # noqa: BLE001
            continue
        es = df["elapsed_seconds"].to_numpy().astype("float64")
        real = es >= 0
        n = int(real.sum())
        if n:
            out[str(u)] = float((es[real] < 86400).mean())
        if len(out) % 250 == 0 and out:
            print("  %d users..." % len(out), flush=True)
    json.dump(out, open(CACHE, "w"))
    print("cached same-day share for %d users -> %s" % (len(out), CACHE))


def load(tag, mode):
    p = "result/%s%s.jsonl" % ("RWKV-" if mode == "ahead" else "RWKV-P-", tag)
    if not os.path.exists(p):
        return None
    d = {}
    for line in open(p):
        r = json.loads(line)
        d[int(r["user"])] = r["metrics"]["LogLoss"]
    return d


def wilcoxon(diffs):
    """One-sided paired signed-rank p that the treatment is WORSE. scipy if present, else a
    normal approximation -- n is 2500, so the approximation is accurate."""
    try:
        from scipy.stats import wilcoxon as w
        return float(w(diffs, alternative="greater").pvalue)
    except Exception:                                       # noqa: BLE001
        import math
        nz = [d for d in diffs if d != 0]
        n = len(nz)
        if n < 10:
            return float("nan")
        order = sorted(range(n), key=lambda i: abs(nz[i]))
        ranks = [0.0] * n
        for r, i in enumerate(order, 1):
            ranks[i] = r
        wpos = sum(ranks[i] for i in range(n) if nz[i] > 0)
        mu = n * (n + 1) / 4.0
        sd = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
        z = (wpos - mu) / sd
        return 0.5 * math.erfc(z / math.sqrt(2))


def main():
    if "--shares" in sys.argv:
        build_shares()
        return 0

    res = {}
    for mode in ("ahead", "imm"):
        c, t = load(CTRL, mode), load(TREAT, mode)
        if c is None or t is None:
            print("missing results for %s (%s / %s) -- arms not finished" % (mode, CTRL, TREAT))
            return 1
        users = sorted(set(c) & set(t))
        diffs = [t[u] - c[u] for u in users]                # positive = e2s WORSE
        res[mode] = dict(users=users, diffs=diffs,
                         c=st.fmean(c[u] for u in users), t=st.fmean(t[u] for u in users),
                         delta=st.fmean(diffs), p=wilcoxon(diffs))

    print("PAIR: %s (end-to-END control) vs %s (end-to-START)   n=%d\n"
          % (CTRL, TREAT, len(res["ahead"]["users"])))
    print("%-6s %11s %11s %12s %12s" % ("mode", "control", "e2s", "delta", "p(e2s worse)"))
    for m in ("ahead", "imm"):
        r = res[m]
        print("%-6s %11.6f %11.6f %+12.6f %12.3g" % (m, r["c"], r["t"], r["delta"], r["p"]))

    print("\n--- PREDICTION 1: e2s worse by +0.0001 to +0.0005 in both modes")
    for m in ("ahead", "imm"):
        d = res[m]["delta"]
        verdict = "as predicted" if 0.0001 <= d <= 0.0005 else (
            "SMALLER than predicted" if 0 < d < 0.0001 else
            "LARGER than predicted" if d > 0.0005 else "WRONG SIGN -- e2s is BETTER")
        print("  %-6s %+0.6f  %s" % (m, d, verdict))

    print("\n--- PREDICTION 2: imm degrades MORE than ahead (the leak is about the CURRENT review)")
    da, di = res["ahead"]["delta"], res["imm"]["delta"]
    if di > da:
        print("  CONFIRMED: imm %+0.6f vs ahead %+0.6f (ratio %.2f)" % (di, da, di / da if da else float('nan')))
    else:
        print("  ★ REFUTED: ahead moved more (%+0.6f vs imm %+0.6f). The effect is NOT primarily"
              % (da, di))
        print("    a leak into the current review's rating; look at curve placement instead.")

    print("\n--- PREDICTION 3: the effect is concentrated in SAME-DAY reviews")
    if not os.path.exists(CACHE):
        print("  (no share cache -- run with --shares on an idle machine)")
    else:
        shares = json.load(open(CACHE))
        for m in ("ahead", "imm"):
            us, ds = res[m]["users"], res[m]["diffs"]
            pairs = [(shares[str(u)], d) for u, d in zip(us, ds) if str(u) in shares]
            if len(pairs) < 100:
                print("  %-6s too few users with a share" % m)
                continue
            pairs.sort()
            k = len(pairs) // 4
            lo = st.fmean(d for _, d in pairs[:k])
            hi = st.fmean(d for _, d in pairs[-k:])
            print("  %-6s bottom quartile same-day share: %+0.6f   top quartile: %+0.6f  %s"
                  % (m, lo, hi,
                     "concentrated as predicted" if abs(hi) > abs(lo) * 1.5 else
                     "NOT concentrated -- suggests the refit, not the interval"))

    print("\nREMINDER: a regression here is the SIZE OF THE CORRECTION, not a rejection. Deploy")
    print("computes end-to-start whatever we train on, so the honest number is the e2s one and")
    print("the champion's published figure is optimistic by this much as a deploy estimate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
