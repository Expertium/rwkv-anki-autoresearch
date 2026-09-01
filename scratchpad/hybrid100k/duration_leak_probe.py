"""Is end-to-END's advantage information, or a LEAK? A decisive, cheap test.

CONTEXT. Andrew ran FSRS-7 on all 10k users under both interval definitions:

    LogLoss   end-to-end 0.3179   end-to-start 0.3182   (end-to-start WORSE by 0.0003)
    AUC       end-to-end 0.7520   end-to-start 0.7515   (end-to-start WORSE by 0.0005)
    RMSE      6.36% both

Two readings fit that, and they imply opposite decisions:

  A. end-to-END is genuinely more informative, so the current definition should stay.
  B. end-to-END LEAKS. It is answer(k) - answer(k-1), so it silently contains duration(k) --
     the length of the review being predicted. That quantity does not exist at prediction time
     and it correlates with the outcome, because a review the user struggles with takes longer.
     A larger interval lowers predicted retrievability, which is exactly the right direction to
     look good on a review that was about to be failed. A benchmark that trains and scores both
     arms self-consistently REWARDS the leak.

THE TEST THAT SEPARATES THEM. Hold the end-to-START gap fixed, then ask whether duration(k)
still separates outcomes. Within a narrow band of end-to-start gap, the elapsed forgetting time
is already fully described, so duration(k) carries NO legitimate information about decay. Any
residual predictive power there is leak by construction.

Reported as a STRATIFIED AUC (concordance pooled within gap bins). 0.50 means no leak.

TWO CONTROLS, both necessary:
  * the long-interval stratum. duration(k) predicts the outcome EVERYWHERE -- that alone proves
    nothing. It only leaks where it is a material FRACTION of the interval, which is the
    same-day population. If the conditional AUC is high in both strata but the interval only
    moves in one, the mechanism is confirmed and its scope is bounded.
  * a shuffled-duration arm. Stratified AUC has a finite-sample floor when bins are coarse;
    shuffling duration within each bin measures that floor directly instead of assuming it.

Usage: python scratchpad/hybrid100k/duration_leak_probe.py [n_users] [n_bins]
CPU-only, single-threaded, minutes.
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

ROOT = r"C:/Users/Andrew/anki-revlogs-10k/revlogs"
SAME_DAY_S = 86_400


def auc(score, pos):
    """Mann-Whitney concordance. Returns (auc, n_pos*n_neg) or (nan, 0) if degenerate."""
    n_pos = int(pos.sum())
    n_neg = int(len(pos) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return np.nan, 0
    r = pd.Series(score).rank().to_numpy()          # average ranks handle ties correctly
    u = r[pos].sum() - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg), n_pos * n_neg


def stratified_auc(score, pos, strata):
    """Concordance computed inside each stratum, pooled by pair count."""
    num = den = 0.0
    used = 0
    for b in np.unique(strata):
        m = strata == b
        a, w = auc(score[m], pos[m])
        if w:
            num += a * w
            den += w
            used += 1
    return (num / den if den else np.nan), den, used


def qbins(x, n):
    """Quantile bins that tolerate heavy ties (short gaps repeat a lot)."""
    edges = np.unique(np.quantile(x, np.linspace(0, 1, n + 1)))
    return np.digitize(x, edges[1:-1], right=True)


def report(name, dur, fail, gap, n_bins, rng):
    if len(dur) < 1000:
        print("  %-22s too few rows (%d)" % (name, len(dur)))
        return
    raw, _ = auc(dur, fail)
    strata = qbins(gap, n_bins)
    cond, pairs, used = stratified_auc(dur, fail, strata)

    # the finite-sample floor: same bins, duration shuffled inside each one
    sh = dur.copy()
    for b in np.unique(strata):
        m = strata == b
        v = sh[m]
        rng.shuffle(v)
        sh[m] = v
    floor, _, _ = stratified_auc(sh, fail, strata)

    frac = np.median(dur / np.maximum(gap + dur, 1e-9))   # duration as a share of end-to-END
    print("  %-22s rows %9d   fail %5.1f%%   AUC(dur) %.4f   "
          "AUC(dur | gap) %.4f   shuffled floor %.4f   dur/interval median %6.3f%%   bins %d"
          % (name, len(dur), 100.0 * fail.mean(), raw, cond, floor, 100.0 * frac, used))
    return cond, floor


def main():
    n_users = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    n_bins = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    rng = np.random.default_rng(1234)

    files = sorted(glob.glob(os.path.join(ROOT, "**", "*.parquet"), recursive=True))
    step = max(1, len(files) // n_users)
    files = files[::step][:n_users]

    dur_l, fail_l, gap_l = [], [], []
    for f in files:
        df = pd.read_parquet(f, columns=["rating", "duration", "elapsed_seconds"])
        if df.empty:
            continue
        es = df["elapsed_seconds"].to_numpy().astype("float64")
        d = df["duration"].to_numpy().astype("float64") / 1000.0
        keep = es >= 0                       # drop the -1 first-review sentinel
        gap = np.maximum(es[keep] - d[keep], 0.0)     # end-to-START
        dur_l.append(d[keep])
        fail_l.append(df["rating"].to_numpy()[keep] == 1)
        gap_l.append(gap)

    dur = np.concatenate(dur_l)
    fail = np.concatenate(fail_l)
    gap = np.concatenate(gap_l)
    end_to_end = gap + dur

    print("users %d   reviews with a real gap %s   bins %d" % (len(files), f"{len(dur):,}", n_bins))
    print()
    print("AUC(dur)          = does the review's own duration predict failing it? (leak channel exists)")
    print("AUC(dur | gap)    = ...still, at a FIXED end-to-start gap? (leak is real, not confounding)")
    print("shuffled floor    = the same statistic with duration permuted inside each bin (the null)")
    print()

    same = end_to_end < SAME_DAY_S
    out = {}
    out["all"] = report("all rows", dur, fail, gap, n_bins, rng)
    out["same"] = report("same-day", dur[same], fail[same], gap[same], n_bins, rng)
    out["long"] = report("longer than a day", dur[~same], fail[~same], gap[~same], n_bins, rng)

    print()
    print("HOW MUCH INTERVAL DOES THE LEAK ACTUALLY MOVE (this is what bounds its effect):")
    for label, m in (("same-day", same), ("longer than a day", ~same)):
        if m.sum() < 100:
            continue
        share = dur[m] / np.maximum(end_to_end[m], 1e-9)
        print("  %-22s share of rows moving 10%% or more: %6.2f%%   p90 share %6.2f%%"
              % (label, 100.0 * (share >= 0.10).mean(), 100.0 * np.quantile(share, 0.90)))

    print()
    c = out["same"]
    if c and not np.isnan(c[0]) and c[0] - c[1] > 0.01:
        print("=> READING B SUPPORTED: at a fixed end-to-start gap, duration(k) still separates")
        print("   outcomes well above the shuffled floor. end-to-END therefore injects an")
        print("   outcome-correlated, prediction-time-unavailable quantity into the interval,")
        print("   concentrated in exactly the same-day rows where it is a material share of it.")
        print("   A self-consistent benchmark rewards that, so the 0.0003 is not evidence that")
        print("   end-to-end is the better DEPLOY definition.")
    else:
        print("=> READING B NOT SUPPORTED: duration adds nothing at a fixed gap. The end-to-end")
        print("   advantage is not explained by a leak and needs another explanation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
