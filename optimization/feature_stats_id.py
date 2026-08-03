"""Derive the STATISTICS constants for the timestamp features, from `anki-revlogs-10k-id`.

WHY THIS EXISTS: `optimization/FUTURE_FEATURES.md` "IMPLEMENTATION PLAN" lists four code sites for
the features rebuild. Three are mechanical (append column names, write the derivations, repoint the
dataset root). The fourth -- **`STATISTICS` (`data_processing.py:43+`), a mean/std per new
continuous column** -- is a MEASUREMENT, and it was the last real unknown blocking the rebuild.
This computes them, CPU-only, so it can run while the GPU is busy.

★ THE DESIGN POINT: it recomputes the NINE EXISTING constants in the same pass. Those are hardcoded
upstream values whose derivation was never written down (which sample? which users? are the
"missing" -1 rows included in the mean?). Reproducing them from raw parquet is the only way to know
that the recipe here matches the one the live features were built with -- otherwise the new columns
would be normalized on a subtly different convention than the 24 they sit beside, which is exactly
the silent train/eval/deploy mismatch CLAUDE.md section 9 exists to catch. Treat a mismatch on the
existing constants as a BUG IN THIS SCRIPT, not as a correction to upstream.

The "missing" convention is the ambiguity that matters. Upstream scales as

    np.where(x == -1, 0, np.log(1 + 1e-5 + x))   then   (v - mean) / std

so a -1 row takes log-value 0 and lands at -mean/std after standardization. Whether `mean` was
computed over the non-missing rows only, or over the whole column with missing folded to 0, is not
recoverable from the code. Both are computed here and printed side by side; the one that reproduces
the hardcoded numbers is the convention to use for the new columns.

Read-only on the dataset. One user at a time, scalar accumulators, so RAM stays flat -- it is safe
to run beside training (the RAM guard's floor is the thing to respect, not the CPU).

    .venv/Scripts/python.exe optimization/feature_stats_id.py --users 300
    .venv/Scripts/python.exe optimization/feature_stats_id.py --users 40 --out scratchpad/fs.json
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(r"C:\Users\Andrew\anki-revlogs-10k-id")
SECONDS_PER_DAY = 86400

# Sample from the TRAIN half only. Normalization constants are part of the model, so deriving them
# from 5001-10000 would leak eval-set statistics into every candidate's inputs.
TRAIN_USER_LO, TRAIN_USER_HI = 1, 5000

# Anki ids are epoch-MILLISECONDS; the default deck and default preset both use the sentinel id 1.
# FUTURE_FEATURES.md measured the population as cleanly bimodal (real stamp, or the sentinel), so
# any threshold between them works. 1e11 ms = 1973, far below Anki's existence and far above 1.
TIMESTAMP_MIN_MS = 1e11

# The hardcoded constants this script must reproduce (data_processing.py:43+).
UPSTREAM = {
    "elapsed_days": (1.51, 1.62),
    "elapsed_days_cumulative": (2.14, 2.25),
    "elapsed_seconds": (9.96, 5.21),
    "elapsed_seconds_cumulative": (10.86, 5.8),
    "duration": (8.9, 1.07),
    "diff_new_cards": (2.945, 2.011),
    "diff_reviews": (4.64, 2.59),
    "cum_new_cards_today": (2.55, 1.41),
    "cum_reviews_today": (4.59, 1.30),
}


class Acc:
    """Streaming mean/std. Also tracks how often the value was negative or missing, which is the
    diagnostic that matters for the new difference features (a card created before its deck is a
    real Anki state, not a bug -- but it decides whether the column needs a sign-safe transform)."""

    __slots__ = ("n", "s", "ss", "n_neg", "n_miss", "lo", "hi")

    def __init__(self):
        self.n = 0
        self.s = 0.0
        self.ss = 0.0
        self.n_neg = 0
        self.n_miss = 0
        self.lo = math.inf
        self.hi = -math.inf

    def add(self, v):
        v = np.asarray(v, dtype=np.float64)
        v = v[np.isfinite(v)]
        if v.size == 0:
            return
        self.n += v.size
        self.s += float(v.sum())
        self.ss += float(np.dot(v, v))
        if v.size:
            self.lo = min(self.lo, float(v.min()))
            self.hi = max(self.hi, float(v.max()))

    def note_raw(self, raw):
        """Count negatives / missings on the RAW (pre-log) values."""
        raw = np.asarray(raw)
        self.n_neg += int((raw < 0).sum())

    @property
    def mean(self):
        return self.s / self.n if self.n else float("nan")

    @property
    def std(self):
        if self.n < 2:
            return float("nan")
        var = self.ss / self.n - self.mean**2
        return math.sqrt(max(var, 0.0))

    def as_dict(self):
        return {
            "n": self.n, "mean": round(self.mean, 4), "std": round(self.std, 4),
            "min": None if self.lo == math.inf else round(self.lo, 4),
            "max": None if self.hi == -math.inf else round(self.hi, 4),
            "n_neg_raw": self.n_neg,
        }


def log_t(x):
    """Upstream's transform for the elapsed_* / duration family."""
    return np.log(1.0 + 1e-5 + np.asarray(x, dtype=np.float64))


def log3(x):
    """Upstream's transform for the count family (diff_new_cards etc.): np.log(3 + x)."""
    return np.log(3.0 + np.asarray(x, dtype=np.float64))


def signed_log(x):
    """For differences that can legitimately be negative (card created before its deck).

    sign(x) * log1p(|x|) -- monotone, zero-preserving, and it keeps the two sides on one axis
    instead of needing a separate is-negative dim.
    """
    x = np.asarray(x, dtype=np.float64)
    return np.sign(x) * np.log1p(np.abs(x))


def user_ids(n_users, seed=0):
    """Spread the sample across the train half. Ids are not size-ordered, so a stride is a fair
    sample and is reproducible without a seed."""
    lo, hi = TRAIN_USER_LO, TRAIN_USER_HI
    stride = max(1, (hi - lo + 1) // n_users)
    return list(range(lo, hi + 1, stride))[:n_users]


def deck_depths(df_decks):
    """Depth in the A::B::C tree via parent_id. 0 = top level. Returns {deck_id: depth}."""
    parent = dict(zip(df_decks["deck_id"], df_decks["parent_id"]))
    known = set(parent)
    depth = {}

    def resolve(d, seen):
        if d in depth:
            return depth[d]
        p = parent.get(d, 0)
        if p == 0 or p not in known or p in seen:
            depth[d] = 0
            return 0
        seen.add(d)
        v = resolve(p, seen) + 1
        depth[d] = v
        return v

    for d in parent:
        resolve(d, set())
    return depth


def process_user(uid, acc, cov, circ):
    r = pd.read_parquet(DATA / "revlogs" / f"user_id={uid}")
    if len(r) < 2:
        return 0
    c = pd.read_parquet(DATA / "cards", filters=[("user_id", "=", uid)])
    d = pd.read_parquet(DATA / "decks", filters=[("user_id", "=", uid)])
    c = c.drop(columns=["user_id"])
    d = d.drop(columns=["user_id"])

    r["review_th"] = range(1, len(r) + 1)
    df = r.merge(c, on="card_id", how="left", validate="many_to_one")
    df = df.merge(d, on="deck_id", how="left", validate="many_to_one")
    assert len(df) == len(r)

    # ---------- the EXISTING nine, recomputed (the self-check) ----------
    df["elapsed_days_cumulative"] = df.groupby("card_id")["elapsed_days"].cumsum()
    df["elapsed_seconds_cumulative"] = df.groupby("card_id")["elapsed_seconds"].cumsum()
    df["is_first_review"] = (df["elapsed_days"] == -1).astype(int)
    df["cum_new_cards"] = df["is_first_review"].cumsum()
    df["diff_new_cards"] = df.groupby("card_id")["cum_new_cards"].diff().fillna(0)
    df["diff_reviews"] = np.maximum(
        0, -1 + df.groupby("card_id")["review_th"].diff().fillna(0))
    df["cum_new_cards_today"] = df.groupby("day_offset")["is_first_review"].cumsum()
    df["cum_reviews_today"] = df.groupby("day_offset").cumcount()

    for name in ("elapsed_days", "elapsed_days_cumulative",
                 "elapsed_seconds", "elapsed_seconds_cumulative"):
        v = df[name].to_numpy()
        present = v != -1
        acc[f"{name}__present"].add(log_t(v[present]))
        acc[f"{name}__withmiss"].add(np.where(v == -1, 0.0, log_t(np.maximum(v, 0))))
    acc["duration__present"].add(log_t(df["duration"].to_numpy()))
    acc["duration__withmiss"].add(log_t(df["duration"].to_numpy()))
    for name in ("diff_new_cards", "diff_reviews", "cum_new_cards_today", "cum_reviews_today"):
        acc[f"{name}__present"].add(log3(df[name].to_numpy()))
        acc[f"{name}__withmiss"].add(log3(df[name].to_numpy()))

    # ---------- the NEW timestamp features ----------
    rt = df["review_time"].to_numpy(dtype=np.int64)          # epoch ms, SHOW time
    cid = df["card_id"].to_numpy(dtype=np.float64)
    did = pd.to_numeric(df["deck_id"], errors="coerce").to_numpy(dtype=np.float64)
    pid = pd.to_numeric(df["preset_id"], errors="coerce").to_numpy(dtype=np.float64)
    nid = pd.to_numeric(df["note_id"], errors="coerce").to_numpy(dtype=np.float64)

    cov["rows"] += len(df)
    cov["card_ts"] += int(np.sum(cid >= TIMESTAMP_MIN_MS))
    cov["note_ts"] += int(np.nansum(nid >= TIMESTAMP_MIN_MS))
    cov["deck_ts"] += int(np.nansum(did >= TIMESTAMP_MIN_MS))
    cov["preset_ts"] += int(np.nansum(pid >= TIMESTAMP_MIN_MS))

    # 1. seconds since ANY review (upgrades the integer-day #10 to sub-day resolution)
    gap = np.diff(rt) / 1000.0
    gap = np.maximum(gap, 0.0)
    acc["t_since_any_review"].add(log_t(gap))

    # 2. user tenure = seconds since the user's first-ever review
    acc["user_tenure"].add(log_t((rt - rt[0]) / 1000.0))

    # 3. creation -> first review, a per-CARD property broadcast to the card's rows
    first_rt = df.groupby("card_id")["review_time"].transform("first").to_numpy(dtype=np.int64)
    ok = cid >= TIMESTAMP_MIN_MS
    c2f = (first_rt[ok] - cid[ok]) / 1000.0
    acc["creation_to_first_review"].note_raw(c2f)
    acc["creation_to_first_review"].add(signed_log(c2f))
    acc["creation_to_first_review__poslog"].add(log_t(np.maximum(c2f, 0.0)))

    # 4/5. deck-derived ages (99.5% coverage per FUTURE_FEATURES.md)
    okd = (did >= TIMESTAMP_MIN_MS) & (cid >= TIMESTAMP_MIN_MS)
    if okd.any():
        cmd_ = (cid[okd] - did[okd]) / 1000.0
        acc["card_minus_deck_creation"].note_raw(cmd_)
        acc["card_minus_deck_creation"].add(signed_log(cmd_))
        acc["deck_age_at_review"].add(log_t(np.maximum((rt[okd] - did[okd]) / 1000.0, 0.0)))

    # 6. preset age -- defined for ~1 row in 14; kept to size the low-value add-on honestly
    okp = pid >= TIMESTAMP_MIN_MS
    if okp.any():
        acc["preset_age_at_review"].add(
            log_t(np.maximum((rt[okp] - pid[okp]) / 1000.0, 0.0)))

    # 7. creation-batch size: how many of the user's cards were created near this one
    card_ids = np.sort(c["card_id"].to_numpy(dtype=np.float64))
    card_ids = card_ids[card_ids >= TIMESTAMP_MIN_MS]
    if card_ids.size:
        for label, win_ms in (("1min", 60_000.0), ("1h", 3_600_000.0), ("1d", 86_400_000.0)):
            lo = np.searchsorted(card_ids, cid[ok] - win_ms, side="left")
            hi = np.searchsorted(card_ids, cid[ok] + win_ms, side="right")
            acc[f"creation_batch_{label}"].add(log3(hi - lo))
        lo1h = np.searchsorted(card_ids, cid[ok] - 3_600_000.0, side="left")
        pos = np.searchsorted(card_ids, cid[ok], side="left") - lo1h
        acc["creation_batch_pos_1h"].add(log3(pos))

    # 8. deck depth in the A::B::C tree (needs no new export -- see FUTURE_FEATURES.md)
    if len(d):
        depths = deck_depths(d)
        dep = np.array([depths.get(x, 0) for x in df["deck_id"].fillna(-1)], dtype=np.float64)
        acc["deck_depth"].add(dep)

    # 9. time-of-day concentration. No constant needed (sin/cos are bounded), but the mean
    # resultant length R says whether "deviation from the user's usual hour" can carry signal at
    # all: R ~ 0 would mean users review uniformly round the clock and the feature is dead.
    theta = (rt % 86_400_000) / 86_400_000.0 * 2 * np.pi
    S, C = float(np.sin(theta).sum()), float(np.cos(theta).sum())
    circ["R"].append(math.hypot(S, C) / len(theta))
    return len(df)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=300)
    ap.add_argument("--out", default="optimization/feature_stats_id.json")
    args = ap.parse_args()

    if not DATA.exists():
        raise SystemExit(f"dataset not found: {DATA}")

    from collections import defaultdict
    acc = defaultdict(Acc)
    cov = defaultdict(int)
    circ = {"R": []}

    uids = user_ids(args.users)
    print(f"sampling {len(uids)} users from the TRAIN half {TRAIN_USER_LO}-{TRAIN_USER_HI} "
          f"(stride {uids[1] - uids[0] if len(uids) > 1 else 0})", flush=True)
    t0 = time.time()
    rows = 0
    done = 0
    for i, uid in enumerate(uids):
        try:
            rows += process_user(uid, acc, cov, circ)
            done += 1
        except Exception as e:                       # a missing/odd user must not kill the sweep
            print(f"  user {uid}: SKIPPED ({type(e).__name__}: {e})", flush=True)
        if (i + 1) % 25 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(uids)} users, {rows:,} reviews, {el/60:.1f} min "
                  f"(eta {el/(i+1)*(len(uids)-i-1)/60:.1f} min)", flush=True)

    print(f"\n{done} users, {rows:,} reviews, {(time.time()-t0)/60:.1f} min\n")

    # ---- the self-check ----
    print("=" * 78)
    print("SELF-CHECK -- recomputed vs the hardcoded constants in data_processing.py")
    print("  (a mismatch means THIS SCRIPT is wrong, not upstream)")
    print("=" * 78)
    print(f"{'constant':<32} {'upstream':>16} {'present-only':>16} {'missing->0':>16}")
    worst = 0.0
    for name, (um, us) in UPSTREAM.items():
        p, w = acc[f"{name}__present"], acc[f"{name}__withmiss"]
        print(f"{name+'_mean':<32} {um:>16.4f} {p.mean:>16.4f} {w.mean:>16.4f}")
        print(f"{name+'_std':<32} {us:>16.4f} {p.std:>16.4f} {w.std:>16.4f}")
        worst = max(worst, min(abs(p.mean - um), abs(w.mean - um)) / max(abs(um), 1e-9))
    print(f"\nworst relative mean error (best convention per constant): {worst*100:.1f}%")

    # ---- the new constants ----
    print("\n" + "=" * 78)
    print("NEW CONSTANTS -- paste into data_processing.py STATISTICS")
    print("=" * 78)
    new_names = [
        "t_since_any_review", "user_tenure", "creation_to_first_review",
        "creation_to_first_review__poslog", "card_minus_deck_creation",
        "deck_age_at_review", "preset_age_at_review", "creation_batch_1min",
        "creation_batch_1h", "creation_batch_1d", "creation_batch_pos_1h", "deck_depth",
    ]
    for n in new_names:
        a = acc[n]
        if a.n == 0:
            print(f'    # "{n}": NO DATA')
            continue
        neg = f"  (raw negative on {a.n_neg/a.n*100:.1f}%)" if a.n_neg else ""
        print(f'    "{n}_mean": {a.mean:.4f},\n    "{n}_std": {a.std:.4f},'
              f'   # n={a.n:,} range [{a.lo:.2f}, {a.hi:.2f}]{neg}')

    print("\n" + "=" * 78)
    print("COVERAGE (share of REVIEW ROWS whose id is a real epoch-ms stamp, not the "
          "sentinel/NaN)")
    print("=" * 78)
    for k in ("card_ts", "note_ts", "deck_ts", "preset_ts"):
        print(f"  {k:<12} {cov[k]/max(cov['rows'],1)*100:6.2f}%   ({cov[k]:,} / {cov['rows']:,})")

    R = np.array(circ["R"])
    print(f"\ntime-of-day concentration R over {len(R)} users: mean {R.mean():.3f}, "
          f"median {np.median(R):.3f}, p10 {np.percentile(R,10):.3f}, "
          f"p90 {np.percentile(R,90):.3f}")
    print("  (R=0 uniform round the clock -> the 'usual hour' feature would be dead; "
          "R=1 all reviews at one instant)")

    out = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": str(DATA),
        "users_sampled": done,
        "user_range": [TRAIN_USER_LO, TRAIN_USER_HI],
        "reviews": rows,
        "selfcheck": {n: {"upstream_mean": m, "upstream_std": s,
                          "present_mean": round(acc[f'{n}__present'].mean, 4),
                          "present_std": round(acc[f'{n}__present'].std, 4),
                          "withmiss_mean": round(acc[f'{n}__withmiss'].mean, 4),
                          "withmiss_std": round(acc[f'{n}__withmiss'].std, 4)}
                      for n, (m, s) in UPSTREAM.items()},
        "new": {n: acc[n].as_dict() for n in new_names if acc[n].n},
        "coverage_rows": dict(cov),
        "tod_R": {"mean": float(R.mean()), "median": float(np.median(R))} if len(R) else None,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
