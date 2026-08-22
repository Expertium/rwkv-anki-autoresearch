"""The report: workload ratio at nominal DR, calibration of both arms, and the ratio again
at MATCHED REALIZED retention.

The three tables answer three different questions and only the third one is an efficiency
claim.

  1. NOMINAL   -- at the same stated desired retention, how do the two workloads compare?
                  This is what was asked for. It is only an efficiency comparison if both
                  models deliver the retention they promise.

  2. CALIBRATION -- do they? Each arm's own scheduling curve is evaluated at the interval
                  that actually happened, and compared with what actually happened. An
                  overconfident model asks for longer intervals and looks cheaper while
                  quietly under-delivering retention, so this table decides whether table 1
                  can be read as efficiency at all.

  3. MATCHED   -- workload at equal REALIZED retention. Each arm's nominal DR axis is
                  mapped through its own empirical calibration curve, so "80% realized" is
                  looked up on each side at whatever nominal DR that arm needs to actually
                  achieve 80%. This is the comparison that is not confounded by
                  over/under-confidence.

Usage: .venv/Scripts/python.exe scratchpad/workload/analyze.py [--fsrs-prefix fsrs] ...
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from combine import DR_LEVELS, DR_COLS, daily_workload  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"


def load_pairs(fsrs_prefix):
    pairs = {}
    for f in sorted(OUT.glob("%s_u*.parquet" % fsrs_prefix)):
        uid = int(f.stem.split("_u")[1])
        r = OUT / ("rwkv_u%d.parquet" % uid)
        if not r.exists():
            continue
        a, b = pd.read_parquet(f), pd.read_parquet(r)
        if len(a) != len(b) or not (a["review_th"].to_numpy() == b["review_th"].to_numpy()).all():
            print("  SKIP u%d: arms disagree on rows" % uid)
            continue
        pairs[uid] = (a, b)
    return pairs


DATA = Path(r"C:\Users\Andrew\anki-revlogs-10k")


def queue_mask(uid, review_th, queue):
    """Which scheduling decisions count.

    'review' (default) keeps only the windows during which the card sat in Anki's REVIEW
    queue -- row j counts iff the card's NEXT review has state == 2.

    WHY THIS IS THE DEFAULT. Without it the metric measures the learning queue: sum(1/L)
    weights short intervals enormously, and 64-98% of the unfiltered workload comes from
    intervals the 1-day floor had to rewrite, at EVERY desired-retention level. Those are
    same-day learning and relearning steps, which in Anki are driven by fixed learning
    steps, not by FSRS or RWKV. Including them makes the ratio a measurement of the floor
    rather than of the two algorithms.

    The state column is the card's state BEFORE its review (0 new, 1 learning, 2 review,
    3 relearning, 4 filtered), which is why the test is on the NEXT review's state: that
    is the state the card is in during the window whose workload we are attributing.

    Reading it from the raw parquet needs no rebuild of the arms' outputs: get_rwkv_data
    assigns review_th = 1..N in file order before any sort, so raw row i is review_th i+1.
    Asserted below rather than assumed.
    """
    raw = pd.read_parquet(DATA / "revlogs" / ("user_id=%d" % uid))
    assert len(raw) == len(review_th), (
        "u%d: raw revlogs has %d rows, arm output %d -- review_th is no longer the raw "
        "row index and the state join would be silently misaligned" % (uid, len(raw), len(review_th)))
    assert (review_th == np.arange(1, len(raw) + 1)).all(), "review_th is not 1..N in order"
    if queue == "all":
        return np.ones(len(raw), dtype=bool)
    state = raw["state"].to_numpy()
    card = raw["card_id"].to_numpy()
    nxt_state = np.full(len(raw), -1, dtype=np.int64)
    order = np.argsort(card, kind="stable")   # stable -> within a card, review_th order
    cs, ss = card[order], state[order]
    tmp = np.full(len(raw), -1, dtype=np.int64)
    tmp[:-1] = np.where(cs[:-1] == cs[1:], ss[1:], -1)
    nxt_state[order] = tmp
    return nxt_state == 2


def actual_gap_days(uid, review_th):
    """The interval the user's REAL schedule used, per review: the gap to that card's next
    review, in days. Row j's gap is row j+1's elapsed_days.

    This is the reference arm for an ABSOLUTE check that neither LogLoss nor the ratio can
    provide. The user's observed retention is a fact; feed that same number to a model as
    its desired retention and the workload it asks for should land near the workload the
    user actually carried. A model whose inverted intervals are systematically too short
    will demand several times the real review load at the retention the user demonstrably
    achieved -- and no amount of good LogLoss at the horizons that happened would reveal
    that, because inverting the curve to a fixed DR is an extrapolation.
    """
    raw = pd.read_parquet(DATA / "revlogs" / ("user_id=%d" % uid))
    assert len(raw) == len(review_th)
    ed = raw["elapsed_days"].to_numpy(dtype=np.float64)
    card = raw["card_id"].to_numpy()
    order = np.argsort(card, kind="stable")
    cs = card[order]
    tmp = np.full(len(raw), np.nan)
    tmp[:-1] = np.where(cs[:-1] == cs[1:], ed[order][1:], np.nan)
    gap = np.full(len(raw), np.nan)
    gap[order] = tmp
    return gap


def workload_curve(df, n_days, floor_days, mode, min_cards, row_mask=None):
    """mean W over the days that clear min_cards, one value per DR level, plus the active
    card count (identical for both arms, so they average over exactly the same days) and
    the part of W contributed by intervals the floor rewrote."""
    ws, fl, cs = [], [], None
    for col in DR_COLS:
        w, c = daily_workload(df, col, n_days, floor_days, mode, row_mask=row_mask)
        f, _ = daily_workload(df, col, n_days, floor_days, mode, only_floored=True,
                              row_mask=row_mask)
        ws.append(w)
        fl.append(f)
        cs = c
    return np.array(ws), np.array(fl), cs


def calibration_curve(pred, y, n_bins):
    """Empirical realized retention as a function of predicted retention.

    Equal-COUNT bins, not equal-width: predictions pile up near 1, and equal-width bins
    would put almost every row in one bin and then report a calibration curve built from
    a handful of rows at the low end.
    """
    o = np.argsort(pred)
    p, yy = pred[o], y[o]
    edges = np.linspace(0, len(p), n_bins + 1).astype(int)
    xs, ys, ns = [], [], []
    for i in range(n_bins):
        s = slice(edges[i], edges[i + 1])
        if edges[i + 1] - edges[i] < 30:
            continue
        xs.append(float(p[s].mean()))
        ys.append(float(yy[s].mean()))
        ns.append(int(edges[i + 1] - edges[i]))
    return np.array(xs), np.array(ys), np.array(ns)


def sign_test_p(ratios):
    """Two-sided exact sign test that the per-user ratio is centred on 1.

    A sign test rather than a t-test on log-ratio: the per-user spread here runs from 0.20
    to 2.59, is heavy-tailed, and n is 25 -- so the direction of each user's effect is the
    part of the data that can be trusted, and the magnitudes are not.
    """
    from math import comb
    r = np.asarray(ratios, dtype=float)
    r = r[np.isfinite(r) & (r > 0)]
    n = int((r != 1).sum())
    if n == 0:
        return 1.0
    k = int((r > 1).sum())
    k = min(k, n - k)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2.0 ** n)
    return float(min(1.0, 2.0 * tail))


def realized_at(nominal, xs, ys):
    """Realized retention this arm actually delivers when it claims `nominal`."""
    return float(np.interp(nominal, xs, ys))


def nominal_for_realized(target, xs, ys):
    """Inverse: the nominal DR this arm must be set to in order to realize `target`.
    ys is non-decreasing in principle; enforce it so the inverse is well defined, and
    return NaN outside the range the data actually covers rather than extrapolating."""
    ym = np.maximum.accumulate(ys)
    if target < ym[0] or target > ym[-1]:
        return float("nan")
    return float(np.interp(target, ym, xs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fsrs-prefix", default="fsrs")
    ap.add_argument("--floor-days", type=float, default=1.0)
    ap.add_argument("--mode", default="alive", choices=["alive", "persist"])
    ap.add_argument("--min-cards", type=int, default=20)
    ap.add_argument("--queue", default="review", choices=["review", "all"],
                    help="'review' counts only windows where the card sat in Anki's review "
                         "queue; 'all' counts learning and relearning steps too, where the "
                         "1-day floor dominates and the algorithms barely differ")
    ap.add_argument("--bins", type=int, default=20)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    pairs = load_pairs(args.fsrs_prefix)
    if not pairs:
        print("no matched user pairs in %s" % OUT)
        return
    print("=" * 78)
    print("WORKLOAD-EFFICIENCY REPLAY   FSRS-7 (%s) vs RWKV-Curve (iter-53 champion)"
          % args.fsrs_prefix)
    print("users %d   floor %.3g d   activity=%s   queue=%s   min_cards=%d"
          % (len(pairs), args.floor_days, args.mode, args.queue, args.min_cards))
    print("=" * 78)

    ratios, wf_all, wr_all, wa_all, floor_share = [], [], [], [], []
    daily_med = []
    cal = {"fsrs": [[], [], []], "rwkv": [[], [], []]}
    per_user = {}
    for uid, (a, b) in sorted(pairs.items()):
        n_days = int(max(a["day_offset"].max(), b["day_offset"].max())) + 1
        qm = queue_mask(uid, a["review_th"].to_numpy(), args.queue)
        WF, FF, cf = workload_curve(a, n_days, args.floor_days, args.mode, args.min_cards, qm)
        WR, FR, cr = workload_curve(b, n_days, args.floor_days, args.mode, args.min_cards, qm)
        # the reference arm: the schedule the user actually followed
        gap = actual_gap_days(uid, a["review_th"].to_numpy())
        act = a[["card_id", "day_offset"]].copy()
        act["ivl_act"] = np.where(np.isnan(gap), 1e9, gap)
        WA, _ = daily_workload(act, "ivl_act", n_days, args.floor_days, args.mode,
                               row_mask=qm & ~np.isnan(gap))
        assert np.allclose(cf, cr), "u%d: arms see different active-card counts" % uid
        sel = (cf >= args.min_cards) & (WF > 0).all(0) & (WR > 0).all(0)
        if not sel.any():
            continue
        mf, mr = WF[:, sel].mean(1), WR[:, sel].mean(1)
        # ⚠ POOLED, not the mean of daily ratios. Andrew's formulation is
        # average(W_FSRS/W_RWKV), but a per-DAY ratio has an unstable denominator: on days
        # when RWKV's active cards all sit on long intervals, W_RWKV is near zero and the
        # day's ratio explodes. Measured: user 6711's mean-of-daily-ratios at DR=80% is
        # 386.7 while its total-workload ratio is 1.52, and that one user dragged the
        # cross-user MEAN to 16.9 against a median of 1.05.
        # sum(W_FSRS)/sum(W_RWKV) over the same days is the quantity that actually means
        # something -- total reviews one algorithm would cost over the whole history
        # against the other -- and it is what "how much more work" asks for.
        ratios.append(mf / mr)
        daily_med.append(np.median(WF[:, sel] / WR[:, sel], axis=1))
        wf_all.append(mf)
        wr_all.append(mr)
        wa_all.append(float(WA[sel].mean()))
        floor_share.append((FF[:, sel].mean(1) / mf, FR[:, sel].mean(1) / mr))
        for key, d in (("fsrs", a), ("rwkv", b)):
            m = d["has_next"].to_numpy().astype(bool)
            cal[key][0].append(d["pred"].to_numpy()[m])
            cal[key][1].append(d["y"].to_numpy()[m])
            cal[key][2].append(gap[m])
        per_user[uid] = {"ratio": ratios[-1].tolist(), "w_fsrs": mf.tolist(),
                         "w_rwkv": mr.tolist(), "w_actual": wa_all[-1],
                         "n_days": int(sel.sum()), "n_reviews": int(len(a))}

    R = np.array(ratios)
    print("\n1. WORKLOAD RATIO AT NOMINAL DESIRED RETENTION  (reviews/day, per user)")
    print("   per-user ratio = total FSRS reviews / total RWKV reviews over that user's")
    print("   whole history, at that DR. Summarised across users below.")
    print("   %-6s %9s %9s %11s %9s %8s"
          % ("DR", "median", "geomean", "p25..p75", "frac>1", "p"))
    print("   " + "-" * 72)
    for k, dr in enumerate(DR_LEVELS):
        c = R[:, k]
        print("   %-6s %9.3f %9.3f %5.2f..%-5.2f %9.2f %8.3f"
              % ("%d%%" % round(dr * 100), np.median(c),
                 np.exp(np.log(c).mean()), np.percentile(c, 25), np.percentile(c, 75),
                 (c > 1).mean(), sign_test_p(c)))
    print("   ratio > 1 = FSRS-7 costs MORE reviews/day, i.e. RWKV-Curve is more efficient.")
    FS = np.array([f[0] for f in floor_share])
    RS = np.array([f[1] for f in floor_share])
    print("")
    print("   share of that workload coming from intervals the 1-day floor REWROTE")
    print("   (high = the row measures the floor, not the algorithms)")
    print("   %-6s %12s %12s" % ("DR", "FSRS-7", "RWKV-Curve"))
    print("   " + "-" * 32)
    for k, dr in enumerate(DR_LEVELS):
        print("   %-6s %11.1f%% %11.1f%%"
              % ("%d%%" % round(dr * 100), 100 * np.median(FS[:, k]), 100 * np.median(RS[:, k])))

    WA = np.array(wa_all)
    WFm0 = np.array(wf_all)
    WRm0 = np.array(wr_all)
    print("")
    print("   ABSOLUTE CHECK: each arm's workload vs the load the user ACTUALLY carried")
    print("   (median over users of W_model / W_actual; 1.0 = the model asks for exactly")
    print("   the real review load at that DR)")
    print("   %-6s %12s %12s" % ("DR", "FSRS/actual", "RWKV/actual"))
    print("   " + "-" * 32)
    for k, dr in enumerate(DR_LEVELS):
        print("   %-6s %12.2f %12.2f"
              % ("%d%%" % round(dr * 100), np.median(WFm0[:, k] / WA),
                 np.median(WRm0[:, k] / WA)))
    print("   mean actual review-queue load: %.2f reviews/day" % WA.mean())

    # Does the ratio depend on collection size? This is what decides whether the expensive
    # large-user phase is worth running: if the ratio is flat in size, phase 1's small and
    # medium users already answer the question and 1.28M more reviews buy nothing.
    if len(per_user) >= 6:
        uids = sorted(per_user)
        sizes = np.array([per_user[u]["n_reviews"] for u in uids], dtype=np.float64)
        print("\n   ratio vs collection size (Spearman rho over %d users)" % len(uids))
        print("   %-6s %8s   %s" % ("DR", "rho", "reads as"))
        print("   " + "-" * 52)
        rank_s = np.argsort(np.argsort(sizes))
        for k, dr in enumerate(DR_LEVELS):
            v = np.array([per_user[u]["ratio"][k] for u in uids])
            rank_v = np.argsort(np.argsort(v))
            rho = float(np.corrcoef(rank_s, rank_v)[0, 1])
            note = ("flat" if abs(rho) < 0.4 else
                    "RWKV relatively better on BIG collections" if rho > 0 else
                    "RWKV relatively better on SMALL collections")
            print("   %-6s %+8.3f   %s" % ("%d%%" % round(dr * 100), rho, note))

    print("\n2. CALIBRATION ON THE SAME ROWS  (each arm's scheduling curve at the real interval)")
    curves = {}
    print("   %-6s %10s %10s %10s %10s" % ("arm", "logloss", "mean pred", "mean obs", "bias"))
    print("   " + "-" * 52)
    for key in ("fsrs", "rwkv"):
        p = np.clip(np.concatenate(cal[key][0]), 1e-6, 1 - 1e-6)
        y = np.concatenate(cal[key][1])
        ll = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
        xs, ys, ns = calibration_curve(p, y, args.bins)
        curves[key] = (xs, ys)
        print("   %-6s %10.4f %10.4f %10.4f %+10.4f"
              % (key.upper(), ll, p.mean(), y.mean(), p.mean() - y.mean()))
    print("   bias > 0 = the arm claims more retention than it delivers (overconfident),")
    print("   so its intervals at a given nominal DR are longer than that DR justifies.")

    print("")
    print("   CALIBRATION BY HORIZON  (does either curve decay too fast with t?)")
    print("   Bias = mean predicted minus mean observed, on the rows whose ACTUAL interval")
    print("   fell in each bucket. A curve that over-decays reads increasingly NEGATIVE as")
    print("   the horizon grows: it under-predicts recall exactly where it has to")
    print("   extrapolate, and inverting to a fixed DR depends on that extrapolation.")
    EDGES = [0, 1, 3, 7, 21, 60, 180, 1e9]
    NAMES = ["<1d", "1-3d", "3-7d", "7-21d", "21-60d", "60-180d", ">180d"]
    print("   %-9s %9s %9s %11s %11s" % ("horizon", "n", "observed", "FSRS bias", "RWKV bias"))
    print("   " + "-" * 54)
    gh = np.concatenate(cal["fsrs"][2])
    pf = np.concatenate(cal["fsrs"][0])
    yf = np.concatenate(cal["fsrs"][1])
    pr = np.concatenate(cal["rwkv"][0])
    for i, name in enumerate(NAMES):
        m = (gh >= EDGES[i]) & (gh < EDGES[i + 1]) & ~np.isnan(gh)
        if m.sum() < 200:
            continue
        print("   %-9s %9d %9.4f %+11.4f %+11.4f"
              % (name, int(m.sum()), yf[m].mean(),
                 pf[m].mean() - yf[m].mean(), pr[m].mean() - yf[m].mean()))

    print("\n   realized retention at each nominal DR (from the empirical calibration curve)")
    print("   %-6s %12s %12s" % ("nominal", "FSRS-7", "RWKV-Curve"))
    print("   " + "-" * 32)
    for dr in DR_LEVELS:
        print("   %-6s %12.4f %12.4f" % (
            "%d%%" % round(dr * 100),
            realized_at(dr, *curves["fsrs"]), realized_at(dr, *curves["rwkv"])))

    print("\n3. WORKLOAD RATIO AT MATCHED REALIZED RETENTION")
    print("   Each arm is set to whatever nominal DR its own calibration curve says it")
    print("   needs to actually deliver the target; workload is then read off that arm's")
    print("   own W(DR) curve by log-log interpolation. NaN = the target is outside the")
    print("   range that arm's predictions actually cover.")
    print("   Summarised PER USER, like table 1, so the two are comparable. The DR mapping")
    print("   comes from the pooled calibration curve (a per-user curve is far too noisy to")
    print("   invert); each user's own W(DR) is then read at the mapped nominal DR.")
    print("   %-8s %9s %9s %10s %10s %9s %8s"
          % ("realized", "nom FSRS", "nom RWKV", "median", "geomean", "frac>1", "p"))
    print("   " + "-" * 68)
    WFu = np.array(wf_all)      # (users, DR)
    WRu = np.array(wr_all)
    x = np.array(DR_LEVELS)
    matched = []
    for target in DR_LEVELS:
        nf = nominal_for_realized(target, *curves["fsrs"])
        nr = nominal_for_realized(target, *curves["rwkv"])
        if not (np.isfinite(nf) and np.isfinite(nr)):
            print("   %-8s %9s %9s %10s %10s %9s %8s"
                  % ("%d%%" % round(target * 100), "-", "-", "-", "-", "-", "-"))
            continue
        # W(DR) is smooth and strongly curved, so interpolate log W against DR.
        wf = np.exp([np.interp(nf, x[::-1], np.log(WFu[i])[::-1]) for i in range(len(WFu))])
        wr = np.exp([np.interp(nr, x[::-1], np.log(WRu[i])[::-1]) for i in range(len(WRu))])
        rr = wf / wr
        p = sign_test_p(rr)
        matched.append({"realized": target, "nom_fsrs": nf, "nom_rwkv": nr,
                        "median": float(np.median(rr)),
                        "geomean": float(np.exp(np.log(rr).mean())),
                        "frac_gt1": float((rr > 1).mean()), "p": p})
        print("   %-8s %9.4f %9.4f %10.3f %10.3f %9.2f %8.3f"
              % ("%d%%" % round(target * 100), nf, nr, np.median(rr),
                 np.exp(np.log(rr).mean()), (rr > 1).mean(), p))
    print("   p = two-sided sign test that the per-user ratio is centred on 1.")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            "config": vars(args), "dr_levels": DR_LEVELS,
            "nominal_ratio_median": np.median(R, 0).tolist(),
            "nominal_ratio_mean": R.mean(0).tolist(),
            "calibration": {k: {"pred": v[0].tolist(), "obs": v[1].tolist()}
                            for k, v in curves.items()},
            "matched": matched, "per_user": per_user,
        }, indent=1), encoding="utf-8")
        print("\nwrote %s" % args.json_out)


if __name__ == "__main__":
    main()
