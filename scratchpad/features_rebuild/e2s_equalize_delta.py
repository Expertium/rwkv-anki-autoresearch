"""How much does the equalize set move if the label filter sees END-TO-START intervals?

THE GAP. `find_equalize_test_reviews.py` reads the raw parquet and calls `create_features`
DIRECTLY -- it never goes through `data_processing.get_rwkv_data`, which is where our
end-to-start correction lives. So `label_filter_db` is built from END-TO-END `elapsed_seconds`
even now that training and eval use end-to-start.

That matters because with `SECS = true` the srs-benchmark filter is not interval-independent:

    features/base.py:127   delta_t_secs = elapsed_seconds / 86400
    features/base.py:227   delta_t := delta_t_secs          (when use_secs_intervals)
    features/base.py:284   return df[df["delta_t"] > 0]

so `delta_t > 0` drops rows whose gap floors to zero -- and which rows those ARE depends on the
interval definition. Andrew 2026-09-01: "We should have delta_t > 0 though, to make our
methodology closer to that of srs-benchmark." We DO have it; what we do not have is an
end-to-start-aware version of it.

WHAT THIS MEASURES, using the REAL filter rather than a re-implementation: `create_features` is
run twice on the same user -- once on the raw frame, once with `elapsed_end_to_start_published`
applied first -- and the two surviving `review_th` sets are compared. It then reports whether the
dropped rows are EASIER than average, because that is what decides whether the change moves mean
LogLoss on its own (srs-benchmark's dropped rows were 2.7x easier: 6.09% vs 16.14% failure).

Usage: e2s_equalize_delta.py [n_users] [stride] [first_user]
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

ROOT = "C:/Users/Andrew/anki-revlogs-10k/revlogs"


def surviving(df, e2s):
    """review_th values that survive create_features, with or without the e2s correction."""
    from config import Config, create_parser
    from features import create_features
    import rwkv.id_features as idf

    parser = create_parser()
    args, _ = parser.parse_known_args([])
    cfg = Config(args)
    cfg.model_name = "FSRS-6"   # what find_equalize_*.toml sets via ALGO
    cfg.include_short_term = True
    cfg.use_secs_intervals = True

    d = df.copy()
    if e2s:
        os.environ["RWKV_E2S_PUBLISHED"] = "1"
        d = idf.elapsed_end_to_start_published(d)
    else:
        os.environ.pop("RWKV_E2S_PUBLISHED", None)
    out = create_features(d, config=cfg)
    return out


def main():
    n_users = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    stride = int(sys.argv[2]) if len(sys.argv) > 2 else 311
    first = int(sys.argv[3]) if len(sys.argv) > 3 else 5001

    tot_e2e = tot_e2s = 0
    drop_rows = []
    kept_fail = kept_n = 0

    for k in range(n_users):
        u = first + k * stride
        p = os.path.join(ROOT, "user_id=%d" % u)
        if not os.path.isdir(p):
            continue
        raw = pd.read_parquet(p)
        try:
            a = surviving(raw, e2s=False)
            b = surviving(raw, e2s=True)
        except ValueError as err:
            print("  user %d skipped: %s" % (u, err))
            continue

        sa, sb = set(a["review_th"]), set(b["review_th"])
        tot_e2e += len(sa)
        tot_e2s += len(sb)
        lost = sa - sb
        gained = sb - sa
        if lost:
            sub = a[a["review_th"].isin(lost)]
            drop_rows.append(sub)
        kept_fail += int((a["y"] == 0).sum())
        kept_n += len(a)
        print("  user %-6d e2e %-7d  e2s %-7d   lost %-5d gained %-4d  (%.4f%% of e2e)"
              % (u, len(sa), len(sb), len(lost), len(gained),
                 100.0 * len(lost) / max(len(sa), 1)))

    print()
    print("TOTAL   e2e %d   e2s %d   net %+d  (%.4f%%)"
          % (tot_e2e, tot_e2s, tot_e2s - tot_e2e,
             100.0 * (tot_e2s - tot_e2e) / max(tot_e2e, 1)))

    if drop_rows:
        d = pd.concat(drop_rows)
        fail_dropped = 100.0 * (d["y"] == 0).mean()
        fail_all = 100.0 * kept_fail / max(kept_n, 1)
        print("dropped rows: n=%d   failure rate %.2f%%   vs %.2f%% overall  (%.2fx %s)"
              % (len(d), fail_dropped, fail_all,
                 (fail_all / fail_dropped) if fail_dropped else float("inf"),
                 "EASIER" if fail_dropped < fail_all else "harder"))
        # Binomial 1-sigma on the dropped set, so a direction is not read off noise. The 8-user
        # sample said 1.24x EASIER and the 24-user sample said 0.86x HARDER -- the same statistic,
        # opposite sign, which is why this now prints an interval instead of a verdict.
        n = len(d)
        se = 100.0 * ((fail_dropped / 100.0) * (1 - fail_dropped / 100.0) / max(n, 1)) ** 0.5
        z = (fail_dropped - fail_all) / se if se else 0.0
        print("   dropped-set failure rate SE %.2fpp  ->  %+.1f sigma from the overall rate" % (se, z))
        print()
        if abs(z) < 2.0:
            print("=> Difficulty of the dropped rows is NOT distinguishable from average here, so")
            print("   the size change alone gives no predictable direction for mean LogLoss.")
        elif fail_dropped < fail_all:
            print("=> The dropped rows are EASIER than average, so removing them RAISES mean")
            print("   LogLoss by itself. That is a re-base, not a regression.")
        else:
            print("=> The dropped rows are HARDER than average, so removing them LOWERS mean")
            print("   LogLoss by itself. That is a re-base, not an improvement.")
    else:
        print("no rows dropped in this sample")

    # ⚠ NOT A RESULT IF NOTHING RAN. The first version of this script printed "no rows dropped"
    # after skipping every user on a config error -- the exact false-green shape it exists to
    # avoid producing elsewhere.
    if tot_e2e == 0:
        print()
        print("*** NO USER WAS PROCESSED -- nothing was measured. This is a FAILED probe, not a")
        print("    null result. Check the model name / data path before reading anything into it.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
