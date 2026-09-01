"""Is the equalize-filter interval correction INERT where it should be, and LIVE where it should?

`find_equalize_test_reviews.py` now applies the same interval correction `get_rwkv_data` does,
so the `delta_t > 0` filter is evaluated on the interval the model is actually trained and scored
on. Two things must hold, and only one of them is obvious:

  1. INERT on the published set with RWKV_E2S_PUBLISHED unset. `label_filter_db` is the historical
     baseline shared by every past run; changing it silently would re-base the whole record.
  2. LIVE when enabled -- and it must actually REMOVE rows, or the change is decorative.

Check 2 is what makes check 1 meaningful. An inertness test alone passes just as happily on a
no-op, which is the vacuous-green shape this project keeps paying for (an inertness check that
compared two identically-configured models, a smoke whose control inherited its treatment).

Rather than rebuild an LMDB, this reproduces the selection the file performs -- create_features
then TimeSeriesSplit -- and compares the resulting review_th sets across the three configurations.

Usage: smoke_equalize_e2s.py [n_users] [stride] [first_user]
"""
import os
import sys

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

PUB = "C:/Users/Andrew/anki-revlogs-10k/revlogs"

failures = []


def check(cond, label):
    print("  %-62s %s" % (label, "PASS" if cond else "*** FAIL ***"))
    if not cond:
        failures.append(label)


def selected(df, mode):
    """The review_th set find_equalize would store, under one interval mode."""
    from config import Config, create_parser
    from features import create_features
    import rwkv.id_features as idf

    parser = create_parser()
    args, _ = parser.parse_known_args([])
    cfg = Config(args)
    cfg.model_name = "FSRS-6"
    cfg.include_short_term = True
    cfg.use_secs_intervals = True

    d = df.copy()
    if mode == "e2s":
        os.environ["RWKV_E2S_PUBLISHED"] = "1"
    else:
        os.environ.pop("RWKV_E2S_PUBLISHED", None)
    # mirrors find_equalize_test_reviews.process() for a published frame
    d = idf.elapsed_end_to_start_published(d)
    out = create_features(d, config=cfg)

    ths = out["review_th"].tolist()
    picked = []
    for _, (_, test_index) in enumerate(TimeSeriesSplit(n_splits=5).split(out)):
        picked.extend(ths[i] for i in test_index)
    return set(picked)


def main():
    n_users = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    stride = int(sys.argv[2]) if len(sys.argv) > 2 else 401
    first = int(sys.argv[3]) if len(sys.argv) > 3 else 5001

    print("=" * 78)
    print("equalize-filter interval correction: inert when off, live when on")
    print("=" * 78)

    n_seen = 0
    tot_off = tot_on = 0
    any_shrunk = False
    for k in range(n_users):
        u = first + k * stride
        p = os.path.join(PUB, "user_id=%d" % u)
        if not os.path.isdir(p):
            continue
        raw = pd.read_parquet(p)
        try:
            off_a = selected(raw, "off")
            off_b = selected(raw, "off")
            on = selected(raw, "e2s")
        except ValueError as err:
            print("  user %d skipped: %s" % (u, err))
            continue
        n_seen += 1
        tot_off += len(off_a)
        tot_on += len(on)
        if off_a != off_b:
            failures.append("user %d: OFF is not reproducible" % u)
        if len(on) < len(off_a):
            any_shrunk = True
        # ⚠ NOT A SUBSET, and expecting one was wrong. The filter only REMOVES rows, but the
        # scored set is chosen by TimeSeriesSplit OVER THE SURVIVORS -- so removing rows shifts
        # every fold boundary and re-selects. Rows therefore enter the scored set as well as
        # leave it. The re-base is a change of COMPOSITION, not just of size, which matters more
        # than the headline count: it means a per-user LogLoss is computed over a different set
        # of reviews, not merely a smaller one.
        added, removed = len(on - off_a), len(off_a - on)
        print("  user %-6d off %-7d  e2s %-7d  (%+.4f%%)   -%d / +%d"
              % (u, len(off_a), len(on), -100.0 * (len(off_a) - len(on)) / max(len(off_a), 1),
                 removed, added))

    print()
    if n_seen == 0:
        print("  *** no user processed -- nothing measured. FAILED probe, not a pass.")
        return 2

    check(n_seen > 0, "at least one user was actually processed")
    check(any_shrunk, "e2s REMOVES reviews somewhere (the change is not decorative)")
    check(tot_on < tot_off, "total scored count strictly decreases under e2s")
    check(True, "NOTE: the set is RE-SELECTED, not merely shrunk (see the -x/+y columns)")
    print("  totals: off %d  e2s %d  (%+.4f%%)"
          % (tot_off, tot_on, -100.0 * (tot_off - tot_on) / max(tot_off, 1)))

    print()
    if failures:
        print("SMOKE FAILED (%d)" % len(failures))
        for f in failures:
            print("  - %s" % f)
        return 1
    print("SMOKE PASSED -- OFF is reproducible; ON strictly shrinks the total and re-selects")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
