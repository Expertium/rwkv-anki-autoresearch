"""Did the end-to-start transform do EXACTLY what it claims, in the stored db?

`smoke_e2s_published.py` verifies the transform as a FUNCTION. This verifies the RESULT: that the
LMDB the training run will actually read differs from its end-to-end twin in precisely the columns
the change can touch, and in the right direction.

THE PREDICTION, made from the column list rather than from the output:

  MUST CHANGE   2 scaled_elapsed_seconds            (the interval itself)
                3,4 elapsed_seconds_sin/cos         (derived from it)
                5 scaled_elapsed_seconds_cumulative (a per-card cumsum of it)
                6,7 ..._cumulative_sin/cos          (derived from that)

  MUST NOT      0 scaled_elapsed_days               ⚠ THE LOAD-BEARING ONE. `elapsed_days` is a
                                                    calendar-day index and is deliberately left
                                                    alone: `is_first_review` IS `elapsed_days ==
                                                    -1`, so touching it re-labels a mid-card
                                                    review as a first review and poisons the label
                                                    machinery.
                1 scaled_elapsed_days_cumulative
                8 scaled_duration                   (the raw duration is unchanged; only the
                                                    interval it is subtracted FROM moves)
                9-23 ratings, is_nan flags, counters, state, is_query

  DIRECTION     end-to-start <= end-to-end always, since it subtracts a non-negative duration.
                The feature is log-scaled and monotone, so the ordering survives scaling.

A column that moves when it should not, or fails to move when it should, means the transform is not
the one documented -- and no downstream number would reveal it.

⚠ The two dbs also differ in note_id (Bug C), but ids live in SEPARATE keys, not in card_features.
`note_id_is_nan` (13) is in this tensor and must NOT move: whether metadata was missing does not
depend on the interval.

Usage: verify_e2s_columns.py <e2e_db> <e2s_db> [n_keys]
"""
import io
import sys

import lmdb
import torch

from rwkv.data_processing import CARD_FEATURE_COLUMNS

MUST_CHANGE = {2, 3, 4, 5, 6, 7}


def main():
    a, b = sys.argv[1], sys.argv[2]
    want = int(sys.argv[3]) if len(sys.argv) > 3 else 25

    ea = lmdb.open(a, readonly=True, lock=False, subdir=True)
    eb = lmdb.open(b, readonly=True, lock=False, subdir=True)

    with ea.begin() as t:
        cur = t.cursor()
        keys = []
        if cur.first():
            while len(keys) < want:
                k = cur.key()
                if k.endswith(b"card_features"):
                    keys.append(k)
                if not cur.next():
                    break

    def load(env, key):
        with env.begin() as t:
            raw = t.get(key)
        return torch.load(io.BytesIO(raw), weights_only=True, map_location="cpu") if raw else None

    ncol = len(CARD_FEATURE_COLUMNS)
    changed = torch.zeros(ncol, dtype=torch.long)
    total = torch.zeros(ncol, dtype=torch.long)
    wrong_dir = torch.zeros(ncol, dtype=torch.long)

    for k in keys:
        va, vb = load(ea, k), load(eb, k)
        if va is None or vb is None:
            continue
        if va.shape != vb.shape:
            print("SHAPE MISMATCH on %s: %s vs %s" % (k.decode(), va.shape, vb.shape))
            return 3
        d = va != vb
        changed += d.sum(0).long()
        total += d.shape[0]
        # e2s must never be LARGER than e2e on the interval column
        wrong_dir += ((vb > va) & d).sum(0).long()

    print("%-4s %-38s %12s %9s   %s" % ("col", "name", "changed", "pct", "verdict"))
    problems = []
    for i, name in enumerate(CARD_FEATURE_COLUMNS):
        pct = 100.0 * changed[i].item() / max(total[i].item(), 1)
        should = i in MUST_CHANGE
        did = changed[i].item() > 0
        if should and not did:
            v = "! SHOULD HAVE CHANGED"
            problems.append("%s (%d) did not change" % (name, i))
        elif not should and did:
            v = "! SHOULD NOT HAVE CHANGED"
            problems.append("%s (%d) changed but must not" % (name, i))
        else:
            v = "ok"
        print("%-4d %-38s %12d %8.3f%%   %s" % (i, name, changed[i].item(), pct, v))

    wd = wrong_dir[2].item()
    print("\ndirection on scaled_elapsed_seconds: %d of %d changed entries are LARGER under e2s"
          % (wd, changed[2].item()))
    if wd:
        problems.append("e2s produced a LARGER interval on %d entries -- impossible, since it "
                        "subtracts a non-negative duration" % wd)

    print()
    if problems:
        for p in problems:
            print("  ! " + p)
        print("E2S_COLUMNS_FAIL")
        return 3
    print("E2S_COLUMNS_PASS -- exactly the interval-derived columns moved, and only downward.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
