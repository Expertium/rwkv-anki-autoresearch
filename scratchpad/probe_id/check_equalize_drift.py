"""Does the -id dataset change the EQUALIZE selection (i.e. the `size` gate)?

WHY: `label_filter_db` stores, per user, the review_ths that count in the benchmark. The selection
in find_equalize_test_reviews.process() is POSITIONAL -- TimeSeriesSplit(n_splits=5) over the frame
that survives create_features' outlier/non-continuity filtering. So it is not enough that day_offset
barely moves: if the filter drops a DIFFERENT NUMBER of rows on -id, every split boundary shifts and
the user's whole equalized set changes. `size` is acceptance gate #1 ("IDENTICAL to champion; any
change = a pipeline bug"), so this has to be known before the rebuild, not after.

Runs create_features on both datasets for the same users and compares surviving row count and the
exact selected review_th list. Read-only, CPU-only, seconds per user.

    .venv/Scripts/python.exe scratchpad/probe_id/check_equalize_drift.py 486 1 2 3
"""
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import Config, create_parser  # noqa: E402
from features import create_features  # noqa: E402

PUB = Path(r"C:\Users\Andrew\anki-revlogs-10k")
IDS = Path(r"C:\Users\Andrew\anki-revlogs-10k-id")

parser = create_parser()
args, _ = parser.parse_known_args([])
config = Config(args)
# ⚠ ALGO is "FSRS-6", NOT "RWKV" -- rwkv/find_equalize_5k_h1.toml:8. The vendored create_features
# rejects "RWKV" outright, so getting this wrong makes every user error out.
config.model_name = "FSRS-6"
config.include_short_term = True      # SHORT = true in the same toml
config.use_secs_intervals = True      # SECS  = true


def selection(path, user_id, drop_review_time=True):
    df = pd.read_parquet(path / "revlogs" / f"{user_id=}")
    raw_rows = len(df)
    if drop_review_time and "review_time" in df.columns:
        # -id carries an extra column; create_features is shared upstream code that never saw it.
        df = df.drop(columns=["review_time"])
    try:
        df = create_features(df.copy(), config=config)
    except ValueError as err:
        return raw_rows, None, f"ValueError: {err}"
    if len(df) == 0:
        return raw_rows, None, "empty after filtering"
    ths = []
    for _, (_, test_index) in enumerate(TimeSeriesSplit(n_splits=5).split(df)):
        ths.extend(int(df.iloc[i]["review_th"]) for i in test_index)
    return raw_rows, (len(df), ths), None


def main():
    users = [int(x) for x in sys.argv[1:]] or [486, 1, 2, 3, 17, 101]
    print(f"{'user':>6} {'raw rows':>10} {'kept pub':>9} {'kept id':>8} {'size pub':>9} "
          f"{'size id':>8}  verdict")
    n_diff = 0
    n_err = 0
    for u in users:
        rp, sp, ep = selection(PUB, u)
        ri, si, ei = selection(IDS, u)
        if ep or ei:
            # ⚠ An error must NEVER be reportable as "no drift" -- the first version of this script
            # `continue`d without counting, and printed "0 of 6 users would change" when it had in
            # fact compared nothing at all. Silence is not success.
            n_err += 1
            print(f"{u:>6} {rp:>10} {'-':>9} {'-':>8} {'-':>9} {'-':>8}  ERR pub={ep} id={ei}")
            continue
        kp, tp = sp
        ki, ti = si
        same = (tp == ti)
        n_diff += (not same)
        verdict = "IDENTICAL" if same else (
            f"** DIFFERS ** ({sum(a != b for a, b in zip(tp, ti))} th mismatches)"
            if len(tp) == len(ti) else "** DIFFERS ** (different length)")
        print(f"{u:>6} {rp:>10} {kp:>9} {ki:>8} {len(tp):>9} {len(ti):>8}  {verdict}")
    n_cmp = len(users) - n_err
    print(f"\n{n_diff} of {n_cmp} COMPARED users would change their equalized set under -id"
          f"  ({n_err} errored)")
    if n_err and n_cmp == 0:
        raise SystemExit("nothing was compared -- fix the errors above before reading any verdict")
    if n_diff:
        print("=> label_filter_db MUST be rebuilt, and the `size` gate will legitimately move.")
    else:
        print("=> on this sample the equalize selection is stable; still confirm on the "
              "NaN-affected users before relying on it.")


if __name__ == "__main__":
    main()
