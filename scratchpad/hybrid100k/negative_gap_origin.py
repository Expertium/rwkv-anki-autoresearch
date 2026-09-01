"""Why does end-to-start look 1000x more negative on the published set than in PR #2?

PR open-spaced-repetition/anki-revlogs-dataset-builder#2 measures end-to-start going negative on
**2 rows in 2,306,229** (worst -2.0 s), computed PER CARD from real start/end timestamps.
`INTERVAL_HANDOFF.md` (mine) reports **0.559% of same-day rows** needing the clamp, computed as
`elapsed_seconds - duration(k)` on the PUBLISHED set. Those cannot both describe the same thing.

THE HYPOTHESIS: upstream builds `elapsed_seconds` by diffing the WHOLE FRAME in protobuf order and
only overwrites `state == 0` rows with the -1 sentinel (`build_parquet.py`). Protobuf order is
per-card blocks, so the first row of each block gets a CROSS-CARD diff -- the gap to a different
card's review, which can be milliseconds. Subtracting this review's own multi-second duration from
a millisecond gap yields a large negative that has nothing to do with the interval definition.

If that is right, my 0.559% is an artifact of the published column, PR #2's number is the real
property of the correction, and the handoff needs fixing.

Usage: python scratchpad/hybrid100k/negative_gap_origin.py [n_users]
CPU-only, 1 thread.
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

ROOT = r"C:/Users/Andrew/anki-revlogs-10k/revlogs"


def main():
    n_users = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    files = sorted(glob.glob(os.path.join(ROOT, "**", "*.parquet"), recursive=True))
    step = max(1, len(files) // n_users)
    files = files[::step][:n_users]

    tot = neg = neg_crosscard = neg_samecard = 0
    same_card_rows = 0
    worst_same = 0.0
    for f in files:
        df = pd.read_parquet(f, columns=["card_id", "elapsed_seconds", "duration"])
        if df.empty:
            continue
        es = df["elapsed_seconds"].to_numpy().astype("float64")
        dur = df["duration"].to_numpy().astype("float64") / 1000.0
        card = df["card_id"].to_numpy()
        # "the previous ROW in frame order is the same card" -- which is what upstream's whole-frame
        # .diff() silently assumes and what a per-card computation would require.
        prev_same = np.empty(len(df), dtype=bool)
        prev_same[0] = False
        prev_same[1:] = card[1:] == card[:-1]

        real = es >= 0                       # not the -1 first-review sentinel
        corrected = es - dur
        bad = real & (corrected < 0)

        tot += int(real.sum())
        neg += int(bad.sum())
        neg_crosscard += int((bad & ~prev_same).sum())
        neg_samecard += int((bad & prev_same).sum())
        same_card_rows += int((real & prev_same).sum())
        if (bad & prev_same).any():
            worst_same = min(worst_same, float(corrected[bad & prev_same].min()))

    print(f"users {len(files)}   rows with a real (non-sentinel) gap: {tot:,}")
    print(f"  of those, previous ROW is the SAME card: {same_card_rows:,} "
          f"({100.0 * same_card_rows / tot:.1f}%)")
    print()
    print(f"end-to-start goes NEGATIVE on {neg:,} rows ({100.0 * neg / tot:.4f}% of real gaps)")
    print(f"  ... previous row is a DIFFERENT card (cross-card diff): {neg_crosscard:,} "
          f"({100.0 * neg_crosscard / max(neg, 1):.1f}% of the negatives)")
    print(f"  ... previous row is the SAME card (a genuine overlap) : {neg_samecard:,} "
          f"({100.0 * neg_samecard / max(neg, 1):.1f}%)")
    if neg_samecard:
        print(f"      worst same-card value: {worst_same:.1f} s")
    print()
    if neg and neg_crosscard / neg > 0.5:
        print("=> CONFIRMED: most negatives come from upstream's CROSS-CARD diff, not from the")
        print("   interval correction. PR #2's per-card figure is the honest one, and the")
        print("   handoff's 0.559% overstates the clamp rate for a per-card computation.")
    else:
        print("=> NOT confirmed: the negatives are genuine same-card overlaps. The discrepancy")
        print("   with PR #2 has another cause and must be found before either number is quoted.")
    rate = 100.0 * neg_samecard / max(same_card_rows, 1)
    print(f"\nper-card negative rate (the quantity PR #2 reports): {rate:.6f}% "
          f"= {neg_samecard} of {same_card_rows:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
