"""How much would an end-to-start interval definition change same-day reviews?

Andrew, 2026-08-29: should srs-benchmark get a THIRD table using the corrected interval?

THE THREE DEFINITIONS. A revlog row is written when the user ANSWERS, so `id` is the answer time
and `duration` is how long that review took. Writing show(k) = id(k) - duration(k):

    answer-to-answer  =  id(k) - id(k-1)              <- what `elapsed_seconds` stores today
    show-to-show      =  show(k) - show(k-1)          <- the `-id` dataset's naive diff
    END-TO-START      =  show(k) - id(k-1)            <- what Andrew wants: the gap during which
                                                         the memory actually decays

    end_to_start = elapsed_seconds - duration(k)

⚠ It is the CURRENT review's duration that comes off, not the previous one. Easy to get backwards:
`show-to-show` is the one that differs from answer-to-answer by duration(k-1).

★ THE FEASIBILITY POINT: `duration` and `elapsed_seconds` are BOTH in the public
`anki-revlogs-10k`. So the corrected interval is a one-line transform of data everybody already
has -- no new dataset, no HF upload, no reprocessing. That matters for whether this is worth
proposing upstream.

This measures the size of the change, because "is it worth a table" is a quantitative question:
if durations are negligible against same-day gaps, the third table equals the second one.

Usage: python scratchpad/hybrid100k/interval_def_effect.py [n_users]
CPU-only, ~1 thread.
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
    if not files:
        raise SystemExit("dataset not found at " + ROOT)
    # stride so the sample is not the first N (smallest) users
    step = max(1, len(files) // n_users)
    files = files[::step][:n_users]

    rows = []
    for f in files:
        df = pd.read_parquet(f, columns=["card_id", "elapsed_seconds", "duration",
                                         "elapsed_days"])
        # elapsed_seconds == -1 is the first-review sentinel; keep only real gaps
        df = df[df["elapsed_seconds"] >= 0]
        if df.empty:
            continue
        dur_s = df["duration"].to_numpy(dtype=np.float64) / 1000.0
        gap = df["elapsed_seconds"].to_numpy(dtype=np.float64)
        same_day = df["elapsed_days"].to_numpy() == 0
        rows.append(pd.DataFrame({"gap": gap, "dur": dur_s, "same_day": same_day}))

    d = pd.concat(rows, ignore_index=True)
    d["corrected"] = d["gap"] - d["dur"]
    d["frac"] = np.where(d["gap"] > 0, d["dur"] / d["gap"], np.nan)

    tot = len(d)
    sd = d[d["same_day"]]
    ld = d[~d["same_day"]]

    print(f"users sampled: {len(files)}   reviews with a real gap: {tot:,}")
    print(f"  same-day (elapsed_days == 0): {len(sd):,}  ({100.0 * len(sd) / tot:.1f}%)")
    print(f"  longer-interval             : {len(ld):,}  ({100.0 * len(ld) / tot:.1f}%)")

    def block(name, x):
        if x.empty:
            print(f"\n{name}: none")
            return
        fr = x["frac"].dropna()
        print(f"\n{name}  (n={len(x):,})")
        print(f"  median gap                 : {x['gap'].median():,.1f} s")
        print(f"  median review duration     : {x['dur'].median():.2f} s")
        print("  duration as a FRACTION of the gap:")
        for q in (0.10, 0.25, 0.50, 0.75, 0.90, 0.99):
            print(f"      p{int(q * 100):<3d} {fr.quantile(q) * 100:8.2f}%")
        print(f"  gap shrinks by >=10%       : {100.0 * (fr >= 0.10).mean():.1f}% of these rows")
        print(f"  gap shrinks by >=50%       : {100.0 * (fr >= 0.50).mean():.1f}% of these rows")
        neg = (x["corrected"] < 0).mean()
        print(f"  corrected gap goes NEGATIVE: {100.0 * neg:.3f}% of these rows")

    block("SAME-DAY reviews", sd)
    block("LONGER-INTERVAL reviews", ld)

    print("\n--- what this means for a third table ---")
    fr_sd = sd["frac"].dropna()
    fr_ld = ld["frac"].dropna()
    print(f"  same-day  : the interval moves by a median {fr_sd.median() * 100:.1f}%")
    print(f"  long-term : the interval moves by a median {fr_ld.median() * 100:.4f}%")
    print("  A third table can therefore only differ from the WITH-same-day table. On the")
    print("  without-same-day table the correction is numerically invisible.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
