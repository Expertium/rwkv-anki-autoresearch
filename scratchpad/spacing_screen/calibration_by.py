"""One probe pass, many bucketings: where is the champion actually MIS-CALIBRATED?

Generalises violation_calibration.py, which answered one question (are monotonicity violations
errors? -- no) and had to walk 90k reviews to do it. The walk is the expensive part and the
per-review record is cheap, so this records `(p, y, elapsed_seconds, nth_review_of_card, rating)`
once and slices it several ways. Adding a question later costs a slice, not another 90 minutes.

WHY CALIBRATION IS THE RIGHT LENS FOR ALL OF THEM. A loss comparison across groups is confounded by
difficulty -- hard rows cost more whatever the model does -- but a well-calibrated model has
`mean(y) == mean(p)` *within every subgroup*, however hard. So a non-zero gap in a bucket is a
directional error the model could in principle be trained out of, and a zero gap means there is
nothing there to win, no matter how much loss the bucket carries.

THE QUESTION THIS RUN IS FOR (plan rank 9, horizon reweighting of the curve loss): the proposal says
long intervals are RARE and HARD, so the curve objective is dominated by short `t` and the model
underfits the tail. Two halves, and both are checkable here:
  * RARE  -- the row counts per `t` bucket, free.
  * HARD  -- excess loss is NOT the test (long intervals are genuinely harder; that is not an error).
             The test is whether the model is systematically BIASED at long `t`.
There is a real mechanism to expect one: the GRU curve head is a 3-component mixture of
`(1+t/S)^-d`, i.e. ~9 degrees of freedom per review, so it CANNOT fit an arbitrary curve shape. If
the true forgetting curve leaves that family at long `t`, the residual shows up as a calibration
drift that reweighting could plausibly attack. If instead the gap is flat in `t`, the objective is
already unbiased across horizons and reweighting has no target -- the same verdict shape that killed
the spacing lever.

ALSO SLICED, free: calibration by review index (does the model drift as a card's history grows?) and
by rating, which is the control -- `after Hard` was the worst-calibrated group in the first run
(-0.0251) and should stay that way here.

TRAIN-RANGE USERS ONLY -- the VAL half 5001-7500 is reserved for gate decisions.
"""
import os
import sys

sys.path.insert(0, os.getcwd())  # run from the repo root

os.environ.setdefault("RWKV_ARCH_MODULE", "scratchpad/track2_a18/architecture_d80_lora4_cnd.py")
for _k, _v in (("RWKV_INTERLEAVE", "1"), ("RWKV_GRU_HEAD", "3"), ("RWKV_PAVA_LAMBDA", "0.2"),
               ("RWKV_NO_AHEAD_RESIDUAL", "1"), ("RWKV_STRIP_L0_VLORA", "1"),
               ("RWKV_ZERO_FEATURES", "22"), ("RWKV_STATE_CLAMP_TAU", "300"),
               ("RWKV_STATE_CLAMP_WINDOW", "32768"),
               ("RWKV_STRIP_CMIX", "user_id:0,user_id:1,user_id:2,preset_id:0,preset_id:1,"
                                   "preset_id:2,deck_id:1,deck_id:2,card_id:1"),
               ("RWKV_CHAMP_CKPT", "scratchpad/iter45_kddecay/i45_d_10935.pth")):
    os.environ.setdefault(_k, _v)

from pathlib import Path

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(4)
import rwkv.run_as_rnn as rnn_mod

DATA = Path("../anki-revlogs-10k")
DAY = 86400.0
USERS = [int(u) for u in sys.argv[1:]] or [107, 136, 156, 178, 203]
OUT = Path("scratchpad/spacing_screen/calib_records.npz")


def load_user_df(user_id):
    df = pd.read_parquet(DATA / "revlogs" / f"{user_id=}")
    dc = pd.read_parquet(DATA / "cards", filters=[("user_id", "=", user_id)])
    dc.drop(columns=["user_id"], inplace=True)
    dd = pd.read_parquet(DATA / "decks", filters=[("user_id", "=", user_id)])
    dd.drop(columns=["user_id", "parent_id"], inplace=True)
    df = df.merge(dc, on="card_id", how="left", validate="many_to_one")
    df = df.merge(dd, on="deck_id", how="left", validate="many_to_one")
    df["review_th"] = range(1, df.shape[0] + 1)
    return df


def collect():
    P, Y, T, N, R, U = [], [], [], [], [], []
    for uid in USERS:
        torch.manual_seed(uid)
        df = load_user_df(uid)
        srs = rnn_mod.RNNProcess(path=os.environ["RWKV_CHAMP_CKPT"],
                                 device=torch.device("cpu"), dtype=torch.float32)
        prev, nth, prat = {}, {}, {}
        for _, row in df.iterrows():
            cid = row["card_id"]
            k = nth.get(cid, 0)
            if cid in prev:
                P.append(float(srs.predict_func(prev[cid], row["elapsed_seconds"])))
                Y.append(1.0 if int(row["rating"]) >= 2 else 0.0)
                T.append(float(row["elapsed_seconds"]))
                N.append(k)                       # 0-based index of this review within its card
                # ⚠ the PREVIOUS press, not this row's. Slicing by this row's rating would
                # condition on the OUTCOME (y == rating>=2), which is degenerate -- it reports
                # y=1.0 exactly for Hard/Good/Easy and y=0 for Again, measuring nothing. The
                # state-setting press is the one that could bias the prediction being scored.
                R.append(prat.get(cid, 0))
                U.append(uid)
            prev[cid] = srs.process_row(row)
            prat[cid] = int(row["rating"])
            nth[cid] = k + 1
        print(f"user {uid}: {len(df):,} reviews -> {len(P):,} records so far", flush=True)
    return (np.array(P), np.array(Y), np.array(T), np.array(N), np.array(R),
            np.array(U))


def table(title, labels, masks, p, y, note=""):
    print(f"\n=== {title} ===")
    if note:
        print(note)
    print(f"  {'bucket':<20} {'n':>9} {'mean p':>9} {'mean y':>9} {'gap':>10} {'95% CI':>10} {'logloss':>9}")
    for lab, m in zip(labels, masks):
        if m.sum() < 50:
            print(f"  {lab:<20} {int(m.sum()):>9,}   (too few)")
            continue
        pp, yy = p[m], y[m]
        pc = np.clip(pp, 1e-6, 1 - 1e-6)
        ll = -(yy * np.log(pc) + (1 - yy) * np.log(1 - pc))
        gap = yy.mean() - pp.mean()
        ci = 1.96 * yy.std(ddof=1) / np.sqrt(yy.size)
        flag = "  <-- biased" if abs(gap) > ci else ""
        print(f"  {lab:<20} {yy.size:>9,} {pp.mean():>9.4f} {yy.mean():>9.4f} "
              f"{gap:>+10.4f} {ci:>10.4f} {ll.mean():>9.4f}{flag}")


def main():
    if OUT.exists() and os.environ.get("REUSE", "1") == "1":
        d = np.load(OUT)
        p, y, t, n, r = d["p"], d["y"], d["t"], d["n"], d["r"]
        # `u` is the marker for the 2026-08-17 fix: the user column and the previous-press
        # rating landed together, so its absence dates the whole npz.
        stale_npz = "u" not in d
        if stale_npz:
            print("  (this npz predates the user column -- delete it and set REUSE=0 to"
                  " regenerate, so recalibration_prize.py can hold out BY USER)")
        print(f"reusing {OUT} ({p.size:,} records)")
    else:
        stale_npz = False
        p, y, t, n, r, u = collect()
        np.savez_compressed(OUT, p=p, y=y, t=t, n=n, r=r, u=u)
        print(f"\nwrote {OUT} ({p.size:,} records)")

    # --- rank 9: is the model BIASED at long horizons? ---
    edges = [0, 1, 3, 7, 21, 60, 180, 1e9]
    names = ["<1d", "1-3d", "3-7d", "7-21d", "21-60d", "60-180d", ">180d"]
    td = t / DAY
    masks = [(td >= lo) & (td < hi) for lo, hi in zip(edges[:-1], edges[1:])]
    table("CALIBRATION BY HORIZON t (plan rank 9's premise)", names, masks, p, y,
          note="  RARE = the n column. HARD is NOT the test -- long gaps are genuinely harder.\n"
               "  The test is the GAP: flat in t => the objective is already unbiased across\n"
               "  horizons and horizon reweighting has no target.")
    tot = sum(int(m.sum()) for m in masks)
    long_share = 100.0 * sum(int(m.sum()) for m in masks[4:]) / max(tot, 1)
    print(f"  -> share of scored rows at t >= 21d: {long_share:.1f}%")

    # --- free slice: does calibration drift as a card accumulates history? ---
    nedges = [0, 1, 2, 4, 8, 16, 10 ** 9]
    nnames = ["1st pred", "2nd", "3rd-4th", "5th-8th", "9th-16th", "17th+"]
    table("CALIBRATION BY REVIEW INDEX", nnames,
          [(n >= lo) & (n < hi) for lo, hi in zip(nedges[:-1], nedges[1:])], p, y)

    # --- control: the first run found `after Hard` worst-calibrated; it should reappear ---
    # ⚠ REFUSE to print this from a pre-fix npz. Those cached `r` values are the CURRENT row's
    # rating, so the table would be degenerate (y == rating>=2 by definition) -- and it would look
    # like a finding, complete with "<-- biased" flags on every row. A tool that emits a known-wrong
    # table is worse than one that declines. `u` is the marker: both columns landed in the same fix.
    if stale_npz:
        print("\n=== CALIBRATION BY THE PREVIOUS PRESS: SKIPPED ===")
        print("  The cached npz predates the previous-press fix, so its rating column holds the")
        print("  CURRENT row's rating and this table would be degenerate. Delete")
        print("  calib_records.npz and re-run with REUSE=0 (~90 min CPU) to get it.")
    else:
        table("CALIBRATION BY THE PREVIOUS (state-setting) PRESS",
              ["(first)", "Again", "Hard", "Good", "Easy"],
              [r == k for k in (0, 1, 2, 3, 4)], p, y,
              note=("  The press that WROTE the state making this prediction. Slicing by the\n"
                    "  CURRENT row's rating instead would condition on the outcome\n"
                    "  (y == rating>=2) and report y=1.0000 exactly -- degenerate.\n"
                    "  Cross-check: the first screen found `after Hard` the worst-calibrated\n"
                    "  group at -0.0251, which should reappear here."))


if __name__ == "__main__":
    main()
