"""Screen for the SPACING-EFFECT MONOTONICITY proposal -- does the champion actually violate it?

The proposal (PROPOSALS.md, curve-shape family) is to penalise the model when predicted stability
DECREASES after a successful review, on the grounds that this is a real SRS structural fact that
nothing in the architecture imposes. True -- the GRU head gives monotone-in-t and convex-in-t by
construction, but nothing constrains behaviour across REVIEW COUNT.

But an unimposed constraint is only worth a 5.5 h run if the model actually breaks it. This is the
same question the decay-floor probe asked ("is the bound binding?") and answered with a no, killing
three proposals for ~20 minutes of CPU.

METHOD. Run the champion through the deploy RNN path on whole user histories, CPU only. After each
review the model stores that card's forgetting curve; `predict_func` evaluates it at ANY horizon, so
we read each curve at fixed reference horizons and compare consecutive reviews OF THE SAME CARD:

    violation  <=>  review n+1 succeeded (rating >= 2)  AND  R_{n+1}(t_ref) < R_n(t_ref)

Comparing at a FIXED t_ref is what makes this a statement about the model rather than about the
schedule -- the actual intervals differ between reviews, so comparing each curve at its own interval
would conflate "stability changed" with "the interval changed".

CONTROL: the same count after FAILED reviews (rating == 1), where a decrease is legitimate and
expected. If the failed-review rate is not clearly higher, the measurement is not sensitive and the
numbers mean nothing.
"""
import os
import sys

sys.path.insert(0, os.getcwd())  # run from the repo root

os.environ.setdefault("RWKV_ARCH_MODULE", "scratchpad/track2_a18/architecture_d80_lora4_cnd.py")
os.environ.setdefault("RWKV_INTERLEAVE", "1")
os.environ.setdefault("RWKV_GRU_HEAD", "3")
os.environ.setdefault("RWKV_PAVA_LAMBDA", "0.2")
os.environ.setdefault("RWKV_NO_AHEAD_RESIDUAL", "1")
os.environ.setdefault("RWKV_STRIP_L0_VLORA", "1")
os.environ.setdefault("RWKV_ZERO_FEATURES", "22")
os.environ.setdefault("RWKV_STATE_CLAMP_TAU", "300")
os.environ.setdefault("RWKV_STATE_CLAMP_WINDOW", "32768")
os.environ.setdefault("RWKV_STRIP_CMIX",
                      "user_id:0,user_id:1,user_id:2,preset_id:0,preset_id:1,preset_id:2,"
                      "deck_id:1,deck_id:2,card_id:1")
os.environ.setdefault("RWKV_CHAMP_CKPT", "scratchpad/iter45_kddecay/i45_d_10935.pth")

from pathlib import Path
import numpy as np
import pandas as pd
import torch

torch.set_num_threads(4)
import rwkv.run_as_rnn as rnn_mod

DATA = Path("../anki-revlogs-10k")
DAY = 86400.0
HORIZONS = {"1d": DAY, "7d": 7 * DAY, "30d": 30 * DAY, "180d": 180 * DAY}
USERS = [int(u) for u in sys.argv[1:]] or [107, 136, 156]


def load_user_df(user_id):
    df = pd.read_parquet(DATA / "revlogs" / f"{user_id=}")
    df["review_th"] = range(1, df.shape[0] + 1)
    dc = pd.read_parquet(DATA / "cards", filters=[("user_id", "=", user_id)])
    dc.drop(columns=["user_id"], inplace=True)
    dd = pd.read_parquet(DATA / "decks", filters=[("user_id", "=", user_id)])
    dd.drop(columns=["user_id", "parent_id"], inplace=True)
    df = df.merge(dc, on="card_id", how="left", validate="many_to_one")
    df = df.merge(dd, on="deck_id", how="left", validate="many_to_one")
    df["review_th"] = range(1, df.shape[0] + 1)
    return df


def main():
    tot = {h: {"succ": 0, "succ_viol": 0, "fail": 0, "fail_viol": 0,
               "drop": [], "gain": [], "delta": []} for h in HORIZONS}
    # SANITY: predicted retention at a fixed horizon must TREND UP over a card's life. If it does
    # not, the probe is measuring the wrong quantity and no violation rate from it means anything.
    trend = {h: {"up": 0, "down": 0, "first": [], "last": []} for h in HORIZONS}
    by_rating = {r: {"n": 0, "viol": 0} for r in (1, 2, 3, 4)}
    first_R, last_R = {}, {}

    for uid in USERS:
        torch.manual_seed(uid)
        df = load_user_df(uid)
        srs = rnn_mod.RNNProcess(path=os.environ["RWKV_CHAMP_CKPT"],
                                 device=torch.device("cpu"), dtype=torch.float32)
        prev_R = {}          # card_id -> {horizon: R after that card's previous review}
        n_pairs = 0
        for i, row in df.iterrows():
            cid = row["card_id"]
            curve = srs.process_row(row)
            R = {h: float(srs.predict_func(curve, t)) for h, t in HORIZONS.items()}
            rat = int(row["rating"])
            if cid not in first_R:
                first_R[cid] = R
            last_R[cid] = R
            if cid in prev_R:
                n_pairs += 1
                success = rat >= 2
                by_rating[rat]["n"] += 1
                if R["30d"] < prev_R[cid]["30d"]:
                    by_rating[rat]["viol"] += 1
                for h in HORIZONS:
                    d = R[h] - prev_R[cid][h]
                    b = tot[h]
                    b["delta"].append(d)
                    if success:
                        b["succ"] += 1
                        if d < 0:
                            b["succ_viol"] += 1
                            b["drop"].append(-d)
                        else:
                            b["gain"].append(d)
                    else:
                        b["fail"] += 1
                        if d < 0:
                            b["fail_viol"] += 1
            prev_R[cid] = R
        for cid in first_R:
            for h in HORIZONS:
                trend[h]["first"].append(first_R[cid][h])
                trend[h]["last"].append(last_R[cid][h])
                if last_R[cid][h] > first_R[cid][h]:
                    trend[h]["up"] += 1
                else:
                    trend[h]["down"] += 1
        first_R.clear(); last_R.clear()
        print(f"user {uid}: {len(df):,} reviews, {n_pairs:,} consecutive same-card pairs", flush=True)

    print("\n=== does the champion violate stability monotonicity across review count? ===")
    print(f"{'horizon':>8} {'after SUCCESS':>26} {'after FAILURE (control)':>26} {'median drop':>12}")
    for h in HORIZONS:
        b = tot[h]
        sv = 100.0 * b["succ_viol"] / max(b["succ"], 1)
        fv = 100.0 * b["fail_viol"] / max(b["fail"], 1)
        md = float(np.median(b["drop"])) if b["drop"] else 0.0
        mg = float(np.median(b["gain"])) if b["gain"] else 0.0
        print(f"{h:>8} {b['succ_viol']:>9,}/{b['succ']:<9,} ={sv:5.1f}% "
              f"{b['fail_viol']:>9,}/{b['fail']:<9,} ={fv:5.1f}%   {md:.5f} (median gain {mg:.5f})")

    print("")
    print("=== SANITY: does R at a fixed horizon trend up over a card's life? ===")
    for h in HORIZONS:
        t = trend[h]
        n = t["up"] + t["down"]
        print(f"  {h:>5}: cards ending HIGHER than they started {100.0*t['up']/max(n,1):5.1f}%   "
              f"mean first {np.mean(t['first']):.4f} -> mean last {np.mean(t['last']):.4f}")
    print("")
    print("=== per-step mean change (all pairs) ===")
    for h in HORIZONS:
        d = np.array(tot[h]["delta"])
        print(f"  {h:>5}: mean {d.mean():+.5f}  median {np.median(d):+.5f}  "
              f"p10 {np.percentile(d,10):+.4f}  p90 {np.percentile(d,90):+.4f}")
    print("")
    print("=== violation rate at 30d, split by the button pressed ===")
    for r, name in ((1, "Again"), (2, "Hard"), (3, "Good"), (4, "Easy")):
        b = by_rating[r]
        print(f"  {name:<6} {b['viol']:>7,}/{b['n']:<7,} = {100.0*b['viol']/max(b['n'],1):5.1f}%")


if __name__ == "__main__":
    main()
