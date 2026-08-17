"""SECOND spacing-effect screen: is a monotonicity violation an ERROR, or correct inference?

The first screen (monotonicity_probe.py) established that the constraint BINDS -- 39.7% of Good and
38.3% of Easy transitions lower predicted retention at a fixed horizon. That is necessary but NOT
sufficient: a constraint the model violates is only worth imposing if the violations are WRONG. If
the model drops R after a Good press and the card really does then perform worse, the drop is
correct inference and a regulariser that forbids it can only cost accuracy.

THE TEST, and why it is not the obvious one. The obvious test -- compare logloss on reviews
following a violation vs following a non-violation -- is CONFOUNDED, and by a mechanism the first
screen already measured: violations concentrate on hard cards (lapse rate rises 1.9% -> 46.4% with
review count, rho 0.4867), and hard cards carry higher loss whatever the model does. That comparison
would report a difference under the null.

CALIBRATION is the confound-free version. For every review we have the model's own predicted
retention p and the realised outcome y. Ask, WITHIN each group, whether p matches y:

    violation is an ERROR      =>  the drop was spurious, p is too LOW,  so mean(y) - mean(p) > 0
    violation is INFERENCE     =>  p already reflects the true difficulty, so the gap is ~0

Card difficulty cannot fake this: a hard card has a low p AND a low y, and the gap stays near zero.
Only a systematically MIS-SET state moves the gap. The non-violating Good/Easy group is the control
-- what the model's calibration gap looks like when it does not violate -- and the binned table is
the strict form, comparing outcomes only between rows at MATCHED predicted probability.

Indexing (worth stating, since an off-by-one would invert the conclusion): a violation detected at
review k of a card is a property of the state written AFTER k, so it is charged to the prediction of
review k+1, which is the first prediction that state produces.

Instrument: identical to monotonicity_probe.py, whose predict_func path reproduces the certified
reference_iter41 py_pred_ahead at exactly 0.000e+00 (verify_probe.py).

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
T_REF = 30 * DAY          # the horizon the first screen reported its button split at
USERS = [int(u) for u in sys.argv[1:]] or [107, 136, 156, 178, 203]


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


def summarize(name, rows):
    if not rows:
        return None
    p = np.array([r[0] for r in rows], dtype=np.float64)
    y = np.array([r[1] for r in rows], dtype=np.float64)
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    ll = -(y * np.log(pc) + (1 - y) * np.log(1 - pc))
    gap = y.mean() - p.mean()
    # SE of the gap: y is Bernoulli, p is a fixed prediction, so the sampling noise is y's
    se = y.std(ddof=1) / np.sqrt(y.size)
    print(f"  {name:<34} n={y.size:>7,}  mean p={p.mean():.4f}  mean y={y.mean():.4f}  "
          f"gap={gap:+.4f} +/-{1.96*se:.4f}  logloss={ll.mean():.4f}")
    return {"p": p, "y": y, "gap": gap, "se": se}


def binned(name, d, edges):
    if d is None:
        return
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (d["p"] >= lo) & (d["p"] < hi)
        if m.sum() < 30:
            out.append("        --   ")
            continue
        out.append(f"{d['y'][m].mean() - d['p'][m].mean():+.4f}({m.sum():>5,})")
    print(f"  {name:<34} " + " ".join(out))


def main():
    # groups keyed by what happened on the transition that WROTE the state making this prediction
    groups = {"goodeasy_violation": [], "goodeasy_ok": [], "hard": [], "again": [], "first": []}

    for uid in USERS:
        torch.manual_seed(uid)
        df = load_user_df(uid)
        srs = rnn_mod.RNNProcess(path=os.environ["RWKV_CHAMP_CKPT"],
                                 device=torch.device("cpu"), dtype=torch.float32)
        prev = {}   # card_id -> (curve, R_at_T_REF, group_label_for_the_next_prediction)
        for _, row in df.iterrows():
            cid = row["card_id"]
            rat = int(row["rating"])
            if cid in prev:
                curve_prev, r_prev, label = prev[cid]
                p = float(srs.predict_func(curve_prev, row["elapsed_seconds"]))
                groups[label].append((p, 1.0 if rat >= 2 else 0.0))
            else:
                curve_prev, r_prev = None, None
            curve = srs.process_row(row)
            r_now = float(srs.predict_func(curve, T_REF))
            if r_prev is None:
                label = "first"                       # no previous curve to compare against
            elif rat >= 3:
                label = "goodeasy_violation" if r_now < r_prev else "goodeasy_ok"
            elif rat == 2:
                label = "hard"
            else:
                label = "again"
            prev[cid] = (curve, r_now, label)
        print(f"user {uid}: {len(df):,} reviews", flush=True)

    print("\n=== is a monotonicity violation an ERROR or correct inference? ===")
    print("(gap = mean(outcome) - mean(predicted). >0 means the model UNDER-predicts, i.e. the drop")
    print(" it applied was too big. ~0 means the prediction already matches reality.)")
    dv = summarize("after Good/Easy WITH violation", groups["goodeasy_violation"])
    do = summarize("after Good/Easy, no violation", groups["goodeasy_ok"])
    summarize("after Hard", groups["hard"])
    summarize("after Again", groups["again"])
    summarize("first review of a card", groups["first"])

    if dv is not None and do is not None:
        diff = dv["gap"] - do["gap"]
        se = np.sqrt(dv["se"] ** 2 + do["se"] ** 2)
        print(f"\n  violation-minus-control calibration gap: {diff:+.4f} +/- {1.96*se:.4f} (95% CI)")
        print("  -> the lever has a real target ONLY if this is clearly POSITIVE.")

    print("\n=== the strict form: calibration gap WITHIN matched predicted-probability bins ===")
    edges = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0]
    print("  " + " " * 34 + " ".join(f"{lo:.2f}-{hi:.2f}".center(14)
                                     for lo, hi in zip(edges[:-1], edges[1:])))
    binned("after Good/Easy WITH violation", dv, edges)
    binned("after Good/Easy, no violation", do, edges)


if __name__ == "__main__":
    main()
