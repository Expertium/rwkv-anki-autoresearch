"""ONE deploy-RNN pass on the realcyc checkpoint (gen-5 layout, -id parquet), MANY screens.

Records, per predicted row (the prediction made at row k for row k+1 of the same card):
  p       = curve at t_{k+1} from the state after row k, duration IN   (what training scores on real rows)
  p0      = the same from a NON-committing probe of row k with scaled_duration = 0
            (what deploy / the rectified metric serves: the pressed-grade probe)
  y       = 1 if rating_{k+1} >= 2
  t       = elapsed_seconds of row k+1 (end-to-start, the -id convention)
  rat     = rating of row k+1 (the ORDINAL label the curve never sees)
  prat    = rating of row k (the state-setting press)
  n       = 0-based index of row k+1 within its card
  u       = user id
Screens fed (PROPOSALS ranked queue 2026-09-04):
  rank 2 duration dropout : by-user mean BCE(p0) - BCE(p)  = the current-duration half of the
                            rectification penalty on THIS checkpoint (kill if < +0.0004)
  rank 1 ordinal target   : among successes with t >= 1 d, Hard share must FALL and Easy share
                            RISE with the decile of logit p (|rho| >= 0.8); AUC(Easy vs Hard)
                            > 0.75 => already separated, dead
  rank 10 recalibration   : the by-user calibration gap (mean y - mean p), train-range only
TRAIN-RANGE USERS ONLY (the VAL half is reserved for gate decisions).

Usage: screen_pass.py [--limit N] [users...]      (REUSE=0 forces a re-walk)
"""
import os
import sys

sys.path.insert(0, os.getcwd())
# realcyc's env (run_realcyc.cmd), minus the training-only vars
_ENV = {
    "RWKV_ARCH_MODULE": "scratchpad/track2_a18/architecture_d80_lora4_cnd.py",
    "RWKV_INTERLEAVE": "1", "RWKV_GRU_HEAD": "3", "RWKV_PAVA_LAMBDA": "0.2",
    "RWKV_NO_AHEAD_RESIDUAL": "1", "RWKV_STRIP_L0_VLORA": "1",
    "RWKV_STATE_CLAMP_TAU": "300", "RWKV_STATE_CLAMP_WINDOW": "32768",
    "RWKV_STRIP_CMIX": "user_id:0,user_id:1,user_id:2,preset_id:0,preset_id:1,preset_id:2,deck_id:1,deck_id:2,card_id:1",
    "RWKV_ID_FEATURES": "1", "RWKV_REAL_CYCLES": "1", "RWKV_ZERO_FEATURES": "",
    "RWKV_CHAMP_CKPT": "scratchpad/realcyc/rc_d_10935.pth",
}
for _k, _v in _ENV.items():
    os.environ[_k] = _v

from pathlib import Path

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(4)
import rwkv.run_as_rnn as rnn_mod
from rwkv.run_as_rnn import scale_duration, scale_state

DATA = Path(r"C:\Users\Andrew\anki-revlogs-10k-id")
DAY = 86400.0
OUT = Path("scratchpad/proposals_2026-09-04/screen_records.npz")

args = [a for a in sys.argv[1:] if not a.startswith("--")]
LIMIT = 0
if "--limit" in sys.argv:
    LIMIT = int(sys.argv[sys.argv.index("--limit") + 1])
    args = [a for a in args if a != str(LIMIT)]
USERS = [int(u) for u in args] or [107, 136, 156, 178, 203, 1207, 2207, 3207, 4207, 4807]


from rwkv.data_processing import get_rwkv_data  # noqa: E402  (the -id pipeline's own frame builder)


def collect():
    """Walk each user's get_rwkv_data frame (the same feature values the LMDB holds) through the
    deploy RNN, feeding frame records to `run()` directly as scratchpad/workload/rwkv_arm.py does.
    The prediction made AT row k is scored on row k's own label triple (the card's next review),
    so no per-card bookkeeping is needed. The probe is evaluated BEFORE the committing step, on the
    same pre-state, with skip=True."""
    P, P0, Y, T, RAT, PRAT, N, U = [], [], [], [], [], [], [], []
    for uid in USERS:
        torch.manual_seed(uid)
        df = get_rwkv_data(DATA, uid).sort_values("review_th", kind="stable").reset_index(drop=True)
        df["nth"] = df.groupby("card_id").cumcount()
        if LIMIT:
            df = df.iloc[:LIMIT]
        srs = rnn_mod.RNNProcess(path=os.environ["RWKV_CHAMP_CKPT"],
                                 device=torch.device("cpu"), dtype=torch.float32)
        with torch.inference_mode():
            for row in df.to_dict("records"):
                has = int(row["has_label"]) == 1
                if has:
                    r0 = dict(row)
                    r0["scaled_duration"] = 0.0
                    curve0, _ = srs.run(r0, skip=True)      # the pressed-grade probe, non-committing
                curve, _ = srs.run(row, skip=False)          # the real step (commits the states)
                if has:
                    t = float(row["label_elapsed_seconds"])
                    P.append(float(srs.predict_func(curve, t)))
                    P0.append(float(srs.predict_func(curve0, t)))
                    Y.append(float(row["label_y"]))
                    T.append(t)
                    RAT.append(int(row["label_rating"]))
                    PRAT.append(int(row["rating"]))
                    N.append(int(row["nth"]))
                    U.append(uid)
        print(f"user {uid}: {len(df):,} reviews -> {len(P):,} records so far", flush=True)
    return {k: np.array(v) for k, v in dict(p=P, p0=P0, y=Y, t=T, rat=RAT, prat=PRAT, n=N, u=U).items()}


def bce(p, y):
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(pc) + (1 - y) * np.log(1 - pc))


def spearman(a, b):
    ra = pd.Series(a).rank().values
    rb = pd.Series(b).rank().values
    return float(np.corrcoef(ra, rb)[0, 1])


def auc(pos, neg):
    # probability a random positive scores above a random negative (rank-based)
    x = np.concatenate([pos, neg])
    r = pd.Series(x).rank().values
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    if OUT.exists() and os.environ.get("REUSE", "1") == "1" and not LIMIT:
        d = dict(np.load(OUT))
        print(f"reusing {OUT} ({d['p'].size:,} records)")
    else:
        d = collect()
        if not LIMIT:
            np.savez_compressed(OUT, **d)
            print(f"wrote {OUT} ({d['p'].size:,} records)")
    p, p0, y, t, rat, prat, n, u = (d[k] for k in ("p", "p0", "y", "t", "rat", "prat", "n", "u"))
    users = sorted(set(u.tolist()))

    print("\n=== RANK 2 (duration dropout): the current-duration half of the rectification penalty ===")
    per = []
    for uid in users:
        m = u == uid
        per.append((uid, bce(p[m], y[m]).mean(), bce(p0[m], y[m]).mean(), int(m.sum())))
    for uid, a, b, k in per:
        print(f"  user {uid:>5}  n={k:>7,}  BCE(dur in) {a:.6f}  BCE(dur 0) {b:.6f}  cost {b - a:+.6f}")
    costs = np.array([b - a for _, a, b, _ in per])
    print(f"  BY-USER MEAN cost of zeroing the current duration: {costs.mean():+.6f}   (kill line +0.0004;"
          f" iter 31 measured +0.001451 on the published set)   min {costs.min():+.6f} max {costs.max():+.6f}")

    print("\n=== RANK 1 (ordinal target): is the k+1 grade a monotone function of the model's own R? ===")
    succ = (y == 1) & (t >= DAY)
    lp = np.log(np.clip(p[succ], 1e-6, 1 - 1e-6) / (1 - np.clip(p[succ], 1e-6, 1 - 1e-6)))
    g = rat[succ]
    dec = pd.qcut(lp, 10, labels=False, duplicates="drop")
    hard, good, easy, dm = [], [], [], []
    for q in sorted(set(dec.tolist())):
        m = dec == q
        hard.append((g[m] == 2).mean()); good.append((g[m] == 3).mean()); easy.append((g[m] == 4).mean())
        dm.append(float(lp[m].mean()))
    print(f"  successes with t >= 1 d: {int(succ.sum()):,}   grade shares Hard/Good/Easy ="
          f" {(g == 2).mean():.3f}/{(g == 3).mean():.3f}/{(g == 4).mean():.3f}")
    print("  decile of logit p  |  Hard share  Good share  Easy share")
    for i in range(len(dm)):
        print(f"    {i:>2}  (lp~{dm[i]:+.2f})  |  {hard[i]:.3f}       {good[i]:.3f}       {easy[i]:.3f}")
    rh, re_ = spearman(dm, hard), spearman(dm, easy)
    print(f"  Spearman(decile, Hard share) = {rh:+.3f} (want <= -0.8)   Spearman(decile, Easy share) = {re_:+.3f} (want >= +0.8)")
    a_eh = auc(lp[g == 4], lp[g == 2]); a_gh = auc(lp[g == 3], lp[g == 2])
    print(f"  AUC(logit p; Easy vs Hard) = {a_eh:.3f} (dead if > 0.75; expected 0.58-0.68)   AUC(Good vs Hard) = {a_gh:.3f}")

    print("\n=== RANK 10 (recalibration): by-user calibration gap, train-range ===")
    gaps = np.array([y[u == uid].mean() - p[u == uid].mean() for uid in users])
    print(f"  mean(y) - mean(p): by-user mean {gaps.mean():+.5f}  min {gaps.min():+.5f}  max {gaps.max():+.5f}"
          f"   (kill if |mean| < 0.001)")
    gaps0 = np.array([y[u == uid].mean() - p0[u == uid].mean() for uid in users])
    print(f"  same for the duration-0 probe:   {gaps0.mean():+.5f}")


if __name__ == "__main__":
    main()
