"""CPU screen for ranked-queue rank 7 (auxiliary next-interval regression head, Caruana-style).

Question: does the trunk ALREADY encode the scheduler's next interval? The proposal's kill rule is a
ridge probe R^2 > 0.85 from the head trunk output to log(1 + label_elapsed_seconds). The deploy RNN
does not expose x_w, but the GRU curve head's own (w, S, d) triple (9 numbers, a fixed function of
x_w) is what the curve is built from, so a probe on it bounds the same question from the head's side:
if 9 numbers already give R^2 > 0.85 the head carries the interval and the aux loss is redundant.
Also reports the correlation between the probe's residual and the per-row ahead BCE (the second
kill rule: ~0 => the information the aux task would add is unrelated to ahead error).

Walks a few TRAIN-range users through realcyc's deploy RNN (same env as screen_pass.py), fits ridge
on all-but-one user, scores R^2 on the held-out user (leave-one-user-out over the set).
Usage: aux_probe.py [users...]        default: 107 136 156 203 1207 4207
"""
import os
import sys

sys.path.insert(0, os.getcwd())
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
import torch

torch.set_num_threads(4)
import rwkv.run_as_rnn as rnn_mod
from rwkv.data_processing import get_rwkv_data

DATA = Path(r"C:\Users\Andrew\anki-revlogs-10k-id")
USERS = [int(u) for u in sys.argv[1:]] or [107, 136, 156, 203, 1207, 4207]


def collect():
    X, T, Y, P, U = [], [], [], [], []
    for uid in USERS:
        torch.manual_seed(uid)
        df = get_rwkv_data(DATA, uid).sort_values("review_th", kind="stable").reset_index(drop=True)
        srs = rnn_mod.RNNProcess(path=os.environ["RWKV_CHAMP_CKPT"], device=torch.device("cpu"), dtype=torch.float32)
        with torch.inference_mode():
            for row in df.to_dict("records"):
                curve, _ = srs.run(row, skip=False)
                if int(row["has_label"]) != 1:
                    continue
                _, w, s_raw, d_raw = curve
                feats = torch.cat([w.reshape(-1), s_raw.reshape(-1), d_raw.reshape(-1)]).float().numpy()
                t = float(row["label_elapsed_seconds"])
                X.append(feats); T.append(t); Y.append(float(row["label_y"])); U.append(uid)
                P.append(float(srs.predict_func(curve, t)))
        print(f"user {uid}: {len(df):,} reviews -> {len(X):,} records", flush=True)
    return np.array(X), np.array(T), np.array(Y), np.array(P), np.array(U)


def ridge(Xtr, ytr, lam=1e-2):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Z = (Xtr - mu) / sd
    Z1 = np.hstack([Z, np.ones((len(Z), 1))])
    A = Z1.T @ Z1 + lam * np.eye(Z1.shape[1]); A[-1, -1] -= lam
    beta = np.linalg.solve(A, Z1.T @ ytr)
    return mu, sd, beta


def predict(model, X):
    mu, sd, beta = model
    Z1 = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
    return Z1 @ beta


def main():
    X, T, Y, P, U = collect()
    y = np.log1p(np.clip(T, 1.0, None))
    users = sorted(set(U.tolist()))
    r2s, corr = [], []
    for held in users:
        tr, te = U != held, U == held
        model = ridge(X[tr], y[tr])
        yhat = predict(model, X[te])
        ss_res = ((y[te] - yhat) ** 2).sum(); ss_tot = ((y[te] - y[te].mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot
        pc = np.clip(P[te], 1e-6, 1 - 1e-6)
        bce = -(Y[te] * np.log(pc) + (1 - Y[te]) * np.log(1 - pc))
        resid = np.abs(y[te] - yhat)
        c = float(np.corrcoef(resid, bce)[0, 1])
        r2s.append(r2); corr.append(c)
        print(f"  held-out user {held:>5}: n={int(te.sum()):>7,}  R^2 {r2:.3f}   corr(|resid|, ahead BCE) {c:+.3f}")
    print(f"MEAN held-out R^2 = {np.mean(r2s):.3f}   (kill line 0.85: above it the head already carries the interval)")
    print(f"MEAN corr(|resid|, ahead BCE) = {np.mean(corr):+.3f}   (kill line ~0: the aux task's information is unrelated to ahead error)")


if __name__ == "__main__":
    main()
