"""Is the CURVE HEAD lossy relative to its own input? The decisive screen for a richer ahead head.

Motivation (headrank_screen.py, 2026-09-07): reading the SAME 80-d trunk output (participation-ratio
rank 10.0), the rating head's hidden spreads over rank 10.9 while the curve head's collapses to
**2.4** -- 3 dims carry 90% of its variance, for a 9-number output (w, S, d) x N=3. So across every
review the forgetting curve is drawn from a ~2.4-parameter family, which is about what FSRS has, in a
model whose whole premise is a flexible learned curve. That is either (a) the trunk carries no more
curve-relevant information, or (b) the head cannot exploit what is there. Only (b) is a lever.

THE TEST. Take the trunk output x at each labelled row -- exactly what the curve head is handed --
and fit a leave-one-user-out logistic probe on [x, basis(log t)] -> y. Compare its held-out BCE with
the MODEL'S OWN ahead prediction on the same rows.
  probe >= model            -> the head is not lossy; the limit is the trunk or the supervision, and a
                              richer head is not the lever. (Expected if the curve family is right.)
  probe << model by >0.002  -> the head discards information its own input contains: a richer ahead
                              head is worth a run, and this bounds the prize.
The probe is deliberately UNCONSTRAINED (no monotonicity, arbitrary function of x and t), so it is an
upper bound on what any head reading the same x could do -- it can also exploit the very
non-monotonicity the deploy contract forbids, which is why the number is a bound and not a target.

⚠ Train-range users only; the VAL half is reserved for gates.
Usage: head_bottleneck.py [users...]
"""
import os
import sys
import time

os.chdir(r"C:\Users\Andrew\rwkv-anki-autoresearch")
sys.path.insert(0, os.getcwd())
_ENV = {
    "RWKV_ARCH_MODULE": "scratchpad/track2_a18/architecture_d80_lora4_cnd.py",
    "RWKV_INTERLEAVE": "1", "RWKV_GRU_HEAD": "3", "RWKV_PAVA_LAMBDA": "0.2",
    "RWKV_NO_AHEAD_RESIDUAL": "1", "RWKV_STRIP_L0_VLORA": "1",
    "RWKV_STATE_CLAMP_TAU": "300", "RWKV_STATE_CLAMP_WINDOW": "32768",
    "RWKV_STRIP_CMIX": "user_id:0,user_id:1,user_id:2,preset_id:0,preset_id:1,preset_id:2,deck_id:1,deck_id:2,card_id:1",
    "RWKV_ID_FEATURES": "1", "RWKV_REAL_CYCLES": "1", "RWKV_ZERO_FEATURES": "",
}
for k, v in _ENV.items():
    os.environ.setdefault(k, v)
os.environ.setdefault("RWKV_CHAMP_CKPT", "scratchpad/realcyc/rc_d_10935.pth")
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(4)
import rwkv.run_as_rnn as rnn_mod
from rwkv.data_processing import get_rwkv_data

DATA = Path(r"C:\Users\Andrew\anki-revlogs-10k-id")
USERS = [int(u) for u in sys.argv[1:]] or [107, 136, 156, 178, 203, 1207]
EPS = 1e-6


def bce(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


X, P, Y, T, U = [], [], [], [], []
buf = {}
t0 = time.time()
for uid in USERS:
    torch.manual_seed(uid)
    df = get_rwkv_data(DATA, uid).sort_values("review_th", kind="stable").reset_index(drop=True)
    proc = rnn_mod.RNNProcess(path=os.environ["RWKV_CHAMP_CKPT"], device=torch.device("cpu"), dtype=torch.float32)

    def hook(_m, _i, o):
        buf["x"] = o.detach().float().reshape(-1).numpy().copy()
    h = proc.rnn.prehead_norm.register_forward_hook(hook)
    with torch.inference_mode():
        for row in df.to_dict("records"):
            curve, _ = proc.run(row, skip=False)
            if int(row["has_label"]) == 1:
                t = float(row["label_elapsed_seconds"])
                X.append(buf["x"]); P.append(float(proc.predict_func(curve, t)))
                Y.append(float(row["label_y"])); T.append(t); U.append(uid)
    h.remove()
    print(f"user {uid}: {len(df):,} rows -> {len(X):,} labelled [{time.time()-t0:.0f}s]", flush=True)

X = np.stack(X).astype(np.float32)
P, Y, T, U = (np.asarray(v) for v in (P, Y, T, U))
out = "scratchpad/arch_2026-09-07/head_dump.npz"
np.savez_compressed(out, x=X, p=P, y=Y, t=T, u=U)
print(f"wrote {out}: {len(X):,} labelled rows, trunk dim {X.shape[1]}")
print("now fit with: .venv/Scripts/python.exe scratchpad/arch_2026-09-07/head_probe_fit.py " + out)
