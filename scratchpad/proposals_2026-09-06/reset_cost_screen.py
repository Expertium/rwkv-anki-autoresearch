"""Chunk-reset cost screen (rank 13 chunk-continuous training / a learned initial state), CPU, deploy RNN.

Training resets EVERY stream's state at each 16,384-row chunk boundary; deploy never resets. How much
ahead/imm loss does a cold start cost, and how fast does it recover?
  (A) CONTINUOUS: one RNNProcess over the user's first 2*W rows; record per-labelled-row BCE(ahead) and
      CE(imm) on rows [W, 2W).
  (B) RESET: a fresh RNNProcess fed only rows [W, 2W) (what training sees for the second chunk).
Reset cost = mean over rows [W,2W) of loss(B) - loss(A), and the same by position bins after the reset
(the recovery curve). Kill line (pre-registered in literature.md 09-04, rank 13): the reset-cost curve is
flat past ~16k rows (i.e. the cost concentrates in the first few thousand rows and the by-chunk mean
cost is < 0.0005) => chunk-continuous training / a learned init has nothing to recover.
Usage: reset_cost_screen.py <user> [W=16384]
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
    "RWKV_CHAMP_CKPT": "scratchpad/realcyc/rc_d_10935.pth",
}
for k, v in _ENV.items():
    os.environ[k] = v
from pathlib import Path
import numpy as np
import torch

torch.set_num_threads(4)
import rwkv.run_as_rnn as rnn_mod
from rwkv.data_processing import get_rwkv_data

DATA = Path(r"C:\Users\Andrew\anki-revlogs-10k-id")
uid = int(sys.argv[1])
W = int(sys.argv[2]) if len(sys.argv) > 2 else 16384


def bce(p, y):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def run(rows, start, record_from):
    """Feed rows[start:] through a fresh process; return {row_index: (ahead_bce, imm_ce)} for labelled
    rows with index >= record_from."""
    srs = rnn_mod.RNNProcess(path=os.environ["RWKV_CHAMP_CKPT"], device=torch.device("cpu"), dtype=torch.float32)
    out = {}
    with torch.inference_mode():
        for i in range(start, len(rows)):
            row = rows[i]
            has = int(row["has_label"]) == 1
            curve, p_logits = srs.run(row, skip=False)
            if has and i >= record_from:
                t = float(row["label_elapsed_seconds"])
                p = float(srs.predict_func(curve, t))
                y = float(row["label_y"])
                out[i] = (bce(p, y), 0.0)  # imm dropped: ahead is the binding mode
    return out


t0 = time.time()
df = get_rwkv_data(DATA, uid).sort_values("review_th", kind="stable").reset_index(drop=True)
rows = df.iloc[: 2 * W].to_dict("records")
print(f"user {uid}: {len(df):,} reviews, using the first {len(rows):,}; W={W}", flush=True)
A = run(rows, 0, W)
print(f"  continuous pass done in {time.time()-t0:.0f}s, {len(A)} labelled rows in [W,2W)", flush=True)
t1 = time.time()
B = run(rows, W, W)
print(f"  reset pass done in {time.time()-t1:.0f}s, {len(B)} labelled rows", flush=True)
keys = sorted(set(A) & set(B))
da = np.array([B[k][0] - A[k][0] for k in keys])
di = np.array([B[k][1] - A[k][1] for k in keys])
pos = np.array(keys) - W
print(f"RESET COST over [W,2W): ahead {da.mean():+.5f}  imm {di.mean():+.5f}  (n={len(keys)}; continuous ahead {np.mean([A[k][0] for k in keys]):.4f} imm {np.mean([A[k][1] for k in keys]):.4f})")
print("recovery curve (rows after the reset): bin  n  d_ahead  d_imm")
edges = [0, 256, 512, 1024, 2048, 4096, 8192, W]
for a, b in zip(edges[:-1], edges[1:]):
    m = (pos >= a) & (pos < b)
    if m.sum():
        print(f"  [{a:>5},{b:>5})  {m.sum():>5}  {da[m].mean():+.5f}  {di[m].mean():+.5f}")
print(f"cost concentrated in first 2048 rows: ahead {da[pos < 2048].sum()/max(da.sum(),1e-9):.2f} of the total (if total > 0)")
print(f"done in {time.time()-t0:.0f}s")
