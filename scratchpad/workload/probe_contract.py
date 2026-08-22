"""What do the two possible shortcuts in the RWKV scheduling path actually cost?

The deploy contract needs FIVE forward passes per review (4 counterfactual buttons for
PAVA + 1 to advance the state with the real duration), which is 20.5 reviews/s on one
thread. Two shortcuts are available and both are tempting at 2.5x / 5x:

  A  FULL CONTRACT : 4 button heads, PAVA-rectified, duration zeroed.   (5 fwd)
  B  NO PAVA       : the pressed button's own raw curve, duration zeroed. (2 fwd)
  C  NO PAVA, REAL DURATION : the state-advancing forward's own curve.    (1 fwd)

This measures the interval each produces on the same reviews, so the choice is made on a
number instead of on intuition. Reported as the distribution of log-ratio vs A, per DR
level, because an interval comparison is multiplicative.

C is expected to differ most: it feeds the model a duration it would not have at
scheduling time, which is the exact leak the deploy contract's point 1 exists to prevent.

Usage: .venv/Scripts/python.exe scratchpad/workload/probe_contract.py <uid> [n_reviews]
"""
import sys
import os

sys.path.insert(0, os.getcwd())
from scratchpad.workload.env_champ import apply, CHAMPION_CKPT  # noqa: E402

apply()

from pathlib import Path  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)

from rwkv.data_processing import get_rwkv_data  # noqa: E402
from rwkv.run_as_rnn import RNNProcess  # noqa: E402
from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG  # noqa: E402
from scratchpad.workload.rwkv_arm import DR_LEVELS, build_grid, invert  # noqa: E402

UID = int(sys.argv[1]) if len(sys.argv) > 1 else 5100
N = int(sys.argv[2]) if len(sys.argv) > 2 else 600

DATA = Path(r"C:\Users\Andrew\anki-revlogs-10k")
df = get_rwkv_data(DATA, UID).sort_values("review_th", kind="stable").reset_index(drop=True)
proc = RNNProcess(CHAMPION_CKPT, "cpu", torch.float32, DEFAULT_ANKI_RWKV_CONFIG)
rnn = proc.rnn
torch.manual_seed(1234)
t_grid = build_grid(rnn.s_max)
log_t = np.log(t_grid.numpy().astype(np.float64))

K = len(DR_LEVELS)
ivl = {k: np.zeros((N, K)) for k in "ABC"}
ratings = df["rating"].to_numpy(dtype=np.int64)

with torch.inference_mode():
    for i, row in enumerate(df.head(N).to_dict("records")):
        feats = proc.get_tensor(row)
        cid, nid, did, pid = row["card_id"], row["note_id"], row["deck_id"], row["preset_id"]
        for d, k in ((proc.card_states, cid), (proc.note_states, nid),
                     (proc.deck_states, did), (proc.preset_states, pid)):
            d.setdefault(k, None)
        st = (proc.card_states[cid], proc.note_states[nid],
              proc.deck_states[did], proc.preset_states[pid], proc.global_state)
        b = ratings[i] - 1

        heads = rnn.button_heads(feats, *st)
        # A: rectified across all four buttons
        ivl["A"][i] = invert(rnn.button_curves(heads, t_grid)[b].double().numpy(),
                             log_t, DR_LEVELS)[0]
        # B: same heads, but the pressed button's RAW curve -- no cross-button pooling
        ah, w, s_raw, d_raw, _ = heads
        raw_b = rnn.curve_p(ah[b:b + 1], w[b:b + 1], s_raw[b:b + 1], d_raw[b:b + 1],
                            t_grid.reshape(-1, 1)).double().numpy().ravel()
        ivl["B"][i] = invert(raw_b, log_t, DR_LEVELS)[0]

        out = rnn.review(feats, *st)
        # C: the state-advancing forward's own curve, i.e. WITH the real duration
        raw_c = rnn.curve_p(out[0], out[1], out[2], out[3],
                            t_grid.reshape(-1, 1)).double().numpy().ravel()
        ivl["C"][i] = invert(raw_c, log_t, DR_LEVELS)[0]

        (proc.card_states[cid], proc.note_states[nid], proc.deck_states[did],
         proc.preset_states[pid], proc.global_state) = out[5:10]

print("user %d, %d reviews -- interval vs the FULL CONTRACT (A), log-ratio stats" % (UID, N))
print("")
for name in ("B", "C"):
    label = {"B": "B  no PAVA        (2 fwd, 2.5x faster)",
             "C": "C  no PAVA + real duration (1 fwd, 5x faster)"}[name]
    print(label)
    print("  %-5s %10s %10s %10s %10s" % ("DR", "median", "p90 |.|", "max |.|", "frac>5%"))
    for k, dr in enumerate(DR_LEVELS):
        lr = np.log(ivl[name][:, k] / ivl["A"][:, k])
        print("  %-5s %+10.4f %10.4f %10.4f %9.1f%%" % (
            "%d%%" % round(dr * 100), np.median(lr), np.percentile(np.abs(lr), 90),
            np.abs(lr).max(), 100 * (np.abs(lr) > np.log(1.05)).mean()))
    print("")
print("log-ratio 0 = identical. +0.10 = the shortcut gives a 10.5% LONGER interval.")
