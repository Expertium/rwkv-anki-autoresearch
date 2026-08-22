"""How accurate is the RWKV arm's interval inversion?

The FSRS arm bisects 60 times, so its interval is exact to machine precision. The RWKV arm
instead samples the rectified curve on a 441-point log-t grid and reads t(DR) off by linear
interpolation, because one grid call is far cheaper than seven bisections on a curve that
costs four forward passes to obtain. That trade is only sound if the interpolation error is
negligible, and a systematic error here would bias every workload number.

So: invert on the grid, then evaluate the EXACT rectified curve at the returned t* and see
how far R(t*) is from the target. Reported as |R(t*) - DR|, which is the quantity that
matters -- an interval error only counts insofar as it lands at the wrong retention.

Usage: .venv/Scripts/python.exe scratchpad/workload/probe_inversion.py [uid] [n_reviews]
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
N = int(sys.argv[2]) if len(sys.argv) > 2 else 250

df = get_rwkv_data(Path(r"C:\Users\Andrew\anki-revlogs-10k"), UID)
df = df.sort_values("review_th", kind="stable").reset_index(drop=True)
proc = RNNProcess(CHAMPION_CKPT, "cpu", torch.float32, DEFAULT_ANKI_RWKV_CONFIG)
rnn = proc.rnn
torch.manual_seed(1234)
t_grid = build_grid(rnn.s_max)
log_t = np.log(t_grid.numpy().astype(np.float64))
ratings = df["rating"].to_numpy(dtype=np.int64)

err = []
clamped = 0
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
        curve = rnn.button_curves(heads, t_grid)[b].double().numpy()
        ivl, hit_lo, hit_hi, _ = invert(curve, log_t, DR_LEVELS)
        # exact rectified curve at the returned interval
        exact = rnn.button_curves(
            heads, torch.tensor(ivl, dtype=torch.float32))[b].double().numpy()
        ok = ~(hit_lo | hit_hi)
        clamped += int((~ok).sum())
        err.append(np.where(ok, np.abs(exact - np.array(DR_LEVELS)), np.nan))
        out = rnn.review(feats, *st)
        (proc.card_states[cid], proc.note_states[nid], proc.deck_states[did],
         proc.preset_states[pid], proc.global_state) = out[5:10]

E = np.array(err)
print("user %d, %d reviews, %d-point grid" % (UID, N, len(t_grid)))
print("  %-6s %12s %12s %12s" % ("DR", "median |dR|", "p99 |dR|", "max |dR|"))
for k, dr in enumerate(DR_LEVELS):
    c = E[:, k]
    c = c[~np.isnan(c)]
    if not len(c):
        print("  %-6s %12s %12s %12s" % ("%d%%" % round(dr * 100), "-", "-", "-"))
        continue
    print("  %-6s %12.3e %12.3e %12.3e"
          % ("%d%%" % round(dr * 100), np.median(c), np.percentile(c, 99), c.max()))
print("  clamped (target unreachable in [1 s, exp(s_max)]): %d of %d cells"
      % (clamped, N * len(DR_LEVELS)))
print("")
print("|dR| is how far the returned interval actually lands from the requested retention.")
print("Compare against the DR spacing of 0.05: anything <= 1e-3 is far below the level at")
print("which the choice of DR level itself matters.")
