"""How fast is the RWKV deploy scheduling path, per review, on 1 CPU thread?

Three costs are separated, because they scale differently and only one of them is
avoidable:
  advance   -- review() with the REAL duration. One forward. Mandatory.
  heads     -- button_heads(): FOUR forwards (one per counterfactual button), duration
               zeroed. Needed because PAVA pools ACROSS buttons, so the rectified curve
               for the pressed button cannot be had from one forward.
  invert    -- curve evaluation on a t-grid + inversion for the 7 DR levels. Pure
               arithmetic on already-computed heads.

Usage: .venv/Scripts/python.exe scratchpad/workload/probe_speed.py [user_id] [n_reviews]
"""
import sys, os, time
sys.path.insert(0, os.getcwd())
from scratchpad.workload.env_champ import apply, CHAMPION_CKPT
apply()

from pathlib import Path
import torch
from rwkv.data_processing import get_rwkv_data
from rwkv.run_as_rnn import RNNProcess
from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG

UID = int(sys.argv[1]) if len(sys.argv) > 1 else 5001
N = int(sys.argv[2]) if len(sys.argv) > 2 else 300
torch.set_num_threads(1)

D = Path(r"C:\Users\Andrew\anki-revlogs-10k")
df = get_rwkv_data(D, UID)
print("user %d: %d reviews, %d cards" % (UID, len(df), df["card_id"].nunique()))

proc = RNNProcess(CHAMPION_CKPT, "cpu", torch.float32, DEFAULT_ANKI_RWKV_CONFIG)
rnn = proc.rnn
print("card_features_dim =", rnn.card_features_dim, " gru_on =", rnn.gru_on,
      " has pava_theta =", hasattr(rnn, "pava_theta"))

# the ID encodings are torch.randint draws; seed so a rerun reproduces.
torch.manual_seed(1234)

rows = df.head(N).to_dict("records")
t_grid = torch.exp(torch.linspace(0.0, float(rnn.s_max), 128))

t_adv = t_heads = t_inv = 0.0
n_adv = n_heads = 0
with torch.inference_mode():
    for r in rows:
        feats = proc.get_tensor(r)
        cid, nid, did, pid = r["card_id"], r["note_id"], r["deck_id"], r["preset_id"]
        for d, k in ((proc.card_states, cid), (proc.note_states, nid),
                     (proc.deck_states, did), (proc.preset_states, pid)):
            d.setdefault(k, None)

        t0 = time.perf_counter()
        heads = rnn.button_heads(feats, proc.card_states[cid], proc.note_states[nid],
                                 proc.deck_states[did], proc.preset_states[pid],
                                 proc.global_state)
        t1 = time.perf_counter()
        curves = rnn.button_curves(heads, t_grid)   # (4, T)
        t2 = time.perf_counter()
        out = rnn.review(feats, proc.card_states[cid], proc.note_states[nid],
                         proc.deck_states[did], proc.preset_states[pid],
                         proc.global_state)
        t3 = time.perf_counter()
        (proc.card_states[cid], proc.note_states[nid], proc.deck_states[did],
         proc.preset_states[pid], proc.global_state) = out[5:10]

        t_heads += t1 - t0; t_inv += t2 - t1; t_adv += t3 - t2
        n_heads += 1; n_adv += 1

tot = t_heads + t_inv + t_adv
print("")
print("per review over %d reviews (1 thread):" % N)
print("  advance (1 fwd)          %8.2f ms   %5.1f%%" % (1000 * t_adv / N, 100 * t_adv / tot))
print("  button_heads (4 fwd)     %8.2f ms   %5.1f%%" % (1000 * t_heads / N, 100 * t_heads / tot))
print("  curve on %3d-pt grid     %8.2f ms   %5.1f%%" % (len(t_grid), 1000 * t_inv / N, 100 * t_inv / tot))
print("  TOTAL                    %8.2f ms  -> %.1f reviews/s" % (1000 * tot / N, N / tot))
print("")
print("  a 17k-review user would take %.1f min at this rate" % (17459 * tot / N / 60))
