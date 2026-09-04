"""CPU screen for ranked-queue rank 5 (multi-horizon button ordering on the probes): how often do the
4 counterfactual button curves CROSS at horizons other than the label's t?

At every 20th real row of a few train-range users, on realcyc's deploy RNN: take the 4 button heads
(the same counterfactual states `insert_probes` builds: the row re-fed with each grade, duration
zeroed) and evaluate the RAW (unrectified) curves at horizons {label t, 1 d, 7 d, 30 d, 180 d}.
Report, per horizon: the fraction of rows with an adjacent-button ORDER VIOLATION (R_b > R_{b+1} for
some b, Again<Hard<Good<Easy expected), the fraction whose button order differs from the order at the
label's t, and the median |R_Good - R_Hard| at 30 d.
Kill rule (pre-registered in domain.md): crossings on < 3% of rows at EVERY horizon => the constraint
is already satisfied and the hinge would be inert. Also dead if the median Good-Hard gap at 30 d is
< 0.01 (the buttons barely separate; ordering is not the binding structure).
Usage: button_probe.py [users...]    default 107 136 203 1207
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
from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG

DATA = Path(r"C:\Users\Andrew\anki-revlogs-10k-id")
USERS = [int(u) for u in sys.argv[1:]] or [107, 136, 203, 1207]
DAY = 86400.0
HORIZONS = {"1d": DAY, "7d": 7 * DAY, "30d": 30 * DAY, "180d": 180 * DAY}
EVERY = 20


def raw_button_curves(rnn, heads, t_list):
    ahead_logits, w, s_raw, d_raw, _ = heads
    t = torch.as_tensor(t_list, dtype=torch.float32).reshape(-1, 1)
    raw = torch.stack([rnn.curve_p(ahead_logits[k:k + 1], w[k:k + 1], s_raw[k:k + 1], d_raw[k:k + 1], t)
                       for k in range(4)])          # (4, T)
    return raw.reshape(4, -1).float().numpy()


def main():
    names = ["label_t"] + list(HORIZONS)
    viol = {n: [] for n in names}
    order_differs = {n: [] for n in names[1:]}
    gap30 = []
    n_rows = 0
    for uid in USERS:
        torch.manual_seed(uid)
        df = get_rwkv_data(DATA, uid).sort_values("review_th", kind="stable").reset_index(drop=True)
        proc = rnn_mod.RNNProcess(path=os.environ["RWKV_CHAMP_CKPT"], device=torch.device("cpu"), dtype=torch.float32)
        rnn = proc.rnn
        with torch.inference_mode():
            for i, row in enumerate(df.to_dict("records")):
                if int(row["has_label"]) == 1 and i % EVERY == 0:
                    feats = proc.get_tensor(row)
                    cid, nid, did, pid = row["card_id"], row["note_id"], row["deck_id"], row["preset_id"]
                    for d, k in ((proc.card_states, cid), (proc.note_states, nid), (proc.deck_states, did), (proc.preset_states, pid)):
                        d.setdefault(k, None)
                    heads = rnn.button_heads(feats, proc.card_states[cid], proc.note_states[nid],
                                             proc.deck_states[did], proc.preset_states[pid], proc.global_state)
                    t_lab = max(1.0, float(row["label_elapsed_seconds"]))
                    ts = [t_lab] + [HORIZONS[h] for h in names[1:]]
                    R = raw_button_curves(rnn, heads, ts)   # (4 buttons, len(ts))
                    order_lab = tuple(np.argsort(R[:, 0]))
                    for j, n in enumerate(names):
                        col = R[:, j]
                        viol[n].append(bool(np.any(col[:-1] > col[1:] + 1e-7)))
                        if j > 0:
                            order_differs[n].append(tuple(np.argsort(col)) != order_lab)
                    gap30.append(abs(R[2, names.index("30d")] - R[1, names.index("30d")]))
                    n_rows += 1
                proc.run(row, skip=False)
        print(f"user {uid}: {len(df):,} reviews -> {n_rows:,} probed rows so far", flush=True)
    print(f"\nprobed rows: {n_rows:,}   (every {EVERY}th labelled row, {len(USERS)} users)")
    print(f"{'horizon':<9} {'crossing rate':>14} {'order != order@label_t':>24}")
    for n in names:
        od = f"{np.mean(order_differs[n]):.3f}" if n in order_differs else "   --"
        print(f"{n:<9} {np.mean(viol[n]):>14.3f} {od:>24}")
    print(f"median |R_Good - R_Hard| at 30 d: {np.median(gap30):.4f}   (dead if < 0.01)")
    worst = max(np.mean(viol[n]) for n in names[1:])
    print("VERDICT: " + ("DEAD -- crossings < 3% at every horizon" if worst < 0.03 else
                        f"ALIVE -- max crossing rate {worst:.3f} at an off-label horizon"))


if __name__ == "__main__":
    main()
