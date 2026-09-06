"""Sensitivity of the curve head to its own hardcoded Dropout(0.1) inside head_w (srs_model.py:744-750):
ahead BCE on realcyc with that dropout ON (train mode, seeded) vs OFF, 2 train users, all labelled rows.
The states are advanced with dropout OFF (skip=False run); the ON prediction re-evaluates the heads only."""
import os, sys, time
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
for k, v in _ENV.items(): os.environ[k] = v
from pathlib import Path
import numpy as np, torch
torch.set_num_threads(4)
import rwkv.run_as_rnn as rnn_mod
from rwkv.data_processing import get_rwkv_data
DATA = Path(r"C:\Users\Andrew\anki-revlogs-10k-id")
USERS = [int(u) for u in sys.argv[1:]] or [107, 136]
eps = 1e-6
def bce(p, y): p = np.clip(p, eps, 1 - eps); return -(y*np.log(p) + (1-y)*np.log(1-p))
XS = []
def hook_n(m, i, o): XS.append(o.detach().clone())
t0 = time.time(); res = {}
for uid in USERS:
    torch.manual_seed(uid)
    df = get_rwkv_data(DATA, uid).sort_values("review_th", kind="stable").reset_index(drop=True)
    proc = rnn_mod.RNNProcess(path=os.environ["RWKV_CHAMP_CKPT"], device=torch.device("cpu"), dtype=torch.float32)
    rnn = proc.rnn
    drop = [m for m in rnn.head_w if isinstance(m, torch.nn.Dropout)]
    assert len(drop) == 1 and abs(drop[0].p - 0.1) < 1e-9, drop
    h = rnn.prehead_norm.register_forward_hook(hook_n)
    b_off, b_on, ys = [], [], []
    g = torch.Generator().manual_seed(1234)
    with torch.inference_mode():
        for row in df.to_dict("records"):
            XS.clear()
            curve, _ = proc.run(row, skip=False)
            if int(row["has_label"]) != 1: continue
            t = max(1.0, float(row["label_elapsed_seconds"])); y = float(row["label_y"])
            p_off = float(proc.predict_func(curve, t).reshape(()))
            x = XS[-1]
            # ON: re-run head_w with dropout active (seeded), everything else identical
            drop[0].train(); torch.manual_seed(int(torch.randint(0, 2**31-1, (1,), generator=g)))
            x_w = rnn.head_w(x).float(); drop[0].eval()
            w_l = torch.nn.functional.linear(x_w, rnn.gru_w_weight, rnn.gru_w_bias)
            s = torch.nn.functional.linear(x_w, rnn.gru_s_weight, rnn.gru_s_bias)
            d = torch.nn.functional.linear(x_w, rnn.gru_d_weight, rnn.gru_d_bias)
            wsm = torch.softmax(w_l, dim=-1)
            p_on = float(rnn.gru_forgetting_curve(wsm, s, d, torch.tensor([[t]])).reshape(()))
            b_off.append(bce(p_off, y)); b_on.append(bce(p_on, y)); ys.append(y)
    h.remove()
    res[uid] = (np.mean(b_off), np.mean(b_on), len(b_off))
    print(f"user {uid}: n={len(b_off)} ahead BCE dropout OFF {np.mean(b_off):.5f}  ON {np.mean(b_on):.5f}  delta {np.mean(b_on)-np.mean(b_off):+.5f}  [{time.time()-t0:.0f}s]", flush=True)
a = np.array(list(res.values())); print(f"BY-USER MEAN delta (ON - OFF): {np.mean(a[:,1]-a[:,0]):+.5f}")
