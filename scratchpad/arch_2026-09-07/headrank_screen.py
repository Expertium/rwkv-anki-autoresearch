"""Effective rank (participation ratio) of the curve head's hidden activation x_w (320-d) and the
rating head's x_p on realcyc, over real rows of 2 train users. Screen for head_fc_mult 4 -> 6."""
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
    "RWKV_CHAMP_CKPT": os.environ.get("RWKV_CHAMP_CKPT", "scratchpad/realcyc/rc_d_10935.pth"),
}
for k, v in _ENV.items(): os.environ[k] = v
from pathlib import Path
import numpy as np, torch
torch.set_num_threads(4)
import rwkv.run_as_rnn as rnn_mod
from rwkv.data_processing import get_rwkv_data
DATA = Path(r"C:\Users\Andrew\anki-revlogs-10k-id")
USERS = [int(u) for u in sys.argv[1:]] or [107, 136]
XW, XP, X = [], [], []
def hook_w(m, i, o): XW.append(o.detach().float().reshape(-1).numpy().copy())
def hook_p(m, i, o): XP.append(o.detach().float().reshape(-1).numpy().copy())
def hook_n(m, i, o): X.append(o.detach().float().reshape(-1).numpy().copy())
t0 = time.time()
for uid in USERS:
    torch.manual_seed(uid)
    df = get_rwkv_data(DATA, uid).sort_values("review_th", kind="stable").reset_index(drop=True)
    proc = rnn_mod.RNNProcess(path=os.environ["RWKV_CHAMP_CKPT"], device=torch.device("cpu"), dtype=torch.float32)
    rnn = proc.rnn
    h1 = rnn.head_w.register_forward_hook(hook_w); h2 = rnn.head_p.register_forward_hook(hook_p); h3 = rnn.prehead_norm.register_forward_hook(hook_n)
    with torch.inference_mode():
        for row in df.to_dict("records"):
            proc.run(row, skip=False)
    h1.remove(); h2.remove(); h3.remove()
    print(f"user {uid}: {len(df)} rows [{time.time()-t0:.0f}s]", flush=True)
def report(name, A):
    A = np.stack(A); A = A - A.mean(0)
    s = np.linalg.svd(A, compute_uv=False); e = s**2 / (s**2).sum()
    pr = 1.0 / (e**2).sum()
    k90 = int(np.searchsorted(np.cumsum(e), 0.90) + 1); k99 = int(np.searchsorted(np.cumsum(e), 0.99) + 1)
    dead = int((A.std(0) < 1e-6).sum())
    print(f"{name}: dim {A.shape[1]}, rows {A.shape[0]}, participation-ratio rank {pr:.1f}, dims for 90% var {k90}, 99% var {k99}, dead dims {dead}")
report("x (prehead_norm, trunk out)", X); report("x_w (curve head hidden)", XW); report("x_p (rating head hidden)", XP)
