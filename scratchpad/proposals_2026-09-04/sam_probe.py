"""CPU screen for ranked-queue rank 6 (SAM, decay-only): is realcyc's minimum SHARP at SAM's scale?

Sharpness = L(w + rho * g / ||g||) - L(w) on real training chunks, with g the gradient of the REAL
training loss (get_loss's average_loss) at the realcyc checkpoint, rho in {0.01, 0.02, 0.05}, ||.||
the global 2-norm over all parameters (SAM's ascent step). Reports the MAX over chunks (a median cannot
see the sharp directions; the iter-51 lesson) and the median.
Kill rule (pre-registered in literature.md): gap at rho = 0.05 below 0.002 (~0.5% of the loss) => the
minimum is already flat at SAM's scale, the penalty sits at the noise floor, dead.
Second screen, free: the train-vs-val gap from realcyc's own WS log (last 500 steps' train loss vs the
final validation loss) -- if ~0 there is no generalization gap to close.

Runs on CPU with the pure-PyTorch kernel (the model is float32, dropout off so the loss is
deterministic; grads enabled). ~12 chunks from train-range users, smallest chunk each.
Usage: sam_probe.py [n_users=12]
"""
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.getcwd())
_ENV = {
    "RWKV_ARCH_MODULE": "scratchpad/track2_a18/architecture_d80_lora4_cnd.py",
    "RWKV_INTERLEAVE": "1", "RWKV_GRU_HEAD": "3", "RWKV_PAVA_LAMBDA": "0.2",
    "RWKV_NO_AHEAD_RESIDUAL": "1", "RWKV_STRIP_L0_VLORA": "1",
    "RWKV_STATE_CLAMP_TAU": "300", "RWKV_STATE_CLAMP_WINDOW": "32768",
    "RWKV_STRIP_CMIX": "user_id:0,user_id:1,user_id:2,preset_id:0,preset_id:1,preset_id:2,deck_id:1,deck_id:2,card_id:1",
    "RWKV_ID_FEATURES": "1", "RWKV_REAL_CYCLES": "1", "RWKV_ZERO_FEATURES": "",
    "RWKV_NO_JIT": "1",
}
for _k, _v in _ENV.items():
    os.environ[_k] = _v

import lmdb
import numpy as np
import torch

torch.set_num_threads(6)
from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG
from rwkv.model.srs_model import SrsRWKV
from rwkv.prepare_batch import get_data, prepare

DB = "F:/rwkv_lmdb/train_db_5k_h1_id5"
CKPT = "scratchpad/realcyc/rc_d_10935.pth"
N_USERS = int(sys.argv[1]) if len(sys.argv) > 1 else 12
RHOS = [0.01, 0.02, 0.05]


def to_f32(pb):
    """The LMDB stores bf16; the CPU model is fp32. Cast every floating tensor field (and lists of
    them) of the PreparedBatch up, leave integer index tensors alone."""
    import dataclasses

    def cast(v):
        if torch.is_tensor(v):
            return v.float() if v.is_floating_point() else v
        if isinstance(v, list):
            return [cast(x) for x in v]
        if isinstance(v, tuple):
            return tuple(cast(x) for x in v)
        return v
    for f in dataclasses.fields(pb):
        setattr(pb, f.name, cast(getattr(pb, f.name)))
    return pb


def load_chunks():
    env = lmdb.open(DB, map_size=400_000_000_000, readonly=True, lock=False)
    out = []
    with env.begin(write=False) as txn:
        uid = 101
        while len(out) < N_USERS and uid < 400:
            raw = txn.get(f"{uid}_batches".encode())
            uid += 1
            if raw is None:
                continue
            batches = json.loads(raw)
            b = min(batches, key=lambda x: x[2])
            key = (uid - 1, b[0], b[1], b[2])
            out.append((key, to_f32(prepare([get_data(txn, key, device="cpu")], seed=1234, probe_density=0.08).to("cpu"))))
    env.close()
    return out


def loss_of(model, pb):
    st = model.get_loss(pb)
    return st.average_loss


def main():
    torch.manual_seed(0)
    model = SrsRWKV(anki_rwkv_config=DEFAULT_ANKI_RWKV_CONFIG)
    sd = torch.load(CKPT, map_location="cpu", weights_only=True)
    model.load_state_dict(sd, strict=True)
    model = model.float()
    model.eval()                      # dropout off: the loss must be a deterministic function of w
    params = [p for p in model.parameters() if p.requires_grad]
    chunks = load_chunks()
    print(f"{len(chunks)} chunks from {DB}", flush=True)
    rows = []
    for key, pb in chunks:
        model.zero_grad(set_to_none=True)
        torch.set_grad_enabled(True)
        l0 = loss_of(model, pb)
        l0.backward()
        grads = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in params]
        gnorm = torch.sqrt(sum((g.float() ** 2).sum() for g in grads)).item()
        gaps = {}
        with torch.no_grad():
            for rho in RHOS:
                for p, g in zip(params, grads):
                    p.add_(g * (rho / (gnorm + 1e-12)))
                with torch.enable_grad():
                    pass
                l1 = loss_of(model, pb).item()
                for p, g in zip(params, grads):
                    p.sub_(g * (rho / (gnorm + 1e-12)))
                gaps[rho] = l1 - l0.item()
        rows.append((key, l0.item(), gnorm, gaps))
        print(f"  user {key[0]:>4} rows {key[3]:>6,}  L0 {l0.item():.5f}  ||g|| {gnorm:.4f}  " +
              "  ".join(f"gap@{r}={gaps[r]:+.5f}" for r in RHOS), flush=True)
    for rho in RHOS:
        g = np.array([r[3][rho] for r in rows])
        print(f"rho={rho}: MAX gap {g.max():+.5f}   median {np.median(g):+.5f}   min {g.min():+.5f}")
    g5 = np.array([r[3][0.05] for r in rows])
    print(f"KILL LINE: gap@0.05 < 0.002 everywhere => flat. MAX gap@0.05 = {g5.max():+.5f} -> "
          + ("SHARP (lever alive)" if g5.max() >= 0.002 else "FLAT (dead)"))

    # second screen: the train-val gap from realcyc's own WS log
    logs = sorted(glob.glob("scratchpad/realcyc/ws_*.log"), key=os.path.getmtime)
    if logs:
        txt = open(logs[-1], encoding="utf-8", errors="replace").read().replace("\r", "\n")
        steps = re.findall(r"^0 \d+ \d+, all: ([\d.]+), ahead: ([\d.]+) \([\d.]+\), imm: ([\d.]+)", txt, flags=re.M)
        vals = re.findall(r"Mean ahead validation loss: ([\d.]+) \([\d.]+\), imm: ([\d.]+)", txt)
        if steps and vals:
            last = np.array(steps[-500:], dtype=float)
            print(f"WS log {os.path.basename(logs[-1])}: last-500-step TRAIN ahead {last[:,1].mean():.4f} imm {last[:,2].mean():.4f}"
                  f"   final VAL ahead {float(vals[-1][0]):.4f} imm {float(vals[-1][1]):.4f}"
                  f"   gap (val - train) ahead {float(vals[-1][0]) - last[:,1].mean():+.4f} imm {float(vals[-1][1]) - last[:,2].mean():+.4f}"
                  "   (NOTE: train rows include dropout + all rows; val is the 10-user set -- a coarse gap only)")


if __name__ == "__main__":
    main()
