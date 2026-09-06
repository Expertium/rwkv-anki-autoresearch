"""CPU screen for LAWA-style WS-checkpoint averaging as the DECAY warm start (no LMDB, no GPU).
Forward-only on 3-4 parquet users (first 8192 rows each, deploy-identical feature path):
  L(avg of rc_ws_{K..10935}) vs L(rc_ws_10935)   for K in {6000, 8000, 9000}   (uniform average)
Kill: the WS-average is not lower than the WS-final by >= 0.002 on the median chunk => the constant-LR
iterate is not oscillating around a lower valley floor, nothing for the decay to start from.
Also reports, per round of the interleave, ||grad||/||W|| of the trunk tensors on one chunk (deep-supervision
screen): kill if round-0 tensors' median relative gradient is >= 1/3 of the last round's (no starvation).
"""
import os, sys, time
sys.path.insert(0, os.getcwd())
_ENV = {
    "RWKV_ARCH_MODULE": "scratchpad/track2_a18/architecture_d80_lora4_cnd.py",
    "RWKV_INTERLEAVE": "1", "RWKV_GRU_HEAD": "3", "RWKV_PAVA_LAMBDA": "0.2",
    "RWKV_NO_AHEAD_RESIDUAL": "1", "RWKV_STRIP_L0_VLORA": "1",
    "RWKV_STATE_CLAMP_TAU": "300", "RWKV_STATE_CLAMP_WINDOW": "32768",
    "RWKV_STRIP_CMIX": "user_id:0,user_id:1,user_id:2,preset_id:0,preset_id:1,preset_id:2,deck_id:1,deck_id:2,card_id:1",
    "RWKV_ID_FEATURES": "1", "RWKV_REAL_CYCLES": "1", "RWKV_ZERO_FEATURES": "",
    "RWKV_NO_JIT": "1", "RWKV_DROPOUT_SCALE": "0.5",
}
for k, v in _ENV.items():
    os.environ[k] = v
from pathlib import Path
import dataclasses
import numpy as np
import torch
torch.set_num_threads(4)
from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG
from rwkv.model.srs_model import SrsRWKV
from rwkv.data_processing import get_rwkv_data, create_sample
from rwkv.prepare_batch import prepare

DATA = Path(r"C:\Users\Andrew\anki-revlogs-10k-id")
D = "scratchpad/realcyc"
USERS = [int(a) for a in sys.argv[1:]] or [110, 107, 136, 156]


def to_f32(pb):
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


def build(uid):
    df = get_rwkv_data(DATA, uid).iloc[:1500]
    s = create_sample(user_id=uid, section_df=df, equalize_review_ths=[], dtype=torch.float32, device=torch.device("cpu"))
    return to_f32(prepare([s], seed=1234, probe_density=0.08).to("cpu")), len(df)


def W(step):
    return torch.load(os.path.join(D, f"rc_ws_{step}.pth"), map_location="cpu", weights_only=True)


steps_all = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 10935]
model = SrsRWKV(anki_rwkv_config=DEFAULT_ANKI_RWKV_CONFIG).float()
model.eval()
batches = [build(u) for u in USERS]
print("built", [(u, n) for u, (_, n) in zip(USERS, batches)], flush=True)


def losses(sd):
    model.load_state_dict(sd, strict=True)
    out = []
    with torch.no_grad():
        for pb, _ in batches:
            st = model.get_loss(pb)
            out.append((float(st.average_loss), float(st.ahead_avg), float(st.imm_avg)))
    return np.array(out)


base = losses(W(10935))
print("WS-final rc_ws_10935: per-chunk [total, ahead, imm]\n", base, flush=True)
final = W(10935)
for K in (6000, 8000, 9000, 10000):
    ks = [s for s in steps_all if s >= K]
    acc = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in final.items() if torch.is_tensor(v) and v.is_floating_point()}
    for s in ks:
        sd = W(s)
        for k in acc:
            acc[k] += sd[k].float()
    avg = dict(final)
    for k in acc:
        avg[k] = (acc[k] / len(ks)).to(final[k].dtype)
    la = losses(avg)
    d = la - base
    print(f"avg of {len(ks)} ckpts [{K}..10935]: delta vs WS-final per chunk total {np.round(d[:,0],5)} | median total {np.median(d[:,0]):+.5f} ahead {np.median(d[:,1]):+.5f} imm {np.median(d[:,2]):+.5f}", flush=True)
# distance between consecutive WS checkpoints vs distance to the average (oscillation signature)
def flat(sd):
    return torch.cat([sd[k].float().reshape(-1) for k in sorted(sd) if torch.is_tensor(sd[k]) and sd[k].is_floating_point()])
f9, f10, f11 = flat(W(9000)), flat(W(10000)), flat(W(10935))
print(f"||W10935-W10000|| {float((f11-f10).norm()):.4f}  ||W10000-W9000|| {float((f10-f9).norm()):.4f}  cos(step9->10, step10->11) {float(((f10-f9)@(f11-f10))/((f10-f9).norm()*(f11-f10).norm())):+.4f}  ||W|| {float(f11.norm()):.2f}")

# deep-supervision screen: relative gradient by interleave round on one chunk (the decay-final checkpoint)
model.load_state_dict(torch.load(os.path.join(D, "rc_d_10935.pth"), map_location="cpu", weights_only=True), strict=True)
model.zero_grad(set_to_none=True)
torch.set_grad_enabled(True)
st = model.get_loss(batches[0][0])
st.average_loss.backward()
rows = []
for n, p in model.named_parameters():
    if p.grad is None or p.dim() < 2:
        continue
    import re
    m = re.search(r"(card_id|note_id|deck_id|preset_id|user_id)\D*(\d+)", n)
    if not m:
        continue
    rows.append((m.group(1), int(m.group(2)), n, float(p.grad.norm() / (p.norm() + 1e-12))))
import collections
by = collections.defaultdict(list)
for s, l, n, r in rows:
    by[(s, l)].append(r)
print("relative grad ||g||/||W|| by (stream, layer) [median over the stream's 2-D tensors]:")
for k in sorted(by):
    print(f"  {k[0]:<10} layer {k[1]}: median {np.median(by[k]):.2e}  n={len(by[k])}")
