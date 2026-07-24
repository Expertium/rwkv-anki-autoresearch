"""Stage-by-stage diff trace: run the full forward twice (baseline vs query-rows-
zeroed) and record where the perturbation dies. Hooks capture features2card output
and every stream module's output."""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ["RWKV_NO_JIT"] = "1"
os.environ["RWKV_BASELINE_CELL"] = "gru"
os.environ["RWKV_ARCH_MODULE"] = "scratchpad/track2_a9/architecture_d128_cmix1_user3_card2_note1.py"
os.environ["RWKV_GRU_HEAD"] = "2"
os.environ["RWKV_NO_AHEAD_RESIDUAL"] = "1"
os.environ["RWKV_ZERO_FEATURES"] = "22"
sys.path.insert(0, os.getcwd())

import json

import lmdb
import torch

from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG
from rwkv.model.srs_model import SrsRWKV
from rwkv.prepare_batch import get_data, prepare

torch.manual_seed(0)
db = lmdb.open("train_db_5k_h1", readonly=True, lock=False)
with db.begin(write=False) as txn:
    key = None
    for user in range(1, 200):
        raw = txn.get(f"{user}_batches".encode())
        if raw is None:
            continue
        for b in json.loads(raw):
            if 1500 <= b[2] <= 4000:
                key = (user, b[0], b[1], b[2])
                break
        if key:
            break
    sample = get_data(txn, key, device="cpu")
db.close()
batch = prepare([sample], seed=1234)

model = SrsRWKV(anki_rwkv_config=DEFAULT_ANKI_RWKV_CONFIG)
model.load_state_dict(torch.load("scratchpad/baseline_gru/bgruws_7000.pth",
                                 weights_only=True, map_location="cpu"))
model.float().eval()

start0 = batch.start.float()
qrows = start0[:, 23] > 0.5
sA = start0.clone()
sA[qrows] = 0.0

captured = {}


def grab(name):
    def hook(mod, inp, out):
        captured[name] = out.detach().clone() if torch.is_tensor(out) else out
    return hook


hooks = [model.features2card.register_forward_hook(grab("features2card"))]
for i, m in enumerate(model.rwkv_modules):
    hooks.append(m.register_forward_hook(grab(f"stream{i}")))
    # streams run once per split; keep ALL calls
    captured.setdefault(f"stream{i}_all", [])


def grab_multi(name):
    def hook(mod, inp, out):
        captured[name].append(out.detach().clone())
    return hook


for h in hooks:
    h.remove()
hooks = [model.features2card.register_forward_hook(grab("features2card"))]
for i, m in enumerate(model.rwkv_modules):
    hooks.append(m.register_forward_hook(grab_multi(f"stream{i}_all")))


def run(start):
    for i in range(len(model.rwkv_modules)):
        captured[f"stream{i}_all"] = []
    with torch.no_grad():
        outs = model.forward_batch(start, batch.sub_gather, batch.sub_gather_lens,
                                   batch.time_shift_selects, batch.skips,
                                   batch.num_data)
    snap = {"features2card": captured["features2card"]}
    for i in range(len(model.rwkv_modules)):
        snap[f"stream{i}"] = [t for t in captured[f"stream{i}_all"]]
    snap["final"] = outs
    return snap


a = run(start0)
b = run(sA)

d = (a["features2card"] - b["features2card"]).abs()
print(f"features2card: max|d| all rows {d.max():.3e}, at qrows {d[qrows].max():.3e}, "
      f"at real rows {d[~qrows].max():.3e}")
for i in range(len(model.rwkv_modules)):
    for j, (ta, tb) in enumerate(zip(a[f"stream{i}"], b[f"stream{i}"])):
        dd = (ta - tb).abs().max().item()
        print(f"stream {i} split {j}: out shape {tuple(ta.shape)}, max|d| = {dd:.3e}")
for k, (ta, tb) in enumerate(zip(a["final"], b["final"])):
    print(f"final[{k}]: max|d| = {(ta - tb).abs().max().item():.3e}")
