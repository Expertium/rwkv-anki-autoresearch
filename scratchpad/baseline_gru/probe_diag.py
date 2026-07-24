"""Diagnose WHY the trained GRU ignores query-row features (probe_sensitivity_check
FAILED test A). Inspect the real PreparedBatch structures the smoke never saw:
per-module skip dtypes/counts, gather index ranges, and a direct module-level
sensitivity call on the card stream."""
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
from rwkv.config import RWKV_SUBMODULES
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
print(f"user {key[0]}: sample.skips dtype={sample.skips.dtype}, "
      f"n={sample.skips.numel()}, sum={int(sample.skips.sum())}")

batch = prepare([sample], seed=1234)
n_rows = batch.start.shape[0]
print(f"batch.start {tuple(batch.start.shape)} {batch.start.dtype}; "
      f"num_data={batch.num_data}; labels {tuple(batch.labels.shape)}")

for i, name in enumerate(RWKV_SUBMODULES):
    for j, (g, L, ts, sk) in enumerate(zip(
            batch.sub_gather[i], batch.sub_gather_lens[i],
            batch.time_shift_selects[i], batch.skips[i])):
        g = g.view(-1)
        print(f"module {i} ({name}) split {j}: sub_len={L}, "
              f"B'={g.numel() // L}, gather[min={int(g.min())},max={int(g.max())}] "
              f"n_neg={int((g < 0).sum())}, skip dtype={sk.dtype} "
              f"sum={int(sk.view(-1).to(torch.int64).sum())}/{sk.numel()}")

# --- direct module-level sensitivity on the card stream ---------------------------
model = SrsRWKV(anki_rwkv_config=DEFAULT_ANKI_RWKV_CONFIG)
model.load_state_dict(torch.load("scratchpad/baseline_gru/bgruws_7000.pth",
                                 weights_only=True, map_location="cpu"))
model.float().eval()

start0 = batch.start.float()
with torch.no_grad():
    x = model.features2card(model._apply_input_feat_mask(start0)
                            if model.input_feat_mask_on else start0)
    g = batch.sub_gather[0][0]
    L = batch.sub_gather_lens[0][0]
    module_in = torch.index_select(x, 0, torch.clamp(g, min=0)).view(-1, L, x.size(-1))
    skip_BT = batch.skips[0][0].view(-1, L)
    ts_BT = batch.time_shift_selects[0][0].view(-1, L)
    m0 = model.rwkv_modules[0]
    out0 = m0(module_in, time_shift_select_BT=ts_BT, skip_BT=skip_BT)

    # find a TRUE skip position and perturb the module input there
    pos = torch.nonzero(skip_BT.to(torch.bool))
    print(f"card split0: module_in {tuple(module_in.shape)}, "
          f"skip positions {pos.shape[0]}")
    if pos.shape[0]:
        b0, t0 = int(pos[0][0]), int(pos[0][1])
        pert = module_in.clone()
        pert[b0, t0] += 1.0
        out1 = m0(pert, time_shift_select_BT=ts_BT, skip_BT=skip_BT)
        d_at = (out1 - out0)[b0, t0].abs().max().item()
        d_all = (out1 - out0).abs().max().item()
        print(f"perturb module_in at skip ({b0},{t0}): |dout| at pos = {d_at:.3e}, "
              f"max anywhere = {d_all:.3e}  (probe alive iff > 0)")
