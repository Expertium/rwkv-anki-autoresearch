"""Iter-23 probe-row smoke (BUILD_NOTES.md smoke plan).

Part A (CPU): insert_probes unit checks on a real train chunk -- row accounting, probe
content (differs from target ONLY at duration + grade one-hot within the 92), labels
(has_label=0, les/review_th preserved), skip=True, id copies, repack validity (permutation,
grouping, ascending in-group order), eligibility (no in-chunk-first targets), placement
(4 probes contiguous immediately before the target), meta flat-index consistency after
prepare().

Part B (CUDA, tiny chunk, coexists with the running A2): the CROWN JEWEL invisibility
check -- with fixed weights, every real/query row's curve + p outputs must be unchanged
by probe insertion (proves skip-commit masking + token-shift invisibility end-to-end
through the packed batch path), then a lambda>0 end-to-end loss/backward with
pava_theta.grad live.

Env: d=32 champion recipe (H=2/K=16, ZERO_FEATURES=22, NO_AHEAD_RESIDUAL=1), JIT on.
"""

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
os.chdir(REPO)
os.environ.setdefault("OMP_NUM_THREADS", "7")
os.environ["RWKV_N_HEADS"] = "2"
os.environ["RWKV_HEAD_DIM"] = "16"
os.environ["RWKV_ZERO_FEATURES"] = "22"
os.environ["RWKV_NO_AHEAD_RESIDUAL"] = "1"
os.environ["RWKV_PAVA_LAMBDA"] = "0.1"
os.environ["RWKV_DETERMINISTIC"] = "0"

import json

import lmdb
import numpy as np
import torch

from rwkv.prepare_batch import (
    _COL_DUR, _COL_R1, _LBL_HAS_LABEL, _PROBE_DUR_SCALED,
    get_data, insert_probes, prepare,
)
from rwkv.config import RWKV_SUBMODULES

FAIL = 0


def check(name, cond, extra=""):
    global FAIL
    print(f"{'PASS' if cond else 'FAIL'} {name}{' ' + extra if extra else ''}")
    if not cond:
        FAIL += 1


DB = "train_db_5k_h1"
env = lmdb.open(DB, map_size=400_000_000_000, readonly=True, lock=False)
with env.begin(write=False) as txn:
    # smallest single chunk among the first users -> tiny CUDA footprint
    best = None
    for uid in range(1, 40):
        raw = txn.get(f"{uid}_batches".encode())
        if raw is None:
            continue
        for b in json.loads(raw):
            if best is None or b[2] < best[1][2]:
                best = (uid, b)
    uid, b = best
    key = (uid, b[0], b[1], b[2])
    print(f"using user {uid} chunk {b} ({b[2]} rows)")
    data = get_data(txn, key, device="cpu")

DENS, SEED = 0.5, 1234
data2, meta = insert_probes(data, DENS, SEED)
n = data.card_features.size(0)
m = meta.pos4.shape[0]
check("row accounting", data2.card_features.size(0) == n + 4 * m, f"n={n} m={m}")

# recompute the old->new position map exactly as insert_probes does
sk = data.skips.numpy()
lab = data.global_labels.float().numpy()
is_t = np.zeros(n, dtype=bool)
# meta.target holds NEW positions; invert via src: build from data2 sizes
cf_old, cf_new = data.card_features.float().numpy(), data2.card_features.float().numpy()
lab_new = data2.global_labels.float().numpy()

# placement + content per probe group
ok_place = ok_content = ok_labels = ok_ids = ok_press = True
for j in range(m):
    tgt = int(meta.target[j])
    if list(meta.pos4[j]) != [tgt - 4, tgt - 3, tgt - 2, tgt - 1]:
        ok_place = False
    for k in range(4):
        p = int(meta.pos4[j][k])
        dif = np.nonzero(cf_new[p, :92] != cf_new[tgt, :92])[0]
        allowed = {_COL_DUR, _COL_R1, _COL_R1 + 1, _COL_R1 + 2, _COL_R1 + 3}
        if not set(dif.tolist()) <= allowed:
            ok_content = False
        if cf_new[p, _COL_DUR] != torch.tensor(_PROBE_DUR_SCALED, dtype=data.card_features.dtype).float().item():
            ok_content = False
        oh = cf_new[p, _COL_R1:_COL_R1 + 4]
        if oh[k] != 1 or oh.sum() != 1:
            ok_content = False
        if lab_new[p, _LBL_HAS_LABEL] != 0 or not data2.skips[p]:
            ok_labels = False
        if lab_new[p, 0] != lab_new[tgt, 0]:  # label_elapsed_seconds preserved
            ok_labels = False
        for sub in RWKV_SUBMODULES:
            if int(data2.ids[sub][p]) != int(data2.ids[sub][tgt]):
                ok_ids = False
    if lab_new[tgt, _LBL_HAS_LABEL] != 1:
        ok_labels = False
    press = int(meta.pressed[j])
    if cf_new[tgt, _COL_R1 + press] != 1:
        ok_press = False
check("probes contiguous before target", ok_place)
check("probe content: only duration+grade differ, imputed+one-hot correct", ok_content)
check("probe labels: has_label=0, skip, les preserved; target has_label=1", ok_labels)
check("probe ids == target ids (all 5 streams)", ok_ids)
check("pressed == target grade argmax", ok_press)

# repack validity per stream
new_n = n + 4 * m
ok_perm = ok_groups = True
for sub in RWKV_SUBMODULES:
    md = data2.modules[sub]
    fp = md.from_perm.numpy()
    tp = md.to_perm.numpy()
    if sorted(fp.tolist()) != list(range(new_n)) or not (tp[fp] == np.arange(new_n)).all():
        ok_perm = False
    ids = data2.ids[sub].numpy()
    pos = 0
    for l, bcount in zip(md.split_len.tolist(), md.split_B.tolist()):
        for _ in range(bcount):
            seg = fp[pos:pos + l]
            if not (ids[seg] == ids[seg[0]]).all() or not (np.diff(seg) > 0).all():
                ok_groups = False
            pos += l
    if pos != new_n:
        ok_perm = False
    # every entity's rows form exactly one group
    if sum(l * c for l, c in zip(md.split_len.tolist(), md.split_B.tolist())) != new_n:
        ok_perm = False
check("repack: from_perm/to_perm are inverse permutations, sizes add up", ok_perm)
check("repack: groups are single-entity, in ascending row order", ok_groups)

# eligibility: no target may be the in-chunk first REAL occurrence of its card
cards = data.ids["card_id"].numpy()
real_idx = np.nonzero(~sk)[0]
_, fpos = np.unique(cards[real_idx], return_index=True)
firsts = set(real_idx[fpos].tolist())
# map targets back to OLD rows: target new pos - 4*(rank of target among picks by pos)
old_targets = [int(meta.target[j]) - 4 * (j + 1) for j in range(m)]
check("eligibility: no in-chunk-first targets",
      all(t not in firsts for t in old_targets))
check("eligibility: all targets real+labeled",
      all((not sk[t]) and lab[t, _LBL_HAS_LABEL] == 1 for t in old_targets))

# prepare() meta flat consistency (single sample -> base offset 0, but padded T)
batch0 = prepare([data], seed=SEED, probe_density=0.0)
batchP = prepare([data], seed=SEED, probe_density=DENS)
check("batch0 has no probe channel", batch0.probe_rows is None)
check("batchP probe channel shapes",
      batchP.probe_rows is not None and batchP.probe_rows.shape == (m, 4)
      and batchP.probe_target.shape == (m,) and batchP.probe_query.shape == (m,))
gT = batchP.labels.shape[1]
check("flat indices in range", bool((batchP.probe_rows < gT).all()))
# stored card_features = 24 base cols; prepare() appends 40 id-code + 28 cycle dims -> 92
NB = cf_new.shape[1]
sp = batchP.start.float().numpy()
ok_flat = all(
    (sp[int(batchP.probe_rows[j, k])][:NB] == cf_new[int(meta.pos4[j][k])]).all()
    for j in range(min(m, 20)) for k in range(4)
)
check("flat probe_rows point at the probe feature rows", ok_flat)
# STRONGER: in the assembled 92-dim input, probe differs from target ONLY at duration +
# grade (same ids -> same per-batch id codes; same day_offsets -> same cycle encodings)
allowed = {_COL_DUR, _COL_R1, _COL_R1 + 1, _COL_R1 + 2, _COL_R1 + 3}
ok_92 = all(
    set(np.nonzero(sp[int(batchP.probe_rows[j, k])] != sp[int(batchP.probe_target[j])])[0].tolist())
    <= allowed
    for j in range(min(m, 20)) for k in range(4)
)
check("assembled 92-dim probe row differs from target only at duration+grade", ok_92)
# query rows: is_query label set, before the probes
labP = batchP.labels.float().numpy()[0]
ok_q = all(labP[int(batchP.probe_query[j])][6] == 1
           and int(batchP.probe_query[j]) < int(batchP.probe_rows[j, 0])
           for j in range(m))
check("probe_query rows are is_query rows before the probes", ok_q)

if FAIL:
    print(f"{FAIL} FAILURES in part A -- aborting before CUDA part")
    sys.exit(1)

# ---------------- Part B: CUDA invisibility + e2e loss ----------------
if not torch.cuda.is_available():
    print("CUDA unavailable -- part B skipped (NOT a pass)")
    sys.exit(2)

from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG
from rwkv.model.srs_model import SrsRWKV

torch.manual_seed(0)
model = SrsRWKV(DEFAULT_ANKI_RWKV_CONFIG)
model.eval()  # dropout inert
with torch.no_grad():
    model.w_linear.weight.normal_(std=0.5)
    model.w_linear.bias.normal_(std=0.5)
    model.p_linear.weight.normal_(std=0.5)
model = model.to("cuda")

b0 = batch0.to("cuda")
bP = batchP.to("cuda")
# stored features are bf16; the smoke model runs fp32 (train uses selective_cast bf16)
b0.start = b0.start.float()
bP.start = bP.start.float()
with torch.no_grad():
    s0 = model.get_loss(b0)
    # lambda>0 but keep loss graph off for the pure comparison
    sP = model.get_loss(bP)
check("both forwards finite", s0 is not None and sP is not None)

# map: old row t -> new position (recompute like insert_probes)
pick_old = np.array(sorted(old_targets))
is_t = np.zeros(n, dtype=bool)
is_t[pick_old] = True
new_pos_old = np.arange(n) + 4 * np.cumsum(is_t)
c0 = s0.p_curve[0].float().cpu().numpy()[:n]
cP = sP.p_curve[0].float().cpu().numpy()[new_pos_old]
i0 = s0.p_imm[0].float().cpu().numpy()[:n]
iP = sP.p_imm[0].float().cpu().numpy()[new_pos_old]
dc = np.abs(c0 - cP).max()
di = np.abs(i0 - iP).max()
check("INVISIBILITY: real/query curve outputs unchanged", dc < 1e-5, f"max|d|={dc:.3g}")
check("INVISIBILITY: real/query p outputs unchanged", di < 1e-5, f"max|d|={di:.3g}")

# probe outputs live + bounded
pc = sP.p_curve[0].float().cpu().numpy()[batchP.probe_rows.numpy().ravel()]
check("probe curves finite in (0,1)", bool(np.isfinite(pc).all() and (pc > 0).all() and (pc < 1).all()))

# e2e loss with grad
model.train()
sT = model.get_loss(bP)
check("pava stat wired", float(sT.pava_loss_avg) > 0 or float(sT.pava_pool_frac) >= 0)
sT.average_loss.backward()
g = model.pava_theta.grad
check("pava_theta grad exists+finite", g is not None and bool(torch.isfinite(g).all()))
print(f"pava_loss={float(sT.pava_loss_avg):.4f} pool_frac={float(sT.pava_pool_frac):.3f} "
      f"theta_grad={None if g is None else g.tolist()}")

print("ALL_PASS" if FAIL == 0 else f"{FAIL} FAILURES")
sys.exit(1 if FAIL else 0)
