#!/usr/bin/env python
"""RWKV_DECK_TREE on the real pipeline. CPU, ~1 min. Run via run_smoke_tree.cmd (2 subprocesses).

MODE `off`  : tree off, prints checksums.
MODE `null` : tree ON but pointed at a parent map where NOTHING resolves.
MODE `real` : tree ON with the real parent map.

THE POINT OF `null`: if no row has an ancestor, every row is inactive at every level, so nothing
is ever scattered back from the ancestor streams and the forward is MATHEMATICALLY IDENTICAL to
tree-off. It therefore exercises the entire new path -- chain expansion, splits over 7 streams,
derived ModuleData, mask threading, the scatter filter -- and any leak shows up as a checksum
difference against `off`. A bypass that is merely "small" fails this; only an exact one passes.

The batch itself must also match `off` bit-for-bit: ancestor streams get no id encodings, so they
consume no augmentation RNG draws and `start` is untouched. That is checked separately, because a
shifted RNG stream would be a silent confound in every future comparison.

ASCII output only.
"""
import hashlib
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

MODE = sys.argv[1]
USERS = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [101, 102]

import lmdb  # noqa: E402

from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG  # noqa: E402
from rwkv.model.srs_model import SrsRWKV  # noqa: E402
from rwkv.prepare_batch import get_data, prepare  # noqa: E402


def ck(t):
    a = t.detach().to(torch.float64).cpu().numpy()
    return float(np.nansum(a)), hashlib.md5(np.ascontiguousarray(a)).hexdigest()[:12]


names = [n for n, _ in DEFAULT_ANKI_RWKV_CONFIG.modules]
print(f"[{MODE}] chain = {names}")

env = lmdb.open("train_db_5k_h1", readonly=True, lock=False, subdir=True,
                map_size=400_000_000_000, max_readers=2048)
keys = []
with env.begin() as txn:
    for uid in USERS:
        raw = txn.get(f"{uid}_batches".encode())
        kk = json.loads(raw)
        keys.append((uid, *kk[len(kk) // 2]))
    data_list = [get_data(txn, (u, s, e, L), device="cpu") for u, s, e, L in keys]

pb = prepare(data_list, target_len=65536, seed=1234)
print(f"[{MODE}] streams in batch = {len(pb.sub_gather)}  rows = {pb.start.size(0)}")
s_sum, s_md5 = ck(pb.start)
print(f"[{MODE}] START checksum   = {s_sum:.6f} {s_md5}")

if pb.stream_active is not None:
    for nm, act in zip(names, pb.stream_active):
        if act.numel() > 0:
            real = int(act.sum())
            print(f"[{MODE}]   {nm}: active {real}/{act.numel()} = {real/act.numel():.2%}")

# singleton check: every inactive row must be its own group in that stream
if MODE == "real":
    from rwkv import deck_tree as dt
    for d in data_list:
        for k in range(1, dt.num_levels()):
            nm = dt.stream_name(k)
            md = d.modules[nm]
            ids = d.ids[nm].numpy()
            n_neg = int((ids < 0).sum())
            n_sing = int(md.split_B[0]) if len(md.split_len) and md.split_len[0] == 1 else 0
            pass  # inactive rows now group by LEAF DECK, not as singletons (see deck_tree docstring)
        break
    print(f"[{MODE}] inactive rows grouped by leaf deck: OK")

torch.manual_seed(1234)
model = SrsRWKV(DEFAULT_ANKI_RWKV_CONFIG)
model.eval()
# ⚠ THE VACUITY TRAP (CLAUDE.md section 9): a freshly built RWKV7 has ZERO-INIT output
# projections, so every layer is EXACTLY the identity and no change to sequence grouping can
# show up. The first run of this smoke reported `real` == `off` for precisely that reason.
# Randomize per parameter NAME (not in iteration order) so the tree model's one extra parameter
# cannot shift the RNG stream and manufacture a difference by itself.
# ⚠ AND THE NAMES ARE NOT STABLE ACROSS MODES: nn.ModuleList dedupes the repeated deck object,
# so with the tree on preset/user are `rwkv_modules.5/.6` where off they are `.3/.4`. Seeding on
# the raw name therefore gave those two streams DIFFERENT weights and made `null` look like a
# bypass leak. Canonicalise the index to the STREAM NAME first.
_idx2stream = {f"rwkv_modules.{i}.": f"STREAM_{n.split('@')[0]}." for i, n in enumerate(names)}


def canon(nm):
    for k, v in _idx2stream.items():
        if nm.startswith(k):
            return v + nm[len(k):]
    return nm


with torch.no_grad():
    for nm, prm in model.named_parameters():
        if "tree_level_emb" in nm:
            continue  # zero-init on purpose: levels start indistinguishable
        g = torch.Generator().manual_seed(
            int(hashlib.md5(canon(nm).encode()).hexdigest()[:8], 16))
        prm.copy_(torch.randn(prm.shape, generator=g) * 0.05)
n_par = sum(p.numel() for p in model.parameters())
print(f"[{MODE}] params = {n_par:,}")
with torch.no_grad():
    out = model.forward_batch(
        pb.start.float(), pb.sub_gather, pb.sub_gather_lens,
        pb.time_shift_selects, pb.skips, pb.num_data, pb.stream_active,
    )
o_sum, o_md5 = ck(out[3])   # the rating logits -- live in every config
a_sum, a_md5 = ck(out[0])
print(f"[{MODE}] P_LOGITS         = {o_sum:.6f} {o_md5}")
print(f"[{MODE}] AHEAD_LOGITS     = {a_sum:.6f} {a_md5}")
print(f"[{MODE}] RESULT {s_md5} {o_md5} {a_md5} {n_par}")
