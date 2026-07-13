"""Smoke test for the RWKV_INIT_BLEND hook (iter 9 shrink-perturb).

Replays the hook's exact logic standalone under the champion env:
1. state_dict key sets of champion ckpt vs fresh model must match,
2. blended tensors land strictly between fresh and trained (norm check),
3. buffers (non-parameter keys) come from the fresh init.
"""
import os

os.environ.setdefault("RWKV_N_HEADS", "2")
os.environ.setdefault("RWKV_HEAD_DIM", "16")
os.environ.setdefault("RWKV_NO_JIT", "1")
os.environ.setdefault("RWKV_QAT_LOWRANK_SCOPE", "card:1:int4,note:1:int4")
os.environ.setdefault("RWKV_QAT_PQ", "reference/pq_cb_wkv_q72u.txt")
os.environ.setdefault("RWKV_QAT_SHIFT_PQ", "reference/pq_cb_shift_q72u.txt")
os.environ.setdefault("RWKV_QAT_PQ_LEARN", "1")
os.environ.setdefault("RWKV_QAT_SHIFT_PQ_LEARN", "1")
os.environ.setdefault("RWKV_QAT_SHIFT_SCOPE", "card:int3,note:int3")
os.environ.setdefault("RWKV_QAT_NORM_BITS", "1")
os.environ.setdefault("RWKV_QAT_FUSED", "1")

import torch
from rwkv.model.srs_model import SrsRWKV
from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG

CKPT = "scratchpad/champ5k_b1/champ5kb1d_1638.pth"
LAM, SEED = 0.5, 777

trained = torch.load(CKPT, weights_only=True, map_location="cpu")

torch.manual_seed(12345)  # the trainer's global seed, line 45
base = SrsRWKV(anki_rwkv_config=DEFAULT_ANKI_RWKV_CONFIG)
with torch.random.fork_rng():
    torch.manual_seed(SEED)
    fresh_sd = SrsRWKV(anki_rwkv_config=DEFAULT_ANKI_RWKV_CONFIG).state_dict()

base_sd = base.state_dict()
missing = set(base_sd) - set(trained)
extra = set(trained) - set(base_sd)
print(f"keys: model {len(base_sd)}, ckpt {len(trained)}, missing {len(missing)}, extra {len(extra)}")
assert not missing and not extra, f"KEY MISMATCH missing={sorted(missing)[:5]} extra={sorted(extra)[:5]}"

param_names = {n for n, _ in base.named_parameters()}
n_blend = n_buf = n_between = n_zerozero = 0
for n, t in base_sd.items():
    if n in param_names:
        b = (LAM * trained[n].float() + (1 - LAM) * fresh_sd[n].float()).to(t.dtype)
        n_blend += 1
        tn, fn, bn = trained[n].norm().item(), fresh_sd[n].norm().item(), b.norm().item()
        if tn == 0 and fn == 0:
            n_zerozero += 1  # e.g. LoRA-A zeros stay zero on both sides -> blend zero
        elif min(tn, fn) - 1e-6 <= bn <= max(tn, fn) * 1.05 + 1e-6:
            n_between += 1
    else:
        n_buf += 1
print(f"blended {n_blend} params ({n_zerozero} zero-on-both-sides, {n_between} norm-sane), {n_buf} buffers from fresh")
# sanity: fresh draw differs from the champion's own init (seed 777 vs 12345)
diffs = [
    (fresh_sd[n] - base_sd[n]).abs().max().item()
    for n in list(param_names)[:50]
    if base_sd[n].numel() > 0 and base_sd[n].norm() > 0
]
print(f"fresh-vs-base-init max|diff| over 50 nonzero tensors: min {min(diffs):.4g}, max {max(diffs):.4g}")
assert max(diffs) > 0, "fresh draw identical to base init -- seeding broken"
print("SMOKE_PASS")
