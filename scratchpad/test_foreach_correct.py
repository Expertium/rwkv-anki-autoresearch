"""Prove the foreach-vectorized copy_downcast_ / transfer_child_grad_to_master are BIT-IDENTICAL
to the original per-param loops. CPU, deterministic, no training needed."""
import os
os.environ["RWKV_NO_JIT"] = "1"
import torch
from rwkv.model.srs_model import SrsRWKV, is_excluded
from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG
from rwkv.train_rwkv import transfer_child_grad_to_master

torch.manual_seed(0)
dtype = torch.bfloat16


def fresh():
    master = SrsRWKV(DEFAULT_ANKI_RWKV_CONFIG)
    child = SrsRWKV(DEFAULT_ANKI_RWKV_CONFIG).selective_cast(dtype)
    with torch.no_grad():
        for p in master.parameters():
            p.copy_(torch.randn_like(p))
    return master, child


# ---- copy_downcast_ : new (foreach) vs old (loop) ----
master, child = fresh()
child.copy_downcast_(master, dtype=dtype)          # NEW foreach
mp = dict(master.named_parameters())
mism = 0
for name, param in child.named_parameters():       # OLD loop, recomputed for comparison
    target = torch.float32 if is_excluded(name) else dtype
    expect = mp[name].data.to(target)
    if not torch.equal(param.data, expect):
        mism += 1
print(f"copy_downcast_: {mism} mismatched params (expect 0)")

# ---- transfer_child_grad_to_master : new (foreach) vs old (loop) ----
masterA, childA = fresh()
masterB, childB = fresh()
# identical child grads in both
with torch.no_grad():
    gA = {n: torch.randn_like(p) for n, p in childA.named_parameters()}
    for n, p in childA.named_parameters():
        p.grad = gA[n].clone()
    for n, p in childB.named_parameters():
        p.grad = gA[n].clone()
    # pre-seed some master grads to exercise the accumulate path
    mpA = dict(masterA.named_parameters()); mpB = dict(masterB.named_parameters())
    seed = {n: torch.randn_like(p) for n, p in masterA.named_parameters()}
    for n in seed:
        mpA[n].grad = seed[n].clone()
        mpB[n].grad = seed[n].clone()

transfer_child_grad_to_master(masterA, childA)     # NEW foreach

# OLD loop on B
for name, param in childB.named_parameters():
    master_param = mpB[name]
    if param.grad is not None:
        with torch.no_grad():
            if master_param.grad is None:
                master_param.grad = torch.zeros_like(master_param, requires_grad=True)
            master_param.grad.add_(param.grad.to(torch.float32))
        param.grad.zero_()

gmism = cmism = 0
for n in mpA:
    if not torch.equal(mpA[n].grad, mpB[n].grad):
        gmism += 1
for (n, pa), (_, pb) in zip(childA.named_parameters(), childB.named_parameters()):
    if not torch.equal(pa.grad, pb.grad):
        cmism += 1
print(f"transfer_grad: {gmism} master-grad mismatches, {cmism} child-grad-zero mismatches (expect 0, 0)")
print("OK" if (mism == 0 and gmism == 0 and cmism == 0) else "FAIL")
