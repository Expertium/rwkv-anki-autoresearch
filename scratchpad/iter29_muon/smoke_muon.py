"""Iter-29 Muon smoke (CPU, standalone — no GPU/pipeline needed).

A. Adam-delegation bit-exactness: MuonAdamW with use_muon=False everywhere must track
   torch.optim.AdamW bit-for-bit over 50 steps on a toy model (validates the functional
   adamw call signature + state layout on THIS torch version).
B. Muon path: use_muon=True on the matrix groups -> loss decreases on a toy regression,
   momentum buffers exist, orthogonalization returns finite values, wd applied.
C. state_dict round-trip: save + load mid-run, trajectories identical afterward.
"""
import copy
import os
import sys

import torch

sys.path.insert(0, os.getcwd())
from rwkv.muon import MuonAdamW, zeropower_via_newtonschulz5

torch.manual_seed(0)


def make_model():
    return torch.nn.Sequential(
        torch.nn.Linear(16, 64), torch.nn.Tanh(), torch.nn.Linear(64, 8)
    )


def groups_for(model, use_muon):
    mats, rest = [], []
    for n, p in model.named_parameters():
        (mats if p.ndim >= 2 else rest).append(p)
    g = [
        {"params": mats, "weight_decay": 0.01, "lr": 1e-3},
        {"params": rest, "weight_decay": 0.0, "lr": 1e-3},
    ]
    if use_muon:
        g[0]["use_muon"] = True
        g[0]["wd_lr_scale"] = 1e-3 / 0.02
        g[0]["lr"] = 0.02
    return g


X = torch.randn(256, 16)
Y = torch.randn(256, 8)


def run(model, opt, steps=50):
    losses = []
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(X), Y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


# --- A: bit-exact Adam delegation -------------------------------------------------
m1 = make_model()
m2 = copy.deepcopy(m1)
o1 = torch.optim.AdamW(groups_for(m1, False), betas=(0.9, 0.999), eps=1e-18)
o2 = MuonAdamW(groups_for(m2, False), betas=(0.9, 0.999), eps=1e-18)
run(m1, o1)
run(m2, o2)
max_d = max((p1 - p2).abs().max().item() for p1, p2 in zip(m1.parameters(), m2.parameters()))
print(f"A. adam-delegation max param diff after 50 steps: {max_d:.3e}")
assert max_d == 0.0, "Adam delegation is not bit-exact"

# --- B: Muon path trains ----------------------------------------------------------
m3 = make_model()
o3 = MuonAdamW(groups_for(m3, True), betas=(0.9, 0.999), eps=1e-18)
losses = run(m3, o3, steps=100)
print(f"B. muon loss step1={losses[0]:.4f} step100={losses[-1]:.4f}")
assert losses[-1] < losses[0] * 0.7, "Muon path failed to train"
assert all(torch.isfinite(p).all() for p in m3.parameters())
n_bufs = sum(1 for p in m3.parameters() if "momentum_buffer" in o3.state.get(p, {}))
print(f"   momentum buffers: {n_bufs} (expect 2 matrices)")
assert n_bufs == 2
O = zeropower_via_newtonschulz5(torch.randn(64, 16))
sv = torch.linalg.svdvals(O.float())
print(f"   NS5 singular values of a random 64x16: min={sv.min():.3f} max={sv.max():.3f} (want ~1)")
assert 0.5 < sv.min() and sv.max() < 1.6

# --- C: state_dict round-trip -----------------------------------------------------
m4 = make_model()
o4 = MuonAdamW(groups_for(m4, True), betas=(0.9, 0.999), eps=1e-18)
run(m4, o4, steps=20)
sd_m, sd_o = copy.deepcopy(m4.state_dict()), copy.deepcopy(o4.state_dict())
ref = run(m4, o4, steps=10)
m5 = make_model()
m5.load_state_dict(sd_m)
o5 = MuonAdamW(groups_for(m5, True), betas=(0.9, 0.999), eps=1e-18)
o5.load_state_dict(sd_o)
res = run(m5, o5, steps=10)
print(f"C. resume trajectory max loss diff: {max(abs(a - b) for a, b in zip(ref, res)):.3e}")
assert ref == res, "state_dict round-trip diverged"
print("SMOKE_ALL_PASS")
