"""Iter-30 cautious-wd smoke (CPU, standalone).

A. cautious_wd=False single-step EXACTNESS: MuonAdamW's Muon branch must equal the
   hand-computed original update p' = p*(1-wd_eff) - lr*scale*NS5(nesterov_upd)
   bit-for-bit (proves the O_full refactor changed nothing on the champion path).
B. cautious_wd=True single-step EXACTNESS + mask semantics: p' must equal
   p*(1 - wd_eff*(p*O<0)) - lr*scale*O; mask fraction ~0.5 on random data.
C. cautious ON toy training: loss decreases, params finite.
D. state_dict round-trip with the new group key: trajectories identical after resume.
"""
import copy
import os
import sys

import torch

sys.path.insert(0, os.getcwd())
from rwkv.muon import MuonAdamW, zeropower_via_newtonschulz5

torch.manual_seed(0)


def one_step_check(cautious):
    torch.manual_seed(42)
    p0 = torch.randn(64, 32)
    g = torch.randn(64, 32)
    lr, wd, momentum, wd_lr_scale = 0.02, 0.01, 0.95, 1e-3 / 0.02

    p = p0.clone().requires_grad_(True)
    opt = MuonAdamW([{ "params": [p], "lr": lr, "weight_decay": wd,
                       "use_muon": True, "wd_lr_scale": wd_lr_scale,
                       "cautious_wd": cautious }])
    p.grad = g.clone()
    opt.step()

    # hand-computed reference (original ordering)
    buf = g.clone()                       # zeros.mul(m).add(g)
    upd = g.add(buf, alpha=momentum)      # nesterov
    O = zeropower_via_newtonschulz5(upd)
    wd_eff = lr * wd_lr_scale * wd
    ref = p0.clone()
    if cautious:
        mask = (p0 * O) < 0
        ref.mul_(1.0 - wd_eff * mask.to(ref.dtype))
        frac = mask.float().mean().item()
        assert 0.2 < frac < 0.8, f"mask fraction {frac} implausible"
        print(f"   cautious mask fraction: {frac:.3f}")
    else:
        ref.mul_(1.0 - wd_eff)
    scale = max(1.0, 64 / 32) ** 0.5
    ref.add_(O, alpha=-lr * scale)
    d = (p.detach() - ref).abs().max().item()
    print(f"{'B' if cautious else 'A'}. cautious={cautious} single-step max diff: {d:.3e}")
    assert d == 0.0, "update formula mismatch"


one_step_check(False)
one_step_check(True)


def make_model():
    return torch.nn.Sequential(
        torch.nn.Linear(16, 64), torch.nn.Tanh(), torch.nn.Linear(64, 8)
    )


def groups_for(model):
    mats, rest = [], []
    for n, p in model.named_parameters():
        (mats if p.ndim >= 2 else rest).append(p)
    return [
        {"params": mats, "weight_decay": 0.01, "lr": 0.02,
         "use_muon": True, "wd_lr_scale": 1e-3 / 0.02, "cautious_wd": True},
        {"params": rest, "weight_decay": 0.0, "lr": 1e-3},
    ]


X = torch.randn(256, 16)
Y = torch.randn(256, 8)


def run(model, opt, steps):
    losses = []
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(X), Y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


m = make_model()
o = MuonAdamW(groups_for(m), betas=(0.9, 0.999), eps=1e-18)
losses = run(m, o, 100)
print(f"C. cautious training loss step1={losses[0]:.4f} step100={losses[-1]:.4f}")
assert losses[-1] < losses[0] * 0.7
assert all(torch.isfinite(p).all() for p in m.parameters())

m2 = make_model()
o2 = MuonAdamW(groups_for(m2), betas=(0.9, 0.999), eps=1e-18)
run(m2, o2, 20)
sd_m, sd_o = copy.deepcopy(m2.state_dict()), copy.deepcopy(o2.state_dict())
ref = run(m2, o2, 10)
m3 = make_model()
m3.load_state_dict(sd_m)
o3 = MuonAdamW(groups_for(m3), betas=(0.9, 0.999), eps=1e-18)
o3.load_state_dict(sd_o)
res = run(m3, o3, 10)
print(f"D. resume trajectory max loss diff: {max(abs(a - b) for a, b in zip(ref, res)):.3e}")
assert ref == res
print("SMOKE_ALL_PASS")
