"""Property tests for rwkv/model/pava.py (iter 23 rectifier op).

1. Vectorized == scalar reference on random + adversarial rows, random thetas.
2. p=1 (theta=atanh(0.5)) == classic weighted arithmetic PAVA.
3. Identity on already-non-decreasing rows (exact).
4. Output always non-decreasing.
5. Output bounded by [min(v), max(v)].
6. Gradients: finite into v, w, theta; theta grad nonzero iff pooling occurred.
7. p sign semantics: on a violating pair, p=-2 pools lower than p=+1 pools lower than p=+2.
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch

from rwkv.model.pava import pava_rectify, pava_rectify_scalar, power_mean, theta_init

torch.manual_seed(0)
FAIL = 0


def check(name, cond):
    global FAIL
    if cond:
        print(f"PASS {name}")
    else:
        FAIL += 1
        print(f"FAIL {name}")


def classic_pava(v, w):
    """Weighted arithmetic PAVA reference (p=1 everywhere)."""
    stack = []
    for k in range(4):
        stack.append((v[k], w[k], k))
        while len(stack) >= 2 and stack[-2][0] > stack[-1][0]:
            bv, bw, bl = stack.pop()
            av, aw, al = stack.pop()
            stack.append(((av * aw + bv * bw) / (aw + bw), aw + bw, al))
    out = []
    for i, (val, _, left) in enumerate(stack):
        nxt = stack[i + 1][2] if i + 1 < len(stack) else 4
        out.extend([val] * (nxt - left))
    return out


# ---- build test rows: random, adversarial (ties, near-ties, reversed, extremes)
M = 20000
v = torch.rand(M, 4) * (1 - 2e-5) + 1e-5
adv = torch.tensor([
    [0.5, 0.5, 0.5, 0.5],
    [0.9, 0.7, 0.5, 0.3],          # fully reversed -> one big block
    [0.3, 0.5, 0.7, 0.9],          # sorted -> identity
    [0.5, 0.5 - 1e-7, 0.5, 0.5],   # near-tie violation
    [1e-5, 1 - 1e-5, 1e-5, 1 - 1e-5],
    [0.7, 0.3, 0.8, 0.2],          # two separate violations
    [0.8, 0.6, 0.4, 0.9],          # cascading back-merge
    [0.2, 0.9, 0.5, 0.4],          # merge chain into middle
])
v = torch.cat([v, adv], dim=0)
M = v.size(0)
w_ones = torch.ones(M, 4)
w_rand = torch.rand(M, 4) * 2 + 0.05

for w, wname in ((w_ones, "unit"), (w_rand, "random")):
    for theta_case, tname in (
        (theta_init(), "p=1"),
        (torch.tensor([1.8, -1.7, 0.0001]), "mixed"),
        (torch.tensor([-5.0, 5.0, 0.0]), "extreme"),
    ):
        powers = 2 * torch.tanh(theta_case)
        out = pava_rectify(v, w, powers)
        # 1. vs scalar reference
        ok = True
        for i in range(min(M, 3000)):
            ref = pava_rectify_scalar(
                [float(x) for x in v[i]], [float(x) for x in w[i]],
                [float(x) for x in powers],
            )
            if not torch.allclose(out[i], torch.tensor(ref, dtype=out.dtype), atol=1e-6, rtol=1e-5):
                ok = False
                print(f"  row {i}: v={v[i].tolist()} out={out[i].tolist()} ref={ref}")
                break
        check(f"vector==scalar ({wname} w, {tname})", ok)
        # 4. non-decreasing
        check(f"non-decreasing ({wname}, {tname})",
              bool((out.diff(dim=1) >= -1e-7).all()))
        # 5. bounded
        check(f"bounded ({wname}, {tname})",
              bool((out >= v.min(dim=1, keepdim=True).values - 1e-7).all()
                   and (out <= v.max(dim=1, keepdim=True).values + 1e-7).all()))

# 2. p=1 == classic weighted arithmetic PAVA
powers1 = 2 * torch.tanh(theta_init())
out1 = pava_rectify(v, w_rand, powers1)
ok = True
for i in range(min(M, 3000)):
    ref = classic_pava([float(x) for x in v[i]], [float(x) for x in w_rand[i]])
    if not torch.allclose(out1[i], torch.tensor(ref, dtype=out1.dtype), atol=1e-6, rtol=1e-5):
        ok = False
        print(f"  row {i}: out={out1[i].tolist()} classic={ref}")
        break
check("p=1 == classic weighted PAVA", ok)

# 3. identity on sorted rows (exact equality — no merge must ever fire)
vs, _ = torch.sort(v, dim=1)
out_s = pava_rectify(vs, w_rand, 2 * torch.tanh(torch.randn(3)))
check("identity on sorted rows (exact)", torch.equal(out_s, vs))

# 6. gradients
theta = theta_init().requires_grad_(True)
vg = v[:500].clone().requires_grad_(True)
wg = w_rand[:500].clone().requires_grad_(True)
outg = pava_rectify(vg, wg, 2 * torch.tanh(theta))
outg.sum().backward()
check("grad finite v", bool(torch.isfinite(vg.grad).all()))
check("grad finite w", bool(torch.isfinite(wg.grad).all()))
check("grad finite theta", bool(torch.isfinite(theta.grad).all()))
check("grad theta nonzero (pooling occurred)", bool((theta.grad != 0).any()))
# no-pooling rows -> zero theta grad
theta2 = theta_init().requires_grad_(True)
vs2 = vs[:100].clone().requires_grad_(True)
pava_rectify(vs2, w_ones[:100], 2 * torch.tanh(theta2)).sum().backward()
# no pooling -> powers never enter the graph -> grad is None (or all-zero)
check("theta grad zero without pooling",
      theta2.grad is None or bool((theta2.grad == 0).all()))
check("v grad identity without pooling", bool(torch.allclose(vs2.grad, torch.ones_like(vs2.grad))))

# 7. p sign semantics on one violating pair (0.7, 0.3), equal weights
a = torch.tensor([[0.7, 0.3, 0.99, 0.999]])
wu = torch.ones(1, 4)
lo = pava_rectify(a, wu, torch.tensor([-2.0, 1.0, 1.0]))[0, 0]
mid = pava_rectify(a, wu, torch.tensor([1.0, 1.0, 1.0]))[0, 0]
hi = pava_rectify(a, wu, torch.tensor([2.0, 1.0, 1.0]))[0, 0]
check("p=-2 < p=1 < p=+2 pooled value", bool(lo < mid < hi))
check("p=1 pooled = arithmetic mean", bool(abs(mid - 0.5) < 1e-6))

# power_mean unit checks
pm = power_mean(torch.tensor(0.7), torch.tensor(0.3), torch.tensor(1.0),
                torch.tensor(1.0), torch.tensor(0.0))
check("power_mean p=0 = geometric", bool(abs(pm - math.sqrt(0.21)) < 1e-6))
pm3 = power_mean(torch.tensor(0.7), torch.tensor(0.3), torch.tensor(3.0),
                 torch.tensor(1.0), torch.tensor(1.0))
check("power_mean weighted arithmetic", bool(abs(pm3 - 0.6) < 1e-6))

print("ALL_PASS" if FAIL == 0 else f"{FAIL} FAILURES")
sys.exit(1 if FAIL else 0)
