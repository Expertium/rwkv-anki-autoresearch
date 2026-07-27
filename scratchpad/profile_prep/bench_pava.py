"""How expensive is pava_rectify, structurally? (CPU-only, zero GPU, safe during a live run.)

The GPU cost of this function is dominated by things a kernel-time profile cannot see:
  * ~15 tiny tensor ops per back-merge iteration, up to 6 iterations => a few hundred kernel
    LAUNCHES on an (M,4) tensor that does almost no arithmetic;
  * `bool(merge.any())` (pava.py:91) forces a device->host SYNC once per iteration.

Both are measurable structurally on CPU: count the dispatched aten ops (each is one GPU launch)
and count how many back-merge iterations actually run before the early `break`.

Usage: python scratchpad/profile_prep/bench_pava.py
"""
import os
import sys
import time
import torch
from torch.utils._python_dispatch import TorchDispatchMode

sys.path.insert(0, os.getcwd())
from rwkv.model.pava import pava_rectify, theta_init  # noqa: E402


class OpCounter(TorchDispatchMode):
    """Counts dispatched aten ops -- on CUDA each of these is a kernel launch."""

    def __init__(self):
        self.n = 0
        self.per_op = {}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = str(func)
        self.n += 1
        self.per_op[name] = self.per_op.get(name, 0) + 1
        return func(*args, **(kwargs or {}))


def make_batch(M, pool_frac, seed=0):
    """(M,4) values whose rows violate monotonicity at rate ~pool_frac (what the traces report)."""
    g = torch.Generator().manual_seed(seed)
    v = torch.rand((M, 4), generator=g) * 0.8 + 0.1
    ordered = v.sort(dim=1).values
    keep = torch.rand((M, 1), generator=g) > pool_frac  # rows already monotone -> no merging
    return torch.where(keep, ordered, v)


def main():
    powers = 2.0 * torch.tanh(theta_init())
    print(f"powers (2*tanh(theta_init)) = {powers.tolist()}")
    print()
    print(f"{'M':>8s} {'pool_frac':>10s} {'aten ops':>9s} {'ops/call':>9s} {'cpu ms':>9s}"
          f" {'est GPU launch ms @6us':>23s}")
    for M in (1_000, 10_000, 100_000):
        for pf in (0.10, 0.98):
            v = make_batch(M, pf)
            w = torch.ones_like(v)
            with OpCounter() as c:
                pava_rectify(v, w, powers)
            ops = c.n
            t0 = time.perf_counter()
            reps = 20
            for _ in range(reps):
                pava_rectify(v, w, powers)
            dt = (time.perf_counter() - t0) / reps * 1e3
            print(f"{M:8d} {pf:10.2f} {ops:9d} {ops:9d} {dt:9.3f} {ops * 6e-3:23.2f}")

    # Which ops dominate the launch count, and how many syncs?
    v = make_batch(100_000, 0.98)
    w = torch.ones_like(v)
    with OpCounter() as c:
        pava_rectify(v, w, powers)
    print("\ntop dispatched ops (each = one GPU kernel launch):")
    for k in sorted(c.per_op, key=lambda x: -c.per_op[x])[:10]:
        print(f"  {c.per_op[k]:4d}  {k}")

    # count back-merge iterations actually executed at realistic pooling rates
    print("\nback-merge iterations actually executed (the `break` at pava.py:92):")
    for pf in (0.10, 0.50, 0.98):
        v = make_batch(50_000, pf)
        w = torch.ones_like(v)
        iters = 0
        slots = torch.arange(4)
        bv, bw = v.clone(), w.clone()
        lp = slots.expand(v.size(0), 4).clone()
        for k in range(1, 4):
            for _ in range(k):
                lk = lp[:, k]
                j_c = (lk - 1).clamp(min=0)
                left_v = bv.gather(1, j_c.unsqueeze(1)).squeeze(1)
                merge = (lk > 0) & (left_v > bv[:, k])
                if not bool(merge.any()):
                    break
                iters += 1
                left_w = bw.gather(1, j_c.unsqueeze(1)).squeeze(1)
                from rwkv.model.pava import power_mean
                m_val = power_mean(left_v, bv[:, k], left_w, bw[:, k], powers[j_c])
                new_l = lp.gather(1, j_c.unsqueeze(1)).squeeze(1)
                rng = (slots.unsqueeze(0) >= new_l.unsqueeze(1)) & (slots.unsqueeze(0) <= k)
                upd = merge.unsqueeze(1) & rng
                bv = torch.where(upd, m_val.unsqueeze(1), bv)
                bw = torch.where(upd, (left_w + bw[:, k]).unsqueeze(1), bw)
                lp = torch.where(upd, new_l.unsqueeze(1), lp)
        print(f"  pool_frac {pf:.2f}: {iters}/6 iterations ran -> {iters} device syncs per call")


if __name__ == "__main__":
    main()
