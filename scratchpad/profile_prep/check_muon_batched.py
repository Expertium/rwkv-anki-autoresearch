"""Does RWKV_MUON_BATCHED=1 agree with the per-parameter path, and how many dispatches does it save?

CPU-only, seconds, safe to run while the GPU is busy.

Two questions, both answered by measurement rather than inspection:
  1. AGREEMENT -- batched bmm is equivalent in exact arithmetic but not in float, so this
     reports the actual max |diff| on realistic shapes instead of claiming "equivalent".
  2. DISPATCH COUNT -- the whole point of the change. Counts aten ops both ways.
"""
import os
import sys

sys.path.insert(0, os.getcwd())
import torch  # noqa: E402
from torch.utils._python_dispatch import TorchDispatchMode  # noqa: E402


class OpCounter(TorchDispatchMode):
    def __init__(self):
        self.n = 0
        self.mm = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.n += 1
        if "mm" in str(func) or "bmm" in str(func):
            self.mm += 1
        return func(*args, **(kwargs or {}))


def build(seed=0):
    """A param set shaped like the real model: many repeats of a few shapes."""
    g = torch.Generator().manual_seed(seed)
    shapes = [(80, 80)] * 60 + [(80, 120)] * 40 + [(240, 80)] * 30 + [(80, 4)] * 20 + [(128, 80)] * 27
    ps = []
    for s in shapes:
        p = torch.nn.Parameter(torch.randn(s, generator=g) * 0.02)
        p.grad = torch.randn(s, generator=g) * 0.01
        ps.append(p)
    return ps


def run(batched, seed=0):
    os.environ["RWKV_MUON_BATCHED"] = "1" if batched else "0"
    import importlib

    import rwkv.muon as m
    importlib.reload(m)
    assert m._MUON_BATCHED == batched, "env flag did not take effect after reload"
    ps = build(seed)
    opt = m.MuonAdamW(
        [dict(params=ps, lr=0.02, weight_decay=0.01, use_muon=True, wd_lr_scale=0.05)],
        muon_momentum=0.95, ns_steps=5,
    )
    with OpCounter() as c:
        opt.step()
    return [p.detach().clone() for p in ps], c.n, c.mm


def main():
    print(f"params: 177 matrices over 5 distinct shapes\n")
    ref, n_ref, mm_ref = run(False)
    bat, n_bat, mm_bat = run(True)

    worst = 0.0
    worst_rel = 0.0
    for a, b in zip(ref, bat):
        d = (a - b).abs().max().item()
        scale = a.abs().max().item() or 1.0
        worst = max(worst, d)
        worst_rel = max(worst_rel, d / scale)

    print(f"{'':22s} {'aten ops':>10s} {'mm/bmm':>8s}")
    print(f"{'per-parameter (now)':22s} {n_ref:10d} {mm_ref:8d}")
    print(f"{'batched':22s} {n_bat:10d} {mm_bat:8d}")
    print(f"{'reduction':22s} {n_ref / max(n_bat,1):9.2f}x {mm_ref / max(mm_bat,1):7.2f}x")
    print()
    print(f"max |diff| vs per-param path: {worst:.3e}  (relative {worst_rel:.3e})")
    # bf16 carries ~3 decimal digits; the NS iterate is normalized to ~1, so ~1e-2 relative
    # is the honest bar for "same computation, different reduction order".
    if worst_rel < 1e-2:
        print("AGREE -- within bf16 reduction-order noise, as expected (NOT bit-exact by design)")
    else:
        print("DISAGREE -- larger than bf16 reduction-order noise; this is a BUG, not float drift")


if __name__ == "__main__":
    main()
