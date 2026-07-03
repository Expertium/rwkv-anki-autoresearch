"""Golden-output generator for the qat_lr kernel speedup (skip-elision + parallel PQ + warp power-loop).

Runs the CURRENT build's rwkv7_wkv_qat_lr_{forward,backward}_float on fixed seeded inputs (realistic
skip pattern: t=0 never skipped, ~50% query-style skips) for BOTH the int-N path and the PQ path,
and saves every output tensor. After the kernel edit + rebuild, run with `check` to compare BIT-EXACTLY.

Usage:
  python scratchpad/qat_speed/golden_gen.py gen    -> writes scratchpad/qat_speed/golden.pt
  python scratchpad/qat_speed/golden_gen.py check  -> reruns and torch.equal-compares vs golden.pt
"""
import os
import sys

import torch

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, ROOT)
import rwkv, rwkv.model  # noqa: registers torch.ops.rwkv

GOLDEN = os.path.join(os.path.dirname(__file__), "golden.pt")
SHAPES = [(48, 91, 2, 16), (4, 1100, 2, 16)]  # short-T many-B (card-like) + multi-chunk long-T
QMAX = 7.0  # int4


def make_inputs(B, T, H, K, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    def rnd(*shape, scale=1.0):
        return (torch.randn(*shape, generator=g) * scale).float().cuda().contiguous()
    r = rnd(B, T, H, K, scale=0.5)
    k = rnd(B, T, H, K, scale=0.5)
    v = rnd(B, T, H, K, scale=0.5)
    # w: decay in (0,1); a: in (0,1); kd: unit-ish direction entries
    w = torch.rand(B, T, H, K, generator=g).float().mul(0.6).add(0.35).cuda().contiguous()
    a = torch.rand(B, T, H, K, generator=g).float().cuda().contiguous()
    kd = rnd(B, T, H, K, scale=0.3)
    skip = (torch.rand(B, T, generator=g) < 0.5)
    skip[:, 0] = False  # "Cannot skip the start"
    skip = skip.cuda().contiguous()
    grad = rnd(B, T, H, K, scale=0.5)
    return r, k, v, w, a, kd, skip, grad


def load_codebook():
    path = os.path.join(ROOT, "reference", "pq_cb_m2b8.txt")
    with open(path) as fh:
        lines = [ln for ln in fh if ln.strip()]
    m, bits, sub, kk, ncent = (int(x) for x in lines[0].split()[:5])
    rows = [[float(x) for x in ln.split()] for ln in lines[1:]]
    flat = []
    idx = 0
    for role in range(4):
        vals = rows[idx:idx + m * ncent]
        idx += m * ncent
        if role < 2:
            flat.extend(x for row in vals for x in row)
    cbt = torch.tensor(flat, dtype=torch.float32, device="cuda")
    return cbt, m, sub, ncent


def run_all():
    out = {}
    for pq in (False, True):
        if pq:
            cbt, m, sub, ncent = load_codebook()
            torch.ops.rwkv.rwkv7_set_pq_codebook(cbt, m, sub, ncent)
        else:
            torch.ops.rwkv.rwkv7_set_pq_codebook(torch.zeros(1, device="cuda"), 0, 0, 0)
        for si, (B, T, H, K) in enumerate(SHAPES):
            r, k, v, w, a, kd, skip, grad = make_inputs(B, T, H, K, seed=1234 + si)
            fwd_out, ckpt = torch.ops.rwkv.rwkv7_wkv_qat_lr_forward_float.default(
                r, k, v, w, a, kd, skip, QMAX)
            grads = torch.ops.rwkv.rwkv7_wkv_qat_lr_backward_float.default(
                r, k, v, w, a, kd, skip, ckpt, grad, QMAX)
            tag = f"{'pq' if pq else 'int'}_s{si}"
            out[f"{tag}_out"] = fwd_out.cpu()
            out[f"{tag}_ckpt"] = ckpt.cpu()
            for gi, gt in enumerate(grads):
                out[f"{tag}_grad{gi}"] = gt.cpu()
    torch.cuda.synchronize()
    return out


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "gen"
    res = run_all()
    if mode == "gen":
        torch.save(res, GOLDEN)
        print(f"GOLDEN saved: {len(res)} tensors -> {GOLDEN}")
        for name in sorted(res):
            t = res[name]
            print(f"  {name}: {tuple(t.shape)} finite={torch.isfinite(t).all().item()}")
    else:
        gold = torch.load(GOLDEN, weights_only=True)
        bad = []
        for name in sorted(gold):
            if name not in res:
                bad.append(f"{name}: MISSING")
                continue
            if not torch.equal(gold[name], res[name]):
                d = (gold[name] - res[name]).abs()
                nz = (gold[name] != res[name]).sum().item()
                bad.append(f"{name}: MISMATCH n={nz} maxabs={d.max().item():.3e}")
        if bad:
            print("BITEXACT_FAIL")
            for b in bad:
                print(" ", b)
            sys.exit(1)
        print(f"BITEXACT_PASS all {len(gold)} tensors identical")


if __name__ == "__main__":
    main()
