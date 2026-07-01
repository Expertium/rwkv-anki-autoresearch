"""Parity test for the K-dynamic WKV kernel (K=16 support). Compares RWKV7_WKV.apply (the CUDA kernel)
vs reference_rwkv7 (pure-PyTorch, K-agnostic) for FORWARD + BACKWARD grads, at:
  - K=32 (must still match -> proves the change is a no-op at K=32, no regression)
  - K=16 (the NEW path: H=2 heads of 16, d_model=32 -- same width as the champion)
and for BOTH kernel paths: time-parallel (T=2000 >= 3*512) and sequential (T=200).
fp32 (kernel float path vs fp32 reference). Run: PYTHONPATH=. .venv\\Scripts\\python.exe scratchpad/test_k16_wkv.py
"""
import torch
import rwkv.model  # registers torch.ops.rwkv (loads the compiled kernel)
from rwkv.model.rwkv_ops import RWKV7_WKV, reference_rwkv7

torch.manual_seed(0)
dev = "cuda"


def make_inputs(B, T, H, K):
    r = 0.5 * torch.randn(B, T, H, K, device=dev)
    v = 0.5 * torch.randn(B, T, H, K, device=dev)
    a = torch.sigmoid(torch.randn(B, T, H, K, device=dev))
    kd = torch.nn.functional.normalize(torch.randn(B, T, H, K, device=dev), dim=-1, p=2.0)
    k = kd * a
    w = (0.85 + 0.14 * torch.rand(B, T, H, K, device=dev)).float()
    skip = torch.zeros(B, T, dtype=torch.bool, device=dev)
    skip[:, 50] = True
    return [x.contiguous() for x in (r, k, v, w, a, kd, skip)]


def leaf(x):
    y = x.clone().detach()
    y.requires_grad_(True)
    return y


def maxabs(x, y):
    return (x.float() - y.float()).abs().max().item()


cases = [
    (2, 2000, 4, 32, "K=32 time-parallel"),
    (2, 200, 4, 32, "K=32 sequential"),
    (2, 2000, 2, 16, "K=16 time-parallel"),
    (2, 200, 2, 16, "K=16 sequential"),
    (2, 2000, 4, 16, "K=16 H=4 time-parallel"),
]
allok = True
for (B, T, H, K, tag) in cases:
    r, k, v, w, a, kd, skip = make_inputs(B, T, H, K)
    grad_seed = torch.randn(B, T, H, K, device=dev)

    ck = [leaf(t) for t in (r, k, v, a, kd)]
    ok = RWKV7_WKV.apply(ck[0], ck[1], ck[2], w, ck[3], ck[4], skip)
    (ok.float() * grad_seed).sum().backward()
    gk = [t.grad.float().clone() for t in ck]

    cr = [leaf(t) for t in (r, k, v, a, kd)]
    orf = reference_rwkv7(cr[0], cr[1], cr[2], w, cr[3], cr[4], skip)
    (orf.float() * grad_seed).sum().backward()
    gr = [t.grad.float().clone() for t in cr]

    df = maxabs(ok, orf)
    gnames = ["r", "k", "v", "a", "kd"]
    gd = {n: maxabs(x, y) for n, x, y in zip(gnames, gk, gr)}
    dg = max(gd.values())
    ok_flag = max(df, dg) < 2e-3
    allok = allok and ok_flag
    print(f"{tag:24s} fwd|d|={df:.2e}  grad|d|={dg:.2e}  [{ {n: f'{v:.1e}' for n,v in gd.items()} }]  {'OK' if ok_flag else 'FAIL'}")

print("\n" + ("ALL PASS" if allok else "*** SOME FAILED ***"))
