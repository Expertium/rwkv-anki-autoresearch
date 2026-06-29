"""Fake-quantize a checkpoint's 2D matmul weights (per-output-channel symmetric int-N) and save a
DEQUANTIZED fp32 checkpoint -- to MEASURE the accuracy impact of PTQ before implementing int8
storage/inference in the Rust engine. Norms / biases / tiny (<8) weights stay fp32.

Per-output-channel symmetric: for W (out,in), scale[o] = max|W[o,:]| / (2^(bits-1)-1);
q = round(W/scale).clamp(+/-qmax); dequant = q*scale. (Linear does x @ W^T, so each OUTPUT
channel = one row of W gets its own scale -- the standard, lowest-error scheme.)

Usage: python optimization/quant_ptq.py <in.pth> <out.pth> <bits>
"""
import os
import sys

import torch

inp, out, bits = sys.argv[1], sys.argv[2], int(sys.argv[3])
qmax = 2 ** (bits - 1) - 1

sd = torch.load(inp, map_location="cpu", weights_only=True)
new = {}
nq = nk = 0
bytes_orig = bytes_q = 0.0
for k, v in sd.items():
    if torch.is_tensor(v) and v.is_floating_point() and v.dim() == 2 and min(v.shape) >= 8:
        W = v.float()
        scale = (W.abs().amax(dim=1, keepdim=True) / qmax).clamp_min(1e-12)  # (out,1)
        q = torch.round(W / scale).clamp(-qmax, qmax)
        new[k] = (q * scale).to(v.dtype)
        nq += 1
        bytes_q += q.numel() * bits / 8.0 + scale.numel() * 4  # packed intN + fp32 scales
        bytes_orig += v.numel() * 4
    else:
        new[k] = v
        nk += 1
        if torch.is_tensor(v):
            bytes_orig += v.numel() * 4
            bytes_q += v.numel() * 4

os.makedirs(os.path.dirname(out), exist_ok=True)
torch.save(new, out)
print(f"quantized {nq} 2D weights (min-dim>=8), kept {nk} others; bits={bits}")
print(f"approx weight storage: fp32 {bytes_orig/1024:.1f} KiB -> intN {bytes_q/1024:.1f} KiB "
      f"({bytes_orig/max(bytes_q,1):.2f}x smaller)")
