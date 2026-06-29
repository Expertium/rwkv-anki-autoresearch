"""Average (model-soup) two or more checkpoints with identical architecture.

Usage: python optimization/soup.py OUT.pth IN1.pth IN2.pth [IN3.pth ...]
Averages tensors elementwise (floats); non-float buffers taken from the first ckpt.
The inputs must share the SAME architecture (same keys/shapes) -- e.g. seeds of one arch.
"""
import sys
import torch

out_path = sys.argv[1]
in_paths = sys.argv[2:]
assert len(in_paths) >= 2, "need >=2 checkpoints to soup"

sds = [torch.load(p, map_location="cpu", weights_only=True) for p in in_paths]
keys = sds[0].keys()
souped = {}
for k in keys:
    t0 = sds[0][k]
    if torch.is_floating_point(t0):
        acc = torch.zeros_like(t0, dtype=torch.float32)
        for sd in sds:
            acc += sd[k].float()
        souped[k] = (acc / len(sds)).to(t0.dtype)
    else:
        souped[k] = t0  # ints/bools: take first
torch.save(souped, out_path)
print(f"souped {len(in_paths)} ckpts -> {out_path} ({len(souped)} tensors)")
