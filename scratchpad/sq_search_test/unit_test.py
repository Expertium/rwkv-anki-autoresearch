"""Bit-exactness unit test for the RWKV_SHIFT_SQ_SEARCH rewrite (fake_pq_shift nearest-centroid).

Checks, on GPU fp32, for the real q72u shapes (8192-row chunks x 16 dims vs 4096-entry catalog)
plus small-catalog and ragged-tail shapes:
  (1) sqrt(clamp_min(_sq_dist_rows(a,b),0)) == torch.cdist(a,b)  BITWISE (proves the augmented
      matmul replicates aten::_euclidean_dist exactly on this torch/CUDA build), and
  (2) argmin indices: _clamp_argmin_rows(_sq_dist_rows(a,b)) == cdist(a,b).argmin(1) EXACTLY,
      including adversarial near-ties (rows equal to centroids, duplicated centroids, tiny
      perturbations at fp32 epsilon scale).
"""
import os
import sys

sys.path.insert(0, r"C:\Users\Andrew\rwkv-anki-autoresearch")
os.environ.setdefault("RWKV_SHIFT_SQ_SEARCH", "1")

import torch  # noqa: E402
from rwkv.model.rwkv_model import _sq_dist_rows, _clamp_argmin_rows  # noqa: E402

torch.manual_seed(0)
dev = "cuda"
fails = 0


def check(a, b, tag):
    global fails
    ref = torch.cdist(a, b)
    d2 = _sq_dist_rows(a, b)
    recon = d2.clamp_min(0).sqrt()
    bit = torch.equal(recon, ref)
    idx_new = _clamp_argmin_rows(d2)
    idx_ref = ref.argmin(dim=1)
    idx_ok = torch.equal(idx_new, idx_ref)
    n_diff = (idx_new != idx_ref).sum().item()
    print(f"{tag:40s} bitwise_dist={'PASS' if bit else 'FAIL'}  "
          f"argmin={'PASS' if idx_ok else f'FAIL ({n_diff} rows differ)'}")
    if not (bit and idx_ok):
        fails += 1


# real q72u shift shapes: unit vectors (norm 1) x learned catalog, m=2 sub=16 ncent=4096
for trial in range(5):
    a = torch.randn(8192, 16, device=dev)
    a = a / a.norm(dim=1, keepdim=True).clamp_min(1e-20)
    b = torch.randn(4096, 16, device=dev) * 0.3
    check(a, b, f"q72u chunk 8192x16 vs 4096 (t{trial})")

# ragged tail chunk (N % 8192)
a = torch.randn(3313, 16, device=dev)
a = a / a.norm(dim=1, keepdim=True).clamp_min(1e-20)
b = torch.randn(4096, 16, device=dev) * 0.3
check(a, b, "ragged tail 3313x16 vs 4096")

# small catalog (one-shot path shapes, legacy m2b8)
a = torch.randn(60000, 16, device=dev)
a = a / a.norm(dim=1, keepdim=True).clamp_min(1e-20)
b = torch.randn(256, 16, device=dev) * 0.3
check(a, b, "one-shot 60000x16 vs 256")

# adversarial: rows EXACTLY equal to centroids (d=0 vs clamp ties)
b = torch.randn(4096, 16, device=dev) * 0.3
a = b[torch.randint(0, 4096, (8192,), device=dev)].clone()
check(a, b, "rows == centroids (exact zeros)")

# adversarial: duplicated centroids (guaranteed exact ties -> first-min index must match)
b_dup = b.clone()
b_dup[2048:] = b_dup[:2048]
check(a, b_dup, "duplicated centroids (exact ties)")

# adversarial: epsilon-perturbed near-ties
a2 = a + torch.randn_like(a) * 1e-7
check(a2, b_dup, "eps-perturbed near-ties")

print("ALL_PASS" if fails == 0 else f"FAILURES={fails}")
sys.exit(0 if fails == 0 else 1)
