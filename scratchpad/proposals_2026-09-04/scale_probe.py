"""CPU screen for ranked-queue rank 11 (Muon coverage of the 26 `*scale*` matrices, the last 2-D weights
still on AdamW after iter 53 moved the LoRAs).

Kill rule (literature.md): if the training UPDATE of these matrices is already isotropic, Muon's
orthogonalisation has nothing to regularise. Measure, per tensor, the anisotropy of the update
W_final - W_init: sigma_max / ||dW||_F (1.0 = rank-1 update, 1/sqrt(min(m,n)) = white). Compare with
the LoRA tensors' ratio (iter 53's population, which Muon DID pay on) and with the Muon-run matrices.
Kill line: scale tensors' median ratio < 0.5. Also report their share of the total update energy.
Usage: scale_probe.py [init_ckpt] [final_ckpt]
"""
import sys

import numpy as np
import torch

init = sys.argv[1] if len(sys.argv) > 1 else "scratchpad/realcyc/rc_ws_50.pth"
final = sys.argv[2] if len(sys.argv) > 2 else "scratchpad/realcyc/rc_d_10935.pth"
a = torch.load(init, map_location="cpu", weights_only=True)
b = torch.load(final, map_location="cpu", weights_only=True)


def ratio(k):
    dw = (b[k].float() - a[k].float())
    if dw.dim() != 2 or min(dw.shape) < 2:
        return None
    s = torch.linalg.svdvals(dw)
    return float(s[0] / (dw.norm() + 1e-12)), float(dw.norm()), tuple(dw.shape)


groups = {"scale": [], "lora": [], "other2d": []}
for k in b:
    if not torch.is_tensor(b[k]) or b[k].dim() != 2:
        continue
    r = ratio(k)
    if r is None:
        continue
    g = "scale" if "scale" in k else ("lora" if "lora" in k else "other2d")
    groups[g].append((k, *r))
tot = sum(n ** 2 for g in groups.values() for _, _, n, _ in g)
for g, rows in groups.items():
    rs = np.array([r for _, r, _, _ in rows]); en = sum(n ** 2 for _, _, n, _ in rows)
    white = np.array([1 / np.sqrt(min(sh)) for _, _, _, sh in rows])
    print(f"{g:<8} n={len(rows):>3}  anisotropy sigma_max/||dW||_F: median {np.median(rs):.3f}  min {rs.min():.3f}  max {rs.max():.3f}"
          f"   (white-noise reference median {np.median(white):.3f})   share of update energy {100 * en / tot:.1f}%")
sc = np.array([r for _, r, _, _ in groups["scale"]])
print("scale tensors, one line each:")
for k, r, n, sh in sorted(groups["scale"], key=lambda t: -t[1])[:8]:
    print(f"  {k:<48} shape {sh}  ratio {r:.3f}  ||dW|| {n:.4f}")
print("VERDICT: " + ("DEAD -- scale updates already isotropic (median < 0.5)" if np.median(sc) < 0.5
                     else f"ALIVE -- scale updates anisotropic (median {np.median(sc):.3f}); Muon's orthogonalisation has something to act on"))
