"""PREREG P3 for cmixgraft: did the decay USE the restored capacity? CPU, seconds.

The six restored channel mixers start with W_v exactly zero (that is what makes the graft an
identity). P3: at the end of the decay each restored W_v must have a Frobenius norm of at least 10%
of the median trained W_v elsewhere in the model. Below that the decay never used the capacity and
the run is UNINTERPRETABLE, not a null -- measure this BEFORE reading the logloss.

Usage: python scratchpad/cmixgraft/engage_probe.py [ckpt]   (default scratchpad/ws10/w10g_d_21870.pth)
Exit 0 = engaged, 1 = not engaged.
"""
import statistics
import sys

import torch

CKPT = sys.argv[1] if len(sys.argv) > 1 else "scratchpad/ws10/w10g_d_21870.pth"
RESTORED = [f"rwkv_modules.{m}.blocks.{b}.channel_mixer.W_v.weight" for m in (3, 4) for b in (0, 1, 2)]

sd = torch.load(CKPT, map_location="cpu", weights_only=False)
if isinstance(sd, dict) and "model" in sd and not any(k.startswith("features2card") for k in sd):
    sd = sd["model"]
wv = {k: v.float() for k, v in sd.items() if k.endswith("channel_mixer.W_v.weight")}
missing = [k for k in RESTORED if k not in wv]
if missing:
    raise SystemExit(f"restored W_v missing from {CKPT}: {missing}")
neigh = {k: v.norm().item() for k, v in wv.items() if k not in RESTORED and v.numel() > 1}
if not neigh:
    raise SystemExit("no trained neighbour W_v to compare against")
ref = statistics.median(neigh.values())
print(f"{CKPT}")
print(f"neighbour trained W_v ({len(neigh)}): median ||W_v||_F = {ref:.4f}")
ok = True
for k in RESTORED:
    n = wv[k].norm().item()
    r = n / ref
    flag = "engaged" if r >= 0.10 else "NOT engaged"
    ok = ok and r >= 0.10
    print(f"  {k:<50} ||W_v||_F {n:.4f}  = {r:6.1%} of the median  {flag}")
print("P3 HOLDS -- the restored capacity was used" if ok else
      "P3 FAILS -- the run is UNINTERPRETABLE, not a null")
sys.exit(0 if ok else 1)
