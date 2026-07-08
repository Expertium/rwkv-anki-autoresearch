"""Compare the three A/B arms: sha256 of files, then tensor-level state_dict diffs."""
import hashlib
import torch

BASE = r"C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\sq_search_test"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


for step in (50, 110):
    for kind in ("", "optim_"):
        hs = {arm: sha(f"{BASE}\\{arm}_{kind}{step}.pth") for arm in ("sq0a", "sq0b", "sq1")}
        ctrl = "PASS" if hs["sq0a"] == hs["sq0b"] else "FAIL"
        ab = "MATCH" if hs["sq1"] == hs["sq0a"] else "DIFF"
        print(f"{kind or 'model_'}{step}: control(sq0a==sq0b)={ctrl}  sq1-vs-sq0a={ab}  {hs}")

print()
for step in (50, 110):
    a = torch.load(f"{BASE}\\sq0a_{step}.pth", weights_only=True, map_location="cpu")
    b = torch.load(f"{BASE}\\sq1_{step}.pth", weights_only=True, map_location="cpu")
    ka, kb = set(a.keys()), set(b.keys())
    if ka != kb:
        print(f"step {step}: KEY SET DIFFERS  only_sq0a={sorted(ka - kb)[:8]}  only_sq1={sorted(kb - ka)[:8]}")
    n_diff, worst_key, worst = 0, None, 0.0
    for k in sorted(ka & kb):
        ta, tb = a[k].float(), b[k].float()
        if ta.shape != tb.shape:
            print(f"step {step}: SHAPE DIFF {k}: {tuple(ta.shape)} vs {tuple(tb.shape)}")
            n_diff += 1
            continue
        d = (ta - tb).abs().max().item()
        if d > 0:
            n_diff += 1
            if d > worst:
                worst, worst_key = d, k
    print(f"step {step}: {n_diff}/{len(ka & kb)} common tensors differ; worst |diff|={worst:.3e} at {worst_key}")
