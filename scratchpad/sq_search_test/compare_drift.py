"""Quantify run-to-run drift (sq0a vs sq0b, identical env) vs the rewrite's drift (sq0a vs sq1),
plus line-by-line step-trace comparison of all three arms."""
import json
import torch

BASE = r"C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\sq_search_test"


def diff_stats(pa, pb):
    a = torch.load(pa, weights_only=True, map_location="cpu")
    b = torch.load(pb, weights_only=True, map_location="cpu")
    worst, n_diff, n = 0.0, 0, 0
    rel_worst = 0.0
    for k in a.keys() & b.keys():
        ta, tb = a[k].float(), b[k].float()
        if ta.shape != tb.shape:
            continue
        n += 1
        d = (ta - tb).abs().max().item()
        if d > 0:
            n_diff += 1
        denom = ta.abs().max().item()
        if denom > 0:
            rel_worst = max(rel_worst, d / denom)
        worst = max(worst, d)
    return worst, rel_worst, n_diff, n


for step in (50, 110):
    for pair in (("sq0a", "sq0b"), ("sq0a", "sq1")):
        w, rw, nd, n = diff_stats(f"{BASE}\\{pair[0]}_{step}.pth", f"{BASE}\\{pair[1]}_{step}.pth")
        print(f"step {step} {pair[0]} vs {pair[1]}: worst|diff|={w:.3e}  worst rel={rw:.3e}  ({nd}/{n} tensors differ)")

print()
traces = {}
for arm in ("sq0a", "sq0b", "sq1"):
    traces[arm] = [json.loads(l) for l in open(f"{BASE}\\{arm}_trace.jsonl", encoding="utf-8")]
n = min(len(t) for t in traces.values())
print(f"trace lengths: {[len(t) for t in traces.values()]}")


def first_div(a, b):
    for i in range(n):
        if traces[a][i] != traces[b][i]:
            return i + 1, traces[a][i], traces[b][i]
    return None, None, None


for pair in (("sq0a", "sq0b"), ("sq0a", "sq1")):
    s, ra, rb = first_div(*pair)
    if s is None:
        print(f"{pair[0]} vs {pair[1]}: traces IDENTICAL over {n} steps (4-decimal logloss)")
    else:
        print(f"{pair[0]} vs {pair[1]}: first divergence at step {s}: {ra} vs {rb}")
        # max divergence over the run
        ma = max(abs(traces[pair[0]][i]["ahead"] - traces[pair[1]][i]["ahead"]) for i in range(n))
        mi = max(abs(traces[pair[0]][i]["imm"] - traces[pair[1]][i]["imm"]) for i in range(n))
        print(f"   max |ahead diff|={ma:.4f}  max |imm diff|={mi:.4f} over {n} steps")
