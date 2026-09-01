"""Where do the champion's 558,212 parameters actually live?

A <=100k budget cannot be planned from the headline number. This groups every
parameter by (stream, role) so a redesign can see which pools are big enough to be
worth attacking and which are already rounding errors.

Stream names come from CFG.modules (index -> name), NEVER a hardcoded list: the _cnd
arch runs card,note,deck,... while earlier ones ran card,deck,note,..., and a fixed
list silently mislabels deck as note (the exact bug model_stats.py records fixing).

Run under the champion's full env -- see env.sh. Without RWKV_STRIP_CMIX the total is
782,710, i.e. a different model.
"""
import os, sys
sys.path.insert(0, os.getcwd())
import torch
torch.set_num_threads(1)

from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG as CFG
from rwkv.model.srs_model_rnn import SrsRWKVRnn

model = SrsRWKVRnn(CFG)
IDX2NAME = {i: n for i, (n, _) in enumerate(CFG.modules)}


def classify(name):
    parts = name.split(".")
    stream = "(model)"
    if parts[0] == "rwkv_modules":
        stream = IDX2NAME[int(parts[1])]
        stream = stream[:-3] if stream.endswith("_id") else stream
    low = name.lower()
    if "features2card" in low:
        role = "input_fc"
    elif stream == "(model)":
        role = "heads"                       # every remaining model-level tensor is a head
    elif "lora" in low or "_a." in low or "gate" in low and "w" in low:
        role = "lora"
    elif "channel_mixer" in low or "cmix" in low or "ffn" in low:
        role = "cmix"
    elif "ln" in parts or "norm" in low or "scale" in low or "lerp" in low:
        role = "norm/lerp"
    else:
        role = "wkv_core"
    return stream, role


tot, by, cnt = 0, {}, {}
for n, p in model.named_parameters():
    k = classify(n)
    by[k] = by.get(k, 0) + p.numel()
    cnt[k] = cnt.get(k, 0) + 1
    tot += p.numel()

layers = {n: c.n_layers for n, c in CFG.modules}
print("TOTAL %d params   d_model=%d  n_heads=%d" % (tot, CFG.d_model, CFG.modules[0][1].n_heads))
print("layers: %s" % layers)
print()
roles = ["wkv_core", "lora", "cmix", "norm/lerp", "input_fc", "heads"]
streams = [n[:-3] if n.endswith("_id") else n for n, _ in CFG.modules] + ["(model)"]
hdr = "%-8s" % "stream" + "".join("%11s" % r for r in roles) + "%11s%8s" % ("total", "%")
print(hdr); print("-" * len(hdr))
for s in streams:
    row = [by.get((s, r), 0) for r in roles]
    st = sum(row)
    if not st:
        continue
    print("%-8s" % s + "".join("%11d" % v if v else "%11s" % "." for v in row)
          + "%11d%7.1f%%" % (st, 100 * st / tot))
print("-" * len(hdr))
rt = [sum(by.get((s, r), 0) for s in streams) for r in roles]
print("%-8s" % "TOTAL" + "".join("%11d" % v for v in rt) + "%11d" % tot)
print("%-8s" % "%" + "".join("%10.1f%%" % (100 * v / tot) for v in rt))
print()
per_card = sum(by.get(("card", r), 0) for r in roles)
ctx = sum(by.get((s, r), 0) for s in ["note", "deck", "preset", "user"] for r in roles)
print("card stream (the per-card recurrence)   %7d  %5.1f%%" % (per_card, 100 * per_card / tot))
print("context streams (note/deck/preset/user) %7d  %5.1f%%" % (ctx, 100 * ctx / tot))
print("input FC + heads                        %7d  %5.1f%%"
      % (tot - per_card - ctx, 100 * (tot - per_card - ctx) / tot))
