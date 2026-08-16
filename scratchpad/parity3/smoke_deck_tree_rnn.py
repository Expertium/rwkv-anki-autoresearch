#!/usr/bin/env python
"""RWKV_DECK_TREE in the DEPLOY path (SrsRWKVRnn). CPU, seconds.

The training-side proof is scratchpad/deck_tree/smoke_tree.py (an all-inactive parent map
reproduces the tree-off forward BIT-FOR-BIT, while a real map moves it). This is the same
argument for the recurrent path, which is a separate implementation and therefore a separate
opportunity to diverge silently -- the exact failure the three-way-parity rule exists for
(PAVA was trained but never evaluated for a whole phase).

Three modes, one subprocess each (ScriptModule bakes the first construction's flags):
  off   -- no tree
  null  -- tree ON, every level marked INACTIVE  => must equal `off` EXACTLY
  real  -- tree ON, levels active                => must differ

⚠ Weights are randomized before comparing: a freshly built RWKV7 has zero-init output
projections, so every layer is the identity and ANY structural change looks like a null.
Seeded by canonicalised parameter name, because nn.ModuleList dedupes the shared deck object
and renumbers the streams after it.

ASCII output only.
"""
import hashlib
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

MODE = sys.argv[1]
from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG  # noqa: E402
from rwkv.model.srs_model_rnn import SrsRWKVRnn  # noqa: E402

names = [n for n, _ in DEFAULT_ANKI_RWKV_CONFIG.modules]
torch.manual_seed(1234)
m = SrsRWKVRnn(DEFAULT_ANKI_RWKV_CONFIG)
m.eval()

_idx2stream = {f"rwkv_modules.{i}.": f"STREAM_{n.split('@')[0]}." for i, n in enumerate(names)}


def canon(nm):
    for k, v in _idx2stream.items():
        if nm.startswith(k):
            return v + nm[len(k):]
    return nm


with torch.no_grad():
    for nm, prm in m.named_parameters():
        if "tree_level_emb" in nm:
            continue
        g = torch.Generator().manual_seed(int(hashlib.md5(canon(nm).encode()).hexdigest()[:8], 16))
        prm.copy_(torch.randn(prm.shape, generator=g) * 0.05)

n_feat = m.features2card[0].in_features
g = torch.Generator().manual_seed(7)
feats = torch.randn(6, n_feat, generator=g) * 0.3

st = {k: None for k in ("card", "note", "deck", "preset", "glob")}
n_lvl = (max(m.tree_level) + 1) if m.tree_on else 0
anc = [None] * n_lvl
acc = []
with torch.no_grad():
    for t in range(feats.size(0)):
        kw = {}
        if m.tree_on:
            kw["deck_ancestor_states"] = anc
            # `null` marks every level inactive; `real` activates them from review 1 on
            kw["deck_ancestor_active"] = [False] * n_lvl if MODE == "null" \
                else [t >= 1] * n_lvl
        out = m.review(feats[t:t + 1], st["card"], st["note"], st["deck"],
                       st["preset"], st["glob"], **kw)
        # review() returns (..., next states...); take them by position from the tail
        acc.append(float(out[4].double().sum()))  # rating logits: ahead is all-zero under NO_AHEAD_RESIDUAL
        st["card"], st["note"], st["deck"], st["preset"], st["glob"] = out[-5:]

ck = hashlib.md5(("|".join(f"{v:.10f}" for v in acc)).encode()).hexdigest()[:12]
print(f"[{MODE}] tree_on={m.tree_on} levels={m.tree_level}")
print(f"[{MODE}] params={sum(p.numel() for p in m.parameters()):,}")
print(f"[{MODE}] RESULT {ck} sum={sum(acc):.8f}")
