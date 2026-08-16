#!/usr/bin/env python
"""What do the ancestor streams cost in BATCH SHAPE, not just layer count? CPU, ~1 min.

The naive cost model says 8 extra layer-steps over the same n rows == ~1.6x. That model assumes
the extra streams have the same shape profile as the deck stream, and ancestor grouping could
easily break it in two ways worth knowing BEFORE spending 5.5 h of GPU:

  1. PADDING WASTE. greedy_splits buckets sequences by length and pads to the bucket. Ancestor
     decks pool many child decks, so their sequences are longer AND more unequal -- a long tail
     next to ~50% singletons could blow up padded volume.
  2. LOST PARALLELISM. The WKV recurrence is sequential within a sequence and parallel across
     them. Fewer, longer sequences means a smaller batch dim, which costs wall-clock even at
     identical total work.

Reports padded volume (the kernel's actual B*T) and the length profile per stream.
ASCII output only.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import lmdb  # noqa: E402

from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG  # noqa: E402
from rwkv.prepare_batch import get_data, prepare  # noqa: E402

USERS = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "101,102,103,104").split(",")]
names = [n for n, _ in DEFAULT_ANKI_RWKV_CONFIG.modules]
depths = [c.n_layers for _, c in DEFAULT_ANKI_RWKV_CONFIG.modules]

env = lmdb.open("train_db_5k_h1", readonly=True, lock=False, subdir=True,
                map_size=400_000_000_000, max_readers=2048)
with env.begin() as txn:
    dl = []
    for uid in USERS:
        kk = json.loads(txn.get(f"{uid}_batches".encode()))
        dl.append(get_data(txn, (uid, *kk[len(kk) // 2]), device="cpu"))

pb = prepare(dl, target_len=65536, seed=1234)
n_rows = sum(d.card_features.size(0) for d in dl)
print(f"users={USERS}  real rows={n_rows:,}")
print("")
print(f"  {'stream':12s} {'depth':>5s} {'padded B*T':>12s} {'waste':>7s} {'seqs':>7s} "
      f"{'maxT':>7s} {'meanT':>8s}  layer-step volume")
tot_vol = 0
for i, nm in enumerate(names):
    padded = 0
    seqs = 0
    maxT = 0
    for g, L in zip(pb.sub_gather[i], pb.sub_gather_lens[i]):
        padded += g.numel()
        seqs += g.numel() // L
        maxT = max(maxT, L)
    vol = padded * depths[i]
    tot_vol += vol
    waste = 1.0 - n_rows / max(padded, 1)
    print(f"  {nm:12s} {depths[i]:5d} {padded:12,} {waste:6.1%} {seqs:7,} {maxT:7,} "
          f"{padded/max(seqs,1):8.1f}  {vol:,}")
print("")
print(f"  TOTAL layer-step volume (padded rows x layers) = {tot_vol:,}")
print(f"  layer-steps = {sum(depths)}")
