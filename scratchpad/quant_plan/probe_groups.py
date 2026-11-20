"""Count training groups per epoch for a user range and MAX_TRAIN_GLOBAL_LEN.

Offline replica of `rwkv/train_rwkv.py::get_groups` (same packing loop), read-only, so we can size
a decay run BEFORE launching it without touching the GPU.

Why it is needed: the anchor redo must replay the decay that iter36 actually ran, which is 124
steps (see NOTES.md "Anchor investigation"). To hit exactly 124 steps on the 5k train db we need
its group count per epoch for the user range we train on.

Usage:
  .venv/Scripts/python.exe scratchpad/quant_plan/probe_groups.py [--db PATH] [--users 1 100]
      [--max 16384 66000] [--limit-users N]
"""
import argparse
import json
import math
import random
from collections import Counter

import lmdb

DB = "F:/rwkv_lmdb/train_db_5k_h1_adaptive92"
DB_SIZE = 400_000_000_000


def chunks_for(db, size, users):
    """(user, batch0, batch1, chunk_len) tuples exactly as get_groups builds them."""
    env = lmdb.open(db, map_size=size, readonly=True, lock=False)
    keys = []
    missing = []
    with env.begin(write=False) as txn:
        for u in users:
            raw = txn.get(f"{u}_batches".encode())
            if raw is None:
                missing.append(u)
                continue
            for batch in json.loads(raw):
                keys.append((u, batch[0], batch[1], batch[2]))
    env.close()
    return keys, missing


def count_groups(keys, max_len, seed=0):
    random.seed(seed)
    random.shuffle(keys)
    keys = sorted(keys, key=lambda x: x[3], reverse=True)
    n = 0
    skipped = 0
    l = 0
    while l < len(keys):
        _, _, _, size = keys[l]
        max_batch = math.floor(max_len / size - 1e-6)
        if max_batch == 0:
            skipped += 1
            l += 1
            continue
        r = l - 1
        while r + 1 < len(keys) and r + 1 - l + 1 <= max_batch:
            r += 1
        if l <= r:
            n += 1
        l = r + 1
    return n, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DB)
    ap.add_argument("--db-size", type=int, default=DB_SIZE)
    ap.add_argument("--users", type=int, nargs=2, default=[1, 100])
    ap.add_argument("--max", type=int, nargs="+", default=[16384, 66000])
    args = ap.parse_args()

    users = list(range(args.users[0], args.users[1] + 1))
    keys, missing = chunks_for(args.db, args.db_size, users)
    print(f"db={args.db}")
    print(f"users {args.users[0]}..{args.users[1]}: {len(users)} asked, {len(missing)} MISSING "
          f"{missing[:10]}{' ...' if len(missing) > 10 else ''}")
    if not keys:
        raise SystemExit("no chunks found for that user range")
    sizes = sorted(k[3] for k in keys)
    print(f"chunks={len(keys)}  chunk size min={sizes[0]} med={sizes[len(sizes) // 2]} "
          f"max={sizes[-1]}  total chunk rows={sum(sizes)}")
    print(f"chunks per user: {Counter(k[0] for k in keys).most_common(3)} ...")
    for m in args.max:
        n, skipped = count_groups(list(keys), m)
        print(f"MAX_TRAIN_GLOBAL_LEN={m}: groups/epoch={n}  chunks skipped (chunk>MAX)={skipped}"
              f"  -> EPOCHS for 124 steps = {124 / n:.4f}" if n else "no groups")
    print("RESULT " + " | ".join(
        f"MAX={m} groups={count_groups(list(keys), m)[0]}" for m in args.max))


if __name__ == "__main__":
    main()
