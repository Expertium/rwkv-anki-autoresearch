"""Scan the real training database for the TA3 split invariant (read-only, CPU, single process).

Why: the first chain epoch aborted at step 4930 on 2026-11-15 with `TA3: mode 'ahead' has no
labelled rows in this phase`, because the old split used ONE cut at round(0.5 * T) on the raw time
axis. This scan measures how often a real chunk's label layout breaks the OLD rule and proves the
NEW rank rule never breaks.

It reads each stored chunk's `global_labels` and reproduces the prepared label layout exactly: with
`RWKV_PROBE_DENSITY=0`, `RWKV_R2_HEAD` unset and the clock fix off -- true for every current arm --
nothing else modifies `has_label`/`is_query`. 98.2% of training groups hold exactly ONE chunk (42,592
of 43,354 measured), so the per-chunk scan covers the dominant case; multi-chunk shapes (several rows
sharing one time position) are covered exhaustively by `scratchpad/ta3/test_split.py`.

A pool of 4 workers hung on 2026-11-16 (workers busy, no results returned for 20 min), so this version
is deliberately single-process and prints progress every 250 users. Expect ~5-10 min for 5,000 users.

Usage: .venv/Scripts/python.exe scratchpad/ta3/scan_batch_layouts.py [--users 1-5000]
"""
import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch  # noqa: E402
import lmdb  # noqa: E402

from rwkv import ta3  # noqa: E402
from rwkv.prepare_batch import load_tensor  # noqa: E402

DB = "F:/rwkv_lmdb/train_db_5k_h1_adaptive92"
DB_SIZE = 400_000_000_000
HAS, ISQ = ta3._HAS_LABEL, ta3._IS_QUERY


def counts(mask_a, mask_i):
    return int(mask_a.sum().item()), int(mask_i.sum().item())


def scan(users, db=DB):
    env = lmdb.open(db, map_size=DB_SIZE, readonly=True, lock=False, readahead=False, max_readers=512)
    total = Counter()
    examples = []
    t0 = time.time()
    try:
        with env.begin() as txn:
            for k, user in enumerate(users):
                raw = txn.get(f"{user}_batches".encode())
                if raw is None:
                    total["users_missing"] += 1
                    continue
                for b in json.loads(raw):
                    start, end, length = b[0], b[1], b[2]
                    prefix = f"{user}_{start}-{end}_{length}_"
                    lab = load_tensor(txn, prefix + "global_labels", device="cpu")
                    labels = lab.unsqueeze(0) if lab.dim() == 2 else lab.view(1, lab.shape[0], lab.shape[-1])
                    T = int(labels.shape[1])
                    has = labels[..., HAS] > 0.5
                    isq = labels[..., ISQ] > 0.5
                    ahead, imm = has & ~isq, has & isq
                    n_a, n_i = counts(ahead, imm)
                    total["chunks"] += 1
                    total[f"T_bin_{T // 2048}k"] += 1
                    if n_a == 0:
                        total["ahead_0"] += 1
                    if n_i == 0:
                        total["imm_0"] += 1
                    if n_a < 2:
                        total["ahead_lt2"] += 1
                    if n_i < 2:
                        total["imm_lt2"] += 1
                    # OLD rule: one cut at round(0.5 * T) on the raw time axis
                    cut = max(1, min(T - 1, int(round(0.5 * T))))
                    t = torch.arange(T).view(1, T)
                    for name, mask in (("ahead", ahead), ("imm", imm)):
                        a = int((mask & (t < cut)).sum().item())
                        q = int((mask & (t >= cut)).sum().item())
                        if (a + q) and (a == 0 or q == 0):
                            total[f"OLD_{name}_empty_phase"] += 1
                            if len(examples) < 6:
                                examples.append({"user": user, "chunk": [start, end, length], "T": T,
                                                 "n_ahead": n_a, "n_imm": n_i, name: [a, q],
                                                 "OLD_EMPTY": True})
                    # NEW rule: rank split per group
                    in_a = ta3._mode_split(labels, ta3.FRAC)
                    for name, mask in (("ahead", ahead), ("imm", imm)):
                        n = int(mask.sum().item())
                        a = int((mask & in_a).sum().item())
                        q = n - a
                        if a + q != n:
                            total["NEW_partition_broken"] += 1
                        if n >= 2 and (a == 0 or q == 0):
                            total[f"NEW_{name}_empty_phase"] += 1
                            examples.append({"user": user, "NEW_FAIL": name, "n": n, "a": a, "q": q})
                        if n == 1 and (a, q) != (1, 0):
                            total[f"NEW_{name}_single_not_in_A"] += 1
                if (k + 1) % 250 == 0:
                    print(f"  {k + 1}/{len(users)} users, {time.time() - t0:.0f}s, "
                          f"{total['chunks']} chunks", flush=True)
                    print("   " + json.dumps({kk: total[kk] for kk in sorted(total)
                                              if not kk.startswith("T_bin_")}), flush=True)
    finally:
        env.close()
    return total, examples, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DB)
    ap.add_argument("--users", default="1-5000")
    args = ap.parse_args()
    lo, hi = (int(x) for x in args.users.split("-"))
    total, examples, secs = scan(list(range(lo, hi + 1)), args.db)
    print(json.dumps({k: v for k, v in sorted(total.items()) if not k.startswith("T_bin_")}, indent=2))
    print("T bins:", json.dumps({k: v for k, v in sorted(total.items()) if k.startswith("T_bin_")}))
    print("examples:", json.dumps(examples[:6], default=str))
    print(f"SCAN_DONE in {secs:.0f}s")


if __name__ == "__main__":
    main()
