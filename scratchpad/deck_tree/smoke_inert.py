#!/usr/bin/env python
"""Is the deck-tree change INERT with RWKV_DECK_TREE unset? CPU, seconds.

This is the guard the running iter-49 chain needs: its EVAL phase is a new process that will
import whatever is on disk then, so anything the lever changed unconditionally reaches it.
Two things were not inert by default and this checks both, plus the one real refactor:

1. `tree_level_emb` must NOT exist when the lever is off -- a Parameter that always exists adds a
   state_dict key, and every checkpoint written before today would then fail to load.
2. The scripted forward must still compile without it (the reference lives in a jit.ignore body).
3. insert_probes' grouping was extracted to deck_tree.build_module_data. That runs on EVERY batch
   regardless of the flag, so it must be byte-identical to the code it replaced. Checked against a
   verbatim copy of the pre-refactor algorithm on adversarial id arrays (ties, singletons,
   negatives, one huge group, already-sorted, reverse-sorted).

ASCII output only.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
assert not os.environ.get("RWKV_DECK_TREE"), "run this WITHOUT RWKV_DECK_TREE set"

import torch  # noqa: E402

from rwkv.deck_tree import build_module_data  # noqa: E402


def old_grouping(arr, new_n):
    """VERBATIM copy of the pre-refactor block from prepare_batch.insert_probes."""
    order = np.argsort(arr, kind="stable")
    sorted_ids = arr[order]
    starts = np.concatenate(([0], np.nonzero(np.diff(sorted_ids))[0] + 1))
    ends = np.concatenate((starts[1:], [new_n]))
    lens = ends - starts
    buckets = {}
    for s, e, l in zip(starts, ends, lens):
        buckets.setdefault(int(l), []).append(order[s:e])
    locs_parts = []
    split_len = []
    split_B = []
    for l in sorted(buckets):
        split_len.append(l)
        split_B.append(len(buckets[l]))
        locs_parts.extend(buckets[l])
    from_perm = np.concatenate(locs_parts)
    to_perm = np.empty(new_n, dtype=np.int64)
    to_perm[from_perm] = np.arange(new_n)
    return (np.array(split_len, dtype=np.int32), np.array(split_B, dtype=np.int32),
            from_perm, to_perm)


def check_grouping():
    rng = np.random.default_rng(0)
    cases = {
        "random_small_alphabet": rng.integers(0, 5, 200).astype(np.int64),
        "random_big_alphabet": rng.integers(0, 500, 2000).astype(np.int64),
        "all_singletons": np.arange(300, dtype=np.int64),
        "all_one_group": np.zeros(300, dtype=np.int64),
        "one_huge_rest_single": np.concatenate(
            [np.zeros(500, dtype=np.int64), np.arange(1, 51, dtype=np.int64)]),
        "already_sorted": np.repeat(np.arange(50, dtype=np.int64), 7),
        "reverse_sorted": np.repeat(np.arange(50, 0, -1, dtype=np.int64), 7),
        "negatives_mixed": np.concatenate(
            [-np.arange(1, 101, dtype=np.int64), rng.integers(0, 9, 100).astype(np.int64)]),
        "huge_ids": (np.int64(314159265358979323) - rng.integers(0, 3, 150)).astype(np.int64),
        "single_row": np.array([7], dtype=np.int64),
    }
    bad = 0
    for name, arr in cases.items():
        n = arr.shape[0]
        a = build_module_data(arr, n)
        b = old_grouping(arr, n)
        same = all(np.array_equal(x, y) for x, y in zip(a, b))
        # invariants the chain relies on, independent of the old code
        fp, tp = a[2], a[3]
        perm_ok = np.array_equal(np.sort(fp), np.arange(n))
        inv_ok = np.array_equal(tp[fp], np.arange(n))
        cover = int((a[0].astype(np.int64) * a[1].astype(np.int64)).sum())
        print(f"  {name:24s} n={n:5d} identical={same} perm={perm_ok} inverse={inv_ok} "
              f"covers={cover == n}")
        if not (same and perm_ok and inv_ok and cover == n):
            bad += 1
    return bad


def check_model_inert():
    from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG
    from rwkv.model.srs_model import SrsRWKV

    names = [n for n, _ in DEFAULT_ANKI_RWKV_CONFIG.modules]
    print(f"  chain = {names}")
    assert not any("@" in n for n in names), "chain must be untouched with the lever off"
    m = SrsRWKV(DEFAULT_ANKI_RWKV_CONFIG)
    has_emb = any("tree_level_emb" in k for k in m.state_dict())
    n_par = sum(p.numel() for p in m.parameters())
    print(f"  tree_on={m.tree_on} tree_level_emb in state_dict={has_emb} params={n_par:,}")
    assert not has_emb, "tree_level_emb LEAKED into the state_dict with the lever off"
    assert not m.tree_on
    if os.environ.get("RWKV_NO_JIT", "0") != "1":
        torch.jit.script(m)
        print("  torch.jit.script(SrsRWKV) OK without tree_level_emb")
    else:
        print("  (RWKV_NO_JIT=1 -- scripting not exercised; rerun without it)")
    return 0


print("== grouping refactor is byte-identical ==")
bad = check_grouping()
print("== model is inert with RWKV_DECK_TREE unset ==")
bad += check_model_inert()
print("")
print("INERT: PASS" if bad == 0 else f"INERT: FAIL ({bad})")
sys.exit(1 if bad else 0)
