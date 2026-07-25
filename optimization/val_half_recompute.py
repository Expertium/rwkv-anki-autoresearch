"""Recompute by-user-mean LogLoss on ANY user subrange from the stored per-user result
jsonls -- no GPU, no re-eval (Andrew 2026-07-25: "you can recompute log loss for A0 or A4
on 2.5k users, right?").

Why it works: result/RWKV-<tag>.jsonl and RWKV-P-<tag>.jsonl hold one record per user
({"metrics": {...}, "user": N, "size": M}), and the benchmark metric is the UNWEIGHTED
mean of per-user LogLoss. So a run evaluated on the full 5001-10000 range can be scored on
the val half (5001-7500) exactly as if it had been evaluated there -- the per-user numbers
are identical, only the set being averaged changes. This makes the pre-split rows
(A0..A8, iters <=28) directly comparable to the post-split ones, which the tables' 'v'
marker currently warns they are not.

Usage:
  python optimization/val_half_recompute.py                     # all track-2 tags
  python optimization/val_half_recompute.py iter23 iter26       # explicit tags
  python optimization/val_half_recompute.py --lo 7501 --hi 10000 track2_a15   # test half
"""
import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULT = ROOT / "result"

TRACK2_DEFAULT = [
    "track2_a0", "track2_a1", "track2_a2", "track2_a3", "track2_reanchor",
    "track2_a5", "track2_a6", "track2_a7", "track2_a8", "track2_a9",
    "track2_a10", "track2_a11", "track2_a12", "track2_a13", "track2_a14",
    "track2_a15",
]
# display names (A4 was written under the tag 'reanchor')
ALIAS = {"track2_reanchor": "A4 (reanchor)"}


def load(tag, kind):
    p = RESULT / f"RWKV-{tag}.jsonl" if kind == "ahead" else RESULT / f"RWKV-P-{tag}.jsonl"
    if not p.exists():
        return None
    out = {}
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        out[int(r["user"])] = float(r["metrics"]["LogLoss"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="*", default=None)
    ap.add_argument("--lo", type=int, default=5001)
    ap.add_argument("--hi", type=int, default=7500)
    args = ap.parse_args()
    tags = args.tags or TRACK2_DEFAULT

    print(f"by-user mean LogLoss over users {args.lo}-{args.hi} "
          f"(recomputed from stored per-user jsonls)\n")
    print(f"{'run':<18} {'n':>5}  {'ahead':>9}  {'imm':>9}   {'full-range n':>12}")
    print("-" * 62)
    rows = []
    for tag in tags:
        a, i = load(tag, "ahead"), load(tag, "imm")
        if a is None or i is None:
            print(f"{ALIAS.get(tag, tag):<18} {'--':>5}  (missing result jsonl)")
            continue
        users = sorted(u for u in a.keys() & i.keys() if args.lo <= u <= args.hi)
        if not users:
            print(f"{ALIAS.get(tag, tag):<18} {'0':>5}  (no users in range)")
            continue
        ma = sum(a[u] for u in users) / len(users)
        mi = sum(i[u] for u in users) / len(users)
        rows.append((tag, len(users), ma, mi, len(a)))
        print(f"{ALIAS.get(tag, tag):<18} {len(users):>5}  {ma:>9.6f}  {mi:>9.6f}   {len(a):>12}")

    if len(rows) > 1:
        base = rows[-1]
        print(f"\ndeltas vs {ALIAS.get(base[0], base[0])} (positive = that run is WORSE):")
        for tag, n, ma, mi, _ in rows[:-1]:
            print(f"  {ALIAS.get(tag, tag):<18} ahead {ma - base[2]:+.6f}   imm {mi - base[3]:+.6f}")


if __name__ == "__main__":
    main()
