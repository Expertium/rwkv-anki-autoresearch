"""How big is the 750-user roster's eval work, so the corruption check's GPU cost can be scaled?

Reads only the `{user}_batches` metadata from the val LMDB (same quantity eval_sharded.py sizes with:
the largest section length per user).
"""
import json
import sys
from pathlib import Path

import lmdb

ROOT = Path(__file__).resolve().parents[3]
VAL_DB = "F:/rwkv_lmdb/test_db_5k_adaptive92"
DB_SIZE = 250_000_000_000


def sizes(users):
    env = lmdb.open(VAL_DB, map_size=DB_SIZE, readonly=True, lock=False, max_readers=256,
                    readahead=False)
    out = {}
    with env.begin() as txn:
        for u in users:
            raw = txn.get(f"{u}_batches".encode())
            if raw is None:
                continue
            batches = json.loads(raw)
            out[u] = sum(int(b[2]) for b in batches)      # b = [start_th, end_th, len]
    env.close()
    return out


def main():
    roster = json.loads((ROOT / "scratchpad" / "quant_plan" /
                         "roster750_seed20261119.json").read_text(encoding="utf-8"))
    users = roster["users"] if isinstance(roster, dict) else roster
    r = sizes([int(u) for u in users])
    cor = json.loads((ROOT / "scratchpad" / "quant_plan" / "corruption" /
                      "corrupt50.json").read_text(encoding="utf-8"))
    c = {int(x["user"]): int(x["section_rows"]) for x in cor["selected"]}
    print(f"roster750: {len(r)} users, {sum(r.values()):,} section rows")
    print(f"corrupt50: {len(c)} users, {sum(c.values()):,} section rows")
    print(f"ratio corrupt50/roster750 rows = {sum(c.values()) / sum(r.values()):.3f}")
    big = sorted(c.items(), key=lambda kv: -kv[1])[:5]
    print("largest corrupt50 sections:", big)


if __name__ == "__main__":
    sys.exit(main())
