"""Count training groups in train_db_5k_h1 at MAX=110000 -> optimization/groups_5k.json.

Run ONCE after build STEP3 (train_db 1-5000) completes. hp_tuner_5k.py loads the result as
GROUPS_PER_EPOCH (the 2-epoch WS budget + decay_ratio math need the real per-epoch step count).
"""
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

from rwkv.train_rwkv import get_groups  # noqa: E402

TRAIN_DB = "train_db_5k_h1"
DB_SIZE = 400_000_000_000
MAX_LEN = 110000
USERS = list(range(1, 5001))


def main():
    os.chdir(ROOT)
    groups = get_groups(TRAIN_DB, DB_SIZE, MAX_LEN, USERS)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "groups_5k.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"groups_per_epoch": len(groups), "train_db": TRAIN_DB,
                   "max_train_global_len": MAX_LEN, "users": [1, 5000]}, fh, indent=2)
    print(f"GROUPS_PER_EPOCH {len(groups)} -> {out}")


if __name__ == "__main__":
    main()
