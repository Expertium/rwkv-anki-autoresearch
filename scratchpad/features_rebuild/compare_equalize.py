"""Do two databases agree on WHICH reviews are scored? Compares per-user equalized counts.

WHY THIS IS A REAL CHECK AND NOT A FORMALITY. `size` -- the number the benchmark scores and gate
#1 compares -- IS the stored `label_is_equalize` count, and that comes from the LABEL FILTER DB
(verified 2026-09-02 against `test_db_5k_e2s`). Generations 3 and 4 use the SAME
`label_filter_db_id`, so their equalized counts must be IDENTICAL. Anything else is a build bug:
rows dropped, a different filter picked up, or a chunking change.

That makes this the one integrity check a features rebuild can run against its predecessor without
training anything. `check_db.py` verifies entry counts and column width -- i.e. that the shape is
right -- and cannot see whether the same REVIEWS ended up marked as scored.

Usage: compare_equalize.py <db_a> <size_a> <db_b> <size_b> <first_user> [n_users] [stride]
Exit 0 = identical. 1 = they differ. 2 = nothing compared (vacuous).
"""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import lmdb  # noqa: E402
import numpy as np  # noqa: E402

from rwkv.prepare_batch import get_data  # noqa: E402

_HAS, _EQ = 4, 5


def equalized_counts(db, size, users):
    env = lmdb.open(db, map_size=size, readonly=True, lock=False)
    out = {}
    with env.begin(write=False) as txn:
        for u in users:
            raw = txn.get(f"{u}_batches".encode())
            if raw is None:
                continue
            n = 0
            for b in json.loads(raw):
                d = get_data(txn, (u, b[0], b[1], b[2]), device="cpu")
                lab = d.global_labels.float().numpy()
                sk = d.skips.numpy()
                n += int(((lab[:, _EQ] > 0.5) & (lab[:, _HAS] > 0.5) & sk).sum())
            out[u] = n
    return out


def main():
    db_a, size_a, db_b, size_b, first = (
        sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), int(sys.argv[5]))
    n_users = int(sys.argv[6]) if len(sys.argv) > 6 else 20
    stride = int(sys.argv[7]) if len(sys.argv) > 7 else 53
    users = [first + k * stride for k in range(n_users)]

    a = equalized_counts(db_a, size_a, users)
    b = equalized_counts(db_b, size_b, users)
    common = sorted(set(a) & set(b))
    print("[equalize] A %s" % db_a)
    print("[equalize] B %s" % db_b)
    print("[equalize] compared %d users" % len(common))
    if not common:
        print("[equalize] *** no users in common -- nothing compared. VACUOUS, not a pass.")
        return 2
    if sum(a[u] for u in common) == 0:
        print("[equalize] *** every count is zero -- the label filter did not apply. NOT a pass.")
        return 2

    bad = [u for u in common if a[u] != b[u]]
    print("[equalize] total scored: A %d   B %d"
          % (sum(a[u] for u in common), sum(b[u] for u in common)))
    if bad:
        print("[equalize] *** %d of %d users DIFFER" % (len(bad), len(common)))
        for u in bad[:10]:
            print("[equalize]     user %d: A %d, B %d" % (u, a[u], b[u]))
        print("[equalize] Both generations use the same label filter, so identical counts are")
        print("[equalize] REQUIRED. A difference is a build bug, not a dataset property.")
        return 1
    print("[equalize] OK -- identical scored sets on all %d users" % len(common))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
