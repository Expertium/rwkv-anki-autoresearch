"""Does a BUILT database actually carry the Bug C fix? Checks the artifact, not the code.

assert_bugc_fixed.py proves `nan_id_fill` is correct. That is a different claim from "the database
on disk was built with it", and conflating the two is exactly the error retracted on 2026-09-01: a
fix can be live in code while the artifact in hand predates it. A database is dated by when it was
BUILT, not by what it is named, so the only trustworthy check reads the bytes.

THE MEASUREMENT. For cards flagged `note_id_is_nan`, count distinct stored note ids against
distinct card ids. Bug C fixed => one placeholder per card, ratio 1.0. Bug C present => far fewer
(published collapses ~98.3%, -id ~37.2%, both far below 1.0).

Run with the same RWKV_ID_FEATURES as the db was built with -- the column layout depends on it.

Usage: check_db_idfill.py <db_path> <db_size> <first_user> [n_users] [stride]
Exit 0 = fixed. 47 = Bug C present. 48 = nothing measured (vacuous).
"""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import lmdb  # noqa: E402

from rwkv.data_processing import CARD_FEATURE_COLUMNS  # noqa: E402
from rwkv.prepare_batch import get_data  # noqa: E402

COL_NAN = CARD_FEATURE_COLUMNS.index("note_id_is_nan")


def main():
    db, size, first = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    n_users = int(sys.argv[4]) if len(sys.argv) > 4 else 20
    stride = int(sys.argv[5]) if len(sys.argv) > 5 else 53

    env = lmdb.open(db, map_size=size, readonly=True, lock=False)
    cards, phs = set(), set()
    with env.begin(write=False) as txn:
        for k in range(n_users):
            u = first + k * stride
            raw = txn.get(f"{u}_batches".encode())
            if raw is None:
                continue
            for b in json.loads(raw):
                d = get_data(txn, (u, b[0], b[1], b[2]), device="cpu")
                cf = d.card_features.float().numpy()
                m = cf[:, COL_NAN] > 0.5
                if m.any():
                    cards.update(int(x) for x in d.ids["card_id"].numpy()[m])
                    phs.update(int(x) for x in d.ids["note_id"].numpy()[m])

    print("[idfill] %s" % db)
    print("[idfill] NaN-note cards %d -> distinct placeholder note ids %d" % (len(cards), len(phs)))
    if not cards:
        print("[idfill] *** no NaN-note cards found -- nothing was measured. VACUOUS, not a pass.")
        return 48
    ratio = len(phs) / len(cards)
    print("[idfill] ratio %.4f" % ratio)
    if ratio > 0.999:
        print("[idfill] OK -- one note per NaN-note card, Bug C fix is IN THE ARTIFACT")
        return 0
    print("[idfill] *** Bug C IS PRESENT in this database (expected ratio 1.0)")
    return 47


if __name__ == "__main__":
    raise SystemExit(main())
