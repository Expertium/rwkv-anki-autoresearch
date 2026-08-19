"""Gate an LMDB on ENTRY COUNT and, optionally, stored feature WIDTH. Exit 0 = pass.

WHY ENTRIES AND NOT EXISTENCE. `data_processing` and `find_equalize_test_reviews` are resumable:
they skip users already present. A pre-existing directory therefore reports success in seconds
having done nothing at all -- which is exactly what the de-risk build did at 17:27 on 2026-08-19,
whose dbs were left over from 2026-08-16. An `if not exist ...mdb` guard passes that vacuously,
because the file is there and is even the right size (LMDB preallocates its map).

WHY WIDTH TOO. The whole point of this rebuild is that the feature vector got wider. Checking the
stored tensor's second dimension is the one check that fails if `RWKV_ID_FEATURES` did not reach
the worker processes -- the failure mode that would otherwise produce a complete, plausible,
2-4-day db carrying the OLD layout. Same family as the QAT env that was parsed and then discarded:
a banner proves a value was computed, never that it was used.

Usage: check_db.py <path> <min_entries> [expected_width]
"""
import io
import sys

import lmdb
import torch

path = sys.argv[1]
min_entries = int(sys.argv[2])
want_width = int(sys.argv[3]) if len(sys.argv) > 3 else 0

try:
    env = lmdb.open(path, readonly=True, lock=False, subdir=True)
except lmdb.Error as e:
    print("CHECK_DB FAIL %s: cannot open (%s)" % (path, e))
    sys.exit(2)

with env.begin() as txn:
    entries = txn.stat()["entries"]
    width = None
    if want_width:
        cur = txn.cursor()
        found = cur.set_range(b"")
        while found:
            k, v = cur.item()
            if k.endswith(b"card_features"):
                try:
                    width = int(torch.load(io.BytesIO(v), map_location="cpu",
                                           weights_only=True).shape[1])
                except Exception as e:                       # noqa: BLE001
                    print("CHECK_DB FAIL %s: cannot decode card_features (%s)" % (path, e))
                    sys.exit(4)
                break
            found = cur.next()

print("CHECK_DB %s entries=%d width=%s" % (path, entries, width))

if entries < min_entries:
    print("CHECK_DB FAIL: %d entries < required %d -- the build did nothing, or stopped early. "
          "A resumable builder over a pre-existing dir reports success having skipped everything."
          % (entries, min_entries))
    sys.exit(3)

if want_width:
    if width is None:
        print("CHECK_DB FAIL: no card_features key found, so width is unverifiable")
        sys.exit(5)
    if width != want_width:
        print("CHECK_DB FAIL: stored width %d != expected %d -- RWKV_ID_FEATURES did not reach "
              "the workers, and this db carries the OLD layout despite looking complete."
              % (width, want_width))
        sys.exit(6)

print("CHECK_DB OK")
sys.exit(0)
