"""Does a BUILT generation-5 database actually carry the real-time cycles? Checks the artifact.

`RWKV_REAL_CYCLES=1` is read at import, so a rebuild launched without it silently reproduces
generation 4 under a generation-5 name: same entry counts, same Bug C ratio, width 46 instead of
70 -- and check_db would catch the width, but nothing would catch a width-70 db whose last 24
columns are garbage. "The flag is live in code" and "this database was built with it" are
different claims (the featA2 retraction), so this reads the stored card_features back and
asserts structure, not just shape:

  * stored width is 70 (46 + 24);
  * columns 46..69 are twelve (sin, cos) pairs on the unit circle -- to bf16 tolerance, since the
    LMDB stores bf16 and its roundoff puts sin^2+cos^2 within ~0.01 of 1 (a 1e-3 tolerance
    "failed" the real layout once already);
  * NEGATIVE CONTROL: the same pair test on the window shifted by one column must FAIL, or the
    test cannot tell position and a pass means nothing;
  * the first-review halves are constant within a card (the pairs at offsets 2,3 of each period).

Usage: check_db_cycles.py <db_path> <db_size> <first_user> [n_users] [stride]
Exit 0 = cycles present and well-formed. 49 = missing/malformed. 48 = nothing measured (vacuous).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import lmdb  # noqa: E402

from rwkv.prepare_batch import get_data  # noqa: E402

from rwkv import id_features as _idf  # noqa: E402

# Derived from the live layout, not hardcoded: the first version said 46..70 and the layout moved
# to 45..69 the same afternoon when day_of_week was dropped as well. The env must carry the flags
# the db was built with -- asserted, because a mismatch here would test the wrong columns.
assert _idf.enabled() and _idf.real_cycles_enabled(), \
    "run with RWKV_ID_FEATURES=1 RWKV_REAL_CYCLES=1 -- the layout this checks depends on both"
WIDTH = _idf.card_feature_width()
CYC_HI = WIDTH
CYC_LO = WIDTH - len(_idf.CYCLE_COLUMNS)
TOL = 0.02


def pair_err(block):
    seg = block.reshape(block.shape[0], -1, 2)
    return float(np.abs((seg ** 2).sum(-1) - 1.0).max())


def main():
    db, size, first = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    n_users = int(sys.argv[4]) if len(sys.argv) > 4 else 12
    stride = int(sys.argv[5]) if len(sys.argv) > 5 else 53

    env = lmdb.open(db, map_size=size, readonly=True, lock=False)
    worst_true, best_shift, n_rows, n_chunks, bad_const = 0.0, 0.0, 0, 0, 0
    with env.begin(write=False) as txn:
        for k in range(n_users):
            u = first + k * stride
            raw = txn.get(f"{u}_batches".encode())
            if raw is None:
                continue
            for b in json.loads(raw):
                d = get_data(txn, (u, b[0], b[1], b[2]), device="cpu")
                cf = d.card_features.float().numpy()
                if cf.shape[1] != WIDTH:
                    print("[cycles] *** %s: stored width %d, expected %d" % (db, cf.shape[1], WIDTH))
                    return 49
                real = cf[~d.skips.numpy()]
                if real.shape[0] == 0:
                    continue
                n_chunks += 1
                n_rows += real.shape[0]
                worst_true = max(worst_true, pair_err(real[:, CYC_LO:CYC_HI]))
                best_shift = max(best_shift, pair_err(real[:, CYC_LO - 1:CYC_HI - 1]))
                # first-review halves constant within a card: per period p, dims lo+4p+2, lo+4p+3
                # -- except the 7 d and 365 d periods, which have ONLY the first half (2 dims).
                cards = d.ids["card_id"].numpy()[~d.skips.numpy()]
                col = CYC_LO
                for p in (3, 7, 30, 100, 365.25, 3650, 36500):
                    if p not in (7, 365.25):
                        col += 2                      # skip the review-time half
                    first_sin = real[:, col]
                    for cid in np.unique(cards)[:50]:
                        v = first_sin[cards == cid]
                        if v.size > 1 and float(np.abs(v - v[0]).max()) > 0.02:
                            bad_const += 1
                    col += 2

    print("[cycles] %s" % db)
    print("[cycles] chunks %d  rows %d  width %d" % (n_chunks, n_rows, WIDTH))
    if n_rows == 0:
        print("[cycles] *** nothing measured -- VACUOUS, not a pass")
        return 48
    print("[cycles] cols %d..%d as 12 (sin,cos) pairs: max |sin^2+cos^2-1| = %.4f  (tol %.2f)"
          % (CYC_LO, CYC_HI - 1, worst_true, TOL))
    print("[cycles] negative control, cols %d..%d:     max |sin^2+cos^2-1| = %.4f  (must exceed tol)"
          % (CYC_LO - 1, CYC_HI - 2, best_shift))
    print("[cycles] first-review halves non-constant within a card: %d (must be 0)" % bad_const)
    ok = worst_true < TOL and best_shift > TOL and bad_const == 0
    print("[cycles] %s" % ("OK -- real-time cycles are IN THE ARTIFACT and well-formed" if ok
                            else "*** cycles missing or malformed"))
    return 0 if ok else 49


if __name__ == "__main__":
    raise SystemExit(main())
