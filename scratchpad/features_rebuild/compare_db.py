"""Assert two LMDBs are structurally IDENTICAL -- same entry count, same stored feature width.

WHY THIS EXISTS, AND WHY A THRESHOLD IS NOT ENOUGH. `check_db.py` gates on `entries >= min`, which
catches a db that never got built. It cannot catch a db that got built with a DIFFERENT NUMBER OF
ROWS -- and that is the failure that just cost srs-benchmark a result.

Andrew, 2026-08-30: end-to-start scored 0.318275 against end-to-end's 0.317944, and two thirds of
that gap turned out to be the DENOMINATOR: `delta_t > 0` deleted 0.172% of reviews whose corrected
gap floored to zero, and those rows were 2.7x EASIER than average (6.09% failure vs 16.14%).
Deleting the easiest rows raises mean logloss all by itself. With sizes matched the gap fell to
0.000111.

Our pipeline should not have that problem -- we have no `delta_t > 0` filter, we keep every row and
only mark which ones count, and the e2s tomls reuse the same `label_filter_db` and user ranges. But
"should not" is a prediction, and gate #1 exists precisely because a review-count change is the
signature of a pipeline bug. So it gets CHECKED, not assumed.

⚠⚠ SCOPE -- THIS COMPARES DBS BUILT THE SAME WAY, AND IS TOO STRICT ACROSS GENERATIONS
(learned 2026-08-31). `test_db_5k` holds 340,576 entries and `test_db_5k_e2s` holds 170,384,
almost exactly 2x -- and that gap is CHUNKING, not content. Checked against the evals themselves:
featA2 (on the 170,384-entry db) and iter 53 (on the 340,576-entry one) report **0 per-user `size`
mismatches over 2,500 users and an identical 128,800,080 total reviews scored**. The same reviews
are scored either way; only the number of LMDB records they are packed into differs.

So an entry-count difference here means "these were built with different chunking" and NOT
necessarily "these score different reviews". Use this script for arms of one experiment, built by
the same code in the same generation (fixc vs e2s), where a difference really would be a bug.

**The authoritative gate-#1 check is per-user `size` in the eval jsonls**, because that is the
quantity the gate is actually about. Had this script been treated as the last word, it would have
rejected a perfectly valid champion comparison.

Exit 0 = identical. Exit 3 = a real difference. Exit 2 = could not open.

Usage: compare_db.py <db_a> <db_b> [expected_width]
"""
import sys

import lmdb
import torch


def stats(path, want_width):
    try:
        env = lmdb.open(path, readonly=True, lock=False, subdir=True)
    except lmdb.Error as e:
        print("FAIL %s: cannot open (%s)" % (path, e))
        sys.exit(2)
    with env.begin() as txn:
        entries = txn.stat()["entries"]
        width = None
        if want_width is not None:
            cur = txn.cursor()
            if cur.first():
                # the stored value is a serialized sample; the feature matrix is [T, width]
                try:
                    obj = torch.load(__import__("io").BytesIO(cur.value()), weights_only=False)
                except Exception as e:                      # noqa: BLE001 - diagnostic only
                    print("  (width unreadable on %s: %s)" % (path, e))
                    obj = None
                if obj is not None:
                    for v in (obj.values() if isinstance(obj, dict) else []):
                        if torch.is_tensor(v) and v.dim() == 2:
                            width = int(v.shape[1])
                            break
    env.close()
    return entries, width


def main():
    a, b = sys.argv[1], sys.argv[2]
    want = int(sys.argv[3]) if len(sys.argv) > 3 else None

    ea, wa = stats(a, want)
    eb, wb = stats(b, want)
    print("A %-46s entries %12s  width %s" % (a, f"{ea:,}", wa))
    print("B %-46s entries %12s  width %s" % (b, f"{eb:,}", wb))

    bad = []
    if ea != eb:
        bad.append("ENTRY COUNT DIFFERS by %d (%.4f%%) -- rows were filtered differently, which is"
                   " exactly the confound that broke the srs-benchmark interval comparison"
                   % (eb - ea, 100.0 * (eb - ea) / max(ea, 1)))
    if wa is not None and wb is not None and wa != wb:
        bad.append("WIDTH DIFFERS: %s vs %s -- the arms do not share a feature layout" % (wa, wb))
    if want is not None:
        for name, w in ((a, wa), (b, wb)):
            if w is not None and w != want:
                bad.append("%s width %s, expected %s" % (name, w, want))

    if bad:
        for x in bad:
            print("  ! " + x)
        print("COMPARE_DB FAIL")
        return 3
    print("COMPARE_DB PASS -- identical entry count%s" % ("" if wa is None else " and width"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
