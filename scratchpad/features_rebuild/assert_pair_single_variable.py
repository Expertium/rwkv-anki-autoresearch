"""Assert two LMDBs differ in EXACTLY ONE thing: the interval features.

This is the guard for the fixc/e2s pair -- the control and treatment of the interval experiment.
Three conditions, and the third is the one that matters most:

  1. IDENTICAL entry counts.        A different count means rows were filtered differently, which
                                    is the confound that broke the srs-benchmark comparison.
  2. IDENTICAL id streams.          The NaN-placeholder fill does not depend on the interval, so
                                    card/note/deck/preset/user must match byte-for-byte. If they
                                    do not, the two dbs were built by different code and the pair
                                    measures the interval PLUS whatever else moved -- exactly the
                                    Bug C confound found between `_fix` and `_e2s`.
  3. card_features must DIFFER.     ⚠ THE ANTI-FALSE-GREEN CHECK. If `RWKV_E2S_PUBLISHED` leaked
                                    into the control's build, the "control" is a second copy of
                                    the treatment: conditions 1 and 2 would both PASS, the pair
                                    would look immaculate, and the experiment would measure
                                    nothing while reporting a clean null.

That third condition is the rgate lesson generalised. There, a control arm inherited its treatment
from `os.environ` and the inertness check passed VACUOUSLY at 0.000e+00 while comparing two treated
models. A pair-checker that only tests for SAMENESS can never catch that; it has to also require
the intended difference to be present.

Usage: assert_pair_single_variable.py <control_db> <treatment_db> [n_keys]
Exit 0 = a valid single-variable pair. Exit 3 = not one.
"""
import collections
import io
import sys

import lmdb
import torch


def main():
    a, b = sys.argv[1], sys.argv[2]
    want = int(sys.argv[3]) if len(sys.argv) > 3 else 40

    ea = lmdb.open(a, readonly=True, lock=False, subdir=True)
    eb = lmdb.open(b, readonly=True, lock=False, subdir=True)

    with ea.begin() as t:
        na = t.stat()["entries"]
    with eb.begin() as t:
        nb = t.stat()["entries"]

    problems = []
    print("A (control)   %-46s entries %s" % (a, f"{na:,}"))
    print("B (treatment) %-46s entries %s" % (b, f"{nb:,}"))
    if na != nb:
        problems.append("entry counts differ (%d vs %d)" % (na, nb))

    # collect a sample of id keys and feature keys
    with ea.begin() as t:
        cur = t.cursor()
        id_keys, feat_keys = [], []
        if cur.first():
            while len(id_keys) < want or len(feat_keys) < want:
                k = cur.key()
                if k.endswith(b"_id_") and len(id_keys) < want:
                    id_keys.append(k)
                elif k.endswith(b"card_features") and len(feat_keys) < want:
                    feat_keys.append(k)
                if not cur.next():
                    break

    def load(env, key):
        with env.begin() as t:
            raw = t.get(key)
        return torch.load(io.BytesIO(raw), weights_only=True, map_location="cpu") if raw else None

    # ---- 2. ids must be IDENTICAL ----
    id_diff = collections.Counter()
    id_tot = collections.Counter()
    for k in id_keys:
        va, vb = load(ea, k), load(eb, k)
        if va is None or vb is None:
            problems.append("id key missing on one side: %s" % k.decode())
            continue
        name = "_".join(k.decode()[:-4].rsplit("_", 2)[-2:])
        va, vb = va.reshape(-1), vb.reshape(-1)
        m = min(len(va), len(vb))
        id_tot[name] += m
        id_diff[name] += int((va[:m] != vb[:m]).sum())
    moved = sorted(n for n in id_tot if id_diff[n])
    print("\nid streams checked: %s" % ", ".join(sorted(id_tot)) or "(none)")
    if moved:
        for n in moved:
            print("   ! %s differs on %d of %d (%.2f%%)"
                  % (n, id_diff[n], id_tot[n], 100.0 * id_diff[n] / id_tot[n]))
        problems.append("id streams differ (%s) -- the dbs were built by different code" %
                        ", ".join(moved))
    else:
        print("   all id streams IDENTICAL")

    # ---- 3. features must DIFFER ----
    feat_tot = feat_diff = 0
    checked = 0
    for k in feat_keys:
        va, vb = load(ea, k), load(eb, k)
        if va is None or vb is None:
            continue
        checked += 1
        va, vb = va.reshape(-1), vb.reshape(-1)
        m = min(len(va), len(vb))
        feat_tot += m
        feat_diff += int((va[:m] != vb[:m]).sum())
    pct = 100.0 * feat_diff / max(feat_tot, 1)
    print("\ncard_features: %d of %d entries differ (%.3f%%) over %d keys"
          % (feat_diff, feat_tot, pct, checked))
    if checked and feat_diff == 0:
        problems.append("card_features are IDENTICAL -- the interval lever did not take, so the "
                        "'control' is a second copy of the treatment and the experiment measures "
                        "nothing")

    print()
    if problems:
        for p in problems:
            print("  ! " + p)
        print("PAIR_INVALID")
        return 3
    print("PAIR_VALID -- same rows, same entities, different intervals. Exactly one variable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
