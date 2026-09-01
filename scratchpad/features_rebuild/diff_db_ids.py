"""Do two LMDBs store the SAME entity ids for the same keys? A confound check, not a smoke test.

WHY. `train_db_5k_h1_fix` / `test_db_5k_fix` were built 2026-08-21. The Bug C fix (`nan_id_fill`,
int64 placeholder arithmetic now SHARED by training and deploy) landed 2026-08-26. The `_e2s` dbs
were built 2026-08-30, i.e. WITH it. If that fix changed stored ids, the e2s arm is not a
single-variable interval experiment -- it bundles the interval change with a note-identity fix, and
the sibling of that fix (Bug A) was independently worth +0.000148 / +0.000169.

Entry counts already match (170,384 / 1,483,984), so structure is not the question. VALUES are.

The Bug C signature is a JUMP IN DISTINCT ids on the NaN-filled streams (note_id above all), with
`card_id` untouched -- card_id is never placeholder-filled. If card_id also moved, the cause is
something else and must be found before either db is trusted.

⚠ SCHEMA: ids are NOT stored inside one pickled sample. Each stream is its own key,
"<user>_<range>_<n>_<stream>_id_", holding a tensor. The first version of this file assumed a dict
per key and mis-parsed the stream name, collapsing all five streams into one bucket -- it reported
a real 4.57% difference with no way to say whether it lived in note_id or in card_id, which mean
completely different things.

Usage: diff_db_ids.py <db_a> <db_b> [n_id_keys]
Exit 0 = ids identical on the sample. Exit 3 = they differ (a confound).
"""
import collections
import io
import sys

import lmdb
import torch


def stream_of(key):
    t = key.decode()
    assert t.endswith("_id_"), t
    return "_".join(t[:-4].rsplit("_", 2)[-2:])


def main():
    a, b = sys.argv[1], sys.argv[2]
    want = int(sys.argv[3]) if len(sys.argv) > 3 else 200

    ea = lmdb.open(a, readonly=True, lock=False, subdir=True)
    eb = lmdb.open(b, readonly=True, lock=False, subdir=True)

    with ea.begin() as txn:
        cur = txn.cursor()
        keys = []
        if cur.first():
            while len(keys) < want:
                k = cur.key()
                if k.endswith(b"_id_"):
                    keys.append(k)
                if not cur.next():
                    break

    def load(env, key):
        with env.begin() as txn:
            raw = txn.get(key)
        return torch.load(io.BytesIO(raw), weights_only=True, map_location="cpu") if raw else None

    tot = collections.Counter()
    dif = collections.Counter()
    da = collections.defaultdict(set)
    db = collections.defaultdict(set)
    missing = 0

    for k in keys:
        va, vb = load(ea, k), load(eb, k)
        if va is None or vb is None:
            missing += 1
            continue
        name = stream_of(k)
        va, vb = va.reshape(-1), vb.reshape(-1)
        m = min(len(va), len(vb))
        tot[name] += m
        dif[name] += int((va[:m] != vb[:m]).sum())
        da[name].update(va[:m].tolist())
        db[name].update(vb[:m].tolist())

    print("id keys sampled %d   (missing on one side: %d)" % (len(keys), missing))
    print()
    print("%-12s %13s %13s %9s   %11s %11s"
          % ("stream", "entries", "DIFFERING", "pct", "distinct A", "distinct B"))
    for name in sorted(tot):
        pct = 100.0 * dif[name] / max(tot[name], 1)
        print("%-12s %13d %13d %8.3f%%   %11d %11d"
              % (name, tot[name], dif[name], pct, len(da[name]), len(db[name])))

    print()
    if not sum(dif.values()):
        print("=> IDS IDENTICAL on this sample. The dbs differ only in what they were meant to.")
        return 0

    moved = [n for n in tot if dif[n]]
    print("=> IDS DIFFER on: %s" % ", ".join(sorted(moved)))
    if "card_id" in moved:
        print("   ⚠ card_id MOVED. card_id is never placeholder-filled, so this is NOT Bug C and")
        print("     the cause must be identified before either db is used for anything.")
    else:
        print("   Confined to placeholder-filled streams, and distinct counts ROSE -- the Bug C")
        print("     signature (int64 fill restoring per-card note placeholders that float64")
        print("     rounding had merged). Real, explained, and still a CONFOUND: any experiment")
        print("     spanning these two dbs measures the interval change PLUS this.")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
