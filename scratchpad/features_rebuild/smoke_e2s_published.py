"""RWKV_E2S_PUBLISHED: does the published-set end-to-start correction do exactly what it says?

Run BEFORE the rebuild, because a rebuild bakes the result into an LMDB and a wrong interval is
invisible afterwards -- the column has the same name, the same dtype and a plausible magnitude.

The five things that can go wrong, one case each:

  1. INERT when unset. Every existing db and run must be untouched.
  2. The ARITHMETIC is `elapsed_seconds - duration/1000`, subtracting THIS review's duration --
     not the previous one, which is the -id formula. Checked against an independent recompute.
  3. The -1 FIRST-REVIEW SENTINEL survives. A sentinel that became 0 would silently convert a
     first review into a same-instant repeat.
  4. NEGATIVES clamp to 0 *before* the int cast. Flooring -0.4 gives -1, which mints a fake first
     review -- the same sentinel-collision class that has bitten this pipeline twice.
  5. It REFUSES the -id set. Running both corrections would subtract two durations.

Usage: .venv/Scripts/python.exe scratchpad/features_rebuild/smoke_e2s_published.py [n_users]
CPU-only, seconds. Reads the real dataset; writes nothing.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

PUB = r"C:/Users/Andrew/anki-revlogs-10k/revlogs"


def main():
    n_users = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    import rwkv.id_features as idf

    users = [1, 2, 3, 17, 101, 333, 555, 1001][:n_users]
    fails = []

    def check(name, cond, detail=""):
        print(("  PASS  " if cond else "  FAIL  ") + name + (("  -- " + detail) if detail else ""))
        if not cond:
            fails.append(name)

    frames = {}
    for u in users:
        p = os.path.join(PUB, "user_id=%d" % u)
        if os.path.isdir(p):
            frames[u] = pd.read_parquet(p)
    if not frames:
        raise SystemExit("no users found under " + PUB)
    print("users loaded: %s  (%s rows)"
          % (sorted(frames), "{:,}".format(sum(len(d) for d in frames.values()))))

    print("\n[1] unset -> INERT")
    os.environ.pop("RWKV_E2S_PUBLISHED", None)
    same = all(idf.elapsed_end_to_start_published(d)["elapsed_seconds"].equals(d["elapsed_seconds"])
               for d in frames.values())
    check("every frame returned unchanged", same)

    print("\n[2] set -> the arithmetic is exactly elapsed_seconds - duration(k)/1000")
    os.environ["RWKV_E2S_PUBLISHED"] = "1"
    moved = kept = neg = 0
    for u, d in frames.items():
        out = idf.elapsed_end_to_start_published(d)["elapsed_seconds"].to_numpy()
        es = d["elapsed_seconds"].to_numpy().astype("float64")
        dur = d["duration"].to_numpy().astype("float64") / 1000.0
        want = np.floor(np.maximum(es - dur, 0.0))
        sent = es == -1
        want[sent] = -1.0
        if not np.array_equal(out, want.astype("int64")):
            check("user %d matches an independent recompute" % u, False)
            return 1
        moved += int(((out != es) & ~sent).sum())
        kept += int(sent.sum())
        neg += int(((es - dur < 0) & ~sent).sum())
    check("all users match an independent recompute", True,
          "%s rows changed" % "{:,}".format(moved))
    check("the correction is NOT a no-op", moved > 0, "%s rows changed" % "{:,}".format(moved))

    print("\n[3] the -1 first-review sentinel survives")
    d = frames[users[0]]
    out = idf.elapsed_end_to_start_published(d)["elapsed_seconds"].to_numpy()
    es = d["elapsed_seconds"].to_numpy()
    check("sentinel count preserved", int((out == -1).sum()) == int((es == -1).sum()),
          "%d vs %d" % (int((out == -1).sum()), int((es == -1).sum())))
    check("sentinels are exactly the same ROWS", bool(np.array_equal(out == -1, es == -1)))

    print("\n[4] negatives clamp to 0, never to the -1 sentinel")
    # a hand-made frame whose gap is smaller than the duration
    tiny = pd.DataFrame({"elapsed_seconds": [-1, 5, 30, 0], "duration": [9000, 9000, 1000, 400]})
    o = idf.elapsed_end_to_start_published(tiny)["elapsed_seconds"].to_numpy()
    check("first review stays -1", o[0] == -1, str(o))
    check("5 s gap with a 9 s review -> 0, not -1", o[1] == 0, str(o))
    check("30 s gap with a 1 s review -> 29", o[2] == 29, str(o))
    check("no new -1 was minted", int((o == -1).sum()) == 1, str(o))
    check("real data has rows that need the clamp", neg > 0, "%d rows" % neg)

    print("\n[5] it REFUSES the -id set (whose correction uses the other formula)")
    faux = pd.DataFrame({"elapsed_seconds": [10], "duration": [1000], "review_time": [1],
                         "card_id": [1]})
    try:
        idf.elapsed_end_to_start_published(faux)
        check("raises on a frame carrying review_time", False, "it did not raise")
    except AssertionError as e:
        check("raises on a frame carrying review_time", True, str(e)[:60] + "...")

    print()
    if fails:
        print("SMOKE FAILED: " + ", ".join(fails))
        return 1
    print("SMOKE PASSED: RWKV_E2S_PUBLISHED is inert when unset, computes "
          "elapsed_seconds - duration(k), preserves the sentinel, clamps negatives to 0, and "
          "refuses the -id set.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
