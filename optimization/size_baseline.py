"""Gate #1 (`size`), made concrete and LINEAGE-SCOPED.

Gate #1 says a candidate's per-user equalized review count must be IDENTICAL to the champion's,
because `size` is a property of the data and the filters -- so any change is a pipeline bug. That
is still exactly right. What was missing is *what defines a lineage*, and the answer turns out to
be narrower than "the database generation":

    `size` IS the stored `label_is_equalize` count, and that comes from the LABEL FILTER DB.

Verified two ways on 2026-09-02:
  * directly -- per-user equalized counts read out of `test_db_5k_e2s` match the `size` field in
    `RWKV-e2sc.jsonl` exactly (users 5001/5137/5613/6104/7499);
  * across four eval databases spanning the entire published lineage -- `test_db_5k` (July),
    `_fix` (08-21), `_fixc` (08-31) and `_e2s` (08-30) -- **0 per-user mismatches out of 2,500 and
    an identical 128,800,080 total**.

So the practical rule:

  | lineage                                   | label filter          | baseline |
  |-------------------------------------------|-----------------------|----------|
  | published, incl. `_fix` / `_fixc` / `_e2s`| `label_filter_db`     | one      |
  | `-id`, gen 3 and gen 4                    | `label_filter_db_id`  | another  |

**The end-to-start switch did NOT move `size`, so the published baseline carries across it
unbroken** -- which makes the gate more useful than it would be otherwise, because it still bridges
that transition and can still catch a pipeline bug there. What moves the baseline is swapping the
LABEL FILTER, i.e. moving to `-id`.

Corollary worth using: gen 3 and gen 4 share `label_filter_db_id`, so their sizes must be
IDENTICAL. A difference is a build bug, not a dataset property.

⚠ Our pipeline has no `delta_t > 0` filter -- srs-benchmark does, and that is why the interval
change deleted 0.172% of its reviews while ours keeps every row and only marks it. Do not carry
their behaviour over when reasoning about this gate.

Usage:
  size_baseline.py snapshot <lineage> <result.jsonl>    write optimization/size_baseline_<lineage>.json
  size_baseline.py check    <lineage> <result.jsonl>    compare; exit 0 pass, 1 mismatch, 2 unusable
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def path_for(lineage):
    return os.path.join(HERE, "size_baseline_%s.json" % lineage)


def sizes(jsonl):
    out = {}
    for line in open(jsonl):
        line = line.strip()
        if line:
            r = json.loads(line)
            if "size" in r:
                out[str(r["user"])] = r["size"]
    return out


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        return 2
    cmd, lineage, jsonl = sys.argv[1], sys.argv[2], sys.argv[3]
    if not os.path.exists(jsonl):
        print("missing result jsonl: %s" % jsonl)
        return 2
    cur = sizes(jsonl)
    if not cur:
        print("*** no `size` fields found in %s -- nothing measured, NOT a pass" % jsonl)
        return 2

    if cmd == "snapshot":
        blob = {
            "lineage": lineage,
            "source": os.path.basename(jsonl),
            "n_users": len(cur),
            "total": sum(cur.values()),
            "sizes": cur,
        }
        with open(path_for(lineage), "w", encoding="utf-8", newline="\n") as f:
            json.dump(blob, f, indent=1, sort_keys=True)
        print("wrote %s" % path_for(lineage))
        print("  lineage %s   users %d   total scored %d"
              % (lineage, len(cur), sum(cur.values())))
        return 0

    if cmd == "check":
        p = path_for(lineage)
        if not os.path.exists(p):
            print("*** no baseline for lineage '%s' (%s)." % (lineage, p))
            print("    A lineage without a baseline cannot be gated -- snapshot one FIRST from a")
            print("    run you trust, rather than treating the first candidate as the reference.")
            return 2
        base = json.load(open(p))["sizes"]
        common = sorted(set(base) & set(cur), key=int)
        if not common:
            print("*** no overlapping users -- nothing compared, NOT a pass")
            return 2
        bad = [u for u in common if base[u] != cur[u]]
        print("lineage %s   compared %d users (baseline %s)"
              % (lineage, len(common), os.path.basename(p)))
        print("  total scored: baseline %d   candidate %d"
              % (sum(base[u] for u in common), sum(cur[u] for u in common)))
        if bad:
            print("  *** SIZE GATE FAIL: %d of %d users differ" % (len(bad), len(common)))
            for u in bad[:10]:
                print("      user %s: baseline %d, candidate %d" % (u, base[u], cur[u]))
            if len(bad) > 10:
                print("      ... and %d more" % (len(bad) - 10))
            print("  `size` is a property of the data and the label filter, so a difference WITHIN")
            print("  a lineage is a pipeline bug. Across lineages, use that lineage's own baseline.")
            return 1
        print("  SIZE GATE PASS: 0 of %d users differ" % len(common))
        return 0

    print("unknown command %r" % cmd)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
