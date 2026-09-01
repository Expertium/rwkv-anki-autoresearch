"""Clone the champion WS toml with a different MAX_TRAIN_GLOBAL_LEN, for the speed sweep.

Usage: python scratchpad/dispatch/write_max_toml.py <src_toml> <max_len> <out_toml>

MAX is the WKV batch dimension. It is NOT a pure speed lever -- it also sets how many groups (and
therefore how many optimizer steps) an epoch has, which is why iter 34's move to 65536 cost
0.0003 at the old LR. This generator exists for BENCHMARKS only; adopting a different MAX for a
real run needs the LR retuned with it (phase 5).

The asserts are the point: a benchmark that silently ran at the wrong MAX would compare two
identical configs and report a null.
"""
import io
import re
import sys


def main():
    src, max_len, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    s = io.open(src, encoding="utf-8").read()

    pat = re.compile(r"^MAX_TRAIN_GLOBAL_LEN[ \t]*=[ \t]*(\d+)[ \t]*$", re.M)
    m = pat.search(s)
    assert m, "no MAX_TRAIN_GLOBAL_LEN line in " + src
    old = int(m.group(1))
    s2 = pat.sub("MAX_TRAIN_GLOBAL_LEN = %d" % max_len, s, count=1)

    # Assert on the OUTPUT, not on the intent -- the mk53/mk54 lesson.
    m2 = pat.search(s2)
    assert m2 and int(m2.group(1)) == max_len, "the rewrite did not take"
    assert s2.count("MAX_TRAIN_GLOBAL_LEN") == s.count("MAX_TRAIN_GLOBAL_LEN")
    # Nothing data-affecting may move: same db, same users, same fetch processes.
    for key in ("TRAIN_DATASET_LMDB_PATH", "VALIDATE_DATASET_LMDB_PATH",
                "LABEL_FILTER_LMDB_PATH", "NUM_FETCH_PROCESSES", "EPOCHS"):
        a = re.search("^" + key + r"\s*=.*$", s, re.M)
        b = re.search("^" + key + r"\s*=.*$", s2, re.M)
        assert (a is None) == (b is None) and (a is None or a.group(0) == b.group(0)), key

    io.open(out, "w", encoding="utf-8", newline="\n").write(s2)
    print("wrote %s   MAX %d -> %d" % (out, old, max_len))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
