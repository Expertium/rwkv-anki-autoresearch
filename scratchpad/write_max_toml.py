"""Clone a WS toml with a different MAX_TRAIN_GLOBAL_LEN. For MAX sweeps only.

⚠ A sweep must compare **reviews_per_sec**, never steps_per_sec: changing MAX changes both how
many steps an epoch has and how much work a step does, so steps/s is not comparable across arms.
(`optimization/TRAINING_SPEED.md` records a whole round of results voided by that mistake.)
"""
import io
import sys


def main():
    if len(sys.argv) != 4:
        print("usage: write_max_toml.py <src.toml> <new_max> <dst.toml>")
        return 2
    src, new_max, dst = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    lines = io.open(src, encoding="utf-8").read().split("\n")
    out, seen = [], False
    for line in lines:
        if line.startswith("MAX_TRAIN_GLOBAL_LEN"):
            out.append("MAX_TRAIN_GLOBAL_LEN = %d" % new_max)
            seen = True
        else:
            out.append(line)
    # Assert on the OUTPUT, not on the input having been as expected: a silent no-op here would
    # make every arm of the sweep run at the SAME max and report a flat curve as a finding.
    assert seen, "MAX_TRAIN_GLOBAL_LEN not found in " + src
    text = "\n".join(out)
    assert ("MAX_TRAIN_GLOBAL_LEN = %d" % new_max) in text
    io.open(dst, "w", encoding="utf-8", newline="\n").write(text)
    print("wrote %s with MAX=%d" % (dst, new_max))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
