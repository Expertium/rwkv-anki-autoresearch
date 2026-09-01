"""Are two training runs BIT-IDENTICAL? Compares the printed loss trace line for line.

Exit 0 = identical. Exit 1 = they diverge (with the first divergence shown). Exit 2 = unusable
input (no losses found, or different step counts) -- which must NOT be read as "identical".

WHY EXIT 2 EXISTS AND IS SEPARATE. A comparison that finds nothing to compare would otherwise
return "no differences" and be read as a pass. That is the vacuous-guard failure this project has
hit before: a smoke that passes because it never ran the thing it was testing. If either trace has
no loss lines, that is a FAILURE OF THE TEST, not a property of the code.

The trace line looks like:
    0 705 706, all: 0.818597, ahead: 0.3006 (0.3006), imm: 0.435
`all` is the full-precision training loss and is the sensitive one -- a trajectory perturbation
shows up there first and grows.

Usage: cmp_traces.py <log_a> <log_b>
"""
import io
import re
import sys

PAT = re.compile(r"^\s*\d+\s+\d+\s+(\d+),\s*all:\s*([0-9.]+),\s*ahead:\s*([0-9.]+)")


def parse(path):
    out = []
    for line in io.open(path, encoding="utf-8", errors="replace"):
        m = PAT.match(line)
        if m:
            out.append((int(m.group(1)), m.group(2), m.group(3)))
    return out


def main():
    a, b = parse(sys.argv[1]), parse(sys.argv[2])
    print("trace A: %d loss lines   trace B: %d loss lines" % (len(a), len(b)))

    if not a or not b:
        print("  ! one trace has NO loss lines -- the test did not run. This is NOT a pass.")
        return 2
    if len(a) != len(b):
        print("  ! different step counts (%d vs %d) -- not comparable" % (len(a), len(b)))
        return 2

    for i, (ra, rb) in enumerate(zip(a, b)):
        if ra != rb:
            print("  ! FIRST DIVERGENCE at line %d (step %s):" % (i + 1, ra[0]))
            print("      A  all=%s  ahead=%s" % (ra[1], ra[2]))
            print("      B  all=%s  ahead=%s" % (rb[1], rb[2]))
            later = sum(1 for x, y in zip(a[i:], b[i:]) if x != y)
            print("      %d of the remaining %d lines differ" % (later, len(a) - i))
            return 1

    print("  BIT-IDENTICAL across all %d steps (all + ahead, full printed precision)" % len(a))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
