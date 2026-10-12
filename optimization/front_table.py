"""Insert a row into research_5k.md's front table IN NUMBER ORDER.

Andrew 2026-09-21: *"Please keep iterations in order"*. Ten record scripts each carried
`idx = [k for k, l in enumerate(s) if l.startswith("| 87 `lfwu20k`")]` + `s.insert(idx[0] + 1, ...)`
-- a HARDCODED anchor, so every new row landed directly beneath it and the block grew DOWNWARD in
reverse: 87, then 94, 93, 92, 91, 90, 89, 88. The numbers were right (they are assigned at verdict
time, AGENTS.md's completion-order rule); the ROWS read backwards.

The fix is to anchor on the DATA rather than on a string: append after the highest-numbered row
that exists, then ASSERT the whole column came out ascending. The assert is the part that matters
-- an anchor can drift silently, an ordering check cannot.
"""
import io
import re

FRONT = "optimization/research_5k.md"
# a front-table row: "| 94 `tag` ..." or "| **76** `tag` ...". Rows numbered F1-F6 (the
# features-phase baselines that landed between 61 and 62) carry no integer and are skipped.
ROW = re.compile(r"^\| \*{0,2}(\d+)\*{0,2}[ `|]")


def _nums(lines):
    return [(k, int(m.group(1))) for k, l in enumerate(lines) for m in [ROW.match(l)] if m]


def insert_row(front, path=FRONT):
    lines = io.open(path, encoding="utf-8").read().split("\n")
    rows = _nums(lines)
    assert rows, "no numbered rows found in " + path
    at = max(rows, key=lambda r: r[1])[0]
    lines.insert(at + 1, front)
    after = [n for _, n in _nums(lines)]
    assert after == sorted(after), (
        "front-table numbers are not ascending after the insert: %s"
        % [n for i, n in enumerate(after) if i and n < after[i - 1]])
    io.open(path, "w", encoding="utf-8", newline="\n").write("\n".join(lines))
    return at + 1


def check(path=FRONT):
    nums = [n for _, n in _nums(io.open(path, encoding="utf-8").read().split("\n"))]
    bad = [n for i, n in enumerate(nums) if i and n < nums[i - 1]]
    return nums, bad
