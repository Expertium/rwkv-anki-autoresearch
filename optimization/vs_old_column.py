"""Maintain research_5k.md's "vs old" column: how far each row still is from the OLD d=128 model.

Andrew 2026-08-12: "add a column that shows by how much log loss is worse compared to the old
model, with negative being better (we're not there yet though)."

    value = row - baseline,  POSITIVE = still worse than the old model, NEGATIVE = beaten it.

BASELINE = the old d=128 leaderboard model (`pretrain/RWKV_trained_on_101_4999.pth`, unquantized)
restricted to the **VAL half 5001-7500** -- the only user set candidates are ever scored on, so it
is the like-for-like target. Its full-range 5001-10000 numbers (0.296385 / 0.264905) are NOT the
right comparison for these rows and are deliberately not used here.

⚠ ONE HONEST CAVEAT, and it makes the AHEAD column PESSIMISTIC. The baseline was measured
2026-07-03, before the rectified gate; rows from iter 33 on are RECTIFIED (the deploy-honest
metric, marked with the vr superscript). Rectification costs `ahead` roughly +0.0019..0.0036
depending on the model, and it is a cost the baseline never paid. So the true ahead gap is smaller
than this column shows by about that much. `imm` is closer to comparable -- the rectifier does not
touch the rating head -- up to the ~0.0003 probe-insertion noise.
This column is therefore a PROGRESS INDICATOR, not a gate. The gate is always vs the current
champion (see CLAUDE.md "ACCEPTANCE GATE").

Idempotent: inserts the column if missing, recomputes it if present. Re-run after adding a row.

    python optimization/vs_old_column.py [--check]
"""
import io
import os
import re
import sys

MD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "research_5k.md")
BASE_AHEAD = 0.294612
BASE_IMM = 0.263561
COL = "vs old (a / i)"
NUM = re.compile(r"^\**([0-9]*\.?[0-9]+)")


def fmt(row_val, base):
    return f"{row_val - base:+.4f}"


def main():
    check = "--check" in sys.argv
    text = io.open(MD, encoding="utf-8").read()
    lines = text.split("\n")
    h = next(i for i, l in enumerate(lines) if l.startswith("| iter | trained on"))
    end = next((i for i in range(h + 2, len(lines)) if not lines[i].startswith("| ")), len(lines))

    hdr = [c.strip() for c in lines[h].split("|")][1:-1]
    have = COL in hdr
    pos = hdr.index(COL) if have else hdr.index("imm") + 1

    changed = 0
    for i in range(h, end):
        cells = [c.strip() for c in lines[i].split("|")][1:-1]
        if i == h:
            new = COL
        elif i == h + 1:
            new = "---"
        else:
            # The reference row IS the baseline, and its printed numbers are the FULL
            # 5001-10000 range, not the VAL half -- so a delta there would compare two different
            # user sets and read as though the baseline were 0.0018 worse than itself.
            status = cells[hdr.index("status") if have else hdr.index("status")]
            if "target" in status or "reference" in status:
                new = "— (ref)"
            else:
                a, m = NUM.match(cells[2]), NUM.match(cells[3])
                new = (f"{fmt(float(a.group(1)), BASE_AHEAD)} / {fmt(float(m.group(1)), BASE_IMM)}"
                       if a and m else "—")
        if have:
            if cells[pos] != new:
                cells[pos] = new
                changed += 1
        else:
            cells.insert(pos, new)
            changed += 1
        lines[i] = "| " + " | ".join(cells) + " |"

    if check:
        print(f"{'would change' if changed else 'up to date'}: {changed} cell(s)")
        return 1 if changed else 0
    io.open(MD, "w", encoding="utf-8", newline="\n").write("\n".join(lines))
    print(f"{'inserted' if not have else 'refreshed'} '{COL}' column "
          f"({end - h - 2} rows, {changed} cell(s) written)")
    print(f"baseline (VAL half 5001-7500): ahead {BASE_AHEAD} / imm {BASE_IMM}; "
          f"positive = still worse than the old d=128 model")
    return 0


if __name__ == "__main__":
    sys.exit(main())
