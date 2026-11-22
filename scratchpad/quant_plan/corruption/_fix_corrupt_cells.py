"""Repair the four corruption cells written by _record_corrupt29_32.py.

Its ROWS tails ended in `| pending |`, so replacing the whole tail with `**PASS** (...)` also consumed
the CELL SEPARATOR in front of the corruption column, gluing the verdict onto the previous (gate) cell.

Fix: before every occurrence of the verdict marker, ensure the two preceding characters are `| `.
Rows 15-18 already carry the separator (their cells were written by an earlier, correct recorder), so
they are left untouched automatically.
"""
import sys
from pathlib import Path

RUN = Path(__file__).resolve().parents[1]
RESULTS = RUN / "RESULTS.md"
MARKER = b"**PASS** (158,986"
SEP = b"| "


def main() -> int:
    raw = RESULTS.read_bytes()
    out = bytearray()
    pos = 0
    fixed = 0
    while True:
        i = raw.find(MARKER, pos)
        if i < 0:
            out += raw[pos:]
            break
        before = raw[max(0, i - 2):i]
        out += raw[pos:i]
        if before != SEP:
            if before.endswith(b"|"):
                out += b" "          # separator present, but no space after it
            else:
                out += SEP
            fixed += 1
        out += MARKER
        pos = i + len(MARKER)
    if fixed == 0:
        print("nothing to fix (all verdict cells already separated)")
        return 0
    RESULTS.write_bytes(bytes(out))
    print(f"separators restored before {fixed} verdict cells")
    return 0


if __name__ == "__main__":
    sys.exit(main())
