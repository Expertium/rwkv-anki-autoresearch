"""One-shot: collapse the stray blank line my NOTES.md bullet left after `## Live status`.

The heading line ends with CRLF and the bullet I inserted used LF, so the join produced two blank
lines. This rewrites only that junction and leaves the rest of the file byte-for-byte alone.
"""
import sys
from pathlib import Path

P = Path(__file__).resolve().parents[3] / "scratchpad" / "quant_plan" / "NOTES.md"
text = P.open(encoding="utf-8", newline="").read()
needle = "not a parser change.\n\r\n\r\n* 2026-11-22 11:50"
repl = "not a parser change.\r\n\r\n* 2026-11-22 11:50"
if needle not in text:
    print("junction not found - nothing to do")
    sys.exit(0)
P.open("w", encoding="utf-8", newline="").write(text.replace(needle, repl, 1))
print("collapsed the junction after the 12:55 bullet")
