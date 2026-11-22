"""Restore the '## Live status' heading in NOTES.md (an earlier edit replaced it by accident)."""
import re
from pathlib import Path

NOTES = Path(__file__).resolve().parents[1] / "NOTES.md"
text = NOTES.open(encoding="utf-8", newline="").read()

if "## Live status" in text:
    raise SystemExit("heading already present - nothing to do")

text, n = re.subn(r"(?m)^(\* 2026-11-22 11:50 - )", r"## Live status\r\n\r\n\1", text, count=1)
if n != 1:
    raise SystemExit(f"expected 1 anchor, got {n}")

with NOTES.open("w", encoding="utf-8", newline="") as fh:
    fh.write(text)
print("heading restored")
