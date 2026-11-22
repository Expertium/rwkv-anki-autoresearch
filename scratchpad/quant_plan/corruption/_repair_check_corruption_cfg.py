"""Repair the chain-14 insertion in corruption/check_corruption.py (literal `\\1\\n` written)."""
from pathlib import Path

p = Path(__file__).resolve().parent / "check_corruption.py"
text = p.open(encoding="utf-8", newline="").read()

bad = '\\1\\n    "l2i8n2i8"'
if bad not in text:
    raise SystemExit("the broken marker is not there - nothing to repair")

good = ('    "l3i8s4n4s2": (RUN / "qat_it36l3i8s4n4s2" / "qat_124.pth", "l3i8s4n4s2", "shrink point 4"),'
        '\r\n    "l2i8n2i8"')
text = text.replace(bad, good, 1)

with p.open("w", encoding="utf-8", newline="") as fh:
    fh.write(text)
print("repaired")
