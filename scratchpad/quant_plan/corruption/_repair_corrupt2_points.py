"""Repair the chain-14 insertion in it36corrupt2_chain.py (the replacement wrote a literal `\1\n`)."""
from pathlib import Path

p = Path(__file__).resolve().parents[1] / "it36corrupt2_chain.py"
text = p.open(encoding="utf-8", newline="").read()

bad = '    \\1\\n    ("l2i8n2i8"'
if bad not in text:
    raise SystemExit("the broken marker is not there - nothing to repair")

good = ('    ("l3i8s4n4s2", "it36l3i8s4n4s2", "shrink point 4 (note int4 + int2 shifts)"),'
        '\r\n    ("l2i8n2i8"')
text = text.replace(bad, good, 1)

with p.open("w", encoding="utf-8", newline="") as fh:
    fh.write(text)
print("repaired")
