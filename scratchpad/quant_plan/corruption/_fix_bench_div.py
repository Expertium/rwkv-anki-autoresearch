"""One-shot: fix the one line the Rust compiler rejected in the compress_bench prototype.

`z.column_mut(j) /= n;` is not a valid left-hand side (E0067) - a column view is not a place you can
compound-assign to with nalgebra's types here. Replaced with an explicit element loop.
"""
from pathlib import Path

P = Path(__file__).resolve().parents[3] / "rust" / "rwkv-infer" / "src" / "model.rs"
text = P.open(encoding="utf-8", newline="").read()
old = "                if n.is_finite() && n > 1e-30 {\r\n                    z.column_mut(j) /= n;\r\n                }"
new = ("                if n.is_finite() && n > 1e-30 {\r\n"
       "                    for i in 0..K {\r\n"
       "                        z[(i, j)] /= n;\r\n"
       "                    }\r\n"
       "                }")
if old not in text:
    raise SystemExit("the marker is not there - nothing changed")
P.open("w", encoding="utf-8", newline="").write(text.replace(old, new, 1))
print("patched the normalization loop")
