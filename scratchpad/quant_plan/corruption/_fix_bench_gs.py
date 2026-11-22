"""One-shot: fix the borrow conflict in the compress_bench prototype's Gram-Schmidt.

`let d = z.column(l).dot(&z.column(j)); z.column_mut(j).axpy(-d, &z.column(l), 1.0);` borrows `z`
mutably and immutably at once (E0502). Copying column l into a small temp vector removes the conflict
and is the same arithmetic.
"""
from pathlib import Path

P = Path(__file__).resolve().parents[3] / "rust" / "rwkv-infer" / "src" / "model.rs"
text = P.open(encoding="utf-8", newline="").read()
old = ("                    for l in 0..j {\r\n"
       "                        let d = z.column(l).dot(&z.column(j));\r\n"
       "                        z.column_mut(j).axpy(-d, &z.column(l), 1.0);\r\n"
       "                    }")
new = ("                    for l in 0..j {\r\n"
       "                        let mut tmp = na::SVector::<f32, K>::zeros();\r\n"
       "                        tmp.copy_from(&z.column(l));\r\n"
       "                        let d = tmp.dot(&z.column(j));\r\n"
       "                        for i in 0..K {\r\n"
       "                            z[(i, j)] -= d * tmp[i];\r\n"
       "                        }\r\n"
       "                    }")
if old not in text:
    raise SystemExit("the marker is not there - nothing changed")
P.open("w", encoding="utf-8", newline="").write(text.replace(old, new, 1))
print("patched the Gram-Schmidt inner loop")
