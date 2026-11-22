"""One-shot: fix the bench's reconstruction-error metric.

`(uf * vf.transpose()).iter().copied().collect()` iterates a nalgebra SMatrix in COLUMN-major order,
while `compress_wkv_state`'s flat buffer is ROW-major. Comparing the two compared a matrix with its
transpose, which is why every configuration reported a constant relative error of ~1.4 no matter how
many iterations it ran. Extract row-major instead.
"""
from pathlib import Path

P = Path(__file__).resolve().parents[3] / "rust" / "rwkv-infer" / "src" / "model.rs"
text = P.open(encoding="utf-8", newline="").read()
helper = ("    /// Flatten a KxK matrix to the ROW-major layout `compress_wkv_state` works in (nalgebra's\n"
          "    /// own iterator is column-major, so a direct comparison would compare a matrix with its\n"
          "    /// transpose).\n"
          "    fn row_major(m: &na::SMatrix<f32, K, K>) -> Vec<f32> {\n"
          "        let mut out = Vec::with_capacity(K * K);\n"
          "        for r in 0..K {\n"
          "            for c in 0..K {\n"
          "                out.push(m[(r, c)]);\n"
          "            }\n"
          "        }\n"
          "        out\n"
          "    }\n\n")
anchor = "    fn rel_frob(x: &[f32], y: &[f32]) -> f32 {"
if "fn row_major" in text:
    raise SystemExit("the helper is already there; nothing to do")
text = text.replace(anchor, helper + anchor, 1)
old = "let proto: Vec<f32> = (uf * vf.transpose()).iter().copied().collect();"
new = "let proto: Vec<f32> = row_major(&(uf * vf.transpose()));"
n = text.count(old)
text = text.replace(old, new)
P.open("w", encoding="utf-8", newline="").write(text)
print(f"added the row-major helper and replaced {n} comparisons")
