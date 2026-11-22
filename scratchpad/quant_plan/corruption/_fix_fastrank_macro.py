"""One-shot fix for the two build errors in the E2 fast-rank path:

1. `rankr_fast_static<const K, const R>` cannot compile: `const B: usize = R + 1;` inside a
   generic fn is E0401 ("nested items are independent from their parent item"), and the
   const-generic form of nalgebra's fixed-size types cannot satisfy `ToTypenum` (the fixed-size
   SymmetricEigen needs a concrete dimension; the const-expr form needs the unstable
   `generic_const_exprs`). Replaced by a MACRO that pastes LITERAL sizes, which monomorphizes to
   exactly the same specialized code, with no dynamic allocation on the hot path. The macro is
   defined BEFORE its use because `macro_rules!` is textual.

2. main.rs constructs `fast::FastLayerState` in four places (the tensor<->fast state conversions
   and the two self-tests): each now needs the new `basis` field, seeded empty.

Math is unchanged from the version that was already reviewed: subspace iteration on the NORMALIZED
matrix, Rayleigh-Ritz on the (R+1)x(R+1) projection, sigma recomputed from the ORIGINAL A.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "rust" / "rwkv-infer" / "src"
print(f"repo root: {ROOT}")

NEW_BLOCK = '''/// FAST rank-r truncation for one KxK state: warm-started block subspace iteration + Rayleigh-Ritz.
/// `q` is this (head of this stream)'s block from the previous step, column-major, (R+1)*K floats; it is
/// overwritten with the new block. Returns false when the shape is not covered or the iteration went
/// degenerate, so the caller can fall back to the reference.
fn rankr_fast(
    a: &nalgebra::DMatrix<f32>,
    an: &nalgebra::DMatrix<f32>,
    k: usize,
    r: usize,
    q: &mut [f32],
    uf: &mut nalgebra::DMatrix<f32>,
    vf: &mut nalgebra::DMatrix<f32>,
) -> bool {
    match (k, r) {
        (32, 2) => rankr_fast_impl!(32, 2, a, an, &mut *q, &mut *uf, &mut *vf),
        (32, 3) => rankr_fast_impl!(32, 3, a, an, &mut *q, &mut *uf, &mut *vf),
        (32, 4) => rankr_fast_impl!(32, 4, a, an, &mut *q, &mut *uf, &mut *vf),
        (16, 2) => rankr_fast_impl!(16, 2, a, an, &mut *q, &mut *uf, &mut *vf),
        (8, 2) => rankr_fast_impl!(8, 2, a, an, &mut *q, &mut *uf, &mut *vf),
        _ => false, // any other shape: keep the reference path
    }
}

/// One (K, r) shape of the fast path, monomorphized by a MACRO rather than by const generics: the tracked
/// block width is B = R+1, a `const B: usize = R + 1;` inside a generic fn cannot be used in type position
/// (E0401), and the const-generic form of nalgebra's fixed-size types cannot satisfy `ToTypenum` (its
/// fixed-size SymmetricEigen needs a concrete dimension, and const-generic arithmetic in types needs the
/// unstable `generic_const_exprs`). Pasting LITERAL sizes gives the same specialized code with no dynamic
/// allocation on the hot path. `a`, `an`, `q`, `uf`, `vf` are passed in explicitly because macro hygiene
/// hides the caller's locals.
///
/// Math (the reference is `rankr_reference`, the definition of correct here): iterate the block towards
/// the dominant left subspace of the NORMALIZED `an` (Z = an an^T Q + reorthogonalization), then
/// Rayleigh-Ritz on the (R+1)x(R+1) projection G = (an^T Q)^T (an^T Q), whose top-R eigenvectors give the
/// best R directions inside the tracked block; the (R+1)-th is discarded. sigma is recomputed from the
/// ORIGINAL `a` and sqrt(sigma) is split into both factors, exactly as the reference does.
macro_rules! rankr_fast_impl {
    ($k_lit:literal, $r_lit:literal, $a:expr, $an:expr, $q:expr, $uf:expr, $vf:expr) => {{
        const K: usize = $k_lit;
        const R: usize = $r_lit;
        const B: usize = R + 1; // the R wanted directions + 1 extra, so a NEW one can still enter
        let a: &nalgebra::DMatrix<f32> = $a;
        let an: &nalgebra::DMatrix<f32> = $an;
        let q: &mut [f32] = $q;
        let uf: &mut nalgebra::DMatrix<f32> = $uf;
        let vf: &mut nalgebra::DMatrix<f32> = $vf;
        if q.len() < B * K {
            return false;
        }
        use nalgebra::{SMatrix, SymmetricEigen};
        let an_s = SMatrix::<f32, K, K>::from_fn(|i, j| an[(i, j)]);
        // Seed from the previous step's block; a zero, non-finite or degenerate seed means a cold start.
        let mut block = SMatrix::<f32, K, B>::zeros();
        let mut warm_ok = true;
        for j in 0..B {
            for i in 0..K {
                block[(i, j)] = q[j * K + i];
            }
            let n = block.column(j).norm();
            if !n.is_finite() || n < 1e-6 {
                warm_ok = false;
            }
        }
        if warm_ok && !mgs_into::<K, B>(&mut block) {
            warm_ok = false;
        }
        if !warm_ok {
            block = cold_block::<K, B>(&an_s);
        }
        for _ in 0..fast_iters() {
            let t = an_s.transpose() * block; // an^T Q   (KxB)
            let mut z = an_s * t; // an an^T Q (KxB) -- the KxK Gram matrix is never formed
            if !mgs_into::<K, B>(&mut z) {
                return false;
            }
            block = z;
        }
        let t = an_s.transpose() * block; // KxB
        let g = t.transpose() * t; // BxB: the projection of an an^T onto span(block)
        let eig = SymmetricEigen::new(g);
        let mut order: Vec<usize> = (0..B).collect();
        order.sort_by(|&i, &j| {
            let x = if eig.eigenvalues[i].is_finite() { eig.eigenvalues[i] } else { f32::NEG_INFINITY };
            let y = if eig.eigenvalues[j].is_finite() { eig.eigenvalues[j] } else { f32::NEG_INFINITY };
            y.partial_cmp(&x).unwrap()
        });
        let mut w = SMatrix::<f32, B, R>::zeros();
        for j in 0..R {
            let c = order[j];
            for i in 0..B {
                w[(i, j)] = eig.eigenvectors[(i, c)];
            }
        }
        let mut u = block * w; // KxR
        if !mgs_into::<K, R>(&mut u) {
            return false;
        }
        for j in 0..R {
            let u_col = u.column(j).into_owned(); // unit left singular vector of an (= of a)
            let v_un = a.transpose() * &u_col; // = sigma * v  (the ORIGINAL a -> the true sigma)
            let sigma = v_un.norm();
            if !sigma.is_finite() || sigma <= 1e-20 {
                continue;
            }
            let sj = sigma.sqrt();
            for i in 0..K {
                uf[(i, j)] = u_col[i] * sj;
                vf[(i, j)] = (v_un[i] / sigma) * sj;
            }
        }
        // next step's block: the R refined directions + one freshly refreshed extra direction
        for j in 0..R {
            for i in 0..K {
                block[(i, j)] = u[(i, j)];
            }
        }
        let mut rotated = block;
        refresh_extra::<K, R, B>(&an_s, &mut rotated);
        for j in 0..B {
            for i in 0..K {
                q[j * K + i] = rotated[(i, j)];
            }
        }
        true
    }};
}
'''


def load(p: Path):
    text = p.open(encoding="utf-8", newline="").read()
    nl = "\r\n" if text.count("\r\n") >= text.count("\n") - text.count("\r\n") else "\n"
    return text, nl


# ---------------------------------------------------------------- model.rs: the macro
p = SRC / "model.rs"
text, nl = load(p)
start = text.index("/// FAST rank-r truncation for one KxK state:")
end = text.index("/// In-place rank-r SVD-truncation + quantization of a flat row-major")
old = text[start:end]
assert "fn rankr_fast_static<const K: usize, const R: usize>(" in old, "the function to replace is not there"
assert "rankr_fast_static::<" in old, "the dispatcher does not call the function"
print(f"model.rs: replacing {len(old)} chars (rankr_fast + rankr_fast_static) with the macro block")
text = text[:start] + NEW_BLOCK.replace("\n", nl) + nl + text[end:]
# the r>=2 branch's comment pointed at the old function name
text = text.replace("see `rankr_fast_static`", "see `rankr_fast`")
assert "rankr_fast_static" not in text, "a stale reference to the old function survives"
p.open(encoding="utf-8", newline="", mode="w").write(text)
print("  wrote model.rs")

# ----------------------------------------------------------------------- main.rs: basis
p = SRC / "main.rs"
text, nl = load(p)
needle = "warm_shift: Vec::new(),"
lines = text.split(nl)
hits = 0
out = []
for line in lines:
    out.append(line)
    if line.strip() == needle:
        indent = line[: len(line) - len(line.lstrip())]
        out.append(indent + "basis: Vec::new()," + " // no fast-rank basis on a freshly built state")
        hits += 1
assert hits == 4, f"expected 4 FastLayerState literals, patched {hits}"
p.open(encoding="utf-8", newline="", mode="w").write(nl.join(out))
print(f"main.rs: added `basis: Vec::new(),` to {hits} FastLayerState literals")

print("\ndone - model.rs macro + main.rs basis fields")
