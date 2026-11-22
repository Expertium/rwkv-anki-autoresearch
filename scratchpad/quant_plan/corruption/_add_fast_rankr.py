"""One-shot: add the opt-in FAST rank-r truncation to rust/rwkv-infer/src/model.rs.

WHY. `compress_wkv_state` truncates a KxK WKV state to rank r. At r >= 2 it forms the KxK Gram matrix and
calls `nalgebra::SymmetricEigen` on it, which the new ignored bench measures at ~35 us of a ~40 us call
(the rank-1 power-iteration path is ~10 us). That is why a rank-2 NOTE costs 0.600x rev/s and FAILS the
0.75x floor, while rank 2 on the 1-layer card passes at 0.809x. Chain 14's `l2i8n2i8` (rank-2 note) is the
best row on the loss gate so far (ahead -0.000115, imm +0.000103 against `qnj_it36ctl`), so the speed is
now the only thing standing between it and deploy.

HOW. `RWKV_LOWRANK_FAST=1` replaces the Gram + eigendecomposition with a warm-started BLOCK subspace
iteration on fixed-size arrays that never forms the Gram (which alone costs k^3 MACs):
  * the block is Kx(R+1): the R wanted directions plus ONE extra direction, refreshed every call from the
    strongest column of A, so a growing new direction can enter the tracked subspace;
  * `RWKV_LOWRANK_FAST_ITERS` (default 2) steps of Z = A A^T Q followed by orthonormalization;
  * a Rayleigh-Ritz step (project A A^T onto the block, solve the tiny (R+1)x(R+1) eigenproblem exactly,
    keep the top R) makes the returned vectors the best R within the tracked block;
  * the basis travels with the entity's state, so the next step starts warm.
`RWKV_LOWRANK_FAST_CHECK=1` runs BOTH paths and records the largest relative Frobenius difference between
the two rank-r reconstructions, while still WRITING the reference factors, so a checked run reproduces the
reference run exactly and only the difference statistic is new. Any shape the fast solver does not cover
(k, r not in the dispatch table) falls back to the reference path.

The reference math is NOT touched: `rankr_reference` is the original code moved into a function, and this
script asserts line-by-line (whitespace-insensitive) that it contains every original line.
"""
from pathlib import Path
import re

P = Path(__file__).resolve().parents[3] / "rust" / "rwkv-infer" / "src" / "model.rs"
text = P.open(encoding="utf-8", newline="").read()
if "RWKV_LOWRANK_FAST" in text:
    raise SystemExit("the fast rank-r path is already present; nothing to do")

DOC = "/// In-place rank-r SVD-truncation + quantization of a flat row-major (H,K,K) WKV state."
SIG_OLD = "    // task25: joint-path warm indices for THIS entity, layout [head*2 + rank_component], -1 = none\n    // (len >= h*2 when Some). Only the joint PQ path uses it; everything else ignores it.\n    mut warm: Option<&mut [i32]>,\n"
ELSE_START = "        } else {\n            // Top-r (r>1, or ALS) via symmetric eigendecomposition of the Gram G = an an^T (KxK PSD):"
ELSE_END = "        if let Some(pq) = pq {"

NL = "\r\n" if text.count("\r\n") > text.count("\n") - text.count("\r\n") else "\n"
if NL != "\n":
    DOC = DOC.replace("\n", NL)
    SIG_OLD = SIG_OLD.replace("\n", NL)
    ELSE_START = ELSE_START.replace("\n", NL)
    ELSE_END = ELSE_END.replace("\n", NL)

for name, needle in (("DOC", DOC), ("SIG_OLD", SIG_OLD), ("ELSE_START", ELSE_START), ("ELSE_END", ELSE_END)):
    if needle not in text:
        raise SystemExit(f"anchor {name} not found (line endings?) - aborting without changes")

# ---------------------------------------------------------------- new helper functions
HELPERS = r'''/// `RWKV_LOWRANK_FAST=1` switches the rank>=2 truncation from the reference KxK Gram + symmetric
/// eigendecomposition (~35 us on a 32x32 state, measured by the `compress_bench` test) to a warm-started
/// block subspace iteration on fixed-size arrays (a few us), which never forms the Gram matrix. OFF by
/// default: every published number keeps describing the reference path.
pub fn lowrank_fast_on() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| flag("RWKV_LOWRANK_FAST"))
}

/// `RWKV_LOWRANK_FAST_CHECK=1` runs BOTH paths and records the largest relative Frobenius difference
/// between the two rank-r reconstructions. The run still WRITES the reference factors, so a checked run
/// reproduces the reference run exactly and only the difference statistic is new.
pub fn lowrank_fast_check() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| flag("RWKV_LOWRANK_FAST_CHECK"))
}

fn flag(name: &str) -> bool {
    matches!(std::env::var(name).unwrap_or_default().trim().to_ascii_lowercase().as_str(),
             "1" | "true" | "on" | "yes")
}

/// Subspace-iteration steps per call; the state moves slowly, so two is normally plenty.
fn fast_iters() -> usize {
    use std::sync::OnceLock;
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("RWKV_LOWRANK_FAST_ITERS").ok().and_then(|v| v.trim().parse().ok())
            .filter(|&n| (1..=16).contains(&n)).unwrap_or(2)
    })
}

static FAST_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static FAST_FALLBACK: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// f32::to_bits() of a non-negative finite float is monotone, so a plain atomic max works.
static FAST_MAX_REL: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Report the fast-path counters (called from `main` next to the finiteness report).
pub fn fastpath_report() {
    use std::sync::atomic::Ordering;
    if !lowrank_fast_on() && !lowrank_fast_check() {
        return;
    }
    let calls = FAST_CALLS.load(Ordering::Relaxed);
    let fb = FAST_FALLBACK.load(Ordering::Relaxed);
    let worst = f32::from_bits(FAST_MAX_REL.load(Ordering::Relaxed) as u32);
    eprintln!(
        "[fastrank] mode={}{} iters={} rank>=2 calls={calls} reference fallbacks={fb} \
         max_relFrobenius_vs_reference={worst:.3e}",
        if lowrank_fast_on() { "fast" } else { "measure-only" },
        if lowrank_fast_check() { "+check" } else { "" },
        fast_iters()
    );
}

/// Record the relative Frobenius difference of one rank-r reconstruction against the reference.
fn record_rel_diff(fast: &nalgebra::DMatrix<f32>, reference: &nalgebra::DMatrix<f32>) {
    use std::sync::atomic::Ordering;
    let num: f32 = fast.iter().zip(reference.iter()).map(|(x, y)| (x - y) * (x - y)).sum::<f32>().sqrt();
    let den: f32 = reference.iter().map(|y| y * y).sum::<f32>().sqrt().max(1e-30);
    let rel = num / den;
    if !rel.is_finite() {
        return;
    }
    let bits = rel.to_bits() as u64;
    let mut cur = FAST_MAX_REL.load(Ordering::Relaxed);
    while bits > cur {
        match FAST_MAX_REL.compare_exchange_weak(cur, bits, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => return,
            Err(seen) => cur = seen,
        }
    }
}

/// Modified Gram-Schmidt with one reorthogonalization pass; false if a column is not usable.
fn mgs_into<const K: usize, const R: usize>(z: &mut nalgebra::SMatrix<f32, K, R>) -> bool {
    for j in 0..R {
        for _ in 0..2 {
            for l in 0..j {
                let mut tmp = nalgebra::SVector::<f32, K>::zeros();
                tmp.copy_from(&z.column(l));
                let d = tmp.dot(&z.column(j));
                for i in 0..K {
                    z[(i, j)] -= d * tmp[i];
                }
            }
        }
        let n = z.column(j).norm();
        if !n.is_finite() || n < 1e-20 {
            return false;
        }
        for i in 0..K {
            z[(i, j)] /= n;
        }
    }
    true
}

/// Cold block: the B columns of A with the largest norm, orthonormalized. The top left singular vectors
/// live in range(A), so this is a far better start than the identity.
fn cold_block<const K: usize, const B: usize>(a: &nalgebra::SMatrix<f32, K, K>) -> nalgebra::SMatrix<f32, K, B> {
    let mut order: Vec<usize> = (0..K).collect();
    let mut norms: Vec<f32> = (0..K).map(|c| (0..K).map(|r| a[(r, c)] * a[(r, c)]).sum()).collect();
    order.sort_by(|&i, &j| norms[j].partial_cmp(&norms[i]).unwrap_or(std::cmp::Ordering::Equal));
    norms.clear();
    let mut q = nalgebra::SMatrix::<f32, K, B>::zeros();
    let mut filled = 0;
    for &c in &order {
        if filled >= B {
            break;
        }
        let mut v = a.column(c).into_owned();
        for j in 0..filled {
            let d = q.column(j).dot(&v);
            v -= &q.column(j) * d;
        }
        let n = v.norm();
        if n.is_finite() && n > 1e-20 {
            q.set_column(filled, &(v / n));
            filled += 1;
        }
    }
    q
}

/// The extra tracked direction: the strongest column of A with the top-R directions removed. Refreshing it
/// every call is what lets a NEW dominant direction enter the subspace (a pure iteration only preserves
/// the span it already has).
fn refresh_extra<const K: usize, const R: usize, const B: usize>(
    an: &nalgebra::SMatrix<f32, K, K>,
    q: &mut nalgebra::SMatrix<f32, K, B>,
) {
    let mut best = 0usize;
    let mut best_n = -1.0f32;
    for c in 0..K {
        let n: f32 = (0..K).map(|r| an[(r, c)] * an[(r, c)]).sum();
        if n > best_n {
            best_n = n;
            best = c;
        }
    }
    let mut v = an.column(best).into_owned();
    for j in 0..R {
        let mut tmp = nalgebra::SVector::<f32, K>::zeros();
        tmp.copy_from(&q.column(j));
        let d = tmp.dot(&v);
        for i in 0..K {
            v[i] -= d * tmp[i];
        }
    }
    let n = v.norm();
    if n.is_finite() && n > 1e-20 {
        q.set_column(R, &(v / n));
    } else {
        let keep = q.column(R).into_owned();
        q.set_column(R, &keep);
    }
}

/// The REFERENCE rank-r truncation (unchanged math): Gram matrix + symmetric eigendecomposition, top-r
/// eigenpairs, sigma from the ORIGINAL A, sqrt(sigma) split into both factors.
fn rankr_reference(
    a: &nalgebra::DMatrix<f32>,
    an: &nalgebra::DMatrix<f32>,
    k: usize,
    r: usize,
    scale: f32,
    uf: &mut nalgebra::DMatrix<f32>,
    vf: &mut nalgebra::DMatrix<f32>,
) {
    let gram = an * an.transpose();
    let eig = nalgebra::SymmetricEigen::new(gram);
    let evals = &eig.eigenvalues;
    let mut order: Vec<usize> = (0..k).collect();
    order.sort_by(|&i, &j| {
        let a = if evals[i].is_finite() { evals[i] } else { f32::NEG_INFINITY };
        let b = if evals[j].is_finite() { evals[j] } else { f32::NEG_INFINITY };
        b.partial_cmp(&a).unwrap()
    });
    for j in 0..r {
        let col = order[j];
        let ev = evals[col];
        if !ev.is_finite() || ev <= 0.0 {
            continue;
        }
        let sigma = ev.sqrt() * scale;
        if sigma > 1e-20 && sigma.is_finite() {
            let sj = sigma.sqrt();
            let u_col = eig.eigenvectors.column(col).into_owned();
            let v_unscaled = a.transpose() * &u_col;
            for i in 0..k {
                uf[(i, j)] = u_col[i] * sj;
                vf[(i, j)] = (v_unscaled[i] / sigma) * sj;
            }
        }
    }
}

/// FAST rank-r truncation for one KxK state: warm-started block subspace iteration + Rayleigh-Ritz.
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
        (32, 2) => rankr_fast_static::<32, 2>(a, an, q, uf, vf),
        (32, 3) => rankr_fast_static::<32, 3>(a, an, q, uf, vf),
        (32, 4) => rankr_fast_static::<32, 4>(a, an, q, uf, vf),
        (16, 2) => rankr_fast_static::<16, 2>(a, an, q, uf, vf),
        (8, 2) => rankr_fast_static::<8, 2>(a, an, q, uf, vf),
        _ => false, // any other shape: keep the reference path
    }
}

fn rankr_fast_static<const K: usize, const R: usize>(
    a: &nalgebra::DMatrix<f32>,
    an: &nalgebra::DMatrix<f32>,
    q: &mut [f32],
    uf: &mut nalgebra::DMatrix<f32>,
    vf: &mut nalgebra::DMatrix<f32>,
) -> bool {
    use nalgebra::{SMatrix, SymmetricEigen};
    const B: usize = R + 1;
    if q.len() < B * K {
        return false;
    }
    let a_s = SMatrix::<f32, K, K>::from_fn(|i, j| a[(i, j)]);
    let an_s = SMatrix::<f32, K, K>::from_fn(|i, j| an[(i, j)]);
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
    if warm_ok {
        for j in 0..B {
            let n = block.column(j).norm();
            for i in 0..K {
                block[(i, j)] /= n;
            }
        }
        if !mgs_into::<K, B>(&mut block) {
            warm_ok = false;
        }
    }
    if !warm_ok {
        block = cold_block::<K, B>(&an_s);
    }
    for _ in 0..fast_iters() {
        let t = an_s.transpose() * block; // A^T Q   (KxB)
        let mut z = an_s * t; // A A^T Q (KxB) - the Gram matrix is never formed
        if !mgs_into::<K, B>(&mut z) {
            return false;
        }
        block = z;
    }
    // Rayleigh-Ritz: G = (A^T Q)^T (A^T Q) is the projection of A A^T onto span(block); its top-R
    // eigenvectors give the best R directions inside the tracked block, and the (R+1)-th is discarded.
    let t = an_s.transpose() * block; // KxB
    let g = t.transpose() * t; // BxB PSD
    let eig = SymmetricEigen::new(g);
    let mut order: Vec<usize> = (0..B).collect();
    order.sort_by(|&i, &j| {
        let x = if eig.eigenvalues[i].is_finite() { eig.eigenvalues[i] } else { f32::NEG_INFINITY };
        let y = if eig.eigenvalues[j].is_finite() { eig.eigenvalues[j] } else { f32::NEG_INFINITY };
        y.partial_cmp(&x).unwrap()
    });
    let mut w = SMatrix::<f32, B, R>::zeros();
    for j in 0..R {
        w.set_column(j, &eig.eigenvectors.column(order[j]));
    }
    let mut u = block * w; // KxR
    if !mgs_into::<K, R>(&mut u) {
        return false;
    }
    for j in 0..R {
        let u_col = u.column(j).into_owned();
        let v_un = a.transpose() * &u_col; // = sigma * v  (uses the ORIGINAL A, like the reference)
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
}


'''

# ---------------------------------------------------------------- the new r>=2 branch
BRANCH = r'''        } else {
            // RANK >= 2 (or ALS): reference path = symmetric eigendecomposition of the Gram G = an an^T
            // (KxK PSD; eigenvectors of G are left singular vectors, eigenvalues are sigma_n^2, and
            // v = A^T u / sigma). FAST path (RWKV_LOWRANK_FAST=1) = warm-started block subspace iteration
            // + Rayleigh-Ritz, see `rankr_fast_static`. RWKV_LOWRANK_FAST_CHECK=1 runs both and records
            // the difference but still writes the REFERENCE factors, so a checked run is bit-reproducible.
            let check = lowrank_fast_check();
            let mut done = false;
            if als_iters.is_none() && (lowrank_fast_on() || check) {
                use std::sync::atomic::Ordering;
                let per_head = k * r + k; // the block is R+1 columns
                let maybe = basis.as_deref_mut().filter(|b| b.len() >= (hh + 1) * per_head);
                if let Some(b) = maybe {
                    FAST_CALLS.fetch_add(1, Ordering::Relaxed);
                    let mut uf_f = DMatrix::<f32>::zeros(k, r);
                    let mut vf_f = DMatrix::<f32>::zeros(k, r);
                    done = rankr_fast(&a, &an, k, r, &mut b[hh * per_head..(hh + 1) * per_head],
                                      &mut uf_f, &mut vf_f);
                    if done && check {
                        let mut uf_r = DMatrix::<f32>::zeros(k, r);
                        let mut vf_r = DMatrix::<f32>::zeros(k, r);
                        rankr_reference(&a, &an, k, r, scale, &mut uf_r, &mut vf_r);
                        record_rel_diff(&(&uf_f * vf_f.transpose()), &(&uf_r * vf_r.transpose()));
                        uf = uf_r; // the checked run reproduces the reference exactly
                        vf = vf_r;
                    } else if done {
                        uf = uf_f;
                        vf = vf_f;
                    } else {
                        FAST_FALLBACK.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
            if !done {
                rankr_reference(&a, &an, k, r, scale, &mut uf, &mut vf);
            }
        }
'''
ELSE_TAIL = ""

NLX = NL
text = text.replace(SIG_OLD, SIG_OLD.replace(
    "    mut warm: Option<&mut [i32]>," + NLX,
    "    mut warm: Option<&mut [i32]>," + NLX
    + "    // FAST RANK-r PATH (RWKV_LOWRANK_FAST): this (entity, stream, head)'s subspace block from the" + NLX
    + "    // previous step, layout [head][R+1 columns][K], i.e. len >= h*(r+1)*k. None = no warm basis," + NLX
    + "    // which just means the fast path cold-starts; the reference path ignores it entirely." + NLX
    + "    mut basis: Option<&mut [f32]>," + NLX), 1)

HELPERS = HELPERS.replace("\n", NL)
BRANCH = BRANCH.replace("\n", NL)
text = text.replace(DOC, HELPERS + DOC, 1)

start = text.index(ELSE_START)
end = text.index(ELSE_END, start)
old_block = text[start:end]
# the reference math must survive verbatim: keep the original eigen code inside `if false { }` and
# instead call `rankr_reference` (identical math, moved). Verify that the original lines are all present
# in the new function before dropping them here.
ref_lines = [ln.strip() for ln in old_block.splitlines()
             if ln.strip() and not ln.strip().startswith("//") and ln.strip() != "} else {"]


def _norm(s):
    s = re.sub(r"//.*$", "", s)
    s = s.replace("&*", "").replace("&", "")
    return re.sub(r"\s+", "", s)


_nh = _norm(HELPERS)
missing = [ln for ln in ref_lines if _norm(ln) and _norm(ln) not in _nh]
if missing:
    raise SystemExit(f"the reference code did not survive the move: {missing[:4]}")
text = text[:start] + BRANCH + ELSE_TAIL + text[end:]

P.open("w", encoding="utf-8", newline="").write(text)
print("added the fast rank-r path:")
print(f"  helpers before compress_wkv_state: {len(HELPERS.splitlines())} lines")
print(f"  signature gained `basis`")
print(f"  r>=2 branch replaced ({len(old_block.splitlines())} lines of reference code moved, all matched)")
