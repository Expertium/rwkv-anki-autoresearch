//! RWKV-7 SRS model, RNN (sequential) inference form, ported from
//! rwkv/model/srs_model_rnn.py + rwkv_rnn_model.py. CPU, f32, batch size 1.
//!
//! Weights are loaded by their PyTorch state_dict names from a safetensors file,
//! so the mapping to the Python module tree is unambiguous.

use anyhow::{anyhow, Result};
use candle_core::{DType, Device, Tensor, D};
use std::collections::HashMap;

// Model dims (H heads, K head-dim, C d_model) and per-stream layer counts are DERIVED from
// the weight shapes at load time (see Model::load) so the engine auto-adapts to any arch.
const LN_EPS: f64 = 1e-5;
const GN_EPS: f64 = 64e-5;
const L2_EPS: f64 = 1e-12;

type TMap = HashMap<String, Tensor>;

fn get<'a>(m: &'a TMap, k: &str) -> Result<&'a Tensor> {
    m.get(k).ok_or_else(|| anyhow!("missing weight: {k}"))
}

/// Debug: print sum / L2 norm / first 3 values of a tensor (matches debug_review0.py).
fn summ(name: &str, t: &Tensor) {
    let v: Vec<f32> = t.flatten_all().unwrap().to_vec1().unwrap();
    let sum: f32 = v.iter().sum();
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    eprintln!(
        "{name:16} shape {:?} sum {sum:+.6} norm {norm:.6} head [{:.6}, {:.6}, {:.6}]",
        t.dims(),
        v[0],
        v[1],
        v[2]
    );
}

#[allow(dead_code)] // superseded by Model::lin (pre-transposed weights); kept for reference
fn linear(x: &Tensor, w: &Tensor, b: Option<&Tensor>) -> Result<Tensor> {
    let y = x.matmul(&w.t()?)?;
    match b {
        Some(b) => Ok(y.broadcast_add(b)?),
        None => Ok(y),
    }
}

/// Round-trip a tensor through symmetric per-tensor int-N quantization, simulating int8/int4 STATE
/// storage: quantize for storage, dequantize for the next step's fp32 compute. qmax = 2^(bits-1)-1.
fn quant_roundtrip(t: &Tensor, qmax: f64) -> Result<Tensor> {
    let amax = t.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()? as f64;
    let scale = (amax / qmax).max(1e-12);
    let q = t.affine(1.0 / scale, 0.0)?.round()?.clamp(-qmax, qmax)?;
    Ok(q.affine(scale, 0.0)?)
}

/// Quantize a state tensor to the INTEGER codes (the stored values) at qmax: returns (codes, scale).
/// Stored = round(t/scale) in [-qmax, qmax]; dequant = code*scale. For int2 (qmax=1) codes are {-1,0,1}.
pub fn quant_codes(t: &Tensor, qmax: f64) -> Result<(Vec<f32>, f64)> {
    let amax = t.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()? as f64;
    let scale = (amax / qmax).max(1e-12);
    let codes: Vec<f32> = t.affine(1.0 / scale, 0.0)?.round()?.clamp(-qmax, qmax)?
        .flatten_all()?.to_vec1()?;
    Ok((codes, scale))
}

/// In-place symmetric per-vector int-N quant (mirrors `quant_roundtrip` exactly): scale = amax/qmax
/// (>=1e-12), store round(x/scale) clamped to [-qmax,qmax], dequant = code*scale. Used by the fast
/// engine's shift-vector quantization so it matches the candle path.
pub fn quant_vec_inplace(v: &mut [f32], qmax: f64) {
    let amax = v.iter().fold(0f32, |m, &x| m.max(x.abs())) as f64;
    let scale = (amax / qmax).max(1e-12);
    for x in v.iter_mut() {
        *x = (((*x as f64) / scale).round().clamp(-qmax, qmax) * scale) as f32;
    }
}

/// Per-stream state-compression config, shared by the candle and fast inference paths so both compress
/// identically. Built once from the RWKV_* env vars in `Model::load` and handed to the fast engine.
#[derive(Clone, Default)]
pub struct CompressCfg {
    pub lowrank: std::collections::HashMap<usize, (usize, Option<f64>)>, // stream -> (rank, factor qmax)
    pub quant_qmax: std::collections::HashMap<usize, f64>,               // stream -> full-matrix qmax
    pub quant_shifts: bool,
    /// RWKV_STATE_SHIFT_LEVEL=intN: override the token-shift bit-width INDEPENDENTLY of the WKV factor
    /// level, for streams that are already compressed (quant_shifts on). None = shifts follow the WKV
    /// level (default). Lets card = WKV int4 (256 b) + shifts int2 (128 b) = 384 b — a finer-WKV split
    /// than int3-everything at the same size. Only lowers the shift bits; the WKV factor level is untouched.
    pub shift_qmax_override: Option<f64>,
    pub percol: bool,
    pub hadamard: bool,
    pub four_level: bool,
    pub mixed53: bool,
    pub compand: Option<f64>,
    pub vqmax: Option<f64>,
    pub als: Option<usize>,
    /// Error-feedback / noise-shaping (Idea EF): carry a per-card quant-error buffer, add it back to the
    /// fresh state BEFORE re-compressing each step, then update e = compensated - reconstructed. Cancels the
    /// compounding DC bias of the low-rank factor quantization. POC carries `e` at full precision (over budget).
    pub ef: bool,
    /// EF budget knobs: compress the carried `e` buffer itself (so the DEPLOY-honest, budget-compliant `e`
    /// is what feeds back). `ef_erank` = low-rank truncation of `e` (its bias is low-rank → rank-1 may
    /// suffice); `ef_elevel` = quantize `e`'s factors to this qmax. Both None = full-precision `e` (POC).
    pub ef_erank: Option<usize>,
    pub ef_elevel: Option<f64>,
    /// Product-quantization codebook for the rank-2 factor directions (RWKV_LOWRANK_PQ=<file>). When set,
    /// REPLACES per-factor int-N quant with codebook encoding of the sign-canonicalized unit directions.
    pub pq: Option<std::sync::Arc<PqCodebook>>,
    /// RWKV_EF_PQ: also PQ the carried error buffer's direction (reuse `pq`) so the stabilizer is cheap
    /// (~96 b for a rank-1 `e` vs 256 b at int4) → lets PQ factors + PQ'd `e` fit ≤256 b. Needs ef_erank set.
    pub ef_pq: bool,
    /// RWKV_SHIFT_PQ=<file>: product-quantize the TOKEN-SHIFT vectors instead of int-N. 2 roles
    /// (0 = t_xshift, 1 = c_xshift), each C-dim vector normalized, chunked into m sub-vectors and coded
    /// by the nearest centroid; norm kept as the scale. Replaces the int-N shift quant for compressed
    /// streams when quant_shifts is on. m4b8 on C=32 → 4×8 idx + 8 b norm = 40 b/vector (int4 = 128+).
    /// FAST-PATH ONLY (same precedent as `pq`; candle stays for parity A/B of non-PQ paths).
    pub shift_pq: Option<std::sync::Arc<PqCodebook>>,
    /// RWKV_SHIFT_ROT=<file>: LEARNED per-role orthogonal pre-rotation for the shift PQ (SpinQuant
    /// adapted to product quantization — moves cross-chunk correlation the product codebooks can't
    /// express). File: line1 `C`, then 2 role blocks (0=t, 1=c) of C rows x C floats (row-major R).
    /// Applied rotate -> encode_decode -> unrotate; norms are rotation-invariant. Global, amortized.
    pub shift_rot: Option<std::sync::Arc<Vec<f32>>>,
}

/// Apply the shift pre-rotation in place: y = R x (forward) or x = R^T y (transpose). `rot` is one
/// role's row-major C x C block. Small C (32): a plain O(C^2) loop is fine on the replay path.
pub(crate) fn rot_apply(rot: &[f32], x: &mut [f32], transpose: bool) {
    let c = x.len();
    let mut y = vec![0f32; c];
    for (i, yi) in y.iter_mut().enumerate() {
        let mut s = 0f32;
        for j in 0..c {
            s += if transpose { rot[j * c + i] } else { rot[i * c + j] } * x[j];
        }
        *yi = s;
    }
    x.copy_from_slice(&y);
}

/// Batched state quant: t is (B,H,K,K); compute a PER-CARD (per leading-B) per-tensor amax so each
/// card gets its own scale, matching the B=1 `quant_roundtrip` exactly (which scales over the whole
/// (H,K,K)). A single global amax would couple cards and break parity.
fn quant_roundtrip_batched(t: &Tensor, qmax: f64) -> Result<Tensor> {
    // amax over (H,K,K) for each b -> (B,1,1,1)
    let amax = t.abs()?.max_keepdim(3)?.max_keepdim(2)?.max_keepdim(1)?;
    let scale = amax.affine(1.0 / qmax, 0.0)?.clamp(1e-12, f64::INFINITY)?; // (B,1,1,1)
    let inv = scale.recip()?;
    let q = t.broadcast_mul(&inv)?.round()?.clamp(-qmax, qmax)?;
    Ok(q.broadcast_mul(&scale)?)
}

/// Symmetric per-matrix quant roundtrip of a small factor matrix in place (mirrors quant_codes:
/// scale = amax/qmax, store round(x/scale) in [-qmax,qmax], dequant = code*scale).
fn quant_factor_inplace(m: &mut nalgebra::DMatrix<f32>, qmax: f64) {
    let amax = m.iter().fold(0f32, |a, &x| a.max(x.abs())) as f64;
    let scale = (amax / qmax).max(1e-12);
    for x in m.iter_mut() {
        *x = (((*x as f64) / scale).round().clamp(-qmax, qmax) * scale) as f32;
    }
}

/// Per-COLUMN (per rank-component) symmetric int-N quant: each column gets its OWN scale, so a
/// small-singular-value column is not crushed by the dominant column's scale. This is the "channel
/// scaling equalizes quantization difficulty" fix that makes int2 low-rank viable (RWKV_LOWRANK_PERCOL).
/// Cost = r extra scales/factor (a few bytes) vs the 2*K*r factor codes -- negligible.
fn quant_factor_percol_inplace(m: &mut nalgebra::DMatrix<f32>, qmax: f64) {
    for mut col in m.column_iter_mut() {
        let amax = col.iter().fold(0f32, |a, &x| a.max(x.abs())) as f64;
        let scale = (amax / qmax).max(1e-12);
        for x in col.iter_mut() {
            *x = (((*x as f64) / scale).round().clamp(-qmax, qmax) * scale) as f32;
        }
    }
}

/// Quantize one value to the symmetric 4-LEVEL 2-bit grid {-1.5,-0.5,+0.5,+1.5}*scale (uses all 4
/// codes of a 2-bit field, vs ternary int2's 3). Nearest level = floor(x/scale) clamped to [-2,1] + 0.5.
fn q4_level(x: f32, scale: f64) -> f32 {
    let y = (x as f64) / scale;
    let lvl = y.floor().clamp(-2.0, 1.0) + 0.5; // nearest of {-1.5,-0.5,0.5,1.5}
    (lvl * scale) as f32
}

/// Per-COLUMN symmetric uniform int-N quant where each rank-component column gets its OWN qmax (level
/// budget). Powers the PACKED MIXED-RESOLUTION int2 scheme (RWKV_LOWRANK_MIXED53): the dominant rank
/// component (col 0) is quantized to a FINER 5-level grid {-2,-1,0,1,2}*scale (qmax=2, scale=amax/2)
/// while the second (col 1) keeps coarse ternary {-1,0,1} (qmax=1). BOTH keep an exact zero (symmetric
/// uniform -> unbiased), avoiding the no-zero bias that sank the 4-level scheme. Honest storage: one
/// col-0 entry (5 levels) PACKS with one col-1 entry (3 levels) into 4 bits (5*3=15 < 16 combos), so a
/// rank-2 factor = 64 such pairs = 256 bits -- the int2 budget, but with extra resolution where 82% of
/// the matrix energy lives (col 0). Scales are per-column (data-driven), no global magic constants.
fn quant_factor_mixed_inplace(m: &mut nalgebra::DMatrix<f32>, qmax_per_col: &[f64]) {
    for (j, mut col) in m.column_iter_mut().enumerate() {
        let qmax = qmax_per_col.get(j).copied().unwrap_or(1.0);
        let amax = col.iter().fold(0f32, |a, &x| a.max(x.abs())) as f64;
        let scale = (amax / qmax).max(1e-12);
        for x in col.iter_mut() {
            *x = (((*x as f64) / scale).round().clamp(-qmax, qmax) * scale) as f32;
        }
    }
}

/// Per-COLUMN COMPANDED symmetric int-N quant: before uniform quantization each entry is passed
/// through a signed power-law companding curve c = sign(u)*|u|^p (u = x/amax in [-1,1], p<1), uniform-
/// quantized, then expanded u_hat = sign(c)*|c|^(1/p). With p<1 the reconstruction levels are placed
/// DENSE NEAR ZERO (quadratically for p=0.5) and sparse near the max -- so small factor entries stay
/// small instead of being snapped up to a coarse uniform step. This targets the real driver of state-
/// compression log-loss: keeping near-zero entries near-zero (see research_log -- the 4-level & mixed53
/// schemes died by INFLATING small entries even though they kept an exact zero). Keeps an exact zero
/// (u=0 -> c=0 -> 0). p is a single shared shape constant (NOT per-matrix-fitted); a win must be
/// insensitive to it. amax is per-column (data-driven). No extra storage vs plain int-N.
fn quant_factor_compand_inplace(m: &mut nalgebra::DMatrix<f32>, qmax: f64, p: f64) {
    for mut col in m.column_iter_mut() {
        let amax = col.iter().fold(0f32, |a, &x| a.max(x.abs())) as f64;
        if amax < 1e-20 {
            continue;
        }
        for x in col.iter_mut() {
            let u = (*x as f64) / amax; // [-1, 1]
            let c = u.signum() * u.abs().powf(p); // compress: spread small |u| out
            let qc = (c * qmax).round().clamp(-qmax, qmax) / qmax; // uniform N-level, keeps 0
            let uhat = qc.signum() * qc.abs().powf(1.0 / p); // expand back
            *x = (uhat * amax) as f32;
        }
    }
}

/// 4-LEVEL symmetric 2-bit factor quant (technique #4): uses all 4 codes ({-1.5,-0.5,0.5,1.5}*scale,
/// scale = amax/1.5) instead of ternary int2's {-1,0,1} -- ~33% finer at the SAME 2-bit storage, still
/// symmetric (no zero-point/bias). per_col gives each rank-component its own scale (stacks with #1).
fn quant_factor_4level_inplace(m: &mut nalgebra::DMatrix<f32>, per_col: bool) {
    if per_col {
        for mut col in m.column_iter_mut() {
            let amax = col.iter().fold(0f32, |a, &x| a.max(x.abs())) as f64;
            let scale = (amax / 1.5).max(1e-12);
            for x in col.iter_mut() {
                *x = q4_level(*x, scale);
            }
        }
    } else {
        let amax = m.iter().fold(0f32, |a, &x| a.max(x.abs())) as f64;
        let scale = (amax / 1.5).max(1e-12);
        for x in m.iter_mut() {
            *x = q4_level(*x, scale);
        }
    }
}

/// Normalized (orthogonal AND symmetric) Sylvester-Hadamard matrix, size k x k. Returns None unless k
/// is a power of 2. Because H == H^T and H*H == I, the SAME matrix rotates the factors (before quant)
/// and un-rotates them (after) -- the QuIP#/QuaRot/ButterflyQuant incoherence trick (technique #3):
/// rotating a low-rank factor spreads its energy across all K dims so low-bit (esp. ternary int2)
/// quantization has lower error. Free at deploy (H is fixed/known -> no extra storage), O(K^2) here.
fn hadamard_matrix(k: usize) -> Option<nalgebra::DMatrix<f32>> {
    if k == 0 || (k & (k - 1)) != 0 {
        return None; // Sylvester construction needs a power of 2
    }
    let mut h = nalgebra::DMatrix::<f32>::from_element(1, 1, 1.0);
    let mut n = 1;
    while n < k {
        let mut h2 = nalgebra::DMatrix::<f32>::zeros(2 * n, 2 * n);
        for i in 0..n {
            for j in 0..n {
                let v = h[(i, j)];
                h2[(i, j)] = v;
                h2[(i, j + n)] = v;
                h2[(i + n, j)] = v;
                h2[(i + n, j + n)] = -v;
            }
        }
        h = h2;
        n *= 2;
    }
    Some(h * (1.0 / (k as f32).sqrt()))
}

/// Low-rank roundtrip of a (H,K,K) WKV state: per head, replace the KxK matrix with its rank-r SVD
/// truncation A_r = (U_r sqrt(S_r)) (V_r sqrt(S_r))^T. The deploy model stores the two Kxr factors
/// (2*K*r floats) instead of the full K*K -- the 0.15 KB card path. If `factor_qmax` is Some, the
/// factors are additionally quantized (the real deploy size = 2*K*r codes at that bit-width). Applying
/// this per recurrence step == the deploy per-persist model (a card advances 1 step per review, state
/// persisted between reviews). Uses a fast top-r truncation (Gram + symmetric eigendecomposition),
/// NOT a full SVD -- the full SVD converges pathologically slowly on near-low-rank states.
/// Candle wrapper: extract the (H,K,K) state to a flat Vec, compress IN PLACE via `compress_wkv_state`
/// (the SHARED core, also used by the plain-Rust fast engine so both produce identical states), rebuild.
fn lowrank_roundtrip(
    t: &Tensor,
    rank: usize,
    factor_qmax: Option<f64>,
    per_col: bool,
    hadamard: bool,
    four_level: bool,
    mixed53: bool,
    compand: Option<f64>,
    v_qmax: Option<f64>,
    als_iters: Option<usize>,
) -> Result<Tensor> {
    let (h, k, k2) = t.dims3()?;
    assert_eq!(k, k2, "WKV state must be square KxK");
    let mut data: Vec<f32> = t.flatten_all()?.to_vec1()?;
    compress_wkv_state(
        &mut data, h, k, rank, factor_qmax, per_col, hadamard, four_level, mixed53, compand, v_qmax,
        als_iters, None, // PQ is fast-path only (like EF); candle stays for parity A/B of the non-PQ paths
        None, // no PQ -> no warm indices
    );
    Ok(Tensor::from_vec(data, (h, k, k), t.device())?)
}

/// Product-quantization codebook for rank-2 WKV factor DIRECTIONS. Roles: 0=u1, 1=v1, 2=u2, 3=v2 (rank
/// order = dominant first). GLOBAL + fixed (trained offline on a dev corpus) → amortized across all cards,
/// so its size does NOT count against the per-card budget; only the per-card indices + scales do. Each
/// K-dim unit direction is split into `m` sub-vectors of `sub_dim=K/m`; each sub-vector is coded by the
/// nearest of `ncent=2^bits` centroids (an index of `bits` bits). Bits/direction = m*bits.
pub struct PqCodebook {
    pub m: usize,
    pub sub_dim: usize,
    pub k: usize,
    pub ncent: usize,
    cb: Vec<Vec<f32>>, // [role*m + pos] -> flat ncent*sub_dim centroids (centroid c at [c*sub_dim..])
    /// RWKV_PQ_NORM_BITS: quantize the per-direction norm scalar to n bits, uniform in log2 domain over
    /// a fixed per-codebook range (octaves). Ranges are corpus-derived globals (2026-07-04: WKV √σ spans
    /// log2 [-2.5,-0.6], shift norms [2.4,2.7] — layernorm pins them). None = exact f32 norm (legacy).
    pub norm_bits: Option<u32>,
    pub norm_lo_log2: f32,
    pub norm_hi_log2: f32,
    /// JOINT-UV mode (task23, the m2b12 "index bits ≠ catalog size" principle on the WKV side):
    /// header sub_dim == 2*k with m == 1 → the single catalog holds concat(u_unit, v_unit) 32-dim
    /// entries; ONE index per head selects BOTH factor directions (u/v correlation captured), each
    /// half rescaled by its own (norm-quantized) norm. File = 1 centroid block instead of 4.
    pub joint: bool,
    /// task25 (port of the CUDA warm-start, RWKV_QAT_NO_WARM analog): callers that compress the SAME
    /// entity's state/shift on consecutive reviews can pass the previous winning index as a warm bound —
    /// the vectors drift slowly, so most of the catalog prunes after a few dims. PROVABLY pick-identical
    /// (see the scan comments). `RWKV_NO_WARM=1` disables it (bitwise-A/B / paranoid fallback).
    pub warm_enabled: bool,
}

impl PqCodebook {
    /// Parse the text codebook written by `scratchpad/pq_train.py`:
    /// line1 `m bits sub_dim k ncent`, then 4*m blocks (role-major, then pos) of `ncent` centroid rows.
    pub fn load(path: &str) -> anyhow::Result<Self> {
        Self::load_roles(path, 4)
    }

    /// Same format but with `n_roles` role blocks (WKV factor codebooks = 4 roles u1,v1,u2,v2;
    /// token-shift codebooks = 2 roles t_xshift,c_xshift — written by `scratchpad/pq_train_shift.py`).
    pub fn load_roles(path: &str, n_roles: usize) -> anyhow::Result<Self> {
        let txt = std::fs::read_to_string(path)?;
        let mut lines = txt.lines().filter(|l| !l.trim().is_empty());
        let hdr = lines.next().ok_or_else(|| anyhow!("empty PQ codebook file"))?;
        let h: Vec<usize> = hdr.split_whitespace().map(|x| x.parse()).collect::<Result<_, _>>()?;
        anyhow::ensure!(h.len() >= 5, "bad PQ header");
        let (m, sub_dim, k, ncent) = (h[0], h[2], h[3], h[4]);
        // joint-uv detection: one catalog of concat(u,v) entries (only meaningful for WKV, n_roles=4)
        let joint = sub_dim == 2 * k && m == 1;
        let n_roles = if joint { 1 } else { n_roles };
        let mut cb = Vec::with_capacity(n_roles * m);
        for _ in 0..n_roles * m {
            let mut flat = Vec::with_capacity(ncent * sub_dim);
            for _ in 0..ncent {
                let ln = lines.next().ok_or_else(|| anyhow!("PQ codebook truncated"))?;
                for tok in ln.split_whitespace() {
                    flat.push(tok.parse::<f32>()?);
                }
            }
            anyhow::ensure!(flat.len() == ncent * sub_dim, "PQ centroid block size mismatch");
            cb.push(flat);
        }
        let warm_enabled = std::env::var("RWKV_NO_WARM").map(|v| v != "1").unwrap_or(true);
        Ok(Self { m, sub_dim, k, ncent, cb, norm_bits: None, norm_lo_log2: 0.0, norm_hi_log2: 0.0, joint, warm_enabled })
    }

    /// Enable norm quantization (see field docs). Called by Model::load per codebook with its range.
    pub fn set_norm_quant(&mut self, bits: u32, lo_log2: f32, hi_log2: f32) {
        self.norm_bits = Some(bits);
        self.norm_lo_log2 = lo_log2;
        self.norm_hi_log2 = hi_log2;
    }

    /// The RWKV_PQ_NORM_BITS norm quantizer (see field docs). Identity when norm_bits is None.
    #[inline]
    fn quant_norm(&self, norm: f32) -> f32 {
        match self.norm_bits {
            None => norm,
            Some(0) => {
                // 0 bits: norm = the FIXED range midpoint (nothing stored per card at all)
                ((self.norm_lo_log2 + self.norm_hi_log2) * 0.5).exp2()
            }
            Some(bits) => {
                // store norm at n bits, uniform in log2 over the fixed per-codebook range
                let levels = ((1u32 << bits) - 1) as f32;
                let t = (norm.log2() - self.norm_lo_log2) / (self.norm_hi_log2 - self.norm_lo_log2);
                let q = (t * levels).round().clamp(0.0, levels);
                (self.norm_lo_log2 + q / levels * (self.norm_hi_log2 - self.norm_lo_log2)).exp2()
            }
        }
    }

    /// In place: normalize `col` (K-dim) to unit, replace each sub-vector by its nearest centroid, rescale
    /// by the original norm. `role` selects the codebook set (0=u1,1=v1,2=u2,3=v2).
    #[inline]
    pub(crate) fn encode_decode(&self, role: usize, col: &mut [f32]) {
        self.encode_decode_warm(role, col, None);
    }

    /// `encode_decode` with a warm-start bound (task25, port of the CUDA kernel's warm-started scan).
    /// `warm` (len >= m): per-chunk previous winning index for THIS entity+role, or -1; updated in place.
    /// PROVABLY PICK-IDENTICAL to the plain scan: (a) the serial first-strict-min == argmin by
    /// (distance, then lower index), and the scan below keeps exactly that order via the explicit tie
    /// rule; (b) d is a monotone non-decreasing sum of squares, so a candidate whose partial sum already
    /// fails the update predicate can never win — pruning it is safe; (c) rustc emits strict-IEEE f32
    /// (no reassociation/contraction), so distances are bit-equal to the plain scan's.
    pub(crate) fn encode_decode_warm(&self, role: usize, col: &mut [f32], mut warm: Option<&mut [i32]>) {
        let mut norm = col.iter().map(|x| x * x).sum::<f32>().sqrt();
        if !norm.is_finite() || norm < 1e-20 {
            return;
        }
        let inv = 1.0 / norm; // chunks normalized by the TRUE norm (centroid match unaffected)
        norm = self.quant_norm(norm);
        for p in 0..self.m {
            let s = p * self.sub_dim;
            let cents = &self.cb[role * self.m + p];
            let wc: Option<usize> = match (self.warm_enabled, warm.as_deref()) {
                (true, Some(w)) => (w[p] >= 0 && (w[p] as usize) < self.ncent).then(|| w[p] as usize),
                _ => None,
            };
            let mut best = 0usize; // centroid-0 fallback if every distance is non-finite (as before)
            let mut bestd = f32::INFINITY;
            // warm candidate first (exact bound), then the rest; one loop body = one FP op sequence
            for c in wc.into_iter().chain((0..self.ncent).filter(|&cc| Some(cc) != wc)) {
                let base = c * self.sub_dim;
                let mut d = 0f32;
                let mut alive = true;
                let mut j = 0usize;
                while j < self.sub_dim {
                    let je = (j + 8).min(self.sub_dim);
                    while j < je {
                        let diff = col[s + j] * inv - cents[base + j];
                        d += diff * diff;
                        j += 1;
                    }
                    alive = d < bestd || (d == bestd && c < best); // survival == update predicate
                    if !alive {
                        break;
                    }
                }
                if alive {
                    bestd = d;
                    best = c;
                }
            }
            if let Some(w) = warm.as_deref_mut() {
                w[p] = best as i32;
            }
            let base = best * self.sub_dim;
            for j in 0..self.sub_dim {
                col[s + j] = cents[base + j] * norm;
            }
        }
    }

    /// JOINT-UV encode (self.joint == true): match concat(u/‖u‖, v/‖v‖) (2K-dim) against the single
    /// catalog, reconstruct u from the winner's first half × quant(‖u‖) and v from the second half ×
    /// quant(‖v‖). Matching uses the TRUE norms (like encode_decode); a degenerate norm on EITHER
    /// factor leaves BOTH untouched (the pair is one code — half a code is meaningless).
    #[inline]
    pub(crate) fn encode_decode_joint(&self, u: &mut [f32], v: &mut [f32]) {
        self.encode_decode_joint_warm(u, v, None);
    }

    /// `encode_decode_joint` with a warm-start bound (task25). `warm` = the previous winning index for
    /// THIS (entity, head, rank-component), or -1; updated in place. Pick-identity proof: see
    /// `encode_decode_warm` — same three-part argument, u-half then v-half accumulation order unchanged.
    pub(crate) fn encode_decode_joint_warm(&self, u: &mut [f32], v: &mut [f32], warm: Option<&mut i32>) {
        let k = u.len();
        let nu = u.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nv = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if !nu.is_finite() || nu < 1e-20 || !nv.is_finite() || nv < 1e-20 {
            return;
        }
        let (iu, iv) = (1.0 / nu, 1.0 / nv);
        let (qu, qv) = (self.quant_norm(nu), self.quant_norm(nv));
        let cents = &self.cb[0];
        let wc: Option<usize> = match (self.warm_enabled, warm.as_deref()) {
            (true, Some(&w)) => (w >= 0 && (w as usize) < self.ncent).then_some(w as usize),
            _ => None,
        };
        let mut best = 0usize; // centroid-0 fallback if every distance is non-finite (as before)
        let mut bestd = f32::INFINITY;
        for c in wc.into_iter().chain((0..self.ncent).filter(|&cc| Some(cc) != wc)) {
            let base = c * self.sub_dim;
            let mut d = 0f32;
            let mut alive = true;
            let mut j = 0usize;
            while j < k {
                let je = (j + 8).min(k);
                while j < je {
                    let diff = u[j] * iu - cents[base + j];
                    d += diff * diff;
                    j += 1;
                }
                alive = d < bestd || (d == bestd && c < best);
                if !alive {
                    break;
                }
            }
            let mut j = 0usize;
            while alive && j < k {
                let je = (j + 8).min(k);
                while j < je {
                    let diff = v[j] * iv - cents[base + k + j];
                    d += diff * diff;
                    j += 1;
                }
                alive = d < bestd || (d == bestd && c < best);
            }
            if alive {
                bestd = d;
                best = c;
            }
        }
        if let Some(w) = warm {
            *w = best as i32;
        }
        let base = best * self.sub_dim;
        for j in 0..k {
            u[j] = cents[base + j] * qu;
            v[j] = cents[base + k + j] * qv;
        }
    }
}

/// In-place rank-r SVD-truncation + quantization of a flat row-major (H,K,K) WKV state. The SHARED
/// compression core called by BOTH the candle `lowrank_roundtrip` and the plain-Rust fast engine
/// (`fast.rs`), so the two inference paths produce bit-identical compressed states. Each head's KxK is
/// read into a local matrix BEFORE its result is written back, so in-place mutation is safe.
#[allow(clippy::too_many_arguments)]
pub fn compress_wkv_state(
    data: &mut [f32],
    h: usize,
    k: usize,
    rank: usize,
    factor_qmax: Option<f64>,
    per_col: bool,
    hadamard: bool,
    four_level: bool,
    mixed53: bool,
    compand: Option<f64>,
    v_qmax: Option<f64>,
    als_iters: Option<usize>,
    pq: Option<&PqCodebook>,
    // task25: joint-path warm indices for THIS entity, layout [head*2 + rank_component], -1 = none
    // (len >= h*2 when Some). Only the joint PQ path uses it; everything else ignores it.
    mut warm: Option<&mut [i32]>,
) {
    use nalgebra::DMatrix;
    let r = rank.min(k);
    // QuIP#/QuaRot incoherence rotation (#3): built once, reused per head. None if K not a power of 2.
    let hmat = if hadamard { hadamard_matrix(k) } else { None };
    for hh in 0..h {
        let off = hh * k * k;
        // our layout is row-major (row r, col c) at off + r*k + c
        let a = DMatrix::<f32>::from_row_slice(k, k, &data[off..off + k * k]);
        // Top-r truncated SVD via symmetric eigendecomposition of the Gram matrix G = A A^T (KxK PSD):
        // eigenvectors of G are the left singular vectors of A, eigenvalues are sigma^2; the right
        // singular vector is v = A^T u / sigma. This is FAST + ROBUST on near-low-rank states -- a
        // symmetric eigensolver has none of the Golub-Kahan slow-convergence pathology that nalgebra's
        // full SVD hits on the ~30 clustered near-zero singular values (which made the per-step note
        // low-rank gate hang for tens of minutes). Validated == full-SVD rank-2 recon to ~1e-15.
        // A is NORMALIZED by its max-abs before forming the Gram (the product A A^T squares magnitudes
        // and would overflow f32 for a state that has grown large over a long review history -> NaN
        // eigenvalues); eigenvalues are unscaled afterward (sigma = scale * sqrt(eig)).
        let amax = a.iter().fold(0f32, |m, &x| m.max(x.abs()));
        let scale = if amax.is_finite() && amax > 1e-30 { amax } else { 1.0 };
        let an = &a * (1.0 / scale);
        let mut uf = DMatrix::<f32>::zeros(k, r);
        let mut vf = DMatrix::<f32>::zeros(k, r);
        if r == 1 && als_iters.is_none() {
            // RANK-1 FAST PATH: power-iterate to the top left singular vector of `an` (Gram eigenvectors
            // are scale-invariant, so it's also A's), converged tightly so it matches the full-eig top
            // vector to ~1e-6 -- parity-preserving -- while avoiding the full KxK symmetric eigendecomp
            // AND forming the Gram. (Sign is arbitrary but cancels in the uf*vf^T outer product.)
            let mut u = nalgebra::DVector::<f32>::from_element(k, 1.0 / (k as f32).sqrt());
            for _ in 0..64 {
                let atu = an.transpose() * &u; // (= sigma_n * v_n)
                let nu = &an * &atu; // (= an an^T u)
                let nrm = nu.norm();
                if !nrm.is_finite() || nrm < 1e-30 {
                    break;
                }
                let nu = nu / nrm;
                let dot = u.dot(&nu).abs();
                u = nu;
                if 1.0 - dot < 1e-7 {
                    break; // converged to the dominant eigenvector
                }
            }
            let v_un = &a.transpose() * &u; // = sigma * v  (uses ORIGINAL A -> true sigma)
            let sigma = v_un.norm();
            if sigma > 1e-20 && sigma.is_finite() && u.iter().all(|x| x.is_finite()) {
                let sj = sigma.sqrt();
                for i in 0..k {
                    uf[(i, 0)] = u[i] * sj;
                    vf[(i, 0)] = (v_un[i] / sigma) * sj;
                }
            }
        } else {
            // Top-r (r>1, or ALS) via symmetric eigendecomposition of the Gram G = an an^T (KxK PSD):
            // eigenvectors of G are left singular vectors, eigenvalues are sigma_n^2; v = A^T u / sigma.
            let gram = &an * an.transpose();
            let eig = nalgebra::SymmetricEigen::new(gram);
            let evals = &eig.eigenvalues;
            let mut order: Vec<usize> = (0..k).collect();
            // NaN-safe descending sort: non-finite eigenvalues -> -inf (sort last, never picked as top-r),
            // and a PROPER total order (a bare partial_cmp.unwrap_or(Equal) panics on NaN evals from int2).
            order.sort_by(|&i, &j| {
                let a = if evals[i].is_finite() { evals[i] } else { f32::NEG_INFINITY };
                let b = if evals[j].is_finite() { evals[j] } else { f32::NEG_INFINITY };
                b.partial_cmp(&a).unwrap()
            });
            for j in 0..r {
                let col = order[j];
                let ev = evals[col];
                if !ev.is_finite() || ev <= 0.0 {
                    continue; // skip degenerate/non-finite components (graceful rank reduction)
                }
                let sigma = ev.sqrt() * scale; // unscale -> true singular value
                if sigma > 1e-20 && sigma.is_finite() {
                    let sj = sigma.sqrt(); // split sqrt(sigma) symmetrically into both factors
                    let u_col = eig.eigenvectors.column(col).into_owned();
                    let v_unscaled = &a.transpose() * &u_col; // = sigma * v  (Kx1, uses original A)
                    for i in 0..k {
                        uf[(i, j)] = u_col[i] * sj;
                        vf[(i, j)] = (v_unscaled[i] / sigma) * sj;
                    }
                }
            }
        }
        if let Some(pq) = pq {
            // PRODUCT QUANTIZATION of the factor directions (replaces int-N quant). Per rank component j:
            // sign-canonicalize (flip u's dominant entry positive, flip v with it so u*v^T is invariant --
            // matches pq_train.py), then codebook-encode each K-dim unit direction (norm kept as the scale).
            // Only rank<=2 (codebook roles = u1,v1,u2,v2). uf/vf are column-major: col j at [j*k..(j+1)*k].
            let us = uf.as_mut_slice();
            let vs = vf.as_mut_slice();
            for j in 0..r.min(2) {
                let o = j * k;
                let mut am = 0f32;
                let mut sgn = 1f32;
                for i in 0..k {
                    let a = us[o + i].abs();
                    if a > am {
                        am = a;
                        sgn = if us[o + i] >= 0.0 { 1.0 } else { -1.0 };
                    }
                }
                if sgn < 0.0 {
                    for i in 0..k {
                        us[o + i] = -us[o + i];
                        vs[o + i] = -vs[o + i];
                    }
                }
                if pq.joint {
                    // joint-uv catalog: ONE code per (head, rank-component) selects both directions
                    let wref = warm.as_deref_mut().map(|w| &mut w[hh * 2 + j]);
                    pq.encode_decode_joint_warm(&mut us[o..o + k], &mut vs[o..o + k], wref);
                } else {
                    pq.encode_decode(2 * j, &mut us[o..o + k]);
                    pq.encode_decode(2 * j + 1, &mut vs[o..o + k]);
                }
            }
        } else if let Some(qmax) = factor_qmax {
            let vq = v_qmax.unwrap_or(qmax);
            if let Some(n_als) = als_iters {
                // DIRECT FACTOR OPTIMIZATION (alternating least squares with a per-iter quantize step):
                // refine the QUANTIZED factors to reduce ||A - U V^T||_F. Init from the SVD factors
                // (quantized), then alternate the closed-form LS solve for one factor given the other,
                // re-quantizing each time so quant error in V is COMPENSATED when re-solving U (and vice
                // versa) -- strictly lower Frobenius than post-hoc quant. Uses per-column uniform quant
                // (keeps an exact zero). NOTE: optimizes Frobenius, which is anti-correlated with log-loss
                // here (it down-weights the small entries that matter) -- judged ONLY by run_eval log-loss.
                quant_factor_percol_inplace(&mut uf, qmax);
                quant_factor_percol_inplace(&mut vf, vq);
                for _ in 0..n_als {
                    // U <- A V (V^T V)^-1, then re-quantize
                    let vtv = vf.transpose() * &vf; // r x r
                    if let Some(inv) = vtv.try_inverse() {
                        let unew = (&a * &vf) * inv; // k x r
                        if unew.iter().all(|x| x.is_finite()) {
                            uf = unew;
                            quant_factor_percol_inplace(&mut uf, qmax);
                        }
                    }
                    // V <- A^T U (U^T U)^-1, then re-quantize
                    let utu = uf.transpose() * &uf; // r x r
                    if let Some(inv) = utu.try_inverse() {
                        let vnew = (a.transpose() * &uf) * inv; // k x r
                        if vnew.iter().all(|x| x.is_finite()) {
                            vf = vnew;
                            quant_factor_percol_inplace(&mut vf, vq);
                        }
                    }
                }
            } else {
                // #3 incoherence: rotate the factor K-dim into the spread-energy basis before quant.
                if let Some(hm) = &hmat {
                    uf = hm * &uf;
                    vf = hm * &vf;
                }
                // Quantize ONE factor at its own qmax `q` via the selected method. uf and vf can use
                // DIFFERENT qmax (asymmetric per-factor bit allocation, e.g. U=int4 V=int3 = 224 bits).
                let quant_one = |m: &mut nalgebra::DMatrix<f32>, q: f64| {
                    if let Some(p) = compand {
                        // dense-near-zero companded quant (keeps small entries small); per-column amax.
                        quant_factor_compand_inplace(m, q, p);
                    } else if mixed53 && q <= 1.5 {
                        // PACKED 5x3 mixed resolution (int2 level): col0 -> 5-level, col1 -> ternary.
                        quant_factor_mixed_inplace(m, &[2.0, 1.0]);
                    } else if four_level && q <= 1.5 {
                        quant_factor_4level_inplace(m, per_col); // #4: all 4 codes of the 2-bit field.
                    } else if per_col {
                        quant_factor_percol_inplace(m, q);
                    } else {
                        quant_factor_inplace(m, q);
                    }
                };
                quant_one(&mut uf, qmax);
                quant_one(&mut vf, vq);
                // un-rotate (H is symmetric orthogonal -> H*(H*uf_q) recovers uf with the quant error
                // introduced in the incoherent basis). No-op when hmat is None.
                if let Some(hm) = &hmat {
                    uf = hm * &uf;
                    vf = hm * &vf;
                }
            }
        }
        let a_r = &uf * vf.transpose(); // (k,k)
        for rr in 0..k {
            for cc in 0..k {
                data[off + rr * k + cc] = a_r[(rr, cc)];
            }
        }
    }
}

fn sigmoid(x: &Tensor) -> Result<Tensor> {
    Ok(candle_nn::ops::sigmoid(x)?)
}

fn silu(x: &Tensor) -> Result<Tensor> {
    Ok((x * sigmoid(x)?)?)
}

/// softplus(x) = log(1+exp(x)), stable form: relu(x) + log(1+exp(-|x|)).
fn softplus(x: &Tensor) -> Result<Tensor> {
    let m = x.relu()?;
    let a = (x.abs()?.neg()?.exp()? + 1.0)?.log()?;
    Ok((m + a)?)
}

fn layer_norm(x: &Tensor, w: &Tensor, b: &Tensor, eps: f64) -> Result<Tensor> {
    let mean = x.mean_keepdim(D::Minus1)?;
    let xc = x.broadcast_sub(&mean)?;
    let var = xc.sqr()?.mean_keepdim(D::Minus1)?;
    let xn = xc.broadcast_div(&var.affine(1.0, eps)?.sqrt()?)?;
    Ok(xn.broadcast_mul(w)?.broadcast_add(b)?)
}

/// GroupNorm with `groups` groups over a (1, C) row vector.
fn group_norm(x: &Tensor, w: &Tensor, b: &Tensor, groups: usize, eps: f64) -> Result<Tensor> {
    let c = x.dim(1)?;
    let cs = c / groups;
    let xr = x.reshape((groups, cs))?;
    let mean = xr.mean_keepdim(D::Minus1)?;
    let xc = xr.broadcast_sub(&mean)?;
    let var = xc.sqr()?.mean_keepdim(D::Minus1)?;
    let xn = xc.broadcast_div(&var.affine(1.0, eps)?.sqrt()?)?;
    let xn = xn.reshape((1, c))?;
    Ok(xn.broadcast_mul(w)?.broadcast_add(b)?)
}

/// Batched GroupNorm: x is (B, C), `groups` groups act per-row. Mirrors `group_norm` with a leading B.
fn group_norm_batched(x: &Tensor, w: &Tensor, b: &Tensor, groups: usize, eps: f64) -> Result<Tensor> {
    let bsz = x.dim(0)?;
    let c = x.dim(1)?;
    let cs = c / groups;
    let xr = x.reshape((bsz, groups, cs))?;
    let mean = xr.mean_keepdim(D::Minus1)?;
    let xc = xr.broadcast_sub(&mean)?;
    let var = xc.sqr()?.mean_keepdim(D::Minus1)?;
    let xn = xc.broadcast_div(&var.affine(1.0, eps)?.sqrt()?)?;
    let xn = xn.reshape((bsz, c))?;
    Ok(xn.broadcast_mul(w)?.broadcast_add(b)?)
}

/// L2-normalize each head row of a (H, K) tensor over the K dim (torch eps=1e-12).
/// Batch-agnostic: also works on (B, H, K) since it reduces over the last dim and broadcasts back.
fn l2norm_heads(x: &Tensor) -> Result<Tensor> {
    let n = x.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?; // (H,1)
    let n = n.clamp(L2_EPS, f64::INFINITY)?;
    Ok(x.broadcast_div(&n)?)
}

/// torch.lerp(start, end, weight) = start + weight*(end-start).
fn lerp(start: &Tensor, end: &Tensor, weight: &Tensor) -> Result<Tensor> {
    let diff = end.broadcast_sub(start)?;
    Ok(start.broadcast_add(&weight.broadcast_mul(&diff)?)?)
}

/// Per-layer RNN state: (time_xshift, time_state_HKK, channel_xshift).
#[derive(Clone)]
pub struct LayerState {
    pub t_xshift: Tensor, // (1,C)
    pub t_state: Tensor,  // (H,K,K)
    pub c_xshift: Tensor, // (1,C)
}

pub type StreamState = Vec<LayerState>;

/// Batched per-layer RNN state: a leading B (batch of independent cards) dim on every tensor.
/// Used ONLY by the `*_batched` query path (JSchoreels-style queue scoring). The B=1 `LayerState`
/// path is left untouched so its bit-exact parity is preserved.
#[derive(Clone)]
pub struct BatchedLayerState {
    pub t_xshift: Tensor, // (B,C)
    pub t_state: Tensor,  // (B,H,K,K)
    pub c_xshift: Tensor, // (B,C)
}

pub type BatchedStreamState = Vec<BatchedLayerState>;

/// Stack B per-card `StreamState`s into one `BatchedStreamState` (cat the (1,C) shifts along dim 0,
/// stack the (H,K,K) WKV states into (B,H,K,K)). All inputs must have the same layer count.
pub fn stack_stream_states(states: &[StreamState]) -> Result<BatchedStreamState> {
    let n_layers = states[0].len();
    let mut out = Vec::with_capacity(n_layers);
    for l in 0..n_layers {
        let t_xshift = Tensor::cat(&states.iter().map(|s| s[l].t_xshift.clone()).collect::<Vec<_>>(), 0)?;
        let c_xshift = Tensor::cat(&states.iter().map(|s| s[l].c_xshift.clone()).collect::<Vec<_>>(), 0)?;
        let t_state = Tensor::stack(&states.iter().map(|s| s[l].t_state.clone()).collect::<Vec<_>>(), 0)?;
        out.push(BatchedLayerState { t_xshift, t_state, c_xshift });
    }
    Ok(out)
}

pub struct Model {
    w: TMap,
    lin_wt: TMap, // linear weights pre-transposed to (in,out) + contiguous at load, keyed "<prefix>.weight"
    dev: Device,
    h: usize,              // n_heads (derived from weights)
    k: usize,              // head dim = c / h
    c: usize,              // d_model (derived from weights)
    stream_layers: Vec<usize>, // layers per stream (derived by counting blocks)
    s_space: Tensor,       // (1,128) forgetting-curve time constants
    point_space: Vec<f32>, // (128) interp grid
    // Per-stream STATE quant: module_idx -> qmax (127=int8, 7=int4). Empty = fp32 everywhere.
    // Allows MIXED bits across streams (e.g. card int4 + note int8). See load() for env parsing.
    state_quant_qmax: std::collections::HashMap<usize, f64>,
    // Per-stream LOW-RANK card-state truncation: module_idx -> (rank, optional factor qmax). When set
    // for a module, the per-step WKV state is replaced by its rank-r SVD truncation (and factors
    // optionally quantized) INSTEAD of full-matrix quant -- the 0.15 KB card path (step 4). See load().
    state_lowrank: std::collections::HashMap<usize, (usize, Option<f64>)>,
    // RWKV_QUANT_SHIFTS=1: also quantize the (1-D, non-low-rankable) token-shift vectors of any
    // COMPRESSED stream at its bit-width, so the deploy SIZE accounting is honest (shifts otherwise
    // stay fp32, which alone blows the 0.15 KB card budget). Off by default -> past numbers reproduce.
    quant_shifts: bool,
    // RWKV_STATE_SHIFT_LEVEL=intN: override token-shift bit-width independently of the WKV factor level
    // (only for already-compressed streams). None = shifts follow the WKV level. See CompressCfg doc.
    shift_qmax_override: Option<f64>,
    // RWKV_LOWRANK_PERCOL=1: quantize low-rank factors with a PER-COLUMN (per rank-component) scale
    // instead of one shared scale -> makes int2 low-rank viable (the small-sigma column isn't crushed).
    lowrank_percol: bool,
    // RWKV_LOWRANK_HADAMARD=1 (#3): rotate low-rank factors by a Hadamard (QuIP#/QuaRot incoherence)
    // matrix before quant and un-rotate after -> spreads energy across K so low-bit quantizes cleaner.
    // Free (H fixed, no extra storage); needs K a power of 2 (else silently skipped). Stacks with percol.
    lowrank_hadamard: bool,
    // RWKV_LOWRANK_4LEVEL=1 (#4): use all 4 codes of a 2-bit field for int2 low-rank factors
    // ({-1.5,-0.5,0.5,1.5}*scale) instead of ternary {-1,0,1}. Only affects the int2 level. Free.
    lowrank_4level: bool,
    // RWKV_LOWRANK_MIXED53=1: PACKED 5x3 mixed-resolution int2 -- col0 (dominant component) at 5 levels,
    // col1 at ternary; 5*3=15<16 packs to 4 bits/pair = 256 bits at rank 2. Only affects the int2 level.
    lowrank_mixed53: bool,
    // RWKV_LOWRANK_COMPAND=<p>: companded factor quant with signed power-law shape p (<1 -> levels
    // dense near zero, keeps small entries small). None = off (plain uniform int-N).
    lowrank_compand: Option<f64>,
    // RWKV_LOWRANK_VLEVEL=int3/int4/...: override the SECOND factor's (V) bit-width independently of U
    // (the scope level). Enables asymmetric per-factor budgets, e.g. U=int4 V=int3 = 224 bits at rank 1.
    lowrank_vqmax: Option<f64>,
    // RWKV_LOWRANK_ALS=<n>: run n iterations of alternating-least-squares factor optimization (quantized
    // U,V minimizing ||A - U V^T||_F) instead of post-hoc SVD-then-quantize. None = off. Frobenius is
    // anti-correlated with log-loss here, so this is judged only by run_eval.
    lowrank_als: Option<usize>,
    // Plain-Rust f32 forward (no candle in the hot loop) -- the deployment speed path, transferred from
    // the parent rwkv-anki-autoresearch engine. Parity-gated vs the candle review_batched (<1e-5; see
    // main.rs --verify-fast). Dim-agnostic: derives H/K/C from weight shapes, so it works for the
    // H=2/K=16 champion (two 16x16 per-head WKV matrices) as well as the old H=1/K=32. Wins ~4.8x at
    // B=1 (the sequential single-user recalc path); near candle at large B (gemm ceiling).
    pub fast: crate::fast::FastModel,
}

impl Model {
    pub fn load(path: &str, dev: Device) -> Result<Self> {
        let w = candle_core::safetensors::load(path, &dev)?;
        // Derive dims from the weight shapes so the engine auto-adapts to any arch.
        let c = get(&w, "prehead_norm.weight")?.dim(0)?;
        let h = get(&w, "rwkv_modules.0.blocks.0.time_mixer.k_scale_linear.weight")?.dim(0)?;
        let k = c / h;
        let mut stream_layers = Vec::new();
        for m in 0..5 {
            let mut l = 0;
            while w.contains_key(&format!(
                "rwkv_modules.{m}.blocks.{l}.time_mixer.layer_norm.weight"
            )) {
                l += 1;
            }
            stream_layers.push(l);
        }
        // forgetting_curve s_space: length = num_curves, DERIVED from w_linear out-features
        // (so the engine auto-adapts to SRS-head-width changes, e.g. iter29's 128->64).
        let num_curves = get(&w, "w_linear.weight")?.dim(0)?;
        let num_points = get(&w, "ahead_linear.weight")?.dim(0)?;
        let s_max = 22.0f32;
        let s_spread = 18.5f32;
        let s_scale = (s_max - s_spread).exp();
        let s_space: Vec<f32> = (0..num_curves)
            .map(|i| {
                let l = 18.5f32 * i as f32 / (num_curves as f32 - 1.0);
                0.1 + (l.exp() - 1.0) * s_scale
            })
            .collect();
        // interp point_space: length = num_points, grid identical formula, different consts
        let max_e = 21.0f32;
        let p_spread = 18.5f32;
        let p_scale = (max_e - p_spread).exp();
        let point_space: Vec<f32> = (0..num_points)
            .map(|i| {
                let l = 18.5f32 * i as f32 / (num_points as f32 - 1.0);
                0.5 + (l.exp() - 1.0) * p_scale
            })
            .collect();
        // STATE quantization (weights stay fp32). Two env vars build a per-stream qmax map:
        //   RWKV_STATE_QUANT       = default level (int8=127, int4=7) for streams without an override
        //   RWKV_STATE_QUANT_SCOPE = comma list selecting streams; each entry is "name" (use default
        //                            level) or "name:int4"/"name:int8" (explicit) -> MIXED bits.
        //                            ""/"all" = every stream at the default level (legacy).
        // Rationale: card & note are the EXPENSIVE-at-deploy streams (many cards/notes) AND have SHORT
        // recurrence (few reviews per card/note), so quantizing them is far milder than the long-
        // recurrence user/global streams that sank the all-streams attempt. Quant SHRINKS state w/o
        // cutting capacity (unlike layer-cutting, which costs imm -- see iter38).
        let parse_level = |s: &str| -> Option<f64> {
            match s {
                "int8" => Some(127.0),
                "int6" => Some(31.0),
                "int5" => Some(15.0),
                "int4" => Some(7.0),
                "int3" => Some(3.0), // symmetric 3-bit = 7 levels {-3..3}*scale (rank-1 = 192 bits)
                "int2" => Some(1.0), // symmetric 2-bit = ternary {-scale,0,+scale}; 0.27 KiB/card
                _ => None,
            }
        };
        let name_to_idx = |n: &str| -> usize {
            match n {
                "card" => 0,
                "deck" => 1,
                "note" => 2,
                "preset" => 3,
                "user" => 4,
                other => panic!("unknown stream in RWKV_STATE_QUANT_SCOPE: {other}"),
            }
        };
        let default_qmax = parse_level(std::env::var("RWKV_STATE_QUANT").unwrap_or_default().as_str());
        let scope = std::env::var("RWKV_STATE_QUANT_SCOPE").unwrap_or_default();
        let mut state_quant_qmax: std::collections::HashMap<usize, f64> = std::collections::HashMap::new();
        if scope.is_empty() || scope == "all" {
            if let Some(q) = default_qmax {
                for i in 0..5 {
                    state_quant_qmax.insert(i, q);
                }
            }
        } else {
            for entry in scope.split(',') {
                let entry = entry.trim();
                let (name, qmax) = match entry.split_once(':') {
                    Some((n, lvl)) => (n, parse_level(lvl).expect("bad level in RWKV_STATE_QUANT_SCOPE (use int4/int8)")),
                    None => (
                        entry,
                        default_qmax.expect("RWKV_STATE_QUANT must be set (int4/int8) when a SCOPE entry omits :level"),
                    ),
                };
                state_quant_qmax.insert(name_to_idx(name), qmax);
            }
        }
        // LOW-RANK card-state truncation (step 4, the 0.15 KB path). RWKV_STATE_LOWRANK_SCOPE = comma
        // list of "name:rank" or "name:rank:int4"/"name:rank:int8"/"name:rank:int2" (factors also
        // quantized). Applied per recurrence step INSTEAD of full-matrix state quant for that stream.
        let mut state_lowrank: std::collections::HashMap<usize, (usize, Option<f64>)> =
            std::collections::HashMap::new();
        let lr_scope = std::env::var("RWKV_STATE_LOWRANK_SCOPE").unwrap_or_default();
        if !lr_scope.is_empty() {
            for entry in lr_scope.split(',') {
                let parts: Vec<&str> = entry.trim().split(':').collect();
                let name = parts[0];
                let rank: usize = parts
                    .get(1)
                    .expect("RWKV_STATE_LOWRANK_SCOPE entry needs name:rank")
                    .parse()
                    .expect("bad rank in RWKV_STATE_LOWRANK_SCOPE");
                let fqmax = parts
                    .get(2)
                    .map(|lvl| parse_level(lvl).expect("bad factor level in LOWRANK scope (int2/int4/int8)"));
                state_lowrank.insert(name_to_idx(name), (rank, fqmax));
            }
        }
        let quant_shifts = std::env::var("RWKV_QUANT_SHIFTS")
            .map(|v| v == "1" || v == "true")
            .unwrap_or(false);
        // RWKV_STATE_SHIFT_LEVEL=intN: quantize token-shifts at intN instead of the WKV factor level.
        let shift_qmax_override = std::env::var("RWKV_STATE_SHIFT_LEVEL")
            .ok()
            .and_then(|v| parse_level(v.as_str()));
        let lowrank_percol = std::env::var("RWKV_LOWRANK_PERCOL")
            .map(|v| v == "1" || v == "true")
            .unwrap_or(false);
        let lowrank_hadamard = std::env::var("RWKV_LOWRANK_HADAMARD")
            .map(|v| v == "1" || v == "true")
            .unwrap_or(false);
        let lowrank_4level = std::env::var("RWKV_LOWRANK_4LEVEL")
            .map(|v| v == "1" || v == "true")
            .unwrap_or(false);
        let lowrank_mixed53 = std::env::var("RWKV_LOWRANK_MIXED53")
            .map(|v| v == "1" || v == "true")
            .unwrap_or(false);
        let lowrank_compand = std::env::var("RWKV_LOWRANK_COMPAND")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|p| *p > 0.0);
        let lowrank_vqmax = std::env::var("RWKV_LOWRANK_VLEVEL")
            .ok()
            .and_then(|v| parse_level(v.as_str()));
        let lowrank_als = std::env::var("RWKV_LOWRANK_ALS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|n| *n > 0);
        // Idea EF (error-feedback / noise shaping): carry a per-card quant-error buffer between steps and
        // add it back before re-compressing, to cancel the compounding DC bias of factor quantization.
        // POC-only in the FAST path; `e` is full-precision (over budget) to measure the idea's ceiling.
        let lowrank_ef = std::env::var("RWKV_LOWRANK_EF")
            .map(|v| v == "1" || v == "true")
            .unwrap_or(false);
        // EF budget knobs: shrink the carried `e` to a deploy-honest size (low-rank + quantized).
        let ef_erank = std::env::var("RWKV_EF_ERANK")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|n| *n > 0);
        let ef_elevel = std::env::var("RWKV_EF_ELEVEL")
            .ok()
            .and_then(|v| parse_level(v.as_str()));
        // RWKV_PQ_NORM_BITS=<n>: quantize the per-direction norm scalars (WKV √σ + shift norms) to n bits,
        // log2-uniform over fixed corpus-derived ranges (WKV [-3,0] octaves, shifts [2.2,2.9]). Cuts the
        // per-card norm cost from 8 b/scalar to n b/scalar; PTQ-applicable (norms carry magnitude only).
        // n=0 = FIXED midpoint norm (ZERO stored bits/scalar); int5==int4==int3==int2 proved free (task22).
        let norm_bits: Option<u32> = std::env::var("RWKV_PQ_NORM_BITS")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .filter(|n| *n <= 8);
        // Product quantization of the rank-2 factor directions: load the offline-trained codebook file.
        let pq = std::env::var("RWKV_LOWRANK_PQ").ok().filter(|s| !s.is_empty()).map(|path| {
            let mut cb = PqCodebook::load(&path)
                .unwrap_or_else(|e| panic!("failed to load RWKV_LOWRANK_PQ '{path}': {e}"));
            if let Some(b) = norm_bits {
                cb.set_norm_quant(b, -3.0, 0.0);
            }
            std::sync::Arc::new(cb)
        });
        let ef_pq = std::env::var("RWKV_EF_PQ").map(|v| v == "1" || v == "true").unwrap_or(false);
        // Product quantization of the token-shift vectors (2 roles: t_xshift, c_xshift). Fast-path only.
        let shift_pq = std::env::var("RWKV_SHIFT_PQ").ok().filter(|s| !s.is_empty()).map(|path| {
            let mut cb = PqCodebook::load_roles(&path, 2)
                .unwrap_or_else(|e| panic!("failed to load RWKV_SHIFT_PQ '{path}': {e}"));
            if let Some(b) = norm_bits {
                cb.set_norm_quant(b, 2.2, 2.9);
            }
            std::sync::Arc::new(cb)
        });
        // RWKV_SHIFT_ROT=<file>: learned per-role orthogonal pre-rotation for the shift PQ.
        // File: line1 `C`, then 2 role blocks of C rows x C floats. See CompressCfg docs.
        let shift_rot = std::env::var("RWKV_SHIFT_ROT").ok().filter(|s| !s.is_empty()).map(|path| {
            let txt = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("failed to read RWKV_SHIFT_ROT '{path}': {e}"));
            let mut it = txt.split_whitespace();
            let rc: usize = it.next().and_then(|t| t.parse().ok())
                .unwrap_or_else(|| panic!("RWKV_SHIFT_ROT '{path}': bad header"));
            assert_eq!(rc, c, "RWKV_SHIFT_ROT dim {rc} != model C {c}");
            let vals: Vec<f32> = it.map(|t| t.parse::<f32>()
                .unwrap_or_else(|e| panic!("RWKV_SHIFT_ROT '{path}': bad float: {e}"))).collect();
            assert_eq!(vals.len(), 2 * c * c, "RWKV_SHIFT_ROT '{path}': want {} floats", 2 * c * c);
            std::sync::Arc::new(vals)
        });
        // Pre-transpose every 2D linear weight (out,in) -> (in,out) contiguous ONCE, so the
        // per-token matmul needs no .t() / re-contiguous. Norm weights are 1D and skipped.
        let mut lin_wt: TMap = HashMap::new();
        for (key, t) in w.iter() {
            if key.ends_with(".weight") && t.dims().len() == 2 {
                lin_wt.insert(key.clone(), t.t()?.contiguous()?);
            }
        }
        let s_space_t = Tensor::from_vec(s_space.clone(), (1, num_curves), &dev)?;
        // Build the plain-Rust fast forward from the same f32 weight maps + derived dims.
        let compress_cfg = CompressCfg {
            lowrank: state_lowrank.clone(),
            quant_qmax: state_quant_qmax.clone(),
            quant_shifts,
            shift_qmax_override,
            percol: lowrank_percol,
            hadamard: lowrank_hadamard,
            four_level: lowrank_4level,
            mixed53: lowrank_mixed53,
            compand: lowrank_compand,
            vqmax: lowrank_vqmax,
            als: lowrank_als,
            ef: lowrank_ef,
            ef_erank,
            ef_elevel,
            pq,
            ef_pq,
            shift_pq,
            shift_rot,
        };
        let fast = crate::fast::FastModel::build(
            &w, &lin_wt, c, h, k, stream_layers.clone(), num_curves, num_points,
            s_space.clone(), point_space.clone(), compress_cfg,
        )?;
        Ok(Self {
            w,
            lin_wt,
            dev,
            h,
            k,
            c,
            stream_layers,
            s_space: s_space_t,
            point_space,
            state_quant_qmax,
            state_lowrank,
            quant_shifts,
            shift_qmax_override,
            lowrank_percol,
            lowrank_hadamard,
            lowrank_4level,
            lowrank_mixed53,
            lowrank_compand,
            lowrank_vqmax,
            lowrank_als,
            fast,
        })
    }

    /// (H heads, K head-dim, C d_model) derived from the weights.
    pub fn dims(&self) -> (usize, usize, usize) {
        (self.h, self.k, self.c)
    }

    /// Layers per stream [card, deck, note, preset, user].
    pub fn stream_layers(&self) -> &[usize] {
        &self.stream_layers
    }

    fn ln(&self, x: &Tensor, prefix: &str, eps: f64) -> Result<Tensor> {
        layer_norm(
            x,
            get(&self.w, &format!("{prefix}.weight"))?,
            get(&self.w, &format!("{prefix}.bias"))?,
            eps,
        )
    }

    fn lin(&self, x: &Tensor, prefix: &str, bias: bool) -> Result<Tensor> {
        // weight already (in,out) + contiguous (pre-transposed at load) -> direct matmul, no .t().
        let wt = get(&self.lin_wt, &format!("{prefix}.weight"))?;
        let y = x.matmul(wt)?;
        match bias {
            true => Ok(y.broadcast_add(get(&self.w, &format!("{prefix}.bias"))?)?),
            false => Ok(y),
        }
    }

    /// features2card: Linear(92->512)->SiLU->LayerNorm(512)->Linear(512->128)->SiLU
    fn features2card(&self, feats: &Tensor) -> Result<Tensor> {
        let x = silu(&self.lin(feats, "features2card.0", true)?)?;
        let x = self.ln(&x, "features2card.2", LN_EPS)?;
        let x = silu(&self.lin(&x, "features2card.3", true)?)?;
        Ok(x)
    }

    /// One RWKV7 time-mixer layer (RNN form). Returns (out, v0_out, new_t_xshift, new_t_state).
    #[allow(clippy::too_many_arguments)]
    fn time_mixer(
        &self,
        p: &str,
        layer_id: usize,
        in_x: &Tensor,
        v0: Option<&Tensor>,
        st: Option<(&Tensor, &Tensor)>,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        #[allow(non_snake_case)]
        let (H, K, C) = (self.h, self.k, self.c); // dims derived from weights
        let x = self.ln(in_x, &format!("{p}.layer_norm"), LN_EPS)?;
        let (xshift, s_prev) = match st {
            Some((xs, s)) => (xs.clone(), s.clone()),
            None => (
                x.clone(),
                Tensor::zeros((H, K, K), DType::F32, &self.dev)?,
            ),
        };
        let diff = xshift.broadcast_sub(&x)?; // (end - start) component reused

        // 8-way lerp -> r,k,v,d,a,g,k_scale,v_scale. Fused: compute all 8 inputs in one
        // broadcast (8,C) = x + rkvdag_lerp*diff, then slice rows (far fewer candle ops/layer).
        let lerp_w = get(&self.w, &format!("{p}.rkvdag_lerp"))?.reshape((8, C))?; // (8,C)
        let all_inp = x.broadcast_add(&lerp_w.broadcast_mul(&diff)?)?; // (8,C)
        let inp = |i: usize| -> Result<Tensor> { Ok(all_inp.narrow(0, i, 1)?) };
        let inp_r = inp(0)?;
        let inp_k = inp(1)?;
        let inp_v = inp(2)?;
        let inp_d = inp(3)?;
        let inp_a = inp(4)?;
        let inp_g = inp(5)?;
        let inp_ks = inp(6)?;
        let inp_vs = inp(7)?;

        let r = self.lin(&inp_r, &format!("{p}.W_r"), false)?;
        let k = self.lin(&inp_k, &format!("{p}.W_k"), false)?;
        let k_scale = sigmoid(&self.lin(&inp_ks, &format!("{p}.k_scale_linear"), true)?)?; // (1,H)
        let v_scale = sigmoid(&self.lin(&inp_vs, &format!("{p}.v_scale_linear"), true)?)?; // (1,H)

        // v + v0 mixing (layer 0 sets v0)
        let (v, v0_out) = if layer_id == 0 {
            let v = self.lin(&inp_v, &format!("{p}.W_v"), false)?;
            (v.clone(), v)
        } else {
            let v_lerp = sigmoid(&self.lora_simple(&inp_v, &format!("{p}.v_lora_simple"))?)?;
            let wv = self.lin(&inp_v, &format!("{p}.W_v"), false)?;
            let v0 = v0.ok_or_else(|| anyhow!("v0 missing for layer>0"))?;
            (lerp(&wv, v0, &v_lerp)?, v0.clone())
        };

        let a = sigmoid(&self.lora_simple(&inp_a, &format!("{p}.a_lora_simple"))?)?;
        let g = self.lin(
            &sigmoid(&self.lin(&inp_g, &format!("{p}.lora_A_g"), false)?)?,
            &format!("{p}.lora_B_g"),
            false,
        )?;

        // decay: _d = -0.5 - softplus(-d_lora_mlp(d)); w = exp(-exp(_d))
        let d_mlp = self.lora_mlp(&inp_d, &format!("{p}.d_lora_mlp"))?;
        let _d = softplus(&d_mlp.neg()?)?.neg()?.affine(1.0, -0.5)?;
        let w_decay = _d.exp()?.neg()?.exp()?;

        // reshape to heads
        let to_hk = |t: &Tensor| -> Result<Tensor> { Ok(t.reshape((H, K))?) };
        let k_h0 = l2norm_heads(&to_hk(&k)?)?;
        let k_h0 = k_h0.broadcast_mul(&k_scale.reshape((H, 1))?)?; // (H,K)
        let r_h = to_hk(&r)?;
        let v_h = l2norm_heads(&to_hk(&v)?)?;
        let v_h = v_h.broadcast_mul(&v_scale.reshape((H, 1))?)?;
        let w_h = to_hk(&w_decay)?;
        let a_h = to_hk(&a)?;
        let kd_h = k_h0.clone(); // k_deformed = k before *a
        let k_h = (&k_h0 * &a_h)?;

        // WKV single_timestep
        let (out_hk, next_s) = single_timestep(H, K, &r_h, &k_h, &v_h, &w_h, &a_h, &kd_h, &s_prev)?;

        let out_flat = out_hk.reshape((1, C))?;
        let out_gn = group_norm(
            &out_flat,
            get(&self.w, &format!("{p}.out_group_norm.weight"))?,
            get(&self.w, &format!("{p}.out_group_norm.bias"))?,
            H,
            GN_EPS,
        )?;

        // r_k bonus: (r*bonus*k).sum(-1,keepdim) * v
        let bonus_p = get(&self.w, &format!("{p}.bonus"))?.reshape((H, K))?;
        let term = (&r_h * &bonus_p)?;
        let term = (&term * &k_h)?.sum_keepdim(D::Minus1)?; // (H,1)
        let bonus = term.broadcast_mul(&v_h)?; // (H,K)
        let bonus_flat = bonus.reshape((1, C))?;

        let out2 = self.lin(
            &(&g * &(out_gn + bonus_flat)?)?,
            &format!("{p}.W_o"),
            false,
        )?;
        let out = (in_x + out2)?;
        Ok((out, v0_out, x, next_s))
    }

    fn lora_simple(&self, x: &Tensor, p: &str) -> Result<Tensor> {
        let a = self.lin(x, &format!("{p}.A"), false)?;
        self.lin(&a, &format!("{p}.B_and_lamb"), true)
    }

    fn lora_mlp(&self, x: &Tensor, p: &str) -> Result<Tensor> {
        let a = self.lin(x, &format!("{p}.A"), false)?.tanh()?;
        self.lin(&a, &format!("{p}.B_and_lamb"), true)
    }

    /// One RWKV7 channel-mixer layer. Returns (out, new_c_xshift).
    fn channel_mixer(&self, p: &str, in_x: &Tensor, xshift: Option<&Tensor>) -> Result<(Tensor, Tensor)> {
        #[allow(non_snake_case)]
        let C = self.c;
        let x = self.ln(in_x, &format!("{p}.layer_norm"), LN_EPS)?;
        let xs = match xshift {
            Some(t) => t.clone(),
            None => x.clone(),
        };
        let lerp_k = get(&self.w, &format!("{p}.lerp_k"))?.reshape((1, C))?;
        let mixed = lerp(&x, &xs, &lerp_k)?;
        let k = self.lin(&mixed, &format!("{p}.W_k"), false)?;
        let k = k.relu()?.sqr()?;
        let o = self.lin(&k, &format!("{p}.W_v"), false)?;
        let out = (in_x + o)?;
        Ok((out, x))
    }

    /// Run one RWKV stream (n layers) over a single token. Returns (out, new_state).
    fn run_stream(
        &self,
        module_idx: usize,
        n_layers: usize,
        input: &Tensor,
        state: Option<&StreamState>,
    ) -> Result<(Tensor, StreamState)> {
        let mut x = input.clone();
        let mut v0: Option<Tensor> = None;
        let mut new_state: StreamState = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            let tp = format!("rwkv_modules.{module_idx}.blocks.{l}.time_mixer");
            let cp = format!("rwkv_modules.{module_idx}.blocks.{l}.channel_mixer");
            let ls = state.map(|s| &s[l]);
            let t_st = ls.map(|s| (&s.t_xshift, &s.t_state));
            let (xt, v0_out, t_xshift, t_state) =
                self.time_mixer(&tp, l, &x, v0.as_ref(), t_st)?;
            // Simulate per-card STATE storage by round-tripping the recurrent WKV matrix each step
            // (worst-case accumulation == the deploy per-persist model). t_xshift/c_xshift are tiny ->
            // left fp32. Low-rank (the 0.15 KB path) takes precedence over full-matrix quant per stream.
            let t_state = if let Some(&(rank, fqmax)) = self.state_lowrank.get(&module_idx) {
                lowrank_roundtrip(
                    &t_state,
                    rank,
                    fqmax,
                    self.lowrank_percol,
                    self.lowrank_hadamard,
                    self.lowrank_4level,
                    self.lowrank_mixed53,
                    self.lowrank_compand,
                    self.lowrank_vqmax,
                    self.lowrank_als,
                )?
            } else if let Some(&qmax) = self.state_quant_qmax.get(&module_idx) {
                quant_roundtrip(&t_state, qmax)?
            } else {
                t_state
            };
            // Optionally quantize the 1-D shift vectors (part of the persisted state) at this stream's
            // bit-width so the deploy size is honest. Off unless RWKV_QUANT_SHIFTS=1.
            let shift_qmax: Option<f64> = if self.quant_shifts {
                let base = self
                    .state_lowrank
                    .get(&module_idx)
                    .and_then(|&(_, fq)| fq)
                    .or_else(|| self.state_quant_qmax.get(&module_idx).copied());
                // Override the LEVEL for already-compressed streams; leave uncompressed streams unquantized.
                base.map(|b| self.shift_qmax_override.unwrap_or(b))
            } else {
                None
            };
            let t_xshift = match shift_qmax {
                Some(q) => quant_roundtrip(&t_xshift, q)?,
                None => t_xshift,
            };
            v0 = Some(v0_out);
            let c_st = ls.map(|s| &s.c_xshift);
            let (xc, c_xshift) = self.channel_mixer(&cp, &xt, c_st)?;
            let c_xshift = match shift_qmax {
                Some(q) => quant_roundtrip(&c_xshift, q)?,
                None => c_xshift,
            };
            x = xc;
            new_state.push(LayerState {
                t_xshift,
                t_state,
                c_xshift,
            });
        }
        Ok((x, new_state))
    }

    /// Full forward over all 5 chained streams + heads.
    /// states: [card, deck, note, preset, user] in chain order.
    /// Returns (out_ahead_logits(1,128), out_w(1,128), out_p_logits(1,4), new_states).
    pub fn review(
        &self,
        feats: &Tensor,
        states: &[Option<StreamState>; 5],
    ) -> Result<(Tensor, Tensor, Tensor, [StreamState; 5])> {
        let dbg = std::env::var("RWKV_DEBUG").is_ok();
        let mut x = self.features2card(feats)?;
        if dbg {
            summ("features2card", &x);
        }
        // chain streams
        let mut new: Vec<StreamState> = Vec::with_capacity(5);
        for m in 0..5 {
            let (xo, ns) = self.run_stream(m, self.stream_layers[m], &x, states[m].as_ref())?;
            x = xo;
            new.push(ns);
            if dbg {
                summ(&format!("stream{m}"), &x);
            }
        }
        let global_encoding = x;

        let xh = self.ln(&global_encoding, "prehead_norm", LN_EPS)?;
        if dbg {
            summ("prehead_norm", &xh);
        }

        // head_w -> w_linear -> softmax  (128 curve weights)
        let hw = self.lin(&xh, "head_w.0", true)?.relu()?;
        let hw = self.ln(&hw, "head_w.2", LN_EPS)?;
        let hw = self.lin(&hw, "head_w.4", true)?;
        let out_w_logits = self.lin(&hw, "w_linear", true)?;
        let out_w = candle_nn::ops::softmax(&out_w_logits, D::Minus1)?;

        // head_ahead_logits -> ahead_linear  (128 points)
        let ha = self.lin(&xh, "head_ahead_logits.0", true)?.relu()?;
        let out_ahead_logits = self.lin(&ha, "ahead_linear", true)?;

        // head_p -> p_linear  (4-way)
        let hp = self.lin(&xh, "head_p.0", true)?.relu()?;
        let out_p_logits = self.lin(&hp, "p_linear", true)?;

        if dbg {
            summ("out_p_logits", &out_p_logits);
            summ("out_w", &out_w);
            summ("out_ahead_logits", &out_ahead_logits);
        }
        let new_arr: [StreamState; 5] = new
            .try_into()
            .map_err(|_| anyhow!("stream count mismatch"))?;
        Ok((out_ahead_logits, out_w, out_p_logits, new_arr))
    }

    /// imm probability = 1 - softmax(out_p_logits)[again]
    pub fn imm_prob(&self, out_p_logits: &Tensor) -> Result<f32> {
        let p = candle_nn::ops::softmax(out_p_logits, D::Minus1)?; // (1,4)
        let again: f32 = p.narrow(1, 0, 1)?.reshape(())?.to_scalar()?;
        Ok(1.0 - again)
    }

    /// forgetting_curve(out_w, elapsed_seconds) -> probability scalar.
    fn forgetting_curve(&self, out_w: &Tensor, elapsed: f32) -> Result<f32> {
        let e = elapsed.max(1.0);
        // 0.9^(e/s) = exp(ln(0.9) * e / s)
        let ln09 = 0.9f64.ln() as f32;
        let inv_s = self.s_space.recip()?;
        let pw = inv_s.affine((ln09 * e) as f64, 0.0)?.exp()?; // exp(ln09*e/s)
        let summed = (out_w * pw)?.sum_keepdim(D::Minus1)?; // (1,1)
        let s: f32 = summed.reshape(())?.to_scalar()?;
        Ok(1e-5 + (1.0 - 2e-5) * s)
    }

    /// interp(out_ahead_logits, elapsed) -> logit residual scalar.
    fn interp(&self, out_ahead_logits: &Tensor, elapsed: f32) -> Result<f32> {
        let e = elapsed.max(1.0);
        let ps = &self.point_space;
        // bisect_left (torch.searchsorted default, right=False)
        let mut right = ps.partition_point(|&v| v < e);
        if right < 1 {
            right = 1;
        }
        if right > ps.len() - 1 {
            right = ps.len() - 1;
        }
        let left = right - 1;
        let xl = ps[left];
        let xr = ps[right];
        let logits: Vec<f32> = out_ahead_logits.reshape((ps.len(),))?.to_vec1()?;
        let yl = logits[left];
        let yr = logits[right];
        let val = yl + (yr - yl) * (e - xl) / (xr - xl);
        Ok(1e-5 + (1.0 - 2e-5) * val)
    }

    /// Combined ahead prediction from a stored curve at a given elapsed_seconds.
    pub fn predict_ahead(
        &self,
        out_ahead_logits: &Tensor,
        out_w: &Tensor,
        elapsed: f32,
    ) -> Result<f32> {
        let p_raw = self.forgetting_curve(out_w, elapsed)?;
        let logit_raw = (p_raw / (1.0 - p_raw)).ln();
        let residual = self.interp(out_ahead_logits, elapsed)?;
        let logit = logit_raw + residual;
        Ok(1.0 / (1.0 + (-logit).exp()))
    }

    // ---------------------------------------------------------------------------------------------
    // Batched query path (B independent cards, one forward step each). Mirrors the B=1 methods with
    // a leading B dim. Used for JSchoreels-style queue scoring (read-only). B=1 path is untouched.
    // ---------------------------------------------------------------------------------------------

    /// Batched time-mixer. in_x is (B,C); state shifts (B,C), state matrix (B,H,K,K).
    #[allow(clippy::too_many_arguments)]
    fn time_mixer_batched(
        &self,
        p: &str,
        layer_id: usize,
        in_x: &Tensor,
        v0: Option<&Tensor>,
        st: Option<(&Tensor, &Tensor)>,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        #[allow(non_snake_case)]
        let (H, K, C) = (self.h, self.k, self.c);
        let bsz = in_x.dim(0)?;
        let x = self.ln(in_x, &format!("{p}.layer_norm"), LN_EPS)?;
        let (xshift, s_prev) = match st {
            Some((xs, s)) => (xs.clone(), s.clone()),
            None => (x.clone(), Tensor::zeros((bsz, H, K, K), DType::F32, &self.dev)?),
        };
        let diff = xshift.broadcast_sub(&x)?; // (B,C)

        // 8-way lerp -> r,k,v,d,a,g,k_scale,v_scale. Per-row form: inp_i = x + lerp_w[i]*diff,
        // identical math to the B=1 fused (8,C) version (a (1,C) lerp row broadcasts over (B,C)).
        let lerp_w = get(&self.w, &format!("{p}.rkvdag_lerp"))?.reshape((8, C))?; // (8,C)
        let inp = |i: usize| -> Result<Tensor> {
            let row = lerp_w.narrow(0, i, 1)?; // (1,C)
            Ok(x.broadcast_add(&row.broadcast_mul(&diff)?)?) // (B,C)
        };
        let inp_r = inp(0)?;
        let inp_k = inp(1)?;
        let inp_v = inp(2)?;
        let inp_d = inp(3)?;
        let inp_a = inp(4)?;
        let inp_g = inp(5)?;
        let inp_ks = inp(6)?;
        let inp_vs = inp(7)?;

        let r = self.lin(&inp_r, &format!("{p}.W_r"), false)?;
        let k = self.lin(&inp_k, &format!("{p}.W_k"), false)?;
        let k_scale = sigmoid(&self.lin(&inp_ks, &format!("{p}.k_scale_linear"), true)?)?; // (B,H)
        let v_scale = sigmoid(&self.lin(&inp_vs, &format!("{p}.v_scale_linear"), true)?)?; // (B,H)

        let (v, v0_out) = if layer_id == 0 {
            let v = self.lin(&inp_v, &format!("{p}.W_v"), false)?;
            (v.clone(), v)
        } else {
            let v_lerp = sigmoid(&self.lora_simple(&inp_v, &format!("{p}.v_lora_simple"))?)?;
            let wv = self.lin(&inp_v, &format!("{p}.W_v"), false)?;
            let v0 = v0.ok_or_else(|| anyhow!("v0 missing for layer>0"))?;
            (lerp(&wv, v0, &v_lerp)?, v0.clone())
        };

        let a = sigmoid(&self.lora_simple(&inp_a, &format!("{p}.a_lora_simple"))?)?;
        let g = self.lin(
            &sigmoid(&self.lin(&inp_g, &format!("{p}.lora_A_g"), false)?)?,
            &format!("{p}.lora_B_g"),
            false,
        )?;

        let d_mlp = self.lora_mlp(&inp_d, &format!("{p}.d_lora_mlp"))?;
        let _d = softplus(&d_mlp.neg()?)?.neg()?.affine(1.0, -0.5)?;
        let w_decay = _d.exp()?.neg()?.exp()?;

        let to_hk = |t: &Tensor| -> Result<Tensor> { Ok(t.reshape((bsz, H, K))?) };
        let k_h0 = l2norm_heads(&to_hk(&k)?)?;
        let k_h0 = k_h0.broadcast_mul(&k_scale.reshape((bsz, H, 1))?)?; // (B,H,K)
        let r_h = to_hk(&r)?;
        let v_h = l2norm_heads(&to_hk(&v)?)?;
        let v_h = v_h.broadcast_mul(&v_scale.reshape((bsz, H, 1))?)?;
        let w_h = to_hk(&w_decay)?;
        let a_h = to_hk(&a)?;
        let kd_h = k_h0.clone();
        let k_h = (&k_h0 * &a_h)?;

        let (out_hk, next_s) =
            single_timestep_batched(bsz, H, K, &r_h, &k_h, &v_h, &w_h, &a_h, &kd_h, &s_prev)?;

        let out_flat = out_hk.reshape((bsz, C))?;
        let out_gn = group_norm_batched(
            &out_flat,
            get(&self.w, &format!("{p}.out_group_norm.weight"))?,
            get(&self.w, &format!("{p}.out_group_norm.bias"))?,
            H,
            GN_EPS,
        )?;

        let bonus_p = get(&self.w, &format!("{p}.bonus"))?.reshape((H, K))?;
        let term = r_h.broadcast_mul(&bonus_p)?;
        let term = (&term * &k_h)?.sum_keepdim(D::Minus1)?; // (B,H,1)
        let bonus = term.broadcast_mul(&v_h)?; // (B,H,K)
        let bonus_flat = bonus.reshape((bsz, C))?;

        let out2 = self.lin(&(&g * &(out_gn + bonus_flat)?)?, &format!("{p}.W_o"), false)?;
        let out = (in_x + out2)?;
        Ok((out, v0_out, x, next_s))
    }

    /// Batched channel-mixer. in_x is (B,C).
    fn channel_mixer_batched(
        &self,
        p: &str,
        in_x: &Tensor,
        xshift: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        #[allow(non_snake_case)]
        let C = self.c;
        let x = self.ln(in_x, &format!("{p}.layer_norm"), LN_EPS)?;
        let xs = match xshift {
            Some(t) => t.clone(),
            None => x.clone(),
        };
        let lerp_k = get(&self.w, &format!("{p}.lerp_k"))?.reshape((1, C))?;
        let mixed = lerp(&x, &xs, &lerp_k)?;
        let k = self.lin(&mixed, &format!("{p}.W_k"), false)?;
        let k = k.relu()?.sqr()?;
        let o = self.lin(&k, &format!("{p}.W_v"), false)?;
        let out = (in_x + o)?;
        Ok((out, x))
    }

    /// Batched single-step over one RWKV stream (n layers). Returns (out (B,C), new_state).
    fn run_stream_batched(
        &self,
        module_idx: usize,
        n_layers: usize,
        input: &Tensor,
        state: Option<&BatchedStreamState>,
    ) -> Result<(Tensor, BatchedStreamState)> {
        let mut x = input.clone();
        let mut v0: Option<Tensor> = None;
        let mut new_state: BatchedStreamState = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            let tp = format!("rwkv_modules.{module_idx}.blocks.{l}.time_mixer");
            let cp = format!("rwkv_modules.{module_idx}.blocks.{l}.channel_mixer");
            let ls = state.map(|s| &s[l]);
            let t_st = ls.map(|s| (&s.t_xshift, &s.t_state));
            let (xt, v0_out, t_xshift, t_state) =
                self.time_mixer_batched(&tp, l, &x, v0.as_ref(), t_st)?;
            let t_state = match self.state_quant_qmax.get(&module_idx) {
                Some(&qmax) => quant_roundtrip_batched(&t_state, qmax)?,
                None => t_state,
            };
            v0 = Some(v0_out);
            let c_st = ls.map(|s| &s.c_xshift);
            let (xc, c_xshift) = self.channel_mixer_batched(&cp, &xt, c_st)?;
            x = xc;
            new_state.push(BatchedLayerState { t_xshift, t_state, c_xshift });
        }
        Ok((x, new_state))
    }

    /// Batched forward over all 5 chained streams + heads. feats is (B,92); each state is (B,...).
    /// Returns (out_ahead_logits (B,np), out_w (B,nc), out_p_logits (B,4), new_states).
    pub fn review_batched(
        &self,
        feats: &Tensor,
        states: &[Option<BatchedStreamState>; 5],
    ) -> Result<(Tensor, Tensor, Tensor, [BatchedStreamState; 5])> {
        let mut x = self.features2card(feats)?; // (B,C); features2card is last-dim ops -> batch-fine
        let mut new: Vec<BatchedStreamState> = Vec::with_capacity(5);
        for m in 0..5 {
            let (xo, ns) =
                self.run_stream_batched(m, self.stream_layers[m], &x, states[m].as_ref())?;
            x = xo;
            new.push(ns);
        }
        let xh = self.ln(&x, "prehead_norm", LN_EPS)?;

        let hw = self.lin(&xh, "head_w.0", true)?.relu()?;
        let hw = self.ln(&hw, "head_w.2", LN_EPS)?;
        let hw = self.lin(&hw, "head_w.4", true)?;
        let out_w_logits = self.lin(&hw, "w_linear", true)?;
        let out_w = candle_nn::ops::softmax(&out_w_logits, D::Minus1)?;

        let ha = self.lin(&xh, "head_ahead_logits.0", true)?.relu()?;
        let out_ahead_logits = self.lin(&ha, "ahead_linear", true)?;

        let hp = self.lin(&xh, "head_p.0", true)?.relu()?;
        let out_p_logits = self.lin(&hp, "p_linear", true)?;

        let new_arr: [BatchedStreamState; 5] = new
            .try_into()
            .map_err(|_| anyhow!("stream count mismatch"))?;
        Ok((out_ahead_logits, out_w, out_p_logits, new_arr))
    }

    /// Batched imm probability = 1 - softmax(out_p_logits)[again] per card. Returns B values.
    pub fn imm_prob_batched(&self, out_p_logits: &Tensor) -> Result<Vec<f32>> {
        let p = candle_nn::ops::softmax(out_p_logits, D::Minus1)?; // (B,4)
        let again: Vec<f32> = p.narrow(1, 0, 1)?.flatten_all()?.to_vec1()?;
        Ok(again.iter().map(|a| 1.0 - a).collect())
    }
}

/// RWKV-7 WKV single timestep (matches rwkv_ops.single_timestep).
/// state' = state*w(cols) - (state@kd)@(a*kd)^T + v@k^T ; out = state'@r
#[allow(non_snake_case)]
fn single_timestep(
    n_heads: usize,
    head_dim: usize,
    r: &Tensor, // (H,K)
    k: &Tensor,
    v: &Tensor,
    w: &Tensor,
    a: &Tensor,
    kd: &Tensor,
    s_prev: &Tensor, // (H,K,K)
) -> Result<(Tensor, Tensor)> {
    let (H, K) = (n_heads, head_dim);
    let col = |t: &Tensor| -> Result<Tensor> { Ok(t.reshape((H, K, 1))?) };
    let row = |t: &Tensor| -> Result<Tensor> { Ok(t.reshape((H, 1, K))?) };

    // Both the decay and the remove term use the ORIGINAL state (Python evaluates the
    // whole RHS before reassigning): state*w(cols) - (state@kd)@(a*kd)^T
    let decay = s_prev.broadcast_mul(&row(w)?)?; // scale each column j by w[j]
    let sk = s_prev.matmul(&col(kd)?)?; // (H,K,1) -- from s_prev, NOT the decayed state
    let akd = row(&(a * kd)?)?; // (H,1,K)
    let s = (decay - sk.matmul(&akd)?)?;
    let s = (s + col(v)?.matmul(&row(k)?)?)?; // + v k^T
    let out = s.matmul(&col(r)?)?.reshape((H, K))?;
    Ok((out, s))
}

/// Batched WKV single timestep. r/k/v/w/a/kd are (B,H,K); s_prev is (B,H,K,K). candle's matmul
/// batches over the leading (B,H) dims, so the math is identical to `single_timestep` per-card.
#[allow(non_snake_case)]
fn single_timestep_batched(
    bsz: usize,
    n_heads: usize,
    head_dim: usize,
    r: &Tensor,
    k: &Tensor,
    v: &Tensor,
    w: &Tensor,
    a: &Tensor,
    kd: &Tensor,
    s_prev: &Tensor, // (B,H,K,K)
) -> Result<(Tensor, Tensor)> {
    let (B, H, K) = (bsz, n_heads, head_dim);
    let col = |t: &Tensor| -> Result<Tensor> { Ok(t.reshape((B, H, K, 1))?) };
    let row = |t: &Tensor| -> Result<Tensor> { Ok(t.reshape((B, H, 1, K))?) };

    let decay = s_prev.broadcast_mul(&row(w)?)?; // (B,H,K,K) * (B,H,1,K)
    let sk = s_prev.matmul(&col(kd)?)?; // (B,H,K,1)
    let akd = row(&(a * kd)?)?; // (B,H,1,K)
    let s = (decay - sk.matmul(&akd)?)?;
    let s = (s + col(v)?.matmul(&row(k)?)?)?;
    let out = s.matmul(&col(r)?)?.reshape((B, H, K))?;
    Ok((out, s))
}
