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
/// The two card-feature columns the button probes rewrite. These are the CARD block's own indices
/// (rwkv/model/srs_model_rnn.py: `CARD_FEATURE_COLUMNS.index(...)` = 8 and 9), and the card block
/// is the PREFIX of the input -- verified empirically on the reference traces: feats_proc[.,9..13]
/// is a valid one-hot on every row and feats_imm has 0.0 at both.
///
/// ⚠ THESE TWO ARE LAYOUT-STABLE AND THAT IS CHECKED, NOT ASSUMED. `RWKV_ID_FEATURES=1` drops
/// `scaled_state` at index 22 and appends 23 columns, so every index AFTER 22 shifts. 8 and 9 sit
/// before the drop point and are therefore unmoved in both layouts. (A column after it would need
/// to be derived per layout -- the same trap that made a features smoke pass vacuously in Python.)
pub(crate) const COL_DUR: usize = 8;
pub(crate) const COL_R1: usize = 9;

/// The input width used to be `const FEATURE_DIM: usize = 92`. It is now DERIVED at load from
/// `features2card.0.weight`, like H/K/C/stream_layers/arch already were -- the file's own header
/// says those are derived "so the engine auto-adapts to any arch", and this constant was the one
/// exception. `RWKV_ID_FEATURES=1` makes the real width 114, so a hardcoded 92 would have refused
/// to load a features model at all (and `RWKV_ZERO_FEATURES` would have refused to mask it).
/// See `Model::feature_dim` / `FastModel::feature_dim`; both read the SAME tensor, so the two
/// paths cannot disagree -- the divergence class this crate is most prone to.
const GN_EPS: f64 = 64e-5;
const L2_EPS: f64 = 1e-12;

type TMap = HashMap<String, Tensor>;

fn get<'a>(m: &'a TMap, k: &str) -> Result<&'a Tensor> {
    m.get(k).ok_or_else(|| anyhow!("missing weight: {k}"))
}

/// The string map a safetensors file carries under `__metadata__` (empty if it has none -- every
/// file exported before 2026-09-12). candle's loader returns tensors only, so the header is read
/// here directly from the format: an 8-byte little-endian header length, then that many bytes of
/// JSON. Only the header is read, never the tensor data.
fn safetensors_metadata(path: &str) -> Result<HashMap<String, String>> {
    use std::io::Read;
    let mut f = std::fs::File::open(path)?;
    let mut len8 = [0u8; 8];
    f.read_exact(&mut len8)?;
    let n = u64::from_le_bytes(len8);
    if n > 100_000_000 {
        return Err(anyhow!("{path}: safetensors header claims {n} bytes -- not a safetensors file?"));
    }
    let mut hdr = vec![0u8; n as usize];
    f.read_exact(&mut hdr)?;
    let v: serde_json::Value = serde_json::from_slice(&hdr)
        .map_err(|e| anyhow!("{path}: unreadable safetensors header: {e}"))?;
    let mut out = HashMap::new();
    if let Some(meta) = v.get("__metadata__").and_then(|m| m.as_object()) {
        for (k, val) in meta {
            if let Some(s) = val.as_str() {
                out.insert(k.clone(), s.to_string());
            }
        }
    }
    Ok(out)
}

/// True iff `k` is a STRIPPED component's leftover dummy.
///
/// The training side removes a sublayer by replacing it with a `d_model=1, n_heads=1` module
/// (rwkv_model.py / rwkv_rnn_model.py) rather than deleting it, so the state dict stays
/// symmetric and older checkpoints keep loading. The dummy is therefore exactly `(1,1)` and
/// "was this stripped?" is a SHAPE question -- no env vars needed on the deploy side.
///
/// The test is exact-`[1,1]` on purpose: a real rank-1 LoRA is `(1, d_model)` and a real
/// rank-2 one is `(2, d_model)` (verified against t2a18d_5586.pth), so any `min(dim)==1`
/// heuristic would silently strip a genuine rank-1 projection.
fn is_stripped(m: &TMap, k: &str) -> bool {
    m.get(k).is_some_and(|t| t.dims() == [1, 1])
}

/// Curve-head outputs for one review (candle path).
///
/// The classic head predicts only mixture weights over a FIXED log-spaced grid of stabilities.
/// The GRU head predicts stability and decay per curve as well, so the grid disappears and the
/// stored curve needs all three. Keeping them in one value (rather than widening the review()
/// tuple) means the ~20 call sites that only want the rating head stay untouched.
#[derive(Clone, Debug)]
pub struct CurveOut {
    /// (B, num_curves) post-softmax mixture weights.
    pub w: Tensor,
    /// `Some((s_raw, d_raw))` under the GRU head, each (B, num_curves). Raw = pre-clamp,
    /// pre-exp; the exp(clamp(.)) happens at curve-evaluation time, as in the Python.
    pub gru: Option<(Tensor, Tensor)>,
}

/// Architecture features detected from the checkpoint's weight SHAPES.
///
/// Everything here used to be an env-var recipe flag on the training side. At deploy the
/// checkpoint is the only input, so each one is recovered from the weights instead -- which
/// also means a mismatch cannot happen silently: `Model::load` prints this and refuses the
/// combinations that would otherwise run at a wrong width (see `detect_arch`).
#[derive(Clone, Debug)]
pub struct ArchFlags {
    /// `[stream][layer]` -- does this layer have a real channel mixer? (`RWKV_STRIP_CMIX`)
    pub has_cmix: Vec<Vec<bool>>,
    /// `[stream][layer]` -- does this layer have a real v_lora / v0 mixing?
    /// (`RWKV_STRIP_L0_VLORA`; layer 0 never mixes v0 regardless -- it *produces* it.)
    pub has_vlora: Vec<Vec<bool>>,
    /// `Some(N)` = GRU power-curve head with N curves, replacing the fixed-basis `w_linear`
    /// mixture: `R(t) = sum_i w_i * (1 + t/(1e-7+S_i))^(-d_i)`. (`RWKV_GRU_HEAD=N`)
    pub gru_curves: Option<usize>,
    /// Is the piecewise-linear ahead residual live? False = curve is the pure mixture, monotone
    /// in t by construction. (`RWKV_NO_AHEAD_RESIDUAL=1` strips `ahead_linear` to a dummy.)
    pub ahead_residual: bool,
    /// PAVA rectifier junction powers `theta` (3 floats), if the checkpoint carries them.
    /// `p = 2*tanh(theta)`; absent = classic arithmetic PAVA (p=1) for models trained without it.
    pub pava_theta: Option<Vec<f32>>,
    /// Per-step Frobenius clamp on the carried WKV state. Recipe flag, not a weight, so it comes
    /// from the environment (`RWKV_STATE_CLAMP_TAU`). See TRACK2_PORT_PLAN.md on why this is
    /// deliberately NOT bit-identical to the training clamp.
    pub state_clamp_tau: Option<f32>,
    /// Round-robin layer schedule across streams instead of stream-by-stream (`RWKV_INTERLEAVE=1`,
    /// the iter-41 champion). Recipe flag, not a weight -- NOT detectable from shapes, which is
    /// exactly why an engine built from the checkpoint alone would silently compute a different
    /// model. Front-loaded placement only (iter 44's "spread" variant was rejected).
    pub interleave: bool,
}

/// GAP 7, single source of truth: the execution order under the interleaved schedule, as
/// `(module, layer)` pairs. Round `r` runs layer `r` of every stream still deep enough, in module
/// order — FRONT-LOADED placement, matching the champion (iter 44's endpoint-anchored "spread"
/// variant was rejected as a tie and is deliberately not implemented).
///
/// Exists as one function because the loop is needed by THREE engines (candle B=1, candle batched,
/// fast). Three transcriptions with "keep in sync" comments is exactly how the fast/candle
/// divergences this crate is prone to get started, and unlike a numeric drift a schedule bug is
/// invisible to every shape check — it just computes a different model.
pub fn interleave_visits(stream_layers: &[usize]) -> Vec<(usize, usize)> {
    let n_rounds = stream_layers.iter().copied().max().unwrap_or(0);
    let mut out = Vec::with_capacity(stream_layers.iter().sum());
    for r in 0..n_rounds {
        for (m, &depth) in stream_layers.iter().enumerate() {
            if r < depth {
                out.push((m, r));
            }
        }
    }
    out
}

impl ArchFlags {
    /// One-line summary, printed at load. The whole point of shape-detection is that a
    /// checkpoint/engine mismatch is visible rather than silently numerically wrong.
    pub fn describe(&self) -> String {
        let strip = |v: &Vec<Vec<bool>>| -> String {
            let mut out = Vec::new();
            for (m, layers) in v.iter().enumerate() {
                for (l, live) in layers.iter().enumerate() {
                    if !*live {
                        out.push(format!("{m}:{l}"));
                    }
                }
            }
            if out.is_empty() { "none".into() } else { out.join(",") }
        };
        format!(
            "arch: head={} ahead_residual={} pava={} clamp={} sched={} | stripped cmix=[{}] vlora=[{}]",
            match self.gru_curves {
                Some(n) => format!("gru{n}"),
                None => "basis".into(),
            },
            self.ahead_residual,
            match &self.pava_theta {
                Some(t) => format!("{:?}", t),
                None => "no".into(),
            },
            match self.state_clamp_tau {
                Some(t) => format!("{t}"),
                None => "off".into(),
            },
            if self.interleave { "interleaved" } else { "sequential" },
            strip(&self.has_cmix),
            strip(&self.has_vlora),
        )
    }
}

/// Recover [`ArchFlags`] from weight shapes + the two recipe env vars that leave no trace in
/// the weights.
fn detect_arch(
    w: &TMap,
    stream_layers: &[usize],
    meta: &HashMap<String, String>,
) -> Result<ArchFlags> {
    let mut has_cmix = Vec::new();
    let mut has_vlora = Vec::new();
    for (m, &n_layers) in stream_layers.iter().enumerate() {
        let mut cm = Vec::new();
        let mut vl = Vec::new();
        for l in 0..n_layers {
            cm.push(!is_stripped(w, &format!("rwkv_modules.{m}.blocks.{l}.channel_mixer.W_k.weight")));
            vl.push(!is_stripped(
                w,
                &format!("rwkv_modules.{m}.blocks.{l}.time_mixer.v_lora_simple.A.weight"),
            ));
        }
        has_cmix.push(cm);
        has_vlora.push(vl);
    }

    let gru_curves = match w.get("gru_w_weight") {
        Some(t) => Some(t.dim(0)?),
        None => None,
    };
    // Guard the highest-risk line in the loader: with the GRU head the old `w_linear` is a (1,1)
    // dummy, and reading num_curves off it would silently give a width-1 mixture that computes
    // "successfully" and wrongly. Only one of the two heads may be live.
    let w_linear_stripped = is_stripped(w, "w_linear.weight");
    match (gru_curves, w_linear_stripped) {
        (Some(_), false) => return Err(anyhow!(
            "checkpoint has BOTH a GRU head (gru_w_weight) and a live w_linear -- ambiguous curve head"
        )),
        (None, true) => return Err(anyhow!(
            "checkpoint has NO curve head: w_linear is a (1,1) dummy and gru_w_weight is absent"
        )),
        _ => {}
    }

    let ahead_residual = !is_stripped(w, "ahead_linear.weight");
    let pava_theta = match w.get("pava_theta") {
        Some(t) => Some(t.flatten_all()?.to_vec1::<f32>()?),
        None => None,
    };
    // RECIPE flags: not detectable from any weight shape. Since 2026-09-12 the export writes them
    // into the safetensors metadata, with the stream order's precedence (Model::load): env +
    // metadata must AGREE or the load is refused; metadata alone is used; env alone, or neither,
    // is exactly the old behaviour (every file exported before that date has no metadata).
    // Measured the same day: a run missing RWKV_INTERLEAVE alone misses by up to 0.66 per review.
    let env_set = |k: &str| std::env::var(k).ok().filter(|s| !s.is_empty());
    let parse_tau = |s: &str, src: &str| -> Result<Option<f32>> {
        let v: f32 = s.parse().map_err(|_| anyhow!("bad state clamp tau in {src}: {s}"))?;
        Ok(if v > 0.0 { Some(v) } else { None })
    };
    let state_clamp_tau = match (env_set("RWKV_STATE_CLAMP_TAU"), meta.get("state_clamp_tau")) {
        (Some(e), Some(m)) => {
            let (te, tm) = (parse_tau(&e, "RWKV_STATE_CLAMP_TAU")?, parse_tau(m, "the weights' metadata")?);
            if te != tm {
                return Err(anyhow!(
                    "RWKV_STATE_CLAMP_TAU={e:?} disagrees with the weights' metadata ({m:?}). \
                     Refusing; unset the variable to use the weights' value."
                ));
            }
            te
        }
        (Some(e), None) => parse_tau(&e, "RWKV_STATE_CLAMP_TAU")?,
        (None, Some(m)) => {
            eprintln!("[rwkv-infer] state clamp tau from the weights' metadata: {m}");
            parse_tau(m, "the weights' metadata")?
        }
        (None, None) => None,
    };

    let interleave = match (env_set("RWKV_INTERLEAVE"), meta.get("interleave")) {
        (Some(e), Some(m)) => {
            if (e == "1") != (m == "1") {
                return Err(anyhow!(
                    "RWKV_INTERLEAVE={e:?} disagrees with the weights' metadata ({m:?}). \
                     Refusing: one of the two schedules computes a different model. Unset the \
                     variable to use the weights' value."
                ));
            }
            e == "1"
        }
        (Some(e), None) => e == "1",
        (None, Some(m)) => {
            eprintln!("[rwkv-infer] interleave from the weights' metadata: {m}");
            m == "1"
        }
        (None, None) => false,
    };
    if let Some(p) = meta.get("interleave_placement") {
        if p != "front" {
            return Err(anyhow!(
                "the weights were exported with interleave placement {p:?}; this engine implements \
                 the front-loaded placement only (iter 44's spread variant was never ported)"
            ));
        }
    }

    Ok(ArchFlags {
        has_cmix,
        has_vlora,
        gru_curves,
        ahead_residual,
        pava_theta,
        state_clamp_tau,
        interleave,
    })
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
    crate::finite::vec_raw(v); // observation only (RWKV_FINITE_CHECK)
    let amax = v.iter().fold(0f32, |m, &x| m.max(x.abs())) as f64;
    // A non-finite input makes amax non-finite, hence scale inf, hence every output NaN - so the exit
    // count below is the honest "did quantization produce garbage" number. The value itself is kept.
    let scale = (amax / qmax).max(1e-12);
    for x in v.iter_mut() {
        *x = (((*x as f64) / scale).round().clamp(-qmax, qmax) * scale) as f32;
    }
    crate::finite::vec_quant(v);
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
    pub shift_level: std::collections::HashMap<usize, f64>,
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
    /// FIXED-BASIS (codec) state coding: stream -> (m, coefficient qmax). Instead of re-describing a
    /// BASIS every step (the rank-r path stores 2*K*r factors), store the m x m coefficient block
    /// C = P^T W Q against a FIXED K x m basis pair (P, Q) that ships with the model. Bits/stream =
    /// layers * (m*m*qbits + 64*shift_bits). A codec basis, NOT a codebook (no discrete entries, no
    /// nearest-neighbour lookup, no per-user vectors). Applied INSTEAD of the rank-r round-trip for
    /// that stream. Set RWKV_STATE_FIXEDBASIS_SCOPE + RWKV_FIXEDBASIS_PATH.
    pub fb: std::collections::HashMap<usize, (usize, Option<f64>)>,
    /// The fixed bases themselves: (module_idx, layer) -> (P, Q), each K x stride row-major. Arc-shared
    /// because a basis is a MODEL-side artifact, identical for every card -- not per-card state.
    pub fb_basis: std::sync::Arc<std::collections::HashMap<(usize, usize), (Vec<f32>, Vec<f32>)>>,
    /// RWKV_SHIFT_FIXEDBASIS_PATH=<file>: FIXED-BASIS (codec) coding of the persisted TOKEN-SHIFT
    /// vectors - the `fb` idea applied to the 2 x C shift payload a layer persists. The artifact
    /// carries `mu` (C floats) + `P` (C x stride) per key; keys are TIERED exactly like the state
    /// codec's lookup (`<stream>__r<role>__L<layer>` -> `<stream>__r<role>` -> `r<role>`) and are
    /// resolved ONCE at load into (module, layer, role). `shift_fb_m` is the coefficient count; the
    /// coefficient LEVEL is the stream's shift level (RWKV_STATE_SHIFT_LEVEL / the state level), so
    /// the shift budget becomes 2*m*level_bits per layer instead of 64*level_bits.
    /// NOT a codebook: a fixed linear transform, no discrete entries, no per-user vectors, ZERO
    /// state bits for mu/P. Precedence: RWKV_SHIFT_PQ > this > int-N.
    pub shift_fb: std::sync::Arc<std::collections::HashMap<(usize, usize, u8), ShiftBasis>>,
    pub shift_fb_m: usize,
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
        None, // no fast-rank basis: this wrapper is the PARITY REFERENCE, the fast path is engine-only
    );
    Ok(Tensor::from_vec(data, (h, k, k), t.device())?)
}

/// FIXED-BASIS (codec) state coding: W -> P * quant(P^T W Q) * Q^T, one scale per head. `p` and `q`
/// are K x `stride` row-major basis matrices that ship with the model (only the first `m` columns are
/// read). This replaces the rank-r round-trip's per-step re-description of a BASIS (2*K*r numbers)
/// with an m x m COEFFICIENT block against a basis fixed at deploy (m*m numbers), and its cost
/// (2*K*K*m + 2*K*m*m flops) sits below the rank>=2 Gram + symmetric eigensolve, so it is faster too.
/// NOT a codebook: no discrete entries, no nearest-neighbour lookup, no per-user vectors.
pub fn fixedbasis_roundtrip(
    data: &mut [f32],
    h: usize,
    k: usize,
    m: usize,
    p: &[f32],
    q: &[f32],
    stride: usize,
    qmax: Option<f64>,
) {
    assert!(
        m > 0 && m <= stride && m <= k,
        "fixedbasis: bad block size m={m} (stride={stride}, k={k})"
    );
    assert!(
        p.len() >= k * stride && q.len() >= k * stride,
        "fixedbasis: basis too small ({} / {} vs {})",
        p.len(),
        q.len(),
        k * stride
    );
    crate::finite::state_raw(data); // observation only (RWKV_FINITE_CHECK)
    // The bases ship as K x stride row-major, so with stride = K every inner-loop access below would
    // sit `stride * 4` bytes (a fresh cache line) from the previous one, in all four loops. Transposing
    // them ONCE per call costs ~0.9 us on a 32x32 state (see `compress_bench::bench_fixedbasis_layout`)
    // and makes every inner loop contiguous: measured 1.70x at m=6 and 1.70-1.80x at m=8..16, with
    // max|diff| = 0 because the accumulation ORDER in each loop is unchanged. The result is therefore
    // bit-identical to the strided version, which is what the Python<->Rust parity checks rely on.
    let pt = transpose_basis(p, k, stride);
    let qt = transpose_basis(q, k, stride);
    let mut tmp = vec![0f32; k * m]; // W Q       (k x m)
    let mut c = vec![0f32; m * m]; //  P^T W Q   (m x m) -- the stored payload
    let mut u = vec![0f32; k * m]; //  P C       (k x m)
    for hh in 0..h {
        let w = &mut data[hh * k * k..(hh + 1) * k * k];
        for i in 0..k {
            let wrow = &w[i * k..i * k + k];
            for t in 0..m {
                let qrow = &qt[t * k..t * k + k];
                let mut s = 0f32;
                for j in 0..k {
                    s += wrow[j] * qrow[j];
                }
                tmp[i * m + t] = s;
            }
        }
        for s0 in 0..m {
            let prow = &pt[s0 * k..s0 * k + k];
            for t in 0..m {
                let mut s = 0f32;
                for i in 0..k {
                    s += prow[i] * tmp[i * m + t];
                }
                c[s0 * m + t] = s;
            }
        }
        if let Some(qmax) = qmax {
            quant_vec_inplace(&mut c, qmax);
        }
        for i in 0..k {
            let prow = &p[i * stride..i * stride + stride];
            for t in 0..m {
                let mut s = 0f32;
                for s0 in 0..m {
                    s += prow[s0] * c[s0 * m + t];
                }
                u[i * m + t] = s;
            }
        }
        for i in 0..k {
            let urow = &u[i * m..i * m + m];
            for j in 0..k {
                let qrow = &q[j * stride..j * stride + stride];
                let mut s = 0f32;
                for t in 0..m {
                    s += urow[t] * qrow[t];
                }
                w[i * k + j] = s;
            }
        }
        crate::finite::state_quant(w); // observation only
    }
}

/// Transpose a K x `stride` row-major basis into a `stride` x K one: `out[t*k + i] = a[i*stride + t]`.
/// The codec's contiguous fast path needs both orientations; this bridges the on-disk layout.
pub fn transpose_basis(a: &[f32], k: usize, stride: usize) -> Vec<f32> {
    let mut out = vec![0f32; k * stride];
    for i in 0..k {
        let row = &a[i * stride..i * stride + stride];
        for t in 0..stride {
            out[t * k + i] = row[t];
        }
    }
    out
}

/// One key's token-shift codec basis (see `fixedbasis_vec_inplace`): `mu` (C floats) plus `P`
/// (C x `stride`, row-major). `pt` is P transposed ONCE at load so both inner loops below are
/// contiguous - the state codec's own layout fix (1.7-1.8x there, bit-identical because the
/// accumulation order is unchanged).
pub struct ShiftBasis {
    pub stride: usize,
    pub mu: Vec<f32>,
    pub p: Vec<f32>,
    pt: Vec<f32>,
}

impl ShiftBasis {
    pub fn new(stride: usize, mu: Vec<f32>, p: Vec<f32>) -> Self {
        let c = mu.len();
        let pt = transpose_basis(&p, c, stride);
        Self { stride, mu, p, pt }
    }
}

/// FIXED-BASIS (codec) round-trip of ONE persisted token-shift vector:
/// `x -> mu + P * quant(P^T (x - mu))`. `mu`/`P` ship with the model (ZERO state bits), so the coded
/// payload is only the `m` coefficients; the quantizer is the SAME symmetric per-vector int-N rule
/// `quant_vec_inplace` applies (scale = amax/qmax per vector), so the level semantics are unchanged.
/// Cost is 2*C*m flops. NOT a codebook: a fixed linear transform, no discrete entries, no
/// nearest-neighbour lookup, no per-user vectors.
pub fn fixedbasis_vec_inplace(v: &mut [f32], b: &ShiftBasis, m: usize, qmax: f64) {
    let c = v.len();
    assert!(
        m > 0 && m <= b.stride && b.mu.len() == c && b.p.len() >= c * b.stride,
        "shift fixedbasis: bad shapes (C={c}, stride={}, m={m})",
        b.stride
    );
    crate::finite::vec_raw(v); // observation only (RWKV_FINITE_CHECK)
    let mut coef = vec![0f32; m];
    for (j, cj) in coef.iter_mut().enumerate() {
        let prow = &b.pt[j * c..j * c + c];
        let mut s = 0f32;
        for i in 0..c {
            s += (v[i] - b.mu[i]) * prow[i];
        }
        *cj = s;
    }
    quant_vec_inplace(&mut coef, qmax);
    for i in 0..c {
        let prow = &b.p[i * b.stride..i * b.stride + m];
        let mut s = b.mu[i];
        for (j, &cj) in coef.iter().enumerate() {
            s += prow[j] * cj;
        }
        v[i] = s;
    }
    crate::finite::vec_quant(v); // observation only
}

/// Candle wrapper for ONE shift vector (...,C): codec-round-trip every C-long row in place.
fn shift_fb_roundtrip_tensor(t: &Tensor, b: &ShiftBasis, m: usize, qmax: f64) -> Result<Tensor> {
    let c = t.dims().last().copied().unwrap_or(0);
    let mut data: Vec<f32> = t.flatten_all()?.to_vec1()?;
    if c > 0 {
        for chunk in data.chunks_mut(c) {
            fixedbasis_vec_inplace(chunk, b, m, qmax);
        }
    }
    Ok(Tensor::from_vec(data, t.dims().to_vec(), t.device())?)
}

/// Candle wrapper for `fixedbasis_roundtrip` (H,K,K), mirroring `lowrank_roundtrip`: the parity
/// reference for the fast engine's copy of the same math.
fn fixedbasis_roundtrip_tensor(
    t: &Tensor,
    m: usize,
    p: &[f32],
    q: &[f32],
    qmax: Option<f64>,
) -> Result<Tensor> {
    let (h, k, k2) = t.dims3()?;
    assert_eq!(k, k2, "WKV state must be square KxK");
    let stride = (p.len() / k).max(1);
    let mut data: Vec<f32> = t.flatten_all()?.to_vec1()?;
    fixedbasis_roundtrip(&mut data, h, k, m, p, q, stride, qmax);
    Ok(Tensor::from_vec(data, (h, k, k), t.device())?)
}

/// Batched candle wrapper (B,H,K,K). The coefficient scale is per CARD (per leading dim), matching
/// `quant_roundtrip_batched` and the fast engine's per-entity loop.
fn fixedbasis_roundtrip_batched(
    t: &Tensor,
    m: usize,
    p: &[f32],
    q: &[f32],
    qmax: Option<f64>,
) -> Result<Tensor> {
    let (b, h, k, k2) = t.dims4()?;
    assert_eq!(k, k2, "WKV state must be square KxK");
    let stride = (p.len() / k).max(1);
    let mut data: Vec<f32> = t.flatten_all()?.to_vec1()?;
    let per = h * k * k;
    for bi in 0..b {
        fixedbasis_roundtrip(&mut data[bi * per..(bi + 1) * per], h, k, m, p, q, stride, qmax);
    }
    Ok(Tensor::from_vec(data, (b, h, k, k), t.device())?)
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
    /// 2026-09-19 fast scan: ||centroid||^2 per block, for the dot-product PRESCREEN (see `nearest`).
    cn2: Vec<Vec<f32>>,
    /// the same blocks TRANSPOSED (dimension j of every centroid contiguous at [j*ncent..]): the prescreen is then
    /// sub_dim axpy passes over ncent floats, which vectorizes across centroids (throughput-bound, not latency-bound).
    cbt: Vec<Vec<f32>>,
    /// RWKV_PQ_SCAN=exact restores the serial early-exit scan (the verification reference). Picks are identical.
    fast_scan: bool,
    /// 2026-09-19 (evening): 13-bit integer copy of each block for the INTEGER prescreen (`I16Blk`); RWKV_PQ_SCAN=f32
    /// keeps the float prescreen. Empty when off.
    cb16: Vec<I16Blk>,
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
    /// RESIDUAL VQ (2026-09-20, joint mode only): the catalog holds `stages` blocks of `ncent`
    /// centroids and each stage codes the RESIDUAL left by the earlier ones, so the reconstruction is
    /// the SUM of `stages` picks and the per-head index cost is `stages * bits`. A single index cannot
    /// be fitted or searched past ~b15, so this is the only way to spend a large bit grant. The header
    /// carries the stage count as an optional 6th field; a file without it is a 1-stage catalog and
    /// takes exactly the code path it always took.
    pub stages: usize,
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
        // optional 6th header field = residual-VQ stage count (see the `stages` field). Absent = 1.
        let stages = if h.len() > 5 { h[5] } else { 1 };
        anyhow::ensure!(stages >= 1 && stages <= 8, "PQ stages out of range: {stages}");
        anyhow::ensure!(stages == 1 || joint, "residual VQ (stages > 1) is joint-mode only");
        let mut cb = Vec::with_capacity(n_roles * m * stages);
        for _ in 0..n_roles * m * stages {
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
        let cn2: Vec<Vec<f32>> = cb
            .iter()
            .map(|blk| blk.chunks_exact(sub_dim).map(|c| c.iter().map(|x| x * x).sum::<f32>()).collect())
            .collect();
        let cbt: Vec<Vec<f32>> = cb
            .iter()
            .map(|blk| {
                let mut t = vec![0f32; blk.len()];
                for c in 0..ncent {
                    for j in 0..sub_dim {
                        t[j * ncent + c] = blk[c * sub_dim + j];
                    }
                }
                t
            })
            .collect();
        let fast_scan = std::env::var("RWKV_PQ_SCAN").map(|v| v != "exact").unwrap_or(true);
        let int_scan = fast_scan && std::env::var("RWKV_PQ_SCAN").map(|v| v != "f32").unwrap_or(true);
        let cb16: Vec<I16Blk> = if int_scan { cb.iter().map(|blk| I16Blk::build(blk, ncent, sub_dim)).collect() } else { Vec::new() };
        Ok(Self { m, sub_dim, k, ncent, cb, cn2, cbt, fast_scan, cb16, norm_bits: None, norm_lo_log2: 0.0, norm_hi_log2: 0.0, joint, stages, warm_enabled })
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
        if self.fast_scan {
            let mut scratch: Vec<f32> = Vec::new();
            let mut q = vec![0f32; self.sub_dim];
            for p in 0..self.m {
                let s = p * self.sub_dim;
                let blk = role * self.m + p;
                for j in 0..self.sub_dim {
                    q[j] = col[s + j] * inv; // the same product the serial scan forms per element
                }
                let best = nearest(&q, &self.cb[blk], &self.cbt[blk], &self.cn2[blk], self.cb16.get(blk));
                if let Some(w) = warm.as_deref_mut() {
                    w[p] = best as i32;
                }
                let base = best * self.sub_dim;
                for j in 0..self.sub_dim {
                    col[s + j] = self.cb[blk][base + j] * norm;
                }
            }
            return;
        }
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
        if self.stages > 1 {
            // RESIDUAL VQ: stage 0 codes the unit pair, every later stage codes what the earlier
            // stages left behind, and the reconstruction is their SUM -- the same order the CUDA
            // kernel accumulates in (`c_pq_stages`), so train and deploy pick the same centroids.
            // The warm-start bound is not used here: it is a speed optimization with a per-stage slot
            // in the kernel, and the deploy state cache carries one slot per (head, component).
            let mut acc = vec![0f32; 2 * k];
            let mut q = vec![0f32; 2 * k];
            for st in 0..self.stages {
                for j in 0..k {
                    q[j] = u[j] * iu - acc[j];
                    q[k + j] = v[j] * iv - acc[k + j];
                }
                let best = if self.fast_scan {
                    nearest(&q, &self.cb[st], &self.cbt[st], &self.cn2[st], self.cb16.get(st))
                } else {
                    nearest_exact(&q, &self.cb[st], self.ncent, self.sub_dim)
                };
                let base = best * self.sub_dim;
                for j in 0..2 * k {
                    acc[j] += self.cb[st][base + j];
                }
            }
            if let Some(w) = warm {
                *w = -1; // no single winning index to carry forward
            }
            for j in 0..k {
                u[j] = acc[j] * qu;
                v[j] = acc[k + j] * qv;
            }
            return;
        }
        if self.fast_scan {
            let mut q = vec![0f32; 2 * k];
            for j in 0..k {
                q[j] = u[j] * iu;
                q[k + j] = v[j] * iv;
            }
            let mut scratch: Vec<f32> = Vec::new();
            let best = nearest(&q, &self.cb[0], &self.cbt[0], &self.cn2[0], self.cb16.first());
            if let Some(w) = warm {
                *w = best as i32;
            }
            let base = best * self.sub_dim;
            for j in 0..k {
                u[j] = self.cb[0][base + j] * qu;
                v[j] = self.cb[0][base + k + j] * qv;
            }
            return;
        }
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

/// Plain argmin with first-strict-min tie-breaking -- the `RWKV_PQ_SCAN=exact` reference for one
/// residual-VQ stage. The single-stage exact scan keeps its own early-exit loop, which is the same
/// function written for the u-half/v-half accumulation order; this one takes an already-built query.
fn nearest_exact(q: &[f32], cents: &[f32], ncent: usize, sub_dim: usize) -> usize {
    let (mut best, mut bestd) = (0usize, f32::INFINITY);
    for c in 0..ncent {
        let base = c * sub_dim;
        let mut d = 0f32;
        for j in 0..sub_dim {
            let diff = q[j] - cents[base + j];
            d += diff * diff;
        }
        if d < bestd {
            bestd = d;
            best = c;
        }
    }
    best
}

#[inline(always)]
fn prescreen_body(q: &[f32], cbt: &[f32], cn2: &[f32], out: &mut [f32]) {
    let n = cn2.len();
    out.copy_from_slice(cn2);
    for (j, &qj) in q.iter().enumerate() {
        let w = -2.0 * qj;
        let col = &cbt[j * n..(j + 1) * n];
        for (o, &x) in out.iter_mut().zip(col.iter()) {
            *o += w * x;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn prescreen_avx2(q: &[f32], cbt: &[f32], cn2: &[f32], out: &mut [f32]) {
    prescreen_body(q, cbt, cn2, out)
}

fn prescreen_scores(q: &[f32], cbt: &[f32], cn2: &[f32], out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe { prescreen_avx2(q, cbt, cn2, out) };
            return;
        }
    }
    prescreen_body(q, cbt, cn2, out)
}

/// Nearest centroid to the (already normalized) query `q`, PICK-IDENTICAL to the serial scan
/// (argmin by exact serial distance, then lower index; centroid 0 if nothing is finite).
/// Stage 1: approximate scores ||c||^2 - 2 q.c for every centroid (vectorizable; = distance - ||q||^2).
/// Stage 2: every centroid within PRESCREEN_EPS of the best approximate score is re-measured with the
/// EXACT serial sum of squared differences, in index order, strict-less update. The true winner is always
/// in that set while the approximation error stays under EPS/2: q and the centroids are unit-scale, so
/// the error of a <=64-term f32 sum is ~1e-5 against an EPS of 1e-3. A non-finite best score (cannot
/// happen for finite inputs) falls back to the full exact scan.
const PRESCREEN_EPS: f32 = 1e-3;
thread_local! { static NEAREST_SCRATCH: std::cell::RefCell<(Vec<f32>, Vec<i32>, Vec<i16>)> = std::cell::RefCell::new((Vec::new(), Vec::new(), Vec::new())); }

/// INTEGER prescreen block. Centroids and the query are rounded to 13-bit integers (|x| <= 4095, so a sum of <= 32
/// pair products stays inside i32); dims are stored in PAIRS, interleaved per centroid ([c0_j, c0_j+1, c1_j, c1_j+1, ..]),
/// so one `madd_epi16` gives q_j c_j + q_j+1 c_j+1 for 8 centroids = 16 multiply-adds per instruction, on half the bytes
/// of the f32 table. Integer sums are exact, so the prescreen gives the same numbers on every CPU and in the scalar
/// fallback. The rounding error of the dot product is BOUNDED (|q.c - sq sc qi.ci| <= 0.5 sq sc ||qi||_1 + 0.5 sq max||c||_1),
/// and every centroid within twice the score bound (+ slack) of the best approximate score is re-measured exactly,
/// so the pick is identical to the serial scan.
pub(crate) struct I16Blk {
    pairs: Vec<i16>, // [pair p][centroid c][2]
    npairs: usize,
    sc: f32,    // centroid scale: c ~ sc * ci
    c1max: f32, // max over centroids of ||c||_1
}
impl I16Blk {
    fn build(blk: &[f32], ncent: usize, d: usize) -> Self {
        let amax = blk.iter().fold(0f32, |m, x| m.max(x.abs()));
        let sc = if amax > 0.0 && amax.is_finite() { amax / 4095.0 } else { 1.0 };
        let npairs = (d + 1) / 2;
        let mut pairs = vec![0i16; npairs * ncent * 2];
        let mut c1max = 0f32;
        for c in 0..ncent {
            let mut l1 = 0f32;
            for j in 0..d {
                let x = blk[c * d + j];
                l1 += x.abs();
                pairs[(j / 2) * ncent * 2 + c * 2 + (j & 1)] = (x / sc).round().clamp(-4095.0, 4095.0) as i16;
            }
            c1max = c1max.max(l1);
        }
        Self { pairs, npairs, sc, c1max }
    }
}
#[inline(always)]
fn int_scores_body(qi: &[i16], blk: &I16Blk, acc: &mut [i32]) {
    let n = acc.len();
    for p in 0..blk.npairs {
        let (q0, q1) = (qi[2 * p] as i32, qi[2 * p + 1] as i32);
        let col = &blk.pairs[p * n * 2..(p + 1) * n * 2];
        for c in 0..n {
            acc[c] += q0 * col[2 * c] as i32 + q1 * col[2 * c + 1] as i32;
        }
    }
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn int_scores_avx2(qi: &[i16], blk: &I16Blk, acc: &mut [i32]) {
    use std::arch::x86_64::*;
    let n = acc.len();
    let n8 = n / 8 * 8;
    for p in 0..blk.npairs {
        let (q0, q1) = (qi[2 * p], qi[2 * p + 1]);
        let qv = _mm256_set1_epi32(((q1 as u16 as i32) << 16) | (q0 as u16 as i32));
        let col = blk.pairs.as_ptr().add(p * n * 2);
        let mut c = 0;
        while c < n8 {
            let cv = _mm256_loadu_si256(col.add(2 * c) as *const __m256i);
            let a = acc.as_mut_ptr().add(c) as *mut __m256i;
            _mm256_storeu_si256(a, _mm256_add_epi32(_mm256_loadu_si256(a), _mm256_madd_epi16(cv, qv)));
            c += 8;
        }
        while c < n {
            acc[c] += q0 as i32 * *col.add(2 * c) as i32 + q1 as i32 * *col.add(2 * c + 1) as i32;
            c += 1;
        }
    }
}

#[inline(always)]
fn scores_from_int_body(out: &mut [f32], cn2: &[f32], acc: &[i32], k2: f32) {
    for ((o, &n2), &a) in out.iter_mut().zip(cn2.iter()).zip(acc.iter()) {
        *o = n2 + k2 * a as f32;
    }
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn scores_from_int_avx2(out: &mut [f32], cn2: &[f32], acc: &[i32], k2: f32) {
    scores_from_int_body(out, cn2, acc, k2)
}
fn scores_from_int(out: &mut [f32], cn2: &[f32], acc: &[i32], k2: f32) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { scores_from_int_avx2(out, cn2, acc, k2) };
        }
    }
    scores_from_int_body(out, cn2, acc, k2)
}
/// min of a slice in 8 independent lanes (min is exact, so the lane order cannot change the result; NaN never wins,
/// like the `v < smin` loop this replaces).
fn slice_min(x: &[f32]) -> f32 {
    let mut lanes = [f32::INFINITY; 8];
    let mut it = x.chunks_exact(8);
    for ch in &mut it {
        for l in 0..8 {
            lanes[l] = if ch[l] < lanes[l] { ch[l] } else { lanes[l] };
        }
    }
    let mut m = f32::INFINITY;
    for &v in lanes.iter().chain(it.remainder().iter()) {
        if v < m {
            m = v;
        }
    }
    m
}

fn nearest(q: &[f32], cents: &[f32], cbt: &[f32], cn2: &[f32], i16blk: Option<&I16Blk>) -> usize {
    NEAREST_SCRATCH.with(|sc| {
        let mut g = sc.borrow_mut();
        let (f, a, qi) = &mut *g;
        nearest_with(q, cents, cbt, cn2, i16blk, f, a, qi)
    })
}
#[allow(clippy::too_many_arguments)]
fn nearest_with(q: &[f32], cents: &[f32], cbt: &[f32], cn2: &[f32], i16blk: Option<&I16Blk>, scratch: &mut Vec<f32>, acc: &mut Vec<i32>, qi: &mut Vec<i16>) -> usize {
    let d = q.len();
    let ncent = cn2.len();
    scratch.resize(ncent, 0.0);
    let qmax = q.iter().fold(0f32, |m, x| m.max(x.abs()));
    let mut eps = PRESCREEN_EPS;
    match i16blk {
        Some(blk) if qmax > 0.0 && qmax.is_finite() => {
            let sq = qmax / 4095.0;
            qi.clear();
            qi.resize(blk.npairs * 2, 0);
            let mut q1 = 0f32;
            for j in 0..d {
                let v = (q[j] / sq).round().clamp(-4095.0, 4095.0);
                qi[j] = v as i16;
                q1 += v.abs();
            }
            acc.clear();
            acc.resize(ncent, 0);
            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("avx2") {
                    unsafe { int_scores_avx2(qi, blk, acc) };
                } else {
                    int_scores_body(qi, blk, acc);
                }
            }
            #[cfg(not(target_arch = "x86_64"))]
            int_scores_body(qi, blk, acc);
            let k2 = -2.0 * sq * blk.sc;
            scores_from_int(scratch, cn2, acc, k2);
            // score error <= 2 * dot error; candidates within twice that of the best, plus the float slack
            let bound = 0.5 * sq * blk.sc * q1 + 0.5 * sq * blk.c1max;
            eps = 4.0 * bound + PRESCREEN_EPS;
        }
        _ => prescreen_scores(q, cbt, cn2, scratch),
    }
    let smin = slice_min(scratch);
    let all = !smin.is_finite();
    let lim = smin + eps;
    let (mut best, mut bestd) = (0usize, f32::INFINITY);
    let mut ncand = 0u64;
    for c0 in (0..ncent).step_by(64) {
        let c1 = (c0 + 64).min(ncent);
        // one vectorizable test per 64 scores; the index order of the recheck (and so the tie rule) is unchanged
        if !all && !scratch[c0..c1].iter().fold(false, |any, &v| any | (v <= lim)) {
            continue;
        }
        for c in c0..c1 {
        if all || scratch[c] <= lim {
            ncand += 1;
            let base = c * d;
            let mut dist = 0f32;
            for j in 0..d {
                let diff = q[j] - cents[base + j];
                dist += diff * diff;
            }
            if dist < bestd {
                bestd = dist;
                best = c;
            }
        }
        }
    }
    CAND_N.fetch_add(ncand, std::sync::atomic::Ordering::Relaxed);
    CAND_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    best
}
pub static CAND_N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static CAND_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The rank-1 power iteration of `compress_wkv_state` on STACK matrices (no heap traffic per head / per iteration).
/// Same nalgebra generic kernels, same operation order as the dynamic path. Returns (u, sigma*v, sigma).
fn rank1_static<const K: usize>(a_rows: &[f32], scale: f32) -> Option<(Vec<f32>, Vec<f32>, f32)> {
    use nalgebra::{SMatrix, SVector};
    let a = SMatrix::<f32, K, K>::from_row_slice(a_rows);
    let an = a * (1.0 / scale);
    let ant = an.transpose();
    let mut u = SVector::<f32, K>::from_element(1.0 / (K as f32).sqrt());
    for _ in 0..64 {
        let atu = ant * u;
        let nu = an * atu;
        let nrm = nu.norm();
        if !nrm.is_finite() || nrm < 1e-30 {
            break;
        }
        let nu = nu / nrm;
        let dot = u.dot(&nu).abs();
        u = nu;
        if 1.0 - dot < 1e-7 {
            break;
        }
    }
    let v_un = a.transpose() * u;
    let sigma = v_un.norm();
    Some((u.iter().copied().collect(), v_un.iter().copied().collect(), sigma))
}

/// `RWKV_LOWRANK_FAST=1` switches the rank>=2 truncation from the reference KxK Gram + symmetric
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
/// Counts of the recorded differences in buckets (<1e-3, <1e-2, <1e-1, <1, >=1). The MAX alone cannot
/// separate a single cold-start outlier -- the first compression of a state has no warm basis -- from a
/// systematic error, and the deploy decision needs both facts.
static REL_BUCKETS: [std::sync::atomic::AtomicU64; 5] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];

/// Report the fast-path counters (called from `main` next to the finiteness report).
pub fn fastpath_report() {
    use std::sync::atomic::Ordering;
    if !lowrank_fast_on() && !lowrank_fast_check() {
        return;
    }
    let calls = FAST_CALLS.load(Ordering::Relaxed);
    let fb = FAST_FALLBACK.load(Ordering::Relaxed);
    let worst = f32::from_bits(FAST_MAX_REL.load(Ordering::Relaxed) as u32);
    let bk: Vec<u64> = REL_BUCKETS.iter().map(|b| b.load(Ordering::Relaxed)).collect();
    eprintln!(
        "[fastrank] mode={}{} iters={} rank>=2 calls={calls} reference fallbacks={fb} \
         max_relFrobenius_vs_reference={worst:.3e} rel_buckets={}/{}/{}/{}/{}",
        if lowrank_fast_on() { "fast" } else { "measure-only" },
        if lowrank_fast_check() { "+check" } else { "" },
        fast_iters(),
        bk[0],
        bk[1],
        bk[2],
        bk[3],
        bk[4]
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
    let bucket = if rel < 1e-3 {
        0
    } else if rel < 1e-2 {
        1
    } else if rel < 1e-1 {
        2
    } else if rel < 1.0 {
        3
    } else {
        4
    };
    REL_BUCKETS[bucket].fetch_add(1, Ordering::Relaxed);
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
        (32, 2) => rankr_fast_impl!(32, 2, a, an, &mut *q, &mut *uf, &mut *vf),
        (32, 3) => rankr_fast_impl!(32, 3, a, an, &mut *q, &mut *uf, &mut *vf),
        (32, 4) => rankr_fast_impl!(32, 4, a, an, &mut *q, &mut *uf, &mut *vf),
        (16, 2) => rankr_fast_impl!(16, 2, a, an, &mut *q, &mut *uf, &mut *vf),
        (8, 2) => rankr_fast_impl!(8, 2, a, an, &mut *q, &mut *uf, &mut *vf),
        _ => false, // any other shape: keep the reference path
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
    // FAST RANK-r PATH (RWKV_LOWRANK_FAST): this (entity, stream, head)'s subspace block from the
    // previous step, layout [head][R+1 columns][K], i.e. len >= h*(r+1)*k. None = no warm basis,
    // which just means the fast path cold-starts; the reference path ignores it entirely.
    mut basis: Option<&mut [f32]>,
) {
    use nalgebra::DMatrix;
    // DEBUG ONLY (2026-09-19): RWKV_DEBUG_JITTER=<rel> multiplies every state element by 1 +/- rel (deterministic hash
    // sign) before compression -- a stand-in for another CPU's float order, to measure how far quantized states diverge.
    {
        use std::sync::OnceLock;
        static J: OnceLock<f32> = OnceLock::new();
        static CNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let j = *J.get_or_init(|| std::env::var("RWKV_DEBUG_JITTER").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0));
        if j != 0.0 {
            let c = CNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            for (i, x) in data.iter_mut().enumerate() {
                let hsh = (c.wrapping_mul(0x9E3779B97F4A7C15) ^ (i as u64).wrapping_mul(0xC2B2AE3D27D4EB4F)).wrapping_mul(0x165667B19E3779F9);
                *x *= if (hsh >> 40) & 1 == 0 { 1.0 + j } else { 1.0 - j };
            }
        }
    }
    crate::finite::state_raw(data); // observation only (RWKV_FINITE_CHECK)
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
            if let Some((uu, vv, sg)) = match k {
                16 => rank1_static::<16>(&data[off..off + k * k], scale),
                32 => rank1_static::<32>(&data[off..off + k * k], scale),
                8 => rank1_static::<8>(&data[off..off + k * k], scale),
                _ => None,
            } {
                if sg > 1e-20 && sg.is_finite() && uu.iter().all(|x| x.is_finite()) {
                    let sj = sg.sqrt();
                    for i in 0..k {
                        uf[(i, 0)] = uu[i] * sj;
                        vf[(i, 0)] = (vv[i] / sg) * sj;
                    }
                }
            } else {
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
            }
        } else {
            // RANK >= 2 (or ALS): reference path = symmetric eigendecomposition of the Gram G = an an^T
            // (KxK PSD; eigenvectors of G are left singular vectors, eigenvalues are sigma_n^2, and
            // v = A^T u / sigma). FAST path (RWKV_LOWRANK_FAST=1) = warm-started block subspace iteration
            // + Rayleigh-Ritz, see `rankr_fast`. RWKV_LOWRANK_FAST_CHECK=1 runs both and records
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
        crate::finite::state_quant(&data[off..off + k * k]); // observation only
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
    /// Input width, derived from `features2card.0.weight` at load: 92 for the published layout,
    /// 114 under `RWKV_ID_FEATURES=1`. Was a hardcoded const until 2026-09-02.
    feature_dim: usize,
    /// GAP 8. `stream_slot[m]` = which CANONICAL entity slot module `m` implements, where the
    /// canonical order is the fixed `[card, deck, note, preset, user]` every caller uses to key
    /// its per-entity state maps. Identity for the historical arch; `[0,2,1,3,4]` for the iter-41
    /// champion, whose execution order is card->NOTE->DECK->preset->user.
    ///
    /// Callers pass and receive states in CANONICAL order and need to know nothing about the
    /// arch's execution order -- the permutation lives here, once. Before this existed the
    /// engine fed DECK's state into the note module (and vice versa) on any reordered arch: every
    /// shape matched, so nothing failed, it just computed a different model. The Python deploy
    /// mirror routes states by NAME for exactly this reason (the positional body it replaced had
    /// the same bug); this is the Rust half of that fix.
    stream_slot: [usize; 5],
    /// Architecture features recovered from the weight shapes (stripped sublayers, curve-head
    /// type, ahead residual, PAVA powers). See [`ArchFlags`] / `detect_arch`.
    arch: ArchFlags,
    /// (1,num_curves) fixed forgetting-curve time constants. `None` under a GRU head, which
    /// predicts S per review instead of reading a grid.
    s_space: Option<Tensor>,
    point_space: Vec<f32>, // (num_points) interp grid; EMPTY when the ahead residual is stripped
    // Per-stream STATE quant: module_idx -> qmax (127=int8, 7=int4). Empty = fp32 everywhere.
    // Allows MIXED bits across streams (e.g. card int4 + note int8). See load() for env parsing.
    state_quant_qmax: std::collections::HashMap<usize, f64>,
    // Per-stream LOW-RANK card-state truncation: module_idx -> (rank, optional factor qmax). When set
    // for a module, the per-step WKV state is replaced by its rank-r SVD truncation (and factors
    // optionally quantized) INSTEAD of full-matrix quant -- the 0.15 KB card path (step 4). See load().
    state_lowrank: std::collections::HashMap<usize, (usize, Option<f64>)>,
    // FIXED-BASIS (codec) state coding (approach 1): module_idx -> (m, coefficient qmax). Applied
    // INSTEAD of the rank-r round-trip for that stream, layer by layer. See CompressCfg::fb.
    state_fixedbasis: std::collections::HashMap<usize, (usize, Option<f64>)>,
    // The fixed K x stride basis pairs, one per (module_idx, layer). Arc-shared, model-side.
    fb_basis: std::sync::Arc<std::collections::HashMap<(usize, usize), (Vec<f32>, Vec<f32>)>>,
    // Token-shift FIXED-BASIS codec: (module, layer, role) -> mu/P, plus the mode count. Empty =
    // inert (the shifts keep the int-N round-trip). See CompressCfg::shift_fb.
    shift_fb: std::sync::Arc<std::collections::HashMap<(usize, usize, u8), ShiftBasis>>,
    shift_fb_m: usize,
    // RWKV_QUANT_SHIFTS=1: also quantize the (1-D, non-low-rankable) token-shift vectors of any
    // COMPRESSED stream at its bit-width, so the deploy SIZE accounting is honest (shifts otherwise
    // stay fp32, which alone blows the 0.15 KB card budget). Off by default -> past numbers reproduce.
    quant_shifts: bool,
    // RWKV_STATE_SHIFT_LEVEL=intN: override token-shift bit-width independently of the WKV factor level
    // (only for already-compressed streams). None = shifts follow the WKV level. See CompressCfg doc.
    // RWKV_STATE_SHIFT_LEVEL, per stream: module_idx -> shift qmax. A BARE level in the env fills all
    // five streams; a "card:int4,note:int8" list fills only the named ones. See CompressCfg::shift_level.
    shift_level: std::collections::HashMap<usize, f64>,
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
        let mut w = candle_core::safetensors::load(path, &dev)?;
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
        // Which architecture is this checkpoint? Recovered from weight shapes -- see detect_arch.
        // The recipe facts the export wrote into the safetensors metadata (stream order, layer
        // schedule, clamp threshold); empty for files exported before 2026-09-12.
        let meta = safetensors_metadata(path)?;
        let arch = detect_arch(&w, &stream_layers, &meta)?;
        eprintln!("[rwkv-infer] {}", arch.describe());

        // RWKV_ZERO_FEATURES: the input-feature mask the Python model applies at its own input
        // (srs_model.py:314, srs_model_rnn.py:50). It had NO Rust counterpart, which is a
        // three-way-parity break: the mask lives inside the Python module, so the exported trace
        // carries RAW features and this engine consumed the columns Python had thrown away. It
        // hid because the parity gate scores MEAN LogLoss and the champion's masked dim (22,
        // scaled_state) has a small input weight -- but it was costing max per-review
        // |rust-python| 1.59e-3, and with the mask applied that drops to 2.28e-6 (~700x).
        // NOTE this also corrects the record: that per-review spread was written up as
        // "accumulated float divergence over a ~5,000-step recurrence, not a formula error". It
        // was a formula error.
        //
        // Zeroing input column j is exactly equivalent to zeroing feature j (y = Wx+b is linear in
        // x), so it is done ONCE here on the weight rather than per-call on the features: free at
        // runtime, and impossible to forget at one of the three call sites (model.rs x2, fast.rs).
        // Input width, DERIVED rather than assumed -- 92 for the published layout, 114 under
        // RWKV_ID_FEATURES=1. Read once here and stored on the model so every consumer agrees.
        let feature_dim = get(&w, "features2card.0.weight")?.dims2()?.1;

        if let Ok(spec) = std::env::var("RWKV_ZERO_FEATURES") {
            let dims: Vec<usize> = spec
                .split(',')
                .filter_map(|t| t.trim().parse::<usize>().ok())
                .collect();
            if !dims.is_empty() {
                let key = "features2card.0.weight";
                let wt = get(&w, key)?;
                let (out, n_in) = wt.dims2()?;
                let mut buf = wt.to_vec2::<f32>()?;
                for &d in &dims {
                    if d >= n_in {
                        return Err(anyhow!("RWKV_ZERO_FEATURES dim {d} >= input width {n_in}"));
                    }
                    for row in buf.iter_mut().take(out) {
                        row[d] = 0.0;
                    }
                }
                let flat: Vec<f32> = buf.into_iter().flatten().collect();
                w.insert(key.to_string(), Tensor::from_vec(flat, (out, n_in), &dev)?);
                eprintln!("[rwkv-infer] zeroed input features {dims:?} (RWKV_ZERO_FEATURES)");
            }
        }

        // Curve-head width. Under the GRU head the stabilities S and decays d are PREDICTED per
        // review, so there is no fixed basis grid at all: s_space stays None and any attempt to
        // use the classic mixture errors instead of quietly evaluating a grid the model never saw.
        // (num_curves must NOT come from w_linear here -- under a GRU head that is a (1,1) dummy
        // and would silently make the mixture width 1. detect_arch has already rejected the
        // ambiguous cases, so this match is exhaustive and safe.)
        let (num_curves, s_space) = match arch.gru_curves {
            Some(n) => (n, None),
            None => {
                // forgetting_curve s_space: length = num_curves, DERIVED from w_linear
                // out-features (so the engine auto-adapts to head-width changes, e.g. 128->64).
                let n = get(&w, "w_linear.weight")?.dim(0)?;
                let s_max = 22.0f32;
                let s_spread = 18.5f32;
                let s_scale = (s_max - s_spread).exp();
                let grid: Vec<f32> = (0..n)
                    .map(|i| {
                        let l = 18.5f32 * i as f32 / (n as f32 - 1.0);
                        0.1 + (l.exp() - 1.0) * s_scale
                    })
                    .collect();
                (n, Some(grid))
            }
        };
        // interp point_space: length = num_points, grid identical formula, different consts.
        // Empty when the ahead residual is stripped -- the curve is then the pure mixture and
        // nothing may interpolate. `interp` rejects an empty grid rather than degenerating.
        let (num_points, point_space) = if arch.ahead_residual {
            let n = get(&w, "ahead_linear.weight")?.dim(0)?;
            let max_e = 21.0f32;
            let p_spread = 18.5f32;
            let p_scale = (max_e - p_spread).exp();
            let grid: Vec<f32> = (0..n)
                .map(|i| {
                    let l = 18.5f32 * i as f32 / (n as f32 - 1.0);
                    0.5 + (l.exp() - 1.0) * p_scale
                })
                .collect();
            (n, grid)
        } else {
            (0usize, Vec::new())
        };
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
        // ⚠ STREAM NAME -> MODULE INDEX. This MUST follow the checkpoint's own execution order, and
        // since iter 41 the champion's is card->NOTE->DECK->preset->user (arch
        // architecture_d80_lora4_cnd.py), not the historical card->deck->note->preset->user. The
        // safetensors name modules positionally (`rwkv_modules.{N}`), so the order is not
        // recoverable from the checkpoint -- it is a recipe fact and comes from the environment,
        // like state_clamp_tau. Default = the historical order, so every pre-iter-41 invocation is
        // unchanged.
        // WHY THIS MATTERS: with the old hardcoded map, `card:1:int4,note:1:int4` on a _cnd
        // checkpoint quantizes modules 0 and 2 = card and DECK. Wrong stream, and deck state is
        // 5,760 floats vs note's 1,440 -- so both the accuracy effect and the deploy size
        // accounting would be wrong while every shape check still passed. Same class of bug as
        // model_stats.py's positional deck/note labels (fixed 2026-08-09).
        //
        // ★ 2026-09-12: THE ORDER NOW TRAVELS WITH THE WEIGHTS. Relying on the environment alone
        // failed exactly as predicted: two PQ-catalog corpus dumps (2026-09-08 / 09-10) ran a `_cnd`
        // checkpoint without RWKV_STREAM_ORDER, i.e. mis-wired (the note corpus was another stream,
        // card states 15.5% off), and a parity run without it misses by ~0.5 per review.
        // export_rnn_trace.py now writes `stream_order` into the safetensors METADATA from the
        // exported model's own stream names. Precedence:
        //   env + metadata -> they must AGREE, else refuse (a wrong recipe must not run silently);
        //   metadata only  -> use it;
        //   env only       -> use it (every file exported before 2026-09-12: unchanged behaviour);
        //   neither        -> the historical default, with a loud warning on stderr.
        // Every new message goes to STDERR: stdout is captured into corpus files, whose first line
        // scratchpad/lfqat/check_corpus_order.py reads.
        let split_order = |s: &str| -> Vec<String> {
            s.split(',')
                .map(|t| t.trim().to_string())
                .filter(|t| !t.is_empty())
                .collect()
        };
        let meta_order = meta.get("stream_order").cloned();
        let stream_order: Vec<String> = match (std::env::var("RWKV_STREAM_ORDER").ok(), meta_order) {
            (Some(env), Some(meta)) => {
                let (e, m) = (split_order(&env), split_order(&meta));
                if e != m {
                    return Err(anyhow!(
                        "RWKV_STREAM_ORDER={env:?} disagrees with the stream order the weights were \
                         exported with ({meta:?}, safetensors metadata of {path}). Refusing: one of \
                         the two would run a mis-wired model. Unset the variable to use the weights' order."
                    ));
                }
                e
            }
            (Some(env), None) => split_order(&env),
            (None, Some(meta)) => {
                eprintln!("[rwkv-infer] stream order from the weights' metadata: {meta}");
                split_order(&meta)
            }
            (None, None) => {
                eprintln!(
                    "[rwkv-infer] WARNING: neither RWKV_STREAM_ORDER nor the weights' metadata \
                     names the stream order; assuming the HISTORICAL card,deck,note,preset,user. \
                     A card,note,deck (`_cnd`) checkpoint runs MIS-WIRED under this default."
                );
                split_order("card,deck,note,preset,user")
            }
        };
        assert_eq!(
            stream_order.len(),
            5,
            "RWKV_STREAM_ORDER must name exactly 5 streams, got {stream_order:?}"
        );
        let name_to_idx = |n: &str| -> usize {
            stream_order
                .iter()
                .position(|s| s == n)
                .unwrap_or_else(|| panic!("unknown stream {n:?}; RWKV_STREAM_ORDER={stream_order:?}"))
        };
        // GAP 8 (see the field doc): module m implements canonical entity stream_slot[m].
        const CANON: [&str; 5] = ["card", "deck", "note", "preset", "user"];
        let mut stream_slot = [0usize; 5];
        for (m, name) in stream_order.iter().enumerate() {
            stream_slot[m] = CANON
                .iter()
                .position(|c| c == name)
                .unwrap_or_else(|| panic!("RWKV_STREAM_ORDER names unknown stream {name:?}"));
        }
        {
            let mut seen = stream_slot.to_vec();
            seen.sort_unstable();
            assert_eq!(
                seen,
                vec![0, 1, 2, 3, 4],
                "RWKV_STREAM_ORDER must be a permutation of {CANON:?}, got {stream_order:?}"
            );
        }
        if stream_slot != [0, 1, 2, 3, 4] {
            println!(
                "[rwkv-infer] stream order {stream_order:?} -> module->canonical slot map {stream_slot:?}"
            );
        }
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
        // FIXED-BASIS (codec) state coding (approach 1, 2026-11-22):
        //   RWKV_STATE_FIXEDBASIS_SCOPE = "card:8:int8,note:8:int8" (name:m[:level]; m = coefficient
        //     block size, level = its bit-width) + RWKV_FIXEDBASIS_PATH = the fitted-basis text file.
        //   File: line 1 = entry count, then per entry
        //     <name> <layer> <stride> <K*stride floats P row-major> <K*stride floats Q row-major>
        //   Only the first m <= stride columns are read, so ONE fitted file serves every m.
        let mut state_fixedbasis: std::collections::HashMap<usize, (usize, Option<f64>)> =
            std::collections::HashMap::new();
        let fb_scope = std::env::var("RWKV_STATE_FIXEDBASIS_SCOPE").unwrap_or_default();
        if !fb_scope.trim().is_empty() {
            for entry in fb_scope.split(',') {
                let parts: Vec<&str> = entry.trim().split(':').collect();
                let msz: usize = parts
                    .get(1)
                    .expect("RWKV_STATE_FIXEDBASIS_SCOPE entry needs name:m[:level]")
                    .parse()
                    .expect("bad m in RWKV_STATE_FIXEDBASIS_SCOPE");
                let lvl = parts
                    .get(2)
                    .map(|l| parse_level(l).expect("bad level in FIXEDBASIS scope (int4/int8)"));
                assert!(msz >= 1 && msz <= k, "RWKV_STATE_FIXEDBASIS_SCOPE: need 1 <= m <= K");
                state_fixedbasis.insert(name_to_idx(parts[0]), (msz, lvl));
            }
        }
        let mut fb_basis: std::collections::HashMap<(usize, usize), (Vec<f32>, Vec<f32>)> =
            std::collections::HashMap::new();
        if !state_fixedbasis.is_empty() {
            let path = std::env::var("RWKV_FIXEDBASIS_PATH").expect(
                "RWKV_STATE_FIXEDBASIS_SCOPE is set but RWKV_FIXEDBASIS_PATH is not; refusing to silently measure uncompressed state",
            );
            let txt = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("failed to read RWKV_FIXEDBASIS_PATH '{path}': {e}"));
            let mut it = txt.split_whitespace();
            let n: usize = it
                .next()
                .and_then(|t| t.parse().ok())
                .unwrap_or_else(|| panic!("RWKV_FIXEDBASIS_PATH '{path}': line 1 must be the entry count"));
            for e in 0..n {
                let name = it
                    .next()
                    .unwrap_or_else(|| panic!("FIXEDBASIS '{path}': entry {e}: missing name"));
                let layer: usize = it
                    .next()
                    .and_then(|t| t.parse().ok())
                    .unwrap_or_else(|| panic!("FIXEDBASIS '{path}': entry {e}: bad layer"));
                let stride: usize = it
                    .next()
                    .and_then(|t| t.parse().ok())
                    .unwrap_or_else(|| panic!("FIXEDBASIS '{path}': entry {e}: bad stride"));
                let take = |it: &mut std::str::SplitWhitespace| -> Vec<f32> {
                    (0..k * stride)
                        .map(|_| {
                            it.next()
                                .and_then(|t| t.parse().ok())
                                .unwrap_or_else(|| panic!("FIXEDBASIS '{path}': entry {e}: short basis"))
                        })
                        .collect()
                };
                let p = take(&mut it);
                let q = take(&mut it);
                fb_basis.insert((name_to_idx(name), layer), (p, q));
            }
            for (&(mi, l), (p, q)) in fb_basis.iter() {
                assert_eq!(p.len(), q.len(), "FIXEDBASIS '{path}': P/Q length mismatch at {mi}/L{l}");
                if let Some(&(mm, _)) = state_fixedbasis.get(&mi) {
                    let stride = p.len() / k;
                    assert!(
                        mm <= stride,
                        "FIXEDBASIS '{path}': stream {mi} layer {l} needs m={mm} but its basis has {stride} columns"
                    );
                }
            }
            let rows: Vec<String> = {
                let mut r: Vec<String> = state_fixedbasis
                    .iter()
                    .map(|(&mi, &(mm, fq))| format!("module {mi}: m={mm} qmax={fq:?}"))
                    .collect();
                r.sort();
                r
            };
            eprintln!(
                "[fixedbasis] scope [{}] with {} basis entries from {path}",
                rows.join(", "),
                fb_basis.len()
            );
        }
        let fb_basis = std::sync::Arc::new(fb_basis);
        let mut shift_fb: std::collections::HashMap<(usize, usize, u8), ShiftBasis> =
            std::collections::HashMap::new();
        let shift_fb_m: usize = std::env::var("RWKV_SHIFT_FIXEDBASIS_MODES")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(8);
        let shift_fb_path = std::env::var("RWKV_SHIFT_FIXEDBASIS_PATH").unwrap_or_default();
        // RWKV_SHIFT_FIXEDBASIS_SCOPE="note": only these streams get the codec; every other stream keeps
        // the int-N shift round-trip (a basis fitted on one stream is 10-20x worse on another).
        let shift_fb_scope: Vec<String> = std::env::var("RWKV_SHIFT_FIXEDBASIS_SCOPE")
            .unwrap_or_default()
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        if !shift_fb_path.is_empty() {
            let txt = std::fs::read_to_string(&shift_fb_path).unwrap_or_else(|e| {
                panic!("failed to read RWKV_SHIFT_FIXEDBASIS_PATH '{shift_fb_path}': {e}")
            });
            let mut it = txt.split_whitespace();
            let n: usize = it.next().and_then(|t| t.parse().ok()).unwrap_or_else(|| {
                panic!("RWKV_SHIFT_FIXEDBASIS_PATH '{shift_fb_path}': line 1 must be the entry count")
            });
            let mut raw: std::collections::HashMap<String, (usize, Vec<f32>, Vec<f32>)> =
                std::collections::HashMap::new();
            for e in 0..n {
                let name = it
                    .next()
                    .unwrap_or_else(|| panic!("SHIFT-FB '{shift_fb_path}': entry {e}: missing key"))
                    .to_string();
                let stride: usize = it.next().and_then(|t| t.parse().ok()).unwrap_or_else(|| {
                    panic!("SHIFT-FB '{shift_fb_path}': entry {e}: bad stride")
                });
                let mut take = |cnt: usize| -> Vec<f32> {
                    (0..cnt)
                        .map(|_| {
                            it.next().and_then(|t| t.parse::<f32>().ok()).unwrap_or_else(|| {
                                panic!("SHIFT-FB '{shift_fb_path}': entry {e}: short basis (need {cnt} floats)")
                            })
                        })
                        .collect()
                };
                let mu = take(c);
                let p = take(c * stride);
                raw.insert(name, (stride, mu, p));
            }
            // Resolve the TIERS once: (module, layer, role) -> the most specific key present.
            for (mi, sname) in stream_order.iter().enumerate() {
                if !shift_fb_scope.is_empty() && !shift_fb_scope.iter().any(|s| s == sname) {
                    continue;
                }
                for l in 0..stream_layers[mi] {
                    for role in 0..2u8 {
                        let cands = [
                            format!("{sname}__r{role}__L{l}"),
                            format!("{sname}__r{role}"),
                            format!("r{role}"),
                        ];
                        let hit = cands.iter().find_map(|cand| raw.get(cand));
                        if let Some((stride, mu, p)) = hit {
                            assert!(
                                shift_fb_m <= *stride,
                                "RWKV_SHIFT_FIXEDBASIS_PATH '{shift_fb_path}': {sname}/L{l}/r{role} needs m={shift_fb_m} but its basis has {stride} columns"
                            );
                            shift_fb.insert((mi, l, role), ShiftBasis::new(*stride, mu.clone(), p.clone()));
                        }
                    }
                }
            }
            let tiers = raw.len();
            eprintln!(
                "[shift-fixedbasis] m={shift_fb_m}, {} (module,layer,role) keys resolved from {tiers} tiers ({shift_fb_path})",
                shift_fb.len()
            );
        }
        let shift_fb = std::sync::Arc::new(shift_fb);
        let quant_shifts = std::env::var("RWKV_QUANT_SHIFTS")
            .map(|v| v == "1" || v == "true")
            .unwrap_or(false);
        // RWKV_STATE_SHIFT_LEVEL=intN: quantize token-shifts at intN instead of the WKV factor level.
        // BARE level ("int4") = every compressed stream; PER-STREAM list ("card:int4,note:int8") =
        // only the named streams, the rest follow their WKV factor level. Same shape as the quant scope.
        let mut shift_level: std::collections::HashMap<usize, f64> = std::collections::HashMap::new();
        for entry in std::env::var("RWKV_STATE_SHIFT_LEVEL").unwrap_or_default().split(',') {
            let entry = entry.trim();
            if entry.is_empty() {
                continue;
            }
            match entry.split_once(':') {
                Some((n, lvl)) => {
                    if let Some(q) = parse_level(lvl) {
                        shift_level.insert(name_to_idx(n), q);
                    }
                }
                None => {
                    if let Some(q) = parse_level(entry) {
                        for i in 0..5 {
                            shift_level.insert(i, q);
                        }
                    }
                }
            }
        }
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
                let (lo, hi) = norm_range_env("RWKV_PQ_NORM_RANGE_WKV", -3.0, 0.0);
                cb.set_norm_quant(b, lo, hi);
            }
            std::sync::Arc::new(cb)
        });
        let ef_pq = std::env::var("RWKV_EF_PQ").map(|v| v == "1" || v == "true").unwrap_or(false);
        // Product quantization of the token-shift vectors (2 roles: t_xshift, c_xshift). Fast-path only.
        let shift_pq = std::env::var("RWKV_SHIFT_PQ").ok().filter(|s| !s.is_empty()).map(|path| {
            let mut cb = PqCodebook::load_roles(&path, 2)
                .unwrap_or_else(|e| panic!("failed to load RWKV_SHIFT_PQ '{path}': {e}"));
            if let Some(b) = norm_bits {
                let (lo, hi) = norm_range_env("RWKV_PQ_NORM_RANGE_SHIFT", 2.2, 2.9);
                cb.set_norm_quant(b, lo, hi);
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
        // The GRU head's three linears are ROOT-level Parameters on the Python side, named
        // `gru_w_weight` / `gru_w_bias` rather than a `gru_w` submodule -- a TorchScript
        // constraint (CLAUDE.md sec 9: scripted hook bodies may not call submodules, so these
        // are bare Parameters + F.linear). That underscore means they miss the `.weight` test
        // above and would never reach lin_wt. Alias them into the dotted convention so the
        // ordinary lin()/linear() helpers find them, in BOTH engines (fast.rs builds its maps
        // from these same two, below).
        let mut bias_aliases = Vec::new();
        for base in ["gru_w", "gru_s", "gru_d"] {
            if let Some(t) = w.get(&format!("{base}_weight")) {
                lin_wt.insert(format!("{base}.weight"), t.t()?.contiguous()?);
            }
            if let Some(t) = w.get(&format!("{base}_bias")) {
                bias_aliases.push((format!("{base}.bias"), t.clone()));
            }
        }
        for (k, t) in bias_aliases {
            w.insert(k, t);
        }
        let s_space_t = match &s_space {
            Some(g) => Some(Tensor::from_vec(g.clone(), (1, num_curves), &dev)?),
            None => None,
        };
        // Build the plain-Rust fast forward from the same f32 weight maps + derived dims.
        let compress_cfg = CompressCfg {
            lowrank: state_lowrank.clone(),
            quant_qmax: state_quant_qmax.clone(),
            quant_shifts,
            shift_level: shift_level.clone(),
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
            fb: state_fixedbasis.clone(),
            fb_basis: fb_basis.clone(),
            shift_fb: shift_fb.clone(),
            shift_fb_m,
        };
        let fast = crate::fast::FastModel::build(
            &w, &lin_wt, c, h, k, stream_layers.clone(), stream_slot, num_curves, num_points,
            s_space.clone().unwrap_or_default(), point_space.clone(), arch.clone(), compress_cfg,
        )?;
        Ok(Self {
            w,
            lin_wt,
            dev,
            h,
            k,
            c,
            stream_layers,
            feature_dim,
            stream_slot,
            arch,
            s_space: s_space_t,
            point_space,
            state_quant_qmax,
            state_lowrank,
            quant_shifts,
            shift_level,
            lowrank_percol,
            lowrank_hadamard,
            lowrank_4level,
            lowrank_mixed53,
            lowrank_compand,
            lowrank_vqmax,
            lowrank_als,
            state_fixedbasis,
            fb_basis,
            shift_fb,
            shift_fb_m,
            fast,
        })
    }

    /// (H heads, K head-dim, C d_model) derived from the weights.
    pub fn dims(&self) -> (usize, usize, usize) {
        (self.h, self.k, self.c)
    }

    /// Input width derived from the weights: 92 for the published layout, 114 under the
    /// `RWKV_ID_FEATURES` timestamp features. Callers that build feature rows must use this
    /// rather than assuming a width.
    pub fn feature_dim(&self) -> usize {
        self.feature_dim
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
    /// Per-(batch,head) Frobenius clamp on the carried WKV state:
    /// `S *= tau / max(tau, ||S||_F)`, with non-finite heads zeroed. Mirrors
    /// `RWKV7RNNTimeMixer.clamp_state` in rwkv_rnn_model.py.
    ///
    /// ⚠ Deliberately NOT bit-identical to training. Training clamps between windows of
    /// `state_clamp_window` steps, but a one-step-at-a-time engine cannot reconstruct those
    /// boundaries -- they depend on how `prepare_batch` packed the batch -- so it clamps every
    /// step. The two agree EXACTLY wherever ||S|| <= tau (the factor is then exactly 1.0 and
    /// this returns the input untouched) and differ only on already-diverging states, where
    /// per-step is the more conservative choice. See TRACK2_PORT_PLAN.md.
    ///
    /// The early return is what makes that guarantee real: in the overwhelmingly common case
    /// nothing is allocated and no float is touched, so parity cannot drift through here.
    fn clamp_state(&self, s: &Tensor) -> Result<Tensor> {
        let tau = match self.arch.state_clamp_tau {
            Some(t) => t as f64,
            None => return Ok(s.clone()),
        };
        let dims = s.dims().to_vec();
        let nd = dims.len();
        let per = dims[nd - 1] * dims[nd - 2]; // K*K
        let norms: Vec<f32> = s
            .flatten_from(nd - 2)?
            .sqr()?
            .sum(D::Minus1)?
            .sqrt()?
            .flatten_all()?
            .to_vec1()?;
        if norms.iter().all(|n| n.is_finite() && (*n as f64) <= tau) {
            return Ok(s.clone()); // inert -- bit-exact
        }
        let mut v: Vec<f32> = s.flatten_all()?.to_vec1()?;
        for (i, &n) in norms.iter().enumerate() {
            let sl = &mut v[i * per..(i + 1) * per];
            if !n.is_finite() {
                sl.iter_mut().for_each(|x| *x = 0.0);
            } else if (n as f64) > tau {
                let f = (tau / n as f64) as f32;
                sl.iter_mut().for_each(|x| *x *= f);
            }
        }
        Ok(Tensor::from_vec(v, dims, s.device())?)
    }

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
            let ls = state.map(|s| &s[l]);
            let (xo, v0_out, layer_state) = self.run_layer(module_idx, l, &x, v0.as_ref(), ls)?;
            x = xo;
            v0 = Some(v0_out);
            new_state.push(layer_state);
        }
        Ok((x, new_state))
    }

    /// ONE layer of one stream: the body `run_stream` used to inline, extracted so the interleaved
    /// schedule (`RWKV_INTERLEAVE`, iter 41) can visit layers round-by-round across streams.
    ///
    /// `v0` is the caller's STREAM-LOCAL value-residual: `None` at that stream's layer 0 (which
    /// *produces* it) and `Some` thereafter. Interleaving does not thread v0 across streams --
    /// getting that wrong is invisible in a shape check and yields plausible-looking numbers.
    fn run_layer(
        &self,
        module_idx: usize,
        l: usize,
        input: &Tensor,
        v0: Option<&Tensor>,
        ls: Option<&LayerState>,
    ) -> Result<(Tensor, Tensor, LayerState)> {
        let x = input;
        {
            let tp = format!("rwkv_modules.{module_idx}.blocks.{l}.time_mixer");
            let cp = format!("rwkv_modules.{module_idx}.blocks.{l}.channel_mixer");
            let t_st = ls.map(|s| (&s.t_xshift, &s.t_state));
            let (xt, v0_out, t_xshift, t_state) = self.time_mixer(&tp, l, x, v0, t_st)?;
            // Bound the carried state before anything else sees it -- the clamp is part of the
            // recurrence in training, not a storage step, so it goes ahead of the compression
            // round-trip below.
            let t_state = self.clamp_state(&t_state)?;
            // Simulate per-card STATE storage by round-tripping the recurrent WKV matrix each step
            // (worst-case accumulation == the deploy per-persist model). t_xshift/c_xshift are tiny ->
            // left fp32. Low-rank (the 0.15 KB path) takes precedence over full-matrix quant per stream.
            // FIXED-BASIS (codec) state coding takes precedence over the rank-r path for a stream that
            // has a basis for THIS layer (RWKV_STATE_FIXEDBASIS_SCOPE + RWKV_FIXEDBASIS_PATH).
            let fb_here = self.state_fixedbasis.get(&module_idx).and_then(|&(mm, fq)| {
                self.fb_basis.get(&(module_idx, l)).map(|pq| (mm, fq, pq))
            });
            let t_state = if let Some((mm, fq, pq)) = fb_here {
                fixedbasis_roundtrip_tensor(&t_state, mm, &pq.0, &pq.1, fq)?
            } else if let Some(&(rank, fqmax)) = self.state_lowrank.get(&module_idx) {
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
                    .state_fixedbasis
                    .get(&module_idx)
                    .and_then(|&(_, fq)| fq)
                    .or_else(|| self.state_lowrank.get(&module_idx).and_then(|&(_, fq)| fq))
                    .or_else(|| self.state_quant_qmax.get(&module_idx).copied());
                // Override the LEVEL for already-compressed streams; leave uncompressed streams unquantized.
                base.map(|b| self.shift_level.get(&module_idx).copied().unwrap_or(b))
            } else {
                None
            };
            let t_xshift = match shift_qmax {
                Some(q) => match self.shift_fb.get(&(module_idx, l, 0)) {
                    Some(fb) => shift_fb_roundtrip_tensor(&t_xshift, fb, self.shift_fb_m, q)?,
                    None => quant_roundtrip(&t_xshift, q)?,
                },
                None => t_xshift,
            };
            let c_st = ls.map(|s| &s.c_xshift);
            // A stripped channel mixer skips the sublayer AND its residual add, and its shift
            // state is never written (mirrors RWKV7RNNLayer.forward: `return x_BC, v0_BC,
            // (time_state, channel_state)`). We must still store SOMETHING in LayerState, so the
            // incoming shift passes through untouched; it is never read again, and a stripped
            // layer should be excluded from deploy state-size accounting.
            let (xc, c_xshift) = if self.arch.has_cmix[module_idx][l] {
                let (xc, cs) = self.channel_mixer(&cp, &xt, c_st)?;
                let cs = match shift_qmax {
                    Some(q) => match self.shift_fb.get(&(module_idx, l, 1)) {
                        Some(fb) => shift_fb_roundtrip_tensor(&cs, fb, self.shift_fb_m, q)?,
                        None => quant_roundtrip(&cs, q)?,
                    },
                    None => cs,
                };
                (xc, cs)
            } else {
                let carried = match c_st {
                    Some(t) => t.clone(),
                    None => Tensor::zeros_like(&xt)?,
                };
                (xt.clone(), carried)
            };
            Ok((
                xc,
                v0_out,
                LayerState {
                    t_xshift,
                    t_state,
                    c_xshift,
                },
            ))
        }
    }

    /// Full forward over all 5 chained streams + heads.
    /// states: [card, deck, note, preset, user] in chain order.
    /// Returns (out_ahead_logits, curve, out_p_logits(1,4), new_states), where out_ahead_logits
    /// is `None` when the piecewise-linear residual is stripped and `curve` carries the
    /// mixture weights (+ per-review S and d under a GRU head).
    pub fn review(
        &self,
        feats: &Tensor,
        states: &[Option<StreamState>; 5],
    ) -> Result<(Option<Tensor>, CurveOut, Tensor, [StreamState; 5])> {
        let dbg = std::env::var("RWKV_DEBUG").is_ok();
        let mut x = self.features2card(feats)?;
        if dbg {
            summ("features2card", &x);
        }
        // GAP 8: `states` arrives in CANONICAL entity order [card, deck, note, preset, user];
        // module m consumes slot self.stream_slot[m] and its output is written back to that same
        // slot, so the returned array is canonical too and callers stay arch-agnostic.
        let slot = self.stream_slot;
        // chain streams -- sequentially, or round-robin under the iter-41 interleaved schedule
        let mut new: Vec<StreamState> = Vec::with_capacity(5);
        if self.arch.interleave {
            // GAP 7. Round r runs layer r of every stream that has one, in module order; the
            // single hidden vector threads through in that order. Worth +0.000489 ahead /
            // +0.000612 imm on its own (iter 42's de-bundle control), so it is not skippable.
            //
            // The parallel trainer has to re-anchor each stream's gather to the canonical row
            // layout every round (srs_model.py::_interleaved_streams) because prepare() builds
            // each stream's indices against the previous stream's output layout. A ONE-REVIEW
            // engine has no such problem: one row, five states held separately.
            //
            // Placement is FRONT-LOADED (stream m runs its layer r in round r, for r < depth_m) --
            // the champion's, and the only one shipped: the endpoint-anchored "spread" variant was
            // iter 44 and was rejected as a tie.
            let mut v0s: [Option<Tensor>; 5] = Default::default();
            let mut per_stream: Vec<StreamState> = vec![Vec::new(); 5];
            for m in 0..5 {
                per_stream[slot[m]].reserve(self.stream_layers[m]);
            }
            for (m, l) in interleave_visits(&self.stream_layers) {
                let ls = states[slot[m]].as_ref().map(|s| &s[l]);
                let (xo, v0_out, layer_state) = self.run_layer(m, l, &x, v0s[m].as_ref(), ls)?;
                x = xo;
                v0s[m] = Some(v0_out); // STREAM-LOCAL: never threaded across streams
                per_stream[slot[m]].push(layer_state);
                if dbg {
                    summ(&format!("ilv m{m}l{l}"), &x);
                }
            }
            new = per_stream;
        } else {
            new = vec![Vec::new(); 5];
            for m in 0..5 {
                let (xo, ns) =
                    self.run_stream(m, self.stream_layers[m], &x, states[slot[m]].as_ref())?;
                x = xo;
                new[slot[m]] = ns;
                if dbg {
                    summ(&format!("stream{m}"), &x);
                }
            }
        }
        let global_encoding = x;

        let xh = self.ln(&global_encoding, "prehead_norm", LN_EPS)?;
        if dbg {
            summ("prehead_norm", &xh);
        }

        // head_w -> curve head (fixed-basis mixture weights, or GRU w/S/d)
        let hw = self.lin(&xh, "head_w.0", true)?.relu()?;
        let hw = self.ln(&hw, "head_w.2", LN_EPS)?;
        let hw = self.lin(&hw, "head_w.4", true)?;
        let out_curve = self.curve_head(&hw)?;

        // head_ahead_logits -> ahead_linear (the piecewise-linear residual). When the residual is
        // stripped the ENTIRE head is dummied -- head_ahead_logits.0 is (1,1) too -- so this is
        // skipped rather than computed and thrown away: running it would be a shape-mismatch
        // matmul against a (1,C) input. Skipping is also the point (dead compute: ~131.7k params
        // of matmul per review at d=128).
        let out_ahead_logits = match self.arch.ahead_residual {
            true => {
                let ha = self.lin(&xh, "head_ahead_logits.0", true)?.relu()?;
                Some(self.lin(&ha, "ahead_linear", true)?)
            }
            false => None,
        };

        // head_p -> p_linear  (4-way)
        let hp = self.lin(&xh, "head_p.0", true)?.relu()?;
        let out_p_logits = self.lin(&hp, "p_linear", true)?;

        if dbg {
            summ("out_p_logits", &out_p_logits);
            summ("out_w", &out_curve.w);
            if let Some(al) = &out_ahead_logits {
                summ("out_ahead_logits", al);
            }
        }
        let new_arr: [StreamState; 5] = new
            .try_into()
            .map_err(|_| anyhow!("stream count mismatch"))?;
        Ok((out_ahead_logits, out_curve, out_p_logits, new_arr))
    }

    /// imm probability = 1 - softmax(out_p_logits)[again]
    /// The four button intervals for the review Anki is about to show.
    ///
    /// Port of `SrsRWKVRnn.button_heads` / `button_curves` / `button_intervals`
    /// (rwkv/model/srs_model_rnn.py) — gap 6 of TRACK2_PORT_PLAN.md. Steps:
    ///
    /// 1. build 4 counterfactual rows = this review with the grade one-hot swapped to
    ///    Again/Hard/Good/Easy and `scaled_duration` ZEROED on all four;
    /// 2. run each read-only against the current states (they are probes: they read state and
    ///    must never advance it — hence `states` is borrowed and the new states are dropped);
    /// 3. rectify the four curves with PAVA at every probe;
    /// 4. solve each rectified curve for `desired_retention`.
    ///
    /// The probe construction lives HERE rather than in the caller on purpose: if a caller built
    /// the four rows itself it could feed them inconsistent inputs, and the button comparison
    /// would silently stop being a comparison. Duration is zeroed because the real press duration
    /// is not known until after the press — using it would make the displayed interval depend on
    /// how long the user stared at the card, and displayed would stop equalling scheduled.
    /// 0.0 is the pipeline's own "no press yet" encoding, the value every query row already
    /// carries (verified on the reference traces: feats_imm has 0.0 at COL_DUR on every row).
    pub fn button_intervals(
        &self,
        feats: &Tensor,
        states: &[Option<StreamState>; 5],
        desired_retention: f32,
    ) -> Result<[f32; 4]> {
        // one forward per button, state discarded -- 4 forwards per press, not per probe
        let mut heads: Vec<(Option<Tensor>, CurveOut, Tensor)> = Vec::with_capacity(4);
        for k in 0..4 {
            let row = self.probe_row(feats, k)?;
            let (ahead, curve, p_logits, _dropped_states) = self.review(&row, states)?;
            heads.push((ahead, curve, p_logits));
        }
        let powers = crate::pava::junction_powers(self.arch.pava_theta.as_ref());
        // Closure is called once per bisection probe; it re-rectifies every time, because pooling
        // couples the buttons (rectifying after solving gives a different answer). Cheap: the
        // heads do not depend on t, so this is closed-form arithmetic on cached head outputs.
        let mut err: Option<anyhow::Error> = None;
        let curves_at = |t: f32| -> [f32; 4] {
            let mut raw = [0f32; 4];
            for k in 0..4 {
                match self.predict_ahead(heads[k].0.as_ref(), &heads[k].1, t) {
                    Ok(p) => raw[k] = p,
                    Err(e) => {
                        if err.is_none() {
                            err = Some(e);
                        }
                        raw[k] = 0.5;
                    }
                }
            }
            crate::pava::pava_rectify(raw, [1.0; 4], powers)
        };
        let out = crate::pava::solve_intervals(
            curves_at,
            desired_retention,
            1.0,
            self.s_max_seconds(),
            50,
        );
        if let Some(e) = err {
            return Err(e);
        }
        Ok(out)
    }

    /// Upper bracket for the interval solver: e^22 s (~113 years), the largest time constant the
    /// curve head can represent. Matches `SrsRWKVRnn.button_intervals`' `math.exp(self.s_max)`.
    fn s_max_seconds(&self) -> f32 {
        22.0f32.exp()
    }

    /// One counterfactual probe row: `scaled_duration` zeroed, grade one-hot set to `button`.
    fn probe_row(&self, feats: &Tensor, button: usize) -> Result<Tensor> {
        let mut v: Vec<f32> = feats.flatten_all()?.to_vec1()?;
        let fd = self.feature_dim;
        if v.len() != fd {
            return Err(anyhow!("probe_row: expected {fd} features, got {}", v.len()));
        }
        v[COL_DUR] = 0.0;
        for k in 0..4 {
            v[COL_R1 + k] = if k == button { 1.0 } else { 0.0 };
        }
        Ok(Tensor::from_vec(v, (1, fd), &self.dev)?)
    }

    pub fn imm_prob(&self, out_p_logits: &Tensor) -> Result<f32> {
        let p = candle_nn::ops::softmax(out_p_logits, D::Minus1)?; // (1,4)
        let again: f32 = p.narrow(1, 0, 1)?.reshape(())?.to_scalar()?;
        Ok(1.0 - again)
    }

    /// Curve-head parameters from the shared trunk feature `hw`.
    ///
    /// Classic: one linear + softmax over a fixed grid of stabilities.
    /// GRU (srs_model.py::gru_forgetting_curve): three tiny linears off the SAME trunk feature --
    /// w (softmaxed), plus raw S and d which stay raw here and are exp(clamp(.))'d at evaluation
    /// time, exactly as the Python does.
    fn curve_head(&self, hw: &Tensor) -> Result<CurveOut> {
        match self.arch.gru_curves {
            None => {
                let logits = self.lin(hw, "w_linear", true)?;
                Ok(CurveOut { w: candle_nn::ops::softmax(&logits, D::Minus1)?, gru: None })
            }
            Some(_) => {
                let w = candle_nn::ops::softmax(&self.lin(hw, "gru_w", true)?, D::Minus1)?;
                let s_raw = self.lin(hw, "gru_s", true)?;
                let d_raw = self.lin(hw, "gru_d", true)?;
                Ok(CurveOut { w, gru: Some((s_raw, d_raw)) })
            }
        }
    }

    /// GRU power-curve recall. Mirrors srs_model_rnn.py::gru_forgetting_curve EXACTLY:
    ///   s = exp(clamp(s_raw,-25,25)); d = exp(clamp(d_raw,-25,25)); t = max(1,elapsed)
    ///   r = sum_i w_i * exp(-d_i * log1p(t / (1e-7 + s_i)))
    /// The log1p form is the Python's, not the algebraically equal (1+t/S)^(-d) -- keep it, so
    /// the float rounding matches too.
    fn gru_forgetting_curve(&self, c: &CurveOut, elapsed: f32) -> Result<f32> {
        let (s_raw, d_raw) = c.gru.as_ref().ok_or_else(|| {
            anyhow!("gru_forgetting_curve called on a classic fixed-basis checkpoint")
        })?;
        let t = elapsed.max(1.0);
        let w: Vec<f32> = c.w.flatten_all()?.to_vec1()?;
        let sr: Vec<f32> = s_raw.flatten_all()?.to_vec1()?;
        let dr: Vec<f32> = d_raw.flatten_all()?.to_vec1()?;
        let mut r = 0f32;
        for i in 0..w.len() {
            let s = sr[i].clamp(-25.0, 25.0).exp();
            let d = dr[i].clamp(-25.0, 25.0).exp();
            r += w[i] * (-d * (t / (1e-7 + s)).ln_1p()).exp();
        }
        Ok(1e-5 + (1.0 - 2e-5) * r)
    }

    /// forgetting_curve(out_w, elapsed_seconds) -> probability scalar.
    fn forgetting_curve(&self, out_w: &Tensor, elapsed: f32) -> Result<f32> {
        let e = elapsed.max(1.0);
        // 0.9^(e/s) = exp(ln(0.9) * e / s)
        let ln09 = 0.9f64.ln() as f32;
        let s_space = self.s_space.as_ref().ok_or_else(|| {
            anyhow!("classic forgetting_curve called on a GRU-head checkpoint (no fixed basis grid)")
        })?;
        let inv_s = s_space.recip()?;
        let pw = inv_s.affine((ln09 * e) as f64, 0.0)?.exp()?; // exp(ln09*e/s)
        let summed = (out_w * pw)?.sum_keepdim(D::Minus1)?; // (1,1)
        let s: f32 = summed.reshape(())?.to_scalar()?;
        Ok(1e-5 + (1.0 - 2e-5) * s)
    }

    /// interp(out_ahead_logits, elapsed) -> logit residual scalar.
    fn interp(&self, out_ahead_logits: &Tensor, elapsed: f32) -> Result<f32> {
        let e = elapsed.max(1.0);
        let ps = &self.point_space;
        if ps.len() < 2 {
            // Empty grid = the ahead residual was stripped. Reaching here means a caller skipped
            // the arch.ahead_residual branch; the old code would have underflowed `ps.len()-1`.
            return Err(anyhow!("interp called with no point_space (ahead residual is stripped)"));
        }
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
        out_ahead_logits: Option<&Tensor>,
        curve: &CurveOut,
        elapsed: f32,
    ) -> Result<f32> {
        let p_raw = match self.arch.gru_curves {
            Some(_) => self.gru_forgetting_curve(curve, elapsed)?,
            None => self.forgetting_curve(&curve.w, elapsed)?,
        };
        // No residual -> the curve is the pure mixture, monotone in t by construction. NOT a
        // plain early return: Python still ADDS 1e-5 in logit space, because its zeroed logits
        // go through interp's `1e-5 + (1 - 2e-5) * 0` (srs_model_rnn.ahead_residual documents
        // this, and run_as_rnn.predict_func calls interp unconditionally). Worth ~2.5e-6 in
        // probability -- far inside tolerance, but parity means parity.
        let residual = match (self.arch.ahead_residual, out_ahead_logits) {
            (false, _) => 1e-5f32,
            (true, Some(al)) => self.interp(al, elapsed)?,
            (true, None) => {
                return Err(anyhow!("ahead residual is live but no ahead logits were stored"))
            }
        };
        let logit_raw = (p_raw / (1.0 - p_raw)).ln();
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
            let ls = state.map(|s| &s[l]);
            let (xo, v0_out, layer_state) =
                self.run_layer_batched(module_idx, l, &x, v0.as_ref(), ls)?;
            x = xo;
            v0 = Some(v0_out);
            new_state.push(layer_state);
        }
        Ok((x, new_state))
    }

    /// ONE batched layer -- the body `run_stream_batched` used to inline, extracted so the
    /// interleaved schedule can walk rounds. Mirrors `run_layer`; keep the three in sync
    /// (this one, `run_layer`, and `fast::FastModel::run_layer`).
    fn run_layer_batched(
        &self,
        module_idx: usize,
        l: usize,
        input: &Tensor,
        v0: Option<&Tensor>,
        ls: Option<&BatchedLayerState>,
    ) -> Result<(Tensor, Tensor, BatchedLayerState)> {
        let x = input;
        {
            let tp = format!("rwkv_modules.{module_idx}.blocks.{l}.time_mixer");
            let cp = format!("rwkv_modules.{module_idx}.blocks.{l}.channel_mixer");
            let t_st = ls.map(|s| (&s.t_xshift, &s.t_state));
            let (xt, v0_out, t_xshift, t_state) =
                self.time_mixer_batched(&tp, l, x, v0, t_st)?;
            // clamp_state is shape-generic: it reduces over the trailing (K,K) and iterates the
            // flattened leading dims, so (B,H,K,K) works exactly as (H,K,K) does.
            let t_state = self.clamp_state(&t_state)?;
            let fb_here = self.state_fixedbasis.get(&module_idx).and_then(|&(mm, fq)| {
                self.fb_basis.get(&(module_idx, l)).map(|pq| (mm, fq, pq))
            });
            let t_state = if let Some((mm, fq, pq)) = fb_here {
                fixedbasis_roundtrip_batched(&t_state, mm, &pq.0, &pq.1, fq)?
            } else {
                match self.state_quant_qmax.get(&module_idx) {
                    Some(&qmax) => quant_roundtrip_batched(&t_state, qmax)?,
                    None => t_state,
                }
            };
            let c_st = ls.map(|s| &s.c_xshift);
            // Stripped channel mixer: skip the sublayer and its residual (see run_stream).
            let (xc, c_xshift) = if self.arch.has_cmix[module_idx][l] {
                self.channel_mixer_batched(&cp, &xt, c_st)?
            } else {
                let carried = match c_st {
                    Some(t) => t.clone(),
                    None => Tensor::zeros_like(&xt)?,
                };
                (xt.clone(), carried)
            };
            Ok((xc, v0_out, BatchedLayerState { t_xshift, t_state, c_xshift }))
        }
    }

    /// Batched forward over all 5 chained streams + heads. feats is (B,92); each state is (B,...).
    /// Returns (out_ahead_logits (B,np) or None, curve (B,nc), out_p_logits (B,4), new_states).
    pub fn review_batched(
        &self,
        feats: &Tensor,
        states: &[Option<BatchedStreamState>; 5],
    ) -> Result<(Option<Tensor>, CurveOut, Tensor, [BatchedStreamState; 5])> {
        let mut x = self.features2card(feats)?; // (B,C); features2card is last-dim ops -> batch-fine
        // GAPS 7 + 8, mirroring Model::review -- keep in sync.
        let slot = self.stream_slot;
        let mut new: Vec<BatchedStreamState> = vec![Vec::new(); 5];
        if self.arch.interleave {
            let mut v0s: Vec<Option<Tensor>> = vec![None; 5];
            for m in 0..5 {
                new[slot[m]].reserve(self.stream_layers[m]);
            }
            for (m, l) in interleave_visits(&self.stream_layers) {
                let ls = states[slot[m]].as_ref().map(|s| &s[l]);
                let (xo, v0_out, layer_state) =
                    self.run_layer_batched(m, l, &x, v0s[m].as_ref(), ls)?;
                x = xo;
                v0s[m] = Some(v0_out); // STREAM-LOCAL
                new[slot[m]].push(layer_state);
            }
        } else {
            for m in 0..5 {
                let (xo, ns) =
                    self.run_stream_batched(m, self.stream_layers[m], &x, states[slot[m]].as_ref())?;
                x = xo;
                new[slot[m]] = ns;
            }
        }
        let xh = self.ln(&x, "prehead_norm", LN_EPS)?;

        // Same curve head as the B=1 path -- softmax is last-dim, so it is batch-agnostic.
        let hw = self.lin(&xh, "head_w.0", true)?.relu()?;
        let hw = self.ln(&hw, "head_w.2", LN_EPS)?;
        let hw = self.lin(&hw, "head_w.4", true)?;
        let out_curve = self.curve_head(&hw)?;

        let out_ahead_logits = match self.arch.ahead_residual {
            true => {
                let ha = self.lin(&xh, "head_ahead_logits.0", true)?.relu()?;
                Some(self.lin(&ha, "ahead_linear", true)?)
            }
            false => None,
        };

        let hp = self.lin(&xh, "head_p.0", true)?.relu()?;
        let out_p_logits = self.lin(&hp, "p_linear", true)?;

        let new_arr: [BatchedStreamState; 5] = new
            .try_into()
            .map_err(|_| anyhow!("stream count mismatch"))?;
        Ok((out_ahead_logits, out_curve, out_p_logits, new_arr))
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

#[cfg(test)]
mod schedule_tests {
    use super::interleave_visits;

    /// The champion's depths (card 2, note 1, deck 4, preset 3, user 3) under the `_cnd` order.
    const CHAMP: [usize; 5] = [2, 1, 4, 3, 3];

    #[test]
    fn matches_the_python_schedule_for_the_champion() {
        // Python prints, for these depths:
        //   sched=[[0,1,-1,-1], [0,-1,-1,-1], [0,1,2,3], [0,1,2,-1], [0,1,2,-1]]
        // i.e. 4 rounds, 13 layer-steps, hierarchy order within each round. Transcribed by hand
        // from that banner so the test fails if either side drifts.
        let expect = vec![
            (0, 0), (1, 0), (2, 0), (3, 0), (4, 0), // round 0
            (0, 1), (2, 1), (3, 1), (4, 1),         // round 1: note (depth 1) sits out
            (2, 2), (3, 2), (4, 2),                 // round 2: card done too
            (2, 3),                                 // round 3: deck alone
        ];
        assert_eq!(interleave_visits(&CHAMP), expect);
    }

    #[test]
    fn every_layer_runs_exactly_once() {
        let v = interleave_visits(&CHAMP);
        assert_eq!(v.len(), CHAMP.iter().sum::<usize>(), "13 layer-steps total");
        let mut sorted = v.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), v.len(), "no layer visited twice");
        for (m, &d) in CHAMP.iter().enumerate() {
            for l in 0..d {
                assert!(v.contains(&(m, l)), "module {m} layer {l} never runs");
            }
        }
    }

    #[test]
    fn a_streams_layers_run_in_ascending_order() {
        // v0 correctness depends on this: layer 0 PRODUCES the stream's value residual and every
        // later layer consumes it, so a stream's layer 0 must be its first visit. This is the
        // property that is invisible to a shape check.
        let v = interleave_visits(&CHAMP);
        for m in 0..CHAMP.len() {
            let layers: Vec<usize> = v.iter().filter(|(mm, _)| *mm == m).map(|(_, l)| *l).collect();
            assert!(layers.windows(2).all(|w| w[0] < w[1]), "module {m} layers out of order");
            assert_eq!(layers.first(), Some(&0), "module {m} does not start at layer 0");
        }
    }

    #[test]
    fn uniform_depths_are_a_clean_round_robin() {
        assert_eq!(
            interleave_visits(&[2, 2]),
            vec![(0, 0), (1, 0), (0, 1), (1, 1)]
        );
    }

    #[test]
    fn degenerate_depths_do_not_panic() {
        assert!(interleave_visits(&[]).is_empty());
        assert!(interleave_visits(&[0, 0]).is_empty());
        // A zero-depth stream simply never appears; the others are unaffected.
        assert_eq!(interleave_visits(&[0, 2]), vec![(1, 0), (1, 1)]);
    }
}

/// RWKV_PQ_NORM_RANGE_{WKV,SHIFT}="lo,hi" (2026-09-24): override a fixed norm-quant range (octaves). The defaults
/// were fitted in the d=32 era; unset = the old behaviour. Must equal the QAT run's RWKV_QAT_NORM_RANGE_{WKV,SHIFT}.
fn norm_range_env(name: &str, lo: f32, hi: f32) -> (f32, f32) {
    match std::env::var(name) {
        Ok(v) if !v.is_empty() => {
            let p: Vec<f32> = v.split(',').map(|x| x.trim().parse().expect("bad norm range")).collect();
            assert!(p.len() == 2 && p[1] > p[0], "{name} must be lo,hi with hi > lo");
            eprintln!("[norm range] {name} = [{}, {}] octaves", p[0], p[1]);
            (p[0], p[1])
        }
        _ => (lo, hi),
    }
}

#[cfg(test)]
mod fixedbasis_tests {
    //! The fixed-basis (codec) round-trip must satisfy exactly what the scheme claims: m = K with no
    //! quantization is an EXACT reconstruction (P, Q orthonormal), a state that lives in the basis's
    //! top-m subspace survives m < K, and the coefficient block is what actually gets quantized.
    //! Run: cargo test --release fixedbasis -- --nocapture
    use super::fixedbasis_roundtrip;

    const K: usize = 32;

    /// A deterministic orthonormal K x K pair (QR of an LCG-filled matrix).
    fn orth(seed: u64) -> Vec<f32> {
        let mut s = seed | 1;
        let mut rnd = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 11) as f32 / (1u64 << 53) as f32) * 2.0 - 1.0
        };
        let mut m: Vec<Vec<f32>> = (0..K).map(|_| (0..K).map(|_| rnd()).collect()).collect();
        // Gram-Schmidt: m rows -> orthonormal rows (row-major K x K).
        for i in 0..K {
            for j in 0..i {
                let d: f32 = (0..K).map(|t| m[i][t] * m[j][t]).sum();
                for t in 0..K {
                    let v = m[j][t];
                    m[i][t] -= d * v;
                }
            }
            let n: f32 = (0..K).map(|t| m[i][t] * m[i][t]).sum::<f32>().sqrt().max(1e-9);
            for t in 0..K {
                m[i][t] /= n;
            }
        }
        m.into_iter().flatten().collect()
    }

    fn rel_err(a: &[f32], b: &[f32]) -> f32 {
        let num: f32 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>().sqrt();
        let den: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        num / den
    }

    #[test]
    fn m_equals_k_without_quant_is_exact() {
        let p = orth(11);
        let q = orth(22);
        let mut s = orth(33);
        let orig = s.clone();
        fixedbasis_roundtrip(&mut s, 1, K, K, &p, &q, K, None);
        let e = rel_err(&s, &orig);
        assert!(e < 1e-5, "m=K, qmax=None must be exact, rel={e}");
    }

    #[test]
    fn in_subspace_state_survives_small_m() {
        // A rank-2 state built INSIDE the basis's top-2 subspace: modes along P[0],P[1] and Q[0],Q[1].
        let p = orth(11);
        let q = orth(22);
        let mut s = vec![0f32; K * K];
        for i in 0..K {
            for j in 0..K {
                s[i * K + j] = 3.0 * p[i * K] * q[j * K] + 1.0 * p[i * K + 1] * q[j * K + 1];
            }
        }
        let orig = s.clone();
        fixedbasis_roundtrip(&mut s, 1, K, 2, &p, &q, K, None);
        let e = rel_err(&s, &orig);
        assert!(e < 1e-4, "a state inside the top-2 subspace must survive m=2, rel={e}");
        // m=1 keeps the dominant mode only: 9/(9+1) = 0.9 of the energy, i.e. rel 0.316.
        let mut s1 = orig.clone();
        fixedbasis_roundtrip(&mut s1, 1, K, 1, &p, &q, K, None);
        let e1 = rel_err(&s1, &orig);
        assert!((0.25..0.45).contains(&e1), "m=1 must keep only the dominant mode, rel={e1}");
    }

    #[test]
    fn narrow_stride_is_read_only_up_to_m() {
        // The file stores a stride-8 basis; asking for m=4 must read the first 4 columns only.
        let p8 = orth(11);
        let q8 = orth(22);
        let mut s4 = orth(44);
        let mut s8 = s4.clone();
        fixedbasis_roundtrip(&mut s4, 1, K, 4, &p8, &q8, K, None);
        fixedbasis_roundtrip(&mut s8, 1, K, 8, &p8, &q8, K, None);
        assert!(rel_err(&s4, &s8) > 1e-3, "m=4 and m=8 must differ on a full-rank state");
    }

    #[test]
    fn quantization_hits_the_coefficient_block() {
        // A state that LIVES in the basis's top-8 subspace (decaying modes along the basis's own first
        // 8 directions): m=8 is then EXACT, so whatever error is left is the coefficient quantization
        // alone -- which is what this check is about. A random full-rank state can only measure the
        // subspace projection (~m/K of the energy), not the quantization.
        let p = orth(11);
        let q = orth(22);
        let mut orig = vec![0f32; K * K];
        let mut sigma = 3.0f32;
        for s in 0..8 {
            for i in 0..K {
                for j in 0..K {
                    orig[i * K + j] += sigma * p[i * K + s] * q[j * K + s];
                }
            }
            sigma *= 0.8;
        }
        let mut exact = orig.clone();
        fixedbasis_roundtrip(&mut exact, 1, K, 8, &p, &q, K, None);
        let e0 = rel_err(&exact, &orig);
        assert!(e0 < 1e-5, "m=8 on an in-subspace state must be exact, rel={e0}");
        let mut a = orig.clone();
        let mut c = orig.clone();
        fixedbasis_roundtrip(&mut a, 1, K, 8, &p, &q, K, Some(127.0));
        fixedbasis_roundtrip(&mut c, 1, K, 8, &p, &q, K, Some(1.0));
        let ea = rel_err(&a, &orig);
        let ec = rel_err(&c, &orig);
        assert!(ea < 0.05, "int8 on the coefficient block must be a small precision error, rel={ea}");
        assert!(ec > ea * 2.0, "int2 must be clearly worse than int8 ({ec} vs {ea})");
    }
}

#[cfg(test)]
mod compress_bench {
    //! Ignored benchmark + prototype for the rank>=2 compression cost (run with --ignored --nocapture).
    //!
    //! The deploy engine measures a rank-2 NOTE at 0.600x rev/s and rank-3 at 0.581x, both below the
    //! 0.75x floor, while rank 1 (a power iteration on stack matrices) is 0.90x. The suspicion is that
    //! the rank>=2 branch - which forms the KxK Gram matrix and calls `nalgebra::SymmetricEigen` - is
    //! the whole cost. This module measures that, and prototypes the cheap alternative: subspace
    //! iteration on fixed-size arrays, which never forms the Gram matrix and can be warm-started from
    //! the previous step's basis because the state moves slowly.
    use super::compress_wkv_state;
    use nalgebra as na;
    use std::hint::black_box;
    use std::time::Instant;

    const K: usize = 32;
    const R: usize = 2;

    /// A KxK state with a geometric singular spectrum: sum_i decay^i * k_i v_i^T with |k| = |v| = 1.
    fn synth(decay: f32, terms: usize, seed: u64) -> Vec<f32> {
        let mut out = vec![0.0f32; K * K];
        let mut s = 1.0f32;
        let mut rng = seed | 1;
        let mut rnd = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            ((rng >> 11) as f32 / (1u64 << 53) as f32) * 2.0 - 1.0
        };
        for _ in 0..terms {
            let k: Vec<f32> = (0..K).map(|_| rnd()).collect();
            let v: Vec<f32> = (0..K).map(|_| rnd()).collect();
            let kn = k.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
            let vn = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
            for i in 0..K {
                for j in 0..K {
                    out[i * K + j] += s * (k[i] / kn) * (v[j] / vn);
                }
            }
            s *= decay;
        }
        out
    }

    /// The production exact rank-2 truncation: `compress_wkv_state` with NO quantization at all, so the
    /// buffer it writes back IS A_2 = U_2 V_2^T.
    fn reference(a: &[f32]) -> Vec<f32> {
        let mut buf = a.to_vec();
        compress_wkv_state(&mut buf, 1, K, R, None, false, false, false, false, None, None, None, None,
                           None, None);
        buf
    }

    /// PROTOTYPE: top-2 left singular subspace by subspace iteration on fixed-size arrays. `q` is the
    /// basis (columns = orthonormal K-vectors) and is updated in place. Never forms A A^T.
    fn subit(a: &na::SMatrix<f32, K, K>, q: &mut na::SMatrix<f32, K, R>, iters: usize) {
        for _ in 0..iters {
            let t = a.transpose() * &*q; // KxR
            let mut z = a * t; // A A^T q  (KxR)
            for j in 0..R {
                for _ in 0..2 {
                    // modified Gram-Schmidt with one reorthogonalization pass
                    for l in 0..j {
                        let mut tmp = na::SVector::<f32, K>::zeros();
                        tmp.copy_from(&z.column(l));
                        let d = tmp.dot(&z.column(j));
                        for i in 0..K {
                            z[(i, j)] -= d * tmp[i];
                        }
                    }
                }
                let n = z.column(j).norm();
                if n.is_finite() && n > 1e-30 {
                    for i in 0..K {
                        z[(i, j)] /= n;
                    }
                }
            }
            *q = z;
        }
    }

    /// Factor the prototype's basis exactly as the production path does: sigma from the ORIGINAL A,
    /// then split sqrt(sigma) into both factors.
    fn proto_factors(
        a: &na::SMatrix<f32, K, K>,
        q: &na::SMatrix<f32, K, R>,
    ) -> (na::SMatrix<f32, K, R>, na::SMatrix<f32, K, R>) {
        let mut uf = na::SMatrix::<f32, K, R>::zeros();
        let mut vf = na::SMatrix::<f32, K, R>::zeros();
        for j in 0..R {
            let u = q.column(j).into_owned();
            let v_un = a.transpose() * &u; // = sigma * v
            let sigma = v_un.norm();
            if sigma.is_finite() && sigma > 1e-20 {
                let sj = sigma.sqrt();
                for i in 0..K {
                    uf[(i, j)] = u[i] * sj;
                    vf[(i, j)] = (v_un[i] / sigma) * sj;
                }
            }
        }
        (uf, vf)
    }

    /// Flatten a KxK matrix to the ROW-major layout `compress_wkv_state` works in (nalgebra's
    /// own iterator is column-major, so a direct comparison would compare a matrix with its
    /// transpose).
    fn row_major(m: &na::SMatrix<f32, K, K>) -> Vec<f32> {
        let mut out = Vec::with_capacity(K * K);
        for r in 0..K {
            for c in 0..K {
                out.push(m[(r, c)]);
            }
        }
        out
    }

    fn rel_frob(x: &[f32], y: &[f32]) -> f32 {
        let num: f32 = x.iter().zip(y).map(|(p, q)| (p - q) * (p - q)).sum::<f32>().sqrt();
        let den: f32 = y.iter().map(|q| q * q).sum::<f32>().sqrt().max(1e-30);
        num / den
    }

    /// A cold basis: the two columns of A with the largest norm, orthonormalized (the top left singular
    /// vectors live in range(A), so this starts far better than the identity does).
    fn cold_basis(a: &na::SMatrix<f32, K, K>) -> na::SMatrix<f32, K, R> {
        let mut order: Vec<usize> = (0..K).collect();
        order.sort_by(|&i, &j| {
            let ni: f32 = (0..K).map(|r| a[(r, i)] * a[(r, i)]).sum();
            let nj: f32 = (0..K).map(|r| a[(r, j)] * a[(r, j)]).sum();
            nj.partial_cmp(&ni).unwrap()
        });
        let c0 = a.column(order[0]).into_owned().normalize();
        let c1 = a.column(order[1]).into_owned();
        let d = c1.dot(&c0);
        let c1o = (c1 - c0.clone() * d).normalize();
        let mut q = na::SMatrix::<f32, K, R>::zeros();
        q.set_column(0, &c0);
        q.set_column(1, &c1o);
        q
    }

    fn us_per(secs: f64, n: usize) -> f64 {
        secs * 1e6 / n as f64
    }

    #[test]
    #[ignore]
    fn bench_rank2_paths() {
        let n = 2000usize;
        let cases: Vec<(&str, Vec<f32>)> = vec![
            ("lowrank(0.85x8)", synth(0.85, 8, 12345)),
            ("slow(0.98x16)", synth(0.98, 16, 999)),
            ("flat(1.0x32)", synth(1.0, 32, 4242)),
        ];
        for (name, a) in &cases {
            let sm = na::SMatrix::<f32, K, K>::from_row_slice(a);
            let gram = sm * sm.transpose();
            let t0 = Instant::now();
            for _ in 0..n {
                let e = na::SymmetricEigen::new(black_box(gram));
                black_box(e.eigenvalues[0]);
            }
            let t_eig = us_per(t0.elapsed().as_secs_f64(), n);
            let t0 = Instant::now();
            for _ in 0..n {
                black_box(black_box(&sm) * black_box(&sm).transpose());
            }
            let t_gram = us_per(t0.elapsed().as_secs_f64(), n);

            let mut buf = a.clone();
            let t0 = Instant::now();
            for _ in 0..n {
                buf.copy_from_slice(a);
                compress_wkv_state(&mut buf, 1, K, 1, None, false, false, false, false, None, None,
                                   None, None, None, None);
            }
            let t_r1 = us_per(t0.elapsed().as_secs_f64(), n);
            let ref2 = reference(a);
            let t0 = Instant::now();
            for _ in 0..n {
                buf.copy_from_slice(a);
                compress_wkv_state(&mut buf, 1, K, R, None, false, false, false, false, None, None,
                                   None, None, None, None);
            }
            let t_r2 = us_per(t0.elapsed().as_secs_f64(), n);

            let mut line = format!(
                "{name:16} gram {t_gram:6.2}us eig {t_eig:6.2}us compress r1 {t_r1:6.2}us r2 {t_r2:6.2}us"
            );
            for iters in [1usize, 2, 4, 8] {
                let mut q = cold_basis(&sm);
                subit(&sm, &mut q, iters);
                let (uf, vf) = proto_factors(&sm, &q);
                let proto: Vec<f32> = row_major(&(uf * vf.transpose()));
                let err = rel_frob(&proto, &ref2);
                let t0 = Instant::now();
                for _ in 0..n {
                    let mut q2 = cold_basis(black_box(&sm));
                    subit(black_box(&sm), &mut q2, iters);
                    black_box(&q2);
                }
                let t = us_per(t0.elapsed().as_secs_f64(), n);
                line += &format!(" | cold{iters} {t:5.2}us err {err:.2e}");
            }
            // warm start: the state drifts a little between steps, so re-converge from the PREVIOUS basis
            let mut warm = cold_basis(&sm);
            subit(&sm, &mut warm, 8);
            let mut d = a.clone();
            for (i, x) in d.iter_mut().enumerate() {
                *x = *x * 1.01 + 0.001 * ((i % 7) as f32 - 3.0);
            }
            let drifted = na::SMatrix::<f32, K, K>::from_row_slice(&d);
            let ref_drift = reference(&d);
            for iters in [1usize, 2] {
                let mut q = warm;
                subit(&drifted, &mut q, iters);
                let (uf, vf) = proto_factors(&drifted, &q);
                let proto: Vec<f32> = row_major(&(uf * vf.transpose()));
                let err = rel_frob(&proto, &ref_drift);
                let t0 = Instant::now();
                for _ in 0..n {
                    let mut q2 = black_box(warm);
                    subit(black_box(&drifted), &mut q2, iters);
                    black_box(&q2);
                }
                let t = us_per(t0.elapsed().as_secs_f64(), n);
                line += &format!(" | warm{iters} {t:5.2}us err {err:.2e}");
            }
            println!("{line}");
        }
    }

    /// PROTOTYPE BENCH (2026-11-23): is the BASIS LAYOUT the reason the codec's engine path is slower
    /// than the rank-1 path? `fixedbasis_roundtrip` reads its bases with a stride of `stride` floats,
    /// so consecutive inner-loop accesses are `stride * 4` bytes apart - at stride 32 that is a fresh
    /// cache line for EVERY element, in all four loops (two K x m projections and the K x K
    /// reconstruction). This times the production function against the SAME math with pre-transposed
    /// bases (every inner loop walks contiguous memory) and asserts the two outputs are bit-identical,
    /// because the accumulation order is unchanged.
    fn random_basis(k: usize, stride: usize, seed: u64) -> Vec<f32> {
        let mut rng = seed | 1;
        let mut rnd = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            ((rng >> 11) as f32 / (1u64 << 53) as f32) * 2.0 - 1.0
        };
        (0..k * stride).map(|_| rnd()).collect()
    }

    fn transpose_basis(a: &[f32], k: usize, stride: usize) -> Vec<f32> {
        let mut out = vec![0f32; k * stride];
        for i in 0..k {
            for t in 0..stride {
                out[t * k + i] = a[i * stride + t];
            }
        }
        out
    }

    /// The same math as `super::fixedbasis_roundtrip`, but every inner loop is contiguous: `pt`/`qt`
    /// are the transposes of `p`/`q`. The accumulation order in each of the four loops is IDENTICAL
    /// to the production function, so the result must be bit-for-bit equal.
    #[allow(clippy::too_many_arguments)]
    fn fb_contiguous(
        data: &mut [f32],
        h: usize,
        k: usize,
        m: usize,
        p: &[f32],
        pt: &[f32],
        q: &[f32],
        qt: &[f32],
        stride: usize,
        qmax: Option<f64>,
    ) {
        let mut tmp = vec![0f32; k * m]; // W Q     (k x m)
        let mut c = vec![0f32; m * m]; //   P^T W Q (m x m)
        let mut u = vec![0f32; k * m]; //   P C     (k x m)
        for hh in 0..h {
            let w = &mut data[hh * k * k..(hh + 1) * k * k];
            for i in 0..k {
                let wrow = &w[i * k..i * k + k];
                for t in 0..m {
                    let qrow = &qt[t * k..t * k + k];
                    let mut s = 0f32;
                    for j in 0..k {
                        s += wrow[j] * qrow[j];
                    }
                    tmp[i * m + t] = s;
                }
            }
            for s0 in 0..m {
                let prow = &pt[s0 * k..s0 * k + k];
                for t in 0..m {
                    let mut s = 0f32;
                    for i in 0..k {
                        s += prow[i] * tmp[i * m + t];
                    }
                    c[s0 * m + t] = s;
                }
            }
            if let Some(qmax) = qmax {
                super::quant_vec_inplace(&mut c, qmax);
            }
            for i in 0..k {
                let prow = &p[i * stride..i * stride + stride];
                for t in 0..m {
                    let mut s = 0f32;
                    for s0 in 0..m {
                        s += prow[s0] * c[s0 * m + t];
                    }
                    u[i * m + t] = s;
                }
            }
            for i in 0..k {
                let urow = &u[i * m..i * m + m];
                for j in 0..k {
                    let qrow = &q[j * stride..j * stride + stride];
                    let mut s = 0f32;
                    for t in 0..m {
                        s += urow[t] * qrow[t];
                    }
                    w[i * k + j] = s;
                }
            }
        }
    }

    #[test]
    #[ignore]
    fn bench_fixedbasis_layout() {
        use super::fixedbasis_roundtrip;
        let n = 5000usize;
        let stride = K;
        let p = random_basis(K, stride, 777);
        let q = random_basis(K, stride, 778);
        let pt = transpose_basis(&p, K, stride);
        let qt = transpose_basis(&q, K, stride);
        let a = synth(0.92, 12, 31337);
        for m in [4usize, 6, 8, 16] {
            let mut buf = a.clone();
            fixedbasis_roundtrip(&mut buf, 1, K, m, &p, &q, stride, None);
            let mut buf2 = a.clone();
            fb_contiguous(&mut buf2, 1, K, m, &p, &pt, &q, &qt, stride, None);
            let diff = buf
                .iter()
                .zip(buf2.iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0f32, f32::max);
            let t0 = Instant::now();
            for _ in 0..n {
                let mut b = a.clone();
                fixedbasis_roundtrip(black_box(&mut b), 1, K, m, &p, &q, stride, None);
            }
            let t_prod = us_per(t0.elapsed().as_secs_f64(), n);
            let t0 = Instant::now();
            for _ in 0..n {
                let mut b = a.clone();
                fb_contiguous(black_box(&mut b), 1, K, m, &p, &pt, &q, &qt, stride, None);
            }
            let t_cont = us_per(t0.elapsed().as_secs_f64(), n);
            let t0 = Instant::now();
            for _ in 0..n {
                let mut b = a.clone();
                let ptv = transpose_basis(&p, K, stride);
                let qtv = transpose_basis(&q, K, stride);
                fb_contiguous(black_box(&mut b), 1, K, m, &p, &ptv, &q, &qtv, stride, None);
            }
            let t_trans = us_per(t0.elapsed().as_secs_f64(), n);
            println!(
                "m={m:2}  production {t_prod:7.3} us   contiguous {t_cont:7.3} us ({:.2}x)   \
                 on-the-fly transpose {t_trans:7.3} us ({:.2}x)   max|diff| {diff:.3e}",
                t_prod / t_cont,
                t_prod / t_trans
            );
        }
    }
}
