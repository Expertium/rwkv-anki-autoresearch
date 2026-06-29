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

/// Low-rank roundtrip of a (H,K,K) WKV state: per head, replace the KxK matrix with its rank-r SVD
/// truncation A_r = (U_r sqrt(S_r)) (V_r sqrt(S_r))^T. The deploy model stores the two Kxr factors
/// (2*K*r floats) instead of the full K*K -- the 0.15 KB card path. If `factor_qmax` is Some, the
/// factors are additionally quantized (the real deploy size = 2*K*r codes at that bit-width). Applying
/// this per recurrence step == the deploy per-persist model (a card advances 1 step per review, state
/// persisted between reviews). Uses a fast top-r truncation (Gram + symmetric eigendecomposition),
/// NOT a full SVD -- the full SVD converges pathologically slowly on near-low-rank states.
fn lowrank_roundtrip(t: &Tensor, rank: usize, factor_qmax: Option<f64>) -> Result<Tensor> {
    use nalgebra::DMatrix;
    let (h, k, k2) = t.dims3()?;
    assert_eq!(k, k2, "WKV state must be square KxK");
    let data: Vec<f32> = t.flatten_all()?.to_vec1()?;
    let mut out = vec![0f32; data.len()];
    let r = rank.min(k);
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
        let gram = &an * an.transpose();
        let eig = nalgebra::SymmetricEigen::new(gram);
        let evals = &eig.eigenvalues;
        let mut order: Vec<usize> = (0..k).collect();
        // NaN-safe descending sort (a non-finite eigenvalue sorts last, so it is never picked as top-r).
        order.sort_by(|&i, &j| {
            evals[j]
                .partial_cmp(&evals[i])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut uf = DMatrix::<f32>::zeros(k, r);
        let mut vf = DMatrix::<f32>::zeros(k, r);
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
        if let Some(qmax) = factor_qmax {
            quant_factor_inplace(&mut uf, qmax);
            quant_factor_inplace(&mut vf, qmax);
        }
        let a_r = &uf * vf.transpose(); // (k,k)
        for rr in 0..k {
            for cc in 0..k {
                out[off + rr * k + cc] = a_r[(rr, cc)];
            }
        }
    }
    Ok(Tensor::from_vec(out, (h, k, k), t.device())?)
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
                "int4" => Some(7.0),
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
        // Pre-transpose every 2D linear weight (out,in) -> (in,out) contiguous ONCE, so the
        // per-token matmul needs no .t() / re-contiguous. Norm weights are 1D and skipped.
        let mut lin_wt: TMap = HashMap::new();
        for (key, t) in w.iter() {
            if key.ends_with(".weight") && t.dims().len() == 2 {
                lin_wt.insert(key.clone(), t.t()?.contiguous()?);
            }
        }
        let s_space_t = Tensor::from_vec(s_space, (1, num_curves), &dev)?;
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
                lowrank_roundtrip(&t_state, rank, fqmax)?
            } else if let Some(&qmax) = self.state_quant_qmax.get(&module_idx) {
                quant_roundtrip(&t_state, qmax)?
            } else {
                t_state
            };
            // Optionally quantize the 1-D shift vectors (part of the persisted state) at this stream's
            // bit-width so the deploy size is honest. Off unless RWKV_QUANT_SHIFTS=1.
            let shift_qmax: Option<f64> = if self.quant_shifts {
                self.state_lowrank
                    .get(&module_idx)
                    .and_then(|&(_, fq)| fq)
                    .or_else(|| self.state_quant_qmax.get(&module_idx).copied())
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
