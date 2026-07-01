//! Plain-Rust f32 batched forward (no candle in the hot loop). Mirrors `Model::review_batched`
//! exactly (transcribed op-for-op) but on flat Vec<f32> buffers with batched matmuls that reuse
//! each weight matrix across all B cards -> no per-op tensor allocation / candle dispatch. Parity
//! gated vs the candle path (<1e-5). All dims derived from weight shapes (arch-agnostic).
//!
//! Layout: activations are card-major flat: card b's `dim` values at [b*dim .. (b+1)*dim].
//! WKV state is [b][h][i][j] at ((b*H + h)*K + i)*K + j.

use anyhow::{anyhow, Result};
use candle_core::Tensor;
use std::collections::HashMap;

const LN_EPS: f32 = 1e-5;
const GN_EPS: f32 = 64e-5; // matches model.rs (NOT 1e-5)
const L2_EPS: f32 = 1e-12;

/// A weight as flat f32 + its 2 logical dims (1-D weights use d1 = 1).
struct FastW {
    v: Vec<f32>,
    d0: usize,
    d1: usize,
}

pub struct FastLayerState {
    pub t_xshift: Vec<f32>, // B*C
    pub t_state: Vec<f32>,  // B*H*K*K
    pub c_xshift: Vec<f32>, // B*C
}
pub type FastStreamState = Vec<FastLayerState>;

pub struct FastModel {
    pub c: usize,
    pub h: usize,
    pub k: usize,
    pub stream_layers: Vec<usize>,
    pub num_curves: usize,
    pub num_points: usize,
    fw: HashMap<String, FastW>,  // raw weights (norms, lerps, bias, bonus, ...) as f32
    fwt: HashMap<String, FastW>, // linear weights pre-transposed to (in,out) f32
}

fn to_vec(t: &Tensor) -> Result<Vec<f32>> {
    Ok(t.flatten_all()?.to_vec1::<f32>()?)
}

impl FastModel {
    /// Build from the candle weight maps (raw `w` + pre-transposed `lin_wt`) and derived dims.
    pub fn build(
        w: &HashMap<String, Tensor>,
        lin_wt: &HashMap<String, Tensor>,
        c: usize,
        h: usize,
        k: usize,
        stream_layers: Vec<usize>,
        num_curves: usize,
        num_points: usize,
    ) -> Result<Self> {
        let mut fw = HashMap::new();
        for (key, t) in w.iter() {
            let dims = t.dims();
            let (d0, d1) = match dims.len() {
                1 => (dims[0], 1usize),
                2 => (dims[0], dims[1]),
                _ => (t.elem_count(), 1usize), // flatten anything else (e.g. bonus (H,K))
            };
            fw.insert(key.clone(), FastW { v: to_vec(t)?, d0, d1 });
        }
        let mut fwt = HashMap::new();
        for (key, t) in lin_wt.iter() {
            let dims = t.dims(); // (in,out)
            fwt.insert(key.clone(), FastW { v: to_vec(t)?, d0: dims[0], d1: dims[1] });
        }
        Ok(Self { c, h, k, stream_layers, num_curves, num_points, fw, fwt })
    }

    fn raw(&self, key: &str) -> Result<&FastW> {
        self.fw.get(key).ok_or_else(|| anyhow!("fast: missing raw weight {key}"))
    }
    fn lt(&self, key: &str) -> Result<&FastW> {
        self.fwt.get(key).ok_or_else(|| anyhow!("fast: missing lin weight {key}"))
    }

    /// y(B,out) = x(B,in) @ Wt(in,out) [+ bias(out)]. Wt reused across all B (cache-friendly).
    fn linear(&self, x: &[f32], b: usize, prefix: &str, bias: bool) -> Result<Vec<f32>> {
        let wt = self.lt(&format!("{prefix}.weight"))?;
        let (inn, out) = (wt.d0, wt.d1);
        let bvec = if bias { Some(&self.raw(&format!("{prefix}.bias"))?.v) } else { None };
        let mut y = vec![0f32; b * out];
        for bi in 0..b {
            let xr = &x[bi * inn..bi * inn + inn];
            let yr = &mut y[bi * out..bi * out + out];
            if let Some(bv) = bvec {
                yr.copy_from_slice(&bv[..out]);
            }
            // accumulate over in; weight row-major (in,out) -> wt.v[i*out + o]
            for i in 0..inn {
                let xi = xr[i];
                if xi == 0.0 {
                    continue;
                }
                let wrow = &wt.v[i * out..i * out + out];
                for o in 0..out {
                    yr[o] += xi * wrow[o];
                }
            }
        }
        Ok(y)
    }

    /// LayerNorm over the last `dim` of (B,dim) with weight/bias under `prefix`.
    fn layernorm(&self, x: &[f32], b: usize, dim: usize, prefix: &str) -> Result<Vec<f32>> {
        let wv = &self.raw(&format!("{prefix}.weight"))?.v;
        let bv = &self.raw(&format!("{prefix}.bias"))?.v;
        let mut y = vec![0f32; b * dim];
        for bi in 0..b {
            let xr = &x[bi * dim..bi * dim + dim];
            let mean: f32 = xr.iter().sum::<f32>() / dim as f32;
            let var: f32 = xr.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / dim as f32;
            let inv = 1.0 / (var + LN_EPS).sqrt();
            let yr = &mut y[bi * dim..bi * dim + dim];
            for j in 0..dim {
                yr[j] = (xr[j] - mean) * inv * wv[j] + bv[j];
            }
        }
        Ok(y)
    }

    /// GroupNorm over (B,C) with `groups` groups (group size C/groups), weight/bias under prefix.
    fn groupnorm(&self, x: &[f32], b: usize, prefix: &str) -> Result<Vec<f32>> {
        let (c, g) = (self.c, self.h);
        let cs = c / g;
        let wv = &self.raw(&format!("{prefix}.weight"))?.v;
        let bv = &self.raw(&format!("{prefix}.bias"))?.v;
        let mut y = vec![0f32; b * c];
        for bi in 0..b {
            for gi in 0..g {
                let off = bi * c + gi * cs;
                let xr = &x[off..off + cs];
                let mean: f32 = xr.iter().sum::<f32>() / cs as f32;
                let var: f32 = xr.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / cs as f32;
                let inv = 1.0 / (var + GN_EPS).sqrt();
                for j in 0..cs {
                    let idx = gi * cs + j;
                    y[off + j] = (xr[j] - mean) * inv * wv[idx] + bv[idx];
                }
            }
        }
        Ok(y)
    }

    fn lora_simple(&self, x: &[f32], b: usize, p: &str) -> Result<Vec<f32>> {
        let a = self.linear(x, b, &format!("{p}.A"), false)?;
        self.linear(&a, b, &format!("{p}.B_and_lamb"), true)
    }
    fn lora_mlp(&self, x: &[f32], b: usize, p: &str) -> Result<Vec<f32>> {
        let mut a = self.linear(x, b, &format!("{p}.A"), false)?;
        for v in a.iter_mut() {
            *v = v.tanh();
        }
        self.linear(&a, b, &format!("{p}.B_and_lamb"), true)
    }

    /// One time-mixer layer. in_x (B,C). Returns (out(B,C), v0_out(B,C), new_t_xshift(B,C), new_state(B*H*K*K)).
    #[allow(clippy::type_complexity)]
    fn time_mixer(
        &self,
        p: &str,
        layer_id: usize,
        in_x: &[f32],
        b: usize,
        v0: Option<&[f32]>,
        st: Option<(&[f32], &[f32])>,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> {
        let (c, h, k) = (self.c, self.h, self.k);
        let x = self.layernorm(in_x, b, c, &format!("{p}.layer_norm"))?;
        let zeros;
        let (xshift, s_prev): (&[f32], &[f32]) = match st {
            Some((xs, s)) => (xs, s),
            None => {
                zeros = vec![0f32; b * h * k * k];
                (&x, &zeros)
            }
        };
        // diff = xshift - x
        let mut diff = vec![0f32; b * c];
        for i in 0..b * c {
            diff[i] = xshift[i] - x[i];
        }
        // 8 lerp inputs: inp_i = x + lerp_w[i]*diff  (lerp_w is (8,C))
        let lerp_w = &self.raw(&format!("{p}.rkvdag_lerp"))?.v; // 8*C
        let make_inp = |i: usize| -> Vec<f32> {
            let lw = &lerp_w[i * c..i * c + c];
            let mut out = vec![0f32; b * c];
            for bi in 0..b {
                for j in 0..c {
                    let idx = bi * c + j;
                    out[idx] = x[idx] + lw[j] * diff[idx];
                }
            }
            out
        };
        let inp_r = make_inp(0);
        let inp_k = make_inp(1);
        let inp_v = make_inp(2);
        let inp_d = make_inp(3);
        let inp_a = make_inp(4);
        let inp_g = make_inp(5);
        let inp_ks = make_inp(6);
        let inp_vs = make_inp(7);

        let r = self.linear(&inp_r, b, &format!("{p}.W_r"), false)?;
        let kk = self.linear(&inp_k, b, &format!("{p}.W_k"), false)?;
        let mut k_scale = self.linear(&inp_ks, b, &format!("{p}.k_scale_linear"), true)?; // (B,H)
        sigmoid_(&mut k_scale);
        let mut v_scale = self.linear(&inp_vs, b, &format!("{p}.v_scale_linear"), true)?; // (B,H)
        sigmoid_(&mut v_scale);

        let (v, v0_out) = if layer_id == 0 {
            let v = self.linear(&inp_v, b, &format!("{p}.W_v"), false)?;
            (v.clone(), v)
        } else {
            let mut v_lerp = self.lora_simple(&inp_v, b, &format!("{p}.v_lora_simple"))?;
            sigmoid_(&mut v_lerp);
            let wv = self.linear(&inp_v, b, &format!("{p}.W_v"), false)?;
            let v0 = v0.ok_or_else(|| anyhow!("v0 missing for layer>0"))?;
            // lerp(wv, v0, v_lerp) = wv + v_lerp*(v0 - wv)
            let mut v = vec![0f32; b * c];
            for i in 0..b * c {
                v[i] = wv[i] + v_lerp[i] * (v0[i] - wv[i]);
            }
            (v, v0.to_vec())
        };

        let mut a = self.lora_simple(&inp_a, b, &format!("{p}.a_lora_simple"))?;
        sigmoid_(&mut a);
        let mut g_a = self.linear(&inp_g, b, &format!("{p}.lora_A_g"), false)?;
        sigmoid_(&mut g_a);
        let g = self.linear(&g_a, b, &format!("{p}.lora_B_g"), false)?;

        // decay: _d = -0.5 - softplus(-d_mlp); w_decay = exp(-exp(_d))
        let d_mlp = self.lora_mlp(&inp_d, b, &format!("{p}.d_lora_mlp"))?;
        let mut w_decay = vec![0f32; b * c];
        for i in 0..b * c {
            let _d = -0.5 - softplus(-d_mlp[i]);
            w_decay[i] = (-(_d.exp())).exp();
        }

        // reshape to heads + l2norm + scale. All (B,H,K) flat == (B,C) flat (C=H*K).
        let mut k_h0 = kk.clone();
        l2norm_heads_(&mut k_h0, b, h, k);
        scale_heads_(&mut k_h0, &k_scale, b, h, k); // *k_scale[b,h]
        let mut v_h = v.clone();
        l2norm_heads_(&mut v_h, b, h, k);
        scale_heads_(&mut v_h, &v_scale, b, h, k);
        let r_h = &r; // (B,C)=(B,H,K)
        let w_h = &w_decay;
        let a_h = &a;
        let kd_h = k_h0.clone(); // k_deformed = k before *a
        let mut k_h = k_h0.clone();
        for i in 0..b * c {
            k_h[i] *= a_h[i];
        }

        // WKV recurrence
        let (out_hk, next_s) = single_timestep(b, h, k, r_h, &k_h, &v_h, w_h, a_h, &kd_h, s_prev);

        // out group norm over (B,C)
        let out_gn = self.groupnorm(&out_hk, b, &format!("{p}.out_group_norm"))?;

        // r_k bonus: term = (r_h * bonus * k_h).sum(K) per head; bonus = term * v_h
        let bonus_p = &self.raw(&format!("{p}.bonus"))?.v; // (H,K) flat = C
        let mut combined = vec![0f32; b * c];
        for bi in 0..b {
            for hh in 0..h {
                let base = bi * c + hh * k;
                let mut term = 0f32;
                for j in 0..k {
                    term += r_h[base + j] * bonus_p[hh * k + j] * k_h[base + j];
                }
                for i in 0..k {
                    combined[base + i] = out_gn[base + i] + term * v_h[base + i];
                }
            }
        }
        // gate then W_o
        for i in 0..b * c {
            combined[i] *= g[i];
        }
        let out2 = self.linear(&combined, b, &format!("{p}.W_o"), false)?;
        let mut out = vec![0f32; b * c];
        for i in 0..b * c {
            out[i] = in_x[i] + out2[i];
        }
        Ok((out, v0_out, x, next_s))
    }

    /// One channel-mixer layer. Returns (out(B,C), new_c_xshift(B,C)).
    fn channel_mixer(
        &self,
        p: &str,
        in_x: &[f32],
        b: usize,
        xshift: Option<&[f32]>,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let c = self.c;
        let x = self.layernorm(in_x, b, c, &format!("{p}.layer_norm"))?;
        let xs = xshift.unwrap_or(&x);
        let lerp_k = &self.raw(&format!("{p}.lerp_k"))?.v; // (C,)
        let mut mixed = vec![0f32; b * c];
        for bi in 0..b {
            for j in 0..c {
                let idx = bi * c + j;
                mixed[idx] = x[idx] + lerp_k[j] * (xs[idx] - x[idx]);
            }
        }
        let mut kk = self.linear(&mixed, b, &format!("{p}.W_k"), false)?;
        for v in kk.iter_mut() {
            let r = v.max(0.0);
            *v = r * r; // relu^2
        }
        let o = self.linear(&kk, b, &format!("{p}.W_v"), false)?;
        let mut out = vec![0f32; b * c];
        for i in 0..b * c {
            out[i] = in_x[i] + o[i];
        }
        Ok((out, x))
    }

    fn run_stream(
        &self,
        m: usize,
        n_layers: usize,
        input: &[f32],
        b: usize,
        state: Option<&FastStreamState>,
    ) -> Result<(Vec<f32>, FastStreamState)> {
        let mut x = input.to_vec();
        let mut v0: Option<Vec<f32>> = None;
        let mut new_state: FastStreamState = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            let tp = format!("rwkv_modules.{m}.blocks.{l}.time_mixer");
            let cp = format!("rwkv_modules.{m}.blocks.{l}.channel_mixer");
            let ls = state.map(|s| &s[l]);
            let t_st = ls.map(|s| (s.t_xshift.as_slice(), s.t_state.as_slice()));
            let (xt, v0_out, t_xshift, t_state) =
                self.time_mixer(&tp, l, &x, b, v0.as_deref(), t_st)?;
            v0 = Some(v0_out);
            let c_st = ls.map(|s| s.c_xshift.as_slice());
            let (xc, c_xshift) = self.channel_mixer(&cp, &xt, b, c_st)?;
            x = xc;
            new_state.push(FastLayerState { t_xshift, t_state, c_xshift });
        }
        Ok((x, new_state))
    }

    fn features2card(&self, feats: &[f32], b: usize) -> Result<Vec<f32>> {
        let mut x = self.linear(feats, b, "features2card.0", true)?;
        silu_(&mut x);
        let hidden = x.len() / b;
        let mut x = self.layernorm(&x, b, hidden, "features2card.2")?;
        x = self.linear(&x, b, "features2card.3", true)?;
        silu_(&mut x);
        Ok(x)
    }

    /// Full batched forward. feats (B*92). Returns (out_ahead(B*np), out_w(B*nc), out_p(B*4), new_states).
    #[allow(clippy::type_complexity)]
    pub fn review_batched(
        &self,
        feats: &[f32],
        b: usize,
        states: &[Option<FastStreamState>; 5],
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, [FastStreamState; 5])> {
        let mut x = self.features2card(feats, b)?;
        let mut new: Vec<FastStreamState> = Vec::with_capacity(5);
        for m in 0..5 {
            let (xo, ns) = self.run_stream(m, self.stream_layers[m], &x, b, states[m].as_ref())?;
            x = xo;
            new.push(ns);
        }
        let xh = self.layernorm(&x, b, self.c, "prehead_norm")?;

        // head_w -> softmax (num_curves)
        let mut hw = self.linear(&xh, b, "head_w.0", true)?;
        relu_(&mut hw);
        let hwn = hw.len() / b;
        let hw = self.layernorm(&hw, b, hwn, "head_w.2")?;
        let hw = self.linear(&hw, b, "head_w.4", true)?;
        let out_w_logits = self.linear(&hw, b, "w_linear", true)?;
        let mut out_w = out_w_logits;
        softmax_rows_(&mut out_w, b, self.num_curves);

        // head_ahead (num_points)
        let mut ha = self.linear(&xh, b, "head_ahead_logits.0", true)?;
        relu_(&mut ha);
        let out_ahead = self.linear(&ha, b, "ahead_linear", true)?;

        // head_p (4)
        let mut hp = self.linear(&xh, b, "head_p.0", true)?;
        relu_(&mut hp);
        let out_p = self.linear(&hp, b, "p_linear", true)?;

        let new_arr: [FastStreamState; 5] =
            new.try_into().map_err(|_| anyhow!("stream count mismatch"))?;
        Ok((out_ahead, out_w, out_p, new_arr))
    }

    /// imm prob per card = 1 - softmax(out_p)[again]. out_p is (B,4).
    pub fn imm_prob(&self, out_p: &[f32], b: usize) -> Vec<f32> {
        let mut v = out_p.to_vec();
        softmax_rows_(&mut v, b, 4);
        (0..b).map(|bi| 1.0 - v[bi * 4]).collect()
    }
}

// ---- elementwise helpers (in-place) ----
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
fn sigmoid_(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = sigmoid(*v);
    }
}
fn silu_(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v *= sigmoid(*v);
    }
}
fn relu_(x: &mut [f32]) {
    for v in x.iter_mut() {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
}
fn softplus(x: f32) -> f32 {
    // relu(x) + ln(1 + exp(-|x|))
    x.max(0.0) + (1.0 + (-x.abs()).exp()).ln()
}

/// L2-normalize each head (K-slice) of a (B,H,K) flat buffer over K (eps=1e-12).
fn l2norm_heads_(x: &mut [f32], b: usize, h: usize, k: usize) {
    for bi in 0..b {
        for hh in 0..h {
            let base = (bi * h + hh) * k;
            let mut ss = 0f32;
            for j in 0..k {
                ss += x[base + j] * x[base + j];
            }
            let inv = 1.0 / ss.sqrt().max(L2_EPS);
            for j in 0..k {
                x[base + j] *= inv;
            }
        }
    }
}

/// Multiply each head K-slice of (B,H,K) by a per-(b,h) scalar from scale(B,H).
fn scale_heads_(x: &mut [f32], scale: &[f32], b: usize, h: usize, k: usize) {
    for bi in 0..b {
        for hh in 0..h {
            let s = scale[bi * h + hh];
            let base = (bi * h + hh) * k;
            for j in 0..k {
                x[base + j] *= s;
            }
        }
    }
}

/// Row-wise softmax over `dim` of a (B,dim) flat buffer.
fn softmax_rows_(x: &mut [f32], b: usize, dim: usize) {
    for bi in 0..b {
        let r = &mut x[bi * dim..bi * dim + dim];
        let mx = r.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0f32;
        for v in r.iter_mut() {
            *v = (*v - mx).exp();
            sum += *v;
        }
        let inv = 1.0 / sum;
        for v in r.iter_mut() {
            *v *= inv;
        }
    }
}

/// WKV single timestep, batched. r/k/v/w/a/kd are (B,H,K) flat; s_prev is (B,H,K,K) flat.
/// Returns (out(B,H,K), s_new(B,H,K,K)).
#[allow(clippy::too_many_arguments)]
fn single_timestep(
    b: usize,
    h: usize,
    k: usize,
    r: &[f32],
    kk: &[f32],
    v: &[f32],
    w: &[f32],
    _a: &[f32],
    kd: &[f32],
    s_prev: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    // NOTE: a*kd is folded as akd[j]; caller passes a separately but we need a[j]*kd[j].
    // We recompute akd here from a and kd. (a passed as _a to keep signature parallel.)
    let mut s_new = vec![0f32; b * h * k * k];
    let mut out = vec![0f32; b * h * k];
    for bi in 0..b {
        for hh in 0..h {
            let vb = (bi * h + hh) * k; // base into (B,H,K)
            let sb = vb * k; // base into (B,H,K,K)
            for i in 0..k {
                // sk[i] = sum_j s_prev[i,j]*kd[j]
                let mut sk = 0f32;
                let row = &s_prev[sb + i * k..sb + i * k + k];
                for j in 0..k {
                    sk += row[j] * kd[vb + j];
                }
                let vi = v[vb + i];
                let dst = &mut s_new[sb + i * k..sb + i * k + k];
                for j in 0..k {
                    // s_new[i,j] = s_prev[i,j]*w[j] - sk*a[j]*kd[j] + v[i]*k[j]
                    dst[j] = row[j] * w[vb + j] - sk * _a[vb + j] * kd[vb + j] + vi * kk[vb + j];
                }
            }
            // out[i] = sum_j s_new[i,j]*r[j]
            for i in 0..k {
                let row = &s_new[sb + i * k..sb + i * k + k];
                let mut acc = 0f32;
                for j in 0..k {
                    acc += row[j] * r[vb + j];
                }
                out[vb + i] = acc;
            }
        }
    }
    (out, s_new)
}
