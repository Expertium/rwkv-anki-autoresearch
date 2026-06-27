//! RWKV-7 SRS model, RNN (sequential) inference form, ported from
//! rwkv/model/srs_model_rnn.py + rwkv_rnn_model.py. CPU, f32, batch size 1.
//!
//! Weights are loaded by their PyTorch state_dict names from a safetensors file,
//! so the mapping to the Python module tree is unambiguous.

use anyhow::{anyhow, Result};
use candle_core::{DType, Device, Tensor, D};
use std::collections::HashMap;

pub const H: usize = 4; // heads
pub const K: usize = 32; // head dim
pub const C: usize = 128; // d_model
const LN_EPS: f64 = 1e-5;
const GN_EPS: f64 = 64e-5;
const L2_EPS: f64 = 1e-12;

// layers per stream, in chain order: card, deck, note, preset, user
pub const STREAM_LAYERS: [usize; 5] = [3, 4, 2, 3, 4];

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

fn linear(x: &Tensor, w: &Tensor, b: Option<&Tensor>) -> Result<Tensor> {
    let y = x.matmul(&w.t()?)?;
    match b {
        Some(b) => Ok(y.broadcast_add(b)?),
        None => Ok(y),
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

/// L2-normalize each head row of a (H, K) tensor over the K dim (torch eps=1e-12).
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

pub struct Model {
    w: TMap,
    dev: Device,
    s_space: Tensor,       // (1,128) forgetting-curve time constants
    point_space: Vec<f32>, // (128) interp grid
}

impl Model {
    pub fn load(path: &str, dev: Device) -> Result<Self> {
        let w = candle_core::safetensors::load(path, &dev)?;
        // forgetting_curve s_space (num_curves=128)
        let n = 128usize;
        let lin: Vec<f32> = (0..n).map(|i| 18.5f32 * i as f32 / (n as f32 - 1.0)).collect();
        let s_max = 22.0f32;
        let s_spread = 18.5f32;
        let s_scale = (s_max - s_spread).exp();
        let s_space: Vec<f32> = lin
            .iter()
            .map(|&l| 0.1 + (l.exp() - 1.0) * s_scale)
            .collect();
        // interp point_space (num_points=128), grid identical formula, different consts
        let max_e = 21.0f32;
        let p_spread = 18.5f32;
        let p_scale = (max_e - p_spread).exp();
        let point_space: Vec<f32> = lin
            .iter()
            .map(|&l| 0.5 + (l.exp() - 1.0) * p_scale)
            .collect();
        let s_space_t = Tensor::from_vec(s_space, (1, n), &dev)?;
        Ok(Self {
            w,
            dev,
            s_space: s_space_t,
            point_space,
        })
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
        let wt = get(&self.w, &format!("{prefix}.weight"))?;
        let b = if bias {
            Some(get(&self.w, &format!("{prefix}.bias"))?)
        } else {
            None
        };
        linear(x, wt, b)
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
        let x = self.ln(in_x, &format!("{p}.layer_norm"), LN_EPS)?;
        let (xshift, s_prev) = match st {
            Some((xs, s)) => (xs.clone(), s.clone()),
            None => (
                x.clone(),
                Tensor::zeros((H, K, K), DType::F32, &self.dev)?,
            ),
        };
        let diff = xshift.broadcast_sub(&x)?; // (end - start) component reused

        // 8-way lerp -> r,k,v,d,a,g,k_scale,v_scale
        let lerp_w = get(&self.w, &format!("{p}.rkvdag_lerp"))?; // (8,1,1,128)
        let inp = |i: usize| -> Result<Tensor> {
            let wi = lerp_w.narrow(0, i, 1)?.reshape((1, C))?;
            Ok(x.broadcast_add(&wi.broadcast_mul(&diff)?)?)
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
        let (out_hk, next_s) = single_timestep(&r_h, &k_h, &v_h, &w_h, &a_h, &kd_h, &s_prev)?;

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
            v0 = Some(v0_out);
            let c_st = ls.map(|s| &s.c_xshift);
            let (xc, c_xshift) = self.channel_mixer(&cp, &xt, c_st)?;
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
            let (xo, ns) = self.run_stream(m, STREAM_LAYERS[m], &x, states[m].as_ref())?;
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
        let logits: Vec<f32> = out_ahead_logits.reshape((128,))?.to_vec1()?;
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
}

/// RWKV-7 WKV single timestep (matches rwkv_ops.single_timestep).
/// state' = state*w(cols) - (state@kd)@(a*kd)^T + v@k^T ; out = state'@r
fn single_timestep(
    r: &Tensor, // (H,K)
    k: &Tensor,
    v: &Tensor,
    w: &Tensor,
    a: &Tensor,
    kd: &Tensor,
    s_prev: &Tensor, // (H,K,K)
) -> Result<(Tensor, Tensor)> {
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
