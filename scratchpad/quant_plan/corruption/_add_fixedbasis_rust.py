"""Add the FIXED-BASIS (codec) state-coding path to the Rust deploy engine (approach 1, 2026-11-22).

WHY: the rank-r path stores U_r and V_r (2*K*r numbers) because it re-describes a BASIS every step.
The fixed-basis scheme instead stores the m x m coefficient block C = P^T W Q against a FIXED basis
pair (P, Q) that ships with the model -> m*m numbers per layer, and the deploy cost is two small
matmuls instead of a Gram + eigensolve. A codec basis (like a DCT basis), NOT a codebook: no discrete
entries, no nearest-neighbour lookup, no per-user vectors.

WHAT: adds a shared core `fixedbasis_roundtrip` (model.rs, right after `lowrank_roundtrip`, so both
inference paths call the SAME code and stay bit-identical), two candle wrappers, the CompressCfg /
Model plumbing, and the fast-engine branch in `fast.rs::compress_stream_state`.

Env surface (both must be set):
  RWKV_STATE_FIXEDBASIS_SCOPE = "card:8:int8,note:8:int8"   (name:m[:level]; m = block size)
  RWKV_FIXEDBASIS_PATH = text file, line 1 = entry count, then per entry:
        <name> <layer> <stride> <K*stride float P,row-major> <K*stride float Q,row-major>
  Only the first m <= stride columns are used, so ONE fitted file serves every m.

The path is INERT unless the scope is set. Run: .venv/Scripts/python.exe <this file>
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MODEL = ROOT / "rust" / "rwkv-infer" / "src" / "model.rs"
FAST = ROOT / "rust" / "rwkv-infer" / "src" / "fast.rs"
CRLF = b"\r\n"
LF = b"\n"


def patch(path: Path, pairs, eol: bytes):
    raw = path.read_bytes()
    for i, (old, new) in enumerate(pairs):
        old_b = old.replace(b"\n", eol)
        new_b = new.replace(b"\n", eol)
        n = raw.count(old_b)
        if n != 1:
            raise SystemExit(f"[{path.name}] anchor {i} matched {n} times (need 1):\n{old[:180]}")
        raw = raw.replace(old_b, new_b, 1)
    path.write_bytes(raw)
    print(f"patched {path} ({len(raw)} bytes, {len(pairs)} anchors)")


# ---------------------------------------------------------------- model.rs
CORE = b'''
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
    let mut tmp = vec![0f32; k * m]; // W Q       (k x m)
    let mut c = vec![0f32; m * m]; //  P^T W Q   (m x m) -- the stored payload
    let mut u = vec![0f32; k * m]; //  P C       (k x m)
    for hh in 0..h {
        let w = &mut data[hh * k * k..(hh + 1) * k * k];
        for i in 0..k {
            for t in 0..m {
                let mut s = 0f32;
                for j in 0..k {
                    s += w[i * k + j] * q[j * stride + t];
                }
                tmp[i * m + t] = s;
            }
        }
        for s0 in 0..m {
            for t in 0..m {
                let mut s = 0f32;
                for i in 0..k {
                    s += p[i * stride + s0] * tmp[i * m + t];
                }
                c[s0 * m + t] = s;
            }
        }
        if let Some(qmax) = qmax {
            quant_vec_inplace(&mut c, qmax);
        }
        for i in 0..k {
            for t in 0..m {
                let mut s = 0f32;
                for s0 in 0..m {
                    s += p[i * stride + s0] * c[s0 * m + t];
                }
                u[i * m + t] = s;
            }
        }
        for i in 0..k {
            for j in 0..k {
                let mut s = 0f32;
                for t in 0..m {
                    s += u[i * m + t] * q[j * stride + t];
                }
                w[i * k + j] = s;
            }
        }
        crate::finite::state_quant(w); // observation only
    }
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
/// `quant_roundtrip_batched` and the fast engine\'s per-entity loop.
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
'''

CFG_FIELDS = b'''    pub shift_rot: Option<std::sync::Arc<Vec<f32>>>,
'''

CFG_FIELDS_NEW = b'''    pub shift_rot: Option<std::sync::Arc<Vec<f32>>>,
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
'''

MODEL_FIELDS = b'''    state_lowrank: std::collections::HashMap<usize, (usize, Option<f64>)>,
'''

MODEL_FIELDS_NEW = b'''    state_lowrank: std::collections::HashMap<usize, (usize, Option<f64>)>,
    // FIXED-BASIS (codec) state coding (approach 1): module_idx -> (m, coefficient qmax). Applied
    // INSTEAD of the rank-r round-trip for that stream, layer by layer. See CompressCfg::fb.
    state_fixedbasis: std::collections::HashMap<usize, (usize, Option<f64>)>,
    // The fixed K x stride basis pairs, one per (module_idx, layer). Arc-shared, model-side.
    fb_basis: std::sync::Arc<std::collections::HashMap<(usize, usize), (Vec<f32>, Vec<f32>)>>,
'''

ENV_OLD = b'''        let quant_shifts = std::env::var("RWKV_QUANT_SHIFTS")
'''

ENV_NEW = b'''        // FIXED-BASIS (codec) state coding (approach 1, 2026-11-22):
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
                let mut take = |it: &mut std::str::SplitWhitespace| -> Vec<f32> {
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
        let quant_shifts = std::env::var("RWKV_QUANT_SHIFTS")
'''

CFG_INIT_OLD = b'''            shift_rot,
        };
'''

CFG_INIT_NEW = b'''            shift_rot,
            fb: state_fixedbasis.clone(),
            fb_basis: fb_basis.clone(),
        };
'''

SELF_INIT_OLD = b'''            lowrank_als,
            fast,
        })
'''

SELF_INIT_NEW = b'''            lowrank_als,
            state_fixedbasis,
            fb_basis,
            fast,
        })
'''

CANDLE_OLD = b'''            let t_state = if let Some(&(rank, fqmax)) = self.state_lowrank.get(&module_idx) {
'''

CANDLE_NEW = b'''            // FIXED-BASIS (codec) state coding takes precedence over the rank-r path for a stream that
            // has a basis for THIS layer (RWKV_STATE_FIXEDBASIS_SCOPE + RWKV_FIXEDBASIS_PATH).
            let fb_here = self.state_fixedbasis.get(&module_idx).and_then(|&(mm, fq)| {
                self.fb_basis.get(&(module_idx, l)).map(|pq| (mm, fq, pq))
            });
            let t_state = if let Some((mm, fq, pq)) = fb_here {
                fixedbasis_roundtrip_tensor(&t_state, mm, &pq.0, &pq.1, fq)?
            } else if let Some(&(rank, fqmax)) = self.state_lowrank.get(&module_idx) {
'''

CANDLE_SHIFT_OLD = b'''                let base = self
                    .state_lowrank
                    .get(&module_idx)
                    .and_then(|&(_, fq)| fq)
                    .or_else(|| self.state_quant_qmax.get(&module_idx).copied());
'''

CANDLE_SHIFT_NEW = b'''                let base = self
                    .state_fixedbasis
                    .get(&module_idx)
                    .and_then(|&(_, fq)| fq)
                    .or_else(|| self.state_lowrank.get(&module_idx).and_then(|&(_, fq)| fq))
                    .or_else(|| self.state_quant_qmax.get(&module_idx).copied());
'''

BATCHED_OLD = b'''            let t_state = match self.state_quant_qmax.get(&module_idx) {
                Some(&qmax) => quant_roundtrip_batched(&t_state, qmax)?,
                None => t_state,
            };
'''

BATCHED_NEW = b'''            let fb_here = self.state_fixedbasis.get(&module_idx).and_then(|&(mm, fq)| {
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
'''

TEST_MOD = b'''#[cfg(test)]
mod fixedbasis_tests {
    //! The fixed-basis (codec) round-trip must satisfy exactly what the scheme claims: m = K with no
    //! quantization is an EXACT reconstruction (P, Q orthonormal), a state that lives in the basis\'s
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
        // A rank-2 state built INSIDE the basis\'s top-2 subspace: modes along P[0],P[1] and Q[0],Q[1].
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
        // int8 on the m x m block is a small precision error; int2 on the same block is much larger.
        let p = orth(11);
        let q = orth(22);
        let orig = orth(55);
        let mut a = orig.clone();
        let mut c = orig.clone();
        fixedbasis_roundtrip(&mut a, 1, K, 8, &p, &q, K, Some(127.0));
        fixedbasis_roundtrip(&mut c, 1, K, 8, &p, &q, K, Some(1.0));
        let ea = rel_err(&a, &orig);
        let ec = rel_err(&c, &orig);
        assert!(ea < 0.35, "int8 on the block must be mild, rel={ea}");
        assert!(ec > ea * 2.0, "int2 must be clearly worse than int8 ({ec} vs {ea})");
    }
}

'''

TEST_ANCHOR = b'''#[cfg(test)]
mod compress_bench {
'''

if __name__ == "__main__":
    raw = MODEL.read_bytes()
    if b"fn fixedbasis_roundtrip(" in raw:
        raise SystemExit("model.rs already has the fixed-basis core; nothing to do")

    insert_after = (
        b"    Ok(Tensor::from_vec(data, (h, k, k), t.device())?)\n"
        b"}\n"
    )
    insert_after_b = insert_after.replace(b"\n", CRLF)
    n = raw.count(insert_after_b)
    if n != 1:
        raise SystemExit(f"lowrank_roundtrip tail anchor matched {n} times")
    raw = raw.replace(insert_after_b, insert_after_b + CORE.replace(b"\n", CRLF), 1)

    # the new test module goes before the existing test module
    ta = TEST_ANCHOR.replace(b"\n", CRLF)
    if raw.count(ta) != 1:
        raise SystemExit("compress_bench anchor is not unique")
    raw = raw.replace(ta, TEST_MOD.replace(b"\n", CRLF) + ta, 1)
    MODEL.write_bytes(raw)
    print(f"model.rs: core + tests inserted ({len(raw)} bytes)")

    pairs = [
        (CFG_FIELDS.decode(), CFG_FIELDS_NEW.decode()),
        (MODEL_FIELDS.decode(), MODEL_FIELDS_NEW.decode()),
        (ENV_OLD.decode(), ENV_NEW.decode()),
        (CFG_INIT_OLD.decode(), CFG_INIT_NEW.decode()),
        (SELF_INIT_OLD.decode(), SELF_INIT_NEW.decode()),
        (CANDLE_OLD.decode(), CANDLE_NEW.decode()),
        (CANDLE_SHIFT_OLD.decode(), CANDLE_SHIFT_NEW.decode()),
        (BATCHED_OLD.decode(), BATCHED_NEW.decode()),
    ]
    patch(MODEL, [(a.encode(), b.encode()) for a, b in pairs], CRLF)

    fast_pairs = [
        (b"""    fn compress_stream_state(
        &self,
        m: usize,
        b: usize,
""", b"""    fn compress_stream_state(
        &self,
        m: usize,
        l: usize,
        b: usize,
"""),
        (b"""        let want_basis = match cfg.lowrank.get(&m) {
            Some(&(rank, _))
                if rank.min(k) >= 2
                    && cfg.als.is_none()
""", b"""        let want_basis = match cfg.lowrank.get(&m) {
            Some(&(rank, _))
                if rank.min(k) >= 2
                    && cfg.als.is_none()
                    && !cfg.fb.contains_key(&m) // the fixed-basis path reads no warm basis
"""),
        (b"""        if let Some(&(rank, fqmax)) = cfg.lowrank.get(&m) {
            if cfg.ef {
""", b"""        // FIXED-BASIS (codec) round-trip: takes precedence over the rank-r path when this stream has
        // a basis for THIS layer (RWKV_STATE_FIXEDBASIS_SCOPE + RWKV_FIXEDBASIS_PATH).
        let fbb = cfg
            .fb
            .get(&m)
            .and_then(|&(mm, fq)| cfg.fb_basis.get(&(m, l)).map(|pq| (mm, fq, pq)));
        if let Some((mm, fq, pq)) = fbb {
            for bi in 0..b {
                crate::model::fixedbasis_roundtrip(
                    &mut t_state[bi * hkk..(bi + 1) * hkk],
                    h,
                    k,
                    mm,
                    &pq.0,
                    &pq.1,
                    pq.0.len() / k,
                    fq,
                );
            }
        } else if let Some(&(rank, fqmax)) = cfg.lowrank.get(&m) {
            if cfg.ef {
"""),
        (b"""        if cfg.quant_shifts {
            let base = cfg
                .lowrank
                .get(&m)
                .and_then(|&(_, fq)| fq)
                .or_else(|| cfg.quant_qmax.get(&m).copied());
""", b"""        if cfg.quant_shifts {
            let base = cfg
                .fb
                .get(&m)
                .and_then(|&(_, fq)| fq)
                .or_else(|| cfg.lowrank.get(&m).and_then(|&(_, fq)| fq))
                .or_else(|| cfg.quant_qmax.get(&m).copied());
"""),
        (b"""                self.compress_stream_state(m, b, &mut t_state, &mut t_xshift, &mut c_xshift, e_prev, ls)
""", b"""                self.compress_stream_state(m, l, b, &mut t_state, &mut t_xshift, &mut c_xshift, e_prev, ls)
"""),
    ]
    patch(FAST, fast_pairs, LF)
    print("done")
