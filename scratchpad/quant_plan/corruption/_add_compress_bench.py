"""One-shot: append an ignored benchmark/prototype test module to rust/rwkv-infer/src/model.rs.

model.rs has MIXED line endings (CRLF-dominant), so `edit` cannot match a multi-line block in it.
Appending at the end of the file is safe and does not disturb the existing text.

What the test does (all CPU, no GPU; run with `cargo test --release -- --ignored --nocapture`):
  * times the three things the rank>=2 compress path pays for - forming the KxK Gram matrix,
    `nalgebra::SymmetricEigen::new` on it, and the whole `compress_wkv_state` call at rank 1 vs rank 2;
  * runs a PROTOTYPE rank-2 subspace-iteration solver (fixed-size arrays, never forms the Gram matrix)
    and measures its speed AND its reconstruction error against the production eigen path, on states
    with fast-decaying, slow-decaying and flat singular spectra, cold- and warm-started.

Nothing here is production code; the numbers decide whether a fast rank-r engine path is worth building.
"""
from pathlib import Path

P = Path(__file__).resolve().parents[3] / "rust" / "rwkv-infer" / "src" / "model.rs"
text = P.open(encoding="utf-8", newline="").read()
nl = "\r\n" if text.count("\r\n") > text.count("\n") - text.count("\r\n") else "\n"
if "mod compress_bench" in text:
    raise SystemExit("the bench module is already there; nothing to do")

MOD = r'''
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
                           None);
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
                        let d = z.column(l).dot(&z.column(j));
                        z.column_mut(j).axpy(-d, &z.column(l), 1.0);
                    }
                }
                let n = z.column(j).norm();
                if n.is_finite() && n > 1e-30 {
                    z.column_mut(j) /= n;
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
                                   None, None, None);
            }
            let t_r1 = us_per(t0.elapsed().as_secs_f64(), n);
            let ref2 = reference(a);
            let t0 = Instant::now();
            for _ in 0..n {
                buf.copy_from_slice(a);
                compress_wkv_state(&mut buf, 1, K, R, None, false, false, false, false, None, None,
                                   None, None, None);
            }
            let t_r2 = us_per(t0.elapsed().as_secs_f64(), n);

            let mut line = format!(
                "{name:16} gram {t_gram:6.2}us eig {t_eig:6.2}us compress r1 {t_r1:6.2}us r2 {t_r2:6.2}us"
            );
            for iters in [1usize, 2, 4, 8] {
                let mut q = cold_basis(&sm);
                subit(&sm, &mut q, iters);
                let (uf, vf) = proto_factors(&sm, &q);
                let proto: Vec<f32> = (uf * vf.transpose()).iter().copied().collect();
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
                let proto: Vec<f32> = (uf * vf.transpose()).iter().copied().collect();
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
}
'''

text = text.rstrip("\r\n") + nl + MOD.replace("\n", nl)
P.open("w", encoding="utf-8", newline="").write(text)
print(f"appended the compress_bench module ({len(MOD.splitlines())} lines, newline {nl!r})")
