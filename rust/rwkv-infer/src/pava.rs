//! PAVA rectifier + the button-interval solver — the deploy-side operators that turn four raw
//! recall curves into four ordered, monotone scheduling intervals.
//!
//! Ported from `rwkv/model/pava.py::pava_rectify_scalar` (the scalar reference, which the
//! vectorized torch path is itself gated against) and `SrsRWKVRnn.button_intervals`.
//!
//! WHY this lives in the engine and not in training: the rectifier is the model's button-ordering
//! GUARANTEE, not a loss term. Until 2026-07-26 it existed only inside the training loss, so eval
//! and this engine both computed a model we never intended to ship — see the three-way parity rule
//! in CLAUDE.md §9. Both halves must apply it or none of them do.

/// |p| below this uses the geometric-mean branch (matches pava.py's P_EPS).
const P_EPS: f32 = 1e-3;

/// Weighted power mean of two values. p -> 0 degenerates to the weighted geometric mean, which is
/// why the small-|p| branch exists rather than letting `a.powf(1/p)` blow up.
fn pmean(a: f32, b: f32, wa: f32, wb: f32, p: f32) -> f32 {
    if p.abs() < P_EPS {
        ((wa * a.ln() + wb * b.ln()) / (wa + wb)).exp()
    } else {
        ((wa * a.powf(p) + wb * b.powf(p)) / (wa + wb)).powf(1.0 / p)
    }
}

/// Pool-adjacent-violators on 4 button values with 3 learned junction powers.
///
/// Enforces P_Again <= P_Hard <= P_Good <= P_Easy by pooling any out-of-order adjacent pair into
/// their weighted power mean, repeatedly, until the sequence is non-decreasing. `powers[j]` is the
/// exponent used at the junction between slots j and j+1; p = 1 is ordinary arithmetic PAVA, which
/// is the correct fallback for a checkpoint with no `pava_theta`.
pub fn pava_rectify(v: [f32; 4], w: [f32; 4], powers: [f32; 3]) -> [f32; 4] {
    // stack entries: (value, weight, leftmost slot)
    let mut stack: Vec<(f32, f32, usize)> = Vec::with_capacity(4);
    for k in 0..4 {
        stack.push((v[k], w[k], k));
        while stack.len() >= 2 && stack[stack.len() - 2].0 > stack[stack.len() - 1].0 {
            let (bv, bw, bl) = stack.pop().unwrap();
            let (av, aw, al) = stack.pop().unwrap();
            let p = powers[bl - 1]; // junction between slot bl-1 and bl
            stack.push((pmean(av, bv, aw, bw, p), aw + bw, al));
        }
    }
    // blocks are contiguous and already in slot order; expand back to 4 slots
    let mut out = [0f32; 4];
    for i in 0..stack.len() {
        let (val, _, left) = stack[i];
        let next = if i + 1 < stack.len() { stack[i + 1].2 } else { 4 };
        for slot in left..next {
            out[slot] = val;
        }
    }
    out
}

/// Learned junction powers `p = 2*tanh(theta)`, or all-ones (classic arithmetic PAVA) when the
/// checkpoint carries no `pava_theta`. Mirrors `SrsRWKVRnn.button_curves`.
pub fn junction_powers(theta: Option<&Vec<f32>>) -> [f32; 3] {
    match theta {
        Some(t) if t.len() == 3 => [
            2.0 * t[0].tanh(),
            2.0 * t[1].tanh(),
            2.0 * t[2].tanh(),
        ],
        _ => [1.0, 1.0, 1.0],
    }
}

/// Solve each rectified button curve for the elapsed time at which it falls to `desired_retention`.
///
/// `curves_at(t) -> [f32; 4]` must return the four RECTIFIED recall probabilities at time `t`.
/// ⚠ Rectification MUST happen inside that closure, i.e. at every bisection probe: pooling couples
/// the four buttons, so evaluating raw curves and rectifying the answer afterwards gives a
/// different (wrong) result. This is cheap regardless — the heads do not depend on t, so a probe is
/// closed-form arithmetic and the RWKV forward runs exactly 4 times per press, not 4 times per probe.
///
/// Bisection is geometric (sqrt of the bracket), because intervals span seconds to years and a
/// linear split would waste nearly all its iterations at the top of the range.
pub fn solve_intervals<F>(
    mut curves_at: F,
    desired_retention: f32,
    lo_s: f32,
    hi_s: f32,
    iters: usize,
) -> [f32; 4]
where
    F: FnMut(f32) -> [f32; 4],
{
    let mut lo = [lo_s; 4];
    let mut hi = [hi_s; 4];
    for _ in 0..iters {
        // Each button has its own bracket, so probe each at its own midpoint. The closure returns
        // all four curves per call; we use button k's own value, which is the diagonal.
        let mut mid = [0f32; 4];
        for k in 0..4 {
            mid[k] = (lo[k] * hi[k]).sqrt();
        }
        for k in 0..4 {
            let r = curves_at(mid[k])[k];
            if r > desired_retention {
                lo[k] = mid[k];
            } else {
                hi[k] = mid[k];
            }
        }
    }
    let mut out = [0f32; 4];
    for k in 0..4 {
        out[k] = (lo[k] * hi[k]).sqrt();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn already_ordered_is_untouched() {
        let v = [0.1f32, 0.3, 0.6, 0.9];
        let out = pava_rectify(v, [1.0; 4], [1.0; 3]);
        for k in 0..4 {
            assert!((out[k] - v[k]).abs() < 1e-6, "{out:?} != {v:?}");
        }
    }

    #[test]
    fn violation_is_pooled_to_a_tie() {
        // slots 1,2 are out of order -> they pool to their mean (0.5), and the result is ordered
        let out = pava_rectify([0.1, 0.7, 0.3, 0.9], [1.0; 4], [1.0; 3]);
        assert!(out.windows(2).all(|p| p[0] <= p[1] + 1e-6), "not ordered: {out:?}");
        assert!((out[1] - 0.5).abs() < 1e-6 && (out[2] - 0.5).abs() < 1e-6, "{out:?}");
    }

    #[test]
    fn output_is_always_ordered() {
        // a descending input must come out completely flat (everything pools into one block)
        let out = pava_rectify([0.9, 0.6, 0.3, 0.1], [1.0; 4], [1.0; 3]);
        assert!(out.windows(2).all(|p| p[0] <= p[1] + 1e-6), "not ordered: {out:?}");
        assert!((out[0] - out[3]).abs() < 1e-6, "expected one flat block: {out:?}");
    }

    #[test]
    fn powers_shift_the_pooled_value_but_not_the_ordering() {
        let v = [0.1f32, 0.8, 0.2, 0.9];
        let arith = pava_rectify(v, [1.0; 4], [1.0; 3]);
        let geo = pava_rectify(v, [1.0; 4], [0.0; 3]); // p->0 = geometric mean branch
        assert!(geo.windows(2).all(|p| p[0] <= p[1] + 1e-6), "not ordered: {geo:?}");
        // geometric mean of a violating pair is below the arithmetic one
        assert!(geo[1] < arith[1], "geo {geo:?} not below arith {arith:?}");
    }

    #[test]
    fn theta_maps_to_powers() {
        let p = junction_powers(Some(&vec![0.0, 10.0, -10.0]));
        assert!((p[0] - 0.0).abs() < 1e-6);
        assert!((p[1] - 2.0).abs() < 1e-4); // 2*tanh(10) -> 2
        assert!((p[2] + 2.0).abs() < 1e-4);
        assert_eq!(junction_powers(None), [1.0, 1.0, 1.0]);
    }

    #[test]
    fn solver_finds_the_target_and_orders_intervals() {
        // synthetic monotone curves, button k decaying with time constant s_k (increasing in k)
        let s = [1e3f32, 1e4, 1e5, 1e6];
        let curves = |t: f32| {
            let mut v = [0f32; 4];
            for k in 0..4 {
                v[k] = (-t / s[k]).exp();
            }
            pava_rectify(v, [1.0; 4], [1.0; 3])
        };
        let iv = solve_intervals(curves, 0.9, 1.0, 1e9, 60);
        assert!(iv.windows(2).all(|p| p[0] <= p[1] * (1.0 + 1e-4)), "unordered: {iv:?}");
        for k in 0..4 {
            let r = curves(iv[k])[k];
            assert!((r - 0.9).abs() < 1e-3, "button {k}: R({}) = {r}, want 0.9", iv[k]);
        }
        // lower retention must mean longer intervals
        let iv8 = solve_intervals(curves, 0.8, 1.0, 1e9, 60);
        for k in 0..4 {
            assert!(iv8[k] > iv[k], "R=0.8 not longer than R=0.9 at button {k}");
        }
    }
}
