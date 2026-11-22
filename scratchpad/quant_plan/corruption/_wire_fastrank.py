"""One-shot: plumb the new `basis: Option<&mut [f32]>` argument of `compress_wkv_state`
through every caller, and carry the RWKV_LOWRANK_FAST subspace block in the fast engine's
per-entity state (fast.rs), then report it from main.

The E2 model.rs HALF is already applied (helpers + signature + the r>=2 dispatch), so the
crate does NOT compile until the callers pass the 15th argument. This script does that.

Line endings: model.rs / fast.rs are CRLF-dominant, main.rs is LF. Every anchor is written
with "\n" and rewritten with the file's own NL before matching, so a file's mixture of
endings cannot defeat a match (the failure mode seen with `edit` earlier today).
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "rust" / "rwkv-infer" / "src"
print(f"repo root: {ROOT}")


def load(p: Path):
    text = p.open(encoding="utf-8", newline="").read()
    nl = "\r\n" if text.count("\r\n") >= text.count("\n") - text.count("\r\n") else "\n"
    return text, nl


def apply(p: Path, pairs, label):
    text, nl = load(p)
    print(f"\n== {label} ({p.name}) nl={'CRLF' if nl == chr(13) + chr(10) else 'LF'} ==")
    for i, (old, new) in enumerate(pairs, 1):
        o, n = old.replace("\n", nl), new.replace("\n", nl)
        cnt = text.count(o)
        if cnt != 1:
            print(f"  [{i}] ABORT: anchor found {cnt} times (want 1): {o.strip()[:80]!r}")
            sys.exit(1)
        text = text.replace(o, n)
        print(f"  [{i}] ok: {n.strip().splitlines()[0][:90]}")
    p.open(encoding="utf-8", newline="", mode="w").write(text)
    print(f"  wrote {p.name}")


# ---------------------------------------------------------------- model.rs call sites
MODEL = SRC / "model.rs"
apply(MODEL, [
    # 1. the candle wrapper: no per-entity basis on that path
    ("""    compress_wkv_state(
        &mut data, h, k, rank, factor_qmax, per_col, hadamard, four_level, mixed53, compand, v_qmax,
        als_iters, None, // PQ is fast-path only (like EF); candle stays for parity A/B of the non-PQ paths
        None, // no PQ -> no warm indices
    );""",
     """    compress_wkv_state(
        &mut data, h, k, rank, factor_qmax, per_col, hadamard, four_level, mixed53, compand, v_qmax,
        als_iters, None, // PQ is fast-path only (like EF); candle stays for parity A/B of the non-PQ paths
        None, // no PQ -> no warm indices
        None, // no fast-rank basis: this wrapper is the PARITY REFERENCE, the fast path is engine-only
    );"""),
    # 2. the compress_bench reference() helper -- stays on the reference path on purpose
    ("""        compress_wkv_state(&mut buf, 1, K, R, None, false, false, false, false, None, None, None, None,
                           None);""",
     """        compress_wkv_state(&mut buf, 1, K, R, None, false, false, false, false, None, None, None, None,
                           None, None);"""),
    # 3.+4. the two bench timing calls (rank 1 / rank R)
    ("""                compress_wkv_state(&mut buf, 1, K, 1, None, false, false, false, false, None, None,
                                   None, None, None);""",
     """                compress_wkv_state(&mut buf, 1, K, 1, None, false, false, false, false, None, None,
                                   None, None, None, None);"""),
    ("""                compress_wkv_state(&mut buf, 1, K, R, None, false, false, false, false, None, None,
                                   None, None, None);""",
     """                compress_wkv_state(&mut buf, 1, K, R, None, false, false, false, false, None, None,
                                   None, None, None, None);"""),
], "model.rs")

# ------------------------------------------------------------------------- fast.rs
FAST = SRC / "fast.rs"
apply(FAST, [
    # 1. the new state field
    ("""    pub warm_wkv: Vec<i32>,   // B*H*2 joint-WKV picks ([bi][head][rank_comp]); -1 = none
    pub warm_shift: Vec<i32>, // B*2*m shift-PQ picks ([bi][role][pos]); -1 = none
}""",
     """    pub warm_wkv: Vec<i32>,   // B*H*2 joint-WKV picks ([bi][head][rank_comp]); -1 = none
    pub warm_shift: Vec<i32>, // B*2*m shift-PQ picks ([bi][role][pos]); -1 = none
    /// RWKV_LOWRANK_FAST warm subspace block for the rank>=2 WKV truncation, layout
    /// [bi][head][R+1 columns][K] = B*H*(rank+1)*K floats (empty while that path is off). Travels with
    /// the state exactly like the warm PQ picks, and a length mismatch seeds ZEROS, which the fast path
    /// reads as "cold start" (a zero column has no direction). SPEED ONLY: the reference path ignores
    /// it, and under RWKV_LOWRANK_FAST_CHECK=1 the reference factors are what gets written.
    pub basis: Vec<f32>,
}"""),
    # 2. the return tuple of the stream forward
    ("""            let (e_state, warm_wkv, warm_shift) = if persist {
                self.compress_stream_state(m, b, &mut t_state, &mut t_xshift, &mut c_xshift, e_prev, ls)
            } else {
                (Vec::new(), Vec::new(), Vec::new())
            };
            Ok((
                xc,
                v0_out,
                FastLayerState { t_xshift, t_state, c_xshift, e_state, warm_wkv, warm_shift },
            ))""",
     """            let (e_state, warm_wkv, warm_shift, basis) = if persist {
                self.compress_stream_state(m, b, &mut t_state, &mut t_xshift, &mut c_xshift, e_prev, ls)
            } else {
                (Vec::new(), Vec::new(), Vec::new(), Vec::new())
            };
            Ok((
                xc,
                v0_out,
                FastLayerState { t_xshift, t_state, c_xshift, e_state, warm_wkv, warm_shift, basis },
            ))"""),
    # 3. the return doc line
    ("""new warm_wkv, new warm_shift)""", """new warm_wkv, new warm_shift, new fast-rank basis)"""),
    # 4. the return type
    ("""    ) -> (Vec<f32>, Vec<i32>, Vec<i32>) {""", """    ) -> (Vec<f32>, Vec<i32>, Vec<i32>, Vec<f32>) {"""),
    # 5. seed the basis (inserted after the warm_shift seeding)
    ("""        let mut warm_shift: Vec<i32> = ls
            .map(|s| s.warm_shift.clone())
            .filter(|v| v.len() == want_shift)
            .unwrap_or_else(|| vec![-1; want_shift]);""",
     """        let mut warm_shift: Vec<i32> = ls
            .map(|s| s.warm_shift.clone())
            .filter(|v| v.len() == want_shift)
            .unwrap_or_else(|| vec![-1; want_shift]);
        // RWKV_LOWRANK_FAST: this (entity, stream, layer)'s subspace block. Allocated ONLY while the
        // fast path (or its check) is on AND the stream is a rank>=2 low-rank one that actually reads it
        // (rank 1 never does, and the ALS path keeps the reference solver), so an untouched recipe
        // allocates and copies nothing. Seeded from the entity's previous step, size-checked -- a
        // mismatch (e.g. a state built before this feature) falls back to a cold start.
        let want_basis = match cfg.lowrank.get(&m) {
            Some(&(rank, _))
                if rank.min(k) >= 2
                    && cfg.als.is_none()
                    && (crate::model::lowrank_fast_on() || crate::model::lowrank_fast_check()) =>
            {
                b * h * (rank.min(k) + 1) * k
            }
            _ => 0,
        };
        let mut basis: Vec<f32> = ls
            .map(|s| s.basis.clone())
            .filter(|v| v.len() == want_basis)
            .unwrap_or_else(|| vec![0.0; want_basis]);"""),
    # 6. the EF-compensated state stays cold (like its warm slot)
    ("""                    crate::model::compress_wkv_state(
                        slice, h, k, rank, fqmax, cfg.percol, cfg.hadamard, cfg.four_level,
                        cfg.mixed53, cfg.compand, cfg.vqmax, cfg.als, cfg.pq.as_deref(), None,
                    );""",
     """                    crate::model::compress_wkv_state(
                        slice, h, k, rank, fqmax, cfg.percol, cfg.hadamard, cfg.four_level,
                        cfg.mixed53, cfg.compand, cfg.vqmax, cfg.als, cfg.pq.as_deref(), None,
                        None, // EF stays cold here too (see the warm note above)
                    );"""),
    # 7. the EF error re-compress
    ("""                        crate::model::compress_wkv_state(
                            eb, h, k, er, cfg.ef_elevel, cfg.percol, false, false, false, None, None,
                            None, epq, None,
                        );""",
     """                        crate::model::compress_wkv_state(
                            eb, h, k, er, cfg.ef_elevel, cfg.percol, false, false, false, None, None,
                            None, epq, None,
                            None, // the carried error is a different matrix: no shared basis
                        );"""),
    # 8. the LIVE path: pass this entity's own slice
    ("""            } else {
                for bi in 0..b {
                    let wslice = (!warm_wkv.is_empty())
                        .then(|| &mut warm_wkv[bi * h * 2..(bi + 1) * h * 2]);
                    crate::model::compress_wkv_state(
                        &mut t_state[bi * hkk..(bi + 1) * hkk], h, k, rank, fqmax, cfg.percol,
                        cfg.hadamard, cfg.four_level, cfg.mixed53, cfg.compand, cfg.vqmax, cfg.als,
                        cfg.pq.as_deref(), wslice,
                    );
                }
            }""",
     """            } else {
                let per_head = k * rank.min(k) + k; // the fast block is R+1 columns per head
                for bi in 0..b {
                    let wslice = (!warm_wkv.is_empty())
                        .then(|| &mut warm_wkv[bi * h * 2..(bi + 1) * h * 2]);
                    let bslice = (!basis.is_empty())
                        .then(|| &mut basis[bi * h * per_head..(bi + 1) * h * per_head]);
                    crate::model::compress_wkv_state(
                        &mut t_state[bi * hkk..(bi + 1) * hkk], h, k, rank, fqmax, cfg.percol,
                        cfg.hadamard, cfg.four_level, cfg.mixed53, cfg.compand, cfg.vqmax, cfg.als,
                        cfg.pq.as_deref(), wslice, bslice,
                    );
                }
            }"""),
    # 9. return the updated block
    ("""        (e_new, warm_wkv, warm_shift)
    }""", """        (e_new, warm_wkv, warm_shift, basis)
    }"""),
    # 10. tile it with the rest of the per-entity state
    ("""                            warm_wkv: tile_b1_i32(&ls.warm_wkv, n),
                            warm_shift: tile_b1_i32(&ls.warm_shift, n),
                        });""",
     """                            warm_wkv: tile_b1_i32(&ls.warm_wkv, n),
                            warm_shift: tile_b1_i32(&ls.warm_shift, n),
                            basis: tile_b1(&ls.basis, n),
                        });"""),
], "fast.rs")

# ------------------------------------------------------------------------- main.rs
MAIN = SRC / "main.rs"
apply(MAIN, [
    ("""    let bad = finite::report_total();""",
     """    crate::model::fastpath_report(); // RWKV_LOWRANK_FAST counters + worst relative Frobenius diff
    let bad = finite::report_total();"""),
], "main.rs")

print("\nall call sites wired")
