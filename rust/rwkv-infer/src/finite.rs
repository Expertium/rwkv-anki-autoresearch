//! Non-finiteness and blow-up observation for the DEPLOY engine (Andrew, 2026-11-22).
//!
//! WHY THIS EXISTS. The engine silently HIDES a blow-up: `san()` in `main.rs` maps a non-finite
//! p(recall) to 0.5 (its own comment says the int2 recurrence "can transiently blow up to inf on some
//! power users"), and `compress_wkv_state` skips degenerate components. So a state that has gone
//! non-finite looks like an ordinary max-entropy prediction instead of a failure. The Python-side
//! corruption gate (`scratchpad/quant_plan/corruption/finite_probe.py`) answers that question for the
//! TRAINING path; this module answers it for the path that actually ships.
//!
//! WHAT IT COUNTS. Per user, on the compressed streams:
//!   * `state_vals` / `state_raw_bad`  - the fresh WKV state values entering one compression, and how
//!     many of them are non-finite (this is the recurrence blowing up, before any quantization);
//!   * `state_quant_bad`              - non-finite values in the compressed (dequantized) state that
//!     gets written back and persisted (this is quantization producing garbage);
//!   * `compressions`                 - compression calls seen (one per stream per step per entity);
//!   * `vec_raw_bad` / `vec_quant_bad`- the same pair for the token-shift vectors handled by
//!     `quant_vec_inplace` (also used by the full-matrix state quant);
//!   * `p_vals` / `p_bad`             - every p(recall) value passed through `san()`, and how many were
//!     non-finite (the number that `san()` would otherwise turn into 0.5);
//!   * `state_maxabs` / `vec_maxabs`  - the largest |value| seen in the state and in the token-shift
//!     vectors, kept APART because the two live on different scales (the reference users read a state
//!     max-abs near 0.35 and a shift max-abs near 5.2). This is the state-GROWTH signal.
//!
//! COST. Everything is behind `RWKV_FINITE_CHECK`, read once into a `OnceLock`, so a normal run pays a
//! single predictable branch per call and no per-element scan.
//!
//!   RWKV_FINITE_CHECK unset / "0" / "off" -> off (default)
//!   RWKV_FINITE_CHECK=1 / "true"         -> count and report per user + a total line
//!   RWKV_FINITE_CHECK=strict             -> the same, and exit code 3 if ANYTHING was non-finite
//!
//! The Python driver is `scratchpad/quant_plan/rust_finite.py`. Parity note: this module only OBSERVES;
//! it never changes a value, so a run with the check on is bit-identical to one with it off.

use std::sync::atomic::{AtomicU64, Ordering};

const OFF: u8 = 0;
const COUNT: u8 = 1;
const STRICT: u8 = 2;

static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();

static STATE_VALS: AtomicU64 = AtomicU64::new(0);
static STATE_RAW_BAD: AtomicU64 = AtomicU64::new(0);
static STATE_Q_BAD: AtomicU64 = AtomicU64::new(0);
static COMPRESSIONS: AtomicU64 = AtomicU64::new(0);
static VEC_VALS: AtomicU64 = AtomicU64::new(0);
static VEC_RAW_BAD: AtomicU64 = AtomicU64::new(0);
static VEC_Q_BAD: AtomicU64 = AtomicU64::new(0);
static P_VALS: AtomicU64 = AtomicU64::new(0);
static P_BAD: AtomicU64 = AtomicU64::new(0);
/// f32::to_bits() of a NON-NEGATIVE finite float is monotone in value, so a plain atomic max works.
/// STATE and VEC are tracked APART on purpose: on the reference users the Python probe reads a state
/// max-abs near 0.35 while the token-shift vectors reach 5.2, so one mixed maximum is misleading - it
/// reports the shift vectors and hides whether the recurrent state itself is growing.
static STATE_MAXABS: AtomicU64 = AtomicU64::new(0);
static VEC_MAXABS: AtomicU64 = AtomicU64::new(0);
/// A deliberately planted non-finite value (RWKV_FINITE_SELFTEST=1). It exists so the STRICT path can be
/// END-TO-END tested on a healthy model: a safety check that has never been seen to fire is not a check.
/// It is counted separately from the per-user counters (which reset per user) so it survives the loop.
static SELFTEST_BAD: AtomicU64 = AtomicU64::new(0);

pub fn mode() -> u8 {
    *MODE.get_or_init(|| {
        let v = std::env::var("RWKV_FINITE_CHECK").unwrap_or_default();
        match v.trim().to_ascii_lowercase().as_str() {
            "" | "0" | "false" | "off" => OFF,
            "strict" => STRICT,
            _ => COUNT,
        }
    })
}

#[inline]
pub fn on() -> bool {
    mode() != OFF
}

#[inline]
pub fn strict() -> bool {
    mode() == STRICT
}

pub fn announce() {
    if on() {
        eprintln!(
            "[finite] check ON ({}) - counting non-finite state, quantized-state, shift-vector and \
             p(recall) values; the run is otherwise unchanged",
            if strict() { "strict" } else { "count" }
        );
        selftest();
    }
}

/// RWKV_FINITE_SELFTEST=1 plants one non-finite count so the STRICT exit path can be verified.
fn selftest() {
    let armed = std::env::var("RWKV_FINITE_SELFTEST")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    if armed {
        SELFTEST_BAD.fetch_add(1, Ordering::Relaxed);
        eprintln!("[finite] SELFTEST: one non-finite value planted on purpose (strict mode must exit 3)");
    }
}

#[inline]
fn bump_maxabs(slot: &AtomicU64, x: f32) {
    let bits = x.abs().to_bits() as u64;
    let mut cur = slot.load(Ordering::Relaxed);
    while bits > cur {
        match slot.compare_exchange_weak(cur, bits, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => return,
            Err(seen) => cur = seen,
        }
    }
}

/// Fresh WKV state entering one compression: the "did the recurrence blow up" count.
pub fn state_raw(s: &[f32]) {
    if !on() {
        return;
    }
    COMPRESSIONS.fetch_add(1, Ordering::Relaxed);
    STATE_VALS.fetch_add(s.len() as u64, Ordering::Relaxed);
    for &x in s {
        if x.is_finite() {
            bump_maxabs(&STATE_MAXABS, x);
        } else {
            STATE_RAW_BAD.fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// Compressed state written back: the "did quantization produce garbage" count.
pub fn state_quant(s: &[f32]) {
    if !on() {
        return;
    }
    for &x in s {
        if !x.is_finite() {
            STATE_Q_BAD.fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// A vector about to be quantized by `quant_vec_inplace` (token shifts, or a full-matrix state).
pub fn vec_raw(v: &[f32]) {
    if !on() {
        return;
    }
    VEC_VALS.fetch_add(v.len() as u64, Ordering::Relaxed);
    for &x in v {
        if x.is_finite() {
            bump_maxabs(&VEC_MAXABS, x);
        } else {
            VEC_RAW_BAD.fetch_add(1, Ordering::Relaxed);
        }
    }
}

pub fn vec_quant(v: &[f32]) {
    if !on() {
        return;
    }
    for &x in v {
        if !x.is_finite() {
            VEC_Q_BAD.fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// A p(recall) value on its way into `san()` - the value `san()` would silently rewrite to 0.5.
pub fn p(v: f32) {
    if !on() {
        return;
    }
    P_VALS.fetch_add(1, Ordering::Relaxed);
    if !v.is_finite() {
        P_BAD.fetch_add(1, Ordering::Relaxed);
    }
}

fn take(c: &AtomicU64) -> u64 {
    c.swap(0, Ordering::Relaxed)
}

/// Per-user line. Counters are READ AND RESET, so each line describes that user alone.
pub fn report_user(u: i64) {
    if !on() {
        return;
    }
    let sv = take(&STATE_VALS);
    let srb = take(&STATE_RAW_BAD);
    let sqb = take(&STATE_Q_BAD);
    let ncomp = take(&COMPRESSIONS);
    let vv = take(&VEC_VALS);
    let vrb = take(&VEC_RAW_BAD);
    let vqb = take(&VEC_Q_BAD);
    let pv = take(&P_VALS);
    let pb = take(&P_BAD);
    let sm = f32::from_bits(take(&STATE_MAXABS) as u32);
    let vm = f32::from_bits(take(&VEC_MAXABS) as u32);
    eprintln!(
        "[finite] user={u} compressions={ncomp} state_vals={sv} state_raw_bad={srb} \
         state_quant_bad={sqb} vec_vals={vv} vec_raw_bad={vrb} vec_quant_bad={vqb} \
         p_vals={pv} p_bad={pb} state_maxabs={sm:.6e} vec_maxabs={vm:.6e}"
    );
}

/// Total line; returns the number of non-finite values seen in this user's slice of the run.
pub fn report_total() -> u64 {
    if !on() {
        return 0;
    }
    let sv = take(&STATE_VALS);
    let srb = take(&STATE_RAW_BAD);
    let sqb = take(&STATE_Q_BAD);
    let ncomp = take(&COMPRESSIONS);
    let vv = take(&VEC_VALS);
    let vrb = take(&VEC_RAW_BAD);
    let vqb = take(&VEC_Q_BAD);
    let pv = take(&P_VALS);
    let pb = take(&P_BAD);
    let st = SELFTEST_BAD.load(Ordering::Relaxed);
    let bad = srb + sqb + vrb + vqb + pb + st;
    let sm = f32::from_bits(STATE_MAXABS.load(Ordering::Relaxed) as u32);
    let vm = f32::from_bits(VEC_MAXABS.load(Ordering::Relaxed) as u32);
    eprintln!(
        "[finite] TOTAL compressions={ncomp} state_vals={sv} state_raw_bad={srb} \
         state_quant_bad={sqb} vec_vals={vv} vec_raw_bad={vrb} vec_quant_bad={vqb} \
         p_vals={pv} p_bad={pb} state_maxabs={sm:.6e} vec_maxabs={vm:.6e} \
         planted={st} -> non_finite_total={bad}"
    );
    bad
}
