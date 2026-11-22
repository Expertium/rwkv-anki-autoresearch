"""One-shot: add a bucket histogram to the fast-rank difference statistic.

`max_relFrobenius_vs_reference` alone cannot answer the deploy question: a single cold-start call (the
first compression of a state has no warm basis) can set the maximum while every warm call is two orders
of magnitude better. The buckets (<1e-3, <1e-2, <1e-1, <1, >=1) show which it is. Counters only, so
the measurement stays observation-only.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
p = ROOT / "rust" / "rwkv-infer" / "src" / "model.rs"
text = p.open(encoding="utf-8", newline="").read()
nl = "\r\n" if text.count("\r\n") >= text.count("\n") - text.count("\r\n") else "\n"


def sub(old, new, label):
    global text
    o, n = old.replace("\n", nl), new.replace("\n", nl)
    if text.count(o) != 1:
        print(f"ABORT [{label}]: anchor found {text.count(o)} times")
        sys.exit(1)
    text = text.replace(o, n)
    print(f"ok [{label}]")


sub("""static FAST_MAX_REL: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);""",
    """static FAST_MAX_REL: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Counts of the recorded differences in buckets (<1e-3, <1e-2, <1e-1, <1, >=1). The MAX alone cannot
/// separate a single cold-start outlier -- the first compression of a state has no warm basis -- from a
/// systematic error, and the deploy decision needs both facts.
static REL_BUCKETS: [std::sync::atomic::AtomicU64; 5] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];""", "bucket statics")

sub("""    let worst = f32::from_bits(FAST_MAX_REL.load(Ordering::Relaxed) as u32);
    eprintln!(
        "[fastrank] mode={}{} iters={} rank>=2 calls={calls} reference fallbacks={fb} \\
         max_relFrobenius_vs_reference={worst:.3e}",
        if lowrank_fast_on() { "fast" } else { "measure-only" },
        if lowrank_fast_check() { "+check" } else { "" },
        fast_iters()
    );""",
    """    let worst = f32::from_bits(FAST_MAX_REL.load(Ordering::Relaxed) as u32);
    let bk: Vec<u64> = REL_BUCKETS.iter().map(|b| b.load(Ordering::Relaxed)).collect();
    eprintln!(
        "[fastrank] mode={}{} iters={} rank>=2 calls={calls} reference fallbacks={fb} \\
         max_relFrobenius_vs_reference={worst:.3e} rel_buckets={}/{}/{}/{}/{}",
        if lowrank_fast_on() { "fast" } else { "measure-only" },
        if lowrank_fast_check() { "+check" } else { "" },
        fast_iters(),
        bk[0],
        bk[1],
        bk[2],
        bk[3],
        bk[4]
    );""", "report line")

sub("""    let rel = num / den;
    if !rel.is_finite() {
        return;
    }""",
    """    let rel = num / den;
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
    REL_BUCKETS[bucket].fetch_add(1, Ordering::Relaxed);""", "bucket bump")

p.open(encoding="utf-8", newline="", mode="w").write(text)
print("wrote model.rs")
