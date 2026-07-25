# Vendored from JSchoreels/anki — READ-ONLY REFERENCE

**Source:** https://github.com/JSchoreels/anki (branch `main`, fetched 2026-07-25) — an
unofficial Anki desktop fork that ships the **old (upstream 2.76M-param) RWKV** as a live
scheduler, with both a Python inference package and a full Rust engine inside `rslib`.

**Why it's here (Andrew's directive, 2026-07-25):** *"check what functions he has for
calculating card/note/deck/preset/global states and p(recall), copy them. We'll need them
eventually anyway, for iterations of speeding up CPU inference."* This is the only known
production-shaped CPU implementation of this model family, so it is the natural reference
for our roadmap step 4 (speed) and step 6 (CPU-only inference). See `INDEX.md` for the
function map.

## ⚠ LICENSE — AGPL-3.0-or-later

Anki and this fork are **GNU AGPL v3 or later** (the file headers say so explicitly:
`Copyright: Ankitects Pty Ltd and contributors / License: GNU AGPL, version 3 or later`).
Consequences to respect:

- These files are kept **unmodified**, in this clearly-marked vendor directory, for
  reference. Nothing here is imported or compiled by our code.
- If we **copy code** from here into `rust/rwkv-infer` or anywhere we distribute, that work
  becomes AGPL-derived and must be released under AGPL-3.0-or-later with attribution. That
  is fine for the end goal (the model ships *inside* Anki, which is AGPL anyway) but it is a
  deliberate licensing decision, not an accident — flag it to Andrew before shipping any
  copied block, and keep provenance comments on copied functions.
- Prefer **reimplementing from the documented behaviour** where the code is simple (e.g. the
  AVX2 helpers are ~30 lines of standard intrinsics); copy verbatim only where matching
  numerics matter.

## What was fetched

| file | size | what |
|---|---|---|
| `rust/mod.rs` | 302 KB | the whole Rust RWKV engine: weights, states, streams, heads, batching, state compression |
| `rust/bulk.rs` | 47 KB | bulk/replay path (many reviews at once) |
| `rust/matmul.rs` | 1.9 KB | macOS-only BLAS (Accelerate `cblas_sgemm`) wrapper |
| `rust/rwkv.rs` | 43 KB | `rslib/src/scheduler/rwkv.rs` — scheduler-side glue |
| `rust/rwkv_bench.rs`, `rust/rwkv_predict_bench.rs` | 35 + 30 KB | standalone CPU benchmarks |
| `rust/x86_simd.patch` | 251 lines | the `codex/rwkv-x86-simd` branch diff (AVX2/FMA `dot_product` + `add_scaled_in_place` with runtime detection, scalar fallback, equivalence tests) |
| `python/*.py` | ~43 KB | `qt/aqt/rwkv_inference/` — the PyTorch RNN inference package actually shipped in the fork |

Not fetched: the 11 MB `.pth`/`.bin` weight blobs (we have our own), the Qt/TS UI, and the
test suites (`qt/tests/test_rwkv_*.py`, 370 KB+) — fetch on demand if we need their fixtures.
