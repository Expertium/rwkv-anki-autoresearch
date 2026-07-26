# Porting `rwkv-infer` to the track-2 architecture — the gating work

**Why this is now the top item (2026-07-26).** The track-2 width ladder closed (A15 = d=96 /
808,762 params / 3.41×; A16/A17/A18 all rejected — see `optimization/research_5k_verbose.md`),
and `optimization/CPU_INFERENCE.md` showed that in the only engine we can currently measure a
**4.5× arithmetic cut bought just 1.24× wall-clock and plateaued** — param count has decoupled
from the metric Anki users feel. Nothing further in track 2 reaches a user until this engine
runs the champion. Andrew, 2026-07-25: *"I told you to do ablations hoping that fewer params →
faster CPU inference in Anki."*

## What already works (checked, not assumed)

`model.rs` is **already dimension-agnostic**: `load()` derives `c` from `prehead_norm.weight`,
`h` from `k_scale_linear.weight`, `k = c/h`, and `stream_layers` by counting blocks (model.rs
:903-919). The stale "K=32-hardwired" note in CLAUDE.md was superseded by `1d3b5b8`. So d=96 /
H=3 / K=32 needs no work on that axis.

## The five real gaps (all detectable from weight SHAPES — no env flags needed in Rust)

A15's checkpoint (`scratchpad/track2_a15/t2a15d_5586.pth`, 420 tensors) shows the training-side
strips leave **1×1 dummy tensors** behind, which makes auto-detection trivial and robust:

| # | gap | detection | work |
|---|---|---|---|
| 1 | **GRU curve head** replaces `w_linear` | `gru_w_weight` present (2,384); `w_linear.weight` is (1,1) | implement `R(t) = Σ wᵢ·(1 + t/(1e-7+Sᵢ))^(−dᵢ)` with `w = softmax(gru_w·x+b)`, `S = exp(clamp(gru_s·x+b, −25, 25))`, `d = exp(clamp(gru_d·x+b, −25, 25))`; N = `gru_w_weight.dim(0)` (=2) |
| 2 | **stripped channel mixers** | `channel_mixer.W_k.weight` is (1,1) | skip the whole channel-mixer sublayer (identity) for that layer |
| 3 | **stripped L0 v_lora** | `time_mixer.v_lora_simple.A.weight` is (1,1) | skip v0 mixing at that layer (`v0 = v`) |
| 4 | **no ahead residual** | `ahead_linear.weight` is (1,1) | curve = pure mixture; skip the `interp(out_ahead_logits, t)` term |
| 5 | **per-step state clamp** (τ=300, window 32768) | not in weights — recipe flag | apply the same clamp the training/eval path uses, else parity drifts on long histories |

A15's exact strip map (auto-detected, for the parity test): card L0 v_lora, card L1 cmix;
deck L0 v_lora, deck L1/L2 cmix; note L0 v_lora; preset L0 v_lora + L0/L1/L2 cmix;
user L0 v_lora + L0/L1/L2 cmix.

## Order of work

1. **Loader tolerance + auto-detect** — treat 1×1 tensors as "absent"; record per-layer
   `has_cmix` / `has_vlora`; read `num_curves` from `gru_w_weight` when present, else
   `w_linear`. *No math changes yet; the model should load and report its shape.*
2. **GRU head math** in both the candle path (`model.rs`) and the fast path (`fast.rs`).
3. **Skip logic** in the per-layer loop for cmix/v_lora; state clamp.
4. **Parity gate** — extend `verify_rust.py` to the track-2 champion: export
   `t2a15d_5586.pth` → safetensors, run the 3 reference users, require |rust − python| within
   the usual float tolerance. **A15 parity is the definition of done for the port.**
5. **THEN measure** — `optimization/measure_throughput.py` on A0/A14/A15 and compare against
   the Python-path curve in `CPU_INFERENCE.md`. This is the experiment that finally answers
   whether the ablations bought user-visible speed.
6. **Then optimize** (roadmap step 4), in this order of expected value:
   - **AVX2/FMA `dot_product` + `add_scaled_in_place`** — we have none; the reference
     implementation is `vendor/jschoreels_anki/rust/x86_simd.patch` (⚠ AGPL — see that
     directory's `NOTICE.md`; these are ~30 lines of standard intrinsics we can also write
     independently).
   - **Batched GEMM path** — the reference fork routes many per-review matvecs through
     `cblas_sgemm` on macOS (`vendor/.../matmul.rs`, `bulk.rs`) but falls back to scalar on
     x86. An x86 batched path is unclaimed and is the direct fix for the overhead-bound
     profile.
   - Single-thread by default: the CPU bench showed 1 thread beats 3 and 6 on this workload.

## Non-goals for this port

Track-1 deploy extras (PAVA rectifier, GRU_HEAD=3, Muon — training-only) and the q72u
quantization path already in the engine. Keep them untouched; the track-2 model is plain fp32
for now, and quantization is a separate axis that already works.
