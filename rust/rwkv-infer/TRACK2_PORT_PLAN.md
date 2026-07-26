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
| 6 | **PAVA rectifier on the 4 button curves** (added 2026-07-26) | `pava_theta` present (3 floats) | the deploy-side operator, not a training detail — see below |

### Gap 6 in detail — the rectifier is part of the model, not the loss

Andrew, 2026-07-26: *"Eval should score the rectified model, of course"* and *"implement
[it] everywhere: training+eval+CPU inference"*. Until that day the rectifier existed only
inside the training loss, so **the Python eval and the Rust engine both computed a model we
never intended to ship**. Python eval is fixed (`RWKV_EVAL_PAVA=1`); Rust is this gap.

What the engine must do when Anki asks for the four button intervals:

1. Build 4 counterfactual review rows — the current review with the grade one-hot swapped to
   Again/Hard/Good/Easy and **`scaled_duration` set to 0.0**, identical in every other
   respect. (0.0 is the pipeline's "no press yet" encoding, the same value query rows carry;
   it implies ≈7.3 s given `scale_duration(x) = (log(10+x) − 8.9)/1.07`. The old
   `duration_median.json` constant is retired.)
2. Run each through the model to get 4 recall curves. **These are skip rows: they read the
   state and must NOT advance it** (`rwkv7_cuda.cu`: `if (skip) state_xy = in_state_xy`).
3. Apply PAVA pooling-to-tie with the learned junction powers `p = 2·tanh(θ)`, uniform
   weights — port `rwkv/model/pava.py` (~40 lines; the scalar reference
   `pava_rectify_scalar` is the one to translate, and it is trivially correct).
4. Solve each rectified curve for its interval. Guaranteed ordered, because the curves are
   monotone in t by construction (no ahead residual) and now ordered across buttons.
5. The **real** duration enters only the state update after the press — so the displayed
   interval is stable while the user reads it, and displayed == scheduled.

The vendored fork does 1/2/4 and a PAVA at step 3 (`intervals_for_pava_adjusted_samples`,
`simulated_answer_input` varies only the grade) — same design, Andrew's in both places, so
it is a template rather than independent corroboration. Ours differs by having 3 *learned*
powers instead of a fixed arithmetic mean.

**The Python reference now exists** (`11ab7e0`): `SrsRWKVRnn.button_heads` /
`button_curves` / `button_intervals` implement exactly steps 1–5, and
`scratchpad/eval_pava/smoke_rnn_buttons.py` is the property test to mirror in Rust. Port
against that, not against the training path — it is the one place all five steps appear
together. Two things it learned the hard way, both worth copying:

- `pava_theta` must be **in the loaded state dict**, or a PAVA checkpoint cannot be opened
  at all (Python's `load_state_dict` is strict; Rust's loader should likewise refuse a
  model whose `pava_theta` it silently ignored).
- The interval solver must apply the rectifier **at every bisection probe**, since
  rectification couples the four buttons — evaluating a raw curve and rectifying afterwards
  gives a different answer. Cheap regardless: the heads do not depend on `t`, so a probe is
  closed-form arithmetic and the RWKV forward runs exactly 4 times per press.

## Gaps 2, 3 and 5 were PYTHON-RNN gaps too — now closed there (2026-07-26, `921ac76`)

`rwkv/model/rwkv_rnn_model.py` implemented **none** of `RWKV_STRIP_CMIX`,
`RWKV_STRIP_L0_VLORA` or `RWKV_STATE_CLAMP_*` — they lived only in `rwkv_model.py`. Since
`verify_rust.py` gates Rust *against* the Python RNN, that side had to land first; it has,
and `scratchpad/parity3/parity_train_vs_rnn.py` now holds RWKV7 (parallel) against
RWKV7RNN (recurrent) on identical weights at ~1e-6 across seven cases. Three things the
Rust port should copy rather than rediscover:

- `stream_name` must be **stamped onto each stream's config** or `RWKV_STRIP_CMIX` matches
  `":<layer>"` and strips nothing while appearing to comply. Silent, and it makes the
  engine compute a different model than the one trained.
- The strips leave **1×1 dummies**, so the state dict stays symmetric — which is exactly
  the shape-detection the table above relies on.
- **The state clamp cannot be bit-identical to training**, and Rust should not pretend
  otherwise. Training clamps between windows of `state_clamp_window` steps; a
  one-step-at-a-time engine cannot reconstruct those boundaries, since they depend on how
  `prepare_batch` packed the batch. Clamp every step. The two agree exactly wherever
  ‖S‖ ≤ τ (the factor is then exactly 1.0), and differ only on already-diverging states,
  where per-step is the more conservative choice. Note this in the parity gate's tolerance
  story rather than chasing an exact match that does not exist.

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
   **★ DONE 2026-07-26 — `PARITY: PASS` for A18, imm 0.000035 / ahead 0.000044 vs tol 0.0005.**
   The first-ever parity verification of the track-2 architecture, so steps 1–3 are certified.
   Target regenerated as `reference_a18/` (self-contained at exactly 0.000e+00); see CLAUDE.md
   §11 for the exact procedure and the 9.6e-3 max-per-review note. The history below is kept
   because the *reason* it was red is the reusable lesson.
   ⚠ **The gate WAS red and its wiring is a trap (found 2026-07-26).** `verify_rust.py` does **not** run the engine: it scores
   `reference/rust_pred_<user>.json` files that a previous manual run left behind, so
   `RWKV_WEIGHTS` does not influence its verdict at all. The files sitting there were from
   **Jun 30**, left over from the quantization-ladder sweep (note the `_int4_`, `_i2_`,
   `_qatq_` siblings), which is why the gate reported `PARITY: FAIL` with an identical
   `dpred 6.8e-01` across three different crate versions *and* three different weight files.
   The real procedure is: run the binary **from the repo root** (it resolves
   `reference/trace_user_*` relative to CWD) → it writes `preds/rust_pred_*.json` → copy those
   into `reference/` → then score. Done that way with
   `RWKV_WEIGHTS=reference/rwkv_ref_558.safetensors` (the model `reference/ref_metrics.json`
   actually names), the gate still fails, but far less: imm 0.004425, ahead 0.024390 against a
   0.0005 tolerance. **Unresolved:** whether that residual is wrong weights, a stale trace, the
   per-user id-encoding seed scheme, or a genuine engine regression since June. `preds/` and
   `reference/rust_pred_*` are untracked, so nothing was lost by regenerating them.
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

Muon is training-only — nothing ships. The q72u quantization path already in the engine
stays untouched; the track-2 model is plain fp32 for now, and quantization is a separate
axis that already works.

⚠ **The PAVA rectifier and GRU_HEAD are NOT non-goals** (corrected 2026-07-26). An earlier
version of this line listed them as "track-1 deploy extras, training-only", which was
wrong twice over: the GRU head is gap 1 above, and the rectifier is the model's button
ordering guarantee, not a training detail. Both ship. That misfiling is the same error
that let PAVA go unevaluated from iter 23 to iter 30 — see the three-way parity rule in
CLAUDE.md §9.
