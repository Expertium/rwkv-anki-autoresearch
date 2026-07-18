# State-norm clamp (A3-instability fix) — design + validation record (2026-07-18)

## Diagnosis (probe_a3_nan.py, user 5002, A3 ckpt t2a3d_5586)

- A3's 129 eval NaN-skips are all **large single-segment users** (5002 = 565k rows,
  5064 = 634k, 5150 = 359k; A0's 9501 = 968k). A3 lowered the blow-up horizon from
  ~1M (A0) to ~360k rows.
- **First non-finite site: `rwkv_modules.1.blocks.2.time_mixer.out_group_norm`** —
  the NOTE stream's last layer, i.e. the value normalizing the WKV recurrence output.
  All kernel INPUTS finite and modest (absmax ≤ 30). The explosion is born inside the
  state recurrence.
- **fp32 NaNs at the identical site** → weight-level (A0-class), NOT a bf16 artifact
  (bf16 shares fp32's exponent range; non-finite means the state truly passed ~3e38).
- Growth curve (windowed measurement, tau=1e30): healthy head-state Frobenius norms
  are **~5–30 across ALL layers/streams**; the divergent note-L2 head hits **2.8e3 by
  t=33k, 1.2e6 at 131k, 4e12 at 229k, inf ≈ 295k** — clean exponential, ~100×/window.
  GroupNorm hides arbitrary finite scale downstream, so nothing breaks until the state
  itself overflows.

## Mechanism (`windowed_clamped_wkv` in rwkv/model/rwkv_model.py)

Env: `RWKV_STATE_CLAMP_TAU` (0 = off = **byte-identical**, branch never taken),
`RWKV_STATE_CLAMP_WINDOW` (default 32768), `RWKV_STATE_CLAMP_LOG=1` (measurement mode).

When tau > 0 and a stream's padded per-entity T > window: run the WKV via the
**stateful kernel** (RWKV7_WKV_Stateful, fp32 carried state) in T-windows and
soft-shrink each head's state between windows:

    S *= tau / max(tau, ||S||_F)      # exactly 1.0 while ||S|| <= tau

- `@torch.jit.ignore` free function (autograd-Function tuple returns aren't
  scriptable); calls no submodules (iter-16 rule). CUDA-only.
- Non-finite head-states (belt-and-braces) are zero-reset with a printed warning;
  with tau=300 this never fires (shrink prevents reaching inf).
- Training is structurally untouched at MAX=32768 (chunk T never exceeds one window);
  vals and eval (long segments) get the guard via the same env.
- QAT interaction: streams on the quant_aware_rwkv7 path (card/note in QAT runs) skip
  this branch — clamp inside the per-step reference at QAT time if still needed.

## τ = 300, window = 32768

10× above the healthy max (~30), 10× below the first divergent observation (2.8e3).
Worst observed intra-window growth from a shrunk state: ~×3e6 → mid-window peak ~1e9,
still 29 orders below fp32 overflow. GroupNorm absorbs the transient scale.

## Validation

- **Off-path**: tau=0 → branch untaken, byte-identical by construction.
- **Scripted-forward compile** (user 5003, JIT): passes.
- **Engaged-inert** (tau=1e30, window 4096, user 5003): loss differs from one-shot by
  ~2e-7/2e-6 — sequential-vs-parallel kernel accumulation order, numerics-level. NOT
  bit-exact: engaged streams (T > window) carry ~1e-6-level noise even when the shrink
  never fires. Acceptable; the clamp is part of the candidate recipe it ships with.
- **Divergent users rescued** (tau=300): 5002 → finite 0.3513/0.3352 (13 shrinks,
  0 resets); 5064 → finite 0.4787/0.4663 (two streams' L2 shrinking, 0 resets).

## Deploy note

The Rust RNN equivalent is a per-step (or every-N-steps) version of the same shrink —
trivial; queued with the rest of the deploy contract. The clamp is a guard, not a
cure: the runaway channels remain in the weights. If SHRINK lines ever show up on
ordinary users, escalate to a train-time state-norm penalty as its own gated iteration.

## First consumer

Track-2 A5 bundle (scratchpad/track2_a5/): GRU head + L0-v_lora strip + this clamp,
2,115,359 params (−8.84% vs A4), gate = ratio ≤0.0001/100k both modes vs A4 on full
n=5000. Also retro-available: re-evaling A3's 129 skipped users with the clamp would
complete A3's n=5000 (queued, low priority — A3 is superseded by A5 if A5 passes).
