# Optimization research narrative — ARCHIVE

This file is the verbose research narrative (reasoning, breakthroughs, measured lever costs, dead
ends) that used to live inline in `CLAUDE.md`. It was moved here on 2026-06-29 to declutter the
handover. **The canonical numeric record is `optimization/log.md`** (iteration table + State-
quantization section + QAT section, regenerated from `log.jsonl` / `quant_log.jsonl` / `qat_log.jsonl`).
CLAUDE.md keeps only the current champion, the compact lesson bank, and the active plan; everything
historical/explanatory is here.

---

## Champion lineage & frontier history

**Iteration 0 baseline:** `rwkv_ref_558.pth`, d_model=128, 2,762,884 params, 51.0 KiB/card,
ahead 0.374046 / imm 0.319475, throughput 181.8 rev/s (B=1).

**Lineage to the iter36 champion:** iter21 (209,312 / 0.315078, heads 128) → iter29 (192,800 /
0.312980, heads 64) → iter31 (192,800 / 0.315438, 8.5 KiB) → iter35 (192,800 / 0.316508, 4.25 KiB,
note-grow) → **iter36 (192,800 / imm 0.313864 / ahead 0.347959, 4.25 KiB fp32, deck-grow =
deploy-optimal)**. iter36 = iter31 + card 2→1 compensated by growing the CHEAP card-adjacent **deck**
stream → [1,4,3,3,3]. STRICTLY DOMINATES iter35 (same state/params, better imm + note=3 not 4 =
cheaper deploy). At 4.25 KiB its imm ~matches the 12.75 KiB iter29 (0.312980) — validating Andrew's
ENTITY-COUNT insight: cheap deck-grow beats note-grow for compensating a card cut. State shrank 3×
across the session (12.75 → 8.5 → 4.25 KiB).

**Other frontier points (superseded as champion but on the accuracy/size frontier):**
- **iter23 (best accuracy):** d=64 (N_HEADS=2), [3,3,2,3,3], LoRA 4, channel=1.0, WSD + 4-epoch decay.
  555,324 params (4.98×), state 25.5 KiB, ahead 0.343852 / imm 0.301092, throughput 247.2 rev/s.
  (Supersedes iter16's imm 0.304314 — identical arch, just a longer decay.)
- **iter21/22 (max compression at the time):** d=32 (N_HEADS=1), [3,3,2,3,3], LoRA 16/16/8/16,
  channel=1.0, WSD + 4-epoch decay. 209,312 params (13.2×), state 12.75 KiB, ahead ~0.3486 /
  imm ~0.3152, throughput 244.4 rev/s B=1 but 1.057× FASTER than iter16 under 3-thread load
  (Wilcoxon p=9.77e-4 PASS).
- **iter18 (lean alt):** d=64 card 3→2, 527k params, 17.0 KiB, imm 0.313217.

## The d=32 breakthrough (iters 20–22)

d=32's 13.2× compression is RELIABLE. The earlier d=32 rejection blamed ~0.007 imm variance — but
that was a **decay-length artifact**, not instability. A 4-epoch decay (vs the usual 2) anneals into
a flatter, reproducible, BETTER minimum: 3 independent seeds gave imm 0.315773 / 0.315078 / 0.315237
(spread 0.0007, ~10× tighter than the 2-epoch 0.0070), all passing both gates with ~0.006 margin.
d=32 is the d_model FLOOR (K=32 must stay for the kernel). KEY FINDINGS: (1) optimal LoRA rank FLIPS
with d_model (d=64 over-capacity → cut LoRA to 4; d=32 starved → raise LoRA to 16); (2) the LONGER
4-epoch decay is a GENERAL win — the old 2-epoch default was too short: it both tightens d=32's
variance AND lowers d=64's imm by −0.0032 (iter16 0.304314 → iter23 0.301092).

## Param-reduction lever menu (measured costs)

Champion param breakdown (run `scratchpad/param_breakdown.py` on the 209,312-param arch): RWKV blocks
145,628 (69.6%) · `ahead_linear` 16,512 (7.9%, 128×128 SRS curve head) · `w_linear` 16,512 (7.9%) ·
`features2card` 16,288 (7.8%, 92→128→32 input FC) · small heads ~14k.

- **SRS-head width** — ✅ DONE (iter29): `num_curves`/`num_points` 128→64, −16,512 params AND improved
  both LogLosses (resolution was over-provisioned). 64→48/32 possible but diminishing.
- **card stream 3→2 + grow ungated stream** — ✅ DONE (iter31): card→note rebalance, state 12.75→8.5
  KiB at constant params. card→user (iter32) was WORSE; card 3→2 alone (iter30) passed but burned budget.
- **card 2→1** — ✅ DONE (iter36): 8.5→4.25 KiB, compensated by deck-grow.
- **FC/head inner width** (`head_fc_mult`=4) — ❌ FAILED (iter33): 4→2 gave −12% params but imm +0.0526
  CATASTROPHIC (ahead robust). The 4×d_model WIDTH is critical capacity for the imm path (w_head
  curve-mixture + p_head rating). Keep 4. Maybe-surgical: cut ONLY features_fc (input encoder).
- **note 3→2 layer-cut** — ❌ FAILED (iter38, [1,5,2,3,3] +0.0018 imm): shrink note STATE via quant, not layers.
- **LoRA ranks** (champion 16/16/8/16 every stream): cut on over-provisioned streams (no state change).
- **STATE via K<32 is BLOCKED:** the CUDA training kernel hardwires K=32 (`rwkv7_cuda.cu` `const int
  K=32`, shared-mem `32*(32+1)`, `dim3 block_dim(32,32)`, warp-shuffle `offset=16` = one 32-lane warp
  per state row). `reference_rwkv7` (`rwkv_ops.py`) IS K-agnostic but is a Python `for t in range(T)`
  loop — far too slow to train (T up to 66000). AND a K-split (K=16/H=2) shrinks STATE but NOT params
  (projections are C×C regardless of K). So K<32 needs a real CUDA-kernel rewrite — defer.

## State quantization (PTQ on the iter36 champion) — full results

Validated on the 17 smallest of users 101-200 (full RNN export of the larger users is infeasible —
all huge: min 5229, median 33k reviews → built a FAST feature-only exporter
`scratchpad/export_features_fast.py`, bit-identical). Engine: `RWKV_STATE_QUANT_SCOPE` takes per-stream
mixed bits, e.g. `card:int4,note:int8`; levels int8/int4/int2. Deltas vs fp32 rust baseline (budget +0.0015):
- card int8 +0.000002 (card 1.06 KiB) | card int4 +0.000355 (card 0.53 KiB) | card+note int8 imm
  +0.000118 / ahead +0.000217 (note 3.19 KiB) | card int4+note int8 +0.000470/+0.000577 (0.53+3.19 KiB;
  worst user 9528 16.7 GB→3.6 GB) | card int4+note int4 +0.003569/+0.005360 (passes iter0 but >2× budget
  → NOTE INT4 via PTQ REJECTED) | card int2 (ternary, 0.27 KiB) +0.001249 imm (passes but 83% of budget;
  int4→int2 ~4× the cost for half the card state → int4 = card sweet spot for PTQ) | card int2+note int8
  +0.001319/+0.000669.
- **RULE: quant aggressiveness ∝ 1/recurrence-length** — card (short recurrence) tolerates int4/maybe
  int2; note (medium) wants int8; deck/preset/user (long) stay fp32.
- The ALL-STREAMS blanket version FAILS (int8 imm +0.0025 over budget, int4 +0.093 catastrophic) — sunk
  by the long-recurrence user/global streams. Scoped card/card+note quant is the win.
- Weight PTQ int8/int4 (iter27/28): accuracy fine but NO speed win (B=1 ≈ fp32; ~3× SLOWER under
  multi-stream/bulk load via candle QMatMul rayon oversubscription); file size not a priority. REJECTED.

## Batching (2026-06-29) — full results

Batched single-step QUERY forward as SEPARATE `*_batched` fns in `model.rs` (B=1 untouched). PARITY
PASS: `verify_rust.py` bit-exact (B=1, LL diff 0.000000); `--verify-batched` users 107/136/156 (real
B=738..1014) batched==B=1 within 1.8e-7. Speed-vs-RAM Pareto (`scratchpad/sweep_pareto.py` →
`pareto_speed_ram.png`/`pareto_data.csv`, synthetic states): throughput climbs near-free in RAM to a
KNEE at B=128 (~11,300 rev/s, 29 MB) ≈20× batched-B=1, ≈40× true B=1 single-query, then BACK-BENDS
(B=512 8244 / B=2048 7255 = dominated). Thread count IRRELEVANT (`thread_sweep.py`: RAYON 1..32 ×
B 16..2048 all overlap; B=128 = 11.2–11.7k regardless) — the K=32 matmuls don't parallelize, so the
back-bend is an INTRINSIC L2/L3 CACHE cliff (B=128→256), NOT oversubscription. Single-threaded deploy
loses nothing. Recommend batch ~128, single-thread, NOT the fork's 512. Deploy ref: JSchoreels/anki
fork batches per-user QUEUE SCORING = batched single-step (B,C) query forward over B cards (default
512) from warmed states (NOT a replay); state-quant payoff loads right here. See BATCHING_PLAN.md.

## QAT integration + iter39/40 results — full narrative

**Integration (2026-06-29):** solved the "no per-step state" obstacle — in training each stream
reshapes to `(-1, sub_len, d_model)` so the CARD/NOTE streams are SHORT per-entity sequences; a
per-step fake-quant reference loop over them is cheap (spike: B=5000×T=30 = 33ms fwd, GPU). Engine:
`rwkv/model/rwkv_ops.py::quant_aware_rwkv7` + `fake_quant_state` (per-(B) amax over H,K,K, STE gradient
— matches Rust `quant_roundtrip_batched`). Wired via `state_qmax` on RWKV7Config/RWKV7TimeMixer
(default inf=off → fast kernel; ≠inf → quant-aware per-step path). `architecture.py` reads
`RWKV_QAT_SCOPE="card:int2,note:int4"` (mirrors Rust scope). `RWKV_NO_JIT=1` disables torch.jit (in
rwkv_model.py + srs_model.py) so the quant loop runs as plain Python. DEFAULT path verified UNCHANGED
(still JIT-scripts, 192,800 params, eval byte-for-byte identical; quant branch only compiled, not taken).

**iter39 (decay-only QAT from iter36 WS-final, card int2 + note int4) — QAT WINNER:** champ_fp32 imm
0.296064 / ahead 0.326631; qat_fp32 imm 0.298520 / ahead 0.327760; qat+deploy-quant imm 0.298538 /
ahead 0.327633. KEY: pure quant cost on the QAT'd model = +0.000018 imm (PTQ card int4+note int4 was
+0.003569; int2 worse) → QAT fully dissolves the quant penalty in BOTH modes. COST: the decay-only
fine-tune raised fp32 by +0.002456 imm, so deploy is +0.0025 vs champ fp32 — but at card int2+note int4
= 0.27+1.59 KiB (≈3× smaller than PTQ's recommended card int4+note int8 0.53+3.19). Hits Andrew's IDEAL
config within the iter0 gate. Weights `reference/rwkv_iter39_124.safetensors`. Tooling:
`scratchpad/{qat_spike,qat_eval,pth_to_sft}.py`, `scratchpad/run_qat_eval.sh`.

**iter40 (full WS+QAT FROM SCRATCH) — REJECTED (negative result):** deploy imm 0.310306 / ahead
0.338650 = +0.0118 WORSE than iter39. fp32 ft-regress jumped to +0.0098 imm AND the quant cost itself
rose to +0.0045 (vs ~0 for iter39). Training from RANDOM INIT with int2/int4 quant noise from step 0
(high LR) converges to a much worse minimum. **LESSON: QAT must WARM-START from a good fp32 checkpoint
(champion), NEVER from scratch.**

## Creative / non-standard ideas (seed list — extend freely)

Hitting 1 KB state + further param cuts needs INVENTED methods, not textbook PTQ:
- **per-persist (not per-step) state quant** — Anki keeps state fp32 in memory during a session and
  quantizes ONLY when persisting; the drift is far milder than the per-step round-trip.
- **low-rank / factored card WKV state (★ evidence-backed, but NOT a free memory win — see math)** — store
  S (K×K) as U·Vᵀ (rank r≪K) → 2Kr floats vs K². The `--dump-card-state` 32×32 grid showed the card state is
  near rank-1 (every row a scalar multiple of one column-pattern — the S=Σv·kᵀ outer-product structure).
  **MEMORY MATH (Andrew 2026-06-29):** rank-1 fp32 = 64 floats × 4 B = **256 B = EXACTLY int2-full
  (1024 × 2 bit)** — a TIE, not cheaper (an earlier note here wrongly said "cheaper than int2"). rank-2 fp32
  (512 B) LOSES to int2. Pure-fp32 low-rank CANNOT beat int2 on memory (rank-1 is the 64-float floor). To go
  UNDER int2 you must quantize the FACTORS: rank-1 int8 = 64 B (4× below int2), rank-1 int4 = 32 B (8×) — but
  this stacks low-rank error × quant error. The token-shifts (64 fp32 floats = 256 B) also become the floor
  once WKV shrinks below them. So: low-rank's memory win = ONLY rank-1 + int8/int4 factors; its fp32-only value
  is accuracy-at-equal-bytes (no quant noise), which QAT largely already captured. Recurrence adds rank each
  step → re-factor (truncated SVD) each persist, or keep a fixed-rank running approx; measure on the 2k loop.
- mixed-precision keeping only outlier channels fp32 (RWKVQuant proxy idea); product/vector quantization
  of the state with a tiny learned codebook; learned state *compression* head (autoencoder bottleneck on
  the persisted state); structured pruning of dead channel-mix/LoRA dims; weight-sharing/tying across
  layers; non-uniform (log/μ-law) state quant matched to the WKV value distribution; the K<32 kernel route.
- RWKV-edge (2412.10856v4, `scratchpad/rwkvedge.txt`): SVD low-rank on W_r/k/v/o, FFN sparsity (cuts
  params, not state). The 1-bit FFN-activation-predictor trick (eq.4-5) does NOT fit our tiny 32-neuron
  FFN (paper's own small-model caveat; our bottleneck is WKV not FFN) — but its STATIC analog does:
  cut `channel_mixer_factor` 1.0→0.5.

## GPU-training & low-rank-gate speedups (step 3, 2026-06-29) — full narrative

NEW PHASE PLAN step 3 = "maximally speed up GPU training + the low-rank gate," arch-agnostic, untimed
(GPU training speed doesn't gate per protocol). Profiled with `scratchpad/profile_train.py` (mirrors
`train_rwkv.main_loop`'s per-step body, fetches batches SYNCHRONOUSLY via `get_data`+`prepare` — the
async multiprocessing fetcher hangs in a scratch script on Windows spawn; caches N real batches on-device,
times a per-section sync breakdown + an old-vs-new end-to-end body A/B).

**Per-section breakdown (no-JIT, full 31-group workload, ms/step):** copy_downcast_ 22.9, fwd 91.8,
**bwd 200.1 (≈50%)**, transfer_grad 43.3, grad_norm 27.9, clip+opt 11. The 444-param-tensor model spends
**copy+transfer+grad_norm = 94 ms (24%)** in per-param PYTHON LOOPS (launch-bound) + logging syncs; the
rest (fwd+bwd ≈ 290 ms, 73%) is the custom WKV CUDA kernel running at B=1 / K=32 / H=1 LOW PARALLELISM
over long sequences (the user_id stream T up to 20k) — that's COMPUTE/latency-bound (one 32×32 state
evolving sequentially), not launch-bound, and is what makes GPU util ~15%. So the earlier "launch-bound"
read was only half right: the per-param loops + syncs were launch-bound (fixable), the kernel is parallelism-
starved (not fixable without kernel/batch changes).

**Wins implemented (all bit-identical / arch-agnostic):**
1. **`torch._foreach_*` vectorization** of `copy_downcast_` (srs_model.py) and `transfer_child_grad_to_master`
   (train_rwkv.py): group params by dtype, one fused `_foreach_copy_` / `_foreach_add_`+`_foreach_zero_` per
   group instead of ~440 per-param launches each. `copy_`/`add_` cast, so == the per-param loop bit-for-bit
   (proved by `scratchpad/test_foreach_correct.py`: 0 mismatches).
2. **Skip logging-only syncs when `USE_WANDB` is off** (every iter config): `get_grad_norm` does ~440
   `.item()` D2H syncs/step (~28 ms, drains the pipeline) purely for `log["norm"]` → wandb; `log_model`
   similar on validate steps. Gated behind `config.USE_WANDB`.
   → (1)+(2) = **2.53 → 3.07 steps/s (+1.21x)** no-JIT, full workload (8-batch subset agreed: 2.70→3.27).
   The ~65 ms saving is FIXED per step (depends on param count, not work) → larger relative win on cheaper
   short-sequence steps, so 1.21x is a lower bound on the average.
3. **JIT RESTORED via `@torch.jit.ignore` on `quant_aware_rwkv7`.** Discovered JIT (TorchScript) was SILENTLY
   BROKEN in torch 2.12.1+cu130: building the model without `RWKV_NO_JIT` throws an internal assert
   (`outputs_[i]->uses().empty()`) while scripting `RWKV7TimeMixer.forward`. Root cause = the recently-added
   QAT branch calls `quant_aware_rwkv7`, whose per-step loop + `torch.linalg.svd` (in `fake_lowrank_state`)
   isn't TorchScript-able; the scripter compiles ALL branches even though this one never runs under JIT
   (state_qmax=inf, lowrank_rank=0 defaults). This would have CRASHED any plain WS/decay training AND
   `get_result.py` eval (both JIT-on). Marking `quant_aware_rwkv7` `@torch.jit.ignore` (+ a `-> Tensor`
   annotation) makes the scripter treat it as an opaque Python call → the hot kernel path scripts again;
   eager (RWKV_NO_JIT) QAT path is unchanged (`jit.ignore` is transparent in eager; verified off-path ==
   reference). JIT mainly cuts Python dispatch overhead (can't touch the custom kernel), worth ~6–13% on
   the pipelined step. **Combined JIT-on + foreach + sync-removal = 3.48 steps/s = 1.38x over no-JIT old
   body, 1.30x over JIT-on old body.** ⚠ JIT has a ~30–60 s one-time compile → net win only on LONG runs
   (the 1k-user phase); for short 100-user iters JIT is ~neutral and the unconditional foreach/sync win is
   what counts.
4. **`torch.compile` ruled out:** no Triton wheel on Windows (`triton` import fails) → inductor unusable.
   JIT was the only fusion route on this machine, and it's now fixed. CUDA graphs not pursued (variable seq
   shapes need bucketing + the custom autograd.Function complicates capture; the dominant kernel wouldn't shrink).

**Low-rank gate speedup (the second half of step 3):** `lowrank_roundtrip` (Rust) replaced nalgebra's FULL
SVD — which converges pathologically slowly on the real near-low-rank states (singular values 3–32 ≈ 0;
Golub-Kahan grinds on the clustered tiny values; user 187 hung >35 min) — with a top-r truncation via the
**Gram matrix + symmetric eigendecomposition** (eigvecs of A Aᵀ = left singular vecs, eigvals = σ²; right
vec v = Aᵀu/σ). Symmetric eigensolvers have NONE of that slow-convergence pathology. A is normalized by its
max-abs before forming the Gram (the squaring overflows f32 for a state grown large over a long history →
NaN eigenvalues → a panic at review ~4000 on the first try; normalize then unscale σ = scale·√eig fixes it),
plus a NaN-safe sort and skip-non-finite-component guard. Validated == full-SVD rank-2 recon to ~1e-15 in
numpy (`scratchpad/analyze_card_rank.py` style check). RESULT: user 187 both-low-rank = **22 s** (was >35 min
hang); full 17-user both-low-rank gate ~100 s. **note-low-rank is now PRACTICAL in the iteration loop.**
Both-low-rank deploy re-confirmed on ALL 17 users incl 187: **imm 0.288831 / ahead 0.320098, −0.0072 imm /
−0.0065 ahead vs champ_fp32, GATE PASS** (prior 0.289137 was 16 users; 187's low-rank deploy is fine — the
hang/panic was purely SVD numerics, NOT state divergence).
