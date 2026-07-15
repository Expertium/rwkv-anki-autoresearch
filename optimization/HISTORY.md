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


---

## CLAUDE.md optimization-state snapshot (archived 2026-06-30 tidy)

Verbatim copy of the `## Optimization state` section as it stood in CLAUDE.md before the 2026-06-30 declutter. Superseded plans (NEW PHASE PLAN, deck/preset-grow RESUME, stateful-BPTT ROUTE-R narrative, step-4 groundwork, old active agenda) and the full iter36/iter45 champion lineage live here now; CLAUDE.md keeps only the current champion + gate + compact lesson bank.

## Optimization state (steps 4-5-7)

> Full numeric record = `optimization/log.md` (iteration table + State-quant + QAT sections, rebuilt
> from `log.jsonl`/`quant_log.jsonl`/`qat_log.jsonl`). Verbose research narrative (frontier history,
> breakthroughs, measured lever costs, dead-end details) = `optimization/HISTORY.md`. This section
> keeps only the current state, the compact lesson bank, and the active agenda.

**Iteration 0 baseline:** d_model=128, 2,762,884 params, 51.0 KiB/card, ahead 0.374046 / imm 0.319475.
**Gate ceilings (iter0 + 0.0015):** imm <= 0.320975, ahead <= 0.375546. Review count = 6,164,115 (must
be identical every iter). Gates are vs iter0 (a FLOOR), not vs the champion.

**CHAMPION (fp32 arch) = iter36** `[1,4,3,3,3]` (card,deck,note,preset,user), d=32 / K=32 / H=1,
**192,800 params** (14.3x smaller than iter0), per-card state **4.25 KiB fp32**, ahead 0.347959 /
imm 0.313864, throughput 285.6 rev/s (B=1). Rust-parity PASS (bit-exact). Restore =
`optimization/arch_snapshots/arch_iter36.py`.

**DEPLOYED CHAMPION = iter45 weights + LOW-RANK card deploy** (PTQ, no retrain): **card rank-2 int4 factors
(lowrank) + note int2**, with shifts quantized. deploy imm **0.291471** / ahead 0.323603, **deployed state =
card 0.094 KiB (96 B) + note 0.80 KiB** -- BOTH hard targets MET. Gate PASS; BEATS the fp32 champion (imm
-0.004593, ahead -0.003028). Engine: `RWKV_STATE_LOWRANK_SCOPE=card:2:int4 RWKV_STATE_QUANT_SCOPE=note:int2
RWKV_QUANT_SHIFTS=1` on `reference/rwkv_iter45.safetensors`. ★ KEY: low-rank rank-2 int4 is SMALLER (0.27->0.094
KiB) AND MORE ACCURATE than card int2 full (int2 coarsely quantizes all 1024 WKV floats; rank-2 keeps the top-2
SVD comps in int4 = 98.7% energy). Prior all-int2 champion (card int2+note int2): honest deploy imm 0.295833
(shifts int2) / 0.292560 (shifts fp32) -- superseded by low-rank. Same iter45 weights (16-epoch decay-QAT).
**Meets the >=2x note target** (note int2 0.80 KiB). iter44 (8ep) is ~tied (imm 0.295436 / ahead 0.323291 --
better ahead, worse imm); both saved. KEY QAT LESSON: the original 4-epoch decay-QAT (iter43) was UNDERTRAINED;
deploy-imm by decay length = 4ep 0.299469 / 8ep 0.295436 / 16ep 0.292560 (gains shrink -0.0040 -> -0.0029, and
ahead crosses over at 16ep) -> STOP epoch-scaling at ~16. FINDING: the fp32 BASE keeps improving with more
decay (qat_fp32 imm 0.296064->0.292454->0.287818 at 4/8/16ep) => iter36's 2-epoch decay was undertrained; a
longer PLAIN (non-QAT) decay would likely improve the real eval100 benchmark -- revisit as a base improvement.
Lineage: iter39 (int2/int4, +0.0025) -> iter43 (int2/int2, +0.0034) -> iter44 (8ep, -0.0006) -> iter45 (16ep,
-0.0035, champion). PTQ could not reach even card int4+note int4 (+0.0036).

**★ CORRECTION + new low-rank work (2026-06-29, re-derived after the low-rank Rust code was lost from the
working tree and rebuilt this session -- card rank-2 int4 PTQ re-measured at EXACTLY 0.291471, validating it):**
- **iter43/44/45 were NOT real QAT.** The restored champion `architecture.py` (arch_iter36) was MISSING the
  `[QAT]` scope parser that arch_iter41/42 have, so `state_qmax` stayed inf -> fake-quant never ran (the
  iter44/45 logs have ZERO `[QAT]` lines). They were plain LONGER-DECAY fine-tunes + PTQ int2 at the gate.
  So the "more QAT epochs help" lesson is really "more DECAY improves the fp32 base" -> the int2 PTQ penalty
  (+0.003-0.005) was never dissolved => REAL QAT has untapped headroom. The [QAT] + [QAT-LOWRANK] parsers are
  now restored into architecture.py AND the arch_iter36 snapshot.
- **NOTE low-rank also works (PTQ):** both-low-rank (card rank2-int4 + NOTE rank2-int4 + int4 shifts) PTQ on
  iter45 = imm **0.289137** / ahead 0.321056 -- the BEST deploy yet (beats fp32 champ -0.0069), at the
  SMALLEST state (card 96 B + note ~288 B vs note int2 816 B). BUT note low-rank's per-step 3-layer nalgebra
  SVD makes the GATE ~20x slower (~20-25 min vs ~100s); note int2 already meets the >=2x target, so note
  low-rank is lower-ROI (revisit if the extra -0.002 imm + 2.8x note shrink is worth the eval/QAT cost).
- **REAL low-rank QAT (iter46) = DEAD END (naive STE).** `fake_lowrank_state` (STE rank-r SVD truncation +
  int-N factor quant, matches the Rust deploy) in rwkv_ops.py, wired via `RWKV_QAT_LOWRANK_SCOPE`. iter46 =
  8-epoch decay, card rank2-int4 low-rank QAT + note int2 QAT. RESULT: deploy imm **0.303617 -- WORSE** than
  the card low-rank PTQ (0.291471, by +0.012) and worse than champ_fp32 (+0.0076). The low-rank deploy cost
  on the QAT model BALLOONED to +0.0103 (vs ~+0.0037 PTQ). WHY: rank-2 truncation is a STRUCTURAL change, so
  the identity STE gradient gives NO signal to concentrate energy in the top-2 singular dirs (unlike int-quant,
  where small element-wise error makes STE work) -> the model drifts toward HARDER-to-low-rank states. LESSON:
  **low-rank stays PTQ** (PTQ low-rank already BEATS int2 + hits 0.15 KB); int-quant stays QAT. A
  differentiable-SVD QAT could be tried but PTQ already suffices. The infra (fake_lowrank_state, parsers) is
  kept for the int-quant QAT path it also enables.

**Note on the two state numbers:** "4.25 KiB" is the *fp32, pre-quant* card state (1,088 floats x 4 B
- a pure arch property that `model_stats.py`/`scratchpad/params_for_arch.py` report). The *deployed*
card state is the quantized figure: int8 1.06 / int4 0.53 / **int2 0.27 KiB**. Same 1,088 floats,
different storage bits (deployed KiB = floats x bits / 8 / 1024). `params_for_arch.py` now prints both.

**★★ HARD TARGETS (Andrew 2026-06-29) — BOTH MET 2026-06-29: (A) card state -> 0.15 KB [✓ MET: 0.094 KiB
(96 B) via rank-2-int4 low-rank card WKV + int4 shifts, deploy imm 0.291471 PASS, BEATS fp32 champ];
(B) note state -> >=2x smaller [✓ MET: note int2 0.80 KiB via QAT].** The deployed champion now hits both
(see DEPLOYED CHAMPION above). Memory math (card = 1,024 WKV floats [32x32 matrix] + 64 token-shift
floats [2 vectors; 1-D so only quantizable, not low-rankable]):
- **int2 quant ALONE bottoms out at 256 B** (1,024 floats x 2 bit) -> CANNOT reach 0.15 KB by quant
  alone; MUST cut the float COUNT (low-rank WKV, or smaller K via the kernel route).
- **Card to 0.15 KB path (PRIMARY = low-rank + quantized FACTORS, sidesteps the K=32 kernel block):**
  rank-1 WKV int4 (32 B) + shifts int4 (32 B) = **64 B (0.06 KiB)**; rank-1 int8 + shifts int8 = 128 B;
  rank-2 int4 + shifts int4 = 96 B. All clear 0.15 KB. (Dump shows card state IS near rank-1.) Stacks
  low-rank err x quant err -> measure on the 2k loop. Alt (BLOCKED): H=2/K=16 + int2 = 144 B (CUDA
  kernel rewrite or slow chunked-PyTorch proof).
- **Note >=2x path [✓ DONE iter43]:** note int4 (1.59 KiB) -> **note int2 via QAT = exactly 2x (0.80 KiB)**
  WORKED (deploy imm 0.299469, +0.0034 vs champ fp32, PASS) -- exactly as predicted (QAT rescued note int2
  just as it made card int2 nearly free). Further cuts (if ever needed) via low-rank note WKV + quant.
  Dimension cuts are HARD: note layers 3->2 rejected (iter38,
  costs imm); note d_model<32 (K<16) is K=32-kernel-BLOCKED. NOTE matters MOST for total memory at deploy
  (3 layers => note int4 1.59 KiB is ~6x the card 0.27 KiB per entity; MEASURED: notes ~= 0.9x cards
  across the 10k dataset, so a 1M-card user has ~900k notes -> note state is the DOMINANT deploy memory,
  ~4-5x the card-state total for a power user. See scratchpad/entity_counts_10k.csv + [[dataset-entity-counts]]).
- **These RAISE the value of low-rank (now PRIMARY, not lowest-priority) and of the deck/preset grow**
  (iter41/42 build the accuracy headroom to afford card-0.15 + note-int2). See RESUME step 4.

### Engine (`rust/rwkv-infer`)
fp32 + pre-transpose + lerp-fusion (+8.7%). **Auto-derives num_curves/num_points AND per-stream layer
counts from weight shapes** - adapts to any arch with no code change. State quant via
`RWKV_STATE_QUANT_SCOPE="card:int2,note:int4"` (per-stream mixed bits int8/int4/int2; omitted streams
stay fp32). Batching: `*_batched` query forward (B=1 path untouched, parity bit-exact); **optimal
B~128, single-thread** (intrinsic L2/L3 cache knee at B=128->256; thread count irrelevant). Rust modes:
`--verify-batched`, `--bench-batched`, `--sweep-batched`, `--bench-synth`. See `rust/rwkv-infer/BATCHING_PLAN.md`.
**★ FAST LOW-RANK SVD (2026-06-29, step-3 win):** `lowrank_roundtrip` no longer uses nalgebra's FULL SVD
(which converged pathologically slowly on near-low-rank states -> the note-low-rank gate HUNG; user 187
ran >35 min). Replaced with a top-r truncation via **Gram matrix + symmetric eigendecomposition**
(eigvecs of A Aᵀ = left singular vecs, eigvals = sigma²; right vec v = Aᵀu/sigma). A is normalized by its
max-abs before forming the Gram (the product squares magnitudes -> f32 overflow -> NaN eigenvalues for a
state grown large over a long history; normalize, then unscale sigma). NaN-safe sort + skip non-finite
comps. Validated == full-SVD rank-2 recon to ~1e-15 (numpy). RESULT: user 187 both-low-rank now **22 s**
(was a >35 min hang); the full 17-user both-low-rank gate runs in ~100 s. note-low-rank is now PRACTICAL
in the iteration loop. **Both-low-rank deploy re-confirmed on ALL 17 users (incl 187): imm 0.288831 /
ahead 0.320098, beats fp32 champ by -0.0072 imm / -0.0065 ahead, GATE PASS** (the prior 0.289137 was 16
users w/o 187; 187's low-rank deploy is fine -- the earlier hang/panic was purely the SVD numerics, not divergence).

### LESSON BANK - do NOT re-run these dead ends (full numbers in log.md / HISTORY.md)
- ✅ **Kept:** SRS heads 128->64 (iter29) · card->deck rebalance (compensation order **deck > preset >
  user**, NOT note) · card 2->1 (iter36) · 4-epoch decay (general win, tightens variance) · scoped
  state-quant **card int4 + note int8 ~free** (the 1-KB lever) · QAT makes card int2 + note int4
  essentially free (+0.000018 quant cost) WHEN warm-started from the champion · **QAT note int2 = >=2x
  note target MET (iter43-45)** · **LONGER decay-QAT (8-16 epochs, warm-started) makes the deployed
  int2+int2 model BEAT the fp32 champion** (iter45 deploy imm -0.0035 vs champ; the 4ep decay was
  undertrained) -- saturates ~16ep (best imm@16, best ahead@8) · **LOW-RANK card WKV (rank-2, int4 factors)
  BEATS int2 -- smaller (0.27->0.094 KiB) AND more accurate (-0.0044 imm): rank-2 keeps the top-2 SVD comps
  in int4 (98.7% energy) vs int2's coarse 3-level on all 1024 floats. Card 0.15 KB target MET via PTQ.** ·
  shifts must be quantized too for honest deploy size (RWKV_QUANT_SHIFTS): +0.0033 imm at int2, +0.0011 at int4.
- ❌ **Failed:** FC/head-width 4->2 (imm +0.0526, imm-critical) · note 3->2 layer-cut (iter38, +0.0018
  - shrink note STATE via quant, not layers) · all-streams blanket state-quant (long-recurrence
  user/global sink it) · note int4 via PTQ (>2x budget) · weight PTQ int8/int4 (no speed win) ·
  **QAT from scratch (iter40, +0.0118 - MUST warm-start from a good fp32 ckpt)** · **naive low-rank QAT
  (iter46, STE rank-2 truncation): deploy +0.0076 vs champ, WORSE than low-rank PTQ -- STE can't guide a
  structural rank change; low-rank stays PTQ, int-quant stays QAT**.
- ⚡ **GPU-training + gate SPEEDUPS (2026-06-29, step 3 -- arch-agnostic, untimed/non-gating):**
  (a) **`copy_downcast_` + `transfer_child_grad_to_master` vectorized with `torch._foreach_*`** (one fused
  kernel per dtype group vs ~440 per-param launches each) -- BIT-IDENTICAL (verified); (b) **`get_grad_norm`
  (~440 `.item()` syncs/step) + `log_model` skipped when `USE_WANDB` is off** (logging-only) -- in
  `train_rwkv.main_loop`. Together **+1.21x** no-JIT (2.53->3.07 steps/s, full 31-group workload). (c) **JIT
  RESTORED via `@torch.jit.ignore` on `quant_aware_rwkv7`** -- the QAT-lowrank addition (torch.linalg.svd in
  the per-step loop) had SILENTLY broken TorchScript scripting (internal assert in torch 2.12.1+cu130), which
  would CRASH any plain WS/decay training AND `get_result.py` eval (both JIT-on). Fix lets the scripter skip
  the never-scripted QAT branch -> JIT-on hot path restored, eager QAT path unchanged (off-path==reference).
  Combined **JIT-on + foreach + sync-removal = 3.48 steps/s = 1.38x over the no-JIT old body** (1.30x over
  JIT-on old body). ⚠ JIT has a ~30-60 s one-time compile -> wins only for LONG runs (the 1k-user phase);
  for short 100-user iters it's ~neutral and the foreach/sync win (unconditional) is what matters. **`torch.compile`
  is NOT viable (no Triton on Windows); JIT was the only fusion route and it's now fixed.** Profiler =
  `scratchpad/profile_train.py` (sync section breakdown + old-vs-new body A/B; the dominant fwd+bwd ~90% is the
  custom WKV kernel at low B=1 parallelism -- untouchable without kernel/batch changes).
- 🔒 **Blocked:** K<32 (smaller head dim, the biggest state lever) - the CUDA training kernel hardwires
  K=32; needs a kernel rewrite or a slow K-agnostic chunked-PyTorch proof. Deferred. · `torch.compile`/inductor
  (no Triton wheel on Windows) and CUDA graphs (variable seq shapes + custom autograd.Function) -- not pursued.

### ★★ NEW PHASE PLAN (Andrew 2026-06-29, supersedes the deck/preset-grow RESUME below) ★★
Low-rank investigation is essentially DONE: **both-low-rank PTQ (card rank2-int4 + note rank2-int4 +
int4 shifts) = imm 0.289137** is the best deploy (smallest state too: card 96 B + note ~288 B), and it
is **deploy-viable** -- the per-step SVD is needed at inference (re-truncate the rank-2 state each review,
since a rank-2 state + rank-1 WKV update -> rank-4) BUT costs ~10-40 us/SVD in Rust (~158 us in numpy);
at human review pace that's ~0.6 s over a 1000-review DAY = negligible. The ~20-min gate slowness is ONLY
the benchmark replaying millions of reviews at max speed -- a measurement artifact, not a deploy cost.
Ordered steps:
1. **[✓ DONE 2026-06-29] Clean-confirm both-low-rank -> CHAMPION.** 16-user clean gate (dropped the stuck
   large user 187): deploy imm 0.271665, delta vs champ_fp32 = **-0.005905 imm** (matches the prelim 17-user
   -0.006927 -> validated; absolute 17-user ~0.289). Pure low-rank deploy cost +0.001012. **Both-low-rank PTQ
   (card rank2-int4 + note rank2-int4 + int4 shifts, card 96 B + note ~288 B) is the deployed champion.**
   ⚠ BLOCKER [✓ RESOLVED 2026-06-29 by step 3]: the note-low-rank gate was impractically slow (user 187
   ran >35 min on nalgebra full SVD). FIXED with the fast Gram+eigen truncated SVD -> 187 now 22 s, full
   17-user both-low-rank gate ~100 s. **Re-confirmed on ALL 17 users incl 187: deploy imm 0.288831 / ahead
   0.320098, -0.0072 imm vs champ_fp32, GATE PASS** (the 16-user 0.271665 above was a different/cleaner
   subset; the absolute 17-user number is ~0.289). Both-low-rank PTQ is the deployed champion; gate practical now.
2. **Settle PTQ vs QAT for BOTH-low-rank -> LOCKED = PTQ (low-rank), based on iter46 + mechanism.** iter46
   (card-only low-rank QAT, STE) was a DEAD END (deploy +0.0076, pure quant cost +0.0103 vs PTQ ~+0.001 --
   STE can't guide a STRUCTURAL rank change; this is per-stream physics, so the NOTE case is identical). A
   full both-low-rank QAT to re-confirm was impractical (its deploy gate hung on the slow-SVD issue, now
   FIXED in step 3 -- the gate is fast). So low-rank stays PTQ. ★ ROOT CAUSE of the gate slowness (2026-06-29): nalgebra FULL SVD converges
   SLOWLY on the real near-low-rank states (sing. values 3-32 ~0 -> the iterative SVD grinds on the tiny
   clustered values) -- user 187 (only 1,119 cards) took >35 min. My random-matrix bench (158 us) missed this.
   FIX (step 3, and the RIGHT method anyway): a truncated rank-2 (power/subspace iteration) extracts ONLY the
   top-2 and IGNORES the tiny values -> fast AND well-suited to near-low-rank. DEPLOY is still fine (per-review
   even at ~ms is negligible vs seconds between reviews); only the benchmark (millions of replayed reviews) is
   hit. Path to beat the champion = a BETTER BASE (fp32 base still improving at 16ep: 0.296->0.292->0.288 at
   4/8/16ep -> try 24-32ep plain decay) + both-low-rank PTQ. NO note-int2 QAT (iter47 shelved -- note int2
   0.80 KiB is BIGGER than note low-rank 0.28 KiB).
3. **[✓ DONE 2026-06-29] Maximally speed up GPU training + the low-rank gate (arch-agnostic).** See the
   ⚡ lesson-bank entry for full numbers. GPU TRAINING: profiled (the dominant cost is the custom WKV CUDA
   kernel at B=1 low parallelism over long sequences -- compute-bound, NOT launch-bound the way assumed;
   the launch-bound part was the per-param Python loops + logging syncs). Wins = `torch._foreach_*`
   vectorization of `copy_downcast_`/`transfer_grad` + skip `get_grad_norm`/`log_model` when wandb off
   (+1.21x, bit-identical) + RESTORE JIT via `@torch.jit.ignore` on `quant_aware_rwkv7` (was silently
   broken -> would crash plain WS/eval; combined **3.48 steps/s = 1.38x** over the no-JIT old body).
   `torch.compile` ruled out (no Triton on Windows); CUDA graphs not worth it (variable shapes). LOW-RANK
   GATE: replaced nalgebra full SVD with a fast Gram+symmetric-eigen top-r truncation (see Engine section)
   -> note-low-rank now ~22 s/heavy-user (was a >35 min hang); the both-low-rank gate is practical IN THE
   LOOP. ⚠ JIT one-time compile (~30-60 s) means JIT-on wins for LONG runs (the 1k phase); foreach/sync win
   is unconditional. ALL changes are arch-agnostic (derive shapes at runtime).
4. **NEW RESEARCH PHASE: train 1-1000 / test 1001-2000, GPU-ONLY eval** (the roadmap's 2k loop). Rust/CPU
   ONLY for minimal ~3-user parity checks, NOT the main gate -- the main eval is `get_result.py` (CUDA) on
   1000 users. Focus = **ALGORITHMIC improvements** (the research-y step) while keeping **params AND
   per-entity state size under fixed MAX CAPS** (cap = current champion: ~192,800 params; card 96 B + note
   ~288 B low-rank, or whatever the confirmed champion is). Bigger/cleaner eval signal than the 17-user gate.

### Step-4 GROUNDWORK (Andrew 2026-06-29, IN PROGRESS) -- old-vs-new baseline on the 1k test set
Andrew's pre-step-4 groundwork: (1) eval the OLD RWKV (`pretrain/RWKV_trained_on_5000_10000.pth`, the
original 2.76M d=128 leaderboard model, trained on users 5000-10000) on users 1001-2000, per-user logloss
for BOTH modes (ahead=forgetting-curve, imm=immediate); (2) ENSURE per-user `size` (equalized review count)
is IDENTICAL old-vs-new (proof the preprocessing matches); (3) eval the NEW champion on the same 1k users.
- **DATA WASN'T BUILT**: test_db only had users 101-200, label_filter_db ~100-516. Building 1001-2000 via
  `find_equalize_test_reviews` (label_filter) + `data_processing` (test_db) -- detached `scratchpad/build_eval1k.cmd`
  (configs `find_equalize_eval1k_config.toml` + `data_processing_config_eval1k.toml`, USER 1001-2000). Both
  APPEND (skip `_done` users) so the existing 101-200 gate data is untouched. Monitor `scratchpad/build_eval1k.log`
  (`DONE_EXIT_`). ~1-2 hr, ~50 GB (257 GB free).
- **OLD model needs the d=128 arch**: our srs_model.py diverged (features_fc_mult/head_fc_mult/num_curves/
  num_points config fields the srs-benchmark original lacks), so I transcribed the original into our format =
  `scratchpad/architecture_old_d128.py` (STRICT-loads the old ckpt, 2,762,884 params, exact match). Eval swaps
  it into `rwkv/architecture.py` then restores the champion (`scratchpad/architecture_champion_backup.py`).
- **Eval after build**: `scratchpad/run_eval1k.cmd` (NEW via get_result_new_1k.toml; OLD via arch-swap +
  get_result_old_1k.toml; then `compare_eval1k.py` = size-identity check + by-user-mean logloss + per-user CSV).
- **SMOKE (users 1001-1003) PASSED**: size IDENTICAL old/new (14170/91150/67930); OLD beats NEW on all 3
  (e.g. user 1003 imm old 0.4522 / new 0.7373). The NEW champion was trained on only 100 users + SELECTED on
  101-200, so its 1001-2000 numbers are a generalization FLOOR -- step 4 retrains the arch on 1-1000 to close
  the gap vs the old 5000-user-trained model. The full 1000-user means are the real comparison (3 users = noisy).
- ⚠ get_result.py runs JIT-on -> REQUIRES this session's `@torch.jit.ignore` fix on `quant_aware_rwkv7`
  (else it crashes at model build). Confirmed working (the 11s/100-user eval + the smoke ran JIT-on).

### ★★ DATA-DROP BUG (Andrew 2026-06-29) -- the optimization loop trained on ~5% of the data ★★
While investigating "why is B=1", found that **`get_groups` SILENTLY SKIPS any batch whose size >
MAX_TRAIN_GLOBAL_LEN** (`max_batch = floor(MAX/size); if max_batch==0: continue`). The train_db batches
are large (per-user histories, sizes up to 65,536 ~ the ORIGINAL MAX=66000). The optimization configs use
**MAX_TRAIN_GLOBAL_LEN=20000**, so at 20000: **only 35/212 batches kept = 4.7% review-token coverage, just
20/100 users fully present** (the smallest-history users); the 80 longer-history users are partly/fully
dropped. Coverage by MAX: 20000->4.7%, 40000->16.3%, **66000->100%** (all 212 batches, 170 groups). So the
champion (iter36) trained on ~5% of even its 100 users' data -- almost certainly a big part of its POOR
generalization to 1001-2000 (smoke: old beats new on all 3 users; it never saw long-history users). B=1 is a
symptom: the ~35 surviving batches each ~fill the 20000 budget alone. **Iter-to-iter RANKINGS stay valid (all
used the same 20000 subset), but absolute champion quality is on a biased slice.** FIX = MAX=66000 (full
coverage); feasible on the 12 GB GPU now (d=32 champion, ~16x smaller activations than the original d=128 that
needed 66000 on a 24 GB card). At 66000 you also get B>1 free for small users (histogram B1:148,B2:13,...,B7:1).
- **IN PROGRESS: re-baseline the champion at 66000** (Andrew "do both"): `scratchpad/run_rebaseline.cmd` runs
  `rebase_66k_ws.toml` (from-scratch WS, 1-100, 66000, 6 epochs ~1020 steps) TWICE -> run1=fair champion,
  run1-vs-run2=run-to-run variance. THEN eval run1 on 1001-2000 (new) + old on 1001-2000 -> redo old-vs-new.
  RUN ONLY AFTER build_eval1k finishes (the failed 20000 variance run died from GPU contention with the build's
  data_processing -- evals crashed before writing; trainings were fine). ~30 min/run on a clean GPU.
- **DETERMINISM enabled** (Andrew "enable determinism"): `train_rwkv._maybe_enable_determinism()` (RWKV_DETERMINISTIC
  default 1) pins the TRAINING process RNG + cuBLAS/cuDNN (CUBLAS_WORKSPACE_CONFIG=:4096:8). The custom WKV kernel
  has no atomics (already deterministic; eval is bit-identical). **Augmentation KEPT stochastic** (Andrew's call --
  the per-batch random ID-encodings + time baselines stay in the fetch children, unseeded) -> run-to-run variance
  now isolates the AUGMENTATION-only noise floor. (Andrew is skeptical the augmentation even helps -- ablation TODO.)

### ★★★ REVISED PLAN (Andrew 2026-06-29 late) -- supersedes the NEW PHASE PLAN's step-4 ordering ★★★
**KEY NEW RESULTS this session:**
- **Full-coverage 66000 re-baseline (WS-only, from scratch on 1-100) BEATS the iter36 champion by ~0.013 imm /
  ~0.017 ahead on 101-200** (re-baseline imm 0.2989-0.3006 / ahead 0.330 vs champion imm 0.3139 / ahead 0.3480;
  SAME train users + eval set, only 5%->100% coverage). The data-drop fix is worth ~0.013 imm -- LARGER than the
  ENTIRE optimization loop (iter0 0.3195 -> champion 0.3139 = 0.006). Re-baseline ckpts:
  `scratchpad/rebase_run1/rebase_1020.pth` (WS), `scratchpad/rebase_champ/rebasec_680.pth` (WS + 4-epoch decay).
- **Run-to-run variance (determinism ON, augmentation stochastic) = ~0.0018 imm / 0.0006 ahead (100 users).**
  PURELY augmentation-induced (the two trainings land in different optima -- a correlated shift that does NOT
  average out with more users). NOT <0.0001. => **tuner noise margin ~0.002.**
- **Tuner = GREEDY coordinate descent** (pattern-search / Hooke-Jeeves, ~0.002 noise-margin acceptance, natural
  early-stop), NOT CMA-ES (25-eval budget too small for its covariance) or Bayesian (warmup waste); Optuna TPE as
  a phase-2 on the ~3 most-coupled params. Tune ~6-8 of the ~20 non-arch hyperparams (full inventory in HISTORY).
- **Stateful-BPTT finding:** training chunks (32768-review windows, multiple per user) are trained COLD --
  `RWKV7_WKV.forward` takes NO initial state, and `get_groups` shuffles chunks independently. So (a) B=1 wastes
  parallelism (one ~62k-token chunk fills the 66000 budget; GPU ~15-67% util) and (b) train/eval MISMATCH (eval =
  full history with carry; test_db = 1 batch/user, asserts len==1). Eval is also slow: power users have 700k+
  review histories (~3 min/100 users; the earlier "11s" was a resume-skip artifact).
**ANDREW'S PLAN (ordered):**
0) [DONE] compaction + GitHub=local.
1) **STATEFUL BPTT FIRST** (the speed enabler -> makes everything else faster): chunk smaller + batch across users
   (B>>1) + carry the RNN state across a user's consecutive chunks. Gets speed (high B util) AND learns long
   context AND closes the train/eval mismatch -- "2-3 birds". Needs a CUDA-kernel change (add initial-state input +
   final-state output to the WKV forward/backward). ALSO look for OTHER train + EVAL speedups.
2) **Build train_db for users 1-1000** (only 1-100 exists!) -- WITH the new BPTT chunking. test_db 1001-2000 is
   ALREADY built (this session). This is the prerequisite Andrew's plan implies for "train on 1k".
3) **1k RESEARCH PHASE: train 1-1000 / eval 1001-2000** (GPU get_result), algorithmic improvements under the
   param + per-entity-state caps. OLD baseline = `pretrain/RWKV_trained_on_5000_10000.pth` (2.76M d=128; eval via
   `scratchpad/architecture_old_d128.py` arch-swap, strict-loads). NEW champion logloss MUST include QUANTIZATION
   (deployed = low-rank PTQ): current champ = iter45 fp32 `pretrain/rwkv/opt_qat45/rwkv_iter45_496.pth`; quantized
   eval via the RUST engine on exported traces (`export_features_fast.py --range`) -- per-step SVD too slow in
   Python over power users' full histories.
4) **AUGMENTATION ABLATION:** train with the per-batch augmentation ON vs a FIXED seed, compare logloss -> does the
   randomization actually improve generalization? If not, fix the seed -> reproducible objective (variance ~0) for
   the tuner. (Augmentation = random ID-encoding vectors + random time-of-day baselines, regenerated EVERY batch,
   `prepare(seed=None)`; eval uses fixed seed 1234 -> eval is bit-deterministic.)
PENDING/ARTIFACTS: the 1001-2000 old-vs-new fp32 comparison was STARTED then STOPPED (slow power users; variance
already answered -- don't resume it as-is). Harness ready: `scratchpad/run_rebaseline_eval.cmd` + `compare_rebaseline.py`
(old d=128 arch-swap + iter45 + re-baseline; size-identity check). get_result runs JIT-on (needs the jit.ignore fix).

### ★ STATEFUL BPTT PROGRESS (2026-06-29, step 1 of the revised plan) -- full design = `optimization/STATEFUL_BPTT_PLAN.md`
- **✅ CUDA kernel foundation DONE + verified.** New `RWKV7_WKV_Stateful` (rwkv_ops.py) + ops
  `rwkv7_wkv_{forward,backward}_stateful_{float,bf16,half}` (rwkv7_cuda.cu/rwkv7.cpp, rebuilt). Forward takes
  `state0_BHKK`, returns `(out, final_state_BHKK)`, ALWAYS sequential (the time-parallel scan can't take an
  initial state). Backward forces sequential (saved checkpoint[0]=state0 -> correct nonzero start; truncated
  BPTT drops dS->state0). Non-stateful path BYTE-IDENTICAL (nullptr -> original behavior). Parity
  (`scratchpad/test_stateful_wkv.py`): (A) stateful(state0=0)==non-stateful EXACTLY 0; (B) forward
  split-equivalence fwd([A;B])==[fwd(A);fwd(B,state0=final_A)] EXACTLY 0 (fp32+bf16); (C) truncated-BPTT grads
  vs pure-PyTorch detached-carry ref = 3.8e-6 fp32. ⚠ NOT yet committed (commit-when-asked).
- **KEY: NO train_db rebuild / schema change needed** -- chunks already stored per-user time-ordered with entity
  IDs; carry is a training-loop + model-forward change only. Per-entity carry = 3 tensors/layer (WKV [H,K,K] + 2
  token-shifts [C]); the 5 streams carry INDEPENDENTLY (blueprint = srs_model_rnn.py run()).
- **★ SCOPE FORK (awaiting Andrew):** chained streams force ONE shared chunking, so the only simple B-boost is
  smaller chunks. (R) MEASURE-FIRST: rebuild train_db 1-100 smaller-chunk, train champion @66000 cold, compare
  speed+logloss vs the 32768 re-baseline -- if cheap cold chunks cost little accuracy, full carry may be UNNEEDED
  (32768 cold already BEAT the champion). (F) FULL per-entity carry (the intricate per-entity-mapping +
  synchronized stateful batching, steps 2-3 of the plan). RECOMMEND R first (cheap, evidence-generating, Andrew
  asked "other ways to speed up"); build F only if R's accuracy cost is unacceptable. Kernel foundation kept either
  way. Hold the train_db 1-1000 build until the route is chosen (chunk size baked into the db depends on it).

### ★★ ROUTE R RESULT (2026-06-29) -- DOUBLE SURPRISE, reshapes the plan. Andrew chose R (measure-first).
Trained the champion FRESH on 1-100 at two chunk sizes (fresh WS, MAX=66000, 6 epochs), evaled on 101-200
(size-identity check PASSED -- same reviews). base65k = current 65536-row-chunk db (B~1); sc8k = new 8192-review
-chunk db (chunk `length`~12288-16384 ROWS = history+query, packs B~4 at ~60400 rev/step, 92% full -- NOT
underpacked). Results:
- **base65k: 31,287 rev/s, 15.2 min, imm 0.296890 / ahead 0.329804.**  (reproduces the re-baseline ~0.2969-0.3006)
- **sc8k:    27,345 rev/s, 17.3 min, imm 0.289628 / ahead 0.322033.**
- **Δ (sc8k - base65k): -14% throughput (SLOWER), but imm -0.00726 / ahead -0.00777 (MORE ACCURATE).**
**(1) SPEED:** smaller chunks are SLOWER, not faster -- training is launch-bound (~15% util) and B=4x13k is
intrinsically a hair slower than B=1x65k at equal packing. **The chunk-size/B lever is a DEAD END for speed ->
kills stateful BPTT's speed rationale.** **(2) ACCURACY:** smaller cold chunks are a real win (-0.0073 imm, 4x
the ~0.0018 aug-noise floor; corroborated by the 10-user in-training validation sc8k 0.2898 vs base65k 0.3030).
**sc8k imm 0.2896 is a NEW BEST on 101-200** (vs re-baseline 0.2969, vs iter36 champ 0.3139). Likely cold-start/
windowing regularization (by-user logloss rewards predicting from little context = the common SRS case).
**RESHAPED PLAN:** SHELVE the intricate stateful carry (its speed rationale is gone + smaller-COLD-chunks help,
the OPPOSITE of carrying state); KEEP the verified kernel (cheap, done). Speed levers are now: HIGHER
MAX_TRAIN_GLOBAL_LEN (fewer/fuller steps: 66000->132000->200000 cuts 6ep steps 960->474->306) + genuine per-step
speedups (CUDA graphs etc.), NOT chunk size. ★ Andrew 2026-06-29: do NOT use fewer-epochs-for-ranking -- the tuner
must evaluate every config at FULL training so rankings stay trustworthy (3-epoch ranking != 6-epoch ranking). **IN FLIGHT (route A/B, `scratchpad/route_ab.py`/.log):** (a) confirm sc8k win w/ 2nd
aug seed + sweep 4096-chunk; (b) sc8k @ MAX=132000/200000 for the higher-MAX speed lever (rev/s + accuracy).
NOTE: 100/100 is the workbench now (Andrew, 2026-06-29) -> train_db 1-1000 build is DROPPED (existing 1-100 train
+ 101-200 test suffice). Eval cost ~3.7 min/100 power users (a co-bottleneck for the tuner loop).
- **★ CHAMPION LOGLOSS = DEPLOYED, not fp32 (Andrew 2026-06-29):** a champion's comparison logloss MUST be
  computed with BOTH quantization AND low-rank state enabled (the deploy config), via the Rust engine on 101-200,
  because that is what ships in Anki. The d=128 baseline-to-beat stays fp32 (it's the accuracy TARGET, not
  deployable). So the research-phase eval needs a quant+low-rank Rust pass on 101-200 (export traces via
  export_features_fast.py --range, run with RWKV_STATE_LOWRANK_SCOPE + RWKV_STATE_QUANT_SCOPE + RWKV_QUANT_SHIFTS).
  Current champion rows showing fp32 (e.g. sc8k imm 0.2896) are PLACEHOLDERS until the deployed number is measured.
  See `optimization/log.md` "Baseline to beat" section. [[champion-logloss-deployed]]
- **★★ BOTH card AND note use low-rank + quant (Andrew 2026-06-30):** the deployed config = BOTH-low-rank --
  card state AND note state are EACH rank-2 low-rank with the factors quantized, AND the 1-D token shifts
  quantized too. Champion logloss = the both-low-rank quantized number (NOT card-low-rank/note-int2-only).
  Rust flags = `RWKV_STATE_LOWRANK_SCOPE="card:2:<lvl>,note:2:<lvl>"` + `RWKV_QUANT_SHIFTS=1` (shift bit-width
  follows the stream's low-rank factor level). Trying int2 factors+shifts: sizes int2 = card 48 B (0.047 KiB) +
  note 144 B (0.14 KiB); int4 (prior) = card 96 B (0.094 KiB) + note 288 B (0.28 KiB). [[champion-logloss-deployed]]

### ★★★ RESEARCH PHASE (100/100) -- CHARTER: plan + ACCEPTANCE GATE (Andrew 2026-06-29 night) ★★★
Workbench: train users 1-100 / eval 101-200 (--short --secs). Champion comparison logloss = DEPLOYED
(quant + low-rank, via Rust on 101-200) [[champion-logloss-deployed]]; baseline-to-beat (d=128 trained on
1-100) stays fp32 = the accuracy TARGET. Both live in `optimization/log.md` "Baseline to beat".

**ACCEPTANCE GATE -- a change is ACCEPTED iff ALL hold (record accepted/rejected BINARY per iter in log.md
`status`):**
1. "size" (equalized review count, 101-200) IDENTICAL to champion (data-integrity; any change = pipeline bug).
2. param count <= **225,000** (raised from 192,800 to give headroom to try new things).
3. **card state size UNCHANGED and note state size UNCHANGED** vs champion. deck/preset/global state MAY grow freely.
4. ahead (forgetting-curve): champion_ahead - candidate_ahead **>= 0.0003** (candidate strictly BETTER).
5. imm (immediate):          champion_imm   - candidate_imm   **>= 0.0003** (candidate strictly BETTER).
   => accept ONLY changes that IMPROVE **BOTH** modes by >=0.0003 vs the CURRENT champion (monotonic champion).
   This REPLACES the old iter0-floor (+0.0015) gate for the research phase.
**VARIANCE / augmentation [RESOLVED 2026-06-29]:** 0.0003 was << the old ~0.0024 augmentation variance
(A/B: sc8k seed1 imm 0.2896 vs seed2 0.2920), so a single-run 0.0003 win was NOISE. Andrew's call: DISABLE
the augmentation outright (don't test) -- the variance cripples the gate more than the augmentation helps;
re-enable later. DONE: `train_rwkv.main` now uses a FIXED augmentation seed (env `RWKV_AUGMENT_SEED`, default
1234; set `=none` to restore stochastic). Eval (get_result) was already fixed at 1234. So train+eval are now
DETERMINISTIC -> variance ~0 -> the 0.0003 gate is usable. ⚠ VERIFY variance ~0 with two augmentation-off runs
before relying on it. ALSO: the champion's official number must be RE-MEASURED augmentation-off (the A/B sc8k
numbers were augmentation-ON); the d=128 baseline auto-runs augmentation-off (inherits the new default).

**PLAN (ordered, Andrew):**
1. [DONE] Higher MAX (2 runs) -- see A/B results below.
3. **[DONE] AUGMENTATION DISABLED** (Andrew's call, no ablation): `train_rwkv.main` uses a FIXED augmentation
   seed (env `RWKV_AUGMENT_SEED`, default 1234) -> deterministic objective (variance ~0) so the 0.0003 gate
   works. Re-enable later with `RWKV_AUGMENT_SEED=none`. (Augmentation = random ID-encoding vectors + random
   time-of-day baselines, previously regenerated every batch via the unseeded fetch children.)
4. **Pick the most impactful hyperparameters -> GREEDY coordinate-descent tuner** (Hooke-Jeeves; full HP inventory
   in HISTORY). ★ Andrew 2026-06-29: **TUNE THE CURRENT CHAMPION FIRST** -- before ANY architecture change -- so
   step 5 is explored from a well-tuned baseline (do NOT start the arch search on an untuned model; an untuned
   baseline could make you reject good archs that just needed tuning). AFTER that first tune, run the tuner
   SPARINGLY: only after a VERY BIG architectural change OR several accumulated small changes -- NOT every iteration.
5. **Improve ARCHITECTURE and/or TRAINING pipeline** to lower logloss (AFTER the first tune in step 4). Any change
   that does NOT alter data PREPROCESSING is fair game. Measure on 100/100; accept per the gate above.
- **★ Periodically do a LITERATURE REVIEW on neural-network architectural improvements** (new attention/RNN/SSM
   tricks, normalization, init, gating, etc.) for inspiration -- weave findings into step 5. Seeds + a concrete
   task-5 experiment queue (state-neutral, gated): `optimization/LIT_REVIEW.md`.

### ★★ HP TUNER RESULT (2026-06-30) -- BIG WIN from tuning the champion (step 4 done) ★★
Greedy coordinate-descent tuner = `optimization/hp_tuner.py` (self-driving `loop` cmd, resumable from
`optimization/tuner_log.jsonl`; trial files in `scratchpad/tuner/`; env overrides RWKV_WEIGHT_DECAY/RWKV_CLIP
added to train_rwkv, defaults==champion). Tuned 5 HPs on the 100/100 workbench (sc8k WS, aug-off, deterministic
-> variance 0). **Champion was badly UNDERTUNED.** Per-coordinate winners (objective = ahead+imm, lower better):
- **peak_lr 7e-4 -> 1e-3** (BIG: obj 0.6168->0.6102; both modes improve). 3.5e-4/5e-4/1.4e-3 all worse.
- warmup_steps **200** (default held; 100/400 worse). weight_decay **0.01** (default held; weak lever).
- **clip 0.5 -> 0.25** (small win, imm-driven). **epochs 6 -> 9 -> 12 -> 15** (grid extended [6,9,12,15];
  obj 0.6097 -> 0.6019 -> 0.6012 -> 0.5982): epochs is the SECOND big lever => model was UNDERTRAINED at 6.
  epochs=15 WON (12->15 still gave imm -0.0023, not fully saturated, but the WSD decay phase is the higher-ROI
  next lever than yet more constant-LR epochs).
**FINAL tuned config = {peak_lr 1e-3, warmup 200, wd 0.01, clip 0.25, epochs 15}** -> ahead **0.316252** /
imm **0.281974** (fp32, 101-200). ckpt scratchpad/tuner/hp_epochs_15/hp_epochs_15_2400.pth. vs untuned champion
(0.324173/0.292607): ahead -0.0079 / imm -0.0106 (smashes the 0.0003 gate). **★ vs the d=128 BASELINE-TO-BEAT
(0.320295/0.281913): the d=32 tuned model BEATS d=128 on AHEAD by 0.0040 and TIES it on imm (within 0.0001) --
at 14x fewer params, purely from HP tuning. The arch was never capacity-limited, just undertuned.** NEXT:
(a) ★ WSD DECAY ACCEPTED: WS-15 + 4-epoch cosine decay = ahead **0.314807** / imm **0.280200** (vs WS-15 ahead
-0.0014 / imm -0.0018; BOTH past gate). **★★ NOW BEATS the d=128 baseline on BOTH modes (ahead +0.0055, imm
+0.0017) at 14x fewer params, pure training.** CHAMPION ckpt = scratchpad/tuner/decay15/decay15_640.pth. (b)
lock done (baseline_log; log.md/snapshot pending). (c) ★ DEPLOYED champion MEASURED
(quant+low-rank via Rust; 17-user penalty -> 100u estimate. scratchpad/run_deploy17.sh + deploy_eval_range.py +
export_weights_only.py). Deploy = card rank2-int4 lowrank (0.094 KiB) + note int2 (0.80 KiB) + int4 shifts = BOTH
hard state targets MET. PTQ penalty is TINY -- note int2 +0.0020 imm / +0.0010 ahead; note int4 +0.0011 / -0.0005
-- so NO QAT NEEDED (the low-rank card + well-trained decay states quantize cleanly). Est. 100u deployed note-int2 =
imm ~0.2822 / ahead ~0.3158 -> vs d=128 baseline imm ~TIED (+0.0003), ahead BEATS by 0.0045; note-int4 (1.59 KiB)
imm ~0.2813 / ahead ~0.3143 BEATS d=128 on BOTH. ★ KEY trick: trace INPUTS are weight-INDEPENDENT -> reuse existing
trace_user_{u} + re-export only weights for a fast penalty read. Exact 100u deferred (power-user RNN trace export is
~hours; the 17u penalty is subset-robust, conclusion unambiguous). (d) NOW: task-5 arch experiments (queue in
optimization/LIT_REVIEW.md; top = restore num_curves/num_points 64->128).
OPEN training levers not yet pushed (revisit if needed): WS epochs 15 wasn't fully saturated; decay length=4 (untuned).

### ▶▶ SESSION 2026-06-30 LIVE STATE + RESUME (across compaction) ▶▶
**CHAMPION = WS-15 + 4-epoch decay** (d=32, 192,800 params): fp32 ahead 0.314807 / imm 0.280200; DEPLOYED =
both-low-rank int4 (card 96 B + note 288 B) ~0.3140/0.2806 -- BEATS d=128 baseline (0.320295/0.281913) on BOTH
modes. ckpt scratchpad/tuner/decay15/decay15_640.pth; weights reference/champ_decay15.safetensors. Records in
optimization/baseline_log.jsonl + research_log.md + log.md (4-decimal). Gate: accept iff BOTH modes improve
>=0.0003 vs champion, params<=225k, card/note state fixed.
**RESEARCH FINDINGS:** capacity adds REJECT (exp1 num_curves/points 128, exp2 channel_mixer 1.5, decay8 8-epoch
-- all in research_log.md) => the d=32 model is DATA-LIMITED at 100 users, not capacity-limited. Training levers
(HP tuning, epochs, decay) are the wins. Arch env-overrides added: RWKV_NUM_CURVES/POINTS, RWKV_CHANNEL_MIXER_FACTOR,
RWKV_LORA (default = champion).
**RUNNING NOW (detached, OS-truth monitor -- watchers die on teardown):**
- **build_1500** (PID ~11896): building train_db_sc8k_1500 (users 1000-2499, ~56 GB, sc8k 8192-chunk) for the
  "VARIED DATA, FEW EPOCHS" experiment. ~2-4 hr (rate fluctuates w/ GPU-job CPU load). RESUMABLE -- relaunch
  scratchpad/run_build_1500.cmd if interrupted (skips _done). Monitor scratchpad/build_1500.log (tqdm + DONE_EXIT).
- **ep18** (WS-18 + decay): finishing (gate when done -> research_log).
**QUEUED:**
1. **1500-user experiment** (Andrew's "1 epoch on 1500 users vs 15 epochs on 100"): after build_1500, run
   scratchpad/run_train_1500.cmd (1 epoch WS on 1000-2499, ~2400 steps ~= compute-matched to the champion) ->
   eval 101-200 -> score vs champion. Tests data VARIETY vs REPETITION. (data_processing tweaked to tolerate
   train users absent from label_filter -> empty equalize, metric-only, safe; find_equalize NOT needed.)
2. **★ SPEEDUPS (Andrew PRIORITIZED 2026-06-30) -- RE-DIAGNOSED 2026-06-30 (CORRECTS the earlier
   "fetch/transfer ~1.5-1.85x" claim, which was WRONG about the mechanism):** measured the data-delivery pipeline
   directly (scratchpad/profile_fetch.py single-process + scratchpad/profile_emptycache.py GPU-only) AND read the
   real train logs' per-step `Got:` print. FINDINGS:
   - **FETCHING IS ALREADY HIDDEN -- NOT a lever.** Real-log `data_fetcher.get()` wait = ~2.5-3 s on the FIRST
     batch only (queue warmup), then **~3-7 ms every step after** (7 fetch workers + FETCH_AHEAD=5 fully overlap
     prep+IPC). So `manager.Queue` proxy and the O(B*T) `prepare()` Python loops are OFF the critical path. The
     input-batch `.to(device)` H2D = **~5 ms pageable / ~3 ms pinned (and ~0 ms on the critical path -- the GPU
     pipelines it)**; avg batch is only ~21 MB, not 100+. => async pinned double-buffer prefetch buys ~nothing.
     (`prepare()` itself is ~700-1500 ms/group but invisible because the workers run ahead.)
   - **THE CHEAP WIN = stop calling `torch.cuda.empty_cache()` EVERY step.** train_rwkv.py clears the device cache
     every step for the first 1000 steps (fragmentation-OOM guard) -- measured **+~150 ms/step**. Short research
     runs are 960-2400 steps, so the WHOLE run pays it => ~1.2x for short runs. Added `RWKV_EMPTY_CACHE_EVERY` env
     (default 1 == byte-identical; 50 = periodic; 0 = off). empty_cache is NUMERICS-NEUTRAL (allocator only) -> no
     bit-identical eval needed, just confirm no-OOM (model is tiny, ~6 GB of 12 GB). VALIDATE on a clean machine
     via scratchpad/run_ectest.cmd (A=every1 vs B=every0, train_db_sc8k, 320 steps). [IN FLIGHT after build_1500.]
   - **THE REAL LEVER = the WKV-kernel compute floor (fwd 140 + bwd 403 = ~543 ms/step, ~80% of the step,
     compute-bound).** Only a SMALLER MODEL (cuts d/layers -> smaller WKV matmuls) or a KERNEL REWRITE (K<32, the
     head dim is hardwired to 32) or a bigger effective batch can move it. => task-3 "2x smaller model" is BOTH
     the size win AND the main speed win. CUDA graphs only touch the ~20% launch overhead (~1.1-1.3x, high effort,
     torch.compile Windows-blocked) -> DEFER. Param breakdown (192,800): 5 RWKV streams 75.5% (deck 4L=21.6%,
     note/preset/user 3L=16.2% each, card 1L=5.4%), SRS heads 16.0%, input FC 8.4%; ~10.4k params/d32-layer.
3. **EMA experiment** prepped (scratchpad/run_exp_ema.cmd): WS-15 + EMA(0.999), eval averaged weights vs champion.
**DEPLOY:** int4 both-low-rank is the config (int2 DEFERRED -- per-column scaling rescues it 3.6x to +0.014 but
not free; sort-fix [robustness, no more panics] + RWKV_LOWRANK_PERCOL in the engine). See [[deploy-known-issues]].
**UNCOMMITTED engine/code:** rust/rwkv-infer (sort fix + per-column low-rank), train_rwkv.py (EMA + env overrides +
augmentation seed + RWKV_EMPTY_CACHE_EVERY), data_processing.py (label_filter-optional), architecture.py (env
overrides). Commit-when-asked.

**A/B RESULTS (2026-06-29, full coverage, fresh WS 6-epoch on 1-100, eval 101-200, fp32):**
| run | chunk | MAX | steps | train_min | rev/s | ahead | imm |
|---|---|---|---|---|---|---|---|
| base65k | 65536 | 66000 | 1020 | 15.2 | 31287 | 0.329804 | 0.296890 |
| sc8k (route R) | 8192 | 66000 | 960 | 17.3 | 27345 | 0.322033 | 0.289628 |
| sc8k_s2 (2nd seed) | 8192 | 66000 | 960 | 16.6 | 28166 | 0.321347 | 0.292042 |
| sc4k | 4096 | 66000 | 936 | 16.7 | 27973 | 0.320997 | 0.289527 |
| sc8k_m132 | 8192 | 132000 | 474 | 13.7 | 34755 | 0.329309 | 0.299410 |
| sc8k_m200 | 8192 | 200000 | 306 | 84.3 | 5440 | 0.334451 | 0.308175 |
CONCLUSIONS: **(a)** smaller chunks (8192/4096) beat 65536 by ~0.005-0.007 imm BUT with ~0.0024 seed-variance
(seed2 0.2920); 4096 ~= 8192 (no further gain) -> KEEP 8192. **(b)** higher MAX raises throughput up to a point
(66000->132000: 27k->35k rev/s, launch-bound confirmed) but HURTS accuracy at fixed 6 epochs (imm 0.2896->0.2994)
via fewer updates (960->474); and at MAX=200000 it COLLAPSES (5440 rev/s, 84 min, near-OOM at 12GB -- memory/
cache-bound) AND worst accuracy (imm 0.3082). So **higher MAX is NOT a free speedup** (132000 = throughput peak
but costs accuracy; 200000 = avoid). NET: no cheap training speedup from chunk-size OR MAX; ~15-17 min/100u @
MAX=66000 is ~the floor at adequate updates. Remaining speed levers = genuine per-step (CUDA graphs, high effort;
torch.compile Windows-blocked) or eval parallelism; otherwise live with ~20 min/experiment (train ~16 + eval ~4).
=> RESEARCH-PHASE TRAINING SETUP = sc8k (8192-chunk) db, MAX=66000, WS 6 epochs, augmentation OFF.

### Active agenda (Andrew, priority order) [OLDER -- see NEW PHASE PLAN above]
1. **Param reduction = headline** (helps throughput AND state). Champion 192,800. Big blocks: RWKV
   stacks ~70%, the two 128x128 SRS linears, the input FC. Standard levers mostly spent -> needs
   CREATIVE methods.
2. **State-only wins count** - shrink card+note, grow the CHEAP deck/preset/global. State memory ~
   entity count (many cards/notes, few decks/presets, one global), so **grow deck/preset freely (even
   10x)** to buy back accuracy lost to aggressive card/note quant.
3. **Quantization** - scoped / per-layer / hybrid schemes; QAT warm-started; revisit RWKV-edge
   (`scratchpad/rwkvedge.txt`).
4. **Creative / non-standard** (now PRIMARY for the 0.15 KB card target): **low-rank/factored card WKV
   state + QUANTIZED factors** (rank-1 int4 = 64 B incl shifts; pure-fp32 low-rank only TIES int2, see
   RESUME step 4 math) - the only path under int2's 256 B floor; per-persist state quant;
   mixed-precision outlier channels; learned-codebook / autoencoder state compression; structured
   pruning; weight-tying across layers. Full seed list in HISTORY.md. Measure every idea on the 2k loop.

**▶▶ RESUME (2026-06-29, ACROSS COMPACTION) — autonomous deck/preset-grow plan (Andrew's REFINED 4-step plan):**
"(1) moderate deck+preset grow; (2) aggressive deck+preset grow; pick whichever gives lower log loss; (3) improve
QAT; (3.5) speed up BOTH GPU training and Rust evaluation (HIGH EFFORT); (4) once card int2 + note int4 work well
(via larger deck/preset and/or better QAT), try the two-low-rank-matrices idea to shrink card state further." Run
autonomously. **NOTE the change vs the old plan: do BOTH moderate AND aggressive unconditionally, then PICK the
lower-logloss one — not "aggressive only if moderate is partial."**
- **iter41 = MODERATE grow [1,8,3,6,3]** (deck 4→8, preset 3→6; 265,614 params; CARD STATE UNCHANGED 4.25 KiB fp32
  / 0.27 KiB int2 — deck/preset are ×few-entity cheap). Pipeline `scratchpad/run_iter41_pipeline.cmd` (WS non-QAT →
  warm-started decay-QAT card int2/note int4 → export `reference/rwkv_iter41_124.safetensors` → gate). MONITOR
  `scratchpad/iter41_pipeline.log` (poll `DONE_EXIT_`). arch snapshot = `arch_iter41.py`.
- **iter42 = AGGRESSIVE grow [1,16,3,12,3]** (deck 4→16, preset 3→12 = 4× champion, 2× moderate). Pipeline
  `scratchpad/run_iter42_pipeline.cmd` — it FIRST copies `arch_iter42.py`→`rwkv/architecture.py` (the arch swap is
  baked in), then WS → decay-QAT → export `reference/rwkv_iter42_124.safetensors` → gate. MONITOR
  `scratchpad/iter42_pipeline.log`. **Run iter42 AFTER iter41 fully finishes (no GPU contention + the arch swap must
  not race iter41's python).** Launch via `detach.ps1 -Script <abs run_iter42_pipeline.cmd>`.
- **WHEN EACH DONE:** read its log `=== EVAL ===` block (champ_fp32 / qat_fp32 / qat_quant imm+ahead), **LOG to
  `optimization/qat_log.jsonl`** (mode "moderate grow [1,8,3,6,3] + decay-QAT" / "aggressive grow [1,16,3,12,3] +
  decay-QAT"; fields per the QAT section) then `python optimization/logbook.py rebuild`. SUCCESS = qat_quant imm
  ≤ champ_fp32 (0.296064) ± a hair (recovers iter39's +0.0025). After BOTH: PICK lower qat_quant imm = new champ;
  weigh the extra deck/preset params/state of aggressive vs its accuracy gain ("see if aggressive is worth it").
- **THEN (3) improve QAT + push note int2** = a LONGER WARM-STARTED QAT fine-tune from the WINNING grown WS-final
  (a few stable-LR epochs + decay, quant active, NOT from scratch — iter40 proved from-scratch QAT fails). USE THIS
  to attempt **note int2 (= the >=2x note target, 1.59->0.80 KiB)**: PTQ rejected note int4 but QAT made card int2
  nearly free, so QAT'ing `card:int2,note:int2` (with the grown deck/preset for headroom) is the path to the note
  target. Gate it. Config like the decay one but TRAIN_MODE WS, fewer epochs, LOAD_MODEL=true from the winning WS-final.
- **THEN (3.5) SPEED UP GPU training AND Rust evaluation (HIGH EFFORT, Andrew 2026-06-29).** ★ CONSTRAINT
  (Andrew 2026-06-29): keep every speedup **ARCHITECTURE-AGNOSTIC** — do NOT hardcode the current dims/layers
  ([1,4,3,3,3], d=32, 1 card layer, etc.). The arch WILL keep changing for log-loss/speed gains, so a speedup
  tailored to today's shapes is wasted effort. Derive shapes at runtime (the Rust engine already does this from
  weight shapes; CUDA graphs must shape-bucket whatever appears; batch/QAT/gate-parallelism are all naturally
  general). Profiled 2026-06-29:
  GPU training is **OVERHEAD/launch-bound, NOT compute-bound** — measured ~15% GPU util, 45 W of 200 W, 6/12 GB
  during WS (a d=32 / 200-400k-param model starves the 4070). QAT is ~4x slower still (~0.24 vs ~1.0 steps/s) due
  to its per-step Python fake-quant loop = even smaller/more-frequent launches. Levers (rated):
  - **Rust eval / the GATE — ✓ DONE 2026-06-29 (~8x, the cheapest+biggest win):** `run_qat_eval.sh` now runs the
    per-user rust passes CONCURRENTLY (split user list across processes, each pinned RAYON/OMP=1 so NPROC procs use
    NPROC cores; NPROC arg, default 10). Bit-IDENTICAL to the old sequential gate (verified iter45: same imm 0.292560
    / ahead 0.324638) -> pure speedup. Measured **841s -> ~100s** at NPROC=10. Gate is no longer the bottleneck
    (~1.7 min); training (~5 min) now dominates. Arch-agnostic (loops whatever users appear). Pass NPROC=1 for sequential.
  - **QAT 4x tax -> chunked/boundary quant (med effort, high value):** use the FAST kernel within chunks, fake-quant
    the state only at chunk boundaries instead of every step. Recovers most of the 4x. ★ SYNERGY: if DEPLOY moves to
    per-PERSIST quant (quantize only on save, not every recurrence step) QAT needs only boundary quant -> fast kernel
    AND lower deploy loss = two-for-one (ties to the per-persist creative idea).
  - **Bigger training batch (low-med):** 6 GB free, but entangled with the long user_id stream (T up to 66k);
    MAX_TRAIN_GLOBAL_LEN is a packing cap not a clean batch knob (40k already backfired). ~1.5-2x.
  - **CUDA graphs (HIGH effort, 2-5x):** the classic launch-bound fix; needs shape-bucketing (variable seq lengths
    break static capture) + care around the custom autograd.Function kernel. torch.compile (1.3-2x) may fight the
    custom kernel/JIT. Theoretical ceiling ~5-6x (the 85%-idle headroom) but structure caps easy capture.
  - ROI UPDATE (2026-06-29): gate-parallelism DONE (~8x) made the gate ~1.7 min, and the recent decay-QAT runs
    trained FASTER than profiled (~1.6 steps/s, a 496-step 16ep run in ~5 min -- the "QAT 4x tax" did not bite the
    DECAY phase). So a full QAT iteration is now ~7 min (train ~5 + gate ~1.7). Remaining GPU-training speedups
    (chunked-QAT, batch, CUDA graphs) are now LOW marginal ROI (training ~5 min, high effort, would fight the custom
    kernel). DEFERRED unless a much longer/bigger-arch training run makes GPU time dominate again. Next priority =
    step 4 (low-rank card WKV -> 0.15 KB), the last open hard target.
- **THEN (4) low-rank card WKV state -> the 0.15 KB target [✓ DONE 2026-06-29 -- 0.094 KiB, beats fp32 champ].**
  RESULT: rank-2 int4-factor low-rank card WKV + int4 shifts = 96 B (0.094 KiB), deploy imm 0.291471 PASS, BEATS
  the fp32 champion (-0.0046 imm) AND the int2 champion (-0.0044) -- low-rank is SMALLER *and* MORE ACCURATE than
  int2. Pure PTQ on iter45 weights; QAT-for-lowrank (fake-low-rank-roundtrip in training) is an untried further
  refinement. Engine: `RWKV_STATE_LOWRANK_SCOPE=card:2:int4` (nalgebra SVD per-step) + `RWKV_QUANT_SHIFTS=1`. The
  original plan/math below is retained for reference.  ORIGINAL PLAN: Needed because int2 alone
  floors at 256 B; 0.15 KB requires cutting the float COUNT. Store the card WKV state S (K×K=32×32) as U·Vᵀ (rank
  r≪32) → 2Kr floats vs K². ★ EMPIRICAL RANK SCREEN (2026-06-29, `scratchpad/analyze_card_rank.py`, 20 real card
  states from gate users via --dump-card-state, SVD energy): **rank-1 is TOO LOSSY** (energy mean 0.896, min 0.711;
  relerr up to 0.54 -- the "near rank-1" claim holds only on AVERAGE, real tail of rank-2 cards). **rank-2 is the
  sweet spot** (energy mean 0.987, min 0.944; relerr mean 0.093). rank-4 ~lossless (0.999) but 160 B int4 just
  OVER target. ★ MEMORY MATH (card = 1024 WKV + 64 shift floats; shifts 1-D so quant-only): **rank-2 int4 WKV (64 B)
  + int4 shifts (32 B) = 96 B (0.094 KiB)** clears 0.15 KB with good fidelity; rank-2 int4 WKV + int8 shifts = 128 B
  (0.125 KiB) safer; rank-2 int8 WKV = 160 B over. So TARGET = rank-2, int4 factors. (NOTE: rank-1 fp32 = 256 B =
  int2-full TIE, confirming pure-fp32 low-rank is pointless; the win needs the rank-2-int4 combo.) NEXT: Frobenius
  energy is a PROXY -- must measure LOGLOSS cost of per-step rank-2 truncation propagated through recurrence+heads.
  Build = (a) Rust low-rank card-state mode: after each card recurrence step truncate the WKV state to rank-2 (SVD
  via nalgebra) + quantize factors int4 -- this per-step == the deploy per-persist model (a card advances 1 step per
  review, state persisted between reviews); gate it PTQ-style. (b) if PTQ too lossy, QAT with a fake-low-rank-roundtrip
  (analogous to fake_quant_state -- QAT rescued int2, likely rescues rank-2 too). (c) gate. Alt (BLOCKED): smaller K
  (H=2/K=16 + int2 = 144 B) needs the K=32 CUDA-kernel rewrite -- low-rank sidesteps it.
- **HOW TO RUN AUTONOMOUSLY + ESC/COMPACTION-PROOF:** launch every training as a self-contained `.cmd` via
  `detach.ps1` (parented to WmiPrvSE, survives Esc/teardown/compaction); log to a STABLE repo path
  (`scratchpad/*.log`, NOT session temp); MONITOR via OS truth (poll log / `Get-Process` / ckpt mtime) — detached
  runs give NO tool-completion event. Re-arm a Bash watcher each turn for notifications (watcher is Esc-killable,
  training is not). Beat heartbeat each turn while actively working. Do NOT kill FSRS PIDs (the 67000s-CPU ones).
- STATUS: iter39 = QAT WINNER (deploy card int2+note int4 = 0.27+1.59 KiB, +0.0025 vs champ, PASSES gate — the
  ideal config PTQ couldn't reach). iter40 = REJECTED (from-scratch QAT). iter41 = MODERATE grow in flight (detached
  pipeline, in the FINAL gate phase — slow because 21 layers vs champ 14). iter42 = AGGRESSIVE grow FULLY PREPPED
  (configs `train_rwkv_config_iter42_{ws,qat_decay}.toml`, `arch_iter42.py`, `run_iter42_pipeline.cmd`) — launch
  right after iter41's `DONE_EXIT_`. NEW TARGETS (2026-06-29): card 0.15 KB + note >=2x (see HARD TARGETS above) —
  pursued AFTER the grow/QAT steps via note int2 (QAT) and low-rank card WKV + quantized factors.

**Ops:** Injector now 24/7 (ClaudeLoopController every 3 min; controller.ps1 only acts on stale heartbeat).
Compaction (ONLY sanctioned way, Andrew 2026-06-28) = run `claude-automation/request_compact.ps1 -Focus "<carry-through>"`
+ yield idle + STOP beating the heartbeat. `/compact <focus>` fires only from a FRESH (<=30 min) + FOCUS-bearing
flag (stale/empty = purged, no fire) so it happens ONLY when Claude itself just asked. Never hand-create
`pending_compact.txt`. Papers in
`scratchpad/{rwkvquant,rwkvedge}.txt`; poppler installed (Read tool handles PDFs). Use the CURRENT session's
scratchpad dir for logs (changes each session teardown — check the task-output paths).
**★ ESC-PROOF DETACHED LAUNCHES (2026-06-29):** the user pressing **Esc** (or session teardown) tree-kills
Claude's Bash/PowerShell background jobs — INCLUDING long training runs. WORKAROUND: launch training DETACHED
via WMI so it's parented to WmiPrvSE (a system service), not Claude. Helper: `scratchpad/detach.ps1 -Script
<abs .cmd>` runs the .cmd via `Invoke-CimMethod Win32_Process Create` (returns detached_pid + parent). Write a
per-run `.cmd` wrapper (cd, set env, python -u, redirect to a STABLE repo log path like `scratchpad/<run>.log`
— NOT the session temp dir which rotates on Esc; end with `echo DONE_EXIT_%ERRORLEVEL%`). Then MONITOR via OS
truth (poll the log / the final-checkpoint mtime / Get-Process) — detached runs give NO tool-completion event.
A Claude-side watcher (Bash run_in_background until-loop) is fine for notifications but is itself Esc-killable;
the TRAINING survives, just re-arm the watcher. Example: `scratchpad/run_qat40_decay.cmd` + `detach.ps1`.
**DATA FACT (2026-06-29):** the anki-revlogs-10k dataset has NO absolute timestamp / review-id anywhere (raw
`revlogs` parquet = card_id, day_offset[integer DAY counter], rating, state, duration, elapsed_days,
elapsed_seconds). It was anonymized — time-of-day is UNRECOVERABLE, so a time-of-day input feature is
impossible with this dataset (would need real Anki collections). elapsed_seconds (time-since-last) is already in.


---

# 5k-era LIVE STATE archive (moved verbatim from CLAUDE.md, 2026-07-15 housekeeping)

> Chronological live-state entries 2026-07-03 .. 2026-07-15, superseded by the compact
> CURRENT STATE section in CLAUDE.md. Per-iteration detail also in research_5k_verbose.md.

### LIVE STATE (2026-07-13)
- **★★ TRACK-2 ANCHOR A0 LANDED (2026-07-15 10:40): ahead 0.299857 / imm 0.269030 (n=4993,
  2,762,884 params).** Full detail: research_5k_verbose.md "Track 2 — A0 anchor". Headlines:
  (1) **1-ep budget tax at d=128 = +0.0037/+0.0044 vs the upstream 12-ep .pth** (intersection-
  paired p~0) -- epochs DO matter at 14x params (unlike d=32); structural to track 2, measured
  against A0 not upstream. (2) A0 beats champ5k_plain by 0.0036/0.0042 = what 2.57M extra params
  buy at matched budget. (3) **⚠ the 1-ep d=128 model NaNs on eval chunks >= ~500k tokens** (7
  users skipped, recorded in result/RWKV-track2_a0.nanskip.jsonl; upstream .pth is clean; d=32
  never NaNs) -> ALL track-2 comparisons use the finite-user intersection (paired_pvalue needs an
  intersection mode when A1 lands). fp32-vs-bf16 probe DEFERRED (LMDB batches are bf16; needs a
  cast shim; probe toml staged at scratchpad/track2_a0/probe_fp32.toml). Anchor json + val trace
  (= track-2 vprune ref) = optimization/champion_5k_track2.json; ckpt t2a0d_5586.pth. Fixes
  banked en route (committed): RWKV_EMPTY_CACHE_WINDOW whole-run clears (d=128 envelope creep ->
  WDDM paging), write_decay_setup MAX param (hardcoded 110000 thrashed the d=128 decay; **track-2
  .cmds MUST pass 32768 as arg 10**), get_result re-raise + NaN-skip-whole-user + skip-file
  resume, eval_sharded completeness gate (merged + skipped == rostered or exit 3).
- **★ ITER 15 = DROP REVIEW-STATE FEATURE ACCEPTED (directed, 2026-07-15 13:52) = NEW PLAIN
  CHAMPION: ahead 0.303663 / imm 0.273227** (n=5000, 0 NaN-skips, pipeline 3h09m). NOT worse --
  slightly BETTER both modes (paired vs champ5k_plain: +0.000071 p=1.5e-08 / +0.000221 p=1.6e-42;
  scaled_state was ~noise). Promoted -> champion_5k_plain.json (ckpt iter15d_1638.pth + traces =
  track-1 vprune ref). **RWKV_ZERO_FEATURES=22 IS NOW CHAMPION RECIPE -- set it in ALL future
  track-1 runs + the final QAT run.** Deploy: Anki need not compute review state (dim 22 fed 0).
- **★ fp32 PROBE DONE (2026-07-15 14:20): A0's NaN is WEIGHT-LEVEL** -- the fp32 GPU eval
  (RWKV_EVAL_CAST_FP32=1 shim; LMDB batches are stored bf16) of user 9501's 502,886-token chunk
  NaN'd identically. Structural to the short-budget d=128 anchor; NaN-skip + finite-intersection
  handling stands.
- **★ ITER 16 = PREHEAD OUTPUT GATE REJECTED (2026-07-15 17:17): ahead 0.303652 / imm 0.273409
  = +0.000011 (p=0.97) / -0.000182 (p=1.0) vs iter15 -- no-effect signature; the shared readout
  is not gating-limited. READOUT family 0/1.** Took 3 launches -- TWO INFRA LESSONS (committed
  328394e, c962f95): (1) **@torch.jit.ignore methods must NOT call SUBMODULES** (through
  scripted code the ignored body sees the raw C++ ScriptModule, 'not callable'; the NaN-except
  made attempt 1 a HOLLOW run) -> use Parameters + F.linear (grade_emb's latent same-bug also
  fixed); (2) **root-level direct Parameters are invisible to selective_cast** (root skip
  protects the fp32-excluded heads) -> bf16 child kept fp32 gate params, copy_downcast_ assert
  killed attempt 2 -> root non-excluded Parameters now cast explicitly. Smoke discipline: must
  exercise the SCRIPTED forward + selective_cast/copy_downcast_ chain, not direct Python calls.
- **-> NOW: ITER 17 RUNNING (launched 2026-07-15 17:25, pid 22268, verdict ~20:45): DIRECT
  BINARY-RECALL LOSS TERM (RWKV_PBIN_SCALE=0.5)** -- the benchmark's imm metric (BCE of
  1-P(again)) was computed as a statistic but NEVER entered the training loss ("train what you
  measure"; 0 new params; loss-reweighting family). Hook: instance-float pbin_scale (TorchScript
  reads instance attrs, not env/globals). After iter 17: 1 more track-1 iter (cross-head readout
  mix variant or permutation init), then TRACK-2 A1 (first ablation: layer cuts / d_model cuts /
  mixer cuts / LoRA dims by expected ratio-efficiency vs the per-100k gate; arch file for
  RWKV_ARCH_MODULE; MAX=32768 + decay arg 32768; vprune vs champion_5k_track2.json; comparisons
  on A0's finite-user intersection -- paired_pvalue needs an --intersect mode).**
- **★ A0 LAUNCH SAGA (2026-07-14 evening): launches 5-7.** Launch 4 (pid 20332) crept
  3.6->11.3 GB by step ~4100 (caching-allocator envelope over variable d=128 group shapes; the
  empty-cache guard stops at step 1000 BY DESIGN) -> WDDM paging, 1.06->4.3 s/step. Fix =
  **RWKV_EMPTY_CACHE_WINDOW env** (train_rwkv; default 1000 = old behavior, 0 = whole run).
  Launch 5 (every=50) SATURATED 11.9/12 GB by step ~250 -> killed; **launch 6 = every=1 window=0
  (per-step clears whole run) confirmed healthy 1.07 s/step** -- then a POWER OUTAGE (~19:20)
  rebooted the PC. **Launch 7 (pid 19660, started 23:02) = current, same config, verdict ~13:15
  2026-07-15.** Step-50 val = 0.4119/0.3879 IDENTICAL across launches 5/6/7 (seeded shuffle
  replays exactly; guard cadence numerics-neutral). ⚠ FALSE-ALARM LESSON: a val event at step 50
  (standard early ckpt) was misread as step-1000 -- vals are only comparable at the SAME step.
  Restart-from-scratch (not resume): the train loop has NO group skip on STEP_OFFSET resume; a
  mid-epoch resume on a 1-ep run re-sees early groups, drops the tail, breaks pairing.
- **★ ITER 15 PREPARED + QUEUED (Andrew's directive 2026-07-14): remove the Anki review-state
  input feature (scaled_state = dim 22 of the 92: Filtered/Review/Learn/Relearn) from the small
  model; ACCEPT REGARDLESS of logloss delta (he expects ~none) = deploy simplification.**
  Implemented as **RWKV_ZERO_FEATURES=<comma dims>** (srs_model.py + srs_model_rnn.py): zeroes
  the columns at the model input in train AND eval -> informationally removed (FC bias absorbs
  the constant); LMDBs/params/layout untouched; deploy feeds 0. Plain tensor attr + jit.ignore
  applier (ScriptModule forbids persistent=False buffers; a persistent one would pollute
  state_dict). Smoke ALL_PASS (JIT-on construction both hook states; col-22 influence check).
  Pipeline scratchpad/iter15_nostate/{run_iter15_nostate.cmd,iter15_nostate_ws.toml} = exact
  champ5k_plain recipe + ZERO_FEATURES=22, NO vprune (directed accept must complete), final
  paired_pvalue vs champ5k_plain INFORMATIONAL. **LAUNCH AT A0's DONE_EXIT (GPU handoff,
  ~13:15 2026-07-15); on finish: promote via promote_champion_5k.py --out
  optimization/champion_5k_plain.json --val-trace, record everywhere, provenance
  "adopted (Andrew, directed accept)".**
- **★ RESEARCH ITER 10 REJECTED (2026-07-13 19:48): warmup-only KD from the d=128 teacher
  (Andrew's idea; 800-step annealed target mix from a stored dump, checksum-guarded) = ahead
  0.306907 / imm 0.278222 -- WORSE both modes (-0.000277/-0.000329 vs champ5k_b1, p=1.0 both).**
  Trajectory = iter 9's exactly: led val early (-0.0026/-0.0046 @ step 500), washed out by WS
  end, finished slightly negative. **EARLY-TRAINING-INTERVENTION family 0/2 (shrink-perturb,
  KD warmup) -> DEPRIORITIZED, not closed (conduct rule 5, Andrew 2026-07-13: closing a family
  needs 3-5 in-family variants)** -- so far head starts do not survive 6554 hard-label steps at
  the 1-ep budget; untried variants if revisited: longer/never-zero KD window, KD into decay,
  permutation init.
  KD machinery stays in-repo (RWKV_KD_DUMP_OUT / RWKV_KD_MIX + exit-43 checksum guard, 78caceb).
  ⚠ OPS: the 2-parallel-shard eval WEDGED ON THE CHAMPION ARCH (both shards frozen 66+ min at
  11.7/12 GB, 100% util, full-core CPU each -- two mega-users collided; the iter-5
  elevated-VRAM-only scoping was TOO NARROW). Fix = kill tree + sequential-resume evalfix
  (run_iter10_kd_evalfix.cmd). **RULE UPDATED: ALL evals run SEQUENTIAL shards** (~45 min slower
  than a clean parallel run, never wedges = unattended-safe; iter11 .cmd already updated).
  **Iter 11 = additive GRADE EMBEDDING (Andrew's idea) REJECTED (2026-07-14 01:24): ahead
  0.307481 / imm 0.278801 -- worse both modes (-0.000851/-0.000908, p=1.0), ~2x cross-seed
  noise = real harm, no seed-pair needed.** The 4x32 zero-init bypass around the input MLP
  (RWKV_GRADE_EMB=1, +128 params) distorts the shared trunk more than it helps -- grade info
  was never bottlenecked (4 of 92 dims through the 128-wide fc). Val looked champion-level all
  run; the harm only showed at full eval. GRADE-REPRESENTATION family 0/1, deprioritized
  (rule 5); untried variants: per-stream embeddings, grade-emb into the SRS heads, LayerNorm on
  the bypass. Hook stays (env-gated, default off = byte-identical).
  **Iter 12 = SRS-HEAD RESOLUTION 64->128 REJECTED (2026-07-14 07:01): ahead 0.306899 / imm
  0.278134 -- no effect (-0.000270/-0.000241 vs champ5k_b1, p=1.0 both, inside the ~0.0004
  cross-seed band = the deck/preset null signature).** The 100u "capacity adds fail" lesson does
  NOT flip at 5k for this lever: 64 curves / 64 points are enough resolution for the
  forgetting-curve mixture. Val sat at champion parity all run (WS-end +0.0003/+0.0010),
  consistent with the null. CAPACITY-AT-5K family 0/1 so far. Clean ~5.6h run (WS 2h32m, decay
  38m, sequential eval 2h24m), no incidents.
  **Iter 13 = CHANNEL MIXER 1.0->1.5 REJECTED (2026-07-14 12:41): ahead 0.306788 / imm 0.278164
  = -0.000159/-0.000271 (p=0.999/1.0), no-effect signature. CAPACITY-AT-5K family 0/2** (head
  resolution, FFN width): the d=32 trunk is not capacity-limited at 5k -- the d=128 gap lives
  elsewhere. LAST QAT-ERA ITERATION.
  **★ METHODOLOGY SWITCH (Andrew 2026-07-14) -- supersedes methodology (a) for the research
  phase:** (1) **QAT PARKED until research closes** -- ALL screening runs (both tracks) are
  PLAIN bf16, JIT on, no codebooks (saves ~2h20m/run; plain step 0.385 s vs 1.41 quant-aware);
  ONE quant-aware run of the final champion at the very end, NO per-accept confirmations.
  champion_5k.json (QAT deploy truth, champ5k_b1) FROZEN; plain screening champion ->
  optimization/champion_5k_plain.json (promote_champion_5k.py --out flag added; plain
  candidates use RWKV_VPRUNE_REF=champion_5k_plain.json). Plain vs QAT-era logloss NOT
  comparable. (2) **TWO RESEARCH TRACKS, ~12h alternating blocks, two tables in
  research_5k.md:** Track 1 = improve the d=32 model (gate unchanged: >=0.0003 both + p<1e-4
  both, params <=225k). Track 2 = ABLATE the old d=128 model; gate **UPDATED
  (Andrew 2026-07-15): 100,000*(LL_after-LL_before)/(params_before-params_after) <= 0.0001 in
  BOTH modes** (tightened from per-50k after A0 landed: the plain-vs-plain collapse
  A0->champ5k_plain costs 0.000074/0.000086 per 50k, so the old bar accepted ablations no better
  than the collapse average; the per-100k bar demands ~1.5-1.7x better) (params strictly
  decrease; "before" = current track-2 champion; rows A0,A1,...). Track-2 anchor A0 = d=128 arch
  retrained through OUR plain 1-ep pipeline at MAX=32768 (the track-2 standard). A0 also A/Bs the 1-ep budget at 14x params. TODO
  at A0 launch: env-based arch-module selector in architecture.py (NOT the KD-dump file-swap).
  (3) **POWER-USER-AWARE EVAL LANDED (eval_sharded.py rewritten, dry-run tested):** users >=1M
  work (56 = 11.3% of eval work on 5001-10000; top-7 ~2.1M) run SOLO first (one process,
  7 threads), then 2 parallel LPT shards, then merge -- one call does all phases; worst
  concurrent pair ~2x below the wedge scale; ~1.8x over sequential; resume-safe per phase;
  --solo-threshold 0 = old behavior; RWKV_EVAL_SHARD_DIR overrides the shard dir. d=128 evals
  stay UNSHARDED (one alone ~9 GB). First E2E = the champ5k_plain eval -- watch phase-B VRAM.
  **★ ITER 14 = champ5k_plain ACCEPTED (2026-07-14 15:53) = THE PLAIN SCREENING CHAMPION:
  ahead 0.303734 / imm 0.273448** (n=5000; 3h07m pipeline: WS 91 min @ 0.82 s/step wall, decay
  22 min, eval 75 min). QAT TAX measured at n=5000: +0.002896/+0.004445 (p=0.0) vs champ5k_b1.
  Gap to the d=128 target now +0.0073/+0.0085 (was +0.0102/+0.0134 QAT). Promoted ->
  optimization/champion_5k_plain.json (ckpt champ5kplaind_1638.pth + WS trace + val trace =
  the PLAIN vprune ref for track-1 candidates); champion_5k.json (QAT) FROZEN. The phased eval
  E2E'd FLAWLESSLY: solo 9 min (mega-user 3.9 GB), phase B ~1.8 GB combined (no wedge
  exposure), 1.9x over sequential.
  ⚠ FIXED EN ROUTE: iter-11 RWKV_GRADE_EMB hook broke JIT-on construction (TorchScript
  resolves attrs in dead branches; hidden all QAT era by NO_JIT) -> @torch.jit.ignore
  indirection in srs_model.py, smoke-tested both hook states. train_rwkv swallowed that
  traceback with exit 0 -- the .cmd artifact gate caught it (always gate phases on artifacts).
  **-> NOW: TRACK 2 ANCHOR A0 RUNNING (4th launch, detached pid 20332, 17:02, verdict ~07:15
  tomorrow):** the ORIGINAL d=128 arch (2,762,884 params, in-log confirmed) retrained through
  the plain pipeline via the NEW RWKV_ARCH_MODULE env hook (architecture.py bottom: exec's a
  standalone config file, replaces DEFAULT_ANKI_RWKV_CONFIG wholesale -- bypasses all
  default-build env hooks; scratchpad/architecture_old_d128.py verified). **MAX=32768 -- THE
  TRACK-2 STANDARD (pairing needs it identical across all track-2 runs).** Launch saga:
  MAX=66000 THRASHED (11.85/12 GB WDDM spill, 40 s/step -- the 100u-era "66000 fits" fact
  doesn't transfer, 5k packs fuller groups) and 49152 still thrashed (13.3 s/step, allocator
  bloat on 3x16384 packing); 32768 = 2x16384 clean packing -> 3.6 GB, 1.06 s/step, ~22k
  steps/epoch. ⚠ COVERAGE FACT (probe 2026-07-14): max single batch in train_db_5k_h1 =
  16,384 tokens -> ZERO data drop at ANY MAX >= 16,384 (the "don't go below 66000 = data
  drops" rule was sc8k-era, NOT true of the 5k db). TWO LATENT BUGS FIXED en route:
  (1) train_rwkv's blanket NaN-except now prints the real traceback (bare asserts have empty
  str(e) -- it had hidden the hollow-compile run and this); (2) utils.KeyValueAverage
  .get_value returned via bare assert n>0 -- early groups can have ZERO equalize-counted
  reviews (first seen at small MAX), and the throw landed AFTER backward but BEFORE
  optimizer.step = silently skipped weight updates; now returns NaN (wandb-only consumer).
  Eval = SINGLE process (--shards 1 --solo-threshold 0; d=128 can't share 12 GB). Ends with
  informational paired vs base5k (the 1-ep-budget check at 14x params). A0's finals + val
  trace = the track-2 "before" anchor + its vprune ref.
  Track-1 queue (plain era, ~3h/iter): prehead output gate, cross-head readout mix, loss-term
  reweighting, permutation init (LOW). Track-2 queue after A0: layer cuts / d_model cuts /
  mixer cuts / LoRA dims / head-width cuts, ranked by expected ratio-efficiency.
- **★ RESEARCH ITER 9 REJECTED (2026-07-13 12:58): shrink-perturb init (lam=0.5, fresh seed 777,
  RWKV_INIT_BLEND hook, else exact champion recipe) = ahead 0.307373 / imm 0.278926 -- WORSE both
  modes (-0.000744/-0.001033 vs champ5k_b1, p=1.0 both), beyond the ~0.0004 seed noise = real harm,
  no seed-pair needed.** Trajectory lesson: the warm init LED the champion's VAL curve all WS
  (-0.010 @ step 1000 shrinking to -0.0006 @ 3500) yet ended net NEGATIVE at full eval -- mid-WS
  val leads from a warm start do NOT predict the final verdict. Both lam endpoints (~0 =
  from-scratch champion, ~1 = the 2-ep budget A/B) are champion-level and the midpoint sits below
  -> **data-driven-init scheme A (shrink-perturb at lam=0.5) rejected; family DEPRIORITIZED,
  not closed (conduct rule 5); lam probe {0.3,0.7} judged not worth GPU for now; scheme B
  (permutation init) queued LOW.** The RWKV_INIT_BLEND hook stays (eed7cb5,
  env-gated, plain path untouched). Artifacts: scratchpad/iter9_sp/, result/RWKV[-P]-iter9_sp.jsonl.
  **-> NOW: iter 10 = warmup-only KD from the d=128 teacher** -- machinery committed 78caceb:
  train_rwkv RWKV_KD_DUMP_OUT teacher-dump mode + RWKV_KD_MIX annealed target-mix student mode
  (per-step labels-checksum pairing guard, mismatch = exit 43 never a silent skip; srs_model
  get_loss(kd_mix=) mixes TARGETS exactly -- BCE/CE are linear in the target; window 800 WS steps,
  alpha 1->0; clear RWKV_KD_MIX before decay -- decay replays the epoch-0 stream). Sequence: dump
  smoke KDSTEPS=3 (d=128 VRAM check) -> full 800-step dump (~20 min, scratchpad/iter10_kd/dump
  ~0.9 GB) -> run_iter10_kd.cmd (~4.7h). ⚠ the dump .cmd FILE-SWAPS rwkv/architecture.py --
  never overlap with any other rwkv launch. Queue after 10: SRS-head resolution 64->128 (capacity
  re-test at 5k data -- the 100u "capacity rejects" lesson was data-limitation-scoped), channel
  mixer 1.0->1.5, prehead output gate, cross-head readout mix, loss-term reweighting.
- **★ STATE-SIZE LADDER CLOSED (2026-07-13 08:04): 0 accepted rungs across 5 iterations (4-8).**
  Per-stream arch hooks live (d6fca68): `RWKV_STREAM_HEADS` (H=1 doubles that stream's per-entity
  WKV state ~param-free) + `RWKV_STREAM_LAYERS` (~10.4k params/layer). Verdicts (all paired vs
  iter 2 champ5k_b1, n=5000): **deck H=1** (iter 4) null p=1.0; **preset H=1** (iter 5) null p=1.0;
  **user H=1** (iter 6) NEAR-MISS +0.000345/+0.000258 (imm short by 0.000042, in-seed p 1e-20/1e-29);
  **user H=1 + 4L** (iter 7) mode TRADE (ahead -0.000299 / imm +0.000604); **iter 8 lad_user1b =
  the seed-pair test of iter 6 (seed 4321) came back NULL** -- ahead 0.306674 (-0.000044, p=0.88) /
  imm 0.278039 (-0.000146, p=1.0) = the deck/preset no-effect signature. **Iter 6's signal did not
  replicate -> substantially SEED LUCK; reject stands per the pre-declared branches.** LESSONS:
  (1) no stream is state-capacity-limited at d=32/H=2 -- 2x recurrent memory clears nothing;
  (2) ⚠ in-seed Wilcoxon p (even 1e-29) measures per-user delta consistency, NOT cross-seed
  robustness -- cross-seed spread on the SAME recipe is ~0.0004 both modes, so **any single-run
  margin < ~0.0005 defaults to seed-pair confirmation before acting**; (3) widened vprune
  (0.006/0.008) ran clean across a seed change. Artifacts: scratchpad/lad_user1b/ (laduser1bd_1638
  + cbs), result/RWKV[-P]-lad_user1b.jsonl; pipeline template = scratchpad/lad_user1b/
  {run_lad_user1b.cmd,lad_user1b_ws.toml} (vprune-ON candidate runs; exit-42 branch; sequential
  sharded eval + gate in-.cmd).
  ⚠ EVAL-SHARD VRAM LESSON (2026-07-12): 2-parallel-shard eval WEDGES on elevated-VRAM rungs
  (K=32 streams: chunk-state buffers ~+0.8 GB/shard on 1M-token batches -> WDDM oversubscription,
  100% GPU util at 10-50x slow). RULE: such rungs -> sequential shards (get_result resumes
  per-shard) then eval_sharded relaunch-skip-merge; template in run_lad_user1b.cmd.
- **-> NOW: the >=50-iteration RESEARCH PHASE [[research-phase-conduct]]** (many idea FAMILIES,
  arch + training pipeline, lit review + own ideas, retry near-misses as variants). Queued seeds:
  warmup distillation from the d=128 teacher (design in notes), data-driven init (shrink-perturb/
  permutation-init), cross-head readout mix (PHA analog), LIT_REVIEW.md queue. Iter numbering
  continues from 9. Champion unchanged = iter 2 champ5k_b1 (0.306629/0.277893, 193,724 params).
- **★ HP TUNING CLOSED (2026-07-12): champ5k_t1 (the tuner winner: wd 0.01->0.2 + dropout_scale
  1.0->0.5) REJECTED at full eval** -- ahead 0.307174 / imm 0.278570 = WORSE than champ5k_b1 by
  0.000545/0.000677 (p=1.0 both) despite winning tune-eval 5001-5200 by +0.0008/+0.0010.
  **champ5k_b1 REMAINS CHAMPION; its HPs are confirmed vs 19 alternatives** (peak_lr, warmup, wd,
  clip, decay_ratio, adamw_beta2, dropout_scale, cb_lr_mult all settled at champion values on the
  full-eval verdict). ⚠ LESSON (bank + research_log note): the 200-user tune-eval CANNOT resolve
  sub-0.001 HP effects -- even in-subset paired p=5e-8 inverted at n=5000; any future sub-0.001
  tuner verdict needs full-eval confirmation before adoption. Round-2 levers wired + kept
  (RWKV_ADAMW_BETA2 / RWKV_DROPOUT_SCALE / RWKV_CB_LR_MULT, defaults byte-identical). The
  VALIDATION prune (replaced the sign-biased train-loss rule mid-tuning) ran the whole descent
  clean: 0 kills, no false fires, joint-AND correctly spared single-mode transients (incl.
  cb_lr_mult=10's imm-only breach); its estimated-logloss formula is now window-mean x
  fitted-alpha anchored on the baseline journal row (fa724c0). Trial .cmds now GATE every phase
  on exit codes (d289d9a, after a WS crash cascaded into decaying a step-50 ckpt -- caught before
  the journal). NEXT = state-size ladders (deck <=5x -> preset <=10x -> global <=50x, FULL-eval
  gate each rung), then the >=50-iteration research phase [[research-phase-conduct]].
- *(2026-07-08 era below)*
- **★ FIRST 5k CHAMPION PROMOTED (2026-07-08 18:23): champ5k_r1 = ahead 0.306572 / imm 0.278323**
  (quant-aware q72u + per-run learned cbs, n=5000 both modes, eval 5001-10000). Behind the d=128 fp
  target (0.296385/0.264905) by +0.0102/+0.0134 -- THE GAP THE PHASE NOW CLOSES. champion_5k.json
  carries ckpt champ5kd_3277.pth + cb_wkv_final/cb_shift_final + the 13108-step WS trace (= Wilcoxon
  prune ref). Pipeline wall-clock ~7.0h clean (WS 5h @ ~1.36 s/step real, decay 72 min, eval 66 min
  2-sharded, GPU-bound at 2 shards -> 2 stays the default). TWO LATENT BUGS hit+fixed en route:
  (1) LEARN=1 optim resume param-group mismatch at the WS->decay seam (f71f43b -- cb groups now
  register pre-load when the saved state has them, moments resume); (2) per-user lmdb env leak in
  get_benchmark_info killed eval shard 0 at user 2007 with a bogus ENOENT swallowed to exit 0 --
  the n=5000 finish gate caught it (7d095e3 -- env now opened once/process). Results recorded:
  research_log.jsonl + research_5k.md (p-value col = 1.0/1.0 vs target, honest) + log.md rebuilt.
- **★ LIVE LOSS PLOT (2026-07-08, Andrew asked):** `scratchpad/liveplot/liveplot.py` = matplotlib
  window, champion-vs-candidate WS train loss (ahead+imm panels), EMA-smoothed, paired one-sided
  Wilcoxon p + mean delta per panel, warmup-end + decay-start vlines, 15 s refresh. Auto-discovers
  the newest `*_ws_trace.jsonl` (tuner trials AND champion runs both set RWKV_STEP_TRACE), champion
  ref = champion_5k.json embedded trace -> works for ALL runs; switches to a new trial automatically.
  Relaunch: `detach.ps1 -Script scratchpad/liveplot/run_liveplot.cmd` (survives Esc; close window to
  stop). NOTE: WMI-launching pythonw GUI directly stalls at 0 CPU -- use the .cmd wrapper.
- **★ BUDGET A/B RESOLVED + ADOPTED (2026-07-09 01:40): champ5k_b1 = NEW CHAMPION at HALF budget.**
  WS 1 ep (6554) + 0.25 ep decay (1638), otherwise champ5k_r1's exact recipe. Full-eval finals
  **ahead 0.306629 / imm 0.277893** -- paired vs r1: ahead -0.000058 (p=0.31, indistinguishable),
  imm +0.000430 BETTER (p=6.1e-62). The 2nd WS epoch (same 5000 users reshuffled) adds NOTHING
  (data-variety lesson holds at 5k). SIZE/SPEED accept; **1-ep budget now standard for ALL 5k runs**
  (tuner trials AND research runs; champion pipeline ~3.5h: WS 2h27m + decay 37m + eval 89m).
  Adoption executed: promoted (champion_5k.json = ckpt champ5kb1d_1638.pth + its cbs + 6554-step
  trace = the new prune ref), hp_tuner WS_EPOCHS=1, 2-ep journal archived
  (tuner_5k_log_2ep_era.jsonl), new baseline recorded (5001-5200: 0.294490/0.270492), tuner loop
  RELAUNCHED (1-ep era; 2-ep prune verdicts for peak_lr 7e-4/1.4e-3 will be re-tested at 1 ep).
  Pre-ship note: the final champion should get ONE full-budget (2 ep) confirmation run.
- **★ HP TUNING RUNNING (launched 2026-07-08 18:35, detached pid 4468):** hp_tuner_5k `loop` --
  coordinate descent over peak_lr/warmup/wd/clip/decay_ratio, trials are self-recording full-recipe
  .cmds (WS 2ep + decay + tune-eval 5001-5200, LEARN=1 cbs, Wilcoxon-pruned vs champ5k_r1's trace).
  Baseline recorded (5001-5200 subset: 0.294204/0.270881). Journal optimization/tuner_5k_log.jsonl;
  loop log scratchpad/tuner5k/loop.log; ~6h/full trial, prunes much cheaper. Monitor armed.
- **FETCH WORKERS = 4 EVERYWHERE (Andrew 2026-07-08, RAM):** every training/eval launch uses
  NUM_FETCH_PROCESSES=4 (was 7-10; each worker holds ~2.6 GB at MAX=110000, fetch is over-provisioned --
  ~4 ms get() waits; worker count never affects batch content/order). Already set in: hp_tuner_5k
  (NUM_FETCH), write_decay_setup, write_eval_toml, champ5k_r1_ws.toml (the copy-from template for future
  hand-written WS tomls). Check any NEW toml against this.
- **★ EVAL CPU PATH VECTORIZED (2026-07-08, byte-identical):** extract_p / get_stats / run() raw-gathers
  were per-review Python loops (300k-user cost: extract_p 308->118 ms, get_stats 1151->87 ms x2/user);
  now numpy dict(zip)+searchsorted (`_eq_gather`), exact dtypes preserved. Verified: 6-trial exact-equality
  harness (scratchpad/eval_speed/stats_ab.py ALL_PASS) + E2E GPU A/B 3 users = result jsonls BYTE-IDENTICAL.
  RNN/trace callers auto-fallback to the old loop (tensor dicts). champ5k_r1's eval picks it up.
  FOLLOW-UP at eval launch (~16:40): sample per-shard VRAM/GPU-util -> maybe --shards 3-4 for future evals.
- **★ SHIFT-PQ SEARCH KERNEL BANKED (2026-07-08, direction #3): quant-aware step 1.207 -> 0.996 s/step
  (1.21x; stacked 1.65x over NO_JIT today).** ~45% of the q72u step was the learnable shift-PQ search
  running eager torch.cdist().argmin() (sqrt+clamp+argmin over a never-needed ~1.8 GB N x 4096 distance
  matrix, 16 calls/step). New `rwkv7_pq_argmin` CUDA kernel (row-tiled, SUB-templated, first-strict-min
  ties = cdist semantics; 5.9 vs 23.9 ms/call): index-identical on 330k-row + exact-tie tests, QAT
  goldens BITEXACT_PASS after rebuild, escape hatches RWKV_SHIFT_SEARCH_KERNEL=0 (-> matmul tier) /
  RWKV_SHIFT_SQ_SEARCH=0 (-> cdist). CPU tensors auto-fallback (RNN/Rust paths untouched). ⚠ DISCOVERY:
  the compiled frozen env is NOT run-to-run bit-reproducible (3-arm A/B: identical-env controls diverge
  ~step 27; per-step trace noise <=3e-4, weight drift 1.7e-2 @ 110 steps) — bit-exact E2E gates are
  unattainable under it; unit-level index proofs + noise-class drift comparison are the standard now
  (Wilcoxon prune pairing unaffected: zero-mean noise). Wall-clock gap CLOSED (1184 ms GPU-busy / 1207
  wall = GPU-bound; host-side lever dead). Plain step re-profiled 385 ms = flat tail confirmed.
  Champion-run training now ~4.6 h. Details: research_5k_notes.md "Speedups banked" 2026-07-08.
- **★ QUANT RESEARCH CLOSED + FULLY PORTED (2026-07-08).** The sibling (`rwkv-state-quant`) finished its
  bit-descent 2026-07-07: final champion **q72u = 72 b/layer (9-byte card)**, 2-seed-confirmed, details in
  the CHAMPION "DEPLOY config" block above. Its full 2026-07-07 code stack (CUDA joint-uv/norm-quant/warm
  search + train_rwkv QAT wiring + the complete Rust engine) landed here in `1d3b5b8` (the sibling's Claude
  verified byte-identical champion eval from OUR build); the RESULTS layer (champion artifacts ->
  `reference/`, deploy env, methodology-(a) QAT env in `hp_tuner_5k.py`, lesson bank) ported 2026-07-08.
  Open follow-ups from the port: (i) ~~per-run learnable-cb wiring~~ DONE 2026-07-08 (LEARN=1 in QAT_ENV;
  resolve_run_cbs.py repoints env at WS->decay and decay->eval seams; champion_5k.json carries
  ckpt+cb_wkv+cb_shift; a champion's evals/deploys use ITS OWN cbs), (ii) ~~JIT unverified~~ RESOLVED
  2026-07-08 (scratchpad/jitab A/B/C): TorchScript FIXED on the grafted paths (instance-bool shift_pq_on +
  jit.ignore fake_pq_shift + typed kd tuple) but JIT vs NO_JIT is a WASH (1.643 vs 1.658 s/step);
  **ADOPTED + FROZEN 5k-family env = NO_JIT + the sibling's sanctioned round-4 flags (COMPILE=student +
  ROT_CACHE + FAST_EMB + EMA_FOREACH + NO_MEMFILL) = 1.207 s/step (1.37x). Never flip flags inside the
  family. ⚠ COMPILE runs MUST call vcvars64 first (no cl.exe -> inductor errors swallowed by the
  NaN-except as hollow skipped batches, exit 0). q72u-era quant-aware step at MAX=110000 = 1.21 s (the
  old ~450 ms predates joint-search/shift-PQ/learnable cbs); champion run ~= 5.6 h**, (iii) 5k-phase
  state-size gates: card/note budgets should now be interpreted against the 72-b deploy format.
- *(2026-07-03 era below)*
- **★ QUANT PORT DONE (2026-07-03): the sibling's research is FINISHED and its machinery is IN-REPO.**
  Fused QAT CUDA kernels (full-matrix int-N + rank-1 low-rank with PQ branch, 150-490x over the Python
  loop), PQ codebook `reference/pq_cb_m2b8.txt`, shift-QAT (JIT-annotated here; sibling ran NO_JIT),
  int3 + RWKV_QAT_SHIFT_SCOPE, and train_rwkv **LR+WD clobber fixes** (optim load silently restored saved
  lr/initial_lr/weight_decay over config/env -- affected EVERY warm-started run) + non-finite loss/grad
  guards. Validated here: plain path bit-exact vs golden; PQ parity 3.2e-07; int-N 7.5e-04; 25-step QAT
  smoke green (`scratchpad/qat_parity/`). Deploy recipe + numbers: see CHAMPION section "DEPLOY config".
- **★ QAT KERNELS OPTIMIZED 37x (2026-07-03, bit-exact):** see the SPEED section -- quant-aware 5k runs
  are back to ~6-7 h (were headed for ~30-40 h). Profile hook added: `RWKV_PROFILE_STEP=N` +
  `RWKV_PROFILE_COUNT` in train_rwkv -> bucketed kernel self-time summary, then exit.
- **★ TELEGRAM BRIDGE LIVE (2026-07-03):** Andrew can steer this session from his phone + sees mirrored
  output (see Ops). His injected messages arrive Esc-first (interrupt, then message).
- **★ 5k LMDB BUILD RUNNING (launched 2026-07-03, detached, 6 threads):** `scratchpad/run_build_5k.cmd` ->
  6 sequential resumable steps (find_equalize 5001-10000 -> test_db 5001-10000 (F:) -> train_db 1-5000 (C:)
  -> find_equalize 1-5000 -> test_db 1-5000 -> train_db 5001-10000 (F:)); log `scratchpad/build_5k.log`;
  ~2-4 days. Eval data for 5001-10000 lands FIRST so the d=128 baseline eval can start before the train_dbs
  finish. Monitor via OS truth; the 6 configs are `rwkv/*_5k_*.toml` (PROCESSES=6).
- **★ EVAL SHARDING READY (2026-07-03, Andrew-approved):** `optimization/eval_sharded.py --config
  <eval toml>` = 2-process size-balanced (LPT) full eval, ~1.5-2x wall-clock, numerics-IDENTICAL
  (additive USERS_FILE selector in get_result; merge + means printed). d=32 evals only (two d=128s
  OOM); E2E smoke pending -- watch the first champion-era sharded eval. Details in notes.
- **★ BASELINE-TO-BEAT LANDED (2026-07-03): d=128 on 5001-10000 = ahead 0.2964 / imm 0.2649**
  (0.296385/0.264905, n=5000 both modes, fp unquantized; consistent with the published 10k-pooled
  0.29743/0.26600; recorded in research_5k.md; result jsonls result/RWKV-base5k*.jsonl; arch restored).
- **⚠ GPU HOLD (Andrew 2026-07-04): do NOT launch GPU training/evals — he is running his own quant
  experiments. Champion run waits for his GO.**
- **★ STEP3 DONE 2026-07-04 07:00 (train_db_5k_h1 complete, exit 0; STEP4 find_equalize 1-5000 running).
  `count_groups_5k.py` run: GROUPS_PER_EPOCH = 6554 → groups_5k.json (hp_tuner prereq DONE). Champion-run
  arithmetic: 2 WS ep = 13,108 steps + decay 0.2–0.8 ep → total ~14.4k–18.4k steps ≈ 1.8–2.3 h clean.
  EVERYTHING for the champion run is staged — only the GPU hold gates it.**
- **★ TONIGHT'S DIRECTION (Andrew 2026-07-08, supersedes the NEXT list below where they differ):**
  (1) ADD CODEBOOK LEARNING to 5k runs (per-run learnable cbs: train with RWKV_QAT_PQ_LEARN=1 +
  RWKV_QAT_SHIFT_PQ_LEARN=1, export each run's learned cbs, point that run's quant-aware EVAL + any
  deploy at ITS OWN exported cbs — the promote/champion flow carries cb artifacts with the ckpt);
  (2) TURN JIT ON (A/B TorchScript on the grafted q72u paths: parity + speed; drop RWKV_NO_JIT if clean)
  -> compaction about here; (3) hunt any remaining speedups (profile the q72u quant-aware step — joint
  search / shift-PQ / norm paths are new surface; check the sibling's speed-round flags for portable
  wins); (4) FIRST REAL 5k CHAMPION RUN (champion-HP, quant-aware, RWKV_STEP_TRACE -> promote);
  (5) HP TUNING (hp_tuner_5k); (6) STATE-SIZE KNOBS in this order, each until gain <0.0003 (the phase
  threshold) or its ceiling: deck up to 5x -> preset up to 10x -> global up to 50x. **RULE (write-down,
  Andrew 2026-07-08): card and note state sizes REMAIN FIXED — the only exception is an architectural
  change that makes a card/note state-size change INEVITABLE (not a tuning knob, a structural
  consequence).** (7) then any architectural improvements at my discretion (queued ideas: warmup
  distillation, data-driven init, cross-head readout mix, LIT_REVIEW).
- **NEXT (per methodology g), in order once data allows:** (1) ~~d=128 baseline eval~~ DONE (above);
  (2) ONE champion-HP 5k run with per-step WS trace (RWKV_STEP_TRACE) + quant-aware forward -> promote via
  `promote_champion_5k.py`; (3) HP tune -- `hp_tuner_5k.py` REPOINTED to FULL 5k 2026-07-03 (train 1-5000
  @ MAX=110000, tune-eval 5001-5200, QAT env in every trial's WS+decay+eval, proxy-era journal archived to
  tuner_5k_log_proxyera.jsonl; PREREQ after STEP3: `python optimization/count_groups_5k.py` -> groups_5k.json).
  ALL live 5k tooling now trains on 1-5000 and evals on 5001-10000 ONLY (verified sweep 2026-07-03); the
  100u/1500u dbs are no longer referenced by anything live (kept on disk, C: has 383 GB free). Any TIMING
  numbers taken while build workers run are fetch-contaminated; take final numbers with the build idle.
- Queued analysis (task #18, Andrew 2026-07-03): **irreducible-entropy estimate** -- cross-model
  residual covariance of the TWO disjoint-trained d=128 .pths on users 1-100 (seen by neither) ->
  irreducible-Brier -> Beta-translated LogLoss floor; + constant-retention baselines H(p-bar).
  Design in notes "Queued analysis" section; needs build STEP4+5 (test data for 1-100); ~30 min GPU.
- Queued research ideas: data-driven init (shrink-perturb / permutation-init, post-HP-tune -- notes
  "Queued idea" section); **warmup-only distillation from the d=128 teacher** (Andrew 2026-07-03: soft
  targets from `RWKV_trained_on_101_4999.pth` for the first ~200-800 steps only, annealed 1->0, then hard
  labels so the student can surpass the teacher; STORED-dump design -- teacher+student can't share a
  process (module-level arch config) -- full design + gate fit in the notes "Queued idea" section;
  post-HP-tune; test SEPARATELY from data-driven init, both touch early training); cross-head readout
  mix (PHA analog, LIT_REVIEW, low-med). Lit-review queue: `optimization/LIT_REVIEW.md`. Everything
  through the quant port is COMMITTED + pushed (local == GitHub).

