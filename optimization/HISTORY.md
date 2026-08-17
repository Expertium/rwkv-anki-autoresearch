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



---

## 5k-era LIVE STATE archive (moved out of CLAUDE.md 2026-07-26)

Verbatim copy of the `### CURRENT STATE` section as it stood when the two tracks merged at
A18 and work continued as iter 31. It had grown to 45% of CLAUDE.md, which is re-injected
into every turn; the chronology below is superseded by `research_5k_verbose.md` (per-iter
detail), `research_5k.md` / `log.md` (numbers) and git history. Kept unedited for provenance.

### CURRENT STATE (updated 2026-07-15 — KEEP THIS SECTION SHORT: champions, live run, queue, live rules. Superseded chronology moves to optimization/HISTORY.md "5k-era LIVE STATE archive"; per-iter detail lives in research_5k_verbose.md)

**Champions / anchors:**
- **Track 1 (d=32 plain) CHAMPION = iter 29 `iter29_muon` (accepted 2026-07-21 16:05):
  ahead 0.302033 / imm 0.271440 ON THE VAL HALF (5001–7500, n=2500, 0 nanskips — the
  FIRST val-split verdict; val-half absolutes are NOT comparable to full-range iters
  ≤28), 171,453 params** (`champion_5k_plain.json` = ckpt
  `scratchpad/iter29_muon/iter29d_1638.pth` + WS/val traces = the track-1 vprune ref).
  **= iter 26 + hybrid Muon+AdamW (rwkv/muon.py) — the first OPTIMIZER-family win:
  matrix wd-groups on Muon (lr 0.02, momentum 0.95 nesterov, NS5, aspect-scaled,
  decoupled wd at the AdamW-equivalent rate), rest bit-exact functional AdamW. vs
  iter 26 same-users: ahead +0.000143 (p=2.5e-06), imm +0.000485 (p=6.5e-71, the
  phase's largest imm gain).** Champion recipe env (set ALL in every future track-1 run
  + the final QAT run): RWKV_NO_AHEAD_RESIDUAL=1, RWKV_ZERO_FEATURES=22,
  RWKV_PAVA_LAMBDA=0.1, RWKV_PROBE_DENSITY=0.08, **RWKV_GRU_HEAD=3**,
  RWKV_STRIP_L0_VLORA=1, RWKV_STATE_CLAMP_TAU=300, RWKV_STATE_CLAMP_WINDOW=32768,
  **RWKV_MUON=1, RWKV_MUON_LR=0.02, RWKV_MUON_MOMENTUM=0.95** +
  H=2/K=16 + HP {peak_lr 1e-3, warmup 200, wd 0.01, clip 0.25} + MAX=110000.
  Optimizer is train-time only — nothing ships to Rust. Val-lag lesson now
  BIDIRECTIONAL (Muon trailed the 10-user val all WS tail, won eval decisively).
  PAVA middle-junction power strongly negative in ALL GRU/PAVA iters (−1.44/−1.44/−1.59).
  **Deploy contract:** learned-power PAVA rectifier on the 4 counterfactual button
  predictions (duration imputed to the frozen train median `scratchpad/iter23_pava/
  duration_median.json`) + per-step state clamp — Rust ports queued. Lineage kept:
  iter 26 (0.303942/0.273353 full-range, GRU N=3) → iter 25 (0.304427/0.273441, N=2,
  size-exception accept) → iter 23 (0.304220/0.273423, PAVA champion, 64-basis head) →
  iter 22 (0.304497/0.273539, no-residual re-baseline) → iter 15 (0.303663/0.273227,
  last with-residual); iter 14 = QAT tax ref (+0.0029/+0.0044).
- **★ Track 2 CHAMPION = A18 `track2_a18` (ACCEPTED 2026-07-26 by Andrew's directed verdict
  change; the auto-verdict was reject-on-ratio at 108%/111% of bar): ahead 0.299302 / imm
  0.268390 ON THE VAL HALF (n=2500, 0 nanskips), 557,246 params = 4.95× below the original
  2.76M (79.8% cut); per-card state 2,880 floats (−56% vs A15)** (`champion_5k_track2.json`
  = ckpt `scratchpad/track2_a18/t2a18d_5586.pth` + WS/val traces = the track-2 vprune ref).
  **= d_model 80 (5 heads × K=16) + LoRA decay/a/gate 4, v0-mix 2; arch
  `scratchpad/track2_a18/architecture_d80_lora4.py`.** Andrew's call: *"Let's accept A18 and
  continue track 1 with it"* — **the ≥5× product goal outranks a marginal-RATE gate missed
  by ~10%**; in absolute terms A18 costs only +0.000960 ahead / +0.000532 imm cumulative vs
  A0 (≈1/3 of what the matched-param GRU baseline gave up). Precedent = iters 23/25/26.
  **THE WIDTH LADDER IS CLOSED: two independent draws at d=80 (A17 112%/83%, A18 108%/111%)
  ⇒ genuine accuracy floor; d=64 (A16) is ~180% of bar. 4.95× is the end of the width road.**
  Side-finding: the second LoRA halving is NOT free at d=80 (+0.00002/+0.00009 for −27.5k)
  whereas A14's first halving IMPROVED both modes at d=128 — the lever flips sign as the
  trunk narrows ⇒ **the model is now genuinely capacity-limited**, so further gains must come
  from ALGORITHMS, not shape. Prior champion A15 (0.299031/0.268111, 808,762 params, 3.41×,
  ckpt `scratchpad/track2_a15/t2a15d_5586.pth` — kept as the gate-clean fallback).
  **⚠ CPU-INFERENCE REALITY CHECK (Andrew
  2026-07-25: "I told you to do ablations hoping that fewer params → faster CPU inference
  in Anki"; state size is quantization's job, training speed serves only us): measured
  today in `optimization/CPU_INFERENCE.md` — in the PYTHON RNN path a 4.5× arithmetic cut
  buys only 1.24× wall-clock and PLATEAUS after A14 (A15/A16 width cuts buy ~nothing),
  because that path runs at 0.08–0.30 GMAC/s vs a core's 5–20 = OVERHEAD-bound: cost
  tracks op count (layers × streams), not width. 1 thread beats 3 and 6 → deploy
  single-threaded. The deploy path is Rust (~10× faster, far less per-op overhead) where
  width SHOULD pay off, but `rust/rwkv-infer` does not yet support the track-2 arch (GRU
  head, STRIP_CMIX, parameterized d_model/H) — PORTING IT IS NOW THE GATING WORK for
  answering whether the ablations bought user-visible speed.** Bench:
  `python optimization/cpu_infer_bench.py`. **TRAINING SPEED IS MONOTONE IN WIDTH
  (measured 2026-07-25 on Andrew's question; an earlier "speed did NOT improve" note here
  was WRONG — it anchored on a startup-inclusive first print): median steps/s A0 0.933 →
  A7 1.022 → A9 1.203 → A14 1.200 → A15 1.434 → A16 1.746 = 1.87× faster than A0 at 7.11×
  fewer params — sublinear in params, as the elementwise-dominated profile predicts.**
  **NEXT = ITER 31 (Andrew 2026-07-26, second half of the same message + his naming
  correction "it shouldn't be called A19, it should be iter 31, first table in research
  5k"): carry the three track-1 ALGORITHMIC wins onto the A18 trunk as one bundle — PAVA
  (iter 23) + GRU N=3 (iter 26) + Muon (iter 29), i.e. exactly the iter-29 champion
  recipe's extra flags. Env deltas vs A18: RWKV_PAVA_LAMBDA=0.1 + RWKV_PROBE_DENSITY=0.08,
  RWKV_GRU_HEAD 2→3, RWKV_MUON=1 + RWKV_MUON_LR=0.02 + RWKV_MUON_MOMENTUM=0.95. 558,212
  params (+966). Gate = ordinary ACCURACY iter vs A18 (both modes ≥0.0001 after 4-dp
  rounding + p<0.0001), not the ratio gate. De-bundle precedent if it regresses =
  A10→A11.** Prior anchor A13
  (0.298837/0.267805, 1,468,724 params, the ZERO_FEATURES=22 re-anchor) and A14
  (0.298798/0.267746, 1,380,660, LoRA halving — better both modes).
- Superseded track-2 detail: A13 `track2_a13` (promoted 2026-07-23 10:50, DIRECTED
  re-anchor): ahead 0.298837 / imm 0.267805 val half, 1,468,724 params, ckpt
  `scratchpad/track2_a13/t2a13d_5586.pth`.
  **= A9 arch/recipe + RWKV_ZERO_FEATURES=22 (Anki card-state input removed, Andrew's
  2026-07-22 both-tracks directive; fixes the track-recipe divergence — track 1 has
  zeroed it since iter 15). THE MEASURED PRICE at d=128: ahead +0.000212 / imm
  +0.000190 worse than A9 (both p≈1.0) — OPPOSITE SIGN vs d=32 (iter 15: ~free);
  recorded, directive stands (revert = re-point the json to A9). Full track-2 env now:
  RWKV_ARCH_MODULE=<champion arch>, RWKV_GRU_HEAD=2, RWKV_STRIP_L0_VLORA=1,
  RWKV_ZERO_FEATURES=22, RWKV_STATE_CLAMP_TAU=300, RWKV_STATE_CLAMP_WINDOW=32768,
  RWKV_NO_AHEAD_RESIDUAL=1, RWKV_STRIP_CMIX=user_id:0,user_id:1,user_id:2,preset_id:0,
  preset_id:1,preset_id:2,deck_id:1,deck_id:2,card_id:1.** Prior champion A9
  (0.298625/0.267615 val half, note 2L→1L + L0 strips, BETTER both modes vs A8,
  cleanest run of the chain). Saliency pruning 5/5 since A6.
  **Stability: cleanest run of the chain — ZERO training NaN activity (A8's watch item
  did NOT recur; shallow note appears to have helped).** Lineage: A8 (0.300380/0.269006
  full-range, card 3L→2L + card.L1 strip, per-card state −1/3, the NaN-transients run) →
  A7 (0.300365/0.268966, user 4L→3L + note.L1/deck.L2 strips, imm p=9.1e-118 the
  strongest of the phase) → A0 (d=128 1-ep retrain, 0.299857/0.269030, n=4993, 7
  nanskips — 1-ep budget tax +0.0037/+0.0044 vs the upstream 12-ep .pth) → A1
  (mixers→1.0) → A4 = A1 + NO_AHEAD_RESIDUAL. The d=128 residual price = ahead
  +0.000495 (p=1.0) but imm 0.000062 BETTER — cheaper + more asymmetric than d=32's.
- **QAT deploy truth (FROZEN until research closes) = champ5k_b1** (0.306629/0.277893 quant-aware;
  `champion_5k.json` + its own cbs). At research close the final champion gets ONE 2-ep
  confirmation run + ONE quant-aware run (q72u deploy env + the frozen NO_JIT family flags;
  plain-era vs QAT-era logloss are NOT comparable).

**Iters 17+19 REJECTED: the pbin lever (binary-recall loss term) is CLOSED by dose-response.**
Scale 0.5 (iter 17): imm +0.000387 / ahead −0.000222; scale 0.25 (iter 19, n=4999): imm
+0.000258 (p=1.6e-70, under the bar) / ahead −0.000101 (p=1.0). The trade is ~linear through
zero → NO scale can make both modes improve ≥0.0003. Real, reproducible effect; pure trade.
⚠ Iter 19 also produced the FIRST-EVER d=32 NaN-skip (user 8902, 2.0M-token mega user, on its
1M–2M-token chunk; finite in all prior track-1 runs) — fp32-probe verdict in
research_5k_verbose.md; watch future track-1 evals for nanskips (gate needs --intersect then).

**Iter 18 REJECTED (directed, 2026-07-15 23:45): duration ablation (ZERO_FEATURES=8,22) =
+0.0018 ahead / +0.0024 imm worse — 6-8x the ≤0.0003 tolerance. Review duration is REAL signal
(historical answer times predict retention; nothing else recovers it); deploy keeps feeding it.
Champion recipe stays RWKV_ZERO_FEATURES=22 only.** The honest persistent val deficit predicted
this one (consistent-all-run val gaps mean something; oscillating ones don't).

**Family scoreboard (track 1, plain+QAT eras; conduct rule 5 — 1-2 rejects = deprioritized, NOT
closed):** early-training-intervention 0/2 (shrink-perturb, warmup-KD — both led early val then
washed out; mid-WS val leads do NOT predict verdicts); grade-representation 0/1; capacity-at-5k
0/2 (head resolution 64→128, mixer 1.5 — the d=32 trunk is not capacity-limited at 5k);
state-size ladder 0/5 CLOSED (no stream is state-capacity-limited at d=32/H=2; iter 6's near-miss
died on the seed pair); readout 0/3 WITH SIGNAL (prehead gate null; iter 20's 64-param cross-head
mix improved BOTH modes at p 2e-10/2e-25 but ~2/3 of the bar; iter 21's KxK 16x-capacity variant
ERASED the gain, ahead −0.0009 — the channel is real but capacity-starved is the WRONG diagnosis;
v3 queued = v1 with the delta EXCLUDED from wd);
loss-reweighting 0/2 (pbin 0.5 + 0.25 = linear imm/ahead trade, the SCALE lever is closed by
interpolation — other reweighting ideas like recency/per-rating weights would be new family
members); HP tuning CLOSED (champion HPs confirmed
vs 19 alternatives at full eval); **optimizer 1/2 — Muon ACCEPTED iter 29 (the strongest
family start of the phase: imm +0.000485 p=6.5e-71); cautious wd REJECTED iter 30 (pure
trade: imm +0.00014 / ahead −0.00038 — the pbin shape again); micro-tuning NOT auto-queued;
NorMuon/Polar-Express = deprioritized in-family variants.**
All hooks stay in-repo env-gated, default off: RWKV_KD_DUMP_OUT/
RWKV_KD_MIX, RWKV_INIT_BLEND, RWKV_GRADE_EMB, RWKV_STREAM_HEADS/RWKV_STREAM_LAYERS,
RWKV_PREHEAD_GATE, RWKV_PBIN_SCALE, RWKV_ZERO_FEATURES, RWKV_ARCH_MODULE, RWKV_EVAL_CAST_FP32,
RWKV_MUON (now ON in the champion recipe).

**Live rules (5k phase, both tracks):**
- **⚠ VAL/TEST SPLIT (Andrew 2026-07-21, effective from iter 29 / post-A8): candidates eval
  ONLY the VAL half = users 5001–7500 (n=2500); all verdicts + p-gates run there, pairing vs
  the champion's existing jsonls via `paired_pvalue --intersect`. TEST = 7501–10000 is touched
  ONLY at each track's close (final champion + the 2-ep confirmation + QAT runs) for honest
  numbers — NEVER for decisions.** Delta bars/p-thresholds unchanged (expect ~1.4× noisier SEs
  at n=2500). Training-val 5001–5010 + tuner 5001–6000 already ⊂ val (vprune refs stay valid).
  Eval tomls: `write_eval_toml ... 5001 7500`. Bonus: eval wall-clock halves. Full text:
  research_5k_notes.md methodology amendment.
- **RWKV_NO_AHEAD_RESIDUAL=1 in EVERY future run, both tracks (Andrew 2026-07-16: the
  piecewise-linear curve correction is DISABLED)** — track-1 iters and track-2 A3+ alike;
  A2 grandfathered (mid-flight). Iter 22 measures the cost; re-baseline is Andrew's call.
- **Track-2 (d=128) runs: RWKV_EMPTY_CACHE_EVERY=1 + RWKV_EMPTY_CACHE_WINDOW=0** (whole-run
  per-step clears — allocator-envelope creep → WDDM paging → 4x slowdown otherwise; ~free under
  the ~1 s step). **MAX=32768 EVERYWHERE incl. `write_decay_setup.py` arg 10** (its 110000
  default THRASHED A0's decay; pairing needs MAX identical across all track-2 runs). d=128 evals
  UNSHARDED (`--shards 1 --solo-threshold 0`; one alone ~9 GB). Coverage fact: max single batch
  in train_db_5k_h1 = 16,384 tokens → zero data drop at any MAX ≥ 16,384.
- d=32 evals: phased `eval_sharded.py` (solo mega-users → 2 LPT shards → merge; ~1.9x over
  sequential, wedge-safe; completeness gate = merged+skipped == rostered or exit 3).
  Elevated-VRAM rungs (e.g. K=32 streams) → sequential shards.
- **MID-EPOCH RESUME NOW SUPPORTED (2026-07-23, Andrew's directive after the reboot ate
  A14's 18k steps): RWKV_RESUME_SKIP_GROUPS=1** + the existing ckpt machinery = crash
  recovery losing ≤1000 steps (~15 min). train_rwkv skips the already-trained prefix of
  the deterministically-shuffled group sequence (epoch e skips min(max(done−e·E,0),E);
  shuffles still consumed → order replays exactly). Procedure: `python
  scratchpad/make_resume.py <run_dir> <prefix> <ws_toml>` (finds newest ckpt pair, copies
  optim to the loader name, writes the resume toml), then rerun the WS phase with the
  run's FULL env + RWKV_RESUME_SKIP_GROUPS=1, WITHOUT deleting step-trace files;
  decay/eval phases unchanged. VALIDATED: smoke B1 (mid-epoch-0, 587 keys) + B2
  (whole-epoch-0+partial-1 skip, 258 keys) both EXACT vs the uninterrupted reference.
  Caveats: the resumed tail's dropout draws differ (weights/optim exact — statistically
  equivalent, ≪ cross-seed spread; NOT bit-identical to uninterrupted); a resumed WS's
  grad-stats json covers only the tail. Vals are only comparable at the SAME step.
- **⚠ NO co-tenant GPU work during gate-critical runs (2026-07-23, learned twice in one
  evening):** a d=32 smoke sharing the GPU with A14 first caused ~1e-4 val drift (cuBLAS
  algo selection under memory pressure breaks bit-replay), then a second smoke leg pushed
  VRAM to 11.6/12 GB and BOTH processes froze in a WDDM paging deadlock for 2.7 h (log
  mtimes stuck at the same second; killing the smoke instantly unstuck A14, zero steps
  lost). Smokes needing GPU wait for a free GPU or run tiny/CPU.
- **Seed-pair doctrine (research phase):** any single-run margin < ~0.0005 needs the exact recipe
  re-run at RWKV_AUGMENT_SEED=4321 before acting — cross-seed spread on the same recipe is
  ~0.0004 both modes; in-seed Wilcoxon p (even 1e-29) measures per-user consistency, NOT
  cross-seed robustness.
- **TorchScript hook rules (cost 2 hollow/dead launches in iter 16):** @torch.jit.ignore bodies
  must NOT call submodules (through scripted code they see the raw C++ ScriptModule → 'not
  callable' → the NaN-except turns the run HOLLOW) — use root Parameters + F.linear, names
  containing weight/bias for the wd groups; root-level Parameters are INVISIBLE to
  selective_cast's module walk (cast them explicitly); ScriptModule forbids persistent=False
  buffers (use plain tensor attrs). Smoke tests MUST exercise the SCRIPTED forward +
  selective_cast/copy_downcast_ chain, not direct Python calls. Gate every .cmd phase on exit
  codes AND artifacts (train_rwkv can swallow fatal errors to exit 0).
- FETCH WORKERS = 4 in every training/eval toml (Andrew 2026-07-08, RAM). Live loss plot:
  `detach.ps1 -Script scratchpad/liveplot/run_liveplot.cmd` (auto-discovers the newest
  `*_ws_trace.jsonl`, champion ref from champion json).

**★ anki-revlogs-10k-id DATASET DONE (2026-07-16 00:07, 16.2 GB at
`C:/Users/Andrew/anki-revlogs-10k-id`):** the 10k dataset rebuilt from the raw HF release with
**REAL Anki epoch-ms IDs** (card/note/deck/parent/preset — no factorize) **+ corrected
`review_time = revlog id − taken_millis`** (show time, Andrew's directive; raw answer id =
review_time + duration; day_offset/elapsed_*/sort all use the corrected time). User numbering
== published set (file stems). VERIFIED vs published: user 70 row set identical (720,110 rows,
ratings 1:1 aligned by answer time), day_offset differs on exactly 1 row (show-time crossed the
day rollover — the intended effect); 10,000/10,000 revlog+deck tables, 9,934 card tables (==
published exactly). Builder `scratchpad/dataset_id/build_parquet_id.py` (resumable). Staging
`...-10k-id-raw` (archive + extracted protobufs ~40 GB — deletable once the parquets are
trusted). Follow-on work: a NEW preprocessing pipeline deriving FUTURE_FEATURES.md features
from the real timestamps.

**★ TRACK-2 A1 ACCEPTED (2026-07-16 10:57) = NEW TRACK-2 CHAMPION: all channel mixers → 1.0.**
**2,320,516 params (−442,368 vs A0); intersection (n=4993) ahead 0.299768 = +0.000089 BETTER
(p=2e-4), imm 0.269070 = +0.000040 worse (p=1.0) ⇒ per-100k ratios −0.0000201 / +0.0000090 —
~50× inside the ≤0.0001 gate.** Full-5000 finals 0.300009/0.269324 with **ZERO NaN-skips** (A0
needed 7 — the instability is gone; future track-2 gates can pair on full n=5000).
`champion_5k_track2.json` = A1 (ckpt `scratchpad/track2_a1/t2a1d_5586.pth`, 24 val points = the
track-2 vprune ref). d=32's mixer lesson transfers to d=128; decay-end val was IDENTICAL to A0.
Detail: research_5k_verbose.md. **Track-2 A2 queue (next track-2 block):** user 4L→3L / deck
4L→3L (~149k each), LoRA-dim cuts, d_model 128→96. **A2+ runs must set
RWKV_GRAD_STATS=<out.json> (Andrew's directive 2026-07-16):** records per-param mean|grad| +
mean|grad·w| (SNIP saliency) across all steps + final near-0/near-1 no-op weight stats, to
rank ablation targets; recorder `rwkv/grad_stats.py` (unit-tested), report
`python optimization/grad_stats_report.py <json>` (layer ranking + type-aware no-op suspects).

**Iter 20 REJECTED (2026-07-16 17:55) but = the plain era's strongest positive signal:
cross-head readout mix v1 (RWKV_XHEAD_MIX=1, zero-init (H,H,K) delta on the WKV output
pre-GroupNorm, 194,620 params) improved BOTH modes — ahead +0.000178 (p=2.0e-10), imm
+0.000107 (p=2.0e-25), n=5000, 0 nanskips — first p-gate PASS since iter 15, but both
magnitudes miss the 0.0003 bar.** Smoke lesson: W_o is zero-init → nothing upstream of it is
observable at fresh init (randomize W_o before perturb/grad smoke checks).

**Iter 21 REJECTED (2026-07-16 21:12): cross-head mix v2 (full K×K, 208,060 params) —
ahead −0.000859 worse (p=1.0), imm tied. The 16× capacity erased v1's both-modes gain;
the readout channel is information-poor + regularization-hungry, not capacity-limited.**

**TRACK-2 A2 REJECTED (2026-07-17 07:25): deck 4L→3L = ahead +0.000180 worse (p=1.0) =
per-100k ratio +0.000155 = 1.55× the ≤0.0001 bar (imm +0.000020 = +0.0000172, passes).**
Full n=5000, 0 nanskips (2nd consecutive clean d=128 run). Deck DEPTH is load-bearing for
the ahead/curve pathway; d128-single-layer-cut family 0/1, deprioritized for BUNDLES (the
cut was exactly 5.0% and still failed the price check). ⚠ A2's grad-stats jsons are DEAD
(whole-step-skip bug: layer-0 v_lora_simple.A never receives grads → every step skipped;
FIXED `dcf11f5` — per-param subset accumulation, report refuses dead jsons + lists
never-grad tensors as FREE prune candidates (5×1,024 params at d=128); A3 records
correctly on the same A1 trunk). Detail: research_5k_verbose.md.
**ITER 22 ACCEPTED (Andrew 2026-07-17 ~10:50, directed re-baseline): no-residual cost
ahead +0.000834 / imm +0.000312 vs iter 15 = the price of monotone-in-t. NEW track-1
champion/reference = 0.304497/0.273539; champion_5k_plain.json re-pointed (promote
--val-trace done).**
**A3 (GRU curve head) COMPLETE 2026-07-17 21:12 — REJECTED on the drafted vs-A1 gate,
VERDICT DEFERRED to the no-residual re-anchor.** n=4871 intersection: **imm 0.268403 =
+0.000105 BETTER (p=1.6e-21, FIRST significant track-2 accuracy win)**; ahead 0.299964 =
+0.000443 worse → ratio +0.000228 (2.28× bar) — but CONFOUNDED (A1 is residual-ON; iter 22
priced residual removal alone at +0.000834 ahead at d=32; A3's deficit is ~half that).
**⚠ 129/5000 eval NaN-skips** (instability oscillates through training; deploy-side
state-norm clamp now load-bearing for d=128). Grad-stats (fixed recorder): 10,886
never-grad params (layer-0 v_lora ×5 = free strip); saliency bottom = ALL non-L0 channel
mixers + user.L3.time_mixer = A4 bundle shortlist. Detail research_5k_verbose.md.
**ITER 23 ACCEPTED (VERDICT CHANGED by Andrew 2026-07-18 ~12:55; auto-verdict 01:15 had
been reject-on-magnitude): learnable power-mean PAVA rectifier = NEW TRACK-1 CHAMPION —
adopted for the monotonicity constraint itself (ordered button intervals = product UX),
with accuracy ~free-to-mildly-positive: BOTH modes improved (+0.000278 p=1.3e-33 /
+0.000116 p=8.1e-15 vs iter 22), n=5000, 0 nanskips, 193,727 params. Curve-shape-
constraints family 1/1. Detail research_5k_verbose.md (incl. the changed-verdict
addendum).**
**TRACK-2 A4 RE-ANCHOR DONE + PROMOTED (2026-07-18 12:02): 0.300504/0.269262, n=5000,
0 nanskips, ZERO NaN val windows (the GRU head, not d=128/no-residual, was A3's
destabilizer). A3 DEFERRED VERDICT = ratio gate PASS both modes (−0.0000288/−0.0000221
vs ≤0.0001; A3 BETTER than the fair anchor: ahead +0.000056 p=0.107, imm +0.000043
p=7.6e-05) — but promotion stays BLOCKED by A3's 129-NaN instability (recorded
gate-PASS-unstable); the GRU head (−194,292 params) is VALIDATED as an A5-bundle
component once the state-norm clamp / train-time fix lands. Re-anchor grad-stats:
never-grad 142,592 (dead ahead head 131,712 + 5×L0 v_lora 10,880 = free strip);
saliency bottom = 8 non-L0 channel mixers (~265k = 11.4% of A1) then card.L1/user
time-mixers — consistent with A3's report = robust A5 menu. ⚠ NAMING: "A4 bundle" in
older notes = A5 now (A4 = the re-anchor). Detail research_5k_verbose.md.**
**ITER 24 REJECTED (2026-07-18 15:32): p-head-weighted PAVA pooling = NULL vs iter 23
(ahead +0.000035 p=0.54, imm +0.000002 p=0.03; n=5000, 0 nanskips) — uniform pooling
suffices, iter 23 stays champion, deploy keeps the simpler rectifier. CONFIRMATION
BONUS: vs iter 22 it scored +0.000312 (p=6e-35) / +0.000118 (p=7e-21) — the PAVA gain
reproduced across two independent trainings (~+0.0003 ahead / +0.0001 imm real).
Weighting sub-lever closed. Detail research_5k_verbose.md.**
**ITER 25 ACCEPTED (VERDICT CHANGED by Andrew 2026-07-19 ~10:35; auto-verdict 07:24 had
been reject-on-logloss): GRU power-curve head at d=32 = NEW TRACK-1 CHAMPION on the
SIZE/SPEED exception — parity inside the budget (ahead −0.000207 p=1.0, imm −0.000018
p=0.38 vs iter 23) at 171,066 params (−11.7%); n=5000, 0 nanskips. The d=128 imm win did
NOT transfer (the d=32 trunk is the binding constraint) but both tracks now share the
GRU head. Val-lead lesson strongest instance: led vals nearly all run, lost eval. PAVA
Hard–Good power −1.44 IDENTICAL to iter 23 under a different head. Detail
research_5k_verbose.md (incl. changed-verdict addendum).**
**MEME RUN DONE (2026-07-19 10:53, recorded in optimization/side_experiments.md SE-1):
BLIND RWKV LOSES to FSRS-7 decisively — ahead 0.351922 (+0.034, wins only 7.5% of
users), imm 0.341322 (+0.023, wins 25%); n=5000, 0 nanskips. Intervals+grades are worth
~0.048 ahead LogLoss (~3.5× the full model's margin over FSRS-7). NOT in
research_log.jsonl by design.**
**ITER 26 (GRU N=3) ACCEPTED (VERDICT CHANGED 2026-07-19 ~21:00 — Andrew LOOSENED the
gate to rounded-4dp ≥0.0001 both modes; auto-verdict 20:18 had been reject on the old
0.0003 imm bar): ahead +0.000485 (p=4.4e-42, largest ahead gain of the phase), imm
+0.000088→0.0001 (p=4.8e-09); n=5000, 0 nanskips, 171,453 params = NEW TRACK-1
CHAMPION (recipe now GRU_HEAD=3). Under the new bar iter 20 (xhead v1,
+0.000178/+0.000107, both p≪1e-9) would also have passed → xhead-mix v3 gains queue
priority. PAVA middle junction −1.59 (3rd straight strongly-negative). Detail
research_5k_verbose.md.**
**ITER 27 REJECTED (2026-07-20 00:01): GRU N=4 = ahead −0.000411 / imm −0.000172 worse
than N=3 (p=1.0 both); n=5000, 0 nanskips. THE N-SWEEP PEAKS AT 3 — closed, no N=5;
iter 26 stands. Val-parity lost eval again. Detail research_5k_verbose.md.**
**ITER 28 REJECTED (2026-07-20 14:38): xhead v1 on the iter-26 recipe = ahead −0.000114
/ imm −0.000160 worse (p=1.0 both); n=5000, 0 nanskips. Iter 20's old-recipe gain did
NOT transfer — the readout channel measures NEGATIVE under the GRU head. V3 (wd
exclusion) DEPRIORITIZED with inverted rationale; readout/xhead family 0/3 on current
lineages, closed pending new ideas. Transfer-failure ledger: never graft, re-measure.**
**→ GPU plan (updated 2026-07-22 19:50): A10 + A11 DONE/REJECTED — the de-bundle
SPLIT the damage: user depth FLOORS AT 3L (ahead damage identical ±note strip,
+0.00029, owns the ahead cost — long-recurrence depth serves ahead, cf. A2) and
note.L0's mixer was the imm poison (~+0.00018; last-transform strips are costly).
**A14 DONE/ACCEPTED (2026-07-24 03:30) = NEW TRACK-2 CHAMPION: LoRA dims halved
(decay/a/gate 16→8, v0 8→4, all streams — the first structural cut), 1,380,660
params (−6.0% vs A13, **−50.03% vs 2.76M — halfway mark crossed**), BETTER both
modes (+0.000039 p=0.045 / +0.000059 p=0.0069) — the LoRA ranks were oversized;
a further 8→4 halving is a queue candidate. champion_5k_track2.json = A14 (ckpt
scratchpad/track2_a14/t2a14d_5586.pth; full env = A13's + the lora8 arch module).
**SE-2 BASELINES DONE + CLOSED 2026-07-25 (Andrew's "is RWKV needed?" experiment;
informational, never champion candidates): RWKV-7 WINS by ~0.002 ahead / ~0.003 imm at
matched params — real (4-9× the acceptance bar) but ~1/10 of the ~0.019 margin over
FSRS-7; the other ~0.017 is the shared features/heads/pipeline, which the classic cells
inherit.** Val half n=2500, 0 nanskips: GRU v3 0.300778/0.270525 (1,559,824 p) and LSTM
v3 0.301103/0.270973 (1,488,688 p) vs A13 0.298837/0.267805 (1,468,724 p) — two
independent cell families reproducing the same deficit ⇒ property of the recurrence
CLASS. A14 beats both while 7-11% smaller ⇒ no plan changes. Speed: LSTM 1.74 steps/s >
RWKV 1.24 > GRU 1.18 (RWKV's win is accuracy-per-param, not step rate). ⚠ Two bug
generations are recorded in side_experiments.md SE-2 and are the lasting lessons: v1 had
no query probe (interval-blind, 0.415/0.415); v2 probed correctly but had NO residuals,
so the signal attenuated 3-10×/layer and imm stayed blind — caught only because Andrew
asked "are you sure there are no other bugs?", proven by
`scratchpad/baseline_gru/probe_sensitivity_check.py` (now a reusable post-run gate: zero
the query rows and the imm predictions MUST move). Eval-side fixes banked from the same
work: `RWKV_EVAL_EMPTY_CACHE_EVERY` (default 20 = unchanged) and
`RWKV_RNN_PROBE_CHUNK`/lean no-grad path in rnn_baseline.py (a 2,087,967-row eval batch
made one cuDNN call ask for 20.93 GiB).
**★ THE WIDTH LADDER IS CLOSED (2026-07-26) AND ANDREW TOOK OPTION (b): A18 (d=80 via 5
heads × K=16 + LoRA 4, 557,246 params, 4.95×) IS THE TRACK-2 CHAMPION** — accepted by his
directed verdict change over an auto-reject at 108%/111% of the ratio bar, because the ≥5×
product goal outranks a marginal RATE the run missed by ~10% while costing only
+0.000960/+0.000532 cumulative vs A0. A16 (d=64, 7.11×) REJECTED at ~1.8× the bar; A17
(d=80, 4.72×) missed by 26 millionths (112%/83%). **Two independent draws at d=80 both
~110% ⇒ genuine floor — 4.95× is the end of the width road.** Side-finding: the second LoRA
halving is NOT free at d=80 (+0.00002/+0.00009 for −27.5k) whereas A14's first halving
IMPROVED both modes at d=128 — the lever flips sign as the trunk narrows, i.e. the model is
now truly capacity-limited. A15 kept as the gate-clean fallback.
**★ THE TWO TRACKS HAVE MERGED (Andrew 2026-07-26: "continue track 1 with it" + "it
shouldn't be called A19, it should be iter 31 (first table in research 5k)"). From here
there is ONE lineage: the A18 trunk, numbered as track-1 iterations in research_5k.md's
FIRST table.** The track-2 A-series is closed at A18. ⚠ The old track-1 `params ≤ 225,000`
cap does NOT carry over — it belonged to the d=32 track; this lineage's size story is the
4.95× reduction (558,212 params). Flagged to Andrew rather than silently dropped.
**→ ITER 31 (RUNNING, relaunched 2026-07-26 12:42 on the corrected duration convention,
pid 3136, `scratchpad/iter31_algo/`) = A18 + PAVA + GRU N=3 + Muon**, the three track-1 wins
track 2 never received, bundled (all independently validated; together = the iter-29 recipe;
1 run ≈ 10 h vs ~30 h for three).
Ordinary accuracy gate vs A18 (0.299302/0.268390 val half). **SPEED: measured ~1.1–1.4
steps/s** (4,657 steps in the first 68 min) vs A18's 1.86 — probes add ~30% rows, PAVA runs
eager, Muon adds Newton-Schulz; WS ~4.5 h, verdict ~20:45. 8.8 GB peak reserved. (The 40-step
`BENCH_RESULT` said 0.29 — it understates steady state ~3×, NEVER schedule off it.)
**⚠ ITS EVAL LEG IS UNRECTIFIED** — the `.cmd` was written before `RWKV_EVAL_PAVA` existed
and a RUNNING batch file must not be edited (cmd.exe re-reads it at a saved byte offset).
That is fine and arguably better: the unrectified number is directly comparable to A18's
existing jsonls, so it is the primary gate; the rectified pair (A18 falls back to classic
p=1, having no `pava_theta`) then follows as two separate ~1.25 h evals for the DEPLOY
metric. If the two metrics disagree on the verdict, report both to Andrew rather than
picking one.
⚠ Treat as a hypothesis, not a deposit — they were tuned at d=32 and the
transfer ledger (iter 28, A13's opposite-sign state price) says d=32 wins need re-earning;
the encouraging prior is that A18's own LoRA finding shows the trunk is capacity-limited,
and PAVA/GRU-N add head-side capacity.
**★ DEPLOY-PATH WORK DONE 2026-07-26 (the "implement everywhere" directive + its fallout),
commits `11ab7e0` / `921ac76` / `db85154`:** (1) the Python RNN got the 4-BUTTON API it never
had — `SrsRWKVRnn.button_heads` / `button_curves` / `button_intervals`, plus `pava_theta` in
the state dict (without it `load_state_dict`, which is strict, could not even OPEN a
PAVA-trained checkpoint); probes read the state and never advance it, duration is zeroed on
all four inside the API so a caller cannot get the contract wrong, and the interval solver
bisects on the RECTIFIED curve (the rectifier couples the buttons, so rectifying after
solving gives a different answer). Smoke `scratchpad/eval_pava/smoke_rnn_buttons.py`.
(2) **Gaps 2/3/5 turned out to be PYTHON-RNN gaps, not just Rust ones** —
`RWKV_STRIP_CMIX` / `RWKV_STRIP_L0_VLORA` / `RWKV_STATE_CLAMP_*` lived only in
`rwkv_model.py`, so the deploy twin could not run the merged champion; now ported, with
`stream_name` stamped in `srs_model_rnn.py` (unstamped, STRIP_CMIX matches `":<layer>"` and
strips NOTHING while appearing to comply). ⚠ The state clamp is per-step at deploy vs
per-window in training — equal wherever ‖S‖ ≤ τ (factor exactly 1.0, bit-inert), different
only on already-diverging states; that belongs in the parity gate's tolerance story, not
chased as a bug. (3) New harness `scratchpad/parity3/parity_train_vs_rnn.py`, 7/7 at ~1e-6
— see §9.
**RUST PORT still the highest-value NON-research work** (`rust/rwkv-infer/TRACK2_PORT_PLAN.md`,
committed 1a86f04): `optimization/CPU_INFERENCE.md` shows param count has already decoupled
from the metric users feel (4.5× fewer MACs bought 1.24× wall-clock, plateauing after A14)
and the engine cannot run the track-2 arch at all — but `model.rs` IS already dim-agnostic,
so the 5 gaps are shape-detectable (1×1 dummies): GRU head, stripped cmix, stripped L0
v_lora, no ahead residual, state clamp. ⚠ Iter 31 changes the parity target (GRU N=2→3), so
port against A18/iter31 whichever is champion when the port lands.
Reference implementation + an AVX2/FMA patch to crib from: `vendor/jschoreels_anki/`.
Earlier: A13 promoted (state-feature re-anchor, price +0.00021/+0.00019, opposite
sign vs d=32); A12 REJECTED (preset 3L→2L, imm ratio 1.23× bar) — ALL DEPTH FLOORS
MAPPED: card=2, deck=4, note=1, preset=3, user=3; depth ladder EXHAUSTED. Also
2026-07-23 night: MID-EPOCH RESUME landed (RWKV_RESUME_SKIP_GROUPS=1 +
scratchpad/make_resume.py, smoke-validated EXACT; crashes now lose ≤1000 steps) +
the NO-CO-TENANT-GPU hard rule (see Live rules).**
⚠ EVAL-PATH FETCH-WORKER LEAK IS SYSTEMATIC (A11's run left none — intermittent): every
eval/rerun leaves 1–2 orphan pythons, some spinning a FULL CORE (iter-29's for 14 h,
the A9-rerun's for 8.5 h) — the trainer kills its workers ("Killed processes.") but
the eval path doesn't; CHECK + KILL ORPHAN PYTHONS after every run (spare pythonw =
bridge/controller, ~80000s-CPU = FSRS, and Andrew's liveplot); fix candidate: worker
cleanup in eval_sharded/get_result. Track-1 queue: permutation init (LOW),
fresh-family planning (LIT_REVIEW + FUTURE_FEATURES). 2026-07-21: A8 + iter 29 (Muon)
ACCEPTED, iter 30 (cautious wd) REJECTED; A8's first launch died in the ~02:35
black-screen hang (zero telemetry precursor, driver 610.62; crash combo REMAPPED to
RIGHT Ctrl + SPACE ×2, registry armed + rebooted); A9's first eval WEDGED on user
5747 (transient fetch race; eval_sharded RESUME recovered it). **ITER 28 QUEUED (Andrew 2026-07-19 ~20:50: re-benchmark iter 20 on the new recipe):
xhead-mix v1 EXACT (RWKV_XHEAD_MIX=1, +896 params) on the iter-26 champion recipe —
the old +0.000178/+0.000107 (p 2e-10/2e-25, would pass the NEW gate) was measured vs
the stale iter-15 recipe and must be re-earned (transfer failures are precedented).
Parked pid 21048 on A6's DONE_EXIT (~12:00 tomorrow → verdict ~15:30); tail prints
paired vs BOTH iter 26 and iter 27. If it passes → v3 (wd exclusion) as a follow-up
lever; if it fails → v3 is the in-family retry.** Track-1 queue after: permutation
init (LOW).
⚠ ERRATUM (2026-07-19): module index 1 = the DECK stream (arch order card,deck,note,
preset,user — NOT the RWKV_SUBMODULES order); the A3/A5 "note.L2 diverges" narrative
should read **deck.L2** (CLAMP_NOTES.md corrected; grad reports were always right).
New env for the strip: RWKV_STRIP_CMIX (rwkv_model.py, name:layer list, dummy-mixer
pattern, default off = byte-identical; RWKV7Config gains stream_name, stamped in
SrsRWKV.__init__).
⚠ OPS (cost 2 launches 03:22): PowerShell Set-Content -Encoding utf8 writes a BOM →
tomli dies line 1 col 1 — write tomls via the Write tool or UTF8Encoding($false); and a
crashed run's DONE_EXIT_WSFAIL satisfies downstream waitloop greps → relaunch upstream
first (its cmd truncates its own log), THEN re-park dependents.**
**MEME RUN "BLIND RWKV" QUEUED (Andrew 2026-07-19 ~02:30, recorded SEPARATELY — new
`optimization/side_experiments.md` at verdict, NOT research_log.jsonl): train d=32
WITHOUT interval features and WITHOUT grades (RWKV_ZERO_FEATURES=0-7,9-12,22; duration
kept) — can blind RWKV still beat FSRS-7? TARGET = FSRS-7-sched_penalties-short-secs-
recency on users 5001-10000: by-user mean LogLoss 0.317933 (vs AHEAD mode; our champion
0.304220 → 0.0137 of margin). Parked pid 4460 on iter 25's DONE_EXIT (scratchpad/
meme_blind/, ~3.5h). Recipe deviations (forced): vprune OFF (champion val ref would
false-kill), PAVA OFF (grade probes meaningless), clamp ON (full-n insurance), standard
64-basis head. Cmd tail prints paired-vs-iter23 (the cost of blindness). Interpretation
caveat: day-resolution intervals remain PARTIALLY reconstructible from the cycle
features (rows 22-28 share a per-batch phase → day gaps recoverable) + rows 12/13
(activity since card's last review) — grades are truly gone (duration correlates only).
Andrew's queue order: meme BEFORE further experiments → if iter 25 passes, iter 26
(GRU N=3) parks on the MEME's DONE_EXIT, not iter 25's.**
**ITER 25 QUEUED (Andrew 2026-07-18 ~23:30: "Let's try power curves first, to see if they
improve log loss of the small model"): GRU-faithful power-curve head at d=32
(RWKV_GRU_HEAD=2 + RWKV_STRIP_L0_VLORA=1 + state clamp τ=300 as insurance; full iter-23
champion recipe incl. PAVA; 171,066 params = −11.7%; MIN_STEP=6000). Parked pid 36720,
waitloop on A5's DONE_EXIT (~03:00) → verdict ~06:30. Gate: ≥0.0003 both modes vs iter 23
+ p<0.0001. **If iter 25 PASSES: iter 26 = RWKV_GRU_HEAD=3 (Andrew 2026-07-18 ~23:55 —
"If iter 25 succeeds, try 3"); sweep upward while it keeps winning (ordered-S
cumsum-softplus anti-collapse insurance available if higher N label-switches).** If it
misses: variant A (fixed log-spaced S-grid, weights-only, N≈8–16) is the family sibling. By-construction button-ordering ideas (FOSD/CDF-power head,
shared-shape ordered-S) discussed with Andrew 2026-07-18 — candidate follow-ups in the
curve-shape-constraints family after the power-curve verdicts.**
**Iter 22 REDEFINED (Andrew 2026-07-16 ~23:00) = DISABLE THE PIECEWISE-LINEAR CURVE
CORRECTION, queued behind A2 (detached pid 20584, waitloop on A2's DONE_EXIT → self-starts
~08:30, verdict ~11:45; run dir `scratchpad/iter22_nores`).** Andrew's directive: "check if
RWKV-Curve is using a linear piecewise correction, and if so — disable it for both tracks."
Confirmed: `curve_logits = logit(mixture) + interp(out_ahead_logits, t)` — a learned
64/128-point residual linearly interpolated between log-spaced time points. New flag
**RWKV_NO_AHEAD_RESIDUAL=1** (srs_model + srs_model_rnn) zeroes the residual outside
autograd → curve = pure mixture-of-exponentials, monotone in t BY CONSTRUCTION (supersedes
the cummin variant, which never trained; the raw-mixture BCE term AHEAD_RAW_SCALE=0.5
already supervises the mixture directly). NaN probe moved to out_p_logits under the flag
(zeros can't NaN — eval nanskip + train guard key off that probe). Params unchanged 193,724
(~12.5k now dead at d=32; ~131.7k dead at d=128 — strippable at deploy/in a track-2 bundle).
Smoke ALL_PASS (zero-residual, grad isolation, off-path byte-identity, JIT + NO_JIT).
**MANDATORY RECIPE both tracks from now on: RWKV_NO_AHEAD_RESIDUAL=1 in every future run
(track-1 iters AND track-2 A3+); A2 grandfathered (mid-flight, residual-on — its gate vs A1
is within-family valid).** **Iter 22 gate = ANDREW DECIDES: report both modes' finals,
deltas vs iter 15, p-values, and nan_users to him and WAIT — no auto-accept/reject, no
promotion. Likely outcome: iter 22 becomes the new track-1 REFERENCE (directed re-baseline
à la iter 14/15) since with-residual champions aren't fair gates for no-residual candidates;
track 2 similarly needs a no-residual re-anchor decision at the A2 verdict.**
**Track-1 queue (Andrew 2026-07-16 late, FIXED ORDER — iter 23 DONE/rejected-near-miss):
iter 24 = learnable PAVA + pooling weights from the p-head's button-press probabilities
(Instant mode, RWKV_PAVA_PWEIGHT=1; λ/density unchanged — validated by iter 23).** Then:
xhead-mix v3 (v1 delta excluded from wd), permutation init (LOW). **Duration imputation for the counterfactual probes (Andrew
delegated): ONE shared value across all 4 buttons (causally correct — duration is spent
before the press, independent of which button), = a GLOBAL CONSTANT (train-set median)
frozen into the deploy contract; only duration is imputed (elapsed/etc. are real at both
train and deploy); upgrade path if the audit shows sensitivity = per-user EMA carried
next to the state. Build-time checklist: enumerate ALL outcome-dependent dims of the 92
(INPUT_FEATURES.md) — rating one-hot + duration + any derived — and swap/impute them
consistently in the probe rows.**
**Track-2 sizing recommendation (Andrew 2026-07-16, soft rule): aim for ≥5% param reduction
per iteration, ideally more** — single ~116k layer cuts are borderline (A2 = exactly 5.0%);
future candidates should BUNDLE cuts (e.g. deck+user layers together, LoRA-dim cuts folded
into a bigger ablation) or go structural (d_model 128→96 ≈ 40%+). Track-2 queue after A3:
grad-stats-ranked BUNDLES (single ~116k layer cuts are now proven under-priced — A2's deck
cut failed at exactly 5.0%): user-layer + LoRA-dim bundles, d_model 128→96, head_w squeeze
(~83k, once the GRU head proves N=2 suffices) — re-ranked per A3's (fixed-recorder) report;
now confirmed by A4's report (same bottom tier) — this bundle = **A5** (A4 = the re-anchor).
**+ POWER-CURVE BASIS (Andrew 2026-07-16 late, for A3 bundling): replace the 128 exponential
bases with a handful (N≈8–16) of FSRS-7-style power curves** `R_i(t) = (1 + f_i·t/S_i)^(−c_i)`,
`f_i = 0.9^(−1/c_i) − 1` (pins R_i(S_i)=0.9; form = srs-benchmark `models/fsrs_v7.py`
forgetting_curve), S_i = fixed log-spaced grid, c_i = N learnable decays sigmoid-clamped to
[0.01, 0.95] (init ~0.5). Why few can replace 128: a power curve IS an infinite Gamma-mixture
of exponentials — one basis covers the heavy-tail region that needed dozens of exponentials.
Monotone in t by construction (keeps the no-residual guarantee). **Params at d=128:
w_linear 512→N cuts 65,664 → ~4.1k (−61.5k); + stripping the DEAD ahead head (−131.7k,
zero-risk, residual already disabled) ≈ −193k ≈ 8.3% of A1 before any head_w shrink**
(head_w 82.8k is a further optional squeeze once N is tiny). d=32 port later if it works
(w_linear 64→8 saves ~7.2k ≈ 3.7%). Note for the future hard-ordering option: per-basis c_i
breaks total pointwise order of the basis (curves with different decays cross); a single
SHARED learnable c + S-grid keeps the basis totally ordered (FOSD trick compatible) —
measure both if cheap. **VARIANT B = GRU-FAITHFUL (Andrew 2026-07-17, srs-benchmark models/gru.py — his call,
A3 ANCHOR): predict w, S, AND decay per curve.** ⚠ NAMING (Andrew 2026-07-17): the
benchmark model is called **GRU** — the old GRU-P entry was REMOVED from srs-benchmark
(training-data remnant; never write "GRU-P"). Our env flag = RWKV_GRU_HEAD=N, params
gru_*. GRU uses n_curves=2 and THREE tiny
linears off the trunk feature — w_fc (N logits→softmax), s_fc (exp(clamp(·,−25,25))
stabilities), d_fc (same-form decays) — into R(t) = Σ wᵢ·(1 + t/(1e−7+Sᵢ))^(−dᵢ). Plain
form, no R(S)=0.9 factor pinning. exp ⇒ dᵢ>0 ⇒ EACH curve monotone in t even with
per-curve decays (time-axis monotonicity does NOT need a shared decay — shared d is only
for the future FOSD hard rating-ordering, where the basis must be totally
pointwise-ordered; keep as later variant). Plan: N=2 faithful first (proven on the
leaderboard, label-switching moot at N=2); if it holds, sweep N with ordered-S
(cumsum-softplus) as anti-collapse insurance. Init: zero-init the three head WEIGHTS
(input-independent start, like the current zero-init w_linear) + set BIASES to a sane
prior curve (spread log-S, moderate d). Reuse the head_w trunk; replaces w_linear
(65,664 → ~3.1k at N=2, d=128). NB the current head does NOT predict S at all — fixed
log-spaced S grid (0.1 s→~e^22 s), model predicts only the 128 softmax weights (a
distribution over grid stabilities); grid-power-basis (variant A) = fallback.** ⚠ TorchScript trap (cost smoke_mono v1): old-style ScriptModule bakes
the FIRST construction's env-flag into the compiled class — never two flag values in one
process; ahead_linear is zero-init (like W_o) — randomize before head perturb/grad smokes.

**Queued:** entropy-floor analysis (irreducible-LogLoss estimate from the two disjoint d=128
.pths on users 1-100; design in research_5k_notes.md; ~30 min GPU); future-input-features plan =
`optimization/FUTURE_FEATURES.md` (real-timestamp features; needs a new dataset export — Andrew
2026-07-15); **scheduling-monotonicity plan = `optimization/MONOTONICITY_PLAN.md`** (Andrew
2026-07-16: button intervals can invert, e.g. Again > Hard — constraint must live IN the model;
time-axis stage RESOLVED BY REMOVAL — the piecewise residual is disabled per Andrew's directive,
curve now monotone in t by construction; remaining: audit → counterfactual button-consistency
loss at segment-end states via the shelved stateful kernel → isotonic projection as part of the
model at deploy; = the "curve-shape constraints" track-1 family); `optimization/LIT_REVIEW.md` queue;
deploy-side state-norm clamp (NaN guard, MONOTONICITY_PLAN-adjacent ship-time work).



## 5k-era LIVE STATE archive (moved out of CLAUDE.md 2026-07-30)

The "3-job GPU chain" block, verbatim. All five jobs COMPLETED; every finding in it is
also recorded in `research_5k_verbose.md` (rectified evals + the PAVA/rectification
result, the mode-2 duration decomposition, the mode-3 noise control, iter 32, iter 33)
and in `research_log.jsonl` / `research_5k.md` for iters 32-33. Archived because
CLAUDE.md's CURRENT STATE section is loaded every turn and its own header says KEEP
THIS SECTION SHORT. The two REUSABLE ops lessons that lived at the end of the block
(the `findstr /B` waitloop anchor and detach.ps1's absolute-path requirement) were NOT
archived -- they moved into CLAUDE.md's LIVE RULES, where they belong.

#### LIVE — a 3-job GPU chain, each parked on the previous one's `DONE_EXIT_`
1. **★ DONE 00:28 (2026-07-27): rectified evals — BOTH METRICS AGREE, and the deploy metric is
   5x MORE favourable to iter 31 than the gate it was accepted on.** n=2500, VAL half:

   | metric | A18 | iter 31 | delta | p |
   |---|---|---|---|---|
   | ahead unrect (PRIMARY gate) | 0.299302 | 0.298909 | +0.000393 | 6.0e-26 |
   | imm unrect (PRIMARY gate) | 0.268390 | 0.267637 | +0.000753 | 1.5e-209 |
   | **ahead RECT (deploy)** | 0.302890 | **0.300802** | **+0.002088** | 2.4e-160 |
   | **imm RECT (deploy)** | 0.268670 | **0.267691** | **+0.000979** | 6.0e-294 |

   **★ TRAINING UNDER PAVA HALVES THE DEPLOY-TIME RECTIFICATION COST — the finding this pair was
   run for.** Post-hoc rectification costs A18 (never trained under the constraint)
   **+0.003588** on ahead; it costs iter 31 (trained at `RWKV_PAVA_LAMBDA=0.1`) only **+0.001893**.
   So iter 31's real deploy gain is 5.3x what the unrectified gate reported, and the ahead
   rectification penalty is a *training* problem, not an inherent cost of the rectifier.
   ⚠ **Residual, stated honestly:** even the PAVA-trained model still pays +0.001893 ahead to be
   rectified (0.298909 -> 0.300802), and the rectifier SHIPS, so **0.300802 is what a user gets**.
   Driving that residual toward zero is a live research target (higher lambda? probe density?).
   The +0.001893 is ~19x iter 31's own probe-insertion noise floor (its imm rect-vs-unrect delta is
   only +0.000054), so it is a real effect. Note that noise floor is 5x SMALLER than A18's
   +0.000280 on the identical probe machinery — a per-model sensitivity difference, worth a look
   if the state clamp or PAVA training turns out to be what damps it.
   **WHY TWO METRICS:** iter 31's own eval leg is UNRECTIFIED (its `.cmd` predates
   `RWKV_EVAL_PAVA`, and a RUNNING `.cmd` must never be edited — cmd.exe re-reads it at a saved
   byte offset). That is fine and is the **PRIMARY gate**, being directly comparable to A18's
   existing jsonls; the rectified pair is the **deploy metric**. They agreed in both modes, so the
   "report both, do not pick" instruction did not have to fire.
2. **★ DONE 00:56 (2026-07-27): mode-2 duration diagnostic — ANDREW'S QUESTION IS ANSWERED, and
   the rectifier turns out to be the SMALL half.** iter 31, users 5001-5500 (n=500), additive to
   exactly 0.00e+00:

   | component | ahead |
   |---|---|
   | **duration zeroing (m2 - m0)** | **+0.001451** |
   | PAVA pooling itself (m1 - m2) | +0.000611 |
   | total rect-vs-unrect (m1 - m0) | +0.002062 |

   imm probe-insertion noise = +0.000056 (identical for modes 1 and 2, p=2.1e-8 — a good
   consistency check, since imm depends on probe INSERTION, not on what is substituted), so the
   duration term is ~+0.00140 net; mode 3 pins the ahead-side noise directly.
   **~70% of the deploy penalty is the model losing the current review's duration; only ~30% is
   the monotonicity pooling.** And the 70% is a **TRAIN/DEPLOY MISMATCH**, exactly the class §9's
   three-way-parity directive exists to catch: training feeds the real `scaled_duration` for the
   scored row, deploy CANNOT (Anki must show intervals *before* the press), so the model learns to
   lean on a feature that vanishes at serving time. **=> the strongest iter-33 candidate is to zero
   the current row's duration during TRAINING** (`RWKV_ZERO_FEATURES` already exists and is in use
   for dim 22) so the two paths compute the same quantity. It should cost a little on the
   unrectified gate — which is scored WITH a feature deploy will not have — while removing most of
   the deploy penalty, so it must be judged on BOTH metrics (see QUEUE 0). A `RWKV_PAVA_LAMBDA`
   sweep can only ever attack the 30%.
3. **★ DONE 01:29: mode-3 noise control — the AHEAD noise is EXACTLY ZERO**
   (+0.000000 +/- 0.000014, p=0.33, worse on 253/500 = a coin flip), so `m2-m3` and `m2-m0` agree
   to six decimals and the duration number above was never confounded. That is the point of the
   control: measured, not assumed. It also refutes the blanket "probe insertion costs ~3x the gate"
   generalization — see the ★★ refinement in the probe-insertion bullet above.
4. **★ DONE 09:49 (2026-07-27): iter 32 = full-run DISTILLATION — ACCEPTED on the primary
   (unrectified) gate, both modes.** VAL half, n=2500, 0 nanskips, `size` identical (0/2500),
   params **558,212** and card/note state **2,880/1,440** all UNCHANGED (KD is train-time only —
   nothing ships to Rust).

   | metric | iter 31 | iter 32 | delta | p |
   |---|---|---|---|---|
   | ahead | 0.298909 | **0.298333** | +0.000577 | 2.28e-66 |
   | imm | 0.267637 | **0.267207** | +0.000430 | 3.12e-143 |

   **★ THE ANSWER IT WAS RUN FOR — a training-budget deficit IS partly transferable as a soft
   target.** Against the d=128 teacher on the same VAL half (0.294612/0.263561), iter 31 trailed by
   -0.004297/-0.004076 and iter 32 trails by -0.003721/-0.003645: **KD closed 13.4% of the ahead gap
   and 10.6% of imm.** First direct evidence here — and it also bounds the lever, since distillation
   alone plainly will not close the remaining ~0.0037.
   **KD is nearly FREE:** WS 1.30-1.42 steps/s vs iter 31's 1.44 (~9%); dump 22,346 files / 6.96 GB
   in 1h36m (vs ~3h projected); whole run 1:30:32 -> 9:49:36. `pava_pool_frac` fell 0.92 -> **0.075**
   over WS, i.e. under KD the curve head goes nearly monotone unaided.
   ⚠ **NOT PROMOTED TO CHAMPION, deliberately — `champion_5k_track2.json` still points at iter 31.**
   Two reasons: (a) iter 32's eval is **UNRECTIFIED** (its `.cmd` predates `RWKV_EVAL_PAVA`) and from
   iter 33 the gate basis is the RECTIFIED metric, so iter 32 needs a rectified eval before it can be
   anyone's baseline; (b) iter 33 was already running against iter 31's rectified jsonls when this
   landed, and promoting mid-flight would invalidate that comparison. The rectified eval is a GPU job;
   the GPU is busy until iter 33 finishes. **Throughput unmeasured for the same reason** — KD changes
   no architecture/params/ops so it is iter 31's by construction, recorded as a DERIVATION not a
   measurement; queue the real run when the GPU frees.
   ⚠ **ASK ANDREW — the imm margin +0.000430 is BELOW the ~0.0005 seed-pair threshold** (cross-seed
   spread on an identical recipe is ~0.0004 both modes), so the imm win is unresolved by this single
   run; ahead at +0.000577 is above it. Precedent is mixed: **iter 31 was accepted with ahead
   +0.000393, also below**, without a seed-pair run. Not resolved unilaterally — the doctrine says
   "needs".
   Historical detail of the run below (kept: the v1 false-failure and the deadlock guard are
   reusable lessons).
   ⚠ **v1 (pid 25348) DIED at its smoke gate on a FALSE FAILURE — the check was wrong, not the
   dump.** `DUMP_CHECK_FAIL 5/5, "p_curve outside (0,1): [.., 1.000000]"`. `p_curve` is stored
   **fp16** (`train_rwkv.py:1090`) and fp16 spacing below 1.0 is 4.88e-4, so every teacher output
   above ~0.99951 becomes exactly 1.0 — 9.97% of values. Harmless: it is consumed as a soft **BCE
   target** (`srs_model.py:772`) mixed with hard labels that are themselves exactly 0/1, so a 1.0
   target gives `-log(p)`, finite; and saturation lands precisely where soft and hard targets
   coincide, so almost no signal is lost. `check_dump.py` now tests `[0,1]` **plus an
   ANTI-COLLAPSE condition** (>=10% strictly interior) — a dump degenerated to hard labels is the
   failure that would actually waste the ten hours, and the old open-interval test would have
   PASSED that. Re-verified `DUMP_CHECK_OK`, 7.6 GB projected.
   ⚠ **v2 writes its OWN log, deliberately — a DEADLOCK guard.** mode 3 is still polling
   `iter32_kd.log` for `DONE_EXIT_`, and the script opens its log with `>`, which TRUNCATES.
   Sharing the file would have erased the token mode 3 waits for while v2 waited on mode 3.
   Original v1 spec (unchanged otherwise): ~10 h (teacher dump ~3 h + WS + decay + eval).
   Teacher = `pretrain/RWKV_trained_on_101_4999.pth` under `scratchpad/architecture_old_d128.py`;
   student = the iter-31 recipe unchanged plus `RWKV_KD_MIX` + **`RWKV_KD_ALPHA=0.5`** (new flag,
   2026-07-26: holds alpha FIXED = the classic form; unset keeps iter 10's linear 1->0 ramp
   byte-identical). Decay runs on hard labels. Gate = ordinary accuracy iter vs iter 31; the
   `.cmd` also reports the candidate against the d=128 teacher, i.e. how much of the 0.004 closed.
   **Three things to know if it misbehaves:** (a) **vprune is deliberately OFF** — the
   decay_ratio_0p1 FALSE-KILL scope rule says prune only at MATCHED regularization, and KD
   replaces the target wholesale while validation still scores HARD labels; (b) the teacher must
   set `RWKV_PROBE_DENSITY=0.08` + `RWKV_PROBE_DUR=0.0` even though it has no PAVA, because probes
   are a DATA-side row-layout change and teacher/student must agree or the per-step shape check
   exit-43s; (c) a smoke dump of 5 steps gates the full one on `check_dump.py` — the student's
   checksum proves ALIGNMENT but nothing else proves the tensors are teacher outputs at all, and a
   wrong arch/flag yields perfectly aligned garbage. It also projects the dump's disk footprint
   before committing.


## CLAUDE.md chronology archive (moved out 2026-08-10)

Andrew: *"your Claude.md is like 150k characters or something, trim it down"*. CLAUDE.md was 162,746 chars / 1,888 lines, i.e. ~40k tokens loaded EVERY turn, and the section whose own heading says "KEEP THIS SECTION SHORT" was 54% of it. Nothing below was deleted from the project record -- it is chronology (completed experiments, superseded champions, finished queues) moved here verbatim, with one-line pointers left in CLAUDE.md. Rules, invariants, live state and operative plans stayed.

### superseded champion blocks (iters 39, 36, 35, 34, 32, 31, A18)

#### PREVIOUS CHAMPION = iter 39 `iter39_kda09` (iter-36 recipe with RWKV_KD_ALPHA 0.5 -> 0.9) -- promoted 2026-08-08 22:15
**RECTIFIED (the gate basis): ahead 0.298180 / imm 0.265875** on the VAL half (n=2500) =
+0.000158 / +0.000153 vs iter 36 at p=2.2e-10 / 7.8e-37 -- a clean full-gate pass, and both
margins still clear the TIGHTENED raw >=0.0001 bar adopted 2026-08-10 (they were the smallest
surviving accept when that bar was checked). size 0/2500, nan_users 0, **558,212 params, card/note state
2,880/1,440 unchanged** (alpha is loss-time only; nothing new ships to Rust). Throughput
1823.8 rev/s. ckpt `scratchpad/iter39_kda9/i39_d_10935.pth`; `champion_5k_track2.json` points
at it (trace extracted from the WS log, the iter-35 convention).
**★ THE CHAMPION RECIPE'S `RWKV_KD_ALPHA` IS NOW 0.9** (0.5 was the value iter 32 -> 36; an env
string copied from an older `.cmd` silently trains at the wrong mix). Full env = iter 36's with
that one value changed: seed 4321, KD WS-only from `C:\rwkv_kd_dump\t128_seedpair_65k`,
PAVA lambda 0.2, tuned HPs, MAX=65536.
**The alpha dose curve is MONOTONE UP** (vs alpha 0.5: ahead +0.000115 -> +0.000158, imm
+0.000048 -> +0.000153 at 0.75 -> 0.9; imm ACCELERATING, p=4.4e-21 paired 0.9-vs-0.75), which
MOOTS iter 38's near-miss question -- 0.9 dominates 0.75. Reading: at a 1-ep WS budget the
12-ep teacher's soft targets beat a 50/50 mix; hard labels still anchor the whole decay phase.
Family distillation 3/3 (+1 dominated near-miss). Detail: `research_5k_verbose.md` iter 39.

#### PREVIOUS CHAMPION = iter 36 `iter36_pava02` (iter-35 recipe with RWKV_PAVA_LAMBDA 0.1 -> 0.2) -- DIRECTED-accepted + promoted 2026-08-07
**RECTIFIED (the gate basis): ahead 0.298338 / imm 0.266027** on the VAL half (n=2500) =
**+0.000478 ahead (p=5.1e-67) / -0.000081 imm** vs iter 35. **The mechanical gate FAILED on imm
and Andrew took the trade anyway** (precedent iters 22/23 -- monotonicity constraints accepted
for the constraint, not the logloss): the exchange is **5.9:1** and the imm cost is exactly one
4-dp tick. size 0/2500, nan_users 0, **558,212 params and card/note state 2,880/1,440 unchanged**
(lambda is a loss weight -- nothing new ships to Rust, but see the contract note). Throughput
1769.4 rev/s. ckpt `scratchpad/iter36_pava/i36b_d_10935.pth`; `champion_5k_track2.json` points
at it (trace EXTRACTED from the WS log, same convention as iter 35).
**★★ THE DEPLOY CONTRACT NOW CARRIES `RWKV_PAVA_LAMBDA=0.2` -- SET IT IN EVERY FUTURE RUN ON
THIS TRUNK.** 0.1 was the value from iter 31 through iter 35, so a `.cmd` that copies an older
env string trains at the wrong pressure and its gate is not comparable to this champion.
**Env = iter 35's exactly, with that one value changed** (seed 4321 + KD from
`C:\rwkv_kd_dump\t128_seedpair_65k` still stand).
**WHY IT IS A TRADE AND NOT A WIN:** the rectifier's deploy penalty shrinks with training
pressure (~+0.0019 at lambda 0.1 -> +0.001544 at 0.2 -> +0.001334 at 0.3) while raw
(unrectified) ahead does NOT pay -- the ahead gain is almost entirely recovered deploy cost.
The imm cost is genuine and training-side (its rect-unrect gap matches known probe noise), and
the two responses are monotone in OPPOSITE directions, so **no lambda > 0.1 can pass the
both-modes gate**; the ahead gain is concave (81% captured at 0.2) and the imm cost ~linear.
lambda=0.3 was measured and rejected as the worse point (marginal step trades ~1.1:1). Full
dose-response table: `research_5k_verbose.md` iter 36.

#### PREVIOUS CHAMPION = iter 35 `sp4321_kd` (the tuned recipe + KD restored, seed 4321) -- promoted 2026-08-06 21:20
**RECTIFIED (the gate basis): ahead 0.298816 / imm 0.265946** on the VAL half (n=2500) =
+0.000153 / +0.000271 vs iter 34 at p=5.9e-11 / 7.9e-71 -- and it ALSO beats its same-seed twin
(arm A, tuned-no-KD at 4321) by +0.000160 / +0.000251 at p=4.3e-13 / 3.3e-75, so the accept is
not cross-seed luck. size 0/2500 (vs champion AND twin), nan_users 0. **558,212 params (exact,
incl. pava_theta -- model_stats WITHOUT the PAVA env prints 558,209), card/note state
2,880/1,440 unchanged** (KD is train-time only; nothing new ships to Rust). Throughput 1829.7
rev/s (same deploy model as iter 32/34; deltas are bench noise). ckpt
`scratchpad/seedpair65k/spb_d_10935.pth`; `champion_5k_track2.json` points at it (= the vprune
ref; ⚠ its trace was EXTRACTED from the WS log by `scratchpad/seedpair65k/
extract_trace_from_log.py` -- the arms ran without RWKV_STEP_TRACE -- 4-dp precision, val at
1000-step cadence, fine for vprune).
**Env = iter 34's tuned recipe PLUS `RWKV_AUGMENT_SEED=4321` + `RWKV_KD_MIX=C:\rwkv_kd_dump\
t128_seedpair_65k:10935` + `RWKV_KD_ALPHA=0.5` (KD WS-only, cleared before decay).** Why this
exists: the HP tuner ran WITHOUT KD (the old dump was bound to MAX=32768 + seed 1234), so iter
34 silently dropped iter 32's accepted KD win; iter 35 = the seed pair's arm B restores it.
**★ FUTURE CANDIDATES RUN AT SEED 4321 WITH KD** -- that keeps gates same-seed and reuses the
dump for free. The dump is bound to db/MAX/seed AND probe layout: model-side levers (PAVA
lambda) reuse it; data-side levers (MAX, RWKV_PROBE_DENSITY) need a fresh one (~1.5 h, 7.7 GB).
**The seed pair also CLOSED iter 32's seed caveat and iter 34's robustness caveat:** KD wins at
a second seed (distillation family 2/2), and arm A reproduces iter 34's means to
+0.000006/-0.000020 while per-user losses genuinely differ (mean |d| 0.0012/0.0005, coin-flip
sign) -- cross-seed spread on this recipe ~2e-5, ~20x tighter than the ~0.0004 doctrine figure
(QAT d=32 era; pair-specific observation, not a new rule). Full detail: `research_5k_verbose.md`
iter 35.

#### PREVIOUS CHAMPION = iter 34 `t65_dropout_scale_0p5` (iter-32 env + the MAX=65536 tuned recipe) -- promoted 2026-08-05 22:45
**RECTIFIED (the gate basis): ahead 0.298970 / imm 0.266217** on the VAL half (n=2500) =
+0.001298 / +0.001044 vs iter 32 at p=1.8e-152 / ~0. size 0/2500, nan_users 0. **558,212 params,
card/note state 2,880/1,440 -- all IDENTICAL to iter 32** (training-recipe only; nothing new ships
to Rust). Throughput 1797.6 rev/s. ckpt `scratchpad/tuner65k/t65_dropout_scale_0p5/t65_dropout_scale_0p5d_10935.pth`;
`champion_5k_track2.json` points at it (= the vprune ref).
**Env = iter 32's PLUS:** `MAX_TRAIN_GLOBAL_LEN=65536`, `NUM_FETCH_PROCESSES=2`, the speed stack
(`RWKV_MUON_BATCHED=1 RWKV_NO_JIT=1 RWKV_QAT_COMPILE=1`, cleared before eval), `WARMUP_STEPS=400`,
`RWKV_MUON_LR=0.0025` (8x cut -- THE win, +0.00183), `decay_ratio=1.0` (+0.00145; ⚠ total training
is now 2.0 epochs -- a budget every future run pays), `RWKV_DROPOUT_SCALE=0.5` (+0.000254 subset,
survived full-VAL; ⚠ the env var was a SILENT NO-OP on this trunk until 2026-08-03 -- the fork had
hardcoded the rates). Also 1.68x FASTER to train. Grid detail + lessons: `research_5k_verbose.md`
iter 34; journal `optimization/tuner_5k_log.jsonl` (24 rows).
⚠ warmup 400 and dropout 0.5 were adopted INSIDE the ~0.0008 noise band -- RESOLVED 2026-08-06:
the seed pair's arm A reproduced this recipe at seed 4321 to ~2e-5 both modes; they survive.

#### PREVIOUS CHAMPION = iter 32 `iter32_kd` (iter-31 trunk + full-run distillation) -- promoted 2026-07-27 23:13
**RECTIFIED (deploy, and the gate basis from iter 33 on): ahead 0.300268 / imm 0.267262** on the VAL
half (5001-7500, n=2500) = +0.000534 / +0.000429 vs iter 31 at p=3.4e-54 / 2.9e-144. Unrectified
0.298333 / 0.267207. **Both metrics agree**, so the gate-vs-deploy divergence risk did not bite.
**558,212 params, card/note state 2,880/1,440 -- all IDENTICAL to iter 31** (KD is train-time only;
nothing new ships to Rust). Throughput **1857.4 rev/s** (measured). ckpt
`scratchpad/iter32_kd/iter32d_5586.pth`; `champion_5k_track2.json` points at it.
Env = iter 31's env plus `RWKV_KD_MIX` + `RWKV_KD_ALPHA=0.5`; teacher dump at
`C:
wkv_kd_dump	128_iter32` (22,346 files / 6.96 GB -- do NOT delete, the cheap follow-ups reuse it).
⚠ ~~Seed-pair caveat OPEN~~ **CLOSED 2026-08-06 by iter 35 (the seed pair):** KD wins at seed
4321 too (+0.000160/+0.000251 within-seed, p=4.3e-13/3.3e-75) -- the +0.000429 imm margin was
not seed luck.

#### PREVIOUS CHAMPION = iter 31 `iter31_algo` (A18 trunk + PAVA + GRU N=3 + Muon)
Accepted 2026-07-26, the FIRST merged-lineage iteration. **ahead 0.298909 / imm 0.267637** on the
VAL half (5001-7500, n=2500, 0 nanskips) = +0.000393 / +0.000753 vs A18 at p=6.0e-26 / 1.5e-209;
`size` identical (0/2500 mismatches). **558,212 params** (+966 vs A18 = GRU N=2->3 + `pava_theta`);
per-card state 2,880 floats and note 1,440, both UNCHANGED (PAVA and Muon are train-time only; the
GRU head is a head, not a stream). ckpt `scratchpad/iter31_algo/iter31d_5586.pth`;
`champion_5k_track2.json` now points at it (= the vprune ref). Env = A18's full env below PLUS
`RWKV_GRU_HEAD=3`, `RWKV_PAVA_LAMBDA=0.1`, `RWKV_PROBE_DENSITY=0.08`, `RWKV_MUON=1`,
`RWKV_PROBE_DUR=0.0`.
⚠ A BUNDLE of three changes -- it establishes that the graft transfers to d=80, NOT which part
carries it. Ablation = 3 more runs, deferred pending Andrew.
⚠ Third confirmation that VAL LAG IS BIDIRECTIONAL: iter 31 trailed A18 on val all through WS and
won both modes on eval. Record val lag; never act on it.

#### PREVIOUS CHAMPION = A18 `track2_a18` (the trunk iter 31 builds on)
Accepted 2026-07-26 by Andrew's directed verdict change over an auto-reject at 108%/111% of
the ratio bar: *the >=5x product goal outranks a marginal RATE missed by ~10%*, costing only
+0.000960 ahead / +0.000532 imm cumulative vs A0 (~1/3 of what the matched-param GRU baseline
gave up). Precedent = iters 23/25/26.
- **ahead 0.299302 / imm 0.268390** on the VAL half (5001-7500, n=2500, 0 nanskips);
  **557,246 params = 4.95x below the original 2.76M** (79.8% cut); per-card state 2,880 floats.
- ckpt `scratchpad/track2_a18/t2a18d_5586.pth`; `champion_5k_track2.json` = the vprune ref.
- arch `scratchpad/track2_a18/architecture_d80_lora4.py`: d_model 80 (5 heads x K=16), LoRA
  decay/a/gate 4, v0-mix 2.
- **FULL ENV (set all of these in every run on this trunk):** `RWKV_ARCH_MODULE=<that file>`,
  `RWKV_GRU_HEAD=2`, `RWKV_STRIP_L0_VLORA=1`, `RWKV_ZERO_FEATURES=22`,
  `RWKV_STATE_CLAMP_TAU=300`, `RWKV_STATE_CLAMP_WINDOW=32768`, `RWKV_NO_AHEAD_RESIDUAL=1`,
  `RWKV_STRIP_CMIX=user_id:0,user_id:1,user_id:2,preset_id:0,preset_id:1,preset_id:2,deck_id:1,deck_id:2,card_id:1`.
- Fallback = A15 (0.299031/0.268111, 808,762 params, 3.41x — the gate-clean one).
- **THE WIDTH LADDER IS CLOSED:** two independent draws at d=80 (A17 112%/83%, A18 108%/111%)
  = a genuine accuracy floor; d=64 (A16) is ~1.8x the bar. 4.95x is the end of the width road.
  All depth floors are mapped too (card=2, deck=4, note=1, preset=3, user=3) — ladder exhausted.
  **Side-finding that sets the agenda:** the second LoRA halving is NOT free at d=80
  (+0.00002/+0.00009 for -27.5k) whereas A14's first halving IMPROVED both modes at d=128 —
  the lever flips sign as the trunk narrows, so the model is now genuinely capacity-limited and
  further gains must come from **ALGORITHMS, not shape**.

**Track-1 (d=32) lineage — closed, but its wins carry over.** Last champion iter 29
`iter29_muon` (0.302033/0.271440 val half, 171,453 params). Its three algorithmic wins are
exactly what iter 31 grafts onto the A18 trunk: **PAVA** (iter 23), **GRU N=3** (iter 26),
**hybrid Muon+AdamW** (iter 29, `rwkv/muon.py` — train-time only, nothing ships to Rust).
**Deploy contract:** learned-power PAVA rectifier on the 4 counterfactual button predictions
(current-row duration zeroed on all four) + per-step state clamp.

**QAT deploy truth (FROZEN until research closes)** = champ5k_b1 (0.306629/0.277893
quant-aware; `champion_5k.json` + its own codebooks). At research close the final champion gets
the LONG RUN below + ONE quant-aware run (q72u deploy env + the frozen NO_JIT family flags).
Plain-era and QAT-era logloss are NOT comparable.


### LIVE section chronology (iters 35-44 narratives)

#### LIVE
**THE SEED PAIR IS DONE AND RECORDED AS ITER 35 (2026-08-06)** — chain ran 20.9 h unattended,
clean end-to-end; arm B (tuned + KD) is the NEW CHAMPION (block above), iter 32's seed caveat
and iter 34's robustness caveat both CLOSED. base128_val completed earlier (drift check passed
to 1e-6/user; row 0ᵛ added).
**ITER 36 (PAVA lambda) IS DONE, RECORDED, AND DIRECTED-ACCEPTED AT λ=0.2 (Andrew 2026-08-07)** —
the new champion (block above). Dose pair λ=0.2/0.3 measured; no λ>0.1 passes the both-modes
gate, and Andrew took the 5.9:1 trade at 0.2. **The deploy contract's λ is now 0.2 — set
`RWKV_PAVA_LAMBDA=0.2` in every new run.**
**✗ ITER 37 (by-user loss weighting) DONE + REJECTED (2026-08-08): worse in EVERY size
quartile, INCLUDING the small users it was built to help** (smallest 25%: +0.000143 worse,
coin-flip sign) — the mechanism is REFUTED, not underdosed, so no 1/sqrt(N) retry. Overall
−0.000264/−0.000091 vs iter 36. Reading: small users' losses are driven by generalization from
the shared trunk, which raw data volume trains best; down-weighting big users discards signal.
Family objective-alignment 0/1, deprioritized. Hook stays in-tree (`RWKV_USER_WEIGHT`,
default off, bit-identical off — smoke `scratchpad/parity3/smoke_user_weight.py`). Full
quartile table: `research_5k_verbose.md` iter 37.
**✓ ITER 39 (KD α 0.9) ACCEPTED + PROMOTED 2026-08-08 — the new champion (block above).** The
alpha dose curve is monotone up with imm accelerating; iter 38's near-miss question is MOOT
(0.9 dominates 0.75). iter 38 stays recorded as rejected.
**✗ ITER 40 (KD α 1.0) DONE + REJECTED 2026-08-09 — and it BRACKETS THE PEAK, as designed.**
Ahead flat vs 0.9 (−0.000019, p=0.71); imm genuinely worse (−0.000067, p=1.0). The dose curve
is concave with an interior optimum at ~0.9: imm 0.266027 (α.5) → 0.265979 (.75) → **0.265875
(.9)** → 0.265942 (1.0). **The alpha lever is mapped end-to-end and CLOSED; α=0.9 stands.**
The 10% hard labels in WS still carry real signal. Family distillation: 3 accepts + 1
dominated near-miss + 1 informative endpoint reject. Open in-family if revisited: annealed
alpha / KD-through-decay (decay is HALF of total training at ratio 1.0, pure hard labels).
⚠ Ops (iter 39's first launch): generating a `.cmd` by string replacement missed the backslash
`set DIR=` line (forward-slash-only replace) — killed at minute 4, fixed, relaunched. Replace
BOTH slash styles, and assert no stale refs remain (iter 40's generator does).
**✓ ITER 41 ACCEPTED + PROMOTED 2026-08-09 — the new champion (block above), the phase's
largest architectural gain (+0.000291/+0.000396, p=5e-24/7e-95). TOPOLOGY family opens 1/1.**
**✗✓ ITER 42 (the de-bundle control) DONE 23:30 + REJECTED — AND IT INVERTS THE CHEAP
HYPOTHESIS: INTERLEAVING CARRIES THE WHOLE ITER-41 GAIN, THE REORDER IS A SMALL NEGATIVE.**
Order-alone (arch `_cnd`, sequential) scores **0.298379 / 0.266090** = −0.000489 / −0.000612
vs iter 41 (p=1.0 both) **and −0.000198 / −0.000215 vs iter 39's OLD order** on the identical
sequential recipe. The (order)×(schedule) 2×2, rectified VAL half n=2500:
old+seq (iter 39) 0.298180/0.265875 · new+seq (iter 42) 0.298379/0.266090 · new+interleaved
(iter 41) 0.297889/0.265479 · **old+interleaved = ITER 43, RUNNING**.
**Three consequences.** (1) Holding order fixed, the schedule is worth **+0.000489 / +0.000612**
— it had to overcome the order penalty to deliver iter 41's net gain, so the bundle headline
UNDERSTATES interleaving. (2) **DEPLOY: `rust/rwkv-infer` DOES need the interleave port** — the
cheap escape is closed, and since the champion also ships the reorder, the port carries both.
(3) **Fine-to-coarse is not automatically better**: the granularity intuition (notes ≈0.9×
cards, decks ≈56/user) does not survive measurement — the right axis is information flow, which
is the same story interleaving tells. Chain clean (DONE_EXIT_0, eval 2500/2500 attempt 1, size
0/2500, nan 0, leak guard 0 hits, params/state identical). Detail: `research_5k_verbose.md`
iter 42.
**✗= ITER 43 (the 4th cell) DONE 09:05 + REJECTED AS A TIE — AND IT CLOSES THE ORDER LEVER:
UNDER INTERLEAVING, WITHIN-ROUND STREAM ORDER STOPS MATTERING.** Interleave at the ORIGINAL
order scores **0.297964 / 0.265464** = −0.000075 (p=0.42) / +0.000014 (p=0.098) vs iter 41 —
the least significant pairing of the phase (every other pair has a mode at p<1e-9).
**THE COMPLETED 2×2** (rectified VAL half, n=2500):

| | sequential | interleaved |
|---|---|---|
| old order (card→deck→note) | iter 39: 0.298180/0.265875 | iter 43: 0.297964/0.265464 |
| new order (card→note→deck) | iter 42: 0.298379/0.266090 | iter 41: 0.297889/0.265479 |

**interleave effect** +0.000216/+0.000411 (old order) and +0.000490/+0.000611 (new), all
p<1e-25 — large and robust. **reorder effect** −0.000199/−0.000215 sequentially but
+0.000075/−0.000015 (noise) interleaved. The two changes INTERACT: the reorder was never a
second improvement, it was a cost the schedule paid off.
**Consequences.** (1) Champion UNCHANGED (iter 41) — a tie is not a promotion. (2) **DEPLOY,
actionable: the reorder can be dropped at zero measured cost, removing port gap 8 and letting
`rust/rwkv-infer` keep its existing hardcoded card→deck→note chain** — a Pareto-simplicity call
for Andrew, flagged not taken. (3) The ORDER lever is CLOSED (measured at both schedule
settings). (4) The SCHEDULE is the productive lever and where the next TOPOLOGY iterations go.
⚠ OPS, self-inflicted, zero GPU time lost: WS completed (i43_ws_10935.pth 02:54:37) but the
chain died 27 s later with `DONE_EXIT_WSFAIL_9009` / `'ratchpad' is not recognized` —
**`git rebase --autostash` REWROTE the running `run_iter43.cmd`**, corrupting cmd.exe's saved
read offset. Generalizes CLAUDE.md's live-`.cmd`-edit rule: **no git operation that rewrites the
working tree may touch a running runner's path until its chain reports DONE_EXIT.** Recovered
via `run_iter43b.cmd` (decay+eval from the final ckpt, guard asserting step 10935); collateral
was the WS log truncated to 99 B, so iter 43 has no step trace. Detail:
`research_5k_verbose.md` iter 43.
(Iter 41's own run record: fired 05:57 five min after iter 40's DONE_EXIT; banners verified:
interleave ON, depths [2,1,4,3,3] = the reordered arch, seed 4321:
`RWKV_INTERLEAVE=1` **PLUS the corrected fine-to-coarse stream order** (Andrew 2026-08-09,
after the order audit) via `scratchpad/track2_a18/architecture_d80_lora4_cnd.py` — same
streams/depths, tuple order now card→note→deck→preset→user (depths travel with names:
2,1,4,3,3). A 2-change bundle by direction; under interleaving the within-round order is the
smaller effect. The RNN deploy path routes states BY NAME since this change (was positional
with modules[1]=deck hardcoded — a silent state-cross-wire under any reordered arch;
refactor verified GOLDEN-EXACT). Smoke re-run under the reordered arch: depth-1 oracle
bit-exact, banner order correct, no-grad sets identical. Family = TOPOLOGY (NEW) — Andrew's
"try something more ambitious" directive (2026-08-08; HP tuning has out-gained every
architectural accept combined).** The
sequential form gives the 5-stream chain ONE pass — global context never reaches card-level
processing; interleaving round-robins the SAME layers across scopes (round r = layer r of each
stream, hierarchy order within rounds; depths [2,4,1,3,3] → 4 rounds, 13 layer-steps). Same
params/per-entity states/ops — execution ORDER only. Gathers re-anchored to the canonical
layout model-side (fetch workers + KD dump untouched); v0 stream-local; the RNN mirror reads
the same flag (`srs_model_rnn.py`), trace parity due before any interleaved champion ships.
PRE-VERIFIED (`scratchpad/parity3/smoke_interleave.py`): depth-1 oracle BIT-EXACT vs the
sequential branch, real depths differ, no-grad sets identical, scripted compile OK. Recipe =
iter-39 champion (seed 4321, KD α=0.9, λ=0.2). Tag `RWKV-iter41_ilv`, gate vs iter 39.
Runner `scratchpad/iter41_ilv/run_iter41.cmd` (banner guards on WS AND decay; INTERLEAVE
stays set through eval). ⚠ If it barely misses, the iter-37 risk note applies doubly: a new
TOPOLOGY at the incumbent's HPs — one lr_mult probe before writing the family off.
HP TUNING was ITER 34 (2026-08-05, 24 rows). The previous "3-job GPU
chain" (iter 31's rectified evals, the mode-2 duration decomposition, the mode-3 noise control,
iter 32, iter 33) is COMPLETE and was archived to `optimization/HISTORY.md` "5k-era LIVE STATE
archive (moved out of CLAUDE.md 2026-07-30)" — go there for the verbatim entries. The findings
themselves are in `research_5k_verbose.md`; the numbers are in `research_5k.md` + `log.md`. The
three that still drive decisions are quoted where they are used, not here: **training under PAVA
halves the deploy-time rectification cost** (A18 pays +0.003588 on ahead, iter 31 only +0.001893)
-> the DEPLOY CONTRACT section; **~70% of the deploy penalty is the lost current-row duration and
only ~30% is PAVA pooling** (+0.001451 vs +0.000611) -> the same section's iter-33 rationale; and
**probe-insertion noise is channel- and model-dependent, and ZERO on ahead** -> the
probe-insertion bullet in the file map.


### THE ORDER FROM HERE (items 1-3b done; queue items 0-8 largely done/superseded)

### ★ THE ORDER FROM HERE (Andrew 2026-08-01, explicit): finish HP tuning -> seed pair -> PAVA lambda

1. **iter 34 = the HP tuning itself**, recorded on whatever recipe the tuner leaves. ⚠ The grid runs
   on the 1000-user subset 5001-6000, which is a RANKING PROXY, not a gate — so before iter 34 is
   recorded the winner needs **one eval on the full VAL half 5001-7500** and a paired comparison
   against iter 32's RECTIFIED jsonls. That is eval-only (~2.5 h): the winning trial's decay
   checkpoint already exists under `scratchpad/tuner65k/<trial>/`, so nothing is retrained.
2. **✓ DONE 2026-08-06 — the SEED PAIR ran and is RECORDED AS ITER 35** (champion block above;
   detail in `research_5k_verbose.md` iter 35). Both questions answered YES: KD wins at seed
   4321 (+0.000160/+0.000251 within-seed) and the tuned recipe is seed-robust (~2e-5 mean
   spread). Arm B (tuned + KD) passed the gate vs iter 34 AND vs its same-seed twin -> promoted.
   Actual cost 20.9 h (the ~9.5 h estimate predated decay_ratio=1.0 doubling each arm's decay).
2b. **HP TUNING EXTENDED — two more coordinates (Andrew 2026-08-02: "let's add momentum and beta2
   then").** `muon_momentum` [0.95, 0.9, 0.8, 0.975] and `adamw_beta2` [0.999, 0.98, 0.95] =
   **5 unrun points x ~5.9 h = ~29 h** (they run at the incumbent `decay_ratio=1.0`, whose decay
   phase is as long as WS itself). Runs AFTER the current confirmation eval, which is for the
   pre-extension winner and stays valid as a checkpoint either way; **re-confirm only if the
   winner changes.**
   `muon_momentum` is the high-EV one: Muon's LR turned out 8x too high for this batch size, and
   momentum enters the effective step the same way (~lr/(1-m)) over the same 500,800 params. It is
   bracketed in **(1-m)**, not m — a quantity pinned near 1 cannot be bracketed by ratios.
   ⚠ `RWKV_MUON_MOMENTUM` moved OUT of the fixed trunk env into the per-trial block; a run that
   copies the old trunk string would pin it at 0.95 and silently flatten the coordinate.
2c. **TWO METHODOLOGY DECISIONS SETTLED (Andrew 2026-08-02):**
   * **AUGMENTATION STAYS OFF.** So epochs remain byte-identical replays. The consequence to keep
     in mind: any "more epochs" result measured in this configuration is measured where extra
     epochs *cannot* add data variety — which is exactly why the `champ5k_b1` A/B that pinned
     WS=1 is not evidence against a larger budget.
   * **THE EPOCH BUDGET IS NOT REOPENED NOW — it belongs to the endgame.** *"We could do more
     epochs, but that would slow every future run down, so it's best to do it at the very end."*
     So WS stays 1 epoch for every research iteration, and the budget question is answered ONCE
     by the 10x run after the features rebuild. ⚠ Note `decay_ratio=1.0` already moved total
     training 1.25 -> 2.0 epochs, so the "current budget" is 1.6x what it was — a cost every
     future run now pays, accepted because the accuracy was real (+0.00145 vs the default).
3. **✓ DONE 2026-08-07 — RECORDED AS ITER 36, rejected on gate, directed verdict pending Andrew
   (see LIVE).** Dose pair λ=0.2/0.3: big ahead wins, small linear imm costs — no λ passes both
   modes; λ=0.2 recommended if the trade is taken. Design notes kept below for the record.
   ⚠ `RWKV_PROBE_DENSITY` is data-side — changing it invalidates the KD dump
   (~1.5 h + 7.7 GB per density value); PAVA lambda alone reuses `t128_seedpair_65k` free.
   Candidate runs = the iter-35 champion recipe (seed 4321 + KD) with lambda swapped.
   **WHY THIS ONE:** the rectifier SHIPS, and iter 31 still pays **+0.001893 on ahead** purely to
   be rectified (0.298909 -> 0.300802) — roughly 4x everything the HP tuner just recovered, and the
   single largest identified loss we already know how to attack. It is a TRAINING problem, not an
   inherent cost: A18 (never trained under the constraint) paid +0.003588 and training at
   `PAVA_LAMBDA=0.1` halved it; nobody has asked whether more pressure halves it again. One env
   flag on the existing recipe, so a normal ~9.4 h iteration (the seed-pair arms' measured cost:
3.4 h WS + 3.4 h decay at ratio 1.0 + 2.6 h full-VAL eval). Extends a family that is 1/1 rather
   than reopening a closed one (conduct rule 5).
   ⚠ **Measure BOTH metrics.** This family trades raw accuracy for monotonicity, so it can look
   like a regression on the unrectified number while being a deploy win. The gate basis is now
   RECTIFIED, which is finally the quantity it improves — but record both.
3b. **THE d=128 VAL-HALF RE-EVAL + a research_5k.md row (Andrew 2026-08-02).** Runner staged at
   `scratchpad/base128_val/` (~2.5 h, queue it behind the winner confirmation). Config = **the
   model as intended: NO rectifier, piecewise-linear ahead correction ON** — which is exactly the
   2026-07-03 baseline's setup, since that run predated both `RWKV_EVAL_PAVA` and the
   `NO_AHEAD_RESIDUAL` rule.
   **It re-derives a number we already have** (ahead 0.294612 / imm 0.263561 on 5001-7500) *on
   purpose*: that five-week-old result anchors the entire phase's "~0.0037 behind upstream"
   narrative, and the model code has since gained PAVA, the GRU head, the state clamp, feature
   masking and cmix stripping. All default OFF, so it SHOULD reduce to the original forward — but
   the June parity trace proved a stale reference can invalidate a gate for weeks, and "should" is
   what failed there. New tags (`base128_val`) so the July jsonls are not clobbered;
   `compare_base128.py` reports mean AND max per-user drift and flags a mean move >1e-4.
   **THEN (Andrew's instruction, CONDITIONAL on the drift check passing): add 0.294612/0.263561 to
   `research_5k.md`.** ⚠ Do NOT add them if the re-run disagrees — report the drift instead.
   **WHY the row is worth adding:** the table's iteration-0 row currently carries the **5001-10000**
   numbers (0.2964/0.2649) while every other row is VAL-half (the ᵛ marker), so row 0 is not
   comparable to row 32 as printed. The VAL-half restriction makes the target directly diffable
   against every candidate.
4. **In parallel and CPU-only (no GPU contention): scope the input-features LMDB rebuild.** The
   endgame says start the long-lead item BEFORE the algorithmic loop runs dry, not after; design is
   already in `optimization/FUTURE_FEATURES.md`, and the rebuild is 2-4 days of CPU.

**⚠ BIG-EVAL OPS RULE (learned the hard way 2026-07-29/30):** giant users (5002/5905/5995,
266k-367k reviews) OOM the 12 GB card **iff the DESKTOP is holding several GB of VRAM** — 4.6 GB
during three separate failures vs ~0.5 GB when the same users cleared three evals overnight.
`expandable_segments` does NOT help. **Use `scratchpad/maxval/run_maxval_eval3.cmd` as the
template: NO `del` of the result jsonls**, because `eval_sharded` skips completed users, so a
relaunch only re-risks the remainder. Check `nvidia-smi` before starting a big eval.

**★★ ANDREW 2026-07-28: "Focus solely on speedups after iter 33, then continue with iter 34 once
speedups are exhausted."** So the order is FIXED: (a) record iter 33's verdict (protocol-mandated,
part of finishing it); (b) **speedups ONLY** until the list below is exhausted; (c) then iter 34.
Do NOT interleave research iterations into (b). Full measurements + method rules live in
`optimization/TRAINING_SPEED.md`; the ranked list, with the profile's sizing:

  1. **Re-run the 4-arm speed A/B on a CLEAN GPU (~25 min).** MANDATORY and unfinished: the first
     run was VOID (started 1 min after a 2.5 h eval, inherited its memory, all arms in WDDM paging
     at 12.83 GB vs 8.86 GB clean). So **whether batched Muon actually helps in wall-clock is still
     UNKNOWN** — it is implemented and numerically verified (35x fewer matmul dispatches, 0.000e+00
     CPU difference) but unmeasured. `scratchpad/profile_prep/run_speed_arms.cmd` +
     `parse_speed_arms.py`. Wait for `nvidia-smi` memory.used to settle near idle first.
  2. **MAX=65536 = 1.61x, the biggest lever** (13,899 vs 8,607 reviews/s; 81920 is over the ceiling
     and SLOWER). Config-only, no code risk, and it makes every later experiment 1.6x cheaper.
     ⚠ NOT free: groups 22,346 -> 10,935, i.e. HALF the optimizer steps per epoch, so it needs an
     accuracy run vs iter 32 and probably LR/warmup retuning (batch size is structural). Raise the
     LR question with Andrew when you get here rather than silently retuning.
  3. **`torch.compile`, re-argued.** Shelved at an honest 1.05x — but that was mixer-scoped at
     d=32, BEFORE the 17,346 `cudaLaunchKernel`/step (199 ms) was known; fusion attacks the
     dispatch count itself, which is the actual bottleneck. QAT-JIT also removed the objection to
     its `RWKV_NO_JIT` requirement. Not a free retry (whole-graph compile hits Python 3.12's Dynamo
     C-recursion cap) — mixer-scoped first.
  4. **`indexing_backward_kernel` 32 ms/step** — a THIRD deterministic-indexing site PermGather
     missed. Bit-exact if fixed the same way. ~2% of wall clock.
  5. **`aten::fill_` 7,227 calls/step** — unexplained zeroing for a 558k-param model.
  6. ⚠ **SUPERSEDED 2026-07-31 — 2 workers is NOT enough at MAX=65536; ARM THE RAM GUARD.** With
     `NUM_FETCH_PROCESSES = 2` already in effect, tuner trial 2's two workers hit **24.75 + 24.05 GB
     after ~2.5 h**, leaving **0.7 GB free of 63.9 GB** — deeper into the hang band than any of the
     three recorded hangs. One `EmptyWorkingSet` pass reclaimed **46.6 GB (1.0 -> 47.6 GB free)**
     with the run unaffected (steps advancing, fetch waits still 0.004 s). Each worker now holds the
     mmap pages for a 4x larger batch, so the old "3.8 GB/h" figure is specific to iter 33's
     MAX=16384 and must not be quoted here. **=> launch
     `scratchpad/run_ram_guard.cmd` detached (`-FloorGB 14`) alongside ANY multi-hour unattended
     training.** Original note follows.
     **`NUM_FETCH_PROCESSES` 4 -> 2** — stability, not speed: halves the 3.8 GB/h RAM climb that
     put the box in the 56-63 GB hang band. Free (fetch is 2.3 ms of a 1,450 ms step). DO IT FIRST.
  **SKIP anything targeting KERNEL efficiency** (tensor cores, chunked-matmul/fla, kernel-level CUDA
  graphs): GPU kernels are only 16% of the step, so the ceiling is tiny. That is the profile's main
  strategic message — the wins are FEWER, BIGGER ops, not faster ones.

0. **RESOLVED by the directive above** — kept for the reasoning, since it explains WHY the gate
   moved. **★ ASK ANDREW — THE GATE AND THE DEPLOY METRIC ARE DIFFERENT NUMBERS (surfaced 2026-07-27).**
   Candidates are gated on the **unrectified** logloss, but the rectifier is in the deploy contract,
   so what a user gets is the **rectified** number. iter 31 happened to win both, so nothing is
   wrong today — but the two are not the same quantity and can in principle diverge, which would
   let an iteration be ACCEPTED while being worse for users. This is exactly the principle already
   recorded in [[champion-logloss-deployed]] ("a champion's comparison logloss must be the DEPLOYED
   model, not fp32"), applied to a transform that did not exist when that rule was written.
   ⚠ **Do NOT switch the gate unilaterally** — it re-bases every comparison in the phase, and the
   rect-vs-unrect delta is model-dependent (A18 +0.003588 vs iter 31 +0.001893), so old iterations
   cannot be retro-scored without re-running them. Options to put to Andrew: (a) keep the
   unrectified gate and treat rectified as a reported-alongside deploy number (status quo, cheapest);
   (b) gate on rectified from iter 33 on, accepting that the pre-31 lineage is not comparable;
   (c) require BOTH to pass, which is strictly safer and costs one extra eval per iteration.
1. **Shrink the rectification residual — a cheap, well-posed iter 33 candidate.** iter 31 still pays
   **+0.001893** on ahead to be rectified, and training under PAVA is what recovers such penalties
   (it already halved A18's). Both levers are ONE flag on an existing recipe: raise
   `RWKV_PAVA_LAMBDA` (0.1 today) and/or `RWKV_PROBE_DENSITY` (0.08). Expect a trade — more
   monotonicity pressure should cost raw unrectified accuracy — which is precisely why item 0
   matters: judged on the unrectified gate this family looks like a regression even when it is a
   deploy win. Measure BOTH metrics for any run in it.
2. **DONE 2026-07-27 — iter 32 recorded** in `research_log.jsonl` (60 entries) + `research_5k.md`
   first table + `research_5k_verbose.md` + `logbook.py rebuild`, filed under **distillation**.
   **Both GPU jobs it owed are DONE** (2026-07-27 23:12/23:13, this entry was stale until
   2026-07-30): the **rectified eval** ran and iter 32 was **promoted to champion** on it
   (0.300268/0.267262, the gate basis from iter 33 on), and **throughput was measured at
   1857.4 rev/s** — both recorded in the `research_log.jsonl` row's `ahead_rect`/`imm_rect`/
   `throughput` fields. The **seed-pair question** on the +0.000429 imm margin (below the ~0.0005
   threshold; the rectified eval re-scores the SAME training run, so it confirms the metric, not
   the seed) is **DECIDED — Andrew 2026-07-30: re-run it at a different seed AFTER the HP tune.**
   Design + cost in item 3.
3. **✓ DONE 2026-08-06 — RECORDED AS ITER 35 AND PROMOTED (see the champion block).** The
   design below ran exactly as specified (`scratchpad/seedpair65k/run_seedpair.cmd`); kept for
   the reasoning. **★ SEED-PAIR RE-RUN — SPECIFIED BY ANDREW 2026-07-30: "let's do iter 31 and iter 32 both with
   the same seed after HP tune."** So it is a **TWO-ARM, SEED-MATCHED** test, run AFTER the tuner
   on whatever recipe it leaves: the tuned recipe **WITHOUT KD (the iter-31 arm)** and **WITH KD
   (the iter-32 arm)**, both at **`RWKV_AUGMENT_SEED=4321`**, and the margin taken WITHIN that seed.
   * **Why both arms.** The quantity in doubt is **iter32 − iter31 = +0.000429 imm**, a difference
     BETWEEN two runs. Re-running only the KD arm at 4321 and diffing it against iter 31 at 1234
     would measure seed noise plus KD, not KD. (Contrast the sibling's q72u seed pair, which was a
     WITHIN-run penalty and so needed only one run per seed.)
   * **Why the tuned recipe rather than iter 31/32's original one.** Same cost either way, but this
     version doubles as the KD re-confirmation under the HPs the lineage will actually use — and
     re-confirming an obsolete recipe answers a question nobody will ask again. It does mean the
     seed-4321 margin is compared against a seed-1234 margin measured at MAX=32768 with untuned
     HPs, i.e. "does KD still win at a second seed" rather than a strict 2x2.
     **OPTIONAL 2x2 if a strict one is wanted:** the tuner's winning trial already IS the
     no-KD/seed-1234 arm for free, so only **KD/seed-1234-tuned** is missing (+1 run +1 dump,
     ~5.5 h). Andrew's call; not part of what he specified.
   * **The dump at `C:\rwkv_kd_dump\t128_iter32` is bound to MAX=32768 AND seed 1234.**
     `train_rwkv.py:681-683` states the contract — *"batch-stream identity REQUIRED: same db/MAX/
     seeds; mismatch = hard exit 43, NOT a skipped batch"* — and the student re-checks a labels
     checksum per step (`:1216-1224`). Changing the seed invalidates it, and so does the adopted
     **MAX=65536**, which regroups 22,346 steps into 10,935. Failure is LOUD (exit 43), so this
     cannot silently corrupt a run — but it does mean a fresh dump.
   **=> cost ≈ 2 x ~4.0 h training + ~1.5 h teacher dump ≈ 9.5 h.**
   **OPERATIONAL FACTS for the fresh dump (mapped 2026-08-03 from `run_iter32_kd.cmd`, so the seed
   pair can be launched without re-deriving them):** teacher = `pretrain/RWKV_trained_on_101_4999.pth`
   (the original d=128 model, forward-only); the runner does smoke-5-steps -> `check_dump.py`
   (semantic + disk projection, `--max-gb 60`) -> full dump -> assert `step_<WSSTEPS>.pt` exists.
   **`WSSTEPS` becomes 10,935 at MAX=65536** (was 22,346). Size ~7 GB again (half the files, ~2x each);
   C: has 248 GB free, so the old dump can stay.
   ⚠⚠ **THE RUNNER DOES `if exist "%DUMP%" rmdir /s /q "%DUMP%"` BEFORE THE FULL DUMP.** Reusing the
   name `t128_iter32` would therefore DESTROY the one artifact that can reproduce iter 32 exactly as
   accepted — the thing this very entry says not to delete. **Use a new `%DUMP%` name** (e.g.
   `C:\rwkv_kd_dump\t128_seedpair_65k`).
   ⚠ **THE SAME CORRECTION KILLS THIS ENTRY'S OLD CLAIM** that the annealed-alpha variant and the
   alpha sweep are "student-only re-runs, no dump". That was true at MAX=32768; at the adopted
   MAX=65536 they each need the fresh dump too (one dump serves all of them, so batch them with
   the seed pair rather than paying for it repeatedly). Curve-level distillation (variant 3) still
   needs new dump code on top.
   ⚠ **Do not delete `C:\rwkv_kd_dump\t128_iter32` yet** — it is the only artifact that can
   reproduce iter 32 EXACTLY as accepted (MAX=32768, seed 1234), which is what any re-audit of the
   accepted result would need. It is dead weight for new work, not for verification.
4. **RUST PORT** (`rust/rwkv-infer/TRACK2_PORT_PLAN.md`) — ⚠ **THIS ENTRY WAS STALE; both of its
   "remaining" items are DONE (verified 2026-07-27).** The button API landed in `fast.rs`
   (`button_intervals` at `fast.rs:703`, plus `tile_states_b1` and the `--buttons-fast` /
   `--buttons-fast-selfcheck` drivers), and the measurement it gated is recorded in
   `CPU_INFERENCE.md` Measurement 2: **a 4.96x param cut buys 2.39x rev/s on the Rust path**
   (714 -> ~1,703 rev/s), versus 1.24x and plateauing in the Python RNN path — i.e. **the ablation
   programme DOES pay off for Anki users, in the engine that will actually ship**. That answers
   Andrew's 2026-07-25 question directly. `PARITY: PASS` for A18 (§11), so those throughput
   numbers are backed by verified-correct predictions.
   **What is genuinely left**, in value order: ~~(a) cost of the button API~~ **(a) DONE 2026-07-27
   — `--bench-buttons`, CPU_INFERENCE.md Measurement 3: serving all 4 intervals costs 0.76 ms/card
   = 2.69x a plain prediction, of which 93% is the probe forward and only 7% the 50-step bisection.
   Cheap in absolute terms, and NOT a lever — the solver is noise and the 4 probes already batch to
   2.5x rather than 4x, so nothing cheap is left in it**;
   ~~(b) throughput with QUANTIZED state~~ **(b) DONE 2026-07-27 — Measurement 4. ★ THE 256x STATE
   COMPRESSION COSTS ~3x INFERENCE TIME** (review 0.307 -> 0.917 ms; with buttons 0.839 -> 3.530 ms
   = 4.2x). Absolute cost stays under 4 ms/card, so this is a **Pareto CHOICE to put to Andrew**,
   not a blocker: 9 bytes/card and 3x slower, or 51 KiB/card and fast. The compression work was
   justified on SIZE alone and its time price had never been on the table. Most expensive single
   component = **shift PQ + 1-bit norms** (+68% of baseline in one rung; m2b12L searches 2x4096
   entries per shift vector per layer per stream vs the WKV side's one 1024-entry catalog) — the
   same lever that dominates the BITS, so one change moves both. ⚠ Warm search
   (`warm_wkv`/`warm_shift`, which travel with the state) is load-bearing: a cold-search harness
   reports 6.17x instead of the true 2.99x; (c)
   the AVX2/FMA `dot_product` + `add_scaled_in_place` from `vendor/jschoreels_anki/x86_simd.patch`
   — our engine has NO SIMD, and that is the most portable remaining speed win (⚠ AGPL: flag to
   Andrew before copying, keep provenance comments).
5. **NEW INPUT FEATURES — now on the critical path** (Andrew 2026-07-26; see "THE ENDGAME,
   ORDERED"). **★ Andrew 2026-08-09: the rebuild must DROP Anki's card state
   (new/learning/review/relearning — the `state − 2` column, dim 22) from the input vector** —
   the permanent form of iter 15's ZERO_FEATURES=22 mask; cautions (keep state for the
   FILTERS, renumber mask/Rust dim consumers) recorded in FUTURE_FEATURES.md's plan.
   `optimization/FUTURE_FEATURES.md` + the deck-tree features. **SCOPED 2026-07-27** —
   that doc now carries the four code sites, the F:-side-by-side disk plan (no delete needed), and
   a 100-user de-risk build. **CONSTANTS MEASURED 2026-08-03** (`optimization/feature_stats_id.py`,
   300 users / 24.3M reviews, train half only) and **WALL-CLOCK MEASURED: ~23 h for both DBs**
   (6,671 rev/s on a contended box; an overestimate — see the doc). **SCOPING IS COMPLETE.**
   ⚠ Two blockers that inspection had missed, both found by actually running it: (a) the `-id` swap
   is **NOT** a one-line `DATA_PATH` change — `data_processing.py:408` asserts an exhaustive column
   partition and dies on the extra `review_time`; (b) the landmine below. **★ AND A THIRD, measured
   2026-08-03: `label_filter_db` MUST be rebuilt and the `size` GATE WILL LEGITIMATELY MOVE** —
   `create_features`' outlier/continuity filter amplifies the 0.001% raw-row difference into
   **70% of users getting a different equalized set and 30% a different `size`** (user 17:
   108,870 -> 109,025). So gate #1 ("`size` IDENTICAL, any change = a pipeline bug") must be restated
   as *within a rebuild generation*, and the plan's own de-risk check was invalidated and redesigned.
   ⚠ **That measurement found a LANDMINE: the `-id` rebuild produces NaN features on 3.2% of users**
   (48 rows in 25M, but one bad row poisons a user; ~160 train / ~80 eval users). `build_parquet_id`
   recomputes `elapsed_seconds` from the SHOW time, which goes negative-but-not-`-1` when a duration
   overlaps the next review, and `scale_elapsed_seconds` then takes `log` of a negative. The
   published set has ZERO such rows. One-line clamp + the three forced design corrections (card−deck
   gap is 57% negative; deck ids are timestamps on only 70% of REVIEW rows; preset age stays
   flag-only) are in `FUTURE_FEATURES.md`. Budget question CLOSED: the 10x run happens once, last, after this.
6. Entropy-floor analysis (~30 min GPU; design in `research_5k_notes.md`); permutation init (LOW).
   `pava_loss_avg` / `pava_pool_frac` step-trace fields: DONE (train_rwkv.py, keyed on enablement).
7. **POSTPONED by Andrew 2026-07-27 — the users-vs-epochs ablation** (2,500 users x 2 epochs vs
   iter 31's 5,000 x 1, step count held fixed; a 2-line toml change, ~6-7 h, no rebuild). Design +
   why it matters in `optimization/DATASETS.md`. Do NOT start it unprompted. It is the cheap test
   of whether the model is data-limited at 5,000 users, so it is worth re-offering **before** the
   ~4-day 10x-budget endgame run, whose whole premise is that more epochs buy the +0.0037/+0.0043.
   Same doc's VERDICT: **do not train on FSRS-Anki-20k** (1.5 TB LMDB vs 1.13 TB free; no
   note/deck/preset = a regime that is 0% of deployment; 4.3% of its users leak into our eval half).
8. **★ ANDREW 2026-07-27: "After iter 33 see if you can speed up training." → MEASURED, and the
   answer is in `optimization/TRAINING_SPEED.md`. READ THAT FIRST; the notes below are the older
   plan that led to it.**
   **THE HEADLINE: the step is CPU-DISPATCH-BOUND, not kernel-bound.** At d=80: wall clock
   ~1,450-1,540 ms/step, GPU kernel time only **237 ms (~16%)**, self CPU **915 ms**, and
   **90,576 op dispatches per step** (`cudaLaunchKernel` alone = 17,346 calls / 199 ms). Fetch is
   2.3 ms, re-confirmed as a non-lever. **=> faster kernels are near-pointless; the wins are
   FEWER, BIGGER ops.** This also explains iter 33's 2.83x-per-step cost on 1.13x rows, and
   retro-justifies the tensor-core / chunked-matmul dead ends (both attack the 16%).
   **★ PAVA IS EXONERATED** — 443 dispatches/call is ~1% of 90,576, and ALL stream syncs in the
   whole step total 17 ms. The dead `break` at `pava.py:92` is still worth deleting (bit-exact)
   but it is a tidy-up, not a speedup. **Do not spend effort optimizing PAVA.**
   **`empty_cache` RULED OUT by direct A/B:** 0.6484 -> 0.6887 steps/s = **1.06x**; it guards a
   documented 4x paging failure, so leave it.
   **BEST CONCRETE TARGET = batch Muon's Newton-Schulz** (`muon.py:80` runs it per-parameter in a
   Python loop: 2,658 `aten::mm`/step, 92 ms CPU to do 21.6 ms of GPU work). Group by shape +
   `torch.bmm` => ~10x fewer matmul dispatches, **~6-7% expected**. ⚠ NOT bit-exact (bmm changes
   reduction order), so it needs an accuracy check and the seed-pair doctrine applies.
   Also open: `indexing_backward_kernel` 32 ms/step = a THIRD deterministic-indexing site the
   PermGather work missed (bit-exact if fixed the same way), and `aten::fill_` 7,227 calls/step.
   (Original plan follows — the GPU-needed parts are done; QAT-JIT is staged at `scratchpad/qat_jit/`.)
   **Start with measurement, not ideas: the last profile was at d=32/MAX=110000 and is two
   architectures stale.** Re-profile the CURRENT d=80 trunk with the existing hook —
   `RWKV_PROFILE_STEP=N` + `RWKV_PROFILE_COUNT` (train_rwkv.py:790-795) wraps N steps in
   torch.profiler, prints bucketed self-GPU-time (QAT / wkv scan / wkv recurrence / gemm /
   elementwise) and exits; off by default so training stays byte-identical. ~5 min GPU.
   **★ QAT-JIT: DONE 2026-07-27, and the speed half is a NEGATIVE result — do not quote ~1.38x
   for QAT.** NUMERICS SETTLED: with a null control (two identical-flag runs) clean at 0/160,
   nojit-vs-jit is also **0/160 mismatches** over 80 real training steps on the CUDA
   `qat_lr_rank1` kernels — **`RWKV_NO_JIT=1` is not required by QAT**. SPEED NOT ESTABLISHED:
   identical-flag runs differ by **5.3%** and jit-vs-nojit is only **1.06x**, i.e. inside the
   noise. The ~1.38x came from the NON-QAT body and does not transfer, which fits the dispatch
   finding above: TorchScript removes Python overhead, not `cudaLaunchKernel` overhead.
   **So the "worth ~1.5 days of the 10x run" claim is withdrawn.** ⚠ All three arms peaked at
   **12.807 GB on a 12 GB card** (QAT at MAX=32768 is over the ceiling, into WDDM paging), so
   re-measure at a lower MAX before concluding anything about QAT speed.
   **A NEW lead from iter 33 (2026-07-27):** per-step cost is far more sensitive to `max_batch`
   (GPU parallelism) than to rows — halving MAX cost 2.83x per step on only 1.13x more rows. That
   says the step is parallelism/launch-bound at small B, which points back at the elementwise mass
   (78% of the step at the d=32 profile) rather than the WKV kernel.
   **DO NOT RE-RUN THE KNOWN DEAD ENDS** (all in the SPEED and LESSON BANK sections above): tensor
   cores (Amdahl <1%), `torch.compile` (honest 1.05x, shelved), CUDA graphs (variable shapes,
   1.1-1.3x), the chunked-matmul/fla rewrite (addresses <=18%), and fetch-side work (already hidden,
   ~1 ms waits confirmed again on iter 32 and 33). Already banked and not repeatable: deterministic
   indexing 1.5x, JIT 1.38x, EMPTY_CACHE_EVERY=0 1.12x, the QAT kernel 6.3x, the WKV scratch
   allocator.
   **★ ANDREW 2026-07-27: "Make sure it also profiles PAVA rectification and anything else that
   could be slow." THE EXISTING PROFILE CANNOT — that is a real gap, not a tuning detail.**
   `_print_kernel_profile` buckets by **CUDA kernel NAME**, which works for the WKV kernels and
   gemms but not for PAVA / the GRU head / Muon / the state clamp: those emit only generic
   elementwise/reduce/gather/where kernels, so all of them land in the catch-all "other" bucket that
   was already 78% of the step. **Worse, the summary reports GPU self-time only, so a region that is
   LAUNCH- or SYNC-bound looks nearly free in it while being expensive in wall clock** — and the
   iter-33 finding (cost tracks parallelism, not rows) says that is exactly the regime we are in.
   **BUILT 2026-07-27, ready to wire: `rwkv/profile_regions.py`** — `region("name")` emits a
   `record_function` scope (a shared `nullcontext` when `RWKV_PROFILE_REGIONS` is unset, so
   annotated code stays byte-identical), and `region_report()` prints **CPU and DEVICE time
   side by side per region**, flagging `cpu/device > 3` as OVERHEAD-BOUND. Wire call sites at:
   `pava_rectify` (`model/pava.py:57`), `_pava_probe_loss` / `_pava_rectify_eval`
   (`srs_model.py:549,513`), the GRU curve head, the state clamp (`rwkv_model.py:633`), Muon's
   `zeropower_via_newtonschulz5` (`muon.py:27`) and its optimizer step, plus feature FC / curve head
   / rating head / loss. ⚠ Those are all files iter 33's DECAY phase re-imports, so **apply the
   call-site edits only after iter 33 finishes**; the helper is a new file and touches nothing.
   **★ PAVA MEASURED STRUCTURALLY ALREADY (CPU-only, `scratchpad/profile_prep/bench_pava.py`):
   443 aten ops per call and the count is INDEPENDENT of M** (443 at M=1,000 and at M=100,000)
   = ~2.7 ms of pure launch overhead per call at ~6 us/launch, forward, plus the backward.
   **And the early `break` at `pava.py:92` is DEAD CODE: 6/6 back-merge iterations ran at every
   pooling rate tested (0.10 / 0.50 / 0.98)**, because `merge.any()` reduces over the WHOLE batch
   and with tens of thousands of rows some row always violates. So every call pays **6 device->host
   syncs** for a branch that never fires, and each sync drains the GPU queue and kills CPU run-ahead.
   **=> FIRST CANDIDATE FIX, likely free and bit-exact: delete the `break`.** With `merge` all-False
   the loop body's `torch.where(upd, ...)` is already a no-op, so running it changes nothing
   numerically; the NaN-through-`where`-backward hazard is unchanged (it already applies to
   non-merging rows in iterations that do run). Size it with the region profile before claiming a win.

**⚠ CPU-INFERENCE REALITY CHECK (Andrew 2026-07-25: "I told you to do ablations hoping that
fewer params -> faster CPU inference in Anki").** Measured in `optimization/CPU_INFERENCE.md`:
in the PYTHON RNN path a 4.5x arithmetic cut buys only **1.24x** wall-clock and PLATEAUS after
A14 — that path runs at 0.08-0.30 GMAC/s vs a core's 5-20, so it is OVERHEAD-bound and cost
tracks op count (layers x streams), not width. **1 thread beats 3 and 6 → deploy
single-threaded.** The deploy path is Rust (~10x faster, far less per-op overhead) where width
SHOULD pay off — which is why the port is the gating work for whether the ablations bought
user-visible speed. Bench: `python optimization/cpu_infer_bench.py`.
(Training speed IS monotone in width — median steps/s A0 0.933 -> A16 1.746 = 1.87x faster at
7.11x fewer params, sublinear as the elementwise-dominated profile predicts.)


### deploy-contract narrative (the ZERO_FEATURES bug story, the A/B/global-vs-surgical design, queue item 0) (moved out of CLAUDE.md 2026-08-10)

### ★★ THE DEPLOY CONTRACT IS NOW ONE QUANTITY IN ALL THREE PATHS (Andrew, 2026-07-27)
> *"Everywhere (train+eval+CPU inference): duration of the most recent review zeroed out + PAVA +
> no piecewise correction. And yes, train with zeroing as iter 33."*

This settles the gate question below (QUEUE 0) by removing the divergence instead of choosing a
side: **train, eval and CPU inference must all compute the SAME quantity**, namely
1. the most recent review's duration zeroed, 2. PAVA rectification applied, 3. no piecewise
ahead correction (`RWKV_NO_AHEAD_RESIDUAL=1`, already in every run).

**Consequences, in order of how much they change:**
- **The gate becomes the RECTIFIED metric.** Eval runs `RWKV_EVAL_PAVA=1` from iter 33 on. The
  champion baseline to beat is therefore iter 31's **rectified** VAL-half numbers —
  **ahead 0.300802 / imm 0.267691** (n=2500) — NOT the 0.298909/0.267637 the front table shows.
  Pre-iter-31 rows are unrectified and are NOT comparable; do not retro-score them, the
  rect-vs-unrect delta is model-dependent (A18 +0.003588 vs iter 31 +0.001893).
- **iter 32 straddles the change.** It was launched on the old basis (unrectified eval, no training
  zeroing), so judge it against iter 31 UNRECTIFIED as designed, and run a rectified eval before
  comparing it with anything from iter 33 on.
- **★ `RWKV_ZERO_FEATURES` WAS MISSING FROM `rust/rwkv-infer` — FOUND AND FIXED 2026-07-27, AND IT
  WAS A LIVE BUG, NOT A LATENT ONE.** The mask lives INSIDE the Python module (`srs_model.py:314`,
  `srs_model_rnn.py:50`), so the exported trace carries RAW features (verified: `feats_proc` dim 22
  is nonzero on 45.1% of rows) and the engine was consuming columns Python had thrown away.
  Measured on iter 31, same weights and trace, only the mask differing:

  | | mean \|rust-python\| | max per-review \|diff\| |
  |---|---|---|
  | no mask (how the gate ran) | imm 8e-6 / ahead 1e-6 | **1.59e-03** |
  | mask applied | imm 0.000000 / ahead 0.000000 | **2.28e-06** |

  ~700x tighter. It survived the gate because the gate scores MEAN LogLoss and dim 22's input
  weight is small (column L2 0.844 vs median 8.0) — the error partly cancels across reviews.
  ⚠ **This also CORRECTS §11's explanation of the per-review spread**, which called it "accumulated
  float divergence over a ~5,000-step recurrence ... not a formula error". It was a formula error;
  over the identical recurrence a masked engine agrees to 2.28e-6.
  **Fix:** `model.rs::load` zeroes the named input columns of `features2card.0.weight` once at load.
  Zeroing input column j is exactly equivalent to zeroing feature j (`y = Wx+b` is linear in x), so
  it costs nothing at runtime and cannot be forgotten at one of the three call sites.
  **This was about to get much worse:** iter 33 masks dim 8, whose column L2 is **42.07 — 5x the
  median and 50x dim 22's**. Unmasked that is a large error, not a subtle one. Exactly the failure
  class §9 exists to catch, caught by asking what the deploy path actually computes.
  **→ RECOMMENDED FOLLOW-UP (Andrew's call, not done unilaterally):** bake the mask into the
  **exported safetensors** at export time instead of applying it from an env var at load. Since it
  is just zeroed weight columns, the deploy artifact would then be correct for ANY consumer without
  that consumer knowing the flag exists — and Anki will not be setting `RWKV_ZERO_FEATURES`. The
  env path stays useful for research runs; this only changes what ships.

**⚠ IMPLEMENTATION IS A REAL FORK — read before writing the run.** "The most recent review's
duration" is the CURRENT row's, and in a causal RNN each row's duration is unavailable to its OWN
prediction but perfectly available to every LATER one (at deploy it is history the moment the user
presses). So:
- **(A) GLOBAL zeroing** — `RWKV_ZERO_FEATURES=22,8` (`scaled_duration` is index 8; confirmed
  independently by `CARD_FEATURE_COLUMNS[8]` and Rust `COL_DUR=8`). One flag, exact three-way
  parity by construction, ~free. But it is a SUPERSET of the ask: it also destroys PAST reviews'
  durations, which deploy genuinely has.
- **(B) SURGICAL zeroing** — exclude the row's duration from its own ahead prediction while keeping
  it in the state passed forward. This is precisely what a probe (a skip row) already does, so it
  means `RWKV_PROBE_DENSITY=1.0` with the ahead loss taken from the pressed probe. Faithful, but
  +1 row per scored review is ~+93% rows ≈ **2x training cost** (baseline is ~1.07 rows/scored
  review: 4 probes at density 0.08 inflate the batch ~30%).
Recommendation: run **(A) first** — it is the cheap upper bound on the parity benefit and its
result tells us whether past-duration signal is worth (B)'s 2x. If (A) wins the rectified gate,
take it; if it loses by less than the +0.001451 duration penalty it removes, (B) is justified.

5. **✗ DONE + REJECTED 01:42 (2026-07-29): iter 33 = the duration fix.** Rectified-vs-rectified vs
   the iter-32 champion: **ahead 0.303055 (-0.002787), imm 0.268066 (-0.000805)**, p=1.0 both;
   size identical (0/2500), nan_users 0, params unchanged. **The hypothesis as stated is NOT
   supported** — training without the current row's duration did not shrink the deploy penalty, it
   made the deploy number worse.
   ⚠ **BUT IT CANNOT ATTRIBUTE — three changes shipped together**, and that is the run's design
   fault: (1) the duration withholding (the hypothesis); (2) ⚠ **THIS ITEM WAS WRONG — CORRECTED
   2026-08-10.** It said `RWKV_AHEAD_PROBE_ONLY=1` "dropped the ahead supervision on the ~23.5% of
   rows probes cannot cover". The code does the OPPOSITE (`srs_model.py:1010-1013`, whose own
   comment says "Rows NOT eligible for probes keep the real-row term"): it zeroes `ahead_mask`
   **only at PROBED rows**, so the 23.5% kept full weight and the probed **76.5%** moved onto the
   PAVA probe path — which enters as `pava_lambda * pava_loss` at **lambda=0.1** (`:1114`) versus
   the real-row term's scale 1.0. That is a **10x downweighting of the ahead objective for the
   majority of rows**, with the remaining gradient dominated by a biased subsample (first-in-chunk
   reviews = least history). Bigger confound than the one we recorded; (3) MAX 32768->16384, which
   halves batching and doubles the step count. Any of the three could produce -0.0028, and (2) is
   the most likely single culprit.
   **Retry design, if the family is revisited:** do NOT use probes to withhold duration. Use
   per-row Bernoulli dropout on `scaled_duration` (dim 8) at the model input — no probe-density
   change, no row inflation, no MAX change, no loss reweighting, so only the duration varies.
   iter 18 (permanent removal, -0.0018/-0.0024) brackets the p=1.0 end. Full notes in
   `research_5k_verbose.md`.
   ROBUSTNESS: 31 h, survived a hard hang at step 33,003 + two clean stop/resumes, **total loss 3
   steps of 43,354**.
   Historical detail of the run below.
   **(superseded) ▶ RESUMED 00:23 (2026-07-28) — iter 33 = the duration fix.**
   `scratchpad/iter33_dur/run_iter33_resume.cmd`, **pid 5088**. Verified from the log:
   `[resume-skip] epoch 0: skipping the first 14000 already-trained groups (resume at global step
   14001)`. ~29,354 WS steps remain + decay + rectified eval.
   **Its GATE now pairs against iter 32 RECTIFIED** (the new champion), not iter 31 — that is the
   one substantive change vs the original runner, and it is free because iter 32's rectified jsonls
   now exist. The resume runner also drops the waitloop and the 40-step sanity phase (already
   passed) and fixes the stale "vprune ON min6000" banner.
   ⚠ **`RWKV_MUON_BATCHED` is deliberately NOT set** — the batched Newton-Schulz (13fd1b1) perturbs
   optimizer numerics on GPU and steps 1-14,000 were trained without it; the speedup goes to the
   NEXT run rather than splitting this one across two implementations.
   Stop/resume history: STOPPED 17:38 2026-07-27 at Andrew's request (cable management, not a
   failure) at WS step 14,000 of 43,354.
   **RESUME POINT: WS step 14,000 of 43,354 (32%)** — ckpts `scratchpad/iter33_dur/iter33ws_14000.pth`
   + `iter33ws_optim_14000.pth` (17:33:59). The step trace had 14,934 lines, so a resume re-does
   ~934 steps, inside the documented <=1000-step loss.
   **HOW (the documented mid-epoch resume, LIVE RULES section):** `RWKV_RESUME_SKIP_GROUPS=1` +
   `python scratchpad/make_resume.py scratchpad/iter33_dur iter33ws scratchpad/iter33_dur/iter33_dur_ws.toml`,
   then rerun the WS phase with the run's **FULL env** and **WITHOUT deleting the step-trace files**.
   The resumed tail's dropout draws differ (weights/optim exact) — statistically equivalent, not
   bit-identical. ⚠ The `.cmd` runs WS -> decay -> eval as one chain, so a plain relaunch would redo
   WS from zero; resume the WS phase, then let the remaining phases run.
   **REMAINING: ~16 h WS + 5.7 h decay + 2.5 h eval = ~24 h** at the measured 0.527 steps/s.
   ⚠ While it is stopped the GPU is FREE, so the jobs it was blocking can run first — in value
   order: **iter 32's rectified eval** (unblocks the champion question), `measure_throughput.py` on
   `iter32d_5586.pth`, the **QAT-JIT GPU half** (~15 min, ~1.38x), and the **d=80 re-profile +
   region wiring** (queue item 8, Andrew's speed directive). Several of those are short enough to
   finish before a resumed iter 33 would need the card.
   Original launch record: the waitloop fired 5 min after iter 32's `DONE_EXIT_0`, as designed.
   `scratchpad/iter33_dur/run_iter33_dur.cmd`, **pid 15496**, log
   `scratchpad/iter33_dur/iter33_dur.log`.
   ⚠ Its gate still pairs against **iter 31's** rectified jsonls, which is correct and deliberate —
   iter 32 has no rectified eval yet (see item 4). If iter 32 gets one before iter 33's gate runs,
   the baseline can be re-pointed for free; do NOT re-point it to iter 32's UNRECTIFIED numbers.
   ⚠ **ITS WS BANNER LIES: it prints "vprune ON min6000" but vprune is OFF, as designed.** Verified
   two ways — no `RWKV_VPRUNE_*` env is set anywhere in the `.cmd` (only a REM line), and
   `train_rwkv.py:910` arms vprune only when `RWKV_VPRUNE_REF` is non-empty. A stale echo string
   copied from a template; the `.cmd` could not be corrected once running (editing a live `.cmd`
   corrupts cmd.exe's saved read offset). **Fix the string after the run**, and do not read that log
   line as evidence the run was prunable.
   `SANITY OK` at 09:53:49 — the 40-step VRAM check passed at MAX=16384 with density 1.0, so the
   post-probe row count fits the 12 GB card. WS started 09:53:49.
   ⚠ **★ IT IS A ~31 h RUN, NOT ~16 h — the projection modelled ROWS but not BATCHING (measured
   2026-07-27, 5-min window at steady state: 0.527 steps/s).** WS **43,354 steps -> 22.9 h**, decay
   10,838 -> 5.7 h, rectified eval ~2.5 h; **verdict ~17:00 on 2026-07-28**.
   **Why:** `max_batch = floor(MAX/size)` and the largest chunks in `train_db_5k_h1` are exactly
   16,384 rows, so at MAX=16384 they get `max_batch = 1` — **no batching at all**, where MAX=32768
   gave them 2. Step count rose 1.94x as modelled (43,354 vs iter 32's 22,346) but **per-step cost
   rose 2.83x while rows/step rose only 1.13x** (41.6k post-probe vs iter 32's 36.8k). Fetch waits
   are ~1 ms in both runs, so it is not the loader — it is lost GPU parallelism. (Inference from the
   group counts plus the known elementwise/B x T-shaped profile, not a direct kernel measurement.)
   **No cheap rescue:** raising MAX leaves those 16,384-row chunks at `max_batch = 1` until MAX
   doubles back to 32768, which is the VRAM ceiling that forced 16384; and density 1.0 is
   load-bearing (`RWKV_AHEAD_PROBE_ONLY=1` needs it to cover the ahead rows).
   **=> LESSON for any future MAX change: cost is NOT linear in rows. Halving MAX cost 5.2x WS
   wall-clock here (4h23m -> 22.9 h), because it also halves batching. Estimate from `Number of
   groups` x a measured steps/s, never from row counts alone.**
   ⚠ **It also BLOCKS the queue for ~31 h** — iter 32's rectified eval and throughput both need the
   GPU and the no-co-tenant rule applies, so the champion question stays open until this finishes.
   `RWKV_PROBE_DENSITY=1.0` +
   **`RWKV_AHEAD_PROBE_ONLY=1`** (new flag): the ahead objective moves entirely onto the
   duration-zeroed probe path, which is the quantity deploy serves. Eval is **RECTIFIED**
   (`RWKV_EVAL_PAVA=1`) and the gate pairs against `RWKV-iter31_algo_rect.jsonl`.
   **Projected ~16 h** (WS ~11 h at 2.54x rows, decay ~2.7 h, rectified eval ~2.5 h).
   ⚠ **`MAX_TRAIN_GLOBAL_LEN` lowered 32768 -> 16384 and it is load-bearing.** Probes are inserted
   AFTER grouping, so density 1.0 does not add steps — it makes each batch ~2.54x LONGER, which at
   MAX=32768 is ~83k rows on a 12 GB card. 16384 keeps post-probe rows near iter 31's effective
   ~36.7k and doubles the step count instead. **16384 is the FLOOR, not free**: the largest chunk
   is 16,384 and `get_groups` SILENTLY DROPS anything larger than MAX. The 40-step sanity phase is
   the VRAM check; a `DONE_EXIT_SANITYFAIL` means lower the batch, not abandon the idea.
   ⚠ vprune OFF (objective changed + MAX moved, so step pairing is meaningless). The gate's
   baseline is only the final `paired_pvalue` call, so it can be RE-POINTED for free after iter 32's
   verdict without re-running anything.


### section 11's parity archaeology (the June stale-trace saga, A18 and iter-31 procedures) (moved out of CLAUDE.md 2026-08-10)

**Rust-parity invariant:** `verify_rust.py` (3-user float32) must pass for
the champion arch before "shipping" (re-export trace + match the trained model bit-exactly).
⚠ **CORRECTED 2026-07-26 -- the old instruction here was WRONG and self-confirming.** It said to run
with `RWKV_WEIGHTS=reference/rwkv_iter36_124.safetensors` and that the default `rwkv_ref_558` "will
MISMATCH (wrong-weights, not a regression)". In fact **`verify_rust.py` never runs the engine** -- it
scores `reference/rust_pred_<user>.json` left by an earlier manual run, so `RWKV_WEIGHTS` cannot affect
its verdict, and any weights argument "works" or "fails" identically. Those files were stale (Jun 30,
quant-ladder era), which is how a FAIL with identical dpred across 3 crate versions and 3 weight files
went unnoticed. Correct procedure: **run the binary from the REPO ROOT** (`RWKV_WEIGHTS=...
./rust/rwkv-infer/target/release/rwkv-infer.exe`; it resolves `reference/trace_user_*` relative to CWD)
-> it writes `preds/rust_pred_*.json` -> **copy those into `reference/`** -> `python verify_rust.py`.
`reference/ref_metrics.json` names the reference model: **`rwkv_ref_558.pth`**, not iter36.
**★ SOLVED + GREEN 2026-07-26 -- the ROOT CAUSE was that the June `reference/` trace is not
reproducible by current Python, so the gate was scoring the artifacts, not the port.** New tool
`scratchpad/parity3/trace_selfcontained.py` asks the question that settles it: feed the trace's own
92-dim features back through the Python RNN at review 0 (all states empty = pure forward pass) and see
if it reproduces the `py_pred` frozen in the same file. The June trace FAILS at |d| up to 3.4e-1 -- the
same magnitude as the Rust "error" -- because `architecture.py` has since moved from d=128 to d=32 and
the model code has evolved, so nothing can reproduce those numbers now. **Run this check FIRST whenever
a parity gate looks wrong; a stale reference is far likelier than a broken engine.**
**FIX = regenerate, do not archaeologise.** `export_rnn_trace.py` now honours `RWKV_REF_DIR` (and
`verify_rust.py` too, matching the engine's `RWKV_TRACE_DIR`), so a fresh trace lands beside the old one
instead of clobbering it; `ref_metrics.json` now records the checkpoint + arch module actually exported
(the old code hardcoded "rwkv_ref_558.pth" regardless -- the very thing that made CLAUDE.md's
instruction wrong). The fresh A18 trace is `reference_a18/` and is SELF-CONTAINED at exactly 0.000e+00.
**RESULT -- the first-ever track-2 parity verification: `PARITY: PASS`, imm 0.000035 / ahead 0.000044
vs tol 0.0005** (14x and 11x inside). Procedure:
`RWKV_WEIGHTS=reference_a18/track2_a18.safetensors RWKV_TRACE_DIR=reference_a18 RWKV_STATE_CLAMP_TAU=300
RWKV_PRED_DIR=preds_a18v ./rust/rwkv-infer/target/release/rwkv-infer.exe` -> copy `preds_a18v/*` into
`reference_a18/` -> `RWKV_REF_DIR=reference_a18 python verify_rust.py`.
⚠ ~~Note max per-review |rust-python| = 9.6e-3 ... accumulated float divergence over a ~5,000-step
recurrence, not a formula error.~~ **THIS EXPLANATION WAS WRONG (corrected 2026-07-27).** It WAS a
formula error: `rust/rwkv-infer` had no `RWKV_ZERO_FEATURES` mask, so it consumed input columns the
Python model zeroes at its own input. With the mask implemented, iter 31's max per-review |diff|
falls from 1.59e-3 to **2.28e-6** over that same ~5,000-step recurrence — so accumulated float
divergence was never the story, and a large per-review spread should be read as a SIGNAL that the
two paths compute different formulas, not excused as float noise. Details + the measurement table
are in the deploy-contract section of CURRENT STATE.

**★ THE CHAMPION ITSELF IS NOW PARITY-VERIFIED TOO (iter 31, 2026-07-27): `PARITY: PASS`, imm
0.000008 / ahead 0.000001 vs tol 0.0005** -- 62x and 500x inside, TIGHTER than A18's, and max
per-review |rust-python| 1.59e-3 vs A18's 9.6e-3. This closed a real gap: every parity artifact
until now was A18's, so the **deploy contract's two newest pieces had never run end-to-end** --
**GRU N=3** (A18 is N=2) and **the PAVA rectifier with real learned powers** (the A18 run printed
`pava=no`). The engine loaded both from the checkpoint unaided: `head=gru3
pava=[-0.13663276, -1.517578, 0.00020639747]`. `gru_curves` and `pava_theta` are read from tensor
shapes (`model.rs:142,173`), so neither was hardcoded to A18's values.
Artifacts in `reference_iter31/` (same gitignore convention: only `ref_metrics.json` tracked).
Procedure identical to A18's, with `RWKV_CHAMP_CKPT=scratchpad/iter31_algo/iter31d_5586.pth
RWKV_CHAMP_SFT=iter31_algo.safetensors RWKV_REF_DIR=reference_iter31` on the export and
`RWKV_GRU_HEAD=3 RWKV_PAVA_LAMBDA=0.1` added to A18's env.
⚠ **`trace_selfcontained.py` HAD A BUG THAT MAKES IT LIE, FIXED 2026-07-27 -- if you used it before
this date, distrust the verdict.** It honoured `RWKV_CHAMP_CKPT` but **hardcoded `reference/`** for
the trace, so pointing it at a new trace compared the NEW checkpoint against the OLD June d=128
trace and printed `TRACE_NOT_SELF_CONTAINED` for a directory it never opened. It failed exactly
that way on BOTH fresh traces here (worst 1.0e-1 / 6.2e-2, plausible magnitudes, right shape) and
the tell was that the "stored" values were IDENTICAL across two different models. Doubly nasty
because this is the one tool whose entire job is detecting a stale reference. Post-fix both traces
are self-contained at exactly 0.000e+00, A18 reproducing its recorded verdict.


### superseded d=32 HISTORICAL CHAMPION block (H=2/K=16 on the 1500-user recipe) (moved out of CLAUDE.md 2026-08-10)

### HISTORICAL CHAMPION (SUPERSEDED -- the live champion is A18, see CURRENT STATE) = H=2/K=16 on the 1500-user data-variety recipe  (d=32, 2 heads x K=16; 193,724 params)
- arch `[1,4,3,3,3]` (card,deck,note,preset,user), d_model=32 split as **2 heads x 16 (K=16)** via the NEW
  K<32 CUDA kernel -- this HALVES the per-card WKV state (1088->576 floats; model_stats confirmed) at ~same
  params, ~half the WKV-kernel work, and **~1.16x faster GPU training (WS 1.182 vs 1.020 steps/s)**. Trained on
  users 1000-2499 (`train_db_sc8k_1500`), 1 epoch WS (3351 steps) + 0.27-epoch cosine decay (904 steps). ckpt
  `scratchpad/exp_h2k16/h2k16d_904.pth`; weights `reference/champ_h2k16.safetensors`. Recipe env = RWKV_N_HEADS=2
  RWKV_HEAD_DIM=16 + HP {peak_lr 1e-3, warmup 200, wd 0.01, clip 0.25} + RWKV_EMPTY_CACHE_EVERY=0.
- **fp32: ahead 0.309723 / imm 0.276566** (eval 101-200) -- accuracy PARITY with the prior H=1 champion
  (champ_1500d 0.309706/0.276357; both modes within 0.0002, far inside the +0.0015 efficiency budget), and BEATS
  the d=128 baseline by +0.0106 ahead / +0.0053 imm. Accepted as a **SIZE/SPEED win** (state halved + faster),
  NOT on the +0.0003 monotonic gate. HPs are re-tuned as part of the 5k phase (methodology d), not on 1500u.
- **★ KEY FINDINGS:** (1) DATA VARIETY beats repetition -- "1 epoch on ~1500 varied users" >> "15 epochs on
  100 users" (drove the prior champion jump; the d=32 model is DATA-limited, so the path forward is MORE DATA,
  scale toward 5k). (2) K<32 UNBLOCKED -- the WKV kernel is now K-dynamic (any K dividing 32), so H=2/K=16 gives
  the 2x-smaller-state + faster model that makes 5k-user training practical. PRIOR champions kept as refs:
  champ_1500d (H=1/K=32, 0.309706/0.276357), decay15 (100u, 0.314807/0.280200).
- **DEPLOY config (the sibling's FINAL locked recipe `q72u`, research CLOSED 2026-07-07; results ported
  here 2026-07-08) [[champion-logloss-deployed]]: 72 b/layer = 9-BYTE CARD, 27 B note, 256x compression.**
  Format per layer: m2b12L learnable shift catalog (2 chunks x 4096 entries, 48 b) + JOINT-UV b10 WKV
  catalog (per head ONE 10-bit code into a 1024-entry concat(u,v) 32-dim catalog, 20 b) + 1-bit norms (4 b).
  VAL penalty vs fp32 **+0.00114/+0.00021 (seed 1234) and +0.00115/+0.00040 (seed 4321)** — 2/2 seeds pass
  with margin; best-ever robustness (imm nbad 96-98/400); imm is ~seed-noise-FREE under joint coding.
  **Artifacts (ported to our `reference/`):** `qat_pq_q72u.safetensors` + `pq_cb_wkv_q72u.txt` +
  `pq_cb_shift_q72u.txt`. **Deploy env (Rust):** `RWKV_STATE_LOWRANK_SCOPE=card:1:int4,note:1:int4
  RWKV_QUANT_SHIFTS=1 RWKV_LOWRANK_PERCOL=1 RWKV_LOWRANK_PQ=reference/pq_cb_wkv_q72u.txt
  RWKV_SHIFT_PQ=reference/pq_cb_shift_q72u.txt RWKV_PQ_NORM_BITS=1`. **QAT recipe:** warm-start champion,
  2.0-ep plain QAT (no rotation/anneal/KD), BOTH cbs learnable (`RWKV_QAT_PQ_LEARN=1
  RWKV_QAT_SHIFT_PQ_LEARN=1`), `RWKV_QAT_NORM_BITS=1 RWKV_QAT_SHIFT_SCOPE=card:int3,note:int3`, NO_JIT.
  **The full engine (joint cb + warm search + norm quant) IS in OUR `rust/rwkv-infer` since `1d3b5b8`**
  (byte-identical champion eval verified from the parent build). Full detail: sibling
  `research_log_h2k16.md` + explainer `how_state_compression_works.md`.


### probe-insertion noise measurements (moved out of CLAUDE.md's file map 2026-08-10)

  **★★ REFINED 2026-07-27 BY THE MODE-3 CONTROL — THE NOISE IS CHANNEL- AND MODEL-DEPENDENT, AND
  ON `ahead` IT IS ZERO.** The bullet below generalized A18's imm measurement into a blanket "probe
  insertion costs ~3x the gate"; the direct control says otherwise. iter 31, n=500, `RWKV_EVAL_PAVA=3`
  (probes inserted, nothing substituted): **ahead noise = +0.000000 +/- 0.000014, p=0.33, worse on
  253/500 users** — an exact coin flip. Its imm noise is **+0.000056** (p=2.1e-8), itself 5x below
  A18's +0.000280 on identical probe machinery. So: (1) the duration decomposition was never
  confounded on ahead, which is now VERIFIED rather than hoped; (2) the "never compare rectified to
  unrectified at the 0.0001 gate" rule still stands for **imm**, but must not be quoted for ahead or
  as a fixed magnitude — **measure the control for the model in hand** (`RWKV_EVAL_PAVA=3`, ~30 min
  at n=500); (3) whatever damps it in iter 31 (state clamp? training WITH probes at
  `RWKV_PROBE_DENSITY=0.08`, which A18 lacked?) is untested and would be worth knowing.
  **★ PROBE INSERTION IS NOT NUMERICALLY FREE (measured 2026-07-26 — supersedes "imm must be
  bit-identical rect vs unrect, which proves the probes are non-perturbative").** The reasoning was
  right and the conclusion wrong: probes ARE skip rows, and the token shift does step over them
  (`prepare_batch` advances `last` only on non-skip rows), so in EXACT arithmetic imm is
  unchanged. But +4 rows per scored review inflates the batch ~30%, which re-buckets sequences by
  length and reorders bf16 reductions. Measured on A18, n=2500, `imm` (the channel the rectifier
  cannot reach, hence the clean probe): **mean +0.000280**, 2,425/2,500 users moved, median only
  +1.6e-5 but max 6.4e-3; **magnitude scales with recurrence length** (mean |d| 1.98e-4 at ~4.7k
  reviews/user -> 3.97e-4 at ~179k) and the bias grows with it too (62% -> 78% of users worse).
  One-signed because **LogLoss is CONVEX** — zero-mean noise on a prediction raises it.
  **Consequences:** (1) NEVER compare a rectified eval to an unrectified one at the 0.0001 gate —
  probe insertion alone costs ~3x the gate on imm; compare rect-to-rect only. (2) `mode2 - mode0`
  confounds duration-zeroing with this noise, which is why **`RWKV_EVAL_PAVA=3`** (probes inserted,
  nothing substituted) exists: `m3-m0` = noise, `m2-m3` = the clean duration cost, `m1-m2` = pooling.
  (3) A18 rectified scores ahead **0.302890 vs 0.299302 unrectified (+0.003588, worse on
  2080/2500)** — post-hoc rectification badly hurts a model never trained under the constraint,
  which is 10x the noise and so a real effect, not an artifact.

## HP TUNER (MAX=65536 era) -- the LIVE block archived from CLAUDE.md 2026-08-17

Launched 2026-07-30, completed, and became **iter 34**. Archived because CLAUDE.md still
presented it as `▶ LIVE ... RUNNING` weeks after it finished, while its own family
scoreboard said HP tuning was CLOSED. Verbatim below.

**▶ LIVE: THE HP TUNER IS REBUILT AND RUNNING** (launched 2026-07-30, detached pid 32352 via
`scratchpad/tuner65k/run_tuner_loop.cmd`; loop log `scratchpad/tuner65k/tuner_loop.log`, per-trial
`scratchpad/tuner65k/<name>.log`). Target = recover the -0.0003 that MAX=65536 cost.
`optimization/hp_tuner_5k.py` was rewritten wholesale (the old one targeted d=32 H=2/K=16,
MAX=110000, QUANT-AWARE, WS 2 epochs, eval 101-200 — every one of those wrong). What it does now:

  * recipe = **`scratchpad/maxval/run_maxval.cmd` with the HPs swapped** — the d=80 A18 trunk env,
    PLAIN (no QAT), WS 1 epoch, MAX=65536, NUM_FETCH_PROCESSES=2, the three speed flags during
    training and **cleared before eval**, RECTIFIED eval (`RWKV_EVAL_PAVA=1`) on **5001-6000**.
  * **★ THE BASELINE COST ZERO GPU:** `maxval` IS the default config, and restricting its existing
    rectified jsonls to 5001-6000 gives **ahead 0.299250 / imm 0.266335**, seeded into the journal.
    That subset also RANKS maxval-vs-iter-31 the same way the full VAL half does (+0.000113/+0.000309
    vs +0.000264/+0.000306), so it is a usable proxy — unlike the 200-user one that inverted.
  * **LEVER ORDER LEADS WITH THE LEARNING RATES**, because that is what the batch change implicates.
    Lever 1 is a joint **`lr_mult`** [1.0, 1.41, 2.0, 2.8] scaling **BOTH** `PEAK_LR` (1e-3, the
    AdamW group = 57,412 params) **and `RWKV_MUON_LR`** (0.02, the Muon groups = 500,800 params).
    ⚠ Tuning `peak_lr` alone would have moved only ~10% of the weights — Muon has its own base LR
    and the schedulers scale it proportionally (`train_rwkv.py:188-196`). Then `warmup_steps`
    [200,400,800], `muon_lr_mult` [1.0,0.5,2.0] (re-balance Muon vs AdamW after the joint move),
    `weight_decay` [0.01,0.05,0.1], `clip` [0.25,0.5], `decay_ratio` [0.25,0.4].
  * **11 non-default points x ~4.0 h = ~44 h** if nothing prunes — MEASURED on trial 1, not
    projected: **1.253 steps/s** steady state (5-min window past compile warmup), so WS 10,935
    steps = 2.42 h, decay 2,733 = 0.61 h, rectified eval on 1000 users ~1.0 h. Trials are named
    `t65_*`, trial dir `scratchpad/tuner65k/`, journal `optimization/tuner_5k_log.jsonl` (the old rows were archived to
    `tuner_5k_log_d32qat_era.jsonl` — different arch AND batch, not comparable).
  * **Val-based early pruning is ON** against **`optimization/tuner65k_vprune_ref.json`** (built from
    maxval's own val trajectory + its 5001-6000 finals = a matched reference on this exact trunk and
    batch). `RWKV_VPRUNE_MIN_STEP = max(1000, 2 x the trial's warmup)` so a long-warmup trial is not
    killed for being slow by construction. It matters most for the LR grid, where 2.8x can diverge.
  * **Three guards worth keeping in any future runner:** (1) a 40-step sanity phase that greps the
    sanity log for BOTH `BATCHED Newton-Schulz` and `[compile] torch.compile` — an env typo that
    silently disables a speed flag would cost ~2 h *per trial* across 11 trials; (2) stale-result
    deletion happens in **Python at trial-generation time, not in the `.cmd`**, so the `.cmd`'s
    **three eval attempts with NO `del` between them** keep `eval_sharded`'s resume property for the
    giant-user OOM; (3) the WS exit-code guard, because `write_decay_setup` takes the LATEST ckpt and
    would silently decay+evaluate a half-trained one.
  * **★ THE BAR, STATED CONCRETELY so trials are judged not eyeballed:** "recover what MAX=65536
    cost" means reaching **iter 31's numbers ON THE SAME 1000-user subset = ahead 0.299137 /
    imm 0.266026**. Against the seeded baseline (0.299250/0.266335) that is **+0.000113 ahead and
    +0.000309 imm** — note the two are NOT equal, because MAX hurt imm ~2.7x more than ahead on
    this subset. Anything beyond that bar is net new gain on top of the 1.68x speedup.
  * A sub-0.001 winner still needs **confirming on the full VAL half (5001-7500)** before it becomes
    the recipe — the subset is a ranking proxy, not a gate.

