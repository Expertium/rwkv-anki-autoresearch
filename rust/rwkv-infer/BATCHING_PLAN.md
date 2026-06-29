# Batching plan (agenda item b) — (B,C) inference over N independent cards

## ★ STATUS: IMPLEMENTED + VALIDATED (2026-06-29)
Batched single-step QUERY forward landed as SEPARATE `*_batched` fns in `model.rs` (B=1 path
untouched). New fns: `time_mixer_batched`, `channel_mixer_batched`, `run_stream_batched`,
`review_batched`, `imm_prob_batched`, free `single_timestep_batched`/`group_norm_batched`/
`quant_roundtrip_batched` (per-card amax), `stack_stream_states`. main.rs: `warmup` (sequential
replay) + `gather_batch` + modes `--verify-batched`, `--bench-batched`, `--sweep-batched`,
`--bench-synth` (synthetic states, no warmup -> drives the RAM/speed sweep).
- **Parity PASS:** `verify_rust.py` still bit-exact (B=1 untouched; LL diff 0.000000, max pred diff
  7.76e-7). `--verify-batched` on users 107/136/156 (B=738..1014 real cards): batched imm == B=1 imm
  within **1.8e-7** (float reduction-order noise).
- **Speed-vs-RAM frontier** (iter36, CPU, synthetic states; `scratchpad/sweep_pareto.py` ->
  `scratchpad/pareto_speed_ram.png` + `pareto_data.csv`): throughput climbs near-FREE in RAM up to a
  **knee at B=128 (~11,300 rev/s, 29 MB)** = ~20x the batched-B=1 (~550 rev/s) and ~40x the ~280 rev/s
  true B=1 single-query path. Past the knee the frontier BACK-BENDS: B=256 ~8,500, B=512 ~8,200,
  B=2048 ~7,300 rev/s at 44/78/283 MB -> STRICTLY DOMINATED by B=128. Best rev/s-per-MB = B=32.
- **Thread count is ~IRRELEVANT** (`scratchpad/thread_sweep.py` -> `thread_sweep.png`): swept
  RAYON_NUM_THREADS 1..32 x B 16..2048 -- at every B the 6 lines overlap within noise (B=128 ranges
  11,197-11,694; threads=1 gives 11,315). The tiny K=32 matmuls don't parallelize, so the back-bend
  is an INTRINSIC CACHE bound (sharp cliff B=128->256 = L2/L3 working-set spill), NOT oversubscription
  (my earlier "1 thread beats 32 at B=512" was a single-3s-run artifact). Deploy can run single-thread
  and lose nothing -- frees cores for the rest of Anki.
- **RECOMMENDATION:** batch ~128, single-threaded is fine. NOT the fork's default 512 (past the cache
  knee). B=128 is robust across thread counts; the apex may still shift with a different CPU's cache
  size -- re-sweep on the deploy target if it matters. State-quant payoff (16.7->3.6 GB) is orthogonal
  (persisted state, not this transient working set).
- TODO (optional): CPU-freq-locked clean re-run for publication numbers; wire `RWKV_STATE_QUANT_SCOPE`
  into the batched bench to confirm quant is still ~free batched.

## ★ CORRECTION (from the JSchoreels/anki fork, confirmed 2026-06-28)
The deploy batch is WITHIN ONE USER, and the hot path is a SINGLE-STEP query, not a replay:
- **Warmup (once at startup):** replay the user's full revlog SEQUENTIALLY per card history to build
  the warmed recurrent state. Order-preserving, inherently sequential. NOT the hot path.
- **Queue scoring (HOT PATH):** batch B cards (deck option `rwkv_review_batch_size`, default 512) of
  ONE user; each does ONE forward step from its CURRENT warmed state -> returns retrievability +
  optional Good-interval override. Independent, READ-ONLY, run on Rayon lanes today (each B=1).
- **Answer:** updates that card's recurrent state sequentially (one review = one step).

So the throughput win to build = a **batched single-step (B,C) query forward**: gather B cards'
(card,note,deck,preset) states + the shared global, run the 5 chained streams batched -> B retriev-
abilities. **NO variable-length lockstep/masking needed** (every card does exactly one step) -> far
simpler than the original plan below. The lockstep-replay design below is only relevant if we ever
batch the WARMUP (lower value; it's a one-time cost). PRIORITISE the single-step query batch.
Ties to state-quant: queue scoring LOADS the persisted (quantized) states -> dequant for the (B,C)
forward; the 16.7->3.6 GB memory win is realised exactly here. Real (B,C) batching should beat the
fork's current Rayon-over-B=1 (amortises candle per-op dispatch; same oversubscription that slowed
weight-quant). Bench vs the Rayon approach.

## ★ PRECISE SCOPING (from model.rs, 2026-06-28) — implement as SEPARATE batched fns (keep B=1 untouched -> zero parity risk)
Add `*_batched` fns alongside the B=1 ones; batched state = {t_xshift (B,C), t_state (B,H,K,K), c_xshift (B,C)}.
- **features2card**: already works on (B,92)->(B,C) (pure lin/silu/ln on last dim). No change.
- **single_timestep_batched(H,K, r/k/v/w/a/kd:(B,H,K), s_prev:(B,H,K,K))**: col=reshape(B,H,K,1), row=(B,H,1,K);
  candle matmul broadcasts leading (B,H). decay=s_prev*row(w); sk=s_prev@col(kd); s=decay - sk@row(a*kd);
  s+=col(v)@row(k); out=(s@col(r)).reshape(B,H,K). Mechanical.
- **time_mixer_batched**: in_x (B,C). diff=(B,C). LERP fusion: lerp_w(8,C)->(1,8,C), x/diff->(B,1,C),
  all_inp=(B,8,C), inp(i)=narrow(dim1,i,1)->(B,C). to_hk: reshape (B,H,K). k_scale/v_scale (B,H)->(B,H,1).
  out_hk->(B,C). bonus: r_h(B,H,K)*bonus_p(H,K) broadcast; sum_keepdim(-1)->(B,H,1); *v_h->(B,H,K)->(B,C).
- **channel_mixer_batched**: lerp_k(1,C) broadcasts over (B,C); rest is lin on (B,C). Trivial.
- **HELPERS to verify handle a leading B dim**: `l2norm_heads` (norm over last dim K -> fine on (B,H,K)?),
  `group_norm` (acts per-(H groups) on (B,C) -> CHECK it reshapes by B*H not just H), `lerp`, broadcast_* shapes.
- **review_batched (QUERY only)**: feats(B,92) -> features2card -> chain 5 streams (run_stream_batched, READ-ONLY,
  discard new state) -> heads (lin/softmax already batch-fine) -> out_p_logits (B,4) -> (B,) imm prob.
- **main.rs bench**: after a user's warmup replay, gather B cards' current per-stream states -> stack to (B,...);
  run review_batched once = B predictions; loop T s; count B*iters/s. Compare vs B=1 Rayon. + assert batched==B=1.
- **VALIDATION**: (1) verify_rust.py still bit-exact (B=1 path untouched). (2) new: batched B over a ref user's
  cards == per-card B=1 imm prob (max diff ~f32 eps).

---
## (original plan — lockstep replay, only needed for batching the WARMUP)

**Goal (Andrew):** Anki runs the scheduler in BULK (JSchoreels/anki fork) — predict many cards at once.
Batching N independent card-streams amortizes candle's per-op dispatch overhead (the B=1 bottleneck:
tiny (1,C) tensors). Bigger (B,C) matmuls = the real bulk-throughput win. MUST keep B=1 parity bit-exact.

## What changes (model.rs) — mechanically tractable, all ops broadcast over a leading B dim
- **LayerState**: `t_xshift (1,C)->(B,C)`, `t_state (H,K,K)->(B,H,K,K)`, `c_xshift (1,C)->(B,C)`.
- **time_mixer**: add B leading dim. Already-fine ops: `ln`/`group_norm` (act on last dim), `lin`
  ((B,C)@(C,out)=(B,out)), `l2norm_heads` (norm over last dim). Fix-ups:
  - lerp fusion: `lerp_w (8,C)` -> `(1,8,C)`; `diff (B,C)` -> `(B,1,C)`; `all_inp=(B,8,C)`; `inp(i)=narrow(B,C)`.
  - reshape-to-heads: `(B,C)->(B,H,K)`; scales `(B,H)`; bonus `(B,H,K)`.
- **single_timestep**: add B dim; `col/row` reshape to `(B,H,K,1)`/`(B,H,1,K)`; candle does batched matmul
  over leading (B,H) dims. `s_prev (B,H,K,K)`. Pure mechanical.
- **channel_mixer**: same (B,C) treatment.
- **review()**: `features2card (B,92)->(B,C)`; chain 5 streams; heads -> `(B,num_points)`,`(B,num_curves)`,`(B,4)`.

## The hard part: variable-length replay (main.rs)
Each card has its OWN full history (different lengths) + its own per-review elapsed times. To batch the
REPLAY (where the compute is): step B cards in LOCKSTEP over global t; at step t, cards with len>t are
ACTIVE, others MASKED (freeze their state: `s_next = mask ? s_new : s_prev`, mask=(B,1,1,1)). Predict
(forgetting_curve/interp) per card with `(B,)` elapsed vectors. Skip-logic (the `skip_BT` reviews) already
per-step -> per-card boolean mask.

## Throughput bench
Extend the config-driven bench to load B card traces, run them lockstep-batched, count total reviews/s.
Compare vs B=1 via the paired Wilcoxon (20 trials). Expect big win if candle vectorizes (B,C) matmuls.

## Validation
1. B=1 path must stay BIT-EXACT (verify_rust.py unchanged).
2. New: batched B=N over the 3 ref users must match the B=1 per-card predictions (max diff ~f32 eps).
Keep B=1 as a special case (B=1 of the batched path) OR a separate fast path — measure both.

## Risk / notes
- candle broadcast_* requires explicit dims; watch (B,1,C) vs (1,8,C) shapes.
- masking must freeze ALL of: t_state, t_xshift, c_xshift for finished cards.
- the lerp `start=xshift` init (None case) uses x itself per-card -> init xshift=(B,C)=x at t=0.
