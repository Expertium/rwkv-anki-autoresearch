# Iter 23 build notes — learnable PAVA rectifier + in-sequence probe rows (2026-07-17)

Design deltas fixed tonight, on top of `optimization/MONOTONICITY_PLAN.md` stage 2 (the
spec of record). Implementation follows these exactly.

## Probe-row insertion (prepare-batch time, `rwkv/prepare_batch.py`)

- **Where**: inside `prepare()` — a new probe-insertion pass over each `RWKVSample`
  BEFORE `add_encodings`/packing, gated by `probe_density > 0` (param; callers pass the
  env value only on the TRAIN path — `prepare_data_train_test` "validate" groups and all
  eval paths pass 0 → validation/eval stay probe-free, val traces stay comparable to the
  champion's, benchmark eval untouched).
- **Eligible target rows**: `skips == False & has_label == 1 & is_query == 0` AND NOT the
  in-chunk first occurrence of their `card_id` (np.unique(return_index) on ids["card_id"])
  — guarantees every probe reads a warmed same-card state (chunks are cold-started, so
  global is_first_review is the wrong filter; in-chunk-first is the packing-relevant one.
  Note: a skip row at a subsequence start would actually self-shift benignly — the
  `assert (skip_arr[0]==False).any()` in prepare() is per-BLOCK, not per-slot — but
  warmed-state probes are the design intent anyway).
- **Selection**: `numpy.random.default_rng(hash((seed, user_id, start_th)) & 0x7fffffff)`
  with seed = the prepare fixed_seed (RWKV_AUGMENT_SEED) — deterministic per chunk, stable
  across runs/epochs at fixed augment seed. Density env `RWKV_PROBE_DENSITY` (e.g. 0.08).
- **4 probe rows per target** (ratings 1..4, Again..Easy order), inserted immediately
  BEFORE the target row (order among the co-located skip rows — the existing query row +
  the 4 probes — is irrelevant: skip rows are mutually invisible; token-shift reads the
  last NON-skip row, each skip row advances from the committed state).
- **Probe feature row** = copy of target's 92-dim row, then: grade one-hot cols 9:13 =
  one_hot(r); scaled_duration col 8 = scale_duration(median) constant (see
  duration_median.json); everything else real (elapsed/context/IDs). Card-state col 22
  left as-is (champion recipe RWKV_ZERO_FEATURES=22 zeroes it; deploy feeds 0). is_query
  col 23 stays 0 (probes carry outcome features — they are NOT queries). Column indices
  asserted against CARD_FEATURE_COLUMNS by name at import.
- **Probe labels**: `has_label = 0` (CRITICAL — keeps probes out of ahead_mask =
  (1-is_query)*has_label and every standard loss/metric); label_elapsed_seconds = the
  TARGET's label_elapsed_seconds (so _get_loss's per-row curve evaluation lands at the
  right t for free); label_review_th = -1; skip = True.
- **ids / day_offsets / day_offsets_first / review_ths**: copies of the target row's.
- **Repack**: after insertion, recompute each stream's ModuleData exactly like
  `data_processing.create_sample` does (group row indices by entity id in row order,
  bucket by exact group length → split_len/split_B/from_perm/to_perm) — replicated in
  numpy inside prepare_batch (probe-ON path only; probe-OFF path byte-identical, uses the
  stored ModuleData untouched).
- **New PreparedBatch fields** (empty tensors when off): `probe_rows` (M,4) long — flat
  b*global_T+t indices of the 4 probes in Again..Easy order; `probe_target` (M,) long —
  flat index of the target row; `probe_pressed` (M,) long in 0..3 (target's actual
  rating-1). Extend PreparedBatch.to(device).

## Rectifier + loss (`rwkv/model/srs_model.py`)

- 3 learnable powers: root Parameter `pava_theta` (3,), created when RWKV_PAVA_LAMBDA>0;
  p_j = 2*tanh(theta_j), init theta = atanh(0.5) → p = 1 (classic PAVA). Junction j=0:
  Again-Hard, 1: Hard-Good, 2: Good-Easy. Root-Parameter + jit.ignore accessor pattern
  (grup precedent). Name lacks "weight"/2D → falls into other_params (wd=0) — correct.
- **Pooling op**: exact sequential PAVA with pooling-to-tie, non-DEcreasing target order
  (P_Again ≤ P_Hard ≤ P_Good ≤ P_Easy), pooled value = weighted power mean
  M_p(a,b;wa,wb) = ((wa*a^p + wb*b^p)/(wa+wb))^(1/p) at the JUNCTION's power (junction =
  block boundary, uniquely defined); numerics via exp((1/p)*logsumexp(p*log x + log w) -
  log(sum w)) with geometric-mean switch at |p| < 1e-3. Iter 23 weights = block sizes
  (counts); iter 24 swaps in p-head button probabilities (same op).
- **Vectorized over (M,4)** with torch.where mask simulation of the unrolled n=4
  algorithm (junctions left→right, back-merge checks after each merge; per-slot block
  value/weight/left-pointer tensors). MUST be property-tested against a scalar reference
  implementation (random inputs: vectorized == scalar; p=1 == classic arithmetic PAVA;
  identity on already-ordered inputs; output non-decreasing; finite grads).
- **Loss**: p4 = curve_probs.view(-1)[probe_rows] → rectified → pressed =
  rectified[arange(M), probe_pressed]; BCE(pressed, label_y.view(-1)[probe_target]);
  loss_avg += RWKV_PAVA_LAMBDA * mean. Lambda as instance float (pbin_scale pattern,
  TorchScript reads instance attrs). Counterfactual probes get gradient only through
  pooling. Real rows' losses untouched.
- Stats: append `pava_loss_avg` + `pava_pool_frac` (fraction of probe sets with ≥1
  violation) to SrsRWKVIterStatistics WITH explicit construction in _get_loss (NamedTuple
  under TorchScript — no defaults; _get_loss is the only constructor).

## Smoke plan (scratchpad/iter23_pava/)

1. **Invisibility (crown jewel)**: same weights, same chunk, prepare with density 0 vs
   0.3 → every REAL row's curve_probs/p_logits bit-identical (proves skip-commit masking +
   token-shift invisibility end-to-end through the packed batch path).
2. Pooling unit tests (see above) incl. grad flow into pava_theta.
3. Probe-row content check: inserted rows differ from target ONLY in cols 8, 9:13 (+
   labels/skip); ids equal; probe reads warmed state (target not in-chunk-first).
4. E2E: 2-user tiny CPU get_loss with density 1.0, JIT + NO_JIT; loss finite; backward.
5. Off-path byte-identity: density 0 → PreparedBatch identical to pre-change (golden).

## Env summary (iter 23 run)

RWKV_PAVA_LAMBDA (weight, >0 enables params+loss), RWKV_PROBE_DENSITY (train-only
insertion), + champion recipe (ZERO_FEATURES=22, NO_AHEAD_RESIDUAL=1, H=2/K=16, MAX=110000).
Iter 24 = same + RWKV_PAVA_PWEIGHT=1 (p-head probability weights in the pooling mean).

## Open items

- duration_median.json (computing 2026-07-17 ~01:15, users 1-5000 exact histogram).
- Gate/reference: iter 23 gates vs the no-residual reference (iter 22 verdict = Andrew's
  re-baseline call, ~11:45); vprune ref likewise = iter 22's val trace if re-baselined.
- Deploy: the rectifier IS the deploy button projection (Rust port + parity vectors when
  a rectifier model ships — MONOTONICITY_PLAN stage 3 remnant).
