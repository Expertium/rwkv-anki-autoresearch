# RWKV 5k phase — working notes (supplement to `research_5k.md`)

`research_5k.md` is the human-facing front (one results table only). All detail, reasoning, and
running notes live here. Pre-5k history: `research_log.md` / `log.md` / `HISTORY.md`.

**Front-table conventions:** LogLoss to 4 decimal places; parameter counts exact (e.g. 2,762,884 /
193,724, from `optimization/model_stats.py`); working precision here in the notes may be higher.
`provenance` = **adopted** (idea from literature / existing work) or **invented** (our own idea).
`summary` (rightmost) = pre-registered: ≤15 words, written BEFORE the result is known.
`logloss` = **exact** (training finished, real eval) or **estimated** (run Wilcoxon-pruned early; value =
`champ_final + (cand@s − champ@s)` at the prune step — see methodology pt 9).

## Methodology — governing rules for the 5k phase (Andrew 2026-07-01)
These are the accept/reject rules for every 5k experiment. Hard invariants (never change): the
hierarchy card→note→deck→preset→global, and the same preprocessed 92-dim inputs / existing LMDBs.

1. **Split + accept gate.** Train on one 5k half, eval on the other (train 1–5000 → eval 5001–10000;
   the old d=128 model already has weights → just eval it on 5001–10000, same eval set = fair). A change
   is **accepted only if it beats the current champion by ≥ 0.0003 in BOTH modes** — immediate (imm) AND
   forgetting-curve (ahead). Monotonic-both-modes champion.
2. **Param budget ≤ 225,000** (current champion 193,724 → ~31k headroom for experiments). Reducing params
   is welcome; reducing **both** LogLoss and params is the goal.
3. **Latitude.** Try own ideas and do literature searches freely.
4. **Quant-aware eval (NEW, central).** Every recorded LogLoss is measured **with (fake) card-state and note-state
   quantization applied** — the goal is to beat the old fp big model *while* being more efficient via
   quantization, not just to beat it. The old d=128 baseline stays fp (it is the target).
   **PORTED (2026-07-03): the sibling's fused fake-quant machinery is now in-repo** (see "Quantization
   port" section below). **ENV UPDATED 2026-07-08 to the sibling's FINAL locked recipe q72u** (72 b/layer
   = 9-byte card: joint-uv b10 WKV catalog + m2b12 shift catalog + 1-bit norms + int3 shift scope; 2-seed
   VAL +0.00114/+0.00021 and +0.00115/+0.00040; artifacts in `reference/*q72u*`):
   `RWKV_QAT_LOWRANK_SCOPE=card:1:int4,note:1:int4 RWKV_QAT_PQ=reference/pq_cb_wkv_q72u.txt
   RWKV_QAT_SHIFT_PQ=reference/pq_cb_shift_q72u.txt RWKV_QAT_SHIFT_SCOPE=card:int3,note:int3
   RWKV_QAT_NORM_BITS=1 RWKV_QAT_FUSED=1 RWKV_NO_JIT=1`. Codebooks run FIXED in 5k trials (the sibling's
   full recipe LEARNS them — enabling that here needs per-run cb-export→eval wiring, queued); NO_JIT until
   TorchScript is A/B-verified on the grafted q72u paths (once, at champion-run launch).
5. **State-size rules.** Card and note per-entity state sizes are **FIXED (cannot change).** Deck, preset,
   and global state **may grow** — they're cheap: deck/preset ~5–10×, global even up to ~100× is allowed
   (though unlikely to help much).
6. **Schedule + HP-tuning cadence.** WS = **2 epochs (fixed).** Decay epochs = WS × ratio, ratio ∈
   **[1/10, 1/2.5]** → decay ∈ **[0.2, 0.8] epochs**; the **decay phase is also quant-aware.** Add this
   decay-ratio as an HP-tuner hyperparameter (`optimization/hp_tuner_5k.py`). Do **HP tuning first** (after
   the batch-size sweep, point 8), then re-tune either after several small architectural changes accumulate
   **or** after a major change.
7. **Rust/CPU-deployable only (hard).** Every change must be reproducible in the Rust RNN inference engine
   on CPU (deployable in Anki). No GPU-only tricks in the shipped model.
8. **Batch-size / throughput sweep — do BEFORE HP tuning (Andrew 2026-07-02).** The 5k runs are slow, so
   first pick the fastest effective batch size: sweep **`MAX_TRAIN_GLOBAL_LEN`** (max total reviews packed
   per step = the WKV batch dimension) over ~100 steps each on the 5k train_db, recording steps/s (or
   reviews/s) and peak VRAM. Keep the largest that **almost maxes the 12 GB VRAM** (leave OOM headroom) —
   the champion at 66000 uses only ~6/12 GB, so there's room to grow. Fix batch size FIRST because it's
   structural and LR/warmup depend on it (why it precedes HP tuning). Do NOT go below 66000 (smaller drops
   data via `get_groups`); sweep UPWARD toward the VRAM ceiling. (This is the "bigger effective batch"
   headroom flagged in the SPEED notes.)
   → **DONE 2026-07-02: use `MAX_TRAIN_GLOBAL_LEN = 110000`.** Swept 66k/88k/110k/132k on train_db_sc8k_1500
   (H=2/K=16, free CPU, ~120 steps each via train_rwkv's `RWKV_MAX_STEPS` bench mode; tool
   `scratchpad/batch_sweep.py`). reviews/s: 66k=28,598 (5.90 GB) | 88k=34,928 (7.92 GB) |
   **110k=38,968 (9.44 GB, PEAK)** | 132k=29,397 (12.20 GB, -25%). KEY FINDING: throughput peaks just
   BELOW max-VRAM — 132k (~11.4 GiB) thrashes the allocator (worse throughput + OOM risk on long runs), so
   "almost max VRAM" overshoots; 110k (~3 GiB headroom) is the fastest safe batch (1.36x the 66k floor).
   (VRAM curve is CPU-load-independent; a CPU-contended re-run confirmed identical peaks, ~3x slower wall.)
9. **Wilcoxon early-pruning of doomed runs (Andrew 2026-07-02).** Revised run order: (1) eval the big old
   model on 5001–10000, (2) ONE champion-HP run recording per-step train logloss at EVERY WS step (ahead +
   imm; NOT the decay phase — its step count varies) → this run's eval numbers + trace become the 5k
   champion reference, (3) HP tune. Every later candidate runs with the champion trace loaded and, at every
   300n steps (300, 600, 900, …), computes a one-sided Wilcoxon signed-rank on per-step (candidate −
   champion) over ALL steps so far (growing window); **abort iff BOTH ahead and imm are worse at p < 1e-4**
   (strict α + both-modes ⇒ no false prunes — only abysmal runs die). Pairing is valid because the seeded
   epoch shuffle gives every run the same batch at the same step (same db + MAX + seeds).
   **Estimated final logloss for a pruned run** (goes in the front table, flagged `estimated`):
   `champ_final + (cand@s − champ@s)` at the prune step s, per mode. Worked example: champ final 0.3,
   champ@300 0.7, cand@300 0.75 → estimate 0.35. (The marker also records a mean-diff variant,
   `champ_final + mean(cand−champ)`, which is less single-batch-noisy — reference only.)
   **Implementation:** `train_rwkv.py` env-gated — `RWKV_STEP_TRACE=<path>` (write per-step WS trace),
   `RWKV_PRUNE_REF=optimization/champion_5k.json` (enable pruning), `RWKV_PRUNE_EVERY` (300),
   `RWKV_PRUNE_ALPHA` (1e-4), `RWKV_PRUNE_MIN_STEP` (0; raise past a longer warmup — a big-warmup HP trial
   is worse early BY CONSTRUCTION and could otherwise false-prune). Pruned run: writes
   `<trace>.pruned.json` (p-values + estimates) and exits with code 42.
   **Champion auto-update:** accepting a champion = run `optimization/promote_champion_5k.py --name X
   --trace <ws_trace.jsonl> --final-ahead A --final-imm I` — atomically replaces
   `optimization/champion_5k.json` (the prune reference every candidate loads) and archives the old
   champion's metadata to `champion_5k_history.jsonl`. No hand-editing of stored traces, ever.
   ⚠ Trace comparability requires identical data config (db, MAX_TRAIN_GLOBAL_LEN, seeds) — changing any
   of those invalidates step-pairing and needs a fresh champion trace run.
   **HP tuner integration (done 2026-07-03):** every `hp_tuner_5k.py` trial writes its own WS trace and
   auto-prunes when `champion_5k.json` exists (RWKV_PRUNE_MIN_STEP = 2× the trial's warmup, so
   warmup-heavy configs aren't false-pruned while still climbing). A pruned trial's `.cmd` skips
   decay/eval and runs `record-pruned` — the journal gets the ESTIMATED logloss flagged `"pruned": true`
   and coordinate descent proceeds on it. `status` marks such rows `PRUNED@step (estimated)`.

DONE (2026-07-01): the `decay_ratio` lever (range [1/10, 1/2.5]) is now in `hp_tuner_5k.py`. Still TODO
when the tuner is set up for 5k: repoint its data paths to the 5k train_db, set MAX_TRAIN_GLOBAL_LEN=110000
(batch sweep), recompute GROUPS_PER_EPOCH, and make WS/decay/eval apply fake card- AND note-state quant
(once the sibling's fast fake-quant kernel is copied).

## Setup
- **Train** users 1–5000; **eval** users 5001–10000 (disjoint held-out half).
- **Compute budget:** 2 WS epochs + decay = WS × decay_ratio (ratio ∈ [1/10, 1/2.5], default 0.25 → 0.5 decay ep; cosine).
- **Model:** H=2/K=16 champion (d=32, 2 heads × K=16, layers [1,4,3,3,3], 193,724 params, per-card
  WKV state = two 16×16 per-head matrices). Env: `RWKV_N_HEADS=2 RWKV_HEAD_DIM=16`,
  `RWKV_EMPTY_CACHE_EVERY=0`, `RWKV_DETERMINISTIC=1`, `RWKV_AUGMENT_SEED=1234`, HP from the tuner.

## Baseline to beat
The original leaderboard d=128 model `pretrain/RWKV_trained_on_101_4999.pth` (2.76M params, 4 heads ×
K=32), eval on 5001–10000 (genuine held-out — it trained on 101–4999). Eval via arch-swap
`scratchpad/architecture_old_d128.py` (copy over `rwkv/architecture.py`, eval, swap back), bf16 CUDA,
`get_result`, by-user-mean LogLoss. PENDING — needs the 5001–10000 eval data. Goal: our 194k model
trained on 1–5000 matches/beats it on the same set.

## HP tuning — tune on the FULL 5k, deferred (Andrew 2026-06-30)
Tune HPs on the full 5k (train 1–5000, 2 WS + 0.5 decay), NOT the 1500-proxy. Levers: peak_lr, warmup,
weight_decay, clip; WS epochs fixed at 2, decay fixed at 0.5. Champion HP anchor: 1e-3 / 200 / 0.01 / 0.25.
`optimization/hp_tuner_5k.py` is reusable — re-point its data paths to the 5k train_db, recompute
GROUPS_PER_EPOCH, tune-eval on a subset of 5001–10000.
- **FINDING (2026-06-30): 2 epochs on the 1500-proxy is WORSE than 1 epoch** — proxy baseline (champion
  HPs, 2 WS + 0.5 decay on 1500 users) = 0.318732 / 0.287316 vs the 1-epoch champion 0.309723 / 0.276566
  (+0.009 ahead / +0.011 imm). "Variety beats repetition": revisiting 1500 users twice overfits. And the
  proxy overfits MORE on 1500 than on 5000, so it understates the 2-epoch budget at true 5k scale → the
  proxy is not a faithful surrogate. (Tuner stopped after the baseline; resumable from trial 2.)

## Queued idea — data-driven initialization (Andrew 2026-07-02, do AFTER the 5k HP tune)
Goal: recycle the previous run's compute into a better initial point under the fixed 2-epoch budget.
Andrew's base proposal: record per-layer mean/SD of trained params; next run inits from seeded random
draws matching those moments. Assessment + upgrades (Claude):
- ⚠ Our init is NOT iid everywhere (`rwkv_model.py`): LoRA `A` + k/v-scale linears are DELIBERATE ZEROS
  (silent-start stability), decay bias is a DETERMINISTIC per-channel ramp (-7+5·(i/(C-1))^…), mixing
  matrices are uniform/orthogonal. Blind moment-matching clobbers all three. **Whitelist rule: only touch
  iid-random tensors; keep zeros zero and the ramp a ramp.**
- **Scheme A (preferred, arch unchanged): shrink-perturb** — init = λ·trained + (1−λ)·fresh, λ≈0.4–0.6,
  seeded (Ash & Adams 2020). Keeps solution structure (correlations, ramps, zeros blend correctly),
  restores plasticity; ideal under a fixed small budget. Probe λ ∈ {0.3, 0.5, 0.7}.
- **Scheme B (no direct weight reuse): per-tensor seeded PERMUTATION of trained values** (bootstrap-sample
  if shape changed) — matches the FULL empirical distribution incl. heavy tails, same cost as mean/SD,
  strictly better as a distribution matcher; still honest "from scratch".
- Record stats per tensor ROLE (e.g. "W_r, card stream"), not tensor identity → survives arch edits.
- **Protocol caveats:** an init change is itself a gated experiment; if ADOPTED it changes the protocol →
  re-run the champion under the same init before later ≥0.0003 comparisons. Warm-ish starts may shift
  optimal warmup down (fits the re-tune-after-changes cadence).

## Queued idea — warmup-only distillation from the d=128 teacher (Andrew 2026-07-03, do AFTER the 5k HP tune)
Andrew's proposal: during the first ~200–800 training steps, replace hard labels with the OLD d=128
net's predictions (soft targets carry more information than 0/1 labels); hard labels for the rest of
training so the student can SURPASS the teacher, not converge to it. Assessment + design (Claude):
- **Loss mapping is drop-in** (`srs_model.py::get_loss`): `label_y` (0/1 recall) → teacher's
  `curve_probs` at the same `label_elapsed_seconds` (BCEWithLogits accepts soft targets in [0,1]
  natively — feeds `curve_loss` + `curve_raw_loss` on ahead rows); `label_rating` → teacher's 4-way
  `out_p_probs` (torch CE accepts prob targets since 1.10 — feeds `p_loss` on query rows).
  Regularizer terms unchanged.
- **Teacher = `pretrain/RWKV_trained_on_101_4999.pth`** (the baseline-to-beat). No eval leakage: it
  never saw users 5001–10000. Its targets on 101–4999 are its own train set (overconfident-ish) —
  standard for KD, acceptable.
- **STORE predictions, don't run the teacher in-process** (Andrew's instinct is right): the arch config
  is module-level (`architecture.py`), so teacher+student can't coexist in one process — the d=128 arch
  works via file swap (like `run_base5k_eval.cmd`). Dump mode: run the SAME training data pipeline
  (same db/MAX/seeds → batch composition is deterministic; the Wilcoxon pairing already relies on this)
  with the old arch + no_grad for the first N steps, saving per-row (soft_y, p_probs[4]) fp16 per step
  → ~10 B/review ≈ 0.9 GB at N=800×MAX=110000, ~15 min GPU. Student loads step-indexed files for
  steps ≤ N.
- **Anneal, don't hard-switch:** loss targets = α(t)·teacher + (1−α(t))·hard, α linear 1→0 over the
  KD window (a step-800 cliff is a needless loss-landscape jump). Optional temperature T>1 on p_probs
  (probe later; T=1 first). Make the KD window its OWN knob (fixed step count), decoupled from the
  LR-warmup HP.
- **Gate fit:** accuracy-research change → ≥0.0003-both-modes gate. Training-only: params/state/inputs/
  hierarchy/deploy (methodology e) all unchanged. Batch composition unchanged → per-step Wilcoxon
  pairing stays valid (loss values differ, but pairing compares like-for-like steps... NOTE: early-window
  train-loss trace is against SOFT targets → the per-step prune comparison vs a hard-label champion
  trace is only meaningful AFTER the KD window; set RWKV_PRUNE_MIN_STEP > KD window).
- **Interaction warning:** this and data-driven init (above) both target early training — test
  SEPARATELY, then compose if both pass. Order after the HP tune per methodology (d).

## Queued analysis — irreducible-entropy (LogLoss floor) estimate (Andrew 2026-07-03, task #18)
How low can ANY algorithm go on this data? No assumption-free answer exists (single-draw Bernoulli
mixtures are non-identifiable beyond their mean — p*'s dispersion is invisible without structure), so:
- **Estimator:** cross-model residual covariance. y = p* + noise ⇒ for models with independent errors,
  E[(y−pA)(y−pB)] ≈ E[p*(1−p*)] = irreducible BRIER. We have the perfect pair: the two pretrained d=128
  models were trained on DISJOINT halves (101–4999 / 5000–10000), and **users 1–100 were seen by
  neither** → score both there (get_result RAW=true for per-review preds), average residual products.
  Residual error correlation biases it UP (same arch family) — report as an upper-leaning estimate.
  LogLoss floor then needs one parametric step: Beta-distributed p* within calibration bins (mean from
  calibration, variance from the covariance estimate) → implied E[H(p*)].
- **Baselines for scale (Andrew):** constant predictor at global mean retention → H(p̄) (~0.325 at
  p̄≈0.9), and by-user-mean of per-user H(p̄_u); plus both models' own LogLoss/Brier on the slice.
- **Context:** any model's loss upper-bounds the floor (best: 0.266 imm, 10k). A calibrated model's
  loss = mean entropy of its own predictions (Jensen gap to the floor = structure it blurs). A
  scaling-law asymptote across 100u/1500u/5k would bound the FAMILY floor — optional follow-up.
- **Deps:** test_db + equalize covering users 1–100 (build STEP4+5), d=128 arch swap, QAT env off.
  ~30 min GPU. Insight, not gating — run after the champion run / HP-tune kickoff.
- **★ RESULTS (run early 2026-07-03 — the OLD C: test_db already covered users 1–100).** By-user,
  100 users, 3.68M equalized reviews, mean retention 0.7966. IMM: const-global 0.4376 → const-per-user
  0.3781 → model A 0.2685 / model B 0.2684 → **floor estimate 0.2665 [CI 0.2416–0.2935]**. AHEAD:
  models 0.2992/0.2993, floor 0.2994 (≈ models). **Key finding: the estimator COLLAPSED in the most
  informative way — cross-model covariance (0.0950) ≈ each model's own Brier (0.0955), i.e. the two
  disjoint-trained models make ~the same errors (residual disagreement ~1% of Brier). The family is
  SATURATED: more same-family models/data won't move it; remaining error = true noise + SHARED blind
  spots (indistinguishable here). Floor is upper-leaning; true floor could be lower.** Artifacts:
  `optimization/entropy_floor.py`, raw preds `raw/RWKV{-P,}-floor{A,B}.jsonl`, `scratchpad/run_floor_est.cmd`.
  Side catch: get_result's RAW path had TWO dormant bugs (np-in-JSON; per-user lmdb re-open) — fixed;
  and the ORIGINAL C: test_db's reader lock table is FULL + held by an unidentified process (worked
  around via the `test_db_floor` copy; if it recurs: kill holder or copy data.mdb to a fresh env dir).

## Eval sharding (Andrew approved 2026-07-03) — 2-process full evals
`optimization/eval_sharded.py --config <eval toml>`: sizes all users from the test LMDB's
`{user}_batches` keys, LPT-splits them into 2 size-balanced shards (measured: 338,450,172 vs
338,450,387 — 215-review gap), launches 2 parallel `get_result` processes (3 fetch procs +
OMP_NUM_THREADS=3 each; QAT/arch env inherited), merges shard jsonls into the canonical result
files, prints by-user means. Numerics-IDENTICAL to a single-process eval (users are independent;
selection via the additive `USERS_FILE` key in get_result — absent = original behavior). Resume =
rerun (shards skip done users). Refuses to clobber existing canonical result files. Expect
~1.5–2x wall-clock. ⚠ d=32 evals only (two d=128s would OOM 12 GB); ⚠ E2E smoke still pending —
first champion-era eval should be watched (VRAM via nvidia-smi) before trusting it unattended.
Classic LPT-reordering within ONE process buys nothing (GPU processes users sequentially — total
= sum regardless of order); cross-user batch PACKING would be 2-4x more but shifts bf16 numerics
-> phase-boundary-only change, not adopted.

## Data prep — RUNNING since 2026-07-03 (6 threads, detached)
Launched after the sibling quant research finished (Andrew): `scratchpad/run_build_5k.cmd` detached
(WMI, Esc-proof), all six configs at `PROCESSES = 6`, log `scratchpad/build_5k.log`, ETA ~2–4 days.
Resumable — relaunch the same .cmd after any interruption. Scope: train + eval, BOTH halves.

DBs to build (eval DBs currently cover only ~users 1–200):
- `train_db(1-5000)` sc8k → **C:** (`train_db_5k_h1`, fast M.2, primary run reads every step)
- `train_db(5001-10000)` sc8k → **F:** (`F:/rwkv_lmdb/train_db_5k_h2`, 4 TB USB; C: can't hold both)
- `test_db` (whole-user) both halves → **F:** (`F:/rwkv_lmdb/test_db_5k`, users 1–10000)
- `label_filter` both halves → extends the canonical **C:** `label_filter_db` (FSRS-6 --short --secs)

Disk is NOT the constraint: C: ~455 GB free, F: ~1237 GB free; lmdb `map_size` is a SPARSE file on
Windows (500 GB map → 0 GB actual until written) — monitor FREE space, not logical file size. train_db
~51 MB/user → ~255 GB/half.

TIME is the constraint (why 1 thread was rejected). Smoke rates: find_equalize ~0.42 ms/review, test_db
~0.32, train_db ~0.6–0.8; dataset ~745M reviews → at 1 thread full both-halves ~13 days / primary-only
~6 days; at 4–6 threads ~2–4 days.

Ready-to-run infra (just bump threads then launch):
- 6 configs in `rwkv/`: `find_equalize_5k_{h2,h1}.toml`, `data_processing_test_5k_{h2,h1}.toml`,
  `data_processing_train_5k_{h1,h2}.toml`. All have `PROCESSES = 1` → change to 4–6 before launching.
- Driver `scratchpad/run_build_5k.cmd`: 6 builds sequentially, RESUMABLE (skips done users),
  continue-on-error, logs to `scratchpad/build_5k.log`. Order front-loads the 5001–10000 eval data
  (steps 1–2) so the d=128 baseline eval can run while `train_db(1-5000)` builds.
- Launch detached (survives Esc): `powershell -NoProfile -File scratchpad/detach.ps1 -Script
  C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\run_build_5k.cmd`; monitor via OS truth (tail the
  log + FREE space on C:/F: + python PID). Smoke confirmed: configs parse, find_equalize runs, F:+C: writes work.

## Quantization port (2026-07-03) — the sibling's locked recipe + fused kernels are IN-REPO
Ported from `C:\Users\Andrew\rwkv-state-quant` (research DONE; its final log = `research_log_h2k16.md`):
- **LOCKED deploy recipe @ ~352 b/card:** per WKV 16×16 head-matrix, rank-1 factors (power-iteration,
  split-√σ, sign-canon) PQ-encoded with the fixed global codebook **`reference/pq_cb_m2b8.txt`** (2×dim-8
  sub-vectors, 256 centroids) ≈ 96 b/layer + **int4 token-shifts** ≈ 256 b → card ≈ 352 b, note ≈ 1056 b.
  Deploy env (Rust): `RWKV_STATE_LOWRANK_SCOPE=card:1:int4,note:1:int4 RWKV_LOWRANK_PQ=<codebook>
  RWKV_QUANT_SHIFTS=1 RWKV_LOWRANK_PERCOL=1`.
- **QAT result to beat carried over:** e150_pq (1.5-ep QAT on the h2k16 champion) = VAL **+0.0010 imm /
  −0.0003 ahead** vs fp32 — compressed BEATS fp32 on ahead. Weights `reference/qat_pq_ep150.safetensors`
  (local, gitignored). Key finding: epochs monotone (PQ acts as a regularizer); LR/WD/clip/EMA/co-adapt dead.
- **What was ported:** fused QAT CUDA kernels (`rwkv7_wkv_qat_{forward,backward}` full-matrix int-N;
  `rwkv7_wkv_qat_lr_*` + `qat_lr_rank1` rank-1 low-rank with PQ branch; `rwkv7_set_pq_codebook`; 150–490×
  over the Python loop) in `rwkv7_cuda.cu`/`rwkv7.cpp`; `rwkv_ops.py` autograd wrappers + `_sanitize_state`
  + `maybe_upload_pq_codebook`; `rwkv_model.py` shift-QAT (`fake_quant_shift`, JIT-annotated here — the
  sibling ran NO_JIT); `architecture.py` int3 + `RWKV_QAT_SHIFT_SCOPE`; `train_rwkv.py` **LR- and
  WD-clobber fixes** (optim load restores saved lr/initial_lr/weight_decay, silently overriding config/env
  — now reset after load) + non-finite loss/grad-norm guards. Training env for QAT:
  `RWKV_QAT_LOWRANK_SCOPE=card:1:int4,note:1:int4 RWKV_QAT_PQ=reference/pq_cb_m2b8.txt RWKV_QAT_FUSED=1`.
- **Validated in our tree (2026-07-03):** plain WKV path bit-exact vs golden (QAT additions untouched it);
  PQ parity CUDA-vs-Python-deploy max REL 3.2e-07 (== sibling's number); int-N low-rank parity 7.5e-04
  (== sibling's); 25-step end-to-end QAT smoke from the champion ckpt+optim — all env prints + clobber-fix
  resets fired, losses sane. Parity harness kept in `scratchpad/qat_parity/`. Recipe provenance toml:
  `optimization/qat_pq_ep150_recipe.toml`.

## Speedups banked (detail also in CLAUDE.md)
- 2026-07-01 **Tier 1 DEPLOYED in-place** — production `rwkv/model/RWKV_CUDA.cp312-win_amd64.pyd` is
  byte-identical (SHA256) to the bit-exact-validated build (cudaMalloc/cudaFree → caching-allocator
  scan scratch; ~1.3–1.44× WKV microbench). Real-world WS steps/s A/B deferred to the next training run.
- 2026-07-01 **Tensor cores profiled + KILLED** (`scratchpad/prof_wkv.py`). Only matmuls (scan) are
  ≤1.1% of WKV GPU time, 0.74% at B16×T30000; the other ~96% is per-timestep matrix-VECTOR warp-shuffle
  recurrence (backward `final` ~61%, fwd `final`/`base` ~12/11%, bwd `base` ~11%). Amdahl ceiling <1% →
  cheap tensor-core win DEAD. Only path to TCs = from-scratch chunked-matmul (fla delta-rule) rewrite of
  the recurrence — multi-day + parity-risky (±0.0005 gate; K=16 underfills TC tiles). Revisit only if 5k
  proves painfully slow.
- 2026-07-03 **Real-step re-profile at the 5k regime** (new `RWKV_PROFILE_STEP`/`RWKV_PROFILE_COUNT` env
  hook in train_rwkv; H=2/K=16, MAX=110000, train_db_sc8k_1500): plain step = **578 ms GPU** — elementwise
  "other" 78%, WKV recurrence 18%, gemm 5%. The WKV floor is no longer dominant → the **chunked-matmul
  rewrite is DEAD as a priority** (would address ≤18% of the step).
- 2026-07-03 **QAT kernel speedup — 37× on the qat_lr share, 6.3× on the quant-aware step (bit-exact).**
  The methodology-(a) quant-aware forward was **7.1× slower than plain** (4,122 ms/step, **86.8%** in
  `rwkv7_wkv_qat_lr_{forward,backward}` — every 5k run would have been ~30–40 h instead of ~6–7 h; the
  batch sweep's 38,968 rev/s was measured WITHOUT the QAT env, so the plan's time budget was blind to
  this). Root causes inside `qat_lr_rank1` (per timestep!): single-threaded PQ codebook search (~8k serial
  FMAs on tid 0 while 255 threads idle), ~6 block barriers × ≤64 power iterations, and the whole
  truncation computed-then-DISCARDED on skip (query) rows ≈ half of all rows. Fixes (all bit-exact by
  construction): skip-step elision (block-uniform branch), block-parallel PQ search (identical per-distance
  FMA order + first-strict-min (dist,index) reduction), warp-0-scoped power loop (`__syncwarp`). Verified:
  32-tensor fwd+bwd golden BITEXACT_PASS (int-N + PQ, short-T/many-B + multi-chunk long-T), deploy PQ
  parity re-run max REL 3.2e-07. After: QAT share 3,577→96 ms/step, full step 4,122→**651 ms** = quant-aware
  costs **~13%** over plain. Goldens + harness: `scratchpad/qat_speed/golden_gen.py`.
- 2026-07-03 **Deterministic-indexing speedup — 1.5× on the plain step, BIT-EXACT.** A/B profiling showed
  `RWKV_DETERMINISTIC=1` cost **251 ms of the 578 ms step (43%)** — all in sort-based deterministic
  `index_add`/`indexing_backward` from two gather sites. Fix 1: **PermGather** (`srs_model.py`) — the
  hierarchical stream gather references each row at most once (permutation + `-1` pads), so its backward is
  an index_select by the runtime-built inverse permutation (collision-free scatter, deterministic by
  construction) instead of stock index_add; escape hatch `RWKV_PERM_GATHER=0`. Fix 2: **flat-row time-shift
  gather** (`rwkv_model.py::time_shift_gather`) — `gather(x,1,sel.expand(C))` → `index_select` on flattened
  rows: the deterministic backward sorts B·T keys instead of B·T·C elements and row-adds over C. BOTH
  verified by 10-step E2E training traces bit-identical to the pre-change path (fwd+bwd+optimizer chain).
  Plain det step 578→**384 ms** (det tax now ~57 ms vs the 327 ms non-det floor). **Stacked with the QAT
  fix, the full quant-aware deterministic step = 4,122→450 ms (9.2×); a 5k champion run ≈ 4–5 h.**
- 2026-07-03 **zeros_like→empty_like for the 24 WKV backward grad buffers** — validated bit-exact (goldens
  + 10-step E2E trace; the kernels fully write every slot, incl. the explicit t=0 zeroing for a/kd), but
  measured **≈neutral** (450.0→449.2 ms; only the fp32 w_grad fill vanished). Kept as strictly-less-work.
  LESSON: the 4% bf16 FillFunctor mass is NOT the WKV grad zeros — it's spread through autograd/model
  plumbing. **The speedup hunt has hit the flat tail**: remaining step = 250 ms elementwise mass in dozens
  of small kernels (norms ~8%, residual det-indexing ~6%, fills ~4%, pageable HtoD ~2%).
- 2026-07-03 **torch.compile investigated end-to-end and SHELVED (honest 1.05×).** Andrew caught the stale
  "Windows-blocked" claim — triton-windows 3.7.1 is installed and inductor works. Full trail: (1)
  whole-`get_loss` compile hits Python 3.12's FIXED per-thread C-recursion cap inside Dynamo (immune to
  setrecursionlimit AND to a 64 MB thread stack — `scratchpad/train_bigstack.py`); the RecursionErrors were
  swallowed by the NaN-safety except → HOLLOW steps → a fake 303 ms/step (1.27×) profile and fake
  determinism failures (runs "diverged" because each skipped different steps). (2) Mixer-scoped compile
  (RWKV7TimeMixer/ChannelMixer forwards only) traces cleanly: 0 exceptions, run-to-run determinism PASS,
  honest profile **365 ms vs 384 ms JIT = 1.05×** (elementwise 254→234 ms; WKV/QAT untouchable custom ops).
  5% doesn't buy the costs: NO_JIT mode switch, minutes of compile warmup per run, recompile-storm risk
  across full-epoch shape diversity, numerics break vs the JIT path. Plumbing kept for a future revisit:
  `RWKV_COMPILE=1` (requires RWKV_NO_JIT=1) + inductor determinism knobs in train_rwkv + the big-stack
  launcher. LESSONS: always count "Exception caught" in any NO_JIT/compile run before trusting its
  numbers; eager NO_JIT is run-to-run deterministic (control-verified).
- 2026-07-03 **QAT power-iteration warm-start considered and REJECTED**: warm-starting u across timesteps
  would cut the ≤64-iteration power loop (~2× on the 96 ms QAT share ≈ 11% of step) but breaks the
  train==deploy EXACTNESS of the fake-quant (deploy cold-starts per save) — the guarantee the sibling's
  research was built on. Not worth it at 11%.
- **Remaining honest unknown (post-build clean window): the wall-clock gap.** GPU-busy is 449 ms/step
  (quant-aware) but wall step time under the batch sweep implied ~2+ s — Python/TorchScript-interpreter
  gaps between kernels are unmeasurable under build contention. Measure GPU-idle fraction in a clean
  window; if large, host-side batching of the per-split loop is the next (and last) lever.
