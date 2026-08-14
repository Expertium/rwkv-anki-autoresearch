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

**⚠ VAL/TEST SPLIT AMENDMENT (Andrew 2026-07-21, both tracks, effective from iter 29 / post-A8):**
the 5001–10000 eval half is split into **VAL = users 5001–7500** and **TEST = 7501–10000**.
- **All accept/reject decisions run on VAL only** (candidates eval 5001–7500, n=2500; the delta
  bars and p<0.0001 gates are unchanged — expect ~1.4× noisier paired SEs than at n=5000).
  Candidates pair vs the champion's existing jsonls via `paired_pvalue --intersect` (the
  intersection IS the val set when the champion file is full-range) — no champion re-evals needed.
- **TEST is touched only at each track's close** — one eval of the final champion (plus the
  planned 2-ep confirmation + QAT runs) for the honest numbers; never for decisions. Rationale:
  dozens of adaptive accept/reject decisions against one fixed set overfit to it; the held-out
  test half prices that honestly.
- Already-inside-val, so unchanged: training-time validation users 5001–5010 (vprune refs stay
  valid), hp-tuner tune-eval 5001–6000. Baselines (old d=128 .pth, FSRS-7 refs) have full-range
  jsonls → subset per half as needed, no new GPU work.
- Records: research_5k.md tables stay as-is (val-era rows marked; ≤ iter 28 / ≤ A8 were full
  n=5000); one small "final test numbers" table per track gets filled at close.
- Bonus: eval wall-clock halves (~3 h → ~1.5 h at d=128).

1. **Split + accept gate.** Train on one 5k half, eval on the other (train 1–5000 → eval 5001–10000;
   the old d=128 model already has weights → just eval it on 5001–10000, same eval set = fair). A change
   is **accepted only if it beats the current champion in BOTH modes by a RAW ≥ 0.0001** — immediate
   (imm) AND forgetting-curve (ahead). Monotonic-both-modes champion.
   **★ THRESHOLD HISTORY, and why it moved (Andrew 2026-08-10):** ≥ 0.0003 (through iter 25) →
   ≥ 0.0001 *after 4-dp rounding*, i.e. raw ≥ 0.00005 (2026-07-19, first applied to iter 26) →
   **raw ≥ 0.0001, no rounding step** (2026-08-10). The rounding form was retired because iters
   41/43/44 measured the same-capacity spread between three different execution schedules at
   |Δ| ≤ 7.5e-5 — i.e. the old raw bar of 0.00005 sat *below* the level the data can resolve, so
   it could accept noise. A raw +0.000088 now FAILS where it used to round up to a pass. No past
   accept is invalidated (smallest surviving margins: iter 39 +0.000158/+0.000153, iter 35
   +0.000153/+0.000271). ⚠ The noise floor is budget-dependent — re-derive the bar from the
   calibration's null pair if research moves to a shorter training budget.
   **+ p-gate (Andrew 2026-07-08):** additionally, the paired per-user one-sided Wilcoxon signed-rank
   (candidate vs current champion, same 5000 eval users — the data is already in the result jsonls, zero
   GPU cost) must give **p < 0.0001 in BOTH modes**. Tool: `python optimization/paired_pvalue.py
   --cand-ahead ... --cand-imm ... --champ-ahead ... --champ-imm ...` (exit 0 = p-gate pass; prints a
   `PAIRED_P_JSON` line). Rationale: SE of the by-user mean diff over 5000 users is ~0.0002–0.0003, so a
   bare point estimate clearing 0.0003 is only ~1σ; the paired test turns that into a real significance
   statement and neutralizes eval-side noise (training-seed noise is still covered by the seed-pair
   doctrine for thin margins). Record both p-values in the `p-value` column of `research_5k.md`
   (`ahead / imm`). The p-gate applies to accuracy accepts (monotonic-champion changes); SIZE/SPEED-
   exception accepts (efficiency-budget parity changes) are exempt — they don't claim an improvement.
   Wilcoxon-pruned (estimated) runs never reach the gate anyway; the p-gate is computed on real evals only.
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
   RWKV_QAT_SHIFT_PQ=reference/pq_cb_shift_q72u.txt RWKV_QAT_PQ_LEARN=1 RWKV_QAT_SHIFT_PQ_LEARN=1
   RWKV_QAT_SHIFT_SCOPE=card:int3,note:int3 RWKV_QAT_NORM_BITS=1 RWKV_QAT_FUSED=1 RWKV_NO_JIT=1`.
   **CODEBOOK LEARNING ON (2026-07-08, Andrew's direction #1):** cbs init from the reference q72u
   catalogs and train per-run; because the cb Parameters are process-globals initialized from the env
   files (NOT in the ckpt), the trial cmd repoints the env at each phase seam via
   `scratchpad/resolve_run_cbs.py` (WS-final exports → decay env; decay-final exports → eval env; fails
   LOUD with DONE_EXIT_CBFAIL_* if exports are missing). A champion = weights + ITS learned cbs —
   `promote_champion_5k.py` now records `ckpt`/`cb_wkv`/`cb_shift` in champion_5k.json, and any
   deploy/Rust-parity check of that champion must use those files, not the reference catalogs. NO_JIT
   until TorchScript is A/B-verified on the grafted q72u paths (once, at champion-run launch).
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
   champion) over the **last 1500 paired steps** (RWKV_PRUNE_WINDOW; was a growing full window until the
   2026-07-08 0p0014 audit — full window lags late regressions ~2k steps and kills late-bloomers on stale
   early deficits); **abort iff BOTH ahead and imm are worse at p < 1e-4 at TWO consecutive checkpoints**
   (RWKV_PRUNE_PERSIST=2, added 2026-07-09 — see the null-control entry below). Pairing is valid because the seeded
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
   **NULL CONTROL (2026-07-09, `scratchpad/prune_audit/null_control.py`) — triggered by Andrew's "5 trials,
   5 prunes" suspicion.** Paired two IDENTICAL-config runs (champ5k_r1 epoch 1 vs champ5k_b1, same seed/data
   order/env — differ only by the frozen compiled env's run-to-run noise) through the exact windowed test.
   Result: **no false fire** (the both-modes conjunction held) BUT the margin was thin — run-to-run drift is
   AUTOCORRELATED, and one transient episode (~cp 2400–3600) held imm at p ≤ 6e-15 for 4 consecutive
   checkpoints while ahead simultaneously dipped to 1.7e-3. Single-mode p-values are therefore hugely
   overconfident (1500 paired steps ≠ 1500 independent samples); a joint transient could plausibly false-fire.
   **Fix: RWKV_PRUNE_PERSIST=2** — both modes must be < α at two CONSECUTIVE checkpoints (600 steps). Real HP
   regressions persist by mechanism (replay: 0p0014's collapse strengthened 4500→5100: imm 1.5e-4→1.6e-16→
   2.2e-43); null transients come and go. Cost: real prunes fire ≤300 steps (~6 min) later. Of this era's 5
   prunes, the 3 early ones (7e-4, warmup 400/800: p 1e-36..1e-242 at steps where null noise is tiny) are
   beyond doubt; 0p0014 showed a strengthening real collapse; **0p002 is the one thin verdict** (abrupt
   1.0→1.1e-6 imm collapse in one checkpoint mirrors the null's transient signature — but 2×-optimal LR with
   1.4e-3 already regressing makes "genuinely worse" the strong prior; not worth a re-run).
   **CONFIRMED FALSE KILL → TUNER PRUNING DISABLED (2026-07-09, later the same day).** `hp5k_decay_ratio_0p1`
   (WS config byte-identical to `hp5k_weight_decay_0p1` — decay_ratio only affects the post-WS phase) was
   pruned at 4200 (p 1.2e-11 / 3.4e-45) while its twin survived AND **won eval in both modes**. Two lessons:
   (1) **train-loss pruning is SIGN-BIASED against regularization levers** — wd=0.1 runs persistently
   train-hot vs the wd=0.01 champion trace (that's what regularization does) yet evals better; wd=0.05 was
   only saved from the same fate by persist=2 (joint hit at its final checkpoint) and recorded an honest
   eval. (2) **run-to-run drift scales with the config** — between the two wd=0.1 twins, imm hit p=3e-45
   (the r1/b1 null pair at wd=0.01 peaked at 6e-15), so no fixed α is calibrated across bases. And once the
   descent's base regularization ≠ the reference run's, EVERY subsequent trial carries a systematic offset.
   ⇒ tuner trials now run WITHOUT RWKV_PRUNE_REF (traces kept); the bogus decay_ratio_0p1 row was removed
   from the journal (backup `scratchpad/tuner5k/tuner_5k_log_backup_before_dr0p1_removal.jsonl`). Pruning
   remains valid for research candidates at MATCHED regularization vs the champion (persist=2, α 1e-4), and
   the five LR/warmup-class kills stand (gross-failure magnitudes, corroborated). The prune saved ~8-10 GPU-h
   this era and cost one false kill + one bogus row — net positive but only for the gross-failure class.
   **VALIDATION-BASED PRUNE (the replacement rule, Andrew asked to brainstorm a better one, 2026-07-09).**
   Candidates validate every 500 steps (`VALIDATE_USERS` 5001–5010, ~596k labeled reviews/pass, ~50 s) and
   die iff **BOTH modes' val loss is worse than the champion's val trajectory at the same step by ≥ 0.004
   ahead AND ≥ 0.006 imm** (`RWKV_VPRUNE_DELTA_AHEAD/_IMM`) at **2 consecutive** val checkpoints
   (`RWKV_VPRUNE_PERSIST`), from step 1000 (`RWKV_VPRUNE_MIN_STEP`).
   **Parameterization is early-window by necessity (Andrew's flat-curve catch):** past ~step 2500 the val
   curves are nearly flat (ahead 0.3350→0.3313, imm 0.3182→0.3149 over the last 4000 steps), so no late
   threshold can catch the failures worth catching — peak_lr 7e-4's whole ahead gap was ~+0.003. The signal
   lives at steps 1000–2000 where the curves still drop ~0.01/1000 steps: a slow-convergence disaster sits
   hundreds of steps behind → gaps +0.004–0.011, while the champ5k_r1-vs-b1 twin-null there is ≤ 0.0025
   ahead / 0.0029 imm (≤ 0.0012/0.0005 after 2000). Thresholds = 2–3× the early null per mode; the
   joint-AND at 2 consecutive checkpoints carries the safety (the twins never neared a joint hit).
   Why it's right: val is SIGN-CORRECT for regularization levers (wd=0.1's val would look better, not
   worse), magnitude replaces the uncalibrated Wilcoxon p (autocorrelated diffs), disasters die by ~step
   1500 (saves ~2.8 h of 3.5 h), and LATE-emerging regressions intentionally never fire — the run is
   60–80% done and an honest eval is worth the tail. Wiring: train_rwkv `RWKV_VPRUNE_*` + a
   `<trace>.val.jsonl` sidecar whenever RWKV_STEP_TRACE is on; `promote_champion_5k --val-trace` embeds the
   val arrays (champ5k_b1's were attached from its log via `scratchpad/attach_val_ref.py`); tuner trials
   set VALIDATE_EVERY=500 + RWKV_VPRUNE_REF. Exit 42 + the same marker path (`rule: "val"`, estimates =
   champ_final + val_delta).

DONE (2026-07-01): the `decay_ratio` lever (range [1/10, 1/2.5]) is now in `hp_tuner_5k.py`. Still TODO
when the tuner is set up for 5k: repoint its data paths to the 5k train_db, set MAX_TRAIN_GLOBAL_LEN=110000
(batch sweep), recompute GROUPS_PER_EPOCH, and make WS/decay/eval apply fake card- AND note-state quant
(once the sibling's fast fake-quant kernel is copied).

## Setup
- **Train** users 1–5000; **eval** users 5001–10000 (disjoint held-out half).
- **Compute budget:** **1 WS epoch** + decay = WS × decay_ratio (ratio ∈ [1/10, 1/2.5], default 0.25 →
  0.25 decay ep; cosine). *(2→1 epochs Andrew 2026-07-09, via the champ5k_b1 budget A/B: identical
  recipe at half budget came out ahead −0.000058 (p=0.31) / imm +0.000430 BETTER (p=6.1e-62) vs
  champ5k_r1 on the full paired 5000-user eval — the 2nd pass over the same 5000 users adds nothing;
  the data-variety-beats-repetition lesson holds at 5k. Applies to ALL runs: tuner trials and research
  candidates. Pre-ship: the final champion gets one 2-ep confirmation run.)*
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

## Queued idea — RETRY distillation on the A18 trunk, full-run rather than warmup-only (2026-07-26)

Distillation stands at **0/1, not closed** — and the one attempt ran under conditions that no longer
hold, so the conduct rule's "3-5 in-family variants before writing a family off" has barely started.

**What was actually tried:** iter 10, warmup-only KD from a d=128 teacher onto the **d=32** trunk
(193,724 params). Rejected, worse in both modes. It was filed under **early-training-intervention**
(now 0/2), which is why the family scoreboard shows no distillation entry at all — a mis-filing that
made the idea look untried and simultaneously look closed.

**Why the conditions changed.** At d=32 on 100 users the model was **DATA-limited** — the lesson bank
records that capacity adds (num_curves, channel_mixer width, more epochs) were all rejected there, and
that "the path forward is MORE DATA". A teacher cannot fix a data limit; it only re-expresses labels
the student already has. A18 is the opposite regime: the width ladder closed at a genuine accuracy
floor, and the second LoRA halving **flipped sign** vs A14's first halving at d=128 (+0.00002/+0.00009
for -27.5k, where the same lever had IMPROVED both modes on the wider trunk). That is the signature of
a **capacity-limited** student, which is exactly where soft targets are supposed to pay: they carry
per-example information that a 0/1 label cannot, letting a small net spend its limited capacity on
the teacher's ranking rather than rediscovering it.

**Untried variants, roughly in order of expected value:**
1. **Full-run KD with a fixed weight** (the classic form) instead of warmup-only. Warmup-only is the
   unusual variant, and it is the one that was tested.
2. **Anneal the KD weight** from ~1 to 0 over the run — keeps the "student can surpass the teacher"
   property that motivated warmup-only, without the hard cliff at step N.
3. **Distill the CURVE, not just the scalar.** The teacher's whole forgetting curve at several
   elapsed times is a far richer target than its prediction at the one observed time, and it is
   free — the teacher dump already stores the curve parameters.
4. **Teacher = `pretrain/RWKV_trained_on_101_4999.pth`** (2.76M, 4.96x the student). No eval leakage:
   it never saw 5001-10000. Confirm which teacher iter 10 actually used before treating this as new.

**Infrastructure already exists** — `RWKV_KD_TEACHER` / `RWKV_KD_DUMP_OUT` / `RWKV_KD_STEPS` in
`train_rwkv.py`, with the arch-swap footgun already solved via `RWKV_ARCH_MODULE`. The dump path
means the teacher runs ONCE, not every step, so KD is cheap in wall-clock.

**Judge it as its own family.** Whatever the outcome, record it under *distillation*, not
early-training-intervention, so the count is honest.

### LAUNCHED 2026-07-26 as **iter 32** — variant 1 (fixed alpha), and why the case got stronger

`scratchpad/iter32_kd/` (dump toml + student toml + `run_iter32_kd.cmd` + `check_dump.py`), parked
behind the mode-2 diagnostic. New flag **`RWKV_KD_ALPHA`**: set = alpha held FIXED (the classic
form, variant 1); unset = iter 10's linear 1->0 ramp, byte-identical. Run at 0.5 over all 22,346 WS
steps; decay on hard labels.

**The teacher's edge was quantified the same day and it is mostly BUDGET, which makes KD the right
tool rather than a weaker one.** On the val half the teacher scores 0.294612 / 0.263561 vs iter 31's
0.298909 / 0.267637. Decomposing that +0.0043 / +0.0041 against the record: A0 — the *same d=128
architecture at the same 2,762,884 params*, retrained on our 1.25-epoch recipe — already sat at
0.298342 / 0.267858. So **+0.00373 / +0.00430 is the training recipe, and the whole 4.95x ablation
ladder cost only +0.00096 / +0.00053.** The teacher is not a better *architecture*, it is the same
architecture trained ~10x longer. Distillation is therefore **budget transfer**: the student runs one
epoch while its targets carry twelve. That is a different and better-founded bet than "copy a bigger
model", and it is the reason to expect more here than iter 10's d=32 attempt returned.

Confirm at verdict against `result/RWKV-base5k.jsonl` (the `.cmd` prints it) — the interesting number
is not only "did it beat iter 31" but "how much of the 0.004 did it import".

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

## Incidents
- 2026-07-08 **champ5k_r1 WS->decay seam crash (fixed, f71f43b; ~15 min lost).** First-ever
  LEARN=1 -> LEARN=1 optimizer handoff: train_rwkv registered the learnable-cb param groups only
  AFTER `optimizer.load_state_dict` (correct for warm-starts from 5-group pre-LEARN champion
  optims), but a LEARN=1 run saves 7 groups (5 base + shift-cb + wkv-cb) -> decay's fresh 5-group
  optimizer raised "different number of parameter groups"; the .cmd then fell through to
  DONE_EXIT_CBFAIL_DECAY. Fix: cb Parameters are created up front; when the resumed state's group
  count == base+cb they register BEFORE the load (cb Adam moments resume across the seam --
  the right semantic, values still come from the exported cb files), else the old add-after path
  is unchanged (warm-starts unaffected). Also fixes mid-run crash-resume for any LEARN=1 run.
  Resumed via `run_champ5k_r1_resume.cmd` (decay-onward, same frozen env, appends to the same
  log); WS artifacts were all intact (ckpt + optim + resolved WS cbs).

## Speedups banked (detail also in CLAUDE.md)
- 2026-07-08 **EVAL CPU PATH VECTORIZED (byte-identical, banked mid-champion-run).** The per-review
  Python loops in the eval post-processing were the CPU drag between users: `extract_p` (per-index dict
  builds over every timestep), `get_stats` (per-eq-review gather loop + an up-to-800k-row Python
  rows-list → DataFrame), and `run()`'s per-batch `{**a, **b}` dict rebuilds + per-th raw comprehensions.
  All replaced with numpy mask/`dict(zip())` builds and a sorted-key `searchsorted` gather (`_eq_gather`),
  preserving EXACT dtypes/values (np-scalar keys+values from array iteration; DataFrame dtype matched via
  a one-row probe of the old `np.array(rows)` promotion; within-bin row order preserved → groupby-mean
  bit-identical). Timing (300k-review user): extract_p 308→118 ms, get_stats 1151→87 ms (×2 calls/user).
  Verified: 6-trial exact-equality harness incl. dup-keys + int/float bins (`scratchpad/eval_speed/
  stats_ab.py`, ALL_PASS) + E2E GPU A/B on 3 real users (5005/5033/5044, champ_h2k16 bf16) —
  result jsonls byte-identical (`fc.exe` no differences). RNN/trace callers (run_as_rnn,
  export_rnn_trace) pass tensor dicts → auto-fallback to the untouched original loop. Picked up
  automatically by the champ5k_r1 eval phase (shards import get_result at launch). FOLLOW-UP at eval
  launch: sample per-shard VRAM + GPU util → decide if future evals get --shards 3-4 (d=32 only).
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
- 2026-07-08 **Wall-clock gap RESOLVED (clean window, build done): none.** q72u frozen-env quant-aware
  step: 1184 ms GPU-busy vs 1207 ms wall — fully GPU-bound, host-side batching lever DEAD.
- 2026-07-08 **Shift-PQ search kernel — 1.21x on the q72u quant-aware step (1.207 -> 0.996 s/step,
  65+327 protocol).** Profile of the q72u step (first profile since the joint/shift-PQ/learnable-cb
  port) showed ~45% = the LEARNABLE shift-PQ nearest-centroid search running eager `torch.cdist().
  argmin()` in `fake_pq_shift`: sqrt (173 ms) + clamp (173 ms) + argmin (101 ms) + sgemm (99 ms) over a
  never-needed N x 4096 fp32 distance matrix (~1.8 GB per call, 16 calls/step at MAX=110000).
  Fix ladder (all in `rwkv_model.py::_nearest` + csrc):
  (1) `_sq_dist_rows` — aten::_euclidean_dist's exact augmented matmul minus the sqrt; pre-sqrt values
      bitwise-identical to cdist's mm path (unit-proven incl. exact-tie adversarials); saves the sqrt
      pass only (1.189 s/step). A nested-torch.compile fused clamp+argmin attempt DID NOT ENGAGE
      in-process (only 20 ms of fused triton appeared; the big clamp+argmin stayed eager) — dropped.
  (2) **`rwkv7_pq_argmin` CUDA kernel (the win)**: direct squared-distance accumulation, no
      materialized matrix, first-strict-min ties == torch.cdist().argmin() semantics. v1 one-row-per-
      block was L2-BOUND re-reading the 256 KB catalog per row (28 GB/call -> 30 ms, SLOWER than
      cdist). v2 row-tiled (16 rows/block) 9.0 ms; v3 templated on SUB (compile-time register tiles;
      runtime-indexed tiles were spilling to local memory) **5.9 ms/call vs cdist 23.9** -> ~95 ms/step
      search total. Dispatch: sub 8/16/32 fast path, generic fallback; CPU tensors fall through to the
      matmul tier (RNN/Rust-parity safe). Escape hatches: RWKV_SHIFT_SEARCH_KERNEL=0 (tier 1) and
      RWKV_SHIFT_SQ_SEARCH=0 (tier 2) -> original cdist.
  Correctness: index-identical to cdist on 330k random rows + exact-tie adversarials (0 mismatches);
  goldens BITEXACT_PASS after both rebuilds; eval-path (no_grad) numerics change only on fp32 near-tie
  index flips (none observed). E2E: 3-arm 110-step A/B (sq0a/sq0b control + sq1) — **the frozen env is
  inherently NOT run-to-run reproducible** (controls diverge at step 27, trace noise <=3e-4, weight
  drift 1.7e-2 by step 110; inductor autotune nondeterminism suspected), so bit-exact E2E is
  unattainable for ANY change; the rewrite's drift (diverges step 15, <=6e-4) is the same noise class.
  ⚠ PROTOCOL NOTE: the old "run-to-run variance ~0" doctrine does NOT hold under the compiled frozen
  env — per-step trace noise ~1e-4..3e-4 (zero-mean; Wilcoxon prune pairing still valid).
  Stacked 2026-07-08: 1.643 (NO_JIT) -> 1.207 (sanctioned flags) -> 0.996 s/step (search kernel) =
  1.65x; champion-run training ~4.6 h. Next targets if ever needed: QAT kernels (210 ms, already
  37x-optimized), elementwise mass via compile-all-mixers/recompile-limit raise (PERTURBING — needs
  trajectory revalidation; Dynamo's 8-entry cache cap leaves ~1 of 9 mixer guard-sets eager).

## The ahead-vs-imm information gap (measured 2026-08-11) — the largest quantified headroom on the books

The two scored metrics predict **the same events**, from different information sets. Verified on
the iter-41 champion's own result jsonls (zero GPU cost): the per-user `size` field is **identical
for all 2500 users** between `RWKV-*` (ahead, curve head, scored at the *previous* real row) and
`RWKV-P-*` (imm, rating head, scored at that review's *own* query row). Same event set, same count,
every user.

| | value |
|---|---|
| by-user mean ahead | 0.297889 |
| by-user mean imm | 0.265479 |
| **gap** | **0.032411** |
| imm better than ahead on | **2497 / 2500 users (99.9%)** |
| per-user gap: median / p10 / p90 | 0.0197 / 0.0056 / 0.0680 |

So the model already emits, for every scored review, a **strictly better-informed estimate of the
very label the curve head is being trained on** — better on 99.9% of users, by ~100x a typical
accepted iteration gain. That is what makes privileged self-distillation (imm -> ahead soft
targets) and retrievability-coupling (feeding logit R(t) into the Again logit) the two
highest-headroom proposals in the 2026-08-10 ranking.

⚠ **The gap is an UPPER BOUND, not a reachable target.** The query row sees the intervening
reviews and the exact lag; the ahead head structurally cannot — predicting cold from history *is*
the task. Distillation can transfer the variance-reduction part of the gap (a calibrated soft
target beats a 0/1 label, which is why the external-teacher alpha curve peaks at 0.9), not the
information part. Treat 0.032 as "there is real room here", not as the size of the prize.

## What the training budget is worth — a free corroboration of the endgame premise (2026-08-11)

The budget calibration's c41 arm is the champion recipe at **1/3 budget** (WS 3,645 + decay 3,644
vs 10,935 + 10,935). Paired against iter 41 at full budget on the same 2500 VAL users:

| | ahead | imm |
|---|---|---|
| iter 41 (full) | 0.297889 | 0.265479 |
| c41 (1/3) | 0.299851 | 0.267502 |
| **cost of the 3x cut** | **−0.001961** | **−0.002024** |

Two things follow.

**1. It is ~7x a typical accepted iteration gain (+0.0003).** So short-budget models are markedly
further from convergence — which is fine for *ranking candidates at matched budget*, and exactly
why the bias caveat matters: effects that only pay off near convergence (regularization, added
capacity) will be systematically under-measured there.

**2. It independently corroborates the endgame's premise, and this was not designed in.** The
phase's headline gap — our 1.25-epoch recipe vs upstream's ~12 — is **+0.00373 ahead / +0.00430
imm (mean +0.00402)**. If returns are ~log-linear in budget, the measured 3x step (+0.00199 mean)
scales to a 10x step as `0.00199 x ln(10)/ln(3)` = **+0.00418** — within **4%** of the recorded
gap. Two independent routes to the same number: one from a controlled 3x budget A/B on our own
trunk, the other from a five-week-old comparison against a differently-trained model.

⚠ Log-linearity is an assumption fitted to a single pair of points, and the two arms differ only
in budget whereas upstream also differs in augmentation, peak LR, warmup and MAX. Read it as
"the budget story survives a quantitative check it could have failed", not as a forecast. It does
raise confidence that the ~4-day 10x endgame run buys roughly what it is predicted to buy.

## Budget calibration — the PRE-REGISTERED decision rule (written 2026-08-11 12:05, before arm c43 reported)

Two of the three arms are in. Recorded here *before* the third so the criterion cannot be fitted
to the answer.

**Measured so far.** (a) The 1/3 budget costs **−0.00196 ahead / −0.00202 imm** in absolute terms
(c41 vs iter 41). (b) The known large effect (interleaving) survives the cut with sign and
significance intact but **compressed to a consistent ~65%**: ahead +0.000315 vs +0.000489 (64.4%),
imm +0.000402 vs +0.000612 (65.7%), both still p<1e-33. So short-budget effects are **scaled, not
scrambled** — ordering transfers predictably.

**Still to come.** Arm c43 is the verified full-budget NULL (iter 43 tied iter 41 at p=0.42/0.098),
so its short-budget |Δ| vs c41 **is** the short-budget noise floor, call it N.

**The rule.** At full budget the floor is 7.5e-5 and the bar is 1.0e-4, i.e. the bar sits 1.33x
the floor. Holding that evidential standard at 1/3 budget, and correcting for the 0.65 compression
so the bar means the same thing in full-budget terms:

```
short-budget bar (full-budget-equivalent) = 1.33 x N / 0.65 = 2.05 x N
adopt iff 2.05 x N <= 1.0e-4   =>   N <= 4.9e-5   in BOTH modes
```

**ADOPT 1/3 BUDGET IFF the c41-vs-c43 |Δ| ≤ 4.9e-5 in both modes.** Note what that demands: the
short-budget floor must be *better* than the full-budget 7.5e-5, when the models are further from
convergence — so the honest prior is that this FAILS. If it does, the finding is not "the
calibration was wasted" but "1/3 budget cannot gate 0.0001-class effects", which still licenses
short budget for **screening** larger effects and for **ranking** batches, just not for accepting
champions. Cost of learning it: 15.3 h, versus discovering it by promoting a phantom champion.

### VERDICT (2026-08-11 14:01): DO NOT adopt the 1/3 budget for GATING. Use it for screening/ranking only.

All three arms complete, ~15.3 h, chain clean (`DONE_EXIT_0`).

**The measured short-budget noise floor** (c41 vs c43 — a pairing verified NULL at full budget,
p=0.42/0.098): **ahead |Δ| = 9.0e-5, imm |Δ| = 3e-6** (p=1.00 / 0.71, so both are genuinely null
in significance — it is the MAGNITUDE that matters here, since that is what a bar must clear).

Against the pre-registered criterion (adopt iff |Δ| ≤ 4.9e-5 in both modes): **imm PASSES at
3e-6, ahead FAILS at 9.0e-5. Verdict: do not adopt.** Ahead is the binding constraint, and it
fails by ~1.8x.

**Why, in one line:** on ahead the floor got *worse* (9.0e-5 vs the full-budget 7.5e-5, ~1.2x)
while the signal shrank to 65% — so **signal-to-noise falls ~1.9x**. The effective accept bar at
1/3 budget, expressed in full-budget-equivalent units, would be **1.84e-4 on ahead** versus the
1.0e-4 we accept today. Short budget would silently make us ~2x stricter on ahead and throw away
real candidates, which is the opposite of the intent (more runs, not fewer accepts).

**The asymmetry is itself a finding.** imm's floor (3e-6) is 30x tighter than ahead's (9.0e-5).
The rating head reads the query row directly, so its prediction is dominated by well-determined
current-row features; the curve head is evaluated one review stale, through the recurrence, where
trajectory differences accumulate. That matches the earlier information-gap measurement (imm beats
ahead on 99.9% of users) — the same structural asymmetry shows up as *stability*, not just accuracy.

**What the 15.3 h bought, all of it reusable:**
1. **Gating stays at full budget.** Settled by measurement, not by argument.
2. **Screening/ranking at 1/3 budget is LICENSED** — effects are scaled, not scrambled (~65%
   compression, near-identical in both modes, both p<1e-33). A batch of candidates can be ranked
   short and only the leader confirmed at full budget. That is the two-tier screen, now with a
   measured transfer function instead of a hope.
3. **The 0.65 compression constant** — multiply a short-budget delta by 1/0.65 to estimate its
   full-budget size.
4. **The budget-scaling corroboration** (3x budget = +0.002, projecting to +0.0042 at 10x vs the
   +0.0040 recorded gap) — independent support for the endgame run's premise.
5. **A second null pair** for the noise-floor record: iters 41/43/44 at full budget (±7.5e-5) and
   c41/c43 at 1/3 (ahead 9.0e-5, imm 3e-6).

⚠ Do NOT read this as "short budget is useless". It is unusable for *accepting champions* at our
effect sizes; it is fine for triage. And the bias caveat still stands untested: all three arms were
SCHEDULE changes, so nothing here measures how short budget treats regularization or capacity.

### ...and the follow-up that closes it: SCREENING IS NOT WORTH IT EITHER (Andrew asked "are you sure screening+full run of the best candidate will be cheaper?", 2026-08-11)

It is not, for our candidate pool. The "licensed for screening/ranking" line above is too
generous; this supersedes it. **Standing decision (Andrew): keep doing full runs.**

**1. The saving is 46%, not 67%.** A short run is 5.1 h vs 9.4 h, because the eval is a FIXED
2.9 h and becomes 57% of a short run. Screening K candidates then confirming the leader costs
`5.1K + 9.4` vs `9.4K`, so it only breaks even at **K > 2.2** — at K=2 screening is COSTLIER.

**2. The screen cannot resolve most of our candidates — the fatal one.** Its noise on a PAIRWISE
comparison is `9.0e-5 x sqrt(2)` = 1.27e-4 short = **1.96e-4 in full-budget effect size**. Applied
to the last ten iterations' ahead deltas: resolvable = iters 36, 37, 41, 42; **inside the noise =
iters 35, 38, 39, 40, 43, 44 — six of ten, including TWO ACCEPTED CHAMPIONS (35 and 39).** A screen
that cannot separate an accepted champion from a null is shuffling, not ranking. Post-HP-tuning our
candidates cluster inside ±0.0005 and half inside ±0.0001, which is exactly its blind region.

**3. The two fixes fight each other.** Cheaper screen = halve the ranking eval = noise x sqrt(2) =
blinder. Sharper screen = more eval users = the saving erodes.

**When screening WOULD pay:** a batch of >=3 candidates with WIDE expected spread — a fresh
architecture family, a coarse HP grid, deliberately wild ideas where some are clear losers. Not the
incremental near-bar work that is most of what we do.

**The real lesson about throughput:** our ceiling is set by the FIXED 2.9 h eval and the
dispatch-bound training step, not by the epoch budget. Cutting epochs attacks the half of the
iteration that is already cheapest to shorten and most expensive to shorten *correctly*.

## When do we stop? The balance sheet (Andrew's question, 2026-08-12)

> "The stopping point should be 'when we are reasonably confident that if we enabled QAT and
> increased the epoch budget to ~10+2 recipe of the old model, algorithmic improvements + new input
> features would push log loss below that of the old model'."

That makes stopping arithmetic rather than judgement. The rule:

    still_needed(mode) = (champion - old_model) - budget_credit + QAT_tax

and we stop when we are confident features + remaining algorithmic work cover `still_needed`.

### The three terms, all measured except features

| | ahead | imm | source |
|---|---|---|---|
| gap to old model (VAL half) | +0.00309 | +0.00181 | iter 45 champion vs 0.294612 / 0.263561 |
| **QAT tax** | **+0.00290** | **+0.00445** | `champ5k_plain` vs `champ5k_b1` — matched recipe, QAT env stripped, n=5000, p=0.0 |
| budget credit at ~12 ep | −0.00373 | −0.00430 | A0 (IDENTICAL arch + params to the old model, our 1-ep recipe) vs baseline |
| *same, projected* | *−0.00411* | *−0.00424* | *log-linear from the measured 3x step (+0.00196/+0.00202); agrees to 4%* |
| **STILL NEEDED** | **+0.00225** | **+0.00196** | |

Andrew's guess that QAT "eats 0.0030–0.0050" was right. ⚠ That tax was measured on the **d=32 /
193,724-param** trunk with the q72u config; the current model has **5x the card state** (2,880 vs
576 floats), so it is the least trustworthy number here — see the recommended measurement below.

### Why the algorithmic loop alone cannot get there

Post-HP-tuning rate (iter 35 → 45, 10 attempted iterations, 4 accepted): **ahead +0.000112 and imm
+0.000057 per attempted iteration**. Against the requirement:

* ahead: ~20 more iterations ≈ **7.4 days** of continuous GPU
* imm: ~34 more iterations ≈ **12.6 days**

So "keep iterating until we get there" is a ~2-week GPU commitment with no features and no QAT
work — and imm is the binding mode, which it was not before.

### ★ TWO CHEAP MEASUREMENTS WORTH MORE THAN WEEKS OF ITERATING

1. ~~Rectify the baseline~~ **WITHDRAWN (Andrew, 2026-08-12): "there's no need to rectify the
   baseline since that's not how the original model works, and we want our numbers to be compared
   to the srs-benchmark leaderboard's version."** The target is the PUBLISHED leaderboard entry, so
   the comparison is our deploy-honest (rectified) number against their unrectified one, and we
   absorb the rectification cost ourselves. The ahead gap stands at **+0.00309**; there is no
   discount to be had here. Recorded because the tempting version of this argument will come back:
   a like-for-like rectified comparison would flatter us by ~0.002-0.0036, and it is not the goal.
2. **Measure the QAT tax on the CURRENT d=80 model (~9 h).** imm now binds *because* of QAT
   (+0.00445 of a +0.00196 requirement). That figure comes from a 3x-smaller model with a
   different state-quant config, and it is the single largest term on the sheet.

### The reframe

**The QAT tax on imm is worth more than the entire remaining algorithmic loop.** +0.00445 is ~77
iterations at the recent imm rate; halving it beats a month of the current loop. It has never been
a research target because plain-vs-QAT numbers were never comparable -- but the matched pair above
shows the comparison is available whenever we want it.

⚠ **AND THE "FEATURES ARE FREE, RUN THEM IN PARALLEL" CLAIM WAS WRONG (Andrew, 2026-08-12):
"Pre-processing is CPU-only, sure, but training is obviously not."** Only the LMDB rebuild is
CPU-only. Everything that makes features *count* -- re-basing the champion on the new inputs, then
training and evaluating each candidate -- is GPU work that competes with the algorithmic loop for
the same single 4070, and every pre-rebuild iteration is measured against a champion the rebuild
will invalidate. So features are not a free parallel track; they are a PHASE that largely displaces
the loop, and only their ~2-4 day preprocessing head start can overlap it.
(The same wrong claim is in CLAUDE.md's ORDER FROM HERE and is corrected there too.)

## The C=80 shift-PQ refit: what a bit budget actually buys (2026-08-12, no GPU)

The shipped q72u shift catalog is C=32-shaped (`2 12 16 32 4096`) and **hard-fails** on the d=80
model — `RuntimeError: size of tensor a (32) must match b (80)`. So a refit was mandatory before any
QAT measurement, and the refit has a free choice of capacity. Andrew: do both — then two probe fits
turned the pair into a curve. All fitted on ONE corpus (190,763 TS + 117,382 CS unit vectors from
`reference_iter45` traces), MiniBatchKMeans, identical treatment, **held-out** error on 2,000
withheld vectors per role (at ncent=4096 a fit-set error is meaningless — the catalog memorizes).

| catalog | bits/vector | sub | centroids | card state | TS err | CS err |
|---|---|---|---|---|---|---|
| m4b6 | 24 | 20 | 64 | ~23 B | 0.3604 | 0.3132 |
| **m2b12** | **24** | 40 | 4096 | **~23 B** | **0.1902** | **0.1601** |
| m5b12 | 60 | 16 | 4096 | ~37 B | 0.1734 | 0.1465 |
| m10b12 | 120 | 8 | 4096 | ~60 B | 0.1405 | 0.1216 |

**1. At a FIXED budget, catalog size beats chunk count — decisively.** m4b6 and m2b12 both cost 24
bits, and m4b6 is **1.9x worse**. This independently reconfirms, on the new architecture, the
archived quant-endgame lesson that "per-card cost is INDEX bits — catalog size is FREE (amortized):
fewer/bigger chunks + huge learnable catalogs beat the product form". Do not split the shift vector
more finely to save bits; it is the wrong direction.

**2. Bits do buy fidelity, but at poor and uneven rates.** 24 → 60 b is −8.8% error; 60 → 120 b is
another −19%. The driver is `sub`, not bits per se: 4096 centroids cover `4096^(1/sub)` points per
axis = 1.2 at sub=40, 1.65 at sub=16, 2.7 at sub=8. Shrinking sub helps a lot, but at fixed bits you
can only shrink sub by shrinking the catalog — which finding 1 says costs more than it saves. That
is the vise the scheme is in.
⚠ So my first read of the m2-vs-m5 pair ("capacity is not the lever") was **too strong**. Capacity
IS a lever; it is just an expensive one, and 5x the bits removes only ~26% of the error.

**3. The operative prediction for the QAT arms.** m2b12 and m5b12 differ by only ~9% in
reconstruction error, so their logloss arms should be close. If that holds, the deploy config stays
at the cheap **~23 B card** and the tax — whatever it measures — is NOT mostly shift-code starvation,
which points the reduction work at the WKV side or at QAT placement instead. If instead the two
arms differ materially in logloss, then logloss is far more sensitive to shift fidelity than the
reconstruction error suggests, and that is worth knowing on its own.

**Trainer-vs-runtime encode agreement, checked (2026-08-12).** A catalog is only meaningful if
`fake_pq_shift`'s runtime encode matches what `pq_train_shift.py` assumed when clustering — a
mismatch would silently inflate any measured tax, and it is exactly the class the three-way-parity
rule exists for. Feeding real corpus vectors back through the runtime path and scoring the
trainer's own metric gives **0.1944 on `103_card`** against the trainer's **0.1902** held-out.
They agree. An apparent 13% gap on a naive pooled sample was **sampling composition, not a bug**:
the trainer draws randomly from a pool that `103_card` alone supplies ~1/3 of, whereas taking the
first 400 vectors of each of 14 files equal-weights the small, high-error ones. Per-user error is
genuinely heterogeneous (0.16–0.27), which is worth remembering when reading any single-user quant
diagnostic. Also confirmed en route: both refitted catalogs load and run at C=80, where the shipped
q72u catalog hard-fails.

## ★ THE QAT ENV IS SILENTLY INERT UNDER RWKV_ARCH_MODULE — found 2026-08-12, blocks all tax work

**The three-cell design (Andrew, 2026-08-12) that the measurement now implements:**
1. no QAT, full precision — iter 45's numbers (have);
2. QAT-trained, evaluated quantized — the deploy number;
3. QAT-trained, evaluated at FULL precision — same checkpoint, QAT env off at eval.
(2)−(1) = the full tax. (2)−(3) = **precision degradation** (what quantization costs a model
trained for it). (3)−(1) = **model drift** (what training under fake-quant costs by itself).
The d=32 record already carries this decomposition (`quant_cost`/`finetune_cost` in qat_log) and
supports Andrew's recollection that drift dominated: warm-started decay-QAT #39 had precision
degradation −0.000127/+0.000018 (nothing) vs drift +0.001129/+0.002456.

**But cells 2/3 cannot be produced yet.** Both PTQ probe arms came back with cost **+0.000001 /
−0.000000 and m2b12 == m5b12 exactly** — a 0.19-reconstruction-error shift code cannot cost zero,
so the sim never ran. Root cause, verified in source: `rwkv/architecture.py` applies the
`RWKV_QAT_SCOPE` / `RWKV_QAT_LOWRANK_SCOPE` / `RWKV_QAT_SHIFT_SCOPE` env vars to the DEFAULT
config's layer objects (lines ~145–190), and THEN the `RWKV_ARCH_MODULE` override (line ~244)
replaces `DEFAULT_ANKI_RWKV_CONFIG` wholesale with the module's own config — which has no QAT
fields. The scopes are parsed, their banners print, and the objects they mutated are discarded.
**Every track-2 run (A0 onward) has had QAT env vars silently ignored.** The banner order is the
tell: `[QAT-LOWRANK] set:` prints BEFORE `[ARCH-MODULE] ... <-`.
No historical number is wrong — no track-2 iteration *claimed* to be quant-aware — but the
5k-methodology (a) "quant-aware logloss" convention has been impossible on this trunk, and my
runner's banner guard passed for the wrong reason (the banner lies).

**✓ FIXED 2026-08-12, `70185c7`.** The three scope blocks became `_apply_qat_scopes(layers)`,
called LAST on `DEFAULT_ANKI_RWKV_CONFIG.modules` — i.e. on whatever config is final — via
setattr/getattr with defaults so an override module carrying an older `RWKV7Config` still
imports. The override legitimately owns the *capacity* hooks above it; QAT scope is an
orthogonal deploy-simulation overlay and must survive it.
**Regression test, and it is the one that matters: 10 users, iter 45's checkpoint, q72u PTQ env**

| | ahead | imm |
|---|---|---|
| before the fix | +0.000001 | −0.000000 |
| after the fix | **+0.009276** | **+0.012690** |

**Two guards added so this cannot recur silently.** (1) `scratchpad/qat_tax/assert_qat_live.py`
imports the arch under the run's own env, prints every stream's final
`lowrank_rank`/`state_qmax`/`shift_qmax`, and exits 44 if a stream *named in a scope env* is not
really quantized (it also catches a scope naming a stream the arch does not have — a typo or a
renamed arch). Wired as **phase 0** of `run_arm.cmd`, before any GPU is spent; ~2 s.
(2) the eval banner guard now greps the **shard** logs too — `eval_sharded`'s parent log never
contains them, which is why both arms exited rc 41 while still producing results.
**THE TRANSFERABLE LESSON: a banner proves a value was COMPUTED, never that it was USED.** The
`[QAT-LOWRANK] set:` line was perfectly truthful about what it set; the object was discarded one
step later. Any guard for an env-driven setting must inspect the *consumed* state — here, the
final config — not the parsing log. That is why the assert imports the module instead of grepping.

**THEN the three-cell chain:** decay-QAT (catalog chosen by the re-run PTQ arms) → eval
quantized → eval fp.

## ★★ THE WKV CODEBOOK IN THE QAT RECIPE IS WORSE THAN RANDOM ON THIS TRUNK — 2026-08-12

Found while preparing tax-REDUCTION levers, on CPU, with no GPU cost. It reframes the tax
measurement that was about to run.

`reference/pq_cb_wkv_q72u.txt` (header `1 10 32 16 1024` = joint-uv, 1024 centroids over
concat(u_unit, v_unit)) is the WKV half of the q72u QAT recipe. It was fitted on the **d=32 / H=2**
model. It stays *dimensionally* valid here because K=16 either way — which is exactly why nothing
ever caught it. The shift catalog got refitted for C=80 only because the old one hard-FAILED a
shape assert; the WKV one fails **silently, by being merely wrong**.

**Mean relative L2 reconstruction error on this trunk's own card-stream WKV states** (user 101,
23,240 joint (u,v) vectors, held-out split; `scratchpad/qat_tax/wkv_cb_staleness.py`):

| encoder | held-out err |
|---|---|
| **OLD — the q72u catalog, in the live recipe** | **0.9985** |
| 1024 **RANDOM** unit-pair directions | 0.9576 |
| encode everything to **ZERO** | 1.0000 |
| REFIT — same budget, fitted on d=80 | 0.3032 |
| ORACLE — fitted on the holdout itself | 0.2196 |

**The shipped catalog is worse than random and within 0.15% of the zero-codebook bound.** It is
not "stale", it is uninformative: at 1024 centroids it uses only 454 distinct entries and leaves
the query at ~60° from its nearest centroid.

**Mechanism — it is aimed at the wrong SUBSPACE, which is why it loses to isotropic noise.** Not a
per-head story: the five heads are diffuse, not separate blobs (within-head mean cosine 0.13–0.32),
and the old centroids' mean direction is not badly aligned (median best-cos 0.212). The real split
is the principal subspace — the d=80 data's **top-8 PCs carry 63.8% of the DATA's variance but only
22.6% of the OLD CATALOG's**. Random directions are at least unbiased; a catalog concentrated on
the wrong low-dimensional subspace is actively misaimed.

**THE GENERAL LESSON, and it is the same shape as the QAT-inert bug found the same morning: a
quantizer artifact is validated by its SHAPE and used on its CONTENT.** K is unchanged, so every
assert passed. **Any change to d_model, H, or the state distribution silently invalidates a fitted
codebook, and nothing in the pipeline will say so.** A cheap standing guard: score the catalog's
held-out reconstruction error against a random-catalog control whenever the arch changes — CPU-only,
minutes, and it is what turned this up.

**CONSEQUENCE FOR THE TAX.** The +0.00290/+0.00445 on record, and the +0.009276/+0.012690 PTQ probe
measured this morning, were both taken with this catalog. They are therefore **not the cost of
quantizing this model — they are substantially the cost of destroying its card/note WKV state**.
Re-measuring the tax before refitting would have spent ~11 h of GPU characterizing a broken config.
Revised order: finish the corpus dump → fit a d=80 joint-uv catalog at the SAME 10-bit budget (so
deploy state size is unchanged) → re-run the cheap PTQ probe → only then the three-cell chain.
⚠ The 0.3032 refit number is from ONE user's corpus and its holdout is not independent of its
training split; treat it as directional. The OLD-vs-RANDOM contrast is what is decisive, and that
one does not depend on the refit at all.

**NO HISTORICAL NUMBER IS INVALIDATED — checked, not assumed.** (a) Every runner that sets
`RWKV_QAT_PQ=reference/pq_cb_wkv_q72u.txt` (champ5k_b1/r1/t1, iter10-13, …) is **d=32-era**, where
that catalog matches its model; those loglosses stand. (b) On this d=80 trunk the QAT env was
*inert* until this morning's fix, so no track-2 accuracy number was ever produced with the bad
catalog in the first place. (c) `CPU_INFERENCE.md`'s q72u figures are **speed** (ms/rev), and
codebook-search cost depends on catalog SIZE, not fidelity — they are unaffected. The finding
therefore changes what the tax measurement should measure; it does not retract anything recorded.
⚠ The one live document that needs editing is the methodology-(a) env string (CLAUDE.md:579), which
still names the q72u WKV catalog — hold that edit until the refit is validated by the probe matrix.

**Cross-user honesty check** (train user 101, test user 102; the deployed catalog is fitted on a
few users and applied to thousands, so this is the analogue that matters): OLD **1.0107** — worse
than the zero codebook — REFIT 0.4610, ORACLE 0.2717. Cross-user generalization is well short of
the within-user 0.3032, and even a good WKV catalog stays far lossier than the shift side's 0.19:
1024 centroids in 32 dims is ~1.24 points per axis. So expect the refit to recover much of the
cost without making WKV quantization free.

**THE REFIT IS DONE — `reference/pq_cb_wkv_c80_b10.txt`** (2026-08-12, ~3 min of CPU: 787 MB corpus
= 47,700 card+note WKV states from 7 train-range users via the Rust engine's `--dump-corpus`, then
joint-uv k-means at bits=10). **Byte-compatible drop-in:** identical header `1 10 32 16 1024` and
1024 rows, so the index budget — and therefore deploy state size — is UNCHANGED. Fitted on card and
note states together because `RWKV_QAT_PQ` is one catalog shared by every scoped stream, and on
users 101-156 (train range), so no eval-set leakage.

Held-out mean relative L2, full corpus, both splits:

| split | OLD (live) | REFIT | ORACLE |
|---|---|---|---|
| hold out USER 156 (cross-user, the honest one) | **1.0051** | 0.3973 | 0.1684 |
| random-vector holdout | **1.0026** | 0.3512 | 0.3137 |

**OLD sits at ~1.00 in every configuration measured** — single-user, cross-user, random-split — i.e.
consistently at or past the encode-everything-to-zero bound. The refit cuts error 60-65%.
⚠ Do NOT read the ORACLE column as a floor: it fits 1024 centroids to the holdout itself (3,205
vectors for user 156 = ~3 per centroid), so it is near-memorization and the two splits disagree
wildly (0.17 vs 0.31) for that reason alone. **REFIT-vs-OLD is the solid comparison**; it is the
same holdout scored by two encoders.
NEXT: the 4-arm PTQ probe matrix (old-WKV / new-WKV / WKV-only / shift-only, 10 users each) turns
this reconstruction win into a logloss number and localizes the tax to one side.

**WKV CAPACITY CURVE (2026-08-12, CPU) — MY "FLAT CURVE" PREDICTION WAS WRONG, and that matters.**
I predicted, in the runner, that more bits would buy ~nothing: 1024 centroids in 32 dims is
1024^(1/32) ~ 1.24 points per axis and 4096 is ~1.30, and that reasoning HELD on the shift side
(2.5x the bits bought ~9%). It does not hold here. Held-out cross-user (user 102 out, 49,070 held-out
vectors, 189,430 training):

| bits | centroids | REFIT held-out err | vs previous |
|---|---|---|---|
| 8 | 256 | 0.4580 | |
| 10 | 1024 (**shipped budget**) | 0.3776 | -17.6% |
| 12 | 4096 | 0.3224 | -14.6% |
| 14 | 16384 | 0.2844 | -11.8% |
| — | OLD q72u, any budget | 1.0107 | (worse than encode-to-zero) |

So the WKV joint-uv scheme is **not saturated** at the shipped budget — unlike the shift scheme.
**COST, so this is not mistaken for free:** index bits are per head per layer, and card is 1 layer x
5 heads, so +2 bits = +10 bits/card ~ **+1.25 B on the frozen 9 B/card budget (+14%)**; note likewise
on 27 B. bits=12 is a genuine state-size trade and is **Andrew's call**, not an adoption I make.
**Also visible: the corpus, not just the budget, limits the refit.** At bits=12 the ORACLE reaches
0.2044 against REFIT's 0.3224, and the training set is only 189k vectors from 6 users (46 per
centroid at 4096). More users in the corpus is a FREE axis — no deploy cost at all — and should be
tried before spending state bits. The dump is CPU-only and reuses existing traces.
⚠ ORACLE remains optimistic throughout (11 holdout pts/centroid at bits=12) and is not a floor.

**TWO PROCESS BUGS THIS RUN, both worth keeping:**
1. **A non-ASCII character killed a data point.** A `⚠` in a print statement raised
   `UnicodeEncodeError` under cmd.exe's cp1252 redirect and aborted the bits=14 arm. CLAUDE.md's
   "plain ASCII in shell-written values" rule applies to **Python that prints into a redirected
   log**, not only to shell-authored strings.
2. **`%ERRORLEVEL%` inside a `for` loop is a LIE.** The bits=14 arm crashed and the runner still
   logged `BITS_14_EXIT_0`, because `%ERRORLEVEL%` inside a parenthesised block expands at PARSE
   time, once, before any iteration runs. Every per-iteration exit-code guard written that way is
   vacuous. Use `setlocal enabledelayedexpansion` + `!ERRORLEVEL!`, or `call :label` per iteration
   (which is what run_arm.cmd/probe_cbs.cmd already do, and why theirs are sound).

**THE "BEFORE" NUMBER IS IN — PTQ with the BROKEN WKV catalog, 405 users (2026-08-12).** The arm was
stopped early to free the GPU at Andrew's request; 405 of 500 users were already scored and
`eval_sharded` writes incrementally, so nothing was lost and the remainder resumes by skipping
completed users. At n=405 the answer is not in doubt:

| mode | plain (iter 45) | PTQ, old catalog | cost | p | users worse |
|---|---|---|---|---|---|
| ahead | 0.297697 | 0.311678 | **+0.012233** | 1.0e-65 | 97% |
| imm | 0.265375 | 0.282399 | **+0.014456** | 6.2e-68 | 99% |

This is the honest baseline the refit has to beat, and it is ~3-4x the +0.00290/+0.00445 that the
d=32 era recorded as the QAT tax — consistent with the catalog being not merely imperfect but
uninformative on this trunk. It is a PTQ cost, so a QAT fine-tune should recover much of it; the
question the three cells answer is how much, and the question the probe matrix answers is how much
of it simply disappears when the catalog is correct.
Resume: relaunch the same arm; the completed users are skipped. **Never delete these jsonls.**

## ★ THE 4-ARM PTQ PROBE MATRIX — where the quantization cost actually lives (2026-08-12, 16 min GPU)

All four arms evaluate the SAME plain iter-45 checkpoint on the same 10 users (5001-5010), changing
only what is quantized. Cost = vs iter 45's plain numbers on those users (ahead 0.281997 / imm
0.263988).

| arm | what is quantized | ahead cost | imm cost |
|---|---|---|---|
| `oldwkv` | q72u WKV + m2b12 shift — **today's recipe** | +0.009276 | +0.012690 |
| `newwkv` | **d=80 refit** WKV + m2b12 shift | **+0.006041** | **+0.008507** |
| `wkvonly` | refit WKV only, shift left fp32 | +0.005174 | +0.006893 |
| `shonly` | m2b12 shift only, WKV left fp32 | **+0.000365** | **+0.000720** |

**1. The refit buys +0.003235 ahead / +0.004183 imm** — 35%/33% of the PTQ cost, **for zero extra
deploy bytes** (identical header, identical 1024 rows). That is a larger effect than any single
algorithmic iteration in this phase, obtained from ~3 min of CPU k-means.

**2. The WKV side is ~14x the shift side** (+0.005174 vs +0.000365 ahead; 10x on imm). **This
settles the m2b12-vs-m5b12 question without running it:** the ENTIRE shift-side cost is +0.0004 /
+0.0007, so m5b12's extra ~14 B/card could recover at most a fraction of an already negligible
term. **m2b12 is the deploy choice**, and the second 87-min arm that was killed to free the GPU
would have been measuring noise. Quantization work belongs on the WKV side.

**3. The two halves are roughly additive, mildly super-additive:** wkvonly + shonly = +0.005540 /
+0.007613 against +0.006041 / +0.008507 measured together, i.e. the pair costs +0.0005 / +0.0009
MORE than the sum. So the errors interact slightly but per-side tuning is meaningful rather than
misleading — worth knowing before optimising either half in isolation.

⚠ n=10 users. Fine for effects of this size (the smallest, shonly, is still ~5x the n=2500 noise
floor and these are paired on identical weights and inputs), but not a gate-grade number.
**NEXT: the three-cell chain is running with `pq_cb_wkv_c80_b10.txt` + m2b12** (launched 14:31,
`scratchpad/qat_tax/launch_chain_m2b12.cmd`, ~11 h). PTQ cost is what QAT has to recover; the d=32
record says a warm-started decay-QAT recovers nearly all of it (precision degradation ~0), and
whether that still holds at +0.006/+0.0085 is exactly what cells 2 and 3 measure.

## DEPLOY STATE BUDGET ON THE CURRENT TRUNK — bits/card, bits/note, and who actually has state
(Andrew asked both questions 2026-08-12)

**PER-ENTITY COST: card = 185 bits (23.1 B), note = 105 bits (13.1 B).**

| stream | layers | WKV | shift | total |
|---|---|---|---|---|
| card | 2 | 2 x 5 heads x (10-bit joint-uv index + 1-bit norm) = 110 b | 3 vectors x (2x12 + 1) = 75 b | **185 b** |
| note | 1 | 1 x 5 x 11 = 55 b | 2 vectors x 25 = 50 b | **105 b** |

Shift vectors per state = one per layer plus one more per layer whose channel-mix is live (card
layer 1 is in `RWKV_STRIP_CMIX`, hence 3 not 4). **The formula is ANCHORED, not derived in a
vacuum:** applied to the d=32 config (H=2, card 1 layer, note 3) it returns exactly **9 B/card and
27 B/note**, which is what `champion_5k.json` records as the frozen deploy truth. Compression vs
fp32 is 484x (card) / 439x (note).

**★ CARD/NOTE DOMINANCE HAS FLIPPED, and the stored guidance was stale.** The old conclusion —
"note state dominates total deploy memory ~4-5x, so the note target matters more than card" — was a
property of the d=32 arch (card 1 layer, note 3). The `_cnd` trunk is **card 2, note 1**, and
reviewed notes are only **0.67x** reviewed cards, so **card state now outweighs note state 3.4:1**
(it was 0.64:1). Any future state-reduction effort should target CARD first on this trunk. This
ratio is an architecture property — re-derive it whenever stream depths change.

**WHO HAS STATE: only entities that have been REVIEWED.** Sizing off collection counts is wrong,
and wrong by ~50x for the users that would set the worst case. 250-user sample: median **5,061 cards
/ 2,616 notes ever reviewed** against a collection median of 8,866 / 7,461 (72% / 56% reviewed).
**The million-card users are imported shared decks, not superhuman studying** — user 629 has
1,256,705 cards and reviewed **25,395 (2.0%)**; user 9528 has 1,170,003 and reviewed **10,974
(0.9%)**. (Andrew's "that's 150 new cards a day for 20 years" was the right thing to disbelieve.)

Resulting per-user footprint (reviewed entities x the bits above):

| user | rev cards | rev notes | card | note | TOTAL | same user, d=32 cfg |
|---|---|---|---|---|---|---|
| median | 5,061 | 2,616 | 114.3 KB | 33.5 KB | **147.8 KB** | 113.5 KB |
| mean | 9,683 | 5,323 | 218.7 KB | 68.2 KB | 286.9 KB | 225.5 KB |
| p90 | 24,829 | 13,480 | 560.7 KB | 172.8 KB | 733.5 KB | 573.7 KB |
| max in sample | 76,312 | 42,267 | 1.68 MB | 0.53 MB | **2.21 MB** | 1.74 MB |

So the d=80 trunk costs ~30% MORE per user than the frozen d=32 deploy config — card state grew
(1 layer -> 2, H 2 -> 5) faster than note state shrank (3 layers -> 1).
⚠ "max in sample" is the max of 250 users, not of 10,000; the global worst case is higher.

### FULL REVIEWED-ENTITY CENSUS — all 9,934 users (2026-08-12, `scratchpad/reviewed_entity_counts.py`)
Per-user CSV: `scratchpad/reviewed_entity_counts_10k.csv` (percentiles recomputable without
re-reading 745M reviews). Definitions: cards reviewed >=1 time = distinct card_id in that user's
revlogs; notes = distinct note_id whose card_id appears in the revlogs.

| metric | mean | median | p90 | p99 | max |
|---|---|---|---|---|---|
| **cards reviewed >=1 time** | 9,582 | **4,826** | 24,074 | 61,059 | **246,801** |
| **notes with >=1 reviewed card** | 5,347 | **2,578** | 13,928 | 36,378 | **76,887** |
| (collection cards) | 20,982 | 8,513 | 55,854 | 128,395 | 1,256,705 |
| (reviews) | 72,990 | 31,070 | 180,883 | 543,161 | 3,910,718 |

Totals: **95,183,694 reviewed cards / 53,118,620 reviewed notes**. Reviewed notes / reviewed cards
= **0.690 median** (vs 0.9x for COLLECTION counts — reviewed notes are relatively scarcer).

**★ THE DEPLOY WORST CASE IS A DIFFERENT USER THAN THE DATASET WORST CASE.** The 1.26M-card user
(629) reviewed 25,395 = 2.0% -> 811 KB of state. The real maximum is **uid 8902: 246,801 reviewed
cards out of a 15,254-card COLLECTION**, with 2.83M reviews — i.e. heavy deck CHURN, cards reviewed
then deleted and replaced. Top-5-by-collection and top-5-by-state are nearly disjoint sets, so any
sizing argument that starts from collection counts targets the wrong users entirely.
Churn is dataset-wide: **10% of users have MORE reviewed card_ids than collection rows** (p90 ratio
2.50); reviewed-but-deleted cards median 569/user, mean 3,245, max 231,547.

**Deploy footprint at 185 b/card + 105 b/note:**

| | mean | median | p90 | p99 | max |
|---|---|---|---|---|---|
| KB/user | 284.9 | **146.1** | 718.8 | 1,765.4 | **5,678.6** |

⚠ The max assumes state is NEVER pruned. In a real Anki deployment a deleted card's state would go
with the card, so 5.68 MB is an upper bound and uid 8902's is largely historical. The honest planning
numbers are the median (146 KB) and p99 (1.77 MB).

### ★ THE QAT WALL-CLOCK TAX AT d=80, MEASURED: 3.8x, not 1.7x and certainly not +13%
Measured 2026-08-12 on the running qtaxc_m2b12 decay (60 steps in 180 s with the CPU otherwise
idle): **0.333 steps/s**, against the plain recipe's **1.253 steps/s** (tuner trial 1, same MAX,
same trunk) = **3.76x slower**. KD is on in BOTH, so the ratio isolates QAT.

This is exactly the re-measurement the endgame section asked for and it lands close to Andrew's
recollection ("IIRC it ended by being like 3x slower") rather than either recorded figure:
- **+13%** was a PROFILED GPU KERNEL SHARE (651 vs 578 ms) at d=32, not wall-clock.
- **1.7x** was d=32 WALL-CLOCK (`champ5k_plain` vs `champ5k_b1`).
- **3.76x** is d=80 wall-clock — and the growth is expected: **per-card state is 2,880 floats here
  vs 576 at d=32 (5x)**, and the fake-quant work (rank-1 SVD + PQ search + norm quant) scales with
  state size, while the rest of the step does not. CLAUDE.md predicted this would need re-measuring
  "at d=80" for precisely that reason.

**CONSEQUENCE FOR THE 10x-BUDGET ENDGAME.** The plan costs the QAT arm at ~1.7x. At 3.76x, a
QAT-throughout 10x run would be ~4x the plain arm, not ~2x. This strengthens the already-recommended
option A (a warm-started ~2-epoch QAT fine-tune on the plain arm's final) over option B (a second
full 10x run with QAT active throughout): B now costs roughly a week of GPU on its own.
⚠ Measured on the DECAY phase with KD reading from the dump; a WS-phase measurement could differ
slightly, but not by enough to change the conclusion.

**⚠ CORRECTION to the chain's own runner comment (2026-08-12): the decay is 10,935 steps, not
2,733.** `decay_ratio` has been **1.0** since iter 34 ("decay is HALF of all training"), and
`qtaxc_m2b12_decay.toml` carries `EPOCHS = 1.0` — byte-identical in that respect to iter 45's decay
toml, so the single-variable comparison is intact; only my step-count estimate was wrong (it came
from the retired 0.25 ratio, which is also where CLAUDE.md's "decay 63 -> 40 min" speedup figure
comes from — that line predates iter 34 and should not be read as the current decay cost).
The comment inside `run_qat_chain.cmd` still says 2733/~1.0 h and CANNOT be fixed while the file is
executing (cmd.exe resumes from a byte offset); fix it after the run.
**Revised budget at the measured 0.336 steps/s:** phase A 9.0 h, cell 2 7.2 h, cell 3 3.0 h =
**19.3 h**, not the ~11 h projected. This does not change any verdict, only the wall clock.
**And it need not be decided in advance: an eval is interruptible and its partial results are
valid** — `eval_sharded` writes incrementally and resumes by skipping completed users, which the
PTQ arm demonstrated today (stopped at 405/500, answer already at p~1e-65). Cutting both evals to
1,000 users would save 6.1 h if the GPU is wanted sooner.

**SWEEP FOR THE SAME BUG ELSEWHERE — clean (2026-08-12).** Having found one artifact that was
validated by shape and wrong by content, the obvious question is whether others are. Checked every
codebook in `reference/`:

| file | header (m bits sub c ncent) | status |
|---|---|---|
| `pq_cb_wkv_q72u.txt` | `1 10 32 16 1024` | **the bug** — K=16 matches at d=80, so it loads and is wrong |
| `pq_cb_wkv_c80_b10.txt` | `1 10 32 16 1024` | the refit; identical header by design |
| `pq_cb_shift_q72u.txt` | `2 12 16 32 4096` | c=32 -> hard-FAILS a shape assert at C=80, i.e. self-announcing |
| `pq_cb_shift_c80_m2b12.txt` | `2 12 40 80 4096` | refit, in use |
| `pq_cb_shift_c80_m5b12.txt` | `5 12 16 80 4096` | refit, unused (shift side is only ~1/14th of the cost) |
| `pq_cb_m2b8.txt` | `2 8 8 16 256` | ALSO K=16-compatible, but referenced only by docs and two d=32-era **kernel** bit-exactness goldens (`qat_parity/parity_lr_pq.py`, `qat_speed/golden_gen.py`) — those test the kernel, not accuracy on this trunk, so they stay self-consistent |

Every `.cmd` still naming a q72u catalog (champ5k_b1/r1/t1, iter9-13, jitab) is **d=32-era**, where
it matches its own model. And `rust/rwkv-infer` applies no codebook unless `RWKV_LOWRANK_PQ` is set,
so there is no baked-in default waiting to be shipped. **Conclusion: exactly one artifact was
silently compatible, and it is now replaced.**

**FREE PREVIEW FROM THE PAIRED TRAINING LOSSES (2026-08-12 23:40, zero GPU).** The QAT decay and
iter 45's plain decay start from the SAME WS checkpoint with the same seed, db and MAX, so their
10,935 steps pair exactly (10,935 paired steps, no gaps). Mean train loss, QAT minus plain:

| window | all | ahead | imm |
|---|---|---|---|
| first 500 steps | +0.01204 | +0.00340 | +0.00774 |
| last 1000 steps | +0.01443 | +0.00385 | +0.00970 |
| last 300 steps | +0.01457 | +0.00398 | +0.00967 |

**The gap is a near-constant OFFSET from step 500 onward, not something the fine-tune closes** — if
anything it widens slightly (plain improves 0.0042 over the decay, QAT only 0.0019). A cost that is
present immediately and never trains away looks like fixed PRECISION DEGRADATION rather than
accumulating MODEL DRIFT — which would **invert the d=32 finding** (decay-QAT #39 was essentially
free: -0.000127 / +0.000018) and would mean the lever is the quantizer, not the training schedule.
⚠ Read this as a direction, not a result: it is TRAIN loss measured UNDER fake-quant, so it is the
analogue of cell 2 alone and cannot produce the split; and it is not the rectified by-user eval
metric the gate uses. Cells 2 and 3 settle it. What it does establish is that the tax will not be
~0 with the refit catalog in place.

### ★★ THE QAT TAX, MEASURED ON THIS TRUNK — and the balance sheet gets WORSE (2026-08-13, partial)
**FINAL, cell 2 complete on the full VAL half n=2,500 (p below float precision; 92% / 98% of
users worse): FULL TAX = +0.004185 ahead / +0.006219 imm.**
(The partial read at n=1,572 gave +0.004238 / +0.006228 — it moved by 5e-5 / 1e-5, so the early
call was sound.)

**FULL TAX = +0.004185 ahead / +0.006219 imm**, against the d=32-era placeholder of +0.00290 /
+0.00445. **The tax is ~1.5x larger on this trunk than the number the balance sheet was carrying**,
and the comparison is if anything generous to us: the old figure came from `champ5k_b1`, which ran
QAT throughout WS+decay, whereas this is the warm-started decay-only placement that the d=32 record
says is the *cheap* one. So the trunk change dominates — 5x the card state, 5 heads rank-1-encoded
instead of 2 — not the schedule.

Recomputing Andrew's stopping rule `still_needed = (champion - old) - budget_credit + QAT_tax` with
the measured term:

| | ahead | imm |
|---|---|---|
| gap to old model (VAL half) | +0.00309 | +0.00181 |
| budget credit at ~12 ep | -0.00373 | -0.00430 |
| QAT tax — was (d=32 placeholder) | +0.00290 | +0.00445 |
| **QAT tax — measured, d=80 (final, n=2500)** | **+0.00419** | **+0.00622** |
| STILL NEEDED — was | +0.00225 | +0.00196 |
| **STILL NEEDED — now** | **+0.00360** | **+0.00374** |

At the post-tuning rate (+0.000112 ahead / +0.000057 imm per attempted iteration) the requirement
goes from **20/34 iterations to 32/66** — ~12 and ~24 days of continuous GPU. The stopping point
moved AWAY.

**★ THE OPERATIVE RATIO: 0.001 off the QAT tax == ~9 algorithmic iterations on ahead and ~18 on
imm** (~3.3 and ~6.5 GPU-days). That is the argument for continuing quantizer work rather than
returning to the algorithmic loop, and it is why the WKV refit — +0.003235/+0.004183 of PTQ cost
recovered for ZERO deploy bytes and ~3 min of CPU — is the highest-value change of the phase so far.
Without it this tax would be materially worse.
⚠ Partial (1,572 of 2,500 users) and cell 3 still pending, so the split between precision
degradation and model drift is not yet in. The train-loss preview says precision; if that holds, the
remaining levers are quantizer-side (catalog capacity, a larger fitting corpus, a different scheme
on the WKV half) rather than schedule-side.

### ★★ CELL 3 INVALIDATES ITS OWN LABEL: "model drift" is not measurable for STRUCTURAL quantization
Cell 3 (the QAT checkpoint evaluated at FULL PRECISION) came back at **0.403091 ahead / 0.547622
imm** (n=858 partial) against plain's 0.297697 / 0.265375. That is not drift, it is a model being
fed inputs it has never seen. Verified NOT a harness bug before interpreting: cell 2 and cell 3 name
the **same** checkpoint in their tomls; the QAT banners are **absent** from cell 3's shard log (0
matches, so quantization really is off); no exceptions or tracebacks; the `nan`s are `auc` on
single-class users. Cell 3 measured exactly what it was asked to.

**The decomposition's hidden assumption.** `(3)-(1) = model drift` presumes the QAT weights are a
valid full-precision model. They are not here. Our config is
`RWKV_QAT_LOWRANK_SCOPE=card:1:int4,note:1:int4` — **rank-1 truncation** of each 16x16 WKV state,
plus PQ codebook and 1-bit norms. Under QAT the recurrence CARRIES the quantized state, so the model
learns dynamics built around rank-1 states; full-rank input is off-distribution. The signature is
that **precision degradation comes out NEGATIVE** (-0.101 ahead / -0.276 imm): the model is *better*
with its quantizer than without. A model that needs its quantizer has no meaningful fp32 score.

**Why the d=32 record disagreed, and why that is consistent.** `qat_log` #39 computed this exact
split (`quant_cost` = cell2-cell3, `finetune_cost` = cell3-cell1) and got +0.000018/-0.000127 and
+0.002456/+0.001129 — sane, and the origin of "drift dominated". But #39 quantized
**`card int2 + note int4`, plain int-N ROUNDING with no low-rank**. Removing rounding is a small
perturbation; removing a rank-1 projection is not. **The three-cell design is sound for VALUE
quantization and breaks for STRUCTURAL quantization.** Andrew's recollection was right about the
experiment he remembered; it does not transfer to this config.

**The split that IS meaningful here** — use the PTQ arm as the second axis instead of an fp32 eval:

| | ahead | imm |
|---|---|---|
| quantization cost, UNADAPTED (PTQ, refit catalog, n=10) | +0.0060 | +0.0085 |
| quantization cost, ADAPTED (**FULL TAX**, n=2500) | **+0.0042** | **+0.0062** |
| **recovered by the QAT fine-tune** | +0.0018 (30%) | +0.0023 (27%) |

There is no separate drift term to report, and none is needed: the deploy number is the full tax.

**WHERE THIS POINTS THE REMAINING WORK.** At d=32 with int-N, QAT *dissolved* the quant penalty to
~0. With rank-1+PQ the residual after adaptation is still +0.0042/+0.0062, so this quantizer is much
harder to adapt to — the lever is its STRUCTURE, not the schedule. Concrete candidate: **rank-2
instead of rank-1** (11 -> 22 bits/head, i.e. card ~23 B -> ~37 B) — a real state-size trade and
therefore Andrew's call, not an adoption. Precedent is favourable: the d=32 ladder found rank-2 int4
BEAT int2, "smaller AND more accurate". ⚠ Also note cell 3 cost ~3 h of GPU to produce a number that
cannot be used; a 10-user version would have exposed the negative-degradation signature in minutes.
**Cheap-probe-before-long-run applies to VALIDITY, not just to cost.**

### ⚠ INCIDENT 2026-08-13: an incomplete kill produced a symptom that looked like an algorithmic finding
The learnable-codebook run (`run_cblearn.cmd`) advanced **8 steps in 55 minutes** and I diagnosed it
as "the learnable shift codebook is computationally infeasible" — 1.31M trainable centroids in a
non-fused PyTorch path, a plausible story. **It was wrong.** With the GPU exclusive the same run
does **0.3333 steps/s, identical to the plain QAT decay's 0.333** — learnable codebooks are free in
wall-clock. The stall was a co-tenant `get_result` from the previous chain that I had failed to
kill, i.e. the documented WDDM-paging deadlock at ~11.7/12 GB.

**Three transferable lessons:**
1. **Killing a `cmd.exe` wrapper does NOT kill its python children.** They survive, reparented, and
   a process-tree walk rooted at the (now dead) wrapper no longer finds them. **Kill deepest-first
   and then VERIFY BY COMMAND-LINE**, never by tree membership alone — `Get-CimInstance Win32_Process
   ... | Where CommandLine -match 'rwkv|eval_sharded|get_result'` is the check that actually closes.
2. **The runners RETRY.** `run_qat_chain.cmd` / `run_arm.cmd` launch a second eval attempt on
   nonzero exit, so killing an eval's python while the wrapper lives makes the wrapper *relaunch* it.
   A partial kill can actively resurrect the work you meant to stop.
3. **★ A resource-contention symptom impersonated an algorithmic property.** "1 step/hour" reads
   exactly like "this feature is too expensive", and acting on it would have silently changed the
   experiment (dropping shift-codebook learning) on the basis of an artefact. **Before concluding
   that a feature is slow, confirm the GPU is exclusively yours** — `nvidia-smi` plus a command-line
   survivor check costs seconds. This is the same class as the qat-inert and stale-codebook bugs
   found this week: a measurement that is self-consistent and answers a question you did not ask.

**⚠ WAITLOOP TRAP, THIRD VARIANT (2026-08-13): an APPENDED log lets a PREVIOUS run's terminal token
satisfy the wait.** `cblearn.log` is opened with `>>`, so the aborted first attempt's
`DONE_EXIT_23` sat in the file when the relaunch began; a watcher grepping `^DONE_EXIT_` fired
instantly and reported the run finished. The documented fix for the earlier variant — ANCHOR the
pattern with `/B` so prose cannot match — does not help here, because this IS a real terminal line,
just from a run that is over.
**The fix that works: scope the search to lines AFTER the current run's START marker.**

    awk '/CHAIN START/{buf=""} {buf=buf$0"\n"} END{printf "%s", buf}' run.log | grep -qE "^DONE_EXIT_"

Three variants of this trap are now on record — prose mentioning the token, `DONE_EXIT_WSFAIL`
satisfying a `DONE_EXIT` grep, and now a stale token from a previous append. The general rule:
**a completion signal must be unambiguous in BOTH content and TIME.** Either scope to the current
run (above), or write per-run log files (a fresh `%STAMP%` name), which the phase logs already do
and the summary logs do not.

### ★★ LEARNABLE CODEBOOKS: the QAT penalty is largely CATALOG STALENESS *WITHIN* THE RUN
Paired train-loss screen at 4,367 of 10,935 steps (three runs from the same WS-final, same seed,
same batch order: plain / QAT-fixed-catalogs / QAT-learnable-catalogs). Penalty = QAT minus plain.

| steps | ahead: fixed | learn | closed | imm: fixed | learn | closed |
|---|---|---|---|---|---|---|
| 0-999 | +0.00334 | +0.00218 | 34.6% | +0.00766 | +0.00514 | 32.9% |
| 1000-1999 | +0.00388 | +0.00152 | 60.8% | +0.00914 | +0.00389 | 57.4% |
| 2000-2999 | +0.00461 | +0.00159 | 65.6% | +0.01055 | +0.00400 | 62.1% |
| 3000-3999 | +0.00470 | +0.00156 | **66.7%** | +0.01077 | +0.00401 | **62.7%** |

**The raw penalty columns carry the mechanism, and they CORRECT an earlier reading.** With FIXED
catalogs the penalty GROWS monotonically (+0.0033 -> +0.0047 ahead, +0.0077 -> +0.0108 imm); with
LEARNABLE catalogs it is FLAT (+0.0022 -> +0.0016, +0.0051 -> +0.0040). On 2026-08-12 I read the
widening fixed-catalog gap as "the QAT fine-tune is not recovering the cost". That was wrong. The
real cause: **training moves the model's state distribution, so a catalog fitted before the run goes
progressively stale DURING it.** Learnable catalogs track the drift; fixed ones fall behind.

**This unifies the week's two quantization findings into one root cause** — a catalog fitted to one
state distribution and applied to another. Across model generations that produced the
worse-than-random d=32 catalog; within a single run it produces the growing QAT penalty. Both are
fixed by making the catalog follow the states rather than freezing it.

⚠ Train loss under fake-quant, not the rectified by-user eval; screens rank, they do not gate. Valid
here because all three runs share quantizer structure, regularization and seed (the documented
train-loss-prune bias applies to REGULARIZATION levers). **Conditional projection, to be confirmed
not quoted:** a ~65%/63% cut would move the tax from +0.004185/+0.006219 to roughly
+0.0015/+0.0023, and `still_needed` from +0.00360/+0.00374 to roughly +0.0009/+0.0000.

### ★ WS:DECAY SPLIT FOR THE 10x ENDGAME RUN = **10+2** (Andrew asked Claude to decide, 2026-08-13)

**The evidence that appears to favour a long decay is CONFOUNDED — this corrects a belief I had been
carrying.** iter 34's `decay_ratio 0.25 -> 1.0` (+0.00145) was never a pure ratio change;
`research_5k_verbose.md` records the caveat at the time: it also took **total training from 1.25 to
2.0 epochs**. Against the log-linear budget curve (calibrated on the measured 3x step, +0.00196),
pure budget explains **+0.00084** of it. So the ratio itself is worth at most ~**+0.0006, from a
single confounded point**. **We have NO matched-budget evidence that a long decay beats a short
one**, and "our tuning prefers ratio 1.0" should not be quoted as if we did.

**QAT length does not constrain the choice.** With learnable catalogs the penalty closure saturates
by step ~4,000 = **0.37 epochs**, so even a 1.5-epoch decay gives QAT ~4x what it needs. QAT only
*looks* coupled to the decay because the runners happen to enable it for exactly that phase; the two
are separable and should be reasoned about separately.

**So the decision is made on cost, where the spread is large** (arm 2's QAT window scales with the
decay, at the measured 3.76x QAT slowdown):

| split | arm 1 plain | arm 2 QAT | total |
|---|---|---|---|
| 6+6 | 30.3 h | 58.9 h | 89.2 h |
| 8+4 | 30.3 h | 40.8 h | 71.1 h |
| **10+2** | 30.3 h | **22.7 h** | **53.0 h** |
| 11+1 | 30.3 h | 13.6 h | 43.9 h |

**10+2 chosen:** 36 h cheaper than 6+6 for a benefit we cannot demonstrate; it is standard WSD
practice (decay 10-20% of total); and it is the configuration upstream actually used at ~12 epochs,
i.e. a known-good point rather than an extrapolation from our unusual 1-epoch regime. 11+1 is
rejected — it saves 9 h but drops below the conventional range with nothing behind it.
⚠ **The +0.0006 excess is being SPENT, not disproven.** If it should be settled instead, the
de-confounding test is two runs at a FIXED 2-epoch budget — 1+1 vs 1.6+0.4 — for ~10 h plus evals.
Against a one-shot 53 h run that is a defensible insurance premium; it is Andrew's call whether the
delay is worth it.

### ★★★ QAT IMPROVEMENT #1 CONFIRMED: LEARNABLE CODEBOOKS CUT THE TAX ~45% (FINAL, n=2500)
Single-variable vs `qtaxc_m2b12`: the ONLY difference is `RWKV_QAT_PQ_LEARN=1` +
`RWKV_QAT_SHIFT_PQ_LEARN=1`. Same WS-final, same KD alpha 0.5, same starting catalogs, same
schedule. Paired on the same users:

| | plain | tax, fixed cb | tax, LEARNABLE cb | cut | p |
|---|---|---|---|---|---|
| ahead | 0.297697 | +0.004185 | **+0.002286** | **45.4%** | 4.3e-269 |
| imm | 0.265375 | +0.006219 | **+0.003486** | **43.9%** | < float precision |

**FINAL on the full VAL half (n=2500).** The effect was stable at every sample size measured --
10-user probe 44.7%/39.3%, n=569 47.0%/43.0%, n=1186 46.1%/43.0%, n=2500 45.4%/43.9% -- so the
early reads were trustworthy and the partial-eval habit is vindicated again.
**Costs nothing:** deploy size is unchanged (catalog VALUES move, not structure) and wall-clock is
unchanged (0.3333 vs 0.335 steps/s). It also closes the export->eval wiring gap CLAUDE.md had
listed as queued -- the eval consumed the catalogs exported at the final checkpoint.

**Effect on the stopping-point balance sheet:**

| | ahead | imm |
|---|---|---|
| still_needed with the d=32 placeholder tax | +0.00225 | +0.00196 |
| still_needed with the measured fixed-cb tax | +0.00360 | +0.00374 |
| **still_needed with learnable catalogs (FINAL)** | **+0.00165** | **+0.00100** |

At the post-tuning algorithmic rate that is **~15 and ~18 more iterations, not 32 and 66** (~5.5
and ~6.5 GPU-days instead of ~12 and ~24). **imm stops being the binding mode**, which it became
only because of the QAT term.
Nothing here changes the deploy recipe -- this is the SAME quantizer per Andrew's 2026-08-13
constraint, trained better. **Recommend adopting learnable catalogs as the DEFAULT for every
quant-aware run, and especially for the 10x endgame**, where the state distribution moves far more
over 12.5 epochs than over 1 and a frozen catalog would decay correspondingly further.

### ★★ WHERE THE REMAINING QAT TAX LIVES — and a THIRD stale-artifact finding (2026-08-13, CPU only)
Reconstruction-error ladder on real card/note WKV states (7,500 head-states each), decomposing the
frozen recipe's error into its three stages:

| stage | CARD | NOTE |
|---|---|---|
| exact rank-1 only (the STRUCTURAL floor) | 0.4353 | 0.3049 |
| + codebook directions | 0.6623 (+0.2269) | 0.5383 (+0.2334) |
| + 1-bit norm | 0.8148 (+0.1525) | 0.7813 (+0.2431) |
| **share: rank-1 / codebook / norm** | **53% / 28% / 19%** | **39% / 30% / 31%** |

**★ THE NORM RANGE IS STALE FOR THE NOTE STREAM.** The engine quantizes **√σ** log-uniformly over a
**hardcoded** [-3,0] octaves (`rwkv_ops.py` -> `rwkv7_set_norm_quant(cb, bits, -3.0, 0.0)`), a range
the Rust side documents as *"corpus-derived globals (2026-07-04: WKV √σ spans log2 [-2.5,-0.6])"* —
i.e. derived on the **d=32** model. On this trunk:

| stream | log2(√σ) p1 / median / p99 | inside [-3,0] | 1-bit norm err: hardcoded -> FITTED |
|---|---|---|---|
| card | -3.09 / -1.93 / +0.02 | 96.6% | 0.2748 -> 0.2757 (no gain) |
| **note** | **-3.52 / -2.79 / -1.49** | **67.5%** | **0.5113 -> 0.1756 (2.9x better)** |

**32.4% of note norms are CLAMPED to one value.** Card is fine, which is precisely why a single
global range survived unnoticed — it was fitted to card-like states. This also explains the ladder
above (norm = 31% of note's error vs 19% of card's).
⚠ **Method note: my FIRST pass used the wrong quantity** (σ₁/amax, which is ≥1 by construction, so
no [-3,0] window could ever contain it) and reported "100% outside". Reading the engine's own
comment gave the right quantity. Verify what a quantizer quantizes before diagnosing its range.

**RANKED NEXT STEPS, all inside the frozen recipe (Andrew 2026-08-13):**
1. **Per-stream norm ranges** — zero extra bits (two floats), ~3x less norm error on note. Requires
   `norm_lo_log2/norm_hi_log2` to become per-scope instead of global. **Testable by a 10-user PTQ
   probe with no training at all.**
2. **Learnable norm levels** — the generalization of 1: learn the 2 levels instead of fitting a
   range, exactly as the catalogs were just learned. Zero bits, subsumes 1.
3. **Per-stream catalogs** — card and note are visibly different distributions (rank-1 floor 0.435
   vs 0.305) sharing one catalog. Per-entity INDEX bits unchanged; the d=32 endgame already found
   catalog size is free because it amortizes.
4. **Low-rank-friendly regularization** — rank-1 truncation is the largest single term (53%/39%) and
   is structurally frozen, but the model can be TRAINED to emit more nearly rank-1 states. The only
   lever that touches the dominant term. Deploy unchanged, no extra forward.
5. fp32-teacher KD during QAT — classic, now behind the cheaper in-budget items.

**★ THE PATTERN, third instance this week:** an artifact fitted to one state distribution and applied
to another — stale catalog ACROSS generations (worse than random), stale catalog WITHIN a run (the
growing QAT penalty, fixed by learnable catalogs, ~45%), and now a stale norm RANGE across
generations. **Anything in the quantizer that was ever "corpus-derived" must be re-derived when the
trunk changes, and is better learned than frozen.**

**⚠ TWO SELF-INFLICTED BUGS IN THE IMMCORR HELPER (2026-08-14), both cheap but worth the rule:**
1. **A Windows path inside GENERATED file content became control characters.** The toml was written
   through a shell heredoc whose header comment contained `C:\rwkv_kd_dump\t128_seedpair_65k`; the
   `\r` and `\t` were emitted as a literal CR and TAB, and tomli died with
   *"Found invalid character '\r' (at line 2, column 6)"*. **Never put backslash paths in generated
   file content** -- use forward slashes, or write the file with the Write tool instead of a
   heredoc. (Verify with `sorted({c for c in open(f,'rb').read() if c<32 and c!=10})` == [].)
2. **The helper runner did not gate on its exit code**, echoing `DONE_EXIT_0` unconditionally. So a
   hard toml failure reported success, the chain proceeded, and the analysis ran against an empty
   dump. The repo rule *"gate every phase on exit codes AND artifacts"* was written for the long
   training chains; it applies to three-line helper runners too, and this is the second time this
   week a runner reported success for work that did not happen (the first was the QAT-inert banner).
Both fixed; the dump runner now gates on `%ERRORLEVEL%` AND on step files existing.

### ★★★ THE 1-BIT NORM IS ~HALF THE QUANTIZATION COST (norm probe, 2026-08-14, 12 min GPU)
Three PTQ arms, same plain iter-45 checkpoint, refit catalogs, 10 users, only `RWKV_QAT_NORM_BITS`
changing:

| | norm OFF (exact fp32) | **1 bit (DEPLOY)** | 2 bits |
|---|---|---|---|
| ahead cost vs plain | +0.002997 | **+0.006041** | +0.003774 |
| imm cost vs plain | +0.004941 | **+0.008507** | +0.005718 |

**The 1-bit norm alone costs +0.003044 ahead (50% of the whole PTQ cost) and +0.003566 imm (42%).**
**One more bit recovers ~75% of it** (+0.002267 / +0.002789).

**★ RECONSTRUCTION ERROR MISPREDICTED THIS, in the UNFAVOURABLE direction.** The ladder put the norm
at 19% (card) / 31% (note) of reconstruction error; in logloss it is ~half the cost. Third
mispredict of the week -- shift catalog (9% error gap -> ~0.0004 logloss, over-predicted), WKV
catalog (collapse -> ~0.006, under-predicted), now the norm. **Treat reconstruction error as a
generator of hypotheses to probe, never as a proxy for logloss, in EITHER direction.**

**THE PRICE OF THE SECOND BIT:** +1 bit per head per layer = **+10 bits/card, +5 bits/note** ->
185 -> 195 b/card (23.1 -> 24.4 B, **+5.4%**) and 105 -> 110 b/note (+4.8%). At the measured
algorithmic rate that +0.0023/+0.0028 is worth roughly **20 and 49 iterations**. ⚠ It DOES break
Andrew's 2026-08-13 "keep the current quantization recipe" constraint, so it is his call -- but the
ratio is lopsided enough that not surfacing it would be the error.

**ORDER OF WORK (free first):**
1. **Per-stream norm ranges** -- zero bits. Offline: note's 1-bit norm error 0.5113 -> 0.1756 (2.9x)
   from fitting the range; 32.4% of note norms currently clamp.
2. **Learnable norm levels** -- zero bits, the same move that bought 45% on the direction catalog.
   The norm grid is FIXED even in the learnable-codebook run (catalogs learn DIRECTIONS only), so
   this is complementary to that win, not overlapping.
3. Only if 1-2 leave most of the +0.003 on the table, put the +1 bit trade to Andrew.
⚠ n=10 users -- fine for ranking levers of this size, but confirm on ~500 before any state-size
change.

> **★★ SUPERSEDED SAME DAY -- ITEMS 1 AND 2 ARE CLOSED, AND THE REASONING ABOVE IS THE INSTRUCTIVE
> PART.** Everything in this block was measured under FROZEN catalogs. Under learnable catalogs the
> 2-bit norm came back NULL, and the mechanism falsifies item 2's stated premise directly: **learned
> centroids are not unit-norm.** They absorb magnitude into their own length (length spread widened
> **2.43x**), so "catalogs learn DIRECTIONS only" stopped being true the moment they became learnable,
> and the norm grid is no longer the sole carrier of magnitude. Item 1 falls the same way -- a range
> fitted per stream re-encodes what centroid length already encodes -- and item 3's +1-bit trade
> should be re-priced against the LEARNABLE baseline before ever being put to Andrew, since the
> +0.003 it was meant to buy was measured against the frozen one. Item 4 (rank-1 regularization) is
> unaffected and is the live `qtaxf_r1reg` run.
> **The transferable lesson, which is why this block is kept rather than deleted: a learnable
> component silently absorbs the levers around it.** The offline reconstruction numbers above were
> not wrong -- they were measured on a configuration that no longer exists. Before adding a lever
> beside something that learns, measure what it has ALREADY absorbed.

**⚠ THE FIRST imm-independence RESULT WAS VOID -- masking bug, caught by a sanity number (2026-08-14).**
It printed a verdict ("the independence mechanism is real, incremental R^2 = 0.0100") that must NOT
be quoted: **`p_curve` is valid ONLY on ahead rows (`has_label & ~is_query`) and `p_imm` ONLY on
query rows (`has_label & is_query`) -- DISJOINT row sets.** The eval joins the two heads by
`label_review_th` (srs_model.py ~1352); my script masked both to `has_label` and so compared
predictions from DIFFERENT reviews.
**The tell was a sanity line I had put in for exactly this reason: "our ahead 1.98285" against an
eval value of 0.298.** A 6.6x impossible number is what stopped the verdict being believed --
without that row the analysis would have looked plausible and shipped a wrong conclusion. Keep
printing quantities whose correct magnitude is known.
Fixed: the dump now stores `label_review_th` and the analysis joins ahead-row -> query-row per
review (the teacher shares the row layout exactly, both dumps walking the identical stream).
Re-queued behind the 2-bit-norm run -- it needs a fresh dump and must not run co-tenant with it.

**⚠ TRANSIENT CUDA FAULT, NOT A 2-BIT BUG (2026-08-14).** The first 2-bit-norm launch died at the
step-50 validation with `CUDA error: an illegal memory access` on validation user 5008. The relaunch
walked the SAME users and passed cleanly (5008 -> 0.346), so **it did not reproduce even under
`RWKV_DETERMINISTIC=1`**. Treat isolated illegal-access crashes here as transient first and re-run
once before diagnosing -- this machine also has a history of hard black-screen hangs, so GPU-level
transients have precedent. ⚠ Determinism flags do NOT protect against them: they fix the numerics,
not the hardware.

**LATENT HAZARD FOUND WHILE DIAGNOSING (unproven as the cause, but real).** The learnable-codebook
backward does
`atomicAdd(&g_pq_cb_grad[ci * c_pq_subdim + x], ...)` guarded only by `ci >= 0` -- **there is no
upper-bound check** -- into `__device__ float g_pq_cb_grad[32768]`, a fixed buffer that exactly fits
the joint-uv shape (1024 centroids x 32 dims = 32,768). So ANY out-of-range centroid index writes out
of bounds, and a NaN in the distance search (argmin over NaNs returns garbage) is a plausible route
to one.

> **⚠ AUDITED 2026-08-14 -- THIS WAS OVERSTATED, and the real defect is a different one.** Reading the
> kernel end to end: `ci` **cannot** be out of range. It is `sc_best`, which is either a `c` from the
> `c < c_pq_ncent` scan or the literal 0 all-bad fallback (`rwkv7_cuda.cu:536,579`), the recording is
> initialized to `-1` **unconditionally before any degenerate-norm bail** (`:482-485`), and upload
> already enforces `n == want` and `n <= 32768` (`:1180-1181`). The NaN route is closed too: an
> all-NaN distance set hits the `fi == 0x7fffffff` fallback to centroid 0. So (a) and (b) as written
> were not the risk, and chasing them would have been wasted work.
> **THE REAL LATENT DEFECT, narrower and genuinely silent:** the recording buffer is
> `__shared__ int rec_idx_chunk[CHUNK_LEN * 8]` -- a **hardcoded 8-int stride per timestep** -- but
> the kernel writes `2*c_pq_m` entries per slot (`:483`) and reads `[c*8 + c_pq_m + p]`
> (`:1114,:1136`). **Role-mode `m > 4` spills slot `c` into slot `c+1`**, so one timestep's centroid
> picks overwrite the next one's: wrong gradients, no fault, no warning. Joint mode is exempt
> (`m == 1`, enforced kernel-side). It binds only on the LEARN path, which is exactly the path now
> adopted as default, and `m` is a natural thing to raise when experimenting with catalog structure.
> **Guarded at upload in `rwkv_ops.maybe_upload_pq_codebook`** -- deliberately Python-side, so it
> needed no `.pyd` rebuild and could not disturb the live chain -- plus a comment at the declaration.
> **The lesson is about the audit, not the bug: a hazard reported from a single suspicious line was
> wrong in both directions** -- it named a risk the surrounding invariants already excluded, and
> missed a real one three lines away. Trace the invariant to its source before filing the hazard.

**EARLY SIGNAL (step-50 validation, identical batch):** 2-bit norm **0.3259 / 0.3101** vs the 1-bit
run's 0.3269 / 0.3110 -- the expected direction, though 50 steps proves nothing on its own.

### ★★ THE 2-BIT NORM BUYS ~NOTHING UNDER LEARNABLE CATALOGS -- and the PTQ probe over-promised
Paired train-loss screen at 2,980 of 10,935 steps (both runs from the same WS-final, same seed and
batch order; the ONLY difference is `RWKV_QAT_NORM_BITS` 1 -> 2):

| window | ahead pen 1-bit | 2-bit | closed | imm pen 1-bit | 2-bit | closed |
|---|---|---|---|---|---|---|
| 0-999 | +0.00218 | +0.00208 | 4.7% | +0.00514 | +0.00489 | 5.0% |
| 1000-1999 | +0.00152 | +0.00161 | **-5.5%** | +0.00389 | +0.00393 | **-1.1%** |

Noise around zero in both modes. The same screen showed the learnable-catalog change at 34.6% in its
FIRST window and 60.8% in its second, so it resolves effects of this size easily.

**★ WHY THE PTQ PROBE OVER-PROMISED (+0.003044 / +0.003566, "half the quantization cost").** It
measured a model that had never trained under quantization. Under QAT with **learnable** catalogs the
centroids are free parameters and are NOT unit-normalized, so the catalog can carry magnitude in
centroid LENGTH and substitute for norm precision. The coarse norm is absorbed. This also predicts
the gap should SHRINK with further training (the catalogs keep adapting), not grow.

**THE GENERAL LESSON, and it is the counterpart to the reconstruction-vs-logloss one: PTQ cost does
not predict QAT cost when another part of the quantizer can adapt to compensate.** The norm penalty
was real for a frozen model and largely illusory for a trained one. Any future "component X is N% of
the cost" probe must be re-asked under QAT before it justifies spending deploy bits.
=> **the +5.4% card-state bit looks unjustified**; recommend NOT adopting the 2-bit norm.
⚠ Train loss at 27% of the run, not an eval. The run continues (killing it would idle the GPU while
the free fixes are implemented -- both need kernel changes) and will land a definitive number.

### ★★★ THE LEARNABLE CATALOG ALREADY DOES PER-STREAM MAGNITUDE ADAPTATION -- norm/catalog levers CLOSED
Two CPU checks, prompted by the 2-bit norm null and done BEFORE building per-stream norm ranges.

**1. The catalog spends real capacity on LENGTH.** Learned (10,935 steps) vs its k-means init:

| | joint-norm min / med / max | spread | std/med |
|---|---|---|---|
| initial | 1.0867 / 1.3005 / 1.4116 | 1.30x | 0.0454 |
| learned | 1.1178 / 1.5165 / 2.1281 | 1.90x | **0.1101 (2.43x wider)** |

Median centroid grew 16.9% (p90 +37%), and the movement splits about evenly into DIRECTION
(med 19.85 deg) and LENGTH (med 16.9%). The catalog is encoding magnitude in centroid length --
exactly what a 1-bit norm is too coarse to express.

**2. Card and note use DISJOINT parts of the catalog**, so one shared table can specialise per
stream: card 501 distinct centroids, note 220, **only 25 shared (3.6%)**; Bhattacharyya overlap
**0.0219**; shared probability mass **0.78%**.

**=> THREE LEVERS CLOSE AT ONCE, on mechanism rather than on a null result:**
* **per-stream norm ranges** -- the catalog already adapts per stream AND per centroid, which is
  strictly finer than a per-stream range;
* **learnable norm levels** -- same mechanism, same redundancy;
* **per-stream catalogs** (my own earlier item 3) -- the streams are ALREADY disjoint in one table.
A fourth signal agrees: only **721 of 1024 centroids are used at all**, so catalog CAPACITY is not
binding either. This also retro-explains the 2-bit null and the sibling's "norm axis bottoms out at
1 bit": 1 bit is an interior optimum because the catalog covers what the norm cannot -- confirmed
now from BOTH sides (below 1 bit = +0.004 cliff; above 1 bit = nothing).

**WHAT REMAINS: the term the catalog provably cannot touch.** Rank-1 truncation is 53% (card) /
39% (note) of the reconstruction error and is structurally frozen by the recipe -- but the MODEL can
be trained to emit more nearly rank-1 states. A penalty on the state's off-top-singular energy is
training-only, needs no kernel or deploy change, and is now the top remaining QAT lever;
fp32-teacher KD during QAT is second.
⚠ Corpus states come from the PLAIN model and are encoded with the LEARNED catalog; under QAT the
states shift. The disjointness (0.78% shared mass) is far too stark for that to overturn, but the
centroid-usage counts themselves are from 900 states/stream and would rise with more data.

### RANK-1 REGULARIZER (iter qtaxf_r1reg, launched 2026-08-14 14:15) -- and Andrew's objection
`RWKV_QAT_RANK1_REG=0.05`, single-variable vs `qtaxd_cblearn`. Training-only, no kernel change, no
deploy change. Penalty = 1 - max(conc(k), conc(v)) where conc = the participation ratio
||M^T M||_F^2 / ||M||_F^4 (1 for rank-1, 1/r for r equal directions).

**Why a proxy on (k,v) and not on the state:** the true state S lives inside a custom CUDA autograd
Function whose backward accepts only the OUTPUT gradient, so a loss on the returned checkpoints
would not reach the weights at all; the differentiable Python path materializes S but loops per
timestep (T up to 65536). Since S is a decay-weighted sum of outer products k_i v_i^T, it is near
rank-1 exactly when the per-head k (or v) vectors align -- computable from the kernel INPUTS.
Verified on synthetic cases: all-parallel k -> penalty 0.000000; random k,v -> 0.9209 against the
theoretical 1-1/K = 0.9375; gradients flow; skip rows excluded; **inert when the env is unset**.

**★ ANDREW'S OBJECTION (2026-08-14), which is the right one to raise:** *"if making states more
rank-1 could lower log loss, the model would learn to do that anyway."*
**The mechanistic counter: the STE is structurally blind to the truncation error.** The
straight-through estimator passes dL/dS backward AS IF the truncation were identity, so the model
sees the truncation's consequence in the forward loss but its gradient never contains a term for
"reduce the truncation error" -- only "given this mangling, do better". SGD cannot find what the
gradient does not point at; this is exactly why quantization-friendly regularization exists as a
technique. ⚠ But the objection retains force and the proxy is blunt (k-alignment is sufficient, not
necessary, and ignores the decay weighting), so a null is quite plausible. Prior: low.

**THE DECISIVE INTERMEDIATE MEASUREMENT (~10 min CPU, once a checkpoint exists).** Dump states from
the new checkpoint and re-measure the exact-rank-1 floor (today: **card 0.4353 / note 0.3049**):
* floor does NOT move -> the regularizer is inert; kill early; the objection is untested.
* floor moves but logloss does not -> **Andrew's objection is CONFIRMED empirically**: the model was
  already as rank-1 as is useful, and the STE blindness does not bind.
* both move -> the STE-blindness mechanism was real.
This is what makes the run worth its 18 h even under a low prior: every outcome answers something.

**★ EARLY READ AT STEP 225 (2026-08-14 14:28) -- THE PROXY IS MOVING HARD AND THE TASK LOSS IS NOT.**
Both runs start from the SAME WS-final checkpoint and are deterministic single-variable twins, so at
step 0 the gap in the logged `all` is EXACTLY the penalty term (lambda x penalty, lambda = 0.05).
Reading it off the paired logs, no extra compute:

| step | cblearn `all` | r1reg `all` | diff | => penalty | => max concentration |
|---|---|---|---|---|---|
| 0 | 0.676316 | 0.692383 | +0.016067 | 0.3213 | 0.679 |
| 5 | 0.807554 | 0.825557 | +0.018003 | 0.3601 | 0.640 |
| 50 | 1.080769 | 1.094608 | +0.013839 | 0.2768 | 0.723 |
| 100 | 0.956799 | 0.968196 | +0.011397 | 0.2279 | 0.772 |
| 200 | 0.628888 | 0.633642 | +0.004754 | 0.0951 | 0.905 |

(past step ~0 the diff also carries a little task-loss divergence, but at step 200 that is only
~0.0009 on ahead, so the penalty dominates.)

**Three things this already establishes.** (1) **The regularizer is NOT inert** -- it drives its own
objective 0.321 -> 0.095, a 3.4x reduction, in 200 of 10,935 steps. So outcome (a) above is already
ruled out and a null cannot be explained away as "the lever never engaged". (2) **The states were far
from rank-1 to begin with**: concentration 0.679 at the champion checkpoint, not the ~0.95 one might
assume from "rank-1 truncation is cheap". (3) **The task loss is unmoved while the proxy collapses**
-- at step 225 ahead is 0.1773 vs the control's 0.1782 and imm is identical to 3 dp. If that holds to
the gate, it is **Andrew's objection confirmed, in a STRONGER form than he stated it**: not merely
"the model would have learned it anyway", but "the states can be driven MUCH more rank-1 and the task
loss does not care" -- i.e. rank-1-ness is achievable, cheap, and worthless, so the STE blindness is
real but does not bind on anything that matters.
⚠ **What this does NOT yet establish, and why the state dump still runs:** the penalty falling is the
r1reg model's own trajectory; the control's penalty is never computed (lambda = 0 there), so part of
the fall could be the decay phase's natural drift rather than the regularizer. The dump measures the
exact-rank-1 floor for BOTH checkpoints and settles it. Also, a proxy at 0.905 concentration need not
translate into a lower truncation floor -- the proxy ignores the decay weighting, which is exactly
its known blunt edge.
⚠ **If the floor DOES move and logloss still does not, do NOT re-run at a larger lambda** -- that
would be the same experiment with more force behind an answer already given. A larger lambda is only
indicated in the other branch (floor unmoved despite the proxy collapsing), which would mean the
proxy and the floor are decoupled and the LEVER, not the dose, is wrong.

**★★ THE BASELINE FOR THAT CHECK WAS WRONG, AND IT WOULD HAVE MANUFACTURED A FAKE WIN (2026-08-14).**
Building the measurement tool (`scratchpad/qat_tax/rank1_floor.py`, CPU, ~1 min, now saved and
re-runnable) and pointing it at the EXISTING iter-45 corpus does not reproduce the ladder's
"card 0.4353 / note 0.3049". It gives:

| | card | note |
|---|---|---|
| ladder (2026-08-13, ad-hoc script, 7,500 head-states, NOT saved) | 0.4353 | 0.3049 |
| **`rank1_floor.py` (183,480 / 55,020 head-states, 7 users)** | **0.3733** | **0.3729** |

**The disagreement is not sampling: it goes in OPPOSITE directions on the two streams** (card lower,
note higher), and per-user card means span only 0.353-0.439. The new number is the trustworthy one --
its formula, `sqrt(1 - sigma_1^2 / sum sigma_i^2)`, was checked against an EXPLICIT rank-1 truncation
and agrees to **2.4e-07** -- while the ladder's script no longer exists to interrogate.
**Why this mattered more than a bookkeeping fix:** the decisive check was going to compare the r1reg
floor against **0.4353**. Any r1reg value near the true control (~0.373) would then have read as a
**-0.06 improvement, i.e. a large regularizer win that is entirely a change of metric** -- and it
would have landed in the same write-up as a null logloss, producing the maximally confusing
conclusion "the states became much more rank-1 and it bought nothing", *stated with a fabricated
magnitude*. The lesson is the cheap one: **re-measure the baseline with the same tool you will
measure the candidate with, before the candidate exists.**
⚠ **One downstream claim loses its stated support:** per-stream catalogs were ranked partly on "card
and note are visibly different distributions (rank-1 floor 0.435 vs 0.305)". They are NOT different
on this measure -- 0.3733 vs 0.3729. That lever is still closed, but on the disjoint-centroid
evidence (0.78% shared mass), which is independent and measured.

**★ AND THE SECOND HALF OF THE SAME MISTAKE, caught one step later: THE CONTROL WAS THE WRONG
CHECKPOINT.** With the baseline metric fixed, the first end-to-end run compared `qtaxf_r1reg_d_50`
(50 steps into the regularized decay) against **iter45's FINAL** corpus. Paired entity-for-entity
(12,268 card / 3,547 note states, identical counts) it looked emphatic -- card 0.3681 -> 0.3328,
note 0.3538 -> 0.2767, i.e. -10% and -22% relative after only 50 steps. **But those two checkpoints
also differ by 10,885 steps of ordinary training**, so the gap is not attributable to the
regularizer. The matched control is `qtaxd_cblearn_d_50` -- same WS-final, same step, differing
ONLY by `RWKV_QAT_RANK1_REG` -- and it exists, as does `qtaxd_cblearn_d_10935` for the final
comparison. The runner's built-in control phase was removed; it now measures ONE checkpoint and the
comparison is made by diffing two runs.
**The pattern across both catches is one sentence: a difference is only attributable to the variable
you changed if the comparison holds everything else fixed -- the METRIC (same tool) and the
TRAJECTORY (same step).** Each was individually capable of manufacturing a large fake effect, and
they compound.
⚠ I briefly wrote down a third claim from the confounded pair -- "decay training pushes states toward
HIGHER rank over time" -- and it does not survive either: the matched control `qtaxd_cblearn_d_50`
reads 0.3831 card against iter45-final's 0.3681, i.e. the opposite direction, and those two differ in
CONFIG (QAT + learnable catalogs) as well as step. **There is no supported trajectory claim here in
either direction; do not quote one.** Confounded comparisons do not just inflate the effect you were
looking for, they also generate confident side-observations that are artifacts.

### ★★ THE RANK-1 REGULARIZER DEMONSTRABLY MOVES THE DEPLOY QUANTITY (matched, 2026-08-14)
`qtaxf_r1reg_d_50` vs `qtaxd_cblearn_d_50` -- same WS-final, same step 50, same 3 users, same tool,
differing ONLY by `RWKV_QAT_RANK1_REG=0.05`. Identical state counts (12,268 card / 3,547 note), so it
is paired entity-for-entity:

| stream | control | r1reg | delta | relative |
|---|---|---|---|---|
| card | 0.3831 | 0.3328 | -0.0503 | **-13.1%** |
| note | 0.3556 | 0.2767 | -0.0789 | **-22.2%** |

**This settles the branch the whole check existed to settle, and it settles it EARLY.** The
regularizer does not merely drive its own proxy (k/v alignment, penalty 0.3213 -> 0.0951 over 200
steps); it moves the **exact rank-1 truncation error of the state** -- the quantity the deploy
quantizer actually pays -- by 13% and 22% relative after **fifty steps**, with 10,885 still to run.
**Consequences, both worth acting on:**
1. **"The regularizer was too weak / try a larger lambda" is DEAD as an explanation** for any null
   that follows. It engaged, hard, at lambda=0.05. A null now means the thing itself does not pay.
2. **The gate therefore tests exactly one proposition**, cleanly: *does making the states markedly
   more rank-1 improve the quantized logloss?* If the eval comes back null, that is **Andrew's
   objection confirmed empirically in its strong form** -- rank-1-ness is achievable, cheap in
   training terms, and worthless -- and the STE-blindness counter-argument is refuted as a
   *practical* matter even though it remains true as a *mechanical* one (the gradient really does
   lack the term; the term simply is not worth having).
⚠ **Do NOT convert -13% / -22% reconstruction into an expected logloss gain.** That is precisely the
error the week has already made three times in both directions (shift catalog over-predicted, WKV
catalog under-predicted, norm bits under-predicted ~2.6x). Reconstruction generates hypotheses; only
the eval scores them.

**★★ THE PENALTY IS NOT MONOTONE -- IT BOTTOMS OUT AND CLIMBS BACK (step 6,800 read, 2026-08-14
20:00).** Paired window means of `r1reg - cblearn` on the logged training loss (the twins share a
checkpoint and their task components are ~identical, so `d_all / lambda` estimates r1reg's own
penalty):

| step window | d_all | ~penalty | d_ahead | d_imm |
|---|---|---|---|---|
| 1-200 | +0.01087 | 0.2175 | +0.00001 | -0.00001 |
| 200-600 | +0.00351 | 0.0702 | +0.00001 | -0.00006 |
| 600-1200 | +0.00161 | 0.0323 | -0.00004 | -0.00019 |
| **1200-2400** | +0.00125 | **0.0249 (min)** | -0.00003 | -0.00030 |
| 2400-4000 | +0.00179 | 0.0357 | -0.00005 | -0.00027 |
| 4000-5600 | +0.00294 | 0.0588 | +0.00001 | -0.00008 |
| 5600-6800 | +0.00323 | 0.0646 | -0.00004 | -0.00003 |

**Reading it.** (1) The states are driven hard toward rank-1 (0.32 at step 0 -> 0.025 by ~1,800), and
then the system gives ground back: the penalty is **2.6x its minimum** by step 6,800 and still
rising. At fixed lambda that is a moving equilibrium, and the plausible driver is that this recipe is
non-stationary by construction -- **the PQ catalogs are LEARNING** (cblearn is the base), so the
quantization error surface the task loss sits on keeps changing underneath. (2) The task components
are flat throughout: `d_ahead` within +/-5e-5 all run, `d_imm` reaching -0.0003 mid-run and returning
to ~0. On TRAIN loss, under quantization, the regularized model is indistinguishable from its twin.
**★ THIS RETRACTS AN EXTRAPOLATION I MADE FROM THE STEP-50 RESULT.** I reported the floor moving
-13% / -22% "with 10,885 steps still to run", which implied it would move further. The penalty
trajectory says the opposite is likely: most of the rank-1 structure was imposed in the first ~1,800
steps and is being partially surrendered thereafter. **The step-50 floor is therefore NOT a
lower bound on the final model's floor, and must not be quoted as the run's result.** The final
matched measurement (`qtaxf_r1reg_d_10935` vs `qtaxd_cblearn_d_10935`) is mandatory, not
confirmatory. The step-50 number retains exactly one job: proving the lever ENGAGES, which kills the
"try a larger lambda" escape hatch regardless of where the floor ends up.
⚠ The flat `d_ahead`/`d_imm` predict a null at the gate, but train loss is not eval logloss and this
is a quantized forward -- treat it as a prior, not a result.

**PRE-REGISTERED DECISION RULE (written 2026-08-14 17:30, BEFORE the eval exists).** Recording this
now because the floor result is suggestive and the temptation to pick a favourable framing afterwards
is exactly what pre-registration is for.
* **Comparison:** `qtaxf_r1reg` vs **`qtaxd_cblearn`** -- the single-variable twin, both quant-aware,
  both on the VAL half 5001-7500. NOT vs iter 45 (that is the plain-vs-QAT tax, a different question)
  and NOT vs the old frozen-catalog arms. Paired jsonls are already on disk for the control.
* **Gate:** the STANDARD both-modes research rule -- raw improvement >= 0.0001 in BOTH ahead and imm,
  each with paired one-sided Wilcoxon p < 0.0001. **The curve-side exception does NOT apply**: this
  lever changes the WKV state itself, i.e. the shared trunk, so it genuinely can move both modes and
  must be held to both. (Contrast iter 46, where `.detach()` and an untouched `p_loss` made the
  one-sided rule mechanically justified.)
* **What a null means:** given the floor DID move, a null is a substantive result, not a failed
  experiment -- Andrew's objection confirmed in its strong form. File it as such, close the
  low-rank-friendly-regularization family (item 4 of the ranked list), and do NOT retry at a larger
  lambda; the dose is not the issue.
* **What a WIN means:** the STE-blindness mechanism is real AND worth exploiting, which would make
  lambda a tunable and open the sub-family (schedule it like KD, apply it only in decay, etc.).
* **The seed-pair doctrine still binds:** a margin under ~0.0005 needs the recipe re-run at
  `RWKV_AUGMENT_SEED=4321` before it is leaned on. In-seed p-values measure per-user consistency,
  never cross-seed robustness.
