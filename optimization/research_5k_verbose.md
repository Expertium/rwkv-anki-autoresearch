# 5k phase — verbose per-iteration notes (AI-only)

Companion to [research_5k.md](research_5k.md) (whose `summary` column is capped at 20 words,
Andrew 2026-07-13). This file holds the full reasoning/ops trail per iteration; the
machine-readable source of truth is `research_log.jsonl` (`note` fields). Andrew doesn't need to
read this file — it exists so no context is lost between sessions.

## iter 0 — d=128 target (adopted)
Old d=128 leaderboard model (`pretrain/RWKV_trained_on_101_4999.pth`), unquantized; the fp target
to beat on 5001–10000. Evaluated 2026-07-03, n=5000 both modes, full precision: ahead 0.296385 /
imm 0.264905. Consistent with the published 10k-pooled 0.29743/0.26600.

## iter 1 — champ5k_r1 (invented, accepted)
The first 5k champion (starting point). H=2/K=16, quant-aware q72u with per-run learnable cbs,
champion HPs, 2ep WS + 0.5ep decay. Behind the iter-0 fp target by 0.0102/0.0134 — the gap the
phase closes. Promoted 2026-07-08, superseded by iter 2. Pipeline wall-clock ~7.0h; two latent
bugs fixed en route (LEARN=1 optim resume param-group mismatch at the WS→decay seam f71f43b;
per-user lmdb env leak killing eval shard 0 at user 2007, 7d095e3).

## iter 2 — champ5k_b1 (invented, accepted — CURRENT champion)
Iter 1's recipe at HALF budget: WS 1ep (6554) + 0.25ep decay (1638). vs iter 1 paired: ahead
−0.000058 (p=0.31, indistinguishable), imm +0.000430 BETTER (p=6.1e-62) — the 2nd epoch adds
nothing (data-variety lesson holds at 5k). SIZE/SPEED accept; 1-ep budget ADOPTED for all 5k runs
(Andrew 2026-07-08). Promoted 2026-07-09. Pre-ship note: the final champion should get ONE
full-budget (2 ep) confirmation run.

## iter 3 — champ5k_t1 (invented, rejected)
The hp_tuner_5k winner (wd 0.01→0.2 + dropout_scale 1.0→0.5; 20-trial coordinate descent on
tune-eval 5001–5200) at the standard budget. REJECTED: its +0.0008/+0.0010 subset win INVERTED at
n=5000 (−0.000545/−0.000677 vs iter 2, p=1.0 both) — the descent overfit the 200-user subset
(even an in-subset paired p=5e-8 didn't transfer). Champion HPs confirmed vs 19 alternatives; HP
tuning CLOSED for the phase; future tuning uses a 1000-user tune-eval (Andrew 2026-07-12).

## iter 4 — lad_deck1 (invented, rejected)
State-size ladder, deck rung 1: deck n_heads 2→1 at fixed d_model (deck per-entity WKV state
1.89x, −264 params; card/note unchanged). REJECTED: no gain — ahead −0.000271 / imm −0.000238 vs
iter 2 (p=1.0 both, ~1σ of zero = no effect). The deck stream is not state-capacity-limited;
deck knob closed.

## iter 5 — lad_preset1 (invented, rejected)
Ladder, preset rung 1: preset n_heads 2→1 (preset per-entity WKV state 1.89x, −198 params).
REJECTED: ahead −0.000215 / imm −0.000445 vs iter 2 (p=1.0 both) — the long-recurrence prior
didn't materialize; H=1 free state 0/2 at this point. Ops: the 2-parallel-shard eval wedged (WDDM
oversubscription from preset-K=32 chunk-state buffers, ~+0.8 GB/shard on 1M-token batches; both
shards 50–85+ min on mega-users at 11.5/12 GB); fixed via get_result per-shard resume run
sequentially then eval_sharded relaunch-skip-merge (run_lad_preset1_evalfix.cmd, 5338c49).

## iter 6 — lad_user1 (invented, rejected — near-miss)
Ladder, user rung 1: user n_heads 2→1 (user per-entity WKV state 1728→3264 floats, −198 params).
REJECTED but the first rung with a REAL signal: ahead +0.000345 (clears ≥0.0003), imm +0.000258
(misses the bar by 0.000042), both overwhelmingly significant (p 1.3e-20 / 1.5e-29; deck/preset
were p=1.0). The user stream IS the state-sensitive one, as the blanket-quant lesson predicted.
WS validation tracked the champion within noise the whole run (the 10-user val set can't see a
0.0003 effect). Wall-clock: WS 2h26m, decay 37m, sequential eval 2h24m; no incidents.

## iter 7 — lad_user2 (invented, rejected — mode trade)
Ladder, user rung 2: iter 6 + user layers 3→4 (user state 4352 floats = 2.52x champion, +10.4k
params, 203,928 total). REJECTED — a mode TRADE: ahead −0.000299 WORSE (p=1.0), imm +0.000604
better (p=7.8e-143). vs iter 6: ahead −0.000643 / imm +0.000346 — the 4th user layer buys imm
calibration at ahead's expense. Attribution: user state↑ → +ahead; user depth↑ → +imm −ahead.

## iter 8 — lad_user1b (invented, rejected — null seed-pair)
Seed-pair test of iter 6: exact user-H=1 recipe at RWKV_AUGMENT_SEED=4321. REJECTED — NULL:
ahead −0.000044 / imm −0.000146 vs iter 2 (p 0.88/1.0) — the deck/preset no-effect signature.
Iter 6's seed-1234 signal did not replicate → substantially seed luck (in-seed p measures
per-user delta consistency, NOT cross-seed robustness). Cross-seed spread on the same recipe
≈0.0004 both modes → margins <~0.0005 default to seed-pair confirmation. STATE-SIZE LADDER
CLOSED, 0 accepted rungs — no stream is state-capacity-limited at d=32/H=2. Widened vprune
(0.006/0.008) ran clean across the seed change.

## iter 9 — iter9_sp (adopted: Ash & Adams 2020, rejected)
Shrink-perturb init: init = 0.5·champion_final + 0.5·fresh seeded draw (RWKV_INIT_BLEND hook;
zeros/ramps preserved), else exact champion recipe. REJECTED — worse both modes: ahead −0.000744
/ imm −0.001033 vs iter 2 (p=1.0 both), beyond the ~0.0004 seed noise = real harm. The warm init
led the val curve all WS (−0.010 at step 1000 → −0.0006 at 3500) but ended net negative at full
eval — at fixed 1-ep budget on the same data, λ=0.5 inheritance neither keeps the champion basin
nor explores freely; both λ-endpoints are champion-level so the interior is a dip → λ probe not
worth GPU. Scheme A rejected; family DEPRIORITIZED, not closed (conduct rule 5, Andrew
2026-07-13: closing a family needs 3–5 in-family variants); scheme B (permutation init) queued
LOW. The RWKV_INIT_BLEND hook stays (eed7cb5, env-gated, plain path untouched).

## iter 10 — iter10_kd (invented: Andrew's unsourced idea, rejected)
Warmup-only KD from the d=128 teacher: first 800 WS steps on annealed mixed targets α·teacher +
(1−α)·hard (α linear 1→0) from a stored 800-step dump with a per-step labels-checksum pairing
guard (mismatch = exit 43); hard labels after; RWKV_KD_MIX cleared before decay (decay replays
the epoch-0 stream, checksum can't catch a misfire there). REJECTED — worse both modes: ahead
−0.000277 / imm −0.000329 (p=1.0 both). Trajectory = iter 9's exactly: led val early
(−0.0026/−0.0046 at step 500, still leading at 1500), washed out by WS end, finished slightly
negative. EARLY-TRAINING-INTERVENTION family 0/2 → DEPRIORITIZED, not closed (conduct rule 5:
closing a family needs 3–5 in-family variants); so far head starts don't survive 6554 hard-label
steps — untried variants if revisited: longer/never-zero KD window, KD extended into decay,
permutation init. KD machinery stays in-repo (RWKV_KD_DUMP_OUT / RWKV_KD_MIX, 78caceb).
Ops: the parallel eval wedged on the CHAMPION arch (both shards frozen 66+ min at 11.7/12 GB,
100% util, full-core CPU each — two mega-users collided; the iter-5 elevated-VRAM-only scoping
was too narrow). Killed tree + sequential-resume evalfix (run_iter10_kd_evalfix.cmd). RULE:
ALL evals now run sequential shards (~45 min slower than clean parallel, never wedges).

## iter 11 — iter11_gemb (invented: Andrew's unsourced idea, rejected)
Additive grade embedding: x = features2card(f) + grade_onehot @ E, E 4×32 ZERO-INIT bypass
around the shared input MLP (a literal one-hot→embedding swap is a no-op — the first Linear
already embeds the 4 grade columns; the bypass frees grade info from the fc→32 squeeze; matmul
form keeps ahead-mode query rows at exactly zero). RWKV_GRADE_EMB=1 hook in srs_model.py, +128
params (193,852); else exact champion recipe. REJECTED — worse both modes: ahead −0.000851 /
imm −0.000908 (p=1.0 both), ~2x beyond cross-seed noise = real harm, no seed-pair needed. NOT a
near-miss (rule 2 doesn't force a variant). Val looked champion-level all run — the harm only
showed at full eval. Interpretation: the unregularized linear bypass injects the one-hot
straight into the trunk all 5 streams share, skipping the MLP's SiLU/LayerNorm; plausibly
distorts the shared representation more than it helps (grade was never bottlenecked — 4 of 92
dims through a 128-wide fc is plenty). GRADE-REPRESENTATION family 0/1, deprioritized (rule 5);
untried variants: per-stream additive embeddings (+640 params), grade embedding into the SRS
heads instead of the trunk, LayerNorm on the bypass. First run under the all-sequential-eval
rule: clean, ~5.6h. Hook stays in-repo (env-gated, default off = byte-identical).

## iter 12 — iter12_hres (invented, rejected)
SRS-head resolution 64→128 (RWKV_NUM_CURVES=128 + RWKV_NUM_POINTS=128): capacity re-test at 5k
data of the 100u exp1 reject (that "capacity adds fail" lesson was data-limitation-scoped). Pure
params (+16.5k → 210,236 ≤ 225k cap), ZERO state cost, Rust auto-derives head dims from weight
shapes; else exact champ5k_b1 recipe. REJECTED — no effect: ahead −0.000270 / imm −0.000241 vs
iter 2 (p=1.0 both), magnitude inside the ~0.0004 cross-seed band = the deck/preset no-effect
signature. The 100u lesson does NOT flip at 5k for this lever: 64 basis curves / 64 sample
points are enough resolution for the forgetting-curve mixture. Val trace sat at champion parity
the whole run (WS-end +0.0003/+0.0010), fully consistent with the null. CAPACITY-AT-5K family
0/1 so far — channel mixer 1.0→1.5 is the next in-family variant (iter 13). Wall-clock: WS
2h32m, decay 38m, sequential eval 2h24m (~5.6h), no incidents (second clean run under the
all-sequential-eval rule).

## iter 13 — iter13_cmix (invented, rejected) — LAST QAT-ERA ITERATION
Channel mixer factor 1.0→1.5 (RWKV_CHANNEL_MIXER_FACTOR=1.5, per-block FFN width): the second
capacity-at-5k variant. Pure params (+14.3k → 208,060), zero state cost; else exact champ5k_b1
recipe. REJECTED — no effect: ahead −0.000159 (p=0.999) / imm −0.000271 (p=1.0), inside the
~0.0004 cross-seed band. CAPACITY-AT-5K family 0/2 (SRS-head resolution, channel mixer): the
d=32 trunk is not capacity-limited at 5k in the heads or the FFN width — the d=128 gap
(+0.0102/+0.0134) lives elsewhere (plausibly stream width/recurrent capacity, which the H=1
state ladder also failed to buy). Val led mid-WS (to −0.0026 ahead at 4500) and washed out by
WS end — another washout instance. Clean ~5.6h, no incidents.

## METHODOLOGY SWITCH (2026-07-14, after iter 13) — plain screening + two tracks
Andrew's decisions, prompted by the "why 5.6h?" audit (upstream rwkv unchanged since vendoring —
the time was ours): (1) **QAT PARKED until the end of research** — screening is plain-vs-plain
bf16 (saves ~2h20m/run: quant-aware step 1.41 s vs 0.385 s plain); ONE quant-aware run of the
final champion at close; no per-accept quant confirmations. champion_5k.json (QAT deploy truth)
is frozen; plain screening champion → champion_5k_plain.json (promote --out flag added).
(2) **Power-user-aware eval** (implemented, first E2E = champ5k_plain): users ≥1M work (56 =
11.3% of eval work; top-7 ~2.1M each) run solo first, then 2 parallel LPT shards — worst
concurrent pair halves vs the wedge scale; expected ~1.8x over sequential, ~11% off unrestricted
parallel. eval_sharded.py rewritten (solo phase + RWKV_EVAL_SHARD_DIR override; dry-run tested);
--solo-threshold 0 restores old behavior. (3) **Track 2: ablate d=128** — retrain the old arch
through the current pipeline as anchor A0 (MAX=66000 fits 12 GB; the upstream 12-ep .pth is not
budget-comparable), then cut params; gate = 50k·ΔLL/Δparams ≤ 0.0001 BOTH modes. Context: the
whole d=128→d=32 collapse cost 0.0002/50k ahead / 0.00026/50k imm, so the bar demands cuts
~2–2.6x more efficient than the global average. Alternate ~12h blocks (~5 track-1 iters vs 1
track-2 iter). Track 2 needs its own vprune ref (A0's val trace; pairing needs identical MAX/db)
and an env-based arch-module selector (to avoid the KD-dump file-swap footgun) — build at A0
launch. (4) 1-ep-budget check at d=128 rides along free: if A0 ≈ the 12-ep upstream number, the
budget lesson transfers to 14x params.

## iter 14 — champ5k_plain (invented, ACCEPTED — the plain screening champion)
champ5k_b1's exact recipe with all QAT env stripped (plain bf16, JIT on, no codebooks), step+val
trace on, no vprune (it IS the new reference). **Finals: ahead 0.303734 / imm 0.273448**;
paired vs champ5k_b1 = **the QAT tax at n=5000: +0.002896 / +0.004445 (p=0.0 both)**. Gap to
the d=128 upstream target shrinks from +0.0102/+0.0134 to +0.0073/+0.0085. Promoted →
champion_5k_plain.json (ckpt champ5kplaind_1638.pth + 6554-step WS trace + val trace = the
plain vprune ref); champion_5k.json (QAT deploy truth) frozen. Wall-clock 3h07m: WS 91 min
(0.82 s/step wall = 1.7x faster than quant-aware), decay 22 min, eval 75 min — FIRST E2E of the
power-user-aware phased eval, flawless: solo 56 users in 9 min (first mega-user 3.9 GB/81%
util), phase B two shards 64 min at ~1.8 GB combined VRAM (no wedge exposure), merge exact
(1.9x over the 145-min sequential QAT eval). En-route fix committed: the iter-11
RWKV_GRADE_EMB hook crashed JIT-on model construction (TorchScript resolves attributes in dead
branches; hidden all QAT era by NO_JIT) → @torch.jit.ignore indirection, smoke-tested both
hook states. train_rwkv swallowed that traceback with exit 0 — the .cmd's decay-setup artifact
gate caught it (keep gating phases on artifacts, not exit codes).

## Track 2 — A0 anchor (2026-07-15): d=128 retrained at the 1-ep plain budget

**ANCHOR — ahead 0.299857 / imm 0.269030 (n=4993, eval 5001–10000).** The original d=128
arch (2,762,884 params, `RWKV_ARCH_MODULE=scratchpad/architecture_old_d128.py`) retrained
through the exact plain track-1 recipe: 1 ep WS (22,346 steps @ 1.07 s/step, 6h40m) +
0.25 ep cosine decay (5,586 steps, ~1.6h), seed 1234, **MAX=32768 = the track-2 standard**
(66000 and 49152 both thrash 12 GB at d=128; max single batch in train_db_5k_h1 = 16,384
tokens → zero data drop at any MAX ≥ 16,384). Anchor json (val trace = track-2 vprune ref):
`optimization/champion_5k_track2.json`; ckpt `scratchpad/track2_a0/t2a0d_5586.pth`.

**Key numbers (intersection-paired, n=4993):**
- vs upstream 12-ep `.pth` (base5k): **+0.003714 ahead / +0.004376 imm worse, p≈0** — the
  1-ep budget tax at d=128. Contrast d=32, where the 2nd epoch added nothing (champ5k_b1
  A/B): the 14×-param model keeps learning from reshuffled data. Track-2 ablations are
  measured against A0, so this tax is structural to the track, not a bug.
- vs champ5k_plain (d=32, 193,724 params, same budget): **−0.003637 / −0.004163 better** —
  what 2.57M extra params buy at matched budget; the descent A1, A2, … will map where that
  0.004 actually lives.

**⚠ NaN instability of the 1-ep d=128 model (7 users skipped, n=4993):** users 6701, 6810,
7873, 8060, 8746, 9501, 9813 — the model emits NaN logits on eval chunks ≥ ~500k tokens
(smallest failing: 502,886; content-dependent, not pure length — 6810's first 1M chunk
passed, its second failed). The upstream 12-ep .pth evals all 5000 users clean, and d=32
models never NaN → property of the SHORT-BUDGET d=128 training (MAX=32768 never exercises
the >32k-token recurrence regime; decay params presumably sit near the no-decay edge for
some channels). Skips are recorded in `result/RWKV-track2_a0.nanskip.jsonl`; ALL track-2
comparisons use the finite-user intersection. fp32-vs-bf16 probe deferred (LMDB batches are
stored bf16; needs a cast shim) — queued behind iter 15.

**Pipeline fixes banked en route (all committed):** RWKV_EMPTY_CACHE_WINDOW (whole-run
per-step clears; the d=128 allocator envelope creeps to WDDM paging past the old 1000-step
guard window — launch 4 died at 4.3 s/step, launch 5 at every=50 saturated 11.9 GB by step
250); write_decay_setup MAX param (its hardcoded 110000 thrashed the decay phase);
get_result re-raises instead of swallowing crashes to exit 0, NaN-skips users whole (no
partial rows — partial stats would change equalized size) with skip-file resume;
eval_sharded completeness gate (merged + nan-skipped must equal rostered, ahead set == imm
set, else exit 3). Reproducibility note: step-50 and step-1000 vals were IDENTICAL across
launches 4/5/6/7 — the seeded shuffle + guard cadence are numerics-neutral; and vals are
only comparable at the same step (a step-50 val misread as step-1000 caused a false alarm).

## Iter 15 — drop the review-state input feature (2026-07-15): ACCEPTED (directed), new plain champion

**ahead 0.303663 / imm 0.273227 (n=5000, complete, 0 NaN-skips) — NOT worse than champ5k_plain;
in fact slightly better in both modes** (paired: ahead +0.000071 p=1.5e-08, imm +0.000221
p=1.6e-42 — below the 0.0003 gate and inside the ~0.0004 cross-seed band, but consistently
positive per-user: `scaled_state` was ~noise for the model). **Andrew's directive** (2026-07-14):
remove the Anki review state (Filtered/Review/Learn/Relearn) from inputs and accept regardless
of delta — a deploy simplification (Anki doesn't need to compute/supply review state).

**Mechanism:** `RWKV_ZERO_FEATURES=22` (new generic env hook, srs_model.py + srs_model_rnn.py):
zeroes listed input dims at the model input in train AND eval — a constant-zero column is
informationally identical to removal (the input FC's bias absorbs it) while LMDBs, batch layout
and params (193,724) stay untouched; deploy feeds 0 for dim 22. Plain-tensor-attr +
`@torch.jit.ignore` applier (ScriptModule forbids non-persistent buffers; a persistent one would
pollute state_dict). Dim map: `data_processing.CARD_FEATURE_COLUMNS`[22] = `scaled_state`
(= state − 2), confirmed against the grade-emb 9:13 rating precedent.

**Consequences:** new plain champion → `champion_5k_plain.json` (ckpt iter15d_1638.pth + WS/val
traces = the track-1 vprune ref). **ALL future track-1 runs AND the final QAT confirmation run
must set `RWKV_ZERO_FEATURES=22`** — it is now part of the champion recipe. Exact champ5k_plain
recipe otherwise; WS 6554 steps, decay 1638, phased eval 75 min (solo mega-users clean — the
d=32 model has no trace of the d=128 NaN instability); pipeline 3h09m.

### A0 NaN probe result (2026-07-15 14:20): weight-level, NOT a bf16 artifact

fp32 GPU eval of user 9501's failing 502,886-token chunk (RWKV_EVAL_CAST_FP32=1 shim — LMDB
batches are stored bf16) **NaN'd identically**. The 1-ep d=128 model's long-horizon instability
is in the weights, not the precision: some channels' effective decay admits state growth that
overflows even fp32 within ~500k steps. Structural to the short-budget anchor; the per-user
NaN-skip + finite-intersection comparison handling stands. (En-route fix: get_result's teardown
sort_jsonl now exists-guards — a nanskip-only run never creates the result files.)

## Iter 16 — prehead output gate (2026-07-15): REJECTED (null)

**ahead 0.303652 / imm 0.273409 (n=5000)** — vs iter15: +0.000011 (p=0.97) / −0.000182 (p=1.0)
= the no-effect signature. `x * (2·sigmoid(Wx+b))` between prehead norm/dropout and the three
heads (zero-init = exact identity at start, +1,056 params): the shared readout is not
gating-limited. READOUT family 0/1. Hook stays (`RWKV_PREHEAD_GATE`, default off).

**Two infra lessons banked (the run took 3 attempts):** (1) a `@torch.jit.ignore` method must
NOT call a SUBMODULE — invoked through scripted code the ignored body sees the raw C++
ScriptModule (`'torch._C.ScriptModule' object is not callable`) and train_rwkv's NaN-except
turned every step into a silent skip = a HOLLOW run; caught by the monitor's exception spam.
Parameters + `F.linear` is the safe form (proven by iter15's feat-mask full run); the dormant
grade_emb hook had the same latent bug, fixed. (2) root-level direct Parameters are invisible
to `selective_cast`'s module walk (the root skip protects the fp32-excluded heads) → the bf16
child kept fp32 gate params and `copy_downcast_`'s dtype assert killed attempt 2 pre-step-1;
root-level non-excluded Parameters now cast explicitly. Smoke v2 now exercises the SCRIPTED
forward path AND the selective_cast + copy_downcast_ chain — v1 (direct Python calls only)
missed both failure modes.

### Iter 16 — prehead output gate (REJECTED 2026-07-15 17:17)

(Recorded in the front table; TorchScript infra lessons in CLAUDE.md CURRENT STATE. Null verdict:
ahead +0.000011 p=0.97 / imm -0.000182 p=1.0 vs iter15 — the shared readout is not gating-limited.)

### Iter 17 — direct binary-recall loss term (REJECTED 2026-07-15 20:32): a real MODE TRADE

**Idea ("train what you measure"):** the benchmark's imm metric is the BCE of 1−P(again) at query
rows (`p_binary_loss` in srs_model). It was computed as a wandb statistic but NEVER entered the
training loss (which optimizes the 4-way rating CE + curve BCE + aux terms). Iter 17 added
`+ 0.5 * mean(p_binary_loss over query rows)` (RWKV_PBIN_SCALE=0.5, instance-float hook —
TorchScript reads instance attrs, not env/globals; 0 new params; exact iter-15 recipe otherwise).

**Finals (n=5000, 0 NaN-skips): ahead 0.303885 / imm 0.272840** — vs iter15 champion:
**imm +0.000387 BETTER (p=1.7e-173, clears the ≥0.0003 bar); ahead −0.000222 WORSE (p=1.0)** →
REJECT on the both-modes gate. The first NON-null track-1 effect of the plain era: loss
reweighting genuinely moves the imm metric, but pays for it in ahead — shared-trunk capacity
shifts from the curve head toward the rating/binary objective. The WS val trajectory showed the
same signature live (imm led at most checkpoints, up to −0.0016 at step 4500; ahead oscillated
around/behind parity; decay-end val 0.3260/0.3078).

**Family: LOSS-REWEIGHTING 0/1, with a real effect — variants queued (conduct rule 2):**
RWKV_PBIN_SCALE=0.25 (halve the pressure; hope: keep ~half the imm gain at ~no ahead cost), or
pbin + AHEAD_SCALE up-weighted to rebalance. Run after the directed iter 18 (duration ablation)
and the track-2 A1 block. Clean pipeline: WS 91 min (never vprune-threatened), decay 22 min,
phased eval 76 min. Hook stays env-gated, default off.

### Iter 18 — review-duration ablation (directed, REJECTED 2026-07-15 23:45): duration is real signal

**Andrew's directive:** drop the review-duration input (dim 8, scaled_duration) alongside the
already-dropped review-state (dim 22) — RWKV_ZERO_FEATURES=8,22 on the exact iter-15 recipe.
**Directed gate: accept iff BOTH modes degrade ≤ 0.0003** (mirror of the add-gate threshold).

**Finals (n=5000, 0 NaN-skips): ahead 0.305465 / imm 0.275640 = +0.001802 / +0.002413 worse
than iter15 — REJECTED at 6–8× the tolerance.** Since query rows already zero duration (it is
answer-derived), this measured purely the HISTORICAL-duration contribution to the sequence
encoding — and it is large. Slow answers mark weak memories; no other input feature recovers
that signal. Deploy keeps feeding duration (trivially available in Anki). Unlike iters 9–13,
the persistent ~+0.002 joint val deficit was an honest predictor of the final verdict — val
gaps mean something when they are consistent across the whole run rather than oscillating.
Champion recipe stays RWKV_ZERO_FEATURES=22. Feature-ablation family: 1 accept (state,
~free) / 1 reject (duration, harmful to drop).

### Track-2 A1 — all channel mixers → 1.0 (ACCEPTED 2026-07-16 10:57): new track-2 champion

**Target choice:** the five streams' channel mixers (cmf 2.0 card/deck/note/preset, 1.5 user)
held 972,800 params = 35% of A0's 2,762,884 — the single biggest coherent block, and track-1
had already shown mixer width contributes ~nothing at 5k data (iter 13, d=32). Cut all to 1.0
via `scratchpad/track2_a1/architecture_d128_cmix1.py` (RWKV_ARCH_MODULE): **2,320,516 params
(−442,368)**. Exact A0 recipe otherwise (1 ep WS + 0.25 decay, MAX=32768 everywhere,
EMPTY_CACHE_EVERY=1 WINDOW=0, unsharded eval).

**Gate math (per-100k, both ≤ 0.0001 required; Δparams 442,368 ⇒ allowed degradation
0.000442/mode):** on the n=4993 finite intersection vs A0 (paired_pvalue --intersect):
ahead 0.299768 = **+0.000089 BETTER** (p=2.0e-4); imm 0.269070 = +0.000040 worse (p=1.0).
Ratios: **ahead −0.0000201, imm +0.0000090** — imm used 9% of the budget, ahead is negative
(free win). ACCEPTED with ~50× margin. Full-eval finals (all 5000): 0.300009/0.269324.

**Findings:** (1) the d=32 mixer lesson TRANSFERS to d=128 — FFN width is dead weight at 5k
data regardless of scale; (2) **A0's NaN instability is GONE** — 0 NaN-skips over all 5000
users (A0: 7 mega-chunk users) — either the narrower mixers remove the overflow path or the
retrain lottery landed stable weights; future track-2 gates can pair on full n=5000; (3) val
trajectory: behind A0 only in the first ~1000 steps (mixer capacity mostly matters early),
then parity/trade to the end; decay-end val IDENTICAL (0.3225/0.3040 vs 0.3225/0.3041).
Timing: WS 6h37m @ 1.07 s/step (same as A0 — mixer FLOPs weren't the bottleneck), decay
1h38m, eval 2h35m. Promoted → champion_5k_track2.json (A2's "before" + vprune ref).

**A2 queue by expected ratio-efficiency:** user 4L→3L / deck 4L→3L (~149k each; the
user-stream H=1 near-miss at d=32 hints long-recurrence streams have slack), LoRA-dim cuts,
d_model 128→96 (bigger surgery, keep for later).

### Iter 19 — pbin at scale 0.25 (REJECTED 2026-07-16 14:20): dose-response closes the pbin lever

**Hypothesis (conduct rule 2, from iter 17):** halving the binary-recall loss pressure
(RWKV_PBIN_SCALE=0.25) might keep part of iter 17's real imm gain (+0.000387) while shedding
its ahead cost (−0.000222). Exact iter-15 recipe otherwise (RWKV_ZERO_FEATURES=22, vprune vs
champion_5k_plain).

**Finals (n=4999): ahead 0.303825 / imm 0.273024. On the intersection vs iter15
(champ 0.303723/0.273282, paired_pvalue --intersect): imm +0.000258 BETTER (p=1.6e-70) but
under the 0.0003 bar; ahead −0.000101 worse (p=1.0). REJECTED.**

**Key finding — the trade is ~LINEAR in scale:** 0.5 → imm +0.000387 / ahead −0.000222;
0.25 → imm +0.000258 / ahead −0.000101. Both modes interpolate smoothly through zero, so no
scale can make BOTH improve ≥0.0003 — a pure trade can never pass a both-modes gate. **The
pbin-scale lever is exhausted by interpolation** (not merely 2 samples); loss-reweighting
family stands 0/2 with a real, reproducible, dose-responsive effect. Other reweighting ideas
(recency weights, per-rating weights) would be genuinely new family members if revisited.

**NEW FAILURE MODE — first-ever d=32 NaN-skip:** user 8902 (2.0M-token mega user, finite in
every prior track-1 run; iter15 scored 0.0022/0.0002 on 1,768,035 reviews) NaN'd on its
1.0M–2.1M-token eval chunk. Until now this instability class was d=128-only (A0's 7 skips).
fp32 probe (DTYPE=float + RWKV_EVAL_CAST_FP32, same ckpt, user 8902 only): **NaN PERSISTS on
the exact same chunk → weight-level, A0-class** (chunks 0 and 2 finite; scratchpad/
iter19_pbin025/probe32.log). Could be trained-weight lottery rather than pbin causally, but
either way the candidate would have been a worse deploy than the champion. Probe recipe note:
DTYPE=float alone crashes on mixed dtypes (LMDB batches are bf16) — the shim env is required. Merge/completeness gate handled it
correctly (4999 + 1 = 5000 rostered); gate ran manually with --intersect (the pipeline's
template gate exits 1 on set mismatch — future track-1 .cmds should add --intersect only when
a nanskip appears, since full-n pairing is stricter evidence).

**Val trajectory:** imm better at 9/12 WS checkpoints, ahead a coin flip around zero — the
mid-run vals previewed the trade honestly. Timing: WS 93m (never prune-threatened), decay 22m,
phased eval 76m. Artifacts scratchpad/iter19_pbin025/ (iter19d_1638.pth kept),
result/RWKV[-P]-iter19_pbin025.jsonl + .nanskip.jsonl.

**NaN LAYER DIAGNOSIS (Andrew's request, 2026-07-16 14:30, `scratchpad/iter19_pbin025/
diag_nan_layer.py` + `diag_nan.log`):** hooks on all 454 modules, fp32, NO_JIT, both chunks.
**Creator = the WKV state recurrence in the DECK stream's LAST layer (`rwkv_modules.1.blocks.3
.time_mixer`, deck = the 4-layer stack)** — every pre-WKV projection (W_r/W_k/W_v, LoRAs,
norms) is finite; the first NaN tensor is the recurrence output feeding out_group_norm. NaN
starts at token ≈541,159 of the 2.0M-token chunk and poisons ~65% of positions (everything
after), then cascades through the channel mixer into the note stream and the whole model. NO
Inf at any module boundary → the overflow lives inside the per-step state accumulation
(Inf−Inf / Inf×0 within a step yields NaN directly). Mechanism: RWKV-7's state update
(decay + a-scaled removal + write) is not guaranteed contractive; a mega-entity sequence
(one deck ≈ the user's whole 2M-review history) runs ~10⁵–10⁶ consecutive steps through one
state, so a learned (w,a,k) combo with per-step gain marginally >1 compounds to fp32 overflow
— same class as A0's d=128 mega-chunk NaNs (chunk 0 of the same user survives: content-
dependent). Deck is the natural first victim: deepest stack + longest per-entity segments.
**Prevention menu:** (a) deploy/eval-side state-norm clamp (renorm S when ‖S‖∞ > τ~1e4;
O(1)/step, exact when inactive, a few lines in the Rust RNN engine + kernel guard) — QUEUED
for ship time; real Anki power users will produce exactly these sequence lengths; (b)
training-side contractivity margin (bound `a` / penalize state norm) — heavier, only if a
future CHAMPION exhibits the property (iter15 and all other track-1 ckpts are clean on all
5000 users); (c) the eval NaN-guard already handles it honestly (skip + record + intersect).

### Iter 20 — cross-head readout mix v1 (REJECTED 2026-07-16 17:55): first p-gate pass, magnitudes short

**Design:** RWKV_XHEAD_MIX=1 in rwkv_model.py — a zero-init per-channel delta mix across the
2 heads applied to the WKV recurrence output BEFORE out_group_norm: out[g,k] += Σ_h
out[h,k]·delta[h,g,k]. The per-head GroupNorm + elementwise gate make this NOT absorbable
by W_o (a post-norm linear would be). +H·H·K = 64 params/layer × 14 layers = 194,620 total.
wd pulls the delta toward 0 = toward champion behavior. Smoke lesson: **W_o is zero-init, so
at fresh init nothing upstream of W_o is observable and no grad flows to the mix** — the
smoke had to randomize W_o before its perturb/grad checks (smoke_xmix.py).

**Finals (n=5000, 0 NaN-skips): ahead 0.303485 / imm 0.273120 = +0.000178 / +0.000107 BETTER
than iter 15, p = 2.0e-10 / 2.0e-25 — the p-gate PASSES (first candidate since iter 15), but
both magnitudes miss the ≥0.0003 bar → REJECTED.** The strongest positive signal of the
plain era: consistent per-user improvement in both modes, just too small. Readout family
0/2 now WITH signal (prehead gate was null — gating the shared trunk does nothing, but
letting heads exchange information does something real). Val was parity all run — a ~0.0002
effect is below the 10-user val set's resolution, so mid-run vals could not have seen it.

**→ ITER 21 (conduct rule 2): same hook, richer parameterization — full per-head-pair K×K
matrices,** delta (H,H,K,K), out[g,j] += Σ_h Σ_k out[h,k]·delta[h,g,k,j]; v1 is exactly v2's
diagonal (j=k). +1024 params/layer = 208,060 total (under the 225k cap). If the information
channel saturates at v1's level, v2 lands in the same place and the family closes honestly;
if the scalar mix was the bottleneck, v2 has 16× the capacity to carry it over the bar.
Pipeline 3h16m clean (WS 97m, decay 24m, eval 75m).

### Iter 21 — cross-head mix v2, full K×K (REJECTED 2026-07-16 21:12): capacity erased the signal

**Design:** RWKV_XHEAD_MIX=2 — iter 20's hook with the delta widened from per-channel scalars
(H,H,K) to full per-head-pair K×K maps (H,H,K,K): out[g,j] += Σ_hk out[h,k]·delta[h,g,k,j];
v1 is exactly v2's diagonal. +1024 params/layer = 208,060 total. Same zero-init/wd/recipe.

**Finals (n=5000, 0 NaN-skips): ahead 0.304522 = −0.000859 WORSE (p=1.0), imm 0.273208 =
+0.000019 tied (p=0.033). REJECTED decisively.** The 16× capacity didn't carry v1's signal
over the bar — it destroyed it: ahead regressed ~5× beyond v1's total gain. Interpretation:
the cross-head channel is information-poor and regularization-hungry — 64 wd-pulled scalars
extracted a real +0.00018/+0.00011, while 14k free parameters let the mix distort the
per-head GroupNorm geometry faster than they learn anything. Readout family 0/3
(prehead null / v1 near-miss with real p-gate-passing signal / v2 harmful).

**V3 candidate (queued for the NEXT track-1 block, after the A2 block):** v1's exact 64-param
hook with the delta EXCLUDED from weight decay — rename the param so train_rwkv's
'"weight" in name' filter routes it to the wd=0 group. Rationale: wd=0.01 continuously pulls
the scalars toward zero; v1's effect plateaued at ~2/3 of the bar, and the equilibrium
magnitude scales inversely with wd. Zero new capacity, targets exactly the observed failure
mode ("right direction, too small"). If v3 also lands under the bar, the family closes at
0/4 with the honest conclusion "cross-head readout information is real but worth <0.0003".

Val trajectory tracked the champion with slightly more scatter than v1 (no persistent
deficit) — third confirmation that mid-run vals cannot resolve sub-0.001 finals. Pipeline
3h14m clean (WS 95m, decay 23m, eval 76m).

### Track-2 A2 — deck 4L→3L (REJECTED 2026-07-17 07:25): deck depth is load-bearing for ahead

Deck stream 4→3 layers on the A1 arch (`scratchpad/track2_a2/architecture_d128_cmix1_deck3.py`),
2,320,516 → **2,204,412 params (−116,104 = exactly 5.0%)**, exact A1 recipe (1 ep WS + 0.25 ep
decay, seed 1234, MAX=32768, per-step cache clears). Full n=5000 pairing, **0 NaN-skips**
(second consecutive clean d=128 run — A0's ≥500k-token overflow stays gone with mixers at 1.0).

**Finals: ahead 0.300189 / imm 0.269344** vs A1 0.300009/0.269324 → ahead **+0.000180 worse**
(p=1.0), imm +0.000020 worse (p=0.96). Ratio gate (≤0.0001/100k both modes): ahead
**+0.000155 = 1.55× the bar → FAIL**; imm +0.0000172 (pass with 6× margin). The allowed
degradation at Δparams=116,104 was 0.000116/mode; ahead spent 0.000180. Verdict: the deck
stream's 4th layer earns its 82.9k params on the curve pathway — mirrors d=32, where deck
kept 4L as the largest stream after every rebalance. d128-single-layer-cut family 0/1,
deprioritized in favor of BUNDLES (Andrew's ≥5% sizing rule: this was exactly 5.0% and still
failed the price check — future cuts must buy more per point of logloss).

Decay-end val 0.3229/0.3043 vs A1's 0.3225/0.3040 — the small consistent val deficit again
predicted the eval sign (iter-18 lesson: persistent gaps mean something; oscillating ones
don't). Pipeline: WS 5h54m @ ~1.06 s/step (never vprune-threatened), decay 1h30m, unsharded
eval 2h27m (8,821 s), total 9h54m clean.

**Grad-stats recording DEAD** (the run's other deliverable): first live use of
`RWKV_GRAD_STATS` exposed a whole-step-skip bug — the 5 layer-0 `v_lora_simple.A` tensors
never receive grads (v0-mix applies only above layer 0), so `any(g is None)` skipped EVERY
step; both A2 jsons have steps_counted=0 for all 474 tensors. Fixed in `dcf11f5` (per-param
subset accumulation; report refuses dead jsons and lists never-grad tensors as free prune
candidates — those 5×1,024 params are themselves strippable). A2's ranking forfeited; A3
records correctly on the same A1 trunk.

**Next = A3 GRU-faithful curve head** (RWKV_GRU_HEAD=2: three tiny linears predict w/S/decay
for N=2 power curves, replaces w_linear + strips the dead ahead head; 2,126,224 = −8.37% vs
A1; built + fully smoked overnight incl. bit-exact off-path). A2's rejection means the drafted
launch cmd runs unpatched (A1 arch + A1 champion refs were the defaults). Launches after
iter 22 frees the GPU (~11:45).

### Iter 22 — no-residual cost measurement (COMPLETE 2026-07-17 10:30, verdict = ANDREW)

RWKV_NO_AHEAD_RESIDUAL=1 on the exact iter-15 recipe: the learned piecewise-linear ahead
correction zeroed → curve = pure mixture-of-exponentials, **monotone in elapsed time by
construction** (MONOTONICITY_PLAN.md stage-1-by-removal, Andrew's directive). 193,724 params
(~12.5k now dead, strippable at deploy).

**Finals (n=5000, 0 NaN-skips): ahead 0.304497 / imm 0.273539** vs iter 15's
0.303663/0.273227 → **ahead +0.000834 worse (p=1.0), imm +0.000312 worse (p=1.0)** — the
measured price of the monotonicity guarantee. Val trajectory tracked the champion within
noise the entire run (a +0.005 ahead spike at step 1500 was transient; WS-end 0.3287/0.3110 ≈
parity; decay-tail 0.3271/0.3087): the 10-user val set cannot resolve the curve-shape
flexibility the residual was buying — the cost only appeared at full eval. Pipeline 3h09m
clean (WS 91 min, never vprune-threatened; decay 23 min; sharded eval 75 min).

**No auto-verdict — reported to Andrew.** Options as framed at redefinition: (a) directed
re-baseline (iter 22 = new track-1 reference; recommended — the flag is already mandatory in
every future run in both tracks, so a with-residual champion is not a fair gate), (b) treat
as too expensive and revisit the constraint. If (a): promote via `promote_champion_5k.py
--val-trace` and iter 23 (learnable PAVA, built + smoked) gates vs iter 22.

Ops lesson from the same hour (cost one dead launch): Write-tool-authored `.cmd` files are
LF-only and cmd.exe silently dies on them — convert to CRLF before `detach.ps1`, and always
pass detach.ps1 an ABSOLUTE script path (the WMI-spawned cmd.exe starts in system32).

**Iter 22 VERDICT (Andrew 2026-07-17 ~10:50): ACCEPTED as directed re-baseline.** New track-1
plain champion/reference = iter22_nores (0.304497/0.273539); `champion_5k_plain.json`
re-pointed (6,554-step WS trace + val trace = the new vprune ref). Iter 15 stays in the
record as the last with-residual champion; the +0.0008/+0.0003 is the accepted price of the
monotone-in-t guarantee. Iter 23 (learnable PAVA) gates vs iter 22, >=0.0003 both modes.

### Track-2 A3 — GRU-faithful curve head (REJECTED-pending-re-anchor 2026-07-17 21:20)

RWKV_GRU_HEAD=2 on the A1 arch: three tiny fp32 linears off the shared `head_w` trunk
predict per-row (w, S, d) for N=2 power curves R(t)=Σ wᵢ(1+t/Sᵢ)^(−dᵢ) (srs-benchmark GRU
class, exp-clamped ⇒ monotone in t by construction); legacy w_linear + the dead ahead head
→ 1×1 dummies. **2,320,516 → 2,126,224 params (−194,292 = 8.37%).** First no-residual
track-2 run (the head forces it structurally). vprune MIN_STEP=6000 (zero-init prior curve
= mismatched-at-init; in hindsight unneeded — step-1000 val was ahead −0.011 BETTER than A1
same-step; the head converges off its prior in <1000 steps).

**Three findings:**

1. **Accuracy (n=4,871 intersection vs A1): imm 0.268403 = +0.000105 BETTER (p=1.6e-21) —
the FIRST statistically significant track-2 accuracy improvement.** Ahead 0.299964 =
+0.000443 worse (p=1.0) → ratios +0.000228 (2.28× the ≤0.0001 bar, FAIL) / −0.000054
(pass). **Confounded:** A1 carries the piecewise residual; A3 cannot; iter 22 priced
residual-removal ALONE at +0.000834 ahead (d=32). A3's ahead deficit is ~half that → the
GRU head itself plausibly IMPROVES ahead against a fair no-residual anchor. **Final verdict
deferred to the re-anchor**: A1 arch + RWKV_NO_AHEAD_RESIDUAL=1 (queued overnight; needed
anyway — every future track-2 run is no-residual by the mandatory recipe, so the track-2
reference must be re-anchored exactly as track 1 was with iter 22).

2. **Instability: 129/5,000 eval users NaN-skipped** (A0: 7; A1/A2: 0). The ≥500k-token
bf16 overflow returned under the GRU head's training trajectory and OSCILLATES: vals NaN'd
steps 3000–16000, recovered 17000+ (0.3246/0.3059 WS-end, healthy), NaN'd again in decay;
decay-end weights skip 2.6% of full histories. Not deployable as-is — the queued
deploy-side state-norm clamp (or a train-time fix) is now load-bearing for ANY d=128
no-residual config, not just A3. Ops note: mid-eval nanskip polls must read the SHARD file
(`RWKV-track2_a3-s0.nanskip.jsonl`) — the merged name only appears at the end.

3. **Grad-stats (fixed recorder, first valid d=128 recording): 10,886 params NEVER receive
grads** — layer-0 `v_lora_simple` A+B+bias across all 5 streams (v0-mix only applies above
layer 0) = a free strip in any future arch. Saliency bottom tier = ALL non-L0 channel
mixers (preset.L1, user.L1/L2/L3, note.L1, card.L1, deck.L1/L2/L3) + `user.L3.time_mixer`
→ the A4 bundle shortlist (mixer-mass thinning + user 4L→3L, bundled to clear ≥5%).

Pipeline: WS 6h35m @ ~1.06 s/step, decay 1h38m, single-process eval 2h23m, clean exits.
Launch bookkeeping: two dead launches (~5 min lost) — LF-only .cmd (Write tool) killed
cmd.exe silently + relative detach path; then a step-50-val misread killed a healthy
launch. Artifacts scratchpad/track2_a3/ (t2a3d_5586.pth kept), result/RWKV[-P]-track2_a3.jsonl.

### Iter 23 — learnable power-mean PAVA rectifier (REJECTED 2026-07-18 01:15): the closest miss yet

MONOTONICITY_PLAN.md stage 2, Andrew's fixed queue (23 = unweighted, 24 = p-head-weighted).
The champion iter-22 recipe + `RWKV_PAVA_LAMBDA=0.1` + `RWKV_PROBE_DENSITY=0.08`: 8% of
eligible labeled rows get 4 counterfactual button-probe rows (grade one-hot swapped
Again..Easy, duration imputed to the frozen train-median constant, has_label=0) inserted
before them; the 4 curve-head retention estimates at the probe rows pass through a
sequential PAVA whose 3 junction pair-merges are weighted generalized power means with
learnable powers p_j = 2·tanh(θ_j), init θ=atanh(0.5) → p=1 = classic PAVA; loss =
λ·BCE(rectified pressed-button probability, ahead label), train-branch only (val/eval
probe-free by construction → comparable to iter 22). Params 193,727 (+3 thetas).

**Finals (n=5000, 0 NaN-skips): ahead 0.304220 / imm 0.273423** vs iter 22
0.304497/0.273539 → **BOTH modes improved: ahead +0.000278 (p=1.3e-33), imm +0.000116
(p=8.1e-15)**. P-gate passes both modes with enormous margin; magnitude gate fails —
ahead misses the 0.0003 bar by **0.000022**, imm reaches ~1/3 of it. REJECTED, but this
is the strongest positive result of the plain era (iter 20 was +0.000178/+0.000107) and
the second-ever both-modes-positive candidate. The monotonicity loss is ~free-to-mildly-
positive for accuracy at this dose — the constraint acts as a regularizer on the curve
head rather than a tax.

**Learned junction powers (decay ckpt): Again–Hard p≈−0.0008 (geometric mean), Hard–Good
p≈−1.44 (harmonic side), Good–Easy p≈+0.53.** All three moved decisively off classic-PAVA
p=1. p<1 pulls a violating pair toward the LOWER retention estimate — the model wants
soft, pessimistic pooling, strongest at the middle junction (where iter-17/19 showed the
Hard/Good boundary carries the pbin mode-trade too). This is real learned structure, and
it transfers directly to iter 24's interpretation.

Val trajectory: parity with the champion the whole run (oscillating ±0.001 by checkpoint,
imm mildly favoring the candidate mid-WS; WS-end 0.3288/0.3106 vs 0.3287/0.3110; decay-end
0.3270/0.3086 vs 0.3271/0.3087 — indistinguishable at n=10 users, the +0.0003 effect only
resolvable at full eval). Probe-loss trajectory NOT recoverable — the step-trace writer
records ahead/imm only; `pava_loss_avg`/`pava_pool_frac` never reached the jsonl (wire them
into the trace writer if a future PAVA iter needs the trajectory). Pipeline: WS 105m
(never vprune-threatened), decay 26m, phased sharded eval 76m, total 3h27m clean.

**VERDICT CHANGED — ACCEPTED (Andrew, 2026-07-18 ~12:55, directed):** "let's accept it. Not
because of log loss improvements, but just to make Anki user's experience nicer so that answer
buttons have clearly ordered intervals... we're accepting the simple monotonicity constraint
just for the sake of the constraint itself." Iter 23 = the NEW track-1 champion/reference
(0.304220/0.273423; champion_5k_plain.json re-pointed, promote --val-trace done). The
learnable-PAVA loss (λ=0.1, density=0.08) joins the mandatory track-1 recipe; at deploy the
learned-power rectifier becomes a model component applied to the 4 counterfactual button
predictions (duration imputed to the frozen train-median constant) — Rust-side port queued
alongside the state-norm clamp. Iter 24 keeps the NORMAL acceptance criteria, now vs iter 23:
the sophisticated (p-head-weighted) variant replaces the simple one only if it provides real
benefit (≥0.0003 both modes + p<0.0001; its cmd tail prints vs-iter22 — stale, re-gate vs
iter 23 at record time).

**Next = iter 24 (pweight variant, conduct rule 2: near-miss → variant implementation):**
identical config + `RWKV_PAVA_PWEIGHT=1` — pooling weights = the p-head's button-press
softmax at the paired query row (Instant mode) instead of uniform. Rationale: PAVA-merging
with press-probability weights makes the rectified estimate a proper posterior blend —
violations between a likely and an unlikely button should mostly defer to the likely one;
uniform weighting overcorrects the likely button's estimate. λ/density unchanged
(validated by iter 23's neutral-to-positive accuracy). Launches behind the track-2
re-anchor (waitloop). Artifacts scratchpad/iter23_pava/ (iter23d_1638.pth kept),
result/RWKV[-P]-iter23_pava.jsonl.

### Iter 24 — p-head-weighted PAVA pooling (REJECTED 2026-07-18 15:32): uniform suffices

`RWKV_PAVA_PWEIGHT=1` on the exact iter-23 config: the three junction merges weight their
power means by the p-head's Instant-mode button-press softmax at the paired query row
instead of uniformly. **Finals 0.304185/0.273421 (n=5000, 0 NaN-skips) — vs iter 23:
ahead +0.000035 (p=0.54), imm +0.000002 (p=0.03) = the null-effect signature.** The
sophisticated variant provides no benefit over the simple accepted one, so per Andrew's
directive iter 23 stays champion and the mandatory recipe keeps unweighted pooling
(deploy stays simpler too: no p-head softmax needed inside the rectifier).

The run's real value is CONFIRMATION: vs iter 22 it scored **+0.000312 (p=6.0e-35) /
+0.000118 (p=7.1e-21)** — two independent trainings (23 and 24 differ only in pooling
weights) reproduced the PAVA gain almost exactly (+0.000278/+0.000116 vs
+0.000312/+0.000118), with ahead this time OVER the 0.0003 bar. The rectifier's accuracy
effect is real, reproducible, and worth ~+0.0003 ahead / ~+0.0001 imm on top of being
the product constraint. Learned powers [−0.49, −1.27, +0.74] vs iter 23's
[0.00, −1.44, +0.53]: same qualitative shape (soft pooling, harmonic-side middle
junction) — the weighting shifted where the powers settle but not the outcome.
Weighting sub-lever CLOSED; unexplored family members if revisited: per-junction λ,
probe-density sweep. The cmd tail printed the drafted-era stale gate vs iter 22; the
recorded verdict is the rerun vs iter 23 (`paired_pvalue --intersect`). Pipeline: WS
105m, decay 26m, sharded eval 78m, clean. Artifacts scratchpad/iter24_pweight/
(iter24d_1638.pth kept), result/RWKV[-P]-iter24_pweight.jsonl.

### Track-2 A4 — the no-residual re-anchor (ACCEPTED + PROMOTED 2026-07-18 12:02)

A1 arch + `RWKV_NO_AHEAD_RESIDUAL=1`, exact A1 recipe otherwise — the directed re-baseline
planned at A3's verdict: every future track-2 run is no-residual by the mandatory recipe, so
the track-2 reference had to be re-anchored exactly as track 1 was with iter 22. Params
2,320,516 unchanged (142,592 now dead/strippable — see grad-stats below). Promoted via
`promote_champion_5k` → `champion_5k_track2.json` (22,346-step WS trace + val trace = the
track-2 vprune ref; ckpt `t2red_5586.pth`). **All future track-2 candidates gate vs
0.300504/0.269262 on FULL n=5000** — the A0 intersection era ends.

**Finals (n=5000, 0 NaN-skips): ahead 0.300504 / imm 0.269262.** The d=128 residual price
(paired vs A1, informational): **ahead +0.000495 worse (p=1.0), imm 0.000062 BETTER
(p=1.1e-07)** — a sharper asymmetry than d=32's +0.000834/+0.000312 (iter 22): at d=128 the
piecewise residual bought only ahead curve-shape and was mildly *hurting* imm. (The tail's
"P-GATE FAIL" banners are the tool's accept-gate formatting, not a verdict — the re-baseline
is directed.)

**A3's deferred verdict (paired vs THIS anchor, n=4871 intersect): ratio gate PASS both
modes.** A3 is BETTER than the fair anchor in both: ahead +0.000056 (p=0.107, n.s.), imm
+0.000043 (p=7.6e-05). Ratios at Δparams=194,292: **−0.0000288 / −0.0000221** vs the ≤0.0001
bar — the GRU curve head strips 8.37% of params at zero-to-negative accuracy cost. **Promotion
stays BLOCKED by A3's instability** (129/5000 eval NaN users; recorded as gate-PASS-unstable):
the head is validated as an **A5-bundle component** once the state-norm clamp (deploy/eval) or
a train-time stability fix lands. (Naming: "A4 bundle" in pre-re-anchor notes = this A5 —
A4 is the re-anchor itself.)

**Stability: zero NaN val windows + 0 eval nanskips** (3rd clean d=128 run of the last 4) —
the GRU head's training trajectory, not d=128/no-residual, was A3's destabilizer. Val
trajectory was a clean descent all run: WS-end 0.3250/0.3064, decay-end 0.3228/0.3040 ≈ A1
parity (0.3225/0.3040) — the ahead cost was invisible at n=10 val resolution, same lesson
as iter 22.

**Grad-stats (`t2re_grad_stats_ws.json`, fixed recorder, 2nd valid d=128 recording):
never-grad = 142,592 params** — the dead ahead head 131,712 (head_ahead_logits 65,536+512 +
ahead_linear 65,536+128) + the 5× layer-0 `v_lora_simple` 10,880 — a free strip in any
bundle. Saliency bottom tier = **8 non-L0 channel mixers** (ascending: preset.L1, user.L2,
user.L3, user.L1, note.L1, deck.L1, preset.L2, deck.L2 — ~33.2k each, ~265k total = 11.4% of
A1), then card.L1/user.L2/user.L3 time-mixers. Consistent with A3's report on a different
head config → the ranking is robust, head-independent signal. **A5 bundle menu:** free strip
142,592 + bottom-mixer mass (pick ~4–8) + optionally user 4L→3L and/or the GRU head (with
stability fix) — easily clears the ≥5% sizing rule with headroom to spare.

Pipeline: WS 6h38m @ ~1.07 s/step (22,346 steps, never vprune-threatened), decay 1h39m
(5,586 steps), single-process eval 2h27m (8,804 s), DONE_EXIT_0 12:01:55, total ~10h47m
clean. Iter 24's waitloop detected the release and started 12:03:16. Ops note: the whole
verdict was executed by a DIFFERENT session than the one that launched the run (the original
died at 01:32 taking its monitor with it; recovery = the compact focus preserved in
controller.log + these docs — the on-disk record carried everything). Artifacts
scratchpad/track2_reanchor/ (t2red_5586.pth kept), result/RWKV[-P]-track2_reanchor.jsonl.

### Track-2 A5 — GRU head + free strip + state clamp (ACCEPTED 2026-07-19 03:21): new champion

The grad-stats-ranked bundle on the A4 anchor: (1) the GRU curve head (`RWKV_GRU_HEAD=2`,
validated by A3's deferred gate pass), (2) the layer-0 v_lora strip (`RWKV_STRIP_L0_VLORA=1`,
never-grad on A3+A4 recordings — 1×1 dummies keep TorchScript happy), (3) the state-norm
clamp (`RWKV_STATE_CLAMP_TAU=300`, window 32768 — built same-day from the A3-instability
probe; design + validation in `scratchpad/statenorm/CLAMP_NOTES.md`). **2,320,516 →
2,115,359 params (−205,157 = −8.84%).** Channel-mixer thinning deliberately deferred to A6
so the bundle's only unvalidated piece was the clamp.

**Finals: ahead 0.300532 / imm 0.269127 — full n=5000, ZERO NaN-skips** (A3 with the same
head lost 129 users). Paired vs A4: ahead −0.000028 (p=0.99, noise); **imm +0.000136 BETTER
(p=4.2e-38)** — the GRU head's imm advantage reproduced across two independent trainings.
Ratio gate (≤0.0001/100k both modes): ahead **+0.0000136** (7× inside), imm **−0.0000663**
(negative = better) → **ACCEPTED, new track-2 champion** (`champion_5k_track2.json`
promoted, = the track-2 vprune ref).

**The clamp earned its place.** Training transients (the instability oscillates through WS
exactly as in A3): 1 NaN-skipped train batch (~step 3855), val-time SHRINK/RESET activity
peaking mid-WS (at worst the divergent head overflowed the norm within nearly every 32k
window) — yet every val checkpoint stayed full-n. Mechanism note: the Frobenius norm (sum
of squares) overflows at entry-scale ~1e19, so the RESET is a conservative early trigger
~19 orders before outputs poison — which is why no user was ever lost. Eval with FINAL
weights: 3 self-healed resets on one 1.1M-token mega-user, 0 skips.

**Bonus: WS trained ~1.67× faster than A4 (3h58m vs 6h37m, same 22,345 steps).** A4 still
computed the dead ahead head's full per-row forward+backward (only the residual ADD was
zeroed); A5's dummy strip removes it, plus w_linear 65.7k → ~3.1k. Decay 1h41m, clamped
eval 3h04m. Grad-stats: never-grad = only the 21 dummy placeholders; saliency bottom =
non-L0 channel mixers for the third consistent recording (user.L1, preset.L1, deck.L1,
user.L2, preset.L2 lead) = the A6 thinning shortlist.

Ops lesson (cost two instant launch failures at 03:22): PowerShell `Set-Content -Encoding
utf8` writes a BOM → `tomli` dies at line 1 col 1. Write tomls via the Write tool or
`UTF8Encoding($false)`. Second-order trap: the BOM-crashed iter 25's `DONE_EXIT_WSFAIL`
line satisfied the meme run's waitloop grep and cascaded the failure — after fixing, the
relaunch order (iter 25 first, whose cmd truncates its own log, THEN the parked meme run)
restored clean chaining. Artifacts scratchpad/track2_a5/ (t2a5d_5586.pth kept),
result/RWKV[-P]-track2_a5.jsonl.

### Iter 25 — GRU power-curve head at d=32 (REJECTED 2026-07-19 07:24): the d=128 win doesn't transfer

Andrew's directive ("Let's try power curves first, to see if they improve log loss of the
small model"): `RWKV_GRU_HEAD=2` + `RWKV_STRIP_L0_VLORA=1` on the full iter-23 champion
recipe (PAVA included — the probe loss is head-agnostic), state clamp as insurance.
**193,727 → 171,066 params (−11.7%).**

**Finals: ahead 0.304427 / imm 0.273441 (n=5000, 0 nanskips) — vs iter 23: ahead
−0.000207 WORSE (p=1.0), imm −0.000018 tie (p=0.38). REJECTED**; power curves do not
improve the small model. The GRU head's d=128 imm advantage (A3 +0.000105, A5 +0.000136,
both p≪1e-20) did not transfer to d=32 — consistent with the d=32 trunk, not the
curve-head family, being the binding constraint (echoes the capacity-at-5k family: the
64-basis mixture is simply sufficient at this scale). Iter 26 (N=3, conditional on a
pass) does not run. Variant A (fixed log-spaced S-grid, weights-only) remains the family
sibling but the family is deprioritized at d=32.

**Val-lead lesson, strongest instance yet:** iter 25 led iter 23's val trace at most
checkpoints — WS-end −0.0014/−0.0007 better, decay-end −0.0005/−0.0004 better, the best
pre-eval position any track-1 candidate has held — and still lost eval by 0.0002.
n=10-user val leads predict nothing at the 0.0003 scale.

**Size-exception option (Andrew's call, deliberately not auto-invoked):** under the
SIZE/SPEED efficiency budget (both modes within +0.0015; params −11.7%) iter 25 could be
accepted as a size win. Not invoked because the directive was logloss, ahead −0.000207
at p=1.0 is a real regression that burns champion budget, and d=32 *weight* savings are
not deploy-relevant (deploy cost = per-card state, unchanged here).

**PAVA powers are a stable data property:** iter 25 learned [−0.30, **−1.44**, +0.34] vs
iter 23's [0.00, **−1.44**, +0.53] — the Hard–Good junction converged to −1.44
identically under a completely different curve head.

Pipeline: WS 119m (the clamp's windowed sequential path slows the long-user vals), decay
26m, sharded eval 93m, clean; the first launch died on the toml BOM (see the A5 section).
Artifacts scratchpad/iter25_gru/ (iter25d_1638.pth kept), result/RWKV[-P]-iter25_gru.jsonl.
The meme_blind run's waitloop fired on the DONE_EXIT and started 07:26.

**VERDICT CHANGED — ACCEPTED (Andrew, 2026-07-19 ~10:35, directed size-exception accept):**
"Alright, let's accept iter 25 then." Accuracy parity inside the +0.0015 efficiency budget
at −11.7% params ⇒ **iter 25 = NEW track-1 champion (171,066 params, 0.304427/0.273441)**;
`champion_5k_plain.json` re-pointed (promote --val-trace done). The mandatory track-1
recipe now adds `RWKV_GRU_HEAD=2` + `RWKV_STRIP_L0_VLORA=1` + the state clamp
(`RWKV_STATE_CLAMP_TAU=300 WINDOW=32768`) to NO_AHEAD_RESIDUAL + ZERO_FEATURES=22 + PAVA.
Strategic upside: BOTH tracks now run the GRU head — the eventual track merge no longer
has a head schism, and the Rust deploy port gets *simpler* (three tiny linears + closed-
form power curves R(t)=Σwᵢ(1+t/Sᵢ)^(−dᵢ) instead of the 64-basis softmax mixture; the
learned-power PAVA rectifier applies to its counterfactual predictions unchanged). Iter 26
(GRU N=3) becomes the natural next accuracy iter, gated normally vs iter 25.

### Iter 26 — GRU head N=3 (auto-REJECTED 2026-07-19 20:18, FLAGGED for Andrew): largest ahead gain of the phase

`RWKV_GRU_HEAD=3` on the iter-25 champion recipe; 171,453 params (+387). Restarted from
scratch after the PC-shutdown pause (deterministic relaunch confirmed: step-50 val
identical to the killed launch). **Finals 0.303942/0.273353 (n=5000, 0 nanskips) —
vs iter 25: ahead +0.000485 (p=4.4e-42), THE LARGEST single-iteration ahead improvement
of the 5k phase and comfortably over the 0.0003 bar; imm +0.000088 (p=4.8e-09),
highly significant but ~1/3 of the bar.** The strict monotonic gate fails on imm
magnitude alone → auto-verdict rejected, flagged (both prior flags flipped to accepts).
Reading: the third curve buys real curve-shape resolution — ahead IS the curve task —
while imm sits near its trunk-limited ceiling. PAVA powers [−0.84, −1.59, −0.26]: the
middle junction lands strongly negative for the third straight iteration. Sweep
directive ("sweep upward while it keeps winning") reads as alive — both modes improved —
so **iter 27 = N=4 launched immediately** (gate tail prints paired vs BOTH iter 25 and
iter 26). Pipeline: WS ~112m, decay 26m, eval 90m, clean. Artifacts
scratchpad/iter26_gru3/ (iter26d_1638.pth kept), result/RWKV[-P]-iter26_gru3.jsonl.

### Track-2 A7 — user 4L→3L + mixer strips (ACCEPTED 2026-07-21 01:07): better in BOTH modes at −9.4%

The bundle: user stream 4L→3L (`scratchpad/track2_a7/architecture_d128_cmix1_user3.py`,
−116,104 — removes user.L3's time AND channel mixer) + next-tier mixer strips note_id:1
+ deck_id:2 (−66,304). **1,949,624 → 1,767,226 params (−9.36% vs A6, −26.4% vs A4).**

**Finals 0.300365/0.268966 (n=5000, 0 nanskips, 0 clamp resets) — vs A6: ahead +0.000064
BETTER (p=1.3e-07); imm +0.000270 BETTER (p=9.1e-118, the strongest p-value of the
entire 5k phase).** Ratio gate moot — both deltas are improvements. The user stream's
4th layer was actively hurting imm (over-capacity drag), exactly what four consecutive
grad recordings flagging user as the lowest-saliency stream predicted. Sharp contrast
with A2 (deck 4L→3L cost ahead +0.000180 — deck depth loads the curve path): saliency
ranking, not stream symmetry, is the guide. imm 0.268966 = the best full-n track-2 imm
(below even the A0 anchor's intersection value). WS 5h47m (each strip keeps training
faster), decay 1h28m, eval 2h49m.

**A8 (launched 01:25, from A7's own grad recording):** card.L1.channel_mixer is back at
tier-1-freeness saliency (1.2e-7) and BOTH card.L2 units rank bottom-tier → card 3L→2L
+ card.L1 mixer strip = 1,617,975 params (−8.45% vs A7, −41% vs the original 2.76M),
with the deploy bonus of a smaller per-card state. Gate vs A7. Artifacts
scratchpad/track2_a7/ (t2a7d_5586.pth kept), result/RWKV[-P]-track2_a7.jsonl.

### Track-2 A8 — card 3L→2L + card.L1 mixer strip (ACCEPTED 2026-07-21 12:45): −8.45% at ~zero cost; stability watch item

The bundle (from A7's grad recording): card stream 3L→2L
(`scratchpad/track2_a8/architecture_d128_cmix1_user3_card2.py`, −116,104) + card.L1
channel-mixer strip (RWKV_STRIP_CMIX now 8 entries, −33,152). **1,767,226 → 1,617,975
params (−8.45% vs A7, −41.4% vs the original 2.76M)**; also cuts per-card d=128 deploy
state by 1/3 (2 card layers instead of 3).

**Finals 0.300380/0.269006 (full n=5000, 0 nanskips, COMPLETE 5000/5000) — vs A7: ahead
+0.0000155 worse (p=0.59), imm +0.0000402 worse (p=0.97) → per-100k ratios +0.0000104 /
+0.0000269 vs the ≤0.0001 bar = 10× / 3.7× INSIDE. ACCEPTED on the ratio gate** —
essentially free at −149,251 params. Saliency-guided pruning is now 4/4 since A6. Eval
clamp: 1,066 soft SHRINKs / 0 RESETs (lighter than A6's 16k).

**Stability watch item — the phase's first training-time instability since the clamp
landed:** every ~500-step val pass hit (a) 2 deterministic NaN batch-skips on val users
5047 + 5052 (short streams, below the clamp window → no-clamp path; the train-loop guard
skipped them) and (b) recurring 1-head/layer-1 non-finite RESET containment on a mega
val user's ~327k-token stream (window boundaries t=32768…327680). Determinism proof: the
machine died at ~02:35 in the recurring black-screen hang (zero telemetry precursor,
driver 610.62) mid-WS; the from-scratch relaunch (02:51) replayed val-for-val bit-exact
INCLUDING both NaN users and the RESET pattern. None of it reached the final eval
(clean 5000/5000), but A5–A7 trained clean → **card 2L looks stability-negative; carry
into A9 bundling and the QAT close.** Val summaries print roster-n (595795) even when
batches were skipped — mean-only effect, vprune unaffected (skips flatter the candidate;
vprune kills only on worse).

Ops: WS 5h36m, decay 1h20m, eval 2h48m. Grad stats recorded both phases
(t2a8_grad_stats_ws.json + _decay.json) → A9 shortlist. **Methodology cutover: A8 is the
last full-range (5001–10000) gated track-2 iter — Andrew's val/test split (val
5001–7500 for verdicts, test 7501–10000 only at track close) applies from the next
candidates on** (iter 29's parked cmd already re-pointed). Artifacts
scratchpad/track2_a8/ (t2a8d_5586.pth kept), result/RWKV[-P]-track2_a8.jsonl;
champion_5k_track2.json = A8 (24 val points, the track-2 vprune ref).

### Track-2 A18 — d=80 + LoRA 8→4 (auto-verdict REJECTED 2026-07-26 10:30; **VERDICT CHANGED BY ANDREW → ACCEPTED, NEW TRACK-2 CHAMPION**): the second draw at d=80 ⇒ THE WIDTH LADDER IS DONE

The in-family retry A17 earned. Same width (5 heads × K=16) plus the second LoRA halving,
**557,246 params = 4.95× below the original 2.76M** — effectively Andrew's target — with the
allowance rising to 0.000252/mode, enough that A17's *measured* cost would have cleared it
at 99%/74%.

**Val half n=2500, 0 nanskips: ahead 0.299302 (+0.000271 = 108% of bar), imm 0.268390
(+0.000279 = 111%). REJECTED in both modes.**

**Two draws now agree, so d=80 is a genuine floor, not the unlucky draw A17 alone could not
rule out:** A17 landed 112%/83%, A18 108%/111% — both ~110% of their (different) allowances,
from independently trained models. The width ladder for track 2 is therefore **closed at
A15's d=96 / 808,762 params / 3.41×**.

**Secondary finding worth keeping: the second LoRA halving is NOT free at this width.** It
cost +0.00002 ahead / +0.00009 imm relative to A17 for 27,520 fewer params — break-even at
best — whereas A14's *first* halving at d=128 actually IMPROVED both modes. The same lever
flipping sign as the trunk narrows is direct evidence that at d=80 even the LoRA ranks are
load-bearing, i.e. the model is genuinely capacity-limited rather than merely trimmed.

### ⚠ The ≥5× goal and the ratio gate are now in direct conflict — Andrew's call

Width and LoRA cuts cannot reach ≥5× inside the gate. But note *what the gate measures*: it
is a marginal **rate** (logloss per 100k params removed), not an absolute budget. In
absolute terms A18 is cheap — **cumulative vs A0 it costs +0.000960 ahead / +0.000532 imm at
4.95×**, which is about a third of what the GRU baseline gave up (SE-2: +0.0019/+0.0027 while
being *larger*) and roughly half of one accepted track-1 iteration's gain. So the two live
options are:

- **(a) Keep A15** — 808,762 params (3.41×), cumulative +0.000689/+0.000253, gate-clean.
- **(b) Take A18 as a deliberate goal-driven exception** — 557,246 params (4.95×), cumulative
  +0.000960/+0.000532, gate-violating on the margin but tiny in total. Precedent exists:
  Andrew has overridden auto-verdicts before (iters 23, 25, 26) when the framing was wrong.

⚠ **Whichever he picks, it does not reach users yet.** Per `CPU_INFERENCE.md`, param count
has already decoupled from CPU rev/s in the only engine we can measure (a 4.5× arithmetic cut
bought 1.24× wall-clock and plateaued), and `rust/rwkv-infer` cannot run the track-2 arch at
all. **The Rust port is now the highest-value work in this track** — it is what converts any
of these cuts into something an Anki user feels.

### ANDREW'S ANSWER (2026-07-26): option (b) — A18 accepted, and the track continues on it

> *"Let's accept A18 and continue track 1 with it. Add the algorithmic improvements (PAVA,
> GRU n_head=3, Muon) to it"*

So `champion_5k_track2.json` now points at `scratchpad/track2_a18/t2a18d_5586.pth`
(557,246 params, 4.95×, per-card state 2,880 floats, val-half 0.299302/0.268390), and it is
the vprune reference for everything that follows. The ratio gate is not repealed — it simply
lost a tie-break against an explicit product goal, on a rung it missed by ~10% of a *marginal
rate* while costing under a thousandth of a nat in total. Every measurement above stands
unchanged; only the verdict moved.

The second half of his message redirects the track: with the width road closed, the next
gains come from **algorithms rather than shape**. Track 1 found three at d=32 that track 2
never received — PAVA (iter 23), GRU N=3 (iter 26) and Muon (iter 29) — and the next run
carries all three onto the A18 trunk at once. Bundling is deliberate: each is independently
validated, together they are exactly the iter-29 champion recipe, and one run costs ~10 h
against ~30 h. If the bundle regresses, the de-bundle precedent is A10→A11.

Worth flagging as a real possibility rather than a formality: these three were tuned on a
d=32 trunk, and the transfer-failure ledger (iter 28's xhead mix, A13's state-feature price
landing opposite in sign to d=32's) says a d=32 win is a *hypothesis* at d=80, not a
deposit. What makes this one better-founded than most is that the trunk is now demonstrably
capacity-limited (A18's own LoRA finding), and PAVA/GRU-N are head-side changes that add
capacity where the trunk cannot.

### ⇒ THE TWO TRACKS MERGE — the next run is **iter 31**, not A19

I launched it as `track2_a19` and Andrew corrected the naming within minutes:

> *"Yeah, it shouldn't be called A19, it should be iter 31 (first table in research 5k)"*

Which settles what "continue track 1 with it" meant: not a track-2 run that borrows track-1
ideas, but **one merged lineage** — the A18 trunk, numbered as ordinary research iterations
in this document's first table, continuing from iter 30. The track-2 A-series is closed at
A18, and its ratio gate retires with it: from iter 31 on, candidates face the ordinary
accuracy gate (both modes ≥0.0001 after 4-dp rounding, p<0.0001), measured against A18 as
the reigning champion.

⚠ **One inherited rule does NOT survive the merge and I am flagging it rather than dropping
it silently: the track-1 `params ≤ 225,000` cap.** It was written for the d=32 lineage,
where 225k was a real ceiling; the merged model is 558,212 params and its size story is the
4.95× cut from 2.76M. Reading the cap literally would reject the champion Andrew just
accepted, so it cannot mean what it used to. Until he says otherwise I treat it as retired
and the 4.95× reduction as the standing size result.

The rename cost ~4 minutes of training (I killed the run at step 126 of 22,346). Worth it:
checkpoint prefixes, trace filenames, result tags and log paths all embed the run name, and
a mislabelled lineage is the kind of thing that quietly corrupts a record months later.

**Speed, measured on that first launch and worth recording:** this recipe runs at ~0.7–0.9
steps/s against A18's 1.86 — probe rows add ~30% more rows, PAVA's rectifier runs eager
inside a `@torch.jit.ignore` method, and Muon adds a Newton-Schulz iteration per matrix.
Peak reserved 8.8 GB of 12 GB. So WS ~7–8 h and ~10 h end-to-end, i.e. **every merged-lineage
iteration costs about twice a plain track-2 one.** Not a bug — it is the price of the three
features — but it halves iteration velocity and is worth knowing before planning a long
queue. (Footnote on the 40-step sanity bench: it reported 0.29 steps/s, and iter 29's bench
reported 0.275 against ~0.93 actual. The bench systematically understates steady state by
~3×; use it for wiring and VRAM, never for scheduling.)

### Track-2 A17 — d_model 96→80 via 5×K=16 (REJECTED 2026-07-26 05:03 by 26 millionths): noise-limited, retry launched

The intermediate rung between A15's passing 96 and A16's failing 64, taken as **5 heads ×
K=16** (K=16 is proven — the track-1 champion has run it since the H2K16 acceptance, and the
WKV kernel is K-dynamic). **584,766 params = −27.7% vs A15, 4.72× below the original 2.76M.**
A bonus the gate does not score: K=16 shrinks the WKV matrix per layer from 3,072 to 1,280
floats, so **per-card state falls 6,528 → 2,880 (−56%)** — the largest state cut of the track.

**Val half n=2500, 0 nanskips: ahead 0.299281 (+0.000250), imm 0.268296 (+0.000185) vs A15.
imm PASSES at ratio +0.0000826 (83% of the ≤0.0001 bar); ahead misses at +0.0001116 (112%),
i.e. a raw overage of 0.000026 against an allowance of 0.000224.**

**⚠ Read this verdict as noise-limited, not as a floor.** 26 millionths of logloss is roughly
**15× inside the ~0.0004 cross-seed spread** the seed-pair doctrine documents for identical
recipes, so a single run cannot separate "d=80 genuinely costs too much" from "unlucky
draw". The training-val agreed it was ambiguous: A17 traded modes with A15 for the whole run
(behind at 3k/11k, *better* on ahead at 14k, level at 18k) — quite unlike A16, which held a
consistent both-modes deficit and then failed decisively at ~1.8× the bar.

**Response per the near-miss conduct rule** (an idea that barely misses gets a different
implementation, not a writeoff): **A18 = the same width plus the second LoRA halving
(decay/a/gate 8→4, v0-mix 4→2)**, launched 05:10. It buys 27,520 more params of allowance
(0.000224 → 0.000252/mode), which A17's *measured* cost would clear at 99% ahead / 74% imm,
and A14 established that the first LoRA halving was free-to-positive at d=128. At 557,246
params it also lands at **4.95× — effectively Andrew's ≥5× target**. If A18 lands at ~112%
of bar too, then d=80 is a real floor and the honest move is to stop cutting width and put
the effort into the Rust engine instead (see `CPU_INFERENCE.md` — param count has already
decoupled from the CPU rev/s that users actually feel).

Speed footnote: A17 was the fastest run of the phase, WS 3 h 21 m at 1.861 steps/s (K=16
halves the WKV kernel work on top of the narrower trunk).

### Track-2 A16 — d_model 96→64 (REJECTED 2026-07-25 23:29): the WIDTH FLOOR is between 96 and 64

The obvious follow-through to A15, and it would have crossed Andrew's target in one step:
`N_HEADS` 3→2 at K=32, i.e. `d_model` 96→64, **388,032 params = −52.0% vs A15 and 7.11×
below the original 2.76M**. It did not survive the gate.

**Val half n=2500, 0 nanskips: ahead 0.299863 (+0.000832 worse), imm 0.268831 (+0.000720)
vs A15. REJECTED — per-100k ratios +0.0001978 ahead (198% of the ≤0.0001 bar) and
+0.0001711 imm (171%), against an allowance of 0.000421/mode.**

**This is a floor, not noise.** A15's 41% cut spent 41% (ahead) and 64% (imm) of its
allowance; A16's 52% cut spent ~180% of a *larger* allowance — the cost per parameter
roughly quadrupled in a single rung. Two further signs it is a genuine capacity limit
rather than a pathway quirk: (a) both modes failed *together*, unlike the depth rungs
A10–A12 where ahead and imm dissociated (user depth owned ahead, note.L0's mixer owned
imm); (b) the 10-user training-val called it in advance — WS-final gap 0.00063/0.00130,
the first rung where the val gap exceeded the allowance in BOTH modes (A15's was
0.00067/0.00011 and it passed). Worth remembering as a weak but real predictor: the val
overstates ahead and understates imm, but a both-modes-over-allowance val gap has now
predicted a both-modes rejection.

Clean, fast run otherwise: WS 3 h 35 m at 1.746 steps/s — the fastest of the phase (and the
data point that made the training-speed ladder monotone, A0 0.933 → A16 1.746 = 1.87×).

**Where this leaves the ≥5× goal.** Width alone cannot reach it: 96 passes, 64 fails.
⚠ **The "free dead-param strip" I first proposed here is a MIRAGE** — checking A15's
grad-stats json, its 66 never-grad tensors are all **1×1 dummies** totalling 66 params: the
GRU head already replaced the ahead head structurally, and `srs_model.py` keeps 1×1
stand-ins purely so the scripted dead branches still compile. There is no 74k to reclaim.
Measured ladder of what actually remains:

| option | params | vs 2.76M | allowance vs A15 | note |
|---|---|---|---|---|
| d_model 80 (5 heads × K=16) | 584,766 | 4.72× | 0.000224/mode | state per layer 3,264 → 1,440 (more than halves); K=16 is proven (track-1 champion) |
| + head_fc_mult & features_fc_mult 4→2 | 529,246 | 5.21× | 0.000280/mode | **crosses the goal, but the 100-user era rejected mult 4→2 hard (imm +0.053)** — strong negative prior |
| + LoRA 8→4 | ~556k | ~4.96× | — | the queued A14 follow-up, small |
⚠ And per `optimization/CPU_INFERENCE.md`, none of these will show up as user-visible speed
until `rust/rwkv-infer` can run the track-2 arch — param count has already decoupled from
CPU rev/s in the engine we can measure today.

### Track-2 A15 — THE WIDTH CUT, d_model 128→96 (ACCEPTED 2026-07-25 17:08): −41.4% params, 3.41× below 2.76M

The largest single reduction of the track and the first move on WIDTH: `N_HEADS` 4→3
with head dim K=32 unchanged, so `d_model` 128→96
(`scratchpad/track2_a15/architecture_d96_lora8.py`). **REBUILT on the A14 base** — the
staged pre-A14 file was checked before launch and did already carry the halved LoRAs, so
nothing was silently reverted. **1,380,660 → 808,762 params (−571,898 = −41.4% vs A14;
3.41× below the original 2.76M = 70.7% total reduction).** Per-card state 8,704 → 6,528
floats (−25%), per-note 4,352 → 3,264 — welcome on the deploy side and permitted (state
may shrink).

**Val half n=2500, 0 nanskips, COMPLETE 2500/2500: ahead 0.299031 (+0.000233 worse), imm
0.268111 (+0.000365 worse) vs A14 same-users. ACCEPTED on the ratio gate with room to
spare — per-100k ratios +0.0000407 ahead (41% of the ≤0.0001 bar) and +0.0000638 imm
(64%), against an allowance of 0.000572/mode for the 571,898 params bought.** The p-gate
FAILs by construction: it is the accuracy-IMPROVEMENT gate, and size cuts are judged on
the ratio gate (same as A2/A12, which failed it on ratio, not on p).

**Why it worked: the d=128 trunk was over-wide for this data.** The 10-user training-val
tracked A14 essentially exactly for the entire run — level or better at steps 3k/6k, a
hair behind at 9k/11k/14k, dead level at 17k, and WS-final 0.32568/0.30548 vs A14's
0.32501/0.30537 — despite carrying 41% fewer parameters. That mirrors what A14 found one
level down (the LoRA ranks were oversized): this architecture had slack in its widths,
not in its depths (the depth ladder floors were all real, A10–A12).

**Speed note — ⚠ CORRECTED 2026-07-25 21:10 after Andrew asked "is training getting
faster?"** The original entry here claimed "d=96 did NOT train faster (1.43 vs A14's
~1.24–1.27)", which is self-contradictory — 1.43 IS faster than 1.27. I had anchored on
A15's first steps/s print (0.82, startup-inclusive) and never re-checked against the
run's own median. Measured properly (median of the 52 periodic prints per run, first
dropped): **A15 1.434 steps/s vs A14 1.200 = 1.20× faster**, and the whole track-2 curve
is monotone in width: A0 0.933 → A7 1.022 → A9 1.203 → A14 1.200 → A15 1.434 → A16 1.746
(**1.87× faster than A0 at 7.11× fewer params**). The correct framing is that speed
scales SUBLINEARLY with parameters (7.1× smaller buys 1.9× faster), which is what the
2026-07-03 elementwise-dominated profile predicts — not that shrinking buys no time at
all. H=3 is also the phase's
first non-power-of-2 head count; no kernel trouble (determinism on, zero NaN activity,
zero eval nanskips).

**Next rung is the goal line:** d_model 96→64 (N_HEADS=2, K=32) would land near ~520k
params ≈ 5.3× below 2.76M — Andrew's ≥5× target in ONE more cut. Its ratio-gate allowance
would be ~0.00029/mode (Δ≈289k params), i.e. tighter than A15's, so it is not a
formality. Artifacts: `scratchpad/track2_a15/` (t2a15d_5586.pth kept);
`champion_5k_track2.json` = A15 (24 val points = the new vprune ref).

### Track-2 A14 — LoRA dims halved (ACCEPTED 2026-07-24 03:30): better both modes at −6%; halfway mark crossed

The first STRUCTURAL cut after the depth ladder closed: decay/a/gate LoRAs 16→8 and
the v0-mix LoRA 8→4 across all five streams
(`scratchpad/track2_a14/architecture_d128_lora8.py`; pure arch-module change, no new
code). **1,468,724 → 1,380,660 params (−88,064 = −6.0% vs A13, −50.03% vs the
original 2.76M — the halfway mark).**

**Val half n=2500, 0 nanskips: ahead 0.298798 = +0.000039 BETTER (p=0.045), imm
0.267746 = +0.000059 BETTER (p=0.0069) vs A13 same-users. ACCEPTED — ratio gate moot
(both improvements, à la A7/A9). The LoRA ranks were oversized; a further 8→4 halving
is a queue candidate.** Zero training NaN activity; COMPLETE 2500/2500.

Ops-heavy history: launched 11:15 on 07-23; killed at step ~18k by Andrew's planned
PC-off; relaunched 15:41; then the resume-smoke co-tenancy incident froze it 2.7 h in
a two-process WDDM paging deadlock (17:40→20:21 — VRAM 11.6/12 GB, both logs frozen
the same second; killing the smoke unstuck it instantly, zero steps lost) and left a
harmless ~1e-4 cuBLAS-algo drift vs the first launch's trajectory. Two durable
outcomes: **NO co-tenant GPU work during gate-critical runs** (hard ops rule), and
**mid-epoch RESUME landed the same night** (RWKV_RESUME_SKIP_GROUPS=1 +
scratchpad/make_resume.py, smoke-validated exact on 587 + 258 resumed steps — future
crashes lose ≤1000 steps). Next: the GRU/LSTM stream baselines (Andrew's
is-RWKV-needed experiment, auto-fired on A14's DONE_EXIT), then A15 = d_model 128→96
REBUILT on the A14 base (halved LoRAs + N_HEADS=3). Artifacts scratchpad/track2_a14/
(t2a14d_5586.pth kept); champion_5k_track2.json = A14 (24 val points, vprune ref).

### Track-2 A13 — state-feature re-anchor (PROMOTED 2026-07-23 10:50): removal costs +0.0002/+0.0002 at d=128 — opposite sign vs d=32

The Andrew-directed recipe fix (2026-07-22 "It should be removed entirely, from both
track 1 and track 2 models"): the A9 champion arch + recipe with
**RWKV_ZERO_FEATURES=22** — the Anki card-state input (New/Learning/Review/...,
feature dim 22) zeroed at input, as track 1 has done since iter 15; track 2 had never
adopted it (recipe divergence). Pure re-baseline à la A4/iter 22: params unchanged
1,468,724, NO gate, promoted to track-2 anchor at completion.

**The measured price at d=128 (val half n=2500, 0 nanskips): ahead 0.298837 =
+0.000212 worse / imm 0.267805 = +0.000190 worse than A9 same-users, both p≈1.0
(systematically worse per-user). OPPOSITE SIGN vs the d=32 measurement** (iter 15:
removal ~free-to-slightly-better, ~−0.0001) — the d=128 model was extracting real
signal from the state feature that the d=32 trunk evidently cannot use. Small but
consistent; the directive stands (consistency/product decision; the price is
recorded, and reverting = re-pointing champion_5k_track2.json back to A9). All
track-2 runs from A13 on set RWKV_ZERO_FEATURES=22; vprune ref = A13's same-recipe
val trace. Clean run (zero training NaN activity, no wedge).

Grad report under the new recipe: the bottom saliency tier is now entirely
REJECTED-FLOOR territory (deck.L3.cmix #1, user.L2.tm #2, deck.L2.tm #3, note.L0.cmix
#4 — all depth floors or the A11-diagnosed imm poison) → confirms the structural
pivot. Next: **A14 = LoRA-dim halving** (decay/a/gate 16→8, v0-mix 8→4, all streams;
~−86k ≈ −5.9%; pure arch-module change), then head_w squeeze; d_model 128→96 awaits
Andrew's call. Artifacts scratchpad/track2_a13/ (t2a13d_5586.pth kept),
result/RWKV[-P]-track2_a13.jsonl; champion_5k_track2.json = A13 (24 val points).

### Track-2 A12 — preset 3L→2L (REJECTED 2026-07-23 03:00): preset floors at 3L; ALL depth floors mapped

The one untried depth cut: preset 3L→2L on the A9 base (arch
`scratchpad/track2_a12/architecture_d128_cmix1_user3_card2_note1_preset2.py`;
preset.L1/L2 time-mixers ranked #6/#7 in A9's grad report). 1,468,724 → 1,385,767
params (−82,957 = −5.65% vs A9); allowed 0.000083/mode.

**Val half n=2500, 0 nanskips: ahead 0.298699 = +0.000075 worse (ratio 0.0000904 =
0.90× the bar, passes); imm 0.267717 = +0.000102 worse (ratio 0.000123 = 1.23× the
bar, FAILS). REJECTED on imm.** Clean run throughout (zero training NaN activity, no
wedge). **Preset depth floors at 3L — and with it the depth-cut ladder is EXHAUSTED:
card=2 (A8), deck=4 (A2), note=1 (A9), preset=3 (A12), user=3 (A10/A11).** Every
stream is now at its measured depth floor under the ratio gate. Track 2 goes
STRUCTURAL next: first A13 = the Andrew-directed state-feature re-anchor
(RWKV_ZERO_FEATURES=22 on the A9 recipe — pure re-baseline à la A4, fixing the
track-recipe divergence; launched 03:15, verdict ~13:00), then LoRA-dim cuts /
head_w squeeze / d_model 128→96 gate against the new anchor (d_model cut = discuss
with Andrew first). Artifacts scratchpad/track2_a12/; A9 stays champion until A13.

### Track-2 A11 — the A10 de-bundle (REJECTED 2026-07-22 19:40): user depth floors at 3L; note.L0 was the imm poison

A11 = A10 minus the note_id:0 strip (same arch module — user 2L/card 2L/note 1L;
deck.L3 mixer strip kept; note.L0 mixer restored). 1,468,724 → 1,352,620 params
(−7.9% vs A9); allowed 0.000116/mode.

**Val half n=2500, 0 nanskips: ahead 0.298916 = +0.000291 worse → ratio +0.000251 =
2.51× the bar (FAIL); imm 0.267700 = +0.000085 worse → +0.000073 (passes alone).
REJECTED — but the de-bundle splits A10's damage cleanly:** ahead damage is IDENTICAL
with and without the note strip (+0.000293 A10 vs +0.000291 A11) → **user depth
floors at 3L and owns the ahead cost** (echoes A2: long-recurrence stream depth loads
the ahead/curve pathway — deck floors at 4, user at 3); imm damage fell +0.000262 →
+0.000085 → **note.L0's mixer was the imm poison (~+0.00018 imm)** — last-transform
strips are costly (the only strip in the chain that removed a stream's final
transform pair). deck.L3.cmix's own share can't be split from user depth here, but
mixer strips were 7-for-7 harmless before note.L0, so user depth dominates.

**Depth floors now mapped: card=2 (A8), note=1 (A9), user=3 (A7 ok, two 2L fails),
deck=4 (A2). preset 3L→2L = the ONE untried depth cut** (preset.L1/L2 time-mixers
#6/#7 in A9's saliency report) → **A12 = preset 3L→2L alone on the A9 champion base**
(card 2/deck 4/note 1/preset 2/user 3; 1,385,767 params = −5.65% vs A9, allowed
0.000083/mode). After A12 the chain's remaining moves are structural: LoRA-dim cuts,
head_w squeeze, d_model 128→96 (the long-queued ~40% step). Clean run: zero training
NaN events, WS 5h25m, decay 1h12m, eval 1h19m (no wedge). Artifacts
scratchpad/track2_a11/; A9 stays champion + vprune ref.

### Track-2 A10 — user 3L→2L + note.L0/deck.L3 mixer strips (REJECTED 2026-07-22 11:20): the chain's first floor

The bundle (from A9's grad report): user 3L→2L
(`scratchpad/track2_a10/architecture_d128_cmix1_user2_card2_note1.py`, −82,957 —
user.L1/L2 time-mixers ranked #1/#4) + mixer strips note_id:0 (the last note mixer,
kept in A9 for caution; #5) + deck_id:3 (#3). 1,468,724 → 1,319,473 params (−10.2%
vs A9, −52.2% vs 2.76M); STRIP_CMIX 10 entries.

**Val half n=2500, 0 nanskips: ahead 0.298918 = +0.000293 worse (ratio +0.000196 =
1.96× the ≤0.0001 bar), imm 0.267877 = +0.000262 worse (+0.000176 = 1.76×), p=1.0
both → REJECTED — the first ratio-gate failure since A2, ending 5 straight accepts.**
The bundle confounds three cuts. Prime suspect = note_id:0: it left the 1-layer note
stream as a BARE time-mixer — the only strip in the chain's history that removed a
stream's last remaining transform pair. User depth was 2-for-2 (A7, A10's cut =
third) and deck.L3 was an ordinary low-saliency mixer strip. **A11 = the bundle MINUS
the note.L0 strip (user 3L→2L + deck.L3 mixer, −116,104 = −7.9% vs A9, allowed
0.000116/mode): a pass banks most of the size AND fingers note.L0 as the poison; a
fail puts the posterior on user depth flooring at 3L.** Stability: 2 isolated
training RESET events (layer-1/1-head containment — milder than A8's recurring
pattern, not A9's zero). WS 4h37m, decay 1h10m, eval 1h17m (no wedge). Artifacts
scratchpad/track2_a10/; A9 stays champion + vprune ref.

### Track-2 A9 — note 2L→1L + L0 mixer strips (ACCEPTED 2026-07-22 04:05): better both modes at −9.2%; cleanest run of the chain

The bundle (from A8's grad report): note stream 2L→1L
(`scratchpad/track2_a9/architecture_d128_cmix1_user3_card2_note1.py`, −82,957 —
note.L1.time_mixer was #2-lowest saliency; **HALVES per-note d=128 deploy state, the
dominant deploy-memory term**) + L0 channel-mixer strips user_id:0 (#1 lowest) +
preset_id:0 (−66,294). **1,617,975 → 1,468,724 params (−9.22% vs A8, −46.8% vs the
original 2.76M)**; STRIP_CMIX 9 entries; note.L0's own mixer deliberately kept.

**First track-2 verdict on the VAL half (5001–7500, n=2500, paired vs A8's full-range
jsonl via --intersect): ahead 0.298625 = +0.000098 BETTER (p=0.35), imm 0.267615 =
+0.000010 BETTER (p=0.60). ACCEPTED — ratio gate moot (both deltas improvements, à la
A7).** Saliency-guided pruning now 5/5 since A6. **Stability: the cleanest d=128 run
of the chain — ZERO training-time NaN activity** (A8 had 2 deterministic val NaN-skips
+ RESET containment every val pass; shallower note didn't worsen it and appears to have
helped), 0 eval nanskips, COMPLETE 2500/2500.

Ops: the first eval attempt WEDGED at 02:11 on user 5747 — fetch deadlock (shard at
0 CPU / 0 GPU for 40 min; first wedge ever on the d=128 `--shards 1` path). Killed the
tree, relaunched with eval_sharded's RESUME (completed users skipped); user 5747 passed
cleanly on retry → transient race, not data-dependent. Also found and killed a LEAKED
iter-29 fetch worker that had been spinning a full core for 14 h (start time matched
the WS launch to the second) — **check for orphan pythons after every run.** WS 4h34m,
decay 1h16m, eval 1h07m + 1h00m rerun. A10 shortlist from A9's grad recording:
user.L1/L2 time-mixers #1/#4 (user depth prunable AGAIN → 3L→2L), deck.L3.channel_mixer
#3 (mixer strip, NOT the A2 depth cut), note.L0.channel_mixer #5 (the kept one — now
justified). Artifacts scratchpad/track2_a9/ (t2a9d_5586.pth kept),
result/RWKV[-P]-track2_a9.jsonl (val half); champion_5k_track2.json = A9 (24 val
points, the track-2 vprune ref).

### Iter 29 — hybrid Muon+AdamW (ACCEPTED 2026-07-21 16:05): first optimizer-family win, first val-split verdict

The modded-nanogpt sweep's one big transferable: the four matrix wd-groups
(decay/channel_mixer/head/encode) move to **Muon** (lr 0.02, momentum 0.95 nesterov,
quintic Newton-Schulz orthogonalization in bf16, aspect-ratio step scale, decoupled wd
at the AdamW-equivalent absolute rate via wd_lr_scale = peak_lr/muon_lr); every other
param stays on AdamW delegated to torch's functional kernel (bit-exact vs
torch.optim.AdamW — smoke-verified over 50 steps). `rwkv/muon.py`; RWKV_MUON unset =
byte-identical plain AdamW. Params unchanged 171,453. A 40-step E2E sanity phase ran
before the real WS (wiring clean).

**FIRST VAL-SPLIT VERDICT (Andrew 2026-07-21): eval = val half 5001–7500 only, n=2500,
paired vs iter 26's full-range jsonl via --intersect (champ val-half means
0.302176/0.271924). Muon: 0.302033/0.271440, 0 nanskips — ahead +0.000143 (rounds to
0.0001, p=2.5e-06), imm +0.000485 (p=6.5e-71, the phase's largest imm gain since the
1500u-era data jumps). Gate PASS on all counts → NEW TRACK-1 CHAMPION.** Val-half
absolute logloss is NOT comparable to full-range rows ≤28 (different user sample —
the val half runs ~0.0018/~0.0014 easier for this lineage).

**Val-lag lesson, now bidirectional:** Muon led the 10-user review-weighted val hugely
at step 500 (−0.008/−0.009), converged to parity by step 1000, then trailed the
champion by +0.001–0.003 through the entire WS tail and decay — and won the real eval
decisively. The 10-user val predicts nothing, in either direction (10 users +
review-weighting ≠ by-user mean over 2500).

Seed-pair caveat recorded: ahead's +0.000143 sits under the ~0.0005 doctrine bar (imm
is far above); consistent with recent-accept precedent (the p-gate is the operative
consistency check). Ops: WS 1h57m (Muon's NS5 adds no visible step cost), decay 25m,
val-half eval 37m — **the split halves eval wall-clock as designed**; paired_pvalue's
intersection floor lowered 4000→2000 (the val-split shape is intended, tool fix
committed). Recipe consequence: **RWKV_MUON=1 RWKV_MUON_LR=0.02
RWKV_MUON_MOMENTUM=0.95 join the mandatory track-1 champion env**; the final QAT close
run must train with Muon too (optimizer is train-time only — nothing ships to Rust).
Next: cautious weight decay = the in-family sibling (iter 30). Artifacts
scratchpad/iter29_muon/ (iter29d_1638.pth kept), result/RWKV[-P]-iter29_muon.jsonl
(val half); champion_5k_plain.json = iter29_muon (15 val points, the track-1 vprune
ref — val traces pair fine across optimizers, same schedule/steps).

### Iter 30 — cautious weight decay (REJECTED 2026-07-21 19:20): a pure imm/ahead trade

The in-family sibling of the accepted iter-29 Muon (modded-nanogpt #43/50):
RWKV_MUON_CAUTIOUS_WD=1 masks the decoupled decay on the Muon matrix groups to only
those coordinates whose applied step agrees with the weight's sign (never fight a
component the update is already shrinking; all wd mass lives on the Muon groups —
other_params run wd=0, so the Muon-branch mask is complete coverage). Implementation
`rwkv/muon.py` (cautious_wd group flag; off-path bit-exact — smoke A proved the
refactor byte-identical on the champion path; masked formula exact, mask fraction
0.500 on random data).

**Val half n=2500, 0 nanskips: ahead 0.302409 = −0.000376 WORSE (p=1.0); imm 0.271301
= +0.000139 BETTER (p=4.2e-11 — would pass the gate alone). REJECTED on ahead.** The
shape echoes the pbin dose-response lesson: one mode pays for the other. Reading:
freeing growing weights from decay pressure helps the imm pathway but hurts the
curve/ahead pathway — regularization asymmetry between the two heads again.
**Optimizer family 1/2** (Muon accepted, cautious-wd rejected); per the scoreboard
rule, Muon-lr/momentum micro-tuning is NOT auto-queued (cautious-wd didn't signal
both modes); NorMuon/Polar-Express stay as possible in-family variants, deprioritized.
The 10-user val ran at parity with iter 29 the whole run — uninformative again. WS
1h57m + decay 25m + val-half eval 45m. Artifacts scratchpad/iter30_cwd/; iter 29
stays champion + vprune ref.

### Iter 28 — xhead-mix v1 re-benchmark (REJECTED 2026-07-20 14:38): the iter-20 effect did not transfer

`RWKV_XHEAD_MIX=1` (the zero-init per-channel (H,H,K) cross-head delta, +896 params →
172,349) on the full iter-26 champion recipe. **Finals 0.304056/0.273513 (n=5000,
0 nanskips) — vs iter 26: ahead −0.000114 (p=1.0), imm −0.000160 (p=1.0), BOTH worse.**
The identical mechanism vs the iter-15 recipe was +0.000178/+0.000107 at p 2e-10/2e-25;
on the GRU-N=3/PAVA/no-residual recipe the channel measures *negative*. Plausible
mechanism: the GRU head restructures what the trunk must deliver, and the readout-mix
channel that helped the 64-basis softmax head is redundant-to-harmful for three tiny
(w,S,d) linears. **The transfer-failure ledger grows** (GRU imm win d=128→d=32; xhead
old→new recipe): old-recipe wins are never grafted, only re-measured. **V3 (wd
exclusion) deprioritized with INVERTED rationale** — it would let a negative-measuring
delta grow. Readout/xhead family: 0/3 on current lineages, effectively closed pending
genuinely new readout ideas. Vals: parity with iter 26 the whole run (decay-end
0.3261/0.3085 vs 0.3260/0.3082) — eval decided, as usual. Pipeline: WS 111m, decay 27m,
eval 87m, clean. Artifacts scratchpad/iter28_xhead/ (iter28d_1638.pth kept),
result/RWKV[-P]-iter28_xhead.jsonl. GPU → track-2 A7.

### Track-2 A6 — channel-mixer thinning (ACCEPTED 2026-07-20 10:50): new champion, −16% vs A4

The grad-stats shortlist cashed in: `RWKV_STRIP_CMIX=user_id:1,user_id:2,preset_id:1,
preset_id:2,deck_id:1` (the bottom-saliency tier, stable across 3 independent
recordings) on the A5 champion recipe. **2,115,359 → 1,949,624 params (−165,735 =
−7.83% vs A5; −16.0% vs A4).** New machinery: `RWKV_STRIP_CMIX` env in rwkv_model.py —
stream:layer list, dummy-mixer + residual-skip pattern (TorchScript-safe), and
`RWKV7Config.stream_name` stamped centrally in SrsRWKV so any arch module works.
Smokes: params exact, correct mixers by true stream name (which exposed the deck/note
ordering erratum), scripted-branch test (dodging the W_v zero-init trap), off-path
byte-identity.

**Finals 0.300429/0.269236 (n=5000, 0 nanskips, 0 clamp resets in eval) — vs A5:
ahead +0.000103 BETTER (p=1.1e-04); imm −0.000109 worse (p=1.0).** Ratio gate:
ahead negative (better), imm +0.0000658 per 100k = 1.5× inside the ≤0.0001 bar →
**ACCEPTED, new track-2 champion.** 165,735 params bought at an imm price of ~0.0001
with an ahead *improvement* — consistent with the stripped mixers competing for
regularization budget rather than contributing signal. Vals tracked A5 at parity the
entire run (WS-end 0.3254/0.3056 vs 0.3256/0.3056; decay-end 0.3229/0.3041 vs
0.3227/0.3038).

**A7 shortlist (from A6's own grad recording — note the diminishing freeness: this
tier's saliencies are ~2-3× the tier just stripped):** next mixers = user.L3, note.L1,
deck.L2, card.L1, deck.L3 (another 165,760), OR pivot structural: user 4L→3L,
d_model 128→96. Pipeline: WS 6h18m, decay 1h33m, eval 2h55m, clean. Artifacts
scratchpad/track2_a6/ (t2a6d_5586.pth kept), result/RWKV[-P]-track2_a6.jsonl.

### Iter 27 — GRU head N=4 (REJECTED 2026-07-20 00:01): the N-sweep peaks at 3

`RWKV_GRU_HEAD=4`, 171,840 params. **Finals 0.304353/0.273526 (n=5000, 0 nanskips) —
vs iter 26 (N=3): ahead −0.000411 WORSE (p=1.0), imm −0.000172 worse (p=1.0)**; vs
iter 25 (N=2) a null/mixed. Clean capacity peak: N=2 = parity at −11.7% params, N=3 =
real both-modes gain, N=4 = regression (the 4th curve overfits/dilutes the weight
softmax). **Sweep CLOSED, no N=5; iter 26 stands as champion.** Val trajectory tracked
N=3 at parity the whole run and lost eval by 0.0004 — the val-lead lesson holds again.
Pipeline: WS 112m, decay 26m, eval 89m, clean; A6 took the GPU on the DONE_EXIT.
Artifacts scratchpad/iter27_gru4/ (iter27d_1638.pth kept), result/RWKV[-P]-iter27_gru4.jsonl.

**VERDICT CHANGED — ACCEPTED under the NEW GATE (Andrew, 2026-07-19 ~21:00):** "let's
change the acceptance criteria: at least 0.0001 on both gates after rounding to 4
decimal points, so 0.000088→0.0001 passes." New magnitude bar (all future iters): each
mode's improvement rounded to 4 decimals ≥ 0.0001 (raw ≥ 0.00005), p<0.0001 both modes
unchanged; was ≥0.0003 through iter 25. Iter 26: ahead 0.0005 ✓, imm 0.0001 ✓ → **NEW
TRACK-1 CHAMPION (0.303942/0.273353, 171,453 params)**; champion_5k_plain.json
re-pointed; recipe now RWKV_GRU_HEAD=3. Iter 27 (N=4, mid-WS) gates vs iter 26 via its
GATE-B tail. Historical note: under this bar iter 20 (xhead-mix v1, +0.000178/+0.000107,
both p≪1e-9 vs iter 15) would also have passed — the xhead-mix v3 queue entry gains
priority accordingly; no retroactive flips (the champion lineage moved on).

## iter 31 — graft PAVA + GRU N=3 + Muon onto the A18 trunk (ACCEPTED, 2026-07-26)

**The first iteration of the merged lineage.** Track-2's A-series closed at A18, and track-1's
d=32 lineage closed at iter 29; this takes track-1's three algorithmic wins and puts them on the
A18 trunk. `scratchpad/iter31_algo/`, ckpt `iter31d_5586.pth`, 558,212 params (+966 vs A18 = the
GRU head going N=2→3, plus `pava_theta`'s 3 floats).

| | A18 | iter 31 | delta | p |
|---|---|---|---|---|
| ahead | 0.299302 | **0.298909** | +0.000393 | 5.99e-26 |
| imm | 0.268390 | **0.267637** | +0.000753 | 1.49e-209 |

n=2500 (val half 5001–7500), **0 nanskips**. Both deltas clear the ≥0.0001-after-4dp-rounding bar
(0.0004 / 0.0008) and both p-values clear 1e-4 by many orders. `size` IDENTICAL to A18 — 0/2500
mismatches, 128,800,080 reviews on both sides. Per-entity state unchanged (card 2,880 floats, note
1,440), as expected: PAVA and Muon are train-time only and the GRU head is a *head*, not a stream.

### The val-lag was a red herring for the third time

iter 31 TRAILED A18 on validation at matched steps all through WS (step 4000: 0.33287/0.31385 vs
A18's 0.33265/0.31356) and then won the eval in both modes. That is now three data points in both
directions — Muon/iter 29 trailed val for the whole run and won; iters 25 and 27 led val and lost.
**The rule stands: record val lag, never act on it.** Had this run been killed on the val trace it
would have cost the largest single accuracy gain of the merged lineage so far.

### What this does and does not establish

It establishes that the graft transfers: three levers tuned at d=32 still pay at d=80, which was
not obvious — the transfer ledger (iter 28) and A13's opposite-sign state price both say d=32 wins
need re-earning, and the LoRA-halving lever had just *flipped sign* between d=128 and d=80.

It does **not** attribute. This is a bundle of three changes in one run, so we know the package
works and not which part carries it. The imm gain (+0.00075) being nearly 2× the ahead gain
(+0.00039) is at least consistent with iter 29's finding that Muon's largest effect is on imm, and
PAVA is curve-side so it can only move ahead — but that is inference, not measurement. A clean
ablation is three more runs (~10 h each); deferred pending Andrew's call, since the bundle is
already accepted and the ablation buys understanding rather than accuracy.

### Two metrics, deliberately

iter 31's own eval leg is UNRECTIFIED, because its `.cmd` predates `RWKV_EVAL_PAVA` and a running
batch file must not be edited (cmd.exe re-reads it at a saved byte offset). That is the number
above and it is the PRIMARY gate, being directly comparable to A18's existing jsonls. A rectified
pair (A18 at classic p=1, iter 31 with its learned powers) runs separately as the deploy metric.
⚠ Note that rect-vs-unrect moves *two* things at once — the pooling and the scored row's duration
going real → 0 — so the rectified comparison cannot attribute its own result either. `RWKV_EVAL_PAVA=2`
was added the same day to separate them (see `research_5k_notes.md`).

### ★ RECTIFIED RESULT (landed 00:28, 2026-07-27) — both metrics agree, deploy is 5.3x better

| metric | A18 | iter 31 | delta | p |
|---|---|---|---|---|
| ahead unrect (PRIMARY gate) | 0.299302 | 0.298909 | +0.000393 | 6.0e-26 |
| imm unrect (PRIMARY gate) | 0.268390 | 0.267637 | +0.000753 | 1.5e-209 |
| **ahead RECT (deploy)** | 0.302890 | **0.300802** | **+0.002088** | 2.4e-160 |
| **imm RECT (deploy)** | 0.268670 | **0.267691** | **+0.000979** | 6.0e-294 |

The two metrics agree in sign in both modes, so the standing "if they disagree, report both to
Andrew — do not pick" instruction did not fire. The *magnitudes* differ a lot, and that difference
is the finding:

**Training under PAVA halves the deploy-time rectification cost.** Rectifying A18 — a model that
never saw the constraint — costs **+0.003588** on ahead. Rectifying iter 31, trained at
`RWKV_PAVA_LAMBDA=0.1`, costs only **+0.001893**. The earlier reading of A18's +0.003588 was
"post-hoc rectification badly hurts a model never trained under the constraint"; this pair supplies
the matched comparison that turns that into a *quantified* claim — ~47% of the penalty is
recoverable by training, i.e. it is a training problem, not an inherent cost of the rectifier.

**Residual, stated plainly:** iter 31 still pays +0.001893 on ahead to be rectified (0.298909 →
0.300802), and the rectifier is in the deploy contract, so **0.300802 is the number a user actually
gets**. Shrinking that residual is a live target (raise `RWKV_PAVA_LAMBDA`? raise probe density?
both are one-flag runs on an existing recipe).

### ★ WHAT THE RECTIFICATION PENALTY IS ACTUALLY MADE OF (mode-2 diagnostic, 2026-07-27)

Andrew asked how much zeroing the current review's duration costs. `RWKV_EVAL_PAVA=2` substitutes
the pressed probe WITHOUT pooling, so modes 0/1/2 split the penalty additively. iter 31, users
5001-5500 (n=500), additivity exact at 0.00e+00:

| component | ahead |
|---|---|
| duration zeroing (m2 - m0) | **+0.001451** |
| PAVA pooling itself (m1 - m2) | +0.000611 |
| total rect-vs-unrect (m1 - m0) | +0.002062 |

by-user mean ahead: unrectified 0.301461 -> raw probe 0.302912 -> rectified 0.303523.
Probe-insertion noise on the imm channel is **+0.000056** (identical for modes 1 and 2, p=2.1e-8,
worse on 284/500 — a good consistency check, since imm depends on probe INSERTION and not on what
gets substituted).

**★ MODE-3 CONTROL LANDED 01:29 (2026-07-27) AND THE AHEAD NOISE IS EXACTLY ZERO**, so the split
above needs no correction at all:

| component | ahead | p | worse on |
|---|---|---|---|
| probe-insertion noise (m3 - m0) | **+0.000000 ± 0.000014** | **0.33** | 253/500 (a coin flip) |
| duration zeroing (m2 - m3) — the clean pair | **+0.001451** | 1.8e-47 | 401/500 |
| PAVA pooling (m1 - m2) | +0.000611 | 3.9e-4 | 262/500 |
| total (m1 - m0) | +0.002062 | 1.7e-49 | 409/500 |

`m2 - m3` and `m2 - m0` agree to six decimals, which is the whole point of running the control: the
duration number was never confounded, and that is now measured rather than assumed.

⚠ **This refutes a generalization we were carrying.** A18's imm noise (+0.000280, n=2500) had been
written up as a blanket "probe insertion costs ~3x the gate". It is **channel- and model-dependent**:
on iter 31 it is 0 on ahead and +0.000056 on imm — 5x below A18 on identical probe machinery and
identical data. The "don't compare rectified to unrectified at the 0.0001 gate" rule survives for
**imm**, but must not be quoted for ahead or as a fixed magnitude; run the ~30 min control for the
model in hand. What damps it in iter 31 is untested — the state clamp and training WITH probes
(`RWKV_PROBE_DENSITY=0.08`, which A18 did not have) are the obvious candidates.

**The headline is that the rectifier is the SMALL half. ~70% of the penalty is the model losing the
current review's duration, and only ~30% is the monotonicity pooling.** That matters because it
redirects the obvious follow-up: raising `RWKV_PAVA_LAMBDA` can only attack the 30%.

**And the 70% is a TRAIN/DEPLOY MISMATCH, not a law of nature** — precisely the failure class the
§9 three-way-parity directive exists to catch. Training feeds the real `scaled_duration` for the
scored row; deploy CANNOT, because Anki has to show intervals *before* the user presses, so the
duration of a review that has not happened yet is unknowable. The model therefore learns to lean on
a feature that vanishes at serving time, and +0.00140 is the bill.
**=> the better iter-33 candidate is to zero the current row's duration during TRAINING** (the
`RWKV_ZERO_FEATURES` mechanism already exists and is already in use for dim 22), so train and deploy
compute the same quantity. Expect it to cost a little on the unrectified metric — that number is
measured WITH the duration the deploy path won't have — while removing most of the deploy penalty.
This is a case where the gate metric and the deploy metric point in opposite directions, so it must
be judged on both (see the gate question in CLAUDE.md QUEUE 0).

**It is not noise.** iter 31's own probe-insertion noise floor — its imm rect-vs-unrect delta, the
channel the rectifier cannot reach — is **+0.000054**, so the ahead penalty is ~19x it. Worth
flagging separately: that floor is **5x smaller than A18's +0.000280** on identical probe
machinery and identical data, i.e. the two models differ in how much bf16 batch re-bucketing
perturbs them. Candidate causes are the state clamp and PAVA training itself; untested.

## iter 32 — full-run distillation from the d=128 teacher (ACCEPTED, 2026-07-27)

**Family: DISTILLATION.** Note that iter 10 was mis-filed under early-training-intervention,
which is why the family scoreboard showed distillation nowhere; with this run the family is
**1/1 at full-run scale**.

**The design.** Teacher = the original 2.76M `pretrain/RWKV_trained_on_101_4999.pth` under
`scratchpad/architecture_old_d128.py`. Student = the iter-31 recipe *unchanged*, plus
`RWKV_KD_MIX` and the new **`RWKV_KD_ALPHA=0.5`** flag (holds alpha FIXED — the classic KD form;
leaving it unset preserves iter 10's linear 1->0 ramp byte-identically). The teacher's
`p_curve`/`p_again` were pre-dumped for all 22,346 WS steps rather than run live; the decay phase
trains on hard labels.

### Result (VAL half 5001-7500, n=2500, unrectified = the primary gate)

| metric | iter 31 | iter 32 | delta | p |
|---|---|---|---|---|
| ahead | 0.298909 | **0.298333** | +0.000577 | 2.28e-66 |
| imm | 0.267637 | **0.267207** | +0.000430 | 3.12e-143 |

`size` identical (0/2500 mismatches), nan_users 0, params **558,212** and per-card / note state
**2,880 / 1,440** floats — all unchanged, as they must be: KD is a training-time-only change and
nothing about it ships to Rust.

### What the run was actually for: how much of the teacher gap is transferable

The +0.0037/+0.0043 deficit to the old d=128 model is, per the decomposition in "Workbench +
baselines", mostly **training budget** (our 1.25 epochs vs upstream's ~12) rather than ablation
damage. A soft target from a model that *had* that budget is the cheapest way to test whether a
budget deficit is transferable at all. Against the d=128 model on the same VAL half
(0.294612 / 0.263561):

| | ahead gap | imm gap |
|---|---|---|
| iter 31 | -0.004297 | -0.004076 |
| iter 32 | -0.003721 | -0.003645 |
| **closed** | **13.4%** | **10.6%** |

So roughly an eighth of the gap moves across as a soft target. That is a real, positive answer to
a question that had never been tested here — and it also bounds the win: distillation alone will
not close the remaining ~0.0037, which is consistent with the budget story rather than against it.

### Cost — KD is nearly free

WS ran at **1.30-1.42 steps/s vs iter 31's 1.44 (~9% slower)**. The per-step read of the teacher
dump is not a bottleneck. The dump itself was 22,346 files / **6.96 GB** on C: and took **1h36m**
against a ~3h projection. Whole run 1:30:32 -> 9:49:36 (dump 1h36m, sanity 2m, WS 4h23m, decay
1h04m, eval 1h15m).

`pava_pool_frac` fell from 0.92 at step 1 to **0.075** at the final WS step — under KD the curve
head becomes nearly monotone on its own, so the rectifier barely has to pool anything.

### Two caveats, both deliberately unresolved

**1. The imm margin is inside the seed band.** +0.000430 is BELOW the ~0.0005 seed-pair
threshold (cross-seed spread on an identical recipe is ~0.0004 in both modes), so the imm win is
not resolved by one run; ahead at +0.000577 is above it. Precedent is mixed — iter 31 was
accepted with ahead +0.000393, also below the threshold, without a seed-pair run. Flagged to
Andrew rather than resolved unilaterally, since the doctrine's own wording is "needs".

**2. NOT promoted to champion yet.** Two independent reasons: (a) iter 32's eval is
**unrectified** — its `.cmd` predates `RWKV_EVAL_PAVA` — and from iter 33 on the gate basis is
the RECTIFIED metric, so iter 32 needs a rectified eval before it can serve as anyone's baseline;
(b) iter 33 was already running against iter 31's rectified jsonls when this landed, and
promoting mid-flight would invalidate that comparison. The rectified eval is a GPU job and the
GPU is busy until iter 33 finishes.

**Throughput** is likewise unmeasured — the no-co-tenant-GPU rule applies while iter 33 is
gate-critical. KD changes no architecture, params or ops, so the deployed model is shape-identical
to iter 31 and its throughput is iter 31's *by construction*. That is a derivation, not a
measurement; a real `measure_throughput.py` run is queued for when the GPU frees.

### Cheap follow-ups that reuse the same dump

The dump is the expensive part and it is on disk at `C:\rwkv_kd_dump\t128_iter32`. The
**annealed-alpha** variant (unset `RWKV_KD_ALPHA`, ramp over the full run) and an **alpha sweep**
are student-only re-runs. Curve-level distillation (matching the teacher's full 128-point curve
rather than its scalar outputs) needs new dump code.

### iter 32 addendum — rectified (deploy) eval + throughput, 2026-07-27 23:12

| metric | iter 31 | iter 32 | delta | p |
|---|---|---|---|---|
| ahead RECT (deploy) | 0.300802 | **0.300268** | +0.000534 | 3.39e-54 |
| imm RECT (deploy) | 0.267691 | **0.267262** | +0.000429 | 2.90e-144 |

**Both metrics agree**, which is the point of running it: unrectified was +0.000577/+0.000430 and
rectified is +0.000534/+0.000429. The gate-vs-deploy divergence that QUEUE item 0 worried about
does not bite here — KD's gain is real for users, not an artifact of which metric is scored.

**Throughput MEASURED: 1857.4 rev/s** (per-user samples 1823.9-1861.9). It had been recorded as a
derivation from iter 31 on the grounds that KD changes no architecture; that reasoning was sound
but it is now an actual measurement.

⚠ **Distillation does NOT shrink the rectification residual.** iter 32 pays +0.001935 on ahead to
be rectified (0.298333 -> 0.300268) vs iter 31's +0.001893 — statistically indistinguishable. So
the residual remains exactly where it was as a research target (QUEUE item 1: raise
`RWKV_PAVA_LAMBDA` and/or `RWKV_PROBE_DENSITY`), and KD is not a route to it.

**PROMOTED** to `champion_5k_track2.json` on the RECTIFIED numbers, since those are the gate basis
from iter 33 onward.
⚠ **The seed-pair caveat stands and is NOT closed by this run.** imm +0.000429 is still under the
~0.0005 threshold, and the rectified eval re-scores the *same training run* — it confirms the
metric, not the seed. Precedent (iter 31 accepted at ahead +0.000393 without a seed pair) supports
promoting; an independent `RWKV_AUGMENT_SEED=4321` run would settle it.

## iter 33 — withhold the current review's duration (REJECTED, 2026-07-29)

**Andrew's deploy-contract directive** (2026-07-27): train, eval and CPU inference must compute ONE
quantity — current-review duration zeroed, PAVA applied, no piecewise correction. iter 33 is the
training half of that: `RWKV_PROBE_DENSITY=1.0` + the new **`RWKV_AHEAD_PROBE_ONLY=1`**, moving the
ahead objective entirely onto the duration-zeroed probe path. First iteration gated on the
**RECTIFIED** metric.

### Result (VAL 5001-7500, n=2500, RECTIFIED both sides)

| metric | iter 32 (champ) | iter 33 | delta | p |
|---|---|---|---|---|
| ahead | 0.300268 | 0.303055 | **-0.002787** | 1.0 |
| imm | 0.267262 | 0.268066 | **-0.000805** | 1.0 |

`size` identical (0/2500), nan_users 0, params 558,212 unchanged. **REJECTED.**

### What it does and does not tell us

**Does:** the hypothesis as stated is not supported. Training without the current row's duration did
not shrink the deploy penalty — it made the deploy number worse. The motivating decomposition
(duration zeroing costs +0.001451 of iter 31's +0.002062 rect-vs-unrect gap) measured the cost of
*removing the feature at eval time*; it did not follow that *training* without it would recover
that cost, and it didn't.

**Does not:** attribute. **Three changes shipped together** — this is the run's design fault and it
should be named plainly:
1. **duration withheld** — the hypothesis;
2. **~~`RWKV_AHEAD_PROBE_ONLY=1` dropped ~23.5% of the ahead supervision.~~ ⚠ THIS WAS WRONG —
   CORRECTED 2026-08-10, and the true version is a bigger confound than the false one.** Verified
   in code at `srs_model.py:1010-1013`: the flag zeroes `ahead_mask` **only at probed rows**
   (`ahead_mask.reshape(-1)[_pt] = 0`), and the comment there says so explicitly — "Rows NOT
   eligible for probes keep the real-row term". So the 23.5% were **never dropped**; they kept
   full weight. What actually happened is the reverse and worse: the probed **76.5%** moved off
   the real-row term (scale `AHEAD_SCALE 0.5 + AHEAD_RAW_SCALE 0.5 = 1.0`, `:1069-1071,1094-1096`)
   and onto the PAVA probe path, which enters the loss as `loss_avg + self.pava_lambda * pava_loss`
   at **lambda = 0.1** (`:1114`) — an effective **10x downweighting** of the ahead objective for
   the majority of rows. Meanwhile the un-probed 23.5%, a *biased* subsample (first-in-chunk
   reviews = the cards with the least history), kept weight 1.0 and so absorbed the great majority
   of the remaining ahead gradient. A 10x cut plus a biased-subsample reweighting is a far larger
   perturbation than the signal loss we thought we had shipped;
3. **MAX 32768 -> 16384**, forced by VRAM at density 1.0, which halves batching and doubles the
   step count (43,354 vs 22,346). Structural, and worth ~1.84x throughput by itself.

Any of the three could produce -0.0028 — and confound 2, correctly understood, is the most likely
single culprit.

**If the family is retried:** do NOT use probes to withhold duration at all. The clean instrument
is per-row Bernoulli dropout on `scaled_duration` (dim 8) at the model input: no probe-density
change, therefore no row inflation, therefore no MAX change and no loss-term reweighting — only
the duration varies. `p = 1.0` is bracketed by iter 18 (permanent removal, -0.0018/-0.0024), so
the dose curve has a known bad end. **This correction is what makes the retry well-posed; the
old prescription ("keep the real-row term for the 23.5%") described fixing something that was
never broken.**

### Robustness note

31 h wall clock. The run survived a hard black-screen hang at WS step 33,003 and two clean
stop/resumes (Andrew's cable management), for a **total training loss of 3 steps** out of 43,354 —
the checkpoint-every-1000 + `RWKV_RESUME_SKIP_GROUPS` discipline did its job.

## iter 34 — HP tuning for the MAX=65536 era (ACCEPTED, 2026-08-05)

**What it is.** Not one run but the tuner's outcome: greedy coordinate descent over 10 levers,
24 journal rows (`optimization/tuner_5k_log.jsonl`), ~6 days of GPU. Andrew's framing fixed the
scope twice: the original goal was "recover the −0.0003 that MAX=65536 cost", and mid-grid he
retargeted it — *"We're looking for algorithmic improvements of any kind, there is no reason to
stop (other than exhausting this particular lever)"* — which added the momentum/beta2 extension
(his call, 2026-08-02) and then wd_head_mult/dropout_scale (2026-08-03).

**Winner recipe** (= `champion_5k_track2.json` -> `t65_dropout_scale_0p5d_10935.pth`): iter 32's
env + MAX=65536 + the 1.68x speed stack + `WARMUP_STEPS 400`, `RWKV_MUON_LR 0.0025` (8x below
the 0.02 tuned at MAX=32768), `decay_ratio 1.0`, `RWKV_DROPOUT_SCALE 0.5`. peak_lr/wd/clip/
momentum/beta2/head-wd all held their defaults.

**Result, rect-vs-rect vs iter 32 on the full VAL half (n=2500):** ahead 0.298970 (+0.001298,
p=1.8e-152), imm 0.266217 (+0.001044, p underflows). size 0/2500 mismatches, nan_users 0, params
558,212 and card/note state unchanged — training-recipe only, nothing new ships to Rust.
Throughput 1797.6 rev/s (deploy-identical to iter 32; the delta vs 1857.4 is bench noise).
Recovered the MAX cost ~4x over on ahead and ~3x on imm, while keeping the 1.68x training
speedup: accuracy and speed moved together, which is rare enough to note.

**What the grid actually learned, in credit order:**
1. **Muon's LR was the whole problem (+0.00183).** The batch doubling had NOT mis-set "the LR" —
   raising the joint LR is monotonically worse across 4 points — it had mis-set *Muon's* LR
   specifically. The AdamW side (10% of params) was fine at 1e-3.
2. **decay_ratio 1.0 (+0.00145)** — with the honest caveat that this lever buys TOTAL TRAINING
   (1.25 -> 2.0 epochs), a ~5.9 h/run budget every future iteration now pays. Andrew accepted
   the budget change implicitly by accepting the recipe; the 2c methodology note (epoch budget
   belongs to the endgame) is unchanged for WS.
3. **Momentum is an interior optimum at 0.95** (0.8 +0.000375, 0.9 +0.000176, 0.975 +0.000235 —
   a clean inverted U). This *inverts* the motivating hypothesis: if the LR cut had been about
   effective step size (~lr/(1−m)), momentum would substitute for it. It does not — the two
   knobs are independent, and the batch change broke only one of them.
4. **beta2 0.999 held** (0.98 +0.000048, 0.95 +0.000097 — mild monotone worsening).
5. **The per-group wd split did not pay**: head-wd flat across 25x (0.2x +0.000064, 5x
   +0.000101). The peak_lr-style "shared knob hides a split optimum" prior was worth testing and
   is now answered for wd.
6. **dropout x0.5 (+0.000254 subset, COIN FLIP by the tuner's own report) — but it survived the
   full-VAL confirmation**, beating the pre-dropout winner there too (+0.000089/+0.000152).
   Coherent with the capacity-limited-trunk story: a model past its width floor wants LESS
   regularization. ⚠ This lever was a SILENT NO-OP until 2026-08-03 — architecture_d80_lora4.py
   had hardcoded the dropout rates at fork time, eating RWKV_DROPOUT_SCALE — so its coordinate
   only exists because the fix landed first. The other two silent-null finds of the same sweep:
   the momentum env emitted twice per .cmd (benign, last-set-wins, removed), and canon() blind
   to JSON-only coordinates (fixed + self-tested).

**Caveats carried forward.** warmup 400 and dropout 0.5 were adopted INSIDE the ~0.0008 noise
band — the recipe carries them because they won, not because they are individually proven. The
seed pair (next in Andrew's order) re-runs the tuned recipe at seed 4321 and doubles as their
robustness check; if the margin collapses there, the suspects are exactly these two levers.

**Ops record:** one 47-min idle-GPU incident (a running loop's SPACE is baked at import; grid
growth mid-run must edit tuner65k_space.json — lesson now in load_space()'s docstring), one
giant-user OOM eval recovered by the designed no-del retry, one stale-confirm-jsonl hazard
caught before it bit (t65confirm files archive at generation time now — they would have made
eval_sharded skip all 2500 users and "confirm" the new checkpoint with the old winner's
numbers).

## iter 35 — the seed pair; KD confirmed at a second seed and restored to the recipe (ACCEPTED, 2026-08-06)

**What ran.** Andrew's spec (2026-07-30): "let's do iter 31 and iter 32 both with the same seed
after HP tune" — a two-arm, seed-matched test on the tuned recipe. One detached chain
(`scratchpad/seedpair65k/run_seedpair.cmd`, 20.9 h unattended): fresh d=128 teacher dump at
seed 4321 + MAX=65536 (`C:\rwkv_kd_dump\t128_seedpair_65k`, 10,935 steps, 7.7 GB — a NEW dir;
the runner rmdir's its target, and reusing `t128_iter32` would have destroyed iter 32's only
exact-reproduction artifact) → arm A `sp4321_nokd` (iter-34 tuned recipe, no KD) → arm B
`sp4321_kd` (same + `RWKV_KD_MIX` + `RWKV_KD_ALPHA=0.5`, KD WS-only). Both arms
`RWKV_AUGMENT_SEED=4321`, full-VAL 5001-7500 RECTIFIED evals, 2500/2500 users on attempt 1,
zero exit-43s (the dump's batch-stream identity held for all 10,935 steps), KD wall-clock cost
~0 (both arms 3.4 h WS).

**Question 1 — does KD still win at a second seed? YES.** Within seed 4321:
(B−A) = **+0.000160 ahead / +0.000251 imm**, p = 4.3e-13 / 3.3e-75, KD better on 1435/2500 and
1629/2500. Smaller than the seed-1234 untuned-recipe margin (+0.000534/+0.000429, iter 32 vs
iter 31 rect) — plausibly the tuned recipe (2.0 epochs, Muon LR ÷8) already captures part of
what KD provided — but the direction and significance are unambiguous. Distillation family 2/2.

**Question 2 — is iter 34 seed-robust? YES, remarkably.** Arm A reproduces the seed-1234
champion's means to **+0.000006 ahead / −0.000020 imm** — and this is genuine stability, not a
replay: the log confirms fetch seed 4321, and per-user losses differ substantially (mean |d|
0.001188 / 0.000501, max |d| 0.087, candidate worse on 1237/2500 and 1238/2500 — exact coin
flips both modes). Real per-user differences cancelling to a near-zero mean. The inside-noise
adoptions (warmup 400, dropout ×0.5) survive their robustness check. Observed cross-seed spread
on this recipe ~2e-5 — ~20× tighter than the ~0.0004 doctrine figure, which came from the QAT
d=32 era; recorded as pair-specific, NOT a new general rule.

**The promotion (arm B → champion).** The tuner's trials ran WITHOUT KD (the old dump was bound
to MAX=32768 + seed 1234), so iter 34 had silently dropped iter 32's accepted KD win; arm B is
the tuned recipe with KD restored — and it beats iter 34: **rect ahead 0.298816 (+0.000153,
p=5.9e-11) / imm 0.265946 (+0.000271, p=7.9e-71)**, 4-dp deltas 0.0002/0.0003, both ≥0.0001.
It also passes against its SAME-SEED twin A (+0.000160/+0.000251), so the accept is not
cross-seed luck. size 0/2500 vs both champion and A; params **558,212 exact** (incl.
`pava_theta` — `model_stats.py` without the PAVA env prints 558,209; set `RWKV_PAVA_LAMBDA`
when checking); card/note state 2,880/1,440 unchanged (KD is train-time only; nothing new ships
to Rust); throughput 1829.7 rev/s (same deploy model as iter 32/34 — deltas are bench noise).
ckpt `scratchpad/seedpair65k/spb_d_10935.pth`; `champion_5k_track2.json` points at it.

**Operational consequences.**
- **The champion is at seed 4321 and its recipe includes KD.** Future candidates should run
  `RWKV_AUGMENT_SEED=4321` WITH KD — that keeps gates same-seed AND reuses the
  `t128_seedpair_65k` dump for free.
- The dump is bound to db/MAX/seed AND the probe layout: data-side levers (MAX,
  `RWKV_PROBE_DENSITY`) need a fresh dump (~1.5 h + 7.7 GB); model-side levers (PAVA lambda)
  reuse it. Keep `t128_iter32` (iter 32's audit artifact).
- The champion json's trace was EXTRACTED from the WS log
  (`scratchpad/seedpair65k/extract_trace_from_log.py`; the arms deliberately ran without
  `RWKV_STEP_TRACE`) — 4-dp precision, val points at 1000-step cadence, step numbering 0-based
  vs native traces' 1-based. Fine for vprune/pairing; noted for provenance.

## iter 36 — PAVA lambda dose pair: a clean monotone trade, λ=0.2 DIRECTED-ACCEPTED (2026-08-07)

**What ran.** Andrew's queued order ("iter 36 = tuning RWKV_PAVA_LAMBDA"). Two dose points on
the iter-35 champion recipe — seed 4321, KD reusing `t128_seedpair_65k` (lambda is model-side,
dump valid), tuned HPs — with ONLY the lambda changed: trial 1 = 0.3
(`scratchpad/iter36_pava/run_iter36.cmd`), trial 2 = 0.2 (`run_iter36_l02.cmd`). Each trial ran
TWO full-VAL evals per the family's measure-both-metrics rule: rectified (the gate basis) and
unrectified (the decomposition). Both chains clean end-to-end (~10.9 h and ~10.8 h), all four
evals 2500/2500 on attempt 1, size 0/2500 vs the champion in both trials, nan_users 0.

**The dose-response (rectified, vs the λ=0.1 champion 0.298816/0.265946):**

| λ | rect ahead | Δahead | rect imm | Δimm | unrect ahead | unrect imm | rect penalty (ahead) |
|---|---|---|---|---|---|---|---|
| 0.1 | 0.298816 | — | 0.265946 | — | (not measured) | (not measured) | ~+0.0019 (iter-31 basis) |
| 0.2 | 0.298338 | **+0.000478** (p=5.1e-67) | 0.266027 | −0.000081 (p=1.0) | 0.296795 | 0.265959 | +0.001544 |
| 0.3 | 0.298225 | **+0.000591** (p=8.4e-82) | 0.266126 | −0.000180 | 0.296892 | 0.266063 | +0.001334 |

Both responses are monotone in opposite directions, so **no λ>0.1 can pass the both-modes
gate**: the ahead gain is concave (81% of the λ=0.3 gain already at 0.2) while the imm cost is
~linear (~−0.00008 per +0.1 of λ). Mechanical verdict: REJECTED, both trials.

**What the decomposition says.** The hypothesis is CONFIRMED on its own terms: training under
more monotonicity pressure keeps buying down the deploy-time rectification penalty
(~+0.0019 → +0.001544 → +0.001334 ahead), and the raw (unrectified) curve barely pays — at
λ=0.3 raw ahead is unchanged and at λ=0.2 it is the best measured (0.296795). The imm
regression is a genuine training-side effect, not an artifact: its rect−unrect gap
(+0.000068/+0.000063) matches the known probe-insertion noise (+0.000056 on iter 31), so the
rectifier itself still cannot touch imm; the cost leaks in through the shared trunk during
training.

**THE VERDICT — ANDREW DIRECTED-ACCEPTED λ=0.2 (2026-08-07).** Precedent iters 22/23, where
monotonicity constraints were also accepted for the constraint rather than the logloss. The
choice was between (a) keep λ=0.1, (b) λ=0.2, (c) λ=0.3; λ=0.2 was the recommendation because
it offers a **5.9:1 exchange** (ahead +0.000478 for imm −0.000081, the imm cost being exactly
one 4-dp tick) whereas the marginal step 0.2→0.3 trades ~1.1:1 — you pay nearly as much imm as
you gain ahead. Both deploy surfaces ship, which is why the trade is real in both directions:
button intervals come from the curve head (the ahead quantity), `retrievability_head` =
1−P(Again) from the rating head (the imm quantity). Promotion-only — the checkpoint already
existed (`scratchpad/iter36_pava/i36b_d_10935.pth`), nothing was retrained. Throughput 1769.4
rev/s; `champion_5k_track2.json` now points at it.

**★ THE DEPLOY CONTRACT NOW CARRIES `RWKV_PAVA_LAMBDA=0.2`** — every future run on this trunk
must set it. 0.1 was the value from iter 31 through iter 35; a run that copies an older env
string silently trains at the wrong pressure and its gate is not comparable.

**Family scoreboard:** curve-shape constraints 2/3 (PAVA accepted iter 23, λ=0.2 accepted here;
λ=0.3 rejected as the worse point of the same lever).
⚠ Cross-seed robustness of the λ effect is untested (single seed 4321, same-seed gates). The
iter-35 pair measured ~2e-5 recipe-level spread on this trunk, so the +0.00048 ahead margin is
far above plausible seed noise; flagged for completeness, not as a blocker.

## iter 37 — by-user loss weighting: refuted in every size quartile (REJECTED, 2026-08-08)

**What ran.** Family **objective alignment** (NEW); direction chosen by Claude under Andrew's
"It's up to you". The gate is the BY-USER mean LogLoss (every user counts once) but the
training loss was a flat mean over ROWS — and the measured imbalance is **4,308× rows/user**
(1,179..5,079,718; the single largest user carried ~0.8% of ALL gradient signal while counting
as 1/5000th of the eval). `RWKV_USER_WEIGHT=1` weights each chunk 1/N_u, mean-normalized so the
global gradient scale (and therefore the tuned LRs) stay sensible. Implementation invariants,
all verified before launch: objective terms only (the reported equalize metrics and `*_n`
counts stay unweighted, so step traces remain comparable across runs); attached in the train
loop, not the fetch workers (batch stream byte-identical ⇒ the KD labels checksum still passes,
confirmed live at step 1); OFF is bit-identical (`scratchpad/parity3/smoke_user_weight.py` —
scripted compile, unit-weight bit-exactness, independent reference-formula match). The runner
greps the `[user-weight] ON` banner in BOTH WS and decay (the silent-null-lever lesson).
Recipe = the iter-36 champion exactly (seed 4321, KD α=0.5, PAVA λ=0.2). Chain clean
end-to-end; eval 2500/2500 on attempt 1; size 0/2500; nan_users 0.

**Verdict: REJECTED — both modes worse.** Rect ahead 0.298603 (−0.000264 vs the champion),
imm 0.266118 (−0.000091), p=1.0 both.

**The finding worth keeping — the mechanism is refuted, not underdosed.** Per-user delta by
size quartile (negative = iter 37 better):

| quartile | size range | mean Δ | iter 37 better on |
|---|---|---|---|
| smallest 25% | 845–8,040 | **+0.000143** | 308/625 (coin flip) |
| 25–50% | 8,045–19,305 | +0.000202 | 290/625 |
| 50–75% | 19,335–52,615 | +0.000371 | 226/625 |
| largest 25% | 52,820–2,566,240 | +0.000341 | 244/625 |

If distribution-matching worked, the small users would improve while the large ones paid.
Instead **everyone** got slightly worse — including the intended beneficiaries. Reading: a
small user's per-user loss is driven by *generalization from the shared trunk*, and the trunk
is trained best by raw data volume; down-weighting the data-rich users just discards gradient
signal (consistent with the standing data-limited lesson). A milder dose (1/√N) would
interpolate toward zero effect — not worth 9.4 h. **Family objective-alignment 0/1,
mechanism-refuted, deprioritized.** The hook stays in-tree, env-gated, default off,
bit-identical when off.

## iter 38 — KD alpha 0.75: the nearest miss of the phase (REJECTED by 2e-6, 2026-08-08)

**What ran.** `RWKV_KD_ALPHA` 0.5 → 0.75 on the iter-36 champion recipe — the first tuning of a
value that was iter 32's opening guess. Family distillation (2/2). Dump reused (alpha is
loss-time); KD WS-only as always; guard greps `alpha=0.7500` in the WS log. Chain clean
end-to-end, eval 2500/2500 on attempt 1, size 0/2500, nan_users 0.

**Verdict: REJECTED — by two millionths.** Rect vs the champion: **ahead 0.298224 (+0.000115
better, p=3.9e-6) / imm 0.265979 (+0.000048 better, p=8.6e-7)**. Both modes improved; the
p-gate passes both modes; ahead clears the magnitude bar — but imm's raw +0.000048 rounds to
0.0000 against the ≥0.00005 requirement (Andrew's 2026-07-19 rounding rule, applied as
written). The bar is the bar; not relitigated unilaterally, but this is the phase's closest
near-miss and is flagged for Andrew: **a directed accept would be promotion-only**
(ckpt `scratchpad/iter38_kda/i38_d_10935.pth`, jsonls banked as `RWKV-iter38_kda075-s0`).

**Follow-up per conduct rule 2 (barely-missed → variant): iter 39 = α 0.9 is running** — the
third dose point maps the curve: monotone-up ⇒ keep climbing toward pure-teacher WS;
down ⇒ 0.75 is the peak and the lever's verdict is "sub-bar positive". Note the WS/decay
split matters here: KD applies to WS only, and decay (half of total training at ratio 1.0)
still runs pure hard labels — an annealed-alpha or KD-through-decay variant is the next
in-family idea if the dose curve alone cannot clear the bar.

**Ops lesson (cost ~13 min GPU):** iter 39's runner was generated from iter 38's by string
replacement using forward-slash paths only — the backslash `set DIR=` line kept pointing at
`iter38_kda`, sending logs (and eventually grad_stats) into the wrong directory. Caught by the
post-launch banner-verify step, killed at minute 4, fixed, relaunched. When generating a `.cmd`
by replacement, replace BOTH slash styles — or write the file fresh. (iter 38's grad stats
preserved as `grad_stats_iter38.json`.)

## iter 39 — KD alpha 0.9: clean accept, the dose curve is monotone up (ACCEPTED, 2026-08-08)

**What ran.** The third point of the alpha sweep (`RWKV_KD_ALPHA` → 0.9) on the iter-36
champion recipe; KD WS-only from the reused dump. Chain clean, eval 2500/2500 attempt 1.

**Verdict: ACCEPTED — the first full-gate pass since iter 35.** Rect vs iter 36:
**ahead 0.298180 (+0.000158, p=2.2e-10) / imm 0.265875 (+0.000153, p=7.8e-37)**; size 0/2500;
params 558,212 exact; card/note state unchanged (alpha is loss-time only — nothing new ships to
Rust); throughput 1823.8 rev/s. Promoted: `champion_5k_track2.json` →
`scratchpad/iter39_kda9/i39_d_10935.pth` (trace extracted from the WS log, same convention as
iters 35/36).

**The dose curve (all rect, vs the α=0.5 champion iter 36):**

| α | ahead Δ | imm Δ |
|---|---|---|
| 0.5 (iter 32's guess) | — | — |
| 0.75 (iter 38) | +0.000115 | +0.000048 |
| 0.9 (iter 39) | +0.000158 | **+0.000153** |

Monotone up in both modes, and **imm is accelerating** (+0.000104 more between 0.75 and 0.9,
p=4.4e-21 paired against the 0.75 run directly). This **moots iter 38's directed-accept
question** — 0.9 dominates 0.75 (vs 0.75: imm better at p=4e-21, ahead +0.000043 not-worse).
Reading: at a 1-epoch WS budget, the 12-epoch teacher's soft targets are simply a better
supervision signal than a 50/50 mix — and the hard labels still anchor the entire decay phase,
which at ratio 1.0 is half of total training.

**★ THE CHAMPION RECIPE'S `RWKV_KD_ALPHA` IS NOW 0.9** (0.5 was the value from iter 32 through
iter 36) — every future run on this trunk must set it; an env string copied from an older `.cmd`
silently trains at the wrong mix.

**iter 40 = α 1.0 is already running** — pure-teacher WS / hard-label decay, the
pretrain-finetune split and the curve's endpoint. Worse ⇒ the peak is bracketed in [0.75, 1.0]
and 0.9 stands; better ⇒ the WS phase wants no hard labels at all. Family distillation 3/3
accepted (+1 dominated near-miss).

## iter 40 — KD alpha 1.0: the endpoint that brackets the peak (REJECTED, 2026-08-09)

**What ran.** `RWKV_KD_ALPHA=1.0` — pure-teacher WS (hard labels contribute nothing during
warmup-stable), hard-label decay: the classic pretrain-finetune split, and the natural
endpoint after 0.9's clean accept. Chain clean, eval 2500/2500 attempt 1, size 0/2500.

**Verdict: REJECTED vs iter 39, exactly as an endpoint probe should fail.** Ahead
0.298199 (−0.000019, p=0.71 — a coin flip); imm 0.265942 (−0.000067, p=1.0 — genuinely
worse). The full dose curve, rectified on the VAL half:

| α | rect ahead | rect imm |
|---|---|---|
| 0.5 (iter 36) | 0.298338 | 0.266027 |
| 0.75 (iter 38) | 0.298224 | 0.265979 |
| **0.9 (iter 39, champion)** | **0.298180** | **0.265875** |
| 1.0 (iter 40) | 0.298199 | 0.265942 |

Concave with an interior optimum at ~0.9 — **the alpha lever is now mapped end-to-end and
closed**. Reading: even at a 1-epoch WS budget, where the 12-epoch teacher's soft targets
dominate the signal, the 10% hard-label component still earns its keep; removing ground truth
entirely from WS costs real imm. Family distillation closes the sweep at 3 accepts (32,
35-arm-B, 39) + 1 dominated near-miss (38) + this informative endpoint reject. Still open
in-family if revisited: annealed alpha, KD-through-decay (decay is HALF of total training at
ratio 1.0 and runs pure hard labels today).

## iter 41 — interleaving + fine-to-coarse order: the topology family opens with the phase's largest architectural gain (ACCEPTED, 2026-08-09)

**What ran.** The 2-change bundle Andrew directed: `RWKV_INTERLEAVE=1` (round-robin layer
schedule — layer r of every stream per round, 4 rounds / 13 layer-steps, same params/states/
ops, execution order only) on `architecture_d80_lora4_cnd.py` (stream tuples reordered to the
true fine-to-coarse card→note→deck→preset→user; the audit had found every historical arch file
ran card→DECK→NOTE while the docs claimed otherwise). Recipe = the iter-39 champion (seed
4321, KD α=0.9 WS-only, PAVA λ=0.2, tuned HPs). Chain fired itself 5 min after iter 40's
DONE_EXIT and ran clean end-to-end; eval 2500/2500 attempt 1.

**Verdict: ACCEPTED — rect ahead 0.297889 (+0.000291, p=5.1e-24) / imm 0.265479 (+0.000396,
p=7.5e-95)** vs iter 39. The largest both-modes architectural gain of the phase — no single
PAVA/GRU/Muon/KD accept moved both modes this much. Gates: size 0/2500; params 558,212 EXACT
(a schedule has no weights); card 2,880 / note 1,440 / deck 5,760 state floats all unchanged;
throughput 1849.8 rev/s (bench noise). WS wall-clock unchanged (~0.91 steps/s — the 13
gather/scatter round-trips through the canonical layout cost nothing measurable).

**Why it works (the mechanism the family was opened on):** the sequential chain gives global
context no path into card-level processing — the card stream is finished before any other
scope has run. Interleaved, every scope's later layers condition on every other scope's
earlier layers. The imm gain (rating head, fed by the full trunk) being the LARGER of the two
is consistent with that story: prediction-time integration of scopes improved.

**Verification behind the accept:** the depth-1 oracle (all depths 1 ⇒ interleaved schedule
degenerates to the sequential order) proved the gather composition BIT-EXACT vs the
battle-tested branch; the smoke was re-run under the reordered champion arch (banner
[2,1,4,3,3], no-grad sets identical, grads finite, scripted compile); the RNN deploy path now
routes states BY NAME (the old positional body would have silently cross-wired deck/note
states under any reordered arch — refactor verified golden-exact against pre-edit behavior).

**Caveats, all recorded in the jsonl row:**
- **Bundle attribution unknown** — and deploy-relevant: if the ORDER alone carries the gain,
  Rust never needs the interleave port. iter 42 = the de-bundle control (sequential +
  reordered order), gate vs iter 41.
- ahead's +0.000291 sits below the old ~0.0005 seed-doctrine line (but 15× the ~2e-5 trunk
  spread iter 35 measured). Flagged, not blocking.
- **Deploy:** the champion is now interleaved + reordered; `rust/rwkv-infer` implements
  neither (its chain is hardcoded sequential card→deck→note). The port plan gains both items;
  pre-ship trace parity (export_rnn_trace vs the by-name RNN mirror) is mandatory.
- `model_stats.py` labels per-stream states by a hardcoded name order — under reordered archs
  the deck/note labels swap (values positional, correct). FIXED 2026-08-09 (`bd473b2`): labels
  now come from the config's own module names; verified under both orders.

*(Attribution resolved by iter 42 below: the interleave carries all of it, and the reorder is
a small negative. The bundle's net gain understates the schedule's effect.)*

## iter 42 — the de-bundle control: interleaving carries it, the reorder is a small negative (REJECTED, 2026-08-09)

**What ran.** The fine-to-coarse ORDER alone: `architecture_d80_lora4_cnd.py` with
**sequential** execution (`RWKV_INTERLEAVE` explicitly cleared, and the runner fails with
`DONE_EXIT_ILVLEAK` if the interleave banner appears in either training phase — a lingering
env var would have silently re-run iter 41). Everything else identical to iters 39/41: seed
4321, KD α=0.9 WS-only, PAVA λ=0.2, tuned HPs, MAX=65536. Chain clean, eval 2500/2500 on
attempt 1, leak guard 0 occurrences, arch banner confirmed `_cnd`.

**Verdict: REJECTED as a candidate — rect ahead 0.298379 (−0.000489) / imm 0.266090
(−0.000612) vs iter 41, p=1.0 both.** But it is the most informative run since the merge,
because the order is not merely *weaker* than the bundle — it is a **negative in its own
right**. Against iter 39, which is the same sequential recipe at the OLD card→deck→note order,
it loses **−0.000198 ahead / −0.000215 imm**.

**The 2×2 (rectified, VAL half, n=2500):**

| | sequential | interleaved |
|---|---|---|
| **old order** (card→deck→note) | iter 39: 0.298180 / 0.265875 | **not yet run** |
| **new order** (card→note→deck) | iter 42: 0.298379 / 0.266090 | iter 41: 0.297889 / 0.265479 |

**Three conclusions.**
1. **Interleaving carries the entire iter-41 gain and more.** Holding the new order fixed, the
   schedule is worth **+0.000489 ahead / +0.000612 imm** (iter 41 − iter 42) — it had to
   overcome the order penalty to deliver its net +0.000291/+0.000396. The bundle's headline
   *understates* the schedule.
2. **Deploy: `rust/rwkv-infer` DOES need the interleave port.** The cheap escape is closed. And
   since the champion ships with the reorder too, the port carries both.
3. **Fine-to-coarse is not automatically better.** The intuition behind the reorder (notes
   ≈0.9× cards, decks ≈56/user, so note is "finer" than deck) does not survive measurement.
   Whatever the deck stream contributes, the note stream benefits from receiving it — the
   right axis is information flow, not granularity. That is also the mechanism story
   interleaving tells, so the two findings agree.

**The implied next run is the missing cell** — interleave + the ORIGINAL order (iter 41 minus
the reorder). If the −0.0002 order penalty is roughly additive under interleaving, it lands
≈0.29769/0.26526 and passes the gate against the current champion. One arch swap on an
existing recipe (~8.6 h), and well-posed either way: if the penalty *vanishes* under
interleaving, that is itself the finding — within-round order stops mattering once every scope
hears every other across rounds.

## iter 43 — the 2×2's fourth cell: under interleaving, stream order stops mattering (REJECTED as a TIE, 2026-08-10)

**What ran.** `RWKV_INTERLEAVE=1` on the **original** stream order
(`architecture_d80_lora4.py`, card→deck→note→preset→user, depths [2,4,1,3,3]) — iter 41 minus
the reorder. Recipe otherwise identical to iters 39/41/42. Guards inverted vs iter 42: the WS
log had to carry the interleave banner and had to *not* name the `_cnd` arch.

**Verdict: REJECTED, but as a statistical TIE, not a loss** — ahead 0.297964 (−0.000075,
p=0.42) / imm 0.265464 (+0.000014, p=0.098) vs the iter-41 champion. This is the least
significant pairing of the entire phase; every other iteration pair has at least one mode at
p<1e-9. size 0/2500, nan_users 0, params 558,212 and card/note/deck state 2,880/1,440/5,760
identical.

**The completed 2×2 (rectified, VAL half, n=2500):**

| | sequential | interleaved |
|---|---|---|
| **old order** (card→deck→note) | iter 39: 0.298180 / 0.265875 | iter 43: 0.297964 / 0.265464 |
| **new order** (card→note→deck) | iter 42: 0.298379 / 0.266090 | iter 41: 0.297889 / 0.265479 |

**Decomposed:**

| effect | at old order / sequential | at new order / interleaved |
|---|---|---|
| **interleave** | +0.000216 / +0.000411 | +0.000490 / +0.000611 |
| **reorder** | −0.000199 / −0.000215 | +0.000075 / −0.000015 (noise) |

**The finding — and it is the alternative flagged at launch.** The interleave effect is large
and robust to order (both cells p<1e-25). The reorder effect is a real negative when the chain
is sequential and **vanishes** when it is interleaved. So the two changes iter 41 bundled do
not merely add — they *interact*, and the mechanism is the same one the family was opened on:
once every scope hears every other scope across rounds, the order they are heard in *within* a
round carries almost no information. The reorder was never a second improvement; it was a cost
that the schedule paid off.

**Four consequences.**
1. **The champion is unchanged (iter 41).** A tie is not a promotion.
2. **Deploy, and this is the actionable one:** the reorder can be dropped at zero measured
   accuracy cost, which removes **port gap 8** from `rust/rwkv-infer` and lets the engine keep
   its existing hardcoded card→deck→note chain. One less thing to get wrong in the deploy path
   for a delta the data cannot distinguish from zero. That is a Pareto-simplicity call for
   Andrew, not a gate decision — recorded and flagged, not acted on.
3. **The ORDER lever is CLOSED.** Measured at both schedule settings: negative sequentially,
   null under interleaving. No reason to try further permutations.
4. **The SCHEDULE lever is where the next TOPOLOGY iterations go** — it is the phase's most
   productive single change and iter 43 shows its gain does not depend on the ordering choice
   that shipped with it.

Family TOPOLOGY: 1 accept + 2 rejects, and both rejects were *controls that changed what we
believe* rather than failed candidates.

**Ops — a self-inflicted chain break, zero GPU time lost.** WS completed normally
(`i43_ws_10935.pth` + `grad_stats.json` at 02:54:37), then the chain died 27 s later with
`DONE_EXIT_WSFAIL_9009` and `'ratchpad' is not recognized`: cmd.exe re-reads a batch file from
a saved byte offset after each command returns, and `git rebase --autostash` had **rewritten
the running `run_iter43.cmd`** on disk (it was committed after launch). Same hazard class as
CLAUDE.md's live-`.cmd`-edit warning, now generalized: **no git operation that rewrites the
working tree may touch a running runner's path until its chain reports DONE_EXIT.** Recovered
by resuming decay+eval from the final WS checkpoint (`run_iter43b.cmd`, new filename, with a
guard asserting step 10935 rather than a partial). Collateral: the garbage line's redirect
truncated the WS log to 99 B, so iter 43 has no step trace — irrelevant here since it is not
promoted, but it would have cost the vprune val trace had it won.

## iter 44 — endpoint-anchored placement: the third indistinguishable schedule (REJECTED as a TIE, 2026-08-10)

**What ran.** `RWKV_ILV_SPREAD=1`, one flag on the iter-41 champion recipe. Front-loading puts
layer *j* in round *j*, so a **shallow stream only ever runs EARLY** and can never consume the
cross-scope context interleaving exists to expose: in the champion (`_cnd`, depths [2,1,4,3,3])
the **note** stream has depth 1, runs in round 0 and never again — it feeds the global context
but never reads it. SPREAD distributes each stream's layers across all rounds with the endpoints
anchored (layer 0 → round 0, last layer → last round, a depth-1 stream goes LAST):
card `[0,-,-,1]`, note `[-,-,-,0]`, deck `[0,1,2,3]`, preset `[0,-,1,2]`, user `[0,-,1,2]`.
Same params, same per-entity states, same op count — **placement only**.

**PREDICTION MADE BEFORE LAUNCH: a gate-passing gain in both modes**, on the reasoning that note
would finally read global context and every stream's final layer would be computed with maximal
context.

**Verdict: REJECTED as a TIE, and the prediction was WRONG** — recorded as such, because a
well-motivated mechanism that produces nothing is the most useful kind of miss. vs iter 41:
ahead 0.297912 (−0.000023, p=0.93) / imm 0.265479 (−0.000001, p=1.0e-4 — rank-significant but
magnitude-null: most users shift slightly one way while the mean is identical to 6 dp,
0.265479 vs 0.265479). size 0/2500, nan_users 0, params 558,212, card/note/deck state unchanged.

**THE REAL FINDING, which needs all four TOPOLOGY runs to see.** Iters 41, 43 and 44 are three
structurally different schedules that are mutually **indistinguishable**:

| pair | Δ ahead | Δ imm |
|---|---|---|
| 41 vs 43 | −0.000075 | +0.000014 |
| 41 vs 44 | −0.000023 | −0.000001 |
| 43 vs 44 | +0.000052 | −0.000015 |

every |Δ| ≤ 7.5e-5 — while **sequential-vs-interleaved** differs by +0.000216..0.000611. So the
binary *"do scopes interact across rounds at all?"* is worth +0.0002..0.0006 in both modes, and
the **details** of how they interact (within-round order, cross-round layer placement) are worth
approximately **zero**. Reading: what the model gains is the EXISTENCE of a cross-scope
information path, not any particular choreography of it; once every scope can see every other,
the specific rendezvous points carry almost no information.

**Consequences.**
1. The **rearrangement sub-family of TOPOLOGY is EXHAUSTED** — order (iters 42/43) and placement
   (44) are both null, and a third rearrangement variant would be a fourth draw from a
   distribution whose spread we have now measured. Further TOPOLOGY gains must change **what** is
   computed, not merely **when**: extra rounds via layer reuse, cross-stream fusion, added
   connectivity.
2. **A useful calibration falls out free:** ±7.5e-5 is the empirical full-budget spread between
   same-capacity schedule variants. It sat AT the then-current raw accept threshold (0.00005,
   i.e. 0.0001 after 4-dp rounding), so a rearrangement-class "win" of exactly one tick had to be
   treated as unresolved rather than a pass. **This measurement is what moved the accept bar to a
   raw 0.0001** (Andrew, 2026-08-10).
3. **Deploy unaffected** — spread is not adopted, so the Rust port's gap 7 is the FRONT-LOADED
   schedule. (Closed and parity-verified 2026-08-11; `interleave_visits()` implements front-loaded
   only, and adopting spread later would need `interleave_schedule()`'s table instead.)

**Chain clean:** `DONE_EXIT_0`, eval 2500/2500 on attempt 1, SPREAD banner verified in BOTH
training phases (the runner fails `DONE_EXIT_NOSPREAD` / `_DECAY` otherwise). Pre-launch smoke
`scratchpad/parity3/smoke_ilv_spread.py` passed 4 checks including two oracles: spread OFF
bit-identical to a literal re-implementation of the old front-loaded loop, and depth-1 spread ==
front-loaded == sequential; plus identical no-grad sets and a scripted compile matching eager
bit-for-bit.

**Implementation note.** One shared `interleave_schedule()` now drives BOTH the training path and
the RNN deploy mirror, and wiring it exposed a latent trap: **the v0 dummy keyed on the ROUND**,
which only coincides with the stream-local layer index under front-loading — so any placement
change would have silently mis-threaded the value residual.

## iter 45 — KD through the decay phase: the distillation family goes 4/4 (ACCEPTED, 2026-08-11)

**What ran.** The champion recipe with the KD env **not cleared** before the decay phase: WS keeps
its tuned `alpha=0.9`, decay gets `alpha=0.5`. **Zero code** — the entire lever is two lines the
runner doesn't execute. Winner of the 2026-08-10 15-proposal ranking.

**Why.** Since iter 34 adopted `decay_ratio=1.0`, the decay phase is **half of all training** and
ran on pure hard labels, while the WS alpha dose curve is monotone up to 0.9 (iters 32/35/38/39/40).
So half of training received no teacher signal, in the only family with a perfect record.

**Why it is well-defined** (the belief that the dump can't be used in decay is wrong, and our own
docs contain the reason): decay writes `STEP_OFFSET=1`, `EPOCHS=1.0` over the same db/MAX/seed, and
`random.seed(12345)` + a single `get_groups` shuffle mean decay step *s* replays byte-identically
the batch of WS step *s* ("epochs are BYTE-IDENTICAL replays"). Self-validating: the per-step
`labels_sum` checksum hard-exits 43 on any misalignment, so a stream mismatch aborts at step 1
rather than training on wrong targets.

**Verdict: ACCEPTED — new champion.** vs iter 41:

| | champion (41) | iter 45 | Δ | p |
|---|---|---|---|---|
| ahead | 0.297889 | **0.297697** | **+0.000192** | 3.9e-47 |
| imm | 0.265479 | **0.265375** | **+0.000104** | 1.4e-82 |

size 0/2500, nan_users 0, params 558,212, card/note/deck state 2,880/1,440/5,760 all unchanged (a
*schedule* of teacher signal has no weights), throughput 1833.5 rev/s vs 1849.8 — identical within
noise, as a training-only change must be.

**PERFECTLY CONTROLLED, and verified rather than assumed.** The WS step trace extracted from the
log is **identical to iter 41's for all 10,935 steps** (`extract_trace.py --compare`). So the two
runs shared recipe and seed exactly, the entire gain is attributable to the **decay phase alone**,
and run-to-run reproducibility at seed 4321 is re-confirmed as a free by-product.

**⚠ MARGIN CAVEAT, stated plainly.** imm clears the raw 0.0001 bar by only **4%** (+0.000104) —
about 1.4× the ±7.5e-5 same-capacity noise floor measured in iter 44, and well inside the ~0.0004
cross-seed spread the seed-pair doctrine warns about. ahead is more comfortable at +0.000192.
Accepted because (a) the written gate passes on both modes with overwhelming per-user consistency,
and (b) single-run-at-seed-4321 has been the operative practice since iter 35 established 4321 as
the standard — iters 36–44 were all judged that way, including accepts 39 (+0.000158/+0.000153) and
41. A second-seed confirmation remains the rigorous move before this number is leaned on hard.

**What it means.** Teacher signal is valuable in **both** phases, not just the plateau. The
annealing intuition — that decay should run on the true objective so the model lands on the real
loss surface — is wrong here, or at least wrong at `alpha=0.5`.

**Open in-family points, cheap because they reuse the same dump:** `alpha_decay=0.9` (does decay
want the same dose as WS?) and `alpha_decay=0.25` (is 0.5 already past the peak?). The WS sweep
peaked at 0.9 and bracketed at 1.0, so the decay curve's shape is not implied by it.

**Chain clean:** `DONE_EXIT_0` 23:30:07, eval 2500/2500 on attempt 1, and the runner's own guards
confirmed the lever was live where it mattered — the decay log carried `[kd-mix] KD ON` and
`alpha FIXED at 0.5`, with `DONE_EXIT_NOKD_DECAY` / `_WRONGALPHA_DECAY` as the failure modes.

## iter 46 — privileged self-distillation (imm → ahead): a clean null, and it explains the 0.032 gap (REJECTED, 2026-08-12)

**What ran.** `RWKV_SELFKD_BETA=0.7`: the ahead objective's target is softened away from the raw 0/1
label toward the model's **own** better-informed imm estimate of the *same* review, taken from the
query row that scores it (join `review_th[q] == label_review_th[r]`; **100.00% coverage** verified on
real LMDB rows, 388,156/388,156 across 5 users, 0 violations). Teacher **detached**. It runs *before*
the external KD mix and softens only the HARD share, so the tuned external alpha is preserved exactly
— autograd-recovered 0.899999976 at every beta. Motivation: the measured ahead-vs-imm information gap
(identical `size` on all 2500 VAL users, imm better on 2497, mean gap **0.032411**).

**Verdict: REJECTED as a TIE**, under the curve-side gate pre-registered before the run reported.

| | champion (45) | iter 46 | Δ | p |
|---|---|---|---|---|
| ahead | 0.297697 | 0.297719 | **−0.000023** | 0.996 (improvement) |
| imm | 0.265375 | 0.265359 | **+0.000016** | 0.014 |

Both inside the ±7.5e-5 noise floor. imm was **not** significantly worse (p_worse = 0.986), so the
gate failed purely on ahead — the mode the lever targets. size 0/2500, nan_users 0, params 558,212.

**WHY IT DID NOTHING — the teacher shares the trunk and the forward pass.** The imm head is not an
independent model; it is a *different head on the same representation, computed in the same pass*.
Classic distillation transfers "dark knowledge" because the teacher is a genuinely different
function — the d=128 teacher behind iters 32/35/39 is a separate, better-trained model, and that
sub-family is 4/4. Here the soft target is a smoothed re-expression of what the student already
computes, so it carries nothing the hard label plus the student's own representation did not already
provide. It reweights; it does not inform.

**★ CONSEQUENCE, which is worth more than the iteration cost: the 0.032 gap is NOT transferable
knowledge.** It is a gap in what the two heads can **see** — the query row has the intervening
reviews and the exact lag, and the ahead path structurally cannot — plus a difference in head form.
Closing it therefore requires changing what the ahead path **computes or is fed**, not what it is
**fit to**. This is strictly stronger than the original "upper bound, not a target" caveat: soft
targets cannot reach any of it.

**Before the self-distillation sub-family is deprioritized** (conduct rule 2 asks for a second
implementation of a near-miss idea): the variant with actual literature support is a teacher that is
*not the same forward pass* — a past checkpoint (mean-teacher / temporal ensembling) or a different
augmentation view. Those are genuinely independent functions. Not queued: Andrew has directed
QAT-tax work next.

**Ops.** The WS phase ran under `run_iter46.cmd`, whose chain died `DONE_EXIT_WSFAIL_3` because I
edited the running runner and "reverted" with `git checkout` — restoring LF as CRLF and shifting
every byte offset cmd.exe resumes from. The WS **checkpoint survived intact**; phase 2
(`run_iter46b.cmd`) completed decay+eval normally (`DONE_EXIT_0`, eval 2500/2500 on attempt 1, decay
guards green for `selfkd 0.7` and base KD `alpha 0.5`). The WS log was truncated, so this iteration
has no step trace — harmless for a reject.

## iter 47 — the rank-1 regulariser: the floor moves 43%/75% and the loss does not (REJECTED, 2026-08-15)

**The lever.** `RWKV_QAT_RANK1_REG=0.05` adds `lambda * (1 - max(conc(k), conc(v)))` to the loss,
where `conc` is the participation ratio `||M^T M||_F^2 / ||M||_F^4` (1 for rank-1, 1/r for r equal
directions), computed per head over the chunk's k and v. Single-variable vs `qtaxd_cblearn`: same
WS-final, same seed, same batch order, both quant-aware with learnable catalogs. Training-only — no
architecture, parameter, state-size or deploy change, and nothing to port.

**Why a proxy on the kernel INPUTS rather than the state.** The state S lives inside a custom CUDA
autograd Function whose backward accepts only the output gradient, so a loss on the returned
checkpoints reaches no weights; the differentiable Python path materialises S but loops per timestep
(T up to 65,536). Since S is a decay-weighted sum of outer products `k_i v_i^T`, it is near rank-1
exactly when the per-head k (or v) vectors align — computable from the inputs. Verified on synthetic
cases (all-parallel k → 0.000000; random k,v → 0.9209 vs the theoretical 1−1/K = 0.9375), gradients
flow, skip rows excluded, inert when the env is unset.

**Andrew's objection, which framed the whole iteration:** *"if making states more rank-1 could lower
log loss, the model would learn to do that anyway."* The counter was narrow and was the only reason
to run it: the STE passes dL/dS backward as if the truncation were the identity, so the gradient
contains no term for reducing truncation error — the model is not declining to be rank-1, it is
blind to the benefit.

### The verdict
| mode | control (`qtaxd_cblearn`) | iter 47 | delta | gate |
|---|---|---|---|---|
| ahead | 0.299983 | 0.300018 | **−0.000035** | FAIL |
| imm | 0.268861 | 0.269041 | **−0.000180** | FAIL |

n=2500, size 0/2500, nan_users 0. Pre-registered as the **both-modes** rule (the curve-side exception
does not apply: this lever changes the WKV state, i.e. the shared trunk). ahead is inside the ±7.5e-5
noise floor — a tie; imm is outside it — a small real regression.

### What makes the iteration worth its 11 hours
**The floor moved enormously and the loss did not.** Matched finals, same tool/users/entities:

| stream | control | iter 47 | change | iter 47 median |
|---|---|---|---|---|
| card | 0.3594 | **0.2043** | −43.2% | 0.1115 |
| note | 0.2689 | **0.0660** | −75.4% | **0.0152** |

The note stream ends essentially exactly rank-1. **So the reconstruction ladder's ranking of rank-1
truncation as the LARGEST term (53% card / 39% note) does not survive as a logloss ranking** — the
remaining QAT tax lives in the codebook and norm terms. This is the fourth reconstruction-vs-logloss
misprediction of the week and the first about the term the ladder was built to prioritise.

**Andrew was right, and the CONTROL arm is the strongest evidence.** With no regulariser at all the
control drifts 0.3831 → 0.3594 card and 0.3556 → **0.2689** note (−24.4%) across the same decay: the
model moves toward rank-1 unaided, as far in note as the regulariser managed in its first 50 steps.
Because what appeared was a small *regression* rather than an exact tie, the rank-1 constraint does
cost the model slightly more than the reduced truncation damage repays.

**Do not retry at another lambda.** The step-50 check proved the lever engages hard (−13% card /
−22% note after fifty steps), so dose is not the issue. Family closed.

### Three method failures, all the same shape
1. **Wrong metric.** The floor check was going to compare against the ladder's "card 0.4353 / note
   0.3049", which is not reproducible (ad-hoc script, never saved) and disagrees in *opposite*
   directions per stream. The saved tool (`rank1_floor.py`, verified against explicit truncation to
   2.4e-07) reads 0.3733 / 0.3729 on the same corpus. Comparing to the old value would have shown a
   ~0.06 fake improvement printed next to a null logloss.
2. **Wrong checkpoint.** The first run compared step-50 against iter45's *final*, which also differs
   by 10,885 training steps. It looked emphatic and meant nothing — and spawned a confident
   side-claim about training raising state rank that the matched control reversed.
3. **Wrong signal.** I read the penalty climbing (0.025 → 0.077) as the model surrendering rank-1
   structure and predicted the final floor would land near the control. It was the opposite: proxy
   and target **decoupled**, via exactly the blunt edge documented when the penalty was built (it
   ignores the decay weighting — a decayed sum of outer products can be near-rank-1 without aligned
   k). **A proxy that can diverge from its target is an engagement detector, not a progress signal.**

**Ops.** `run_r1reg.cmd`, decay 14:15→23:32 (10,935 steps, 0.328 steps/s), probe SANE
(+0.002193 / +0.003540 on n=10), eval 23:36→10:15 (2500/2500, first attempt). The ~10 h eval is
NORMAL for this quant-aware config — the control's was 10h18m; the "~2.9 h" figure in the queue is a
plain eval. Two dead launches while staging the CPU floor check: cmd.exe parses redirection *before*
`REM`, so `->` and `<placeholder>` in comments became redirects (leaving two 0-byte files named
`Andrew's` and `proxy` in the repo root); and a weights export needs the structural flags
(`RWKV_STRIP_CMIX` / `RWKV_STRIP_L0_VLORA` / `RWKV_GRU_HEAD`) because the checkpoint stores 1×1
dummies for stripped components.

## iter 48 — retrievability-coupled rating head (REJECTED, exact tie)

`RWKV_RCOUPLE=1`, single-variable vs iter 45. The curve head's logit R(t), clamped to ±8, is added
to the 4 rating logits through 4 zero-init coefficients, placed **before** the rating softmax so
`out_p_binary` is genuinely coupled (§9 three-way parity: the coupling is a model property, applied
on every row, not a loss-side transform — the mistake PAVA made for 8 iterations). Undetached, so
the better-conditioned imm objective also gains a gradient path into the curve head.

**Result (n=2500, VAL half):** ahead 0.297688 vs 0.297697 (**+0.000009**, p=0.19), imm 0.265362 vs
0.265375 (**+0.000013**, p=0.37). size 0/2500, nan_users 0. Both ~7× inside the ±7.5e-5 noise floor
— a clean tie, rejected on the both-modes gate.

**THE DIAGNOSTIC IS WORTH MORE THAN THE VERDICT.** `rcouple_w` was zero-init, so reading it back
separates "the model declined to use R(t)" from "it used R(t) and gained nothing":

| | Again | Hard | Good | Easy |
|---|---|---|---|---|
| learned `rcouple_w` | **−0.01380** | +0.00739 | −0.00442 | +0.00431 |

It learned a **sign-correct** coupling — higher retrievability lowers the Again logit, which is the
physically right direction — so the lever engaged. But the magnitude is negligible: the maximum
possible shift is 8·|w| = **0.110** against a `p_linear` bias spread of **0.772**, and that maximum
only occurs at the clamp boundary. **=> The rating head is not MISSING retrievability information;
the trunk representation already carries it.** Supplying it explicitly is redundant.

**★★ TAKEN WITH ITER 46, THIS CLOSES THE AHEAD-VS-IMM-GAP FAMILY.** Two structurally different ways
of moving information between the heads both return exact nulls:
* iter 46 — **soft targets** (privileged self-distillation imm→ahead): −0.000023 / +0.000016.
* iter 48 — **an architectural path** (R(t) into the rating logits): +0.000009 / +0.000013.

So the 0.032411 ahead-vs-imm gap is **not an information-routing deficiency**. This is exactly what
`PROPOSALS.md` warned in advance — *"the gap is an UPPER BOUND, not a target"*, because the query row
sees the intervening reviews and the exact lag while the ahead row structurally cannot, and
predicting cold from history **is** the task. That caveat is now demonstrated twice rather than
merely argued. **Do not propose a third routing variant.**
⚠ What is NOT closed: changing what the ahead path is FED (new input features) or what it can
represent. Those attack the information content, not its routing.

**OPS.** The first eval attempt died on a TorchScript **runtime** bug unrelated to this lever:
`take_rank1_penalty()` (iter 47's regulariser) carried `@torch.jit.ignore` with no return
annotation, so TorchScript typed it `-> Tensor`; returning `None` handed scripted code an undefined
tensor and the next `float * Tensor` aborted. Training was intact and only the eval re-ran (~10 min
of GPU lost, not 6.5 h). Root lesson: **compiling is not running** — the compile half of this same
bug was fixed a day earlier and reported as complete. Guard added:
`scratchpad/parity3/smoke_scripted_eval.sh`, a ~90 s one-user scripted eval that refuses to run
under `RWKV_NO_JIT=1`. A plain eval is the ONLY path that scripts the model, which is why this
class of bug is invisible to training and to QAT evals alike.

## iter 50 — the deck tree: `(deck, depth_level)` as a shared-weight loop (PRE-REGISTERED, launched 2026-08-16)

> **★ RE-SCOPED TO L=2 BEFORE THE REAL RUN (2026-08-16 07:0x).** Everything below was written for
> L=3 and is kept as written, because the reasoning for L=3 is still the reasoning for the *next*
> arm. What changed is measured, not argued — see "WHY L=2" and "THE SINGLETON MISTAKE" at the end
> of this section. Live config: **`RWKV_DECK_TREE=2`, 17 layer-steps, 558,292 params** (the level
> embedding is 1x80 at L=2), chain `card_id, note_id, deck_id, deck_id@1, preset_id, user_id`.

**The lever, in Andrew's words:** *"fully take into account deck tree structure, so that
card->note->deck->preset->global becomes card->note->(deck, depth_level)->preset->global."*
`RWKV_DECK_TREE=3` inserts two ancestor streams after `deck_id`, so the chain is
`card_id, note_id, deck_id, deck_id@1, deck_id@2, preset_id, user_id`. Stream `deck_id@k` groups
reviews by the deck's k-th ANCESTOR and is run by **the same deck module object** — depth is a loop
count over the user's tree, not an architecture constant. Only new weights: an `(L-1, d_model)`
level embedding added at each level's layer 0, **160 floats**, so params go 558,212 → **558,372**.

**Why L=3 rather than a single parent link.** The deck-depth histogram **peaks at 4, not 1**
(`scratchpad/deck_tree/level_reach.py`, 40 users, 3.67 M reviews, review-weighted):

| ancestor at distance | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| reviews reached | 49.21% | 38.29% | 31.20% | 20.93% | 7.80% | 1.62% |

Anki users nest decks deeply (`Japanese::Core2k::Stage 1::Vocab`). L=2 would test "parent deck";
L=3 is the smallest L that tests *tree*. Levels 4+ are held back because each costs a full extra
4-layer pass for a shrinking share of the corpus.

**No LMDB rebuild** — `data_processing` drops `parent_id` (:228) but **never factorizes or remaps
`deck_id`** (the only rewrite is NaN → `ID_PLACEHOLDER`, :243-245), so the parquet's own
`deck_id → parent_id` map applies directly to ids already stored. This settles a contradiction
`FUTURE_FEATURES.md` had carried for weeks, and settles it structurally rather than by spot check.

**The bypass, and why it is compute-and-discard rather than omit.** A row with no k-th ancestor
gets a row-unique negative id — a **singleton** sequence, the cheapest thing the WKV kernel can be
handed — and is marked inactive; the model gathers it but never scatters it back, so `x` keeps its
incoming value **exactly**. It cannot simply be dropped from the stream: `prepare()` chains each
stream's gather against the *previous stream's layout*, so a row absent from one stream has no slot
for the next stream to reference. Filtering the scatter half while leaving `parts` unfiltered is the
whole trick.

### THREE-WAY PARITY (§9 requires this written down BEFORE the run)
- **Training optimizes** the same `curve_loss + p_loss` as the champion. The tree changes only how
  `x` is transformed between `features2card` and the heads; no loss term, label, or probe is touched.
- **Eval scores** the same quantity, with `RWKV_DECK_TREE` **left set** — it is a model property,
  not a training trick, and the checkpoint carries `tree_level_emb`. Clearing it at eval would score
  a different function *and* fail to load.
- **CPU inference computes** the same thing: `srs_model_rnn.py` mirrors the loop (shared module,
  same level embedding, inactive levels skipped so neither `x` nor that level's state advances).
  Ancestor next-states are handed back through the caller's list, because `review()`'s 5-tuple of
  named states is the deploy API's shape.
- **State-size budget:** card and note per-entity state is **unchanged**; this adds per-DECK
  ancestor states, and the gate explicitly allows deck/preset/global to grow. Rust port only if
  accepted.

### What was verified before launching
- An **all-inactive parent map reproduces the tree-off forward BIT-FOR-BIT** in both Python paths
  (`smoke_tree.py`, `smoke_deck_tree_rnn.py`), while a real map moves it. This exercises the entire
  new path — chain expansion, splits over 7 streams, derived ModuleData, mask threading, the scatter
  filter — so a bypass that is merely *small* fails it; only an exact one passes.
- The grouping extracted from `insert_probes` into `deck_tree.build_module_data` is **byte-identical
  to the code it replaced** on 10 adversarial id arrays. That path runs on every batch regardless of
  the flag, so it reaches an in-flight run's eval phase; this is the check that made editing
  `rwkv/*.py` mid-chain safe.
- `tree_level_emb` **does not exist** with the lever off (no older checkpoint gains a missing key);
  the reference lives in a `@torch.jit.ignore` body with an explicit return annotation.
- The **scripted `forward_batch` RUNS**, not merely compiles — iter 48's lesson.
- The phase-0 param assert was validated in both directions: 558,372 with the tree, 558,212 without.

### The cost is 1.60x, MEASURED not projected (`scratchpad/deck_tree/shape_cost.py`)
Padded kernel volume (`B*T` x layers, 4 users, 58,644 real rows) goes **1,344,210 -> 2,151,182 =
1.60x**, matching the naive layer-step ratio 21/13 exactly. Two things that could have broken that
model were checked and did NOT:
- **Padding waste FELL.** Ancestor streams waste 45.5% / 37.8% against the deck stream's own 50.1%
  -- the ~50% singleton rows pack into a length-1 bucket with zero padding.
- **Parallelism ROSE.** `deck_id@1` has **13,533 sequences vs deck's 203** (maxT 8,296 vs 12,022),
  because singletons dominate the sequence count. The worry was that ancestor decks pool many child
  decks into fewer, longer sequences and cost wall-clock at equal work; the opposite happens.
So ~1.6x on both time and activation memory is the honest estimate, and 12 GB should hold it.

### ⚠ The confound, stated in advance
13 layer-steps become **21**, i.e. ~1.6x the model compute of the champion. **A win is therefore
"the tree helps" OR "more deck compute helps"**, and disambiguating needs the L=2 arm (17
layer-steps) plus a depth-matched control (e.g. ancestor levels at depth 1, which is compute-neutral
at 15 steps). A tie needs no disambiguation — the usual asymmetry, and the reason this is worth
running before the controls rather than after.

### Two bugs this hit, both worth carrying
1. **A shared config object let the last level overwrite `stream_name`.** `expand_modules` first
   reused the deck `RWKV7Config` for every level; `srs_model` then stamps `cfg.stream_name = name`
   per entry, so the one object ended up carrying `deck_id@2`, and `RWKV_STRIP_CMIX=deck_id:1,deck_id:2`
   **silently stopped matching** — +26,070 params with no error anywhere. Fixed by deep-copying the
   config per level and stamping the **base** name (the QAT scopes now match on the base name too).
   Configs are mutated in place by several consumers; one object per stream entry is the only safe
   shape.
2. **The first smoke reported `real == off`.** A freshly built RWKV7 has **zero-init output
   projections**, so every layer is exactly the identity and no change to sequence grouping can
   show. This is the vacuity trap CLAUDE.md §9 already warns about for `parity_train_vs_rnn.py`, met
   in a new harness. Randomizing fixed it — but seeding by raw parameter name then made `null`
   diverge, because `nn.ModuleList` **dedupes the repeated deck object and renumbers the streams
   after it** (preset/user move from `.3/.4` to `.5/.6`), so those two streams drew different
   weights. Seeding on the canonicalised stream name fixed it properly. **Both failures produced a
   confident, plausible, wrong verdict** — one a false null, one a false leak.

## iter 49 — restoring the user/preset layer-0 channel mixers: +4.7% params, ~nothing (REJECTED, 2026-08-16)

**The lever, zero code:** drop `user_id:0` and `preset_id:0` from `RWKV_STRIP_CMIX`, i.e. give the two
most general streams back the channel mixer at their **entry** layer. `PROPOSALS.md` item #6.
**584,282 params, +26,070 (+4.7%)** vs the iter-45 champion; single-variable otherwise.

**Result (n=2500, VAL half, rectified):** ahead 0.297630 (**+0.000067**, p=0.113) · imm 0.265288
(**+0.000087**, p=5.3e-16) · size 0/2500 · nan_users 0 · `DONE_EXIT_0`.

**REJECTED, for two independent reasons.** Both modes improve, but **both miss the raw ≥0.0001
bar**, and ahead additionally fails the p-gate at p=0.113 — inside the ±7.5e-5 noise floor, i.e. a
coin flip. imm's +0.000087 is a *real* effect by rank (p=5.3e-16: most users improve slightly) yet
still below the accept threshold. The both-modes rule applies rather than the curve-side exception,
because a trunk capacity change can move either mode.

**★ THE FINDING: capacity at the general streams' ENTRY layer is not the bottleneck.** A 4.7%
parameter increase, placed exactly where the stripped-cmix ablations had removed the most, buys an
effect indistinguishable from noise on ahead and one-sixth of the accept bar on imm. **capacity-at-5k
goes 0/3**, and the pattern is now consistent across three different placements: the model is not
capacity-limited at this data scale, it is limited by something else. This is the same conclusion the
100-user era reached from the opposite direction ("the d=32 model is DATA-limited, not
capacity-limited; training levers are the wins"), now confirmed at 5k on a 4.95x smaller trunk.

**⚠ Worth noting for calibration of the accept bar:** both deltas clear the *pre*-2026-08-10
magnitude bar (raw ≥0.00005 via 4-dp rounding), so under the old rule this would have come down to
the p-gate alone. The tightened bar and the p-gate agreed here, which is the reassuring case — but it
is a reminder that the old magnitude bar sat below the measured noise floor, exactly as the
tightening note said.

**Ops:** clean run, `DONE_EXIT_0` at 06:32 after WS 21:03→00:27 and decay 00:27→03:42 (both ~3h15m at
13 layer-steps), eval 03:42→06:32 (2h50m). No orphan fetch workers left behind.

### ★ WHY L=2, AND THE SINGLETON MISTAKE (both measured on the real run, 2026-08-16)

**THE SINGLETON MISTAKE — I optimised the wrong axis, and two pre-flight checks both missed it.**
Rows with no k-th ancestor were first given a **row-unique** id so they formed singleton sequences,
justified as "T=1 is the cheapest thing the WKV kernel can be handed". That is true about the
SEQUENTIAL axis and irrelevant: **the WKV state is per SEQUENCE** (H*K*K floats each), so ~24,600
singletons carry ~100x the state of the deck stream's 203 sequences. Measured live: **11.8 of 12.3
GB VRAM, 0.16 steps/s, and GPU utilisation at 1%** — the card was allocating and writing state, not
computing.
**Both metrics I had checked in advance looked fine**: padded volume was 1.60x (`shape_cost.py`) and
kernel launches went 42 -> 74. **Neither counts B.** Worse, I had written up the high sequence count
as *reassuring* ("parallelism ROSE — 13,533 sequences vs deck's 203"), which is precisely the
observation that should have raised the alarm.
**FIX:** inactive rows group by their **negated leaf deck id** — bounded by decks-per-user instead of
rows. `deck_id@1` 13,533 -> **64** sequences, `deck_id@2` 24,646 -> **57**, total WKV state 172 ->
**174 MB (+1.4%)**, and `prepare()` 1.37 -> 1.25 s. The exact-bypass property is unchanged and was
re-verified bit-for-bit.

**WHY L=2 RATHER THAN L=3.** Even after the fix, L=3 pins VRAM at **11.6-11.8 of 12.3 GB (95%)** and
runs at **0.238 steps/s against the champion's 0.893 — 3.75x slower** where the shape analysis
predicts 1.31x. At 95% occupancy this machine starts WDDM paging (a documented ~4x cliff), so L=3
would cost ~30 h *and* produce a throughput number confounded by paging. L=2 measures **0.360
steps/s (2.48x), VRAM 10.8 GB (88%), util 76%** => ~8.4 h WS + 8.4 h decay + ~3.7 h eval ~= **20.5 h**.

**⚠ THE RESIDUAL 2.48x-vs-1.31x GAP IS REAL COMPUTE, NOT PAGING, and it is a fact about this kernel
worth carrying:** ancestor streams are **low-B / high-T** (B ~= 64 sequences against the card
stream's 11,900). WKV parallelism is B*H, so B=64 gives ~320 parallel units on a 46-SM card —
poor occupancy. The deck stream already has this shape (B=203); the tree adds four more layers *of
the same bad shape*, so wall-clock grows faster than padded volume says. **Any future "add a coarse
stream" proposal should be costed on B, not on rows.**

**What L=2 still tests:** a parent level on **49.21% of reviews** — a real test of "does the deck
hierarchy carry information". If it shows signal, L=3 justifies the memory work; if it is a tie,
deeper levels were never going to rescue it. The confound shrinks correspondingly: 13 -> 17
layer-steps (1.31x), disambiguated by a depth-1 ancestor control if it wins.

### ⚠ RETRACTION, same day: the "2.48x / low-B-high-T" analysis above is WRONG

Everything in the previous subsection about **wall-clock** was measured inside `torch.compile`
warmup and must not be quoted. Timeline of my own numbers for the SAME L=2 configuration:

| window | steps | measured | what it actually was |
|---|---|---|---|
| first | 11 → 35 | 0.238 steps/s | compile warmup |
| second | ~34 → 50 | 0.360 steps/s | still warmup |
| **steady** | **226 → 299** | **0.664 steps/s** | **the real rate** |

**0.664 vs the champion's 0.893 is 1.35x — the shape analysis predicted 1.31x and was right.** There
was no anomaly, so the mechanism I invented for it ("ancestor streams are low-B/high-T, ~320
parallel units on a 46-SM card, the worst shape for this kernel") explains a gap that does not
exist. It is retracted. It may still be true as a statement about the kernel; nothing here is
evidence for it.

**THE RULE THIS EARNS:** *never time a run against a steady-state baseline until compile warmup is
provably over.* On this stack that is several **hundred** steps, not tens — `RWKV_QAT_COMPILE=1`
plus dynamo recompiles on new stream shapes. Projected cost accordingly falls from ~20.5 h to
**~4.6 h WS + ~4.6 h decay + ~3.5 h eval ≈ 13 h**.

**What survives, and why the re-scope to L=2 still stands.** The singleton fix is justified on
*memory*, which was never a timing measurement: 13,533 + 24,646 singleton sequences × 1280 floats ×
4 layers ≈ **780 MB of extra WKV state in fp32** before backward saves, against 64 / 57 sequences
and +1.4% total after. And L=3's **11.6–11.8 of 12.3 GB (95%)** is likewise a direct reading, on a
machine with a documented WDDM paging cliff at that occupancy and a GPU co-tenant (Andrew's FSRS
benchmark). A 10 h run there is fragile for reasons that have nothing to do with steps/s. **So the
decision is unchanged and the stated reason for it is now only memory** — the speed half of the
argument was an artifact.

**This is the third measurement error of the iter-47 family in a week** (wrong metric, wrong
checkpoint, wrong signal — now wrong *window*), and the pattern is identical each time: a confident,
plausible, mechanism-shaped story built on a comparison where one side was not held fixed.

### RESULT — REJECTED as an exact tie (2026-08-16 19:20, `DONE_EXIT_0`)

| | champion (iter 45) | iter 50 | delta | p |
|---|---|---|---|---|
| ahead | 0.297697 | 0.297690 | **+0.000007** | 0.521 |
| imm | 0.265375 | 0.265399 | **−0.000024** | 0.860 |

n=2500, size 0/2500, nan_users 0, params 558,292. Both deltas are **3–10× inside the ±7.5e-5 noise
floor**, and p=0.52 on ahead is a literal coin flip. Both-modes rule (the tree feeds the shared
trunk, so the curve-side exception does not apply).

**★ THE DIAGNOSTIC IS THE RESULT, and it is what makes this null separable.** The level embedding is
zero-init and free to stay there. It did not: it trained to **L2 = 1.766, max|·| = 0.591**, against a
`features2card.0.weight` row-L2 median of **0.829** — roughly **2× the scale of a typical
input-projection row**. So the model marked the parent-deck level distinctly, **used** it, and gained
nothing. This is the same shape as iter 48's zero-init coupling (learned, sign-correct, negligible),
and it rules out the "the lever never engaged" reading that would otherwise make a null uninformative.

**★★ MECHANISM: the 5-stream hierarchy already BRACKETS the parent-deck scope.** `deck_id` pools
per-deck immediately below it; `preset_id` and `user_id` pool more broadly immediately above it. A
parent-deck level is therefore **interpolation between two scopes the model already has**, not a new
kind of evidence. That is why 49.21% of reviews gaining a genuine extra pooling level moves nothing.

**⚠ THIS DEMOTES THE L=3 FOLLOW-UP RATHER THAN MOTIVATING IT.** The pre-registration argued L=3 is
the better test because the depth histogram peaks at 4. That is still true about *reach*, but the
mechanism cuts the other way: **deeper ancestors interpolate even closer to preset/user**, so if the
parent level — the one with the most distinct scope and the widest reach — is a coin flip, level 3
and 4 are a worse bet, not a better one. L=3 should not be run on "we only tested the shallow case";
it needs a new argument, and the memory work it requires (95% VRAM) makes it expensive.

**What this adds to the topology family.** Iters 41–44 established that *the existence* of a
cross-scope information path pays (interleaving, +0.000216..+0.000611) while its choreography does
not (three arrangements indistinguishable at |Δ| ≤ 7.5e-5). Iter 50 closes the remaining direction:
**adding more scopes does not pay either.** The scope ladder card→note→deck→preset→user is
sufficient; the productive lever was letting information move *between* the levels that exist, and
that has already been banked.

**Cost:** WS 4h33m (6:51→11:24), decay 4h27m (11:24→15:51), eval 3h29m (15:51→19:20) ≈ **12.5 h** at
0.66–0.67 steps/s throughout — 1.35× the champion's 0.893, matching the shape prediction.

## iter 51 — Polar-Express Newton-Schulz schedule for Muon (FAILED: NaN at step 411; lever CLOSED on mechanism, 2026-08-16)

`PROPOSALS.md` #5, chosen after Andrew preferred the Muon variations to duration dropout. The run
died hollow — 410 good steps, then `Exception caught. Nan from RWKV-7? Skipping batch.` on every
batch thereafter (3,684 of them). Killed at ~0.5 h; no eval.

### The pre-work was right and is worth keeping
The proposal's motivation reproduces and decomposes cleanly (`scratchpad/iter51_muon/`):
* RMS|σ−1| over all singular values = **0.274**, matching the recorded 0.19–0.31;
* **not precision** — bf16 0.289 vs fp32 0.301;
* **~half is the near-null tail** (median condition number 1.2e4), unliftable in 5 steps and
  undesirable to lift (noise amplification);
* **0.161 remains on the top-90%-energy directions**, which is the attackable part;
* a greedy-minimax per-step schedule cut that to **0.0251 (−84.4%)**, with the control that merely
  rescaling to σ_max≈1 buys only 5.8%.

### ★ WHY IT FAILED, and it inverts how this file described the production constants
The offender is a **(3, 320)** momentum matrix of effective rank 1. Frobenius normalisation
concentrates a thin matrix's energy into σ_max, so σ_max ≈ 1 — and bf16 rounding puts it at
**1.0012**, just above. From there:

| step | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| fitted schedule | 1.229 | 1.835 | 2.860 | 47.9 | **1.9e8** |
| production triple | **0.696** | 1.118 | 0.726 | 1.082 | 0.701 |

**`a+b+c = 0.7010` means p(1) = 0.70 < 1: the production triple CONTRACTS anything at or above 1.**
That is a stability guarantee, not the "deliberate sloppiness" I called it in the runner header and
the commit message. Accuracy at the top of the spectrum *is* p(1) → 1, which converts a contraction
into a marginally stable map — and thin matrices sit exactly on that fixed point by construction.
**Any revival must keep p(1) strictly below 1, which forfeits accuracy precisely where the energy
is.** The flag now raises at import with this reason inline.

### ★★ THE VALIDATION ERROR IS THE REUSABLE LESSON: I FITTED AND CHECKED ON THE WRONG DISTRIBUTION
The schedule was fitted on **step-10935** momentum and deployed from **step 1**. Early momentum is
differently conditioned (median σ_min **2.8e-6** vs **6.6e-5** late), and on it the schedule produced
an update **1.76e7×** baseline. Two compounding mistakes:
1. **Wrong distribution.** A late-training checkpoint is not a sample of the training run.
2. **Wrong statistic.** I reassured myself with a **median** ‖O‖_F ratio (+2.6%, "far below any LR
   sensitivity"). On early buffers the median was a benign 0.87 while the **max** was 1.76e7. *A
   median cannot see a blow-up; only a max can.* Every future numerical-stability check on this
   project should report the max.

A constrained refit (v2, peak capped at the production triple's own 1.20) was tried and **also
diverged** (max ratio 2.75e8) — which is what promoted the diagnosis from "bad fit" to "structural".

### Cost and disposition
~0.5 h GPU, no eval, no champion impact. **Optimizer family stays 1/2** — this never produced a
number, so it is a failed launch rather than a rejected iteration. The lever is closed on mechanism;
`NorMuon` (per-neuron second-moment normalisation) is a *different* mechanism and remains open.

## iter 56 (QAT#2, `qtaxg_i45kd`) — swap the KD teacher to our own plain champion (REJECTED as an exact tie, 2026-08-17 21:34)

**Numbering:** 52–55 were pre-assigned to queued algorithmic iterations while this was still
running, so this finished *before* them and is numbered *after*. It is a **QAT-tax arm**, not an
algorithmic iteration: its comparison basis is the quant-aware twin `qtaxd_cblearn`
(0.299983 / 0.268861), **never iter 45 plain**.

### The lever, and it is exactly one line
`run_i45kd.cmd` is `run_cblearn.cmd` with the KD dump changed and nothing else:

    run_cblearn.cmd:29   set DUMP=C:\rwkv_kd_dump\t128_seedpair_65k   (the d=128 teacher)
    run_i45kd.cmd:29     set DUMP=C:\rwkv_kd_dump\ours_i45_full       (our own PLAIN iter-45 champion)

Same WS-final, same `alpha=0.5`, same learnable catalogs, same refit starting codebooks, same
LR/schedule. **Hypothesis:** a teacher that was never quantized should pull the quantized student
back toward plain-model quality, i.e. pay down the QAT tax.

### Result — REJECTED
| mode | baseline (cblearn) | QAT#2 | delta | p | verdict |
|---|---|---|---|---|---|
| ahead | 0.299983 | 0.299899 | **+0.000084** | 6.239e-04 | misses the raw 0.0001 bar AND p<0.0001 |
| imm | 0.268861 | 0.268931 | **−0.000070** | 1.0 | **worse by rank as well as by mean** |

`size` 128,800,080 IDENTICAL on all 2500; `nan_users` 0; params 558,212 (a teacher swap has no
weights). Both |deltas| sit **inside** the ±7.5e-5 same-capacity noise floor (1.1× and 0.9×), so the
honest reading is *no effect* — not "a small win on ahead". Both-modes rule applies: a KD-teacher
change reshapes the shared trunk, and the curve-side exception is reserved for levers that
structurally cannot move imm.

### ★★ THE FINDING: a minutes-of-CPU screen predicted this ~13 h GPU result, that same morning
`scratchpad/ensemble_screen/teacher_agreement.py` measured the two teachers agreeing at
**r = 0.9460** — because **iter 45 IS the d=128 teacher's own student**, the end of a 4-iteration
lineage (32 → 35 → 39 → 45) each trained against that same dump. A teacher that agrees 0.946 with
the incumbent carries almost no independent signal, so neither **adding** it (the ensemble proposal,
projected +0.000033) nor **swapping** to it (this run, +0.000084 / −0.000070) can do much. Same
order of magnitude, same verdict: nothing.

That screen was run to rank a *different* proposal and incidentally priced this one. **The
generalizable move is to run the screen BEFORE the build, not alongside it** — this is now four
screens that have redirected six candidates, and the first where a full GPU run was spent
confirming a conclusion the screen had already reached.

### Consequence for the QAT tax
**The tax does not live in the teacher.** It stays where the 2026-08-13 measurements put it — the
**codebook and norm terms** — with iter 47 having separately shown the rank-1 truncation term is
nearly empty of logloss. The remaining lever is codebook capacity/fit, which is **unmotivated
rather than refuted**: reconstruction error cannot rank catalogs by logloss (the learned catalog
reconstructs *worse* than the frozen one it improved on), so it needs a logloss A/B justified on its
own terms before spending ~11 h.

**Do not propose a third teacher variant.** Every teacher in this lineage is correlated with the
d=128 dump by construction, and `RWKV_trained_on_5000_10000.pth` is disqualified — it trained on our
entire VAL+TEST halves.

### Ops
The eval survived the 12:53 hard freeze *and* a later accidental power-off. `get_result` skips users
already banked, so each resume cost only the in-flight user; the run completed `DONE_EXIT_0` at
21:34 with all 2500 present. ⚠ `run_i45kd.cmd`'s REM header still describes the PREVIOUS arm
("QAT IMPROVEMENT #1: LEARNABLE CODEBOOKS … single-variable vs qtaxc_m2b12"). The lever is
unambiguous from the `DUMP` line and the LIVE table, but this is precisely the "runner executes
correctly while describing the wrong experiment" hazard `mk54.py` was given asserts against.
