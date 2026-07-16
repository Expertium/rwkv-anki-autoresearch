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
