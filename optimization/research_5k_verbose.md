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
