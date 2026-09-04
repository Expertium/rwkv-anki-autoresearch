# Reject-log steelman proposals (2026-09-04)

Prior of this agent: read every rejected / tied / near-miss iteration and ask what the same idea looks like
once the reason it failed is removed. Reference for every band below = `realcyc` on gen-5 (n=2,499):
**ahead 0.298083 / imm 0.263592**. Stop criterion ahead <= 0.2950 AND imm <= 0.2640 -- so imm already
CLEARS its target by 0.0004 and ahead is 0.0031 short. Every proposal is judged on ahead first.

Two facts from the record that shape the ranking:

* **The largest ahead-specific cost in the record is the deploy rectification penalty: +0.001544 ahead
  on the lambda=0.2 champion** (iter 36's decomposition), ~70% of it the zeroed current-row duration and
  ~30% PAVA pooling (iter 31's split). That is 15x the accept bar and it is paid by ahead only.
* **The cross-family pattern (iters 48/50/57/59/62): a lever that trained and gained nothing removed a
  constraint that was not binding.** So each proposal below names a constraint that is *demonstrably*
  binding by a measurement already in the record, not a new freedom whose need is assumed.

Considered and NOT proposed, with the reason: (a) the graft of iters 49+56 -- the graft policy's trigger
(>=2 rank-significant sub-bar positives in the same mode) is met on imm, but both ahead halves are
noise-floor coin flips, so the expected ahead gain is ~+0.0001 at perfect additivity; (b) an extra
interleave round via layer reuse -- iter 44's finding ("once every scope can see every other, the
rendezvous points carry ~zero") covers a second full round, since that is exactly "seeing every other
again"; (c) restacking iter 56's linear decay -- closed by arithmetic (+0.000057 ahead under perfect
additivity); (d) the calendar-aware curve head -- killed 2026-09-04 by the LOO sweep (the clock gain is
`t_since_any_review`, not a function of t); (e) the d=128 teacher retrain -- excluded by directive.

**One CPU pass feeds three of the five screens.** Run `rwkv.run_as_rnn` on the realcyc checkpoint over
~10 TRAIN-range users + ~10 VAL users on the gen-5 parquet (`RWKV_ID_FEATURES=1 RWKV_REAL_CYCLES=1`,
`RWKV_CHAMP_CKPT=scratchpad/realcyc/rc_d_10935.pth`), recording per scored row: the curve logit at the
label's t, the same logit with the CURRENT row's `scaled_duration` zeroed (a second head evaluation on
the same state -- the probe is a skip row, so the state is untouched), y, t, user, review index,
previous press. `scratchpad/spacing_screen/calibration_by.py` already does the walk and the record;
it needs the env block swapped and the second head call added. ~1-2 h of CPU, no GPU.

---

## 1. Born-again KD from the realcyc checkpoint (a frozen, separate-forward-pass, same-layout teacher)

**Family:** distillation (external-teacher sub-family 4/4; self-distillation 0/1).

**Descends from:** iter 46 (privileged self-distillation, exact tie), iter 54 (QAT#2: our own iter-45
champion swapped in for the d=128 teacher, exact tie vs the d=128 dump), and the lineage's KD-OFF
state. PROPOSALS rank 11 ("Born-again: fresh student, champion as sole teacher") was never run.

**Why the closing evidence does not cover it.** Iter 46's null was diagnosed, in its own write-up, as
"the teacher shares the trunk AND the forward pass" -- a different head on the same representation in
the same pass, so the soft target re-expresses what the student already computes. The same write-up
names the untested variant: a teacher that is NOT the same forward pass. Iter 54 then measured exactly
such a teacher -- iter 45's final checkpoint, same size, same layout, separate frozen pass -- against
the d=128 teacher in a KD decay, and got an EXACT TIE (+0.000084 / -0.000070). Read together with iter
45 (d=128 teacher in decay beats no teacher in decay by +0.000192 / +0.000104, plain): a same-size
separate-pass teacher delivers the decay-phase KD gain in full. CLAUDE.md's line "iter 46 showed a
teacher that is not a bigger/different function distils nothing" over-generalises iter 46 to a case
iter 54 contradicts. The ensemble-teacher demotion (r=0.946, teacher B is teacher A's student) is about
averaging two teachers, not about using one. And KD is OFF on this lineage only because the 92-dim
d=128 teacher cannot forward 109 dims (teacher_114 screen) -- KD never failed here; it was never
available. The realcyc checkpoint is a 109-dim teacher that already exists.

Caveat stated honestly: iter 54's tie compares two teachers, not a teacher against no teacher, and it
was measured in a QAT decay. The inference "same-size teacher > no teacher" chains iter 54 with iter
45; it is not a direct measurement. That is what the band below prices.

**Mechanism.** `rwkv/train_rwkv.py:736-761`: DUMP mode (`RWKV_KD_DUMP_OUT`, `RWKV_KD_TEACHER=
scratchpad/realcyc/rc_d_10935.pth`, `RWKV_KD_STEPS=10935`) walks the deterministic gen-5 batch stream
in no-grad forward and stores `(p_curve, p_imm_all)` per row -- the pattern of
`scratchpad/qat_tax/run_i45_dump_full.cmd`, with the realcyc env instead of the d=128 arch swap.
STUDENT mode (`RWKV_KD_MIX=<dump>:10935`, `RWKV_KD_ALPHA=0.9` in WS, `0.5` in decay) mixes targets in
`srs_model.py::_get_loss` at both consumers of `kd_mix`: `label_y = a*teacher_curve + (1-a)*y`
(curve/ahead) and `p_loss` -> soft-target CE against `a*teacher_p + (1-a)*one_hot` (rating/imm).
Fresh student (BAN), not warm-started. Zero model code; the dump is ~2 h of forward-only GPU. The
decay phase reproduces the epoch-0 batch stream (train_rwkv's own comment), so one dump serves both.

**Why ahead specifically.** Iter 32's full-run KD moved ahead MORE than imm (+0.00058 / +0.00043);
iter 45's decay KD likewise (+0.000192 / +0.000104). The mechanism (target-variance reduction on a
0/1 label whose base rate is ~0.9) is largest where the per-row label is noisiest relative to the
signal, which is the ahead row (one review, one Bernoulli draw, no query context). The ahead label
is also the one PAVA's probe loss consumes (`_pava_probe_loss` targets `label_y`), so the soft target
reaches the rectified quantity directly.

**CPU screen (kills it before the dump).** From the shared pass: realcyc's by-user ahead/imm LogLoss
on the 10 TRAIN-range users against the same recipe's VAL numbers. KD pays only if the teacher's
targets are a de-noised label rather than a memorised one. Kill if the train-minus-val gap exceeds
0.010 in either mode (a memorising teacher hands the student its own training noise back, and the
born-again literature's gains vanish there). Second check, free from the same records: the teacher's
calibration gap on TRAIN users (mean p - mean y); a gap beyond -0.005 means alpha=0.9 will inherit it
and proposal 5's shift should be folded in before the dump.

**Cost:** ~2 h dump + ~10.5 h run (KD runs ~0.92 steps/s vs 0.97 plain). Gate: BOTH modes (KD rewrites
both objectives -- the record's own caveat about `kd_mix`).

**Pre-registered band:** ahead +0.00015..+0.00045, imm +0.00010..+0.00035 (iter 32 scaled down for a
teacher with no size advantage). Null band (both inside +/-7.5e-5) means a same-lineage teacher
carries no dark knowledge at this budget and iter 54's tie was a QAT-decay artefact -- record that as
closing born-again, and the KD-on baseline waits for the d=128 retrain. Abort if either mode is worse
by > 0.0002 (teacher miscalibration inherited).

**Deploy debt:** none (training-only).

**Not additive with the scheduled d=128 teacher** -- when that lands it replaces this teacher; this run
is the lineage's KD-on baseline until then, and it tells the teacher-retrain phase whether same-size
targets already saturate the variance-reduction gain.

---

## 2. Reallocate the objective toward ahead: `IMMEDIATE_SCALE` 1.0 -> 0.5 (equivalently ahead x2), a DIRECTED-ACCEPT candidate

**Family:** objective / loss weighting (0/2: pbin 0.5 and 0.25 -- "a linear imm/ahead trade").

**Descends from:** iters 17 and 19 (`RWKV_PBIN_SCALE`: imm +0.0004 for ahead -0.0002 at 0.5, half of
that at 0.25 -- a linear trade through zero); iter 36 (PAVA lambda, directed-accepted on a 5.9:1
ahead-for-imm trade); iter 37 (by-USER weighting, mechanism-refuted).

**Why the closing evidence does not cover it.** "Loss-reweighting closed by dose-response" was
measured in ONE direction: pbin ADDED an imm-side term and bought imm with ahead. The reverse -- buying
ahead with imm -- has never been run, and it is the direction the stop criterion now wants. Iter 37 is
not this lever: it reweighted USERS, not MODES. The both-modes gate rejects any trade by design, which
is why this must go to Andrew as iter 36 did. What has changed since iter 36 is the arithmetic: imm
already sits 0.0004 under its 0.2640 target and ahead is 0.0031 over 0.2950. Projected through the
endgame (10x budget ~-0.0042 both; QAT +0.0023 ahead / +0.0035 imm): ahead ~0.2962 (fails by 0.0012),
imm ~0.2629 (passes by 0.0011). A <=1:1 imm-for-ahead trade is exactly what closes the criterion.

**Mechanism.** `srs_model.py::_get_loss` (~:1500-1520): `loss_avg = 0.5*ahead_avg + 1.0*immediate_avg
+ 0.5*ahead_raw_avg + ...`. Under `RWKV_NO_AHEAD_RESIDUAL=1` the two ahead terms are IDENTICAL
(`curve_logits == curve_logits_raw`), so the effective weights are ahead 1.0 : imm 1.0, each a per-mode
mean over its own rows. Add `RWKV_IMM_SCALE_MULT` (default 1.0 = byte-identical) multiplying
`IMMEDIATE_SCALE` (and the `kd_p_avg` term if KD is on). Halving imm rather than doubling ahead keeps
the gradient norm scale (and the clip at 0.25, the Muon step) closer to the tuned regime. Note the
PAVA probe term (lambda 0.2) is ahead-side and unchanged.

**Why ahead specifically.** By construction. The trunk is shared and data-limited (capacity 0/3); the
two heads compete for the same representation, and pbin proved the compromise moves linearly with the
weights. The 0.032 ahead-vs-imm gap is intrinsic (family closed), but the SHARE of trunk capacity
spent on each head is a free choice that was never made deliberately -- 1.0:1.0 is the upstream
default.

**CPU screen.** Gradient-conflict measurement at the realcyc checkpoint: on 3-4 small users' batches
(CPU, `reference_rwkv7` path or the CppExtension kernel), backprop `ahead_avg` and `immediate_avg`
SEPARATELY and record, over the trunk parameters, (i) the norm ratio ||g_imm|| / ||g_ahead|| and
(ii) their cosine. Kill if cos > +0.5 (the objectives already pull the same way -- reweighting cannot
trade, and the lever is inert); proceed if cos < +0.2 with ||g_imm|| >= ||g_ahead|| (imm dominates the
shared update and a reweight moves the compromise). Minutes per batch.

**Cost:** one full run ~10.5 h. Gate: fails the standing gate by construction; report the exchange
rate and put it to Andrew as a directed accept (iter 36 precedent). Pre-register the decision rule
BEFORE the run: accept iff ahead >= +0.0002 at p<1e-4 AND the exchange rate ahead:imm >= 1:1 AND imm
stays <= 0.2640.

**Pre-registered band:** ahead +0.00015..+0.00045, imm -0.00015..-0.00050 (pbin's slope mirrored).
Falsifier: ahead inside the floor with imm worse = the trunk's ahead capacity is saturated and the
mode split is not a lever; then the only ahead levers left are input-side.

**Deploy debt:** none.

---

## 3. Duration dropout on input dim 8 -- iter 33's prescribed clean retry, aimed at the 70% half of the rectification penalty

**Family:** deploy-contract alignment (train/eval/deploy compute one quantity), objective-input.

**Descends from:** iter 33 (withhold the current review's duration; REJECTED -0.002787 / -0.000805 --
and the record's own correction says THREE changes shipped and the most likely culprit was a 10x
down-weighting of the ahead objective on 76.5% of rows, not the hypothesis); iter 18 (permanent removal
of the duration feature, -0.0018 / -0.0024 -- the p=1.0 end of the dose curve).

**Why the closing evidence does not cover it.** Iter 33's verbose section states the retry explicitly:
"do NOT use probes to withhold duration at all. The clean instrument is per-row Bernoulli dropout on
`scaled_duration` (dim 8) at the model input: no probe-density change, no row inflation, no MAX change,
no loss-term reweighting -- only the duration varies." It was never run because the phase moved to
HP tuning and then KD. Iter 18 brackets p=1.0 and removed the feature from the STATE too; dropout at
p<1 keeps duration in the state on (1-p) of rows. The constraint this attacks is measured binding: the
current-row duration cost is +0.001451 of iter 31's +0.002062 rect-vs-unrect gap, and the lambda=0.2
champion still pays +0.001544 total. Today the shipped ahead quantity (probe row, duration 0) is
trained only through the PAVA probe loss -- lambda 0.2 x density 0.08 ~ 1.6% of the ahead weight --
while 98.4% of the ahead gradient teaches a curve that reads a feature deploy never supplies.

**Mechanism.** `srs_model.py::forward_batch` (~:1061): `batch_start` (B,T,109) passes through
`_apply_input_feat_mask`. Add, TRAIN ONLY (`self.training`), a Bernoulli mask on column 8
(`CARD_FEATURE_COLUMNS.index("scaled_duration")` -- index 8 is before the drop point at 22, so it is 8
in both layouts) with keep probability 1-p, applied to REAL rows only (query and probe rows already
carry 0.0). Flag `RWKV_DUR_DROP=0.25`, default 0 = byte-identical. Eval/deploy untouched: the probe
zeroes the current row, the state rows keep their durations -- the same quantity as today.

**Why ahead specifically.** imm's query row already has duration 0.0 by convention, so the rating head
learned "0 = not pressed yet" long ago; the CURVE head at a real row never sees a zero except through
the probes. The lever teaches the curve head, at full loss weight and on ~25% of rows, the exact
input distribution the rectified metric scores. The mismatch is ahead-only by construction.

**CPU screen (from the shared pass).** By-user mean ahead LogLoss on the 10 VAL users with the current
row's duration PRESENT vs ZEROED for the prediction only (no PAVA) = the duration half of the penalty
on THIS checkpoint. Kill if < +0.0004 (the ceiling would sit under two accept bars once the imm side
cost is netted). Also record the per-user spread: if the cost lives in a few users, the run's p-gate is
at risk regardless of the mean.

**Cost:** ~10.5 h. Gate: BOTH modes (an input-side change reaches the trunk; the curve-side exception
does not apply). Pre-registered imm harm line: -0.0001 -- the state loses duration on p of rows and
iter 18 says duration is real imm signal.

**Pre-registered band:** rectified ahead +0.00015..+0.00050, imm -0.00010..+0.00005. Falsifier:
rectified ahead inside the floor while the unrectified ahead WORSENS = the curve head cannot serve
both input distributions at this width and dropout only blurs it; then the family is closed at 0/3
(18, 33, this) and the penalty is a deploy fact, not a training target.

**Deploy debt:** none.

---

## 4. A monotone, non-piecewise odds-power residual on the curve logit (the removed residual's constraint-respecting form) -- NEEDS ANDREW'S CONTRACT SIGN-OFF

**Family:** curve-head expressiveness (GRU N-sweep peaked at 3; readout 0/3).

**Descends from:** iter 22 (the piecewise-linear ahead residual removed for monotonicity at a MEASURED
cost of ahead +0.0008 / imm +0.0003 at d=32 -- "the price of monotone-in-t", accepted by directive) and
A4 (the same removal at d=128: +0.000456 ahead). Also iter 27 (N=4 worse than N=3) and iter 28 (xhead).

**Why the closing evidence does not cover it.** The residual was removed because it was NON-MONOTONE,
never because it was useless -- both removals measured a large ahead cost and the record calls it a
price. Iter 27 shows more MIXTURE components do not help, but a mixture in probability space and a
multiplicative term in ODDS space are different families: for a single power curve the odds
`R/(1-R)` are not a power of t, so `logit R - c*log1p(t/tau)` is not representable by re-weighting
components. What it adds is shape freedom in the transition region (near t=0 it is linear in t, in the
tail it merges with d), which is where a 3-component mixture is stiffest. It is monotone decreasing in
t BY CONSTRUCTION (c = softplus(.) >= 0), so PAVA and the monotone-curve requirement hold exactly.
**The contract question is wording:** Andrew's deploy contract says "no piecewise correction". This is
not piecewise and not a correction table; it is a 2-parameter change of the curve family. If he reads
"no correction" broadly, this proposal is dead and should not be built.

**Mechanism.** `head_and_out`: two zero-init linears from `x_w` (the `head_w` trunk output,
`w_head_dim` = 4*80 = 320) give `c_raw, tau_raw` per row (~642 params); `gru_forgetting_curve` returns
`r`; in `_get_loss`, `curve_logits_raw = logit(r) - softplus(c_raw) * log1p(t / exp(tau_raw))`.
Zero-init `c_raw` with a bias of -6 makes the term ~0 at start (byte-identical up to 1e-3 of logit).
Deploy: `rust/rwkv-infer` evaluates the same closed form; the interval solver bisects a monotone
function, so `pava.rs`'s solver needs no change beyond the new curve formula. Fresh parity trace.

**Why ahead specifically.** The term lives entirely on the curve head; the rating head is untouched
except through the shared trunk, so the curve-side exception applies and the pre-registered harm test
is on imm. iter 22's cost was 2.7:1 ahead:imm, which bounds the shape of what is recoverable.

**CPU screen (from the shared pass).** On the realcyc VAL records, fit a GLOBAL (c, tau) by Newton on
by-user-mean BCE (2 params), then a PER-USER (c, tau) with a 2-fold split by row inside each user as an
oracle on how much row-conditioning could add. Kill if the per-user held-out gain < +0.0002 ahead (a
row-wise head is unlikely to beat a free per-user fit by enough to clear the bar). Minutes.

**Cost:** ~10.5 h + a Rust port + a parity trace. Gate: curve-side exception.

**Pre-registered band:** rectified ahead +0.0001..+0.0003 (a monotone 2-parameter term can recover only
the monotone part of the +0.0008 the free-form residual bought); imm -0.00005..+0.00005. Falsifier:
`softplus(c)` trained to ~0 everywhere = the mixture already spans the family; close the head-form
sub-family at 0/2 with iter 27.

**Deploy debt:** forward-pass change; port + fresh trace.

---

## 5. Direct curve-logit recalibration (shift + temperature, 2 params), re-screened with KD OFF

**Family:** calibration / curve-head expressiveness. (Calendar-aware curve is dead; this is not it.)

**Descends from:** iters 55 and 58 (KD `alpha_decay` 0.9 and 0.25 -- the KD-SCHEDULE route to
calibration, closed as an interior optimum at 0.5) and the horizon screen (the champion is
overconfident by -0.00292 on TRAIN users; a 1-param logit shift recovers +0.000115 held out, Platt
+0.000131). Iter 58's write-up: "A direct recalibration -- a learned output temperature or shift on
the curve head -- remains untested, and is now the natural way to collect that gain."

**Why the closing evidence does not cover it.** Iters 55/58 tested whether LESS or MORE teacher
recalibrates the head; both directions lose because KD has a countervailing term (variance
reduction). The direct route has no countervailing term. The honest objection is the opposite one:
the measured overconfidence was ATTRIBUTED to KD, and KD is off on this lineage, so the prize may have
vanished -- which is why this is a screen first. It survives the redundancy test: the GRU head's
logit is `logit(sum_i w_i (1+t/S_i)^-d_i)` (`srs_model.py::gru_forgetting_curve`) and NO free linear
sits after it -- there is no additive logit bias in the parameterisation, so a global shift is a
genuinely new degree of freedom. Note the by-review-index slice (`calib_by_result.txt`): the gap
GROWS with the card's history (5th-8th -0.0051, 9th-16th -0.0037, 17th+ -0.0144), which a plain BCE
optimum inside this family need not remove.

**Mechanism.** `_get_loss`: `curve_logits = a * curve_logits_raw + b` before `torch.sigmoid`, with
`a=1, b=0` init (byte-identical), 2 root Parameters behind `RWKV_CURVE_CALIB=1`. Two routes: (i)
POST-HOC -- fit (a, b) on the shared pass's TRAIN-user records by minimising the BY-USER-mean BCE (the
metric, not the row-pooled loss), then an EVAL-ONLY run with the values baked in (~4 h, no training);
(ii) TRAINED -- learn them end to end (~10.5 h; the trunk co-adapts). Route (i) first; it is the
cheapest iteration in the queue.

**Why ahead specifically.** The shift lives on the curve logit only; the rating head is untouched, so
imm is identical to the last digit under route (i). PAVA pools PROBABILITIES of the 4 probes, so the
shift must be applied inside the model before pooling (train, eval, RNN, Rust identically -- one scalar
pair in `pava.rs`'s caller).

**CPU screen (from the shared pass).** Re-run `recalibration_prize.py`'s fit on the realcyc records
with a BY-USER 2-fold split and by-user weighting. Kill if the held-out by-user prize < +0.0001 in
ahead OR the TRAIN-user gap |mean p - mean y| < 0.001 (no miscalibration to collect). Also slice by
review index: a monotone gap in review index argues for the trained route (the head can condition the
shift), a flat one for post-hoc.

**Cost:** ~1 h CPU + 4 h eval (route i). Gate: curve-side exception (imm cannot move under route i).

**Pre-registered band:** ahead 0..+0.00013; my honest prior is 50% that the screen kills it (BCE on
hard labels with KD off should have removed most of the global gap). Falsifier is the screen itself.

**Deploy debt:** one scalar pair in the Rust curve evaluation.

---

### Ranking by expected ahead gain

| rank | proposal | family | expected ahead | cost | gate |
|---|---|---|---|---|---|
| 1 | Born-again KD from realcyc | distillation | +0.00015..+0.00045 | 2 h dump + 10.5 h | both modes |
| 2 | imm-scale 0.5 (ahead-favoring reweight) | objective | +0.00015..+0.00045 (imm -0.0002..-0.0005) | 10.5 h | DIRECTED (Andrew) |
| 3 | Duration dropout on dim 8 (iter 33's retry) | contract alignment | +0.00015..+0.00050 | 10.5 h | both modes |
| 4 | Odds-power monotone residual | curve-head form | +0.0001..+0.0003 | 10.5 h + port | curve-side; CONTRACT sign-off |
| 5 | Curve-logit recalibration, KD-off re-screen | calibration | 0..+0.00013 | 1 h CPU + 4 h eval | curve-side |

Proposal 5 is ranked last on expected gain and first on cost: its screen is a slice of the shared pass
and its verdict is an eval-only run. Proposals 1 and 3 do not interact (KD targets vs input mask) and
could be chained; 2 interacts with everything through the trunk and should run alone against a
settled reference. Provenance: 1 is `adopted` (Born-Again Networks, Furlanello et al. 2018, arXiv
1805.04770, plus iter 54's own tie); 2, 3, 4, 5 are `invented` -- so the strict alternation rule
places 1 in the next adopted slot and any of 2-5 in the invented one.
