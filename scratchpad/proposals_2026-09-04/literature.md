# Literature-prior proposals, 2026-09-04 (agent 1 of 3)

Reference for every band below: **realcyc**, ahead 0.298083 / imm 0.263592 (n=2,499, gen-5 dbs,
KD OFF). Stop criterion ahead <= 0.2950, so ahead (0.0031 away) is the binding mode. Noise floor
+/-7.5e-5; accept = raw >= 0.0001 in both modes at p<1e-4 (curve-side levers: ahead >= 0.0001,
imm not significantly worse).

Facts these proposals lean on, all from this repo's record:
* The ahead objective is supervised by ONE bit per row (`label_y`) at ONE horizon (`label_elapsed_seconds`),
  while the same real row carries an UNUSED 4-way label: `data_processing.py:431-445` shifts
  `["elapsed_seconds","elapsed_days","y","rating","review_th"]` by -1 per card, so on a real row
  `label_rating` IS the next review's button. `srs_model.py` consumes it only through
  `immediate_mask = is_query * has_label` (line ~1476) -- the ahead position never sees it.
* Muon pays as a REGULARIZER (train edge decays to zero, held-out edge holds; iters 29/53). Explicit
  NOISE regularizers were tuned DOWN (iter 34's tuner took dropout x0.5; wd 0.2 lost at iter 3).
  So "regularization pays" is true only for the geometric kind, not the noise kind.
* KD from a bigger teacher was worth ~+0.0019 over iters 32/35/39/45 and the features lineage runs
  with ZERO teacher signal. The d=128 teacher retrain is scheduled elsewhere and is NOT proposed here.
* Same-forward-pass self-distillation (iter 46) is a clean null; adding a correlated SECOND teacher
  (r=0.946) was priced at ~21% of a +0.00016 change. Neither of those is a frozen, separate-pass,
  no-other-teacher setting.

---

## 1. Ordinal next-rating supervision on the forgetting curve (CORAL-style cutpoints)  — rank 1

**Family:** objective / multi-task supervision (curve-side).

**Provenance (adopted):** Cao, Mirjalili & Raschka 2020, *Rank consistent ordinal regression for
neural networks with application to age estimation* (CORAL; Pattern Recognition Letters 140, arXiv
1901.07884): K-1 binary tasks P(y > r_k) sharing ONE weight vector and differing only in K-1 biases,
which guarantees rank-monotone probabilities. Decomposition per Frank & Hall 2001 (*A simple approach
to ordinal classification*, ECML). Multi-task benefit per Caruana 1997.

**Mechanism in this model.** The next rating is ordinal on memory strength: Again < Hard < Good < Easy.
Today the curve head learns only the Again/not-Again boundary. Implement the CORAL form on the
curve's own logit:

    z(t)        = logit R(t)                      (existing: gru_forgetting_curve, srs_model.py:1014)
    P(r >= 2|t) = sigmoid(z(t))                   = the existing ahead BCE on label_y   (unchanged)
    P(r >= 3|t) = sigmoid(z(t) - b_3)             b_3 > 0, new
    P(r >= 4|t) = sigmoid(z(t) - b_4)             b_4 > b_3, new

Two new BCE terms on the SAME rows and mask as `curve_loss` (`ahead_wmask`, line ~1520), with targets
`(label_rating >= 2)` and `(label_rating >= 3)` after the existing `label_rating - 1` clamp
(line ~1394), scaled by `RWKV_ORD_SCALE` (start 0.25 each; the existing `AHEAD_SCALE = 0.5`). Cutpoints
`b_3, b_4` are 2 scalars (parameterize `b_4 = b_3 + softplus(.)` for order). Optional variant B: make
the cutpoints row-dependent, `b = Linear(x_w -> 2)` from the shared `head_w` trunk output
(`head_and_out`, line 969) -- 642 params -- if the scalar version is inert. Rows with `has_label = 0`
(last review of a card) are already masked; the query rows are excluded by `ahead_mask` exactly as now.
KD is off, so `label_y` is the hard label and the ordinal targets are consistent with it.

**Why it can move AHEAD specifically.** It adds supervision ONLY at the ahead position and only
through `z(t)`, i.e. through the curve parameters (w, S, d) and the trunk that feeds them. A Hard
success at t tells the curve "R(t) was barely above threshold"; an Easy success says "far above". The
binary label throws both away. Under the shared-logit form the three thresholds pin the curve's LEVEL
and SLOPE at t from three cuts instead of one, which is exactly the shape information the GRU head's
(w, S, d) triple encodes. For scale: iter 26 (GRU N=3) was the phase's largest ahead gain (+0.00049)
and came from curve-shape resolution; this supplies shape supervision without new parameters. imm is
touched only through the shared trunk, so this is a CURVE-SIDE lever under the gate exception.

**Cheap CPU screen (minutes).** Re-run `scratchpad/spacing_screen/calibration_by.py` against the
realcyc checkpoint on ~5 train-range users (it already records `(p, y, elapsed_seconds, rating)`;
env needs the gen-5 flags `RWKV_ID_FEATURES=1 RWKV_REAL_CYCLES=1`, `RWKV_ZERO_FEATURES` empty, the
`-id` data path). Then, on SUCCESS rows only, compute AUC(p_champion; Easy vs Hard) and
AUC(p_champion; Good vs Hard). Kill rule: if AUC(Easy vs Hard) > 0.75, the curve already separates the
grades and the extra cuts have nothing to teach. Pre-registered expectation: 0.58-0.68 (the rating
head is trained on this distinction; the curve is not). Also print the success-row rating mix;
if Hard + Easy < 8% of successes the targets are degenerate -- kill.

**Pre-registered band.** ahead **+0.0001 .. +0.0004**; imm **-0.00005 .. +0.00015** (trunk only). Abort
line: ahead worse by > 0.0001 at p<0.05 (the cuts fight the binary boundary). Engagement diagnostic:
`b_3, b_4` must move off init; the ordinal BCEs must fall below their constant-predictor value.

**Redundancy with closed items -- why it is not one.** Not ahead<-imm ROUTING (iters 46/48 moved model
OUTPUTS between heads; this uses a DATA label the ahead path has never been given). Not KD (no
teacher, hard labels). Not `pbin` (iters 17/19 put the ahead label on the RATING head; this is the
reverse direction and lands on the curve). Not capacity (2 params). ⚠ Honest risk: the "use is not
evidence of need" pattern -- the cutpoints will train; report the AUC screen number next to the verdict
so a null is interpretable. Deploy: none (the deployed quantity is still R(t) = sigmoid(z)); the
cutpoints are train-only.

---

## 2. Born-again self-distillation from the FROZEN realcyc checkpoint  — rank 2

**Family:** distillation (the family with the only real hit rate, 4/5).

**Provenance (adopted):** Furlanello, Tschannen, Itti, Anandkumar 2018, *Born Again Neural Networks*
(ICML; arXiv 1805.04770): a student of IDENTICAL architecture trained on a frozen teacher's soft
targets beats the teacher; gains persist over generations and hold without "dark knowledge"
(their permuted-logit control). Mobahi, Farajtabar & Bartlett 2020 (*Self-distillation amplifies
regularization in Hilbert space*, NeurIPS) give the mechanism: self-distillation is a data-dependent
regularizer that shrinks along low-eigenvalue directions -- i.e. the GEOMETRIC kind that pays here.

**Mechanism in this model.** Zero model code. The existing KD-from-dump path (`train_rwkv.py:736-770`
dump mode; `srs_model.py` `kd_mix` at ~1425 and ~1512) with `RWKV_KD_TEACHER =
scratchpad/realcyc/rc_d_10935.pth` and `RWKV_ARCH_MODULE` = the same `_cnd` arch (teacher and student
share the architecture, so no arch swap -- the one place this differs from the d=128 dump runner).
Dump ~2 h on the gen-5 batch stream (augmentation is OFF, seed 4321, so the dump is aligned by
construction; the `labels_sum` checksum applies). Student = realcyc's recipe + `RWKV_KD_MIX` with the
champion's tuned alphas **0.9 WS / 0.5 decay** unchanged (KD alpha schedules are closed; they are not
touched). One generation only.

**Why it can move AHEAD.** The record attributes KD's payoff to target-VARIANCE reduction (alpha peaks
at 0.9), and the ahead target is the noisiest target in the loss: one Bernoulli draw at one horizon.
A frozen same-size teacher supplies a calibrated p(t) for every ahead row, exactly the quantity the
variance argument wants, and BAN's own result is that the teacher need not be bigger. The d=128
teacher gave +0.00058 ahead at iter 32 (its first application); a same-size teacher should return a
fraction of that. It is also the only way to put ANY teacher signal back into the features lineage
before the d=128 retrain lands, and it is dropped the day that teacher exists.

**Cheap CPU screen (minutes).** Calibration of the TEACHER on held-out users, because a BAN student
inherits it (the 2026-08-17 horizon screen: the champion's overconfidence was KD-inherited). Run
`calibration_by.py` + `recalibration_prize.py` on realcyc for ~5 users from the TUNE range
(5001-6000 is inside VAL and is the tuner's range, so allowed for coarse checks). Kill rule: if the
one-parameter logit-shift prize on the teacher exceeds 0.0003, the student will inherit a bias larger
than the gain band and the run is not interpretable -- do not launch (or launch only with the decay
alpha, 0.5, in WS too -- NOT proposed here because that is a schedule change). Second screen, free:
assert the dump's per-step `labels_sum` matches on the first 50 steps of the student -- the
augmentation/KD trap is fatal and silent if the seed differs.

**Pre-registered band.** ahead **+0.0001 .. +0.0004**; imm **+0.0000 .. +0.0004**. Both-modes rule
(KD rewrites both objectives; see the KD-is-not-curve-side note in CLAUDE.md). Abort: either mode
worse by > 0.0002.

**Redundancy -- why it is not iter 46 or the ensemble screen.** Iter 46's teacher was the SAME forward
pass (a different head on the same representation), and iter 46's own conclusion said the fix is a
teacher that is not the same forward pass -- a frozen past checkpoint is the named variant. The
ensemble screen priced a second teacher ADDED to an existing d=128 teacher at r=0.946; here there is
no first teacher, so the target shift is |p_teacher - y_hard| = the whole soft target, not the
difference between two teachers. Not the scheduled teacher retrain (different architecture, different
purpose; this costs 2 h of dump, not a big run). Deploy: none.

---

## 3. Sharpness-Aware Minimization, decay phase first  — rank 3

**Family:** regularization (loss geometry), optimizer-adjacent.

**Provenance (adopted):** Foret, Kleiner, Mobahi & Neyshabur 2021, *Sharpness-Aware Minimization for
Efficiently Improving Generalization* (ICLR; arXiv 2010.01412). Cheap variants: Liu et al. 2022
*Towards Efficient and Scalable Sharpness-Aware Minimization* (LookSAM, CVPR; SAM step every k),
Kwon et al. 2021 ASAM (scale-invariant rho). Reference implementation: `github.com/davda54/sam`.

**Mechanism in this model.** In the training step (`train_rwkv.py` main loop, the backward/step
block around the `scheduler.step()` at :1519): after the normal backward, compute the per-tensor
ascent `e = rho * g / ||g||` over the Muon+AdamW parameter set, add it in place, run a second
forward+backward on the SAME batch, restore the weights, then let `MuonAdamW.step()` consume the
perturbed gradient. Muon's orthogonalization and the wd handling are untouched -- SAM only changes
WHICH gradient the optimizer sees. Flag `RWKV_SAM_RHO` (0 = byte-identical), `RWKV_SAM_EVERY`
(LookSAM; 1 = every step). ⚠ The second forward must reuse the batch's dropout mask or run with
dropout off; take the ascent on the fp32 master model (the bf16 child copies gradients back through
`transfer_child_grad_to_master`, :401). **First cut = DECAY-ONLY** (warm-start from
`rc_ws_10935.pth`, the 36% discount): 2x cost on 3.1 h = ~6.2 h + eval.

**Why it can move AHEAD.** The evidence that this model's wins are GENERALIZATION wins is direct:
Muon's train-loss edge decays to zero while its held-out edge holds (+0.0019, the matched iter-29
pair). SAM buys flatness explicitly instead of as a side effect of an update rule. Both modes
should move; ahead is where the loss surface is noisiest (1-bit labels at a stochastic horizon), so
a flatness bias on the curve parameters is the more plausible beneficiary -- stated as a
hypothesis, not a mechanism claim. Both-modes rule.

**Cheap CPU screen (~20 min).** Sharpness probe on the realcyc checkpoint: on ~16 training chunks
(CPU `reference_rwkv7` path, autograd works there -- the parity harness uses it), compute
`L(w + rho*g/||g||) - L(w)` for rho in {0.01, 0.02, 0.05} and report the MAX over chunks (a median
cannot see a blow-up, and here a median cannot see the sharp directions either). Kill rule: if the
gap at rho = 0.05 is below 0.002 (~0.5% of the loss) the minimum is already flat at SAM's scale and
the penalty is at the noise floor -- dead. Second screen, free: from realcyc's WS log, the last-500-step
train logloss vs the final val logloss at matched rows; if the train-val gap is ~0 there is no
generalization gap to close. (Both screens bound the prize; neither proves it.)

**Pre-registered band.** ahead **+0.0000 .. +0.0003**; imm **+0.0000 .. +0.0003**. Null expected if
Muon already saturates the flatness that matters (that is the counter-hypothesis, and it is the
useful thing to learn: it would say the regularizer reading of Muon is about SPECTRUM, not
flatness). Abort: either mode worse by > 0.0002 (rho too large -- ASAM or halve rho once, then close).

**Redundancy -- why it is not closed.** The closed optimizer items (PolarExpress, NorMuon, LoRA wd,
cautious wd) all change the UPDATE RULE or the norm equilibrium; SAM changes the OBJECTIVE and leaves
Muon exactly as is. It is not a noise regularizer (the kind the tuner turned down): the perturbation
is deterministic and adversarial. Not tried in any form here (`grep -i sharp|SAM` over the record:
nothing). Deploy: none. Cost is the real objection -- 2x per step -- which is why decay-only is the
first cut and LookSAM (every 4 steps) the fallback.

---

## 4. Auxiliary next-interval regression head (the scheduler's stability estimate as a target)  — rank 4

**Family:** objective / multi-task supervision (trunk-side).

**Provenance (adopted):** Caruana 1997 (*Multitask Learning*, Machine Learning 28) for the auxiliary-
task mechanism; the specific target follows Zhu et al. 2017, *What to Do Next: Modeling User
Behaviors by Time-LSTM* (IJCAI) and Mei & Eisner 2017 (*The Neural Hawkes Process*, NeurIPS), where
predicting the time-to-next-event is trained jointly with the event label and improves the label
task on sparse-feedback sequences. Related in-domain: srs-benchmark's `GRU-P` scores at the
scheduler-chosen interval too, i.e. the interval is a property of the history, not noise.

**Mechanism in this model.** On a real row, `label_elapsed_seconds` (the next review's gap) is today
used ONLY as the point at which the curve is evaluated. It is also the interval the user's SCHEDULER
chose from that card's history -- an external, low-noise summary of the card's stability as assessed
by FSRS/SM-2. Add one head `Linear(w_head_dim=320 -> 1)` off the shared `head_w` trunk output in
`head_and_out` (:969), trained with a Huber loss on `log(1 + label_elapsed_seconds)` standardized with
the `elapsed_seconds` stats already in `data_processing.STATISTICS`, masked by `ahead_wmask`, scale
`RWKV_AUXT_SCALE` (start 0.1). 321 params, train-only, dropped at deploy (the head is not read by
`run_as_rnn` or Rust). Query and probe rows excluded by the existing masks.

**Why it can move AHEAD.** The ahead curve at row k must estimate stability from the state; the
scheduler's interval is a dense, near-deterministic function of the same history that the model
gets 1 noisy bit about. Forcing the trunk to carry "what interval will the scheduler pick" at every
row is a regularizer aimed exactly at the stability-bearing directions of the representation
(Caruana's "hints" mechanism), and it is the AHEAD row that needs them -- the query row already gets
the interval as an input feature. imm through the trunk only.

**Cheap CPU screen (minutes).** Linear probe: run realcyc on ~5 train-range users through the CPU RNN
path (reuse `calibration_by.py`'s walk; capture `x_w` per row), fit ridge regression from `x_w` to
`log(1+label_elapsed_seconds)` on 4 users, score R^2 on the 5th. Kill rule: **R^2 > 0.85** means the
trunk already carries the target and the aux loss is redundant -- dead. Expected 0.5-0.75. Second
number, free from the same records: correlation between the probe's residual and the ahead
per-row loss; if ~0, the information the aux task would add is unrelated to ahead error -- dead.

**Pre-registered band.** ahead **+0.00005 .. +0.0002**; imm **-0.00005 .. +0.0001**. Curve-side
exception does NOT apply (it is a trunk-side lever): both-modes rule. Abort: ahead worse by > 0.0001
(the interval target competes with the curve for trunk capacity -- a real possibility at 564k).

**Redundancy -- why it is not closed.** Not routing (data label, not model output). Not a capacity
add (321 train-only params). Not the calendar-curve lever (that conditioned the OUTPUT on the target
clock; this supervises the REPRESENTATION with a quantity that exists at every row). Not `RWKV_RGATE`
(iter 59 injected an FSRS FORM into the recurrence; this injects nothing at inference). ⚠ Honest
risk: the scheduler differs across users (SM-2 vs FSRS versions), so the target is user-conditional;
the preset/user streams exist to absorb that, and the probe R^2 tells us whether they already do.

---

## 5. Muon coverage completion: the 26 `*scale*` (5 x 80) projections  — rank 5

**Family:** optimizer (coverage axis only).

**Provenance (adopted):** Liu et al. 2025, *Muon is Scalable for LLM Training* (Moonlight; arXiv
2502.16982) and Jordan et al. 2024 (Muon, `github.com/KellerJordan/Muon`): the stated rule is Muon
on EVERY 2-D weight except embeddings and the output head, AdamW for vectors. Our `get_optimizer`
(`train_rwkv.py:127-181`) excludes any name containing `scale`, so `k_scale_linear.weight` and
`v_scale_linear.weight` (shape (H=5, C=80), 26 tensors, 10,400 params = 1.8% of the model) are the
LAST 2-D matrices still on AdamW after iter 53 moved the LoRAs.

**Mechanism in this model.** Same shape as iter 53: a sixth Muon group for the 26 tensors at their
current `wd = 0.0`, flag `RWKV_MUON_INCLUDE_SCALE=1`, ~2 lines + a param-identity smoke like
`scratchpad/iter53_muonlora/smoke_muon_lora.py`. These matrices gate `||kappa||` and `||v||` per
head -- the very factor the 2026-08-17 eigenvalue analysis found limits delta-rule authority
(`||kappa||^2 ~ 0.24`), so they are not decorative.

**Why it could move AHEAD.** Iter 53 established that COVERAGE, not descent quality, is the productive
Muon axis (held-out +0.000174/+0.000184 with zero train-loss gain). This is the remaining coverage.
Both modes; no ahead-specific argument -- ranked last for that reason and for size (1.8% of params
vs 4.9% for the LoRAs, so a proportional expectation is ~+0.00006, under the bar).

**Cheap CPU screen (minutes).** From the realcyc checkpoint vs its WS-50 checkpoint, compute the
per-tensor update anisotropy of the 26 matrices (ratio of top singular value to Frobenius norm of
`W_final - W_init`) and compare with the LoRA tensors' ratio at iter 53 (the spectral pre-registration
in PROPOSALS.md gives the numbers). Kill rule: if the scale tensors are already isotropic
(ratio < 0.5) there is nothing for Muon's orthogonalization to regularize -- dead. Also print their
mean |grad| from any `RWKV_GRAD_STATS` json: near-zero gradient means Muon's normalized step would
be pure noise at a fixed norm -- dead (and a reason to keep them on AdamW).

**Pre-registered band.** ahead **+0.0000 .. +0.0001**; imm **+0.0000 .. +0.0001**. Expected outcome:
sub-bar positive or exact tie. Worth running ONLY as a decay-only or chained filler, never as the
main slot -- Andrew's "stop chasing 0.0001" applies, and this is listed so the coverage axis is
formally closed rather than left half-measured.

**Redundancy -- why it is not closed.** Not weight decay on Muon groups (iter 62 closed NORM CONTROL;
this keeps wd = 0). Not descent quality (no NS change). Iter 53's exclusion list was "lora" AND
"scale"; only the first was ever revisited.

---

### Cross-cutting notes

* Proposals 1 and 4 both add train-only supervision at the ahead position and can be grafted in one
  run if both singles come back sub-bar-but-rank-significant (the pre-registered graft policy);
  they target different quantities (curve LEVEL/SLOPE at t vs representation of stability), so they
  are the less-likely-to-interfere pairing. Do not graft 1 with 2 blind: BAN's soft `label_y`
  changes the binary target while the ordinal cuts keep hard targets, and the two would then
  disagree on the Again boundary -- if both win, run 1 on top of 2 with the ordinal targets
  softened by the same teacher.
* Every item is training-only with the deployed forward pass unchanged, so none adds a Rust port
  gap or a parity trace. 1 and 4 add parameters that are dropped at export; the exported
  `weight_names.json` must not list them (assert at export).
* Provenance column: all five are `adopted` (paper or repo named). The next slot in the strict
  alternation is `invented`, so whichever of these runs next takes the slot AFTER it, unless the
  invented queue is empty.
