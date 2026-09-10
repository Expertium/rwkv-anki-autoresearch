# Pre-registration: what arm 2 (`w10qat`) means, written BEFORE the number exists

Recorded 2026-09-10 12:45, with arm 2's decay at step ~4,900 of 21,870 and no number in existence.

## What arm 2 is, and why its difference from arm 1 is a clean QAT tax

ws10's WS-final (`w10_ws_109350.pth`) + a 2-epoch (21,870-step) decay with the full deploy
quantization stack: rank-1 low-rank int4 card/note WKV state, the joint-uv b10 WKV catalog, the
m2b12 shift catalog, 1-bit norms, BOTH catalogs learnable, starting from catalogs refit on ws10's
WS-final (`reference/pq_cb_{wkv,shift}_ws10_*.txt`). Evaluated QUANT-AWARE on the 2,499 VAL users.

**Single variable, verified by diffing the runners, not by reading their labels.** arm 1's and arm
2's runner envs differ in exactly the eight `RWKV_QAT_*` quantization variables (plus paths and
tags); both call `write_decay_setup.py scratchpad/ws10 w10_ws ... 2.0 1e-3 65536` identically.
Arm 2 also puts `cl.exe` on PATH (vcvars64) -- inductor needs a C++ compiler for the QAT graph and
arm 1's plain graph never asked for one; it is part of "QAT + torch.compile", not a separate
variable. **So `arm2 - arm1` IS the QAT tax at the 10+2-epoch budget, with nothing to subtract.**

## ⚠ Fixed before the number existed: the eval would have used the START catalogs

Found 2026-09-10 12:30 at decay step ~4,800. `run_w10qat.cmd`'s phase C inherited the training env,
so it would have evaluated with the START catalogs while the weights were co-adapted to the LEARNED
ones (by step 4000 those had moved 43% WKV / 45% shift in relative L2). The learned centroids are
module globals, not part of the checkpoint; train_rwkv exports them to
`scratchpad/ws10/w10q_d_{wkv,shift}cb_<step>.txt` and nothing loads them back. `qtaxd_cblearn`
re-pointed its eval env at them (`run_cblearn.cmd`), and that step was lost when arm 2's runner was
built from the plain one. The runner's cmd was stopped (the decay python kept running) and phase C
moved to `run_w10qat_eval.cmd`, whose 10-user probe must LOAD the step-21870 learned catalogs, with
learning off, before the full eval runs. So the number below is the deploy configuration.

## The reference numbers

| model | budget | quantized | ahead | imm |
|---|---|---|---|---|
| arm 1 `w10plain` (the control) | 10 + 2 ep | no | 0.297577 | 0.262066 |
| recorded tax, `qtaxd_cblearn` - `iter45_kddecay` | 1 + 1 ep | yes - no | +0.002286 | +0.003486 |
| arm 1 + recorded tax (a naive forecast) | | | 0.299863 | 0.265552 |
| stop criterion (Andrew 2026-09-02) | | | <= 0.2950 | <= 0.2640 |

⚠ The recorded tax is NOT a single-variable comparison with arm 2's: it differs in budget (1+1 vs
10+2), basis (published end-to-end dbs vs gen 5), KD (on vs off), features (92 vs 109 dims) and the
catalog start. It is a prior, not a control.

## Q1 -- does the 10x budget change the QAT tax? (CLEAN within arm 2; the comparison to 1.25 ep is not)

Prediction: **the tax lands near the recorded one** -- ahead +0.0015..+0.0030, imm +0.0024..+0.0045.
Reasons it could move, both weak: ws10's states reconstruct ~8% worse than realcyc's at the same
bits after a refit (0.5587 vs 0.5185 held-out WKV) -- but reconstruction cannot rank working
catalogs (2026-08-15); and the QAT phase is twice as long (2 vs 1 epoch) -- but closure saturates
by ~0.37 epochs.

* **tax ahead < +0.0015** -- the budget SHRINKS the tax: a better-trained model is more robust to
  quantization, and the deploy picture is better than the arithmetic in CLAUDE.md assumed.
* **+0.0015 .. +0.0030** -- unchanged within tolerance; the 1.25-ep tax transfers, and the
  QAT-tax retry attacks the same terms it did then.
* **tax ahead > +0.0030** -- the budget GROWS the tax, the QAT tax becomes the largest lever left,
  and the byte-budget question Andrew answered conditionally ("if that's the only way, alright")
  comes forward.

## Q2 -- which mode pays more?

Prediction: **imm pays more than ahead**, as in both prior measurements (+0.004445 vs +0.002896 at
d=32/n=5000; +0.003486 vs +0.002286 on this trunk). If ahead pays more, the rating head has become
the robust one at 12 epochs, which would be new.

## Q3 -- does anything survive quantization against the stop criterion?

Prediction: **neither mode meets it at deploy.** Ahead cannot (arm 1 is already 0.0026 short in
full precision). imm has 0.0019 of spare in full precision against a tax of ~0.0035, so the naive
forecast is 0.2656 -- short by ~0.0016. **If imm <= 0.2640 at deploy, the imm criterion survives
quantization** and only ahead is binding.

## Q4 -- engagement (checked before the full eval, by the runner, not after)

The probe's log must name `scratchpad/ws10/w10q_d_wkvcb_21870.txt` and
`.../w10q_d_shiftcb_21870.txt` with `learnable=False`, and its cost against arm 1 on users
5001-5010 must sit in [-0.005, +0.02]. A failure there is a broken eval, not a result.

## What arm 2 decides next -- the QAT-tax retry (Andrew 2026-09-10)

Andrew: "I'd rather not increase card/note state size, but if that's the only way, alright." So
zero-byte routes first. What this pre-registration adds, found while writing it:

* **Route (a), "refit the catalogs on the model that deploys", is already inside arm 2.** The
  catalogs are LEARNED against the deploying model's own task loss from a ws10 refit start. A
  post-hoc k-means refit would reconstruct better and rank nothing (the learned catalog
  reconstructs WORSE than its frozen start by design). Do not re-propose it.
* **Route (b), "chunk structure at constant bits", is MOOT on the side that pays.** The WKV
  catalog is already one chunk per head (m=1, joint u+v, 10 bits); the only way to use fewer,
  bigger chunks is to code heads jointly, which needs a 2^20+ entry catalog -- not deployable.
  The shift side is where the chunk axis lives, and it carries ~1/14 of the tax.
* **THE ZERO-BYTE AXIS THAT REMAINS: the SPLIT of the frozen bytes between WKV and shift.** A
  card is ~182 bits (~23 B): WKV 2 layers x 5 heads x (10 + 1) = 110 bits, shift 3 vectors x 24 =
  72 bits. Taking 8 bits from each shift chunk (m2b12 -> m2b8, -24 bits) buys +2 bits per WKV
  head (b10 -> b12, +20 bits) at 4 bits UNDER budget. On reconstruction the WKV side gains ~15%
  (0.3776 -> 0.3224); the shift side's cost at 8 bits is unmeasured (going UP from 12 bits barely
  moved it; going DOWN is where it bites). ⚠ Reconstruction cannot price this -- it needs the
  same ~20 h logloss A/B as a bits rung, one run, not a sweep.
  Bit counts CHECKED against the arch (`architecture_d80_lora4_cnd.py`: card 2 layers, note 1;
  `RWKV_STRIP_CMIX` removes card layer 1's channel mixer, so card carries 3 shift vectors and note
  2): **card 110 + 72 = 182 bits (22.75 B, the record's "~23 B"), note 55 + 48 = 103 bits.** The
  same trade on note is -16 shift bits for +10 WKV bits, 6 under budget.
  **SCREENED 2026-09-10 12:55 on ws10's own corpus, held out BY USER (fit 107 + 136, score 156):**
  WKV b10 -> b12 refit 0.5587 -> 0.5219 (-6.6%, a LOWER bound -- 12 training vectors per centroid);
  shift m2b12 -> m2b8 refit TS 0.4818 -> 0.5339 and CS 0.3968 -> 0.4404 (+11% each), far from the
  random control (1.07). **So the shift side does NOT collapse at 8 bits; the route is alive, not
  free.** A crude linear map (the PTQ probe matrix put the WKV side at ~14x the shift side's cost)
  says net +0.0001..+0.0003, which iter 47 warns may not survive contact with logloss -- a 43%/75%
  cut in rank-1 error there bought nothing. Worth ONE ~30 h A/B only if arm 2's tax says the
  codebook term is where the cost sits.
  ⚠ A side observation that bears on any catalog design: the same b10 catalog scores **0.3629**
  on user 156 when 156 was in its fit and **0.5587** when it was not. WKV states are strongly
  USER-specific, so held-out error is dominated by user diversity -- which the LEARNABLE catalog
  addresses by training on all 5,000 users' task loss, and a 3-user k-means start cannot.

* **A route that needs ANDREW, not a build: distil from the full-precision twin DURING QAT.**
  The recorded 1.25-ep tax had KD ON in both of its arms (alpha 0.5 in the decay); arm 2 runs
  KD-OFF, so if arm 2's tax comes in well above the prior, KD-off is one of the five differences
  and the only one that is cheap to restore. Arm 1 IS the ideal teacher -- same WS, same decay
  batches (seeded), full precision -- so this is the standard QAT-with-distillation recipe
  (e.g. LLM-QAT, arXiv 2305.17888, distils the quantized student from its own fp model), costing a
  ~2 h logit dump plus one QAT decay + eval (~30 h). iter 54 does NOT argue against it: it
  showed the teacher's IDENTITY does not move the tax, with KD on in both arms, never KD vs none.
  ⚠ It reintroduces KD, which Andrew closed on 2026-09-04 ("Let's skip KD") -- in the context of a
  4-day teacher retrain for the research lineage, not this. Ask him; do not build it on inference.

## PHASE D -- the tax split into its parts (added 2026-09-10 19:00, before any number)

Arm 2's eval runner now ends with a decomposition of THIS checkpoint on users 5001-5300: shift
quantization off (S), WKV codebook + norm quant off (C), and the remainder R = T - S - C (rank-1 +
drift; no arm can remove rank-1, cell 3). It decides the bit split mechanically. Design, predictions
and decision rule: `scratchpad/qat_decomp/PREREG.md`. ~2.5 h, never fatal to this runner.

## PREVIEW (added 2026-09-10 16:05, decay step 8,042 of 21,870 -- NOT the verdict)

In-training validation, arm 2 vs arm 1 at matching decay steps, the SAME 10 users (5001-5010,
594,215 rows, row-weighted, 4 dp). Training validation runs the student with its CURRENT learned
catalogs, so this is a consistent read of the tax as it evolves. The same instrument predicted arm
1's verdict within 0.0004 in both modes.

| decay step | ahead tax | imm tax |
|---|---|---|
| 50 | +0.0097 | +0.0128 |
| 2,000 | +0.0028 | +0.0048 |
| 4,000 | +0.0029 | +0.0048 |
| 6,000 | +0.0033 | +0.0048 |
| 8,000 | +0.0031 | +0.0052 |

* **The tax closes within ~2,000 steps (~0.18 epochs) and is FLAT after it.** Consistent with the
  record's "closure saturates by ~0.37 epochs", and sharper: ~90% of the 2-epoch QAT phase buys no
  further closure. Relevant to the COST of every future QAT A/B, not to this one's validity.
* **If the preview transfers as it did for arm 1, arm 2 lands near +0.003 ahead / +0.005 imm** --
  at the top of Q1's band on ahead and ABOVE the pre-registered imm band (+0.0024..+0.0045). That
  would read as "the budget does not shrink the tax, and on imm it grows it", and deployed imm near
  0.267 would MISS the 0.2640 criterion that full precision meets.
* ⚠ Row-weighted 10-user validation is not the by-user 2,499-user gate; do not quote this as the
  tax. It is recorded now so the verdict can be compared with a prediction made before it.
