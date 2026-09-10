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
