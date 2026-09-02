# lorawd -- PRE-REGISTRATION (written 2026-09-02 22:20, before any number exists)

**Slot:** `adopted` (the alternation rule: iters 51-59 were invented, 60/61 Andrew's arms).
**Source:** Moonlight, *Muon is Scalable for LLM Training*, arXiv 2502.16982 (Liu et al. 2025):
Muon without weight decay lets weight norms grow without bound and degrades long runs; adding
decoupled weight decay (matching AdamW's) fixes it. Our own measurement of the same mechanism:
iter 53 put the LoRA matrices on Muon in a `wd=0.0` group and their deployed `||W||_F` ended
**+62.4%** over the champion's, still rising at the end of the run.
**Dose:** 0.05, from the 2026-08-18 time-constant screen (PROPOSALS "RANK 8 RE-SPECIFIED"): the
LR cancels in Muon's norm equilibrium, so the brake's time constant is 1/wd steps; 0.01 = 100k
steps (never engages in a 21,870-step run), 0.05 = 20k (acts on the run's own timescale).

**The lever, one line:** `RWKV_MUON_LORA_WD=0.05` -- decoupled wd on the LoRA Muon group ONLY;
every other group keeps its wd. Default 0.0 = byte-identical (smoke: `smoke_lora_wd.py`, 3 arms,
identity-based, non-vacuous).

**Control:** gen4base (KD-off, gen-4 dbs, id4), the features lineage's baseline -- or realcyc if it
promotes, in which case `mk_lorawd.py realcyc` regenerates the runner before the waiter fires.
Single-variable by construction (generator asserts in both directions). Gate: BOTH-modes rule
(an optimizer change acts on the shared trunk), raw >= +0.0001 each, p < 1e-4 each, size gate
against `size_baseline.py` (the id_e2s lineage baseline snapshotted from gen4base).

## Predictions

* **P1 (direction):** BOTH modes improve. The growth is unregulated and the model is data-limited,
  so a norm brake is a regularizer where regularizers have paid (Muon itself pays as one).
* **P2 (magnitude):** ahead +0.0001..+0.0004, imm +0.0001..+0.0005. Iter 53's own gain was
  +0.000174/+0.000184; a brake on its side effect should be the same order, not larger.
* **P3 (engagement):** final LoRA `||W||_F` growth vs the WS-start value falls from ~+62% to
  under +25%. If it does NOT fall, the flag was inert or the dose too small, and the verdict is
  uninterpretable rather than a null.
* **P4 (counter-hypothesis, pre-registered both ways):** if wd=0.05 REGRESSES both modes, the norm
  growth is load-bearing (the rank-4 bottleneck needs the magnitude) and the 10x endgame needs NO
  brake on this group -- worth knowing for a 4-day run. Then do not retry at 0.02; the family is
  closed on mechanism, with one exception: a regression on imm only would suggest the rating head's
  LoRAs need it while the trunk's do not, which is a different lever (per-stream wd).

## What is measured at verdict
1. ahead/imm on VAL 5001-7500, rectified, vs the control's jsonls (`paired_pvalue.py --intersect`).
2. `size_baseline.py check id_e2s` on the result.
3. LoRA norm growth (start -> end) for candidate and control from the checkpoints -- P3.
4. Cost: WS + decay from F: at ~0.69 steps/s each, eval ~3 h => ~12 h.
