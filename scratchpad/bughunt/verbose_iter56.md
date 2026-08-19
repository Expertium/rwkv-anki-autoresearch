
## iter 56 — linear LR decay shape (`RWKV_DECAY_SHAPE=linear`, dir `iter57_decayshape`) — REJECTED, and the follow-up is closed by arithmetic

**The lever.** Replace the decay phase's default LR shape `1 - sin(pi x / 2)` (mass 0.3634) with a
linear ramp to zero. Single variable vs iter 45: same WS-final, same KD dump, same alpha schedule
(0.5 in decay), same 10,935 steps. Decay-only, so the lever cannot touch WS by construction.
Zero code, ~3.5 h of decay plus a 2.9 h eval.

**The numbers.** 2500 VAL users, `size` 0/2500 mismatches against both references, `nan_users` 0,
params **558,212 exactly unchanged** (a schedule change has no weights).

| comparison | ahead | imm | reading |
|---|---|---|---|
| vs **iter 53** (the champion — THE GATE) | 0.297640 = **−0.000117**, p=0.985 | 0.265271 = **−0.000080**, p=1.000 | both worse → **REJECTED** |
| vs **iter 45** (the recipe it was built on — the CONTROLLED effect) | **+0.000057**, p=6.0e-12 | **+0.000104**, p=3.1e-161 | real, but sub-bar on ahead |

**The basis is confirmed from the run's own log, not from the label.** Its decay log prints the
pre-iter-53 Muon split — `500,800 matrix params on Muon, 57,412 on AdamW` — against iter 53's
`528,320 ... incl 27,520 LoRA in a wd=0 group, 29,892 on AdamW`. So `RWKV_MUON_INCLUDE_LORA` is
genuinely absent here and the two levers are orthogonal. (CLAUDE.md's standing warning applies:
diff the runners, do not read the labels.)

**The asymmetry is the finding.** On **ahead**, +0.000057 sits *inside* the ±7.5e-5 noise floor, so
its reality rests on rank consistency (p=6e-12) rather than on magnitude — and iter 44 is the
standing warning that rank and magnitude can disagree. On **imm**, +0.000104 clears both the floor
and the accept bar. The decay shape does something real to the rating head and almost nothing to the
curve head.

### The decision-relevant result: the obvious follow-up is dead, priced before queueing it

The natural next run is *"does linear decay stack on iter 53?"* — the levers are orthogonal and were
never tested together. Priced under **perfect additivity**, a stacked run would land at
**+0.000057 ahead / +0.000104 imm vs iter 53**, and the ahead half **fails** the 0.0001 bar. So even
the best case does not clear the gate, and no mechanism predicts super-additivity.

**Do not spend 6.1 h on it.** This bounds the schedule-shape family by arithmetic rather than by a
second run — the same move that the CPU screens have been making all week, applied to a result
instead of to a proposal.

### The pre-registration was half wrong, and that is why it was written

`scratchpad/iter57_decayshape/GATE.md`, written at 04:25 with the eval at 2202/2500 and **no result
inspected**, predicted a **null**. The reasoning: iters 41/43/44 showed that same-capacity
rearrangements are mutually indistinguishable at |Δ| ≤ 7.5e-5, and reshaping the decay *reallocates*
the same optimization budget rather than adding to it.

**Ahead behaved exactly that way. imm did not, at p=3e-161.** So the budget-reallocation intuition is
not a general law about this trunk — it holds for the curve head and fails for the rating head. That
is a sharper statement than the verdict, and it only exists because the prediction was committed
before the number.

The same file pre-registered the **rule** as well as the prediction: **BOTH-modes, not the curve-side
exception**, because an LR-schedule change acts on the whole trunk through the optimizer. Fixing that
in advance is what prevents a post-hoc switch to the curve-side rule — which imm's +0.000104 against
ahead's noise-floor +0.000057 would have made genuinely tempting.

### Ops

* Clean run, `DONE_EXIT_0` at 04:40:48 — within a minute of the 04:41 projected from a mid-eval rate
  fit (14.53 users/min).
* Its decay checkpoint is **`scratchpad/iter45_kddecay/i57_d_10935.pth`**, not in its own directory:
  `write_decay_setup.py` writes beside the WS-final it warm-starts from. An empty run directory is
  expected here, not a failure.
* **The log lies about the alpha and the run is correct.** Two `echo` lines carry iter 52's prose, so
  the live log reads `DECAY_OK KD ON alpha 0.9` for a run whose alpha is 0.5 — confirmed 0.5 in the
  decay log itself (`alpha FIXED at 0.5`), and the guard checked 0.5. Nothing computed depends on the
  echo. The defect was predicted hours earlier by the new preflight check and written into `GATE.md`
  *before* the line was emitted.
* **Deploy debt: none** — a training-schedule change, forward pass untouched.
