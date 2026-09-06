# muongates -- pre-registration (written 2026-09-07 01:05, before launch; ADOPTED slot after eqw)

**Lever:** `RWKV_MUON_INCLUDE_GATES=1` -- the **13 `rkvdag_lerp` (8,1,1,80) token-shift mixing gates,
8,320 params (1.48% of the model)**, move into their own Muon group at the weight decay 0.0 they
already had. They are the last NON-degenerate squeeze-2-D tensors on AdamW; they sit there only
because `get_optimizer` keys its matrix branches on `"weight" in name`. Training-only, forward pass
untouched, params identical (563,652).
**Provenance (adopted):** Muon's own rule -- "every 2-D weight except embeddings and the output head"
(Jordan et al. 2024, `KellerJordan/Muon`; Liu et al. 2025, Moonlight, arXiv 2502.16982).
**Control:** the reference at fire time -- eqw if it passes its both-modes gate, else realcyc.
`auto_control.py` decides mechanically; `mk_muongates.py <base>` regenerates the runner so the flag is
the only difference. **muonscale is deliberately NOT carried** (it was rejected; single-variable).
**Gate:** BOTH-modes: ahead AND imm raw >= +0.0001 at p < 1e-4; size 0/2499.

## Why this lever and not another gate tensor

iter 67 measured that the optimizer gain tracks **which** parameters, not update mass: the k/v scale
gates carried 1.2% of the update energy and bought 59% of iter 53's imm gain. `rkvdag_lerp` is the
same population -- a gate -- at **2.0% of the update energy** and a comparable margin over noise:

| group | tensors | params | update energy | anisotropy vs white-noise floor |
|---|---|---|---|---|
| k/v scale (iter 67, paid imm +0.000109) | 26 | 10,400 | 0.9% | 0.653 vs 0.521 |
| **rkvdag_lerp (this lever)** | **13** | **8,320** | **2.0%** | **0.543 vs 0.449** |
| `bonus` -- EXCLUDED | 13 | 1,040 | 1.1% | degenerate |

**`bonus` is excluded on a mechanism, not a hunch.** `MuonAdamW` reshapes as
`p.grad.reshape(p.size(0), -1)` (muon.py:184/210), so `bonus` (1,1,5,16) becomes a **(1, 80) row
vector**, on which Newton-Schulz is a normalisation rather than an orthogonalisation -- Muon would be
a fixed-norm step, not the treatment. The code guard is a property of the reshape (`shape[0] >= 2 and
numel()//shape[0] >= 2`), not a name list, so a future tensor is judged the same way. The smoke
proves BOTH directions: every admitted tensor reshapes to (8, 80), and every tensor rejected for
degeneracy really is degenerate (14/14 PASS).

## Predictions
- **P1 (direction).** imm improves by **+0.00005 .. +0.00025** (iter 67's +0.000109 at 2.2x the update
  energy, discounted for the smaller anisotropy margin). **ahead: genuinely unknown, and that is the
  point of the run** -- iter 53 (LoRAs) moved BOTH modes, iter 67 (delta-rule gates) moved imm only.
  If ahead moves, the gate population is not imm-specific and the optimizer family is still open on
  the binding mode. If it does not, that is 2/2 and the family is closed on ahead with a mechanism.
- **P2 (engagement).** **Deliberately NOT the cumulative-anisotropy probe** -- iter 67 proved that
  criterion un-diagnostic (the LoRA group, on Muon in both arms, moved as much as the treated group).
  Engagement is instead: (a) the runner's WS-log guard requires the `[muon]` banner to name
  "8,320 gate-matrix params in a wd=0.0 group", so a silently inert flag cannot reach a verdict; and
  (b) the per-tensor update NORM of the 13 gates must change relative to the control by more than the
  run-to-run spread of an untreated Muon group measured in the same pair (the LoRA group is the
  built-in control, and it is printed beside them).
- **P3 (the informative failure).** A REGRESSION beyond -0.0001 in either mode means a fixed-norm
  orthogonalised step is wrong for the token-shift lerp specifically -- plausible, because its 8 rows
  are the r/k/v/d/a/g mixing vectors, which have no reason to be mutually orthogonal; Muon would then
  be imposing a structure the parameter's semantics reject. That would also retro-explain why
  `k_scale`/`v_scale` (whose 5 rows are per-head and interchangeable) took to it and this does not.
- **Abort line.** Either mode worse by > 0.0002.

## What it closes
The optimizer COVERAGE axis, completely: with iters 53, 67 and this run, every non-degenerate matrix
in the trunk has a Muon verdict. What remains on AdamW is 1-D (norms, biases, lerp vectors), the
degenerate-reshape `bonus`, and the heads -- all of which Muon's own rule excludes.

## Honest expected value
Low for the BINDING mode. imm already clears its stop criterion by ~0.0005, so an imm-only gain does
not advance the phase-4 goal; ahead is 0.0030 short and is the reason to run this at all. It is
queued because it is ~2 lines on an otherwise-idle GPU, it closes a family, and its ahead half is a
real question with a mechanism behind either answer -- not because a gain is expected.
