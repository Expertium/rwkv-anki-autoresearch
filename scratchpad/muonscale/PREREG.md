# muonscale -- pre-registration (written 2026-09-05 19:25, before launch; ADOPTED slot after hord)

**Lever:** `RWKV_MUON_INCLUDE_SCALE=1` -- the 26 `k_scale_linear.weight` / `v_scale_linear.weight`
projections (each (H=5, C=80), 10,400 params = 1.85% of the model), the LAST 2-D weights still on
AdamW after iter 53 moved the LoRA matrices, get their own Muon group at the wd 0.0 they already had.
Same shape as iter 53 (a name-rule exclusion in `get_optimizer` becomes a Muon group). No new
parameters; deploy unchanged.
**Provenance (adopted):** Muon's stated rule is "every 2-D weight except embeddings and the output
head" (Jordan et al. 2024, `KellerJordan/Muon`; Liu et al. 2025, Moonlight, arXiv 2502.16982).
**Control:** the reference at fire time -- hord if it passed its curve-side gate (then this runner
carries the hinge too), else realcyc; with SAM in the decay iff the reference recipe carries it.
`auto_control.py` decides mechanically. Single-variable: the flag is the only diff (both-direction
generator guards; the [muon] banner must name the scale-matrix group).
**Gate:** BOTH-modes rule (an optimizer change reaches the whole trunk): ahead AND imm raw >= +0.0001
at p < 1e-4; size 0/2499.

**The screen (`scale_probe.py`, realcyc rc_ws_50 -> rc_d_10935):** the scale matrices' training
update is as anisotropic as the LoRAs' (sigma_max/||dW||_F median 0.653 vs 0.649; white-noise
reference 0.447) -- the property iter 53's mechanism (Muon as SPECTRAL regulariser, coverage not
descent) acts on -- but they carry only 1.0% of the update energy (LoRAs 42%).

## Predictions
- **P1.** Both modes move in the SAME direction as iter 53 (up), at a magnitude proportional to the
  update energy share: **+0.00000 .. +0.00005 per mode**, i.e. a null at the gate. This is the
  expected outcome and is worth one run only because it formally closes the coverage axis: after it,
  every 2-D weight except the output heads has been tried on Muon.
- **P2 (engagement).** The scale matrices' update anisotropy on the candidate FALLS (Muon's
  orthogonalisation whitens the update): re-run `scale_probe.py` on the candidate's checkpoints;
  median ratio must drop below 0.55. If it does not, the group was not on Muon (the banner guard
  should have caught that) and the verdict is uninterpretable.
- **P3 (the informative failure).** A REGRESSION in either mode beyond -0.0001 means Muon's
  fixed-norm step is wrong for these gates (`k_scale = sigmoid(Linear(x))` bounds the delta-rule
  authority; a normalised step could push the gate around too fast) -- worth knowing for the
  endgame's optimizer choice.
- **Abort line.** Either mode worse by > 0.0002.

## What it closes
The optimizer COVERAGE axis. With iter 53 (LoRAs) and this run, every matrix in the trunk has a
verdict on Muon; the remaining AdamW parameters are 1-D (norms, biases, time-mix vectors) and the
heads, which Muon's own rule excludes.
