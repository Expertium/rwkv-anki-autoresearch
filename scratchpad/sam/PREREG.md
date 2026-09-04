# sam -- pre-registration (written 2026-09-04 21:30, before launch; ADOPTED slot after ordcut)

**Lever:** `RWKV_SAM_RHO=0.05`, `RWKV_SAM_EVERY=1` -- Sharpness-Aware Minimization (Foret, Kleiner,
Mobahi & Neyshabur 2021, ICLR, arXiv 2010.01412) on the DECAY phase only, warm-started from the base
run's WS-final checkpoint. Each step: gradient at w, ascent e = rho*g/||g|| (global norm), gradient at
w+e with the SAME batch and the SAME dropout masks (RNG restored), weights restored bit-exactly (asserted
on first use), then Muon/AdamW step on the perturbed gradient. The optimizer is untouched.
`rwkv/sam.py` + two hook lines in `train_rwkv.py`; default 0 = the SAM code path is never entered.
**Control:** the base's own decay from the same WS-final (single-variable: the flag is the only diff;
`mk_sam.py` slices the base runner's decay+eval phases and drops its WS phase). The base is chosen
mechanically at fire time (`auto_control.py`): ordcut if it passed its curve-side gate, else the
control ordcut ran against (durdrop if it passed the both-modes gate, else realcyc).
**Gate:** BOTH-modes rule (a trunk-wide optimizer-side change): ahead AND imm raw >= +0.0001 at
p < 1e-4; size 0/2499.

**Why it is worth a run (measured):** the 2026-09-04 sharpness probe on realcyc, 12 real training
chunks, CPU, fp32: L(w + 0.05 g/||g||) - L(w) = median +0.023, min +0.0097 (1.4% of L0), max +0.095.
The minimum is sharp at SAM's scale on every chunk. And this trunk's wins are generalisation wins
(Muon's train-loss edge decays to zero while its held-out edge holds, iters 29/53).

## Predictions
- **P1 (direction).** Both modes improve: band ahead **+0.0000 .. +0.0003**, imm **+0.0000 .. +0.0003**.
  A null (both inside the +/-7.5e-5 floor) is the pre-registered counter-hypothesis: Muon already
  buys the flatness that matters, and the regularizer reading of Muon is about SPECTRUM, not
  sharpness -- worth knowing, and it closes the family at 0/1 with a mechanism.
- **P2 (engagement).** Re-run `sam_probe.py` on the SAM checkpoint: the median gap at rho 0.05 must
  FALL below realcyc's +0.023 (SAM found a flatter point). If it does not, the dose was too small
  and the verdict is uninterpretable; then rho 0.1 is the one retry.
- **P3 (cost).** Decay wall-clock ~2x the base's (the second forward+backward), i.e. ~6.5 h + eval.
- **Abort line.** Either mode worse by > 0.0002 (rho too large: halve once with ASAM-style scaling,
  then close).

## What it is not
Not PolarExpress/NorMuon (those change the UPDATE RULE; SAM changes the OBJECTIVE); not a noise
regularizer (the perturbation is deterministic and adversarial, the kind the tuner turned DOWN was
dropout); not tried in any form here before.
