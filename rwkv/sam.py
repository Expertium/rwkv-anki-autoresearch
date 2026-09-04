"""Sharpness-Aware Minimization for the master/child training loop (RWKV_SAM_RHO, 2026-09-04).

Foret, Kleiner, Mobahi & Neyshabur 2021 (ICLR; arXiv 2010.01412): minimise max_{||e|| <= rho} L(w + e),
with the inner max taken at one gradient-ascent step e = rho * g / ||g||. The optimizer then consumes
the gradient evaluated at the perturbed point. Muon / AdamW are untouched -- SAM only changes WHICH
gradient they see. LookSAM-style thinning (Liu et al. 2022) via RWKV_SAM_EVERY=k: the SAM pass runs on
every k-th step, the others use the plain gradient.

Why it is a candidate here: Muon's train-loss edge decays to zero while its held-out edge holds
(iters 29/53), i.e. this trunk's wins are generalisation wins; and the 2026-09-04 sharpness probe
measured L(w + 0.05 g/||g||) - L(w) at median +0.023 over 12 real chunks on realcyc -- a sharp minimum.

The loop this hooks into (train_rwkv.py): forward + backward on the bf16 CHILD, grads ACCUMULATED into
the fp32 MASTER (transfer_child_grad_to_master), clip on master, optimizer.step() on master, then the
child is re-downcast from master at the next step. The SAM pass sits between the transfer and the clip:
  1. g := master grads (clone); e := rho * g / ||g||_2 (global norm over all params)
  2. master += e; child <- downcast(master); zero child AND master grads
  3. restore the RNG state saved before the first forward (same dropout masks), forward + backward on
     the SAME batch, transfer -> master grads = grad at w + e
  4. master -= e   (weights exactly restored: the add/sub pair is bit-exact in fp32 for these magnitudes
     -- asserted once at first use)
Default RWKV_SAM_RHO=0 => nothing is called, byte-identical.
"""
import os

import torch

SAM_RHO = float(os.environ.get("RWKV_SAM_RHO", "0") or 0)
SAM_EVERY = int(os.environ.get("RWKV_SAM_EVERY", "1") or 1)
SAM_ON = SAM_RHO > 0.0
_checked = {"restore": False}
if SAM_ON:
    print(f"[sam] Sharpness-Aware Minimization ON: rho={SAM_RHO} every={SAM_EVERY} step(s)")


def sam_active(step: int) -> bool:
    return SAM_ON and (step % SAM_EVERY == 0)


def save_rng():
    st = {"cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state()
    return st


def restore_rng(st):
    torch.set_rng_state(st["cpu"])
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state(st["cuda"])


@torch.no_grad()
def _perturb(master, sign, e_list, params):
    if sign > 0:
        torch._foreach_add_(params, e_list)
    else:
        torch._foreach_sub_(params, e_list)


def sam_second_pass(master, child, dtype, rng_state, forward_backward, transfer):
    """Replace master's accumulated grads with the gradient at w + rho*g/||g||.
    forward_backward(): runs the child forward on the SAME batch and calls .backward() on its loss.
    transfer(): accumulates child grads into master grads (transfer_child_grad_to_master)."""
    params = [p for p in master.parameters() if p.grad is not None]
    if not params:
        return
    grads = [p.grad.detach().clone() for p in params]
    gnorm = torch.sqrt(sum((g.float() ** 2).sum() for g in grads))
    scale = SAM_RHO / (gnorm + 1e-12)
    e_list = [g * scale for g in grads]
    # Restore from a SNAPSHOT, not by subtracting e: (w + e) - e is NOT bit-exact in fp32 (the unit
    # test caught exactly that at rho=0.05), and a drifting master would make every SAM step a tiny
    # uncontrolled weight change. A copy back is exact by construction; 564k floats is nothing.
    snapshot = [p.detach().clone() for p in params]
    _perturb(master, +1, e_list, params)
    with torch.no_grad():
        child.copy_downcast_(master, dtype=dtype)
    for p in master.parameters():
        p.grad = None
    child.zero_grad(set_to_none=True)
    restore_rng(rng_state)
    forward_backward()
    transfer()
    with torch.no_grad():
        torch._foreach_copy_(params, snapshot)
    if not _checked["restore"]:
        with torch.no_grad():
            worst = max(float((p.detach() - s).abs().max()) for p, s in zip(params, snapshot))
        assert worst == 0.0, f"[sam] weight restore is not bit-exact (max |dw| = {worst:.3e})"
        _checked["restore"] = True
        print("[sam] first pass: weights restored bit-exactly after the ascent step")
