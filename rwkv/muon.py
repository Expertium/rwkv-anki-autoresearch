"""Hybrid Muon+AdamW optimizer (research iter 29, 2026-07-21).

Muon (Jordan et al., the modded-nanogpt speedrun's backbone optimizer): momentum-SGD
whose update is orthogonalized per 2D weight matrix via a quintic Newton-Schulz
iteration, with an aspect-ratio step scale. Non-matrix params (biases, norms, LoRAs,
scalars) stay on AdamW with numerics identical to torch.optim.AdamW (delegated to the
functional kernel).

Design constraints honored:
- ONE torch.optim.Optimizer subclass: param_groups carry the usual lr/weight_decay keys
  plus `use_muon`, so train_rwkv's LR schedulers (LinearLR/ConstantLR/Cosine multiply
  each group's own base lr), the resume clobber-restore logic, and state_dict save/load
  all work unchanged.
- Weight decay on Muon groups is applied at the SAME absolute per-step rate as the
  champion's AdamW would have applied it (decoupled p *= 1 - lr_adamw_equiv * wd):
  each Muon group carries `wd_lr_scale` = adamw_peak / muon_peak so the schedule-scaled
  Muon lr maps back to the AdamW-equivalent rate. Regularization stays comparable to
  the champion; the optimizer geometry is the only change.
- Default OFF at the call site (RWKV_MUON unset -> plain torch.optim.AdamW, byte-identical).
"""

import os

import torch
from torch.optim.adamw import adamw as _functional_adamw

# RWKV_MUON_BATCHED=1 -> orthogonalize all same-shaped matrices in one batched Newton-Schulz
# and drive momentum with torch._foreach_*. Default OFF, i.e. byte-identical to iter 29-33.
_MUON_BATCHED = os.environ.get("RWKV_MUON_BATCHED", "0") == "1"


@torch.no_grad()
def zeropower_via_newtonschulz5(G, steps: int = 5):
    """Quintic Newton-Schulz orthogonalization (modded-nanogpt reference constants)."""
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.to(torch.bfloat16)
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.mT
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X.to(G.dtype)


@torch.no_grad()
def zeropower_via_newtonschulz5_batched(G, steps: int = 5):
    """Same iteration, over a STACK of equally-shaped matrices: G is (B, M, N).

    Item-for-item mathematically identical to calling the 2D version B times -- `@` on 3D
    tensors is bmm, which is independent per batch element, and the normalizer is taken
    per item. What changes is that B matrices are orthogonalized in ONE dispatch instead of B.

    Motivation (2026-07-27 profile, optimization/TRAINING_SPEED.md): the per-parameter loop
    issued 2,658 `aten::mm` per step costing 92 ms of CPU dispatch to do 21.6 ms of GPU work,
    on a step that is dispatch-bound overall.

    ⚠ NOT bit-exact vs the 2D path: cuBLAS may choose a different algorithm/reduction order
    for bmm than for mm. Equivalent in exact arithmetic, not in float.
    """
    assert G.ndim == 3
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.to(torch.bfloat16)
    transposed = G.size(1) > G.size(2)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(1, 2), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X.to(G.dtype)


class MuonAdamW(torch.optim.Optimizer):
    def __init__(self, param_groups, betas=(0.9, 0.999), eps=1e-8,
                 muon_momentum=0.95, ns_steps=5, cautious_wd=False):
        defaults = dict(lr=1e-3, weight_decay=0.0, betas=betas, eps=eps,
                        use_muon=False, wd_lr_scale=1.0,
                        muon_momentum=muon_momentum, ns_steps=ns_steps,
                        cautious_wd=cautious_wd)
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                lr = group["lr"]
                wd = group["weight_decay"]
                wd_eff = lr * group["wd_lr_scale"] * wd  # AdamW-equivalent decay rate
                momentum = group["muon_momentum"]
                ns_steps = group["ns_steps"]
                live = [p for p in group["params"] if p.grad is not None]
                # ---- batched path (RWKV_MUON_BATCHED=1): one Newton-Schulz per SHAPE ----
                # The loop below is unchanged; it just consumes a precomputed O per param.
                # See zeropower_via_newtonschulz5_batched for why this is not bit-exact.
                precomputed = {}
                if _MUON_BATCHED and len(live) > 1:
                    by_shape = {}
                    for p in live:
                        g2d = p.grad.reshape(p.grad.size(0), -1)
                        st = self.state[p]
                        if "momentum_buffer" not in st:
                            st["momentum_buffer"] = torch.zeros_like(g2d)
                        by_shape.setdefault(tuple(g2d.shape), []).append((p, g2d, st))
                    for shp, items in by_shape.items():
                        bufs = [st["momentum_buffer"] for (_, _, st) in items]
                        g2ds = [g2d for (_, g2d, _) in items]
                        # momentum + nesterov for the whole shape group in 3 dispatches
                        torch._foreach_mul_(bufs, momentum)
                        torch._foreach_add_(bufs, g2ds)
                        upds = torch._foreach_add(g2ds, bufs, alpha=momentum)
                        if len(items) == 1:
                            Os = [zeropower_via_newtonschulz5(upds[0], steps=ns_steps)]
                        else:
                            stacked = torch.stack(upds, dim=0)
                            Ob = zeropower_via_newtonschulz5_batched(stacked, steps=ns_steps)
                            Os = list(Ob.unbind(0))
                        for (p, _, _), O in zip(items, Os):
                            precomputed[p] = O

                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad
                    assert g.ndim >= 2, "use_muon group must hold matrices"
                    g2d = g.reshape(g.size(0), -1)
                    if p in precomputed:
                        O = precomputed[p]  # momentum/nesterov already applied above
                    else:
                        state = self.state[p]
                        if "momentum_buffer" not in state:
                            state["momentum_buffer"] = torch.zeros_like(g2d)
                        buf = state["momentum_buffer"]
                        buf.mul_(momentum).add_(g2d)
                        upd = g2d.add(buf, alpha=momentum)  # nesterov
                        O = zeropower_via_newtonschulz5(upd, steps=ns_steps)
                    O_full = O.reshape(p.shape)
                    if wd_eff != 0.0:
                        if group["cautious_wd"]:
                            # cautious wd (research iter 30; modded-nanogpt #43/50):
                            # decay ONLY coords whose applied step (-lr*scale*O) agrees
                            # with the weight's sign, i.e. w*O < 0 -- never fight a
                            # component the update is already shrinking toward zero.
                            p.mul_(1.0 - wd_eff * (p * O_full < 0).to(p.dtype))
                        else:
                            p.mul_(1.0 - wd_eff)
                    scale = max(1.0, p.size(0) / p.reshape(p.size(0), -1).size(1)) ** 0.5
                    p.add_(O_full, alpha=-lr * scale)
            else:
                params, grads = [], []
                exp_avgs, exp_avg_sqs, state_steps = [], [], []
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    params.append(p)
                    grads.append(p.grad)
                    state = self.state[p]
                    if len(state) == 0 or "exp_avg" not in state:
                        state["step"] = torch.tensor(0.0)
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                    exp_avgs.append(state["exp_avg"])
                    exp_avg_sqs.append(state["exp_avg_sq"])
                    state_steps.append(state["step"])
                if not params:
                    continue
                beta1, beta2 = group["betas"]
                _functional_adamw(
                    params, grads, exp_avgs, exp_avg_sqs, [], state_steps,
                    amsgrad=False, beta1=beta1, beta2=beta2, lr=group["lr"],
                    weight_decay=group["weight_decay"], eps=group["eps"],
                    maximize=False, foreach=None, capturable=False,
                    differentiable=False, fused=None, grad_scale=None, found_inf=None,
                    has_complex=False,
                )
        return loss
