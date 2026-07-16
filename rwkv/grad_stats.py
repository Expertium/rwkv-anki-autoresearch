"""Per-parameter gradient/no-op statistics recorder (Andrew's directive, 2026-07-16).

Purpose: rank ablation targets for the track-2 (d=128) shrink loop. Records, per named
parameter tensor, across all training steps:
  - mean |grad|            ("which params receive learning signal")
  - mean |grad * w|        (first-order saliency, SNIP-style: estimated |dLoss| if the
                            param were zeroed -- robust where plain |grad| misleads: at
                            convergence grads -> 0 for important params too)
and, from the final weights, near-no-op indicators (mean/median/max |w|, fraction near 0,
fraction near 1 -- the additive-vs-multiplicative interpretation happens in the report
script per param TYPE, not here, because the no-op reference differs: biases/deltas -> 0,
norm gains -> 1, lerp factors -> either end, sigmoid-gated scales -> their init).

Wiring (train_rwkv.py, env RWKV_GRAD_STATS=<out.json>, default off = zero cost):
  gs = GradStats(path, master_model)   # after master_model exists
  gs.accumulate()                      # right after transfer_child_grad_to_master
                                       # (RAW fp32 grads, BEFORE clip_grad_norm_)
  gs.dump()                            # at training end AND before the prune sys.exit(42)s

Cost: a handful of torch._foreach_* fused calls per step (~700 tensors, <1 ms); no .item()
syncs until dump. NaN-safe: a step whose grad summary is non-finite is masked out
per-parameter (counted separately), so one NaN batch cannot poison the accumulators.
"""

import json

import torch


class GradStats:
    def __init__(self, path, model):
        self.path = path
        # only params that can receive grads; order frozen here
        self.names, self.params = [], []
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.names.append(n)
                self.params.append(p)
        n = len(self.params)
        dev = self.params[0].device if self.params else "cpu"
        self.numel = torch.tensor([p.numel() for p in self.params],
                                  dtype=torch.float64, device=dev)
        self.acc_g = torch.zeros(n, dtype=torch.float64, device=dev)
        self.acc_gw = torch.zeros(n, dtype=torch.float64, device=dev)
        self.count = torch.zeros(n, dtype=torch.float64, device=dev)
        print(f"[grad-stats] recording {n} param tensors -> {path}")

    @torch.no_grad()
    def accumulate(self):
        grads = [p.grad for p in self.params]
        if any(g is None for g in grads):  # first steps of exotic setups; skip whole step
            return
        g_l1 = torch.stack(torch._foreach_norm(grads, 1)).double()
        gw = torch._foreach_mul(grads, [p.data for p in self.params])
        gw_l1 = torch.stack(torch._foreach_norm(gw, 1)).double()
        g_mean = g_l1 / self.numel
        gw_mean = gw_l1 / self.numel
        ok = torch.isfinite(g_mean) & torch.isfinite(gw_mean)
        self.acc_g += torch.where(ok, g_mean, torch.zeros(()).double().to(g_mean.device))
        self.acc_gw += torch.where(ok, gw_mean, torch.zeros(()).double().to(g_mean.device))
        self.count += ok.double()

    @torch.no_grad()
    def dump(self):
        cnt = self.count.clamp(min=1)
        mean_g = (self.acc_g / cnt).cpu()
        mean_gw = (self.acc_gw / cnt).cpu()
        counts = self.count.cpu()
        out = {}
        for i, (name, p) in enumerate(zip(self.names, self.params)):
            w = p.data.detach().float()
            absw = w.abs()
            out[name] = {
                "mean_abs_grad": float(mean_g[i]),
                "mean_abs_grad_x_w": float(mean_gw[i]),
                "steps_counted": int(counts[i]),
                "numel": int(p.numel()),
                "shape": list(p.shape),
                "final_mean_abs_w": float(absw.mean()),
                "final_median_abs_w": float(absw.median()),
                "final_max_abs_w": float(absw.max()),
                "final_frac_absw_lt_0.01": float((absw < 0.01).float().mean()),
                "final_frac_within_0.01_of_1": float(((w - 1.0).abs() < 0.01).float().mean()),
            }
        with open(self.path, "w") as f:
            json.dump(out, f, indent=1)
        print(f"[grad-stats] wrote {self.path} ({len(out)} params)")
