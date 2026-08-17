#!/usr/bin/env python
"""Is RWKV-7's hardcoded decay floor BINDING on our trained champion? CPU, seconds.

Andrew 2026-08-17 asked why the queue has no EXPRESSIVENESS changes. The strongest RWKV-specific
candidate is a constant nobody has ever questioned. `rwkv_model.py:915-916`:

    _d = -0.5 - softplus(-d_lora)          # so _d is in (-inf, -0.5]
    w  = exp(-exp(_d))                     # so w  is in (0, 0.5452]

That `-0.5` is a hard cap on how FAST any channel may decay: w cannot fall below **0.5452**
(half-life ~1.1 steps), and it saturates -- a d_lora output of 6 already gives 0.546 and 20 gives
0.5452, so pushing harder buys nothing. The slow end is unbounded by contrast (d_lora = -6 gives a
462-step half-life and keeps going). The decay spectrum is therefore clamped at one end and free at
the other, by a constant inherited from upstream.

WHETHER THAT MATTERS IS AN EMPIRICAL QUESTION, and it is cheap to answer: if no channel is pressed
against the floor, relaxing it cannot help and the lever is dead before any GPU is spent. If a real
fraction IS saturated, the model is telling us it wants faster decay than the parameterization can
express -- which for spaced repetition is plausible, since a lapse should be able to wipe state.

Read STATICALLY from the checkpoint. `d_lora_mlp` is `B_and_lamb(tanh(A(x)))`, so with tanh bounded
in [-1, 1] the reachable range of d_lora at any input is exactly
    bias +/- sum_j |B[.,j]|
and the bias alone is the value at zero input. That gives both the resting point and the full
input-driven envelope without running a single batch.

ASCII output only.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

CKPT = sys.argv[1] if len(sys.argv) > 1 else "scratchpad/iter45_kddecay/i45_d_10935.pth"
FLOOR = -0.5


def w_of(d):
    return np.exp(-np.exp(np.clip(d, -50, 50)))


def half_life(w):
    w = np.clip(w, 1e-12, 1 - 1e-12)
    return np.log(0.5) / np.log(w)


def main():
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = sd.get("model", sd)
    rows = []
    for k, v in sd.items():
        if "d_lora_mlp.B_and_lamb.bias" in k:
            base = k[: -len("bias")]
            wt = sd.get(base + "weight")
            if wt is None:
                continue
            b = v.float().numpy()
            reach = wt.float().abs().sum(dim=1).numpy()  # tanh in [-1,1] => |B| row sums
            rows.append((k.replace(".d_lora_mlp.B_and_lamb.bias", ""), b, reach))
    if not rows:
        print("no d_lora_mlp biases found -- wrong checkpoint?")
        return 1

    print(f"{len(rows)} decay-LoRA tensors from {CKPT}")
    print(f"floor constant = {FLOOR} => w cannot go below {w_of(FLOOR):.4f} "
          f"(half-life {half_life(w_of(FLOOR)):.2f} steps)\n")
    print(f"  {'stream.block':>28s} {'C':>4s} {'w@rest med':>11s} "
          f"{'w_fastest med':>14s} {'% within 0.05 of floor':>23s}")

    allf = []
    for name, b, reach in rows:
        d_rest = FLOOR - np.log1p(np.exp(-b))            # softplus(-b), stable via log1p
        # the FASTEST decay each channel can reach: d_lora as large as tanh allows
        d_fast = FLOOR - np.log1p(np.exp(-(b + reach)))
        w_rest, w_fast = w_of(d_rest), w_of(d_fast)
        near = float(np.mean(w_fast <= w_of(FLOOR) + 0.05))
        allf.append(near)
        short = name.replace("rwkv_modules.", "s").replace(".blocks.", ".b").replace(".time_mixer", "")
        print(f"  {short:>28s} {len(b):4d} {np.median(w_rest):11.4f} "
              f"{np.median(w_fast):14.4f} {100*near:22.1f}%")

    print(f"\n  ACROSS ALL TENSORS: {100*float(np.mean(allf)):.1f}% of channels can reach within "
          f"0.05 of the floor")
    print("\nREAD IT AS: the fastest-decay column is what the floor actually constrains. If it sits")
    print("well ABOVE 0.545 everywhere, no channel is pressed against the cap and making the")
    print("constant learnable cannot buy anything -- kill the lever here, for free.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
