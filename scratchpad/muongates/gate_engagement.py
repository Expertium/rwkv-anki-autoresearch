"""P2 engagement probe for muongates -- built to iter 67's lesson.

iter 67's criterion (cumulative-update anisotropy must fall) was UN-DIAGNOSTIC: the LoRA group, which
is on Muon in both arms and therefore cannot have changed optimizer, moved as much as the treated
group. The fix is not a better threshold, it is a CONTROL: measure the treated group's change
relative to groups that CANNOT have changed, in the same checkpoint pair.

Measures, per group, the total displacement ||W_final - W_init||_F, then the candidate/control RATIO
per group:
  rkvdag_lerp  = TREATED    (AdamW -> Muon in the candidate)
  lora         = CONTROL A  (on Muon in BOTH arms)
  scale        = CONTROL B  (on AdamW in BOTH arms; muonscale was rejected and is not carried)
  other 2-D    = CONTROL C  (on Muon in BOTH arms, the bulk of the trunk)
ENGAGED iff the treated group's ratio sits OUTSIDE the range spanned by the control groups' ratios --
i.e. the gates moved differently from everything whose optimizer did not change. A treated ratio
inside the controls' range means the flag changed nothing measurable and the verdict is
uninterpretable (the runner's [muon] banner guard should have caught an inert flag before this).

Usage: gate_engagement.py [cand_init cand_final ctrl_init ctrl_final]
"""
import sys

import numpy as np
import torch

a = sys.argv[1:] if len(sys.argv) > 1 else [
    "scratchpad/muongates/mg_ws_50.pth", "scratchpad/muongates/mg_d_10935.pth",
    "scratchpad/realcyc/rc_ws_50.pth", "scratchpad/realcyc/rc_d_10935.pth"]
ci, cf, ki, kf = a


def group_of(name):
    if "rkvdag_lerp" in name:
        return "rkvdag_lerp (TREATED)"
    if "bonus" in name:
        return "bonus (excluded)"
    if "scale" in name and "weight" in name:
        return "scale (control B)"
    if "lora" in name and "weight" in name:
        return "lora (control A)"
    if "weight" in name:
        return "other 2-D (control C)"
    return "1-D / scalar"


def disp(init, final):
    A = torch.load(init, map_location="cpu", weights_only=True)
    B = torch.load(final, map_location="cpu", weights_only=True)
    out = {}
    for k in A:
        if k in B and A[k].shape == B[k].shape:
            out[k] = float((B[k].float() - A[k].float()).norm())
    return out


cand, ctrl = disp(ci, cf), disp(ki, kf)
keys = sorted(set(cand) & set(ctrl))
by = {}
for k in keys:
    if ctrl[k] > 1e-9:
        by.setdefault(group_of(k), []).append(cand[k] / ctrl[k])
print(f"candidate {ci} -> {cf}\ncontrol   {ki} -> {kf}\n")
print(f"{'group':<24} {'n':>4}  candidate/control displacement ratio (median / p10 / p90)")
med = {}
for g, rs in sorted(by.items()):
    r = np.array(rs)
    med[g] = float(np.median(r))
    print(f"{g:<24} {len(r):>4}  {np.median(r):.4f}  /  {np.percentile(r, 10):.4f}  /  {np.percentile(r, 90):.4f}")
treated = med.get("rkvdag_lerp (TREATED)")
controls = [v for k, v in med.items() if "control" in k]
if treated is None or not controls:
    print("\nINCONCLUSIVE: a required group is missing from the checkpoints")
    raise SystemExit(2)
lo, hi = min(controls), max(controls)
engaged = not (lo <= treated <= hi)
print(f"\ntreated median {treated:.4f}; controls span [{lo:.4f}, {hi:.4f}]")
print(f"VERDICT: {'ENGAGED' if engaged else 'NOT DISTINGUISHABLE'} -- the treated group's displacement "
      f"ratio is {'outside' if engaged else 'inside'} the range of groups whose optimizer did not change.")
print("NOTE: this measures that the lever CHANGED the trajectory, never that the change was good.")
raise SystemExit(0 if engaged else 3)
