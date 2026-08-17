"""Iter 53's separable diagnostic: did moving the LoRA matrices onto Muon change their SPECTRA?

Iter 53 sets RWKV_MUON_INCLUDE_LORA=1, so ~27,520 params (the rank-4/rank-2 LoRA projections the
A18 width ladder introduced) move from AdamW to Muon. The proposal rests on the 2026-08-16 finding
that **Muon is a REGULARIZER at our budget** -- its train-loss edge decays to nothing over 6,554
steps while its held-out edge holds at +0.0019 -- and on the claim that the LoRA matrices are the
most anisotropic in the model, i.e. exactly where spectral regularization should bite hardest.

Both halves are checkable from checkpoints alone, with no GPU and no eval:

  Q1 (the PREMISE, champion only): are the LoRA matrices actually more anisotropic than the
     matrices Muon already manages? If not, the proposal's motivation is wrong regardless of how
     the run scores.
  Q2 (ENGAGEMENT, matched pair): at the SAME step, does the flag visibly flatten their spectra?
     Muon orthogonalizes the update, so the prediction is higher stable rank and a lower condition
     number on the LoRA group and near-nothing elsewhere.

THE CONTROL IS THE POINT. iter 47's first run compared a step-50 candidate against a FINAL champion
-- 10,885 training steps of difference masquerading as a lever -- and spawned a confident side-claim
that the matched control then reversed. i45_ws_1000.pth is the same recipe at the same step under
the same seed with augmentation off, so it differs from i53_ws_1000.pth in the flag and nothing
else. (iter 45's WS is bit-identical to iter 41's for all 10,935 steps, so this control is exact.)

Reported with MAX as well as median: a median cannot see a blow-up, which is the lesson iter 51 paid
for. Run any time a checkpoint pair exists; it does not wait for the verdict.

    .venv/Scripts/python.exe scratchpad/iter53_muonlora/spectra.py [step]
"""
import re
import sys
from pathlib import Path

import numpy as np
import torch

STEP = sys.argv[1] if len(sys.argv) > 1 else "1000"
CAND = Path(f"scratchpad/iter53_muonlora/i53_ws_{STEP}.pth")
CTRL = Path(f"scratchpad/iter45_kddecay/i45_ws_{STEP}.pth")

# The Muon group rule (train_rwkv.py): a param is EXCLUDED from Muon if its name contains 'lora'
# OR 'scale'. But iter 53's flag moves only the LORA-named ones -- the `scale` tensors stay on AdamW
# in BOTH runs. So they are a free INTERNAL NEGATIVE CONTROL: whatever noise floor they show is the
# floor this measurement cannot see past, measured on tensors of the same kind rather than on the
# 80x80 matrices. Lumping them in with the lever (as the first version of this tool did) dilutes the
# engagement number with tensors that cannot have moved.
EXCLUDE = re.compile(r"lora|scale")     # everything AdamW keeps
LEVER = re.compile(r"lora")             # the subset iter 53 actually moves
INERT = re.compile(r"scale")            # AdamW in both runs -> must show ~0


def load(p):
    sd = torch.load(p, map_location="cpu", weights_only=True)
    return sd.get("model", sd)


def stats(w):
    """Spectral shape of a 2-D weight. stable_rank in [1, min(shape)] -- 1 = rank-one-like."""
    m = w.detach().float()
    if m.ndim != 2 or min(m.shape) < 2:
        return None
    s = torch.linalg.svdvals(m)
    fro2 = float((s ** 2).sum())
    if fro2 <= 0:
        return None
    top = float(s[0])
    q = (s ** 2) / fro2                      # spectral energy distribution
    ent = float(-(q * torch.log(q + 1e-12)).sum())
    return {"stable_rank": fro2 / (top ** 2 + 1e-30),
            "cond": top / float(s[-1] + 1e-12),
            "eff_rank": float(np.exp(ent)),   # perplexity of the spectrum
            "fro": float(np.sqrt(fro2)),
            "n": min(m.shape)}


def collect(sd):
    lora, other = {}, {}
    for k, v in sd.items():
        if not torch.is_tensor(v) or v.ndim != 2 or min(v.shape) < 2:
            continue
        st = stats(v)
        if st is None:
            continue
        (lora if EXCLUDE.search(k) else other)[k] = st
    return lora, other


def show(tag, d, key):
    if not d:
        print(f"  {tag:<28} (none)")
        return
    a = np.array([v[key] for v in d.values()])
    # normalised stable rank: 1.0 = perfectly flat spectrum for that matrix's own shape
    print(f"  {tag:<28} n={len(a):>3}  median={np.median(a):8.4f}  "
          f"min={a.min():8.4f}  max={a.max():8.4f}")


def main():
    if not CAND.exists():
        print(f"candidate checkpoint not written yet: {CAND}")
        return
    print(f"candidate {CAND}\ncontrol   {CTRL}\n")
    cand = load(CAND)
    cl, co = collect(cand)

    print("=== Q1 (PREMISE): are the LoRA matrices more anisotropic than Muon's own? ===")
    print("stable rank = ||W||_F^2 / ||W||_2^2 -- LOWER means more energy in one direction.")
    for key in ("stable_rank", "eff_rank"):
        print(f" [{key}]")
        show("LoRA/scale (AdamW today)", cl, key)
        show("everything else (on Muon)", co, key)
    # normalise by each matrix's own max possible, so shapes are comparable
    cln = np.array([v["stable_rank"] / v["n"] for v in cl.values()])
    con = np.array([v["stable_rank"] / v["n"] for v in co.values()])
    print(f" [stable_rank / min(shape) -- 1.0 = perfectly flat for that shape]")
    print(f"  {'LoRA/scale':<28} median={np.median(cln):.4f}")
    print(f"  {'everything else':<28} median={np.median(con):.4f}")
    print(f"  -> premise holds only if the LoRA row is clearly LOWER.")

    if not CTRL.exists():
        print(f"\ncontrol checkpoint missing ({CTRL}) -- Q2 skipped")
        return
    ctrl = load(CTRL)
    print("\n=== Q2 (ENGAGEMENT): candidate vs the step-matched control ===")
    print("Muon orthogonalises the update, so the prediction is HIGHER stable rank on the LoRA")
    print("group and ~nothing elsewhere. A null here means the flag is inert, whatever the eval says.")
    def _sel(name, which):
        if which == "lever":                       # moved onto Muon by the flag
            return bool(LEVER.search(name))
        if which == "inert":                       # AdamW in BOTH runs -- the noise floor
            return bool(INERT.search(name)) and not LEVER.search(name)
        return not EXCLUDE.search(name)            # on Muon in both runs

    for tag, sel in (("lora_* (THE LEVER)", "lever"),
                     ("*scale* (INERT control)", "inert"),
                     ("everything else (on Muon)", "other")):
        rows = []
        for k in cand:
            if not torch.is_tensor(cand[k]) or k not in ctrl:
                continue
            if not _sel(k, sel):
                continue
            a, b = stats(cand[k]), stats(ctrl[k])
            if a is None or b is None:
                continue
            rows.append((k, a, b))
        if not rows:
            print(f"  {tag:<28} (none)")
            continue
        d_sr = np.array([r[1]["stable_rank"] - r[2]["stable_rank"] for r in rows])
        d_fr = np.array([r[1]["fro"] / max(r[2]["fro"], 1e-12) - 1.0 for r in rows])
        rel = np.array([r[1]["stable_rank"] / max(r[2]["stable_rank"], 1e-12) - 1.0 for r in rows])
        print(f"  {tag:<28} n={len(rows):>3}")
        print(f"      d(stable_rank)  median={np.median(d_sr):+.4f}  "
              f"min={d_sr.min():+.4f}  max={d_sr.max():+.4f}")
        print(f"      rel change      median={100*np.median(rel):+.2f}%  "
              f"max={100*rel.max():+.2f}%")
        print(f"      ||W||_F ratio-1 median={100*np.median(d_fr):+.2f}%  "
              f"max={100*np.abs(d_fr).max():.2f}% (abs)")
        if sel == "lever":
            worst = sorted(rows, key=lambda r: -(r[1]["stable_rank"] - r[2]["stable_rank"]))[:4]
            print("      biggest movers:")
            for k, a, b in worst:
                print(f"        {k[-58:]:<58} {b['stable_rank']:.3f} -> {a['stable_rank']:.3f}")




def random_baseline_report():
    """Q1 done properly: stable rank scales with min(shape), so comparing a rank-4 bottleneck to an
    80x80 matrix by raw stable rank is meaningless -- and dividing by min(shape) OVER-corrects,
    because a random (4,80) Gaussian already sits near 0.67 of its max while a random square one
    sits near 0.25. The only fair reference is a Gaussian OF THE SAME SHAPE, computed here rather
    than quoted from Marchenko-Pastur.

    Reported as stable_rank / E[stable_rank of a same-shape Gaussian]: 1.0 = as spread as random,
    below 1.0 = genuinely concentrated. THIS is the number the proposal's premise needs.
    """
    # ⚠ MUST read the CONTROL, not the candidate. Q1 asks about the CHAMPION's premise ("are the
    # LoRA matrices the most anisotropic in the model?"), and the candidate has already had the
    # lever acting on it -- by step 1000 the flag had raised LoRA stable rank 48%, so measuring the
    # premise there measures the treatment. The first version of this tool loaded CAND and the
    # resulting 0.695 was quoted as a property of the champion. It was not.
    torch.manual_seed(0)
    src = CTRL if CTRL.exists() else CAND
    sd = load(src)
    cache = {}

    def base(shape):
        if shape not in cache:
            vals = []
            for _ in range(16):
                g = torch.randn(*shape)
                s = torch.linalg.svdvals(g)
                vals.append(float((s ** 2).sum() / s[0] ** 2))
            cache[shape] = float(np.mean(vals))
        return cache[shape]

    lo, ot = [], []
    for k, v in sd.items():
        if not torch.is_tensor(v) or v.ndim != 2 or min(v.shape) < 2:
            continue
        st = stats(v)
        if st is None:
            continue
        r = st["stable_rank"] / base(tuple(v.shape))
        (lo if EXCLUDE.search(k) else ot).append(r)
    lo, ot = np.array(lo), np.array(ot)
    print("\n=== Q1 REDONE against a SHAPE-MATCHED random baseline ===")
    print("stable_rank / E[same-shape Gaussian].  1.0 = as spread as random init.")
    print(f"  {'LoRA/scale (AdamW today)':<28} n={lo.size:>3}  median={np.median(lo):.4f}  "
          f"min={lo.min():.4f}  max={lo.max():.4f}")
    print(f"  {'everything else (on Muon)':<28} n={ot.size:>3}  median={np.median(ot):.4f}  "
          f"min={ot.min():.4f}  max={ot.max():.4f}")
    print("  -> the premise 'the LoRA matrices are the most anisotropic in the model' needs the")
    print("     LoRA median to be CLEARLY below the other one. Raw stable rank cannot show this.")


if __name__ == "__main__":
    main()
    random_baseline_report()
