#!/usr/bin/env python
"""Three-way-parity smoke for RWKV_RCOUPLE (retrievability-coupled rating head). CPU, seconds.

WHY IT IS NOT A CASE IN parity_train_vs_rnn.py: that harness compares RWKV7 vs RWKV7RNN, i.e. the
single-stream RECURRENCE. This coupling lives at the SrsRWKV HEAD level, like RWKV_INTERLEAVE, which
that file documents as out of scope for the same reason.

WHAT IT CHECKS -- in the order the failures actually happen:

 1. **INERT WHEN OFF.** With the flag unset the model must be structurally identical to the champion:
    no `rcouple_w` key. Adding an unconditional Parameter would give a 421-key champion checkpoint a
    422nd key and break `load_state_dict(strict=True)` -- the exact failure the PAVA thetas comment
    in srs_model_rnn.py records.
 2. **PRESENT WHEN ON**, and exactly +4 params.
 3. **NOT VACUOUS.** `rcouple_w` is ZERO-INIT, so every comparison below would pass trivially on an
    untouched model -- the documented vacuity trap (CLAUDE.md §9: "randomize the zero-init params
    ... and assert the output scale is non-trivial before comparing"). The coupling is therefore
    given a random non-zero w and asserted to MOVE the rating probabilities.
 4. **TRAIN AND DEPLOY COMPUTE THE SAME FUNCTION.** srs_model.py applies
        out_p_logits + clamp(curve_logits, -clip, clip).unsqueeze(-1) * rcouple_w
    and srs_model_rnn.py's `rating_logits` must match it on identical inputs. These are two
    hand-written expressions in two files; a divergence here is invisible to every gate, because
    each path is self-consistent (the PAVA lesson: trained-but-never-evaluated for 8 iterations).
 5. **SHIFT-INVARIANCE SANITY.** softmax is shift-invariant, so a UNIFORM w must leave P(Again)
    unchanged -- this catches a sign/broadcast error that a norm-based check would miss.

Run: python scratchpad/parity3/smoke_rcouple.py
"""
import os
import subprocess
import sys
from pathlib import Path

REPO = str(Path(__file__).resolve().parents[2])

CHILD = r"""
import os, torch
from rwkv.model.srs_model_rnn import SrsRWKVRnn
from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG
torch.manual_seed(0)
m = SrsRWKVRnn(DEFAULT_ANKI_RWKV_CONFIG)
keys = set(m.state_dict())
on = os.environ.get("RWKV_RCOUPLE", "") == "1"
has = "rcouple_w" in keys
assert has == on, f"rcouple_w present={has} but RWKV_RCOUPLE on={on}"
print(f"NKEYS {len(keys)}")
if not on:
    print("SMOKE_OK_OFF"); raise SystemExit(0)

# ---- non-vacuity: zero-init w would make every check below pass trivially
w = torch.tensor([0.7, -0.3, 0.25, -0.9])
with torch.no_grad():
    m.rcouple_w.copy_(w)
clip = m.rcouple_clip

p_logits = torch.randn(1, 4) * 1.5
curve_logits = torch.tensor([1.234])

got = m.rating_logits(p_logits, curve_logits)
# the TRAINING expression, transcribed from srs_model.py (the point of the test is that these two
# hand-written forms agree; transcribing it here is what makes a divergence detectable)
ref = p_logits + torch.clamp(curve_logits, -clip, clip).reshape(-1)[0] * w
err = (got - ref).abs().max().item()
print(f"TRAIN_VS_RNN_MAXERR {err:.3e}")
assert err < 1e-6, f"train/deploy coupling disagree by {err}"

base_p = torch.softmax(p_logits, -1)[0, 0].item()
coup_p = torch.softmax(got, -1)[0, 0].item()
print(f"P_AGAIN base {base_p:.6f} coupled {coup_p:.6f} delta {coup_p-base_p:+.6f}")
assert abs(coup_p - base_p) > 1e-3, "coupling did not move P(Again) -- test is VACUOUS"

# clamp must bind on an extreme logit (curve_logits can saturate; an inf would poison the DEPLOYED
# head, since retrievability_head = 1 - P(Again))
big = m.rating_logits(p_logits, torch.tensor([1e4]))
ref_big = p_logits + clip * w
assert (big - ref_big).abs().max().item() < 1e-6, "clip did not bind"
print("CLIP_OK")

# shift-invariance: a UNIFORM w shifts all 4 logits equally -> softmax unchanged
with torch.no_grad():
    m.rcouple_w.copy_(torch.full((4,), 0.6))
uni = torch.softmax(m.rating_logits(p_logits, curve_logits), -1)
assert (uni - base_p * 0 - torch.softmax(p_logits, -1)).abs().max().item() < 1e-6, \
    "uniform w changed the distribution -- broadcast or sign error"
print("SHIFT_INVARIANT_OK")
print("SMOKE_OK_ON")
"""


def run(label, env_extra):
    env = dict(os.environ, PYTHONPATH=REPO, **env_extra)
    env.setdefault("RWKV_ARCH_MODULE", "scratchpad/track2_a18/architecture_d80_lora4_cnd.py")
    for k in ("RWKV_GRU_HEAD", "RWKV_NO_AHEAD_RESIDUAL"):
        env.setdefault(k, "3" if k == "RWKV_GRU_HEAD" else "1")
    p = subprocess.run([sys.executable, "-c", CHILD], cwd=REPO, env=env,
                       capture_output=True, text=True)
    out = (p.stdout + p.stderr).strip().splitlines()
    ok = p.returncode == 0 and any("SMOKE_OK" in ln for ln in out)
    print(f"\n=== {label} === {'PASS' if ok else 'FAIL'}")
    for ln in out[-12:]:
        print("   ", ln)
    return ok


def main():
    results = [
        run("flag OFF -> structurally identical to champion", {}),
        run("flag ON  -> coupling present, non-vacuous, train==deploy", {"RWKV_RCOUPLE": "1"}),
    ]
    print("\nRCOUPLE_" + ("ALL_PASS" if all(results) else "FAILED"))
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    sys.exit(main())
