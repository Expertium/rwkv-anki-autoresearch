"""Smoke for RWKV_ORD_LAMBDA (2026-09-04): the ordinal one-cut term on the curve logit.

Arms run in their own env (no ambient RWKV_* inheritance -- the rgate lesson).
  OFF: the model has NO ord params, loads the realcyc champion checkpoint STRICTLY (key set
       unchanged => every existing checkpoint still loads), scripts, ord_on False.
  ON : exactly 2 extra state-dict keys (ord_cut_a, ord_cut_c), scripts; _ord_loss semantics on
       synthetic tensors: target = (rating >= Good), at a=c=0 it equals BCE(z, target) bit-for-bit,
       the cut shifts the logit, gradients reach BOTH a and c, and t < 1 s is clamped.
CPU, seconds.
"""
import json
import os
import subprocess
import sys

ARM = r'''
import os, sys, json
sys.path.insert(0, os.getcwd())
import torch
torch.manual_seed(0)
from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG as CFG
from rwkv.model.srs_model import SrsRWKV
m = SrsRWKV(CFG)
out = {"on": bool(m.ord_on), "lambda": float(m.ord_lambda), "keys": len(m.state_dict()),
       "ord_keys": sorted(k for k in m.state_dict() if k.startswith("ord_cut"))}
if not m.ord_on:
    sd = torch.load("scratchpad/realcyc/rc_d_10935.pth", map_location="cpu", weights_only=True)
    m.load_state_dict(sd, strict=True)
    out["strict_load_realcyc"] = True
else:
    z = torch.randn(3, 7) * 3
    rating = torch.randint(0, 4, (3, 7))          # post-clamp semantics: 0=Again 1=Hard 2=Good 3=Easy
    t = torch.rand(3, 7) * 1e6
    l0 = m._ord_loss(z, rating, t)
    ref = torch.nn.functional.binary_cross_entropy_with_logits(z, (rating >= 2).float(), reduction="none")
    out["a0c0_equals_bce"] = bool(torch.equal(l0, ref))
    with torch.no_grad():
        m.ord_cut_a.fill_(1.5)
    l1 = m._ord_loss(z, rating, t)
    ref1 = torch.nn.functional.binary_cross_entropy_with_logits(z - 1.5, (rating >= 2).float(), reduction="none")
    out["shift_applied"] = bool(torch.allclose(l1, ref1))
    with torch.no_grad():
        m.ord_cut_c.fill_(0.3)
    l2 = m._ord_loss(z, rating, t)
    cut = 1.5 + 0.3 * torch.log1p(torch.clamp(t, min=1.0) / 86400.0)
    ref2 = torch.nn.functional.binary_cross_entropy_with_logits(z - cut, (rating >= 2).float(), reduction="none")
    out["t_slope_applied"] = bool(torch.allclose(l2, ref2))
    l2.sum().backward()
    out["grad_a"] = float(m.ord_cut_a.grad.abs().sum()); out["grad_c"] = float(m.ord_cut_c.grad.abs().sum())
    out["t_clamp"] = bool(torch.allclose(m._ord_loss(z, rating, torch.zeros(3, 7)), m._ord_loss(z, rating, torch.ones(3, 7))))
sm = torch.jit.script(m)
out["scripted"] = True
print("ARM_JSON " + json.dumps(out))
'''

base_env = {k: v for k, v in os.environ.items() if not k.startswith("RWKV_")}
base_env.update({
    "RWKV_ARCH_MODULE": "scratchpad/track2_a18/architecture_d80_lora4_cnd.py",
    "RWKV_INTERLEAVE": "1", "RWKV_GRU_HEAD": "3", "RWKV_PAVA_LAMBDA": "0.2",
    "RWKV_NO_AHEAD_RESIDUAL": "1", "RWKV_STRIP_L0_VLORA": "1",
    "RWKV_STATE_CLAMP_TAU": "300", "RWKV_STATE_CLAMP_WINDOW": "32768",
    "RWKV_STRIP_CMIX": "user_id:0,user_id:1,user_id:2,preset_id:0,preset_id:1,preset_id:2,deck_id:1,deck_id:2,card_id:1",
    "RWKV_ID_FEATURES": "1", "RWKV_REAL_CYCLES": "1", "RWKV_ZERO_FEATURES": "",
    "PYTHONIOENCODING": "utf-8",
})


def run(extra):
    r = subprocess.run([sys.executable, "-c", ARM], env=dict(base_env, **extra), capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if line.startswith("ARM_JSON "):
            return json.loads(line[9:])
    print(r.stdout[-2000:]); print(r.stderr[-3000:])
    raise SystemExit("arm produced no result: " + str(extra))


off = run({})
on = run({"RWKV_ORD_LAMBDA": "0.25"})
print("OFF:", off)
print("ON :", on)
ok = True
def check(cond, msg):
    global ok
    print(("  PASS " if cond else "  FAIL ") + msg)
    ok = ok and cond

check(not off["on"] and off["ord_keys"] == [] and off["strict_load_realcyc"], "OFF: no ord params; realcyc checkpoint loads STRICTLY")
check(on["on"] and on["lambda"] == 0.25 and on["ord_keys"] == ["ord_cut_a", "ord_cut_c"], "ON: exactly the two cut params")
check(on["keys"] == off["keys"] + 2, "ON: state dict = OFF + 2 keys")
check(on["a0c0_equals_bce"], "ON: at a=c=0 the term IS the plain BCE on (rating >= Good)")
check(on["shift_applied"] and on["t_slope_applied"], "ON: the cut shifts the logit; the t-slope applies as log1p(t/1d)")
check(on["grad_a"] > 0 and on["grad_c"] > 0, "ON: gradients reach both a and c")
check(on["t_clamp"], "ON: t is clamped at 1 s")
check(off["scripted"] and on["scripted"], "torch.jit.script compiles with and without the flag")
print("SMOKE_ORD " + ("PASS" if ok else "FAIL"))
raise SystemExit(0 if ok else 1)
