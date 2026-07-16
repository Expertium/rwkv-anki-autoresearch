"""Smoke test for RWKV_GRU_HEAD=N (track-2 A3, Andrew 2026-07-17): GRU-faithful curve
head. One subprocess per (flag, arch, JIT) config -- old-style ScriptModule bakes the first
construction's flags per process. Constructing under JIT compiles ALL script_methods incl.
the new _get_loss branches, so construction success = the scripted glue type-checks.

Checks (gru ON): param count (dummies + 3 tiny heads replace w_linear + ahead head),
num_curves override, prior-curve init (uniform w, exp(bias) S, d=0.5), zero residual,
monotone-decreasing curve in t, squash bounds, grad flow (randomize the zero-init gru
weights first -- the recurring trap), dead dummies, RNN mirror symmetry via copy_downcast_
+ bit-equal curve values. Flag OFF byte-identity is covered by golden_offpath.py +
rnn_equiv_check.py."""

import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CHILD = r"""
import math, os, torch
from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG
from rwkv.model.srs_model import SrsRWKV

N = int(os.environ.get("RWKV_GRU_HEAD", "0"))
torch.manual_seed(0)
model = SrsRWKV(DEFAULT_ANKI_RWKV_CONFIG)
model.eval()
d = DEFAULT_ANKI_RWKV_CONFIG.d_model
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"PARAMS={n_params}")
if N == 0:
    raise SystemExit(0)  # param-anchor run only

assert model.num_curves == N, f"num_curves not overridden: {model.num_curves}"
assert model.no_ahead_residual, "gru must force no_ahead_residual"

# --- init sanity BEFORE randomizing: zero weights => input-independent prior curve
x = torch.randn(3, 5, d)
outs = model.head_and_out(x)
assert len(outs) == 6, f"head_and_out arity {len(outs)}"
ah, w, wlog, p, s_raw, d_raw = outs
assert w.shape == (3, 5, N) and s_raw.shape == (3, 5, N) and d_raw.shape == (3, 5, N)
assert bool((w - 1.0 / N).abs().max() < 1e-6), "init w not uniform"
s0 = torch.exp(model.gru_s_bias)
assert bool((torch.exp(s_raw) - s0).abs().max() < 1e-2), "init S != exp(bias)"
assert bool((torch.exp(d_raw) - 0.5).abs().max() < 1e-6), "init d != 0.5"
assert ah.dtype == torch.float32 and bool((ah == 0).all()), "residual not zeroed"
assert not ah.requires_grad, "zero residual inside autograd"
print("init prior PASS (uniform w, S=exp(bias), d=0.5, zero residual)")

# --- randomize the zero-init heads (grad-observability trap) + p head
with torch.no_grad():
    for pn in ("gru_w_weight", "gru_s_weight", "gru_d_weight"):
        getattr(model, pn).normal_(std=0.5)
    model.p_linear.weight.normal_(std=0.5)

# --- monotonicity + bounds on REAL head outputs across a huge t range
xg = torch.randn(4, 7, d, requires_grad=True)
ah2, w2, wlog2, p2, s2, d2 = model.head_and_out(xg)
ts = torch.tensor([1.0, 10.0, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9])
prev = None
for t in ts:
    tv = torch.full((4, 7, 1), float(t))
    cv = model.gru_forgetting_curve(w2, s2, d2, tv)
    assert cv.shape == (4, 7), f"curve shape {cv.shape}"
    assert bool((cv > 0).all() and (cv < 1).all()), "curve out of (0,1)"
    assert torch.isfinite(cv).all(), "curve not finite"
    if prev is not None:
        assert bool((cv <= prev + 1e-12).all()), f"curve INCREASED at t={t}"
    prev = cv
print("monotone-decreasing curve PASS (t in [1, 1e9], bounded, finite)")

# --- grad flow: curve loss reaches gru heads + trunk; dummies stay dead
tv = torch.full((4, 7, 1), 86400.0)
loss = model.gru_forgetting_curve(w2, s2, d2, tv).sum() + p2.sum()
loss.backward()
for pn in ("gru_w_weight", "gru_s_weight", "gru_d_weight",
           "gru_w_bias", "gru_s_bias", "gru_d_bias"):
    g = getattr(model, pn).grad
    assert g is not None and bool((g != 0).any()), f"no grad on {pn}"
assert model.w_linear.weight.grad is None, "dummy w_linear got grad"
assert model.ahead_linear.weight.grad is None, "dummy ahead_linear got grad"
hw_grads = [q.grad for q in model.head_w.parameters()]
assert any(g is not None and bool((g != 0).any()) for g in hw_grads), "head_w trunk dead"
assert xg.grad is not None and bool((xg.grad != 0).any()), "no grad reached the trunk input"
print("grad flow PASS (gru heads + head_w + trunk live; dummies dead)")

# --- RNN mirror: name symmetry via copy_downcast_, then bit-equal curve math
from rwkv.model.srs_model_rnn import SrsRWKVRnn
rnn = SrsRWKVRnn(DEFAULT_ANKI_RWKV_CONFIG)
rnn.eval()
assert rnn.gru_on and rnn.num_curves == N and rnn.no_ahead_residual
rnn.copy_downcast_(model, torch.float32)  # KeyError/assert here = state_dict asymmetry
feats = torch.randn(1, 92)
with torch.inference_mode():
    r = rnn.review(feats, None, None, None, None, None)
assert len(r) == 10, f"rnn review arity {len(r)}"
r_ah, r_w, r_s, r_d, r_p = r[0], r[1], r[2], r[3], r[4]
assert r_w.shape == (1, N) and r_s.shape == (1, N) and r_d.shape == (1, N)
assert bool((r_ah == 0).all()), "rnn residual not zeroed"
t1 = torch.tensor([[123456.0]])
c_f = model.gru_forgetting_curve(r_w, r_s, r_d, t1)
c_r = rnn.gru_forgetting_curve(r_w, r_s, r_d, t1)
assert torch.equal(c_f, c_r), "fwd vs rnn gru curve mismatch"
print("RNN mirror PASS (copy symmetry, arity 10, bit-equal curve)")
print("CHILD_OK")
"""


def run(env_extra, label):
    env = dict(os.environ)
    env.update({"PYTHONPATH": REPO, "CUDA_VISIBLE_DEVICES": ""})
    env.pop("RWKV_GRU_HEAD", None)
    env.pop("RWKV_NO_AHEAD_RESIDUAL", None)
    env.pop("RWKV_NO_JIT", None)
    env.update(env_extra)
    r = subprocess.run([sys.executable, "-c", CHILD], env=env, cwd=REPO,
                       capture_output=True, text=True)
    print(f"--- {label} (exit {r.returncode}) ---")
    print(r.stdout.strip())
    if r.returncode != 0:
        print(r.stderr[-3000:])
        sys.exit(1)
    for line in r.stdout.splitlines():
        if line.startswith("PARAMS="):
            return int(line.split("=")[1])
    return None


D32 = {"RWKV_N_HEADS": "2", "RWKV_HEAD_DIM": "16", "RWKV_ZERO_FEATURES": "22"}
D128 = {"RWKV_ARCH_MODULE": "scratchpad/track2_a1/architecture_d128_cmix1.py"}

p32_off = run(D32, "d32 OFF (param anchor)")
p32_on = run({**D32, "RWKV_GRU_HEAD": "2"}, "d32 GRU N=2, JIT")
p128_off = run(D128, "d128/A1 OFF (param anchor)")
p128_on = run({**D128, "RWKV_GRU_HEAD": "2"}, "d128/A1 GRU N=2, JIT")
run({**D128, "RWKV_GRU_HEAD": "2", "RWKV_NO_JIT": "1"}, "d128/A1 GRU N=2, NO_JIT")

# w_head_dim = 4*d; delta = -(head_ahead + ahead_linear + w_linear) + 3*(4d*N+N) + 6 dummies
assert p32_off == 193724, f"d32 baseline params changed: {p32_off}"
assert p128_off == 2320516, f"d128/A1 baseline params changed: {p128_off}"
exp32 = p32_off - (4224 + 8256 + 8256) + 3 * (128 * 2 + 2) + 6
exp128 = p128_off - (66048 + 65664 + 65664) + 3 * (512 * 2 + 2) + 6
assert p32_on == exp32, f"d32 gru params {p32_on} != expected {exp32}"
assert p128_on == exp128, f"d128 gru params {p128_on} != expected {exp128}"
print(f"param accounting PASS: d32 {p32_off}->{p32_on} (exp {exp32}), "
      f"d128 {p128_off}->{p128_on} (exp {exp128}, cut {p128_off - p128_on} = "
      f"{100 * (p128_off - p128_on) / p128_off:.2f}%)")
print("ALL_PASS")
