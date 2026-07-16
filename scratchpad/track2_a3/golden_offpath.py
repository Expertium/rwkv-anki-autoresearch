"""Off-path byte-identity goldens for the A3 GRU-P curve-head edit (2026-07-17).

Run BEFORE editing srs_model.py / srs_model_rnn.py (mode=gen) to capture the current
head outputs + state_dicts, then AFTER the edit (mode=check) to prove the flag-off path
is bit-identical. One subprocess per arch (RWKV_ARCH_MODULE is read at import time; the
old-style ScriptModule additionally bakes the first construction's flags per process).

Covers: SrsRWKV construction RNG (state_dict), head_and_out outputs, forgetting_curve +
interp values, and SrsRWKVRnn construction + a single review() step from None states.
"""

import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))

CHILD = r"""
import os, sys, torch
mode, out_path = sys.argv[1], sys.argv[2]
from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG
from rwkv.model.srs_model import SrsRWKV
from rwkv.model.srs_model_rnn import SrsRWKVRnn

d = DEFAULT_ANKI_RWKV_CONFIG.d_model
torch.manual_seed(0)
model = SrsRWKV(DEFAULT_ANKI_RWKV_CONFIG)
model.eval()
# zero-init heads block observability -- randomize deterministically (smoke-lesson rule)
torch.manual_seed(1)
with torch.no_grad():
    model.ahead_linear.weight.normal_(std=0.5)
    model.ahead_linear.bias.normal_(std=0.5)
    model.w_linear.weight.normal_(std=0.5)
    model.w_linear.bias.normal_(std=0.5)
    model.p_linear.weight.normal_(std=0.5)

torch.manual_seed(2)
x = torch.randn(3, 5, d)
outs = model.head_and_out(x)
ah, w = outs[0], outs[1]
t = torch.rand(3, 5, 1) * 1e7 + 1.0
curve = model.forgetting_curve(w, t)
resid = model.interp(ah, t)

torch.manual_seed(3)
rnn = SrsRWKVRnn(DEFAULT_ANKI_RWKV_CONFIG)
rnn.eval()
torch.manual_seed(4)
feats = torch.randn(1, 92)
r_outs = rnn.review(feats, None, None, None, None, None)
r_ah, r_w = r_outs[0], r_outs[1]
r_curve = rnn.forgetting_curve(r_w, torch.tensor([[500000.0]]))

blob = {
    "sd": {k: v.clone() for k, v in model.state_dict().items()},
    "head_outs": [o.clone() for o in outs],
    "curve": curve, "resid": resid,
    "rnn_sd": {k: v.clone() for k, v in rnn.state_dict().items()},
    "rnn_ah": r_ah.clone(), "rnn_w": r_w.clone(), "rnn_curve": r_curve.clone(),
}
if mode == "gen":
    torch.save(blob, out_path)
    print(f"golden saved: {out_path} (d={d}, heads={len(outs)})")
else:
    ref = torch.load(out_path, weights_only=False)
    def eq(a, b, name):
        assert torch.equal(a, b), f"MISMATCH {name}"
    assert set(ref["sd"]) == set(blob["sd"]), "fwd state_dict keys changed"
    for k in ref["sd"]:
        eq(ref["sd"][k], blob["sd"][k], f"sd[{k}]")
    # head_and_out may have grown extra (dummy) return slots; the original ones must match
    for i, o in enumerate(ref["head_outs"]):
        eq(o, blob["head_outs"][i], f"head_out[{i}]")
    eq(ref["curve"], blob["curve"], "curve")
    eq(ref["resid"], blob["resid"], "resid")
    # rnn_sd/rnn_* fields are NOT compared: SrsRWKVRnn construction is nondeterministic
    # (rkvdag_lerp = uninitialized torch.empty memory; production always copies weights
    # in). RNN off-path equivalence is proven by rnn_equiv_check.py (shared-weights,
    # old-code-from-git vs new, bit-identical).
    assert set(ref["rnn_sd"]) == set(blob["rnn_sd"]), "rnn state_dict keys changed"
    print(f"byte-identity PASS (d={d}; rnn via rnn_equiv_check.py)")
"""


def run(mode, arch_env, tag):
    env = dict(os.environ)
    env.update({"PYTHONPATH": REPO, "CUDA_VISIBLE_DEVICES": ""})
    env.pop("RWKV_NO_AHEAD_RESIDUAL", None)
    env.pop("RWKV_GRUP_HEAD", None)
    env.update(arch_env)
    out = os.path.join(HERE, f"golden_offpath_{tag}.pt")
    r = subprocess.run([sys.executable, "-c", CHILD, mode, out], env=env, cwd=REPO,
                       capture_output=True, text=True)
    print(f"--- {tag} {mode} (exit {r.returncode}) ---")
    print(r.stdout.strip())
    if r.returncode != 0:
        print(r.stderr[-2500:])
        sys.exit(1)


MODE = sys.argv[1] if len(sys.argv) > 1 else "gen"
run(MODE, {"RWKV_N_HEADS": "2", "RWKV_HEAD_DIM": "16", "RWKV_ZERO_FEATURES": "22"}, "d32")
run(MODE, {"RWKV_ARCH_MODULE": "scratchpad/track2_a1/architecture_d128_cmix1.py"}, "d128")
print("ALL_" + ("SAVED" if MODE == "gen" else "PASS"))
