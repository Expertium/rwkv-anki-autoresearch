"""Do the Python and Rust BUTTON APIs agree? -- gap 6 of TRACK2_PORT_PLAN.md.

The four button intervals are what Anki actually shows the user, and until 2026-07-26 they existed
in neither engine: the PAVA rectifier lived only inside the training loss. Both halves now
implement it, which is exactly the situation CLAUDE.md sec 9's three-way parity rule exists for --
two implementations of one operator, each self-consistent in isolation.

Review 0 with EMPTY state, so the comparison isolates the forward pass + rectifier + solver from
state chaining. Both sides build their own probe rows internally (duration zeroed, grade one-hot
swapped), so this also checks that the two agree on WHAT a probe is, not just on the arithmetic.

Run (with the A18 env, from the repo root):
  PYTHONPATH=. .venv/Scripts/python.exe scratchpad/parity3/buttons_py_vs_rust.py
"""
import os
import subprocess
import sys

import torch
from safetensors.torch import load_file

sys.path.insert(0, os.getcwd())
from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG  # noqa: E402
from rwkv.model.srs_model_rnn import SrsRWKVRnn  # noqa: E402

REF_DIR = os.environ.get("RWKV_REF_DIR", "reference_a18")
CKPT = os.environ.get("RWKV_CHAMP_CKPT", "scratchpad/track2_a18/t2a18d_5586.pth")
EXE = "./rust/rwkv-infer/target/release/rwkv-infer.exe"
USERS = [107, 136, 156]
RETENTIONS = [0.9, 0.8]
# Intervals span seconds..years and the two solvers bisect in float32 from independent forwards,
# so compare RELATIVE. 1% of an interval is far below what a scheduler can express anyway.
REL_TOL = 0.01


def python_intervals(model, user, dr):
    t = load_file(f"{REF_DIR}/trace_user_{user}.safetensors")
    feats = t["feats_imm"][0:1].float()
    heads = model.button_heads(feats, None, None, None, None, None)
    return [float(x) for x in model.button_intervals(heads, desired_retention=dr)]


def rust_intervals(user, dr):
    env = dict(os.environ, RWKV_TRACE_DIR=REF_DIR,
               RWKV_WEIGHTS=f"{REF_DIR}/track2_a18.safetensors")
    p = subprocess.run([EXE, "--buttons", str(user), str(dr)],
                       capture_output=True, text=True, env=env)
    for line in p.stdout.splitlines():
        if "intervals_s" in line:
            return [float(x) for x in line.split("intervals_s")[1].split()]
    raise RuntimeError(f"no intervals in rust output:\n{p.stdout}\n{p.stderr}")


def main():
    torch.set_grad_enabled(False)
    model = SrsRWKVRnn(DEFAULT_ANKI_RWKV_CONFIG).float().eval()
    model.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=True))

    worst = 0.0
    for user in USERS:
        for dr in RETENTIONS:
            py = python_intervals(model, user, dr)
            rs = rust_intervals(user, dr)
            rel = max(abs(a - b) / max(a, b, 1e-9) for a, b in zip(py, rs))
            worst = max(worst, rel)
            print(f"user {user} R={dr}")
            print(f"   py   {['%.1f' % x for x in py]}")
            print(f"   rust {['%.1f' % x for x in rs]}   max rel {rel:.3e}")
            # the property that actually matters to a user, checked on both sides independently
            for tag, iv in (("py", py), ("rust", rs)):
                assert all(iv[k] <= iv[k + 1] * (1 + 1e-4) for k in range(3)), \
                    f"{tag} intervals not ordered Again<=Hard<=Good<=Easy: {iv}"

    print()
    ok = worst < REL_TOL
    print(f"worst relative disagreement: {worst:.3e} (tol {REL_TOL})")
    print("BUTTONS_" + ("MATCH" if ok else "MISMATCH"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
