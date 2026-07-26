"""Is the Rust-parity trace SELF-CONTAINED? -- i.e. do the 92-dim features stored in
reference/trace_user_<u>.safetensors, fed back through the Python RNN, reproduce the
py_pred_* frozen in reference/trace_user_<u>.json?

Why this is the first question to ask (2026-07-26). verify_rust.py compares the Rust engine's
predictions against those frozen py_pred values, so the gate is only meaningful if the trace
carries EVERYTHING the prediction depends on. If Python itself cannot reproduce its own stored
numbers from the trace, then no engine can, and a FAIL says nothing about Rust.

Review 0 of a user is the sharpest probe: all five stream states start empty, so there is no
chaining, no entity routing and no stored curve -- just the forward pass on one feature vector.

Run (needs the d=128 arch of the reference model, which architecture.py no longer is):
  RWKV_ARCH_MODULE=scratchpad/architecture_old_d128.py \
  PYTHONPATH=. .venv/Scripts/python.exe scratchpad/parity3/trace_selfcontained.py
"""
import json
import os
import sys

import torch
from safetensors.torch import load_file

sys.path.insert(0, os.getcwd())
from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG  # noqa: E402
from rwkv.model.srs_model_rnn import SrsRWKVRnn  # noqa: E402

CKPT = os.environ.get("RWKV_CHAMP_CKPT", "pretrain/rwkv/ref_100/rwkv_ref_558.pth")
USERS = [107, 136, 156]


def main():
    cfg = DEFAULT_ANKI_RWKV_CONFIG
    d = cfg.modules[0][1]
    print(f"arch: d_model={d.d_model} n_heads={d.n_heads} "
          f"layers={[c.n_layers for _, c in cfg.modules]}")

    torch.set_grad_enabled(False)
    model = SrsRWKVRnn(cfg).float().eval()
    sd = torch.load(CKPT, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"checkpoint {CKPT}: {len(sd)} tensors, "
          f"missing={len(missing)} unexpected={len(unexpected)}")
    if missing or unexpected:
        print(f"  ⚠ missing[:4]={list(missing)[:4]}")
        print(f"  ⚠ unexpected[:4]={list(unexpected)[:4]}")

    worst = 0.0
    for u in USERS:
        t = load_file(f"reference/trace_user_{u}.safetensors")
        meta = json.load(open(f"reference/trace_user_{u}.json"))
        th = int(t["review_th"][0].item())
        feats = t["feats_imm"][0:1].float()

        # review 0: every stream state empty -> pure forward pass
        torch.manual_seed(u)  # the exporter's seed scheme, in case anything samples
        out = model.review(feats, None, None, None, None, None)
        out_p_logits = out[4]
        imm = float(1.0 - torch.softmax(out_p_logits, dim=-1)[0, 0])

        stored = float(meta["py_pred_imm"][str(th)])
        d_ = abs(imm - stored)
        worst = max(worst, d_)
        print(f"user {u} review_th={th}: recomputed {imm:.8f}  stored {stored:.8f}  "
              f"|d|={d_:.3e}")

    print()
    if worst < 1e-5:
        print(f"TRACE_SELF_CONTAINED (worst {worst:.3e}) -- the frozen py_pred IS reproducible "
              "from the trace features, so verify_rust.py is comparing like with like.")
    else:
        print(f"TRACE_NOT_SELF_CONTAINED (worst {worst:.3e}) -- Python cannot reproduce its own "
              "stored predictions from the trace, so NO engine can. The gate is measuring the "
              "artifacts, not the port.")
    sys.exit(0 if worst < 1e-5 else 1)


if __name__ == "__main__":
    main()
