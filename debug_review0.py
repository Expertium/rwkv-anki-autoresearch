"""Dump review-0 intermediates from the Python model for user 107 (zero state)."""
import torch
import numpy as np
from safetensors.numpy import load_file

from rwkv.model.srs_model_rnn import SrsRWKVRnn
from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG

torch.set_num_threads(7)

t = load_file("reference/trace_user_107.safetensors")
fi = torch.tensor(t["feats_imm"][0:1], dtype=torch.float32)  # (1,92)
fp = torch.tensor(t["feats_proc"][0:1], dtype=torch.float32)

model = SrsRWKVRnn(DEFAULT_ANKI_RWKV_CONFIG)
sd = torch.load("pretrain/rwkv/ref_100/rwkv_ref_558.pth", map_location="cpu", weights_only=True)
model.load_state_dict(sd)
model.eval()

def summ(name, t):
    a = t.detach().reshape(-1).numpy()
    print(f"{name:16s} shape {tuple(t.shape)} sum {a.sum():+.6f} norm {np.linalg.norm(a):.6f} "
          f"head [{a[0]:.6f}, {a[1]:.6f}, {a[2]:.6f}]")


# Manually replicate review()'s stream chain (hooks don't fire through .run()).
with torch.inference_mode():
    x = model.features2card(fi)
    summ("features2card", x)
    for k in range(5):
        x, _st = model.rwkv_modules[k].run(x, None)
        summ(f"stream{k}", x)
    global_encoding = x
    xh = model.prehead_norm(global_encoding)
    summ("prehead_norm", xh)
    out_w_logits = model.w_linear(model.head_w(xh).float())
    out_w = torch.nn.functional.softmax(out_w_logits, dim=-1)
    out_ahead_logits = model.ahead_linear(model.head_ahead_logits(xh).float())
    out_p_logits = model.p_linear(model.head_p(xh).float())

summ("out_p_logits", out_p_logits)
summ("out_w", out_w)
summ("out_ahead_logits", out_ahead_logits)
p_again = torch.softmax(out_p_logits, dim=-1)[0, 0].item()
print(f"imm = 1 - p_again = {1 - p_again:.6f}")
