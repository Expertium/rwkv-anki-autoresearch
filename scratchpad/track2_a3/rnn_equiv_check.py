"""RNN off-path equivalence check for the A3 GRU edit (2026-07-17).

SrsRWKVRnn CONSTRUCTION is nondeterministic (rkvdag_lerp comes from uninitialized
memory -- torch.empty garbage; harmless in production because weights are always copied
in), so a construction-RNG golden can't work. Instead: load the PRE-EDIT module straight
from git HEAD (extracted to preedit_srs_model_rnn.py), give OLD and NEW the SAME weights
via copy_downcast_ from one fwd model, run one review() from None states, and require
bit-identical outputs. Flag off, d=32 champion env.
"""

import importlib.util
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402
from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG  # noqa: E402
from rwkv.model.srs_model import SrsRWKV  # noqa: E402
from rwkv.model.srs_model_rnn import SrsRWKVRnn  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "preedit_srs_model_rnn", os.path.join(HERE, "preedit_srs_model_rnn.py")
)
old_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old_mod)

torch.manual_seed(0)
fwd = SrsRWKV(DEFAULT_ANKI_RWKV_CONFIG)
fwd.eval()
torch.manual_seed(1)
with torch.no_grad():
    fwd.ahead_linear.weight.normal_(std=0.5)
    fwd.ahead_linear.bias.normal_(std=0.5)
    fwd.w_linear.weight.normal_(std=0.5)
    fwd.w_linear.bias.normal_(std=0.5)
    fwd.p_linear.weight.normal_(std=0.5)

old_rnn = old_mod.SrsRWKVRnn(DEFAULT_ANKI_RWKV_CONFIG)
new_rnn = SrsRWKVRnn(DEFAULT_ANKI_RWKV_CONFIG)
old_rnn.eval()
new_rnn.eval()
old_rnn.copy_downcast_(fwd, torch.float32)
new_rnn.copy_downcast_(fwd, torch.float32)

torch.manual_seed(4)
feats = torch.randn(1, 92)
with torch.inference_mode():
    o = old_rnn.review(feats, None, None, None, None, None)
    n = new_rnn.review(feats, None, None, None, None, None)

assert len(o) == 8 and len(n) == 10, f"arity: old {len(o)}, new {len(n)}"
o_ah, o_w, o_p = o[0], o[1], o[2]
n_ah, n_w, n_p = n[0], n[1], n[4]
assert torch.equal(o_ah, n_ah), "ahead logits mismatch"
assert torch.equal(o_w, n_w), "w mismatch"
assert torch.equal(o_p, n_p), "p logits mismatch"
t = torch.tensor([[123456.0]])
assert torch.equal(
    old_rnn.forgetting_curve(o_w, t), new_rnn.forgetting_curve(n_w, t)
), "curve mismatch"
# states too (slots shifted by 2); state tuples mix tensors and ints
def _eq(a, b, tag):
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and torch.equal(a, b), f"{tag} tensor mismatch"
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), f"{tag} len mismatch"
        for j, (x, y) in enumerate(zip(a, b)):
            _eq(x, y, f"{tag}[{j}]")
    elif isinstance(a, dict):
        assert set(a) == set(b), f"{tag} dict keys mismatch"
        for k in a:
            _eq(a[k], b[k], f"{tag}[{k}]")
    else:
        assert a == b, f"{tag} value mismatch: {a} vs {b}"

for i in range(5):
    _eq(o[3 + i], n[5 + i], f"state{i}")
print("RNN off-path equivalence PASS (old vs new, shared weights, bit-identical)")
