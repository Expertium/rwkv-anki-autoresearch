"""End-to-end probe-fix verification on the LIVE mid-WS GRU checkpoint (CPU ONLY --
no GPU co-tenancy with the running WS).

Andrew's challenge (2026-07-24): "You sure the bug was fixed and there are no other
bugs?" The v1 bug: RNNStream returned bare h_prev at skip/query rows, so imm
predictions were blind to the query's features (elapsed time). The smoke test proved
the v2 CODE matches the intended probe semantics on synthetic data; THIS script
proves the TRAINED MODEL's actual predictions condition on query features, end to
end through the heads, on a real training batch:

  A. Zero the ENTIRE feature rows of all query rows -> the imm predictions (4-way
     p-head at query positions) MUST change (any transmission proves the probe path),
     and every output at NON-query positions must stay BIT-IDENTICAL (query rows must
     never leak into committed state -- that would be a new bug).
  B. Scale ONLY the elapsed-time dims (0..7: elapsed days/secs/cumulative/phases) at
     query rows by 1.5 -> imm predictions must move: elapsed time specifically flows.
  C. CONTROL: monkeypatch RNNStream.forward back to v1 semantics (skip rows read the
     committed predecessor state, no probe) and repeat B -> the delta must be EXACTLY
     zero. Proves the test discriminates v1 from v2 (i.e. it would have caught the
     bug), and quantifies how much the fix changes trained predictions.

Feature-dim facts (INPUT_FEATURES.md): dim 23 = the explicit query flag (row
identification); dims 0-7 = elapsed-time family; RWKV_ZERO_FEATURES=22 (card state)
is applied inside forward_batch, untouched here.
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ["RWKV_NO_JIT"] = "1"
os.environ["RWKV_BASELINE_CELL"] = "gru"
os.environ["RWKV_ARCH_MODULE"] = "scratchpad/track2_a9/architecture_d128_cmix1_user3_card2_note1.py"
os.environ["RWKV_GRU_HEAD"] = "2"
os.environ["RWKV_NO_AHEAD_RESIDUAL"] = "1"
os.environ["RWKV_ZERO_FEATURES"] = "22"

sys.path.insert(0, os.getcwd())

import json

import lmdb
import torch

from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG
from rwkv.model.rnn_baseline import RNNStream
from rwkv.model.rwkv_model import time_shift_gather
from rwkv.model.srs_model import SrsRWKV
from rwkv.prepare_batch import get_data, prepare

CKPT = sys.argv[1] if len(sys.argv) > 1 else "scratchpad/baseline_gru/bgruws_7000.pth"

torch.manual_seed(0)

# --- one small real batch from the train db (read-only, no lock: the live run's
# fetch workers keep the same db open; LMDB multi-reader is safe) -----------------
db = lmdb.open("train_db_5k_h1", readonly=True, lock=False)
key = None
with db.begin(write=False) as txn:
    for user in range(1, 200):
        raw = txn.get(f"{user}_batches".encode())
        if raw is None:
            continue
        for b in json.loads(raw):
            if 1500 <= b[2] <= 4000:
                key = (user, b[0], b[1], b[2])
                break
        if key:
            break
    assert key is not None, "no small batch found in users 1..199"
    print(f"batch: user {key[0]}, reviews {key[1]}-{key[2]}, rows {key[3]}")
    sample = get_data(txn, key, device="cpu")
db.close()

batch = prepare([sample], seed=1234)

# --- model at the live WS checkpoint ---------------------------------------------
model = SrsRWKV(anki_rwkv_config=DEFAULT_ANKI_RWKV_CONFIG)
model.load_state_dict(torch.load(CKPT, weights_only=True, map_location="cpu"))
model.float().eval()
print(f"loaded {CKPT}")


def fwd(start):
    with torch.no_grad():
        return model.forward_batch(
            start, batch.sub_gather, batch.sub_gather_lens,
            batch.time_shift_selects, batch.skips, batch.num_data,
        )


def p_binary(outs):
    # imm prediction = 1 - P(Again) from the 4-way rating head (same as _get_loss)
    return 1.0 - torch.softmax(outs[3].float(), dim=-1)[..., 0]


start0 = batch.start.float()          # stored bf16; the CPU model runs fp32
labels = batch.labels.float()
is_query = labels[..., 6] > 0.5
has_label = labels[..., 4] > 0.5
q_lab = is_query & has_label          # rows the imm loss actually scores
qrows = start0[:, 23] > 0.5           # query rows in the flat feature matrix
assert int(qrows.sum()) == int(is_query.sum()), (
    f"query-flag rows {int(qrows.sum())} != grid is_query {int(is_query.sum())}")
print(f"rows {start0.shape[0]}, query rows {int(qrows.sum())}, "
      f"labeled-query {int(q_lab.sum())}, grid {tuple(labels.shape[:2])}")

base = fwd(start0)
pb0 = p_binary(base)

# --- A: zero the whole query rows --------------------------------------------------
sA = start0.clone()
sA[qrows] = 0.0
outA = fwd(sA)
dq_A = (p_binary(outA) - pb0)[q_lab].abs()
real_leak = max(
    (o1 - o0)[~is_query].abs().max().item()
    for o0, o1 in zip(base, outA) if o0.dim() >= 2 and o0.shape[:2] == labels.shape[:2]
)
print(f"A. zero query rows: imm |dP| max {dq_A.max():.4f} mean {dq_A.mean():.4f} "
      f"(must be >0);  non-query outputs max |d| = {real_leak:.2e} (must be 0)")
assert dq_A.max() > 1e-3, "A FAILED: imm blind to query features (v1-style bug)"
assert real_leak == 0.0, "A FAILED: query rows leak into committed/real outputs"

# --- B: scale elapsed-time dims only -----------------------------------------------
sB = start0.clone()
sB[qrows, 0:8] *= 1.5
outB = fwd(sB)
dq_B = (p_binary(outB) - pb0)[q_lab].abs()
print(f"B. elapsed dims x1.5 on query rows: imm |dP| max {dq_B.max():.4f} "
      f"mean {dq_B.mean():.4f} (must be >0)")
assert dq_B.max() > 1e-4, "B FAILED: imm insensitive to elapsed time"

# --- C: v1-semantics control (bare h_prev at skips, no probe) ----------------------
_v2_forward = RNNStream.forward


def _v1_forward(self, in_BTC, time_shift_select_BT, skip_BT):
    if not getattr(self, "_flat_ok", False):
        for layer in self.rnn:
            layer.flatten_parameters()
        self._flat_ok = True
    in_dtype = in_BTC.dtype
    keep_BT = ~skip_BT
    order_i32 = torch.argsort(skip_BT.int(), dim=1, stable=True).to(torch.int32)
    cum_BT = keep_BT.to(torch.int32).cumsum(dim=1)
    take_i32 = (cum_BT - 1).clamp(min=0).to(torch.int32)
    alive = (cum_BT > 0).unsqueeze(-1)
    x = in_BTC.float()
    for i, layer in enumerate(self.rnn):
        x_comp = time_shift_gather(x, order_i32)
        out_comp = self._run_layer_windowed(layer, x_comp)
        x = time_shift_gather(out_comp, take_i32) * alive.to(out_comp.dtype)
        if i + 1 < len(self.rnn) and self.dropout_p > 0:
            x = torch.nn.functional.dropout(x, self.dropout_p, self.training)
    return self.proj(x).to(in_dtype)


RNNStream.forward = _v1_forward
try:
    v1_base = fwd(start0)
    v1_B = fwd(sB)
    dq_v1 = (p_binary(v1_B) - p_binary(v1_base))[q_lab].abs()
    d_fix = (p_binary(v1_base) - pb0)[q_lab].abs()
    print(f"C. v1 semantics: same elapsed perturbation -> imm |dP| max {dq_v1.max():.2e} "
          f"(must be EXACTLY 0);  v1-vs-v2 trained imm gap max {d_fix.max():.4f} "
          f"mean {d_fix.mean():.4f}")
    assert dq_v1.max() == 0.0, "C FAILED: v1 control shows sensitivity?!"
finally:
    RNNStream.forward = _v2_forward

print("PROBE_CHECK_ALL_PASS")
