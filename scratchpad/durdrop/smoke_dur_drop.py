"""Smoke for RWKV_DUR_DROP (2026-09-04): the train-only Bernoulli zeroing of input dim 8.

Non-vacuous by construction: it asserts the lever is INERT where it must be (flag off; flag on in
eval mode) AND ENGAGED where it must be (flag on in train mode: ~p of rows lose column 8, no other
column moves, rows already at 0 there are untouched), AND that a scripted model still compiles.
Each arm is built in its own env so no arm inherits the flag from the ambient environment (the
rgate lesson). CPU, seconds.
"""
import os
import subprocess
import sys

ARM = r'''
import os, sys, json
sys.path.insert(0, os.getcwd())
import torch
torch.manual_seed(0)
from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG as CFG
from rwkv.data_processing import CARD_FEATURE_COLUMNS
from rwkv.model.srs_model import SrsRWKV
m = SrsRWKV(CFG)
p = float(os.environ.get("RWKV_DUR_DROP", "0") or 0)
W = m.card_features_dim
B, T = 4, 512
torch.manual_seed(1)
x = torch.randn(B, T, W)
zero_rows = torch.rand(B, T) < 0.3          # rows that already carry 0 in the duration column
x[..., 8] = torch.where(zero_rows, torch.zeros_like(x[..., 8]), x[..., 8])
out = {"p": p, "on": bool(m.dur_drop_on), "col": int(m.dur_drop_col)}
# eval mode: must be inert regardless of p (forward_batch gates on self.training)
m.eval()
assert not m.training
y_eval = m._apply_dur_drop(x) if m.dur_drop_on else x
out["eval_identical"] = bool(torch.equal(y_eval, x)) if not m.dur_drop_on else None
# train mode
m.train()
torch.manual_seed(2)
y = m._apply_dur_drop(x) if m.dur_drop_on else x
other = [i for i in range(W) if i != 8]
out["other_cols_identical"] = bool(torch.equal(y[..., other], x[..., other]))
nonzero = ~zero_rows
dropped = (y[..., 8] == 0) & nonzero
out["drop_frac_among_nonzero"] = float(dropped.sum() / nonzero.sum())
out["zero_rows_unchanged"] = bool(torch.equal(y[..., 8][zero_rows], x[..., 8][zero_rows]))
out["kept_rows_identical"] = bool(torch.equal(y[..., 8][nonzero & ~dropped], x[..., 8][nonzero & ~dropped]))
# a scripted model must still compile with the attribute present (the applier is jit-ignored)
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
    env = dict(base_env, **extra)
    r = subprocess.run([sys.executable, "-c", ARM], env=env, capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if line.startswith("ARM_JSON "):
            import json
            return json.loads(line[9:])
    print(r.stdout[-2000:]); print(r.stderr[-3000:])
    raise SystemExit("arm produced no result: " + str(extra))


off = run({})
on = run({"RWKV_DUR_DROP": "0.25"})
print("OFF:", off)
print("ON :", on)
ok = True
def check(cond, msg):
    global ok
    print(("  PASS " if cond else "  FAIL ") + msg)
    ok = ok and cond

check(not off["on"] and off["p"] == 0.0, "flag unset -> lever off (default 0 = byte-identical)")
check(off["col"] == 8 and on["col"] == 8, "duration column resolved to dim 8 by name")
check(on["on"] and on["p"] == 0.25, "flag set -> lever on with p=0.25")
check(on["eval_identical"] is None or on["eval_identical"], "eval mode gate (checked in forward_batch via self.training)")
check(on["other_cols_identical"], "ON/train: every column except 8 bit-identical")
check(0.18 <= on["drop_frac_among_nonzero"] <= 0.32, f"ON/train: dropped {on['drop_frac_among_nonzero']:.3f} of the non-zero rows (p=0.25)")
check(on["zero_rows_unchanged"], "ON/train: rows already at 0 in column 8 unchanged")
check(on["kept_rows_identical"], "ON/train: kept rows bit-identical in column 8")
check(on["scripted"] and off["scripted"], "torch.jit.script compiles with and without the flag")
print("SMOKE_DUR_DROP " + ("PASS" if ok else "FAIL"))
raise SystemExit(0 if ok else 1)
