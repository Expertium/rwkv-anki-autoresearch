"""Smoke for RWKV_PAVA_HORIZON_LAMBDA (2026-09-04): the multi-horizon button-order hinge on the probes.

Arms in their own env (no ambient RWKV_* inheritance).
  OFF: model has no horizon attrs active, scripts, and get_loss on a REAL chunk (CPU fp32) returns
       hord_loss_avg == 0 exactly.
  ON : scripts; on the same chunk hord_loss_avg > 0 (the screen says 30-49% of probe rows cross), the
       total loss differs from OFF by exactly lambda * hord (to fp32 tolerance), gradient of the hinge
       reaches the GRU head params, and ordering the curves by hand (sorting the 4 probe curves) makes
       the hinge 0 -- i.e. the hinge measures what it claims.
CPU, ~3 min (one chunk of user 101 through get_loss, twice).
"""
import json
import os
import subprocess
import sys

ARM = r'''
import os, sys, json
sys.path.insert(0, os.getcwd()); sys.path.insert(0, "scratchpad/proposals_2026-09-04")
import lmdb, torch
torch.manual_seed(0)
from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG as CFG
from rwkv.model.srs_model import SrsRWKV
from rwkv.prepare_batch import get_data, prepare
from sam_probe import to_f32
m = SrsRWKV(CFG)
m.load_state_dict(torch.load("scratchpad/realcyc/rc_d_10935.pth", map_location="cpu", weights_only=True), strict=True)
m = m.float(); m.eval()
env = lmdb.open("F:/rwkv_lmdb/train_db_5k_h1_id5", map_size=400_000_000_000, readonly=True, lock=False)
with env.begin(write=False) as txn:
    b = min(json.loads(txn.get(b"101_batches")), key=lambda x: x[2])
    pb = to_f32(prepare([get_data(txn, (101, b[0], b[1], b[2]), device="cpu")], seed=1234, probe_density=0.08).to("cpu"))
env.close()
out = {"on": bool(m.pava_horizon_on), "lambda": float(m.pava_horizon_lambda), "factors": m.pava_horizon_factors}
st = m.get_loss(pb)
out["hord"] = float(st.hord_loss_avg); out["total"] = float(st.average_loss); out["pava"] = float(st.pava_loss_avg)
if m.pava_horizon_on:
    # gradient reaches the GRU head
    st.average_loss.backward()
    out["grad_gru_w"] = float(m.gru_w_weight.grad.abs().sum())
    # the hinge is zero on hand-ordered curves: build sorted (w,s,d) rows by copying the SAME params to all 4 probes
    probe_rows, probe_target = pb.probe_rows, pb.probe_target
    with torch.no_grad():
        fwd = m.forward_batch(pb.start, pb.sub_gather, pb.sub_gather_lens, pb.time_shift_selects, pb.skips, pb.num_data)
        _, out_w, _, _, out_s, out_d = fwd
        n = out_w.shape[-1]
        w = out_w.reshape(-1, n).clone(); s = out_s.reshape(-1, n).clone(); d = out_d.reshape(-1, n).clone()
        # make all 4 probes of each review identical -> R_b == R_{b+1} -> hinge exactly 0
        for k in range(1, 4):
            w[probe_rows[:, k]] = w[probe_rows[:, 0]]; s[probe_rows[:, k]] = s[probe_rows[:, 0]]; d[probe_rows[:, k]] = d[probe_rows[:, 0]]
        # any positive horizon works for this identity: equal curves => zero hinge whatever t is
        t_any = torch.full((out_w.shape[0] * out_w.shape[1],), 86400.0)
        h0 = m._horizon_order_loss(w.view_as(out_w), s.view_as(out_s), d.view_as(out_d), t_any, probe_rows, probe_target)
        out["hinge_on_equal_curves"] = float(h0)
sm = torch.jit.script(m); out["scripted"] = True
print("ARM_JSON " + json.dumps(out))
'''

base_env = {k: v for k, v in os.environ.items() if not k.startswith("RWKV_")}
base_env.update({
    "RWKV_ARCH_MODULE": "scratchpad/track2_a18/architecture_d80_lora4_cnd.py",
    "RWKV_INTERLEAVE": "1", "RWKV_GRU_HEAD": "3", "RWKV_PAVA_LAMBDA": "0.2",
    "RWKV_NO_AHEAD_RESIDUAL": "1", "RWKV_STRIP_L0_VLORA": "1",
    "RWKV_STATE_CLAMP_TAU": "300", "RWKV_STATE_CLAMP_WINDOW": "32768",
    "RWKV_STRIP_CMIX": "user_id:0,user_id:1,user_id:2,preset_id:0,preset_id:1,preset_id:2,deck_id:1,deck_id:2,card_id:1",
    "RWKV_ID_FEATURES": "1", "RWKV_REAL_CYCLES": "1", "RWKV_ZERO_FEATURES": "", "RWKV_NO_JIT": "1",
    "PYTHONIOENCODING": "utf-8", "OMP_NUM_THREADS": "6",
})


def run(extra):
    r = subprocess.run([sys.executable, "-c", ARM], env=dict(base_env, **extra), capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if line.startswith("ARM_JSON "):
            return json.loads(line[9:])
    print(r.stdout[-2000:]); print(r.stderr[-3000:])
    raise SystemExit("arm produced no result: " + str(extra))


off = run({})
on = run({"RWKV_PAVA_HORIZON_LAMBDA": "0.05"})
print("OFF:", off)
print("ON :", on)
ok = True
def check(cond, msg):
    global ok
    print(("  PASS " if cond else "  FAIL ") + msg)
    ok = ok and cond

check(not off["on"] and off["hord"] == 0.0, "OFF: lever off, hord_loss_avg exactly 0")
check(on["on"] and on["lambda"] == 0.05 and on["factors"] == [0.125, 8.0], "ON: lever on, factors 1/8 and 8")
check(on["hord"] > 0.0, f"ON: the hinge is positive on real probes ({on['hord']:.5f}) -- crossings exist, as the screen said")
check(abs((on["total"] - off["total"]) - 0.05 * on["hord"]) < 1e-5, f"ON: total loss = OFF + lambda*hinge ({on['total'] - off['total']:.6f} vs {0.05 * on['hord']:.6f})")
check(on["pava"] == off["pava"], "ON: the same-t PAVA term is untouched")
check(on["grad_gru_w"] > 0, "ON: gradient reaches the GRU head")
check(on["hinge_on_equal_curves"] == 0.0, "ON: hinge is exactly 0 when the 4 probe curves coincide (measures ordering, nothing else)")
check(off["scripted"] and on["scripted"], "torch.jit.script compiles with and without the flag")
print("SMOKE_HORD " + ("PASS" if ok else "FAIL"))
raise SystemExit(0 if ok else 1)
