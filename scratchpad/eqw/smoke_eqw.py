"""Smoke for RWKV_EQUALIZE_LOSS_W (2026-09-06 round, rank 1: train on what is scored).

Arms run in their own env (no ambient RWKV_* inheritance -- the rgate lesson). Both arms run the
REAL get_loss on a REAL gen-5 training chunk (the ordcut lesson: a flag-gated loss change is smoked
by running get_loss with the flag on, never by unit-testing a helper).
  OFF (flag absent): equalize_loss_on False; total loss bit-identical to a second OFF arm with the
       flag set to "1" explicitly (default 1.0 = byte-identical); the realcyc checkpoint loads
       strictly (the lever adds no parameters).
  ON  (0.25): the reported ahead objective MOVES and moves UP (the unscored rows are the easier
       ones, so down-weighting them raises the weighted mean); the equalize METRICS are unchanged;
       the chunk must contain BOTH kinds of rows (the smallest chunk of user 101 is 100% scored and
       made the flag vacuous -- the first check that failed); torch.jit.script compiles; the deploy
       loader accepts the ON checkpoint; a JIT-ON arm (old-style ScriptModule) reproduces the eager
       totals, i.e. the scripted runtime path carries the weighting.
CPU, ~1 min.
"""
import json
import os
import subprocess
import sys

ARM = r'''
import os, sys, json
sys.path.insert(0, os.getcwd()); sys.path.insert(0, "scratchpad/proposals_2026-09-04")
import torch, lmdb
torch.manual_seed(0)
from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG as CFG
from rwkv.model.srs_model import SrsRWKV
from rwkv.prepare_batch import get_data, prepare
from sam_probe import to_f32
m = SrsRWKV(CFG)
out = {"on": bool(m.equalize_loss_on), "w": float(m.equalize_loss_w), "keys": len(m.state_dict())}
sd = torch.load("scratchpad/realcyc/rc_d_10935.pth", map_location="cpu", weights_only=True)
m.load_state_dict(sd, strict=True); out["strict_load_realcyc"] = True
m = m.float(); m.eval()
env = lmdb.open("F:/rwkv_lmdb/train_db_5k_h1_id5", map_size=400_000_000_000, readonly=True, lock=False)
with env.begin(write=False) as txn:
    b = json.loads(txn.get(b"101_batches"))[0]   # the FIRST chunk: 33% scored rows (the smallest chunk is 100% scored and made the flag vacuous)
    pb = to_f32(prepare([get_data(txn, (101, b[0], b[1], b[2]), device="cpu")], seed=1234, probe_density=0.08).to("cpu"))
env.close()
st = m.get_loss(pb)
out["total"] = float(st.average_loss)
out["ahead_eq_avg"] = float(st.ahead_equalize_avg); out["ahead_eq_n"] = int(st.ahead_equalize_n)
# hand-computed weighted means of the row losses under the flag's weights, from the same forward
lab = pb.labels.float()
has_label, is_eq, is_q = lab[..., 4], lab[..., 5], lab[..., 6]
out["frac_scored_real"] = float((is_eq * (1 - is_q) * has_label).sum() / ((1 - is_q) * has_label).sum())
w = is_eq + (1 - is_eq) * float(os.environ.get("RWKV_EQUALIZE_LOSS_W", "1") or 1)
out["ahead_avg_reported"] = float(st.ahead_avg)
out["imm_avg_reported"] = float(st.imm_avg)
import tempfile
from rwkv.run_as_rnn import RNNProcess
tmp = os.path.join(tempfile.gettempdir(), "smoke_eqw_ckpt.pth"); torch.save(m.state_dict(), tmp)
RNNProcess(path=tmp, device=torch.device("cpu"), dtype=torch.float32); out["deploy_loads"] = True; os.remove(tmp)
# JIT: SrsRWKV is an OLD-STYLE ScriptModule when RWKV_NO_JIT is unset -- _get_loss is a script_method and
# get_loss runs it. A JIT-ON arm therefore exercises the SCRIPTED runtime path (the iter-48 lesson: a
# compile-only check missed a runtime type bug; only a plain eval runs scripted code).
import rwkv.model.srs_model as _SM
out["jit"] = _SM.FunctionType is torch.jit.script_method   # read from the MODULE: sam_probe's import sets RWKV_NO_JIT=1 in os.environ after srs_model was imported
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


off = run({"RWKV_NO_JIT": "1"})
off1 = run({"RWKV_NO_JIT": "1", "RWKV_EQUALIZE_LOSS_W": "1"})
on = run({"RWKV_NO_JIT": "1", "RWKV_EQUALIZE_LOSS_W": "0.25"})
off_jit = run({})
on_jit = run({"RWKV_EQUALIZE_LOSS_W": "0.25"})
print("OFF :", off); print("OFF1:", off1); print("ON  :", on); print("OFFj:", off_jit); print("ONj :", on_jit)
ok = True


def check(cond, msg):
    global ok
    print(("  PASS " if cond else "  FAIL ") + msg)
    ok = ok and cond


check(not off["on"] and off["w"] == 1.0 and off["strict_load_realcyc"], "OFF: flag inert, realcyc loads strictly")
check(off1["total"] == off["total"] and not off1["on"], "explicit W=1 is byte-identical to the flag being absent")
check(on["on"] and on["w"] == 0.25 and on["keys"] == off["keys"], "ON: engaged at 0.25, no new parameters")
check(0.05 < on["frac_scored_real"] < 0.95, f"the chunk has both scored and unscored rows (scored share {on['frac_scored_real']:.3f}) -- non-vacuous")
check(on["ahead_eq_avg"] == off["ahead_eq_avg"] and on["ahead_eq_n"] == off["ahead_eq_n"], "the equalize METRIC is untouched by the objective weighting")
check(on["ahead_avg_reported"] != off["ahead_avg_reported"], f"the ahead objective moved: {off['ahead_avg_reported']:.6f} -> {on['ahead_avg_reported']:.6f}")
# with alpha < 1 and the unscored rows EASIER than the scored ones, the weighted mean must be HIGHER
check(on["ahead_avg_reported"] > off["ahead_avg_reported"], "weighted ahead objective > unweighted (unscored rows are the easier ones)")
check(on["total"] != off["total"], "total loss moved")
check(on["deploy_loads"] and off["deploy_loads"], "deploy loader accepts the ON checkpoint")
check(off_jit["jit"] and on_jit["jit"] and not on["jit"], "the JIT arms really ran with JIT on (old-style ScriptModule, scripted _get_loss)")
check(abs(on_jit["total"] - on["total"]) < 1e-4 and abs(off_jit["total"] - off["total"]) < 1e-4,
      f"SCRIPTED (JIT-on) get_loss reproduces eager on the real chunk (ON {on_jit['total']:.6f} vs {on['total']:.6f}; OFF {off_jit['total']:.6f} vs {off['total']:.6f})")
check(on_jit["ahead_avg_reported"] > off_jit["ahead_avg_reported"], "the weighting is live in the SCRIPTED path too")
print("SMOKE_EQW " + ("PASS" if ok else "FAIL"))
raise SystemExit(0 if ok else 1)
