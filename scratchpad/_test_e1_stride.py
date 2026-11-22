"""Unit check for the E1 stride guard (CPU, no trainer). Uses a SUBSAMPLE process per case."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")

CASE = r'''
import os, torch
from rwkv.model import rwkv_ops as ro
s = torch.randn(2, 2, 32, 32)
n_before = len(ro._TAIL_ACC.get("card", []))
for i in range(8):
    ro.accum_tail_penalty(s, 2, "card")
for i in range(8):
    ro.accum_tail_penalty(s, 1, "note")
n_card = len(ro._TAIL_ACC.get("card", []))
n_note = len(ro._TAIL_ACC.get("note", []))
tl = ro.take_tail_penalty()
print(f"RESULT lambda={ro._TAIL_REG} stride={ro._TAIL_STRIDE} card={n_card} note={n_note} "
      f"tail={None if tl is None else float(tl):} cleared={not ro._TAIL_ACC}")
'''


def run(env_extra: dict) -> str:
    env = dict(os.environ)
    env.pop("RWKV_QAT_TAIL_PENALTY", None)
    env.pop("RWKV_QAT_TAIL_STRIDE", None)
    env.update(env_extra)
    env["PYTHONIOENCODING"] = "utf-8"
    p = subprocess.run([PY, "-c", CASE], cwd=str(ROOT), env=env, capture_output=True, text=True)
    out = (p.stdout or "") + (p.stderr or "")
    line = [l for l in out.splitlines() if l.startswith("RESULT")]
    return line[-1] if line else "FAIL rc=%s :: %s" % (p.returncode, out.strip()[-400:])


print("penalty ON, stride 4 :", run({"RWKV_QAT_TAIL_PENALTY": "1.0", "RWKV_QAT_TAIL_STRIDE": "4"}))
print("penalty ON, default  :", run({"RWKV_QAT_TAIL_PENALTY": "1.0"}))
print("penalty ON, stride 1 :", run({"RWKV_QAT_TAIL_PENALTY": "1.0", "RWKV_QAT_TAIL_STRIDE": "1"}))
print("penalty OFF (default):", run({}))
