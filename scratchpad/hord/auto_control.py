"""Pick hord's BASE mechanically when sam reports, regenerate the runner, preflight it.

sam is DECAY-ONLY on realcyc's WS-final, so its gate is the both-modes rule vs realcyc (an optimizer-
side, trunk-wide change): size 0/2499 AND raw >= 1e-4 in both modes AND paired p < 1e-4 in both.
  sam ACCEPT  => `mk_hord.py realcyc --sam` : hord's decay carries SAM; control = sam's numbers
  else        => `mk_hord.py realcyc`       : control = realcyc
Writes hord/CONTROL.txt; exits 0 only if the chosen runner preflights.
Usage: auto_control.py            (called by wait_sam_then_hord.cmd)
"""
import os
import subprocess
import sys

os.chdir(r"C:\Users\Andrew\rwkv-anki-autoresearch")
PY = r".venv\Scripts\python.exe"
env = dict(os.environ, PYTHONIOENCODING="utf-8")


def sh(args):
    r = subprocess.run(args, capture_output=True, text=True, env=env)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


with_sam, reason = False, "sam result files missing -> plain realcyc base"
have = all(os.path.exists(f) for f in ("result/RWKV-sam.jsonl", "result/RWKV-P-sam.jsonl"))
if have:
    rc, out = sh([PY, "scratchpad/realcyc/realcyc_verdict.py", "sam", "realcyc"])
    gate = [l for l in out.splitlines() if l.startswith("gate:")]
    with_sam = bool(gate) and gate[0].strip().endswith("ACCEPT")
    reason = gate[0] if gate else "verdict script produced no gate line:\n" + out[-600:]
args = [PY, "scratchpad/hord/mk_hord.py", "realcyc"] + (["--sam"] if with_sam else [])
rc_g, out_g = sh(args)
rc_f, out_f = sh([PY, "scratchpad/preflight_runner.py", "scratchpad/hord/run_hord.cmd"])
with open("scratchpad/hord/CONTROL.txt", "a") as f:
    f.write(f"\n[auto_control] with_sam={with_sam}\n{reason}\nmk_hord rc={rc_g}\npreflight rc={rc_f}\n")
print(f"hord base = realcyc{' + SAM in decay (control = sam)' if with_sam else ' (control = realcyc)'} ({reason[:160]}); mk_hord rc {rc_g}; preflight rc {rc_f}")
raise SystemExit(0 if (rc_g == 0 and rc_f == 0) else 4)
