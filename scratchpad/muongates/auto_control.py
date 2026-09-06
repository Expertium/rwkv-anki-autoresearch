"""Pick muongates' BASE mechanically when eqw reports, regenerate the runner, preflight it.

eqw's gate is BOTH-modes vs realcyc: realcyc_verdict.py's `gate:` line ends with ACCEPT.
  eqw ACCEPT => `mk_muongates.py eqw`      (its runner carries the scored-set weighting)
  else       => `mk_muongates.py realcyc`
Writes muongates/CONTROL.txt; exits 0 only if the chosen runner preflights.
Usage: auto_control.py            (called by wait_eqw_then_muongates.cmd)
"""
import os
import subprocess

os.chdir(r"C:\Users\Andrew\rwkv-anki-autoresearch")
PY = r".venv\Scripts\python.exe"
env = dict(os.environ, PYTHONIOENCODING="utf-8")


def sh(args):
    r = subprocess.run(args, capture_output=True, text=True, env=env)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


base, reason = "realcyc", "eqw result missing -> base realcyc"
have = all(os.path.exists(f) for f in ("result/RWKV-eqw.jsonl", "result/RWKV-P-eqw.jsonl"))
if have:
    rc, out = sh([PY, "scratchpad/realcyc/realcyc_verdict.py", "eqw", "realcyc"])
    gate = [l for l in out.splitlines() if l.startswith("gate:")]
    passed = bool(gate) and gate[0].strip().endswith("ACCEPT")
    reason = gate[0] if gate else "verdict script produced no gate line:\n" + out[-600:]
    if passed:
        base = "eqw"
rc_g, out_g = sh([PY, "scratchpad/muongates/mk_muongates.py", base])
rc_f, out_f = sh([PY, "scratchpad/preflight_runner.py", "scratchpad/muongates/run_muongates.cmd"])
with open("scratchpad/muongates/CONTROL.txt", "a") as f:
    f.write(f"\n[auto_control] base={base}\n{reason}\nmk rc={rc_g}\npreflight rc={rc_f}\n{out_g[-300:]}\n{out_f[-300:]}\n")
print(f"muongates base = {base} ({reason[:160]}); mk rc {rc_g}; preflight rc {rc_f}")
raise SystemExit(0 if (rc_g == 0 and rc_f == 0) else 4)
