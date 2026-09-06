"""Pick eqw's BASE mechanically when muonscale reports, regenerate the runner, preflight it.

muonscale's gate is BOTH-modes vs the control recorded in muonscale/CONTROL.txt (`control=hord` or
`control=realcyc`): realcyc_verdict.py's `gate:` line ends with ACCEPT.
  muonscale ACCEPT => `mk_eqw.py muonscale`   (its runner carries the scale flag, and hord's hinge iff hord passed)
  else             => `mk_eqw.py <muonscale's control>`  (hord or realcyc)
Writes eqw/CONTROL.txt; exits 0 only if the chosen runner preflights.
Usage: auto_control.py            (called by wait_muonscale_then_eqw.cmd)
"""
import os
import subprocess

os.chdir(r"C:\Users\Andrew\rwkv-anki-autoresearch")
PY = r".venv\Scripts\python.exe"
env = dict(os.environ, PYTHONIOENCODING="utf-8")


def sh(args):
    r = subprocess.run(args, capture_output=True, text=True, env=env)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


ms_ctrl = "realcyc"
if os.path.exists("scratchpad/muonscale/CONTROL.txt"):
    for line in open("scratchpad/muonscale/CONTROL.txt"):
        if line.startswith("control="):
            ms_ctrl = line.strip().split("=", 1)[1]
base, reason = ms_ctrl, f"muonscale result missing -> base {ms_ctrl}"
have = all(os.path.exists(f) for f in ("result/RWKV-muonscale.jsonl", "result/RWKV-P-muonscale.jsonl",
                                          f"result/RWKV-{ms_ctrl}.jsonl", f"result/RWKV-P-{ms_ctrl}.jsonl"))
if have:
    rc, out = sh([PY, "scratchpad/realcyc/realcyc_verdict.py", "muonscale", ms_ctrl])
    gate = [l for l in out.splitlines() if l.startswith("gate:")]
    passed = bool(gate) and gate[0].strip().endswith("ACCEPT")
    reason = gate[0] if gate else "verdict script produced no gate line:\n" + out[-600:]
    if passed:
        base = "muonscale"
rc_g, out_g = sh([PY, "scratchpad/eqw/mk_eqw.py", base])
rc_f, out_f = sh([PY, "scratchpad/preflight_runner.py", "scratchpad/eqw/run_eqw.cmd"])
with open("scratchpad/eqw/CONTROL.txt", "a") as f:
    f.write(f"\n[auto_control] base={base}\n{reason}\nmk rc={rc_g}\npreflight rc={rc_f}\n{out_g[-300:]}\n{out_f[-300:]}\n")
print(f"eqw base = {base} ({reason[:160]}); mk rc {rc_g}; preflight rc {rc_f}")
raise SystemExit(0 if (rc_g == 0 and rc_f == 0) else 4)
