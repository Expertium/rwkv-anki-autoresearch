"""Pick ordcut's CONTROL mechanically when durdrop reports, then regenerate the runner if needed.

Why this exists: a chained waiter fires ~2 min after the previous run's marker, which is too short
for a human verdict (the realcyc -> lorawd hand-off needed the waiter stopped by hand). The accept
gate is mechanical (size 0/2499, raw >= 1e-4 both modes, p < 1e-4 both modes), so the waiter can
apply it: durdrop ACCEPT => ordcut runs on durdrop's recipe (mk_ordcut.py durdrop); otherwise on
realcyc's (the runner already generated). Writes the decision to ordcut/CONTROL.txt and exits 0
only if the chosen runner preflights.

Usage: auto_control.py            (called by wait_durdrop_then_ordcut.cmd)
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


decision = "realcyc"
reason = "durdrop result files missing or incomplete -> control stays realcyc"
have = all(os.path.exists(f) for f in ("result/RWKV-durdrop.jsonl", "result/RWKV-P-durdrop.jsonl"))
if have:
    rc, out = sh([PY, "scratchpad/realcyc/realcyc_verdict.py", "durdrop", "realcyc"])
    verdict_line = [l for l in out.splitlines() if l.startswith("gate:")]
    accepted = bool(verdict_line) and verdict_line[0].strip().endswith("ACCEPT")
    reason = verdict_line[0] if verdict_line else "verdict script produced no gate line:\n" + out[-800:]
    if accepted:
        decision = "durdrop"
        rc2, out2 = sh([PY, "scratchpad/ordcut/mk_ordcut.py", "durdrop"])
        reason += " | regenerated on durdrop: " + out2.strip().splitlines()[-1] if out2.strip() else ""
        if rc2 != 0:
            print(out2); raise SystemExit(3)
rc3, out3 = sh([PY, "scratchpad/preflight_runner.py", "scratchpad/ordcut/run_ordcut.cmd"])
with open("scratchpad/ordcut/CONTROL.txt", "w") as f:
    f.write(f"control={decision}\n{reason}\npreflight_rc={rc3}\n")
print(f"ordcut control = {decision} ({reason[:160]}); preflight rc {rc3}")
raise SystemExit(0 if rc3 == 0 else 4)
