"""Pick SAM's BASE mechanically when ordcut reports, regenerate the decay-only runner, preflight it.

Base = ordcut if ordcut passed its CURVE-SIDE gate against the control it ran on (size 0/2499 AND
paired_pvalue.py --curve-side exit 0 AND ahead raw >= 1e-4), else that control itself (durdrop if
durdrop passed the both-modes gate -- recorded by ordcut/auto_control.py in ordcut/CONTROL.txt --
else realcyc). Writes sam/CONTROL.txt; exits 0 only if the chosen runner preflights.
Usage: auto_control.py            (called by wait_ordcut_then_sam.cmd)
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


ord_ctrl = "realcyc"
if os.path.exists("scratchpad/ordcut/CONTROL.txt"):
    for line in open("scratchpad/ordcut/CONTROL.txt"):
        if line.startswith("control="):
            ord_ctrl = line.strip().split("=", 1)[1]
base, reason = ord_ctrl, f"ordcut result missing -> base = ordcut's control ({ord_ctrl})"
have = all(os.path.exists(f) for f in ("result/RWKV-ordcut.jsonl", "result/RWKV-P-ordcut.jsonl",
                                          f"result/RWKV-{ord_ctrl}.jsonl", f"result/RWKV-P-{ord_ctrl}.jsonl"))
if have:
    rc_s, out_s = sh([PY, "optimization/size_baseline.py", "check", "id_e2s", "result/RWKV-ordcut.jsonl"])
    rc_p, out_p = sh([PY, "optimization/paired_pvalue.py",
                      "--cand-ahead", "result/RWKV-ordcut.jsonl", "--cand-imm", "result/RWKV-P-ordcut.jsonl",
                      "--champ-ahead", f"result/RWKV-{ord_ctrl}.jsonl", "--champ-imm", f"result/RWKV-P-{ord_ctrl}.jsonl",
                      "--intersect", "--curve-side"])
    ahead_raw = None
    for line in out_p.splitlines():
        if line.startswith("PAIRED_P_JSON "):
            import json
            ahead_raw = json.loads(line[14:])["ahead"]["delta"]
    passed = rc_s == 0 and rc_p == 0 and ahead_raw is not None and ahead_raw >= 1e-4
    reason = f"ordcut vs {ord_ctrl}: size rc {rc_s}, curve-side rc {rc_p}, ahead raw {ahead_raw} -> {'ACCEPT' if passed else 'REJECT'}"
    if passed:
        base = "ordcut"
rc_g, out_g = sh([PY, "scratchpad/sam/mk_sam.py", base])
rc_f, out_f = sh([PY, "scratchpad/preflight_runner.py", "scratchpad/sam/run_sam.cmd"])
with open("scratchpad/sam/CONTROL.txt", "a") as f:
    f.write(f"\n[auto_control] base={base}\n{reason}\nmk_sam rc={rc_g}\npreflight rc={rc_f}\n")
print(f"sam base = {base} ({reason[:160]}); mk_sam rc {rc_g}; preflight rc {rc_f}")
raise SystemExit(0 if (rc_g == 0 and rc_f == 0) else 4)
