"""Pick muonscale's BASE mechanically when hord reports, regenerate the runner, preflight it.

hord's gate is CURVE-SIDE vs the control recorded in hord/CONTROL.txt (`control=sam` or `control=realcyc`):
size 0/2499 AND paired_pvalue.py --curve-side exit 0 AND ahead raw >= 1e-4.
  hord ACCEPT => `mk_muonscale.py hord`          (hord's runner already carries the hinge, and SAM iff its decay did)
  else        => `mk_muonscale.py realcyc [--sam]` with --sam iff hord/CONTROL.txt says with_sam=True
Writes muonscale/CONTROL.txt; exits 0 only if the chosen runner preflights.
Usage: auto_control.py            (called by wait_hord_then_muonscale.cmd)
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


hord_ctrl, hord_sam = "realcyc", False
if os.path.exists("scratchpad/hord/CONTROL.txt"):
    for line in open("scratchpad/hord/CONTROL.txt"):
        if line.startswith("control="):
            hord_ctrl = line.strip().split("=", 1)[1]
        if line.startswith("with_sam="):
            hord_sam = line.strip().split("=", 1)[1] == "True"
base, reason = "realcyc", f"hord result missing -> base realcyc{' + SAM' if hord_sam else ''}"
have = all(os.path.exists(f) for f in ("result/RWKV-hord.jsonl", "result/RWKV-P-hord.jsonl",
                                          f"result/RWKV-{hord_ctrl}.jsonl", f"result/RWKV-P-{hord_ctrl}.jsonl"))
if have:
    rc_s, out_s = sh([PY, "optimization/size_baseline.py", "check", "id_e2s", "result/RWKV-hord.jsonl"])
    rc_p, out_p = sh([PY, "optimization/paired_pvalue.py",
                      "--cand-ahead", "result/RWKV-hord.jsonl", "--cand-imm", "result/RWKV-P-hord.jsonl",
                      "--champ-ahead", f"result/RWKV-{hord_ctrl}.jsonl", "--champ-imm", f"result/RWKV-P-{hord_ctrl}.jsonl",
                      "--intersect", "--curve-side"])
    ahead_raw = None
    for line in out_p.splitlines():
        if line.startswith("PAIRED_P_JSON "):
            import json
            ahead_raw = json.loads(line[14:])["ahead"]["delta"]
    passed = rc_s == 0 and rc_p == 0 and ahead_raw is not None and ahead_raw >= 1e-4
    reason = f"hord vs {hord_ctrl}: size rc {rc_s}, curve-side rc {rc_p}, ahead raw {ahead_raw} -> {'ACCEPT' if passed else 'REJECT'}"
    if passed:
        base = "hord"
args = [PY, "scratchpad/muonscale/mk_muonscale.py", base] + (["--sam"] if (base == "realcyc" and hord_sam) else [])
rc_g, out_g = sh(args)
rc_f, out_f = sh([PY, "scratchpad/preflight_runner.py", "scratchpad/muonscale/run_muonscale.cmd"])
with open("scratchpad/muonscale/CONTROL.txt", "a") as f:
    f.write(f"\n[auto_control] base={base} sam_in_decay={hord_sam if base == 'realcyc' else 'as hord'}\n{reason}\nmk rc={rc_g}\npreflight rc={rc_f}\n")
print(f"muonscale base = {base} ({reason[:160]}); mk rc {rc_g}; preflight rc {rc_f}")
raise SystemExit(0 if (rc_g == 0 and rc_f == 0) else 4)
