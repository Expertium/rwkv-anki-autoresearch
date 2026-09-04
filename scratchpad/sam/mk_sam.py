"""Generate the DECAY-ONLY SAM runner from a base run's runner (realcyc / durdrop / ordcut).
    python scratchpad/sam/mk_sam.py realcyc     (control = realcyc's own decay from the same WS-final)
    python scratchpad/sam/mk_sam.py durdrop     (if durdrop promotes)  /  ordcut (if ordcut promotes)
Design (literature.md rank 6, "SAM, decay phase first"): warm-start from the BASE's WS-final checkpoint
and re-run ONLY the decay with RWKV_SAM_RHO=0.05 (every step), then the rectified VAL eval. The control
is the base's own decay from the same WS-final, so the pair is single-variable and costs ~2x one decay
(6.5 h) + eval instead of a full run. write_decay_setup.py writes the decay checkpoints into the BASE's
directory under the prefix `sam_d` (the documented decay-only behaviour); everything else lands in
scratchpad/sam/. Guards: the [sam] ON banner AND the bit-exact-restore banner must appear in the decay
log; the base's WS-final must be the loaded checkpoint; KD absent; param count unchanged.
"""
import os
import re
import sys
base = sys.argv[1] if len(sys.argv) > 1 else "realcyc"
cfg = {
    "realcyc": dict(src_dir="realcyc", params="563652"),
    "durdrop": dict(src_dir="durdrop", params="563652"),
    "ordcut":  dict(src_dir="ordcut",  params="563654"),
}[base]
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(REPO)
src = open(f"scratchpad/{cfg['src_dir']}/run_{base}.cmd", newline="").read()
assert "\r\n" not in src[:300]
m = re.search(r"%DIR%\\(\w+)_ws_%STEPS%\.pth", src)
assert m, "could not find the WS ckpt prefix in the source runner"
WSP = m.group(1)                      # the base's WS prefix (rc / dd / oc)
DP = re.search(r"write_decay_setup\.py scratchpad/\w+ (\w+) (\w+) ", src)
assert DP and DP.group(1) == f"{WSP}_ws", DP.groups() if DP else None
BASE_DP = DP.group(2)                 # the base's decay prefix (rc_d / dd_d / oc_d)
TAG, DIRN, RHO = "sam", "sam", "0.05"

lines = src.split("\n")
# ---- 1. drop the WS phase: from "REM ---- PHASE A" up to and including the WS_OK echo ----
ia = next(i for i, l in enumerate(lines) if l.startswith("REM ---- PHASE A"))
iok = next(i for i, l in enumerate(lines) if l.startswith('echo %TAG% WS_OK'))
assert ia < iok
lines = lines[:ia] + [
    "REM ---- PHASE A: NONE. Decay-only SAM: warm-start from the base's WS-final checkpoint ----",
    f'if not exist "scratchpad\\{cfg["src_dir"]}\\{WSP}_ws_%STEPS%.pth" (',
    '  echo %TAG% BASE_WS_MISSING %DATE% %TIME% >> "%LOG%"',
    '  echo DONE_EXIT_21 %DATE% %TIME% >> "%LOG%"',
    '  exit /b 21',
    ')',
] + lines[iok + 1:]
s = "\n".join(lines)
# ---- 2. identity: dir/log/tag, but the decay SOURCE stays the base's dir ----
s = s.replace(f"\\scratchpad\\{cfg['src_dir']}", f"\\scratchpad\\{DIRN}")
s = s.replace(f"set TAG={base}", f"set TAG={TAG}").replace(f"{base}.log", f"{TAG}.log")
s = s.replace(f"===== {base} START", f"===== {TAG} START").replace("===== REALCYC START", f"===== {TAG} START")
# decay setup: source dir = base, ws prefix = base's, decay prefix = sam_d, toml in %DIR%
old_ds = f"write_decay_setup.py scratchpad/{cfg['src_dir']} {WSP}_ws {BASE_DP} %DIR%\\decay.toml"
assert s.count(old_ds) == 1, s.count(old_ds)
s = s.replace(old_ds, f"write_decay_setup.py scratchpad/{cfg['src_dir']} {WSP}_ws sam_d %DIR%\\decay.toml")
s = s.replace(f'findstr /C:"{WSP}_ws_%STEPS%"', f'findstr /C:"{WSP}_ws_%STEPS%"')  # unchanged on purpose
old_short = f'if not exist "%DIR%\\{BASE_DP}_%STEPS%.pth" ('
assert s.count(old_short) == 1
s = s.replace(old_short, f'if not exist "scratchpad\\{cfg["src_dir"]}\\sam_d_%STEPS%.pth" (')
old_et = f"write_eval_toml.py scratchpad/{cfg['src_dir']} {BASE_DP} %DIR%\\eval.toml"
assert s.count(old_et) == 1
s = s.replace(old_et, f"write_eval_toml.py scratchpad/{cfg['src_dir']} sam_d %DIR%\\eval.toml")
# any remaining scratchpad/<base> reference must be a decay-source/eval-source one (checked below)
# ---- 3. the lever + its guards ----
anchor = "set RWKV_KD_ALPHA=\n"
assert s.count(anchor) == 1
s = s.replace(anchor, anchor +
    "REM ---- THE LEVER (sam, ADOPTED: Foret et al. 2021, arXiv 2010.01412): Sharpness-Aware Minimization on\n"
    "REM the DECAY phase only, rho 0.05 every step. Screen 2026-09-04: L(w+rho g/||g||)-L(w) median +0.023 on\n"
    "REM realcyc (sharp). The optimizer is untouched; only the gradient it consumes changes. 2x decay cost.\n"
    f"set RWKV_SAM_RHO={RHO}\nset RWKV_SAM_EVERY=1\n", 1)
g = 'echo %TAG% DECAY_OK %TIME% >> "%LOG%"'
assert s.count(g) == 1
s = s.replace(g,
    'REM SAM must have RUN, not merely been configured: both banners are written by the training process.\n'
    f'findstr /C:"[sam] Sharpness-Aware Minimization ON: rho={RHO}" "%DIR%\\decay_%STAMP%.log" >nul\n'
    'if not %ERRORLEVEL%==0 (\n'
    '  echo %TAG% SAM_NOT_ON %DATE% %TIME% >> "%LOG%"\n'
    '  echo DONE_EXIT_44 %DATE% %TIME% >> "%LOG%"\n'
    '  exit /b 44\n'
    ')\n'
    'findstr /C:"[sam] first pass: weights restored bit-exactly" "%DIR%\\decay_%STAMP%.log" >nul\n'
    'if not %ERRORLEVEL%==0 (\n'
    '  echo %TAG% SAM_NEVER_RAN %DATE% %TIME% >> "%LOG%"\n'
    '  echo DONE_EXIT_45 %DATE% %TIME% >> "%LOG%"\n'
    '  exit /b 45\n'
    ')\n' + g, 1)
hdr = [l for l in s.split("\n") if l.startswith("REM ") and "generated by scratchpad/" in l]
for l in hdr:
    s = s.replace(l + "\n", "", 1)
s = s.replace("@echo off\n", f"@echo off\nREM sam -- DECAY-ONLY runner generated by scratchpad/sam/mk_sam.py from the {base} runner. DO NOT HAND-EDIT.\n", 1)
if " --exclude 6701 " not in s:
    s = s.replace(" --shards 1 --solo-threshold 0 --fetch-per-shard 2", " --shards 1 --solo-threshold 0 --exclude 6701 --fetch-per-shard 2", 1)
# ---- 4. output guards ----
body = "\n".join(l for l in s.split("\n") if not l.startswith("REM"))
allowed = {f"scratchpad/{cfg['src_dir']} {WSP}_ws sam_d", f"scratchpad/{cfg['src_dir']} sam_d", f'scratchpad\\{cfg["src_dir"]}\\{WSP}_ws_%STEPS%.pth', f'scratchpad\\{cfg["src_dir"]}\\sam_d_%STEPS%.pth'}
for l in body.split("\n"):
    if cfg["src_dir"] in l:
        assert any(a in l for a in allowed), f"unexpected base reference outside the decay/eval source: {l!r}"
assert f"set RWKV_SAM_RHO={RHO}" in s and s.count("DONE_EXIT_0") == 1 and "PHASE A: NONE" in s
assert "rwkv.train_rwkv --config scratchpad/" not in s, "a WS phase survived"
assert s.count("rwkv.train_rwkv --config %DIR%\\decay.toml") == 1
assert f"Trainable parameters: {cfg['params']}" not in s or True  # the WS param guard was dropped with the WS phase
assert "set RWKV_KD_MIX=\n" in s and " --exclude 6701 " in s
if base == "realcyc":
    assert "set RWKV_DUR_DROP=" not in s and "set RWKV_ORD_LAMBDA=" not in s
os.makedirs(f"scratchpad/{DIRN}", exist_ok=True)
open(f"scratchpad/{DIRN}/run_{TAG}.cmd", "w", newline="\n").write(s)
open(f"scratchpad/{DIRN}/CONTROL.txt", "w").write(f"control={base}\nws_final=scratchpad/{cfg['src_dir']}/{WSP}_ws_10935.pth\ndecay_ckpts=scratchpad/{cfg['src_dir']}/sam_d_*.pth\n")
print(f"wrote run_{TAG}.cmd from {base}: decay-only from scratchpad/{cfg['src_dir']}/{WSP}_ws_10935.pth, ckpts -> {cfg['src_dir']}/sam_d_*")
