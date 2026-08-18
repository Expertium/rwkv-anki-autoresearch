"""Generate run_iter54_phase2.cmd -- iter 54's DECAY + EVAL, after its decay phase was skipped.

WHAT HAPPENED (2026-08-18). The resumed WS completed correctly: `i54_ws_10935.pth` written 12:44,
step 10935 reached, `[resume-skip]` confirmed. The chain log then reads

    === DECAY SETUP 12:44:32,61 ===
    DECAY OK (KD ON, alpha 0.5) 12:44:32,67
    === EVAL TOML rect 12:44:32,67 ===
    DONE_EXIT_TOMLFAIL_1

i.e. decay "succeeded" in 0.06 s and the eval-toml step then failed with
`ERROR: no i54_d_<step>.pth`. `write_decay_setup.py` DID run (dsetup log + a correct
`i54_decay.toml`, both mtime 12:44), but `decay_resume8000.log` was never touched -- its mtime is
11:47, from the earlier aborted attempt.

**THE BUG I CAN PROVE, and it is the reusable one: the two KD guards passed by reading a STALE
LOG.** They `findstr` for `[kd-mix] KD ON` and `alpha FIXED at 0.5` in `decay_%STAMP%.log`, and the
aborted 11:47 attempt had left a log of that exact name containing both strings. So the guards
certified a decay phase that had not run in this process. Same family as a stale `DONE_EXIT_` line
firing a waiter: **a guard that reads a file it did not just write is not a guard.**

⚠ **NOT fully explained: why the `train_rwkv` decay line itself did not execute.** The runner was
written 11:49:27 and launched 11:49:34, so the cmd.exe byte-offset hazard does not apply; the block
is syntactically well-formed; `%DIR%`/`%STAMP%` resolved correctly for the neighbouring dsetup
redirect in the same second; and no stray `decay_*.log` exists anywhere. Recorded as unexplained
rather than guessed at.

THIS RUNNER therefore does three things differently:
  1. a FRESH STAMP (`phase2`), so no pre-existing log can satisfy any guard;
  2. it DELETES its own decay log up front, so the KD guards can only ever read what this run wrote;
  3. an ARTIFACT guard after decay -- refuses to build the eval toml unless `i54_d_10935.pth`
     exists -- mirroring the WS guard added after the LOAD_MODEL_FOLDER incident. Exit code 0 is
     not evidence a phase ran.
"""
import io
import re

SRC = "scratchpad/iter54_cmixpow/run_iter54_resume.cmd"
DST = "scratchpad/iter54_cmixpow/run_iter54_phase2b.cmd"

s = io.open(SRC, encoding="ascii", newline="").read()

# --- drop everything from the phase-0 guards through the WS phase; keep from DECAY SETUP on ---
# Cut at the START of the WS phase, not at the decay marker: everything between them IS the WS
# phase, and slicing to the decay marker would carry it along -- which the "a WS phase leaked in"
# guard below caught on the first attempt. env_block keeps cd /d, DIR/LOG/STAMP, the full env, the
# KD dump vars AND phase 0 (the param assert is worth re-running: it proves the arch env is right).
ws_start = s.index("REM RESUMED from step 8000")
head_end = s.index('set RWKV_RESUME_SKIP_GROUPS=\r\necho === DECAY SETUP')
env_block = s[s.index("setlocal"):ws_start]
tail = s[head_end:]

# the resume flag has no meaning here
tail = tail.replace("set RWKV_RESUME_SKIP_GROUPS=\r\n", "", 1)

# fresh stamp so nothing stale can be read
env_block = re.sub(r"^set STAMP=.*$", "set STAMP=phase2b", env_block, count=1, flags=re.M)
# Own log file. iter54.log already carries DONE_EXIT_TOMLFAIL_1 from the 12:44 failure, and
# appending a DONE_EXIT_0 after it would leave one log with two terminal markers -- a waiter
# using `findstr /B` takes whichever it finds, so the file would report success and failure
# at once. A phase-2 run gets a phase-2 log.
env_block = env_block.replace(r"set LOG=%DIR%\iter54.log",
                              r"set LOG=%DIR%\iter54_phase2b.log")

# (2) delete this run's own decay log before writing it, so the KD guards cannot read stale content
#
# ★★ (2b) AND RESET THE KD ALPHA TO 0.5 -- whose absence killed phase 2a after 3.3 h of decay.
# The champion uses alpha 0.9 for WS (iter 39) and 0.5 for DECAY (iter 45). The original runner
# sets 0.9 in its env block and RESETS it to 0.5 on its own line just before the decay -- and that
# reset line lives INSIDE the WS region this generator slices away. Phase 2a therefore decayed at
# 0.9, which is iter 55's lever, and would have made iter 54's number a confounded mixture of two
# experiments. The runner's own KD guard caught it (DONE_EXIT_WRONGALPHA_DECAY) -- but a guard only
# DETECTS a fault, it cannot repair one, so the reset has to be put back explicitly.
DEC = 'echo === DECAY SETUP %TIME% === >> "%LOG%"\r\n'
assert DEC in tail
tail = tail.replace(
    DEC,
    'REM Delete OUR log first: the 12:44 failure had the KD guards certify a decay that never ran,\r\n'
    'REM because an aborted attempt had left a same-named log containing both guard strings.\r\n'
    'if exist "%DIR%\\decay_%STAMP%.log" del "%DIR%\\decay_%STAMP%.log"\r\n'
    'REM The champion decays at alpha 0.5 (iter 45); 0.9 is its WS value (iter 39). The line that\r\n'
    'REM performs this reset sits inside the WS phase, which this runner does not contain.\r\n'
    'set RWKV_KD_ALPHA=0.5\r\n' + DEC, 1)

# (3) artifact guard: the eval toml step must not run without a decay checkpoint
ETOML = 'echo === EVAL TOML rect %TIME% === >> "%LOG%"\r\n'
assert ETOML in tail
tail = tail.replace(
    ETOML,
    'REM Exit code 0 is not evidence the decay ran -- that is exactly how 12:44 failed.\r\n'
    'if not exist "%DIR%\\i54_d_10935.pth" (\r\n'
    '  echo decay produced no i54_d_10935.pth -- refusing to evaluate %TIME% >> "%LOG%"\r\n'
    '  echo DONE_EXIT_DECAYNOFINAL %DATE% %TIME% >> "%LOG%"\r\n'
    '  exit /b 24\r\n'
    ')\r\n' + ETOML, 1)

HEADER = [
    "@echo off",
    "REM =========================================================================================",
    "REM ITER 54 PHASE 2 -- DECAY + EVAL only. Its WS is complete and on disk",
    "REM (i54_ws_10935.pth, 2026-08-18 12:44, resumed from step 8000 after the power outage).",
    "REM The original run's decay phase never executed while its KD guards passed on a STALE log",
    "REM left by an aborted attempt; see mk54_phase2.py for the full account.",
    "REM",
    "REM Differences from run_iter54_resume.cmd: fresh STAMP, deletes its own decay log first, and",
    "REM refuses to evaluate unless the decay checkpoint exists.",
    "REM",
    "REM Do NOT edit this file while it runs (iters 43 and 46 died that way).",
    "REM No angle brackets or arrows in REM lines -- cmd parses redirection before REM.",
    "REM =========================================================================================",
]
out = "\n".join(HEADER) + "\n" + (env_block + tail).replace("\r\n", "\n")
out = out.replace("\n", "\r\n")

# ---------------- OUTPUT GUARDS ----------------
lines = out.split("\r\n")
assert re.search(r"^cd /d ", out, re.M), "cd /d was lost"
setpos, usepos = {}, {}
for n, ln in enumerate(lines):
    m = re.match(r"\s*set\s+([A-Za-z_][A-Za-z0-9_]*)=", ln)
    if m and m.group(1).upper() not in setpos:
        setpos[m.group(1).upper()] = n
    for v in re.findall(r"%([A-Za-z_][A-Za-z0-9_]*)%", ln):
        usepos.setdefault(v.upper(), n)
for v, un in usepos.items():
    if v in {"DATE", "TIME", "ERRORLEVEL", "RANDOM", "CD"}:
        continue
    assert v in setpos, f"%{v}% used at line {un + 1} but never set"
    assert setpos[v] < un, f"%{v}% used at line {un + 1} before being set at line {setpos[v] + 1}"

_el = [n for n, ln in enumerate(lines) if ln.strip().lower() == "endlocal"]
_de = [n for n, ln in enumerate(lines) if ln.strip().startswith("echo DONE_EXIT_0")]
assert _el and _de and min(_el) > max(_de), "endlocal must come AFTER the DONE_EXIT_0 echo"

assert "train_rwkv --config scratchpad/iter54_cmixpow/i54_decay.toml" in out, "decay phase missing"
assert "i54_ws_resume.toml" not in out and "i54_ws.toml" not in out, "a WS phase leaked in"
assert out.count("STAMP=phase2b") == 1
assert out.count("DONE_EXIT_0") == 1
assert 'if exist "%DIR%\\decay_%STAMP%.log" del' in out, "stale-log deletion missing"
assert "i54_d_10935.pth" in out, "decay artifact guard missing"
assert out.count("set RWKV_KD_ALPHA=0.5") == 1, "the decay alpha reset is missing"
assert out.index("set RWKV_KD_ALPHA=0.9") < out.index("set RWKV_KD_ALPHA=0.5") < out.index("i54_decay.toml"), "0.5 must be set after 0.9 and before the decay call"
for ln in lines:
    if ln.strip().upper().startswith("REM"):
        bad = [c for c in "<>&|^" if c in ln]
        assert not bad, "redirection char " + str(bad) + " in REM line: " + ln

io.open(DST, "w", encoding="ascii", newline="").write(out)
print("wrote", DST, len(out), "bytes")
