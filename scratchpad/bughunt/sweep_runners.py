"""Sweep EVERY .cmd runner in the repo for the four STRUCTURAL defects, and report by defect.

WHY BY DEFECT AND NOT BY FILE. The 2026-08-18 audit found the bug rate is flat (21% of commits in
both halves of 24 days) and that the real driver is INHERITANCE: each new run was cloned from its
nearest ancestor, so one ancestor's defect propagates to a whole lineage. `run_iter45.cmd` carries
the endlocal bug and so does every descendant. A per-file list hides that; a per-defect count with
the file list under it shows the lineage directly, which is the evidence for replacing cloning with
one canonical template.

Only STRUCTURAL defects are counted -- things that are wrong no matter when the runner ran. Stale
paths and missing checkpoints are excluded: a finished run legitimately points at artifacts that
have since been deleted, and counting those would drown the signal.

  D1  endlocal precedes the terminal DONE_EXIT_ echo  -> the marker is written to "" and every
      chained waiter polls forever. This is the one that cost 45 min of idle GPU.
  D2  a %VAR% is used before it is set                -> the mk53/mk54 slice; %LOG% expands empty.
  D3  no `cd /d`                                      -> a detached job starts in System32.
  D4  a findstr guard names a value its env does not set (or the reverse).

Usage: .venv/Scripts/python.exe scratchpad/bughunt/sweep_runners.py
"""
import glob
import io
import os
import re
from collections import defaultdict

BUILTIN = {"DATE", "TIME", "ERRORLEVEL", "RANDOM", "CD", "PATH", "TEMP", "TMP",
           "USERPROFILE", "APPDATA", "SYSTEMROOT", "COMSPEC", "NUMBER_OF_PROCESSORS"}

hits = defaultdict(list)
files = sorted(glob.glob("scratchpad/**/*.cmd", recursive=True))
runners = []

for p in files:
    try:
        text = io.open(p, encoding="ascii", errors="replace", newline="").read()
    except OSError:
        continue
    lines = text.replace("\r\n", "\n").split("\n")
    # A "runner" launches training or eval. Waiters and one-liners are out of scope.
    if not re.search(r"train_rwkv|eval_sharded|get_result", text):
        continue
    # A runner INVOKED by another runner (`call ...run_arm.cmd`) legitimately inherits its
    # caller's environment and working directory, so "uses %DIR% without setting it" and "has no
    # cd /d" are correct there, not defects. Skip callees for D2/D3.
    base = os.path.basename(p)
    is_callee = any(base.lower() in io.open(q, encoding="ascii", errors="replace").read().lower()
                    for q in files if q != p and q.endswith(".cmd"))
    runners.append(p)

    # D1 -- endlocal before the terminal marker
    bare_el = [n for n, l in enumerate(lines) if l.strip().lower() == "endlocal"]
    de = [n for n, l in enumerate(lines) if l.strip().startswith("echo DONE_EXIT_")]
    if bare_el and de and min(bare_el) < max(de):
        hits["D1 endlocal before the DONE_EXIT_ marker"].append(p)

    # D2 -- %VAR% used before set
    setpos, usepos = {}, {}
    for n, l in enumerate(lines):
        m = re.match(r"\s*set\s+([A-Za-z_][A-Za-z0-9_]*)=", l)
        if m and m.group(1).upper() not in setpos:
            setpos[m.group(1).upper()] = n
        # A REM line mentioning %LOG% is prose, not a use. baseline_gru documents its log
        # hygiene at line 17 and sets LOG at 27 -- counting the comment reports a false defect.
        if l.strip().upper().startswith("REM"):
            continue
        for v in re.findall(r"%([A-Za-z_][A-Za-z0-9_]*)%", l):
            usepos.setdefault(v.upper(), n)
    bad = [v for v, un in usepos.items()
           if v not in BUILTIN and (v not in setpos or setpos[v] > un)]
    if bad and not is_callee:
        hits["D2 %%VAR%% used before it is set"].append("%s  (%s)" % (p, ",".join(sorted(bad)[:4])))

    # D3 -- no cd /d
    if not re.search(r"^cd /d ", text, re.M) and not is_callee:
        hits["D3 no `cd /d` (a detached job starts in System32)"].append(p)

    # D4 -- guard/value desync on the KD alpha
    for n, l in enumerate(lines):
        m = re.search(r'findstr /C:"alpha FIXED at ([0-9.]+)"', l)
        if not m:
            continue
        env = None
        for prev in lines[:n]:
            mm = re.match(r"\s*set\s+RWKV_KD_ALPHA=(.*)$", prev)
            if mm and mm.group(1).strip():
                env = mm.group(1).strip()
        if env is not None and abs(float(env) - float(m.group(1))) > 1e-9:
            hits["D4 guard names an alpha the env does not set"].append(
                "%s  (guard %s, env %s)" % (p, m.group(1), env))

print("scanned %d .cmd files, %d of them are runners\n" % (len(files), len(runners)))
for k in sorted(hits):
    v = hits[k]
    print("%-52s %3d / %d runners" % (k.split(" ", 1)[0] + " " + k.split(" ", 1)[1][:44], len(v), len(runners)))
    for f in v[:8]:
        print("      " + f)
    if len(v) > 8:
        print("      ... and %d more" % (len(v) - 8))
    print()
if not hits:
    print("no structural defects found")
