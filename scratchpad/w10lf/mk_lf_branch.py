"""Generate a LEAK-FREE decay-only branch off w10lf's WS, and its dry twin (2026-09-11).

WHY. Phase L said the query-row clock leak is MATERIAL (+0.0032 imm on arm 1), so the decay-length
pair and the budget curve no longer run on ws10's leaked WS: their imm conclusions there would be
about a model that reads the leak (part of the budget's +0.0015 imm may be the model learning the
leak better). They are re-based on w10lf, the leak-free 10-epoch run, and w10lf is their control.

Modes:
  budget <ws_step> [decay_epochs]  tag lfb<N>: decay (default 2 ep, like w10lf) from w10lf_ws_<ws_step>,
                                   EVAL ON 300 USERS (5001-5300) -- a curve point, never a gate candidate
                                   (see scratchpad/w10plain/mk_w10budget.py for why 300 suffices).
  decay <epochs>                   tag lfd<ep>: decay <ep> epochs from w10lf_ws_109350, full VAL-half eval.

BUILT FROM ARM 1's RUNNER (scratchpad/w10plain/run_w10plain.cmd) with exactly the substitutions
mk_w10lf.py applies to w10lf's phases B and C: decay from scratchpad/w10lf w10lf_ws, the lever
RWKV_CLOCK_AT_PREV_ANSWER=1800, the decay guards 56/57, the non-fatal eval-banner post-check.
Asserted: the env equals w10lf's own runner (and arm 1's plus the lever), plus RWKV_DECAY_FROM_STEP
in budget mode only; no identity of arm 1 or of ws10 survives outside REM lines; every substitution
fires once; exit codes are unique; the marker precedes endlocal; the checkpoint guard points at the
decay's SOURCE folder (scratchpad/w10lf), where write_decay_setup puts the checkpoints.

⚠ A decay-only branch writes its checkpoints into scratchpad/w10lf, not into its own folder.
⚠ STEPS uses the trainer's arithmetic, int(EPOCHS * 10935), not round() (the 2026-09-10 dry-run bug).

Writes scratchpad/<tag>/run_<tag>.cmd and scratchpad/<tag>dry/run_<tag>dry.cmd
Usage: python scratchpad/w10lf/mk_lf_branch.py budget 22000
       python scratchpad/w10lf/mk_lf_branch.py decay 3
"""
import io
import os
import re
import sys

os.chdir(r"C:\Users\Andrew\rwkv-anki-autoresearch")
ARM1 = "scratchpad/w10plain/run_w10plain.cmd"
LF = "scratchpad/w10lf/run_w10lf.cmd"
T = "1800"
GROUPS = 10935
NL = "\n"


def rd(p):
    return io.open(p, encoding="utf-8", newline="").read().replace("\r\n", NL)


def fmt(ep):
    """2.0 -> '2.0' (as arm 1 and w10lf write it), 3.0 -> '3.0', 0.05 -> '0.05'."""
    s = "%g" % ep
    return s if "." in s else s + ".0"


def code_of(s):
    return NL.join(l for l in s.split(NL) if not l.strip().upper().startswith("REM"))


def env_block(s):
    head = code_of(s).split(".venv\\Scripts")[0]
    out = {}
    for m in re.finditer(r"^set ((?:RWKV_|OMP_|PYTHON)\w*)=(.*)$", head, re.M):
        out.setdefault(m.group(1), m.group(2).rstrip())
    return out


def env_all(s):
    """Every RWKV_/OMP_/PYTHON assignment anywhere in the code, first value wins (phase-B env of w10lf)."""
    out = {}
    for m in re.finditer(r"^set ((?:RWKV_|OMP_|PYTHON)\w*)=(.*)$", code_of(s), re.M):
        out.setdefault(m.group(1), m.group(2).rstrip())
    return out


def sub(text, pat, rep, count=1):
    new, n = re.subn(pat, rep, text)
    if n != count:
        sys.exit(f"REFUSING: substitution fired {n}x (want {count}): {pat}")
    return new


def guard(logname, code_on, code_ran, what):
    return (
        "REM The fix must have REACHED the fetch workers and RUN there: the ON banner prints at import\n"
        "REM in every process, the chunk-count line only from apply_to_sample.\n"
        'findstr /C:"ON: T=%s s" "%%DIR%%\\%s" >nul\n'
        "if not %%ERRORLEVEL%%==0 (\n"
        '  echo %%TAG%% CLOCK_FIX_NOT_ON_%s %%DATE%% %%TIME%% >> "%%LOG%%"\n'
        '  echo DONE_EXIT_%d %%DATE%% %%TIME%% >> "%%LOG%%"\n'
        "  exit /b %d\n"
        ")\n"
        'findstr /C:"s chunks 1 query rows" "%%DIR%%\\%s" >nul\n'
        "if not %%ERRORLEVEL%%==0 (\n"
        '  echo %%TAG%% CLOCK_FIX_NEVER_RAN_%s %%DATE%% %%TIME%% >> "%%LOG%%"\n'
        '  echo DONE_EXIT_%d %%DATE%% %%TIME%% >> "%%LOG%%"\n'
        "  exit /b %d\n"
        ")\n" % (T, logname, what, code_on, code_on, logname, what, code_ran, code_ran))


EVAL_POSTCHECK = (
    "REM Non-fatal post-check: the eval's shard log must carry the ON banner too.\n"
    'findstr /C:"ON: T=%s s" scratchpad\\eval_shards\\shard_s0.log >nul 2>&1\n'
    "if not %%ERRORLEVEL%%==0 (\n"
    '  echo %%TAG%% EVAL_SHARD_LOG_HAS_NO_CLOCK_FIX_BANNER %%DATE%% %%TIME%% >> "%%LOG%%"\n'
    '  echo clock-fix banner missing from the eval shard log > "%%DIR%%\\EVAL_BANNER_MISSING.txt"\n'
    ")\n"
    "copy /y scratchpad\\eval_shards\\shard_s0.log \"%%DIR%%\\eval_shard_%%STAMP%%.log\" >nul 2>&1\n" % T)

PIN = ("REM ---- Branch from an INTERMEDIATE w10lf WS checkpoint. Under WSD the stable phase is at constant\n"
       "REM LR, so this IS a shorter-budget run. Default-unset elsewhere.\n"
       "set RWKV_DECAY_FROM_STEP=%WSSTEPS%\n\n")

PIN_GUARD = (
    "REM The pin must have been HONOURED, or this is w10lf's own decay under another name.\n"
    'findstr /C:"[decay-from] pinned to" "%DIR%\\dsetup_%STAMP%.log" >nul\n'
    "if not %ERRORLEVEL%==0 (\n"
    '  echo %TAG% DECAY_FROM_STEP was not honoured %DATE% %TIME% >> "%LOG%"\n'
    '  echo DONE_EXIT_55 %DATE% %TIME% >> "%LOG%"\n'
    "  exit /b 55\n"
    ")\n\n")


def header(tag, what, dry):
    h = ("@echo off\n"
         "REM %s -- %s. Generated by scratchpad/w10lf/mk_lf_branch.py from arm 1's runner, with the\n"
         "REM substitutions mk_w10lf.py applies to w10lf's decay and eval: decay from w10lf's LEAK-FREE WS,\n"
         "REM RWKV_CLOCK_AT_PREV_ANSWER=%s in the decay and the eval. CONTROL = w10lf. Checkpoints land in\n"
         "REM scratchpad/w10lf, the decay's source folder. Re-based after phase L said MATERIAL (2026-09-11).\n"
         % (tag, what, T))
    if dry:
        h += "REM DRY RUN: 0.05 epochs of decay and 20 eval users -- plumbing only, never a result.\n"
    return h


def build(mode, arg, ep, dry):
    if mode == "budget":
        ws_step = int(arg)
        assert ws_step % 1000 == 0 and 1000 <= ws_step < 109350, ws_step
        base = "lfb%d" % round(ws_step / GROUPS)
        what = ("BUDGET-CURVE point: a %g-epoch decay from w10lf_ws_%d (%.2f epochs of WS), 300-user eval"
                % (ep, ws_step, ws_step / GROUPS))
    else:
        ws_step = 109350
        base = "lfd%g" % ep
        what = "DECAY-LENGTH branch: a %g-epoch decay from w10lf_ws_109350, full VAL-half eval" % ep
    tag = base + ("dry" if dry else "")
    dep = 0.05 if dry else ep
    steps = int(dep * GROUPS)

    arm1 = rd(ARM1)
    body = arm1[arm1.index("setlocal"):]
    for pat, rep in [
        (r"set DIR=C:\\Users\\Andrew\\rwkv-anki-autoresearch\\scratchpad\\w10plain",
         r"set DIR=C:\\Users\\Andrew\\rwkv-anki-autoresearch\\scratchpad\\" + tag),
        (r"set LOG=%DIR%\\w10plain\.log", r"set LOG=%DIR%\\" + tag + ".log"),
        (r"set TAG=w10plain", "set TAG=" + tag),
        (r"scratchpad/write_decay_setup\.py scratchpad/ws10 w10_ws w10p_d",
         "scratchpad/write_decay_setup.py scratchpad/w10lf w10lf_ws " + tag + "_d"),
        (r'findstr /C:"w10_ws_%WSSTEPS%"', 'findstr /C:"w10lf_ws_%WSSTEPS%"'),
        (r"scratchpad\\ws10\\w10p_d_%STEPS%\.pth", r"scratchpad\\w10lf\\" + tag + "_d_%STEPS%.pth"),
        (r"scratchpad/write_eval_toml\.py scratchpad/w10plain w10p_d",
         "scratchpad/write_eval_toml.py scratchpad/" + tag + " " + tag + "_d"),
        (r"echo ===== w10plain START", "echo ===== " + tag + " START"),
        (r"set STEPS=21870", "set STEPS=%d" % steps),
        (r"(1 5000 )2\.0( 1e-3 65536)", r"\g<1>%s\g<2>" % fmt(dep)),
        (r"set WSSTEPS=109350", "set WSSTEPS=%d" % ws_step),
    ]:
        body = sub(body, pat, rep)
    users = "5001 5020" if dry else ("5001 5300" if mode == "budget" else "5001 7500")
    body = sub(body, r"(write_eval_toml\.py.+?)5001 7500(\s*>)", r"\g<1>%s\g<2>" % users)

    a = "set RWKV_KD_ALPHA=\n"
    if body.count(a) != 1:
        sys.exit("REFUSING: the KD_ALPHA anchor is not unique")
    body = body.replace(a, a + "\nREM ---- THE LEVER: the query-row clock-leak fix, in decay, validation and eval ----\n"
                        "set RWKV_CLOCK_AT_PREV_ANSWER=%s\n" % T)
    if mode == "budget":
        a = 'if not exist "%DIR%" mkdir "%DIR%"'
        if body.count(a) != 1:
            sys.exit("REFUSING: the mkdir anchor is not unique")
        body = body.replace(a, PIN + a, 1)
        a = "REM ---- PHASE C"
        if body.count(a) != 1:
            sys.exit("REFUSING: the phase-C anchor is not unique")
        body = body.replace(a, PIN_GUARD + a, 1)
    a = "REM KD must be ABSENT here"
    if body.count(a) != 1:
        sys.exit("REFUSING: the KD-absent anchor is not unique")
    body = body.replace(a, guard("decay_%STAMP%.log", 56, 57, "DECAY") + a)
    a = 'echo %TAG% EVAL_OK %TIME% >> "%LOG%"\n'
    if body.count(a) != 1:
        sys.exit("REFUSING: the EVAL_OK anchor is not unique")
    body = body.replace(a, a + EVAL_POSTCHECK)
    return tag, header(tag, what, dry) + body, steps, ws_step


def check(tag, out, mode, steps, ws_step, dry):
    code = code_of(out)
    stale = ["w10plain", "w10p_d", "w10_ws", "scratchpad/ws10", "scratchpad\\ws10"]
    if steps != 21870:
        stale.append("21870")
    if mode == "budget" or dry:
        stale.append("5001 7500")
    for tok in stale:
        bad = [l for l in code.split(NL) if tok in l]
        if bad:
            sys.exit(f"REFUSING ({tag}): '{tok}' survives on a non-REM line:\n  " + "\n  ".join(bad[:3]))
    e, e_a1, e_lf = env_block(out), env_block(rd(ARM1)), env_all(rd(LF))
    diff = {k: (v, e.get(k)) for k, v in e_a1.items() if e.get(k) != v}
    if diff:
        sys.exit(f"REFUSING ({tag}): env differs from arm 1: {diff}")
    want_extra = {"RWKV_CLOCK_AT_PREV_ANSWER"} | ({"RWKV_DECAY_FROM_STEP"} if mode == "budget" else set())
    extra = set(e) - set(e_a1)
    if extra != want_extra:
        sys.exit(f"REFUSING ({tag}): extra env {sorted(extra)}, want {sorted(want_extra)}")
    lf_decay = {k: v for k, v in e_lf.items() if k != "RWKV_RESUME_SKIP_GROUPS"}
    d2 = {k: (v, e.get(k)) for k, v in lf_decay.items() if k in e_a1 or k == "RWKV_CLOCK_AT_PREV_ANSWER"
          if e.get(k) != v}
    if d2:
        sys.exit(f"REFUSING ({tag}): env differs from w10lf's runner: {d2}")
    if e["RWKV_CLOCK_AT_PREV_ANSWER"] != T:
        sys.exit("REFUSING: the lever has the wrong value")
    if code.index("set RWKV_CLOCK_AT_PREV_ANSWER=") > code.index("rwkv.train_rwkv"):
        sys.exit("REFUSING: the lever is set after training starts")
    for need in ("scratchpad/write_decay_setup.py scratchpad/w10lf w10lf_ws %s_d" % tag,
                 "scratchpad\\w10lf\\%s_d_%%STEPS%%.pth" % tag, "set STEPS=%d" % steps,
                 "set WSSTEPS=%d" % ws_step, 'findstr /C:"ON: T=%s s" "%%DIR%%\\decay_%%STAMP%%.log"' % T):
        if need not in code:
            sys.exit(f"REFUSING ({tag}): missing {need!r}")
    if (mode == "budget") != ("set RWKV_DECAY_FROM_STEP=%WSSTEPS%" in code):
        sys.exit(f"REFUSING ({tag}): the WS-step pin is present iff budget mode")
    codes = re.findall(r"echo DONE_EXIT_(\d+) ", out)
    if len(codes) != len(set(codes)):
        sys.exit(f"REFUSING ({tag}): duplicate exit codes {sorted(int(c) for c in codes)}")
    if out.index("echo DONE_EXIT_0 ") > out.rindex("endlocal"):
        sys.exit("REFUSING: the terminal marker follows endlocal")
    return sorted(int(c) for c in codes)


def main():
    if len(sys.argv) < 3 or sys.argv[1] not in ("budget", "decay"):
        raise SystemExit(__doc__)
    mode, arg = sys.argv[1], sys.argv[2]
    if mode == "budget":
        ep = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
    else:
        ep = float(arg)
        assert 0.25 <= ep <= 6.0 and ep != 2.0, "decay epochs outside range, or 2.0 (that is w10lf itself)"
    for dry in (False, True):
        tag, out, steps, ws_step = build(mode, arg, ep, dry)
        codes = check(tag, out, mode, steps, ws_step, dry)
        path = "scratchpad/%s/run_%s.cmd" % (tag, tag)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        io.open(path, "w", encoding="ascii", newline="\r\n").write(out)
        print("wrote %s  (WS step %d, decay %d steps, exit codes %s)" % (path, ws_step, steps, codes))


if __name__ == "__main__":
    main()
