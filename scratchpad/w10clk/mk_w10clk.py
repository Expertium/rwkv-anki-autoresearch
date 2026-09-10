"""Generate the LEAK-FREE decay branch runner from ARM 1's runner, so the two differ in one thing.

THE ARM. ws10's WS-final -> the same 2-epoch (21,870-step) decay as arm 1 -> the same rectified
VAL-half eval, with ONE change: RWKV_CLOCK_AT_PREV_ANSWER=<T> (rwkv/clock_fix.py) in the decay, its
in-run validation, and the eval. Query rows and PAVA probe rows are moved to "the card appeared at
the previous answer" when their gap is an in-session one, and the ahead label time loses the target
review's in-session gap. So `w10clk - w10plain` is what the query-row clock leak was worth to the
metric at a matched training history, and w10clk's own number is the first deploy-consistent one
this lineage has produced.

⚠ It is a DECAY-ONLY branch: the 10-epoch WS phase was trained with the leak, and 2 epochs of
leak-free decay may not undo all of the reliance the WS learned. So w10clk is a PESSIMISTIC
estimate of a fully leak-free model -- a leak-free WS would score at least as well -- and the
difference to arm 1 is an UPPER bound on what the leak was worth once training can adapt. Phase L
(the eval-only counterfactual on arm 1's unchanged checkpoint) is the no-adaptation bound above it.

Both directions are asserted, as in mk_cmixgraft.py: every substitution must fire, and nothing of
arm 1's identity may survive outside REM lines.

Usage: python scratchpad/w10clk/mk_w10clk.py [T_seconds]      (default 1800)
"""
import io
import os
import re
import sys

os.chdir(r"C:\Users\Andrew\rwkv-anki-autoresearch")

T = sys.argv[1] if len(sys.argv) > 1 else "1800"
assert float(T) > 0, T
SRC = "scratchpad/w10plain/run_w10plain.cmd"
OUT = "scratchpad/w10clk/run_w10clk.cmd"

HEADER = r"""@echo off
REM w10clk -- THE LEAK-FREE DECAY BRANCH. Generated from run_w10plain.cmd (arm 1) by
REM scratchpad/w10clk/mk_w10clk.py; arm 1 is the exact control.
REM
REM THE LEVER, and it is the ONLY difference from arm 1: RWKV_CLOCK_AT_PREV_ANSWER=%T% in the decay,
REM its validation and the eval (rwkv/clock_fix.py). The query row (imm) and the PAVA probe rows
REM (rectified ahead) predict at SHOW time, and their clock columns carried the current review's
REM capped-duration excess; they are moved to the previous answer when the gap is in-session, and
REM the ahead label time loses the target review's in-session gap. Parameters are unchanged.
REM
REM A decay-only branch off a WS trained WITH the leak, so its number is a pessimistic estimate
REM of a fully leak-free model. PREREG: scratchpad/w10clk/PREREG.md.
"""

SUBS = [
    (r"set DIR=C:\\Users\\Andrew\\rwkv-anki-autoresearch\\scratchpad\\w10plain",
     r"set DIR=C:\\Users\\Andrew\\rwkv-anki-autoresearch\\scratchpad\\w10clk"),
    (r"set LOG=%DIR%\\w10plain\.log", r"set LOG=%DIR%\\w10clk.log"),
    (r"set TAG=w10plain", r"set TAG=w10clk"),
    (r"scratchpad/write_decay_setup\.py scratchpad/ws10 w10_ws w10p_d",
     r"scratchpad/write_decay_setup.py scratchpad/ws10 w10_ws w10lk_d"),
    (r"scratchpad\\ws10\\w10p_d_%STEPS%\.pth", r"scratchpad\\ws10\\w10lk_d_%STEPS%.pth"),
    (r"scratchpad/write_eval_toml\.py scratchpad/w10plain w10p_d",
     r"scratchpad/write_eval_toml.py scratchpad/w10clk w10lk_d"),
    (r"echo ===== w10plain START", r"echo ===== w10clk START"),
]

LEVER = (
    "\nREM ---- THE LEVER: the query-row clock-leak fix, in train, validation and eval ----\n"
    "set RWKV_CLOCK_AT_PREV_ANSWER=%s\n" % T)

DECAY_GUARD = (
    "REM The fix must have REACHED the fetch workers and RUN there. clock_fix prints its ON banner at\n"
    "REM import in every process, and the chunk-count line only from apply_to_sample -- i.e. only if\n"
    "REM prepare() really called it. A banner alone proves the flag parsed, never that it was used.\n"
    'findstr /C:"ON: T=%s s" "%%DIR%%\\decay_%%STAMP%%.log" >nul\n'
    "if not %%ERRORLEVEL%%==0 (\n"
    '  echo %%TAG%% CLOCK_FIX_NOT_ON %%DATE%% %%TIME%% >> "%%LOG%%"\n'
    '  echo DONE_EXIT_56 %%DATE%% %%TIME%% >> "%%LOG%%"\n'
    "  exit /b 56\n"
    ")\n"
    'findstr /C:"s chunks 1 query rows" "%%DIR%%\\decay_%%STAMP%%.log" >nul\n'
    "if not %%ERRORLEVEL%%==0 (\n"
    '  echo %%TAG%% CLOCK_FIX_NEVER_RAN %%DATE%% %%TIME%% >> "%%LOG%%"\n'
    '  echo DONE_EXIT_57 %%DATE%% %%TIME%% >> "%%LOG%%"\n'
    "  exit /b 57\n"
    ")\n" % T)

EVAL_POSTCHECK = (
    "REM Non-fatal post-check: the eval's shard log must carry the ON banner too. Non-fatal on\n"
    "REM purpose -- a guard that can destroy a finished ~3 h eval on a banner mismatch is worse than\n"
    "REM what it guards; a marker file makes a miss impossible to overlook at verdict time.\n"
    'findstr /C:"ON: T=%s s" scratchpad\\eval_shards\\shard_s0.log >nul 2>&1\n'
    "if not %%ERRORLEVEL%%==0 (\n"
    '  echo %%TAG%% EVAL_SHARD_LOG_HAS_NO_CLOCK_FIX_BANNER %%DATE%% %%TIME%% >> "%%LOG%%"\n'
    '  echo clock-fix banner missing from the eval shard log > "%%DIR%%\\EVAL_BANNER_MISSING.txt"\n'
    ")\n"
    "copy /y scratchpad\\eval_shards\\shard_s0.log \"%%DIR%%\\eval_shard_%%STAMP%%.log\" >nul 2>&1\n" % T)


def main():
    src = io.open(SRC, encoding="utf-8", newline="").read().replace("\r\n", "\n")
    body = src[src.index("setlocal"):]
    for pat, rep in SUBS:
        body, n = re.subn(pat, rep, body)
        if n == 0:
            sys.exit(f"REFUSING: substitution never fired: {pat}")

    a = "set RWKV_KD_ALPHA=\n"
    if body.count(a) != 1:
        sys.exit("REFUSING: the KD-clear anchor is not unique")
    body = body.replace(a, a + LEVER)

    a = "REM KD must be ABSENT here"
    if body.count(a) != 1:
        sys.exit("REFUSING: the KD-absent anchor is not unique")
    body = body.replace(a, DECAY_GUARD + a)

    a = 'echo %TAG% EVAL_OK %TIME% >> "%LOG%"\n'
    if body.count(a) != 1:
        sys.exit("REFUSING: the EVAL_OK anchor is not unique")
    body = body.replace(a, a + EVAL_POSTCHECK)

    out = HEADER.replace("%T%", T) + body

    code = "\n".join(l for l in out.split("\n") if not l.strip().upper().startswith("REM"))
    for tok in ("w10plain", "w10p_d"):
        bad = [l for l in code.split("\n") if tok in l]
        if bad:
            sys.exit(f"REFUSING: '{tok}' survives on a non-REM line:\n  " + "\n  ".join(bad[:3]))
    if code.count("set RWKV_CLOCK_AT_PREV_ANSWER=%s\n" % T) != 1:
        sys.exit("REFUSING: the lever is not set exactly once")
    if code.index("set RWKV_CLOCK_AT_PREV_ANSWER=") > code.index("rwkv.train_rwkv"):
        sys.exit("REFUSING: the lever is set after the decay starts")
    if "set RWKV_CLOCK_AT_PREV_ANSWER=\n" in code:
        sys.exit("REFUSING: the lever is cleared somewhere -- the eval must run with it")
    codes = re.findall(r"echo DONE_EXIT_(\d+) ", out)
    if len(codes) != len(set(codes)):
        sys.exit(f"REFUSING: duplicate exit codes {sorted(int(c) for c in codes)}")
    if out.index("echo DONE_EXIT_0 ") > out.rindex("endlocal"):
        sys.exit("REFUSING: the terminal marker follows endlocal")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    io.open(OUT, "w", encoding="ascii", newline="\r\n").write(out)
    raw = open(OUT, "rb").read()
    print(f"wrote {OUT}  ({raw.count(b'\n')} lines, {raw.count(b'\r\n')} CRLF)")
    print(f"  T = {T} s, prefix w10lk_d, exit codes {sorted(int(c) for c in codes)}")


if __name__ == "__main__":
    main()
