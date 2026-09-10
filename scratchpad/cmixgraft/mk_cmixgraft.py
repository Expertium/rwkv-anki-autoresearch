"""Generate the cmixgraft branch runner from ARM 1's runner, so the two differ in one thing.

The control for this arm is arm 1 exactly -- same WS history, same 21,870-step decay, same env --
so the runner is arm 1's with the capacity lever applied and nothing else touched. Generating it by
substitution from that file is what makes "single variable" checkable instead of asserted.

BOTH DIRECTIONS ARE ASSERTED, because a substitution generator has two independent failure modes
and this repo has now been bitten by each:
  * stale text leaks IN   -- caught by requiring zero occurrences of the source arm's tokens;
  * a line that SHOULD have changed does not -- invisible to the check above, because no
    substitution fires on it and the line reads as unmodified. That is how run_fixc_arm.cmd ended
    up describing itself as evaluating the e2s db while correctly evaluating fixc.
So every substitution is required to fire at least once AND the output is grepped for the source
arm's identity.

Usage: python scratchpad/cmixgraft/mk_cmixgraft.py
"""
import io
import os
import re
import sys

os.chdir(r"C:\Users\Andrew\rwkv-anki-autoresearch")

SRC = "scratchpad/w10plain/run_w10plain.cmd"
OUT = "scratchpad/cmixgraft/run_cmixgraft.cmd"

ARM1_STRIP = ("user_id:0,user_id:1,user_id:2,preset_id:0,preset_id:1,preset_id:2,"
              "deck_id:1,deck_id:2,card_id:1")
GRAFT_STRIP = "deck_id:1,deck_id:2,card_id:1"
PARAMS = 641862

HEADER = """@echo off
REM cmixgraft -- COARSE-STREAM CAPACITY AT THE REAL BUDGET. Generated from run_w10plain.cmd
REM (arm 1) by scratchpad/cmixgraft/mk_cmixgraft.py; arm 1 is the exact control.
REM
REM THE LEVER, and it is the ONLY difference from arm 1: RWKV_STRIP_CMIX drops the six
REM user_id/preset_id entries, restoring those channel mixers. 563,652 -> 641,862 params.
REM Those streams are per-deck and per-user, so the 9 B/card and 27 B/note deploy contract is
REM untouched -- gate 3 has always allowed deck/preset/global to grow.
REM
REM WHY IT COSTS A DECAY AND NOT A 38 h WS PHASE. The graft is function-preserving at init: a
REM channel mixer returns in + dropout(W_v(...)) and RWKV7ChannelMixer zeroes W_v at construction.
REM scratchpad/cmixgraft/expand_ckpt.py writes w10g_ws_109350.pth (ws10's WS-final with the six
REM mixers grown from their d_model=1 dummy shapes), and smoke_graft.py measures 0.000e+00 against
REM arm 1 over 300 deploy predictions with a PERTURBED control at 1.132e-06 to prove non-vacuity.
REM
REM ANDREW 2026-09-10: "It would be nice to try to grow deck/preset/global, but 38 h is a pretty
REM steep price" and "I'd rather not increase card/note state size". Both are satisfied.
REM
REM PREREG: scratchpad/cmixgraft/PREREG.md -- P1 ahead >= +0.0003, P3 the restored W_v must be
REM non-zero at the end or the run is uninterpretable, and the asymmetry: a WIN is decisive, a
REM NULL is weak. It also prices the one second variable, p=0.005 layer dropout on six blocks.
"""

SUBS = [
    (r"set DIR=C:\\Users\\Andrew\\rwkv-anki-autoresearch\\scratchpad\\w10plain",
     r"set DIR=C:\\Users\\Andrew\\rwkv-anki-autoresearch\\scratchpad\\cmixgraft"),
    (r"set LOG=%DIR%\\w10plain\.log", r"set LOG=%DIR%\\cmixgraft.log"),
    (r"set TAG=w10plain", r"set TAG=cmixgraft"),
    (re.escape("set RWKV_STRIP_CMIX=" + ARM1_STRIP), "set RWKV_STRIP_CMIX=" + GRAFT_STRIP),
    (r"scratchpad/write_decay_setup\.py scratchpad/ws10 w10_ws w10p_d",
     r"scratchpad/write_decay_setup.py scratchpad/ws10 w10g_ws w10g_d"),
    (r'findstr /C:"w10_ws_%WSSTEPS%"', r'findstr /C:"w10g_ws_%WSSTEPS%"'),
    (r"scratchpad\\ws10\\w10p_d_%STEPS%\.pth", r"scratchpad\\ws10\\w10g_d_%STEPS%.pth"),
    (r"scratchpad/write_eval_toml\.py scratchpad/w10plain w10p_d",
     r"scratchpad/write_eval_toml.py scratchpad/cmixgraft w10g_d"),
    # The START banner: a line that SHOULD change and, left alone, changes nothing about
    # behaviour -- which is precisely why it is invisible to a stale-token-in check and why the
    # direction-2 grep caught it here on the first run.
    (r"echo ===== w10plain START", r"echo ===== cmixgraft START"),
]


def main():
    src = io.open(SRC, encoding="utf-8", newline="").read().replace("\r\n", "\n")
    body = src[src.index("setlocal"):]

    for pat, rep in SUBS:
        body, n = re.subn(pat, rep, body)
        if n == 0:
            sys.exit(f"REFUSING: substitution never fired: {pat}")

    # The capacity lever changes the parameter count, so guard it. A wrong count means the strip
    # string did not reach the fetch workers -- the exact failure the gen-5 runners guard against.
    guard = (
        '\nREM ---- PHASE 0c: the lever must have REACHED the model. A banner proves a value was\n'
        'REM computed, never that it was used, so guard the CONSEQUENCE: the restored mixers are\n'
        'REM +78,210 params, and a wrong count means RWKV_STRIP_CMIX did not reach the workers.\n'
        'findstr /C:"Trainable parameters: %s" "%%DIR%%\\decay_%%STAMP%%.log" >nul\n'
        'if not %%ERRORLEVEL%%==0 (\n'
        '  echo %%TAG%% WRONG_PARAM_COUNT -- the capacity lever did not reach the model '
        '%%DATE%% %%TIME%% >> "%%LOG%%"\n'
        '  echo DONE_EXIT_55 %%DATE%% %%TIME%% >> "%%LOG%%"\n'
        '  exit /b 55\n'
        ')\n' % PARAMS)
    anchor = 'REM KD must be ABSENT'
    if anchor not in body:
        anchor = 'findstr /C:"[kd-mix] KD ON"'
    body = body.replace(anchor, guard + anchor, 1)

    out = HEADER + body

    # DIRECTION 2: nothing of the source arm's identity may survive outside provenance comments.
    for tok in ("w10plain", "w10p_d"):
        bad = [ln for ln in out.split("\n")
               if tok in ln and not ln.strip().upper().startswith("REM")]
        if bad:
            sys.exit(f"REFUSING: '{tok}' survives on a non-REM line:\n  " + "\n  ".join(bad[:3]))
    if ARM1_STRIP in out:
        sys.exit("REFUSING: arm 1's strip list survives")
    if GRAFT_STRIP not in out:
        sys.exit("REFUSING: the graft strip list is absent")

    io.open(OUT, "w", encoding="ascii", newline="\r\n").write(out)
    d = open(OUT, "rb").read()
    print(f"wrote {OUT}  ({d.count(chr(10).encode())} lines, "
          f"{d.count(chr(13).encode() + chr(10).encode())} CRLF)")
    print(f"  strip: {GRAFT_STRIP}")
    print(f"  loads scratchpad/ws10/w10g_ws_109350.pth, writes w10g_d_*, param guard {PARAMS:,}")


if __name__ == "__main__":
    main()
