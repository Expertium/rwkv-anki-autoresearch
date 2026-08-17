"""Generate scratchpad/iter57_decayshape/run_iter57.cmd from iter 52's DECAY-ONLY runner.

Iter 57 changes the decay phase's LR SHAPE from the historical `1 - sin(pi x/2)` to `linear`
(RWKV_DECAY_SHAPE=linear). Decay-only, warm-started from the champion's own i45_ws_10935, so it is
a ~3.5 h run rather than ~5.5 h -- and the lever CANNOT touch WS by construction, which makes it a
perfectly controlled single variable.

★ THE SCREEN THAT RE-SPECIFIED THIS LEVER (2026-08-17, pure arithmetic, no GPU). PROPOSALS.md rank 5
said "cosine -> linear / 1-sqrt". That was wrong twice:
  1. We are NOT on a cosine. `1 + cos(pi/2*(1+x))` is identically `1 - sin(pi/2*x)` (agreement
     4.4e-16) -- far more aggressive than a standard cosine. The function's NAME (`cosine_down`)
     is what made it look like one.
  2. `1-sqrt(x)` is not an alternative DIRECTION. It has LR mass 0.3333 vs the current 0.3634 and is
     IDENTICAL at the midpoint (0.293 both). It sits on top of where we already are.
So the only informative direction is MORE LR mass: linear and standard cosine are both 0.5000, i.e.
1.376x current.

★ AND THAT GIVES A BETTER MOTIVATION THAN THE ORIGINAL. iter 34 adopted decay_ratio 0.25 -> 1.0 (4x
the decay steps) and it was the phase's largest gain -- evidence that more decay-LR-mass helps.
`decay_ratio` buys mass by spending WALL-CLOCK; the shape buys 1.376x of it at an IDENTICAL step
count, for free. That is why this is worth a run; "shape is unexplored" was not.

⚠ Pre-registered counter-hypothesis: because linear and standard cosine have the SAME mass (0.5) but
different profiles, a win here is a MASS result, not a shape result, until the two are compared
against each other. If linear wins, the follow-up that separates them is standard cosine at matched
mass -- do not claim "shape matters" from this run alone.

Output guards mirror mk55.py's (see the mk53/mk54 deletion incident).
"""
import io
import os
import re

BS = chr(92)
SRC = "scratchpad/iter52_kdalpha/run_iter52.cmd"
DST = "scratchpad/iter57_decayshape/run_iter57.cmd"

s = io.open(SRC, encoding="ascii", newline="").read().replace("\r\n", "\n")

s = s.replace("scratchpad/iter52_kdalpha", "scratchpad/iter57_decayshape")
s = s.replace("scratchpad" + BS + "iter52_kdalpha", "scratchpad" + BS + "iter57_decayshape")
s = s.replace("iter52_kdalpha", "iter57_decayshape")
s = s.replace("i52_d", "i57_d").replace("i52_decay.toml", "i57_decay.toml")
s = s.replace("i52_eval.toml", "i57_eval.toml")
s = s.replace("iter52.log", "iter57.log")
s = s.replace("ITER52", "ITER57").replace("ITER 52", "ITER 57")

# --- the lever: keep the CHAMPION's decay alpha (0.5), change the LR shape instead ---
OLD_LEVER = ("REM ============ PHASE A: decay from the champion's WS-final, KD alpha 0.9 ============\n"
             "set RWKV_MUON_BATCHED=1\n"
             "set RWKV_NO_JIT=1\n"
             "set RWKV_QAT_COMPILE=1\n"
             "set RWKV_KD_MIX=%DUMP%:%WSSTEPS%\n"
             "set RWKV_KD_ALPHA=0.9")
NEW_LEVER = ("REM ====== PHASE A: decay from the champion's WS-final, LINEAR LR decay shape ======\n"
             "set RWKV_MUON_BATCHED=1\n"
             "set RWKV_NO_JIT=1\n"
             "set RWKV_QAT_COMPILE=1\n"
             "set RWKV_KD_MIX=%DUMP%:%WSSTEPS%\n"
             "REM KD alpha stays at the CHAMPION's 0.5 -- iter 52 is the run that moves it, and\n"
             "REM moving both at once would make neither attributable.\n"
             "set RWKV_KD_ALPHA=0.5\n"
             "REM ---- THE LEVER: 1-sin(pi x/2) (mass 0.3634) becomes 1-x (mass 0.5000, 1.376x) ----\n"
             "set RWKV_DECAY_SHAPE=linear")
assert OLD_LEVER in s, "iter 52's PHASE A block not found verbatim"
s = s.replace(OLD_LEVER, NEW_LEVER)

# the lever must be CLEARED before the eval, like every other training-only env var
s = s.replace("set RWKV_KD_ALPHA=\n", "set RWKV_KD_ALPHA=\nset RWKV_DECAY_SHAPE=\n")

HEADER = [
    "@echo off",
    "REM ===========================================================================================",
    "REM ITER 57: THE DECAY PHASE'S LR SHAPE (RWKV_DECAY_SHAPE=linear). Decay-only, ~3.5 h.",
    "REM Family: schedule. PROPOSALS.md rank 5, RE-SPECIFIED 2026-08-17 by a pure-arithmetic screen.",
    "REM",
    "REM WHAT THE SCREEN FOUND, and it corrects the proposal twice over:",
    "REM   1. WE ARE NOT ON A COSINE. train_rwkv.py's `cosine_down` computes",
    "REM      1 + cos(pi/2*(1+x)), which is IDENTICALLY 1 - sin(pi/2*x) (agreement 4.4e-16) --",
    "REM      much more aggressive than a standard cosine. The NAME is what made it look like one.",
    "REM   2. 1-sqrt(x) IS NOT AN ALTERNATIVE DIRECTION. Its integrated LR multiplier is 0.3333 vs",
    "REM      the current 0.3634, and the two are IDENTICAL at the midpoint (0.293). It sits on top",
    "REM      of where we already are.",
    "REM   So the only informative direction is MORE LR mass: linear and standard cosine are both",
    "REM   0.5000, i.e. 1.376x current.",
    "REM",
    "REM ** THE MOTIVATION IS THEREFORE NOT 'shape is unexplored'. iter 34 adopted decay_ratio",
    "REM 0.25 to 1.0 -- 4x the decay steps -- and it was the phase's largest gain, i.e. evidence",
    "REM that more decay-LR-mass helps. decay_ratio buys mass by spending WALL-CLOCK; the shape buys",
    "REM 1.376x of it at an IDENTICAL step count, for free. That is the argument.",
    "REM",
    "REM ** PRE-REGISTERED COUNTER-HYPOTHESIS: linear and standard cosine have the SAME mass (0.5)",
    "REM and different profiles, so a win here is a MASS result, not a SHAPE result, until those two",
    "REM are compared. If this wins, the follow-up that separates them is cosine at matched mass.",
    "REM Do NOT claim 'shape matters' from this run alone.",
    "REM",
    "REM PERFECTLY CONTROLLED BY CONSTRUCTION: decay-only, warm-started from the champion's own",
    "REM i45_ws_10935, so WS is literally the champion's and the lever cannot touch it. KD alpha",
    "REM stays at the champion's 0.5 (iter 52 is the run that moves it).",
    "REM",
    "REM GATE: PLAIN basis vs iter 45 == 0.297697 ahead / 0.265375 imm on the VAL half, BOTH-modes",
    "REM rule. Params UNCHANGED at 558,212 -- a schedule has no weights. No deploy debt.",
    "REM The default RWKV_DECAY_SHAPE path was verified BIT-IDENTICAL to the historical schedule",
    "REM over all 2,734 decay steps before this was queued, because a chain's later phases import",
    "REM whatever is on disk THEN and iter 53 was mid-flight.",
    "REM",
    "REM Do NOT edit this file while it runs (iters 43 and 46 died that way; git checkout is not a",
    "REM safe undo, because line endings shift the byte offset cmd.exe resumes from).",
    "REM No angle brackets or arrows in REM lines -- cmd parses redirection before REM.",
    "REM ===========================================================================================",
]
body = s[s.index("setlocal"):]
out = "\n".join(HEADER) + "\n" + body

# ---------------- OUTPUT GUARDS (the class mk53/mk54 lacked) ----------------
# EXECUTED body lines only. A REM inside the body may legitimately reference a sibling iteration
# ("KD alpha stays at 0.5 -- iter 52 is the run that moves it"); what must never survive is a stale
# identifier in a line cmd.exe actually runs.
body_exec = "\n".join(
    l for l in out[out.index("setlocal"):].split("\n")
    if not l.strip().upper().startswith("REM")
).lower()
for stale in ("iter52", "iter 52", "i52_", "kdalpha"):
    assert stale not in body_exec, "stale iter-52 text leaked into an EXECUTED line: " + stale
header_low = out[:out.index("setlocal")].lower()
assert "rem iter 57:" in header_low and "rem iter 52:" not in header_low

assert re.search(r"^cd /d ", out, re.M), "cd /d was deleted"
lines = out.split("\n")
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
    assert setpos[v] < un, f"%{v}% used at line {un + 1} before being set at {setpos[v] + 1}"

assert out.count("set RWKV_DECAY_SHAPE=linear") == 1, "the lever must be set exactly once"
assert out.count("set RWKV_DECAY_SHAPE=") == 2, "the lever must also be CLEARED before the eval"
assert out.count("set RWKV_KD_ALPHA=0.5") == 1 and "set RWKV_KD_ALPHA=0.9" not in out, (
    "decay alpha must stay at the champion's 0.5 -- iter 52 is the run that moves it")
assert out.count("DONE_EXIT_0") == 1
assert "write_decay_setup.py scratchpad/iter45_kddecay i45_ws i57_d" in out or "%SRCREL% i45_ws i57_d" in out, (
    "must decay from the champion's WS-final")
for ln in lines:
    if ln.strip().upper().startswith("REM"):
        bad = [c for c in "<>&|^" if c in ln]
        assert not bad, "redirection char " + str(bad) + " in REM line: " + ln

io.open(DST, "w", encoding="ascii", newline="\r\n").write(out)
print("wrote", DST, len(out), "bytes,", out.count(chr(10)), "lines")
