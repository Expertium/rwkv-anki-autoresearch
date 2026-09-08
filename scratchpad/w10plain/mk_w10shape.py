"""Generate the decay-SHAPE branch off the shared ws10 checkpoint (queue item 4).

iter 56 ran `RWKV_DECAY_SHAPE=linear` at 1.25 epochs and was sub-bar but REAL: ahead +0.000057
(inside the +/-7.5e-5 floor, so its reality rests on rank consistency at p=6e-12) and imm +0.000104
(clears both). It was not queued further because, stacked on iter 53 under perfect additivity, it
still failed the ahead bar -- an arithmetic argument at THAT budget.

Why it comes back now: iter 56's decay was 10,935 steps annealing from a 1-epoch stable phase.
Arm 1's is 21,870 steps annealing from a 10-epoch one, so the shape governs eight times as much
training and starts from a differently-conditioned point. The lever is also free -- no parameters,
no deploy change, and the branch is decay-only.

⚠ It is queued AFTER decay length deliberately. Length is the bigger and better-motivated lever
and shape is a refinement of it; if length moves, the shape branch should be regenerated on the
winning length rather than on 2.0.

Usage: python scratchpad/w10plain/mk_w10shape.py [decay_epochs]      (default 2.0)
"""
import io
import os
import sys

B = chr(92)

ep = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
tag = "w10lin" if ep == 2.0 else "w10lin%g" % ep
steps = int(round(ep * 10935))

os.chdir(r"C:\Users\Andrew\rwkv-anki-autoresearch")
src = io.open("scratchpad/w10plain/run_w10plain.cmd", encoding="utf-8", newline="").read()
src = src.replace("\r\n", "\n")
assert "1 5000 2.0 1e-3 65536" in src and "set STEPS=21870" in src

src = src.replace("scratchpad" + B + "w10plain", "scratchpad" + B + tag)
src = src.replace("scratchpad/w10plain", "scratchpad/" + tag)
src = src.replace("set LOG=%DIR%" + B + "w10plain.log", "set LOG=%DIR%" + B + tag + ".log")
src = src.replace("set TAG=w10plain", "set TAG=" + tag)
src = src.replace("w10p_d", tag + "_d")
src = src.replace("w10plain START", tag + " START")
if ep != 2.0:
    src = src.replace("1 5000 2.0 1e-3 65536", "1 5000 %g 1e-3 65536" % ep)
    src = src.replace("set STEPS=21870", "set STEPS=%d" % steps)

SHAPE = """
REM ---- The ONLY difference from arm 1: the decay phase's LR SHAPE (iter 56's lever).
set RWKV_DECAY_SHAPE=linear
"""
anchor = 'if not exist "%DIR%" mkdir "%DIR%"'
assert src.count(anchor) == 1
src = src.replace(anchor, SHAPE + "\n" + anchor, 1)

GUARD = """
REM The flag must actually reach the trainer: a run that silently used the default cosine would
REM look like a clean null and would be indistinguishable from arm 1 by construction.
findstr /C:"decay shape = linear" "%DIR%{B}decay_%STAMP%.log" >nul
if not %ERRORLEVEL%==0 (
  echo %TAG% DECAY_SHAPE did not reach the trainer %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_54 %DATE% %TIME% >> "%LOG%"
  exit /b 54
)
""".replace("{B}", B)
anchor2 = "REM ---- PHASE C"
assert src.count(anchor2) == 1
src = src.replace(anchor2, GUARD + "\n" + anchor2, 1)

body = "\n".join(ln for ln in src.split("\n") if not ln.strip().startswith("REM"))
for tok in ("w10plain", "w10p_d"):
    assert tok not in body, "stale token %r survived outside a REM" % tok
assert body.count("DONE_EXIT_0") == 1
for need in ("set RWKV_DECAY_SHAPE=linear", "decay shape = linear",
             "scratchpad/ws10 w10_ws " + tag + "_d",
             "set WSSTEPS=109350", "DONE_EXIT_54"):
    assert need in body, "missing %r" % need
os.makedirs("scratchpad/" + tag, exist_ok=True)
out = "scratchpad/%s/run_%s.cmd" % (tag, tag)
io.open(out, "w", newline="\r\n").write(src)
print("wrote %s  (linear decay, %g epochs = %d steps)" % (out, ep, steps))
