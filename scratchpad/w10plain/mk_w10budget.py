"""Generate a BUDGET-CURVE point: decay from an INTERMEDIATE ws10 checkpoint.

THE QUESTION. The endgame rests on "+0.0042 at 10x", which is a log-linear extrapolation from a
single measured 3x step (+0.002, 2026-08-11 calibration). Arm 1 tests the endpoint. It does not
test the SHAPE, and the shape is what says whether budget is a lever worth pulling again or one
that has just been exhausted. Under WSD the stable phase runs at constant LR, so decaying from an
intermediate step IS a shorter-budget run -- one expensive WS buys several budget points for the
price of their decays.

WHY NOT USE realcyc AS THE LOW POINT. realcyc's warmup is 400 steps of a 10,935-step stable phase
(3.7%); ws10's is 400 of 109,350 (0.37%). Points taken from THIS run share their warmup, their
data order and their seed, so they are comparable to each other and to arm 1 in a way realcyc is
not. Use realcyc as corroboration, never as the curve's first point.

THE EVAL IS DELIBERATELY SMALL -- 300 users, ~20 min instead of ~2.9 h. A budget curve is a COARSE
question (is the gain linear or saturating, at a scale of 0.002-0.004), and the 200-user lesson
says subsets rank coarsely and must not settle sub-0.001 effects. This is the former. Arm 1 must be
re-scored on the SAME 300 users for the comparison; the curve's own points are all 300-user.

⚠ So a point from this generator is NOT a gate candidate and must never be logged as one.

Usage: python scratchpad/w10plain/mk_w10budget.py <ws_step> [decay_epochs]
       e.g. 21870 (2 ep of WS), 54675 (5 ep). decay_epochs defaults to 2.0, matching arm 1.
"""
import io
import os
import sys

B = chr(92)

if len(sys.argv) < 2:
    raise SystemExit(__doc__)
ws_step = int(sys.argv[1])
ep = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
assert ws_step % 1000 == 0, "checkpoints exist only every 1000 steps (VALIDATE_EVERY)"
assert 1000 <= ws_step <= 109350
# Checkpoints land on multiples of 1000, not on epoch boundaries (10,935 steps), so a "5-epoch"
# point is really 5.03. The tag carries the ROUNDED epoch count and the runner logs the exact
# step, because a curve is read by its x-values and a tag that lies about them is worse than none.
tag = "w10b%d" % round(ws_step / 10935.0)
steps = int(round(ep * 10935))

os.chdir(r"C:\Users\Andrew\rwkv-anki-autoresearch")
src = io.open("scratchpad/w10plain/run_w10plain.cmd", encoding="utf-8", newline="").read()
src = src.replace("\r\n", "\n")
assert "set STEPS=21870" in src and "5001 7500" in src

src = src.replace("scratchpad" + B + "w10plain", "scratchpad" + B + tag)
src = src.replace("scratchpad/w10plain", "scratchpad/" + tag)
src = src.replace("set LOG=%DIR%" + B + "w10plain.log", "set LOG=%DIR%" + B + tag + ".log")
src = src.replace("set TAG=w10plain", "set TAG=" + tag)
src = src.replace("w10p_d", tag + "_d")
src = src.replace("w10plain START", tag + " START")
src = src.replace("set WSSTEPS=109350", "set WSSTEPS=%d" % ws_step)
if ep != 2.0:
    src = src.replace("1 5000 2.0 1e-3 65536", "1 5000 %g 1e-3 65536" % ep)
    src = src.replace("set STEPS=21870", "set STEPS=%d" % steps)
# 300-user eval: coarse by design, see the docstring.
src = src.replace("5001 7500", "5001 5300")

PIN = """
REM ---- Branch from an INTERMEDIATE WS checkpoint. Under WSD the stable phase is at constant LR,
REM so this IS a shorter-budget run. Default-unset elsewhere, so no other runner is affected.
set RWKV_DECAY_FROM_STEP=%WSSTEPS%
"""
anchor = 'if not exist "%DIR%" mkdir "%DIR%"'
assert src.count(anchor) == 1
src = src.replace(anchor, PIN + "\n" + anchor, 1)

GUARD = """
REM The pin must have been HONOURED. write_decay_setup silently takes the latest checkpoint when
REM the variable is unset or misspelt, which would produce arm 1 again under a different tag --
REM a duplicate wearing a curve point's name, and nothing downstream would notice.
findstr /C:"[decay-from] pinned to" "%DIR%{B}dsetup_%STAMP%.log" >nul
if not %ERRORLEVEL%==0 (
  echo %TAG% DECAY_FROM_STEP was not honoured %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_55 %DATE% %TIME% >> "%LOG%"
  exit /b 55
)
""".replace("{B}", B)
anchor2 = "REM ---- PHASE C"
assert src.count(anchor2) == 1
src = src.replace(anchor2, GUARD + "\n" + anchor2, 1)

body = "\n".join(ln for ln in src.split("\n") if not ln.strip().startswith("REM"))
for tok in ("w10plain", "w10p_d", "5001 7500"):
    assert tok not in body, "stale token %r survived outside a REM" % tok
assert body.count("DONE_EXIT_0") == 1
for need in ("set RWKV_DECAY_FROM_STEP=%WSSTEPS%", "set WSSTEPS=%d" % ws_step,
             "5001 5300", "DONE_EXIT_55", "scratchpad/ws10 w10_ws " + tag + "_d"):
    assert need in body, "missing %r" % need
os.makedirs("scratchpad/" + tag, exist_ok=True)
out = "scratchpad/%s/run_%s.cmd" % (tag, tag)
io.open(out, "w", newline="\r\n").write(src)
print("wrote %s  (WS %d steps = %.2f ep, decay %g ep, 300-user eval)"
      % (out, ws_step, ws_step / 10935.0, ep))
