"""Generate a DECAY-LENGTH branch off the shared ws10 checkpoint.

WHY THIS IS THE QUEUE'S NEXT LEVER, and why it could not be asked before. The 10+2 split was
decided on COST, not on evidence: 6+6 = 89 h, 8+4 = 71 h, 10+2 = 53 h, and CLAUDE.md says in as
many words that the ~+0.0006 attributed to a longer decay is "being SPENT, not disproven" because
iter 34's decay_ratio gain is confounded with a budget change. A shared WS checkpoint turns that
from a 43 h question into a 9 h one: every branch starts from the SAME weights, so decay length is
the only variable and arm 1 is the control.

WHY NOT A REGULARISER INSTEAD. `scratchpad/ws10/gap_trace.py` reads ws10's own validation and
finds the ahead train/val gap FLAT through 4 epochs while validation still improves -- the model
is still fit-limited at 10 epochs, so SAM / wd / dropout remain unmotivated (iter 65's condition
is not met). Levers that improve FIT per step are what pay, and decay length is exactly that.

WHAT IT DOES NOT ANSWER. This varies decay length at a FIXED 10-epoch WS, so total budget varies
with it. That is the operationally relevant question for the endgame ("given this WS checkpoint,
how long should the decay be?"), NOT the fixed-total-budget WS:decay de-confound that CLAUDE.md
leaves as Andrew's call. Do not report one as the other.

Usage: python scratchpad/w10plain/mk_w10decay.py <decay_epochs>     e.g. 1.0, 3.0
"""
import io
import os
import sys

B = chr(92)

if len(sys.argv) != 2:
    raise SystemExit(__doc__)
ep = float(sys.argv[1])
assert 0.25 <= ep <= 6.0, "decay epochs outside a sane range"
tag = "w10d%g" % ep
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
src = src.replace("1 5000 2.0 1e-3 65536", "1 5000 %g 1e-3 65536" % ep)
src = src.replace("set STEPS=21870", "set STEPS=%d" % steps)
src = src.replace(
    "REM ---- PHASE B: the 2-EPOCH decay",
    "REM Decay length is the ONLY difference from arm 1 (w10plain, 2.0 epochs): same WS\n"
    "REM checkpoint, same data, same seed, same env. See mk_w10decay.py for why this is the\n"
    "REM queue's next lever and what it does NOT answer.\n"
    "REM ---- PHASE B: the %g-EPOCH decay" % ep)

body = "\n".join(ln for ln in src.split("\n") if not ln.strip().startswith("REM"))
for tok in ("w10plain", "w10p_d", "21870", "2.0 1e-3"):
    assert tok not in body, "stale token %r survived outside a REM" % tok
assert body.count("DONE_EXIT_0") == 1
for need in ("set STEPS=%d" % steps, "1 5000 %g 1e-3 65536" % ep,
             "scratchpad/ws10 w10_ws " + tag + "_d", "set WSSTEPS=109350"):
    assert need in body, "missing %r" % need
os.makedirs("scratchpad/" + tag, exist_ok=True)
out = "scratchpad/%s/run_%s.cmd" % (tag, tag)
io.open(out, "w", newline="\r\n").write(src)
print("wrote %s  (decay %g epochs = %d steps)" % (out, ep, steps))
