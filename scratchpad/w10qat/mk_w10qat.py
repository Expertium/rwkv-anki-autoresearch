"""Generate the w10qat runner: ENDGAME ARM 2, the QAT tax measured at the real budget.

WHY THIS SHAPE. The old plan asked whether arm 2 should be (A) a warm-started QAT fine-tune on
arm 1's FINAL or (B) a second full 10x run with QAT throughout, and recommended A. The shared-WS
design Andrew approved on 2026-09-08 gives a third and strictly better option: run the SAME
2-epoch decay from the SAME WS checkpoint with the QAT env added and nothing else changed.

  * It is how QAT has always been applied here -- decay-only, warm-started -- so it measures the
    cost AS DEPLOYED, which is what the question asks.
  * It avoids the iter-40 lesson (QAT from scratch = +0.0118; it MUST warm-start).
  * And it is a genuinely SINGLE-VARIABLE A/B against arm 1: same weights at the branch point,
    same decay length, same data, same seed. Options A and B could not give that -- A adds ~2
    epochs of extra training on top of arm 1, B changes the whole trajectory.

So arm 2 - arm 1 IS the QAT tax at the 10-epoch budget, with no confound to subtract.

CATALOGS. The gate is `reference/pq_cb_{wkv,shift}_ws10_*.txt`, which do NOT exist yet and are
NOT the gen-5 pair fitted from realcyc. Phase 0 REFUSES to run without them, deliberately: both
deployed catalogs were measured stale on the gen-5 trunk (scratchpad/qat_gen5/FINDING.md -- the
shift one scored WORSE than encoding to zero), a stale catalog passes every shape assert, and
arm 2's whole purpose is to price quantization. Fit them from ws10's own WS-final before launch;
the procedure is ~10 min of CPU and is written down in that FINDING.

Usage: python scratchpad/w10qat/mk_w10qat.py
"""
import io
import os

B = chr(92)  # a literal backslash; bash heredocs mangle escaped ones, so build it explicitly

os.chdir(r"C:\Users\Andrew\rwkv-anki-autoresearch")
src = io.open("scratchpad/w10plain/run_w10plain.cmd", encoding="utf-8", newline="").read()
src = src.replace("\r\n", "\n")

QAT_ENV = """
REM ---- QAT: the ONLY difference from arm 1. Decay-only and warm-started, which is how QAT is
REM deployed here and what iter 40 (+0.0118 from scratch) says it must be.
set RWKV_QAT_LOWRANK_SCOPE=card:1:int4,note:1:int4
set RWKV_QAT_PQ=reference/pq_cb_wkv_ws10_b10.txt
set RWKV_QAT_SHIFT_PQ=reference/pq_cb_shift_ws10_m2b12.txt
set RWKV_QAT_SHIFT_SCOPE=card:int3,note:int3
set RWKV_QAT_NORM_BITS=1
set RWKV_QAT_FUSED=1
set RWKV_QAT_PQ_LEARN=1
set RWKV_QAT_SHIFT_PQ_LEARN=1
"""

GUARD = """
REM ---- PHASE 0: the catalogs must EXIST and be the ws10-fitted ones, and the QAT env must reach
REM the FINAL config. A banner proves a value was computed, never that it was used: on 2026-08-12
REM the whole QAT env was parsed and then discarded by RWKV_ARCH_MODULE for an entire track-2
REM phase, symptomless except for a quantization cost of zero.
if not exist "reference{B}pq_cb_wkv_ws10_b10.txt" (
  echo %TAG% MISSING ws10 WKV catalog -- fit it from the ws10 WS-final first %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_51 %DATE% %TIME% >> "%LOG%"
  exit /b 51
)
if not exist "reference{B}pq_cb_shift_ws10_m2b12.txt" (
  echo %TAG% MISSING ws10 shift catalog %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_52 %DATE% %TIME% >> "%LOG%"
  exit /b 52
)
.venv{B}Scripts{B}python.exe scratchpad/qat_tax/assert_qat_live.py >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo %TAG% QAT env is INERT in the final config %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_53 %DATE% %TIME% >> "%LOG%"
  exit /b 53
)
""".replace("{B}", B)

# 1. identity
src = src.replace("scratchpad" + B + "w10plain", "scratchpad" + B + "w10qat")
src = src.replace("scratchpad/w10plain", "scratchpad/w10qat")
src = src.replace("set LOG=%DIR%" + B + "w10plain.log", "set LOG=%DIR%" + B + "w10qat.log")
src = src.replace("set TAG=w10plain", "set TAG=w10qat")
src = src.replace("w10p_d", "w10q_d")
src = src.replace("w10plain START", "w10qat START")

# 2. QAT env appended to the env block, then the phase-0 guard before the decay
anchor = 'if not exist "%DIR%" mkdir "%DIR%"'
assert src.count(anchor) == 1
src = src.replace(anchor, QAT_ENV + "\n" + anchor, 1)
anchor2 = "REM ---- PHASE B"
assert src.count(anchor2) == 1
src = src.replace(anchor2, GUARD + "\n" + anchor2, 1)

# 3. the eval is QUANT-AWARE too -- the deploy number, not a plain one
src = src.replace(
    "REM ---- PHASE C: rectified VAL-half eval",
    "REM PHASE C is QUANT-AWARE here (the deploy number), so budget ~10 h, not the ~2.9 h a\n"
    "REM plain eval takes -- iter 47's control eval ran 10h18m.\n"
    "REM ---- PHASE C: rectified VAL-half eval")

body = "\n".join(ln for ln in src.split("\n") if not ln.strip().startswith("REM"))
for tok in ("w10plain", "w10p_d"):
    assert tok not in body, "stale token %r survived outside a REM" % tok
assert body.count("DONE_EXIT_0") == 1
for need in ("RWKV_QAT_PQ_LEARN=1", "RWKV_QAT_SHIFT_PQ_LEARN=1", "pq_cb_wkv_ws10_b10.txt",
             "assert_qat_live.py", "scratchpad/ws10 w10_ws w10q_d", "1 5000 2.0 1e-3 65536"):
    assert need in body, "missing %r" % need
os.makedirs("scratchpad/w10qat", exist_ok=True)
io.open("scratchpad/w10qat/run_w10qat.cmd", "w", newline="\r\n").write(src)
print("wrote scratchpad/w10qat/run_w10qat.cmd")
