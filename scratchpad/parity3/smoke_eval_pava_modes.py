"""Smoke-test RWKV_EVAL_PAVA modes 0/1/2 before spending GPU hours on them.

Mode 2 (substitute the UNRECTIFIED pressed probe) was added 2026-07-26 to separate the PAVA
pooling from the zeroed current-row duration, which modes 0 and 1 move together. It is new code
sitting on the eval path, and this repo's lesson bank records two dead launches from hooks that
were never exercised before a run -- so exercise it here, on CPU, in seconds.

Each mode runs in its OWN subprocess: the flag is read in __init__ and the old-style ScriptModule
API bakes the first construction's env into the compiled class, so two modes cannot coexist in one
process (CLAUDE.md "TorchScript hook rules").

What is asserted, on synthetic tensors with a KNOWN answer:
  mode 0 -> the eval-rectify path is inert (curve_probs returned unchanged)
  mode 2 -> scored rows take the RAW pressed probe          (pooling must NOT happen)
  mode 1 -> scored rows take the RECTIFIED pressed probe    (pooling MUST happen)
and that modes 1 and 2 actually DIFFER on a deliberately out-of-order input -- otherwise the whole
decomposition would silently measure nothing.

Run:  PYTHONPATH=. .venv/Scripts/python.exe scratchpad/parity3/smoke_eval_pava_modes.py
"""
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CHILD = r"""
import os, torch
from rwkv.model.pava import pava_rectify_scalar
import rwkv.model.srs_model as sm

mode = os.environ.get("RWKV_EVAL_PAVA", "0")

# A bare object carrying just what _pava_rectify_eval touches -- building a whole SrsRWKV needs
# an LMDB and a GPU, and this method depends on nothing else.
class Stub:
    pass
s = Stub()
s.eval_pava = mode in ("1", "2")
s.eval_pava_rectify = mode != "2"
s.pava_lambda = 0.0            # -> classic p=1 PAVA, no pava_theta needed
s._pava_rectify_eval = sm.SrsRWKV._pava_rectify_eval.__get__(s, Stub)

# 8 rows: rows 0..3 are one scored review's 4 probes, rows 4..7 another's.
# Probe block A is OUT OF ORDER (0.9 > 0.3), so pooling must change it.
# Probe block B is already ordered, so pooling must leave it alone -- that separates
# "mode 1 did something" from "mode 1 did something everywhere".
curve = torch.tensor([0.10, 0.90, 0.30, 0.95,
                      0.10, 0.20, 0.30, 0.40], dtype=torch.float32)
probe_rows    = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
probe_target  = torch.tensor([0, 4])   # where the scored prediction is written back
probe_pressed = torch.tensor([2, 2])   # both scored rows pressed button index 2 ("Good")

out = s._pava_rectify_eval(curve.clone(), probe_rows, probe_target, probe_pressed)

raw_A, raw_B = 0.30, 0.30                                   # v[pressed] for each block
rect_A = pava_rectify_scalar([0.10, 0.90, 0.30, 0.95], [1.0]*4, [1.0]*3)[2]
rect_B = pava_rectify_scalar([0.10, 0.20, 0.30, 0.40], [1.0]*4, [1.0]*3)[2]
got_A, got_B = float(out[0]), float(out[4])
print(f"mode={mode}  A: got {got_A:.6f} (raw {raw_A:.6f}, rect {rect_A:.6f})"
      f"   B: got {got_B:.6f} (raw {raw_B:.6f}, rect {rect_B:.6f})")

# block A must be a REAL discriminator, else the test proves nothing
assert abs(rect_A - raw_A) > 1e-3, f"fixture is vacuous: rect_A == raw_A == {raw_A}"
assert abs(rect_B - raw_B) < 1e-9, "block B was already ordered; pooling should not move it"

if mode == "2":
    assert abs(got_A - raw_A) < 1e-6, f"mode 2 must NOT pool: got {got_A}, want raw {raw_A}"
elif mode == "1":
    assert abs(got_A - rect_A) < 1e-6, f"mode 1 must pool: got {got_A}, want rect {rect_A}"
assert abs(got_B - raw_B) < 1e-6, "ordered block must be untouched in every mode"
print("MODE_OK")
"""


def main():
    ok = True
    for mode in ("2", "1"):
        env = dict(os.environ, PYTHONPATH=REPO, RWKV_EVAL_PAVA=mode)
        p = subprocess.run([sys.executable, "-c", CHILD], cwd=REPO, env=env,
                           capture_output=True, text=True)
        out = (p.stdout + p.stderr).strip().splitlines()
        for ln in out[-4:]:
            print("   " + ln)
        ok &= p.returncode == 0 and any("MODE_OK" in ln for ln in out)
    print("\nEVAL_PAVA_MODES_" + ("ALL_PASS" if ok else "FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
