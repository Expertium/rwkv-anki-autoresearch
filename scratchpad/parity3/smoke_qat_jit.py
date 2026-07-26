"""Can quant-aware training run with TorchScript ON? CPU-only, seconds, no GPU needed.

WHY THIS MATTERS (CLAUDE.md "THE ENDGAME, ORDERED"). The final 10x-epoch run is quant-aware, and
QAT currently forces `RWKV_NO_JIT=1`. The QAT wall-clock tax is ~1.7x, and the decomposition says
almost all of it is LOST JIT (1.13 kernel x 1.38 JIT = 1.56, vs 1.7 observed) rather than kernel
work. So if QAT can keep JIT, roughly **1.5 days comes off a 4-day run** -- the cheapest large win
available, which is why CLAUDE.md flags "JIT on the grafted q72u paths unverified -- A/B once at
champion-run launch".

The GPU A/B is blocked behind the queue, but the question underneath it is NOT a GPU question. It
has two failure modes and both are reachable on CPU in seconds:

  1. COMPILE  -- does the ScriptModule build at all with the QAT env set? `quant_aware_rwkv7`
     carries `@torch.jit.ignore`, whose stated purpose (rwkv_ops.py:520-522) is exactly to let the
     scripter compile RWKV7TimeMixer.forward's hot path around it. Construction is the test,
     because this repo uses OLD-STYLE ScriptModule, which compiles at construction.
  2. DISPATCH -- can scripted code actually CALL the ignored function at runtime? Plain
     `@torch.jit.ignore` dispatches to the Python interpreter, but `torch.jit.unused` /
     `ignore(drop=True)` compiles to a raise instead. That difference is invisible until a tensor
     goes through, and a compile-only test would pass while the real run dies on step 1.

Test 2 is the one that matters and it is deliberately NOT a full SrsRWKV forward: that needs an
LMDB and a batch dict. A minimal ScriptModule calling the same function with synthetic tensors
isolates the dispatch question exactly.

⚠ Passing here does NOT mean JIT-on QAT is adopted. Still required on GPU: bit-exactness vs the
NO_JIT path (the whole point of QAT is that the simulated quantization matches deploy) and the
actual steps/s A/B. This only says whether that GPU experiment is worth queueing.

Each env gets its OWN subprocess -- old-style ScriptModule bakes the first construction's flags
into the compiled class (CLAUDE.md "TorchScript hook rules").

Run:  .venv/Scripts/python.exe scratchpad/parity3/smoke_qat_jit.py
"""
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The champion's QAT recipe (CLAUDE.md "5k-PHASE METHODOLOGY (a)"), minus RWKV_NO_JIT -- which is
# the variable under test. Codebooks are the tracked q72u deploy pair.
QAT_ENV = {
    "RWKV_QAT_LOWRANK_SCOPE": "card:1:int4,note:1:int4",
    "RWKV_QAT_PQ": "reference/pq_cb_wkv_q72u.txt",
    "RWKV_QAT_SHIFT_PQ": "reference/pq_cb_shift_q72u.txt",
    "RWKV_QAT_SHIFT_SCOPE": "card:int3,note:int3",
    "RWKV_QAT_NORM_BITS": "1",
    "RWKV_QAT_FUSED": "1",
}

# iter 31 / A18 trunk arch, so the answer is about the model we will actually run.
ARCH_ENV = {
    "RWKV_ARCH_MODULE": "scratchpad/track2_a18/architecture_d80_lora4.py",
    "RWKV_GRU_HEAD": "3",
    "RWKV_STRIP_L0_VLORA": "1",
    "RWKV_ZERO_FEATURES": "22",
    "RWKV_STATE_CLAMP_TAU": "300",
    "RWKV_STATE_CLAMP_WINDOW": "32768",
    "RWKV_NO_AHEAD_RESIDUAL": "1",
    "RWKV_STRIP_CMIX": ("user_id:0,user_id:1,user_id:2,preset_id:0,preset_id:1,"
                        "preset_id:2,deck_id:1,deck_id:2,card_id:1"),
}

CHILD = r"""
import os, torch

no_jit = bool(os.environ.get("RWKV_NO_JIT"))

# --- TEST 1: does the whole model COMPILE under this env? -------------------------------
from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG
from rwkv.model.srs_model import SrsRWKV
m = SrsRWKV(anki_rwkv_config=DEFAULT_ANKI_RWKV_CONFIG)
base = type(m).__mro__[1].__name__
n = sum(p.numel() for p in m.parameters())
scripted = "Script" in base
# The env is the independent variable, so assert the flag actually took effect -- otherwise a
# silently-eager build would "pass" the JIT-on case and prove nothing.
assert scripted == (not no_jit), f"NO_JIT={no_jit} but base={base}: the flag did not take effect"
print(f"COMPILE_OK no_jit={no_jit} base={base} params={n}")

# --- TEST 2: can SCRIPTED code call the @torch.jit.ignore'd QAT kernel at runtime? -------
# Compile success above says nothing about this: torch.jit.unused / ignore(drop=True) compiles
# to a raise, and that only fires when a tensor goes through.
from rwkv.model.rwkv_ops import quant_aware_rwkv7

class Caller(torch.nn.Module):
    def forward(self, r, k, v, w, a, kd, skip):
        # float args mirror the real call site (rwkv_model.py:831): int4 qmax, rank-1 low-rank.
        return quant_aware_rwkv7(r, k, v, w, a, kd, skip, 7.0, 1, float("inf"))

B, T, H, K = 1, 4, 2, 16
g = torch.Generator().manual_seed(0)
def rnd():
    return torch.randn(B, T, H, K, generator=g, dtype=torch.float32)
r, k, v, a, kd = rnd(), rnd(), rnd(), rnd(), rnd()
w = -torch.rand(B, T, H, K, generator=g).exp()   # decay must be negative before exp()
skip = torch.zeros(B, T, dtype=torch.bool)

mod = Caller()
if not no_jit:
    mod = torch.jit.script(mod)          # the actual question
out = mod(r, k, v, w, a, kd, skip)

assert torch.isfinite(out).all(), "QAT kernel returned non-finite values"
# Guard against a vacuous pass: an all-zero output would satisfy 'finite' while proving nothing
# ran (the same zero-init trap parity_train_vs_rnn.py warns about).
mag = float(out.abs().mean())
assert mag > 1e-8, f"output is ~zero (mean|out|={mag:.2e}) -- nothing was computed"
print(f"DISPATCH_OK scripted_caller={not no_jit} mean_abs_out={mag:.6e} "
      f"checksum={float(out.double().sum()):.10f}")
"""


# The child must live in a REAL .py file, not `python -c`: torch.jit.script compiles from SOURCE,
# and a class defined in a -c string has none ("Can't get source for ..."). That failure looks
# exactly like a genuine JIT-on rejection, so writing the file is what keeps the test honest.
CHILD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_qat_jit_child.py")


def run(tag, extra):
    env = dict(os.environ, PYTHONPATH=REPO)
    for d in (ARCH_ENV, QAT_ENV):
        env.update(d)
    env.pop("RWKV_NO_JIT", None)
    env.update(extra)
    p = subprocess.run([sys.executable, CHILD_PATH], cwd=REPO, env=env,
                       capture_output=True, text=True)
    lines = (p.stdout + p.stderr).strip().splitlines()
    ok = p.returncode == 0 and any("DISPATCH_OK" in ln for ln in lines)
    print(f"\n--- {tag} (rc={p.returncode}) ---")
    for ln in lines:
        if any(t in ln for t in ("COMPILE_OK", "DISPATCH_OK", "Error", "error", "assert",
                                 "Traceback", "RuntimeError")):
            print("   " + ln[:220])
    if not ok:
        for ln in lines[-6:]:
            print("   | " + ln[:220])
    return ok, lines


def main():
    with open(CHILD_PATH, "w", encoding="utf-8") as fh:
        fh.write(CHILD)
    ok_nojit, l0 = run("CONTROL: QAT + RWKV_NO_JIT=1 (the recipe used today)",
                       {"RWKV_NO_JIT": "1"})
    ok_jit, l1 = run("UNDER TEST: QAT with JIT ON (RWKV_NO_JIT unset)", {})

    def csum(lines):
        for ln in lines:
            if "checksum=" in ln:
                return ln.split("checksum=")[1].split()[0]
        return None

    print("\n" + "=" * 72)
    if not ok_nojit:
        # Control failing means the harness/env is wrong, NOT that JIT-on is impossible.
        print("CONTROL FAILED -- the QAT env itself does not run here; the JIT-on result below")
        print("is uninterpretable. Fix the control first (codebook paths? arch module?).")
    elif ok_jit:
        c0, c1 = csum(l0), csum(l1)
        same = c0 is not None and c0 == c1
        print("QAT_JIT: COMPILES AND DISPATCHES. TorchScript can call the ignored QAT kernel,")
        print("so RWKV_NO_JIT is not structurally required by quant-aware training.")
        print(f"CPU checksum eager={c0} scripted={c1} -> {'IDENTICAL' if same else 'DIFFER'}")
        if not same:
            print("  ^ differing CPU checksums are a red flag; investigate before the GPU A/B.")
        print("NEXT (GPU, gated on a free card): bit-exactness of a real training step vs the")
        print("NO_JIT path, then the steps/s A/B. Worth ~1.38x = ~1.5 days off the 10x run.")
    else:
        print("QAT_JIT: FAILS with JIT on (control passed, so the env is fine). RWKV_NO_JIT stays")
        print("mandatory and the ~1.7x QAT tax is NOT recoverable this way -- budget the 10x run")
        print("at ~4 days. The failure text above is the reason; a dispatch failure (unused/")
        print("drop=True) is fixable, a deep compile failure in the QAT path likely is not.")
    print("=" * 72)
    print("SMOKE_QAT_JIT_" + ("PASS" if (ok_nojit and ok_jit) else "FAILED"))
    sys.exit(0 if (ok_nojit and ok_jit) else 1)


if __name__ == "__main__":
    main()
