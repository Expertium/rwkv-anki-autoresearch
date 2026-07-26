"""Construct the SCRIPTED model under every env a queued job will use. Run before launching.

This is the check the lesson bank keeps asking for and the repo did not have. CLAUDE.md's
TorchScript rules cost two dead launches, both the same shape: a hook that was smoke-tested
through a plain-Python path while the real runs build a **ScriptModule**, where TorchScript
resolves attributes even in dead branches and a failure goes SILENT (the NaN-except turns the
run hollow, and train_rwkv can still exit 0). `SrsRWKV(ModuleType)` IS the ScriptModule, so
merely constructing it exercises the compile.

Why it matters more than it looks: edits to `srs_model.py` land on code that ALREADY-PARKED jobs
will import hours later, in a fresh process. A change that is fine in the editing session can
abort a 10-hour training run at 01:00.

Each env combination gets its OWN subprocess -- the old-style ScriptModule API bakes the first
construction's env flags into the compiled class, so two flag values cannot coexist in one
process.

Asserts, per combination: the model constructs, it really is a ScriptModule (not silently a plain
Module, which would make the test vacuous), the parameter count matches `--expect-params` if
given, and the eval-PAVA flag triple is what that mode should imply -- including that the eval
path is INERT when the flag is unset, i.e. training is untouched.

Usage:
  python scratchpad/parity3/smoke_construct_envs.py [--expect-params 558212]
Env: set the run's arch/recipe flags first (RWKV_ARCH_MODULE, RWKV_GRU_HEAD, ...); this script
only varies RWKV_EVAL_PAVA on top of whatever is already exported.
"""
import argparse
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CHILD = r"""
import os, torch
from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG
from rwkv.model.srs_model import SrsRWKV

m = SrsRWKV(anki_rwkv_config=DEFAULT_ANKI_RWKV_CONFIG)
base = type(m).__mro__[1].__name__
n = sum(p.numel() for p in m.parameters())
mode = os.environ.get("RWKV_EVAL_PAVA", "")
ep = getattr(m, "eval_pava", None)
er = getattr(m, "eval_pava_rectify", None)
es = getattr(m, "eval_pava_substitute", None)
print(f"CONSTRUCT_OK base={base} params={n} mode={mode!r} "
      f"eval_pava={ep} rectify={er} substitute={es}")

# A plain Module here would make every one of these checks vacuous.
assert "Script" in base, f"expected a ScriptModule, got {base} -- this test proves nothing"

want = {
    "":  (False, True,  False),   # training / plain eval: the eval-PAVA path must be INERT
    "0": (False, True,  False),
    "1": (True,  True,  True),    # rectified pressed probe   (the deploy metric)
    "2": (True,  False, True),    # raw pressed probe         (duration moved, no pooling)
    "3": (True,  True,  False),   # probes inserted, nothing substituted (bf16-noise control)
}[mode]
assert (ep, er, es) == want, f"flag triple {(ep, er, es)} != expected {want} for mode {mode!r}"
print("FLAGS_OK")
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expect-params", type=int, default=0)
    ap.add_argument("--modes", default=",1,2,3",
                    help="comma-separated RWKV_EVAL_PAVA values; empty entry = unset (training)")
    args = ap.parse_args()

    ok = True
    seen_params = set()
    for mode in args.modes.split(","):
        env = dict(os.environ, PYTHONPATH=REPO)
        env.pop("RWKV_EVAL_PAVA", None)
        if mode:
            env["RWKV_EVAL_PAVA"] = mode
        p = subprocess.run([sys.executable, "-c", CHILD], cwd=REPO, env=env,
                           capture_output=True, text=True)
        line = next((l for l in (p.stdout + p.stderr).splitlines()
                     if "CONSTRUCT_OK" in l), None)
        good = p.returncode == 0 and "FLAGS_OK" in p.stdout
        print(f"  RWKV_EVAL_PAVA={mode or '(unset/training)'}: "
              + (line or (p.stdout + p.stderr).strip().splitlines()[-1:] or ["no output"])[0]
              if not line else f"  {line}")
        if not good:
            for l in (p.stdout + p.stderr).strip().splitlines()[-4:]:
                print("      " + l)
        if line:
            seen_params.add(int(line.split("params=")[1].split()[0]))
        ok &= good

    if len(seen_params) > 1:
        print(f"  FAIL: param count differs across envs {sorted(seen_params)} -- an eval flag "
              f"is changing the MODEL, not just the readout")
        ok = False
    if args.expect_params and seen_params and args.expect_params not in seen_params:
        print(f"  FAIL: params {sorted(seen_params)} != expected {args.expect_params}")
        ok = False

    print("\nCONSTRUCT_ENVS_" + ("ALL_PASS" if ok else "FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
