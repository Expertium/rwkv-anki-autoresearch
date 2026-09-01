"""If a run is pointed at an LMDB of the WRONG feature width, does it fail LOUDLY?

This closes item 3 of the bug-hunt matrix (`scratchpad/bughunt/README.md`) on CPU. That item was
written as "RWKV_ID_FEATURES=1 has never run on a GPU; on the current 92-dim LMDB it SHOULD fail,
and the question is whether it fails loudly" -- but the diagnostic that ran
(`scratchpad/bughunt/diag_idfeat/`) was pointed at `train_db_5k_h1_id`, a MATCHING-width database.
It therefore proved the features path executes and never tested the mismatch at all. A shape
disagreement needs no GPU, so the open half is answerable here.

WHY IT MATTERS MORE THAN IT SOUNDS. The features phase runs two database generations side by side
and CLAUDE.md already records gen 1 and gen 2 as superseded by `*_id3`. Pointing a run at the wrong
generation is a one-character mistake in a toml, and the failure this smoke is looking for is the
silent kind: a width that is absorbed, padded or truncated somewhere, so the run trains happily on
a misaligned feature vector and reports a number.

THE ASYMMETRY THIS FOUND, which is the reason it is worth having:

  * DEPLOY (`srs_model_rnn.py:398`) asserts `card_features.shape == (1, card_features_dim)`.
  * TRAINING (`srs_model.py`) has NO width assert. Its only protection is the first Linear's own
    shape check.

An implicit protection is fine when it actually fires, and worthless when something upstream pads.
So the point of this file is to establish, by EXECUTION, that it fires -- in BOTH directions,
because narrow-into-wide and wide-into-narrow are different failures and only one of them is the
obvious one.

Run:  .venv/Scripts/python.exe scratchpad/parity3/smoke_width_mismatch.py
CPU-only, seconds. Exit 0 = every mismatch is loud.
"""

import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Feed a batch of `fed` feature columns to a model built for its own width, and report what
# happened. Never let the harness supply the width from the same source the model used -- that
# would make the test agree with itself.
CHILD = r'''
import json, os, sys
sys.path.insert(0, %(repo)r)
import torch
torch.set_num_threads(1)
from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG as CFG
from rwkv.model.srs_model import SrsRWKV
from rwkv.model.srs_model_rnn import SrsRWKVRnn

fed = int(os.environ["FED_WIDTH"])
which = os.environ["WHICH"]
out = {"fed": fed, "which": which}
try:
    if which == "train":
        m = SrsRWKV(CFG)
        out["model_width"] = int(m.card_features_dim)
        x = torch.zeros(3, fed)
        m.features2card(x)          # the first thing the real forward does with the features
    else:
        m = SrsRWKVRnn(CFG)
        out["model_width"] = int(m.card_features_dim)
        # `button_heads` is the deploy entry point carrying the explicit width assert
        # (srs_model_rnn.py:398); it is what the PAVA button API calls per review. The five
        # None states mean "fresh card" -- the same call shape buttons_py_vs_rust.py uses.
        # ⚠ The first version omitted them, so EVERY deploy arm raised TypeError for a missing
        # argument instead of for the width. The mismatch cases then "passed" vacuously; only
        # the matched-width arm in [1] revealed it. That arm exists for exactly this.
        m.button_heads(torch.zeros(1, fed), None, None, None, None, None)
    out["result"] = "ACCEPTED"      # no exception: the mismatch was absorbed SILENTLY
except Exception as e:
    out["result"] = "raised"
    out["err"] = type(e).__name__ + ": " + str(e)[:160]
print("RESULT " + json.dumps(out))
''' % {"repo": REPO}


def run(id_features, fed, which):
    env = {k: v for k, v in os.environ.items() if not k.startswith("RWKV_")}
    env.update({"FED_WIDTH": str(fed), "WHICH": which, "PYTHONIOENCODING": "utf-8"})
    if id_features:
        env["RWKV_ID_FEATURES"] = "1"
    p = subprocess.run([sys.executable, "-c", CHILD], capture_output=True, text=True,
                       env=env, cwd=REPO)
    line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
    if not line:
        raise SystemExit("child produced no RESULT:\n" + p.stdout[-2000:] + p.stderr[-2000:])
    import json
    return json.loads(line[-1][len("RESULT "):])


def main():
    fails = []

    def check(name, cond, detail=""):
        print(("  PASS  " if cond else "  FAIL  ") + name + (("  -- " + detail) if detail else ""))
        if not cond:
            fails.append(name)

    # Establish the two widths from the code itself, so the cases below cannot go stale when the
    # column list changes again (it went 44 -> 46 at gen 2 and this smoke must not need editing).
    off = run(False, 0, "train")["model_width"]
    on = run(True, 0, "train")["model_width"]
    print(f"widths: RWKV_ID_FEATURES off = {off}, on = {on}")
    check("the two layouts really differ", off != on, f"{off} vs {on}")

    print("\n[1] MATCHED width is accepted (proves the cases below are not vacuous)")
    for flag, w, tag in ((False, off, "off"), (True, on, "on")):
        for which in ("train", "deploy"):
            r = run(flag, w, which)
            check(f"{tag}/{which}: width {w} accepted", r["result"] == "ACCEPTED",
                  r.get("err", ""))

    print("\n[2] MISMATCHED width must RAISE, in both directions and both paths")
    for flag, w, tag in ((False, on, "off"), (True, off, "on")):
        for which in ("train", "deploy"):
            r = run(flag, w, which)
            loud = r["result"] == "raised"
            check(f"{tag}/{which}: fed {w} into a {r['model_width']}-wide model -> raises",
                  loud, r.get("err", "SILENTLY ACCEPTED -- this is the bug"))

    print()
    if fails:
        print("SMOKE FAILED: " + ", ".join(fails))
        print("A silently accepted mismatch means a run can train on a misaligned feature "
              "vector and report a number. Add an explicit width assert to the offending path.")
        return 1
    print("SMOKE PASSED: a wrong-width LMDB cannot reach training or deploy silently -- "
          "both directions raise in both paths.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
