"""RWKV_ABLATE_FEATURES: does the NAME-based input ablation mean the same thing in training,
in the deploy RNN, and under both feature layouts?

WHY THIS NEEDS ITS OWN SMOKE, like smoke_id_features_width.py before it: the three-way parity
harness `parity_train_vs_rnn.py` is single-stack and compares RWKV7 against RWKV7RNN. It never
constructs SrsRWKV or SrsRWKVRnn, so it structurally cannot see a disagreement about WHICH INPUT
COLUMN a name denotes -- and that is the only thing this flag does.

The four questions, one subprocess each because an old-style ScriptModule bakes the FIRST
construction's env flags into the compiled class:

  1. INERT      flag unset  -> mask is all ones and input_feat_mask_on is False, both classes.
  2. RESOLVE    a real name -> exactly that dim is zeroed, and the TRAIN and DEPLOY mask vectors
                are elementwise IDENTICAL. This is the train-vs-deploy quantity: both files
                resolve the name independently against their own import of CARD_FEATURE_COLUMNS.
  3. TYPO       an unknown name RAISES. Non-vacuity: a silent no-op would produce a candidate
                identical to the champion, i.e. a clean null that reads as "the feature does not
                matter" when it means "the experiment did not run".
  4. ID-LAYOUT  under RWKV_ID_FEATURES=1 the same NAME still resolves, and a name that exists
                ONLY in the new layout (scaled_sibling_gap) resolves there and raises in the old
                one. This is the case RWKV_ZERO_FEATURES cannot serve at all -- it is refused
                under RWKV_ID_FEATURES=1 -- and the reason this flag exists.

THE THIRD PATH (section 9 asks for train / eval / deploy, and this covers two): `rust/rwkv-infer`
has NO name-based flag and does not need one. It already honours `RWKV_ZERO_FEATURES`, which takes
DIMS, and this flag prints the dims it resolved -- so an ablated model deploys by passing that
printed list to the engine. No Rust change is required unless an ablated model becomes a champion,
at which point the right fix is the one already recommended for the existing mask: bake it into the
exported safetensors so the artifact is correct without an env var.

⚠ WHAT THIS SMOKE DOES NOT COVER. Constructing SrsRWKV scripts it, so all four arms prove the
COMPILE half under four env combinations. It does not run a scripted FORWARD, which is where iter
48 lost an eval (a `@torch.jit.ignore` with no return annotation compiled fine and aborted at
runtime). That class cannot arise here -- this change adds no method, only `__init__` code -- but
`scratchpad/parity3/smoke_scripted_eval.sh` is still the guard to run before any launch that sets
the flag, once a GPU is free.

Run:  .venv/Scripts/python.exe scratchpad/parity3/smoke_ablate_features.py
CPU-only, seconds.
"""

import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The child prints one line per class: "<which> <n_zeroed> <zeroed dims>".
CHILD = r'''
import os, sys, json
sys.path.insert(0, %(repo)r)
import torch
from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG as CFG
from rwkv.data_processing import CARD_FEATURE_COLUMNS
from rwkv.model.srs_model import SrsRWKV
from rwkv.model.srs_model_rnn import SrsRWKVRnn

def dims(m):
    v = m.input_feat_mask
    return sorted(int(i) for i in (v == 0).nonzero().flatten().tolist())

out = {}
tr = SrsRWKV(CFG)
out["train"] = {"on": bool(tr.input_feat_mask_on), "dims": dims(tr), "w": int(tr.card_features_dim)}
rn = SrsRWKVRnn(CFG)
out["deploy"] = {"on": bool(rn.input_feat_mask_on), "dims": dims(rn), "w": int(rn.card_features_dim)}
import rwkv.id_features as _idf
out["base"] = _idf.BASE_CARD_FEATURES
out["nnew"] = len(_idf.NEW_COLUMNS)
out["idenc"] = _idf.ID_ENCODING_DIMS
out["ncols"] = len(CARD_FEATURE_COLUMNS)
out["cols"] = list(CARD_FEATURE_COLUMNS)
print("RESULT " + json.dumps(out))
''' % {"repo": REPO}


def run(env_extra, expect_raise=False):
    """Fresh interpreter with a CLEAN env plus exactly env_extra.

    ⚠ The arms must NOT inherit RWKV_* from the ambient environment. A runner that exports the
    lever before calling its own smoke is how the rgate control silently became a second
    treatment arm and passed vacuously at 0.000e+00.
    """
    env = {k: v for k, v in os.environ.items() if not k.startswith("RWKV_")}
    env.update(env_extra)
    env["PYTHONIOENCODING"] = "utf-8"
    p = subprocess.run([sys.executable, "-c", CHILD], capture_output=True, text=True,
                       env=env, cwd=REPO)
    if expect_raise:
        assert p.returncode != 0, "expected a raise, got exit 0:\n" + p.stdout[-2000:]
        return p.stderr
    assert p.returncode == 0, "child failed:\n" + p.stdout[-3000:] + "\n" + p.stderr[-3000:]
    line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
    assert line, "no RESULT line:\n" + p.stdout[-3000:]
    import json
    return json.loads(line[-1][len("RESULT "):])


def main():
    fails = []

    def check(name, cond, detail=""):
        print(("  PASS  " if cond else "  FAIL  ") + name + (("  -- " + detail) if detail else ""))
        if not cond:
            fails.append(name)

    # ---- 1. INERT -------------------------------------------------------------------------
    print("[1] flag unset -> inert")
    r = run({})
    check("train mask off", r["train"]["on"] is False and r["train"]["dims"] == [])
    check("deploy mask off", r["deploy"]["on"] is False and r["deploy"]["dims"] == [])
    check("old layout is 24 cols / width 92",
          r["ncols"] == 24 and r["train"]["w"] == 92,
          f'ncols={r["ncols"]} w={r["train"]["w"]}')
    old_cols = r["cols"]

    # ---- 2. RESOLVE + train/deploy agreement ----------------------------------------------
    print("[2] a real name resolves, and TRAIN == DEPLOY")
    name = "scaled_duration"
    want = old_cols.index(name)
    r = run({"RWKV_ABLATE_FEATURES": name})
    check(f"train zeroes exactly dim {want}", r["train"]["dims"] == [want], str(r["train"]["dims"]))
    check(f"deploy zeroes exactly dim {want}", r["deploy"]["dims"] == [want], str(r["deploy"]["dims"]))
    check("train and deploy masks IDENTICAL", r["train"]["dims"] == r["deploy"]["dims"])
    # two names, order-independent, whitespace-tolerant
    two = sorted([old_cols.index("rating_1"), old_cols.index("day_of_week")])
    r = run({"RWKV_ABLATE_FEATURES": " day_of_week , rating_1 "})
    check("two names, unordered + padded", r["train"]["dims"] == two and r["deploy"]["dims"] == two,
          str(r["train"]["dims"]))

    # ---- 3. TYPO raises (non-vacuity of the guard) -----------------------------------------
    print("[3] an unknown name RAISES (not a silent no-op)")
    err = run({"RWKV_ABLATE_FEATURES": "scaled_duratoin"}, expect_raise=True)
    check("typo raises with the name echoed", "scaled_duratoin" in err, err.strip().splitlines()[-1][:120])
    err = run({"RWKV_ABLATE_FEATURES": "scaled_sibling_gap"}, expect_raise=True)
    check("a NEW-layout name raises in the OLD layout", "scaled_sibling_gap" in err)

    # ---- 4. the -id layout, which is the reason this flag exists ---------------------------
    print("[4] under RWKV_ID_FEATURES=1 (where RWKV_ZERO_FEATURES is refused)")
    r = run({"RWKV_ID_FEATURES": "1", "RWKV_ABLATE_FEATURES": "scaled_sibling_gap"})
    new_cols = r["cols"]
    want = new_cols.index("scaled_sibling_gap")
    # Cross-check rather than hardcode: `data_processing` builds the column LIST and
    # `id_features` computes the WIDTH from BASE_CARD_FEATURES/NEW_COLUMNS, independently. A
    # literal 46/114 here would go stale at the next rebuild (it already did once, at gen 2's
    # 44 -> 46), whereas requiring the two computations to agree is the check that has teeth.
    want_n = r["base"] - 1 + r["nnew"]
    check(f'list and width agree ({want_n} cols / {want_n + r["idenc"]})',
          r["ncols"] == want_n and r["train"]["w"] == want_n + r["idenc"],
          f'ncols={r["ncols"]} w={r["train"]["w"]}')
    check(f"train zeroes exactly dim {want}", r["train"]["dims"] == [want], str(r["train"]["dims"]))
    check("train and deploy masks IDENTICAL", r["train"]["dims"] == r["deploy"]["dims"])
    check("scaled_state is gone from the new layout", "scaled_state" not in new_cols)

    # THE POINT: the same NAME must denote a DIFFERENT dim in the two layouts, and the flag must
    # follow the name. If both layouts gave the same index this test would prove nothing.
    common = "is_query"
    i_old, i_new = old_cols.index(common), new_cols.index(common)
    check(f"'{common}' really moves between layouts ({i_old} -> {i_new})", i_old != i_new)
    r = run({"RWKV_ID_FEATURES": "1", "RWKV_ABLATE_FEATURES": common})
    check(f"name follows the layout, not the index ({i_new})",
          r["train"]["dims"] == [i_new] and r["deploy"]["dims"] == [i_new], str(r["train"]["dims"]))

    print()
    if fails:
        print("SMOKE FAILED: " + ", ".join(fails))
        return 1
    print("SMOKE PASSED: RWKV_ABLATE_FEATURES is inert when unset, resolves by name in both "
          "layouts, agrees between training and deploy, and refuses an unknown name.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
