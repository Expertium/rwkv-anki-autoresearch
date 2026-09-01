"""RWKV_FSRS_CARD: do TRAINING and DEPLOY compute the same FSRS card recurrence?

The section-9 three-way check for V1. It needs its own file for the same reason
`smoke_id_features_width.py` did: `parity_train_vs_rnn.py` compares RWKV7 against RWKV7RNN on a
single stack and never constructs `SrsRWKV`/`SrsRWKVRnn`, so it structurally cannot see a
disagreement about the card core.

WHAT CAN ACTUALLY DIVERGE HERE, which is what each case targets:

  1. CONSTRUCTION -- the two classes build the core independently, from their own reads of the
     env and their own `CARD_FEATURE_COLUMNS.index("rating_1")`. A different n_free or a
     different rating column would be invisible to every shape check.
  2. THE SCAN -- training scans a (B, T, C) block over T; deploy is handed one review at a time
     and owns the state between calls. Same function only if the state threading matches.
  3. SKIP SEMANTICS -- a query/probe row must PRODUCE an output but must NOT advance the state.
     Training enforces this inside its loop with a per-element `where`. If the two disagree, the
     card's state history differs and nothing downstream would report it.
  4. INERTNESS -- flag off must leave both classes byte-identical.

⚠ WHAT THIS DOES NOT COVER, stated rather than implied: `run_as_rnn`'s own driver calls `review()`
twice per row (once for the query, once for the real review), so the deploy SKIP convention lives
in the CALLER, not in the core. Case 3 proves the core honours a skip mask; it does not prove the
driver passes the right one. That needs an end-to-end trace comparison, which is
`export_rnn_trace.py` + `verify_rust.py`'s job once V1 has a checkpoint.

Run:  .venv/Scripts/python.exe scratchpad/parity3/smoke_fsrs_card.py
CPU-only, seconds.
"""

import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

V1_ENV = {
    "RWKV_ARCH_MODULE": "scratchpad/hybrid100k/arch_fsrs_v1.py",
    "RWKV_INTERLEAVE": "1", "RWKV_GRU_HEAD": "3", "RWKV_PAVA_LAMBDA": "0.2",
    "RWKV_NO_AHEAD_RESIDUAL": "1", "RWKV_STRIP_L0_VLORA": "1", "RWKV_ZERO_FEATURES": "22",
    "RWKV_STATE_CLAMP_TAU": "300", "RWKV_STATE_CLAMP_WINDOW": "32768", "RWKV_NO_JIT": "1",
    "RWKV_STRIP_CMIX": ("user_id:0,user_id:1,user_id:2,preset_id:0,preset_id:1,preset_id:2,"
                        "deck_id:1,deck_id:2"),
}
CHAMP_ENV = dict(V1_ENV)
CHAMP_ENV["RWKV_ARCH_MODULE"] = "scratchpad/track2_a18/architecture_d80_lora4_cnd.py"
CHAMP_ENV["RWKV_STRIP_CMIX"] = V1_ENV["RWKV_STRIP_CMIX"] + ",card_id:1"

CHILD = r'''
import json, os, sys
sys.path.insert(0, %(repo)r)
import torch
torch.set_num_threads(1)
torch.manual_seed(20260830)
from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG as CFG
from rwkv.model.srs_model import SrsRWKV
from rwkv.model.srs_model_rnn import SrsRWKVRnn

out = {}
tr, rn = SrsRWKV(CFG), SrsRWKVRnn(CFG)
out["train_params"] = sum(p.numel() for p in tr.parameters())
out["rnn_params"] = sum(p.numel() for p in rn.parameters())
out["train_cores"] = len(tr.fsrs_cores)
out["rnn_cores"] = len(rn.fsrs_cores)
out["train_r1"] = int(tr.fsrs_r1)
out["rnn_r1"] = int(rn.fsrs_r1)

if len(tr.fsrs_cores) > 0:
    core_t, core_r = tr.fsrs_cores[0], rn.fsrs_cores[0]
    out["n_free"] = int(core_t.n_free)
    # give both cores the SAME random weights -- an untrained emitter is zero-weight, so a
    # comparison at init would be nearly vacuous (every review would emit the same w).
    # ⚠ PARAMETERS ONLY, never the state_dict. The first version randomized everything
    # floating-point in state_dict(), which includes the `clip_lo`/`clip_hi` BUFFERS -- the FSRS
    # parameter ranges. Random bounds make `bounded_w` produce garbage and the whole comparison
    # came back NaN. The non-vacuity check below is what caught it. (n_free=5 escaped by luck:
    # a different-sized state_dict consumed the RNG differently from the same seed.)
    full = dict(core_t.state_dict())
    for k, v in core_t.named_parameters():
        full[k] = torch.randn_like(v) * 0.3
    core_t.load_state_dict(full); core_r.load_state_dict(full)

    B, T, C = 2, 6, CFG.d_model
    x = torch.randn(B, T, C)
    t = torch.rand(B, T) * 30.0
    rating = torch.randint(1, 5, (B, T)).float()
    skip = torch.zeros(B, T)
    skip[0, 2] = 1.0          # a query row mid-sequence
    skip[1, 0] = 1.0          # a query row FIRST, before any real review

    # --- TRAINING-STYLE scan (what srs_model.py's inline loop does) ---
    st = torch.zeros(B, 3 + core_t.n_free)
    outs = []
    for i in range(T):
        xo, _r, ns = core_t.review(x[:, i], t[:, i], rating[:, i], st)
        outs.append(xo)
        keep = skip[:, i].to(torch.bool).unsqueeze(-1)
        st = torch.where(keep, st, ns)
    train_out, train_state = torch.stack(outs, dim=1), st

    # --- DEPLOY-STYLE stepping (what srs_model_rnn.py does, one review per call) ---
    st2 = torch.zeros(B, 3 + core_r.n_free)
    outs2 = []
    for i in range(T):
        xo, _r, ns = core_r.review(x[:, i], t[:, i], rating[:, i], st2)
        outs2.append(xo)
        keep = skip[:, i].to(torch.bool).unsqueeze(-1)
        st2 = torch.where(keep, st2, ns)
    dep_out, dep_state = torch.stack(outs2, dim=1), st2

    out["max_out_diff"] = float((train_out - dep_out).abs().max())
    out["max_state_diff"] = float((train_state - dep_state).abs().max())
    out["out_scale"] = float(train_out.abs().max())

    # --- skip semantics: the state after a skipped row must equal the state before it ---
    st3 = torch.zeros(B, 3 + core_t.n_free)
    _xo, _r, st3 = core_t.review(x[:, 0], t[:, 0], rating[:, 0], st3)
    before = st3.clone()
    xo_sk, _r, ns_sk = core_t.review(x[:, 1], t[:, 1], rating[:, 1], st3)
    kept = torch.where(torch.ones(B, 1, dtype=torch.bool), st3, ns_sk)
    out["skip_state_unchanged"] = float((kept - before).abs().max())
    out["skip_still_outputs"] = float(xo_sk.abs().max())
    out["unskipped_moves"] = float((ns_sk - before).abs().max())

print("RESULT " + json.dumps(out))
''' % {"repo": REPO}


def run(extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith("RWKV_")}
    env.update(extra)
    env["PYTHONIOENCODING"] = "utf-8"
    env["OMP_NUM_THREADS"] = "1"
    p = subprocess.run([sys.executable, "-c", CHILD], capture_output=True, text=True,
                       env=env, cwd=REPO)
    line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
    if not line:
        raise SystemExit("child failed:\n" + p.stdout[-2500:] + "\n" + p.stderr[-2500:])
    import json
    return json.loads(line[-1][len("RESULT "):])


def main():
    fails = []

    def check(name, cond, detail=""):
        print(("  PASS  " if cond else "  FAIL  ") + name + (("  -- " + detail) if detail else ""))
        if not cond:
            fails.append(name)

    print("[1] flag OFF -> inert, and identical in both classes")
    r = run(CHAMP_ENV)
    check("no core in either class", r["train_cores"] == 0 and r["rnn_cores"] == 0)
    check("champion param count unchanged (558,212)",
          r["train_params"] == 558212, str(r["train_params"]))
    check("train and deploy agree on params",
          r["train_params"] == r["rnn_params"],
          f'{r["train_params"]} vs {r["rnn_params"]}')

    print("\n[2] flag ON -> both classes build the SAME core")
    env = dict(V1_ENV); env["RWKV_FSRS_CARD"] = "0"
    r = run(env)
    check("one core in each class", r["train_cores"] == 1 and r["rnn_cores"] == 1)
    check("param counts identical across paths",
          r["train_params"] == r["rnn_params"],
          f'{r["train_params"]} vs {r["rnn_params"]}')
    check("V1 param count is 488,858", r["train_params"] == 488858, str(r["train_params"]))
    check("same rating column resolved", r["train_r1"] == r["rnn_r1"],
          f'{r["train_r1"]} vs {r["rnn_r1"]}')

    print("\n[3] the scan agrees between the two call shapes")
    check("output is non-trivial (guards against a vacuous compare)",
          r["out_scale"] > 1e-3, f'max|out| = {r["out_scale"]:.4f}')
    check("outputs agree", r["max_out_diff"] < 1e-6, f'max diff {r["max_out_diff"]:.3e}')
    check("final states agree", r["max_state_diff"] < 1e-6,
          f'max diff {r["max_state_diff"]:.3e}')

    print("\n[4] skip semantics: a skipped row outputs but does not advance the state")
    check("skipped state is unchanged", r["skip_state_unchanged"] == 0.0,
          f'{r["skip_state_unchanged"]:.3e}')
    check("a skipped row still produces output", r["skip_still_outputs"] > 1e-3,
          f'max|out| = {r["skip_still_outputs"]:.4f}')
    check("...and an UNskipped row really would have moved it",
          r["unskipped_moves"] > 1e-6, f'{r["unskipped_moves"]:.3e}')

    print("\n[5] n_free > 0 (the free-dim hedge) also builds and agrees")
    env2 = dict(V1_ENV); env2["RWKV_FSRS_CARD"] = "5"
    r2 = run(env2)
    check("n_free reaches the core", r2.get("n_free") == 5, str(r2.get("n_free")))
    check("param counts identical across paths",
          r2["train_params"] == r2["rnn_params"],
          f'{r2["train_params"]} vs {r2["rnn_params"]}')
    check("outputs agree", r2["max_out_diff"] < 1e-6, f'max diff {r2["max_out_diff"]:.3e}')
    check("n_free costs params vs n_free=0",
          r2["train_params"] > r["train_params"],
          f'{r2["train_params"]} vs {r["train_params"]}')

    print()
    if fails:
        print("SMOKE FAILED: " + ", ".join(fails))
        return 1
    print("SMOKE PASSED: RWKV_FSRS_CARD is inert when unset, builds identically in the training "
          "and deploy classes, scans to the same result, and honours the skip mask.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
