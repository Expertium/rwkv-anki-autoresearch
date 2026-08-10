"""Correctness harness for iter 44's RWKV_ILV_SPREAD (endpoint-anchored layer placement).

The lever changes WHEN each stream's layers run inside the interleaved round schedule, not
what they compute. Three properties must hold, and the first two are oracles -- they have a
known-correct answer that does not depend on believing the new code:

  [1] SPREAD OFF must be BIT-IDENTICAL to the pre-iter-44 code path. The rewrite replaced
      `if r < depth: forward_layer(r, ...)` with a schedule lookup, and with spread off the
      lookup returns exactly `r`, so any difference is a bug in the rewrite. Checked against a
      literal re-implementation of the OLD loop (inlined here, so the check survives the old
      code being deleted).
  [2] DEPTH-1 ORACLE: when every stream has depth 1 there is one round, so front-loaded and
      spread schedules coincide -- outputs must be bit-identical to each other AND to the
      sequential (non-interleaved) chain. This is the same oracle that validated iter 41.
  [3] At REAL depths, spread must actually DIFFER from front-loaded (else the flag is a no-op
      and any "win" would be noise), gradients must be finite, and the no-grad parameter set
      must be IDENTICAL between the two placements (a placement change must not strand a
      parameter -- that would be a silent capacity change masquerading as a schedule change).

Plus the scripted-compile check that CLAUDE.md requires: eval runs with JIT ON, so a smoke
that only tests eager cannot certify the path that scores the model. Run as a subprocess per
env combination, because old-style ScriptModule bakes the first construction's flags.

Run:  .venv/Scripts/python.exe scratchpad/parity3/smoke_ilv_spread.py
"""
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_ENV = {
    "RWKV_ARCH_MODULE": "scratchpad/track2_a18/architecture_d80_lora4.py",
    "RWKV_GRU_HEAD": "3",
    "RWKV_STRIP_L0_VLORA": "1",
    "RWKV_ZERO_FEATURES": "22",
    "RWKV_STATE_CLAMP_TAU": "300",
    "RWKV_STATE_CLAMP_WINDOW": "32768",
    "RWKV_NO_AHEAD_RESIDUAL": "1",
    "RWKV_STRIP_CMIX": "user_id:0,user_id:1,user_id:2,preset_id:0,preset_id:1,"
                       "preset_id:2,deck_id:1,deck_id:2,card_id:1",
    "RWKV_PAVA_LAMBDA": "0.2",
}

WORKER = r'''
import dataclasses, json, os, sys
sys.path.insert(0, REPO_PATH)
os.chdir(REPO_PATH)
import torch, lmdb
from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG
from rwkv.model.srs_model import SrsRWKV, interleave_schedule
from rwkv.prepare_batch import get_data, prepare

MODE = os.environ["SMOKE_MODE"]           # "new" | "oldloop"
cfg = DEFAULT_ANKI_RWKV_CONFIG
if os.environ.get("SMOKE_DEPTH1") == "1":
    # same construction as smoke_interleave.py's oracle
    cfg = dataclasses.replace(
        cfg, modules=[(n, dataclasses.replace(c, n_layers=1)) for n, c in cfg.modules]
    )
torch.manual_seed(7)
model = SrsRWKV(anki_rwkv_config=cfg)
with torch.no_grad():
    for p in model.parameters():
        p.normal_(0.0, 0.1)               # zero-init params would make agreement vacuous
model = model.float(); model.eval()

if MODE == "oldloop":
    # literal pre-iter-44 placement: layer j in round j. Bit-identity target for [1].
    model.ilv_sched = [[(r if r < d else -1) for r in range(max(model.stream_depths))]
                       for d in model.stream_depths]

env = lmdb.open("train_db_5k_h1", map_size=400_000_000_000, readonly=True, lock=False)
with env.begin(write=False) as txn:
    batches = json.loads(txn.get(b"101_batches"))
    b = min(batches, key=lambda x: x[2])
    data = get_data(txn, (101, b[0], b[1], b[2]), device="cpu")
env.close()
pb = prepare([data], seed=1234, probe_density=0.08).to("cpu")

torch.set_grad_enabled(True)
model.zero_grad(set_to_none=True)
out = model.forward_batch(pb.start.float(), pb.sub_gather, pb.sub_gather_lens,
                          pb.time_shift_selects, pb.skips, pb.num_data)
flat = torch.cat([o.float().flatten() for o in out])
flat.square().mean().backward()
nograd = sorted(n for n, p in model.named_parameters() if p.grad is None)
allfinite = all(bool(torch.isfinite(p.grad).all()) for p in model.parameters()
                if p.grad is not None)
print("SMOKE_JSON " + json.dumps({
    "checksum": float(flat.double().sum()),
    "scale": float(flat.abs().max()),
    "n": int(flat.numel()),
    "sched": model.ilv_sched,
    "nograd": nograd,
    "grads_finite": allfinite,
    "scripted": type(model).__name__ != "SrsRWKV" or isinstance(model, torch.jit.ScriptModule),
}))
'''


def run(tag, env_extra, mode="new"):
    env = dict(os.environ)
    env.update(BASE_ENV)
    env.update(env_extra)
    env["SMOKE_MODE"] = mode
    env["PYTHONPATH"] = REPO
    src = WORKER.replace("REPO_PATH", repr(REPO))
    p = subprocess.run([sys.executable, "-c", src], cwd=REPO, env=env,
                       capture_output=True, text=True)
    line = [l for l in p.stdout.splitlines() if l.startswith("SMOKE_JSON ")]
    if not line:
        print(f"--- {tag} FAILED ---\n{p.stdout[-3000:]}\n{p.stderr[-3000:]}")
        sys.exit(1)
    r = json.loads(line[0][len("SMOKE_JSON "):])
    print(f"  {tag:28s} checksum={r['checksum']:.10f} scale={r['scale']:.4f}")
    return r


def main():
    if not os.path.isdir(os.path.join(REPO, "train_db_5k_h1")):
        print("SKIP: train_db_5k_h1 not present")
        return

    ilv = {"RWKV_INTERLEAVE": "1", "RWKV_NO_JIT": "1"}

    print("[1] spread OFF must be bit-identical to the OLD front-loaded loop")
    a = run("new code, spread off", ilv)
    b = run("literal old loop", ilv, mode="oldloop")
    assert a["scale"] > 1e-3, "outputs ~zero -- comparison would be vacuous"
    assert a["sched"] == b["sched"], f"schedules differ: {a['sched']} vs {b['sched']}"
    assert a["checksum"] == b["checksum"], "REWRITE IS NOT BIT-IDENTICAL with spread off"
    print("    OK -- identical schedule and identical checksum\n")

    print("[2] depth-1 oracle: spread == front-loaded == sequential")
    d1 = {"SMOKE_DEPTH1": "1"}
    o_seq = run("sequential", {**d1, "RWKV_NO_JIT": "1"})
    o_front = run("interleaved, spread off", {**d1, **ilv})
    o_spread = run("interleaved, spread ON", {**d1, **ilv, "RWKV_ILV_SPREAD": "1"})
    assert o_spread["sched"] == o_front["sched"], "one round: schedules must coincide"
    assert o_seq["checksum"] == o_front["checksum"] == o_spread["checksum"], (
        f"depth-1 oracle broken: {o_seq['checksum']} / {o_front['checksum']} / "
        f"{o_spread['checksum']}"
    )
    print("    OK -- all three bit-identical at depth 1\n")

    print("[3] real depths: spread must DIFFER, keep grads finite, strand no parameter")
    f = run("front-loaded", ilv)
    s = run("SPREAD", {**ilv, "RWKV_ILV_SPREAD": "1"})
    assert s["sched"] != f["sched"], "spread produced the front-loaded schedule -- no-op flag"
    assert s["checksum"] != f["checksum"], "spread changed nothing numerically -- no-op flag"
    assert s["grads_finite"] and f["grads_finite"], "non-finite gradients"
    assert s["nograd"] == f["nograd"], (
        f"placement changed the no-grad set (+{sorted(set(s['nograd'])-set(f['nograd']))} "
        f"-{sorted(set(f['nograd'])-set(s['nograd']))}) -- a schedule change must not strand "
        f"parameters"
    )
    print(f"    front  sched={f['sched']}")
    print(f"    spread sched={s['sched']}")
    print(f"    OK -- differs, grads finite, {len(s['nograd'])} no-grad params in BOTH\n")

    print("[4] scripted compile (JIT ON -- the configuration eval actually runs)")
    sj = run("SPREAD, scripted", {"RWKV_INTERLEAVE": "1", "RWKV_ILV_SPREAD": "1"})
    assert sj["scale"] > 1e-3
    print("    OK -- compiles and runs scripted\n")

    print("SMOKE OK -- RWKV_ILV_SPREAD is a pure placement change")


if __name__ == "__main__":
    main()
