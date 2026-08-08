"""iter 41 smoke: RWKV_INTERLEAVE (round-robin layer schedule across the 5 streams).

The interleaved forward re-anchors every (stream, split) gather to the canonical layout by
composing the chained permutations prepare() emits (each stream's indices point into the
PREVIOUS stream's output layout). That composition is the risky part, and it has an exact
oracle: with ALL stream depths = 1 the round-robin schedule visits card.L0 -> deck.L0 ->
note.L0 -> preset.L0 -> user.L0 -- the sequential order precisely -- so INTERLEAVE=1 must
reproduce INTERLEAVE=0 BIT-FOR-BIT while routing through the composed gathers. A real chunk
from the real LMDB through the real prepare() (probes included), not synthetic data.

Checks:
  1. depths-all-1, off vs on: bit-identical logits (validates composition + pads + v0 init
     + scatter against the battle-tested sequential branch).
  2. real depths, off vs on: outputs MUST differ (the silent-null-lever lesson: a flag that
     does nothing measures as a perfect null).
  3. the ON child backprops a scalar loss and asserts every param got a finite grad (the
     index_copy/index_select detour must not detach the graph).
  4. scripted compile is implicit: children run JIT-on (no RWKV_NO_JIT), so construction IS
     the TorchScript compile.

Each env combination runs in its own subprocess (old-style ScriptModule bakes env at class
compile). Needs train_db_5k_h1 locally (skips loudly if absent).

Run:  .venv\\Scripts\\python.exe scratchpad/parity3/smoke_interleave.py
"""
import os
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CHILD = r"""
import dataclasses, json, os, sys, torch
import lmdb
from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG
from rwkv.model.srs_model import SrsRWKV
from rwkv.prepare_batch import get_data, prepare

DEPTH1 = os.environ.get("SMOKE_DEPTH1", "0") == "1"
OUT = os.environ["SMOKE_OUT"]
WANT_GRAD = os.environ.get("SMOKE_GRAD", "0") == "1"

cfg = DEFAULT_ANKI_RWKV_CONFIG
if DEPTH1:
    mods = [(n, dataclasses.replace(c, n_layers=1)) for n, c in cfg.modules]
    cfg = dataclasses.replace(cfg, modules=mods)

torch.manual_seed(7)
model = SrsRWKV(anki_rwkv_config=cfg)
with torch.no_grad():
    for p in model.parameters():
        p.normal_(0.0, 0.1)   # zero-init params would make agreement partly vacuous
model = model.float()
model.eval()  # dropout off: the two runs must be deterministic

env = lmdb.open("train_db_5k_h1", map_size=400_000_000_000, readonly=True, lock=False)
with env.begin(write=False) as txn:
    batches = json.loads(txn.get(b"101_batches"))
    b = min(batches, key=lambda x: x[2])   # smallest chunk of user 101
    key = (101, b[0], b[1], b[2])
    data = get_data(txn, key, device="cpu")
env.close()

pb = prepare([data], seed=1234, probe_density=0.08)
pb = pb.to("cpu")

torch.set_grad_enabled(WANT_GRAD)
out = model.forward_batch(
    pb.start.float(), pb.sub_gather, pb.sub_gather_lens,
    pb.time_shift_selects, pb.skips, pb.num_data,
)
# out is a tuple of head tensors; fingerprint the lot
flat = torch.cat([o.float().flatten() for o in out])
nograd = []
if WANT_GRAD:
    loss = flat.square().mean()
    loss.backward()
    bad = [n for n, p in model.named_parameters()
           if p.grad is not None and not torch.isfinite(p.grad).all()]
    assert not bad, f"NON-FINITE grads: {bad[:8]}"
    # params with NO grad exist by design (layer-0 v_lora_simple is structurally dead --
    # the v0=v branch -- plus the GRU-era dummies); the PARENT asserts the interleaved
    # path's no-grad set equals the sequential path's, an oracle with no allowlist.
    nograd = sorted(n for n, p in model.named_parameters() if p.grad is None)
    print(f"GRAD_OK: {len(nograd)} no-grad params (design-dead), 0 non-finite")
torch.save({"flat": flat.detach(), "nograd": nograd}, OUT)
print(f"rows={data.card_features.size(0)} out_numel={flat.numel()} "
      f"checksum={flat.double().sum().item():.10f} scale={flat.abs().max().item():.4f}")
assert flat.abs().max().item() > 1e-3, "outputs ~zero -- comparison would be vacuous"
print("CHILD_OK")
"""


def run_child(tag, extra_env, out_path):
    env = dict(os.environ, PYTHONPATH=REPO, SMOKE_OUT=out_path, **extra_env)
    env.pop("RWKV_NO_JIT", None)   # JIT ON: construction must compile
    p = subprocess.run([sys.executable, "-c", CHILD], cwd=REPO, env=env,
                       capture_output=True, text=True)
    lines = (p.stdout + p.stderr).strip().splitlines()
    print(f"--- {tag}")
    for ln in lines[-4:]:
        print("    " + ln)
    if p.returncode != 0 or not any("CHILD_OK" in ln for ln in lines):
        print("FAILED CHILD, full output:")
        print(p.stdout + p.stderr)
        sys.exit(1)


def main():
    import torch
    if not os.path.isdir(os.path.join(REPO, "train_db_5k_h1")):
        print("SKIP: train_db_5k_h1 not present")
        sys.exit(0)
    td = tempfile.mkdtemp(prefix="ilv_smoke_")
    f = {k: os.path.join(td, k + ".pt") for k in ("d1_off", "d1_on", "full_off", "full_on")}

    run_child("depths=1 sequential", {"SMOKE_DEPTH1": "1"}, f["d1_off"])
    run_child("depths=1 interleaved", {"SMOKE_DEPTH1": "1", "RWKV_INTERLEAVE": "1"}, f["d1_on"])
    a, b = torch.load(f["d1_off"])["flat"], torch.load(f["d1_on"])["flat"]
    assert torch.equal(a, b), \
        f"NOT bit-identical at depth 1: max|d|={(a - b).abs().max().item():.3e}"
    print("[1] depths-all-1: interleaved == sequential BIT-EXACT "
          f"(n={a.numel()}, checksum {a.double().sum().item():.10f})")

    run_child("real depths sequential + grad", {"SMOKE_GRAD": "1"}, f["full_off"])
    run_child("real depths interleaved + grad", {"RWKV_INTERLEAVE": "1", "SMOKE_GRAD": "1"},
              f["full_on"])
    da, db = torch.load(f["full_off"]), torch.load(f["full_on"])
    a, b = da["flat"], db["flat"]
    d = (a - b).abs().max().item()
    assert d > 1e-6, "real-depth interleave did NOT change the output -- flag is a silent null"
    assert da["nograd"] == db["nograd"], (
        "no-grad param sets differ; seq-only: "
        + str(sorted(set(da["nograd"]) - set(db["nograd"]))[:6])
        + " ilv-only: " + str(sorted(set(db["nograd"]) - set(da["nograd"]))[:6]))
    print(f"[2] real depths: outputs differ as they must (max|d|={d:.3e}); "
          f"no-grad sets IDENTICAL ({len(da['nograd'])} design-dead params both paths)")
    print("\nSMOKE OK -- composition bit-exact at the depth-1 oracle, live at real depths")


if __name__ == "__main__":
    main()
