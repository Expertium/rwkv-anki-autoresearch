"""Correctness check for the perm_gather/perm_scatter rewrite of RWKV_INTERLEAVE's per-round
gather/scatter (rwkv/model/srs_model.py::_interleaved_streams).

The OLD path used torch.index_select (clamp+index) for the read and x.index_copy for the
write -- both fall through to PyTorch's deterministic-mode sort-based backward under
RWKV_DETERMINISTIC=1, which _PermGather's own docstring measures at ~43% of a step for the
(structurally identical) stream-gather case. The NEW path exploits the same permutation
invariant (every canonical row is referenced at most once per round) via perm_gather (existing,
already used by the sequential form) and the new perm_scatter (_PermScatterWrite).

Both paths are reachable in the SAME process via RWKV_PERM_GATHER / RWKV_PERM_SCATTER (escape
hatches, default ON = the new path), so unlike smoke_interleave.py (which needs a fresh
ScriptModule per env combo) this can flip mid-process as long as the model stays in eager mode
(RWKV_NO_JIT=1) -- the perm_gather/perm_scatter branch is chosen by a plain Python bool read at
call time, not baked into a compiled graph.

Checks, at REAL depths (not the depth-1 oracle -- this validates a live, non-trivial round
schedule):
  1. old vs new forward output: bit-identical (torch.equal, not "close").
  2. old vs new gradient for EVERY parameter: bit-identical (torch.equal per-tensor).
  3. sanity: output is non-trivial (not all-zero) so equality isn't vacuous.

Run:  .venv/bin/python scratchpad/parity3/smoke_interleave_permscatter.py
"""
import dataclasses
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
os.chdir(REPO)
os.environ["RWKV_NO_JIT"] = "1"
os.environ["RWKV_INTERLEAVE"] = "1"
os.environ.setdefault("RWKV_ARCH_MODULE", "scratchpad/track2_a18/architecture_d80_lora4_cnd.py")
os.environ.setdefault("RWKV_GRU_HEAD", "3")
os.environ.setdefault("RWKV_STRIP_L0_VLORA", "1")
os.environ.setdefault("RWKV_ZERO_FEATURES", "22")
os.environ.setdefault("RWKV_STATE_CLAMP_TAU", "300")
os.environ.setdefault("RWKV_STATE_CLAMP_WINDOW", "32768")
os.environ.setdefault(
    "RWKV_STRIP_CMIX",
    "user_id:0,user_id:1,user_id:2,preset_id:0,preset_id:1,preset_id:2,deck_id:1,deck_id:2,card_id:1",
)
os.environ.setdefault("RWKV_NO_AHEAD_RESIDUAL", "1")

import lmdb
import torch

from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG
from rwkv.model.srs_model import SrsRWKV
from rwkv.prepare_batch import get_data, prepare


def build_model():
    torch.manual_seed(7)
    model = SrsRWKV(anki_rwkv_config=DEFAULT_ANKI_RWKV_CONFIG)
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0.0, 0.1)  # zero-init params would make agreement partly vacuous
    model = model.float()
    model.eval()  # dropout off: both runs must be deterministic
    return model


def load_batch():
    env = lmdb.open("train_db_5k_h1", map_size=400_000_000_000, readonly=True, lock=False)
    with env.begin(write=False) as txn:
        batches = json.loads(txn.get(b"101_batches"))
        b = min(batches, key=lambda x: x[2])  # smallest chunk of user 101
        key = (101, b[0], b[1], b[2])
        data = get_data(txn, key, device="cpu")
    env.close()
    pb = prepare([data], seed=1234, probe_density=0.08)
    return pb.to("cpu")


def run(model, pb, use_perm):
    for m in model.modules():
        if hasattr(m, "use_perm_gather"):
            m.use_perm_gather = use_perm
        if hasattr(m, "use_perm_scatter"):
            m.use_perm_scatter = use_perm
    model.zero_grad(set_to_none=True)
    torch.set_grad_enabled(True)
    out = model.forward_batch(
        pb.start.float(), pb.sub_gather, pb.sub_gather_lens,
        pb.time_shift_selects, pb.skips, pb.num_data,
    )
    flat = torch.cat([o.float().flatten() for o in out])
    loss = flat.square().mean()
    loss.backward()
    grads = {n: (p.grad.clone() if p.grad is not None else None)
             for n, p in model.named_parameters()}
    return flat.detach().clone(), grads


def main():
    if not os.path.isdir(os.path.join(REPO, "train_db_5k_h1")):
        print("SKIP: train_db_5k_h1 not present")
        sys.exit(0)

    model = build_model()
    pb = load_batch()

    print("--- old path (RWKV_PERM_GATHER=0, RWKV_PERM_SCATTER=0: index_select+index_copy)")
    out_old, grad_old = run(model, pb, use_perm=False)
    print(f"    checksum={out_old.double().sum().item():.10f} scale={out_old.abs().max().item():.4f}")
    assert out_old.abs().max().item() > 1e-3, "outputs ~zero -- comparison would be vacuous"

    print("--- new path (RWKV_PERM_GATHER=1, RWKV_PERM_SCATTER=1: perm_gather+perm_scatter)")
    out_new, grad_new = run(model, pb, use_perm=True)
    print(f"    checksum={out_new.double().sum().item():.10f} scale={out_new.abs().max().item():.4f}")

    assert torch.equal(out_old, out_new), (
        f"FORWARD NOT bit-identical: max|d|={(out_old - out_new).abs().max().item():.3e}"
    )
    print(f"[1] forward: bit-identical (n={out_old.numel()})")

    assert set(grad_old.keys()) == set(grad_new.keys())
    n_checked, n_none_both, bad = 0, 0, []
    for name in grad_old:
        go, gn = grad_old[name], grad_new[name]
        if go is None and gn is None:
            n_none_both += 1
            continue
        if go is None or gn is None:
            bad.append((name, "one-sided None"))
            continue
        if not torch.equal(go, gn):
            d = (go - gn).abs().max().item()
            bad.append((name, f"max|d|={d:.3e}"))
            continue
        n_checked += 1
    if bad:
        print("MISMATCHES:")
        for name, why in bad[:20]:
            print(f"    {name}: {why}")
        sys.exit(1)
    print(f"[2] gradients: bit-identical for {n_checked} params "
          f"({n_none_both} None-in-both by design, matches the depth-1 oracle's no-grad set)")
    print("\nSMOKE OK -- perm_gather/perm_scatter rewrite is bit-exact vs index_select/index_copy")


if __name__ == "__main__":
    main()
