"""Unit test of rwkv.sam.sam_second_pass on CPU, fp32, with the REAL model, a REAL gen-5 chunk and the
REAL training loss (get_loss), under realcyc's arch env. Exercises every line of the second pass
except the bf16 downcast (copy_downcast_ is called with float32 here):
  1. after the pass, master's grads are the gradient at w + rho*g/||g||: they DIFFER from g, and with a
     tiny rho they converge back to g (continuity check, relative diff < 1e-3 at rho=1e-6);
  2. the weights are restored bit-exactly (asserted inside the function, and re-checked here);
  3. the same dropout masks are drawn (model.train() + RNG restore): the second-pass loss at rho=0
     equals the first-pass loss exactly;
  4. transfer() ACCUMULATES, so the pass must zero master grads first -- checked by comparing against
     an independent gradient computed at the perturbed point.
CPU, ~5 min (one chunk of user 101, a few forward+backward passes).
"""
import copy
import os
import sys

sys.path.insert(0, os.getcwd())
for k in list(os.environ):
    if k.startswith("RWKV_"):
        del os.environ[k]
os.environ.update({
    "RWKV_ARCH_MODULE": "scratchpad/track2_a18/architecture_d80_lora4_cnd.py",
    "RWKV_INTERLEAVE": "1", "RWKV_GRU_HEAD": "3", "RWKV_PAVA_LAMBDA": "0.2",
    "RWKV_NO_AHEAD_RESIDUAL": "1", "RWKV_STRIP_L0_VLORA": "1",
    "RWKV_STATE_CLAMP_TAU": "300", "RWKV_STATE_CLAMP_WINDOW": "32768",
    "RWKV_STRIP_CMIX": "user_id:0,user_id:1,user_id:2,preset_id:0,preset_id:1,preset_id:2,deck_id:1,deck_id:2,card_id:1",
    "RWKV_ID_FEATURES": "1", "RWKV_REAL_CYCLES": "1", "RWKV_ZERO_FEATURES": "", "RWKV_NO_JIT": "1",
    "RWKV_SAM_RHO": "0.05", "RWKV_SAM_EVERY": "1",
})
import json
import lmdb
import torch

torch.set_num_threads(6)
from rwkv import sam
from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG
from rwkv.model.srs_model import SrsRWKV
from rwkv.prepare_batch import get_data, prepare
from rwkv.train_rwkv import transfer_child_grad_to_master

sys.path.insert(0, "scratchpad/proposals_2026-09-04")
from sam_probe import to_f32  # noqa: E402

DB = "F:/rwkv_lmdb/train_db_5k_h1_id5"


def load_chunk():
    env = lmdb.open(DB, map_size=400_000_000_000, readonly=True, lock=False)
    with env.begin(write=False) as txn:
        b = min(json.loads(txn.get(b"101_batches")), key=lambda x: x[2])
        pb = to_f32(prepare([get_data(txn, (101, b[0], b[1], b[2]), device="cpu")], seed=1234, probe_density=0.08).to("cpu"))
    env.close()
    return pb


def grads_of(m):
    return [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in m.parameters()]


def main():
    torch.manual_seed(0)
    master = SrsRWKV(anki_rwkv_config=DEFAULT_ANKI_RWKV_CONFIG).float()
    master.load_state_dict(torch.load("scratchpad/realcyc/rc_d_10935.pth", map_location="cpu", weights_only=True), strict=True)
    child = copy.deepcopy(master)
    pb = load_chunk()
    ok = True

    def check(c, msg):
        nonlocal ok
        print(("  PASS " if c else "  FAIL ") + msg, flush=True); ok = ok and c

    def one_pass(rho, train_mode=True):
        """first forward/backward on the child (dropout per train_mode), transfer to master, then the
        SAM second pass at rho; returns (first loss, master grads before, master grads after)."""
        sam.SAM_RHO = rho
        for p in master.parameters():
            p.grad = None
        child.zero_grad(set_to_none=True)
        child.copy_downcast_(master, dtype=torch.float32)
        child.train(train_mode)
        torch.manual_seed(123)
        rng = sam.save_rng()
        st = child.get_loss(pb)
        st.average_loss.backward()
        transfer_child_grad_to_master(master=master, child=child)
        g_before = grads_of(master)
        losses = []

        def fb():
            st2 = child.get_loss(pb)
            losses.append(float(st2.average_loss))
            st2.average_loss.backward()
        w_before = [p.detach().clone() for p in master.parameters()]
        sam.sam_second_pass(master, child, torch.float32, rng, fb, lambda: transfer_child_grad_to_master(master=master, child=child))
        w_after = [p.detach().clone() for p in master.parameters()]
        return float(st.average_loss), g_before, grads_of(master), losses[0], w_before, w_after

    # 3. rho = 0: same dropout masks => second-pass loss == first-pass loss exactly, grads identical
    l0, g0, g0b, l0b, w0, w0b = one_pass(0.0)
    check(l0 == l0b, f"rho=0, train mode: second-pass loss == first-pass loss ({l0:.6f}) -> same dropout masks")
    check(all(torch.equal(a, b) for a, b in zip(g0, g0b)), "rho=0: master grads unchanged (zeroed-then-recomputed == first pass, so transfer did not double-count)")
    check(all(torch.equal(a, b) for a, b in zip(w0, w0b)), "rho=0: weights restored bit-exactly")

    # 1. rho = 0.05: grads differ from g; weights restored
    l1, g1, g1b, l1b, w1, w1b = one_pass(0.05)
    diff = torch.sqrt(sum(((a - b) ** 2).sum() for a, b in zip(g1, g1b))) / torch.sqrt(sum((a ** 2).sum() for a in g1))
    check(float(diff) > 1e-3, f"rho=0.05: master grads are the gradient at the perturbed point (relative change {float(diff):.4f})")
    check(l1b > l1, f"rho=0.05: the loss at w+e ({l1b:.6f}) exceeds the loss at w ({l1:.6f}) -- an ascent step")
    check(all(torch.equal(a, b) for a, b in zip(w1, w1b)), "rho=0.05: weights restored bit-exactly")

    # continuity: tiny rho -> grads converge to g
    l2, g2, g2b, _, _, _ = one_pass(1e-6)
    diff2 = torch.sqrt(sum(((a - b) ** 2).sum() for a, b in zip(g2, g2b))) / torch.sqrt(sum((a ** 2).sum() for a in g2))
    check(float(diff2) < 1e-3, f"rho=1e-6: grads converge back to g (relative change {float(diff2):.2e})")

    # 4. independent check of the perturbed gradient in eval mode (no dropout): perturb by hand, backward once
    sam.SAM_RHO = 0.05
    l3, g3, g3b, _, _, _ = one_pass(0.05, train_mode=False)
    gnorm = torch.sqrt(sum((g ** 2).sum() for g in g3))
    with torch.no_grad():
        for p, g in zip(master.parameters(), g3):
            p.add_(g * (0.05 / (gnorm + 1e-12)))
    child.copy_downcast_(master, dtype=torch.float32); child.eval(); child.zero_grad(set_to_none=True)
    child.get_loss(pb).average_loss.backward()
    g_ref = grads_of(child)
    with torch.no_grad():
        for p, g in zip(master.parameters(), g3):
            p.sub_(g * (0.05 / (gnorm + 1e-12)))
    diff3 = torch.sqrt(sum(((a - b) ** 2).sum() for a, b in zip(g3b, g_ref))) / torch.sqrt(sum((a ** 2).sum() for a in g_ref))
    check(float(diff3) < 1e-5, f"eval mode: second-pass grads == an independent gradient at w+e (relative diff {float(diff3):.2e})")
    print("SAM_UNIT " + ("PASS" if ok else "FAIL"))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
