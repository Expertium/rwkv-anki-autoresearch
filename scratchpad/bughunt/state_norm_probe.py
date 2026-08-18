"""Does the production state clamp (RWKV_STATE_CLAMP_TAU=300) ever BIND on the champion?

WHY THIS IS A THREE-WAY-PARITY QUESTION. The clamp is implemented in all three paths, but with
DELIBERATELY different granularity, and `rwkv_rnn_model.py` says so in as many words:

    training clamps between WINDOWS of state_clamp_window (32768) steps, and only when the chunk
    is longer than one window; an RNN sees one step at a time and has no access to the training
    chunk boundaries, so it clamps EVERY step.

The two agree **exactly** wherever `||S||_F` stays under tau, because the factor is then exactly
`tau/max(tau, ||S||) = 1.0` and the multiply is bit-inert. They differ only on states that were
already diverging. So the whole train-vs-deploy question reduces to ONE empirical fact:

    does ||S||_F ever approach 300 on a healthy champion?

If it does not, all three paths compute the same quantity and there is nothing to fix -- a negative
result worth recording, because the alternative is an unfalsified suspicion sitting under the deploy
contract. If it does, training and deploy genuinely diverge on those users and the clamp becomes a
real correctness item.

`scratchpad/parity3/parity_train_vs_rnn.py` cannot answer this: it deliberately SKIPS the parity
assertion for a binding tau (the training clamp is CUDA-only, so a CPU run of the training path does
not clamp at all), and its synthetic model carries ||S|| ~ 0.38, which says nothing about the real
one. No past run logged norms either -- RWKV_STATE_CLAMP_LOG has never been switched on.

METHOD. Run the champion through the DEPLOY RNN path on CPU and record `max ||S||_F` per step,
by wrapping `clamp_state` at runtime IN THIS PROCESS ONLY. No repo file is touched -- editing
`rwkv/*.py` while a chain is running would silently change the next phase, which is a documented
trap here. Reports the maximum over all steps, all heads, all five streams.

Usage: .venv/Scripts/python.exe scratchpad/bughunt/state_norm_probe.py [user] [max_rows]
"""
import os
import sys

# The arch env MUST be set before the import: old-style ScriptModule bakes the first
# construction's flags into the compiled class, so a late setenv is silently ignored.
ENV = {
    "RWKV_ARCH_MODULE": "scratchpad/track2_a18/architecture_d80_lora4_cnd.py",
    "RWKV_INTERLEAVE": "1",
    "RWKV_GRU_HEAD": "3",
    "RWKV_PAVA_LAMBDA": "0.2",
    "RWKV_NO_AHEAD_RESIDUAL": "1",
    "RWKV_STRIP_L0_VLORA": "1",
    "RWKV_ZERO_FEATURES": "22",
    "RWKV_STRIP_CMIX": "user_id:0,user_id:1,user_id:2,preset_id:0,preset_id:1,"
                       "preset_id:2,deck_id:1,deck_id:2,card_id:1",
    "RWKV_STATE_CLAMP_TAU": "300",
    "RWKV_STATE_CLAMP_WINDOW": "32768",
    "RWKV_MUON": "1",
    "RWKV_MUON_INCLUDE_LORA": "1",
    "RWKV_NO_JIT": "1",
}
for k, v in ENV.items():
    os.environ.setdefault(k, v)

import torch  # noqa: E402

sys.path.insert(0, os.getcwd())
from rwkv.model import rwkv_rnn_model  # noqa: E402

TAU = float(ENV["RWKV_STATE_CLAMP_TAU"])
seen = {"max": 0.0, "steps": 0, "binds": 0}
_orig = rwkv_rnn_model.RWKV7RNNTimeMixer.clamp_state


def probed(self, state_BHKK):
    """Record the norm BEFORE clamping -- after it, a binding state reads exactly tau and the
    evidence of how far it overshot is gone."""
    if self.state_clamp_tau > 0.0:
        n = state_BHKK.flatten(2).norm(p=2.0, dim=-1)
        finite = n[torch.isfinite(n)]
        if finite.numel():
            m = float(finite.max())
            seen["max"] = max(seen["max"], m)
            seen["steps"] += 1
            if m > TAU:
                seen["binds"] += 1
    return _orig(self, state_BHKK)


rwkv_rnn_model.RWKV7RNNTimeMixer.clamp_state = probed

CKPT = "scratchpad/iter53_muonlora/i53_d_10935.pth"
user = int(sys.argv[1]) if len(sys.argv) > 1 else 5001
max_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 40000


class Enough(Exception):
    """Stop the walk once enough steps are sampled -- the module-level run() has no row limit."""


_MAX = max_steps


def probed2(self, state_BHKK):
    r = probed(self, state_BHKK)
    if seen["steps"] >= _MAX:
        raise Enough()
    return r


rwkv_rnn_model.RWKV7RNNTimeMixer.clamp_state = probed2

print("probing %s on user %d (<= %d clamp calls), tau=%g" % (CKPT, user, max_steps, TAU))
if not os.path.exists(CKPT):
    print("MISSING CHECKPOINT -- the champion .pth is untracked and single-machine")
    sys.exit(2)

from pathlib import Path  # noqa: E402

from rwkv.run_as_rnn import run as rnn_run  # noqa: E402

try:
    rnn_run(
        data_path=Path("../anki-revlogs-10k"),
        model_path=CKPT,
        label_db_path="label_filter_db",
        label_db_size=40_000_000_000,
        user_id=user,
        verbose=False,
    )
except Enough:
    print("(stopped early at the step cap)")
except Exception as e:  # noqa: BLE001
    print("walk ended: %s: %s" % (type(e).__name__, e))

print("")
print("--- RESULT")
print("clamp_state calls          : %d" % seen["steps"])
print("max ||S||_F over all steps : %.4f" % seen["max"])
print("tau                        : %.1f" % TAU)
print("steps where the clamp BOUND: %d" % seen["binds"])
if seen["steps"] == 0:
    print("INCONCLUSIVE -- clamp_state was never called; the hook did not attach.")
elif seen["binds"] == 0:
    print("DOES NOT BIND: headroom %.0fx. Train and deploy compute the SAME quantity here,"
          % (TAU / max(seen["max"], 1e-9)))
    print("because the clamp factor is exactly 1.0 and the multiply is bit-inert.")
else:
    print("BINDS -- training (per 32768-step window) and deploy (per step) DIVERGE here.")
