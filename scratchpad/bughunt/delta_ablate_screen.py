"""What does the delta rule actually BUY? Inference-time ablation on the champion. CPU, no GPU.

THE STRONGER FORM OF ANDREW'S QUESTION (2026-08-19). "Simplify the delta rule to cut parameters"
measured out weak on the parameter argument: `a_lora` + `k_scale` are 14,625 params (2.62%), the cuts
are NOT free (36.4% / 20.0% of their variance is token-to-token, which a bias cannot reproduce), and
-- decisively -- they are WEIGHTS, not STATE. The binding deploy budget here is per-card state
(9 B/card, frozen), which these do not touch at all.

The interesting question is one level up: **does the delta rule earn its place on this task?** The
2026-08-17 lit review measured that the delta term moves the state-transition eigenvalue by only
~0.15 against a decay of ~0.98 -- "our trunk uses its WKV state almost as a pure exponential-decay
accumulator with a small rank-1 correction; RWKV-7's headline innovation is barely engaged". If that
holds, dropping the term removes `a_lora` (9,360 params) AND the delta work in the WKV recurrence --
which is KERNEL TIME, in training and in the Rust CPU deploy path, where an arithmetic cut has been
measured to convert at 2.39x. That is a real prize, unlike tens of KB of weights.

THIS SCREEN IS DIRECTIONAL, AND ITS ASYMMETRY IS THE POINT. Zeroing `a` on a model TRAINED with the
delta rule is NOT the same experiment as training without it: the weights co-adapted to it, so this
is an UPPER BOUND on the damage, not an estimate of it. That asymmetry is what makes it a good
screen --

    catastrophic here -> the delta rule is load-bearing. Do NOT spend 9.2 h on a retrain.
    mild here         -> a retrain can plausibly recover it. The GPU run is justified.

It reuses `run_as_rnn.run()` unchanged, so the metric is the project's real equalized LogLoss rather
than a hand-rolled one, captured by wrapping `get_stats`. Paired within user.

Usage: .venv/Scripts/python.exe scratchpad/bughunt/delta_ablate_screen.py [n_users]
"""
import os
import sys

ENV = {
    "RWKV_ARCH_MODULE": "scratchpad/track2_a18/architecture_d80_lora4_cnd.py",
    "RWKV_INTERLEAVE": "1", "RWKV_GRU_HEAD": "3", "RWKV_PAVA_LAMBDA": "0.2",
    "RWKV_NO_AHEAD_RESIDUAL": "1", "RWKV_STRIP_L0_VLORA": "1", "RWKV_ZERO_FEATURES": "22",
    "RWKV_STRIP_CMIX": "user_id:0,user_id:1,user_id:2,preset_id:0,preset_id:1,"
                       "preset_id:2,deck_id:1,deck_id:2,card_id:1",
    "RWKV_STATE_CLAMP_TAU": "300", "RWKV_STATE_CLAMP_WINDOW": "32768",
    "RWKV_MUON": "1", "RWKV_MUON_INCLUDE_LORA": "1", "RWKV_NO_JIT": "1",
}
for k, v in ENV.items():
    os.environ.setdefault(k, v)

import torch  # noqa: E402

sys.path.insert(0, os.getcwd())
from rwkv.model import rwkv_rnn_model  # noqa: E402
import rwkv.run_as_rnn as ras          # noqa: E402
from pathlib import Path               # noqa: E402

CKPT = "scratchpad/iter53_muonlora/i53_d_10935.pth"
# Smallest VAL-half users, so a paired two-arm screen finishes in minutes rather than hours.
USERS = [5044, 5100, 5063, 5097, 5048, 5030]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 4

# ---- the ablation ---------------------------------------------------------------------------
# `a = sigmoid(a_lora(x))`, and the delta term is `a * kappa kappa^T`. Forcing the PRE-sigmoid
# logit very negative sends a -> ~1e-13 and leaves every other path untouched, so this isolates
# the delta term rather than the LoRA's contribution to anything else.
ABLATE = {"on": False}
_cls = rwkv_rnn_model.RWKV7RNNTimeMixer
_orig_init = _cls.__init__


def _patched(self, *a, **kw):
    _orig_init(self, *a, **kw)
    if hasattr(self, "a_lora_simple"):
        self.a_lora_simple.register_forward_hook(
            lambda m, i, o: torch.full_like(o, -30.0) if ABLATE["on"] else None)


_cls.__init__ = _patched

# ---- capture the real metric ----------------------------------------------------------------
grabbed = []
_orig_stats = ras.get_stats


def _stats(*a, **kw):
    out = _orig_stats(*a, **kw)
    grabbed.append(out[0])
    return out


ras.get_stats = _stats


def score(user, ablate):
    """Returns (imm_logloss, ahead_logloss) from the project's own get_stats."""
    ABLATE["on"] = ablate
    grabbed.clear()
    ras.run(data_path=Path("../anki-revlogs-10k"), model_path=CKPT,
            label_db_path="label_filter_db", label_db_size=40_000_000_000,
            user_id=user, verbose=False)
    vals = []
    for g in grabbed[:2]:
        m = g.get("metrics", g) if isinstance(g, dict) else {}
        vals.append(float(m.get("LogLoss", float("nan"))))
    return vals if len(vals) == 2 else [float("nan")] * 2


print("delta-rule inference ablation, champion %s" % CKPT)
print("%d smallest VAL users, paired within user\n" % N)
print("%-8s %19s %19s %10s" % ("user", "baseline imm/ahead", "a=0 imm/ahead", "d_imm"))
print("-" * 62)
di, da = [], []
for u in USERS[:N]:
    try:
        b = score(u, False)
        a = score(u, True)
    except Exception as e:  # noqa: BLE001
        print("%-8d skipped (%s: %s)" % (u, type(e).__name__, str(e)[:40]))
        continue
    if any(v != v for v in b + a):
        print("%-8d unusable (nan)" % u)
        continue
    di.append(a[0] - b[0])
    da.append(a[1] - b[1])
    print("%-8d %9.5f %9.5f %9.5f %9.5f %+10.5f"
          % (u, b[0], b[1], a[0], a[1], a[0] - b[0]))

if di:
    mi = sum(di) / len(di)
    ma = sum(da) / len(da)
    print("")
    print("mean cost of removing the delta rule (INFERENCE-TIME UPPER BOUND):")
    print("   imm   %+.5f" % mi)
    print("   ahead %+.5f" % ma)
    worst = max(mi, ma)
    print("")
    if worst > 0.02:
        print("=> LOAD-BEARING. Far too large to call the mechanism idle. A retrain would recover")
        print("   some of it, but do NOT queue a 9.2 h delta-ablation run on the")
        print("   'barely engaged' argument -- the eigenvalue statistic is not the whole story.")
    elif worst > 0.005:
        print("=> MODERATE. Co-adaptation could explain much of this. A retrain is defensible,")
        print("   but budget for a real cost, not a free simplification.")
    else:
        print("=> NEARLY IDLE even BEFORE retraining. RWKV-7's headline mechanism is doing little")
        print("   here, and a trained-without-delta run is well justified: it cuts 9,360 params")
        print("   AND the delta work in the WKV kernel, which the Rust path converts to wall clock.")
