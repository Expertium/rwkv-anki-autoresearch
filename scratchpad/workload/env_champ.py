"""The iter-53 champion's INFERENCE env, set before any rwkv import.

Only the flags that define the FORWARD PASS are here. Training-only flags (Muon, KD,
dropout, clip, weight decay, augmentation seed, empty-cache, probe density, JIT/compile)
are deliberately absent -- they cannot change what a frozen checkpoint computes, and
carrying them would invite the "cloned runner" failure of copying strings that no longer
apply.

Sourced from scratchpad/iter53_muonlora/run_iter53.cmd, cross-checked against
scratchpad/features_ab/run_featA2_evalonly.cmd (the eval-only slice of the same lineage).

IMPORT THIS BEFORE `rwkv.*`. Old-style ScriptModule bakes the first construction's env
flags into the compiled class, so a late set() is silently ignored.
"""
import os

CHAMPION_CKPT = "scratchpad/iter53_muonlora/i53_d_10935.pth"

_ENV = {
    "RWKV_ARCH_MODULE": "scratchpad/track2_a18/architecture_d80_lora4_cnd.py",
    "RWKV_INTERLEAVE": "1",
    "RWKV_GRU_HEAD": "3",
    "RWKV_PAVA_LAMBDA": "0.2",
    "RWKV_NO_AHEAD_RESIDUAL": "1",
    "RWKV_STRIP_L0_VLORA": "1",
    "RWKV_ZERO_FEATURES": "22",
    "RWKV_STATE_CLAMP_TAU": "300",
    "RWKV_STATE_CLAMP_WINDOW": "32768",
    "RWKV_STRIP_CMIX": (
        "user_id:0,user_id:1,user_id:2,preset_id:0,preset_id:1,preset_id:2,"
        "deck_id:1,deck_id:2,card_id:1"
    ),
    # deploy contract point 1: the most recent review's duration is zeroed on the
    # scheduling probe. button_heads reads this.
    "RWKV_PROBE_DUR": "0.0",
    # inference is single-user and CPU-bound here; 1 thread is the measured optimum
    # (CLAUDE.md: "1 thread beats 3 and 6 -> deploy single-threaded").
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}


def apply():
    for k, v in _ENV.items():
        os.environ[k] = v
    # ID_FEATURES must be OFF: the champion is a published-data model (width 92).
    os.environ.pop("RWKV_ID_FEATURES", None)
    return dict(_ENV)
