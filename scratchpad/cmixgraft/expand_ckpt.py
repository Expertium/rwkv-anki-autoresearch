"""Graft the six stripped user/preset channel mixers onto ws10's WS-final, function-preservingly.

THE POINT. Arm 1's Q2 says the 4.95x parameter reduction costs ~0.003 ahead once the budget is
real, and every capacity reject in the record was measured at ~1.25 epochs -- a budget that
provably cannot expose a capacity limit. Testing capacity properly needs a new ~38-60 h WS run,
because an architecture change cannot branch off a shared WS checkpoint. Andrew, 2026-09-10: *"It
would be nice to try to grow deck/preset/global, but 38 h is a pretty steep price."*

It does not have to cost that, because THIS capacity change is function-preserving at init. A
channel mixer returns `in_BTC + dropout(W_v(relu(W_k x)^2))` and `RWKV7ChannelMixer.__init__`
already does `self.W_v.weight.data.zero_()` -- so a freshly constructed mixer is the IDENTITY, and
a model with six of them grafted on starts exactly where ws10 ended. The 2-epoch decay then trains
the new capacity, with arm 1 as the exact control: same WS checkpoint, same decay length, same env
but for `RWKV_STRIP_CMIX`.

WHAT IS GRAFTED. `RWKV_STRIP_CMIX` currently strips nine channel mixers; six of them are
`user_id:0,1,2` and `preset_id:0,1,2`. Those six are restored. This is iter 49's lever exactly
(+26,070 params, +4.7%), which returned a null -- at 1.25 epochs.

⚠⚠ THE SECOND VARIABLE, FOUND BY READING THE BLOCK RATHER THAN THE MIXER, AND IT IS NOT ZERO.
`Block.forward` returns the mixer's output wrapped in `self.dropout` (`dropout_layer`), while the
STRIPPED branch returns `x_BTC` with no dropout at all. `DROPOUT_LAYER = 0.01 * RWKV_DROPOUT_SCALE`
= **0.005** in this recipe. So restoring a mixer also switches on p=0.005 layer dropout for that
block. The graft is therefore identity in EVAL mode exactly, and in TRAIN mode it is identity plus
0.005 layer dropout on six of thirteen layer-steps.
**It is not being "fixed", because every fix is a bigger variable than the bug:** zeroing
`dropout_layer` for those blocks alone is a code change that no other run has, and changing
`RWKV_DROPOUT_SCALE` moves the whole model. It is priced instead -- see PREREG.md. The direction
helps: extra dropout is extra regularisation, and arm 1's gap screen says this model is FIT-limited
at this budget, so it should if anything HURT the treatment arm. A WIN therefore survives the
confound; a NULL is confounded and must not be reported as "capacity does not bind".

Usage: python scratchpad/cmixgraft/expand_ckpt.py [out.pth]
"""
import os
import sys

os.chdir(r"C:\Users\Andrew\rwkv-anki-autoresearch")
sys.path.insert(0, os.getcwd())

SRC = "scratchpad/ws10/w10_ws_109350.pth"
# Named `w10g_ws_<step>` on purpose: write_decay_setup.py finds the WS checkpoint by
# globbing "<prefix>_<step>.pth" and parsing the step as an int, so a suffixed name like
# w10_ws_109350_cmixgraft.pth either fails to match or parses a non-integer step. Give the
# graft its OWN prefix instead of decorating ws10's.
OUT = sys.argv[1] if len(sys.argv) > 1 else "scratchpad/ws10/w10g_ws_109350.pth"

STRIP_ARM1 = ("user_id:0,user_id:1,user_id:2,preset_id:0,preset_id:1,preset_id:2,"
              "deck_id:1,deck_id:2,card_id:1")
STRIP_GRAFT = "deck_id:1,deck_id:2,card_id:1"

_ENV = {
    "RWKV_ARCH_MODULE": "scratchpad/track2_a18/architecture_d80_lora4_cnd.py",
    "RWKV_INTERLEAVE": "1", "RWKV_GRU_HEAD": "3", "RWKV_PAVA_LAMBDA": "0.2",
    "RWKV_NO_AHEAD_RESIDUAL": "1", "RWKV_STRIP_L0_VLORA": "1",
    "RWKV_STATE_CLAMP_TAU": "300", "RWKV_STATE_CLAMP_WINDOW": "32768",
    "RWKV_ID_FEATURES": "1", "RWKV_REAL_CYCLES": "1", "RWKV_ZERO_FEATURES": "",
    "RWKV_DROPOUT_SCALE": "0.5", "RWKV_NO_JIT": "1",
}
for k, v in _ENV.items():
    os.environ[k] = v
os.environ["RWKV_STRIP_CMIX"] = STRIP_GRAFT

import torch                                                     # noqa: E402
from rwkv.model.srs_model import SrsRWKV                         # noqa: E402
import rwkv.architecture as arch                                 # noqa: E402


def main():
    sd_src = torch.load(SRC, map_location="cpu", weights_only=False)
    if isinstance(sd_src, dict) and "model" in sd_src and not any(
            k.startswith("features2card") for k in sd_src):
        sd_src = sd_src["model"]
    print(f"source {SRC}: {len(sd_src)} tensors")

    model = SrsRWKV(arch.DEFAULT_ANKI_RWKV_CONFIG)
    sd_new = model.state_dict()
    print(f"grafted model: {len(sd_new)} tensors, "
          f"{sum(p.numel() for p in model.parameters()):,} params")

    missing = [k for k in sd_new if k not in sd_src]
    unexpected = [k for k in sd_src if k not in sd_new]
    if missing or unexpected:
        raise SystemExit(f"REFUSING: key sets differ ({len(missing)} missing, "
                         f"{len(unexpected)} unexpected) -- the architectures disagree about "
                         f"more than the six channel mixers")

    # A STRIPPED mixer is NOT absent: rwkv_model.py:578-581 builds it from a d_model=1 dummy
    # config, deliberately, so checkpoint interchange keeps working. So the graft is a SHAPE
    # replacement, not a missing-key fill, and looking for missing keys finds nothing at all --
    # which is how the first version of this script "succeeded" into refusing.
    grow = [k for k in sd_new if tuple(sd_new[k].shape) != tuple(sd_src[k].shape)]
    restored = set()
    for k in grow:
        if ".channel_mixer." not in k:
            raise SystemExit(f"REFUSING: {k} changes shape and is not a channel-mixer key")
        restored.add(k.split(".channel_mixer.")[0])
    if len(restored) != 6:
        raise SystemExit(f"REFUSING: expected 6 restored mixers, found {len(restored)}: "
                         f"{sorted(restored)}")
    print(f"restored mixers ({len(restored)}):")
    for r in sorted(restored):
        print(f"  {r}")
    added = sum(sd_new[k].numel() - sd_src[k].numel() for k in grow)
    print(f"{len(grow)} tensors change shape, adding {added:,} params")

    out = dict(sd_src)
    for k in grow:
        out[k] = sd_new[k].clone()

    # THE FUNCTION-PRESERVING CLAIM, asserted rather than trusted: every restored mixer's OUTPUT
    # projection must be exactly zero, which makes `in + dropout(W_v(...))` == `in`.
    nz = [k for k in grow if k.endswith("W_v.weight") and out[k].abs().max().item() != 0.0]
    if nz:
        raise SystemExit(f"REFUSING: {len(nz)} restored W_v are not exactly zero: {nz[:2]}")
    wv = [k for k in grow if k.endswith("W_v.weight")]
    if len(wv) != 6:
        raise SystemExit(f"REFUSING: expected 6 W_v tensors, found {len(wv)}")
    print(f"all {len(wv)} restored W_v are exactly zero -> identity at init")

    torch.save(out, OUT)
    print(f"wrote {OUT}  ({len(out)} tensors)")
    print(f"\nARM 1 strip: {STRIP_ARM1}")
    print(f"GRAFT strip: {STRIP_GRAFT}")


if __name__ == "__main__":
    main()
