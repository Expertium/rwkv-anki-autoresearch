"""How much is the delta rule worth IN THE CARD STREAM SPECIFICALLY?

WHY THIS IS THE DECISIVE SCREEN FOR THE FSRS-CORE IDEA. Andrew's proposal is to replace the
card-level recurrence with FSRS-7's structured (S_long, S_short, D) update, driven by the
trunk. FSRS has no delta rule -- no key-selective erase-then-write. The record says the delta
rule is massively load-bearing (+0.208 imm when `a` is zeroed), which reads as a fatal
objection.

BUT THAT NUMBER WAS MEASURED GLOBALLY. `delta_ablate_screen.py` zeroes `a` in EVERY stream at
once, so it cannot say whether the damage lives in the card stream (which this proposal would
replace) or in the note/deck/preset/user streams (which it would KEEP). Attributing a global
ablation to one component is exactly the mistake the record keeps recording. So:

    card-only cost small  -> the card recurrence is not where the delta rule earns its keep.
                             Replacing it with an FSRS core is a reasonable bet.
    card-only cost ~ global -> the card stream IS the delta rule's home. An FSRS core removes
                             precisely the mechanism that matters, and the idea is expensive.

Three arms, paired within user: baseline, card-only a=0, all-streams a=0. The third arm is a
CALIBRATION, not a repeat -- it re-derives the known global number on the same users and the
same harness, so the card-only figure is read against a control measured here rather than
against a number from a different run.

SAME DIRECTIONAL CAVEAT AS THE ORIGINAL. Zeroing `a` on a model TRAINED with the delta rule is
an UPPER BOUND on the damage, not an estimate of it: the weights co-adapted. That asymmetry is
what makes it a good screen -- catastrophic here kills the idea cheaply; mild here justifies
the GPU run without proving it will work.

CPU-only, one thread. Usage: .venv/Scripts/python.exe scratchpad/hybrid100k/card_delta_ablate.py [n_users]
"""
import contextlib
import io
import os
import sys
from pathlib import Path

ENV = {
    "RWKV_ARCH_MODULE": "scratchpad/track2_a18/architecture_d80_lora4_cnd.py",
    "RWKV_INTERLEAVE": "1", "RWKV_GRU_HEAD": "3", "RWKV_PAVA_LAMBDA": "0.2",
    "RWKV_NO_AHEAD_RESIDUAL": "1", "RWKV_STRIP_L0_VLORA": "1", "RWKV_ZERO_FEATURES": "22",
    "RWKV_STRIP_CMIX": "user_id:0,user_id:1,user_id:2,preset_id:0,preset_id:1,"
                       "preset_id:2,deck_id:1,deck_id:2,card_id:1",
    "RWKV_STATE_CLAMP_TAU": "300", "RWKV_STATE_CLAMP_WINDOW": "32768",
    "RWKV_MUON": "1", "RWKV_MUON_INCLUDE_LORA": "1", "RWKV_NO_JIT": "1",
    "OMP_NUM_THREADS": "1",
}
for k, v in ENV.items():
    os.environ.setdefault(k, v)

import torch  # noqa: E402

torch.set_num_threads(1)
sys.path.insert(0, os.getcwd())
import rwkv.run_as_rnn as ras                        # noqa: E402
from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG as CFG    # noqa: E402
from rwkv.model.srs_model_rnn import SrsRWKVRnn      # noqa: E402

CKPT = "scratchpad/iter53_muonlora/i53_d_10935.pth"
USERS = [5044, 5100, 5063, 5097, 5048, 5030]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
NAMES = [n for n, _ in CFG.modules]
CARD = NAMES.index("card_id")

# "off" | "card" | "all"
MODE = {"v": "off"}
_orig_review = SrsRWKVRnn.review


def _review(self, *a, **kw):
    """Register the ablation hooks once, per stream, so the scope can be chosen per arm.
    The hook forces the PRE-sigmoid logit of `a` to -30 (a -> ~1e-13), which removes the
    delta term and leaves every other path in the mixer untouched."""
    if not getattr(self, "_abl_hooked", False):
        self._abl_hooked = True
        n_hooked = 0
        for i, mod in enumerate(self.rwkv_modules):
            is_card = (i == CARD)
            for blk in mod.blocks:
                tm = getattr(blk, "time_mixer", None)
                if tm is None or not hasattr(tm, "a_lora_simple"):
                    continue
                n_hooked += 1

                def hook(m, inp, out, _is_card=is_card):
                    if MODE["v"] == "all" or (MODE["v"] == "card" and _is_card):
                        return torch.full_like(out, -30.0)
                    return None

                tm.a_lora_simple.register_forward_hook(hook)
        assert n_hooked > 0, "no a_lora_simple found -- the hook point moved"
        print("   [hooked %d time mixers; card stream index %d]" % (n_hooked, CARD),
              file=sys.stderr)
    return _orig_review(self, *a, **kw)


SrsRWKVRnn.review = _review

grabbed = []
_orig_stats = ras.get_stats


def _stats(*a, **kw):
    out = _orig_stats(*a, **kw)
    grabbed.append(out[0])
    return out


ras.get_stats = _stats


def score(user, mode):
    MODE["v"] = mode
    grabbed.clear()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ras.run(data_path=Path("../anki-revlogs-10k"), model_path=CKPT,
                label_db_path="label_filter_db", label_db_size=40_000_000_000,
                user_id=user, verbose=False)
    vals = []
    for g in grabbed[:2]:
        m = g.get("metrics", g) if isinstance(g, dict) else {}
        vals.append(float(m.get("LogLoss", float("nan"))))
    return vals if len(vals) == 2 else [float("nan")] * 2


import functools
print = functools.partial(print, flush=True)
print("DELTA RULE, SCOPED -- champion %s" % CKPT)
print("arms: baseline / card-stream-only a=0 / all-streams a=0 (calibration)\n")
print("%-7s %-19s %-19s %-19s" % ("user", "baseline imm/ahead", "card-only", "all-streams"))
print("-" * 70)
rows = []
for u in USERS[:N]:
    try:
        b = score(u, "off")
        c = score(u, "card")
        a = score(u, "all")
    except Exception as e:  # noqa: BLE001
        print("%-7d skipped (%s: %s)" % (u, type(e).__name__, str(e)[:50]))
        continue
    if any(v != v for v in b + c + a):
        print("%-7d unusable (nan)" % u)
        continue
    rows.append((b, c, a))
    print("%-7d %8.5f %9.5f %8.5f %9.5f %8.5f %9.5f"
          % (u, b[0], b[1], c[0], c[1], a[0], a[1]))

if rows:
    def mean(f):
        return sum(f(r) for r in rows) / len(rows)

    ci = mean(lambda r: r[1][0] - r[0][0])
    ca = mean(lambda r: r[1][1] - r[0][1])
    ai = mean(lambda r: r[2][0] - r[0][0])
    aa = mean(lambda r: r[2][1] - r[0][1])
    print("\nmean cost vs baseline (INFERENCE-TIME UPPER BOUND, n=%d users)" % len(rows))
    print("   card-stream only : imm %+.5f   ahead %+.5f" % (ci, ca))
    print("   all streams      : imm %+.5f   ahead %+.5f" % (ai, aa))
    share = ci / ai if ai else float("nan")
    print("   card share of the global imm cost: %.1f%%" % (100 * share))
    print()
    if abs(ci) < 0.01:
        print("=> The card stream is NOT where the delta rule earns its keep. Replacing the")
        print("   card recurrence with an FSRS-7 core removes little of what matters.")
    elif share < 0.5:
        print("=> Most of the delta rule's value lives OUTSIDE the card stream, which an")
        print("   FSRS-core design keeps. Moderate risk, and the GPU run is defensible.")
    else:
        print("=> The card stream IS the delta rule's home. An FSRS core removes exactly the")
        print("   mechanism that matters -- expect to pay for it, and say so up front.")
