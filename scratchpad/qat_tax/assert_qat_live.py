"""Phase-0 gate: assert the RWKV_QAT_* scopes actually reached the FINAL arch config.

WHY THIS EXISTS (2026-08-12). The scopes used to be applied to the DEFAULT config's layer
objects, and RWKV_ARCH_MODULE then replaced DEFAULT_ANKI_RWKV_CONFIG wholesale -- discarding
every mutation while the banners still printed. Every track-2 run therefore ran with the QAT
env SILENTLY INERT, and the only symptom was a suspiciously zero PTQ cost. Banner-grepping
could not catch it (the banners were truthful about what they set; the object was thrown away).

This imports the arch module under the run's own env and inspects the config the model will
actually be built from. Costs ~2 s and is the regression test for that bug. Run it as the first
phase of any quant-aware .cmd, BEFORE spending GPU hours.

Exit 0 = every stream named in the scope env is really quantized in the final config.
"""
import os
import sys

_QAT_NAME = {"card": "card_id", "deck": "deck_id", "note": "note_id",
             "preset": "preset_id", "user": "user_id"}


def _named(env):
    """Stream names mentioned in a `card:...,note:...` style scope env var."""
    raw = os.environ.get(env, "").strip()
    if not raw:
        return set()
    return {_QAT_NAME[e.strip().split(":")[0]] for e in raw.split(",") if e.strip()}


def main():
    want_lr = _named("RWKV_QAT_LOWRANK_SCOPE")
    want_int = _named("RWKV_QAT_SCOPE")
    want_sh = _named("RWKV_QAT_SHIFT_SCOPE")
    if not (want_lr or want_int or want_sh):
        print("[qat-assert] no QAT scope env set -- nothing to check", flush=True)
        return 0

    from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG as cfg

    problems = []
    for name, lc in cfg.modules:
        rank = getattr(lc, "state_lowrank_rank", 0)
        qmax = getattr(lc, "state_qmax", float("inf"))
        shq = getattr(lc, "state_shift_qmax", float("inf"))
        print(f"[qat-assert] {name:10s} lowrank_rank={rank} state_qmax={qmax} shift_qmax={shq}",
              flush=True)
        if name in want_lr and rank <= 0:
            problems.append(f"{name}: RWKV_QAT_LOWRANK_SCOPE names it but final rank={rank}")
        if name in want_int and qmax == float("inf"):
            problems.append(f"{name}: RWKV_QAT_SCOPE names it but final state_qmax=inf")
        if name in want_sh and shq == float("inf"):
            problems.append(f"{name}: RWKV_QAT_SHIFT_SCOPE names it but final shift_qmax=inf")

    # Also catch the reverse: a scope naming a stream this arch does not have (typo / renamed arch).
    have = {n for n, _ in cfg.modules}
    for env, want in (("RWKV_QAT_LOWRANK_SCOPE", want_lr), ("RWKV_QAT_SCOPE", want_int),
                      ("RWKV_QAT_SHIFT_SCOPE", want_sh)):
        for miss in sorted(want - have):
            problems.append(f"{env} names '{miss}', absent from this arch ({sorted(have)})")

    if problems:
        print("[qat-assert] FAIL -- QAT env is INERT:", flush=True)
        for p in problems:
            print(f"    {p}", flush=True)
        return 44
    print("[qat-assert] PASS -- scopes reached the final config", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
