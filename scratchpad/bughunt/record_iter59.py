"""Append iter 59 (rgate) to research_log.jsonl. Number assigned at VERDICT time."""
import io
import json

NOTE = (
    "REJECTED as an EXACT TIE vs its own basis, and it is the FOURTH consecutive iteration with the "
    "'used the freedom, gained nothing' signature. vs ITER 45 (the controlled comparison; this run "
    "is iter 45's recipe with one variable changed): ahead 0.297725 = -0.000028, imm 0.265446 = "
    "-0.000071. BOTH inside the +/-7.5e-5 noise floor, i.e. a literal tie. vs the CURRENT CHAMPION "
    "iter 53 (the gate): -0.000202 / -0.000255. size 0/2500, nan_users 0, +324 params (8 tensors "
    "across the 2 card layers). BOTH-modes rule: the lever changes `a` in the WKV recurrence, i.e. "
    "the shared trunk, so the curve-side exception does not apply. "
    "*** THE LEVER: FSRS-style retrievability gating of the delta rule's in-context learning rate. "
    "`rhat = (1 + dt/s)^(-d)` in log space from the row's own elapsed time, and `a_logit += gain * "
    "(1 - rhat)` -- FSRS sign, so LOWER expected recall means a LARGER state update. The idea is "
    "that a card you have probably forgotten should overwrite its memory slot harder than one you "
    "still know. "
    "*** THE ENGAGEMENT DIAGNOSTIC, and it is the fourth confirmation: `rgate_gain` is ZERO-INIT and "
    "trained to **-0.3516** on card layer 0 and **+0.0519** on card layer 1. So the model learned a "
    "non-trivial gate and gained nothing measurable. Compare iter 48 (rcouple_w learned and "
    "sign-correct, exact tie), iter 50 (deck level embedding trained to L2=1.766, coin flip), iter "
    "57 (all 4 live cmix exponents moved 2.0 -> 1.26-1.86, exact tie). Four levers, four different "
    "mechanisms, one signature: THIS MODEL USES ANY NEW DEGREE OF FREEDOM IT IS GIVEN, AND USE IS "
    "NOT EVIDENCE OF NEED. "
    "*** THE SIGN IS THE INTERESTING PART, AND IT CONTRADICTS THE FSRS INTUITION THE LEVER WAS BUILT "
    "ON. Layer 0's gain went NEGATIVE (-0.3516), and the lever adds `gain * (1 - rhat)` where "
    "`(1 - rhat)` rises as expected recall falls. A negative gain therefore means the model chose to "
    "update the state LESS when it predicts you have forgotten -- the opposite of the FSRS-motivated "
    "direction the feature was designed to supply, and the opposite of what the code comment asserts "
    "('FSRS sign: lower expected recall => larger state update'). Layer 1 kept the intended sign but "
    "at 1/7 the magnitude. Read charitably: given the freedom, the trunk preferred to DAMP the "
    "delta-rule rate on long gaps rather than boost it. This is a genuine, if small, piece of "
    "evidence against importing FSRS's retrievability intuition into the recurrence -- and it is "
    "consistent with the 2026-08-19 finding that the delta rule is already massively load-bearing "
    "(zeroing `a` costs +0.208 imm), i.e. its rate is not something the model wants perturbed. "
    "*** FAMILY: this is the first entry in an FSRS-structure-injection family (0/1, deprioritized "
    "not closed). It is NOT the same family as iters 46/48, which routed information BETWEEN heads; "
    "this one injects an external functional FORM into the recurrence. A second variant would have "
    "to inject a different structure, and should reckon with the sign result above before assuming "
    "the FSRS direction is the helpful one. "
    "*** OPS: clean 12.3 h run (WS 17:02-21:23, decay to 02:13, eval to 05:18), DONE_EXIT_0. Its "
    "22:25 smoke failure on 2026-08-18 was STALE -- the hermetic fix landed at 22:31 and the smoke "
    "was re-verified on 2026-08-19 WITH the contaminating RWKV_RGATE deliberately set in the ambient "
    "env (OFF 0 keys / ON 8 keys, delta +324, inertness exactly 0.000e+00). This run also carries "
    "the two step-verification gates added 2026-08-19 after preflight found it verified NEITHER "
    "training phase's output. DEPLOY DEBT: none -- rejected. "
    "⚠ READING THE DIAGNOSTIC REQUIRED THE FINAL CHECKPOINT: a lexicographic glob sort returns "
    "`i55_d_50.pth` AFTER `i55_d_10935.pth`, and the step-50 values differ (gain -0.324 vs -0.352). "
    "Sort by the parsed step. Same wrong-checkpoint class as iter 47's first run."
)

row = {
    "exp": "iter55_rgate",
    "number": 59,
    "change": (
        "RWKV_RGATE=card: FSRS-style retrievability gating of the delta rule's in-context learning "
        "rate. rhat = (1 + dt/s)^(-d) computed in log space from the row's elapsed time, then "
        "a_logit += gain * (1 - rhat) with gain zero-init, on the 2 card layers. +324 params."
    ),
    "params": 558536,
    "ahead": 0.297725,
    "imm": 0.265446,
    "d_ahead": -0.000202,
    "d_imm": -0.000255,
    "p_ahead": 1.0,
    "p_imm": 1.0,
    "nan_users": 0,
    "eval_users": "5001-7500",
    "throughput": "n/a",
    "status": "rejected",
    "note": NOTE,
}

P = "optimization/research_log.jsonl"
existing = [json.loads(l) for l in io.open(P, encoding="utf-8") if l.strip()]
assert all(r.get("number") != 59 for r in existing), "iter 59 already recorded"
assert max(r.get("number", 0) for r in existing) == 58, "expected 58 to be the max before this"
with io.open(P, "a", encoding="utf-8", newline="\n") as fh:
    fh.write(json.dumps(row, ensure_ascii=True) + "\n")
print("appended iter 59 (%s), status=%s" % (row["exp"], row["status"]))
