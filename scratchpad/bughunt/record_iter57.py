"""Append iter 57 (cmixpow) to research_log.jsonl. Number assigned at VERDICT time."""
import io
import json

NOTE = (
    "REJECTED as an EXACT TIE vs its own basis, and the DIAGNOSTIC is the result. vs ITER 45 (the "
    "controlled comparison -- this run is iter 45's recipe with one variable changed): ahead "
    "0.297728 = -0.000031, imm 0.265407 = -0.000032. BOTH are ~2.4x INSIDE the +/-7.5e-5 noise "
    "floor, i.e. a literal tie, with p=0.987/0.999 for improvement. vs the CURRENT CHAMPION iter 53 "
    "(the gate): -0.000205 / -0.000215. size 0/2500 against both references, nan_users 0, params "
    "558,225 (+13 over iter 45 = one exponent per channel mixer). BOTH-modes rule (a channel-mixer "
    "change is a trunk change; the curve-side exception does not apply). "
    "*** THE MODEL USED THE FREEDOM AND GAINED NOTHING, AND THIS IS NOW THE THIRD CONSECUTIVE "
    "ITERATION WITH THAT EXACT SIGNATURE. Init is 2.0 (squared ReLU), confirmed by the stripped "
    "sites reading exactly 2.00000. Every LIVE exponent moved substantially: card:0 -> 1.261, "
    "note:0 -> 1.526, deck:0 -> 1.420, deck:3 -> 1.858 -- a 7% to 37% move, and ALL FOUR in the "
    "SAME DIRECTION, downward toward plain ReLU and away from x^2. So the trunk does prefer a "
    "gentler channel-mixer nonlinearity, it moved decisively to get one, and the held-out loss did "
    "not move. Compare iter 48 (rcouple_w learned, sign-correct, negligible) and iter 50 (the deck "
    "level embedding trained to L2=1.766 and bought a coin flip). => THE GENERALIZABLE STATEMENT: "
    "this model will use any new degree of freedom it is given, and USE IS NOT EVIDENCE OF NEED. A "
    "'the parameter trained, so the lever engaged' check proves the lever is not inert; it says "
    "nothing about whether the loss had anything to gain. Three levers, three different mechanisms "
    "(architectural coupling, a new scope, a functional form), same signature. "
    "*** SCOPE CAVEAT, RECORDED BEFORE THE VERDICT (scratchpad/iter54_cmixpow/GATE.md, written "
    "06:55 while it was still training): THE LEVER REACHED 4 OF 13 CHANNEL MIXERS. 13 cmix_pow "
    "params are created and only 4 receive gradients; the 9 dead ones are EXACTLY RWKV_STRIP_CMIX, "
    "verified as a set equality. A stripped channel mixer still constructs its parameters and never "
    "uses them. So the honest claim is 'null on card:0, note:0, deck:0, deck:3', NOT 'learnable "
    "exponents do not help' -- the other 9 sites do not exist in this configuration and were never "
    "tested. BUT the diagnostic makes this a STRONGER null than the caveat suggests: at the four "
    "sites it did reach, the lever was FULLY ENGAGED (up to a 37% move), so this is not a "
    "too-weak-to-matter result. It is decisive where it was tested and silent elsewhere. "
    "*** FAMILY STATUS: expressiveness-vs-capacity is 0/1, i.e. DEPRIORITIZED, NOT CLOSED (conduct "
    "rule 5 needs 3-5 in-family variants). This is the family's first run, opened by Andrew "
    "2026-08-17 precisely because 'capacity-at-5k is 0/3' had been standing in for an argument it "
    "could not support -- so closing it on one run that reached 4 of 13 sites would repeat that "
    "error exactly. A second variant should target a RICHER form at a site that survives "
    "RWKV_STRIP_CMIX, and must clear the redundancy test (an adjacent free linear must not be able "
    "to absorb it -- which a learnable EXPONENT does, since curvature is not absorbable). "
    "*** NOTE THE OVERLAP WITH ITER 49, which restored the user/preset layer-0 channel mixers and "
    "was rejected at +0.000067 ahead (p=0.11). Those are among the sites that carry no exponent "
    "here. The two results concern the same missing mixers from opposite directions: iter 49 added "
    "the mixers back and got nothing; iter 57 made the surviving mixers richer and got nothing. "
    "*** OPS: this is the run whose WS was interrupted by the 2026-08-18 power outage and recovered "
    "by mid-epoch resume from step 8000 (~110 lost steps). The resumed tail's dropout draws differ, "
    "so the number is FAIR but the run is NOT bit-reproducible. Its phase 2a decayed 3.3 h at KD "
    "alpha 0.9 -- iter 55's lever -- because the decay-only generator sliced away the WS region "
    "holding the reset line; its own guard caught it (DONE_EXIT_WRONGALPHA_DECAY) and phase 2b "
    "confirms the repair in its own log (alpha FIXED at 0.5, 74 per-step confirmations). "
    "DONE_EXIT_0 at 10:50:18. DEPLOY DEBT: none -- rejected, so no port is owed."
)

row = {
    "exp": "iter54_cmixpow",
    "number": 57,
    "change": (
        "RWKV_CMIX_POW=1: a learnable per-channel-mixer exponent on the squared-ReLU activation "
        "(init 2.0), +13 params. The first run of the EXPRESSIVENESS-vs-CAPACITY family -- a "
        "richer functional form at fixed capacity, rather than more of the same form. Chosen "
        "because a learnable exponent survives the redundancy test: an adjacent free linear can "
        "absorb a learnable slope but not curvature."
    ),
    "params": 558225,
    "ahead": 0.297728,
    "imm": 0.265407,
    "d_ahead": -0.000205,
    "d_imm": -0.000215,
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
assert all(r.get("number") != 57 for r in existing), "iter 57 already recorded"
assert max(r.get("number", 0) for r in existing) == 56, "expected 56 to be the max before this"
with io.open(P, "a", encoding="utf-8", newline="\n") as fh:
    fh.write(json.dumps(row, ensure_ascii=True) + "\n")
print("appended iter 57 (%s), status=%s" % (row["exp"], row["status"]))
