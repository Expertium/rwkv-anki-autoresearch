"""Append iter 56 (decayshape) to research_log.jsonl. Number assigned at VERDICT time."""
import io
import json

NOTE = (
    "REJECTED against the champion, but it is NOT a null -- the lever has a real, sub-bar effect "
    "and the interesting part is what that costs to pursue. vs the CURRENT CHAMPION iter 53 (the "
    "gate): ahead 0.297640 = -0.000117 (p=0.985 for improvement), imm 0.265271 = -0.000080 "
    "(p=1.000). Both worse, so it fails on both modes. vs ITER 45 (the CONTROLLED comparison, "
    "since this run is iter 45's recipe with one variable changed -- confirmed in its own decay "
    "log, which shows the pre-iter-53 Muon split of 500,800/57,412 rather than iter 53's "
    "528,320/29,892, i.e. RWKV_MUON_INCLUDE_LORA is NOT set here): ahead +0.000057 at p=6.0e-12, "
    "imm +0.000104 at p=3.1e-161. size 0/2500, nan_users 0, params 558,212 EXACTLY unchanged (a "
    "schedule change has no weights). "
    "THE ASYMMETRY: on ahead +0.000057 sits INSIDE the +/-7.5e-5 noise floor, so its reality rests "
    "on rank consistency (p=6e-12) rather than magnitude -- and iter 44 is the standing warning "
    "that rank and magnitude can disagree. On imm +0.000104 clears both the floor and the bar. So "
    "the decay SHAPE does something real to the rating head and almost nothing to the curve head. "
    "=> THE DECISION-RELEVANT RESULT, computed BEFORE queueing anything: the obvious follow-up is "
    "'does linear decay STACK on iter 53?', since the two levers are orthogonal and were never "
    "tested together. Priced under PERFECT ADDITIVITY, a stacked run would sit at +0.000057 ahead "
    "/ +0.000104 imm vs iter 53 -- and the ahead half FAILS the 0.0001 bar. So even the best case "
    "does not clear the gate, and there is no mechanism predicting super-additivity. DO NOT SPEND "
    "6.1 h on it. This is the schedule-shape family bounded by arithmetic rather than by a second "
    "run. "
    "PRE-REGISTERED PREDICTION WAS HALF WRONG, and the prereg is why that is visible "
    "(scratchpad/iter57_decayshape/GATE.md, written at 04:25 with the eval at 2202/2500 and no "
    "result inspected). It predicted a NULL on the reasoning that iters 41/43/44 showed "
    "same-capacity rearrangements are mutually indistinguishable, and that reshaping the decay "
    "reallocates the same optimization budget rather than adding to it. Ahead behaved exactly that "
    "way; imm did not, at p=3e-161. The budget-reallocation intuition is therefore NOT a general "
    "law about this trunk -- it holds for the curve head and fails for the rating head. "
    "The same GATE.md pre-registered the RULE as well as the prediction: BOTH-modes, not the "
    "curve-side exception, because an LR-schedule change acts on the whole trunk through the "
    "optimizer. Fixing that before the numbers landed is what stops a post-hoc switch to the "
    "curve-side rule, which imm's +0.000104 would have made tempting. "
    "OPS: clean run, DONE_EXIT_0 at 04:40:48, projected 04:41 from a mid-eval rate fit. Its decay "
    "checkpoint is scratchpad/iter45_kddecay/i57_d_10935.pth, NOT in its own directory -- "
    "write_decay_setup.py writes beside the WS-final it warm-starts from. Two ECHO lines in the "
    "runner carry iter 52's prose and the live log now reads 'DECAY_OK KD ON alpha 0.9' for a run "
    "whose alpha is 0.5 (confirmed 0.5 in the decay log itself, and the guard checked 0.5). "
    "Nothing computed depends on it; the run is correct and the LOG lies. That defect was "
    "predicted hours earlier by the new preflight check and documented in GATE.md before it was "
    "written. DEPLOY DEBT: none -- a training-schedule change, forward pass untouched."
)

row = {
    "exp": "iter57_decayshape",
    "number": 56,
    "change": (
        "RWKV_DECAY_SHAPE=linear: replace the decay phase's default LR shape 1-sin(pi x/2) "
        "(mass 0.3634) with a linear ramp to zero. Single variable vs iter 45 -- same WS-final, "
        "same KD dump, same alpha schedule (0.5 in decay), same step count. Decay-only, so the "
        "lever cannot touch WS by construction."
    ),
    "params": 558212,
    "ahead": 0.297640,
    "imm": 0.265271,
    "d_ahead": -0.000117,
    "d_imm": -0.000080,
    "p_ahead": 0.985,
    "p_imm": 1.0,
    "nan_users": 0,
    "eval_users": "5001-7500",
    "throughput": "n/a",
    "status": "rejected",
    "note": NOTE,
}

P = "optimization/research_log.jsonl"
existing = [json.loads(l) for l in io.open(P, encoding="utf-8") if l.strip()]
assert all(r.get("number") != 56 for r in existing), "iter 56 already recorded"
assert max(r.get("number", 0) for r in existing) == 55, "expected 55 to be the max before this"
with io.open(P, "a", encoding="utf-8", newline="\n") as fh:
    fh.write(json.dumps(row, ensure_ascii=True) + "\n")
print("appended iter 56 (%s), status=%s" % (row["exp"], row["status"]))
