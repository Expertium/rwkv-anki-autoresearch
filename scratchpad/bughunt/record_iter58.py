"""Append iter 58 (kdalpha025) to research_log.jsonl. Number assigned at VERDICT time."""
import io
import json

NOTE = (
    "REJECTED, and it CLOSES the KD alpha_decay lever by bracketing 0.5 on BOTH sides -- which is "
    "exactly the secondary purpose its pre-registration named. vs ITER 45 (the controlled "
    "comparison; this run is iter 45's recipe with one variable changed): ahead 0.297775 = "
    "-0.000078, imm 0.265407 = -0.000032. vs the CURRENT CHAMPION iter 53 (the gate): -0.000252 / "
    "-0.000215. size 0/2500 against both references, nan_users 0, params 558,212 unchanged. "
    "*** THE BRACKET, all measured against iter 45: alpha_decay 0.25 gives -0.000078 / -0.000032; "
    "0.50 IS iter 45; 0.90 (iter 55) gives -0.000043 / -0.000116. Both directions from 0.5 lose. "
    "So 0.5 is a genuine INTERIOR optimum and the curve is FLAT around it -- every deviation is "
    "within ~1e-4. LEVER CLOSED: do not test another alpha_decay value. iter 45 already showed "
    "some teacher in decay beats none, so the lever is now bounded on all three sides (none, less, "
    "more). "
    "*** MY PRE-REGISTERED PREDICTION WAS WRONG, AND THAT IS THE FINDING. GATE.md and "
    "mk_kdalpha025.py's docstring both predicted this would IMPROVE, on the calibration mechanism: "
    "KD overwrites the target with alpha*teacher + (1-alpha)*hard, so the head inherits the "
    "TEACHER's calibration; the champion is overconfident by -0.00292 over 83,478 predictions and a "
    "one-parameter logit shift recovers +0.000115 held out; variance reduction is an EARLY good "
    "while calibration matters LATE, when the LR anneals and the weights settle into what ships. "
    "That reasoning correctly predicted iter 55's DIRECTION (more teacher in decay = worse). It "
    "failed here. "
    "*** THE GENERALIZABLE LESSON: A ONE-SIDED MECHANISM CANNOT LOCATE A TWO-SIDED OPTIMUM. The "
    "calibration argument explains why MORE teacher hurts. It does not thereby predict that LESS "
    "teacher helps -- that requires the countervailing term, which the argument NAMED (target-"
    "variance reduction) but never quantified. With both terms live, an interior optimum is the "
    "default expectation and 0.5 sitting at the peak is unremarkable. I should have predicted "
    "'0.5 is near-optimal, both directions lose' from the structure of the argument I had already "
    "written down. Before predicting a direction from a mechanism, check whether the mechanism has "
    "a countervailing term; if it does, it predicts an OPTIMUM, not a direction. "
    "*** THE CALIBRATION FINDING ITSELF IS NOT REFUTED -- the model IS overconfident and a logit "
    "shift DOES recover +0.000115 held out. What is refuted is that alpha_decay is the lever that "
    "reaches it. A direct recalibration (a learned output temperature/shift on the curve head) "
    "remains untested and is now the natural way to collect that +0.000115, since the KD-schedule "
    "route is closed. "
    "*** OPS: clean run, DONE_EXIT_0 at 17:00:27. Its START banner and DECAY_OK line both correctly "
    "read 'alpha 0.25' -- the runner it was cloned from would have written 0.9 in both places, and "
    "that prose defect was found and fixed in the generator on 2026-08-19 before this ran. Its "
    "decay checkpoint is scratchpad/iter45_kddecay/kda025_d_10935.pth, not in its own directory. "
    "First run named for its lever with NO number in its path, under the completion-order "
    "convention. DEPLOY DEBT: none -- a training-schedule change, forward pass untouched."
)

row = {
    "exp": "kdalpha025",
    "number": 58,
    "change": (
        "KD alpha_decay 0.5 -> 0.25: LOWER the teacher weight during the DECAY phase, the "
        "untested side of the bracket after iter 55 showed 0.9 loses. Single variable vs iter 45; "
        "decay-only, so the lever cannot touch WS by construction."
    ),
    "params": 558212,
    "ahead": 0.297775,
    "imm": 0.265407,
    "d_ahead": -0.000252,
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
assert all(r.get("number") != 58 for r in existing), "iter 58 already recorded"
assert max(r.get("number", 0) for r in existing) == 57, "expected 57 to be the max before this"
with io.open(P, "a", encoding="utf-8", newline="\n") as fh:
    fh.write(json.dumps(row, ensure_ascii=True) + "\n")
print("appended iter 58 (%s), status=%s" % (row["exp"], row["status"]))
