"""Chain 16: E1 applied to the two NEAR-MISS shrink layouts, GATED on chain 15 actually working.

WHY. Row 17 (`l2i8s4`, 1,280/2,304 = 3,584 bits) is the axis-1 best. Two smaller layouts already
exist and both FAIL the log-loss gate on the immediate head, by very different margins:

    row 21  `l2i8s4n4`  card rank-2 int8 + int4 shifts, note rank-1 int4 + int4
                        1,280/1,536 = 2,816 bits (768 below row 17)   +0.000328 / +0.000638 imm
    row 22  `l3i8s4n4`  card rank-3 int8 + int4 shifts, note rank-1 int4 + int4
                        1,792/1,536 = 3,328 bits (256 below row 17)   +0.000297 / +0.000525 imm

Both failures come from the note's rank-1 int4 tax, and the note's rank-1 int8 tax ALONE is
+0.000258 imm. So the question this chain asks is narrow and cheap: does the tail-energy penalty
(chain 15, E1 - zero bits, zero deploy cost) remove enough of that tax to flip a near-miss into a
pass, and if so which smaller layout does it flip? Row 22 needs only 0.000025; row 21 needs 0.00014.

THE GATE, because a blind run would waste 2-4 h. Before any point it reads the three chain-15 sweeps
and proceeds only if the penalty BEATS ITS OWN fused-off control on the imm head by at least 0.00002
(so the fused-off path cannot be mistaken for the penalty) and the winning run passes the log-loss
gate on both heads. `l2i8s4pf` is that control: row 17's recipe on the Python QAT loop with no
penalty. If the gate fails, every point is skipped, the reason and the numbers are logged, and the
chain ends normally so the queue keeps moving.

POINT ORDER is by value: the layout that saves the most bits first (row 21), each layout with the
lambda chain 15 found better, then the same layouts with the other lambda.

It runs AFTER `it36e1` (chain 15) and shares its BUSY_PAT, including the corruption driver.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import it36ladder_lib as lib  # noqa: E402

BUSY_PAT = ("qat_decay.py|quant_sweep.py|rwkv.train_rwkv|eval_sharded.py|rust_speed.py|rust_grid.py|"
            "rwkv-infer|probe_get_result|check_corruption")

CONTROL = "it36l2i8s4pf"
LAMBDAS = ("it36l2i8s4e1a", "it36l2i8s4e1b")      # lambda 1.0 and 10.0
MIN_GAIN = 0.00002                                # the penalty must win by this much on imm

# (candidate, stamp, steps, ref); the ORDER is rewritten by the gate, see main().
LAYOUTS = {
    "row21": ("l2i8s4n4e1a", "l2i8s4n4e1b"),      # 2,816 bits, 768 below row 17
    "row22": ("l3i8s4n4e1a", "l3i8s4n4e1b"),      # 3,328 bits, 256 below row 17
}
ORDER = ("row21", "row22")

lib.WAIT_MAX_MIN = 2400


def read_sweep(stamp):
    """The imm/ahead deltas of one landed sweep, or None if that sweep has not landed."""
    p = lib.RUN / f"quant_sweep_{stamp}.json"
    if not p.exists():
        return None
    doc = json.loads(p.read_text(encoding="utf-8"))
    rows = doc.get("rows") or []
    if not rows:
        return None
    gate = rows[0].get("gate") or {}
    out = {"bits": rows[0].get("bits")}
    for head in ("imm", "ahead"):
        h = gate.get(head) or {}
        out[head] = h.get("delta")
        out[head + "_ub"] = h.get("ub95")
    return out


def gate(log):
    """Return the ordered (lambda, layout) plan, or an empty list if E1 did not pay off."""
    ctl = read_sweep(CONTROL)
    runs = {s: read_sweep(s) for s in LAMBDAS}
    for name, rec in [("control " + CONTROL, ctl)] + [(s, runs[s]) for s in LAMBDAS]:
        log(f"gate input {name}: {json.dumps(rec)}")
    if ctl is None:
        log("gate: the fused-off control sweep is missing -- cannot separate the penalty from the "
            "fused-off path, so NOTHING runs")
        return []
    scored = [(s, r) for s, r in runs.items() if r and r.get("imm") is not None]
    if not scored:
        log("gate: no chain-15 penalty sweep landed -- NOTHING runs")
        return []
    scored.sort(key=lambda kv: kv[1]["imm"])           # lower imm delta is better
    best, rec = scored[0]
    gain = ctl["imm"] - rec["imm"]
    ok_loss = (rec["imm_ub"] is not None and rec["ahead_ub"] is not None
               and rec["imm_ub"] <= 0.0005 and rec["ahead_ub"] <= 0.0005)
    log(f"gate: best penalty run {best} imm {rec['imm']:+.6f} vs control {ctl['imm']:+.6f} -> "
        f"gain {gain:+.6f} (need >= {MIN_GAIN}); log-loss gate PASS={ok_loss}")
    if gain < MIN_GAIN:
        log("gate: the penalty did NOT reduce the immediate-head tax -- the shrink points cannot "
            "reach the ceiling this way, so NOTHING runs")
        return []
    if not ok_loss:
        log("gate: the penalty run improves the tax but FAILS the log-loss gate itself -- a failing "
            "recipe cannot lift another recipe over the ceiling, so NOTHING runs")
        return []
    other = scored[1][0] if len(scored) > 1 else best
    lambdas = [best, other]
    plan = []
    for layout in ORDER:
        for lam in lambdas:
            plan.append(LAYOUTS[layout][LAMBDAS.index(lam)])
    log(f"gate PASS: point order {plan} (winning lambda from {best})")
    return plan


def main() -> int:
    log = lib.make_logger(lib.RUN / "it36e1shrink_chain.log")
    lock = lib.RUN / "it36e1shrink_chain.lock"
    log("=== it36e1shrink chain start ===")
    if lock.exists():
        log(f"lock found ({lock.read_text(encoding='utf-8').strip()}) -- another copy is going; exiting")
        return 1
    lock.write_text(lib.time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
    log(f"lock taken: {lock}")
    if not lib.wait_for_prev(log, lib.RUN / "it36e1_chain.log",
                             "=== it36e1 chain end ===", BUSY_PAT):
        log("=== it36e1shrink chain end (without running anything) ===")
        return 1

    plan = gate(log)
    for cand in plan:
        log(f"point {cand}@124 -> tag it36{cand}, ref qnj_it36ctl, epochs 2")
        rc = lib.qat_point(log, cand, f"it36{cand}", 124, "qnj_it36ctl", 2)
        if rc != 0:
            log(f"{cand}: point returned rc={rc} -- continuing with the next point")

    log("=== it36e1shrink chain end ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())