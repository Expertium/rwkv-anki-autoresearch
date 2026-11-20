"""One-off: append the TA3 arm's abandoned/negative record to optimization/research_log.jsonl.

Run once. Refuses to append if an entry with number 128 or exp ta3_4exp_92 is already present.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / "optimization" / "research_log.jsonl"

REC = {
    "exp": "ta3_4exp_92",
    "number": 128,
    "params": 192800,
    "ahead": 0.306259,
    "imm": 0.273344,
    "d_ahead": -0.000412,
    "d_imm": -0.000613,
    "p_ahead": 0.9994,
    "p_imm": 1.0,
    "change": (
        "TA3 (Training-Time Testing, Dang 2026) added to the plain 92-input chain: per-step A/Q "
        "loss-mask split by per-group row rank (the fetched batch stream stays byte-identical), a "
        "normalised downhill trial step on phase A (eta 0.3, unit-norm direction over the 28 "
        "channel-mixer weight matrices), the gradient of phase Q at the perturbed weights, an exact "
        "weight restore, and a per-loss-group weighted merge. Nothing else differs from the plain "
        "chain (same start pair QAT46 step-248, same 24-column data, LR shape, WD 0.01, one fetch "
        "worker, 11 GiB cap, 2-user internal validation)."
    ),
    "note": (
        "ABANDONED as a NEGATIVE result on 2026-11-19 at Andrew's decision, mid epoch 4 (step "
        "138,044 of 173,416). Epochs 1-3 completed cleanly (43,354 steps each, 0 tracebacks, 0 "
        "skipped batches, 0 failed validations) and each ran the fixed 750-user screen, which is "
        "DIAGNOSTIC only: the pre-registered primary endpoint was the 4-epoch checkpoint and it was "
        "not reached. Epoch-matched paired screens vs the plain chain (delta = plain minus TA3, "
        "positive = TA3 better): epoch 1 0.309108/0.275626 vs plain 0.308803/0.275131 = -0.000305 "
        "(p 0.9995) / -0.000494 (p 1.0); epoch 2 0.309924/0.273467 vs 0.307044/0.274275 = -0.002880 "
        "(p 1.0) / +0.000808 (p 5.0e-29); epoch 3 0.306259/0.273344 vs 0.305847/0.272732 = -0.000412 "
        "(p 0.9994) / -0.000613 (p 1.0). All three epochs fail the dual-mode gate, the ahead mode is "
        "worse in every epoch, and TA3's ahead curve (0.30911 -> 0.30992 -> 0.30626) is non-monotone "
        "and stuck at the plain epoch-1/epoch-2 level, i.e. TA3 trained SLOWER rather than better. "
        "Throughput was ~31-32 steps/min with TA3 vs ~40 steps/min plain (fetch-bound), so ~23.5 h "
        "per epoch and ~4 days for the chain. No TA3 step failed, so the code path was never the "
        "problem in this arm. Caveats: one chain is screen-level evidence only, and the "
        "A/Q split here is within-batch by row rank because 98.2% of batches hold exactly one user "
        "group, which is a much weaker test-half than the paper's multi-epoch pretraining setting, "
        "so this negative does not refute TA3 at scale. Artifacts preserved: "
        "scratchpad/ta3_fullrun_20261116/ (epochs 1-3 complete, epoch 4 to step 138,000), "
        "scratchpad/ta3_fullrun_20261115/ (the aborted first attempt, kept), "
        "scratchpad/ta3/chain_20261116.log, scratchpad/ta3/NOTES.md."
    ),
    "nan_users": 0,
    "eval_users": (
        "fixed 750-user screen from VAL 5001-7500 (every other eligible one-batch user, 4 "
        "multi-batch users excluded); 750/750 scored with 0 OOM and 0 NaN in both modes for each epoch"
    ),
    "throughput": "~31-32 steps/min with TA3 vs ~40 steps/min plain",
    "status": "abandoned (negative)",
}


def main():
    text = LOG.read_text(encoding="utf-8")
    if '"number": 128' in text or '"exp": "ta3_4exp_92"' in text or '"exp":"ta3_4exp_92"' in text:
        raise SystemExit("record already present; not appending")
    lines = text.splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    lines.append(json.dumps(REC, ensure_ascii=False))
    LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"appended number 128 ({REC['exp']}) to {LOG}")
    print(f"file now has {len(lines)} non-blank lines")


if __name__ == "__main__":
    main()
