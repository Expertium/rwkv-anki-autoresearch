"""Append-only optimization logbook (protocol point 9).

- log.jsonl : machine-readable, one JSON record per iteration, INCLUDES `comment`.
- log.md    : human-readable table regenerated from log.jsonl, EXCLUDES `comment`.

Usage:
  python optimization/logbook.py add record.json   # append a record, rebuild md
  python optimization/logbook.py rebuild           # rebuild md from jsonl
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
JSONL = HERE / "log.jsonl"
MD = HERE / "log.md"
QUANT_JSONL = HERE / "quant_log.jsonl"  # state-quant configs (deploy-time PTQ; not arch iterations)
QAT_JSONL = HERE / "qat_log.jsonl"      # quant-aware-training experiments (deploy measured on 17-user gate)
BASELINE_JSONL = HERE / "baseline_log.jsonl"  # "baseline to beat" rows for the 100/100 research phase
RESEARCH_JSONL = HERE / "research_log.jsonl"  # research-phase experiments (accept iff BOTH modes improve)

COLS = [
    ("number", "#"),
    ("status", "status"),
    ("timestamp", "timestamp"),
    ("ahead", "ahead LL"),
    ("imm", "imm LL"),
    ("params", "params"),
    ("state_kib", "state KiB"),
    ("throughput", "throughput (rev/s)"),
    ("wilcoxon_p", "wilcoxon p"),
    ("review_count_check", "size✓"),
    ("logloss_tolerance_check", "LL✓"),
    ("state_size_check", "state✓"),
    ("summary", "summary"),
]


def load():
    if not JSONL.exists():
        return []
    return [json.loads(l) for l in JSONL.read_text().splitlines() if l.strip()]


def load_quant():
    if not QUANT_JSONL.exists():
        return []
    return [json.loads(l) for l in QUANT_JSONL.read_text().splitlines() if l.strip()]


QUANT_COLS = [
    ("config", "config"),
    ("card_kib", "card KiB"),
    ("note_kib", "note KiB"),
    ("imm_delta", "imm Δ"),
    ("ahead_delta", "ahead Δ"),
    ("gate", "gate"),
    ("status", "verdict"),
]


def quant_cell(rec, key):
    v = rec.get(key, "")
    if key in ("imm_delta", "ahead_delta") and isinstance(v, (int, float)):
        return f"+{v:.4f}" if v >= 0 else f"{v:.4f}"
    if key in ("card_kib", "note_kib") and isinstance(v, (int, float)):
        return f"{v:.2f}"
    return str(v)


def load_qat():
    if not QAT_JSONL.exists():
        return []
    return [json.loads(l) for l in QAT_JSONL.read_text().splitlines() if l.strip()]


def load_baseline():
    if not BASELINE_JSONL.exists():
        return []
    return [json.loads(l) for l in BASELINE_JSONL.read_text().splitlines() if l.strip()]


BASELINE_COLS = [
    ("name", "model"), ("params", "params"), ("d_model", "d"), ("trained_on", "trained on"),
    ("chunk", "chunk"), ("max_train", "MAX"), ("ahead", "ahead LL"), ("imm", "imm LL"), ("note", "note"),
]


def baseline_section_lines():
    recs = load_baseline()
    if not recs:
        return []
    header = "| " + " | ".join(h for _, h in BASELINE_COLS) + " |"
    sep = "|" + "|".join("---" for _ in BASELINE_COLS) + "|"
    lines = [
        "",
        "## Baseline to beat (100/100 research phase, eval users 101-200, --short --secs)",
        "",
        "The research workbench is train users 1-100 / eval 101-200 (Andrew 2026-06-29). The headline",
        "baseline is the ORIGINAL d=128 arch (2.76M params) trained FROM SCRATCH on the SAME 1-100 -- this",
        "isolates architecture from training-data quantity (the published d=128 row trained on 5000 users,",
        "so its far-lower numbers are partly just 50x more data). by-user-mean LogLoss; `imm`=RWKV-P",
        "(immediate), `ahead`=RWKV (forgetting-curve). GOAL: a SMALLER model whose imm/ahead matches or",
        "beats the `old d=128 (trained 1-100)` row.",
        "★ RULE (Andrew 2026-06-29): a CHAMPION's comparison logloss must be the DEPLOYED model -- with",
        "quantization AND low-rank state enabled (measured via the Rust engine on 101-200), NOT fp32 --",
        "since that is what ships. The d=128 baseline stays fp32 (it's the accuracy target, not deployable).",
        "Champion rows showing fp32 are PLACEHOLDERS until their deployed (quant+low-rank) number is measured.",
        "",
        header,
        sep,
    ]
    for r in recs:
        def c(k):
            v = r.get(k, "")
            if k == "params" and isinstance(v, int):
                return f"{v:,}"
            if k in ("ahead", "imm") and isinstance(v, (int, float)):
                return f"{v:.4f}"
            return str(v)
        lines.append("| " + " | ".join(c(k) for k, _ in BASELINE_COLS) + " |")
    lines.append("")
    return lines


def load_research():
    if not RESEARCH_JSONL.exists():
        return []
    return [json.loads(l) for l in RESEARCH_JSONL.read_text().splitlines() if l.strip()]


RESEARCH_COLS = [
    ("exp", "exp"), ("change", "change"), ("params", "params"),
    ("ahead", "ahead LL"), ("imm", "imm LL"),
    ("d_ahead", "Δahead vs champ"), ("d_imm", "Δimm vs champ"),
    ("status", "status"), ("note", "note"),
]


def research_section_lines():
    recs = load_research()
    if not recs:
        return []
    header = "| " + " | ".join(h for _, h in RESEARCH_COLS) + " |"
    sep = "|" + "|".join("---" for _ in RESEARCH_COLS) + "|"
    lines = [
        "",
        "## Research-phase experiments (100/100 workbench)",
        "",
        "Train users 1-100 / eval 101-200 (sc8k 8192-chunk, MAX=66000, augmentation OFF, deterministic).",
        "ACCEPT iff BOTH ahead AND imm improve by >=0.0003 vs the CURRENT champion (params<=225k; card+note",
        "per-entity state unchanged; eval review-count identical). The champion is monotonic. d_ahead/d_imm",
        "are vs the champion at the time (positive = better). Verbose notes live in `research_log.md`.",
        "",
        header,
        sep,
    ]
    for r in recs:
        def c(k):
            v = r.get(k, "")
            if k == "params" and isinstance(v, int):
                return f"{v:,}"
            if k in ("ahead", "imm") and isinstance(v, (int, float)):
                return f"{v:.4f}"
            if k in ("d_ahead", "d_imm"):
                if isinstance(v, (int, float)):
                    return f"{v:+.4f}"
                return "—"
            return str(v)
        lines.append("| " + " | ".join(c(k) for k, _ in RESEARCH_COLS) + " |")
    lines.append("")
    return lines


def qat_section_lines():
    recs = load_qat()
    if not recs:
        return []
    # BOTH modes (imm AND ahead/forgetting-curve) recorded for every QAT experiment.
    cols = [("number", "#"), ("params", "params"), ("mode", "training mode"), ("config", "deploy config"),
            ("deploy_imm", "deploy imm"), ("deploy_ahead", "deploy ahead(fc)"),
            ("quant_cost_imm", "quant cost imm"), ("quant_cost_ahead", "quant cost ahead"),
            ("finetune_cost_imm", "fp32 ft-regress imm"), ("finetune_cost_ahead", "fp32 ft-regress ahead"),
            ("gate", "gate"), ("deploy_state", "state")]
    header = "| " + " | ".join(h for _, h in cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    lines = [
        "",
        "## Quant-aware training (QAT) experiments",
        "",
        "State-QAT: the card/note WKV state is round-tripped through int-N every step during training",
        "(STE gradient), so weights adapt to the deploy-time quant. BOTH modes recorded: imm (RWKV-P)",
        "and ahead = the forgetting-curve mode (RWKV). Numbers are by-user-mean on the **17-user gate**",
        "(rust deploy-quant vs rust fp32), NOT the 100-user kernel eval. TWO SEPARATE costs:",
        "- `quant cost` = deploy(quant) - same-QAT-model fp32 = the cost QAT REMOVES (near 0 = QAT works).",
        "- `fp32 ft-regress` = QAT-model fp32 - champion fp32 = an fp32 regression from the (short) fine-",
        "  tune, NOT a quant effect. Decay-only QAT leaves this positive; full-WS QAT / deck-preset grow",
        "  aim to drive it to ~0. NOTE: for the SAME aggressive config, QAT beats PTQ (PTQ card int2+note",
        "  int4 ~+0.0044 FAILS; QAT total +0.0025 PASSES) -- the ft-regress is the only thing left to kill.",
        "Gate vs iter0 (imm ceiling 0.320975, ahead 0.375546).",
        "",
        header,
        sep,
    ]
    for r in recs:
        def c(k):
            v = r.get(k, "")
            if k == "params" and isinstance(v, int):
                return f"{v:,}"
            if k in ("deploy_imm", "deploy_ahead") and isinstance(v, (int, float)):
                return f"{v:.4f}"
            if k.startswith(("quant_cost", "finetune_cost")) and isinstance(v, (int, float)):
                return f"{v:+.4f}"
            return str(v)
        lines.append("| " + " | ".join(c(k) for k, _ in cols) + " |")
    lines.append("")
    return lines


def quant_section_lines():
    recs = load_quant()
    if not recs:
        return []
    header = "| " + " | ".join(h for _, h in QUANT_COLS) + " |"
    sep = "|" + "|".join("---" for _ in QUANT_COLS) + "|"
    lines = [
        "",
        "## State quantization (deploy-time PTQ on the iter36 champion)",
        "",
        "Per-stream round-trip of the recurrent WKV state through int8/int4/int2 at inference",
        "(weights stay fp32). Deltas are by-user-mean vs the **fp32 Rust baseline** on the 17",
        "smallest of users 101-200 (full RNN export of the larger users is infeasible). Gate is",
        "vs iter0 floor (+0.0015 budget, ceilings imm 0.320975 / ahead 0.375546) -- all PASS the",
        "floor, but `verdict` flags how much of the deploy budget each burns. RULE: quant",
        "aggressiveness is proportional to 1/recurrence-length (card int4 ok, note wants int8,",
        "deck/preset/user stay fp32). KiB = quantized per-card / per-note state size.",
        "",
        header,
        sep,
    ]
    for rec in recs:
        lines.append("| " + " | ".join(quant_cell(rec, k) for k, _ in QUANT_COLS) + " |")
    lines.append("")
    return lines


def cell(rec, key):
    if key in ("ahead", "imm"):
        v = rec.get("logloss", {}).get(key)
        return f"{v:.4f}" if isinstance(v, (int, float)) else "—"
    v = rec.get(key, "")
    if key == "params" and isinstance(v, int):
        return f"{v:,}"
    if key == "throughput":
        if v in (None, ""):
            return "pending"
        return f"{v:.1f}" if isinstance(v, (int, float)) else str(v)
    if key == "wilcoxon_p":
        if v in (None, ""):
            return "n/a"
        # decimal when large, scientific when tiny
        return f"{v:.2e}" if v < 1e-3 else f"{v:.4f}"
    if key == "status":
        return v if v else "?"
    return str(v)


def rebuild_md():
    recs = load()
    header = "| " + " | ".join(h for _, h in COLS) + " |"
    sep = "|" + "|".join("---" for _ in COLS) + "|"
    lines = [
        "# Optimization log (steps 4–5–7)",
        "",
        "Regenerated from `log.jsonl` (do not edit by hand). `comment` is in the jsonl only.",
        "Gates: LL not worse than iter0 by >+0.0015 (both modes); state ≤ iter0; size identical.",
        "Gates are vs ITER0 (a floor), NOT the champion — passing all gates does NOT mean accepted.",
        "status: accepted = kept (adopted as a champion or a valid alternative); rejected = not kept",
        "(failed a gate, OR passed the iter0 floor but unreliable/regressed — e.g. iter11).",
        "",
    ]
    lines.extend(baseline_section_lines())
    lines.extend(research_section_lines())
    lines += ["", "## Iteration table (steps 4-5-7)", "", header, sep]
    for rec in recs:
        lines.append("| " + " | ".join(cell(rec, k) for k, _ in COLS) + " |")
    lines.extend(quant_section_lines())
    lines.extend(qat_section_lines())
    MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    nq = len(load_quant())
    nqat = len(load_qat())
    nb = len(load_baseline())
    nr = len(load_research())
    print(f"rebuilt {MD} ({nb} baseline + {nr} research + {len(recs)} iteration + {nq} quant + {nqat} qat rows)")


def add(record_path):
    rec = json.loads(Path(record_path).read_text())
    with open(JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"appended iteration {rec.get('number')} to {JSONL}")
    rebuild_md()


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "add":
        add(sys.argv[2])
    elif len(sys.argv) >= 2 and sys.argv[1] == "rebuild":
        rebuild_md()
    else:
        print(__doc__)
        sys.exit(1)
