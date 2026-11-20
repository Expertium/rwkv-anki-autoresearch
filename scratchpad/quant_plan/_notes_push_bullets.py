"""One-shot: prepend the two newest bullets under '## Live status' in scratchpad/quant_plan/NOTES.md.

Written as a script because the file is CRLF and the bullet text is long; `edit` failed to match it.
"""
from pathlib import Path

P = Path(__file__).resolve().parent / "NOTES.md"
text = P.read_text(encoding="utf-8")
nl = "\r\n" if "\r\n" in text else "\n"

marker = "## Live status"
idx = text.find(marker)
if idx < 0:
    raise SystemExit("marker not found")

# Insert after the marker line (and its trailing blank line).
after = text.find(nl, idx)          # end of the "## Live status" line
after = text.find(nl, after + len(nl))  # skip the empty line too
if after < 0:
    raise SystemExit("unexpected file shape")

bullets = [
    "* 2026-11-20 22:52 - **PUSHED TO GITHUB.** Commit `69121550` (\"Document the card/note "
    "state-quantization\n  arm, gated on the identical unquantized twin\") is on `origin/main` "
    "(`git ls-remote` confirms it): 421\n  files, 21,889 insertions - the arm's runners, "
    "`RESULTS.md`/`NOTES.md`, the board and research-log\n  updates, the ram-guard/detach repairs, "
    "the adaptive_wd/ta3 notes and runners, and new `.gitignore`\n  rules for the regenerable per-user "
    "prediction dumps (`scratchpad/quant_plan/preds_*/`,\n  `rust_grid_preds/`, `rust_speed_preds/`) "
    "plus the stray root-level files. The 2,158 stale scratchpad\n  paths that are deleted locally "
    "were deliberately NOT committed (they are still on the remote), and\n  `model_weights.txt` (a "
    "derived 4 MB dump from `rewrite_pth_text_helper.py`) is now ignored, not tracked.",

    "* 2026-11-20 22:46 - **Row 0b scored: the 496-step fp32 control (`qnj_it36ctl496`) = 0.321659 "
    "ahead /\n  0.294614 imm**, i.e. **0.012143/0.011598 BETTER than row 0** (ub95 +0.012911/+0.012336, "
    "p 2.0e-108/\n  1.1e-122) and 0.021154/0.019659 better than the iter36 champion. The decay length "
    "moves the loss about\n  25x more than the whole quantization tax does (0.0121/0.0116 vs "
    "0.0005/0.0010), so row 8 (`l1i8` at 496\n  steps) is quoted against THIS twin, never against row "
    "0. `it36qatsmall_chain` job 0 finished rc=0 in\n  20.4 min (22:45:52); the chain is now on the "
    "`l1i4` 124-step QAT (its smoke finished rc=0 in 1.2 min at\n  22:47:03).",
]

insert = nl.join(b.lstrip("\n").replace("\n", nl) for b in bullets) + nl + nl
text = text[:after + len(nl)] + insert + text[after + len(nl):]
P.write_text(text, encoding="utf-8", newline="")
print("NOTES.md lines now:", len(text.splitlines()))
print("bullet markers:", text.count("2026-11-20 22:52"), text.count("2026-11-20 22:46"))
