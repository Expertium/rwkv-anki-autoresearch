"""One-shot: append the Rust-finiteness ignore rules to .gitignore (mixed line endings defeat `edit`)."""
from pathlib import Path

P = Path(__file__).resolve().parents[3] / ".gitignore"
text = P.open(encoding="utf-8", newline="").read()
nl = "\r\n" if "\r\n" in text else "\n"
BLOCK = """# --- Rust finiteness check (finite.rs), added 2026-11-22 ---
# Per-user engine prediction dumps of the finiteness and speed A/B runs (regenerable in minutes by
# rust_finite.py / measure_finite_offpath.py / measure_offpath_oldbuild.py), the throw-away target
# dir plus the pre-change engine binary used as the OFF-path control, and temporary commit-message
# files. The runners and the rust_finite_*.json summaries ARE tracked.
scratchpad/quant_plan/rust_finite_preds/
scratchpad/quant_plan/rust_finite_oldbuild_preds/
scratchpad/quant_plan/rust_finite_offpath_preds_*/
scratchpad/quant_plan/rust_speed_preds_*/
scratchpad/quant_plan/rust_bin/
rust/rwkv-infer/target_offpath_old/
scratchpad/quant_plan/corruption/_commitmsg*.txt
"""
if "rust_finite_preds" in text:
    print("rules already present - nothing to do")
else:
    if not text.endswith(nl):
        text += nl
    text += nl + BLOCK.replace("\n", nl)
    P.open("w", encoding="utf-8", newline="").write(text)
    print(f"appended {len(BLOCK.splitlines())} lines (newline {nl!r})")
