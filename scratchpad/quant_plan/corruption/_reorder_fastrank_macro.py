"""One-shot: `macro_rules!` is textual, so `rankr_fast_impl` must be DEFINED before `rankr_fast`
uses it. The generator block put the dispatcher first; this swaps the two halves in place.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
p = ROOT / "rust" / "rwkv-infer" / "src" / "model.rs"
text = p.open(encoding="utf-8", newline="").read()
nl = "\r\n" if text.count("\r\n") >= text.count("\n") - text.count("\r\n") else "\n"

start = text.index("/// FAST rank-r truncation for one KxK state:")
end = text.index("/// In-place rank-r SVD-truncation + quantization of a flat row-major")
region = text[start:end]
split = region.index("/// One (K, r) shape of the fast path")
dispatch = region[:split].rstrip("\r\n")
macro_def = region[split:].rstrip("\r\n")
assert dispatch.lstrip().startswith("///"), "unexpected split"
assert macro_def.startswith("/// One (K, r) shape"), "unexpected split"
assert "macro_rules! rankr_fast_impl" in macro_def, "the macro body is missing"
assert "fn rankr_fast(" in dispatch, "the dispatcher is missing"

text = text[:start] + macro_def + nl + nl + dispatch + nl + nl + text[end:]
p.open(encoding="utf-8", newline="", mode="w").write(text)
print(f"swapped: macro definition now at line {text[:start].count(chr(10)) + 1}, dispatcher follows")
