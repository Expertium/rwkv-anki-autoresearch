"""Replace the old 'two readings' sentence with Andrew's 2026-11-20 ruling (line-break agnostic)."""
from pathlib import Path

P = Path(__file__).resolve().parent / "RESULTS.md"
t = P.read_text(encoding="utf-8")

start_marker = "The first reading is the literal constraint"
end_marker = 'must be nearly free").'
i = t.find(start_marker)
assert i != -1, "start marker not found"
j = t.find(end_marker, i)
assert j != -1, "end marker not found"
j += len(end_marker)
old = t[i:j]
new = ('The first reading is the anchor comparison; the second is the micro-bar ("quantization itself\n'
       'must be nearly free"), and **Andrew ruled on 2026-11-20 that the second one governs**.')
print("replacing", len(old), "chars:")
print(old)
t = t[:i] + new + t[j:]
P.write_text(t, encoding="utf-8")
print("done; lines:", len(t.splitlines()))