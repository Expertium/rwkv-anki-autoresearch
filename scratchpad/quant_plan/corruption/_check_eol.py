"""One-shot: report line endings of the three docs the corruption check has to touch."""
import pathlib

for n in ("AGENTS.md", "optimization/GPU_RUN_BOARD.md", "scratchpad/quant_plan/NOTES.md"):
    b = pathlib.Path(n).read_bytes()
    print(n, "CRLF", b.count(b"\r\n"), "LF", b.count(b"\n"), "size", len(b))
