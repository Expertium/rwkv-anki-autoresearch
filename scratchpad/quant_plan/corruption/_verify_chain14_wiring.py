"""Verify the chain-14 wiring: shrink waits for it36noterank2, corrupt2's POINTS and the corruption
driver's CONFIGS carry the four new tags."""
import re
import sys
from pathlib import Path

RUN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN))

shrink = (RUN / "it36shrink_chain.py").open(encoding="utf-8").read()
m = re.search(r"chain_main\(\"it36shrink\", \"([^\"]+)\",\s*\"([^\"]+)\"", shrink)
print("shrink waits on:", m.groups() if m else "NOT FOUND")
print("shrink BUSY_PAT has the corruption patterns:",
      "probe_get_result|check_corruption" in shrink)

import it36corrupt2_chain as c2  # noqa: E402
print("corrupt2 POINTS:", len(c2.POINTS))
new = [p[0] for p in c2.POINTS if p[0].startswith(("l2i8n", "l3i8n"))]
print("chain-14 tags in corrupt2:", new)

import importlib.util  # noqa: E402
spec = importlib.util.spec_from_file_location("cc", RUN / "corruption" / "check_corruption.py")
cc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cc)
print("check_corruption CONFIGS has the new tags:",
      [k for k in cc.CONFIGS if k.startswith(("l2i8n", "l3i8n"))])
