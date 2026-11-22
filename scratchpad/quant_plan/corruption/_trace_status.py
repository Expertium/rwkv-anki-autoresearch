"""List the corrupt50 traces that are COMPLETE (both .json and .safetensors present), with sizes.

A half-written trace would make a Rust grid run fail, so the user list for any wide-roster run is
pinned to these. Sharding is safe because `export_user` seeds torch per USER, so the set is the same
regardless of which shard produced it.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
D = RUN / "trace_corrupt50"

roster_p = RUN / "corrupt50.json"
roster = json.loads(roster_p.read_text(encoding="utf-8")) if roster_p.exists() else {}
if isinstance(roster, dict):
    ids = roster.get("users") or roster.get("roster") or list(roster.get("excluded_too_big", []))
else:
    ids = roster
print("roster type:", type(roster).__name__,
      "| keys:", list(roster)[:8] if isinstance(roster, dict) else len(roster))

done = []
for p in sorted(D.glob("trace_user_*.json")):
    u = int(p.stem.split("_")[-1])
    st = p.with_suffix(".safetensors")
    if not st.exists():
        continue
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:                                    # noqa: BLE001
        print(f"  {u}: unreadable ({e})")
        continue
    n = len(doc.get("review_th") or doc.get("rating") or [])
    done.append((u, n, p.stat().st_size + st.stat().st_size))

done.sort(key=lambda t: t[1])
print(f"\ncomplete traces: {len(done)}")
for u, n, sz in done:
    print(f"  {u:5d}  reviews={n:6d}  bytes={sz:,}")
if done:
    print("\nDONE_USERS=" + ",".join(str(u) for u, _, _ in done))
