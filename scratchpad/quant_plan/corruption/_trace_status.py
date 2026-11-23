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

roster_p = HERE / "corrupt50.json"          # the roster lives NEXT TO this script, not in RUN
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
    # The export writes `n_reviews` (the user's whole history) and `size` (the number of review_ths
    # the model actually consumed). Neither `review_th` nor `rating` exists in these traces, which is
    # why this printed 0 for every user before 2026-11-23.
    n = doc.get("n_reviews")
    if n is None:
        n = doc.get("size")
    if n is None:
        n = len(doc.get("equalize_review_ths") or doc.get("review_th") or doc.get("rating") or [])
    la, li = doc.get("py_ahead_logloss"), doc.get("py_imm_logloss")
    done.append((u, n, p.stat().st_size + st.stat().st_size, la, li))

done.sort(key=lambda t: t[1])
print(f"\ncomplete traces: {len(done)}")
for u, n, sz, la, li in done:
    ll = (f"  py_ahead={la:.4f} py_imm={li:.4f}"
          if isinstance(la, float) and isinstance(li, float) else "")
    print(f"  {u:5d}  reviews={n:6d}  bytes={sz:,}{ll}")
if done:
    print("\nDONE_USERS=" + ",".join(str(t[0]) for t in done))

want = roster.get("users") if isinstance(roster, dict) else None
if want:
    got = {t[0] for t in done}
    missing = [u for u in want if u not in got]
    print(f"roster users: {len(want)} | complete traces: {len(got & set(want))} | "
          f"MISSING: {len(missing)} -> {missing}")
