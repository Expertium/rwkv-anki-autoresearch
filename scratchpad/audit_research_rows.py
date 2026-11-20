import json,re
from pathlib import Path
md=Path("optimization/research_5k.md").read_text(encoding="utf-8",errors="replace")
log=[json.loads(x) for x in Path("optimization/research_log.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
rows=[]
for line in md.splitlines():
 m=re.match(r"^\\|\\s*(\\d+)\\s+`?([^`|]+)",line)
 if m: rows.append((int(m.group(1)),m.group(2).strip()))
ids={n for n,_ in rows}
missing=[(r["number"],r.get("exp")) for r in log if isinstance(r,dict) and "number" in r and r["number"] not in ids]
print("table rows",len(rows),"range",(min(ids),max(ids)) if ids else None)
print("JSONL records",len(log))
print("JSONL experiments lacking table rows",missing)
print("latest table rows",rows[-8:])
print("latest JSONL numbered entries",[(r.get("number"),r.get("exp"),r.get("status")) for r in log[-8:] if isinstance(r,dict)])
