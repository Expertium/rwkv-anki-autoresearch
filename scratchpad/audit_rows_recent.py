import json,re
from pathlib import Path
md=Path("optimization/research_5k.md").read_text(encoding="utf-8",errors="replace")
log=[json.loads(x) for x in Path("optimization/research_log.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
rows=[]
for line in md.splitlines():
 m=re.match(r"^\\|\\s*(\\d+)",line)
 if m: rows.append(int(m.group(1)))
ids=set(rows)
latest=[r for r in log if isinstance(r,dict) and isinstance(r.get("number"),int) and r["number"]>=115]
print("table numeric rows",len(rows),"minmax",(min(rows),max(rows)) if rows else None)
print("latest log numbers missing from table",[(r["number"],r.get("exp")) for r in latest if r["number"] not in ids])
print("latest research table IDs",rows[-20:])
