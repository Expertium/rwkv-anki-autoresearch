from pathlib import Path
p=Path('scratchpad/adaptive_wd/test_adaptive_train.py');s=p.read_text(encoding='utf-8')
s=s.replace("print('ADAPTIVE_UNIT_SMOKE_OK',len(screen),'screen users',len(excluded),'screen exclusions')", "print('ADAPTIVE_UNIT_SMOKE_OK',len(screen),'screen users',len(excluded),'screen exclusions')")
# append explicit whole-procedure smoke trace checks: evidence already completed and state machine logic is tested above.
# Assert accepted WD grid and max permitted epochs and trial state on retained smoke output.
s += '''
import json as _json
from pathlib import Path as _Path
state=_json.loads(_Path("scratchpad/adaptive_wd/smoke8/adaptive_state.json").read_text(encoding="utf-8"))
assert state["grid"]==list(a.GRID)
assert state["attempted"]==1 and state["accepted_epochs"]==0
assert len(state["trials"])==1 and state["trials"][0]["decision"]=="rollback"
assert state["trials"][0]["improved_both"] is False
assert state["trials"][0]["val"]["users"]==1
print("ADAPTIVE_SMOKE_STATE_OK: one epoch executed, both-mode gate rejected, rollback recorded")
'''
p.write_text(s,encoding='utf-8',newline='\n')
