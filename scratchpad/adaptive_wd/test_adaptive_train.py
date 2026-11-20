import sys
sys.path.insert(0,'scratchpad/adaptive_wd')
import adaptive_train as a
assert a.SCREEN_COUNT==750
screen,excluded=a.choose_screen_users(r'F:/rwkv_lmdb/test_db_5k_id5')
assert len(screen)==750 and len(set(screen))==750
assert all(5001<=u<=7500 for u in screen)
assert all(u not in screen for u in excluded)
assert a.should_accept({'ahead':1,'imm':1},{'ahead':.99,'imm':.99})
assert not a.should_accept({'ahead':1,'imm':1},{'ahead':.99,'imm':1})
assert [a.next_weight_decay_index(i) for i in range(7)]==[1,2,3,4,5,6,6]
assert a.resume_schedule({"trials":[{"weight_decay":0.01,"decision":"accept"}],"accepted_epochs":1}) == (0,2,0)
assert a.resume_schedule({"trials":[{"weight_decay":0.01,"decision":"accept"},{"weight_decay":0.01,"decision":"rollback"}],"accepted_epochs":1}) == (1,2,1)
print('ADAPTIVE_UNIT_SMOKE_OK',len(screen),'screen users',len(excluded),'screen exclusions')

import json as _json
from pathlib import Path as _Path
state=_json.loads(_Path("scratchpad/adaptive_wd/smoke8/adaptive_state.json").read_text(encoding="utf-8"))
assert state["grid"]==list(a.GRID)
assert state["attempted"]==1 and state["accepted_epochs"]==0
assert len(state["trials"])==1 and state["trials"][0]["decision"]=="rollback"
assert state["trials"][0]["improved_both"] is False
assert state["trials"][0]["val"]["users"]==1
print("ADAPTIVE_SMOKE_STATE_OK: one epoch executed, both-mode gate rejected, rollback recorded")
