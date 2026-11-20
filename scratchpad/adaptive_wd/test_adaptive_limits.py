import sys
sys.path.insert(0, "scratchpad/adaptive_wd")
import adaptive_train as a
# Current run state: three accepted epochs at WD=0.01.
s={"accepted_epochs":3,"trials":[{"weight_decay":0.01,"decision":"accept"} for _ in range(3)]}
assert a.resume_schedule(s)==(0,4,0)
# A rejected epoch advances one WD rung, but does not increase successful epochs.
s={"accepted_epochs":2,"trials":[{"weight_decay":0.01,"decision":"accept"},{"weight_decay":0.01,"decision":"accept"},{"weight_decay":0.02,"decision":"rollback"}]}
assert a.resume_schedule(s)==(2,3,1)
# Seven rejections are represented without exceeding the seven-rung grid.
s={"accepted_epochs":0,"trials":[{"weight_decay":wd,"decision":"rollback"} for wd in a.GRID]}
assert a.resume_schedule(s)==(6,1,7)
assert len(a.GRID)==7
print("ADAPTIVE_LIMIT_RESUME_TESTS_OK")
