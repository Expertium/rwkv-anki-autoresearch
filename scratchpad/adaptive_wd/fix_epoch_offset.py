from pathlib import Path
p=Path('scratchpad/adaptive_wd/adaptive_train.py')
s=p.read_text(encoding='utf-8')
old='incumbent["model"].stem,epoch_target,epoch_target-1,'
new='incumbent["model"].stem,epoch_target,(epoch_target-1)*g+1,'
assert old in s
p.write_text(s.replace(old,new,1),encoding='utf-8',newline='\n')
print('updated resumed epoch STEP_OFFSET')
