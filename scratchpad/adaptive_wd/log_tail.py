from pathlib import Path
p=Path('scratchpad/adaptive_wd/build_train_db.log')
s=p.read_text(encoding='utf-8',errors='replace')
print('logs bytes',p.stat().st_size,'last:',repr(s.splitlines()[-1] if s.splitlines() else 'empty'))
