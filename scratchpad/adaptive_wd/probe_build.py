import json,sys
from pathlib import Path
sys.path.insert(0,str(Path('.').resolve()))
from rwkv.train_rwkv import get_groups
p=Path('scratchpad/adaptive_wd/build_train_db.log')
text=p.read_text(encoding='utf-8',errors='replace')
print('last',text.splitlines()[-1] if text.splitlines() else 'EMPTY')
