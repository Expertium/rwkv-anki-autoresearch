from pathlib import Path
p=Path('scratchpad/adaptive_wd/adaptive_train.py');s=p.read_text(encoding='utf-8')
s=s.replace('''    from rwkv.train_rwkv import get_groups
    groups=get_groups(train_db,400_000_000_000,16384,list(range(1,5001)))
''','''    import sys
    sys.path.insert(0,str(ROOT))
    from rwkv.train_rwkv import get_groups
    groups=get_groups(train_db,400_000_000_000,16384,list(range(1,5001)))
''')
p.write_text(s,encoding='utf-8',newline='\n')
print('ensured repository import path in adaptive main')
