from pathlib import Path
p=Path('scratchpad/adaptive_wd/adaptive_train.py');s=p.read_text(encoding='utf-8')
s=s.replace('''    from rwkv.train_rwkv import get_groups
    groups=get_groups(train_db,400_000_000_000,16384,list(range(1,5001)))
    g=len(groups); del groups
''','''    from rwkv.train_rwkv import get_groups
    groups=get_groups(train_db,400_000_000_000,16384,list(range(1,5001)))
    g=len(groups); del groups
    if args.smoke:
        g=min(g,8)
''')
s=s.replace('''        trained_steps=max(1,int(0.04*g)) if args.smoke else epoch_target*g
''','''        trained_steps=max(1,int(0.04*g)) if args.smoke else epoch_target*g
''')
p.write_text(s,encoding='utf-8',newline='\n')
print('bounded smoke group-count to 8 so smoke path cannot execute a full epoch')
