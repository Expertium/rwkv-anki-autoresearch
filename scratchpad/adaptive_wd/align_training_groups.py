from pathlib import Path
p=Path('scratchpad/adaptive_wd/adaptive_train.py')
t=p.read_text(encoding='utf-8')
t=t.replace('''    make_train_config(train_cfg,cfg,td,incumbent["model"].parent,
                          incumbent["model"].stem,epoch_target,(epoch_target-1)*g+1,
                          train_db,val_db,users)
''','''    make_train_config(train_cfg,cfg,td,incumbent["model"].parent,
                          incumbent["model"].stem,epoch_target,(epoch_target-1)*g+1,
                          train_db,val_db,users)
''')
# align maximum global len with source database recipe (training db built with MAX_BATCH_SIZE 16384)
t=t.replace('''        "NUM_FETCH_PROCESSES":1,"MAX_TRAIN_GLOBAL_LEN":65536,
''','''        "NUM_FETCH_PROCESSES":1,"MAX_TRAIN_GLOBAL_LEN":16384,
''')
t=t.replace('''    groups=get_groups(train_db,400_000_000_000,65536,list(range(1,5001)))
''','''    groups=get_groups(train_db,400_000_000_000,16384,list(range(1,5001)))
''')
t=t.replace('''    if g<1: raise RuntimeError("training DB has no groups")
''','''    if g<1: raise RuntimeError("training DB has no groups")
''')
p.write_text(t,encoding='utf-8',newline='\n')
print('aligned adaptive grouping to dataset builder batch cap')
