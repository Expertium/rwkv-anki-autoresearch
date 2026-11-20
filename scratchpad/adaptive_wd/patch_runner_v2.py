from pathlib import Path
p=Path('scratchpad/adaptive_wd/adaptive_train.py')
s=p.read_text(encoding='utf-8')
s=s.replace('''SCREEN_USERS = tuple(range(5001, 5751))''','''SCREEN_USERS = tuple(range(5751, 6501))''')
s=s.replace('''        "VALIDATE_USERS_START":min(users),"VALIDATE_USERS_END":max(users),
        "VALIDATE_USERS_FILE":str(Path(out).with_name("validation_users.json")),
        "TRAIN_DATASET_LMDB_PATH":str(train_db),"TRAIN_DATASET_LMDB_SIZE":400_000_000_000,
        "VALIDATE_DATASET_LMDB_PATH":str(val_db),"VALIDATE_DATASET_LMDB_SIZE":250_000_000_000,
        "NUM_FETCH_PROCESSES":1,"MAX_TRAIN_GLOBAL_LEN":16384,''','''        "VALIDATE_USERS_START":min(users),"VALIDATE_USERS_END":max(users),
        "VALIDATE_USERS_FILE":str(Path(out).with_name("validation_users.json")),
        "TRAIN_DATASET_LMDB_PATH":str(train_db),"TRAIN_DATASET_LMDB_SIZE":400_000_000_000,
        "VALIDATE_DATASET_LMDB_PATH":str(val_db),"VALIDATE_DATASET_LMDB_SIZE":250_000_000_000,
        "NUM_FETCH_PROCESSES":1,"MAX_TRAIN_GLOBAL_LEN":65536,''')
s=s.replace('''    if not rows[0] or set(rows[0]) != set(rows[1]): raise RuntimeError("empty or unequal mode roster")
    n=len(rows[0]); means=[]''','''    if not rows[0] or set(rows[0]) != set(rows[1]): raise RuntimeError("empty or unequal mode roster")
    if set(rows[0]) != set(users):
        missing=sorted(set(users)-set(rows[0]))
        raise RuntimeError(f"screen eval did not score the full fixed roster (expected {len(users)}, scored {len(rows[0])}, missing {missing[:20]})")
    n=len(rows[0]); means=[]''')
s=s.replace('''        "VALIDATE_USERS_START":min(users),"VALIDATE_USERS_END":max(users),
        "VALIDATE_USERS_FILE":str(Path(out).with_name("validation_users.json")),''','''        "VALIDATE_USERS_START":min(users),"VALIDATE_USERS_END":max(users),
        "VALIDATE_USERS_FILE":str(Path(out).with_name("validation_users.json")),''')
s=s.replace('''    groups=get_groups(train_db,400_000_000_000,16384,list(range(1,5001)))''','''    groups=get_groups(train_db,400_000_000_000,65536,list(range(1,5001)))''')
s=s.replace('''    base=eval_users(run_dir,eval_base,start_model,"baseline",users,val_db,label_db,env0,shards=1)''','''    # Runtime output names are unique; eval_sharded refuses to overwrite any previous result.
    import uuid
    result_tag=run_dir.name+"_"+uuid.uuid4().hex[:8]
    base=eval_users(run_dir,eval_base,start_model,result_tag+"_baseline",users,val_db,label_db,env0,shards=1)''')
s=s.replace('''        make_train_config(train_cfg,cfg,td,incumbent["model"].parent,
                          incumbent["model"].stem,epoch_target,(epoch_target-1)*g+1,
                          train_db,val_db,users)''','''        internal_val_users=users[:2]
        make_train_config(train_cfg,cfg,td,incumbent["model"].parent,
                          incumbent["model"].stem,(0.04 if args.smoke else epoch_target),
                          (1 if args.smoke else (epoch_target-1)*g+1),
                          train_db,val_db,internal_val_users)''')
s=s.replace('''        val=read_validation_log(log)
        step=epoch_target*g
        cp=td/f"adaptive_{step}.pth"; op=td/f"adaptive_optim_{step}.pth"''','''        trainer_val=read_validation_log(log)  # checks trainer's own validation ran; gate uses per-user mean below
        trained_steps=int(0.04*g) if args.smoke else epoch_target*g
        step=trained_steps
        cp=td/f"adaptive_{step}.pth"; op=td/f"adaptive_optim_{step}.pth"''')
s=s.replace('''        prev=incumbent["score"]
        win=val["ahead"]<prev["ahead"] and val["imm"]<prev["imm"]
        rec={"attempt":attempt,"epoch":epoch_target,"weight_decay":wd,"val":val,''','''        val=eval_users(run_dir,eval_base,cp,result_tag+f"_a{attempt:02d}",users,val_db,label_db,env0,shards=1)
        prev=incumbent["score"]
        win=val["ahead"]<prev["ahead"] and val["imm"]<prev["imm"]
        rec={"attempt":attempt,"epoch":epoch_target,"weight_decay":wd,"val":val,"trainer_val":trainer_val,''')
s=s.replace('''    users=list(SCREEN_USERS[:2] if args.smoke else SCREEN_USERS)''','''    users=list(SCREEN_USERS[:2] if args.smoke else SCREEN_USERS)''')
# smoke global-step checkpoint naming uses fraction of total epoch; use computed actual trainer step count
# avoid complete-roster data issue with exact no-multibatch screen before work.
s=s.replace('''    # Count the exact training groups using the same trainer function and current source data.
    from rwkv.train_rwkv import get_groups''','''    # The trainer validation path requires one batch per user. Verify this screen in the
    # LMDB before running; production external evaluation also enforces full roster coverage.
    import lmdb
    venv=lmdb.open(val_db,map_size=250_000_000_000,readonly=True,lock=False,readahead=False,max_readers=512)
    with venv.begin() as txn:
        bad=[]
        for u in users:
            raw=txn.get(f"{u}_batches".encode())
            if raw is None or len(json.loads(raw)) != 1: bad.append(u)
    venv.close()
    if bad: ap.error(f"fixed validation screen must have one DB batch per user; incompatible users: {bad}")
    # Count the exact training groups using the same trainer function and current source data.
    from rwkv.train_rwkv import get_groups''')
p.write_text(s,encoding='utf-8',newline='\n')
print('patched screen, exact epoch count, by-user evaluation gate, and unique result tags')
