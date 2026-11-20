from pathlib import Path
p=Path('scratchpad/adaptive_wd/adaptive_train.py')
s=p.read_text(encoding='utf-8')
s=s.replace('''SCREEN_USERS = tuple(range(5001, 5751))
''','''SCREEN_START, SCREEN_END, SCREEN_COUNT = 5001, 7500, 750

def choose_screen_users(val_db, count=SCREEN_COUNT):
    import lmdb
    env=lmdb.open(val_db,map_size=250_000_000_000,readonly=True,lock=False,
                  readahead=False,max_readers=512)
    eligible=[]; excluded=[]
    with env.begin() as txn:
        for user in range(SCREEN_START,SCREEN_END+1):
            raw=txn.get(f"{user}_batches".encode())
            if raw is None: excluded.append(user); continue
            if len(json.loads(raw)) != 1: excluded.append(user); continue
            eligible.append(user)
    env.close()
    if len(eligible)<count:
        raise RuntimeError(f"Need {count} one-batch VAL users; found {len(eligible)}")
    # Deterministic, near-uniform selection across the full VAL half.
    chosen=[eligible[((2*i+1)*len(eligible))//(2*count)] for i in range(count)]
    return tuple(chosen), tuple(excluded)
''')
s=s.replace('''    users=list(SCREEN_USERS[:2] if args.smoke else SCREEN_USERS)
''','''    screen_users, excluded_users=choose_screen_users(val_db)
    users=list(screen_users[:2] if args.smoke else screen_users)
''')
s=s.replace('''    # The trainer validation path requires one batch per user. Verify this screen in the
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
''','')
s=s.replace('''    state={"grid":list(GRID),"max_attempts":args.max_attempts,"attempted":0,
           "accepted_epochs":0,"users":users,"baseline":base,"trials":[]}
''','''    state={"grid":list(GRID),"max_attempts":args.max_attempts,"attempted":0,
           "accepted_epochs":0,"users":users,"screen_excluded_multibatch_or_missing":list(excluded_users),
           "baseline":base,"trials":[]}
''')
s=s.replace('''        val=eval_users(run_dir,eval_base,cp,result_tag+f"_a{attempt:02d}",users,val_db,label_db,env0,shards=1)
        prev=incumbent["score"]
        win=val["ahead"]<prev["ahead"] and val["imm"]<prev["imm"]
''','''        val_result=eval_users(run_dir,eval_base,cp,result_tag+f"_a{attempt:02d}",users,val_db,label_db,env0,shards=1)
        val={"ahead":val_result["ahead"],"imm":val_result["imm"],"users":val_result["n"]}
        prev=incumbent["score"]
        win=should_accept(prev,val)
''')
s=s.replace('''        if not win:
            rec["decision"]="rollback"
            wd_i+=1
''','''        if not win:
            rec["decision"]="rollback"
            wd_i=next_weight_decay_index(wd_i)
''')
s=s.replace('''def toml_repr(value):
''','''def should_accept(previous, candidate):
    """Strict dual-mode screen rule: both by-user means must go down."""
    return candidate["ahead"] < previous["ahead"] and candidate["imm"] < previous["imm"]

def next_weight_decay_index(current):
    """Move one rung up the fixed WD grid after a rejected epoch; saturate at last rung."""
    return min(current+1,len(GRID)-1)

def toml_repr(value):
''')
p.write_text(s,encoding='utf-8',newline='\n')
print('replaced contiguous screen with stratified 750 one-batch users; added strict decision helpers')
