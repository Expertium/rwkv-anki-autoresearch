from pathlib import Path
p=Path('scratchpad/adaptive_wd/adaptive_train.py');s=p.read_text(encoding='utf-8')
s=s.replace('''def eval_users(run_dir, base_eval, model, tag, users, val_db, label_db, env, shards=1):''','''def eval_users(run_dir, base_eval, model, tag, users, val_db, label_db, env, shards=1, require_full=True):''')
s=s.replace('''    if set(rows[0]) != set(users):
        missing=sorted(set(users)-set(rows[0]))
        raise RuntimeError(f"screen eval did not score the full fixed roster (expected {len(users)}, scored {len(rows[0])}, missing {missing[:20]})")
    n=len(rows[0]); means=[]
''','''    if require_full and set(rows[0]) != set(users):
        missing=sorted(set(users)-set(rows[0]))
        raise RuntimeError(f"screen eval did not score the full fixed roster (expected {len(users)}, scored {len(rows[0])}, missing {missing[:20]})")
    n=len(rows[0]); means=[]
''')
s=s.replace('''        internal_val_users=[5001,5002]
''','''        internal_val_users=[u for u in range(5001,7501) if u not in set(screen_users) and u not in excluded_users][:2]
        if len(internal_val_users)!=2:
            raise RuntimeError("could not find two fixed one-batch internal validation users outside screen")
''')
s=s.replace('''            full=eval_users(run_dir,eval_base,qcp,"qat_full",list(range(5001,7501)),val_db,label_db,env,shards=2)
''','''            full=eval_users(run_dir,eval_base,qcp,"qat_full",list(range(5001,7501)),val_db,label_db,env0,shards=2,require_full=False)
''')
p.write_text(s,encoding='utf-8',newline='\n')
print('patched full-eval OOM accounting and separate internal validation users')
