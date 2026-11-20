from pathlib import Path
p=Path('scratchpad/adaptive_wd/adaptive_train.py');s=p.read_text(encoding='utf-8')
# The smoke's baseline evaluation needs one representative screen user. Then train at 0.04 epoch.
s=s.replace('''    users=list(screen_users[:2] if args.smoke else screen_users)
''','''    users=list(screen_users[:1] if args.smoke else screen_users)
''')
s=s.replace('''    base=eval_users(run_dir,eval_base,start_model,result_tag+"_baseline",users,val_db,label_db,env0,shards=1)
''','''    base=eval_users(run_dir,eval_base,start_model,result_tag+"_baseline",users,val_db,label_db,env0,shards=1,
                    require_full=False if args.smoke else True)
''')
s=s.replace('''    state={"grid":list(GRID),"max_attempts":args.max_attempts,"attempted":0,
''','''    if args.smoke and base["n"] < 1: raise RuntimeError("smoke baseline evaluation produced no scored users")
    state={"grid":list(GRID),"max_attempts":args.max_attempts,"attempted":0,
''')
p.write_text(s,encoding='utf-8',newline='\n')
print('reduced smoke baseline to one user, allowing OOM accounting')
