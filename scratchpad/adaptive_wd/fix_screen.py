from pathlib import Path
p=Path('scratchpad/adaptive_wd/adaptive_train.py')
s=p.read_text(encoding='utf-8')
s=s.replace('SCREEN_USERS = tuple(range(5751, 6501))','SCREEN_USERS = tuple(range(5001, 5751))')
s=s.replace('''        internal_val_users=users[:2]
        make_train_config(train_cfg,cfg,td,incumbent["model"].parent,
''','''        # Trainer validation requires one batch per user. Keep the fixed primary 750-user
        # screen for the production evaluator; for periodic train-time validation use the
        # same stable two-user probe in both modes and do not use it as the promotion metric.
        internal_val_users=[5001,5002]
        make_train_config(train_cfg,cfg,td,incumbent["model"].parent,
''')
s=s.replace('''    base=eval_users(run_dir,eval_base,start_model,result_tag+"_baseline",users,val_db,label_db,env0,shards=1)''','''    base=eval_users(run_dir,eval_base,start_model,result_tag+"_baseline",users,val_db,label_db,env0,shards=1)''')
p.write_text(s,encoding='utf-8',newline='\n')
print('screen reverted to 5001-5750')
