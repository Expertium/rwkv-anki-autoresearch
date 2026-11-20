from pathlib import Path
p=Path('scratchpad/adaptive_wd/adaptive_train.py')
s=p.read_text(encoding='utf-8')
old='''    run_checked([PY, ROOT / "optimization/eval_sharded.py", "--config", cfg,
                 "--shards", shards, "--solo-threshold", 0,
                 "--fetch-per-shard", 1, "--threads-per-shard", 7],
                ROOT, Path(run_dir) / f"eval_{tag}.log", env)
'''
new='''    users_path=Path(run_dir)/f"users_{tag}.json"
    users_path.write_text(json.dumps(list(users)),encoding="utf-8")
    run_checked([PY, ROOT / "optimization/eval_sharded.py", "--config", cfg,
                 "--users-file", users_path, "--shards", shards, "--solo-threshold", 0,
                 "--fetch-per-shard", 1, "--threads-per-shard", 7],
                ROOT, Path(run_dir) / f"eval_{tag}.log", env)
'''
assert old in s
s=s.replace(old,new,1)
old2='''        "VALIDATE_USERS_START":min(users),"VALIDATE_USERS_END":max(users),
'''
new2='''        "VALIDATE_USERS_START":min(users),"VALIDATE_USERS_END":max(users),
        "VALIDATE_USERS_FILE":str(Path(out).with_name("validation_users.json")),
'''
assert old2 in s
s=s.replace(old2,new2,1)
old3='''    shutil.copy2(base,out); edit_toml(out,values)
'''
new3='''    shutil.copy2(base,out); edit_toml(out,values)
    Path(values["VALIDATE_USERS_FILE"]).write_text(json.dumps(list(users)),encoding="utf-8")
'''
assert old3 in s
s=s.replace(old3,new3,1)
p.write_text(s,encoding='utf-8',newline='\n')
print('added explicit 750-user selectors to train/eval runner')
