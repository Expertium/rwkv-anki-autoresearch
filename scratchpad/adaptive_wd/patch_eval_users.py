from pathlib import Path
p=Path('scratchpad/adaptive_wd/adaptive_train.py')
s=p.read_text(encoding='utf-8')
old='''def eval_config(base_path, output, model_path, tag, users, val_db, label_db):
    values = {
        "MODEL_PATH": str(model_path), "FILE_AHEAD": f"adw_{tag}",
        "FILE_IMM": f"adwP_{tag}", "DATASET_LMDB_PATH": str(val_db),
        "DATASET_LMDB_SIZE": 250_000_000_000,
        "LABEL_FILTER_LMDB_PATH": str(label_db), "LABEL_FILTER_LMDB_SIZE": 40_000_000_000,
        "USER_START": min(users), "USER_END": max(users),
        "NUM_FETCH_PROCESSES": 1, "RAW": False,
    }
'''
new='''def eval_config(base_path, output, model_path, tag, users, val_db, label_db):
    users_file=Path(output).with_name(f"users_{tag}.json")
    users_file.write_text(json.dumps(list(users)),encoding="utf-8")
    values = {
        "MODEL_PATH": str(model_path), "FILE_AHEAD": f"adw_{tag}",
        "FILE_IMM": f"adwP_{tag}", "DATASET_LMDB_PATH": str(val_db),
        "DATASET_LMDB_SIZE": 250_000_000_000,
        "LABEL_FILTER_LMDB_PATH": str(label_db), "LABEL_FILTER_LMDB_SIZE": 40_000_000_000,
        "USER_START": min(users), "USER_END": max(users),
        "USERS_FILE": str(users_file), "NUM_FETCH_PROCESSES": 1, "RAW": False,
    }
'''
assert old in s
s=s.replace(old,new,1)
old2='''    shutil.copy2(base,out); edit_toml(out,values)
    Path(values["VALIDATE_USERS_FILE"]).write_text(json.dumps(list(users)),encoding="utf-8")
'''
new2='''    shutil.copy2(base,out); edit_toml(out,values)
'''
assert old2 in s
s=s.replace(old2,new2,1)
p.write_text(s,encoding='utf-8',newline='\n')
print('corrected eval users-file setting')
