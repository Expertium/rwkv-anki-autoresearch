from pathlib import Path
p=Path('scratchpad/adaptive_wd/adaptive_train.py');s=p.read_text(encoding='utf-8')
s=s.replace('''    validation_users_path=run_dir/"validation_users.json"
    validation_users_path.write_text(json.dumps(internal_val_users),encoding="utf-8")
''','''    validation_users_path=run_dir/"validation_users.json"
    validation_users_path.write_text(json.dumps(internal_val_users),encoding="utf-8")
''')
# ensure every per-attempt output dir receives this stable roster before trainer starts
s=s.replace('''        cfg=td/"train.toml"
        # Trainer validation requires one batch per user.''','''        cfg=td/"train.toml"
        (td/"validation_users.json").write_text(json.dumps(internal_val_users),encoding="utf-8")
        # Trainer validation requires one batch per user.''')
p.write_text(s,encoding='utf-8',newline='\n')
print('copy explicit trainer validation roster into each trial directory')
