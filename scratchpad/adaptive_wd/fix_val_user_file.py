from pathlib import Path
p=Path('scratchpad/adaptive_wd/adaptive_train.py');s=p.read_text(encoding='utf-8')
s=s.replace('''    # Training validation has its own fixed 2-user roster. Evaluation metric users are separate.
    internal_val_users=[5001,5002]
    # Runtime output names are unique; eval_sharded refuses to overwrite any previous result.
''','''    # Training validation has its own fixed 2-user roster, disjoint from evaluation roster.
    internal_val_users=[u for u in range(5001,7501) if u not in set(users) and u not in excluded_users][:2]
    if len(internal_val_users)<2: raise RuntimeError("could not select two internal validation users")
    validation_users_path=run_dir/"validation_users.json"
    validation_users_path.write_text(json.dumps(internal_val_users),encoding="utf-8")
    # Runtime output names are unique; eval_sharded refuses to overwrite any previous result.
''')
p.write_text(s,encoding='utf-8',newline='\n')
print('write internal validation roster before training and make it disjoint')
