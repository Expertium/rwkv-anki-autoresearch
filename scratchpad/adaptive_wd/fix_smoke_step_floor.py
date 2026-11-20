from pathlib import Path
p=Path('scratchpad/adaptive_wd/adaptive_train.py');s=p.read_text(encoding='utf-8')
s=s.replace('''        "VALIDATE_USERS_FILE":str(Path(out).with_name("validation_users.json")),''','''        "VALIDATE_USERS_FILE":str(Path(out).with_name("validation_users.json")),''')
s=s.replace('''        # Trainer validation requires one batch per user. Keep the fixed primary 750-user
        # screen for the production evaluator; for periodic train-time validation use the
        # same stable two-user probe in both modes and do not use it as the promotion metric.
''','''        # Trainer validates only against a fixed tiny roster; primary promotion screen is external.
''')
# Make smoke execute enough steps to hit trainer save: ensure minimum 50 steps while remaining tiny.
s=s.replace('''        trained_steps=max(1,int(0.04*g)) if args.smoke else epoch_target*g
''','''        trained_steps=max(50,int(0.04*g)) if args.smoke else epoch_target*g
''')
p.write_text(s,encoding='utf-8',newline='\n')
print('smoke minimum 50 steps to exercise trainer validation/checkpoint')
