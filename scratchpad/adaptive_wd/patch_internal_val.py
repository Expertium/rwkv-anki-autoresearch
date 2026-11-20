from pathlib import Path
p=Path('scratchpad/adaptive_wd/adaptive_train.py')
s=p.read_text(encoding='utf-8')
# One-batch validation rosters need one explicit user per trainer VAL group. Pin both to two users.
s=s.replace('''        internal_val_users=users[:2]
''','''        internal_val_users=[5001,5002]
''')
# Only the final flag controls deployment evaluation; small screen is already the official eval target.
s=s.replace('''    ap.add_argument("--full-eval",action="store_true")''','''    ap.add_argument("--full-eval",action="store_true",help="after final QAT, run canonical full 5001-7500 VAL")''')
# Add final QAT epoch to smoke path later, after adaptive smoke validation, not now.
p.write_text(s,encoding='utf-8',newline='\n')
print('enforced one-batch trainer validation probe')
