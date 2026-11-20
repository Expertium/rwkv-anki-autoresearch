from pathlib import Path
p=Path('scratchpad/adaptive_wd/adaptive_train.py');s=p.read_text(encoding='utf-8')
s=s.replace('''    if args.smoke_from_scratch:
        import sys
        sys.path.insert(0,str(ROOT))
''','''    if args.smoke_from_scratch:
''')
p.write_text(s,encoding='utf-8',newline='\n')
print('cleaned duplicated smoke path injection')
