from pathlib import Path
p=Path('optimization/eval_sharded.py');s=p.read_text(encoding='utf-8')
s=s.replace('''        if line.split("#", 1)[0].split("=", 1)[0].strip() == "USERS_FILE":
            continue
''','')
needle='''    text.append(f'USERS_FILE = "{users_file}"')
'''
if needle not in s: raise SystemExit('append needle missing')
s=s.replace(needle,'''    text = [line for line in text if line.split("#", 1)[0].split("=", 1)[0].strip() != "USERS_FILE"]
    text.append(f'USERS_FILE = "{users_file}"')
''',1)
p.write_text(s,encoding='utf-8',newline='\n')
print('moved inherited USERS_FILE filtering after base lines are read')
