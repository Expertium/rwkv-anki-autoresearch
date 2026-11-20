from pathlib import Path
p=Path('optimization/eval_sharded.py');s=p.read_text(encoding='utf-8')
s=s.replace('''        if line.split("#", 1)[0].split("=", 1)[0].strip() == "USERS_FILE":
            continue
''','')
s=s.replace('''    text.append(f'USERS_FILE = "{users_file}"')
''','''    # Strip any inherited USERS_FILE from the base template before appending this phase's roster.
    text = [line for line in text if line.split("#", 1)[0].split("=", 1)[0].strip() != "USERS_FILE"]
    text.append(f'USERS_FILE = "{users_file}"')
''',1)
p.write_text(s,encoding='utf-8',newline='\n')
