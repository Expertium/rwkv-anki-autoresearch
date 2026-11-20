from pathlib import Path
p=Path('scratchpad/adaptive_wd/adaptive_train.py');s=p.read_text(encoding='utf-8')
s=s.replace('''    for line in lines:
        raw = line.split("#", 1)[0]
        if "=" in raw:
            key = raw.split("=", 1)[0].strip()
            if key in values:
                out.append(f"{key} = {toml_repr(values[key])}")
                seen.add(key)
                continue
        out.append(line)
''','''    for line in lines:
        raw = line.split("#", 1)[0]
        if "=" in raw:
            key = raw.split("=", 1)[0].strip()
            if key in values:
                if key not in seen:
                    out.append(f"{key} = {toml_repr(values[key])}")
                    seen.add(key)
                # Drop every duplicate occurrence of keys we override.
                continue
        out.append(line)
''')
s=s.replace('''        "NUM_FETCH_PROCESSES":1,"MAX_TRAIN_GLOBAL_LEN":16384,"PEAK_LR":0.001,
''','''        "NUM_FETCH_PROCESSES":1,"MAX_TRAIN_GLOBAL_LEN":16384,"PEAK_LR":0.001,
        "VALIDATE_EVERY": max(1,int(0.02*steps)) if isinstance(steps,float) else 500,
''')
p.write_text(s,encoding='utf-8',newline='\n')
print('deduplicated overridden TOML keys; added smoke validation cadence')
