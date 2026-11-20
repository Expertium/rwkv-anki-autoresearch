from pathlib import Path
p=Path(r'C:\Users\Andrew\PycharmProjects\PythonProject\pth to plain text.py')
s=p.read_text(encoding='utf-8')
old='    out = Path(args.output)'
new=old+'
    if not out.is_absolute():
        out = Path(__file__).resolve().parent / out'
if old not in s: raise SystemExit('output path anchor missing')
s=s.replace(old,new,1)
p.write_text(s,encoding='utf-8',newline='\n')
print('patched relative output path')
