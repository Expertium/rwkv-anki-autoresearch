from pathlib import Path
import sys, json
sys.path.insert(0,str(Path.cwd()))
from optimization.eval_sharded import write_phase_toml
base=Path('scratchpad/adaptive_wd/smoke3/eval_smoke3_2155b14a_baseline.toml').read_text(encoding='utf-8')
out=write_phase_toml(base,'probe-test','a','i',[5002,5005],1)
print(out)
