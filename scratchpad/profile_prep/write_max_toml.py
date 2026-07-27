"""Write the MAX-sweep toml with MAX_TRAIN_GLOBAL_LEN set from argv[1]."""
import sys, pathlib
mx = int(sys.argv[1])
src = pathlib.Path("scratchpad/profile_prep/profile_d80_ws.toml").read_text()
out = []
for line in src.splitlines():
    if line.startswith("MAX_TRAIN_GLOBAL_LEN"):
        line = f"MAX_TRAIN_GLOBAL_LEN = {mx}"
    out.append(line)
pathlib.Path("scratchpad/profile_prep/max_sweep_ws.toml").write_text("\n".join(out) + "\n")
print(f"MAX={mx}")
