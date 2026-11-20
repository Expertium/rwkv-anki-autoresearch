from pathlib import Path
p=Path(r'C:\Users\Andrew\PycharmProjects\PythonProject\pth to plain text.py')
s='''import argparse
from pathlib import Path
import sys

import torch


def main():
    parser = argparse.ArgumentParser(description="Write a PyTorch .pth checkpoint as a readable text dump.")
    parser.add_argument("input", nargs="?", default=r"C:\Users\Andrew\rwkv-anki-autoresearch\pretrain\rwkv\opt_qat45\rwkv_iter45_optim_496.pth", help="Input .pth checkpoint")
    parser.add_argument("output", nargs="?", default="model_weights.txt", help="Output text file")
    parser.add_argument("--max-elements", type=int, default=1000000, help="Maximum values printed per tensor")
    args = parser.parse_args()

    src = Path(args.input)
    if not src.is_file():
        parser.error(f"Checkpoint does not exist: {src}")

    # map_location accepts the actual CPU device, not the string "torch.device('cuda')".
    # CPU loading also works when the checkpoint was saved on a CUDA machine and this PC has no CUDA.
    try:
        data = torch.load(src, map_location=torch.device("cpu"), weights_only=False)
    except TypeError:  # Older PyTorch without weights_only argument.
        data = torch.load(src, map_location=torch.device("cpu"))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as f:
        f.write(f"Checkpoint: {src}\n")
        f.write(f"Top-level type: {type(data).__name__}\n\n")
        if not isinstance(data, dict):
            f.write(repr(data) + "\n")
        else:
            for key, value in data.items():
                f.write(f"Key: {key!r}\nType: {type(value).__name__}\n")
                if isinstance(value, torch.Tensor):
                    t = value.detach().cpu()
                    f.write(f"Shape: {tuple(t.shape)}\nDtype: {t.dtype}\nDevice: {t.device}\n")
                    n = t.numel()
                    f.write(f"Elements: {n}\n")
                    if n <= args.max_elements:
                        f.write("Values:\n")
                        f.write(str(t.tolist()) + "\n")
                    else:
                        f.write(f"Values omitted (limit {args.max_elements}); use --max-elements to raise the limit.\n")
                elif isinstance(value, dict):
                    f.write(f"Entries: {len(value)}\n")
                    for sk, sv in value.items():
                        if isinstance(sv, torch.Tensor):
                            st = sv.detach().cpu()
                            f.write(f"  {sk!r}: Tensor(shape={tuple(st.shape)}, dtype={st.dtype}, device=cpu)\n")
                        else:
                            f.write(f"  {sk!r}: {type(sv).__name__} = {repr(sv)[:500]}\n")
                else:
                    f.write(f"Value: {repr(value)[:2000]}\n")
                f.write("-" * 72 + "\n")
    print(f"Saved readable checkpoint dump to: {out}")
    print("Warning: text dumps can be very large and are for inspection, not reload.")


if __name__ == "__main__":
    main()
'''
p.write_text(s,encoding='utf-8',newline='\n')
print('rewrote',p)
