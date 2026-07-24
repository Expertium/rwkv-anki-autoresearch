"""GRU dtype micro-bench (idle GPU): is bf16 hitting a slow fallback vs cuDNN fp32?"""
import time

import torch

B, T, C, H, L = 2, 8192, 128, 128, 4
x32 = torch.randn(B, T, C, device="cuda")

for dtype in (torch.float32, torch.bfloat16, torch.float16):
    try:
        rnn = torch.nn.GRU(C, H, L, batch_first=True).cuda().to(dtype)
        rnn.flatten_parameters()
        x = x32.to(dtype)
        for _ in range(2):  # warmup
            out, _ = rnn(x)
            out.sum().backward()
        torch.cuda.synchronize()
        t0 = time.time()
        n = 5
        for _ in range(n):
            rnn.zero_grad()
            out, _ = rnn(x)
            out.sum().backward()
        torch.cuda.synchronize()
        print(f"{dtype}: {(time.time() - t0) / n * 1000:.1f} ms per fwd+bwd "
              f"(B={B} T={T} C={C} H={H} L={L})")
    except Exception as e:
        print(f"{dtype}: FAILED - {type(e).__name__}: {str(e)[:120]}")
