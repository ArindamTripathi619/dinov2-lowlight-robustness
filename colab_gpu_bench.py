"""Quick T4 performance probe: verifies CUDA stack + throughput for LoRA workload."""
import subprocess, sys, time

# Ensure the deps the LoRA script needs are available
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "scikit-learn", "matplotlib"], check=False)

import torch

print("torch:", torch.__version__, "| cuda avail:", torch.cuda.is_available())
assert torch.cuda.is_available(), "No CUDA!"
dev = torch.device("cuda")
print("device:", torch.cuda.get_device_name(0))
props = torch.cuda.get_device_properties(0)
print(f"VRAM: {props.total_memory/1e9:.1f} GB | SMs: {props.multi_processor_count}")

# Benchmark 1: big matmul (proxy for attention GEMMs)
a = torch.randn(4096, 4096, device=dev)
torch.cuda.synchronize(); t0 = time.time()
for _ in range(20):
    b = a @ a
torch.cuda.synchronize()
mm_s = (time.time() - t0) / 20
print(f"4096x4096 matmul: {mm_s*1000:.1f} ms/op -> ~{2*4096**3/mm_s/1e12:.1f} TFLOPS")

# Benchmark 2: ViT-S-like fwd+bwd step (22M params, batch 64, 224/14=16 -> 256 tokens)
model = torch.nn.Sequential(
    torch.nn.Linear(384, 1408), torch.nn.GELU(), torch.nn.Linear(1408, 384)
).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
x = torch.randn(64, 384, device=dev)
torch.cuda.synchronize(); t0 = time.time()
for _ in range(20):
    opt.zero_grad()
    loss = model(x).pow(2).mean()
    loss.backward(); opt.step()
torch.cuda.synchronize()
step_s = (time.time() - t0) / 20
print(f"toy train step (bs64): {step_s*1000:.1f} ms -> ~{1000*step_s:.1f}s per 1000 steps")

# Package availability
import torchvision, sklearn, matplotlib, PIL
print("packages OK: torchvision", torchvision.__version__, "| sklearn", sklearn.__version__)

# Disk headroom
df = subprocess.run(["df", "-h", "/content"], capture_output=True, text=True).stdout
print("disk:\n" + df.strip().splitlines()[-1])
print("BENCH_OK")
