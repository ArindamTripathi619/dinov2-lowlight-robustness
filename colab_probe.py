import os, sys, torch

print("python:", sys.version.split()[0])
print("torch:", torch.__version__, "| cuda_build:", torch.version.cuda, "| cuda_avail:", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("gpu:", p.name, "| vram_gb:", round(p.total_memory / 1e9, 1))
print("cpu_count:", os.cpu_count())
print("disk_free_gb:", round(int(os.popen("df -BG /content | tail -1").read().split()[3].rstrip("G")), 1))
total_mem_kb = int(os.popen("grep MemTotal /proc/meminfo").read().split()[1])
print("ram_gb:", round(total_mem_kb / 1e6, 1))
