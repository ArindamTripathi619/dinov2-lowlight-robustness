#!/usr/bin/env python3
"""Build the self-contained Kaggle corruption-grid kernel script.

Kaggle script kernels ship ONLY the single ``code_file``, so every runtime
file the kernel needs is embedded as a base64 blob inside one generated
script (``kaggle_corrgrid_tail.py``). The kernel driver extracts the files
to its cwd, mounts/extracts CIFAR-10 from the
``arindamtripathi/cifar10-python`` dataset mount, hard-aborts if CUDA is
missing, then runs ``grid_tail.sh`` (the corruption-grid chain).

Recovery from /tmp wipes (this laptop loses /tmp on reboot):

    .venv/bin/python3 scripts/build_kaggle_kernel.py [--pkg /tmp/kaggle_kernel]

All five packaged inputs resolve to repo-tracked files by default:

    utils.py, run_notebook2.py, run_lora_simple_colab.py -> repo root
    grid_tail.sh                                         -> scripts/kaggle/grid_tail.sh
    jpeg_cka_matrix.csv                                  -> colab_results/sessionE/nb2_jpeg/cka_matrix.csv

Per-file overrides (e.g. a rebuilt local CKA profile):

    ... --source jpeg_cka_matrix.csv=output/notebook2_jpeg/cka_matrix.csv

After a successful build, push with GPU enabled (both knobs matter):

    cd <pkg> && kaggle kernels push -p .
    # enable_gpu: true in kernel-metadata.json AND
    # --accelerator nvidiaTeslaT4 on the push/CLI side won before
"""
import argparse
import base64
import hashlib
import os
import sys

# name -> default repo-tracked source (paths relative to repo root)
DEFAULT_SOURCES = {
    "utils.py": "utils.py",
    "run_notebook2.py": "run_notebook2.py",
    "run_lora_simple_colab.py": "run_lora_simple_colab.py",
    "grid_tail.sh": "scripts/kaggle/grid_tail.sh",
    "jpeg_cka_matrix.csv": "colab_results/sessionE/nb2_jpeg/cka_matrix.csv",
}

# Literal template (not json.dump) so regenerated metadata is byte-identical
# to the metadata used for the successful v6 push.
METADATA_JSON = '''{
  "id": "arindamtripathi/corrgrid-tail",
  "title": "corrgrid-tail",
  "code_file": "kaggle_corrgrid_tail.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "dataset_sources": ["arindamtripathi/cifar10-python"],
  "competition_sources": [],
  "kernel_sources": []
}
'''

# --- Track 2 v8 variant: ViT-B/14 drift-profile kernel (separate slug so the
# grid kernel's provenance stays intact). Sources = the profiling harness.
PROFILE_SOURCES = {
    "utils.py": "utils.py",
    "stats_tools.py": "stats_tools.py",
    "run_drift_proxy.py": "run_drift_proxy.py",
    "run_drift_profile.py": "run_drift_profile.py",
    "vitb_profile.sh": "scripts/kaggle/vitb_profile.sh",
}

PROFILE_METADATA_JSON = '''{
  "id": "arindamtripathi/vitb-drift-profile",
  "title": "vitb-drift-profile",
  "code_file": "kaggle_vitb_profile.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "dataset_sources": ["arindamtripathi/cifar10-python"],
  "competition_sources": [],
  "kernel_sources": []
}
'''

PROFILE_TEMPLATE = '''"""Self-contained ViT-B drift-profile kernel (all package files embedded)."""
import base64, os, shutil, subprocess, zipfile

PKG_FILES = {
__CHUNKS__
}

CWD = os.getcwd()
print("CWD:", CWD)
for name, b64 in PKG_FILES.items():
    with open(os.path.join(CWD, name), "wb") as fh:
        fh.write(base64.b64decode(b64))
    print("staged", name)

# --- CIFAR: dataset mount (bonus) or torchvision fallback ---
hits = []
for root, dirs, files in os.walk("/kaggle/input"):
    for f in files:
        if f.lower().endswith(".zip") and "cifar" in f.lower():
            hits.append(("zip", os.path.join(root, f)))
    for d in dirs:
        if "cifar" in d.lower():
            hits.append(("dir", os.path.join(root, d)))
print("mount hits:", hits)

os.makedirs("data", exist_ok=True)
if not os.path.isdir("data/cifar-10-batches-py") and hits:
    kind, path = hits[0]
    print("using", kind, path)
    try:
        if kind == "zip":
            with zipfile.ZipFile(path) as z:
                top = sorted({n.split("/")[0] for n in z.namelist()})
                print("zip top-level:", top)
                if len(top) == 1 and "cifar" in top[0].lower():
                    z.extractall("data")
                else:
                    z.extractall("data/cifar-10-batches-py")
        else:
            shutil.copytree(path, "data/cifar-10-batches-py")
    except Exception as e:
        print("mount extraction failed:", e, "- falling back to torchvision download")
if not os.path.isdir("data/cifar-10-batches-py") and os.path.isfile("data/data_batch_1"):
    os.makedirs("data/cifar-10-batches-py", exist_ok=True)
    for f in os.listdir("data"):
        p = os.path.join("data", f)
        if os.path.isfile(p) and not f.endswith(".zip"):
            shutil.move(p, "data/cifar-10-batches-py/")
if os.path.isdir("data/cifar-10-batches-py") and os.path.isfile("data/cifar-10-batches-py/data_batch_1"):
    print("data ready:", sorted(os.listdir("data/cifar-10-batches-py"))[:3], "...")
else:
    print("data dir empty - torchvision download=True inside the runner will fetch it")

# --- GPU sanity ---
import torch
print("cuda:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("no GPU attached - aborting before wasting the run")
print("gpu:", torch.cuda.get_device_name(0))

# --- ViT-B/14 drift-profile sweep (cwd = /kaggle/working so outputs are captured) ---
r = subprocess.run(["bash", "vitb_profile.sh"])
csv_ok = os.path.isfile("output/vitb_profile/proxy_profiles.csv")
print("profile exit:", r.returncode, "| csv_ok:", csv_ok)
print("PROFILE_COMPLETE" if (r.returncode == 0 and csv_ok) else "PROFILE_FAILED")
'''

# --- Track 2 v9 variant: ViT-B LoRA arms kernel (separate slug so the
# profile kernel's provenance stays intact). Sources = the arm runner, the
# re-indexed drift consumer CSV, and the 3-arm tail.
V9_SOURCES = {
    "run_lora_simple_colab.py": "run_lora_simple_colab.py",
    "drift_sev5_low_light.csv": "output/vitb_profile/v9_lora/drift_sev5_low_light.csv",
    "v9_lora.sh": "scripts/kaggle/v9_lora.sh",
}

V9_METADATA_JSON = '''{
  "id": "arindamtripathi/vitb-lora-v9",
  "title": "vitb-lora-v9",
  "code_file": "kaggle_vitb_lora_v9.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "dataset_sources": ["arindamtripathi/cifar10-python"],
  "competition_sources": [],
  "kernel_sources": []
}
'''

V9_TEMPLATE = '''"""Self-contained ViT-B LoRA v9 kernel (all package files embedded)."""
import base64, os, shutil, subprocess, zipfile

PKG_FILES = {
__CHUNKS__
}

CWD = os.getcwd()
print("CWD:", CWD)
for name, b64 in PKG_FILES.items():
    with open(os.path.join(CWD, name), "wb") as fh:
        fh.write(base64.b64decode(b64))
    print("staged", name)

# --- CIFAR: dataset mount (bonus) or torchvision fallback ---
hits = []
for root, dirs, files in os.walk("/kaggle/input"):
    for f in files:
        if f.lower().endswith(".zip") and "cifar" in f.lower():
            hits.append(("zip", os.path.join(root, f)))
    for d in dirs:
        if "cifar" in d.lower():
            hits.append(("dir", os.path.join(root, d)))
print("mount hits:", hits)

os.makedirs("data", exist_ok=True)
if not os.path.isdir("data/cifar-10-batches-py") and hits:
    kind, path = hits[0]
    print("using", kind, path)
    try:
        if kind == "zip":
            with zipfile.ZipFile(path) as z:
                top = sorted({n.split("/")[0] for n in z.namelist()})
                print("zip top-level:", top)
                if len(top) == 1 and "cifar" in top[0].lower():
                    z.extractall("data")
                else:
                    z.extractall("data/cifar-10-batches-py")
        else:
            shutil.copytree(path, "data/cifar-10-batches-py")
    except Exception as e:
        print("mount extraction failed:", e, "- falling back to torchvision download")
if not os.path.isdir("data/cifar-10-batches-py") and os.path.isfile("data/data_batch_1"):
    os.makedirs("data/cifar-10-batches-py", exist_ok=True)
    for f in os.listdir("data"):
        p = os.path.join("data", f)
        if os.path.isfile(p) and not f.endswith(".zip"):
            shutil.move(p, "data/cifar-10-batches-py/")
if os.path.isdir("data/cifar-10-batches-py") and os.path.isfile("data/cifar-10-batches-py/data_batch_1"):
    print("data ready:", sorted(os.listdir("data/cifar-10-batches-py"))[:3], "...")
else:
    print("data dir empty - torchvision download=True inside the runner will fetch it")

# --- GPU sanity ---
import torch
print("cuda:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("no GPU attached - aborting before wasting the run")
print("gpu:", torch.cuda.get_device_name(0))

r = subprocess.run(["bash", "v9_lora.sh"])
outputs = [f"output/v9_lora/seed{s}/{arm}/lora_adapters.pt"
           for s in (43, 44) for arm in ("uniform", "drift", "late")]
present = {p: os.path.isfile(p) for p in outputs}
print("v9 exit:", r.returncode, "| outputs:", present)
print("LORA_V9_SEEDS_COMPLETE" if (r.returncode == 0 and all(present.values())) else "LORA_V9_SEEDS_FAILED")
'''

# --- Track 2 v9 ablation variant: ViT-B allocation-rule ablations kernel
# (separate slug so the arms kernel's provenance stays intact). Sources = the
# arm runner (with the §1.4 ablation layers), the re-indexed drift consumer
# CSV, and the 3-arm × 3-seed ablation tail.
V9ABL_SOURCES = {
    "run_lora_simple_colab.py": "run_lora_simple_colab.py",
    "drift_sev5_low_light.csv": "output/vitb_profile/v9_lora/drift_sev5_low_light.csv",
    "v9_ablation.sh": "scripts/kaggle/v9_ablation.sh",
}

V9ABL_METADATA_JSON = '''{
  "id": "arindamtripathi/vitb-lora-abl",
  "title": "vitb-lora-abl",
  "code_file": "kaggle_vitb_lora_abl.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "dataset_sources": ["arindamtripathi/cifar10-python"],
  "competition_sources": [],
  "kernel_sources": []
}
'''

V9ABL_TEMPLATE = '''"""Self-contained ViT-B LoRA ablation kernel (all package files embedded)."""
import base64, os, shutil, subprocess, zipfile

PKG_FILES = {
__CHUNKS__
}

CWD = os.getcwd()
print("CWD:", CWD)
for name, b64 in PKG_FILES.items():
    with open(os.path.join(CWD, name), "wb") as fh:
        fh.write(base64.b64decode(b64))
    print("staged", name)

# --- CIFAR: dataset mount (bonus) or torchvision fallback ---
hits = []
for root, dirs, files in os.walk("/kaggle/input"):
    for f in files:
        if f.lower().endswith(".zip") and "cifar" in f.lower():
            hits.append(("zip", os.path.join(root, f)))
    for d in dirs:
        if "cifar" in d.lower():
            hits.append(("dir", os.path.join(root, d)))
print("mount hits:", hits)

os.makedirs("data", exist_ok=True)
if not os.path.isdir("data/cifar-10-batches-py") and hits:
    kind, path = hits[0]
    print("using", kind, path)
    try:
        if kind == "zip":
            with zipfile.ZipFile(path) as z:
                top = sorted({n.split("/")[0] for n in z.namelist()})
                print("zip top-level:", top)
                if len(top) == 1 and "cifar" in top[0].lower():
                    z.extractall("data")
                else:
                    z.extractall("data/cifar-10-batches-py")
        else:
            shutil.copytree(path, "data/cifar-10-batches-py")
    except Exception as e:
        print("mount extraction failed:", e, "- falling back to torchvision download")
if not os.path.isdir("data/cifar-10-batches-py") and os.path.isfile("data/data_batch_1"):
    os.makedirs("data/cifar-10-batches-py", exist_ok=True)
    for f in os.listdir("data"):
        p = os.path.join("data", f)
        if os.path.isfile(p) and not f.endswith(".zip"):
            shutil.move(p, "data/cifar-10-batches-py/")
if os.path.isdir("data/cifar-10-batches-py") and os.path.isfile("data/cifar-10-batches-py/data_batch_1"):
    print("data ready:", sorted(os.listdir("data/cifar-10-batches-py"))[:3], "...")
else:
    print("data dir empty - torchvision download=True inside the runner will fetch it")

# --- GPU sanity ---
import torch
print("cuda:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("no GPU attached - aborting before wasting the run")
print("gpu:", torch.cuda.get_device_name(0))

r = subprocess.run(["bash", "v9_ablation.sh"])
outputs = [f"output/v9_ablation/seed{s}/{arm}/lora_adapters.pt"
           for s in (42, 43, 44) for arm in ("btau", "topk", "invbeta")]
present = {p: os.path.isfile(p) for p in outputs}
print("v9-abl exit:", r.returncode, "| outputs:", present)
print("LORA_V9_ABL_COMPLETE" if (r.returncode == 0 and all(present.values())) else "LORA_V9_ABL_FAILED")
'''

# Kernel driver template. Keep byte-compatible with the v5 build
# (/tmp/kaggle_kernel/kaggle_corrgrid_tail.py from the successful v6 run).
TEMPLATE = '''"""Self-contained corruption-grid kernel (all package files embedded)."""
import base64, os, shutil, subprocess, zipfile

PKG_FILES = {
__CHUNKS__
}

CWD = os.getcwd()
print("CWD:", CWD)
for name, b64 in PKG_FILES.items():
    with open(os.path.join(CWD, name), "wb") as fh:
        fh.write(base64.b64decode(b64))
    print("staged", name)

# --- CIFAR: dataset mount (bonus) or torchvision fallback ---
hits = []
for root, dirs, files in os.walk("/kaggle/input"):
    for f in files:
        if f.lower().endswith(".zip") and "cifar" in f.lower():
            hits.append(("zip", os.path.join(root, f)))
    for d in dirs:
        if "cifar" in d.lower():
            hits.append(("dir", os.path.join(root, d)))
print("mount hits:", hits)

os.makedirs("data", exist_ok=True)
if not os.path.isdir("data/cifar-10-batches-py") and hits:
    kind, path = hits[0]
    print("using", kind, path)
    try:
        if kind == "zip":
            with zipfile.ZipFile(path) as z:
                top = sorted({n.split("/")[0] for n in z.namelist()})
                print("zip top-level:", top)
                if len(top) == 1 and "cifar" in top[0].lower():
                    z.extractall("data")
                else:
                    z.extractall("data/cifar-10-batches-py")
        else:
            shutil.copytree(path, "data/cifar-10-batches-py")
    except Exception as e:
        print("mount extraction failed:", e, "- falling back to torchvision download")
if not os.path.isdir("data/cifar-10-batches-py") and os.path.isfile("data/data_batch_1"):
    os.makedirs("data/cifar-10-batches-py", exist_ok=True)
    for f in os.listdir("data"):
        p = os.path.join("data", f)
        if os.path.isfile(p) and not f.endswith(".zip"):
            shutil.move(p, "data/cifar-10-batches-py/")
if os.path.isdir("data/cifar-10-batches-py") and os.path.isfile("data/cifar-10-batches-py/data_batch_1"):
    print("data ready:", sorted(os.listdir("data/cifar-10-batches-py"))[:3], "...")
else:
    print("data dir empty - torchvision download=True inside the runners will fetch it")

# --- GPU sanity ---
import torch
print("cuda:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("no GPU attached - aborting before wasting the run")
print("gpu:", torch.cuda.get_device_name(0))

# --- jpeg CKA profile (salvaged from sessionE) ---
os.makedirs("output/nb2_jpeg", exist_ok=True)
shutil.copy("jpeg_cka_matrix.csv", "output/nb2_jpeg/cka_matrix.csv")
print("jpeg cka profile staged")

# --- grid tail (cwd = /kaggle/working so outputs are captured) ---
r = subprocess.run(["bash", "grid_tail.sh"])
print("grid exit:", r.returncode)
print("GRID_COMPLETE" if r.returncode == 0 else "GRID_FAILED")
'''


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--variant", default="grid", choices=["grid", "vitb_profile", "vitb_lora_v9", "vitb_lora_abl"],
                    help="grid = corruption-grid kernel (corrgrid-tail, byte-identical v6); "
                         "vitb_profile = Track 2 v8 ViT-B drift-profile kernel; "
                         "vitb_lora_v9 = Track 2 v9 ViT-B LoRA arms kernel; "
                         "vitb_lora_abl = Track 2 v9 allocation-rule ablations kernel")
    ap.add_argument("--pkg", default="/tmp/kaggle_kernel",
                    help="staging dir receiving the kernel script (default: /tmp/kaggle_kernel)")
    ap.add_argument("--source", action="append", default=[], metavar="NAME=PATH",
                    help="override the packaged source for file NAME (repeatable)")
    return ap.parse_args()


def main():
    args = parse_args()
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if args.variant == "vitb_profile":
        base_sources, template = PROFILE_SOURCES, PROFILE_TEMPLATE
        metadata, out_name = PROFILE_METADATA_JSON, "kaggle_vitb_profile.py"
    elif args.variant == "vitb_lora_v9":
        base_sources, template = V9_SOURCES, V9_TEMPLATE
        metadata, out_name = V9_METADATA_JSON, "kaggle_vitb_lora_v9.py"
    elif args.variant == "vitb_lora_abl":
        base_sources, template = V9ABL_SOURCES, V9ABL_TEMPLATE
        metadata, out_name = V9ABL_METADATA_JSON, "kaggle_vitb_lora_abl.py"
    else:
        base_sources, template = DEFAULT_SOURCES, TEMPLATE
        metadata, out_name = METADATA_JSON, "kaggle_corrgrid_tail.py"

    sources = dict(base_sources)
    for pair in args.source:
        name, _, path = pair.partition("=")
        if name not in sources or not path:
            print(f"error: --source expects NAME=PATH with NAME in {sorted(sources)}; got {pair!r}")
            return 1
        sources[name] = path

    blobs = []
    for name in base_sources:
        path = sources[name]
        abspath = path if os.path.isabs(path) else os.path.join(repo_root, path)
        if not os.path.isfile(abspath):
            print(f"error: source for {name} not found: {abspath}")
            return 1
        data = open(abspath, "rb").read()
        b64 = base64.b64encode(data).decode()
        blobs.append((name, b64))
        md5 = hashlib.md5(data).hexdigest()
        print(f"embedded {name}: {len(data)} bytes -> {len(b64)} b64 (md5 {md5}, src {path})")

    chunks = []
    for name, b64 in blobs:
        lines = [b64[i:i + 96] for i in range(0, len(b64), 96)]
        chunks.append(f'    "{name}": (\n' + "\n".join(f'        "{l}"' for l in lines) + "\n    ),")

    script = template.replace("__CHUNKS__", "\n".join(chunks))

    os.makedirs(args.pkg, exist_ok=True)
    out = os.path.join(args.pkg, out_name)
    with open(out, "w") as f:
        f.write(script)
    print(f"wrote {out}: {len(script)} bytes")

    meta_path = os.path.join(args.pkg, "kernel-metadata.json")
    if not os.path.isfile(meta_path):
        with open(meta_path, "w") as f:
            f.write(metadata)
        print(f"wrote {meta_path} (was missing)")
    else:
        print(f"kept existing {meta_path}")

    print("\npush with GPU (both knobs matter):")
    print(f"  cd {args.pkg} && kaggle kernels push -p .")
    print("  (enable_gpu: true in metadata AND --accelerator nvidiaTeslaT4 on push)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
