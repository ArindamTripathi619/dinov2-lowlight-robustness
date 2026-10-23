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
import json
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
    "agreement_report.py": "agreement_report.py",
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

# --- Track 3 variant: cross-family drift atlas kernel (separate slug so the
# profile kernel's provenance stays intact). Sources = the profiling harness
# + the atlas tail. Dataset: ImageNet-val subset of record + CIFAR legacy.
ATLAS_SOURCES = {
    "utils.py": "utils.py",
    "stats_tools.py": "stats_tools.py",
    "agreement_report.py": "agreement_report.py",
    "run_drift_proxy.py": "run_drift_proxy.py",
    "run_drift_profile.py": "run_drift_profile.py",
    "atlas.sh": "scripts/kaggle/atlas.sh",
}

ATLAS_METADATA_JSON = '''{
  "id": "arindamtripathi/drift-atlas",
  "title": "drift-atlas",
  "code_file": "kaggle_drift_atlas.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "dataset_sources": ["sautkin/imagenet1kvalid", "arindamtripathi/cifar10-python"],
  "competition_sources": [],
  "kernel_sources": []
}
'''

# --- Track 3 v4: byte-identical protocol, FRESH slug. The v3 drift-atlas
# session stalled past Kaggle's 12h cap with no recoverable logs; a new id
# starts a clean session instead of colliding with the one still reporting
# RUNNING, and keeps the stalled kernel's provenance intact for diagnosis.
ATLAS_V4_METADATA_JSON = '''{
  "id": "arindamtripathi/drift-atlas-v4",
  "title": "drift-atlas-v4",
  "code_file": "kaggle_drift_atlas_v4.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "dataset_sources": ["sautkin/imagenet1kvalid", "arindamtripathi/cifar10-python"],
  "competition_sources": [],
  "kernel_sources": []
}
'''

ATLAS_TEMPLATE = '''"""Self-contained cross-family drift-atlas kernel (all package files embedded)."""
import base64, os, shutil, subprocess, zipfile
os.environ.setdefault("PYTHONUNBUFFERED", "1")  # live ATLAS_MARK lines in the Kaggle log

PKG_FILES = {
__CHUNKS__
}

CWD = os.getcwd()
print("CWD:", CWD)
for name, b64 in PKG_FILES.items():
    with open(os.path.join(CWD, name), "wb") as fh:
        fh.write(base64.b64decode(b64))
    print("staged", name)

# --- CIFAR: dataset mount (consistency leg) ---
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

# --- ImageNet-val sanity (atlas of record); layouts differ between
# /kaggle/input/<slug> and /kaggle/input/datasets/<owner>/<slug> ---
im_roots = []
for root, dirs, files in os.walk("/kaggle/input"):
    if "imagenet1kvalid" in dirs:
        im_roots.append(os.path.join(root, "imagenet1kvalid"))
        dirs.remove("imagenet1kvalid")  # don't descend into the 2GB tree
print("imagenet mounts:", im_roots)
if not im_roots:
    raise RuntimeError("imagenet1kvalid mount missing - aborting before wasting the run")

# --- GPU sanity ---
import torch
print("cuda:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("no GPU attached - aborting before wasting the run")
print("gpu:", torch.cuda.get_device_name(0))

r = subprocess.run(["bash", "atlas.sh"])
models = ["resnet50", "convnext_tiny", "vit_small_patch16_224",
          "swin_tiny_patch4_window7_224", "deit_small_patch16_224", "dinov2_vitb14"]
present = {m: os.path.isfile(f"output/profiles/{m}/proxy_profiles.csv") for m in models}
consistency = {m: os.path.isfile(f"output/profiles/cifar10_{m}/proxy_profiles.csv")
               for m in ("resnet50", "convnext_tiny")}
print("atlas exit:", r.returncode, "| profiles:", present, "| cifar leg:", consistency)
final = "ATLAS_COMPLETE" if (r.returncode == 0 and all(present.values())
                             and all(consistency.values())) else "ATLAS_FAILED"
print(final)
try:
    import datetime as _dt
    with open("output/atlas_marks.log", "a") as _fh:
        _fh.write("ATLAS_MARK ts=%s %s\\n"
                  % (_dt.datetime.now().astimezone().isoformat(timespec="seconds"), final))
except Exception as _e:
    print("marks append failed:", _e)
'''

# --- Track 4 variant: SSR S1 smoke kernel (fresh slug; consumes the harvested
# Track-3 profile CSVs as the drift-ranked selection input, SSR_PLAN §8).
SSR_SOURCES = {
    "utils.py": "utils.py",
    "stats_tools.py": "stats_tools.py",
    "agreement_report.py": "agreement_report.py",
    "run_drift_proxy.py": "run_drift_proxy.py",
    "run_drift_profile.py": "run_drift_profile.py",
    "run_ssr.py": "run_ssr.py",
    "src/__init__.py": "src/__init__.py",
    "src/ssr.py": "src/ssr.py",
    "src/aqr.py": "src/aqr.py",
    "src/corruptions.py": "src/corruptions.py",
    "tools/__init__.py": "tools/__init__.py",
    "tools/build_atlas.py": "tools/build_atlas.py",
    "ssr.sh": "scripts/kaggle/ssr.sh",
    "output/profiles/resnet50/drift_profile_sev5_low_light.csv":
        "output/profiles/resnet50/drift_profile_sev5_low_light.csv",
    "output/profiles/resnet50/drift_profile_sev5_jpeg.csv":
        "output/profiles/resnet50/drift_profile_sev5_jpeg.csv",
    "output/profiles/vit_small_patch16_224/drift_profile_sev5_low_light.csv":
        "output/profiles/vit_small_patch16_224/drift_profile_sev5_low_light.csv",
    "output/profiles/vit_small_patch16_224/drift_profile_sev5_jpeg.csv":
        "output/profiles/vit_small_patch16_224/drift_profile_sev5_jpeg.csv",
}

SSR_METADATA_JSON = '''{
  "id": "arindamtripathi/ssr-s1",
  "title": "ssr-s1",
  "code_file": "kaggle_ssr_s1.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "dataset_sources": ["sautkin/imagenet1kvalid"],
  "competition_sources": [],
  "kernel_sources": []
}
'''

SSR_TEMPLATE = '''"""Self-contained SSR S1 smoke kernel (all package files embedded)."""
import base64, os, subprocess
os.environ.setdefault("PYTHONUNBUFFERED", "1")  # live SSR_MARK lines in the Kaggle log

PKG_FILES = {
__CHUNKS__
}

CWD = os.getcwd()
print("CWD:", CWD)
for name, b64 in PKG_FILES.items():
    path = os.path.join(CWD, name)
    os.makedirs(os.path.dirname(path) or CWD, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(base64.b64decode(b64))
    print("staged", name)

# --- ImageNet-val sanity (SSR eval images of record; layouts differ between
# /kaggle/input/<slug> and /kaggle/input/datasets/<owner>/<slug>) ---
im_roots = []
for root, dirs, files in os.walk("/kaggle/input"):
    if "imagenet1kvalid" in dirs:
        im_roots.append(os.path.join(root, "imagenet1kvalid"))
        dirs.remove("imagenet1kvalid")  # don't descend into the 2GB tree
print("imagenet mounts:", im_roots)
if not im_roots:
    raise RuntimeError("imagenet1kvalid mount missing - aborting before wasting the run")

# --- GPU sanity ---
import torch
print("cuda:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("no GPU attached - aborting before wasting the run")
print("gpu:", torch.cuda.get_device_name(0))

r = subprocess.run(["bash", "ssr.sh"])
invocations = [f"{slug.replace(':', '_')}_{corr}"
               for slug in ("resnet50", "timm:vit_small_patch16_224")
               for corr in ("low_light", "jpeg")]
present = {i: os.path.isfile(f"output/ssr/{i}/ssr_results.csv") for i in invocations}
ident = {}
import json as _json
for i in invocations:
    meta = f"output/ssr/{i}/ssr_meta.json"
    if os.path.isfile(meta):
        try:
            ident[i] = bool(_json.load(open(meta))["identity"]["bit_identical"])
        except Exception as e:
            ident[i] = f"unreadable: {e}"
print("ssr exit:", r.returncode, "| results:", present, "| identity:", ident)
final = ("SSR_S1_COMPLETE" if (r.returncode == 0 and all(present.values())
                              and all(v is True for v in ident.values()))
         else "SSR_S1_FAILED")
print(final)
try:
    import datetime as _dt
    with open("output/ssr_marks.log", "a") as _fh:
        _fh.write("SSR_MARK ts=%s %s\\n"
                  % (_dt.datetime.now().astimezone().isoformat(timespec="seconds"), final))
except Exception as _e:
    print("marks append failed:", _e)
'''

# --- Track 4 variant: SSR S2 attachment-point probe kernel (fresh slug;
# same runner + harvested profile CSVs as S1, different driver:
# scripts/kaggle/ssr_s2.sh, scope legs under output/ssr_s2/).
SSR_S2_SOURCES = dict(SSR_SOURCES)
SSR_S2_SOURCES["ssr.sh"] = "scripts/kaggle/ssr_s2.sh"

SSR_S2_METADATA_JSON = '''{
  "id": "arindamtripathi/ssr-s2",
  "title": "ssr-s2",
  "code_file": "kaggle_ssr_s2.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "dataset_sources": ["sautkin/imagenet1kvalid"],
  "competition_sources": [],
  "kernel_sources": []
}
'''

SSR_S2_TEMPLATE = '''"""Self-contained SSR S2 attachment-probe kernel (all package files embedded)."""
import base64, os, subprocess
os.environ.setdefault("PYTHONUNBUFFERED", "1")  # live SSR_MARK lines in the Kaggle log

PKG_FILES = {
__CHUNKS__
}

CWD = os.getcwd()
print("CWD:", CWD)
for name, b64 in PKG_FILES.items():
    path = os.path.join(CWD, name)
    os.makedirs(os.path.dirname(path) or CWD, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(base64.b64decode(b64))
    print("staged", name)

# --- ImageNet-val sanity (SSR eval images of record; layouts differ between
# /kaggle/input/<slug> and /kaggle/input/datasets/<owner>/<slug>) ---
im_roots = []
for root, dirs, files in os.walk("/kaggle/input"):
    if "imagenet1kvalid" in dirs:
        im_roots.append(os.path.join(root, "imagenet1kvalid"))
        dirs.remove("imagenet1kvalid")  # don't descend into the 2GB tree
print("imagenet mounts:", im_roots)
if not im_roots:
    raise RuntimeError("imagenet1kvalid mount missing - aborting before wasting the run")

# --- GPU sanity ---
import torch
print("cuda:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("no GPU attached - aborting before wasting the run")
print("gpu:", torch.cuda.get_device_name(0))

r = subprocess.run(["bash", "ssr.sh"])
invocations = [("resnet50", "profiled"),
               ("resnet50", "sublayer"),
               ("resnet50", "stream"),
               ("timm:vit_small_patch16_224", "profiled"),
               ("timm:vit_small_patch16_224", "sublayer"),
               ("timm:vit_small_patch16_224", "stream"),
               ("timm:vit_small_patch16_224", "norm")]
dirs = ["%s_%s_low_light" % (slug.replace(":", "_"), scope)
        for slug, scope in invocations]
present = {d: os.path.isfile("output/ssr_s2/%s/ssr_results.csv" % d) for d in dirs}
ident = {}
import json as _json
for d in dirs:
    meta = "output/ssr_s2/%s/ssr_meta.json" % d
    if os.path.isfile(meta):
        try:
            ident[d] = bool(_json.load(open(meta))["identity"]["bit_identical"])
        except Exception as e:
            ident[d] = "unreadable: %s" % e
print("ssr exit:", r.returncode, "| results:", present, "| identity:", ident)
final = ("SSR_S2_COMPLETE" if (r.returncode == 0 and all(present.values())
                              and all(v is True for v in ident.values()))
         else "SSR_S2_FAILED")
print(final)
try:
    import datetime as _dt
    with open("output/ssr_s2_marks.log", "a") as _fh:
        _fh.write("SSR_MARK ts=%s %s\\n"
                  % (_dt.datetime.now().astimezone().isoformat(timespec="seconds"), final))
except Exception as _e:
    print("marks append failed:", _e)
'''

# --- Track 4 variant: SSR S3 replication kernel (fresh slug; same runner +
# harvested profile CSVs, driver scripts/kaggle/ssr_s3.sh: stream/sublayer
# across seeds {42,43,44} x {low_light, jpeg} + k-matched controls).
SSR_S3_SOURCES = dict(SSR_SOURCES)
SSR_S3_SOURCES["ssr.sh"] = "scripts/kaggle/ssr_s3.sh"

SSR_S3_METADATA_JSON = '''{
  "id": "arindamtripathi/ssr-s3",
  "title": "ssr-s3",
  "code_file": "kaggle_ssr_s3.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "dataset_sources": ["sautkin/imagenet1kvalid"],
  "competition_sources": [],
  "kernel_sources": []
}
'''

SSR_S3_TEMPLATE = '''"""Self-contained SSR S3 replication kernel (all package files embedded)."""
import base64, os, subprocess
os.environ.setdefault("PYTHONUNBUFFERED", "1")  # live SSR_MARK lines in the Kaggle log

PKG_FILES = {
__CHUNKS__
}

CWD = os.getcwd()
print("CWD:", CWD)
for name, b64 in PKG_FILES.items():
    path = os.path.join(CWD, name)
    os.makedirs(os.path.dirname(path) or CWD, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(base64.b64decode(b64))
    print("staged", name)

# --- ImageNet-val sanity (SSR eval images of record; layouts differ between
# /kaggle/input/<slug> and /kaggle/input/datasets/<owner>/<slug>) ---
im_roots = []
for root, dirs, files in os.walk("/kaggle/input"):
    if "imagenet1kvalid" in dirs:
        im_roots.append(os.path.join(root, "imagenet1kvalid"))
        dirs.remove("imagenet1kvalid")  # don't descend into the 2GB tree
print("imagenet mounts:", im_roots)
if not im_roots:
    raise RuntimeError("imagenet1kvalid mount missing - aborting before wasting the run")

# --- GPU sanity ---
import torch
print("cuda:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("no GPU attached - aborting before wasting the run")
print("gpu:", torch.cuda.get_device_name(0))

r = subprocess.run(["bash", "ssr.sh"])
invocations = [("resnet50", "stream", "low_light", 42, 25),
               ("resnet50", "sublayer", "low_light", 42, 25),
               ("timm:vit_small_patch16_224", "stream", "low_light", 42, 25),
               ("timm:vit_small_patch16_224", "sublayer", "low_light", 42, 25),
               ("resnet50", "stream", "jpeg", 42, 25),
               ("resnet50", "sublayer", "jpeg", 42, 25),
               ("timm:vit_small_patch16_224", "stream", "jpeg", 42, 25),
               ("timm:vit_small_patch16_224", "sublayer", "jpeg", 42, 25),
               ("resnet50", "stream", "low_light", 43, 25),
               ("resnet50", "sublayer", "low_light", 43, 25),
               ("timm:vit_small_patch16_224", "stream", "low_light", 43, 25),
               ("timm:vit_small_patch16_224", "sublayer", "low_light", 43, 25),
               ("resnet50", "stream", "jpeg", 43, 25),
               ("resnet50", "sublayer", "jpeg", 43, 25),
               ("timm:vit_small_patch16_224", "stream", "jpeg", 43, 25),
               ("timm:vit_small_patch16_224", "sublayer", "jpeg", 43, 25),
               ("resnet50", "stream", "low_light", 44, 25),
               ("resnet50", "sublayer", "low_light", 44, 25),
               ("timm:vit_small_patch16_224", "stream", "low_light", 44, 25),
               ("timm:vit_small_patch16_224", "sublayer", "low_light", 44, 25),
               ("resnet50", "stream", "jpeg", 44, 25),
               ("resnet50", "sublayer", "jpeg", 44, 25),
               ("timm:vit_small_patch16_224", "stream", "jpeg", 44, 25),
               ("timm:vit_small_patch16_224", "sublayer", "jpeg", 44, 25),
               ("resnet50", "stream", "low_light", 42, 1),
               ("resnet50", "sublayer", "low_light", 42, 1),
               ("timm:vit_small_patch16_224", "stream", "low_light", 42, 3),
               ("timm:vit_small_patch16_224", "sublayer", "low_light", 42, 3),
               ("resnet50", "stream", "jpeg", 42, 1),
               ("resnet50", "sublayer", "jpeg", 42, 1),
               ("timm:vit_small_patch16_224", "stream", "jpeg", 42, 3),
               ("timm:vit_small_patch16_224", "sublayer", "jpeg", 42, 3)]
dirs = ["%s_%s_k%d_%s_s%d" % (slug.replace(":", "_"), scope, k, corr, seed)
        for slug, scope, corr, seed, k in invocations]
present = {d: os.path.isfile("output/ssr_s3/%s/ssr_results.csv" % d) for d in dirs}
ident = {}
import json as _json
for d in dirs:
    meta = "output/ssr_s3/%s/ssr_meta.json" % d
    if os.path.isfile(meta):
        try:
            ident[d] = bool(_json.load(open(meta))["identity"]["bit_identical"])
        except Exception as e:
            ident[d] = "unreadable: %s" % e
print("ssr exit:", r.returncode, "| results:", present, "| identity:", ident)
final = ("SSR_S3_COMPLETE" if (r.returncode == 0 and all(present.values())
                              and all(v is True for v in ident.values()))
         else "SSR_S3_FAILED")
print(final)
try:
    import datetime as _dt
    with open("output/ssr_s3_marks.log", "a") as _fh:
        _fh.write("SSR_MARK ts=%s %s\\n"
                  % (_dt.datetime.now().astimezone().isoformat(timespec="seconds"), final))
except Exception as _e:
    print("marks append failed:", _e)
'''

# --- Track 4 variant: SSR S4 6-model grid kernel (fresh slug; same runner +
# harvested profile CSVs for the 4 remaining atlas models, driver
# scripts/kaggle/ssr_s4.sh: stream/sublayer(/norm) x {low_light,jpeg} x
# seeds{42,43,44} at k=25 + a clean profiled-dedup control per model). Adds
# src/data.py + src/probes.py (DINOv2 CIFAR-10 frozen-probe protocol).
SSR_S4_SOURCES = dict(SSR_SOURCES)
SSR_S4_SOURCES["ssr.sh"] = "scripts/kaggle/ssr_s4.sh"
SSR_S4_SOURCES["src/data.py"] = "src/data.py"
SSR_S4_SOURCES["src/probes.py"] = "src/probes.py"
for _m in ("convnext_tiny", "swin_tiny_patch4_window7_224",
           "deit_small_patch16_224", "dinov2_vitb14"):
    for _c in ("low_light", "jpeg"):
        _p = f"output/profiles/{_m}/drift_profile_sev5_{_c}.csv"
        SSR_S4_SOURCES[_p] = _p

SSR_S4_METADATA_JSON = '''{
  "id": "arindamtripathi/ssr-s4",
  "title": "ssr-s4",
  "code_file": "kaggle_ssr_s4.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "dataset_sources": ["sautkin/imagenet1kvalid"],
  "competition_sources": [],
  "kernel_sources": []
}
'''

SSR_S4_TEMPLATE = '''"""Self-contained SSR S4 6-model grid kernel (all package files embedded)."""
import base64, os, subprocess
os.environ.setdefault("PYTHONUNBUFFERED", "1")  # live SSR_MARK lines in the Kaggle log

PKG_FILES = {
__CHUNKS__
}

CWD = os.getcwd()
print("CWD:", CWD)
for name, b64 in PKG_FILES.items():
    path = os.path.join(CWD, name)
    os.makedirs(os.path.dirname(path) or CWD, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(base64.b64decode(b64))
    print("staged", name)

# --- ImageNet-val sanity (eval images for the 3 timm models; layouts differ
# between /kaggle/input/<slug> and /kaggle/input/datasets/<owner>/<slug>).
# DINOv2 legs use CIFAR-10 (torchvision download), not this mount. ---
im_roots = []
for root, dirs, files in os.walk("/kaggle/input"):
    if "imagenet1kvalid" in dirs:
        im_roots.append(os.path.join(root, "imagenet1kvalid"))
        dirs.remove("imagenet1kvalid")  # don't descend into the 2GB tree
print("imagenet mounts:", im_roots)
if not im_roots:
    raise RuntimeError("imagenet1kvalid mount missing - aborting before wasting the run")

# --- GPU sanity ---
import torch
print("cuda:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("no GPU attached - aborting before wasting the run")
print("gpu:", torch.cuda.get_device_name(0))

r = subprocess.run(["bash", "ssr.sh"])
invocations = [("timm:convnext_tiny", "stream", "low_light", 42, 25),
               ("timm:convnext_tiny", "sublayer", "low_light", 42, 25),
               ("timm:swin_tiny_patch4_window7_224", "stream", "low_light", 42, 25),
               ("timm:swin_tiny_patch4_window7_224", "sublayer", "low_light", 42, 25),
               ("timm:deit_small_patch16_224", "stream", "low_light", 42, 25),
               ("timm:deit_small_patch16_224", "sublayer", "low_light", 42, 25),
               ("timm:deit_small_patch16_224", "norm", "low_light", 42, 25),
               ("dinov2_vitb14", "stream", "low_light", 42, 25),
               ("dinov2_vitb14", "sublayer", "low_light", 42, 25),
               ("dinov2_vitb14", "norm", "low_light", 42, 25),
               ("timm:convnext_tiny", "stream", "jpeg", 42, 25),
               ("timm:convnext_tiny", "sublayer", "jpeg", 42, 25),
               ("timm:swin_tiny_patch4_window7_224", "stream", "jpeg", 42, 25),
               ("timm:swin_tiny_patch4_window7_224", "sublayer", "jpeg", 42, 25),
               ("timm:deit_small_patch16_224", "stream", "jpeg", 42, 25),
               ("timm:deit_small_patch16_224", "sublayer", "jpeg", 42, 25),
               ("timm:deit_small_patch16_224", "norm", "jpeg", 42, 25),
               ("dinov2_vitb14", "stream", "jpeg", 42, 25),
               ("dinov2_vitb14", "sublayer", "jpeg", 42, 25),
               ("dinov2_vitb14", "norm", "jpeg", 42, 25),
               ("timm:convnext_tiny", "stream", "low_light", 43, 25),
               ("timm:convnext_tiny", "sublayer", "low_light", 43, 25),
               ("timm:swin_tiny_patch4_window7_224", "stream", "low_light", 43, 25),
               ("timm:swin_tiny_patch4_window7_224", "sublayer", "low_light", 43, 25),
               ("timm:deit_small_patch16_224", "stream", "low_light", 43, 25),
               ("timm:deit_small_patch16_224", "sublayer", "low_light", 43, 25),
               ("timm:deit_small_patch16_224", "norm", "low_light", 43, 25),
               ("dinov2_vitb14", "stream", "low_light", 43, 25),
               ("dinov2_vitb14", "sublayer", "low_light", 43, 25),
               ("dinov2_vitb14", "norm", "low_light", 43, 25),
               ("timm:convnext_tiny", "stream", "jpeg", 43, 25),
               ("timm:convnext_tiny", "sublayer", "jpeg", 43, 25),
               ("timm:swin_tiny_patch4_window7_224", "stream", "jpeg", 43, 25),
               ("timm:swin_tiny_patch4_window7_224", "sublayer", "jpeg", 43, 25),
               ("timm:deit_small_patch16_224", "stream", "jpeg", 43, 25),
               ("timm:deit_small_patch16_224", "sublayer", "jpeg", 43, 25),
               ("timm:deit_small_patch16_224", "norm", "jpeg", 43, 25),
               ("dinov2_vitb14", "stream", "jpeg", 43, 25),
               ("dinov2_vitb14", "sublayer", "jpeg", 43, 25),
               ("dinov2_vitb14", "norm", "jpeg", 43, 25),
               ("timm:convnext_tiny", "stream", "low_light", 44, 25),
               ("timm:convnext_tiny", "sublayer", "low_light", 44, 25),
               ("timm:swin_tiny_patch4_window7_224", "stream", "low_light", 44, 25),
               ("timm:swin_tiny_patch4_window7_224", "sublayer", "low_light", 44, 25),
               ("timm:deit_small_patch16_224", "stream", "low_light", 44, 25),
               ("timm:deit_small_patch16_224", "sublayer", "low_light", 44, 25),
               ("timm:deit_small_patch16_224", "norm", "low_light", 44, 25),
               ("dinov2_vitb14", "stream", "low_light", 44, 25),
               ("dinov2_vitb14", "sublayer", "low_light", 44, 25),
               ("dinov2_vitb14", "norm", "low_light", 44, 25),
               ("timm:convnext_tiny", "stream", "jpeg", 44, 25),
               ("timm:convnext_tiny", "sublayer", "jpeg", 44, 25),
               ("timm:swin_tiny_patch4_window7_224", "stream", "jpeg", 44, 25),
               ("timm:swin_tiny_patch4_window7_224", "sublayer", "jpeg", 44, 25),
               ("timm:deit_small_patch16_224", "stream", "jpeg", 44, 25),
               ("timm:deit_small_patch16_224", "sublayer", "jpeg", 44, 25),
               ("timm:deit_small_patch16_224", "norm", "jpeg", 44, 25),
               ("dinov2_vitb14", "stream", "jpeg", 44, 25),
               ("dinov2_vitb14", "sublayer", "jpeg", 44, 25),
               ("dinov2_vitb14", "norm", "jpeg", 44, 25),
               ("timm:convnext_tiny", "profiled", "low_light", 42, 25),
               ("timm:swin_tiny_patch4_window7_224", "profiled", "low_light", 42, 25),
               ("timm:deit_small_patch16_224", "profiled", "low_light", 42, 25),
               ("dinov2_vitb14", "profiled", "low_light", 42, 25),
               ("timm:convnext_tiny", "profiled", "jpeg", 42, 25),
               ("timm:swin_tiny_patch4_window7_224", "profiled", "jpeg", 42, 25),
               ("timm:deit_small_patch16_224", "profiled", "jpeg", 42, 25),
               ("dinov2_vitb14", "profiled", "jpeg", 42, 25)]
dirs = ["%s_%s_k%d_%s_s%d" % (slug.replace(":", "_"), scope, k, corr, seed)
        for slug, scope, corr, seed, k in invocations]
present = {d: os.path.isfile("output/ssr_s4/%s/ssr_results.csv" % d) for d in dirs}
ident = {}
import json as _json
for d in dirs:
    meta = "output/ssr_s4/%s/ssr_meta.json" % d
    if os.path.isfile(meta):
        try:
            ident[d] = bool(_json.load(open(meta))["identity"]["bit_identical"])
        except Exception as e:
            ident[d] = "unreadable: %s" % e
print("ssr exit:", r.returncode, "| results:", present, "| identity:", ident)
final = ("SSR_S4_COMPLETE" if (r.returncode == 0 and all(present.values())
                              and all(v is True for v in ident.values()))
         else "SSR_S4_FAILED")
print(final)
try:
    import datetime as _dt
    with open("output/ssr_s4_marks.log", "a") as _fh:
        _fh.write("SSR_MARK ts=%s %s\\n"
                  % (_dt.datetime.now().astimezone().isoformat(timespec="seconds"), final))
except Exception as _e:
    print("marks append failed:", _e)
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
    ap.add_argument("--variant", default="grid", choices=["grid", "vitb_profile", "vitb_lora_v9", "vitb_lora_abl", "atlas", "atlas_v4", "ssr_s1", "ssr_s2", "ssr_s3", "ssr_s4"],
                    help="grid = corruption-grid kernel (corrgrid-tail, byte-identical v6); "
                         "vitb_profile = Track 2 v8 ViT-B drift-profile kernel; "
                         "vitb_lora_v9 = Track 2 v9 ViT-B LoRA arms kernel; "
                         "vitb_lora_abl = Track 2 v9 allocation-rule ablations kernel; "
                         "atlas = Track 3 cross-family drift-atlas kernel; "
                         "atlas_v4 = atlas under the fresh drift-atlas-v4 slug "
                         "(same sources; hardened atlas.sh driver); "
                         "ssr_s1 = Track 4 SSR S1 smoke kernel (run_ssr.py + "
                         "harvested selection CSVs; hardened ssr.sh driver); "
                         "ssr_s2 = Track 4 SSR S2 attachment-point probe kernel "
                         "(scope legs --attach; hardened ssr_s2.sh driver); "
                         "ssr_s3 = Track 4 SSR S3 replication kernel (stream/sublayer "
                         "x seeds x jpeg + k-matched controls; hardened ssr_s3.sh driver); "
                         "ssr_s4 = Track 4 SSR S4 6-model grid kernel (convnext/swin/deit/"
                         "dinov2 stream/sublayer(/norm) x corr x seeds + profiled-dedup "
                         "control; hardened ssr_s4.sh driver)")
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
    elif args.variant == "atlas":
        base_sources, template = ATLAS_SOURCES, ATLAS_TEMPLATE
        metadata, out_name = ATLAS_METADATA_JSON, "kaggle_drift_atlas.py"
    elif args.variant == "atlas_v4":
        base_sources, template = ATLAS_SOURCES, ATLAS_TEMPLATE
        metadata, out_name = ATLAS_V4_METADATA_JSON, "kaggle_drift_atlas_v4.py"
    elif args.variant == "ssr_s1":
        base_sources, template = SSR_SOURCES, SSR_TEMPLATE
        metadata, out_name = SSR_METADATA_JSON, "kaggle_ssr_s1.py"
    elif args.variant == "ssr_s2":
        base_sources, template = SSR_S2_SOURCES, SSR_S2_TEMPLATE
        metadata, out_name = SSR_S2_METADATA_JSON, "kaggle_ssr_s2.py"
    elif args.variant == "ssr_s3":
        base_sources, template = SSR_S3_SOURCES, SSR_S3_TEMPLATE
        metadata, out_name = SSR_S3_METADATA_JSON, "kaggle_ssr_s3.py"
    elif args.variant == "ssr_s4":
        base_sources, template = SSR_S4_SOURCES, SSR_S4_TEMPLATE
        metadata, out_name = SSR_S4_METADATA_JSON, "kaggle_ssr_s4.py"
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
        try:
            existing_id = json.loads(open(meta_path).read()).get("id")
            wanted_id = json.loads(metadata).get("id")
            if existing_id != wanted_id:
                print(f"WARNING: existing metadata targets {existing_id!r} but variant "
                      f"{args.variant!r} builds {wanted_id!r} - pushing from this dir "
                      f"would update the WRONG kernel; use a fresh --pkg dir.")
        except Exception as e:
            print(f"WARNING: could not parse existing {meta_path} ({e}); verify the "
                  f"kernel id before pushing")

    print("\npush with GPU (both knobs matter):")
    print(f"  cd {args.pkg} && kaggle kernels push -p .")
    print("  (enable_gpu: true in metadata AND --accelerator nvidiaTeslaT4 on push)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
