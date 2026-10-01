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
    ap.add_argument("--pkg", default="/tmp/kaggle_kernel",
                    help="staging dir receiving kaggle_corrgrid_tail.py (default: /tmp/kaggle_kernel)")
    ap.add_argument("--source", action="append", default=[], metavar="NAME=PATH",
                    help="override the packaged source for file NAME (repeatable)")
    return ap.parse_args()


def main():
    args = parse_args()
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    sources = dict(DEFAULT_SOURCES)
    for pair in args.source:
        name, _, path = pair.partition("=")
        if name not in sources or not path:
            print(f"error: --source expects NAME=PATH with NAME in {sorted(sources)}; got {pair!r}")
            return 1
        sources[name] = path

    blobs = []
    for name in DEFAULT_SOURCES:
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

    script = TEMPLATE.replace("__CHUNKS__", "\n".join(chunks))

    os.makedirs(args.pkg, exist_ok=True)
    out = os.path.join(args.pkg, "kaggle_corrgrid_tail.py")
    with open(out, "w") as f:
        f.write(script)
    print(f"wrote {out}: {len(script)} bytes")

    meta_path = os.path.join(args.pkg, "kernel-metadata.json")
    if not os.path.isfile(meta_path):
        with open(meta_path, "w") as f:
            f.write(METADATA_JSON)
        print(f"wrote {meta_path} (was missing)")
    else:
        print(f"kept existing {meta_path}")

    print("\npush with GPU (both knobs matter):")
    print(f"  cd {args.pkg} && kaggle kernels push -p .")
    print("  (enable_gpu: true in metadata AND --accelerator nvidiaTeslaT4 on push)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
