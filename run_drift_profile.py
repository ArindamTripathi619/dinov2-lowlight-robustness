"""
Drift-profile sweep runner (docs/ROADMAP.md section 2, Tracks 2+3 harness)
==========================================================================

Severity x corruption sweep of per-module drift profiles over ONE clean
pass, reusing the run_drift_proxy.py harness (model-agnostic module
selection, pooling recorder, proxies, CKA, agreement, gate).

This is the profiling harness the roadmap names for Track 3's atlas; its
first job is Track 2's v8 kernel: the ViT-B/14 CKA profile under
low_light + jpeg that parameterizes the v9 drift-weighted LoRA arms.

Per (corruption, severity) key it emits the same artifacts as
run_drift_proxy.py — proxy_profiles.csv (all keys), per-key
drift_profile_sev{S}_{corr}.csv (run_lora_simple_colab.py --drift-csv
compatible), agreement_report.{txt,json} (gate evaluated at the highest
severity — the allocation severity), and per-key profile PNGs plus a
modules x keys CKA-drop heatmap.

Usage:
  python3 run_drift_profile.py --model dinov2_vitb14 --n-images 1000 \
      --severities 1,2,3,4,5 --corruptions low_light,jpeg \
      --output output/vitb_profile
  python3 run_drift_profile.py --model resnet50 --n-images 100 \
      --severities 1,5 --corruptions blur --output /tmp/sweep_smoke
"""

import argparse
import hashlib
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch

from utils import (DEVICE, get_corruption, load_cifar10_subset,
                   load_imagenet_subset)
import run_drift_proxy as rdp


def parse_args():
    p = argparse.ArgumentParser(
        description="Severity x corruption drift-profile sweep (Tracks 2+3 harness)")
    p.add_argument("--model", default="dinov2_vitb14",
                   help="dinov2_* (torch.hub) or resnet50 (torchvision)")
    p.add_argument("--n-images", type=int, default=1000)
    p.add_argument("--severities", default="1,2,3,4,5",
                   help="comma-separated degraded severities (0 is the clean pass)")
    p.add_argument("--corruptions", default="low_light,jpeg",
                   help="comma-separated subset of blur,low_light,jpeg,contrast")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dataset", default="cifar10", choices=["cifar10", "imagenet"],
                   help="profiling images: cifar10 test subset (legacy rows) or "
                        "an ImageNet-val extract (Track 3 atlas of record)")
    p.add_argument("--data-root", default=None,
                   help="dataset root (default: ./data for cifar10; required for imagenet — "
                        "expects <root>/<wnid>/*.jpg, e.g. the Kaggle imagenet1kvalid mount)")
    p.add_argument("--output", default="output/drift_profile")
    p.add_argument("--cache-dir", default=None,
                   help="directory for the ImageNet image-list cache (imagenet dataset only): "
                        "the seeded glob+decode over the read-only /kaggle/input FUSE mount "
                        "runs once; later model invocations load byte-identical arrays from "
                        "this cache. Keyed by n/seed/data_root, so a different protocol never "
                        "reuses stale images.")
    p.add_argument("--max-depth", type=int, default=2)
    p.add_argument("--vit-token", default="cls", choices=["cls", "mean"])
    p.add_argument("--rho-threshold", type=float, default=0.8)
    return p.parse_args()


def load_imagenet_cached(n_images, seed, data_root, cache_dir):
    """Seeded ImageNet subset via load_imagenet_subset, cached on local disk.

    The atlas grid invokes run_drift_profile.py once per model (fresh
    processes), each of which used to re-glob ~5000 patterns over 1000 class
    folders on the read-only FUSE mount and re-decode 2000 JPEGs. With this
    cache only the first invocation pays that cost; hits return byte-identical
    arrays, so the matched-noise corruption rng convention (default_rng(1000+
    sev) over the same images) is unaffected.
    """
    os.makedirs(cache_dir, exist_ok=True)
    root_key = hashlib.md5(os.path.abspath(data_root).encode()).hexdigest()[:8]
    key = f"imagenet_n{n_images}_seed{seed}_{root_key}.pkl"
    path = os.path.join(cache_dir, key)
    t0 = time.time()
    if os.path.isfile(path):
        with open(path, "rb") as fh:
            images, labels = pickle.load(fh)
        print(f"[+{time.time() - t0:.1f}s] ImageNet cache HIT: {path} "
              f"({len(images)} images)", flush=True)
        return images, labels
    images, labels = load_imagenet_subset(n_images=n_images, seed=seed,
                                           data_root=data_root)
    with open(path, "wb") as fh:
        pickle.dump((images, labels), fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[+{time.time() - t0:.1f}s] ImageNet cache MISS -> wrote {path} "
          f"({os.path.getsize(path) / 1e6:.0f} MB)", flush=True)
    return images, labels


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    t_wall = time.time()
    corruptions = [c.strip() for c in args.corruptions.split(",") if c.strip()]
    severities = sorted({int(s) for s in args.severities.split(",") if s.strip()})
    if not severities or any(s < 1 or s > 5 for s in severities):
        raise ValueError(f"--severities must be in 1..5, got {severities}")

    print("=" * 72)
    print(f"Drift-profile sweep — model={args.model}, n={args.n_images}, "
          f"severities={severities}, corruptions={corruptions}")
    print(f"Device: {DEVICE}")
    print("=" * 72)

    if args.dataset == "imagenet":
        if not args.data_root:
            raise ValueError("--dataset imagenet requires --data-root pointing at a "
                             "<root>/<wnid>/*.jpg ImageNet-val extract")
        if args.cache_dir:
            images, _labels = load_imagenet_cached(args.n_images, args.seed,
                                                   args.data_root, args.cache_dir)
        else:
            images, _labels = load_imagenet_subset(n_images=args.n_images, seed=args.seed,
                                                   data_root=args.data_root)
    else:
        images, _labels = load_cifar10_subset(n_images=args.n_images, train=False,
                                              seed=args.seed,
                                              data_root=args.data_root or "./data")
    print(f"[+{time.time() - t_wall:.0f}s] images ready ({len(images)}); loading model",
          flush=True)
    model, model_desc = rdp.load_model(args.model, DEVICE)
    print(f"[+{time.time() - t_wall:.0f}s] model ready: {model_desc}", flush=True)
    modules = rdp.select_profiled_modules(model, max_depth=args.max_depth)
    print(f"Profiling {len(modules)} modules: {list(modules)}")

    # ONE clean pass, reused by every (corruption, severity) key.
    print(f"\n[+{time.time() - t_wall:.0f}s] >>> clean pass (severity 0)", flush=True)
    acts_clean = rdp.forward_pass(model, images, modules, args.batch_size, DEVICE,
                                  token_mode=args.vit_token)
    module_order = [m for m in modules if m in acts_clean]

    t0 = time.time()
    rows_by_key, agreement_by_key = {}, {}
    for corr in corruptions:
        fn = get_corruption(corr)
        for sev in severities:
            key = f"{corr}@sev{sev}"
            rng = np.random.default_rng(1000 + sev)  # matched-noise convention
            print(f"\n[+{time.time() - t_wall:.0f}s] >>> {key}", flush=True)
            degraded = [fn(img, sev, rng=rng) for img in images]
            acts_deg = rdp.forward_pass(model, degraded, modules, args.batch_size,
                                        DEVICE, token_mode=args.vit_token)
            rows = rdp.build_profile_rows(acts_clean, acts_deg, module_order,
                                          corr, sev)
            rows_by_key[key] = rows
            agreement_by_key[key] = rdp.profile_agreement(rows)
    print(f"\nsweep passes done in {time.time() - t0:.0f}s "
          f"(wall +{time.time() - t_wall:.0f}s)", flush=True)

    # Gate at the allocation severity (highest swept).
    gate_sev = severities[-1]
    gate_keys = [f"{c}@sev{gate_sev}" for c in corruptions]
    gate = rdp.evaluate_gate({k: agreement_by_key[k] for k in gate_keys},
                             len(module_order), rho_threshold=args.rho_threshold)

    meta = {
        "model": args.model, "model_desc": model_desc,
        "torch_version": torch.__version__, "device": DEVICE,
        "dataset": args.dataset, "data_root": args.data_root,
        "n_images": len(images), "seed": args.seed,
        "severity": ",".join(str(s) for s in severities),
        "corruptions": corruptions,
        "n_modules": len(module_order), "modules": module_order,
        "vit_token": args.vit_token, "max_depth": args.max_depth,
        "gate_severity": gate_sev,
        "cache_dir": args.cache_dir,
        "cov_precision": rdp.COV_DTYPE,
    }

    print("\n>>> outputs")
    rdp.write_proxy_profiles_csv(rows_by_key,
                                 os.path.join(args.output, "proxy_profiles.csv"))
    rdp.write_drift_profile_csvs(rows_by_key, args.output, None)
    rdp.write_agreement_report(rows_by_key, agreement_by_key, gate, meta, args.output)

    print("\n>>> plots")
    for key, rows in rows_by_key.items():
        rdp.plot_profile_panels(rows, agreement_by_key[key], key,
                                os.path.join(args.output, f"profiles_{key}.png"))
    rdp.plot_drift_heatmap(rows_by_key, module_order,
                           os.path.join(args.output, "profile_heatmap.png"))

    print("\n" + "=" * 72)
    print(f"GATE at severity {gate_sev}: {gate['verdict']} "
          f"(best proxy: {gate['best_proxy']})")
    for key, v in gate["per_proxy"].items():
        rho_str = ", ".join(f"{c}={v['rho'][c]:.3f}" for c in gate_keys)
        print(f"  {key:<12} rho: {rho_str}")
    print("=" * 72)


if __name__ == "__main__":
    main()
