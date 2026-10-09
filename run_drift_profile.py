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
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch

from utils import DEVICE, get_corruption, load_cifar10_subset
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
    p.add_argument("--data-root", default="./data")
    p.add_argument("--output", default="output/drift_profile")
    p.add_argument("--max-depth", type=int, default=2)
    p.add_argument("--vit-token", default="cls", choices=["cls", "mean"])
    p.add_argument("--rho-threshold", type=float, default=0.8)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    corruptions = [c.strip() for c in args.corruptions.split(",") if c.strip()]
    severities = sorted({int(s) for s in args.severities.split(",") if s.strip()})
    if not severities or any(s < 1 or s > 5 for s in severities):
        raise ValueError(f"--severities must be in 1..5, got {severities}")

    print("=" * 72)
    print(f"Drift-profile sweep — model={args.model}, n={args.n_images}, "
          f"severities={severities}, corruptions={corruptions}")
    print(f"Device: {DEVICE}")
    print("=" * 72)

    images, _labels = load_cifar10_subset(n_images=args.n_images, train=False,
                                          seed=args.seed, data_root=args.data_root)
    model, model_desc = rdp.load_model(args.model, DEVICE)
    modules = rdp.select_profiled_modules(model, max_depth=args.max_depth)
    print(f"Profiling {len(modules)} modules: {list(modules)}")

    # ONE clean pass, reused by every (corruption, severity) key.
    print("\n>>> clean pass (severity 0)")
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
            print(f"\n>>> {key}")
            degraded = [fn(img, sev, rng=rng) for img in images]
            acts_deg = rdp.forward_pass(model, degraded, modules, args.batch_size,
                                        DEVICE, token_mode=args.vit_token)
            rows = rdp.build_profile_rows(acts_clean, acts_deg, module_order,
                                          corr, sev)
            rows_by_key[key] = rows
            agreement_by_key[key] = rdp.profile_agreement(rows)
    print(f"\nsweep passes done in {time.time() - t0:.0f}s")

    # Gate at the allocation severity (highest swept).
    gate_sev = severities[-1]
    gate_keys = [f"{c}@sev{gate_sev}" for c in corruptions]
    gate = rdp.evaluate_gate({k: agreement_by_key[k] for k in gate_keys},
                             len(module_order), rho_threshold=args.rho_threshold)

    meta = {
        "model": args.model, "model_desc": model_desc,
        "torch_version": torch.__version__, "device": DEVICE,
        "n_images": len(images), "seed": args.seed,
        "severity": ",".join(str(s) for s in severities),
        "corruptions": corruptions,
        "n_modules": len(module_order), "modules": module_order,
        "vit_token": args.vit_token, "max_depth": args.max_depth,
        "gate_severity": gate_sev,
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
