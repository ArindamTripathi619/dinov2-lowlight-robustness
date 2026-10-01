"""Recompute the CKA matrix of record with the UNBIASED estimator (upstream).

Purpose: our drift-weighted LoRA allocation (Phase 3 v2) rests on the sev-5
CKA drop profile computed with the standard (biased) linear CKA at n=500,
d=384. Upstream's audit found the diagonal (self-similarity) terms inflate
similarity when n is small relative to d. This script recomputes the exact
same artifact (output/notebook2/cka_matrix.csv protocol: 500 test images,
seed 42, CLS hooks, low_light) with linear_cka_unbiased and writes
output/notebook2/cka_matrix_unbiased.csv, then reports:

  1. per-block delta (biased vs unbiased) at each severity,
  2. the rank allocation each profile would produce at r=8 (the decision that
     matters), and
  3. a verdict: does the drift-weighted prescription survive the estimator
     correction?

Local CPU: ~35 min (500 images x 6 severities through ViT-S/14).
"""
import csv
import os

import numpy as np
import torch

from utils import (
    DEVICE, load_dinov2, load_cifar10_subset, DINOV2_PREPROCESS, low_light,
)
from stats_tools import linear_cka_unbiased

N_IMAGES = 500
OUT_CSV = "output/notebook2/cka_matrix_unbiased.csv"
BIASED_CSV = "output/notebook2/cka_matrix.csv"
RANK_BUDGET = 8


@torch.no_grad()
def extract_layer_activations(model, image_list, hooks_dict, device, batch_size=64):
    acts = {l: [] for l in hooks_dict}
    for i in range(0, len(image_list), batch_size):
        batch = image_list[i:i + batch_size]
        tensors = torch.stack([DINOV2_PREPROCESS(img) for img in batch]).to(device)
        model(tensors)
        for l, hook in hooks_dict.items():
            acts[l].append(hook.storage.cpu())
    return {l: torch.cat(v, dim=0).numpy() for l, v in acts.items()}


class CLSHook:
    def __init__(self, layer):
        self.storage = None
        self.handle = layer.register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        self.storage = out[:, 0].detach()

    def remove(self):
        self.handle.remove()


def allocation_from_drops(drops, rank_budget=RANK_BUDGET):
    """Exact allocation rule from run_lora_simple_colab.py: round(rank * drop_i / max(drop)),
    blocks rounding below 1 are dropped entirely."""
    m = max(drops)
    alloc = {}
    for i, d in enumerate(drops):
        r = round(rank_budget * d / m)
        if r >= 1:
            alloc[i] = r
    return alloc


def main():
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    model, n_blocks = load_dinov2(DEVICE)
    hooks = {i: CLSHook(block) for i, block in enumerate(model.blocks)}

    images, labels = load_cifar10_subset(n_images=N_IMAGES, train=False)

    print(">>> clean (severity 0) activations")
    clean_acts = extract_layer_activations(model, images, hooks, DEVICE)

    unbiased = {}  # (layer, severity) -> cka_u
    for severity in range(1, 6):
        print(f">>> severity {severity}")
        degraded = [low_light(img, severity) for img in images]
        sev_acts = extract_layer_activations(model, degraded, hooks, DEVICE)
        for l in range(n_blocks):
            unbiased[(l, severity)] = linear_cka_unbiased(clean_acts[l], sev_acts[l])

    for h in hooks.values():
        h.remove()

    biased = {}
    with open(BIASED_CSV) as f:
        for row in csv.DictReader(f):
            biased[(int(row["layer"]), int(row["severity"]))] = float(row["cka"])

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "severity", "cka", "drop_from_clean"])
        for l in range(n_blocks):
            for s in range(1, 6):
                cka = unbiased[(l, s)]
                w.writerow([l, s, f"{cka:.6f}", f"{1.0 - cka:.6f}"])
    print(f"saved {OUT_CSV}")

    print("\n--- per-block sev-5: biased vs unbiased CKA (drop in parens) ---")
    drops_u, drops_b = [], []
    for l in range(n_blocks):
        cb = biased[(l, 5)]
        cu = unbiased[(l, 5)]
        drops_b.append(1.0 - cb)
        drops_u.append(1.0 - cu)
        print(f"  block {l:2d}: biased {cb:.4f} (drop {1-cb:.4f}) | "
              f"unbiased {cu:.4f} (drop {1-cu:.4f}) | delta {cu-cb:+.4f}")

    alloc_b = allocation_from_drops(drops_b)
    alloc_u = allocation_from_drops(drops_u)
    n_params_b = sum(2 * r * 384 + 2 * 384 * r for r in alloc_b.values())  # qkv+proj, d=384
    n_params_u = sum(2 * r * 384 + 2 * 384 * r for r in alloc_u.values())

    print("\n--- rank allocation at r=8 from each profile ---")
    print(f"  biased  : {alloc_b}")
    print(f"  unbiased: {alloc_u}")
    same = alloc_b == alloc_u
    print(f"\nVERDICT: allocations {'IDENTICAL' if same else 'DIFFER'}")
    print(f"  trainable params: biased {n_params_b:,} vs unbiased {n_params_u:,}")
    if not same:
        diff = {k: (alloc_b.get(k), alloc_u.get(k)) for k in
                set(alloc_b) | set(alloc_u) if alloc_b.get(k) != alloc_u.get(k)}
        print(f"  differing blocks (biased, unbiased): {diff}")
        print("  -> drift-weighted arm should be re-run with the unbiased profile"
              " (or the difference noted as estimator-sensitivity in METHODS)")
    else:
        print("  -> drift-weighted prescription is estimator-robust; note the"
              " robustness check in METHODS and keep the original allocation")


if __name__ == "__main__":
    main()
