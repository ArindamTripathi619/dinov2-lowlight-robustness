"""Recompute the layer-wise CKA matrix and persist it as a CSV artifact.

Produces output/notebook2/cka_matrix.csv — the numeric record of record for
the values quoted in docs/RESEARCH.md and README.md (12 layers x 6 severities).
Protocol identical to run_notebook2.py: 500 CIFAR-10 test images, seed 42,
CLS activations via hooks on all 12 blocks, linear CKA vs clean.
"""
import os
import numpy as np
import torch

from utils import (
    DEVICE, load_dinov2, load_cifar10_subset, DINOV2_PREPROCESS, linear_cka,
    low_light,
)

N_IMAGES = 500
OUT_CSV = "output/notebook2/cka_matrix.csv"


@torch.no_grad()
def extract_layer_activations(model, image_list, hooks_dict, device, batch_size=64):
    """Return {layer_idx: (N, D) CLS activations} using registered hooks."""
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


def main():
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    model, n_blocks = load_dinov2(DEVICE)
    hooks = {i: CLSHook(block) for i, block in enumerate(model.blocks)}

    images, labels = load_cifar10_subset(n_images=N_IMAGES, train=False)

    print(">>> clean (severity 0) activations")
    clean_acts = extract_layer_activations(model, images, hooks, DEVICE)

    rows = ["layer,severity,cka,drop_from_clean"]
    for severity in range(1, 6):
        print(f">>> severity {severity}")
        degraded = [low_light(img, severity) for img in images]
        sev_acts = extract_layer_activations(model, degraded, hooks, DEVICE)
        for l in range(n_blocks):
            cka = float(linear_cka(clean_acts[l], sev_acts[l]))
            rows.append(f"{l},{severity},{cka:.6f},{1.0 - cka:.6f}")

    for h in hooks.values():
        h.remove()

    with open(OUT_CSV, "w") as f:
        f.write("\n".join(rows) + "\n")
    print(f"saved {OUT_CSV}")

    print("\n--- CKA drop (clean -> severity 5) by layer ---")
    for l in range(n_blocks):
        drop = [r for r in rows if r.startswith(f"{l},5,")][0].split(",")[-1]
        print(f"  block {l:2d}: drop {float(drop):.4f}")
    drops = {l: float([r for r in rows if r.startswith(f"{l},5,")][0].split(",")[-1])
             for l in range(n_blocks)}
    worst = max(drops, key=drops.get)
    print(f"max drop: block {worst} ({drops[worst]:.4f})")


if __name__ == "__main__":
    main()
