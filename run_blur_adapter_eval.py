#!/usr/bin/env python3
"""Re-evaluate retained LoRA adapter checkpoints whose eval logs were lost.

Closes the sessionE gap: blur_uniform / blur_drift adapters were trained to
convergence on Colab T4 but their evaluation logs were truncated on salvage.
This script rebuilds each LoRA-wrapped backbone from its `lora_adapters.pt`
(same rebuild path proven in run_exdark_baseline.py) and replays the exact
run_lora_simple_colab.py evaluation protocol:

  * test set: load_cifar10_subset(n=1000, seed=42) — same image set as Phases 1/3
  * matched-noise eval corruption: seeded default_rng(1000 + severity)
  * probe: LogisticRegression(max_iter=2000, C=1.0) on severity-0 embeddings,
    70/30 stratified split, random_state=42
  * original-model comparison pass on the identical corrupted arrays

Usage:
    python run_blur_adapter_eval.py --ckpt colab_results/sessionE/blur_drift/lora_adapters.pt \
        --corruption blur --label blur_drift --output output/blur_reeval/blur_drift
    python run_blur_adapter_eval.py --ckpt colab_results/sessionE/blur_uniform/lora_adapters.pt \
        --corruption blur --label blur_uniform --output output/blur_reeval/blur_uniform

Local CPU: ~25-40 min per checkpoint (dominated by 2 x 6 x 1000 embeddings).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from utils import DEVICE, load_cifar10_subset, load_dinov2, DINOV2_PREPROCESS, get_corruption


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--ckpt", required=True, help="path to lora_adapters.pt")
    p.add_argument("--corruption", default="blur",
                   choices=["low_light", "blur", "jpeg", "contrast"],
                   help="eval corruption (must match the corruption the adapters were trained on)")
    p.add_argument("--label", default=None, help="arm label for reports (default: ckpt config layers)")
    p.add_argument("--model", default=None, help="hub model (default: ckpt config model)")
    p.add_argument("--n-images", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42, help="test-set draw seed (Phase 1/3 protocol)")
    p.add_argument("--output", required=True)
    return p.parse_args()


ARGS = parse_args()
os.makedirs(ARGS.output, exist_ok=True)
device = DEVICE
preprocess = DINOV2_PREPROCESS

# ---------------------------------------------------------------------------
# 1. Load checkpoint + rebuild LoRA-wrapped backbone (exdark-proven path)
# ---------------------------------------------------------------------------
print("=" * 60)
print("ADAPTER RE-EVALUATION")
print("=" * 60)
ckpt = torch.load(ARGS.ckpt, map_location="cpu", weights_only=False)
cfg = ckpt.get("config", {})
model_name = ARGS.model or cfg.get("model", "dinov2_vits14")
label = ARGS.label or cfg.get("layers", "unknown")
block_ranks = {int(k): int(v) for k, v in (cfg.get("block_ranks") or {}).items()}
alpha = int(cfg.get("alpha", 16))
base_rank = int(cfg.get("rank", 8))
n_blocks = 12  # ViT-S/14; config trained on this architecture
print(f"ckpt: {ARGS.ckpt}")
print(f"config: model={model_name} corruption={cfg.get('corruption')} layers={cfg.get('layers')} "
      f"rank={base_rank} alpha={alpha} seed={cfg.get('seed')}")
print(f"block_ranks: {block_ranks or '(uniform — falling back to base rank on all blocks)'}")
if cfg.get("corruption") not in (None, ARGS.corruption):
    print(f"WARNING: adapters were trained on '{cfg.get('corruption')}' but evaluating '{ARGS.corruption}'")

model, _ = load_dinov2(model_name=model_name)
model = model.to(device)


class LoRALayer(nn.Module):
    def __init__(self, original_layer, r=8, alpha=16):
        super().__init__()
        self.original = original_layer
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False
        self.lora_A = nn.Parameter(torch.randn(r, original_layer.in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(original_layer.out_features, r))
        self.scaling = alpha / r

    def forward(self, x):
        return self.original(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scaling


if not block_ranks:
    block_ranks = {b: base_rank for b in range(n_blocks)}
for b, block in enumerate(model.blocks):
    r = block_ranks.get(b)
    if r:
        block.attn.qkv = LoRALayer(block.attn.qkv, r=r, alpha=alpha)
        block.attn.proj = LoRALayer(block.attn.proj, r=r, alpha=alpha)
missing, unexpected = model.load_state_dict(ckpt["adapter_state"], strict=False)
n_loaded = len(ckpt["adapter_state"])
print(f"Rebuilt LoRA on {len(block_ranks)} blocks, loaded {n_loaded} adapter tensors "
      f"(missing={len(missing)}, unexpected={len(unexpected)})")
assert n_loaded > 0 and len(unexpected) == 0, "adapter state did not load cleanly — aborting"
model.eval()

# ---------------------------------------------------------------------------
# 2. Test set + matched-noise corrupted arrays (Phase 3 protocol)
# ---------------------------------------------------------------------------
print(f"\n>>> test set: {ARGS.n_images} CIFAR-10 test images (seed {ARGS.seed})")
images, labels = load_cifar10_subset(n_images=ARGS.n_images, seed=ARGS.seed)
corrupt_fn = get_corruption(ARGS.corruption)
eval_rng = {s: np.random.default_rng(1000 + s) for s in range(6)}
degraded_eval = {s: [corrupt_fn(img, s, rng=eval_rng[s]) for img in images] for s in range(6)}


@torch.no_grad()
def get_embeddings(m, image_list, batch_size=64):
    feats = []
    for i in range(0, len(image_list), batch_size):
        batch = image_list[i:i + batch_size]
        tensors = torch.stack([preprocess(img) for img in batch]).to(device)
        out = m(tensors)
        if out.dim() == 3:
            out = out[:, 0, :]
        feats.append(out.cpu())
        print(f"\r    embedded {min(i + batch_size, len(image_list))}/{len(image_list)}", end="", flush=True)
    print()
    return torch.cat(feats, dim=0).numpy()


print("\n>>> adapter-model embeddings")
adapted_pooled = {s: get_embeddings(model, degraded_eval[s]) for s in range(6)}
print("  " + ", ".join(f"sev{s}: {adapted_pooled[s].shape}" for s in range(6)))

# ---------------------------------------------------------------------------
# 3. Probe on severity-0 embeddings, eval across severities (protocol parity)
# ---------------------------------------------------------------------------
X_clean = adapted_pooled[0]
X_train, X_test, y_train, y_test, _, idx_test = train_test_split(
    X_clean, labels, np.arange(len(labels)), test_size=0.3,
    random_state=42, stratify=labels,
)
probe = LogisticRegression(max_iter=2000, C=1.0)
probe.fit(X_train, y_train)
print("\n>>> adapter-model probe evaluation")
adapted_accs = []
for severity in range(6):
    acc = accuracy_score(y_test, probe.predict(adapted_pooled[severity][idx_test]))
    adapted_accs.append(float(acc))
    print(f"  severity {severity}: {acc:.4f}")

# ---------------------------------------------------------------------------
# 4. Original-model comparison pass on the identical corrupted arrays
# ---------------------------------------------------------------------------
print("\n>>> original-model comparison (pristine hub model)")
orig, _ = load_dinov2(model_name=model_name)
orig = orig.to(device)
orig.eval()
orig_pooled = {s: get_embeddings(orig, degraded_eval[s]) for s in range(6)}
X_clean_o = orig_pooled[0]
X_train_o, X_test_o, y_train_o, y_test_o, _, idx_test_o = train_test_split(
    X_clean_o, labels, np.arange(len(labels)), test_size=0.3,
    random_state=42, stratify=labels,
)
probe_o = LogisticRegression(max_iter=2000, C=1.0)
probe_o.fit(X_train_o, y_train_o)
orig_accs = []
for severity in range(6):
    acc = accuracy_score(y_test_o, probe_o.predict(orig_pooled[severity][idx_test_o]))
    orig_accs.append(float(acc))
    print(f"  severity {severity}: {acc:.4f}")

# ---------------------------------------------------------------------------
# 5. Report
# ---------------------------------------------------------------------------
results = {
    "label": label,
    "ckpt": ARGS.ckpt,
    "corruption": ARGS.corruption,
    "model": model_name,
    "ckpt_config": cfg,
    "protocol": {"n_images": ARGS.n_images, "seed": ARGS.seed,
                 "eval_rng": "1000+severity", "split": "70/30 stratified, random_state=42"},
    "adapted_accs": adapted_accs,
    "orig_accs": orig_accs,
}
adapted_mean = float(np.mean(adapted_accs))
orig_mean = float(np.mean(orig_accs))
print("\n" + "=" * 60)
print(f"RE-EVALUATION COMPLETE — arm: {label}")
print("=" * 60)
print(f"{'Severity':<10} {'Original':>10} {'Adapted':>10} {'Delta':>10}")
print("-" * 42)
for i in range(6):
    print(f"{i:<10} {orig_accs[i]:>10.4f} {adapted_accs[i]:>10.4f} {adapted_accs[i]-orig_accs[i]:>+10.4f}")
print(f"\nMean accuracy: Original={orig_mean:.4f}, Adapted={adapted_mean:.4f}, "
      f"Delta={adapted_mean-orig_mean:+.4f}")

with open(os.path.join(ARGS.output, "results.json"), "w") as f:
    json.dump({**results, "orig_mean": orig_mean, "adapted_mean": adapted_mean}, f, indent=2)
with open(os.path.join(ARGS.output, "results.csv"), "w") as f:
    f.write("severity,original,adapted\n")
    for i in range(6):
        f.write(f"{i},{orig_accs[i]:.4f},{adapted_accs[i]:.4f}\n")
print(f"saved {ARGS.output}/results.json and results.csv")
