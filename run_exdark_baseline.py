"""
ExDark Baseline: Real-Dark Validation of the Low-Light Robustness Story
=======================================================================
Phase 3 of the study's sim-to-real bridge. Runs on the Exclusively Dark
(ExDark) dataset — 7,363 REAL low-light photographs, 12 PASCAL-VOC-style
classes (Loh & Chan, CVIU 2019).

What it measures (all with frozen DINOv2 + logistic-regression probe):
  1. Real-dark baseline        — probe accuracy on real dark images (12-class).
  2. Darkness-response curve   — accuracy binned by empirical mean-luminance
                                 quintile (the mirrors lost the lighting-condition
                                 tags, so we compute a continuous darkness axis).
  3. Enhancement control       — same probe on CLAHE-enhanced images: if accuracy
                                 jumps, the failure is luminance-loss-driven on
                                 real data too (mechanism check).
  4. Adapter transfer (opt.)   — with --adapters <lora_adapters.pt>, rebuild the
                                 LoRA-wrapped backbone (trained on SYNTHETIC
                                 CIFAR-10 darkness) and probe again: does
                                 synthetic-dark adaptation transfer to real dark?

Local CPU runtime: ~30-45 min for all 7,363 images x 2 passes (raw + CLAHE).
Outputs: output/exdark/{results.csv, darkness_curve.png, class_accuracy.png}
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from utils import (
    DEVICE, load_dinov2, DINOV2_PREPROCESS, setup_output_dir, save_fig,
)


def parse_args():
    p = argparse.ArgumentParser(description="ExDark real-dark baseline (frozen DINOv2 + linear probe)")
    p.add_argument("--data-root", default="./data/exdark", help="dir containing the 12 class folders")
    p.add_argument("--model", default="dinov2_vits14")
    p.add_argument("--seed", type=int, default=42, help="split seed")
    p.add_argument("--output", default="./output/exdark")
    p.add_argument("--adapters", default=None,
                   help="optional lora_adapters.pt from run_lora_simple_colab.py — enables the sim-to-real transfer test")
    p.add_argument("--adapters-rank", type=int, default=8,
                   help="base rank used when the adapters were trained (fallback if ckpt lacks config)")
    p.add_argument("--max-per-class", type=int, default=None, help="cap images per class (smoke tests)")
    return p.parse_args()


ARGS = parse_args()
OUT = setup_output_dir(ARGS.output)

CLASSES = sorted(d for d in os.listdir(ARGS.data_root)
                 if os.path.isdir(os.path.join(ARGS.data_root, d)))
print("=" * 60)
print("ExDark Real-Dark Baseline")
print(f"  data: {ARGS.data_root}")
print(f"  classes ({len(CLASSES)}): {CLASSES}")
print(f"  model: {ARGS.model}, device: {DEVICE}")
print("=" * 60)

# ---------------------------------------------------------------------------
# 1. Load images + labels + luminance axis
# ---------------------------------------------------------------------------
IMG_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")

images, labels, luminance = [], [], []
for ci, cls in enumerate(CLASSES):
    folder = os.path.join(ARGS.data_root, cls)
    files = sorted(f for f in os.listdir(folder) if f.endswith(IMG_EXTS))
    if ARGS.max_per_class:
        files = files[: ARGS.max_per_class]
    for fname in files:
        img = np.array(Image.open(os.path.join(folder, fname)).convert("RGB"))
        images.append(img)
        labels.append(ci)
        # Empirical darkness axis: mean Rec.601 luminance in [0, 255]
        lum = float((0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]).mean())
        luminance.append(lum)
    print(f"  {cls}: {len(files)} images")

labels = np.array(labels)
luminance = np.array(luminance)
print(f"Total: {len(images)} images")
print(f"Luminance: mean={luminance.mean():.1f}, median={np.median(luminance):.1f}, "
      f"[min={luminance.min():.1f}, max={luminance.max():.1f}]")


def clahe_enhance(img_bgr):
    """CLAHE on the L channel of LAB — standard low-light enhancement control."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[..., 0] = clahe.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


print("Computing CLAHE-enhanced copies...")
images_clahe = [clahe_enhance(img) for img in images]


# ---------------------------------------------------------------------------
# 2. Model(s)
# ---------------------------------------------------------------------------
model, n_blocks = load_dinov2(model_name=ARGS.model)
print(f"DINOv2 loaded, {n_blocks} blocks.")


@torch.no_grad()
def embed(model_, img_list, batch_size=32):
    feats = []
    for i in range(0, len(img_list), batch_size):
        batch = img_list[i:i + batch_size]
        tensors = torch.stack([DINOV2_PREPROCESS(img) for img in batch]).to(DEVICE)
        out = model_(tensors)
        if out.dim() == 3:
            out = out[:, 0, :]
        feats.append(out.cpu())
    return torch.cat(feats, dim=0).numpy()


def rebuild_with_adapters(ckpt):
    """Rebuild LoRA-wrapped backbone matching the training-time block_ranks."""
    import torch.nn as nn

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

    cfg = ckpt.get("config", {})
    block_ranks = {int(k): v for k, v in (cfg.get("block_ranks") or {}).items()}
    if not block_ranks:  # fallback: uniform late-3 assumption from base rank
        block_ranks = {b: ARGS.adapters_rank for b in range(n_blocks - 3, n_blocks)}
    alpha = int(cfg.get("alpha", ARGS.adapters_rank * 2))
    for b, block in enumerate(model.blocks):
        r = block_ranks.get(b)
        if r:
            block.attn.qkv = LoRALayer(block.attn.qkv, r=r, alpha=alpha)
            block.attn.proj = LoRALayer(block.attn.proj, r=r, alpha=alpha)
    missing, unexpected = model.load_state_dict(ckpt["adapter_state"], strict=False)
    n_loaded = sum(1 for k in ckpt["adapter_state"])
    print(f"  Rebuilt LoRA on {len(block_ranks)} blocks, loaded {n_loaded} adapter tensors "
          f"(missing={len(missing)}, unexpected={len(unexpected)})")
    return model


# ---------------------------------------------------------------------------
# 3. Probe protocol (matched split across passes via fixed seed)
# ---------------------------------------------------------------------------
def probe_protocol(feats, tag):
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        feats, labels, np.arange(len(labels)),
        test_size=0.3, random_state=ARGS.seed, stratify=labels,
    )
    probe = LogisticRegression(max_iter=2000, C=1.0)
    probe.fit(X_train, y_train)
    acc = probe.score(X_test, y_test)
    print(f"  [{tag}] 12-class probe accuracy: {acc:.4f}")
    return probe, acc, idx_test, y_test


results = {}

print("\n>>> Pass 1: raw real-dark embeddings")
raw_feats = embed(model, images)
probe_raw, acc_raw, idx_test, y_test = probe_protocol(raw_feats, "raw")
results["raw"] = acc_raw

print("\n>>> Pass 2: CLAHE-enhanced embeddings")
clahe_feats = embed(model, images_clahe)
probe_clahe, acc_clahe, _, _ = probe_protocol(clahe_feats, "clahe")
results["clahe"] = acc_clahe

# --- Darkness-response curve (quintiles of luminance, on the raw pass) ---
print("\n>>> Darkness-response curve (raw pass, luminance quintiles)")
test_lum = luminance[idx_test]
q = np.quantile(test_lum, [0.2, 0.4, 0.6, 0.8])
bins = np.digitize(test_lum, q)  # 0..4, darkest -> brightest
curve = []
preds_raw = probe_raw.predict(raw_feats[idx_test])
for b in range(5):
    mask = bins == b
    if mask.sum() == 0:
        continue
    acc_b = (preds_raw[mask] == y_test[mask]).mean()
    curve.append({
        "bin": b,
        "lum_mean": float(test_lum[mask].mean()),
        "n": int(mask.sum()),
        "accuracy": float(acc_b),
    })
    print(f"  bin {b} (lum~{test_lum[mask].mean():.1f}): acc={acc_b:.4f}  (n={mask.sum()})")

# --- Per-class accuracy (raw pass) ---
class_accs = []
for ci, cls in enumerate(CLASSES):
    mask = y_test == ci
    if mask.sum():
        class_accs.append({"class": cls, "accuracy": float((preds_raw[mask] == y_test[mask]).mean()),
                           "n": int(mask.sum())})

# ---------------------------------------------------------------------------
# 4. Optional: sim-to-real adapter transfer test
# ---------------------------------------------------------------------------
transfer_row = None
if ARGS.adapters and os.path.exists(ARGS.adapters):
    print(f"\n>>> Pass 3: LoRA adapter transfer ({ARGS.adapters})")
    ckpt = torch.load(ARGS.adapters, map_location="cpu", weights_only=False)
    rebuild_with_adapters(ckpt)
    model.eval().to(DEVICE)
    lora_feats = embed(model, images)
    # Freeze everything again for probing (probe trains on frozen features)
    for p_ in model.parameters():
        p_.requires_grad = False
    probe_lora, acc_lora, idx_test_l, y_test_l = probe_protocol(lora_feats, "lora-transfer")
    # same split seed => same test set as raw
    transfer_row = {
        "raw": acc_raw,
        "lora_transfer": acc_lora,
        "delta": acc_lora - acc_raw,
    }
    print(f"  Sim-to-real delta (LoRA vs raw): {acc_lora - acc_raw:+.4f}")
elif ARGS.adapters:
    print(f"\nWARNING: adapters file not found: {ARGS.adapters}")

# ---------------------------------------------------------------------------
# 5. Save artifacts
# ---------------------------------------------------------------------------
import csv as _csv

rows = [{"pass": k, "accuracy": v} for k, v in results.items()]
if transfer_row:
    rows.append({"pass": "lora_transfer", "accuracy": transfer_row["lora_transfer"]})
with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
    w = _csv.DictWriter(f, fieldnames=["pass", "accuracy"])
    w.writeheader()
    w.writerows(rows)
with open(os.path.join(OUT, "darkness_curve.csv"), "w", newline="") as f:
    w = _csv.DictWriter(f, fieldnames=["bin", "lum_mean", "n", "accuracy"])
    w.writeheader()
    w.writerows(curve)
with open(os.path.join(OUT, "class_accuracy.csv"), "w", newline="") as f:
    w = _csv.DictWriter(f, fieldnames=["class", "accuracy", "n"])
    w.writeheader()
    w.writerows(class_accs)
print(f"Saved results.csv / darkness_curve.csv / class_accuracy.csv in {OUT}")

# Plot 1: darkness-response curve
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot([c["lum_mean"] for c in curve], [c["accuracy"] for c in curve],
        marker="o", color="tab:blue")
ax.set_xlabel("Mean luminance of test bin (0-255, brighter = easier)")
ax.set_ylabel("12-class probe accuracy")
ax.set_title(f"DINOv2 on real dark images (ExDark, n={len(idx_test)})")
ax.set_ylim(0, 1)
fig.tight_layout()
save_fig(fig, os.path.join(OUT, "darkness_curve.png"))

# Plot 2: raw vs CLAHE (and optional transfer) bar chart
fig, ax = plt.subplots(figsize=(7, 5))
bars = [("Raw dark", results["raw"])]
bars.append(("CLAHE-enhanced", results["clahe"]))
if transfer_row:
    bars.append(("+ LoRA transfer\n(synthetic-dark trained)", transfer_row["lora_transfer"]))
names_ = [b[0] for b in bars]
accs_ = [b[1] for b in bars]
colors = ["tab:red", "tab:orange", "tab:green"][: len(bars)]
ax.bar(names_, accs_, color=colors)
for i, a in enumerate(accs_):
    ax.text(i, a + 0.01, f"{a:.3f}", ha="center")
ax.set_ylabel("12-class probe accuracy")
ax.set_title("Real-dark baseline and controls (ExDark)")
ax.set_ylim(0, 1)
fig.tight_layout()
save_fig(fig, os.path.join(OUT, "exdark_summary.png"))

# Summary
print("\n" + "=" * 60)
print("EXDARK BASELINE COMPLETE")
print("=" * 60)
print(f"  Real-dark 12-class accuracy: {results['raw']:.4f}")
print(f"  CLAHE-enhanced accuracy:     {results['clahe']:.4f}  "
      f"(enhancement effect {results['clahe'] - results['raw']:+.4f})")
if transfer_row:
    print(f"  LoRA transfer (synthetic->real): {transfer_row['lora_transfer']:.4f} "
          f"({transfer_row['delta']:+.4f} vs raw)")
print(f"  Darkness curve: {len(curve)} luminance bins, "
      f"darkest-bin acc {curve[0]['accuracy']:.3f} -> brightest-bin acc {curve[-1]['accuracy']:.3f}")
print(f"Output directory: {OUT}")
print("=" * 60)
