"""
Simplified LoRA Experiment for DINOv2 Low-Light Robustness
==========================================================
Manual LoRA implementation (no PEFT dependency) to avoid compatibility issues.
Based on CKA findings: drift concentrated in late attention layers (blocks 10-11).

Phase 0 refactor: fully parameterized for the Phase 3 v2 experiment grid.
Defaults reproduce the published run of record exactly (r=8, a=16, mix=0.7,
all blocks, seed 42, dinov2_vits14, low_light, 10 epochs, /content/output).

Examples (Colab):
  python3 run_lora_simple_colab.py --seed 43 --outdir /content/output/seed43
  python3 run_lora_simple_colab.py --layers late --rank 8            # blocks 9-11 only
  python3 run_lora_simple_colab.py --layers drift --rank 8           # CKA-proportional ranks
  python3 run_lora_simple_colab.py --layers fullft                   # full fine-tune baseline
  python3 run_lora_simple_colab.py --rank 16 --mix 0.5 --epochs 10
"""

import sys, os, subprocess, argparse


def parse_args():
    p = argparse.ArgumentParser(description="Phase 3 v2: parameterized LoRA robustness fine-tuning")
    p.add_argument("--rank", type=int, default=8, help="base LoRA rank (drift mode scales it per block)")
    p.add_argument("--alpha", type=int, default=16, help="LoRA alpha (scaling = alpha/rank)")
    p.add_argument("--mix", type=float, default=0.7, help="probability of corruption augmentation per training image")
    p.add_argument("--layers", default="all", choices=["all", "late", "drift", "fullft"],
                   help="where to adapt: all blocks | late blocks 9-11 | CKA-drift-weighted ranks | full fine-tune")
    p.add_argument("--seed", type=int, default=42, help="seed for data draw + torch init/shuffling")
    p.add_argument("--model", default="dinov2_vits14", help="torch.hub DINOv2 entry")
    p.add_argument("--corruption", default="low_light", choices=["low_light", "blur", "jpeg", "contrast"])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--outdir", default=None,
                   help="where plots/CSV/adapters are written (default: /content/output on Colab, ./output_lora elsewhere)")
    p.add_argument("--drift-csv", default=None,
                   help="optional CSV (layer,severity,cka,drop_from_clean) with measured drift; "
                        "default uses the embedded ViT-S/low-light profile")
    return p.parse_args()


ARGS = parse_args()
if ARGS.outdir is None:
    ARGS.outdir = "/content/output" if os.path.isdir("/content") else "./output_lora"
os.makedirs(ARGS.outdir, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

print("=" * 60)
print("LoRA Fine-Tuning Experiment (parameterized)")
print(f"  rank={ARGS.rank} alpha={ARGS.alpha} mix={ARGS.mix} layers={ARGS.layers}")
print(f"  seed={ARGS.seed} model={ARGS.model} corruption={ARGS.corruption} epochs={ARGS.epochs}")
print("=" * 60)

# Install deps
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "scikit-learn", "matplotlib"], check=False)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Seed torch BEFORE any model creation so the LoRA-A init, head init, dropout,
# and DataLoader shuffling are all deterministic functions of --seed. PyTorch
# seeds its global RNG from entropy at process start; seeding after model init
# (a previous version's bug) would leave initialization unseeded.
torch.manual_seed(ARGS.seed)

# === Load CIFAR-10 ===
print("\n>>> Loading CIFAR-10")
raw_ds = torchvision.datasets.CIFAR10(root="./data", train=False, download=True)
rng = np.random.default_rng(ARGS.seed)
idx = rng.choice(len(raw_ds), size=1000, replace=False)
images, labels = [], []
for i in idx:
    img, label = raw_ds[i]
    images.append(np.array(img))
    labels.append(label)
labels = np.array(labels)

# Training subset
train_ds = torchvision.datasets.CIFAR10(root="./data", train=True, download=True)
train_idx = rng.choice(len(train_ds), size=5000, replace=False)
train_images, train_labels = [], []
for i in train_idx:
    img, label = train_ds[i]
    train_images.append(np.array(img))
    train_labels.append(label)
train_labels = np.array(train_labels)
print(f"Test: {len(images)}, Train: {len(train_images)}")

# === Corruption function (dispatches to utils-style implementations) ===
def make_corruption_fn(name):
    """Return a local copy of the corruption fn (self-contained script)."""
    if name == "low_light":
        def fn(image, severity, rng=None):
            brightness_factors = [1.0, 0.75, 0.55, 0.38, 0.25, 0.15]
            noise_std = [0, 2, 4, 6, 9, 13]
            img = image.astype(np.float32) * brightness_factors[severity]
            if noise_std[severity] > 0:
                noise = (rng if rng is not None else np.random).normal(
                    0, noise_std[severity], img.shape
                )
                img += noise
            return np.clip(img, 0, 255).astype(np.uint8)
        return fn
    if name == "blur":
        def fn(image, severity, rng=None):
            from PIL import Image, ImageFilter
            radii = [0.0, 0.5, 1.0, 2.0, 3.5, 5.5]
            if radii[severity] == 0:
                return image.copy()
            return np.array(Image.fromarray(image).filter(
                ImageFilter.GaussianBlur(radius=radii[severity])))
        return fn
    if name == "jpeg":
        def fn(image, severity, rng=None):
            if severity == 0:
                return image.copy()  # JPEG is lossy even at q100; sev0 = pristine
            import io
            from PIL import Image
            qualities = [100, 60, 40, 25, 15, 8]
            buf = io.BytesIO()
            Image.fromarray(image).save(buf, format="JPEG", quality=qualities[severity])
            buf.seek(0)
            return np.array(Image.open(buf).convert("RGB"))
        return fn
    if name == "contrast":
        def fn(image, severity, rng=None):
            factors = [1.0, 0.8, 0.6, 0.4, 0.25, 0.15]
            img = image.astype(np.float32)
            gray = img.mean(axis=2, keepdims=True)
            img = factors[severity] * img + (1.0 - factors[severity]) * gray
            return np.clip(img, 0, 255).astype(np.uint8)
        return fn
    raise ValueError(f"unknown corruption {name!r}")


corrupt_fn = make_corruption_fn(ARGS.corruption)

# === Load DINOv2 ===
print("\n>>> Loading DINOv2")
dinov2 = torch.hub.load('facebookresearch/dinov2', ARGS.model)
dinov2.eval().to(device)

preprocess = T.Compose([
    T.ToTensor(), T.Resize((224, 224), antialias=True),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# === Manual LoRA implementation ===
class LoRALayer(nn.Module):
    """Low-Rank Adaptation for a linear layer."""
    def __init__(self, original_layer, r=8, alpha=16):
        super().__init__()
        self.original = original_layer
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False

        in_features = original_layer.in_features
        out_features = original_layer.out_features

        self.lora_A = nn.Parameter(torch.randn(r, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        self.scaling = alpha / r

    def forward(self, x):
        original_out = self.original(x)
        lora_out = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return original_out + lora_out

print("\n>>> Applying adaptation to attention layers")

# ---- Layer targeting & rank allocation -------------------------------------
# Embedded measured drift (clean->sev5 CKA drop per block) for ViT-S/14 under
# low_light, from output/notebook2/cka_matrix.csv (artifact of record).
# Used by --layers drift when no --drift-csv is supplied.
DEFAULT_DRIFT = [0.578863, 0.610093, 0.443532, 0.465539, 0.528656,
                 0.525862, 0.577079, 0.679069, 0.762785, 0.767158,
                 0.814622, 0.782831]


def load_drift_profile(n_blocks):
    """Per-block CKA drop profile (higher = drifted more)."""
    if ARGS.drift_csv and os.path.exists(ARGS.drift_csv):
        import csv as _csv
        drops = {}
        with open(ARGS.drift_csv) as f:
            for row in _csv.DictReader(f):
                if int(row["severity"]) == 5:
                    drops[int(row["layer"])] = float(row["drop_from_clean"])
        if len(drops) == n_blocks:
            print(f"  Drift profile from {ARGS.drift_csv}")
            return [drops[i] for i in range(n_blocks)]
        print(f"  WARNING: {ARGS.drift_csv} covers {len(drops)}/{n_blocks} blocks; using embedded default")
    elif ARGS.drift_csv:
        print(f"  WARNING: {ARGS.drift_csv} not found; using embedded default")
    return DEFAULT_DRIFT[:n_blocks]


# Decide (block_index -> rank or None) and full-FT mode
FULL_FT = ARGS.layers == "fullft"
block_ranks = {}
if ARGS.layers == "all":
    block_ranks = {b: ARGS.rank for b in range(len(dinov2.blocks))}
elif ARGS.layers == "late":
    block_ranks = {b: ARGS.rank for b in range(len(dinov2.blocks)) if b >= len(dinov2.blocks) - 3}
elif ARGS.layers == "drift":
    drift = load_drift_profile(len(dinov2.blocks))
    dmax = max(drift)
    for b, d in enumerate(drift):
        scaled = int(round(ARGS.rank * d / dmax))
        if scaled >= 1:
            block_ranks[b] = max(1, scaled)
    print("  Drift-weighted ranks:", block_ranks)
elif not FULL_FT:
    raise ValueError(f"unknown --layers {ARGS.layers!r}")

lora_count = 0
for b, block in enumerate(dinov2.blocks):
    r = block_ranks.get(b)
    if r:
        block.attn.qkv = LoRALayer(block.attn.qkv, r=r, alpha=ARGS.alpha)
        block.attn.proj = LoRALayer(block.attn.proj, r=r, alpha=ARGS.alpha)
        lora_count += 2

if FULL_FT:
    print("  FULL FINE-TUNING mode: unfreezing entire backbone + head")

# Freeze everything (LoRA params are unfrozen below; full-FT unfreezes all)
for param in dinov2.parameters():
    param.requires_grad = False

# Unfreeze LoRA params
for module in dinov2.modules():
    if isinstance(module, LoRALayer):
        module.lora_A.requires_grad = True
        module.lora_B.requires_grad = True

if FULL_FT:
    for param in dinov2.parameters():
        param.requires_grad = True

total_params = sum(p.numel() for p in dinov2.parameters())
trainable_params = sum(p.numel() for p in dinov2.parameters() if p.requires_grad)
print(f"Total params: {total_params:,}")
mode_label = "full fine-tune" if FULL_FT else "LoRA/adapters"
print(f"Trainable ({mode_label}): {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
print(f"LoRA modules: {lora_count} across {len(block_ranks)} targeted block(s)")

# === Training dataset ===
class AugmentedCIFAR10(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].copy()
        label = self.labels[idx]
        # Per-sample RNG (seeded by index) instead of np.random global state:
        # DataLoader workers fork without re-seeding NumPy, so a global-RNG-based
        # augmentation would produce correlated streams across workers.
        rng = np.random.default_rng(seed=idx)
        if rng.random() < ARGS.mix:
            severity = int(rng.integers(2, 5))
            img = corrupt_fn(img, severity, rng=rng)
        if rng.random() < 0.5:
            img = np.flip(img, axis=1).copy()
        if self.transform:
            img = self.transform(img)
        return img, label

train_dataset = AugmentedCIFAR10(train_images, train_labels, transform=preprocess)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2,
                          generator=torch.Generator().manual_seed(ARGS.seed))
print(f"Training: {len(train_dataset)} images ({ARGS.mix:.0%} corruption aug, {ARGS.corruption})")

# === Classification head ===
class DINOv2Classifier(nn.Module):
    def __init__(self, backbone, num_classes=10):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.embed_dim, num_classes)

    def forward(self, x):
        out = self.backbone(x)
        if out.dim() == 3:
            cls_token = out[:, 0, :]
        else:
            cls_token = out
        return self.head(cls_token)

classifier = DINOv2Classifier(dinov2).to(device)

# Optimizer — learning rates by adaptation mode:
#   LoRA arms: adapters 5e-5, head 1e-3 (published protocol).
#   fullft: backbone 1e-5 (standard fine-tune LR; 5e-5 would wreck pretrained
#           features), head 1e-3.
# Params collected disjointly (head first, then remaining trainables) so AdamW
# never sees a parameter in two groups.
head_params = list(classifier.head.parameters())
head_ids = {id(p) for p in head_params}
lora_params = [p for p in classifier.parameters() if p.requires_grad and id(p) not in head_ids]
backbone_lr = 1e-5 if FULL_FT else 5e-5
optimizer = torch.optim.AdamW([
    {"params": lora_params, "lr": backbone_lr, "weight_decay": 0.01},
    {"params": head_params, "lr": 1e-3, "weight_decay": 0.01},
])
if FULL_FT:
    print(f"  full-FT optimizer: backbone lr={backbone_lr}, head lr=1e-3")
criterion = nn.CrossEntropyLoss()

# === Train ===
print("\n>>> Training")
num_epochs = ARGS.epochs
train_losses, train_accs = [], []

for epoch in range(num_epochs):
    classifier.train()
    total_loss, correct, total = 0, 0, 0
    for batch_imgs, batch_labels in train_loader:
        batch_imgs, batch_labels = batch_imgs.to(device), batch_labels.to(device)
        optimizer.zero_grad()
        outputs = classifier(batch_imgs)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch_imgs.size(0)
        correct += outputs.argmax(1).eq(batch_labels).sum().item()
        total += batch_imgs.size(0)
    avg_loss = total_loss / total
    acc = correct / total
    train_losses.append(avg_loss)
    train_accs.append(acc)
    print(f"  Epoch {epoch+1}/{num_epochs}: loss={avg_loss:.4f}, acc={acc:.4f}")

# Plot training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(train_losses, marker="o"); ax1.set_title("Training Loss"); ax1.set_xlabel("Epoch")
ax2.plot(train_accs, marker="o", color="green"); ax2.set_title("Training Accuracy"); ax2.set_xlabel("Epoch")
plt.tight_layout()
plt.savefig(os.path.join(ARGS.outdir, "lora_training_curves.png"), dpi=150)
print("Saved lora_training_curves.png")

# === Precompute degraded test sets ONCE (matched-noise evaluation) ===
# The corruption noise is stochastic; drawing it separately for the LoRA and
# original passes would give each slightly different test images. Seeding one
# RNG per severity and reusing the arrays makes the comparison image-matched.
eval_rng = {s: np.random.default_rng(1000 + s) for s in range(6)}
degraded_eval = {
    s: [corrupt_fn(img, s, rng=eval_rng[s]) for img in images] for s in range(6)
}

# === Extract embeddings with LoRA model ===
print("\n>>> Extracting embeddings with LoRA model")
classifier.eval()

@torch.no_grad()
def get_embeddings(model, image_list, batch_size=64):
    feats = []
    for i in range(0, len(image_list), batch_size):
        batch = image_list[i:i+batch_size]
        tensors = torch.stack([preprocess(img) for img in batch]).to(device)
        out = model.backbone(tensors)
        if out.dim() == 3:
            feats.append(out[:, 0, :].cpu())
        else:
            feats.append(out.cpu())
    return torch.cat(feats, dim=0).numpy()

lora_pooled = {}
for severity in range(6):
    lora_pooled[severity] = get_embeddings(classifier, degraded_eval[severity])
    print(f"  severity {severity}: {lora_pooled[severity].shape}")

# === Linear probe ===
print("\n>>> Linear probe evaluation")
X_clean = lora_pooled[0]
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X_clean, labels, np.arange(len(labels)), test_size=0.3, random_state=42, stratify=labels
)

probe = LogisticRegression(max_iter=2000, C=1.0)
probe.fit(X_train, y_train)

lora_accs = []
for severity in range(6):
    acc = accuracy_score(y_test, probe.predict(lora_pooled[severity][idx_test]))
    lora_accs.append(acc)
    print(f"  severity {severity}: {acc:.3f}")

# === Compare with original DINOv2 ===
print("\n>>> Loading original DINOv2 for comparison")
dinov2_orig = torch.hub.load('facebookresearch/dinov2', ARGS.model)
dinov2_orig.eval().to(device)

@torch.no_grad()
def get_orig_embeddings(image_list, batch_size=64):
    feats = []
    for i in range(0, len(image_list), batch_size):
        batch = image_list[i:i+batch_size]
        tensors = torch.stack([preprocess(img) for img in batch]).to(device)
        out = dinov2_orig(tensors)
        feats.append(out.cpu())
    return torch.cat(feats, dim=0).numpy()

orig_pooled = {}
for severity in range(6):
    orig_pooled[severity] = get_orig_embeddings(degraded_eval[severity])

X_clean_orig = orig_pooled[0]
X_train_o, X_test_o, y_train_o, y_test_o, _, idx_test_o = train_test_split(
    X_clean_orig, labels, np.arange(len(labels)), test_size=0.3, random_state=42, stratify=labels
)
probe_orig = LogisticRegression(max_iter=2000, C=1.0)
probe_orig.fit(X_train_o, y_train_o)

orig_accs = []
for severity in range(6):
    acc = accuracy_score(y_test_o, probe_orig.predict(orig_pooled[severity][idx_test_o]))
    orig_accs.append(acc)

# === Generate comparison plots ===
print("\n>>> Generating comparison plots")

# Plot 1: Accuracy comparison
fig, ax = plt.subplots(figsize=(8, 5))
adapted_label = "Full fine-tune" if FULL_FT else "LoRA-adapted DINOv2"
ax.plot(range(6), orig_accs, marker="o", label="Original DINOv2", color="tab:blue")
ax.plot(range(6), lora_accs, marker="s", label=adapted_label, color="tab:green")
ax.set_xlabel("Low-light severity"); ax.set_ylabel("Accuracy")
ax.set_title(f"DINOv2 Original vs {adapted_label}: {ARGS.corruption} Robustness")
ax.set_ylim(0, 1); ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(ARGS.outdir, "lora_vs_orig_accuracy.png"), dpi=150)
print("Saved lora_vs_orig_accuracy.png")

# === Summary ===
print("\n" + "=" * 60)
print("LORA EXPERIMENT COMPLETE")
print("=" * 60)

print(f"\nConfig: corruption={ARGS.corruption}, model={ARGS.model}, layers={ARGS.layers}")
print(f"  rank={ARGS.rank}, alpha={ARGS.alpha}, per-block ranks={block_ranks or 'full-backbone (fullft)'}")
print(f"  mix={ARGS.mix}, seed={ARGS.seed}, epochs={num_epochs}")
print(f"Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
print(f"Training: {num_epochs} epochs, {len(train_images)} images ({ARGS.mix:.0%} {ARGS.corruption} aug)")

print(f"\n{'Severity':<10} {'Original':>10} {'LoRA':>10} {'Delta':>10}")
print("-" * 42)
for i in range(6):
    delta = lora_accs[i] - orig_accs[i]
    print(f"{i:<10} {orig_accs[i]:>10.4f} {lora_accs[i]:>10.4f} {delta:>+10.4f}")

orig_mean, lora_mean = np.mean(orig_accs), np.mean(lora_accs)
print(f"\nMean accuracy: Original={orig_mean:.4f}, LoRA={lora_mean:.4f}, Delta={lora_mean-orig_mean:+.4f}")
print(f"Sev 5 (darkest): Original={orig_accs[5]:.4f}, LoRA={lora_accs[5]:.4f}, Delta={lora_accs[5]-orig_accs[5]:+.4f}")

print("\n--- Verdict ---")
if lora_mean > orig_mean:
    print(f"✅ LoRA improved mean accuracy by {lora_mean-orig_mean:+.4f}")
else:
    print(f"❌ LoRA did not improve mean accuracy (delta: {lora_mean-orig_mean:+.4f})")
if lora_accs[5] > orig_accs[5]:
    print(f"✅ LoRA improved worst-case (sev 5) by {lora_accs[5]-orig_accs[5]:+.4f}")
else:
    print(f"❌ LoRA did not improve worst-case (delta: {lora_accs[5]-orig_accs[5]:+.4f})")
print("=" * 60)
