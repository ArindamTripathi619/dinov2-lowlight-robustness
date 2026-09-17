"""
Simplified LoRA Experiment for DINOv2 Low-Light Robustness
==========================================================
Manual LoRA implementation (no PEFT dependency) to avoid compatibility issues.
Based on CKA findings: drift concentrated in late attention layers (blocks 10-11).
"""

import sys, os, subprocess
os.makedirs("/content/output", exist_ok=True)
os.chdir("/content")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

print("=" * 60)
print("LoRA Fine-Tuning Experiment (Simplified)")
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

# === Load CIFAR-10 ===
print("\n>>> Loading CIFAR-10")
raw_ds = torchvision.datasets.CIFAR10(root="./data", train=False, download=True)
rng = np.random.default_rng(42)
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

# === Low-light function ===
def low_light(image, severity):
    brightness_factors = [1.0, 0.75, 0.55, 0.38, 0.25, 0.15]
    noise_std = [0, 2, 4, 6, 9, 13]
    img = image.astype(np.float32) * brightness_factors[severity]
    if noise_std[severity] > 0:
        img += np.random.normal(0, noise_std[severity], img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)

# === Load DINOv2 ===
print("\n>>> Loading DINOv2")
dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
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

print("\n>>> Applying LoRA to attention layers")
# Apply LoRA to all transformer blocks' attention layers
lora_rank = 8
lora_alpha = 16

lora_count = 0
for block in dinov2.blocks:
    # LoRA on QKV projection
    block.attn.qkv = LoRALayer(block.attn.qkv, r=lora_rank, alpha=lora_alpha)
    # LoRA on output projection
    block.attn.proj = LoRALayer(block.attn.proj, r=lora_rank, alpha=lora_alpha)
    lora_count += 2

# Freeze everything except LoRA params
for param in dinov2.parameters():
    param.requires_grad = False
for param in dinov2.parameters():
    if hasattr(param, 'requires_grad'):
        pass  # Already frozen

# Unfreeze only LoRA params
for module in dinov2.modules():
    if isinstance(module, LoRALayer):
        module.lora_A.requires_grad = True
        module.lora_B.requires_grad = True

total_params = sum(p.numel() for p in dinov2.parameters())
trainable_params = sum(p.numel() for p in dinov2.parameters() if p.requires_grad)
print(f"Total params: {total_params:,}")
print(f"Trainable (LoRA): {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
print(f"LoRA modules: {lora_count} (2 per block x {len(dinov2.blocks)} blocks)")

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
        if rng.random() < 0.7:
            severity = int(rng.integers(2, 5))
            img = low_light(img, severity)
        if rng.random() < 0.5:
            img = np.flip(img, axis=1).copy()
        if self.transform:
            img = self.transform(img)
        return img, label

train_dataset = AugmentedCIFAR10(train_images, train_labels, transform=preprocess)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
print(f"Training: {len(train_dataset)} images (70% with low-light aug)")

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

# Optimizer
# Collect params disjointly: head first, then every remaining trainable (LoRA) param.
# (The head would otherwise appear twice — once via classifier.parameters() and once
# via classifier.head.parameters() — which makes AdamW raise "some parameters appear
# in more than one parameter group".)
head_params = list(classifier.head.parameters())
head_ids = {id(p) for p in head_params}
lora_params = [p for p in classifier.parameters() if p.requires_grad and id(p) not in head_ids]
optimizer = torch.optim.AdamW([
    {"params": lora_params, "lr": 5e-5, "weight_decay": 0.01},
    {"params": head_params, "lr": 1e-3, "weight_decay": 0.01},
])
criterion = nn.CrossEntropyLoss()

# === Train ===
print("\n>>> Training LoRA adapters")
num_epochs = 10
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
plt.savefig("/content/output/lora_training_curves.png", dpi=150)
print("Saved lora_training_curves.png")

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
    degraded = [low_light(img, severity) for img in images]
    lora_pooled[severity] = get_embeddings(classifier, degraded)
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
dinov2_orig = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
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
    degraded = [low_light(img, severity) for img in images]
    orig_pooled[severity] = get_orig_embeddings(degraded)

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
ax.plot(range(6), orig_accs, marker="o", label="Original DINOv2", color="tab:blue")
ax.plot(range(6), lora_accs, marker="s", label="LoRA-adapted DINOv2", color="tab:green")
ax.set_xlabel("Low-light severity"); ax.set_ylabel("Accuracy")
ax.set_title("DINOv2 Original vs LoRA: Low-Light Robustness")
ax.set_ylim(0, 1); ax.legend()
plt.tight_layout()
plt.savefig("/content/output/lora_vs_orig_accuracy.png", dpi=150)
print("Saved lora_vs_orig_accuracy.png")

# === Summary ===
print("\n" + "=" * 60)
print("LORA EXPERIMENT COMPLETE")
print("=" * 60)

print(f"\nLoRA config: r={lora_rank}, alpha={lora_alpha}, targets=[attn.qkv, attn.proj]")
print(f"Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
print(f"Training: {num_epochs} epochs, {len(train_images)} images (70% low-light aug)")

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
