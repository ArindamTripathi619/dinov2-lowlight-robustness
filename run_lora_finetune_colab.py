"""
LoRA Fine-Tuning Experiment for DINOv2 Attention Layers
=========================================================
Based on CKA findings:
- Drift concentrated in late attention layers (blocks 10-11)
- DINOv2 leans heavily on low-frequency features
- Hypothesis: LoRA on attention layers can restore low-light robustness

Approach:
1. Add LoRA adapters to DINOv2's attention Q/K/V/O projections
2. Train on mixed clean + low-light CIFAR-10 (augmented dataset)
3. Remove LoRA head, extract embeddings, evaluate with linear probe
4. Compare against original DINOv2 baseline (from previous experiments)
"""

import sys
import os
import subprocess

os.makedirs("/content/output", exist_ok=True)
os.chdir("/content")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
_orig_show = plt.show
def _noop_show(*a, **kw): pass
plt.show = _noop_show

print("=" * 60)
print("LoRA Fine-Tuning Experiment for DINOv2")
print("=" * 60)

# ========== Cell 0: Install dependencies ==========
print("\n>>> Cell 0: Install dependencies")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "peft", "scikit-learn", "matplotlib", "scipy"], check=False)

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
from peft import LoraConfig, get_peft_model, TaskType

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ========== Cell 1: Load CIFAR-10 ==========
print("\n>>> Cell 1: Load CIFAR-10")
N_IMAGES = 1000

raw_ds = torchvision.datasets.CIFAR10(root="./data", train=False, download=True)
rng = np.random.default_rng(42)
idx = rng.choice(len(raw_ds), size=N_IMAGES, replace=False)

images, labels = [], []
for i in idx:
    img, label = raw_ds[i]
    images.append(np.array(img))
    labels.append(label)
labels = np.array(labels)
print(f"Loaded {len(images)} images")

# Also load training set for LoRA fine-tuning
train_ds = torchvision.datasets.CIFAR10(root="./data", train=True, download=True)
N_TRAIN = 5000  # subset of training data
train_idx = rng.choice(len(train_ds), size=N_TRAIN, replace=False)
train_images, train_labels = [], []
for i in train_idx:
    img, label = train_ds[i]
    train_images.append(np.array(img))
    train_labels.append(label)
train_labels = np.array(train_labels)
print(f"Loaded {len(train_images)} training images for LoRA fine-tuning")

# ========== Cell 2: Low-light function ==========
print("\n>>> Cell 2: Define corruption functions")

def low_light(image, severity):
    brightness_factors = [1.0, 0.75, 0.55, 0.38, 0.25, 0.15]
    noise_std          = [0,    2,    4,    6,    9,    13]
    factor, noise = brightness_factors[severity], noise_std[severity]
    img = image.astype(np.float32) * factor
    if noise > 0:
        img = img + np.random.normal(0, noise, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)

# ========== Cell 3: Load DINOv2 ==========
print("\n>>> Cell 3: Load DINOv2 ViT-S/14")
dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
dinov2.eval().to(device)

dinov2_preprocess = T.Compose([
    T.ToTensor(),
    T.Resize((224, 224), antialias=True),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

print("DINOv2 loaded.")
print(f"  Architecture: {dinov2.__class__.__name__}")
print(f"  Embedding dim: {dinov2.embed_dim}")
print(f"  Num blocks: {len(dinov2.blocks)}")
print(f"  Num heads: {dinov2.blocks[0].attn.num_heads}")

# ========== Cell 4: Apply LoRA to attention layers ==========
print("\n>>> Cell 4: Apply LoRA adapters to attention layers")

# LoRA config targeting attention projections
# CKA showed blocks 10-11 drift most, but we apply to all blocks
# and let the adapters learn which ones matter
lora_config = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION,
    r=8,                    # LoRA rank - low-rank bottleneck
    lora_alpha=16,          # scaling factor (2x rank is standard)
    lora_dropout=0.1,
    target_modules=[
        "attn.qkv",        # Q, K, V combined projection
        "attn.proj",        # Output projection
    ],
    bias="none",
)

dinov2_lora = get_peft_model(dinov2, lora_config)
dinov2_lora.print_trainable_parameters()
dinov2_lora.train()

# Verify LoRA was applied correctly
print("\nLoRA applied modules:")
for name, param in dinov2_lora.named_parameters():
    if "lora" in name:
        print(f"  {name}: {param.shape} (trainable)")

# Count trainable vs total
total_params = sum(p.numel() for p in dinov2_lora.parameters())
trainable_params = sum(p.numel() for p in dinov2_lora.parameters() if p.requires_grad)
print(f"\nTotal params: {total_params:,}")
print(f"Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")

# ========== Cell 5: Create augmented training dataset ==========
print("\n>>> Cell 5: Create augmented training dataset (clean + low-light)")

class AugmentedCIFAR10(Dataset):
    """CIFAR-10 with random low-light augmentation during training."""
    def __init__(self, images, labels, transform=None, augment=True):
        self.images = images
        self.labels = labels
        self.transform = transform
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].copy()
        label = self.labels[idx]

        if self.augment:
            # Randomly apply low-light degradation (severity 2-4)
            if np.random.random() < 0.7:  # 70% chance of degradation
                severity = np.random.randint(2, 5)
                img = low_light(img, severity)

            # Random horizontal flip
            if np.random.random() < 0.5:
                img = np.flip(img, axis=1).copy()

        if self.transform:
            img = self.transform(img)

        return img, label

train_dataset = AugmentedCIFAR10(
    train_images, train_labels,
    transform=dinov2_preprocess,
    augment=True
)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
print(f"Training dataset: {len(train_dataset)} images (70% with low-light augmentation)")

# ========== Cell 6: Train LoRA with classification head ==========
print("\n>>> Cell 6: Train LoRA adapters")

# Add a classification head on top of the LoRA-adapted backbone
class DINOv2Classifier(nn.Module):
    def __init__(self, backbone, num_classes=10):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.config.hidden_size, num_classes)

    def forward(self, x):
        # Get CLS token from backbone
        out = self.backbone(x)
        if hasattr(out, 'last_hidden_state'):
            cls_token = out.last_hidden_state[:, 0, :]
        else:
            cls_token = out[:, 0, :] if out.dim() == 3 else out
        return self.head(cls_token)

classifier = DINOv2Classifier(dinov2_lora).to(device)

# Freeze everything except LoRA params + head
for param in classifier.backbone.parameters():
    param.requires_grad = False
for param in classifier.head.parameters():
    param.requires_grad = True
# Unfreeze LoRA params
for name, param in classifier.backbone.named_parameters():
    if "lora" in name:
        param.requires_grad = True

# Optimizer with different LR for backbone (LoRA) vs head
# Disjoint param groups: head must not appear in both groups (AdamW would raise
# "some parameters appear in more than one parameter group"). Note: PEFT wraps the
# model, so LoRA param names contain "lora" regardless of nesting depth.
head_params = list(classifier.head.parameters())
head_ids = {id(p) for p in head_params}
lora_params = [p for n, p in classifier.named_parameters()
               if "lora" in n and p.requires_grad and id(p) not in head_ids]

optimizer = torch.optim.AdamW([
    {"params": lora_params, "lr": 5e-5, "weight_decay": 0.01},
    {"params": head_params, "lr": 1e-3, "weight_decay": 0.01},
])

criterion = nn.CrossEntropyLoss()
num_epochs = 10

print(f"Training for {num_epochs} epochs...")
train_losses = []
train_accs = []

for epoch in range(num_epochs):
    classifier.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_imgs, batch_labels in train_loader:
        batch_imgs = batch_imgs.to(device)
        batch_labels = batch_labels.to(device)

        optimizer.zero_grad()
        outputs = classifier(batch_imgs)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_imgs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(batch_labels).sum().item()
        total += batch_imgs.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    train_losses.append(avg_loss)
    train_accs.append(accuracy)
    print(f"  Epoch {epoch+1}/{num_epochs}: loss={avg_loss:.4f}, acc={accuracy:.4f}")

print(f"Training complete. Final accuracy: {train_accs[-1]:.4f}")

# Plot training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(train_losses, marker="o")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_title("Training Loss")
ax2.plot(train_accs, marker="o", color="tab:green")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.set_title("Training Accuracy")
plt.tight_layout()
plt.savefig("/content/output/lora_training_curves.png", dpi=150)
print("Saved lora_training_curves.png")

# ========== Cell 7: Remove classifier head, use LoRA backbone for embeddings ==========
print("\n>>> Cell 7: Extract embeddings with LoRA-adapted DINOv2")

# Get the LoRA-adapted backbone
dinov2_lora_backbone = classifier.backbone
dinov2_lora_backbone.eval()

# Hook for layer-wise activations
_dinov2_lora_layer_outputs = {}
def _make_hook_lora(layer_idx):
    def hook(module, inp, out):
        _dinov2_lora_layer_outputs[layer_idx] = out[:, 0, :].detach().cpu()
    return hook

for i, block in enumerate(dinov2_lora_backbone.blocks):
    block.register_forward_hook(_make_hook_lora(i))

@torch.no_grad()
def get_lora_embeddings(image_list, batch_size=64, collect_layers=False):
    pooled = []
    layerwise = {i: [] for i in range(len(dinov2_lora_backbone.blocks))} if collect_layers else None
    for i in range(0, len(image_list), batch_size):
        batch = image_list[i:i+batch_size]
        tensors = torch.stack([dinov2_preprocess(img) for img in batch]).to(device)
        out = dinov2_lora_backbone(tensors)
        if hasattr(out, 'last_hidden_state'):
            pooled.append(out.last_hidden_state[:, 0, :].cpu())
        else:
            pooled.append(out.cpu())
        if collect_layers:
            for l in range(len(dinov2_lora_backbone.blocks)):
                layerwise[l].append(_dinov2_lora_layer_outputs[l])
    pooled = torch.cat(pooled, dim=0).numpy()
    if collect_layers:
        layerwise = {l: torch.cat(v, dim=0).numpy() for l, v in layerwise.items()}
        return pooled, layerwise
    return pooled

# Extract embeddings at all severity levels
lora_pooled = {}
lora_layerwise = {}

for severity in range(6):
    degraded = [low_light(img, severity) for img in images]
    pooled, layerwise = get_lora_embeddings(degraded, collect_layers=True)
    lora_pooled[severity] = pooled
    lora_layerwise[severity] = layerwise
    print(f"severity {severity}: embeddings {pooled.shape}")

# ========== Cell 8: Linear probe evaluation (same protocol) ==========
print("\n>>> Cell 8: Linear probe evaluation")

X_clean_lora = lora_pooled[0]
X_train_l, X_test_l, y_train_l, y_test_l, idx_train_l, idx_test_l = train_test_split(
    X_clean_lora, labels, np.arange(len(labels)), test_size=0.3, random_state=42, stratify=labels
)

probe_lora = LogisticRegression(max_iter=2000, C=1.0)
probe_lora.fit(X_train_l, y_train_l)

lora_accs = []
for severity in range(6):
    X_sev = lora_pooled[severity][idx_test_l]
    acc = accuracy_score(y_test_l, probe_lora.predict(X_sev))
    lora_accs.append(acc)
    print(f"severity {severity}: LoRA accuracy = {acc:.3f}")

# ========== Cell 9: Layer-wise CKA for LoRA model ==========
print("\n>>> Cell 9: Layer-wise CKA for LoRA model")

N_LORA_LAYERS = len(dinov2_lora_backbone.blocks)

def linear_cka(X, Y):
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    hsic = np.linalg.norm(X.T @ Y, 'fro') ** 2
    var1 = np.linalg.norm(X.T @ X, 'fro')
    var2 = np.linalg.norm(Y.T @ Y, 'fro')
    return hsic / (np.sqrt(var1 * var2) + 1e-8)

lora_cka_matrix = np.zeros((N_LORA_LAYERS, 6))
for layer in range(N_LORA_LAYERS):
    clean_acts = lora_layerwise[0][layer]
    for severity in range(6):
        degraded_acts = lora_layerwise[severity][layer]
        lora_cka_matrix[layer, severity] = linear_cka(clean_acts, degraded_acts)

# ========== Cell 10: Compare with original DINOv2 ==========
print("\n>>> Cell 10: Compare LoRA vs Original DINOv2")

# Load original DINOv2 for comparison
dinov2_orig = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
dinov2_orig.eval().to(device)

@torch.no_grad()
def get_orig_embeddings(image_list, batch_size=64):
    feats = []
    for i in range(0, len(image_list), batch_size):
        batch = image_list[i:i+batch_size]
        tensors = torch.stack([dinov2_preprocess(img) for img in batch]).to(device)
        out = dinov2_orig(tensors)
        feats.append(out.cpu())
    return torch.cat(feats, dim=0).numpy()

orig_pooled = {}
for severity in range(6):
    degraded = [low_light(img, severity) for img in images]
    orig_pooled[severity] = get_orig_embeddings(degraded)

X_clean_orig = orig_pooled[0]
X_train_o, X_test_o, y_train_o, y_test_o, idx_train_o, idx_test_o = train_test_split(
    X_clean_orig, labels, np.arange(len(labels)), test_size=0.3, random_state=42, stratify=labels
)

probe_orig = LogisticRegression(max_iter=2000, C=1.0)
probe_orig.fit(X_train_o, y_train_o)

orig_accs = []
for severity in range(6):
    X_sev = orig_pooled[severity][idx_test_o]
    acc = accuracy_score(y_test_o, probe_orig.predict(X_sev))
    orig_accs.append(acc)

# ========== Cell 11: Generate comparison plots ==========
print("\n>>> Cell 11: Generate comparison plots")

# Plot 1: Accuracy comparison
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(range(6), orig_accs, marker="o", label="Original DINOv2", color="tab:blue")
ax.plot(range(6), lora_accs, marker="s", label="LoRA-adapted DINOv2", color="tab:green")
ax.set_xlabel("Low-light severity")
ax.set_ylabel("Linear probe accuracy")
ax.set_title("DINOv2 Original vs LoRA-adapted: Low-Light Robustness")
ax.set_ylim(0, 1)
ax.legend()
plt.tight_layout()
plt.savefig("/content/output/lora_vs_orig_accuracy.png", dpi=150)
print("Saved lora_vs_orig_accuracy.png")

# Plot 2: CKA heatmap for LoRA model
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Original CKA (recomputed)
orig_block_outputs = {}
def _make_hook_orig(layer_idx):
    def hook(module, inp, out):
        orig_block_outputs[layer_idx] = out[:, 0, :].detach().cpu()
    return hook

for i, block in enumerate(dinov2_orig.blocks):
    block.register_forward_hook(_make_hook_orig(i))

orig_layerwise = {}
for severity in range(6):
    degraded = [low_light(img, severity) for img in images]
    layerwise_s = {}
    for i in range(0, len(degraded), 64):
        batch = degraded[i:i+64]
        tensors = torch.stack([dinov2_preprocess(img) for img in batch]).to(device)
        with torch.no_grad():
            _ = dinov2_orig(tensors)
        for l in range(len(dinov2_orig.blocks)):
            if l not in layerwise_s:
                layerwise_s[l] = []
            layerwise_s[l].append(orig_block_outputs[l])
    orig_layerwise[severity] = {l: torch.cat(v, dim=0).numpy() for l, v in layerwise_s.items()}

N_ORIG_LAYERS = len(dinov2_orig.blocks)
orig_cka_matrix = np.zeros((N_ORIG_LAYERS, 6))
for layer in range(N_ORIG_LAYERS):
    clean_acts = orig_layerwise[0][layer]
    for severity in range(6):
        degraded_acts = orig_layerwise[severity][layer]
        orig_cka_matrix[layer, severity] = linear_cka(clean_acts, degraded_acts)

im1 = ax1.imshow(orig_cka_matrix, aspect="auto", cmap="viridis", vmin=0, vmax=1)
plt.colorbar(im1, ax=ax1, label="CKA similarity")
ax1.set_xlabel("Severity")
ax1.set_ylabel("Transformer block")
ax1.set_title("Original DINOv2 CKA")

im2 = ax2.imshow(lora_cka_matrix, aspect="auto", cmap="viridis", vmin=0, vmax=1)
plt.colorbar(im2, ax=ax2, label="CKA similarity")
ax2.set_xlabel("Severity")
ax2.set_ylabel("Transformer block")
ax2.set_title("LoRA-adapted DINOv2 CKA")

plt.tight_layout()
plt.savefig("/content/output/cka_comparison.png", dpi=150)
print("Saved cka_comparison.png")

# Plot 3: CKA drift reduction
orig_drift = orig_cka_matrix[:, 0] - orig_cka_matrix[:, 5]
lora_drift = lora_cka_matrix[:, 0] - lora_cka_matrix[:, 5]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(range(N_ORIG_LAYERS), orig_drift, marker="o", label="Original DINOv2", color="tab:blue")
ax.plot(range(N_LORA_LAYERS), lora_drift, marker="s", label="LoRA-adapted DINOv2", color="tab:green")
ax.set_xlabel("Transformer block")
ax.set_ylabel("CKA drop (clean → darkest)")
ax.set_title("CKA Drift Reduction from LoRA Fine-Tuning")
ax.legend()
plt.tight_layout()
plt.savefig("/content/output/cka_drift_reduction.png", dpi=150)
print("Saved cka_drift_reduction.png")

# ========== Summary ==========
print("\n" + "=" * 60)
print("LORA EXPERIMENT COMPLETE - SUMMARY")
print("=" * 60)

print(f"\nLoRA config: r=8, alpha=16, targets=[attn.qkv, attn.proj]")
print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
print(f"Training: {num_epochs} epochs, {N_TRAIN} images (70% with low-light augmentation)")

print("\n--- Accuracy Comparison ---")
print(f"{'Severity':<10} {'Original':>10} {'LoRA':>10} {'Delta':>10}")
print("-" * 42)
for i in range(6):
    delta = lora_accs[i] - orig_accs[i]
    print(f"{i:<10} {orig_accs[i]:>10.4f} {lora_accs[i]:>10.4f} {delta:>+10.4f}")

print(f"\n--- Overall Improvement ---")
orig_mean = np.mean(orig_accs)
lora_mean = np.mean(lora_accs)
print(f"Mean accuracy (all severities): Original={orig_mean:.4f}, LoRA={lora_mean:.4f}, Delta={lora_mean-orig_mean:+.4f}")
print(f"Accuracy at sev 5 (darkest): Original={orig_accs[5]:.4f}, LoRA={lora_accs[5]:.4f}, Delta={lora_accs[5]-orig_accs[5]:+.4f}")

print(f"\n--- CKA Drift ---")
print(f"Original mean drift: {orig_drift.mean():.4f}")
print(f"LoRA mean drift:     {lora_drift.mean():.4f}")
print(f"Drift reduction:     {orig_drift.mean() - lora_drift.mean():+.4f}")

# Which layers improved most?
drift_reduction = orig_drift - lora_drift
best_layer = np.argmax(drift_reduction)
print(f"Most improved layer: block {best_layer} (reduced by {drift_reduction[best_layer]:.4f})")

print("\n--- Verdict ---")
if lora_mean > orig_mean:
    print(f"✅ LoRA improved mean accuracy by {lora_mean-orig_mean:+.4f}")
else:
    print(f"❌ LoRA did not improve mean accuracy (delta: {lora_mean-orig_mean:+.4f})")

if lora_accs[5] > orig_accs[5]:
    print(f"✅ LoRA improved worst-case (sev 5) by {lora_accs[5]-orig_accs[5]:+.4f}")
else:
    print(f"❌ LoRA did not improve worst-case (sev 5) (delta: {lora_accs[5]-orig_accs[5]:+.4f})")

if lora_drift.mean() < orig_drift.mean():
    print(f"✅ LoRA reduced representational drift")
else:
    print(f"❌ LoRA did not reduce representational drift")

print("=" * 60)
