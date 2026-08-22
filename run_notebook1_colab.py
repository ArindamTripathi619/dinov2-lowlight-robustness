import sys
import os
import subprocess

os.makedirs("/content/output", exist_ok=True)
os.chdir("/content")

# Headless matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
_orig_show = plt.show
def _noop_show(*a, **kw): pass
plt.show = _noop_show

print("=" * 60)
print("Starting Notebook 1: DINOv2 CIFAR-10 Low-Light Robustness")
print("=" * 60)

# ========== Cell 0: Setup ==========
print("\n>>> Cell 0: Setup")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "scikit-learn", "matplotlib"], check=False)

import torch
import torchvision
import torchvision.transforms as T
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ========== Cell 1: Load CIFAR-10 ==========
print("\n>>> Cell 1: Load CIFAR-10 subset")
N_IMAGES = 1000

raw_ds = torchvision.datasets.CIFAR10(root="./data", train=False, download=True)

rng = np.random.default_rng(42)
idx = rng.choice(len(raw_ds), size=N_IMAGES, replace=False)

images = []
labels = []
for i in idx:
    img, label = raw_ds[i]
    images.append(np.array(img))
    labels.append(label)

labels = np.array(labels)
print(f"Loaded {len(images)} images, classes present: {np.unique(labels)}")

# ========== Cell 2: Low-light function ==========
print("\n>>> Cell 2: Low-light degradation function")

def low_light(image, severity):
    brightness_factors = [1.0, 0.75, 0.55, 0.38, 0.25, 0.15]
    noise_std          = [0,    2,    4,    6,    9,    13]
    factor = brightness_factors[severity]
    noise  = noise_std[severity]
    img = image.astype(np.float32) * factor
    if noise > 0:
        img = img + np.random.normal(0, noise, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

# Visual sanity check
fig, axes = plt.subplots(1, 6, figsize=(15, 3))
sample_img = images[0]
for s in range(6):
    axes[s].imshow(low_light(sample_img, s))
    axes[s].set_title(f"severity {s}")
    axes[s].axis("off")
plt.suptitle(f"Original class: {raw_ds.classes[labels[0]]}")
plt.tight_layout()
plt.savefig("/content/output/severity_visual_check.png", dpi=150)
print("Saved severity_visual_check.png")

# ========== Cell 3: Load DINOv2 ==========
print("\n>>> Cell 3: Load DINOv2 ViT-S/14")
dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
dinov2.eval().to(device)

preprocess = T.Compose([
    T.ToTensor(),
    T.Resize((224, 224), antialias=True),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@torch.no_grad()
def get_embeddings(image_list, batch_size=64):
    feats = []
    for i in range(0, len(image_list), batch_size):
        batch = image_list[i:i+batch_size]
        tensors = torch.stack([preprocess(img) for img in batch]).to(device)
        out = dinov2(tensors)
        feats.append(out.cpu())
    return torch.cat(feats, dim=0).numpy()

print("DINOv2 ViT-S/14 loaded.")

# ========== Cell 4: Extract embeddings ==========
print("\n>>> Cell 4: Extract embeddings at every severity")
embeddings_by_severity = {}

for severity in range(6):
    degraded_images = [low_light(img, severity) for img in images]
    embeddings_by_severity[severity] = get_embeddings(degraded_images)
    print(f"severity {severity}: embeddings shape {embeddings_by_severity[severity].shape}")

# ========== Cell 5: Train linear probe ==========
print("\n>>> Cell 5: Train linear probe on CLEAN embeddings")
from sklearn.model_selection import train_test_split

X_clean = embeddings_by_severity[0]
y = labels

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X_clean, y, np.arange(len(y)), test_size=0.3, random_state=42, stratify=y
)

probe = LogisticRegression(max_iter=2000, C=1.0)
probe.fit(X_train, y_train)

clean_test_acc = accuracy_score(y_test, probe.predict(X_test))
print(f"Linear probe accuracy on clean, held-out test images: {clean_test_acc:.3f}")

# ========== Cell 6: Evaluate across severities ==========
print("\n>>> Cell 6: Evaluate probe across all severities")
results = []

for severity in range(6):
    X_sev_test = embeddings_by_severity[severity][idx_test]
    preds = probe.predict(X_sev_test)
    acc = accuracy_score(y_test, preds)
    results.append({"severity": severity, "accuracy": acc})
    print(f"severity {severity}: accuracy = {acc:.3f}")

# ========== Cell 7: Embedding drift ==========
print("\n>>> Cell 7: Embedding drift (cosine similarity)")
def cosine_sim(a, b):
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return (a * b).sum(axis=1)

for r in results:
    sev = r["severity"]
    sims = cosine_sim(embeddings_by_severity[0], embeddings_by_severity[sev])
    r["mean_cosine_sim_to_clean"] = float(sims.mean())

for r in results:
    print(r)

# ========== Cell 8: Plot ==========
print("\n>>> Cell 8: Plot accuracy and embedding drift")
severities = [r["severity"] for r in results]
accs = [r["accuracy"] for r in results]
sims = [r["mean_cosine_sim_to_clean"] for r in results]

fig, ax1 = plt.subplots(figsize=(8, 5))

ax1.set_xlabel("Low-light severity (0 = bright, 5 = darkest)")
ax1.set_ylabel("Linear probe accuracy", color="tab:blue")
ax1.plot(severities, accs, marker="o", color="tab:blue", label="Accuracy")
ax1.tick_params(axis="y", labelcolor="tab:blue")
ax1.set_ylim(0, 1)

ax2 = ax1.twinx()
ax2.set_ylabel("Mean cosine similarity to clean embedding", color="tab:red")
ax2.plot(severities, sims, marker="s", color="tab:red", linestyle="--", label="Embedding similarity")
ax2.tick_params(axis="y", labelcolor="tab:red")

plt.title("DINOv2 (ViT-S/14) robustness to stepwise low-light degradation, CIFAR-10")
fig.tight_layout()
plt.savefig("/content/output/dinov2_lowlight_results.png", dpi=150)
print("Saved dinov2_lowlight_results.png")

# ========== Cell 9: Save CSV ==========
print("\n>>> Cell 9: Save results to CSV")
import csv

with open("/content/output/dinov2_lowlight_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["severity", "accuracy", "mean_cosine_sim_to_clean"])
    writer.writeheader()
    for r in results:
        writer.writerow(r)

print("Saved dinov2_lowlight_results.csv")

# ========== Summary ==========
print("\n" + "=" * 60)
print("NOTEBOOK 1 COMPLETE - SUMMARY")
print("=" * 60)
for r in results:
    print(f"  Severity {r['severity']}: accuracy={r['accuracy']:.4f}, cosine_sim={r['mean_cosine_sim_to_clean']:.4f}")
print(f"\nClean test accuracy: {clean_test_acc:.4f}")
print(f"Accuracy drop (sev 0 -> 5): {results[0]['accuracy'] - results[5]['accuracy']:.4f}")
print("=" * 60)
