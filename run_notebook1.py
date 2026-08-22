!pip install -q scikit-learn scikit-image matplotlib

import torch
import torchvision
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


# --- cell separator ---

N_IMAGES = 1000  # subset size, increase later if you want tighter statistics

raw_ds = torchvision.datasets.CIFAR10(root="./data", train=False, download=True)

# Balanced-ish random subset
rng = np.random.default_rng(42)
idx = rng.choice(len(raw_ds), size=N_IMAGES, replace=False)

images = []   # list of HxWx3 uint8 numpy arrays, original CIFAR-10 resolution (32x32)
labels = []
for i in idx:
    img, label = raw_ds[i]
    images.append(np.array(img))  # PIL -> numpy, shape (32,32,3), uint8
    labels.append(label)

labels = np.array(labels)
print(f"Loaded {len(images)} images, classes present: {np.unique(labels)}")


# --- cell separator ---

def low_light(image, severity):
    """Darken an image stepwise and add shot-noise, simulating low light.
    severity: 0 (no change) through 5 (very dark, noisy).
    """
    brightness_factors = [1.0, 0.75, 0.55, 0.38, 0.25, 0.15]
    noise_std          = [0,    2,    4,    6,    9,    13]

    factor = brightness_factors[severity]
    noise  = noise_std[severity]

    img = image.astype(np.float32) * factor
    if noise > 0:
        img = img + np.random.normal(0, noise, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

# quick visual sanity check
fig, axes = plt.subplots(1, 6, figsize=(15, 3))
sample_img = images[0]
for s in range(6):
    axes[s].imshow(low_light(sample_img, s))
    axes[s].set_title(f"severity {s}")
    axes[s].axis("off")
plt.suptitle(f"Original class: {raw_ds.classes[labels[0]]}")
plt.tight_layout()
plt.show()


# --- cell separator ---

dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
dinov2.eval().to(device)

# DINOv2 expects inputs whose H and W are multiples of the patch size (14).
# CIFAR-10 is 32x32, so we upsample to 224x224 (standard practice for small-image
# datasets when using ViT backbones pretrained at higher resolution).
preprocess = T.Compose([
    T.ToTensor(),
    T.Resize((224, 224), antialias=True),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@torch.no_grad()
def get_embeddings(image_list, batch_size=64):
    """image_list: list of HxWx3 uint8 numpy arrays -> (N, D) embedding matrix"""
    feats = []
    for i in range(0, len(image_list), batch_size):
        batch = image_list[i:i+batch_size]
        tensors = torch.stack([preprocess(img) for img in batch]).to(device)
        out = dinov2(tensors)  # DINOv2's forward already returns the CLS token pooled feature
        feats.append(out.cpu())
    return torch.cat(feats, dim=0).numpy()

print("DINOv2 ViT-S/14 loaded.")


# --- cell separator ---

embeddings_by_severity = {}

for severity in range(6):
    degraded_images = [low_light(img, severity) for img in images]
    embeddings_by_severity[severity] = get_embeddings(degraded_images)
    print(f"severity {severity}: embeddings shape {embeddings_by_severity[severity].shape}")


# --- cell separator ---

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


# --- cell separator ---

results = []

for severity in range(6):
    X_sev_test = embeddings_by_severity[severity][idx_test]
    preds = probe.predict(X_sev_test)
    acc = accuracy_score(y_test, preds)
    results.append({"severity": severity, "accuracy": acc})
    print(f"severity {severity}: accuracy = {acc:.3f}")


# --- cell separator ---

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


# --- cell separator ---

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
plt.savefig("dinov2_lowlight_results.png", dpi=150)
plt.show()


# --- cell separator ---

import csv

with open("dinov2_lowlight_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["severity", "accuracy", "mean_cosine_sim_to_clean"])
    writer.writeheader()
    for r in results:
        writer.writerow(r)

print("Saved dinov2_lowlight_results.csv and dinov2_lowlight_results.png")


# --- cell separator ---

import skimage.color as skcolor

def imagenet_c_brightness(image, severity=1):
    c = [.05, .1, .15, .2, .3][severity - 1]
    x = np.array(image) / 255.
    x = skcolor.rgb2hsv(x)
    x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
    x = skcolor.hsv2rgb(x)
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)

fig, axes = plt.subplots(1, 5, figsize=(13, 3))
for s in range(1, 6):
    axes[s-1].imshow(imagenet_c_brightness(sample_img, s))
    axes[s-1].set_title(f"severity {s}")
    axes[s-1].axis("off")
plt.suptitle("Standard ImageNet-C/CIFAR-10-C brightness corruption (brightens, not darkens)")
plt.tight_layout()
plt.show()
