# Notebook 3 — Mechanistic Ablation + Restoration Analysis
# Builds directly on run_notebook2.py (CKA / frequency findings) and
# run_lora_simple_colab.py (LoRA adaptation). Each cell separated by "# --- cell separator ---".
#
# New questions this notebook answers that the earlier notebooks did not:
#   Q1  Mechanism: is the collapse driven by representational failure or probe failure?
#   Q2  Geometry: does low light shrink effective dimensionality of the embedding space?
#   Q3  Saturation: how much of the damage is due to uint8 quantization (information destroyed)
#        vs recoverable contrast loss? Implemented as a two-stage clipping model.
#   Q4  Zero-shot vs probed accuracy gap.
#   Q5  Ablation: LoRA rank {4, 8, 16} x target-module sets {attn-only, attn+mlp} -> Pareto front.
#   Q6  Classical mitigation baselines: gamma correction and CLAHE, no training at all.
#
# NOTE (output divergence, debt A4): unlike run_notebook3_colab.py / the .ipynb,
# this plain runner does NOT emit output/quantization_results.csv,
# output/geometry_results.csv or output/zero_shot_results.csv (it writes a
# reduced detailed_geometry table plus its own geometry_results summary
# instead). Downstream consumers must not assume the two runners produce the
# same output set.

# --- cell separator ---

# Cell 0: Setup
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "scikit-learn", "scikit-image", "matplotlib", "scipy"], check=False)

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import os, json, csv

os.makedirs("output", exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# --- cell separator ---

# Cell 1: Data — same subsets and seed as notebooks 1/2 so results are directly comparable
N_IMAGES = 1000
N_TRAIN = 5000

raw_ds = torchvision.datasets.CIFAR10(root="./data", train=False, download=True)
rng = np.random.default_rng(42)
idx = rng.choice(len(raw_ds), size=N_IMAGES, replace=False)
images, labels = [], []
for i in idx:
    img, label = raw_ds[i]
    images.append(np.array(img))
    labels.append(label)
labels = np.array(labels)

train_ds = torchvision.datasets.CIFAR10(root="./data", train=True, download=True)
train_idx = rng.choice(len(train_ds), size=N_TRAIN, replace=False)
train_images, train_labels = [], []
for i in train_idx:
    img, label = train_ds[i]
    train_images.append(np.array(img))
    train_labels.append(label)
train_labels = np.array(train_labels)
print(f"Test subset: {len(images)}  Train subset: {len(train_images)}")

# --- cell separator ---

# Cell 2: Corruption functions (identical to notebooks 1/2, plus the two-stage clip model)
def low_light(image, severity):
    brightness_factors = [1.0, 0.75, 0.55, 0.38, 0.25, 0.15]
    noise_std          = [0,    2,    4,    6,    9,    13]
    img = image.astype(np.float32) * brightness_factors[severity]
    if noise_std[severity] > 0:
        img = img + np.random.normal(0, noise_std[severity], img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)

def apply_low_light_to_stage1(image, severity):
    """Stage 1 only: photometric loss WITHOUT the uint8 clip.
    Returns float32 in [0, 255]. Used to disentangle contrast loss from quantization loss."""
    brightness_factors = [1.0, 0.75, 0.55, 0.38, 0.25, 0.15]
    noise_std          = [0,    2,    4,    6,    9,    13]
    img = image.astype(np.float32) * brightness_factors[severity]
    if noise_std[severity] > 0:
        img = img + np.random.normal(0, noise_std[severity], img.shape)
    return np.clip(img, 0, 255)  # float32, no uint8 re-quantization

def two_stage_low_light(image, severity):
    """Explicit two-stage model of the sensor pipeline:
       stage 1: I' = a(sev)*I + eta, real-valued  ->  stage 2: Q = uint8(I')  (irreversible)
    """
    return apply_low_light_to_stage1(image, severity).astype(np.uint8)

# --- cell separator ---

# Cell 3: Classical mitigation baselines (zero training)
import skimage.color
import skimage.exposure as skexposure

def gamma_correct(image, severity, gamma=2.2):
    """Perceptual (gamma-space) brightening: raise the dark image to power 1/gamma.
    Unlike the linear gain, this amplifies shadows more than highlights, which is what
    classic low-light enhancement does. severity sets the strength: exponent = 1 + 0.35*sev."""
    exponent = 1.0 + 0.35 * max(severity, 1)
    x = np.asarray(image).astype(np.float32) / 255.0
    out = np.power(x, 1.0 / exponent) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)

def simple_gain(image, severity):
    """Restore original scale: multiply by 1/a and clip. Recovers contrast, not noise."""
    factors = [1.0, 0.75, 0.55, 0.38, 0.25, 0.15]
    a = factors[severity]
    img = np.asarray(image).astype(np.float32) / a
    return np.clip(img, 0, 255).astype(np.uint8)

def clahe_enhance(image, severity=None):
    """CLAHE on the L channel of LAB — standard low-light enhancement baseline."""
    img = np.asarray(image).astype(np.float32) / 255.0
    lab = skimage.color.rgb2lab(img)
    lab[:, :, 0] = skexposure.equalize_adapthist(lab[:, :, 0] / 100.0, clip_limit=0.03) * 100.0
    return (np.clip(skimage.color.lab2rgb(lab), 0, 1) * 255).astype(np.uint8)

# --- cell separator ---

# Cell 4: Load models and preprocessing
dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
dinov2.eval().to(device)

preprocess = T.Compose([
    T.ToTensor(),
    T.Resize((224, 224), antialias=True),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@torch.no_grad()
def get_embeddings(image_list, batch_size=64, float_input=False):
    """float_input=True for float32 HWC arrays in [0,255] (stage-1 arm): T.ToTensor only
    rescales uint8 input, so we build the normalized tensor manually for floats."""
    feats = []
    for i in range(0, len(image_list), batch_size):
        batch = image_list[i:i + batch_size]
        if float_input:
            tensors = torch.stack([
                T.functional.normalize(
                    T.functional.resize(
                        torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1) / 255.0,
                        (224, 224), antialias=True),
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                for img in batch
            ]).to(device)
        else:
            tensors = torch.stack([preprocess(img) for img in batch]).to(device)
        out = dinov2(tensors)
        feats.append(out.cpu())
    return torch.cat(feats, dim=0).numpy()

print("DINOv2 ViT-S/14 loaded.")

# --- cell separator ---

# Cell 5: Q1 — Mechanism: representational failure vs probe failure
# Logit margin = top1 logit minus top2 logit of the linear probe.
# If margins shrink smoothly with severity but embeddings stay informative
# (probe retrained per severity does much better), the failure is in the probe,
# not the representation. If retrained probes also collapse, representation itself degrades.

def fit_probe(X, y):
    probe = LogisticRegression(max_iter=2000, C=1.0)
    probe.fit(X, y)
    return probe

def logit_margins(probe, X, y):
    probs = probe.decision_function(X)          # (n, 10)
    srt = np.sort(probs, axis=1)
    top1_minus_top2 = srt[:, -1] - srt[:, -2]
    correct = (probe.predict(X) == y).astype(float)
    return correct.mean(), top1_minus_top2.mean(), top1_minus_top2[probe.predict(X) == y].mean()

embeddings_by_severity = {}
for severity in range(6):
    degraded = [low_light(img, severity) for img in images]
    embeddings_by_severity[severity] = get_embeddings(degraded)
    print(f"severity {severity}: extracted")

X_clean = embeddings_by_severity[0]
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X_clean, labels, np.arange(len(labels)), test_size=0.3, random_state=42, stratify=labels
)

fixed_probe = fit_probe(X_train, y_train)

mechanism_rows = []
for severity in range(6):
    acc_fixed, margin_all, margin_correct = logit_margins(fixed_probe, embeddings_by_severity[severity][idx_test], y_test)
    # probe retrained on THIS severity's train split
    X_tr_s = embeddings_by_severity[severity][idx_train]
    probe_s = fit_probe(X_tr_s, y_train)
    acc_adapted = accuracy_score(y_test, probe_s.predict(embeddings_by_severity[severity][idx_test]))
    mechanism_rows.append({
        "severity": severity,
        "acc_fixed_probe": acc_fixed,
        "acc_probe_retrained_on_severity": acc_adapted,
        "mean_logit_margin": margin_all,
        "mean_logit_margin_correct": margin_correct,
    })
    print(mechanism_rows[-1])

# --- cell separator ---

# Cell 6: Q2 — Embedding geometry: participation ratio (effective dimensionality)
def participation_ratio(X):
    S = np.cov(X.T)
    eig = np.linalg.eigvalsh(S)[::-1]
    return float((eig.sum() ** 2) / ((eig ** 2).sum() + 1e-12))

pr_by_severity = []
for severity in range(6):
    pr_clean = participation_ratio(embeddings_by_severity[0])
    pr_dark = participation_ratio(embeddings_by_severity[severity])
    pr_by_severity.append({"severity": severity, "pr_clean": pr_clean, "pr_dark": pr_dark, "ratio": pr_dark / pr_clean})
    print(pr_by_severity[-1])

# --- cell separator ---

# Cell 7: Q3 — Quantization vs contrast: how much damage is irreversible?
# Arm A: full two-stage corruption (contrast loss + uint8 clip)  -> the standard pipeline
# Arm B: stage-1 only, kept float (contrast loss only)           -> no information destroyed
# If acc(B) >> acc(A) at high severity, quantization is a first-class cause, not a rounding detail.

quant_rows = []
for severity in range(6):
    float_imgs = [apply_low_light_to_stage1(img, severity) for img in images]
    uint8_imgs = [two_stage_low_light(img, severity) for img in images]
    E_float = get_embeddings(float_imgs, float_input=True)
    E_uint8 = get_embeddings(uint8_imgs)
    acc_float = accuracy_score(y_test, fixed_probe.predict(E_float[idx_test]))
    acc_uint8 = accuracy_score(y_test, fixed_probe.predict(E_uint8[idx_test]))
    quant_rows.append({"severity": severity, "acc_float_stage1": acc_float, "acc_uint8_full": acc_uint8})
    print(quant_rows[-1])

# --- cell separator ---

# Cell 8: Q4 — Zero-shot head vs linear probe
# DINOv2 ships with a frozen ImageNet head; on 224-upsampled CIFAR-10 crops we can also
# measure probe-free "nearest clean-prototype" accuracy in embedding space.
protos = np.stack([X_train[y_train == c].mean(0) for c in range(10)])
def proto_accuracy(E):
    sims = E @ protos.T / (np.linalg.norm(E, axis=1, keepdims=True) * np.linalg.norm(protos, axis=1) + 1e-8)
    return accuracy_score(y_test, sims.argmax(1))

zero_shot_rows = []
for severity in range(6):
    E = embeddings_by_severity[severity]
    zero_shot_rows.append({
        "severity": severity,
        "acc_probe": accuracy_score(y_test, fixed_probe.predict(E[idx_test])),
        "acc_nearest_prototype": proto_accuracy(E[idx_test]),
    })
    print(zero_shot_rows[-1])

# --- cell separator ---

# Cell 9: Q5 — LoRA ablation: rank x target modules -> Pareto front
class LoRALayer(nn.Module):
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
        return self.original(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scaling

def build_model(rank, alpha, target):
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    model.eval().to(device)
    for block in model.blocks:
        if target in ("attn", "attn_mlp"):
            block.attn.qkv = LoRALayer(block.attn.qkv, r=rank, alpha=alpha)
            block.attn.proj = LoRALayer(block.attn.proj, r=rank, alpha=alpha)
        if target == "attn_mlp":
            block.mlp.fc1 = LoRALayer(block.mlp.fc1, r=rank, alpha=alpha)
            block.mlp.fc2 = LoRALayer(block.mlp.fc2, r=rank, alpha=alpha)
    for p in model.parameters():
        p.requires_grad = False
    for m in model.modules():
        if isinstance(m, LoRALayer):
            m.lora_A.requires_grad = True
            m.lora_B.requires_grad = True
    return model

class AugmentedCIFAR10(Dataset):
    def __init__(self, imgs, lbls, transform):
        self.images, self.labels, self.transform = imgs, lbls, transform
    def __len__(self):
        return len(self.images)
    def __getitem__(self, i):
        img = self.images[i].copy()
        if np.random.random() < 0.7:
            img = low_light(img, np.random.randint(2, 5))
        if np.random.random() < 0.5:
            img = np.flip(img, axis=1).copy()
        return self.transform(img), self.labels[i]

class Classifier(nn.Module):
    def __init__(self, backbone, n_classes=10):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.embed_dim, n_classes)
    def forward(self, x):
        out = self.backbone(x)
        return self.head(out[:, 0, :] if out.dim() == 3 else out)

def train_lora(rank, target, epochs=6):
    torch.manual_seed(42)
    model = build_model(rank, alpha=2 * rank, target=target)
    clf = Classifier(model).to(device)
    lora_params = [p for n, p in clf.named_parameters()
                   if p.requires_grad and not n.startswith("head.")]
    head_params = list(clf.head.parameters())
    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": 5e-5, "weight_decay": 0.01},
        {"params": head_params, "lr": 1e-3, "weight_decay": 0.01},
    ])
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(AugmentedCIFAR10(train_images, train_labels, preprocess),
                        batch_size=64, shuffle=True, num_workers=2)
    clf.train()
    for epoch in range(epochs):
        for bi, (bi_img, bi_lbl) in enumerate(loader):
            bi_img, bi_lbl = bi_img.to(device), bi_lbl.to(device)
            optimizer.zero_grad()
            loss = criterion(clf(bi_img), bi_lbl)
            loss.backward()
            optimizer.step()
    clf.eval()
    return clf

@torch.no_grad()
def extract_backbone(clf, image_list, batch_size=64):
    feats = []
    for i in range(0, len(image_list), batch_size):
        batch = image_list[i:i + batch_size]
        tensors = torch.stack([preprocess(img) for img in batch]).to(device)
        out = clf.backbone(tensors)
        feats.append((out[:, 0, :] if out.dim() == 3 else out).cpu())
    return torch.cat(feats, dim=0).numpy()

ABLATIONS = [
    (4, "attn"),
    (8, "attn"),
    (16, "attn"),
    (8, "attn_mlp"),
    (16, "attn_mlp"),
]

ablation_rows = []
for rank, target in ABLATIONS:
    print(f"\n=== LoRA r={rank} target={target} ===")
    clf = train_lora(rank, target)
    pooled = {s: extract_backbone(clf, [low_light(im, s) for im in images]) for s in range(6)}
    Xc = pooled[0]
    Xtr, Xte, ytr, yte, _, ite = train_test_split(Xc, labels, np.arange(len(labels)),
                                                  test_size=0.3, random_state=42, stratify=labels)
    probe = fit_probe(Xtr, ytr)
    accs = [accuracy_score(yte, probe.predict(pooled[s][ite])) for s in range(6)]
    n_trainable = sum(p.numel() for p in clf.parameters() if p.requires_grad)
    ablation_rows.append({
        "rank": rank, "target": target, "trainable_params": int(n_trainable),
        "acc_sev0": accs[0], "acc_sev3": accs[3], "acc_sev5": accs[5],
        "mean_acc": float(np.mean(accs)),
    })
    print(ablation_rows[-1])
    del clf
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

# --- cell separator ---

# Cell 10: Q6 — Classical mitigation vs LoRA (and vs doing nothing)
# Baseline accuracies from the un-adapted model:
baseline_accs = [accuracy_score(y_test, fixed_probe.predict(embeddings_by_severity[s][idx_test])) for s in range(6)]

mitigation_rows = []
for severity in range(6):
    row = {"severity": severity, "no_mitigation": baseline_accs[severity]}
    for name, fn in [("gain", simple_gain), ("gamma", gamma_correct), ("clahe", clahe_enhance)]:
        restored = [fn(low_light(img, severity), severity) if name != "clahe" else fn(low_light(img, severity)) for img in images]
        E = get_embeddings(restored)
        row[f"acc_{name}"] = accuracy_score(y_test, fixed_probe.predict(E[idx_test]))
    mitigation_rows.append(row)
    print(row)

# --- cell separator ---

# Cell 11: Save everything + summary plots
with open("output/notebook3_results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["severity", "acc_fixed_probe", "acc_probe_retrained_on_severity",
                                      "mean_logit_margin", "mean_logit_margin_correct"])
    w.writeheader()
    for r in mechanism_rows:
        w.writerow(r)

with open("output/ablation_results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["rank", "target", "trainable_params", "acc_sev0", "acc_sev3", "acc_sev5", "mean_acc"])
    w.writeheader()
    for r in ablation_rows:
        w.writerow(r)

with open("output/mitigation_results.csv", "w", newline="") as f:
    if mitigation_rows:
        w = csv.DictWriter(f, fieldnames=list(mitigation_rows[0].keys()))
        w.writeheader()
        for r in mitigation_rows:
            w.writerow(r)

# Plot: mechanism
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot([r["severity"] for r in mechanism_rows], [r["acc_fixed_probe"] for r in mechanism_rows], marker="o", label="Fixed probe (trained on clean)")
ax.plot([r["severity"] for r in mechanism_rows], [r["acc_probe_retrained_on_severity"] for r in mechanism_rows], marker="s", label="Probe retrained on same severity")
ax.set_xlabel("Low-light severity"); ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1)
ax.set_title("Q1: is the failure in the representation or the probe?")
ax.legend(); plt.tight_layout(); plt.savefig("output/mechanism.png", dpi=150); plt.close()

# Plot: geometry
prs = [r["ratio"] for r in pr_by_severity]
plt.figure(figsize=(8, 5))
plt.plot(range(6), prs, marker="o", color="tab:purple")
plt.xlabel("Low-light severity"); plt.ylabel("Participation ratio (dark / clean)")
plt.title("Q2: embedding effective dimensionality under low light")
plt.tight_layout(); plt.savefig("output/geometry.png", dpi=150); plt.close()

# Plot: quantization gap
plt.figure(figsize=(8, 5))
plt.plot([r["severity"] for r in quant_rows], [r["acc_float_stage1"] for r in quant_rows], marker="o", label="Stage-1 only (float, recoverable)")
plt.plot([r["severity"] for r in quant_rows], [r["acc_uint8_full"] for r in quant_rows], marker="s", label="Full two-stage (uint8 clip, irreversible)")
plt.xlabel("Low-light severity"); plt.ylabel("Accuracy"); plt.ylim(0, 1)
plt.title("Q3: how much damage does uint8 quantization cause?")
plt.legend(); plt.tight_layout(); plt.savefig("output/quantization.png", dpi=150); plt.close()

# Plot: ablation Pareto
plt.figure(figsize=(8, 5))
for (rank, target) in ABLATIONS:
    row = next(r for r in ablation_rows if r["rank"] == rank and r["target"] == target)
    plt.scatter(row["trainable_params"], row["mean_acc"], s=90)
    plt.annotate(f"r={rank}\n{target}", (row["trainable_params"], row["mean_acc"]), fontsize=8,
                 textcoords="offset points", xytext=(6, 4))
plt.xscale("log"); plt.xlabel("Trainable parameters (log)"); plt.ylabel("Mean accuracy over severities 0–5")
plt.title("Q5: LoRA rank x target Pareto front")
plt.tight_layout(); plt.savefig("output/ablation_pareto.png", dpi=150); plt.close()

# Plot: mitigation
plt.figure(figsize=(8, 5))
plt.plot(range(6), [r["no_mitigation"] for r in mitigation_rows], marker="o", label="No mitigation")
plt.plot(range(6), [r["acc_gain"] for r in mitigation_rows], marker="s", label="Gain x(1/a)")
plt.plot(range(6), [r["acc_gamma"] for r in mitigation_rows], marker="^", label="Gamma correction")
plt.plot(range(6), [r["acc_clahe"] for r in mitigation_rows], marker="d", label="CLAHE")
plt.xlabel("Low-light severity"); plt.ylabel("Accuracy"); plt.ylim(0, 1)
plt.title("Q6: zero-training classical mitigation")
plt.legend(); plt.tight_layout(); plt.savefig("output/mitigation.png", dpi=150); plt.close()

print("\nNotebook 3 complete. Artifacts in ./output/")
for name in ["mechanism.png", "geometry.png", "quantization.png", "ablation_pareto.png", "mitigation.png"]:
    print(" -", name)
