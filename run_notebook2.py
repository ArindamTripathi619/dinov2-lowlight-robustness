!pip install -q scikit-learn matplotlib scipy

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


# --- cell separator ---

N_IMAGES = 500

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


# --- cell separator ---

def low_light(image, severity):
    brightness_factors = [1.0, 0.75, 0.55, 0.38, 0.25, 0.15]
    noise_std          = [0,    2,    4,    6,    9,    13]
    factor, noise = brightness_factors[severity], noise_std[severity]
    img = image.astype(np.float32) * factor
    if noise > 0:
        img = img + np.random.normal(0, noise, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def _fft_filter(image, cutoff_frac, keep="low"):
    img = image.astype(np.float32)
    out = np.zeros_like(img)
    h, w = img.shape[:2]
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
    max_dist = np.sqrt(cy ** 2 + cx ** 2)
    radius = cutoff_frac * max_dist
    mask = (dist <= radius) if keep == "low" else (dist > radius)
    for ch in range(img.shape[2]):
        f = np.fft.fftshift(np.fft.fft2(img[:, :, ch]))
        f_filtered = f * mask
        out[:, :, ch] = np.real(np.fft.ifft2(np.fft.ifftshift(f_filtered)))
    return np.clip(out, 0, 255).astype(np.uint8)

def low_pass(image, severity):
    cutoffs = [0.9, 0.7, 0.5, 0.35, 0.2]
    return _fft_filter(image, cutoffs[severity - 1], keep="low")

def high_pass(image, severity):
    cutoffs = [0.05, 0.1, 0.15, 0.2, 0.3]
    return _fft_filter(image, cutoffs[severity - 1], keep="high")

fig, axes = plt.subplots(3, 6, figsize=(15, 8))
sample = images[0]
for s in range(6):
    axes[0, s].imshow(low_light(sample, s)); axes[0, s].set_title(f"low-light sev {s}"); axes[0, s].axis("off")
for s in range(1, 6):
    axes[1, s].imshow(low_pass(sample, s)); axes[1, s].set_title(f"low-pass sev {s}"); axes[1, s].axis("off")
    axes[2, s].imshow(high_pass(sample, s)); axes[2, s].set_title(f"high-pass sev {s}"); axes[2, s].axis("off")
axes[1, 0].axis("off"); axes[2, 0].axis("off")
plt.tight_layout(); plt.show()


# --- cell separator ---

dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
dinov2.eval().to(device)

dinov2_preprocess = T.Compose([
    T.ToTensor(),
    T.Resize((224, 224), antialias=True),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

_dinov2_layer_outputs = {}
def _make_hook(layer_idx):
    def hook(module, inp, out):
        _dinov2_layer_outputs[layer_idx] = out[:, 0, :].detach().cpu()
    return hook

for i, block in enumerate(dinov2.blocks):
    block.register_forward_hook(_make_hook(i))

N_DINOV2_LAYERS = len(dinov2.blocks)
print(f"DINOv2 loaded, {N_DINOV2_LAYERS} transformer blocks hooked.")


# --- cell separator ---

@torch.no_grad()
def get_dinov2_embeddings(image_list, batch_size=64, collect_layers=False):
    pooled = []
    layerwise = {i: [] for i in range(N_DINOV2_LAYERS)} if collect_layers else None
    for i in range(0, len(image_list), batch_size):
        batch = image_list[i:i+batch_size]
        tensors = torch.stack([dinov2_preprocess(img) for img in batch]).to(device)
        out = dinov2(tensors)
        pooled.append(out.cpu())
        if collect_layers:
            for l in range(N_DINOV2_LAYERS):
                layerwise[l].append(_dinov2_layer_outputs[l])
    pooled = torch.cat(pooled, dim=0).numpy()
    if collect_layers:
        layerwise = {l: torch.cat(v, dim=0).numpy() for l, v in layerwise.items()}
        return pooled, layerwise
    return pooled

print("Extraction function ready.")


# --- cell separator ---

dinov2_pooled = {}
dinov2_layerwise = {}

for severity in range(6):
    degraded = [low_light(img, severity) for img in images]
    pooled, layerwise = get_dinov2_embeddings(degraded, collect_layers=True)
    dinov2_pooled[severity] = pooled
    dinov2_layerwise[severity] = layerwise
    print(f"severity {severity}: embeddings {pooled.shape}")


# --- cell separator ---

X_clean = dinov2_pooled[0]
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X_clean, labels, np.arange(len(labels)), test_size=0.3, random_state=42, stratify=labels
)

probe = LogisticRegression(max_iter=2000, C=1.0)
probe.fit(X_train, y_train)

dinov2_accs = []
for severity in range(6):
    X_sev_test = dinov2_pooled[severity][idx_test]
    acc = accuracy_score(y_test, probe.predict(X_sev_test))
    dinov2_accs.append(acc)
    print(f"severity {severity}: accuracy = {acc:.3f}")

plt.figure(figsize=(8, 5))
plt.plot(range(6), dinov2_accs, marker="o", color="tab:blue")
plt.xlabel("Low-light severity (0 = bright, 5 = darkest)")
plt.ylabel("Linear probe accuracy")
plt.title("DINOv2 (ViT-S/14) accuracy vs low-light severity, CIFAR-10")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("dinov2_accuracy.png", dpi=150)
plt.show()


# --- cell separator ---

def bootstrap_accuracy_ci(probe, pooled_by_severity, idx_test, y_test, n_boot=1000, ci=95):
    results = []
    n = len(idx_test)
    for severity in range(6):
        X_sev = pooled_by_severity[severity][idx_test]
        preds = probe.predict(X_sev)
        correct = (preds == y_test).astype(float)
        boot_accs = []
        rng_local = np.random.default_rng(0)
        for _ in range(n_boot):
            sample_idx = rng_local.integers(0, n, n)
            boot_accs.append(correct[sample_idx].mean())
        lo, hi = np.percentile(boot_accs, [(100 - ci) / 2, 100 - (100 - ci) / 2])
        results.append({"severity": severity, "mean": correct.mean(), "ci_low": lo, "ci_high": hi})
    return results

dinov2_ci = bootstrap_accuracy_ci(probe, dinov2_pooled, idx_test, y_test)

fig, ax = plt.subplots(figsize=(8, 5))
sev = [r["severity"] for r in dinov2_ci]
mean = [r["mean"] for r in dinov2_ci]
lo = [r["ci_low"] for r in dinov2_ci]
hi = [r["ci_high"] for r in dinov2_ci]
ax.plot(sev, mean, marker="o", color="tab:blue")
ax.fill_between(sev, lo, hi, alpha=0.2, color="tab:blue")
ax.set_xlabel("Low-light severity")
ax.set_ylabel("Accuracy (95% bootstrap CI shaded)")
ax.set_title("DINOv2 accuracy with bootstrap confidence bands")
plt.tight_layout()
plt.savefig("bootstrap_ci.png", dpi=150)
plt.show()

for r in dinov2_ci: print(r)


# --- cell separator ---

def linear_cka(X, Y):
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    hsic = np.linalg.norm(X.T @ Y, 'fro') ** 2
    var1 = np.linalg.norm(X.T @ X, 'fro')
    var2 = np.linalg.norm(Y.T @ Y, 'fro')
    return hsic / (var1 * var2 + 1e-8)

cka_matrix = np.zeros((N_DINOV2_LAYERS, 6))
for layer in range(N_DINOV2_LAYERS):
    clean_acts = dinov2_layerwise[0][layer]
    for severity in range(6):
        degraded_acts = dinov2_layerwise[severity][layer]
        cka_matrix[layer, severity] = linear_cka(clean_acts, degraded_acts)

plt.figure(figsize=(8, 6))
im = plt.imshow(cka_matrix, aspect="auto", cmap="viridis", vmin=0, vmax=1)
plt.colorbar(im, label="CKA similarity to clean")
plt.xlabel("Low-light severity")
plt.ylabel("DINOv2 transformer block (0 = earliest)")
plt.title("Layer-wise representational drift under low light")
plt.tight_layout()
plt.savefig("layerwise_cka.png", dpi=150)
plt.show()

drop = cka_matrix[:, 0] - cka_matrix[:, 5]
print("Layer with largest clean-vs-darkest CKA drop:", int(np.argmax(drop)), "  drop =", drop.max())


# --- cell separator ---

def eval_probe_on_corruption(probe, corruption_fn, severities):
    accs = []
    for severity in severities:
        degraded = [corruption_fn(img, severity) for img in images]
        pooled = get_dinov2_embeddings(degraded, collect_layers=False)
        X_test = pooled[idx_test]
        acc = accuracy_score(y_test, probe.predict(X_test))
        accs.append(acc)
    return accs

lowpass_accs  = eval_probe_on_corruption(probe, low_pass, [1, 2, 3, 4, 5])
highpass_accs = eval_probe_on_corruption(probe, high_pass, [1, 2, 3, 4, 5])

plt.figure(figsize=(8, 5))
plt.plot(range(1, 6), lowpass_accs, marker="o", label="Low-pass (keeps low freq)")
plt.plot(range(1, 6), highpass_accs, marker="s", label="High-pass (keeps high freq)")
plt.plot(range(0, 6), dinov2_accs, marker="^", linestyle="--", color="gray", label="Low-light (reference)")
plt.xlabel("Corruption severity")
plt.ylabel("DINOv2 linear probe accuracy")
plt.title("Which frequency band does DINOv2 actually depend on?")
plt.legend()
plt.tight_layout()
plt.savefig("frequency_test.png", dpi=150)
plt.show()

print("Low-pass accuracies: ", lowpass_accs)
print("High-pass accuracies:", highpass_accs)


# --- cell separator ---

RUN_DINOV3 = False  # flip to True once your HF access request is approved

DINOV3_AVAILABLE = False
dinov3 = None
dinov3_processor = None

if RUN_DINOV3:
    try:
        !pip install -q -U "transformers>=4.56.0" huggingface_hub
        from huggingface_hub import login
        login()
        from transformers import AutoImageProcessor, AutoModel
        DINOV3_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"
        dinov3_processor = AutoImageProcessor.from_pretrained(DINOV3_ID)
        dinov3 = AutoModel.from_pretrained(DINOV3_ID).eval().to(device)
        DINOV3_AVAILABLE = True
        print("DINOv3 loaded successfully.")
    except Exception as e:
        print("DINOv3 could not be loaded — likely access not approved yet, no token, or a network issue.")
        print("Error detail:", repr(e))
        print("Skipping DINOv3 comparison. Everything above this cell is unaffected.")
else:
    print("DINOv3 comparison skipped (RUN_DINOV3 = False). Flip it to True once your HF access is approved.")


# --- cell separator ---

if DINOV3_AVAILABLE:
    from PIL import Image

    @torch.no_grad()
    def get_dinov3_embeddings(image_list, batch_size=64):
        feats = []
        for i in range(0, len(image_list), batch_size):
            batch = [Image.fromarray(img) for img in image_list[i:i+batch_size]]
            inputs = dinov3_processor(images=batch, return_tensors="pt").to(device)
            out = dinov3(**inputs)
            pooled_out = getattr(out, "pooler_output", None)
            if pooled_out is None:
                pooled_out = out.last_hidden_state[:, 0, :]
            feats.append(pooled_out.cpu())
        return torch.cat(feats, dim=0).numpy()

    dinov3_pooled = {}
    for severity in range(6):
        degraded = [low_light(img, severity) for img in images]
        dinov3_pooled[severity] = get_dinov3_embeddings(degraded)
        print(f"severity {severity}: dinov3 embeddings {dinov3_pooled[severity].shape}")

    X_clean_v3 = dinov3_pooled[0]
    X_train_v3, X_test_v3, y_train_v3, y_test_v3, idx_train_v3, idx_test_v3 = train_test_split(
        X_clean_v3, labels, np.arange(len(labels)), test_size=0.3, random_state=42, stratify=labels
    )
    probe_v3 = LogisticRegression(max_iter=2000, C=1.0)
    probe_v3.fit(X_train_v3, y_train_v3)

    dinov3_accs = []
    for severity in range(6):
        acc = accuracy_score(y_test_v3, probe_v3.predict(dinov3_pooled[severity][idx_test_v3]))
        dinov3_accs.append(acc)

    plt.figure(figsize=(8, 5))
    plt.plot(range(6), dinov2_accs, marker="o", label="DINOv2 (ViT-S/14)")
    plt.plot(range(6), dinov3_accs, marker="s", label="DINOv3 (ViT-S/16)")
    plt.xlabel("Low-light severity")
    plt.ylabel("Linear probe accuracy")
    plt.title("DINOv2 vs DINOv3: low-light robustness, CIFAR-10")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig("dinov2_vs_dinov3_accuracy.png", dpi=150)
    plt.show()
    print("DINOv3 accuracies:", dinov3_accs)
else:
    print("Skipped — set RUN_DINOV3 = True in the cell above (after HF approval) to run this comparison.")
