"""
Shared utilities for DINOv2 low-light robustness experiments.

Centralizes corruption functions, preprocessing, embedding extraction,
linear probe evaluation, CKA, and plotting helpers used across all notebooks.
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless by default; callers can override before import
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as T
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Corruption functions
# ---------------------------------------------------------------------------
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
    return np.clip(img, 0, 255).astype(np.uint8)


def _fft_filter(image, cutoff_frac, keep="low"):
    """Apply a circular FFT low-pass or high-pass filter."""
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
    """Low-pass filter (keeps low frequencies). severity 1-5."""
    cutoffs = [0.9, 0.7, 0.5, 0.35, 0.2]
    return _fft_filter(image, cutoffs[severity - 1], keep="low")


def high_pass(image, severity):
    """High-pass filter (keeps high frequencies). severity 1-5."""
    cutoffs = [0.05, 0.1, 0.15, 0.2, 0.3]
    return _fft_filter(image, cutoffs[severity - 1], keep="high")


# ---------------------------------------------------------------------------
# DINOv2 loading & preprocessing
# ---------------------------------------------------------------------------
DINOV2_PREPROCESS = T.Compose([
    T.ToTensor(),
    T.Resize((224, 224), antialias=True),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_dinov2(device=None):
    """Load DINOv2 ViT-S/14 and return (model, num_blocks)."""
    device = device or DEVICE
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    model.eval().to(device)
    return model, len(model.blocks)


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------
@torch.no_grad()
def get_embeddings(model, image_list, batch_size=64, device=None):
    """Extract pooled CLS embeddings from DINOv2.

    Args:
        model: DINOv2 model (already on device).
        image_list: list of HxWx3 uint8 numpy arrays.
        batch_size: inference batch size.
        device: torch device.

    Returns:
        (N, D) numpy array of embeddings.
    """
    device = device or DEVICE
    feats = []
    for i in range(0, len(image_list), batch_size):
        batch = image_list[i:i + batch_size]
        tensors = torch.stack([DINOV2_PREPROCESS(img) for img in batch]).to(device)
        out = model(tensors)
        feats.append(out.cpu())
    return torch.cat(feats, dim=0).numpy()


@torch.no_grad()
def get_embeddings_with_hooks(model, image_list, hooks_dict, batch_size=64, device=None):
    """Extract pooled CLS embeddings AND collect layer-wise hook outputs.

    Args:
        model: DINOv2 model with forward hooks registered.
        image_list: list of HxWx3 uint8 numpy arrays.
        hooks_dict: global dict populated by hooks (keyed by layer index).
        batch_size: inference batch size.
        device: torch device.

    Returns:
        (pooled_numpy, layerwise_dict) where layerwise_dict maps
        layer_idx -> (N, D) numpy array.
    """
    device = device or DEVICE
    n_layers = len(model.blocks)
    pooled = []
    layerwise = {i: [] for i in range(n_layers)}

    for i in range(0, len(image_list), batch_size):
        batch = image_list[i:i + batch_size]
        tensors = torch.stack([DINOV2_PREPROCESS(img) for img in batch]).to(device)
        out = model(tensors)
        pooled.append(out.cpu())
        for l in range(n_layers):
            layerwise[l].append(hooks_dict[l])

    pooled = np.concatenate(pooled, axis=0)
    layerwise = {l: np.concatenate(v, axis=0) for l, v in layerwise.items()}
    return pooled, layerwise


def register_layer_hooks(model, hooks_dict):
    """Register forward hooks on every transformer block.

    hooks_dict is a mutable dict that hooks will write into (layer_idx -> tensor).
    """
    def _make_hook(layer_idx):
        def hook(module, inp, out):
            hooks_dict[layer_idx] = out[:, 0, :].detach().cpu()
        return hook

    for i, block in enumerate(model.blocks):
        block.register_forward_hook(_make_hook(i))


# ---------------------------------------------------------------------------
# Linear probe evaluation
# ---------------------------------------------------------------------------
def train_linear_probe(embeddings, labels, test_size=0.3, random_state=42):
    """Train a logistic regression probe on embeddings.

    Returns:
        (probe, X_test, y_test, idx_test)
    """
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        embeddings, labels, np.arange(len(labels)),
        test_size=test_size, random_state=random_state, stratify=labels,
    )
    probe = LogisticRegression(max_iter=2000, C=1.0)
    probe.fit(X_train, y_train)
    return probe, X_test, y_test, idx_test


def evaluate_across_severities(probe, embeddings_by_severity, idx_test, y_test):
    """Evaluate a trained linear probe at each severity level.

    Returns:
        list of dicts with keys: severity, accuracy, mean_cosine_sim_to_clean.
    """
    X_clean = embeddings_by_severity[0]

    def cosine_sim(a, b):
        a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
        b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
        return (a * b).sum(axis=1)

    results = []
    for severity in sorted(embeddings_by_severity.keys()):
        X_sev = embeddings_by_severity[severity][idx_test]
        preds = probe.predict(X_sev)
        acc = accuracy_score(y_test, preds)
        sims = cosine_sim(X_clean, embeddings_by_severity[severity])
        results.append({
            "severity": severity,
            "accuracy": acc,
            "mean_cosine_sim_to_clean": float(sims.mean()),
        })
    return results


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------
def bootstrap_accuracy_ci(probe, embeddings_by_severity, idx_test, y_test,
                          n_boot=1000, ci=95):
    """Compute bootstrap 95% CIs for accuracy at each severity.

    Bug fix: uses a SEPARATE RNG per severity so bootstrap samples differ.
    """
    results = []
    n = len(idx_test)
    for severity in sorted(embeddings_by_severity.keys()):
        X_sev = embeddings_by_severity[severity][idx_test]
        preds = probe.predict(X_sev)
        correct = (preds == y_test).astype(float)
        boot_accs = []
        # FIX: use severity-specific seed so each level gets different samples
        rng_local = np.random.default_rng(severity)
        for _ in range(n_boot):
            sample_idx = rng_local.integers(0, n, n)
            boot_accs.append(correct[sample_idx].mean())
        lo, hi = np.percentile(boot_accs, [(100 - ci) / 2, 100 - (100 - ci) / 2])
        results.append({
            "severity": severity,
            "mean": correct.mean(),
            "ci_low": lo,
            "ci_high": hi,
        })
    return results


# ---------------------------------------------------------------------------
# CKA (Centered Kernel Alignment)
# ---------------------------------------------------------------------------
def linear_cka(X, Y):
    """Compute linear CKA similarity between two activation matrices."""
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    hsic = np.linalg.norm(X.T @ Y, "fro") ** 2
    var1 = np.linalg.norm(X.T @ X, "fro") ** 2
    var2 = np.linalg.norm(Y.T @ Y, "fro") ** 2
    return hsic / (np.sqrt(var1 * var2) + 1e-8)


def compute_cka_matrix(layerwise_clean, layerwise_by_severity, n_layers):
    """Compute layer x severity CKA matrix.

    Returns:
        (n_layers, n_severities) numpy array.
    """
    severities = sorted(layerwise_by_severity.keys())
    cka_matrix = np.zeros((n_layers, len(severities)))
    for layer in range(n_layers):
        clean_acts = layerwise_clean[layer]
        for j, severity in enumerate(severities):
            degraded_acts = layerwise_by_severity[severity][layer]
            cka_matrix[layer, j] = linear_cka(clean_acts, degraded_acts)
    return cka_matrix


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def setup_output_dir(path="./output"):
    """Create output directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)
    return path


def save_fig(fig, path, dpi=150):
    """Save a matplotlib figure and close it."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_accuracy_vs_severity(results, title="DINOv2 accuracy vs low-light severity",
                              save_path=None):
    """Plot linear probe accuracy across severity levels."""
    severities = [r["severity"] for r in results]
    accs = [r["accuracy"] for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(severities, accs, marker="o", color="tab:blue")
    ax.set_xlabel("Low-light severity (0 = bright, 5 = darkest)")
    ax.set_ylabel("Linear probe accuracy")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    return fig


def plot_accuracy_and_drift(results, title="DINOv2 robustness to low-light degradation",
                            save_path=None):
    """Plot accuracy and cosine similarity drift on dual y-axes."""
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
    ax2.plot(severities, sims, marker="s", color="tab:red", linestyle="--",
             label="Embedding similarity")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    ax1.set_title(title)
    fig.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    return fig


def plot_bootstrap_ci(ci_results, title="DINOv2 accuracy with bootstrap confidence bands",
                      save_path=None):
    """Plot accuracy with bootstrap CI shaded bands."""
    sev = [r["severity"] for r in ci_results]
    mean = [r["mean"] for r in ci_results]
    lo = [r["ci_low"] for r in ci_results]
    hi = [r["ci_high"] for r in ci_results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sev, mean, marker="o", color="tab:blue")
    ax.fill_between(sev, lo, hi, alpha=0.2, color="tab:blue")
    ax.set_xlabel("Low-light severity")
    ax.set_ylabel("Accuracy (95% bootstrap CI shaded)")
    ax.set_title(title)
    fig.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    return fig


def plot_cka_heatmap(cka_matrix, title="Layer-wise representational drift under low light",
                     save_path=None, xlabel="Low-light severity",
                     ylabel="DINOv2 transformer block (0 = earliest)"):
    """Plot CKA heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cka_matrix, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(im, label="CKA similarity to clean")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    return fig


def plot_frequency_test(lowpass_accs, highpass_accs, ref_accs,
                        title="Which frequency band does DINOv2 actually depend on?",
                        save_path=None):
    """Plot frequency-domain corruption results."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, 6), lowpass_accs, marker="o", label="Low-pass (keeps low freq)")
    ax.plot(range(1, 6), highpass_accs, marker="s", label="High-pass (keeps high freq)")
    ax.plot(range(0, 6), ref_accs, marker="^", linestyle="--", color="gray",
            label="Low-light (reference)")
    ax.set_xlabel("Corruption severity")
    ax.set_ylabel("DINOv2 linear probe accuracy")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    return fig


def plot_corruption_grid(images, save_path=None):
    """Plot 3x6 grid: low-light, low-pass, high-pass across severities."""
    sample = images[0]
    fig, axes = plt.subplots(3, 6, figsize=(15, 8))
    for s in range(6):
        axes[0, s].imshow(low_light(sample, s))
        axes[0, s].set_title(f"low-light sev {s}")
        axes[0, s].axis("off")
    for s in range(1, 6):
        axes[1, s].imshow(low_pass(sample, s))
        axes[1, s].set_title(f"low-pass sev {s}")
        axes[1, s].axis("off")
        axes[2, s].imshow(high_pass(sample, s))
        axes[2, s].set_title(f"high-pass sev {s}")
        axes[2, s].axis("off")
    axes[1, 0].axis("off")
    axes[2, 0].axis("off")
    fig.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------
def save_results_csv(results, path):
    """Save results list of dicts to CSV."""
    if not results:
        return
    fieldnames = list(results[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_cifar10_subset(n_images=1000, train=False, seed=42, data_root="./data"):
    """Load a random subset of CIFAR-10.

    Returns:
        (images: list of np.ndarray, labels: np.ndarray)
    """
    import torchvision
    raw_ds = torchvision.datasets.CIFAR10(root=data_root, train=train, download=True)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(raw_ds), size=min(n_images, len(raw_ds)), replace=False)
    images, labels = [], []
    for i in idx:
        img, label = raw_ds[i]
        images.append(np.array(img))
        labels.append(label)
    labels = np.array(labels)
    print(f"Loaded {len(images)} images from CIFAR-10 ({'train' if train else 'test'}), "
          f"classes: {np.unique(labels).tolist()}")
    return images, labels
