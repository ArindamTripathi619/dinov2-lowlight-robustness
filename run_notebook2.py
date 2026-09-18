"""
Notebook 2: DINOv2 Advanced Low-Light Analysis
================================================
Advanced experiment: layer-wise CKA, bootstrap CI, frequency-domain test.

Fixed: removed Jupyter magic, uses shared utils, headless matplotlib,
       bootstrap RNG bug (same seed for all severities), flexible output dir.
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from utils import (
    DEVICE, load_dinov2, load_cifar10_subset, DINOV2_PREPROCESS,
    get_embeddings, get_embeddings_with_hooks, register_layer_hooks,
    low_light, low_pass, high_pass, get_corruption,
    train_linear_probe, evaluate_across_severities, bootstrap_accuracy_ci,
    linear_cka, compute_cka_matrix,
    plot_accuracy_vs_severity, plot_bootstrap_ci, plot_cka_heatmap,
    plot_frequency_test, plot_corruption_grid,
    save_results_csv, save_fig, setup_output_dir,
)


def parse_args():
    p = argparse.ArgumentParser(description="Phase 2: CKA localization, bootstrap CI, frequency test")
    p.add_argument("--corruption", default="low_light", choices=["low_light", "blur", "jpeg", "contrast"],
                   help="corruption family for the severity sweep and CKA (default: low_light)")
    p.add_argument("--model", default="dinov2_vits14",
                   help="torch.hub DINOv2 entry, e.g. dinov2_vits14 or dinov2_vitb14")
    p.add_argument("--n-images", type=int, default=500)
    p.add_argument("--seed", type=int, default=42, help="sample draw seed")
    p.add_argument("--output", default=None,
                   help="output dir (default: output/notebook2[_{corruption}][_{model}])")
    return p.parse_args()


ARGS = parse_args()
CORRUPTION_FN = get_corruption(ARGS.corruption)

if ARGS.output is not None:
    OUT = ARGS.output
else:
    OUT = "./output/notebook2"
    if ARGS.corruption != "low_light":
        OUT += f"_{ARGS.corruption}"
    if ARGS.model != "dinov2_vits14":
        OUT += f"_{ARGS.model.replace('dinov2_', '')}"
OUTPUT_DIR = setup_output_dir(OUT)

print("=" * 60)
print(f"Notebook 2: DINOv2 Advanced Analysis — corruption={ARGS.corruption}, model={ARGS.model}")
print(f"Device: {DEVICE}")
print("=" * 60)

# --- Cell 1: Load data ---
print("\n>>> Cell 1: Load CIFAR-10 subset")
N_IMAGES = ARGS.n_images
images, labels = load_cifar10_subset(n_images=N_IMAGES, train=False, seed=ARGS.seed)

# --- Cell 2: Visualize corruption types ---
print("\n>>> Cell 2: Visualize corruption types")
plot_corruption_grid(images, save_path=os.path.join(OUTPUT_DIR, "corruption_types.png"))

# --- Cell 3: Load DINOv2 with hooks ---
print("\n>>> Cell 3: Load DINOv2 ViT-S/14 with hooks")
model, n_blocks = load_dinov2(model_name=ARGS.model)
hooks_dict = {}
register_layer_hooks(model, hooks_dict)
print(f"DINOv2 loaded, {n_blocks} transformer blocks hooked.")

# --- Cell 4: Extract embeddings across severities ---
print("\n>>> Cell 4: Extract embeddings across all low-light severities")
dinov2_pooled = {}
dinov2_layerwise = {}
for severity in range(6):
    degraded = [CORRUPTION_FN(img, severity) for img in images]
    pooled, layerwise = get_embeddings_with_hooks(model, degraded, hooks_dict)
    dinov2_pooled[severity] = pooled
    dinov2_layerwise[severity] = layerwise
    print(f"  severity {severity}: embeddings {pooled.shape}")

# --- Cell 5: Linear probe ---
print("\n>>> Cell 5: Linear probe accuracy under low light")
probe, X_test, y_test, idx_test = train_linear_probe(dinov2_pooled[0], labels)

dinov2_accs = []
for severity in range(6):
    X_sev_test = dinov2_pooled[severity][idx_test]
    acc = probe.score(X_sev_test, y_test)
    dinov2_accs.append(acc)
    print(f"  severity {severity}: accuracy = {acc:.3f}")

plot_accuracy_vs_severity(
    [{"severity": i, "accuracy": a} for i, a in enumerate(dinov2_accs)],
    title=f"DINOv2 ({ARGS.model.replace('dinov2_', '')}) accuracy vs {ARGS.corruption} severity, CIFAR-10",
    save_path=os.path.join(OUTPUT_DIR, "accuracy_curve.png"),
)

# --- Cell 6: Bootstrap CI ---
print("\n>>> Cell 6: Bootstrap confidence intervals")
dinov2_ci = bootstrap_accuracy_ci(probe, dinov2_pooled, idx_test, y_test)
for r in dinov2_ci:
    print(f"  sev {r['severity']}: {r['mean']:.4f} [{r['ci_low']:.4f}, {r['ci_high']:.4f}]")

plot_bootstrap_ci(
    dinov2_ci,
    title="DINOv2 accuracy with bootstrap confidence bands",
    save_path=os.path.join(OUTPUT_DIR, "bootstrap_ci.png"),
)

# --- Cell 7: Layer-wise CKA ---
print("\n>>> Cell 7: Layer-wise CKA analysis")
cka_matrix = compute_cka_matrix(dinov2_layerwise[0], dinov2_layerwise, n_blocks)

plot_cka_heatmap(
    cka_matrix,
    title=f"Layer-wise representational drift under {ARGS.corruption}",
    save_path=os.path.join(OUTPUT_DIR, "layerwise_cka.png"),
)

drop = cka_matrix[:, 0] - cka_matrix[:, 5]
max_drop_layer = int(np.argmax(drop))
print(f"Layer with largest clean-vs-darkest CKA drop: {max_drop_layer}  drop = {drop[max_drop_layer]:.4f}")

# --- Cell 8: Frequency test ---
print("\n>>> Cell 8: Frequency-domain test")

def eval_probe_on_corruption(probe, corruption_fn, severities, images, model, idx_test, y_test):
    accs = []
    for severity in severities:
        degraded = [corruption_fn(img, severity) for img in images]
        pooled = get_embeddings(model, degraded)
        X_sev = pooled[idx_test]
        acc = probe.score(X_sev, y_test)
        accs.append(acc)
    return accs

lowpass_accs = eval_probe_on_corruption(probe, low_pass, [1, 2, 3, 4, 5],
                                         images, model, idx_test, y_test)
highpass_accs = eval_probe_on_corruption(probe, high_pass, [1, 2, 3, 4, 5],
                                          images, model, idx_test, y_test)

print(f"  Low-pass accuracies:  {[f'{a:.4f}' for a in lowpass_accs]}")
print(f"  High-pass accuracies: {[f'{a:.4f}' for a in highpass_accs]}")

plot_frequency_test(
    lowpass_accs, highpass_accs, dinov2_accs,
    title=f"Frequency dependence vs {ARGS.corruption} sensitivity",
    save_path=os.path.join(OUTPUT_DIR, "frequency_test.png"),
    ref_label=f"{ARGS.corruption} (reference)",
)

# --- Summary ---
print("\n" + "=" * 60)
print("NOTEBOOK 2 COMPLETE - SUMMARY")
print("=" * 60)

print("\n--- Linear Probe Accuracy (trained on clean) ---")
for i, acc in enumerate(dinov2_accs):
    print(f"  Severity {i}: {acc:.4f}")
print(f"  Accuracy drop (sev 0 -> 5): {dinov2_accs[0] - dinov2_accs[5]:.4f}")

print("\n--- Bootstrap 95% CI ---")
for r in dinov2_ci:
    print(f"  Severity {r['severity']}: {r['mean']:.4f} [{r['ci_low']:.4f}, {r['ci_high']:.4f}]")

print("\n--- Layer-wise CKA ---")
print(f"  Layer with largest drift: block {max_drop_layer} (drop={drop[max_drop_layer]:.4f})")
print(f"  Mean CKA drop across all layers: {drop.mean():.4f}")
print(f"  Earliest layer CKA drop (0): {drop[0]:.4f}")
print(f"  Latest layer CKA drop ({n_blocks-1}): {drop[-1]:.4f}")

print("\n--- Frequency Test ---")
print(f"  Low-pass accuracies:  {[f'{a:.4f}' for a in lowpass_accs]}")
print(f"  High-pass accuracies: {[f'{a:.4f}' for a in highpass_accs]}")
print(f"  Low-pass > High-pass: {all(l > h for l, h in zip(lowpass_accs, highpass_accs))}")

print("\n--- Key Findings ---")
if drop[0] > drop[-1]:
    print("  CKA drift concentrated in EARLY layers -> patch-embedding bias hypothesis SUPPORTED")
else:
    print("  CKA drift concentrated in LATE layers -> attention/norm hypothesis SUPPORTED")
if all(l > h for l, h in zip(lowpass_accs, highpass_accs)):
    print("  Low-pass preserved -> DINOv2 leans on LOW-FREQUENCY features (hypothesis SUPPORTED)")
else:
    print("  High-pass preserved better -> DINOv2 uses HIGH-FREQUENCY features (hypothesis REJECTED)")

print(f"\nOutput directory: {OUTPUT_DIR}")
print("=" * 60)
