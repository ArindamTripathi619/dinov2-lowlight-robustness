"""
Notebook 1: DINOv2 CIFAR-10 Low-Light Robustness
=================================================
Baseline experiment: extract DINOv2 embeddings at 6 corruption severity levels,
train a linear probe on clean embeddings, evaluate across severities.

Fixed: removed Jupyter magic, uses shared utils, headless matplotlib, flexible output dir.
Extended (Phase 0): --corruption dispatch (low_light/blur/jpeg/contrast), --model passthrough.
"""

import sys
import os
import argparse

# Allow running from project root
sys.path.insert(0, os.path.dirname(__file__))

from utils import (
    DEVICE, load_dinov2, load_cifar10_subset, get_embeddings,
    get_corruption, train_linear_probe, evaluate_across_severities,
    plot_accuracy_and_drift, save_results_csv, setup_output_dir,
)


def parse_args():
    p = argparse.ArgumentParser(description="Phase 1: probe accuracy across corruption severities")
    p.add_argument("--corruption", default="low_light", choices=["low_light", "blur", "jpeg", "contrast"],
                   help="corruption family to sweep (default: low_light)")
    p.add_argument("--model", default="dinov2_vits14",
                   help="torch.hub DINOv2 entry, e.g. dinov2_vits14 or dinov2_vitb14")
    p.add_argument("--n-images", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42, help="test-sample draw seed")
    p.add_argument("--output", default=None,
                   help="output dir (default: output/notebook1[_{corruption}][_{model}])")
    return p.parse_args()


ARGS = parse_args()
CORRUPTION_FN = get_corruption(ARGS.corruption)

if ARGS.output is not None:
    OUT = ARGS.output
else:
    OUT = "./output/notebook1"
    if ARGS.corruption != "low_light":
        OUT += f"_{ARGS.corruption}"
    if ARGS.model != "dinov2_vits14":
        OUT += f"_{ARGS.model.replace('dinov2_', '')}"
OUTPUT_DIR = setup_output_dir(OUT)

print("=" * 60)
print(f"Notebook 1: DINOv2 Robustness — corruption={ARGS.corruption}, model={ARGS.model}")
print(f"Device: {DEVICE}")
print("=" * 60)

# --- Cell 1: Load data ---
print("\n>>> Cell 1: Load CIFAR-10 subset")
N_IMAGES = ARGS.n_images
images, labels = load_cifar10_subset(n_images=N_IMAGES, train=False, seed=ARGS.seed)

# --- Cell 2: Visual sanity check ---
print("\n>>> Cell 2: Visual sanity check")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 6, figsize=(15, 3))
for s in range(6):
    axes[s].imshow(CORRUPTION_FN(images[0], s))
    axes[s].set_title(f"severity {s}")
    axes[s].axis("off")
fig.suptitle(f"Original class: {['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck'][labels[0]]}")
fig.tight_layout()
from utils import save_fig
save_fig(fig, os.path.join(OUTPUT_DIR, "severity_visual_check.png"))

# --- Cell 3: Load DINOv2 ---
print(f"\n>>> Cell 3: Load DINOv2 ({ARGS.model})")
model, n_blocks = load_dinov2(model_name=ARGS.model)
print(f"DINOv2 loaded, {n_blocks} transformer blocks.")

# --- Cell 4: Extract embeddings ---
print("\n>>> Cell 4: Extract embeddings at every severity")
embeddings_by_severity = {}
for severity in range(6):
    degraded_images = [CORRUPTION_FN(img, severity) for img in images]
    embeddings_by_severity[severity] = get_embeddings(model, degraded_images)
    print(f"  severity {severity}: shape {embeddings_by_severity[severity].shape}")

# --- Cell 5: Train linear probe ---
print("\n>>> Cell 5: Train linear probe on CLEAN embeddings")
probe, X_test, y_test, idx_test = train_linear_probe(
    embeddings_by_severity[0], labels
)
clean_test_acc = probe.score(X_test, y_test)
print(f"Clean test accuracy: {clean_test_acc:.3f}")

# --- Cell 6: Evaluate across severities ---
print("\n>>> Cell 6: Evaluate probe across all severities")
results = evaluate_across_severities(probe, embeddings_by_severity, idx_test, y_test)
for r in results:
    print(f"  severity {r['severity']}: acc={r['accuracy']:.3f}, cos_sim={r['mean_cosine_sim_to_clean']:.4f}")

# --- Cell 7: Plot ---
print("\n>>> Cell 7: Plot accuracy and embedding drift")
plot_accuracy_and_drift(
    results,
    title=f"DINOv2 ({ARGS.model.replace('dinov2_', '')}) robustness to stepwise {ARGS.corruption} degradation, CIFAR-10",
    save_path=os.path.join(OUTPUT_DIR, "corruption_results.png"),
)

# --- Cell 8: Save CSV ---
print("\n>>> Cell 8: Save results to CSV")
save_results_csv(results, os.path.join(OUTPUT_DIR, "corruption_results.csv"))

# --- Summary ---
print("\n" + "=" * 60)
print("NOTEBOOK 1 COMPLETE - SUMMARY")
print("=" * 60)
for r in results:
    print(f"  Severity {r['severity']}: accuracy={r['accuracy']:.4f}, "
          f"cosine_sim={r['mean_cosine_sim_to_clean']:.4f}")
print(f"\nClean test accuracy: {clean_test_acc:.4f}")
print(f"Accuracy drop (sev 0 -> 5): {results[0]['accuracy'] - results[5]['accuracy']:.4f}")
print(f"Output directory: {OUTPUT_DIR}")
print("=" * 60)
