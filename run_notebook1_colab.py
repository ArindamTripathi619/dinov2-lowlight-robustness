"""
Notebook 1 (Colab): DINOv2 CIFAR-10 Low-Light Robustness
==========================================================
Colab runner — installs deps, copies shared utils, then runs the experiment.
"""

import sys
import os
import subprocess

# Setup
os.makedirs("/content/output", exist_ok=True)
os.chdir("/content")

# Install dependencies
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "scikit-learn", "matplotlib"], check=False)

# Make utils available (assumes utils.py is in the same directory as this script,
# or was uploaded to Colab)
sys.path.insert(0, "/content")

# If utils.py isn't on /content yet, copy from project
if not os.path.exists("/content/utils.py"):
    import shutil
    src = os.path.join(os.path.dirname(__file__), "utils.py")
    if os.path.exists(src):
        shutil.copy(src, "/content/utils.py")

from utils import (
    DEVICE, load_dinov2, load_cifar10_subset, get_embeddings,
    low_light, train_linear_probe, evaluate_across_severities,
    plot_accuracy_and_drift, save_results_csv, save_fig, setup_output_dir,
)
import matplotlib.pyplot as plt

OUTPUT_DIR = setup_output_dir("/content/output/notebook1")

print("=" * 60)
print("Starting Notebook 1: DINOv2 CIFAR-10 Low-Light Robustness")
print(f"Device: {DEVICE}")
print("=" * 60)

# --- Cell 1: Load data ---
print("\n>>> Cell 1: Load CIFAR-10 subset")
N_IMAGES = 1000
images, labels = load_cifar10_subset(n_images=N_IMAGES, train=False)

# --- Cell 2: Visual sanity check ---
print("\n>>> Cell 2: Visual sanity check")
fig, axes = plt.subplots(1, 6, figsize=(15, 3))
class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
for s in range(6):
    axes[s].imshow(low_light(images[0], s))
    axes[s].set_title(f"severity {s}")
    axes[s].axis("off")
fig.suptitle(f"Original class: {class_names[labels[0]]}")
fig.tight_layout()
save_fig(fig, os.path.join(OUTPUT_DIR, "severity_visual_check.png"))

# --- Cell 3: Load DINOv2 ---
print("\n>>> Cell 3: Load DINOv2 ViT-S/14")
model, n_blocks = load_dinov2()
print(f"DINOv2 loaded, {n_blocks} transformer blocks.")

# --- Cell 4: Extract embeddings ---
print("\n>>> Cell 4: Extract embeddings at every severity")
embeddings_by_severity = {}
for severity in range(6):
    degraded_images = [low_light(img, severity) for img in images]
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
    title="DINOv2 (ViT-S/14) robustness to stepwise low-light degradation, CIFAR-10",
    save_path=os.path.join(OUTPUT_DIR, "dinov2_lowlight_results.png"),
)

# --- Cell 8: Save CSV ---
print("\n>>> Cell 8: Save results to CSV")
save_results_csv(results, os.path.join(OUTPUT_DIR, "dinov2_lowlight_results.csv"))

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
