#!/usr/bin/env python3
"""Cross-family drift atlas builder (ROADMAP Track 3).

Aggregates the per-model drift profiles written by run_drift_profile.py
(``output/profiles/<model>/drift_profile_sev{S}_{corr}.csv``, one directory
per model, rows in forward/shallow->deep module order) into:

  - ``atlas_summary.csv``  — one row per (model, corruption, severity) with
    the cross-family headline statistic: late-half drift mass (fraction of
    total CKA drop sitting in the deeper half of the profiled modules) plus
    the argmax-drift module.
  - ``atlas_fig1.png``     — the paper's Fig. 1 candidate: drift (1 - CKA)
    vs normalized depth, one line per model, one panel per corruption.

Normalized depth = module rank / (n_modules - 1) in forward order. Families
have different granularities (4 stages for ConvNeXt, 12 blocks for plain
ViTs, 4 stages + 16 bottlenecks for ResNet-50), so absolute indices cannot be
compared across models; rank alignment is the atlas' common axis.

Pure numpy + stdlib + matplotlib (Agg) — importable in torch-free CI.

Usage:
  python3 tools/build_atlas.py --profiles-root output/profiles \
      --severities 5 --corruptions low_light,jpeg --out-dir output/atlas
"""
import argparse
import csv
import os
import sys

import numpy as np

PROFILE_FIELDS = ("layer", "severity", "module", "cka", "cka_unbiased",
                  "drop_from_clean", "cka_unbiased_drop")

# Atlas display names: profile-dir name -> (paper label, family tag)
MODEL_LABELS = {
    "resnet50": ("ResNet-50", "CNN"),
    "convnext_tiny": ("ConvNeXt-T", "CNN"),
    "vit_small_patch16_224": ("ViT-S/16", "ViT"),
    "deit_small_patch16_224": ("DeiT-S", "ViT"),
    "swin_tiny_patch4_window7_224": ("Swin-T", "hybrid"),
    "dinov2_vitb14": ("DINOv2 ViT-B/14", "ViT (self-sup.)"),
    # CIFAR-10 consistency leg (same models, legacy image axis)
    "cifar10_resnet50": ("ResNet-50 (CIFAR-10)", "CNN"),
    "cifar10_convnext_tiny": ("ConvNeXt-T (CIFAR-10)", "CNN"),
}


def read_profile_csv(path):
    """Read one drift_profile_sev{S}_{corr}.csv; rows kept in file (=forward)
    order; numeric fields cast to float, layer to int."""
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({
                "layer": int(r["layer"]),
                "severity": int(r["severity"]),
                "module": r["module"],
                "cka": float(r["cka"]),
                "drop_from_clean": float(r["drop_from_clean"]),
                "cka_unbiased_drop": float(r["cka_unbiased_drop"]),
            })
    if not rows:
        raise ValueError(f"empty profile: {path}")
    return rows


def normalize_depth(n_modules):
    """Rank-based normalized depth in [0, 1] for n_modules forward-ordered
    modules (shallow -> deep). n=1 maps to 0.0 (degenerate but well-defined).
    """
    if n_modules < 1:
        raise ValueError(f"n_modules must be >= 1, got {n_modules}")
    if n_modules == 1:
        return np.array([0.0])
    return np.linspace(0.0, 1.0, n_modules)


def late_half_mass(drops):
    """Fraction of total drift mass in the deeper half of the profile.

    ``drops`` is a shallow->deep array of per-module CKA drops. The late half
    is the ceil(n/2) deepest modules. Returns NaN for an all-zero profile
    (total mass < 1e-12) — a model that does not drift has no "where".
    """
    d = np.asarray(drops, dtype=np.float64)
    if d.size == 0:
        raise ValueError("empty drops")
    total = float(d.sum())
    if total < 1e-12:
        return float("nan")
    start = d.size - (d.size + 1) // 2  # ceil(n/2) deepest modules
    return float(d[start:].sum() / total)


def summarize_profile(model, rows):
    """One summary row per profile file (single corruption+severity)."""
    drops = np.array([r["drop_from_clean"] for r in rows])
    argmax = int(np.argmax(drops))
    return {
        "model": model,
        "corruption": "",  # not in the CSV rows; filled by caller from the filename key
        "severity": rows[0]["severity"],
        "n_modules": len(rows),
        "late_half_mass": late_half_mass(drops),
        "max_drop": float(drops[argmax]),
        "max_drop_module": rows[argmax]["module"],
        "first_drop": float(drops[0]),
        "last_drop": float(drops[-1]),
    }


def collect_profiles(profiles_root, severities, corruptions):
    """Discover <root>/<model>/drift_profile_sev{S}_{corr}.csv.

    Returns {(model, corruption, severity): rows} for the Cartesian product
    that exists on disk; missing combos are simply absent (models vary in
    what ran).
    """
    found = {}
    for model_dir in sorted(os.listdir(profiles_root)):
        model_root = os.path.join(profiles_root, model_dir)
        if not os.path.isdir(model_root):
            continue
        for corr in corruptions:
            for sev in severities:
                path = os.path.join(model_root, f"drift_profile_sev{sev}_{corr}.csv")
                if os.path.isfile(path):
                    found[(model_dir, corr, sev)] = read_profile_csv(path)
    return found


def build_fig1(found, out_path, severities):
    """Drift vs normalized depth; one line per model, one panel per corruption
    (highest swept severity of record).

    matplotlib is imported lazily: the pure-math helpers above must stay
    importable in the torch-/matplotlib-free CI test env (same convention as
    the optional-dep pattern in tests/run_tests.py).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    corruptions = sorted({c for (_m, c, _s) in found})
    sev = max(severities)
    fig, axes = plt.subplots(1, len(corruptions), figsize=(7.5 * len(corruptions), 5.5),
                             squeeze=False)
    for ax, corr in zip(axes[0], corruptions):
        for (model, c, s), rows in sorted(found.items()):
            if c != corr or s != sev:
                continue
            label, family = MODEL_LABELS.get(model, (model, "?"))
            drops = np.array([r["drop_from_clean"] for r in rows])
            depth = normalize_depth(len(rows))
            ax.plot(depth, drops, marker="o", markersize=3, linewidth=1.6,
                    label=f"{label} ({family})")
        ax.set_title(f"{corr} @ severity {sev}")
        ax.set_xlabel("normalized depth (0 = shallowest profiled module)")
        ax.set_ylabel("CKA drop from clean (1 − CKA)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Cross-family drift atlas — where representation damage lives "
                 "(Track 3, ImageNet-val subset of record)", y=1.00)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--profiles-root", default="output/profiles")
    ap.add_argument("--severities", default="5",
                    help="comma-separated severities to aggregate (default 5 = allocation severity)")
    ap.add_argument("--corruptions", default="low_light,jpeg")
    ap.add_argument("--out-dir", default="output/atlas")
    args = ap.parse_args()

    severities = sorted({int(s) for s in args.severities.split(",") if s.strip()})
    corruptions = [c.strip() for c in args.corruptions.split(",") if c.strip()]

    found = collect_profiles(args.profiles_root, severities, corruptions)
    if not found:
        print(f"error: no profiles under {args.profiles_root!r} "
              f"(looked for drift_profile_sev{{S}}_{{corr}}.csv per model dir)")
        return 1
    models = sorted({m for (m, _c, _s) in found})
    print(f"atlas inputs: {len(models)} models × {len(corruptions)} corruptions × "
          f"severities {severities}:")
    for m in models:
        keys = sorted((c, s) for (mm, c, s) in found if mm == m)
        label, family = MODEL_LABELS.get(m, (m, "?"))
        print(f"  {label:<24} {family:<16} {len(keys)} profile files")

    os.makedirs(args.out_dir, exist_ok=True)

    # --- summary CSV ---
    summary_path = os.path.join(args.out_dir, "atlas_summary.csv")
    fields = ["model", "corruption", "severity", "n_modules", "late_half_mass",
              "max_drop", "max_drop_module", "first_drop", "last_drop"]
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for (model, corr, sev), rows in sorted(found.items()):
            s = summarize_profile(model, rows)
            s["corruption"] = corr
            writer.writerow(s)
    print(f"Saved {summary_path}")

    # --- Fig. 1 ---
    build_fig1(found, os.path.join(args.out_dir, "atlas_fig1.png"), severities)

    # --- console verdict table (highest severity, per corruption) ---
    sev = max(severities)
    print(f"\nlate-half drift mass @ {corruptions} sev {sev} "
          "(0.5 = uniform, >0.5 = late-heavy):")
    for (model, corr, s) in sorted(found):
        if s != sev or corr != corruptions[0]:
            continue
        label, _ = MODEL_LABELS.get(model, (model, "?"))
        vals = []
        for c in corruptions:
            key = (model, c, s)
            if key in found:
                dd = np.array([r["drop_from_clean"] for r in found[key]])
                vals.append(f"{c}={late_half_mass(dd):.3f}")
        print(f"  {label:<24} " + "  ".join(vals))
    return 0


if __name__ == "__main__":
    sys.exit(main())
