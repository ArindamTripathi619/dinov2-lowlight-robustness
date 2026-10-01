"""Readout-repair arm at GPU scale — ports upstream's H4 finding to our protocol.

Upstream (pilot n=120): retraining the linear readout on degraded images
recovers +22.5 pp at sev-4 with the backbone untouched (their F7), and
readout repair beats input-space enhancement (their F11). Our narrative
("representations rotate") never separated backbone-side damage from
readout-side staleness. This runner measures exactly that decomposition on
our artifact of record protocol (n=1000, seed 42, same probe/split machinery
as Phases 1-3):

  Arm A (fixed probe):   probe trained on clean embeddings, evaluated at each
                         severity  (== Phase 1 / 'Original' rows)
  Arm B (adapted probe): probe trained on severity-s images of the SAME
                         train fold, evaluated on severity-s test fold
                         (upstream's stricter control — no test-fold reuse)

Outputs per severity: accuracy, Wilson CI, top-1 share, entropy ratio,
dominant class — plus the paired permutation test A-vs-B per severity and
the recovery delta. Writes output/readout_repair/results.csv + summary.json.

Local CPU: ~40 min (embedding extraction dominates). GPU: ~5 min.
"""
import argparse
import json
import os

import numpy as np
import torch

from utils import (
    DEVICE, load_dinov2, load_cifar10_subset, DINOV2_PREPROCESS,
    low_light, get_corruption,
)
from stats_tools import (
    wilson_ci, paired_permutation_test, readout_collapse_metrics,
)

CIFAR_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                 "dog", "frog", "horse", "ship", "truck"]


@torch.no_grad()
def embed(model, images, device, batch_size=64):
    out = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        tensors = torch.stack([DINOV2_PREPROCESS(img) for img in batch]).to(device)
        feats = model(tensors)
        out.append(feats.cpu().numpy())
    return np.concatenate(out, axis=0)


def main():
    p = argparse.ArgumentParser(description="Readout-repair decomposition (upstream H4 at our scale)")
    p.add_argument("--corruption", default="low_light",
                   choices=["low_light", "blur", "jpeg", "contrast"])
    p.add_argument("--model", default="dinov2_vits14")
    p.add_argument("--n-images", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--probe-C", type=float, default=1.0)
    p.add_argument("--output", default="output/readout_repair")
    ARGS = p.parse_args()

    os.makedirs(ARGS.output, exist_ok=True)
    out_csv = os.path.join(ARGS.output, "results.csv")
    out_json = os.path.join(ARGS.output, "summary.json")

    np.random.seed(ARGS.seed)
    torch.manual_seed(ARGS.seed)

    corrupt = get_corruption(ARGS.corruption)
    model, _ = load_dinov2(DEVICE, ARGS.model)
    model.eval()

    print(f">>> drawing {ARGS.n_images} test images (seed {ARGS.seed})")
    images, labels = load_cifar10_subset(n_images=ARGS.n_images, train=False, seed=ARGS.seed)
    labels = np.asarray(labels)

    # Stratified 70/30 split — identical machinery to Phases 1-3
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(labels))
    idx_train, idx_test = train_test_split(idx, test_size=0.3,
                                           random_state=ARGS.seed, stratify=labels)
    y_train, y_test = labels[idx_train], labels[idx_test]

    print(f">>> clean embeddings ({len(images)} images, {DEVICE})")
    emb_clean = embed(model, images, DEVICE)

    print(">>> Arm A: fixed probe (trained on clean)")
    from sklearn.linear_model import LogisticRegression
    probe_fixed = LogisticRegression(max_iter=2000, C=ARGS.probe_C)
    probe_fixed.fit(emb_clean[idx_train], y_train)

    rows = ["corruption,severity,arm,accuracy,ci_low,ci_high,top1_share,"
            "n_classes_used,entropy_ratio,dominant_class,dominant_class_name,p_vs_fixed"]
    summary = {"corruption": ARGS.corruption, "model": ARGS.model,
               "n_images": ARGS.n_images, "seed": ARGS.seed, "severities": []}

    for severity in range(0, 6):
        print(f">>> severity {severity}")
        if severity == 0:
            emb_sev = emb_clean
        else:
            degraded = [corrupt(img, severity) for img in images]
            emb_sev = embed(model, degraded, DEVICE)

        # Arm A: fixed probe
        X_te = emb_sev[idx_test]
        pred_A = probe_fixed.predict(X_te)
        acc_A = float((pred_A == y_test).mean())
        ciA = wilson_ci((pred_A == y_test).astype(float))
        m_A = readout_collapse_metrics(y_test, pred_A)

        # Arm B: severity-adapted probe — trained on degraded versions of the
        # SAME train fold (upstream's stricter control; no test-fold reuse)
        if severity == 0:
            pred_B, ciB, m_B = pred_A, ciA, m_A
            p_val = 1.0
        else:
            X_tr_sev = emb_sev[idx_train]
            probe_adapted = LogisticRegression(max_iter=2000, C=ARGS.probe_C)
            probe_adapted.fit(X_tr_sev, y_train)
            pred_B = probe_adapted.predict(X_te)
            acc_B = float((pred_B == y_test).mean())
            ciB = wilson_ci((pred_B == y_test).astype(float))
            m_B = readout_collapse_metrics(y_test, pred_B)
            p_val = paired_permutation_test((pred_B == y_test).astype(float),
                                            (pred_A == y_test).astype(float),
                                            n_perm=5000, seed=ARGS.seed)["p_value"]

        for arm, pred, ci, m, pv in (("fixed", pred_A, ciA, m_A, 1.0),
                                     ("adapted", pred_B, ciB, m_B, p_val)):
            rows.append(f"{ARGS.corruption},{severity},{arm},{m['accuracy']:.4f},"
                        f"{ci['ci_low']:.4f},{ci['ci_high']:.4f},{m['top1_share']:.4f},"
                        f"{m['n_classes_used']},{m['entropy_ratio']:.4f},"
                        f"{m['dominant_class']},{CIFAR_CLASSES[m['dominant_class']]},{pv:.4f}")

        rec = {"severity": severity,
               "fixed_acc": acc_A, "adapted_acc": float((pred_B == y_test).mean()),
               "recovery_pp": 100 * (float((pred_B == y_test).mean()) - acc_A),
               "fixed_top1_share": m_A["top1_share"],
               "fixed_entropy_ratio": m_A["entropy_ratio"],
               "fixed_dominant": CIFAR_CLASSES[m_A["dominant_class"]],
               "p_paired": pv}
        summary["severities"].append(rec)
        print(f"    fixed {acc_A:.3f} | adapted {rec['adapted_acc']:.3f} "
              f"({rec['recovery_pp']:+.1f} pp) | top1 {m_A['top1_share']:.2f} "
              f"({rec['fixed_dominant']}) | p={pv:.4f}")

    with open(out_csv, "w") as f:
        f.write("\n".join(rows) + "\n")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved {out_csv} and {out_json}")

    print("\n--- readout-collapse signature (fixed probe) ---")
    for rec in summary["severities"]:
        print(f"  sev {rec['severity']}: top1_share={rec['fixed_top1_share']:.2f} "
              f"entropy_ratio={rec['fixed_entropy_ratio']:.2f} "
              f"dominant={rec['fixed_dominant']}")


if __name__ == "__main__":
    main()
