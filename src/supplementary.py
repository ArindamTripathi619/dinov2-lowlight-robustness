"""Supplementary paper experiments (CVPR checklist: efficiency, robustness of
findings, failure analysis, generalization).

Experiments
-----------
- efficiency        Params / FLOPs / latency / throughput for frozen vs LoRA arms.
- failure_analysis  Flip dynamics, per-class breakdown, failure-persistence, grids.
- seed_sensitivity  Re-runs the main curve across split seeds + probe C values.
- cross_dataset     Zero-shot transfer of the CIFAR-10 pipeline to STL-10.

All experiments write CSV/JSON into the harness outdir and are consumable by
make_paper_tables.py / README generation.
"""

import itertools
import time

import numpy as np
import torch

from src.common import (get_device, matplotlib_agg, save_csv, save_json,
                        seed_everything)
from src.corruptions import low_light
from src.data import CIFAR10_CLASS_NAMES, load_cifar10_subsets
from src.models import EmbeddingExtractor, load_dinov2
from src.probes import fit_probe, stratified_split


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _measure_flops(model: torch.nn.Module, device: str):
    """FLOPs of a single 224x224 forward via torch.profiler (best effort)."""
    try:
        from torch.profiler import ProfilerActivity, profile
        was_training = model.training
        model.eval()
        x = torch.randn(1, 3, 224, 224, device=device)
        with torch.no_grad():
            with profile(activities=[ProfilerActivity.CPU], with_flops=True) as prof:
                model(x)
        if was_training:
            model.train()
        return float(sum(e.flops for e in prof.key_averages()))
    except Exception:
        return None


def _measure_latency(model: torch.nn.Module, device: str, n_warmup: int = 5,
                     n_iter: int = 30) -> dict:
    """Median wall-clock latency per image, batch=1, eval mode."""
    model.eval()
    x = torch.randn(1, 3, 224, 224, device=device)
    is_cuda = device.startswith("cuda")
    with torch.no_grad():
        for _ in range(n_warmup):
            model(x)
            if is_cuda:
                torch.cuda.synchronize()
        times = []
        for _ in range(n_iter):
            t0 = time.perf_counter()
            model(x)
            if is_cuda:
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    ms = 1000.0 * float(np.median(times))
    return {"latency_ms_per_image": ms, "throughput_img_per_s": 1000.0 / ms}


def _severities(cfg) -> list:
    return list(range(len(cfg["corruption"]["brightness_factors"])))


# ---------------------------------------------------------------------------
# Efficiency
# ---------------------------------------------------------------------------

def exp_efficiency(cfg, device, outdir):
    """Params / FLOPs / latency for the frozen backbone and LoRA arms.

    Accuracy columns are joined from other experiments' CSVs when present so
    this table is self-contained in the paper.
    """
    import os
    m = cfg["model"]
    seed_everything(cfg["seed"])
    model = load_dinov2(m["backbone"], device)
    rows = []

    frozen = _measure_latency(model, device)
    flops = _measure_flops(model, device)
    rows.append({
        "arm": "frozen_backbone",
        "backbone_params_M": _count_params(model) / 1e6,
        "trainable_params_M": 0.0,
        "flops_G": (flops / 1e9) if flops else "",
        "latency_ms": frozen["latency_ms_per_image"],
        "throughput_img_per_s": frozen["throughput_img_per_s"],
    })

    # LoRA arms: instantiate adapters to count params; latency measured with
    # adapters active (extra compute is the scaled-product path).
    try:
        from src.lora import apply_lora
        for rank in cfg["lora"]["ranks"]:
            seeded = load_dinov2(m["backbone"], device)
            alpha = rank * cfg["lora"]["alpha_multiplier"]
            apply_lora(seeded, rank=rank, alpha=alpha,
                       target=cfg["lora"]["targets"][-1])
            lat = _measure_latency(seeded, device)
            tr = sum(p.numel() for p in seeded.parameters() if p.requires_grad) / 1e6
            rows.append({
                "arm": f"lora_r{rank}",
                "backbone_params_M": _count_params(seeded) / 1e6,
                "trainable_params_M": tr,
                "flops_G": "",
                "latency_ms": lat["latency_ms_per_image"],
                "throughput_img_per_s": lat["throughput_img_per_s"],
            })
    except Exception as e:  # LoRA module drift should not kill the table
        print(f"  [efficiency] lora arms skipped: {e}")

    # Join accuracies if available
    def _rows_safe(path):
        try:
            from src.common import read_csv_rows
            return read_csv_rows(path)
        except Exception:
            return None
    mc = _rows_safe(os.path.join(outdir, "main_curve.csv"))
    la = _rows_safe(os.path.join(outdir, "lora_ablation.csv"))
    if mc:
        clean = next(r for r in mc if int(r["severity"]) == 0)
        dark = [float(r["accuracy"]) for r in mc if int(r["severity"]) > 0]
        rows[0]["acc_clean"] = float(clean["accuracy"])
        rows[0]["acc_dark_mean"] = float(np.mean(dark))
    if la:
        for r in la:
            key = f"lora_r{r['rank']}"
            row = next((x for x in rows if x["arm"] == key), None)
            if row is not None:
                row["acc_dark_mean"] = float(r["mean_acc"])

    save_csv(rows, f"{outdir}/efficiency.csv")
    save_json({"flops_G_frozen": flops / 1e9 if flops else None,
               "device": device, "note": "latency: batch=1 median of 30 runs"},
              f"{outdir}/efficiency.json")
    print(f"  efficiency: frozen {rows[0]['latency_ms']:.1f} ms/img "
          f"({rows[0]['throughput_img_per_s']:.1f} img/s), "
          f"flops={rows[0]['flops_G']}", flush=True)


# ---------------------------------------------------------------------------
# Failure analysis
# ---------------------------------------------------------------------------

def exp_failure_analysis(cfg, device, outdir):
    """How the model fails: flips, persistence, per-class death order, grids."""
    d, m, p = cfg["data"], cfg["model"], cfg["probe"]
    seed_everything(cfg["seed"])
    images, labels, _, _, _, _ = load_cifar10_subsets(d["n_test"], d["n_train"], d["split_seed"])
    model = load_dinov2(m["backbone"], device)
    extractor = EmbeddingExtractor(model, device)
    idx_tr, idx_te = stratified_split(labels, d["test_fraction"], d["split_seed"])
    probe = fit_probe(extractor([low_light(images[i], 0) for i in idx_tr]),
                      labels[idx_tr], C=p["C"], max_iter=p["max_iter"])
    class_names = CIFAR10_CLASS_NAMES

    sev_range = _severities(cfg)
    preds, corrects = {}, {}
    for s in sev_range:
        E = extractor([low_light(images[i], s) for i in idx_te], batch_size=64)
        preds[s] = probe.predict(E)
        corrects[s] = (preds[s] == labels[idx_te]).astype(float)

    clean_ok = corrects[0].astype(bool)
    rows = []
    for s in sev_range:
        ok = corrects[s].astype(bool)
        flip_to_wrong = int(np.sum(clean_ok & ~ok))
        recovered = int(np.sum(~clean_ok & ok))
        rows.append({
            "severity": s,
            "flip_clean_ok_to_wrong": flip_to_wrong,
            "flip_clean_wrong_to_ok": recovered,
            "n_wrong": int((~ok).sum()),
            "accuracy": float(ok.mean()),
        })
    # persistence: failures at sev1 still failing at max severity
    s_max = max(sev_range)
    fail1 = ~corrects[1].astype(bool)
    fail_max = ~corrects[s_max].astype(bool)
    persistence = float((fail1 & fail_max).sum() / max(1, fail1.sum()))
    rows.append({"severity": "persistence_sev1_to_max", "n_wrong": int(fail1.sum()),
                 "accuracy": persistence})

    # per-class death order: sev at which class accuracy first drops below 50%
    by_class_rows = []
    for c, name in enumerate(class_names):
        mask = labels[idx_te] == c
        n_c = int(mask.sum())
        first_below50 = None
        accs = {}
        for s in sev_range:
            acc = float(corrects[s][mask].mean())
            accs[s] = acc
            if first_below50 is None and acc < 0.5:
                first_below50 = s
        by_class_rows.append({"class": name, "n_test": n_c,
                              "acc_clean": accs[0], "acc_sev1": accs[1],
                              "acc_sev3": accs[3], "acc_sev5": accs[s_max],
                              "first_sev_below50": first_below50 if first_below50 is not None else ">" + str(s_max)})

    # qualitative grid: 8 clean-correct -> sev5-wrong examples
    grid_png = ""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        flip_idx = [j for j in range(len(idx_te))
                    if clean_ok[j] and not corrects[s_max].astype(bool)[j]][:8]
        if flip_idx:
            fig, axes = plt.subplots(2, 4, figsize=(11, 5.6))
            for ax, j in zip(axes.ravel(), flip_idx):
                i = idx_te[j]
                ax.imshow(images[i])
                ax.set_title(f"{class_names[labels[i]]} -> "
                             f"{class_names[preds[s_max][j]]}", fontsize=9)
                ax.axis("off")
            fig.suptitle(f"Clean-correct images misclassified at severity {s_max}")
            fig.tight_layout()
            grid_png = f"{outdir}/failure_grid.png"
            fig.savefig(grid_png, dpi=140)
            plt.close(fig)
    except Exception as e:
        print(f"  [failure] grid skipped: {e}")

    save_csv(rows[:-1], f"{outdir}/failure_summary.csv")
    save_json({"persistence_sev1_to_max": persistence,
               "flip_clean_ok_to_wrong_sev5": rows[-2]["flip_clean_ok_to_wrong"] if len(rows) > 1 else None,
               "failure_grid": grid_png}, f"{outdir}/failure_summary.json")
    save_csv(by_class_rows, f"{outdir}/failure_by_class.csv")
    print(f"  failure: sev1->max persistence={persistence:.2f}, "
          f"flips at max sev={rows[-2]['flip_clean_ok_to_wrong']}", flush=True)


# ---------------------------------------------------------------------------
# Seed / hyperparameter sensitivity
# ---------------------------------------------------------------------------

def exp_seed_sensitivity(cfg, device, outdir):
    """Main curve across 3 split seeds x probe C in {0.1, 1, 10}.

    Establishes that the reported degradation is not an artifact of one split
    or one probe regularization choice (CVPR reproducibility item).
    """
    d, m, p = cfg["data"], cfg["model"], cfg["probe"]
    model = load_dinov2(m["backbone"], device)
    extractor = EmbeddingExtractor(model, device)
    sev_range = _severities(cfg)
    rows = []

    for seed in [42, 43, 44]:
        seed_everything(seed)
        images, labels, _, _, _, _ = load_cifar10_subsets(d["n_test"], d["n_train"], seed)
        idx_tr, idx_te = stratified_split(labels, d["test_fraction"], seed)
        E_tr = extractor([low_light(images[i], 0) for i in idx_tr], batch_size=64)
        E_te = {s: extractor([low_light(images[i], s) for i in idx_te], batch_size=64)
                for s in sev_range}
        for C in [0.1, 1.0, 10.0]:
            probe = fit_probe(E_tr, labels[idx_tr], C=C, max_iter=p["max_iter"])
            for s in sev_range:
                acc = float((probe.predict(E_te[s]) == labels[idx_te]).mean())
                rows.append({"split_seed": seed, "probe_C": C, "severity": s,
                             "accuracy": acc})
        accs = {r["severity"]: r["accuracy"] for r in rows[-3 * len(sev_range):]
                if r["probe_C"] == 1.0}
        print(f"  seed={seed}: clean={accs[0]:.3f} sev3={accs[3]:.3f} "
              f"sev5={accs[max(sev_range)]:.3f}", flush=True)

    save_csv(rows, f"{outdir}/seed_sensitivity.csv")

    # summary: per severity mean/std over the 9 (seed, C) runs
    summary = {}
    for s in sev_range:
        accs = [r["accuracy"] for r in rows if r["severity"] == s]
        summary[s] = {"mean": float(np.mean(accs)), "std": float(np.std(accs)),
                      "min": float(np.min(accs)), "max": float(np.max(accs))}
    save_json(summary, f"{outdir}/seed_sensitivity_summary.json")
    print(f"  sensitivity: clean mean={summary[0]['mean']:.3f}±{summary[0]['std']:.3f}, "
          f"sev5 mean={summary[sev_range[-1]]['mean']:.3f}±{summary[sev_range[-1]]['std']:.3f}",
          flush=True)


# ---------------------------------------------------------------------------
# Cross-dataset transfer (STL-10)
# ---------------------------------------------------------------------------

def _load_stl10_shared_classes(n_test: int, seed: int):
    """STL-10 test images restricted to the 9 classes shared with CIFAR-10.

    'monkey' has no CIFAR-10 counterpart and is dropped; STL-10 'car' maps to
    CIFAR-10 'automobile' (the only non-identical shared name pair).
    Returns (images uint8 HWC, labels in CIFAR-10 ids, n_skipped).
    """
    import torchvision
    ds = torchvision.datasets.STL10(root="data", split="test", download=True)
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(ds))

    cifar_names = CIFAR10_CLASS_NAMES
    name_to_cifar = {n: i for i, n in enumerate(cifar_names)}
    # explicit cross-dataset name mapping (car -> automobile)
    stl_to_cifar = {i: name_to_cifar["automobile"] if n == "car" else name_to_cifar[n]
                    for i, n in enumerate(ds.classes) if n != "monkey"}

    images, labels = [], []
    n_skipped = 0
    for j in order:
        img, lbl = ds[j]
        if lbl not in stl_to_cifar:
            n_skipped += 1
            continue
        if len(images) >= n_test:
            break
        images.append(np.asarray(img))                      # HWC uint8, 96x96
        labels.append(stl_to_cifar[lbl])
    return images, np.array(labels), n_skipped


def exp_cross_dataset(cfg, device, outdir):
    """Zero-shot transfer: CIFAR-10-trained probe evaluated on STL-10 under
    the same low-light ladder (same severity scale, no retraining).

    Tests whether the degradation curve is a property of the backbone+corruption
    interaction rather than of CIFAR-10 specifically.
    """
    d, m, p = cfg["data"], cfg["model"], cfg["probe"]
    seed_everything(cfg["seed"])

    cifar_images, cifar_labels, _, _, _, _ = load_cifar10_subsets(
        d["n_test"], d["n_train"], d["split_seed"])
    model = load_dinov2(m["backbone"], device)
    extractor = EmbeddingExtractor(model, device)
    idx_tr, _ = stratified_split(cifar_labels, d["test_fraction"], d["split_seed"])
    probe = fit_probe(extractor([low_light(cifar_images[i], 0) for i in idx_tr]),
                      cifar_labels[idx_tr], C=p["C"], max_iter=p["max_iter"])

    stl_images, stl_labels, n_skipped = _load_stl10_shared_classes(
        d["n_test"], d["split_seed"])
    rows = []
    for s in _severities(cfg):
        E = extractor([low_light(img, s) for img in stl_images], batch_size=64)
        acc = float((probe.predict(E) == stl_labels).mean())
        rows.append({"dataset": "stl10_shared9", "severity": s, "n_test": len(stl_images),
                     "accuracy": acc})
        print(f"  stl10 sev {s}: acc={acc:.3f}", flush=True)

    save_csv(rows, f"{outdir}/cross_dataset.csv")
    save_json({"n_stl10_used": int(len(stl_images)),
               "n_skipped_monkey": n_skipped,
               "note": "probe trained on CIFAR-10 clean train fold; zero-shot on STL-10; car->automobile mapped"},
              f"{outdir}/cross_dataset_summary.json")
    print(f"  cross-dataset: stl10 clean={rows[0]['accuracy']:.3f} "
          f"sev5={rows[-1]['accuracy']:.3f}", flush=True)
