"""
Track 1: Drift-proxy feasibility gate (docs/ROADMAP.md section 2, Track 1)
==========================================================================

A drift proxy much cheaper than CKA that predicts WHERE adaptation pays
off, agreement-checked against per-module CKA on a non-ViT architecture
(torchvision ResNet-50, ImageNet-pretrained).

Protocol (forward-only, no backprop):
  The same images pass through the model twice per corruption family —
  clean (severity 0) and degraded (severity S) — while forward hooks
  record every profiled module's activations. Outputs are pooled to a
  per-sample feature vector (spatial mean for conv maps, CLS token for
  transformer blocks), so memory stays O(modules * n * d).

Per module, from the paired (clean, degraded) activations:

  Proxies (candidates to replace CKA as the cheap profiler):
    energy_drop = sum_i ||a_deg_i - a_clean_i||^2 / sum_i ||a_clean_i||^2
    mean_shift  = ||mu_deg - mu_clean||_2 / ||mu_clean||_2
    cov_shift   = ||Sigma_deg - Sigma_clean||_F / ||Sigma_clean||_F
    cos_drop    = 1 - mean_i cos(a_clean_i, a_deg_i)
  Reference (what the proxies must predict):
    cka / cka_unbiased  linear CKA(clean, degraded), biased + corrected
    cka_unbiased_drop   = 1 - cka_unbiased   (higher = drifted more)

  Scale note: CKA and cos_drop are scale-invariant; energy_drop,
  mean_shift and cov_shift are scale-sensitive (low-light shrinking
  activations registers as drift there). The candidate set deliberately
  spans both regimes — the gate decides which regime tracks CKA.

Go/no-go gate (docs/ROADMAP.md section 2):
  Spearman rho(proxy profile, CKA profile) >= 0.8 on >= 2 corruptions
  AND top-k modules by proxy contain the top-k modules by CKA
  (operationalized: full containment at k = 5 on those corruptions).
  Fallback if it fails: keep full CKA as the profiler — the paper claim
  weakens to "single cheap profiling pass" but does not die.

Outputs (output/drift_proxy/):
  proxy_profiles.csv                per (corruption, module): proxies + CKA
  drift_profile_sev{S}_{corr}.csv   layer,severity,cka,drop_from_clean schema
                                    (compatible with run_lora_simple_colab.py
                                    --drift-csv for later allocation runs)
  agreement_report.{json,txt}       Spearman + top-k containment + gate verdict
  profiles_{corr}.png               profile curves + proxy-vs-CKA scatters
  profile_heatmap.png               CKA drop, modules x corruptions

Usage:
  python3 run_drift_proxy.py --n-images 100     # smoke
  python3 run_drift_proxy.py --n-images 1000    # full pass (CPU OK)

The harness is model-agnostic (Track 3 reuses it): module selection =
containers at depth <= --max-depth plus known pooling leaves; activation
pooling handles conv (4D), token (3D) and flat (2D) outputs.
"""

import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import torchvision.transforms as T
from scipy.stats import spearmanr

from utils import DEVICE, get_corruption, linear_cka, load_cifar10_subset, load_dinov2
from stats_tools import linear_cka_unbiased


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

RESNET_PREPROCESS = T.Compose([
    T.ToTensor(),
    T.Resize((224, 224), antialias=True),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# Identical to utils.DINOV2_PREPROCESS — both families use ImageNet
# mean/std at 224x224, so one transform serves every model in the harness.
PREPROCESS = RESNET_PREPROCESS


def load_model(model_name, device):
    """Load a pretrained model by name. Extend here for Track 3 (timm)."""
    if model_name.startswith("dinov2"):
        model, _ = load_dinov2(model_name=model_name)
        return model, f"torch.hub {model_name}"
    if model_name == "resnet50":
        import torchvision.models as tvm
        model = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V1)
        return model, "torchvision resnet50 IMAGENET1K_V1 (76.130%)"
    raise ValueError(
        f"Unknown model {model_name!r}; supported: resnet50, dinov2_*. "
        "timm-backed models land with Track 3."
    )


# ---------------------------------------------------------------------------
# Model-agnostic module selection + activation recording
# ---------------------------------------------------------------------------

# Leaf modules whose output is the pre-classifier pooled representation.
LEAF_POOL_NAMES = ("avgpool", "global_pool", "gap", "head.global_pool")


def select_profiled_modules(model, max_depth=2):
    """Choose modules to profile: containers at depth <= max_depth (per-stage
    and per-block granularity) plus known pooling leaves.

    Model-agnostic: depth 2 on torchvision ResNet-50 gives the 4 stages + 16
    bottlenecks (+ avgpool as the readout); on ViT it gives the transformer
    blocks; on timm ConvNeXt/Swin it gives stems + stages.

    Returns dict name -> module in forward (top-down) order.
    """
    selected = {}
    for name, module in model.named_modules():
        if name == "":
            continue
        depth = name.count(".")
        is_container = len(list(module.children())) > 0
        is_pool_leaf = any(name == s or name.endswith("." + s) for s in LEAF_POOL_NAMES)
        if (is_container and depth <= max_depth) or is_pool_leaf:
            selected[name] = module
    return selected


def pool_activation(out, token_mode="cls"):
    """Reduce a module output to one feature vector per sample (B, D).

    4D (B, C, H, W) -> spatial mean over H, W      (channel means)
    3D (B, S, D)    -> CLS token ([:, 0, :]) or sequence mean (token_mode)
    2D (B, D)       -> unchanged
    tuple/list      -> first tensor element
    Returns None for anything else (module skipped).
    """
    if isinstance(out, (tuple, list)):
        out = next((t for t in out if torch.is_tensor(t)), None)
        if out is None:
            return None
    if not torch.is_tensor(out):
        return None
    if out.dim() == 4:
        return out.mean(dim=(2, 3))
    if out.dim() == 3:
        return out[:, 0, :] if token_mode == "cls" else out.mean(dim=1)
    if out.dim() == 2:
        return out
    return None


class ActivationRecorder:
    """Forward hooks that pool and accumulate each module's activations."""

    def __init__(self, modules, token_mode="cls"):
        self.token_mode = token_mode
        self.store = {name: [] for name in modules}
        self.handles = [
            module.register_forward_hook(self._make_hook(name))
            for name, module in modules.items()
        ]

    def _make_hook(self, name):
        def hook(_module, _inp, out):
            vec = pool_activation(out, self.token_mode)
            if vec is not None:
                self.store[name].append(vec.detach().to("cpu", torch.float32))
        return hook

    def finalize(self):
        """Concatenate to (n, d) numpy arrays; drop modules that never fired
        (e.g. nn.ModuleList-style containers iterated manually)."""
        acts = {}
        for name, parts in self.store.items():
            if parts:
                acts[name] = torch.cat(parts, dim=0).numpy()
            else:
                print(f"  WARNING: module {name!r} never fired; excluded from profile")
        return acts

    def remove(self):
        for handle in self.handles:
            handle.remove()


@torch.no_grad()
def forward_pass(model, images, modules, batch_size, device, token_mode="cls",
                 preprocess=None):
    """Run images through the model once; return {module_name: (n, d) array}."""
    preprocess = preprocess or PREPROCESS
    recorder = ActivationRecorder(modules, token_mode=token_mode)
    t0 = time.time()
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        tensors = torch.stack([preprocess(img) for img in batch]).to(device)
        model(tensors)
    acts = recorder.finalize()
    recorder.remove()
    print(f"  forward pass: {len(images)} images in {time.time() - t0:.1f}s")
    return acts

# ---------------------------------------------------------------------------
# Proxies + CKA reference, per module
# ---------------------------------------------------------------------------

# Proxy candidates (drift score; higher = drifted more) + report order.
PROXY_KEYS = ["energy_drop", "mean_shift", "cov_shift", "cos_drop"]


def module_proxies(act_clean, act_deg):
    """Forward-only drift proxies + CKA reference for one module.

    act_clean, act_deg: (n, d) float arrays, row-paired (same images).
    """
    Ac = act_clean.astype(np.float64)
    Ad = act_deg.astype(np.float64)
    n, d = Ac.shape
    diff = Ad - Ac

    # --- proxy 1: activation-energy drop ||Da||^2 / ||a||^2 ---
    energy_clean = float((Ac ** 2).sum())
    energy_drop = float((diff ** 2).sum()) / (energy_clean + 1e-12)

    # --- proxy 2: feature-statistics shift (mean, then covariance) ---
    mu_c, mu_d = Ac.mean(0), Ad.mean(0)
    mean_shift = float(np.linalg.norm(mu_d - mu_c)) / (float(np.linalg.norm(mu_c)) + 1e-12)
    Sig_c = (Ac - mu_c).T @ (Ac - mu_c) / (n - 1)
    Sig_d = (Ad - mu_d).T @ (Ad - mu_d) / (n - 1)
    cov_shift = (float(np.linalg.norm(Sig_d - Sig_c, "fro"))
                 / (float(np.linalg.norm(Sig_c, "fro")) + 1e-12))

    # --- proxy 3: cosine-of-activations (cheapest CKA relative) ---
    def _row_norm(M):
        return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
    cos_sim = float((_row_norm(Ac) * _row_norm(Ad)).sum(axis=1).mean())

    # --- reference: CKA, biased (artifact-compatible) + unbiased ---
    cka = float(linear_cka(act_clean, act_deg))
    cka_u = float(linear_cka_unbiased(Ac, Ad))

    return {
        "n_images": int(n), "feat_dim": int(d),
        "energy_drop": energy_drop, "mean_shift": mean_shift,
        "cov_shift": cov_shift, "cos_sim": cos_sim, "cos_drop": 1.0 - cos_sim,
        "cka": cka, "cka_unbiased": cka_u,
        "cka_drop": 1.0 - cka, "cka_unbiased_drop": 1.0 - cka_u,
    }


def build_profile_rows(acts_clean, acts_deg, module_order, corruption, severity):
    """One row per (corruption, module) with proxies + CKA reference."""
    rows = []
    for idx, name in enumerate(module_order):
        p = module_proxies(acts_clean[name], acts_deg[name])
        rows.append({"corruption": corruption, "severity": severity,
                     "layer": idx, "module": name, **p})
    return rows


# ---------------------------------------------------------------------------
# Agreement: proxy profile vs CKA profile, and the go/no-go gate
# ---------------------------------------------------------------------------

def profile_agreement(rows, topk_values=(3, 5, None)):
    """Spearman rho + top-k containment of each proxy vs the CKA profile.

    Reference ranking: cka_unbiased_drop (higher = drifted more). All proxy
    scores are likewise higher = drifted more, so positive rho = agreement.
    topk_values entries may be None meaning ceil(20% of modules).
    """
    n = len(rows)
    ref = np.array([r["cka_unbiased_drop"] for r in rows])
    order_ref = np.argsort(-ref)
    ks = [min(k if k is not None else int(np.ceil(0.2 * n)), n) for k in topk_values]

    out = {}
    for key in PROXY_KEYS:
        proxy = np.array([r[key] for r in rows])
        if np.std(proxy) == 0 or np.std(ref) == 0:
            rho, p = float("nan"), float("nan")
        else:
            rho, p = spearmanr(proxy, ref)
        order_proxy = np.argsort(-proxy)
        out[key] = {
            "spearman_rho": float(rho), "spearman_p": float(p),
            "topk_containment": {
                str(k): len(set(order_ref[:k]) & set(order_proxy[:k])) / k
                for k in ks
            },
        }
    return out


def evaluate_gate(agreement_by_corruption, n_modules, rho_threshold=0.8, topk=5):
    """Track 1 gate: rho >= threshold on >= 2 corruptions AND full top-k
    containment on those corruptions, for at least one proxy candidate.

    Returns dict with verdict GO / SOFT-PASS (rho gate ok, containment
    partial — a judgment call for DPA_DESIGN) / NO-GO, plus per-proxy detail.
    """
    k = min(topk, n_modules)
    per_proxy = {}
    for key in PROXY_KEYS:
        rho, contain = {}, {}
        for corr, ag in agreement_by_corruption.items():
            rho[corr] = ag[key]["spearman_rho"]
            contain[corr] = ag[key]["topk_containment"][str(k)]
        strict = [c for c in rho if rho[c] >= rho_threshold and contain[c] >= 1.0]
        rho_only = [c for c in rho if rho[c] >= rho_threshold]
        per_proxy[key] = {
            "rho": rho, "containment_at_k": contain, "k": int(k),
            "corruptions_passing_strict": strict,
            "corruptions_passing_rho": rho_only,
            "strict_pass": len(strict) >= 2,
        }

    def _score(key):
        return float(np.nansum(list(per_proxy[key]["rho"].values())))

    strict_proxies = [k2 for k2, v in per_proxy.items() if v["strict_pass"]]
    soft_proxies = [k2 for k2, v in per_proxy.items()
                    if not v["strict_pass"] and len(v["corruptions_passing_rho"]) >= 2]
    if strict_proxies:
        verdict, best = "GO", max(strict_proxies, key=_score)
    elif soft_proxies:
        verdict, best = "SOFT-PASS", max(soft_proxies, key=_score)
    else:
        verdict = "NO-GO"
        best = max(per_proxy, key=_score)
    return {"verdict": verdict, "best_proxy": best, "per_proxy": per_proxy,
            "rho_threshold": rho_threshold, "topk": int(k),
            "gate_criterion": (f"Spearman rho >= {rho_threshold} on >= 2 corruptions "
                               f"AND full containment at k={k}")}

# ---------------------------------------------------------------------------
# Outputs: CSVs, report, plots
# ---------------------------------------------------------------------------

PROFILE_FIELDS = ["corruption", "severity", "layer", "module", "n_images",
                  "feat_dim", "energy_drop", "mean_shift", "cov_shift",
                  "cos_sim", "cos_drop", "cka", "cka_unbiased", "cka_drop",
                  "cka_unbiased_drop"]


def write_proxy_profiles_csv(rows_by_corruption, path):
    """All rows, all corruptions: the raw artifact of record."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PROFILE_FIELDS,
                                extrasaction="ignore")
        writer.writeheader()
        for rows in rows_by_corruption.values():
            writer.writerows(rows)
    print(f"  Saved {path}")


def write_drift_profile_csvs(rows_by_key, out_dir, severity=None):
    """Per-key CKA profile in the cka_matrix.csv schema
    (layer,severity,cka,drop_from_clean) that run_lora_simple_colab.py
    --drift-csv consumes for rank allocation.

    Keys are corruption names (severity from the `severity` argument or the
    rows themselves) or "corr@sevS" sweep keys (run_drift_profile.py).
    """
    for key, rows in rows_by_key.items():
        corr = str(key).split("@sev")[0]
        sev = int(rows[0]["severity"]) if rows else severity
        path = os.path.join(out_dir, f"drift_profile_sev{sev}_{corr}.csv")
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["layer", "severity", "module",
                                                   "cka", "cka_unbiased",
                                                   "drop_from_clean",
                                                   "cka_unbiased_drop"])
            writer.writeheader()
            for r in rows:
                writer.writerow({"layer": r["layer"], "severity": sev,
                                 "module": r["module"], "cka": r["cka"],
                                 "cka_unbiased": r["cka_unbiased"],
                                 "drop_from_clean": r["cka_drop"],
                                 "cka_unbiased_drop": r["cka_unbiased_drop"]})
        print(f"  Saved {path}")


def _fmt(v):
    return "nan" if v != v else f"{v:.4f}"


def write_agreement_report(rows_by_corruption, agreement_by_corruption, gate,
                           meta, out_dir):
    """Human-readable txt + machine-readable json gate report."""
    lines = [
        "=" * 72,
        "DRIFT-PROXY FEASIBILITY REPORT (Track 1 gate, docs/ROADMAP.md section 2)",
        "=" * 72,
        f"model        : {meta['model_desc']}",
        f"device       : {meta['device']} | torch {meta['torch_version']}",
        f"images       : {meta['n_images']} CIFAR-10 test, seed {meta['seed']}",
        f"degradation  : severity {meta['severity']} | corruptions: {', '.join(meta['corruptions'])}",
        f"modules      : {meta['n_modules']} profiled (forward-only, no backprop)",
        "",
    ]
    for corr, rows in rows_by_corruption.items():
        lines.append(f"--- {corr} (severity {rows[0]['severity']}) ---")
        header = (f"{'module':<22}{'energy':>9}{'mean':>9}{'cov':>9}{'cosd':>9}"
                  f"{'CKAu':>9}{'CKAu drop':>10}")
        lines.append(header)
        for r in rows:
            lines.append(
                f"{r['module']:<22}{_fmt(r['energy_drop']):>9}"
                f"{_fmt(r['mean_shift']):>9}{_fmt(r['cov_shift']):>9}"
                f"{_fmt(r['cos_drop']):>9}{_fmt(r['cka_unbiased']):>9}"
                f"{_fmt(r['cka_unbiased_drop']):>10}")
        lines.append("")
        lines.append(f"  proxy agreement vs CKA profile (ref = cka_unbiased_drop):")
        for key in PROXY_KEYS:
            a = agreement_by_corruption[corr][key]
            tk = "  ".join(f"top{k}={v:.2f}" for k, v in a["topk_containment"].items())
            lines.append(f"    {key:<12} rho={_fmt(a['spearman_rho'])} "
                         f"(p={_fmt(a['spearman_p'])})  {tk}")
        lines.append("")

    lines += ["-" * 72, "GATE (roadmap: " + gate["gate_criterion"] + ")", "-" * 72]
    for key, v in gate["per_proxy"].items():
        rho_str = ", ".join(f"{c}={_fmt(r)}" for c, r in v["rho"].items())
        c_str = ", ".join(f"{c}={_fmt(r)}" for c, r in v["containment_at_k"].items())
        lines.append(f"  {key:<12} rho: {rho_str}")
        lines.append(f"  {'':<12} top{v['k']} containment: {c_str}")
    best = gate["best_proxy"]
    lines += ["",
              f"  best proxy: {best}",
              f"  VERDICT: {gate['verdict']}",
              ""]
    if gate["verdict"] == "NO-GO":
        lines.append("  Fallback per roadmap: keep full CKA as the profiler (it is")
        lines.append("  forward-only and already proven); claim weakens to 'single cheap")
        lines.append("  profiling pass' — the paper does not die here.")
        lines.append("")

    txt_path = os.path.join(out_dir, "agreement_report.txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved {txt_path}")

    json_path = os.path.join(out_dir, "agreement_report.json")
    with open(json_path, "w") as f:
        json.dump({"meta": meta, "gate": gate,
                   "agreement_by_corruption": agreement_by_corruption},
                  f, indent=2, default=float)
    print(f"  Saved {json_path}")


def _minmax(v):
    v = np.asarray(v, dtype=float)
    lo, hi = np.nanmin(v), np.nanmax(v)
    return (v - lo) / (hi - lo + 1e-12)


def plot_profile_panels(rows, agreement, corruption, out_path):
    """Top: per-module drift profiles (min-max normalized for shape compare).
    Bottom: proxy-vs-CKA scatters with Spearman rho."""
    import matplotlib.pyplot as plt
    from utils import save_fig

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.4, 1])
    ax = fig.add_subplot(gs[0, :])
    x = np.arange(len(rows))
    series = [("cka_unbiased_drop", "CKA drop (reference)"),
              ("energy_drop", "energy drop"), ("mean_shift", "mean shift"),
              ("cov_shift", "cov shift"), ("cos_drop", "1 - cosine")]
    for key, label in series:
        ax.plot(x, _minmax([r[key] for r in rows]), marker="o", markersize=3,
                label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([r["module"] for r in rows], rotation=60, ha="right",
                       fontsize=7)
    ax.set_ylabel("drift score (min-max normalized per curve)")
    ax.set_title(f"Per-module drift profile — {corruption}")
    ax.legend(fontsize=8)
    for j, key in enumerate(PROXY_KEYS):
        ax_s = fig.add_subplot(gs[1, j])
        ax_s.scatter([r["cka_unbiased_drop"] for r in rows],
                     [r[key] for r in rows], s=18, color="tab:blue")
        ax_s.set_xlabel("CKA drop", fontsize=8)
        ax_s.set_ylabel(key, fontsize=8)
        rho = agreement[key]["spearman_rho"]
        ax_s.set_title(f"rho={rho:.3f}", fontsize=9)
    fig.tight_layout()
    save_fig(fig, out_path)


def plot_drift_heatmap(rows_by_corruption, module_order, out_path):
    """CKA drop heatmap, modules (rows) x corruptions (columns)."""
    import matplotlib.pyplot as plt
    from utils import save_fig

    corruptions = list(rows_by_corruption)
    lookup = {c: {r["module"]: r["cka_unbiased_drop"] for r in rows}
              for c, rows in rows_by_corruption.items()}
    M = np.array([[lookup[c][m] for c in corruptions] for m in module_order])
    fig, ax = plt.subplots(figsize=(2.2 + 1.8 * len(corruptions),
                                    0.25 * len(module_order) + 2))
    im = ax.imshow(M, aspect="auto", cmap="magma", vmin=0.0,
                   vmax=max(1e-6, float(np.nanmax(M))))
    plt.colorbar(im, ax=ax, label="CKA drop (unbiased, clean -> degraded)")
    ax.set_xticks(range(len(corruptions)))
    ax.set_xticklabels(corruptions, rotation=30, ha="right")
    ax.set_yticks(range(len(module_order)))
    ax.set_yticklabels(module_order, fontsize=6)
    ax.set_title("Where does drift live? (reference CKA)")
    fig.tight_layout()
    save_fig(fig, out_path)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Track 1: drift-proxy vs CKA agreement gate (docs/ROADMAP.md)")
    p.add_argument("--model", default="resnet50",
                   help="resnet50 (torchvision) or dinov2_* (torch.hub)")
    p.add_argument("--n-images", type=int, default=1000)
    p.add_argument("--severity", type=int, default=5)
    p.add_argument("--corruptions", default="blur,low_light",
                   help="comma-separated subset of blur,low_light,jpeg,contrast")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42, help="image-draw seed")
    p.add_argument("--data-root", default="./data")
    p.add_argument("--output", default="output/drift_proxy")
    p.add_argument("--max-depth", type=int, default=2,
                   help="max module depth for container selection (Track 3: raise)")
    p.add_argument("--vit-token", default="cls", choices=["cls", "mean"],
                   help="pooling for 3D token outputs (transformer blocks)")
    p.add_argument("--rho-threshold", type=float, default=0.8)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    corruptions = [c.strip() for c in args.corruptions.split(",") if c.strip()]

    print("=" * 72)
    print(f"Track 1: drift-proxy feasibility — model={args.model}, "
          f"n={args.n_images}, severity={args.severity}")
    print(f"Device: {DEVICE}")
    print("=" * 72)

    images, _labels = load_cifar10_subset(n_images=args.n_images, train=False,
                                          seed=args.seed, data_root=args.data_root)
    model, model_desc = load_model(args.model, DEVICE)
    modules = select_profiled_modules(model, max_depth=args.max_depth)
    print(f"Profiling {len(modules)} modules: {list(modules)}")

    # Clean pass: severity-0 degradation is the identity for every family,
    # so use the pristine images directly (no corruption rng consumed).
    print("\n>>> clean pass (severity 0)")
    acts_clean = forward_pass(model, images, modules, args.batch_size, DEVICE,
                              token_mode=args.vit_token)
    module_order = [m for m in modules if m in acts_clean]

    rows_by_corruption, agreement_by_corruption = {}, {}
    for corr in corruptions:
        fn = get_corruption(corr)
        rng = np.random.default_rng(1000 + args.severity)  # matched-noise convention
        print(f"\n>>> corruption: {corr} @ severity {args.severity}")
        degraded = [fn(img, args.severity, rng=rng) for img in images]
        acts_deg = forward_pass(model, degraded, modules, args.batch_size, DEVICE,
                                token_mode=args.vit_token)
        rows = build_profile_rows(acts_clean, acts_deg, module_order, corr,
                                  args.severity)
        rows_by_corruption[corr] = rows
        agreement_by_corruption[corr] = profile_agreement(rows)

    gate = evaluate_gate(agreement_by_corruption, len(module_order),
                         rho_threshold=args.rho_threshold)

    meta = {
        "model": args.model, "model_desc": model_desc,
        "torch_version": torch.__version__, "device": DEVICE,
        "n_images": len(images), "seed": args.seed,
        "severity": args.severity, "corruptions": corruptions,
        "n_modules": len(module_order), "modules": module_order,
        "vit_token": args.vit_token, "max_depth": args.max_depth,
    }

    print("\n>>> outputs")
    write_proxy_profiles_csv(rows_by_corruption,
                             os.path.join(args.output, "proxy_profiles.csv"))
    write_drift_profile_csvs(rows_by_corruption, args.output, args.severity)
    write_agreement_report(rows_by_corruption, agreement_by_corruption, gate,
                           meta, args.output)

    print("\n>>> plots")
    for corr, rows in rows_by_corruption.items():
        plot_profile_panels(rows, agreement_by_corruption[corr], corr,
                            os.path.join(args.output, f"profiles_{corr}.png"))
    plot_drift_heatmap(rows_by_corruption, module_order,
                       os.path.join(args.output, "profile_heatmap.png"))

    print("\n" + "=" * 72)
    print(f"GATE VERDICT: {gate['verdict']} (best proxy: {gate['best_proxy']})")
    for key, v in gate["per_proxy"].items():
        rho_str = ", ".join(f"{c}={v['rho'][c]:.3f}" for c in corruptions)
        print(f"  {key:<12} rho: {rho_str}")
    print("=" * 72)


if __name__ == "__main__":
    main()
