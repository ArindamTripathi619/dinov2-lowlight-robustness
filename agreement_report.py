"""Torch-free Track 1/3 gate report rendering, split out of run_drift_proxy.py.

run_drift_proxy.py imports torch at module scope, so nothing in it can be
imported in CI's torch-free environment. The report format itself is plain
text/JSON rendering — including the dataset-label regression of 2026-10-14
(the txt header hardcoded 'CIFAR-10 test' while profiling ImageNet) — and is
tested here (tests/run_tests.py::test_agreement_report_dataset_label).

run_drift_proxy.py re-exports PROXY_KEYS and write_agreement_report, so its
callers (run_drift_profile.py, the proxy runner itself) are unchanged.
"""

import json
import os

# Proxy candidates (drift score; higher = drifted more) + report order.
PROXY_KEYS = ["energy_drop", "mean_shift", "cov_shift", "cos_drop"]

DATASET_LABELS = {"cifar10": "CIFAR-10 test", "imagenet": "ImageNet-val extract"}


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
        f"images       : {meta['n_images']} "
        f"{DATASET_LABELS.get(meta.get('dataset', 'cifar10'), meta.get('dataset'))}, "
        f"seed {meta['seed']}",
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
