#!/usr/bin/env python3
"""Aggregate SSR S4 6-model-grid results (docs/SSR_PLAN.md §7.4, verdict RESEARCH §6.10).

Parses each leg dir ``<slug>_<scope>_k<k>_<corr>_s<seed>`` under output/ssr_s4/,
reads ssr_results.csv, and emits what the S4 verdict rests on:

  1. replication table: ssr_topk delta-vs-frozen as mean ± sd across the 3
     seeds (k = 25) per (model, scope, corruption, severity);
  2. attachment verdict at sev5 low_light — stream vs sublayer (vs norm where
     present) per model: does the S3 "attachment flips the repair" result
     generalize to ConvNeXt/Swin/DeiT/DINOv2?;
  3. profiled-dedup control (seed 42) — the output-identity dedup in
     run_ssr.py makes the profiled arm single-application; report its delta
     and the modules it dropped;
  4. integrity: identity gate bit-identical everywhere + dedup recorded.

Run: python3 scripts/analyze_ssr_s4.py [<harvest>/output/ssr_s4]
"""
import csv
import glob
import json
import math
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SSR4 = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "output", "ssr_s4")

DIR_RE = re.compile(
    r"^(?P<slug>.+)_(?P<scope>profiled|stream|sublayer|norm)_"
    r"k(?P<k>\d+)_(?P<corr>low_light|jpeg)_s(?P<seed>\d+)$")

MODEL_NAMES = {
    "timm:convnext_tiny": "ConvNeXt-T",
    "timm:swin_tiny_patch4_window7_224": "Swin-T",
    "timm:deit_small_patch16_224": "DeiT-S",
    "dinov2_vitb14": "DINOv2-B",
}
ORDER = ["timm:convnext_tiny", "timm:swin_tiny_patch4_window7_224",
         "timm:deit_small_patch16_224", "dinov2_vitb14"]


def name(slug):
    return MODEL_NAMES.get(slug, slug)


def read_leg(res_dir):
    path = os.path.join(res_dir, "ssr_results.csv")
    if not os.path.isfile(path):
        return None
    rows = list(csv.DictReader(open(path)))
    out = {}
    for r in rows:
        out.setdefault(r["arm"], {})[int(r["severity"])] = float(r["delta_vs_frozen"])
    return out


def collect(base):
    legs = []
    for d in sorted(glob.glob(os.path.join(base, "*"))):
        if not os.path.isdir(d):
            continue
        m = DIR_RE.match(os.path.basename(d))
        if not m:
            continue
        meta = m.groupdict()
        meta["slug"] = (meta["slug"].replace("_", ":", 1)
                        if meta["slug"].startswith("timm_") else meta["slug"])
        deltas = read_leg(d)
        if deltas is not None:
            legs.append((meta, deltas, d))
    return legs


def mean_sd(xs):
    if not xs:
        return (float("nan"), float("nan"))
    mu = sum(xs) / len(xs)
    var = sum((x - mu) ** 2 for x in xs) / (len(xs) - 1) if len(xs) > 1 else 0.0
    return mu, math.sqrt(var)


def pct(x):
    return "n/a " if x is None else f"{100 * x:+.1f}"


def main():
    if not os.path.isdir(SSR4):
        print(f"missing {SSR4} — run not harvested yet", file=sys.stderr)
        return 1
    legs = collect(SSR4)
    print(f"parsed {len(legs)} complete legs under {SSR4}")

    # --- 1. replication table: ssr_topk delta across seeds (k=25 legs) ---
    print("\n=== replication (k=25): ssr_topk delta vs frozen, mean±sd across 3 seeds ===")
    agg = {}
    for meta, deltas, _ in legs:
        if meta["k"] != "25" or meta["scope"] == "profiled":
            continue
        key = (meta["slug"], meta["scope"], meta["corr"])
        for sev in sorted(deltas.get("ssr_topk", {})):
            agg.setdefault((key, sev), []).append(deltas["ssr_topk"][sev])
    print(f"{'model':<10}{'scope':<10}{'corr':<11}{'sev':<5}{'n':<4}{'delta mean±sd':<16}")
    for (key, sev), xs in sorted(agg.items(), key=lambda kv: (ORDER.index(kv[0][0][0]) if kv[0][0][0] in ORDER else 99, kv[0][0][1], kv[0][0][2], kv[0][1])):
        slug, scope, corr = key
        mu, sd = mean_sd(xs)
        print(f"{name(slug):<10}{scope:<10}{corr:<11}{sev:<5}{len(xs):<4}{pct(mu)}±{100 * sd:.1f}")

    # --- 2. attachment verdict at sev5 low_light: stream vs sublayer vs norm ---
    print("\n=== attachment verdict (low_light sev5, k25): does scope flip the repair? ===")
    print(f"{'model':<10}{'stream':<12}{'sublayer':<12}{'norm':<12}")
    by = {}
    for meta, deltas, _ in legs:
        if meta["k"] != "25" or meta["corr"] != "low_light" or meta["scope"] == "profiled":
            continue
        v = deltas.get("ssr_topk", {}).get(5)
        by.setdefault(meta["slug"], {})[meta["scope"]] = v
    for slug in ORDER:
        if slug not in by:
            continue
        row = by[slug]
        print(f"{name(slug):<10}{pct(row.get('stream')):<12}"
              f"{pct(row.get('sublayer')):<12}{pct(row.get('norm')):<12}")

    # --- 3. profiled-dedup control (seed 42) ---
    print("\n=== profiled-dedup control (seed 42): single-application profiled arm ===")
    print(f"{'model':<10}{'corr':<11}{'sev5 delta':<13}{'dedup dropped'}")
    for meta, deltas, d in sorted(legs, key=lambda t: (ORDER.index(t[0]['slug']) if t[0]['slug'] in ORDER else 99, t[0]['corr'])):
        if meta["scope"] != "profiled":
            continue
        v = deltas.get("ssr_topk", {}).get(5)
        dropped = []
        mp = os.path.join(d, "ssr_meta.json")
        if os.path.isfile(mp):
            dropped = json.load(open(mp)).get("dedup_dropped", [])
        print(f"{name(meta['slug']):<10}{meta['corr']:<11}{pct(v):<13}{dropped}")

    # --- 4. integrity: identity gate + dedup recorded ---
    ident_ok, ident_bad, dedup_runs = 0, [], 0
    for meta, _, d in legs:
        mp = os.path.join(d, "ssr_meta.json")
        if not os.path.isfile(mp):
            continue
        j = json.load(open(mp))
        if j.get("identity", {}).get("bit_identical"):
            ident_ok += 1
        else:
            ident_bad.append(d)
        if j.get("dedup_dropped"):
            dedup_runs += 1
    print(f"\nidentity bit-identical: {ident_ok} ok, {len(ident_bad)} bad")
    for b in ident_bad:
        print(f"  BROKEN: {b}")
    print(f"legs where dedup dropped same-tensor containers: {dedup_runs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
