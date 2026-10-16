#!/usr/bin/env python3
"""Aggregate SSR S3-r1 replication results (docs/RESEARCH.md §6.9).

Parses each leg dir name ``<slug>_<scope>_k<k>_<corr>_s<seed>`` under
output/ssr_s3/, reads ssr_results.csv, and emits three things the §6.9
verdict rests on:

  1. per (model, scope, corr, severity) the ssr_topk delta-vs-frozen as
     mean ± sd across the 3 replication seeds (k = 25);
  2. the S2 parity check — the low_light seed-42 k25 legs must reproduce
     the S2 headline (ResNet stream +4.3 @sev5, ViT sublayer -1.4 @sev5);
  3. the k-matched control — k=1 (ResNet) / k=3 (ViT) vs the k=25 primary,
     isolating location from scope size.
"""
import csv
import glob
import math
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Defaults are the committed results of record; pass dirs as argv to point
# at a fresh harvest, e.g. `python3 scripts/analyze_ssr_s3.py <harvest>/output/ssr_s3 <harvest>/output/ssr_s2`.
SSR3 = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "output", "ssr_s3")
SSR2 = sys.argv[2] if len(sys.argv) > 2 else os.path.join(REPO, "output", "ssr_s2")

DIR_RE = re.compile(
    r"^(?P<slug>.+)_(?P<scope>profiled|stream|sublayer|norm)_"
    r"k(?P<k>\d+)_(?P<corr>low_light|jpeg)_s(?P<seed>\d+)$")
# S2 legs carry no k/seed in the dir name: <slug>_<scope>_<corr>
DIR_RE_S2 = re.compile(
    r"^(?P<slug>.+)_(?P<scope>profiled|stream|sublayer|norm)_"
    r"(?P<corr>low_light|jpeg)$")


def read_leg(res_dir):
    """Return {arm: {severity: delta}} for one leg, or None if incomplete."""
    path = os.path.join(res_dir, "ssr_results.csv")
    if not os.path.isfile(path):
        return None
    rows = list(csv.DictReader(open(path)))
    out = {}
    for r in rows:
        arm = r["arm"]
        sev = int(r["severity"])
        out.setdefault(arm, {})[sev] = float(r["delta_vs_frozen"])
    return out


def collect(base, pattern=None):
    """Parse all leg dirs under base -> list of (meta, deltas)."""
    regex = pattern or DIR_RE
    legs = []
    for d in sorted(glob.glob(os.path.join(base, "*"))):
        if not os.path.isdir(d):
            continue
        m = regex.match(os.path.basename(d))
        if not m:
            continue
        meta = m.groupdict()
        meta.setdefault("k", "25")
        meta.setdefault("seed", "42")
        meta["slug"] = meta["slug"].replace("_", ":", 1) if meta["slug"].startswith("timm_") else meta["slug"]
        deltas = read_leg(d)
        if deltas is not None:
            legs.append((meta, deltas))
    return legs


def mean_sd(xs):
    if not xs:
        return (float("nan"), float("nan"))
    mu = sum(xs) / len(xs)
    var = sum((x - mu) ** 2 for x in xs) / (len(xs) - 1) if len(xs) > 1 else 0.0
    return mu, math.sqrt(var)


def pct(x):
    return f"{100 * x:+.1f}"


def main():
    if not os.path.isdir(SSR3):
        print(f"missing {SSR3} — run not harvested yet", file=sys.stderr)
        return 1
    legs = collect(SSR3)
    print(f"parsed {len(legs)} complete legs under {SSR3}")

    # --- 1. replication table: ssr_topk delta across seeds (k25 legs) ---
    print("\n=== replication (k=25): ssr_topk delta vs frozen, mean±sd across seeds ===")
    agg = {}
    for meta, deltas in legs:
        if meta["k"] != "25":
            continue
        key = (meta["slug"], meta["scope"], meta["corr"])
        for sev in sorted(deltas.get("ssr_topk", {})):
            agg.setdefault((key, sev), []).append(deltas["ssr_topk"][sev])
    print(f"{'model':<26}{'scope':<10}{'corr':<11}{'sev':<5}{'n':<4}{'delta mean±sd':<16}")
    for (key, sev), xs in sorted(agg.items()):
        slug, scope, corr = key
        mu, sd = mean_sd(xs)
        model = "ResNet50" if "resnet50" in slug else "ViT-S/16"
        print(f"{model:<26}{scope:<10}{corr:<11}{sev:<5}{len(xs):<4}{pct(mu)}±{100 * sd:.1f}")

    # --- 2. S2 parity: low_light seed 42 k25 vs the S2 headline ---
    print("\n=== S2 parity check (low_light, seed 42, k25) ===")
    s2 = collect(SSR2, DIR_RE_S2)
    s2_map = {(m["slug"], m["scope"]): d for m, d in s2}
    for meta, deltas in legs:
        if meta["k"] != "25" or meta["seed"] != "42" or meta["corr"] != "low_light":
            continue
        if meta["scope"] not in ("stream", "sublayer"):
            continue
        key = (meta["slug"], meta["scope"])
        model = "ResNet50" if "resnet50" in meta["slug"] else "ViT-S/16"
        for sev in (1, 5):
            s3_v = deltas.get("ssr_topk", {}).get(sev)
            s2_v = s2_map.get(key, {}).get("ssr_topk", {}).get(sev)
            if s3_v is None:
                continue
            print(f"  {model:<11}{meta['scope']:<9}sev{sev}: S3={pct(s3_v)}  "
                  f"S2={'n/a' if s2_v is None else pct(s2_v)}")

    # --- 3. k-matched control: k25 vs absolute k (seed 42) ---
    print("\n=== k-matched control (seed 42): k25 vs absolute k ===")
    kmap = {}
    for meta, deltas in legs:
        if meta["seed"] != "42":
            continue
        kmap.setdefault((meta["slug"], meta["scope"], meta["corr"], meta["k"]), deltas)
    print(f"{'model':<26}{'scope':<10}{'corr':<11}{'sev':<5}{'k25':<9}{'k_abs':<9}")
    for (slug, scope, corr, k), deltas in sorted(kmap.items()):
        if k == "25":
            continue
        d25 = kmap.get((slug, scope, corr, "25"))
        if not d25:
            continue
        model = "ResNet50" if "resnet50" in slug else "ViT-S/16"
        for sev in (1, 5):
            a = d25.get("ssr_topk", {}).get(sev)
            b = deltas.get("ssr_topk", {}).get(sev)
            if a is None or b is None:
                continue
            print(f"{model:<26}{scope:<10}{corr:<11}{sev:<5}{pct(a):<9}{pct(b):<9}")

    # --- integrity: identity gate held everywhere? ---
    ident_ok, ident_bad = 0, []
    for meta, _ in legs:
        p = os.path.join(SSR3, "_".join([meta["slug"].replace(":", "_"), meta["scope"],
                                         "k" + meta["k"], meta["corr"], "s" + meta["seed"]]),
                         "ssr_meta.json")
        if os.path.isfile(p):
            import json
            bit = json.load(open(p)).get("identity", {}).get("bit_identical")
            if bit:
                ident_ok += 1
            else:
                ident_bad.append(p)
    print(f"\nidentity bit-identical: {ident_ok} ok, {len(ident_bad)} bad")
    for b in ident_bad:
        print(f"  BROKEN: {b}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
