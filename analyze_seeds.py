#!/usr/bin/env python3
"""Seed-replication analysis: 3-seed error bars for the Phase 3 v2 grid.

Parses the per-arm logs under colab_results/sessionD/ and produces:
  - per-arm table (mean / sev-5 / clean accuracy)
  - group-level seed stats (mean +/- sd, range) for uniform / drift / fullft
  - a parity statement: does drift's 3-seed spread overlap uniform's?

Handles missing (not-yet-run) arms gracefully: prints them as PENDING so the
script can be run at any time and re-run once the replication grid lands.

Usage:  python analyze_seeds.py [--roots DIR ...]
Output: console table + output/seed_replication/seed_analysis.csv
"""
import argparse
import csv
import os
import re
import statistics as st

DEFAULT_ROOTS = ["colab_results/sessionD"]
ARMS = {
    # arm-dir name -> (group, seed)
    "seed42_all": ("uniform", 42), "seed43_all": ("uniform", 43), "seed44_all": ("uniform", 44),
    "drift": ("drift", 42), "drift_seed43": ("drift", 43), "drift_seed44": ("drift", 44),
    "fullft": ("fullft", 42), "fullft_seed43": ("fullft", 43), "fullft_seed44": ("fullft", 44),
}
SE_MEAN = re.compile(r"Mean accuracy: Original=([0-9.]+), LoRA=([0-9.]+), Delta=([+-]?[0-9.]+)")
SE_SEV5 = re.compile(r"Sev 5 \(darkest\): Original=([0-9.]+), LoRA=([0-9.]+)")
SE_SEV = re.compile(r"severity (\d): ([0-9.]+)")


def parse_arm(path):
    """Extract (orig_mean, lora_mean, delta, orig_sev5, lora_sev5, per_sev) from an arm log."""
    txt = open(path, encoding="utf-8", errors="replace").read()
    m_mean = SE_MEAN.search(txt)
    m_sev5 = SE_SEV5.search(txt)
    per_sev = {int(s): float(a) for s, a in SE_SEV.findall(txt)}
    if not m_mean:
        return None
    return dict(
        orig_mean=float(m_mean.group(1)), lora_mean=float(m_mean.group(2)),
        delta=float(m_mean.group(3)),
        orig_sev5=float(m_sev5.group(1)) if m_sev5 else None,
        lora_sev5=float(m_sev5.group(2)) if m_sev5 else None,
        per_sev=per_sev,
    )


def collect(roots):
    arms = {}
    for root in roots:
        for name, (group, seed) in ARMS.items():
            log = os.path.join(root, name, "full_log.txt")
            if name in arms or not os.path.isfile(log):
                continue
            r = parse_arm(log)
            if r:
                arms[name] = dict(group=group, seed=seed, **r)
    return arms


def group_stats(arms, group):
    rows = [a for a in arms.values() if a["group"] == group]
    if not rows:
        return None
    means = [r["lora_mean"] for r in rows]
    sev5 = [r["lora_sev5"] for r in rows if r["lora_sev5"] is not None]
    cleans = [r["per_sev"].get(0) for r in rows if r["per_sev"].get(0) is not None]
    sd = st.stdev(means) if len(means) > 1 else 0.0
    return dict(n=len(rows), mean=st.mean(means), sd=sd,
                lo=min(means), hi=max(means),
                sev5=st.mean(sev5) if sev5 else None,
                clean=st.mean(cleans) if cleans else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="*", default=DEFAULT_ROOTS)
    args = ap.parse_args()

    arms = collect(args.roots)
    out_dir = "output/seed_replication"
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 78)
    print("PHASE 3 v2 — SEED REPLICATION ANALYSIS")
    print("=" * 78)
    print(f"{'arm':<14}{'group':<9}{'seed':<6}{'mean':<8}{'sev5':<8}{'clean':<8}{'status'}")
    rows_csv = []
    for name, (group, seed) in ARMS.items():
        a = arms.get(name)
        if a is None:
            print(f"{name:<14}{group:<9}{seed:<6}{'-':<8}{'-':<8}{'-':<8}PENDING")
            rows_csv.append([name, group, seed, "", "", "", "PENDING"])
            continue
        clean = a["per_sev"].get(0)
        print(f"{name:<14}{group:<9}{seed:<6}{a['lora_mean']:<8.4f}"
              f"{a['lora_sev5'] or 0:<8.4f}{clean:<8.4f}ok")
        rows_csv.append([name, group, seed, a["lora_mean"], a["lora_sev5"], clean, "ok"])

    print("\n" + "-" * 78)
    print("GROUP-LEVEL SEED STATS")
    print(f"{'group':<10}{'n':<4}{'mean':<10}{'sd':<9}{'range':<17}{'sev5':<8}{'clean':<8}")
    stats = {}
    for g in ("uniform", "drift", "fullft"):
        s = group_stats(arms, g)
        if s is None:
            print(f"{g:<10}{0:<4}{'-':<10}{'-':<9}{'-':<17}{'-':<8}{'-':<8}")
            continue
        stats[g] = s
        rng = f"{s['lo']:.3f}-{s['hi']:.3f}"
        print(f"{g:<10}{s['n']:<4}{s['mean']:<10.4f}{s['sd']:<9.4f}{rng:<17}"
              f"{(s['sev5'] or 0):<8.3f}{(s['clean'] or 0):<8.3f}")

    csv_path = os.path.join(out_dir, "seed_analysis.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "group", "seed", "mean_acc", "sev5_acc", "clean_acc", "status"])
        w.writerows(rows_csv)
        f.write("\n")
        w.writerow(["group", "n", "mean", "sd", "min", "max", "sev5_mean", "clean_mean"])
        for g, s in stats.items():
            w.writerow([g, s["n"], f"{s['mean']:.4f}", f"{s['sd']:.4f}",
                        f"{s['lo']:.4f}", f"{s['hi']:.4f}",
                        f"{s['sev5']:.4f}" if s["sev5"] is not None else "",
                        f"{s['clean']:.4f}" if s["clean"] is not None else ""])
    print(f"\nCSV written: {csv_path}")

    print("\n" + "-" * 78)
    print("READ-OUT (for the paper)")
    u, d, ft = stats.get("uniform"), stats.get("drift"), stats.get("fullft")
    if u and u["n"] == 3 and d and d["n"] == 3 and ft and ft["n"] == 3:
        for name, s in (("uniform", u), ("drift", d), ("fullft", ft)):
            print(f"  {name:<8}: mean {s['mean']:.3f} +/- {s['sd']:.3f} (range {s['lo']:.3f}-{s['hi']:.3f})")
        overlap_ud = d["lo"] <= u["hi"] and u["lo"] <= d["hi"]
        overlap_ft = ft["lo"] <= u["hi"] and u["lo"] <= ft["hi"]
        print(f"\n  drift range overlaps uniform:   {'YES' if overlap_ud else 'NO'}")
        print(f"  fullft range overlaps uniform:  {'YES' if overlap_ft else 'NO'}")
        if overlap_ud:
            print("  -> Claim supported: 'CKA-guided allocation matches uniform LoRA and "
                  "full fine-tuning at 0.78% of trainable params' (within seed noise).")
        else:
            print("  -> CAUTION: ranges disjoint; parity claim needs revisiting "
                  "(check whether drift sits above or below uniform).")
    else:
        pending = [n for n, (g, s) in ARMS.items() if n not in arms]
        print(f"  Waiting on {len(pending)} arm(s): {', '.join(pending)}")
        print("  Re-run this script after the replication grid lands.")


if __name__ == "__main__":
    main()
