"""Re-index Track-2 v8 drift-profile CSVs for the Phase-3 LoRA consumer.

METHODS §7.8: the v8 CSVs key `layer` by profiled-module row (0-36); the LoRA
consumer (run_lora_simple_colab.py::load_drift_profile) expects dense block
indices 0..n_blocks-1 over the 12 `blocks.k` whole-block rows.

    python3 tools/reindex_drift_for_lora.py --indir output/vitb_profile
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

FIELDS = ["layer", "severity", "module", "drop_from_clean"]


def reindex(src: Path, dst: Path) -> int:
    with open(src, newline="") as f:
        rows = list(csv.DictReader(f))
    blocks = [r for r in rows if re.fullmatch(r"blocks\.\d+", r["module"])]
    blocks.sort(key=lambda r: int(r["module"].split(".")[1]))
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for i, r in enumerate(blocks):
            w.writerow({
                "layer": i,
                "severity": r["severity"],
                "module": r["module"],
                "drop_from_clean": r["drop_from_clean"],
            })
    return len(blocks)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--indir", default="output/vitb_profile")
    p.add_argument("--outdir", default="output/vitb_profile/v9_lora")
    args = p.parse_args()

    indir, outdir = Path(args.indir), Path(args.outdir)
    srcs = sorted(indir.glob("drift_profile_sev5_*.csv"))
    if not srcs:
        raise SystemExit(f"no drift_profile_sev5_*.csv under {indir}")
    for src in srcs:
        corruption = src.stem.removeprefix("drift_profile_sev5_")
        dst = outdir / f"drift_sev5_{corruption}.csv"
        n = reindex(src, dst)
        if n != 12:
            raise SystemExit(f"{src}: expected 12 whole-block rows, got {n}")
        print(f"{src.name} -> {dst} ({n} blocks)")


if __name__ == "__main__":
    main()
