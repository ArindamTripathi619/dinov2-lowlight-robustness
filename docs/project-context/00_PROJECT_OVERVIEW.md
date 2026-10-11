# 00 — Project Overview

## Identity

| Field | Value |
|---|---|
| Name | DINOv2 Low-Light Robustness Study |
| Type | Research codebase (experiments + paper + presentation) |
| Repository | `github.com/ArindamTripathi619/dinov2-lowlight-robustness` (fork) |
| Upstream | `github.com/officialarghya29/dinov2-lowlight-robustness` |
| Ledger HEAD | `90577cbe171cefe98a551d05979bb8d74c6b986c` (`main`) |
| Paper title (working) | "When Vision Goes Dark" (CVPR-style) |
| License/citation | `CITATION.cff` present |

## Problem statement

Foundation vision models such as DINOv2 are widely reported as robust out-of-distribution,
but standard corruption benchmarks do not include an extreme *photometric* low-light axis.
This project asks three questions and answers them with controlled experiments on
CIFAR-10-derived inputs and a frozen backbone:

1. **Measure** — How does frozen DINOv2 + a linear readout degrade across a synthetic
   low-light severity axis (0–5)?
2. **Localize** — Which layers and which input-frequency bands carry the damage, and
   what is the failure mechanism?
3. **Remediate** — Does a cheap, diagnostic-guided intervention (readout repair,
   enhancement, or LoRA rank allocation) recover robustness?

A fourth strand studies whether the synthetic-darkness findings transfer to **real**
dark images (ExDark).

## Core findings (headline)

| Finding | Number |
|---|---|
| Accuracy collapse across severity | mean accuracy 0.913 → 0.083 (sev 0→5) |
| Representation drift (clean-vs-corrupted cosine) | 1.00 → 0.16 |
| Depth localization | CKA drop 0.81 at block 10 (vs 0.58 at block 0) |
| Best synthetic mitigation | +0.22 mean accuracy, 5.1× worst-case over baseline |
| Drift-weighted LoRA | matches uniform LoRA at 0.78% of trainable params |
| Readout staleness | +25.3 pp at sev-4 (p=0.0002), +30.0 pp at sev-5 |
| Sim-to-real (ExDark) | frozen DINOv2 0.725 on 7,363 real dark photos; CLAHE +0.001 |

**Provenance note (verified 2nd pass):** the headline numbers above are the
*full-scale (GPU, n=1000)* values reported in `docs/RESEARCH.md`. The **committed
`results_pilot/` artifacts are pilot-scale (n=60)** and differ accordingly — e.g.
`main_curve.csv` shows 0.933 → 0.108 with cosine drift 1.00 → 0.16;
`cka_by_layer.csv` peaks at 0.816 (block 11); `mechanism_adapted_probes.csv` recovers
+22.5 pp at sev-4. Treat each number with its provenance: full-scale claims are backed
by `docs/RESEARCH.md` + `colab_results/`, pilot claims by `results_pilot/`.

## Three-phase design

- **Phase 1 — Measure.** `run_notebook1*`. Accuracy/noise curves, null tests, bootstrap CIs.
- **Phase 2 — Localize.** `run_notebook2*`. Per-layer CKA, frequency dissociation,
  mechanism probes (logit-lens, AOPC), prediction collapse.
- **Phase 3 — Remediate.** `run_notebook3*`, `run_lora_*`. Enhancement baselines,
  readout repair, LoRA rank allocation (incl. drift-weighted), full-FT baseline.

Plus **ExDark sim-to-real** (`run_exdark_baseline.py`), **ViT-B generality**
(`run_vitb_generality.py`), **corollaries** (`run_corollaries.py`), **collapse analysis**
(`run_collapse_analysis.py`), and **seed sensitivity** (`analyze_seeds.py`).

## Scientific positioning

A *well-executed synthesis study with one incremental mechanism* (per
`docs/PAPER_RELATED_WORK.md` §2.7). Novel claims:
1. Within-model, within-eval-set linkage of frequency fragility to late-layer CKA drift,
   used prescriptively.
2. CKA-allocated LoRA reaching full-FT parity at <1% trainable params on this failure mode
   (directionally extending RepSAM's diagnostic-first allocation).
3. A quantified sim-to-real boundary for synthetic low-light robustness.

Scope is explicitly limited: one small backbone (ViT-S/14, + ViT-B replication), one
dataset (CIFAR-10-derived), a synthetic axis.

## Deliverables in-repo

| Artifact | Location |
|---|---|
| Experiment harness | `run_experiments.py`, `run_notebook{1,2,3}*.py`, `run_lora_*`, runner scripts |
| Core library | `src/` (10 modules) |
| Committed pilot results | `results_pilot/` (23 files) |
| Paper | `paper/` (TeX, auto-generated tables/figures, PDFs) |
| Site build source | `apps/presentation.py` (14 sections, renders the static site) |
| Static website | `tools/export_static_site.py` → `index.html` → Vercel |
| Docs | `README.md`, `docs/*.md` |
| Tests + CI | `tests/run_tests.py`, `.github/workflows/` |

## Status

- PR #1 merged 2026-10-01; PR #2 open (fork→upstream, MERGEABLE); upstream and fork share
  one lineage; both content sets preserved.
- 22/22 unit tests pass; static site published to Vercel at HEAD.
- Track 2 v9 done at n=3 (2026-10-10): drift-weighted LoRA at parity with uniform on
  ViT-B while training 24% fewer params (RESEARCH §6.5); deterministic-noise residual
  closed (DIVERGENCE_REPORT §8.5).
- Open: Track 3 cross-family atlas; v9 allocation ablations; full-scale paper numbers
  (pilot-scale bars/macros currently in the text).

## Where to start reading

1. `README.md` — quickstart and claims registry.
2. `docs/RESEARCH.md` — the science.
3. `docs/project-context/02_ARCHITECTURE.md` + `03_MODULE_REFERENCE.md` — the code.
4. `docs/project-context/09_AI_WORKING_CONTEXT.md` — how to operate safely.
