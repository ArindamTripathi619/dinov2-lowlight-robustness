# 10 — Context Index

*A navigation hub for the `docs/project-context/` knowledge base and the wider repo.*

## The knowledge base (this directory)

| File | Read it for |
|---|---|
| [`00_PROJECT_OVERVIEW.md`](00_PROJECT_OVERVIEW.md) | What the project is, headline findings, status. Start here. |
| [`01_REPOSITORY_MAP.md`](01_REPOSITORY_MAP.md) | Where everything lives; file counts; dead trees. |
| [`02_ARCHITECTURE.md`](02_ARCHITECTURE.md) | Layers, dependency direction, invariants, trust boundaries. |
| [`03_MODULE_REFERENCE.md`](03_MODULE_REFERENCE.md) | Per-module API with line numbers. |
| [`04_EXECUTION_AND_DATA_FLOWS.md`](04_EXECUTION_AND_DATA_FLOWS.md) | Config→results→paper→site flows; corruption & LoRA pipelines. |
| [`05_CONFIGURATION_AND_INTEGRATIONS.md`](05_CONFIGURATION_AND_INTEGRATIONS.md) | Config keys, dependencies, external services, CI/deploy. |
| [`06_TESTING_AND_QUALITY.md`](06_TESTING_AND_QUALITY.md) | Test runner, coverage, CI, verification checklist. |
| [`07_TECHNICAL_DEBT_AND_OPEN_QUESTIONS.md`](07_TECHNICAL_DEBT_AND_OPEN_QUESTIONS.md) | Bugs, duplication, stale docs, open decisions. |
| [`08_EXPLORATION_LEDGER.md`](08_EXPLORATION_LEDGER.md) | Full file-by-file recon record + coverage audit. |
| [`09_AI_WORKING_CONTEXT.md`](09_AI_WORKING_CONTEXT.md) | Operating manual for an agent/human picking this up cold. |
| `10_CONTEXT_INDEX.md` | This index. |

Root-level: [`../../AGENTS.md`](../../AGENTS.md) — agent operating instructions.

## Source docs (outside this directory)

| File | Read it for |
|---|---|
| `README.md` | Top-level quickstart, claims registry, section list. |
| `docs/RESEARCH.md` | Master research narrative + all results. |
| `docs/METHODS.md` | Methods, protocols, reference environment (§10). |
| `docs/DPA_DESIGN.md` | Diagnostic-guided parameter (LoRA rank) allocation design. |
| `docs/ROADMAP.md` | Track-based roadmap and current status. |
| `docs/DIVERGENCE_REPORT.md` | Fork↔upstream history; PR #1 merge record (read as history). |
| `docs/PAPER_RELATED_WORK.md` | Related work and novelty positioning (v1.1). |
| `paper/README.md` | Paper build instructions + **claim→artifact map**. |
| `paper/CVPR2027_CHECKLIST.md` | Submission-readiness audit. |
| `SECURITY.md` / `CITATION.cff` | Policy / citation. |

## Suggested reading orders

**New engineer (code):** `00` → `01` → `02` → `03` → `04` → `06` → `09`.

**Researcher (science):** `README.md` → `docs/RESEARCH.md` → `docs/METHODS.md` →
`docs/DPA_DESIGN.md` → `paper/README.md` (claim map) → `00` (status).

**AI agent about to modify code:** `09` → `07` → `03` (relevant module) →
`docs/DIVERGENCE_REPORT.md` (why code looks the way it does) → `06` (how to verify).

**Reviewer / auditor:** `07` → `08` → `05` (CI/deps) → `06` → `02` (trust boundaries).

## Quick facts

| Fact | Value |
|---|---|
| Ledger HEAD | `90577cbe171cefe98a551d05979bb8d74c6b986c` (`main`) |
| Remotes | `origin` = ArindamTripathi619 fork; `upstream` = officialarghya29 |
| Merge state | PR #1 merged 2026-10-01 (`9bc6b50`); single lineage |
| Tests | `python3 tests/run_tests.py` → 22/22 (custom runner, no pytest) |
| Core library | `src/` — 10 modules |
| Presentation | `index.html` (live site) ← built from `apps/presentation.py` — 1569 lines, 14 sections |
| Committed evidence | `results_pilot/` — 23 artifacts |
| Configs | `configs/experiments.yaml` (full), `configs/pilot_cpu.yaml` (CPU pilot) |
| CI | `ci.yml` (tests, py3.10/3.11), `security.yml` (weekly pip-audit) |
| Deploy | static site via Vercel (`tools/export_static_site.py`) |
| Excluded/ignored | `upstream_src/` (dead), `.vercel/`, `data/`, `output/`, `colab_results/`, `logs/` |

## Claim → artifact (condensed)

Full map in `paper/README.md`; key rows:

| Claim | Artifact |
|---|---|
| Grace/cliff/floor + null tests | `results_pilot/main_curve.csv`, `main_curve_nulltests.json` |
| Late-layer CKA gradient (ViT-S) | `results_pilot/cka_by_layer.csv`, `cka_summary.json` |
| ViT-B/14 replication | `results_pilot/vitb_*.csv`, `vitb_cka_summary.json` |
| Frequency dissociation | `results_pilot/frequency_tests.csv` |
| Readout repair | `results_pilot/mechanism_adapted_probes.csv` |
| Readout degeneracy | `results_pilot/prediction_collapse.csv` |
| Quantization null | `results_pilot/quantization.csv`, `paired_tests.json` |
| Enhancement baselines | `results_pilot/mitigation.csv` |
| Failure persistence/per-class | `results_pilot/failure_summary.json`, `failure_by_class.csv` |
| Seed × C sensitivity | `results_pilot/seed_sensitivity*.csv/json` |
| Efficiency | `results_pilot/efficiency.csv` |
