# 02 — Architecture

## Architectural style

A **research pipeline**, not an application. There is no web server, database, or service
mesh. The architecture is a layered experiment harness with a deliberately thin library
(`src/`) shared by many entry-point scripts, plus two output surfaces (a paper and a static
presentation site).

```
┌──────────────────────────────────────────────────────────────────────────┐
│  Entry points (scripts / notebooks)                                        │
│  run_experiments.py · run_notebook{1,2,3}* · run_lora_* · run_exdark_* ·   │
│  run_vitb_generality.py · run_readout_repair.py · analyze_seeds.py · …     │
└───────────────┬───────────────────────────────────────────────────────────┘
                │  sys.path.insert(0, repo_root); from src.X import …
                ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  src/ library (torch-optional where possible)                              │
│                                                                            │
│  data ──► corruptions ──► models ──► probes ──► analysis                   │
│    │                        │            │          │                      │
│    └────────── common ──────┴────────────┴──────────┘                      │
│                    │                             │                         │
│                  lora                         manifest                     │
│                    │                             │                         │
│              supplementary                       │                         │
└──────────────────────────────────────────────────────────────────────────┘
                │                                   │
                ▼                                   ▼
        results_pilot/*.csv/json            manifest.jsonl (config SHA-256)
                │
     ┌──────────┴───────────┐
     ▼                      ▼
make_paper_{tables,figures}   apps/presentation.py (site build source, 14 sections)
     │                          │
     ▼                          ▼
 paper/results,figures    tools/export_static_site.py ──► static site ──► Vercel
```

## Layers

### 1. Configuration layer
`configs/*.yaml` define all experiment parameters. The harness computes a SHA-256 of the
config and appends one JSON line per experiment to `manifest.jsonl` in the results
directory (config SHA-256 embedded), so any result is bound to the exact config
that produced it. Two configs: full-scale (`experiments.yaml`) and CPU pilot
(`pilot_cpu.yaml`).

### 2. Library layer — `src/`
Ten modules with a mostly linear dependency flow:

- `common` — torch-optional utilities (`_torch()` lazy import, `seed_everything`,
  `get_device`). No heavy deps.
- `data` — CIFAR-10 loading + seeded splits (depends on common).
- `corruptions` — synthetic low-light + band-limited (low/high-pass) transforms.
- `models` — DINOv2 loader via `torch.hub` + feature extraction.
- `probes` — logistic-regression probe, logit-lens, AOPC.
- `analysis` — curve fitting, null/permutation tests, CKA aggregation.
- `lora` — LoRA adapters, rank allocation, training loop.
- `manifest` — result-manifest hashing/provenance.
- `supplementary` — secondary experiments; **lazy-imported** by
  `run_experiments.py:566` so the base harness does not require its deps.

### 3. Orchestration layer — `run_experiments.py`
The single registry-based harness. Exposes an argparse `--experiment` selector with
**11 experiment functions** — `main_curve, cka, frequency, mechanism, quantization,
lora_ablation, mitigation, efficiency, failure_analysis, seed_sensitivity,
cross_dataset` — plus an `all` choice (12 argparse choices). Each key dispatches to
a function that drives the `src/` library and writes into the `--outdir` target
(harness default `results/`); the committed pilot artifacts live in `results_pilot/`,
produced by `run_pilot_cpu.py` (hardcoded `OUTDIR = "results_pilot"`, line 31) and
`run_vitb_generality.py`.

### 4. Alternative entry points
- `run_notebook{1,2,3}.py` reinterpret the three notebooks as scripts; the `_colab`
  variants target GPU Colab. **Notebooks are self-contained** (duplicate `src/` logic,
  do not import `src`).
- LoRA-specific runners (`run_lora_simple_colab.py` = 6-arm grid incl. drift-weighted
  allocation; `run_lora_finetune_colab.py`; `run_lora_smoke_colab.py`).
- Special studies listed in `01_REPOSITORY_MAP.md`.

### 5. Artifact layer
`results_pilot/` holds committed CSVs/JSON + `numbers.tex`. These are the ground truth
for every claim. `make_paper_tables.py` and `make_paper_figures.py` consume them to emit
`paper/results/*.tex` and `paper/figures/*` — never edited by hand.

### 6. Presentation layer
`apps/presentation.py` is a **single-file Streamlit script** — the site's build source — with 14 sections. A
`SECTION_ORDER` list, `SECTION_LABELS`/`SECTION_TITLES` dicts, and a `RENDERERS` dispatch
table drive it. It reads presentation data from `PROJECT/OUTPUT/COLAB` path constants.
Two headless modes (`PRESENTATION_SMOKE`, `PRESENTATION_STATIC`) let
`tools/export_static_site.py` render each section to static HTML/PNG without a live server,
producing the Vercel-deployed site (`index.html`).

### 7. CI/quality layer
`tests/run_tests.py` exercises the torch-free analysis stack (`22/22`). CI installs only
`numpy scipy scikit-learn scikit-image pyyaml` and runs the tests plus an import smoke.
Security workflow runs weekly `pip-audit` on `requirements-audit.txt`.

## Key invariants

1. **Torch is optional at import time.** `src/common.py` lazily imports torch; CI proves
   the analysis stack imports without it.
2. **Config is authoritative.** Manifest SHA-256 binds results to a config.
3. **Seeded corruption protocol** (`1000 + severity`). Not the upstream deterministic
   primitive; kept for byte-comparability of committed results.
4. **Artifacts flow one way:** library → `results_pilot/` → paper/presentation. The
   presentation and paper do not write results.
5. **Self-contained notebooks** are a known duplication hazard versus `src/`.

## Why this shape (design intent, from `docs/DIVERGENCE_REPORT.md`)

The repo is a *merged* codebase: the fork contributed runners, `docs/`, `output/`, and
ExDark/v2 features; upstream contributed `src/`, `tests/`, `paper/`, CI, and `configs/`.
That explains asymmetries (e.g. duplicate stats logic in `stats_tools.py` vs `src/analysis.py`,
self-contained notebooks vs library code). See `07_TECHNICAL_DEBT_AND_OPEN_QUESTIONS.md`.

## Trust boundaries / risk surfaces

- **No network service, no auth, no PII.** The only external I/O is dataset/model download
  (`torch.hub`, CIFAR-10, ExDark) and dependency install.
- **Dynamic import:** `run_experiments.py` lazily imports `src.supplementary`.
- **Deploy surface:** static HTML only (Vercel); no server-side execution.
- **CI supply chain:** third-party GitHub Actions (`checkout@v4`, `setup-python@v5`) and
  `pip-audit`; weekly audit mitigates.
