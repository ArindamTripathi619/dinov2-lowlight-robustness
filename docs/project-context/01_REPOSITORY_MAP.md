# 01 — Repository Map

*Working-tree map at HEAD `90577cb`. Excludes `.git/`, `.venv/`, `__pycache__/`.*

## Top-level layout

```
dinov2-lowlight-robustness/
├── src/                     core importable library (10 modules)
├── apps/                    presentation.py — site build source (1569 lines, 14 sections)
├── tools/                   export_static_site.py — presentation.py → static site
├── scripts/                 build_kaggle_kernel.py + scripts/kaggle/*.sh
├── configs/                 experiments.yaml (full) + pilot_cpu.yaml (CPU pilot)
├── results_pilot/           23 committed pilot CSVs/JSON/numbers.tex (claim artifacts)
├── paper/                   manuscript + auto-generated tables/figures + vendored style
├── docs/                    RESEARCH/METHODS/DPA_DESIGN/ROADMAP/DIVERGENCE/PAPER_RELATED_WORK
│   └── project-context/     ← THIS knowledge base (00–10)
├── tests/                   run_tests.py (custom runner) + __init__.py
├── .github/workflows/       ci.yml, security.yml
├── static/                  generated site images (excluded from analysis)
├── data/  output/  colab_results/  logs/   large run artifacts (excluded)
├── upstream_src/            dead byte-identical copy of src/ (excluded)
├── .vercel/                 generated deploy mirror (excluded)
├── *.py (31)                root runners / analysis / figure builders
├── *.ipynb (4)              self-contained phase notebooks
├── *.sh                     background-launch helpers
├── index.html  vercel.json  static deploy
└── README.md  CITATION.cff  SECURITY.md  requirements*.txt  .gitignore  .vercelignore
```

## Directory-by-directory

### `src/` — core library
Plain Python package; no `__main__`. Imported by root scripts via `sys.path` insertion.
Modules: `common`, `corruptions`, `data`, `models`, `analysis`, `probes`, `lora`,
`manifest`, `supplementary` (+ empty `__init__`). See `03_MODULE_REFERENCE.md`.

### `apps/` — presentation
Single file `presentation.py`. 14 named sections rendered via a `RENDERERS` dispatch
table; supports smoke (`PRESENTATION_SMOKE=1`) and static (`PRESENTATION_STATIC=1
PRESENTATION_SECTION=<name>`) modes used by the static exporter.

### `tools/` — static export
`export_static_site.py` renders each section to HTML/PNG for the Vercel site.

### `scripts/` — remote runners
Kaggle kernel builder + GPU shell wrappers.

### `configs/` — configuration (single source of truth)
`experiments.yaml` (full-scale; embedded SHA-256 into results manifest) and
`pilot_cpu.yaml` (reduced CPU pilot). See `05_CONFIGURATION_AND_INTEGRATIONS.md`.

### `results_pilot/` — committed evidence
23 files. The paper's auto-generated tables/figures and the presentation's numbers read
from here. Do not hand-edit `numbers.tex`.

### `paper/` — manuscript
`main.tex` / `supl.tex` + `references.bib`; vendored `cvpr.sty` +
`ieeenat_fullname.bst`; `Makefile`; `CVPR2027_CHECKLIST.md`; `results/*.tex` (12
auto-generated) and `figures/*` (6 auto-generated). Full-scale numbers are marked pending.

### `docs/` — prose
The research record (see `10_CONTEXT_INDEX.md` for a reading order).

### `tests/` — quality gate
`run_tests.py` is a **custom runner**, not pytest. Covers the torch-free analysis stack.

### `.github/workflows/` — CI
`ci.yml` (tests + torch-free import smoke on py3.10/3.11) and `security.yml` (weekly
pip-audit).

## Root Python scripts (31) — grouped

- **Harness / registry:** `run_experiments.py`
- **Phase runners (local/Colab):** `run_notebook1{,_colab}.py`,
  `run_notebook2{,_colab}.py`, `run_notebook3{,_colab}.py`
- **LoRA:** `run_lora_simple_colab.py`, `run_lora_finetune_colab.py`,
  `run_lora_smoke_colab.py`
- **Special studies:** `run_vitb_generality.py`, `run_corollaries.py`,
  `run_collapse_analysis.py`, `run_drift_profile.py`, `run_drift_proxy.py`,
  `run_blur_adapter_eval.py`, `run_readout_repair.py`, `run_exdark_baseline.py`,
  `run_pilot_cpu.py`, `run_all_colab.py`
- **Analysis:** `analyze_seeds.py`, `cka_recompute.py`, `cka_unbiased_recompute.py`,
  `colab_gpu_bench.py`, `colab_probe.py`
- **Figure/table builders:** `make_paper_tables.py`, `make_paper_figures.py`,
  `make_mitigation_figure.py`, `make_readme_figures.py`
- **Shared libs:** `stats_tools.py`, `utils.py`

## Dead / generated trees (do not treat as source)

| Path | Why |
|---|---|
| `upstream_src/` | byte-identical copy of `src/`; zero references |
| `.vercel/output/static/` | generated deploy mirror |
| `static/gen/`, `__static_*.png` | generated site images |
| `data/`, `output/`, `colab_results/`, `logs/` | datasets & large run outputs |

## File counts (working tree, non-heavy)

- `src/`: 10 python
- root: 31 python, 4 notebooks, 4 shell scripts
- `docs/`: 6 md (+ this `project-context/`)
- `paper/`: 2 tex + 12 generated tex + 6 figures + style/bib/Makefile
- `results_pilot/`: 23 artifacts
- `tests/`: 2, `.github/workflows/`: 2, `configs/`: 2
