# AGENTS.md

Operating instructions for AI agents (and humans) working in the
**DINOv2 Low-Light Robustness Study** repository.

Knowledge base: [`docs/project-context/`](docs/project-context/10_CONTEXT_INDEX.md)
(11 documents — start at `00`, operate from `09`).

## What this repo is

A **research codebase**: a frozen-DINOv2 low-light robustness study on CIFAR-10 with three
phases (measure → localize → remediate), an ExDark sim-to-real strand, a CVPR-style paper
(`paper/`), and a 14-section presentation deployed as a static Vercel site (`index.html`,
built from `apps/presentation.py` by `tools/export_static_site.py`). It is a merged codebase: the fork contributed runners/`docs/`, upstream
contributed `src/`/`tests/`/`paper/`/CI (see `docs/DIVERGENCE_REPORT.md`).

## Hard rules

1. **Do not modify application source unless explicitly asked.** Treat the tree as
   read-only by default. The only sanctioned default output is documentation.
2. **Never hand-edit generated artifacts:** `paper/results/*.tex`, `paper/figures/*`,
   `results_pilot/numbers.tex`, `static/gen/*`, `index.html`. Regenerate them via
   `make_paper_tables.py` / `make_paper_figures.py` / `tools/export_static_site.py`.
3. **Never break byte-comparability** of committed `results_pilot/` artifacts without an
   explicit re-baseline request. Corruption protocol and configs are load-bearing.
4. **No secrets, no destructive git, no dependency changes** unless explicitly requested.
5. **No comments in new code** unless asked.

## Verify before/after any change

```bash
python3 tests/run_tests.py                                    # expect 22/22, exit 0
python -c "import src.analysis, src.corruptions, src.probes, src.common; print('OK')"
PRESENTATION_SMOKE=1 python3 apps/presentation.py             # only if touching the site's source
git status --porcelain
```

There is **no pytest, no lint, no typecheck** in this repo. The custom runner is the gate.
CI (`.github/workflows/ci.yml`) runs the same two commands on Python 3.10/3.11 with a
torch-free dependency set.

## Environment notes

- Torch is **optional at import time** (`src/common.py:_torch`). The analysis stack must
  import without it.
- DINOv2 weights come from `torch.hub` (`facebookresearch/dinov2`); datasets auto-download.
  Full runs are GPU/Colab-oriented and heavy.
- Two configs: `configs/experiments.yaml` (full) and `configs/pilot_cpu.yaml` (CPU pilot;
  note LoRA `epochs: 3` vs 10 — this explains the historical LoRA contradiction).
- Two requirement files: `requirements.txt` (runtime) vs `requirements-audit.txt` (torch-free,
  audited weekly).

## Key entry points

- Harness: `run_experiments.py` (`--experiment` registry; lazy-imports `src.supplementary`).
- Phase runners: `run_notebook{1,2,3}[_colab].py`; LoRA: `run_lora_*_colab.py`.
- Library: `src/` (see `docs/project-context/03_MODULE_REFERENCE.md`).
- Presentation: `index.html` (deployed site); build source: `apps/presentation.py`; exporter: `tools/export_static_site.py`.

## Where to look

| Task | Read first |
|---|---|
| Understand scope/status | `docs/project-context/00_PROJECT_OVERVIEW.md` |
| Navigate the tree | `docs/project-context/01_REPOSITORY_MAP.md` |
| Touch a module | `docs/project-context/03_MODULE_REFERENCE.md` |
| Understand flows | `docs/project-context/04_EXECUTION_AND_DATA_FLOWS.md` |
| Known bugs/decisions | `docs/project-context/07_TECHNICAL_DEBT_AND_OPEN_QUESTIONS.md` |
| Why code looks odd | `docs/DIVERGENCE_REPORT.md` |
| Claim → evidence | `paper/README.md` (claim-to-artifact map) |

## Dead / generated (ignore)

`upstream_src/` (byte-identical copy of `src/`, zero references), `.vercel/`, `static/gen/`,
root `__static_*.png`, and the large `data/` `output/` `colab_results/` `logs/` trees.
