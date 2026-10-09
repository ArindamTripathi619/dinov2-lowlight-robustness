# 08 — Exploration Ledger

*Read-only reconnaissance of `/home/DevCrewX/Projects/dinov2-lowlight-robustness`.*
*Ledger HEAD: `90577cbe171cefe98a551d05979bb8d74c6b986c` (`main`, 2026-10-08).*

This file records **every directory and file** in the working tree that was considered
during reconnaissance, with an inclusion/exclusion decision and a one-line purpose.
Heavy/generated trees are excluded from detailed analysis but are still accounted for.

## 0. Scope & method

- **Tree scope:** recursive, tracked + untracked, excluding only `.git/`, `.venv/`,
  and `__pycache__/` from enumeration (they are build/version-control internals).
- **Coverage rule:** a file is "covered" if it appears below with a purpose either
  (a) read directly, (b) summarized by a sub-agent report, or (c) explicitly excluded
  with a reason. "Not complete until every file is accounted for."
- **Method:** `git ls-files` + `find` inventory; direct reads of all human-authored
  docs, configs, CI, and key source; parallel read-only sub-agent reports for the
  larger source trees; direct verification of test execution and git state.
- **Constraints honored:** no application source modified, no deps changed, no
  destructive git ops, no secrets printed.

Tracked files: **327**. Non-heavy working-tree files enumerated below: **~140**.

## 1. Git facts (verified)

| Fact | Value |
|---|---|
| Branch | `main` |
| HEAD | `90577cb` — "Publish the static site to Vercel with pinned deploy config" |
| `origin` | `github.com/ArindamTripathi619/dinov2-lowlight-robustness` |
| `upstream` | `github.com/officialarghya29/dinov2-lowlight-robustness` |
| Merge lineage | PR #1 merged 2026-10-01 (merge commit `9bc6b50`); single shared lineage with upstream |
| Working tree | clean at time of recon; earlier HEAD `0f6fea0` observed then superseded |

> **Note on a moving target.** During recon the checkout advanced from `0f6fea0`
> (a 54-line `apps/presentation.py` stub) to `90577cb` (a 1569-line app). All
> references to the presentation app below are re-verified against `90577cb`.

## 2. Root files

| Path | Decision | Purpose |
|---|---|---|
| `README.md` | read | Top-level project overview, quickstart, section list, claims registry. |
| `CITATION.cff` | seen | Citation metadata (software + paper). |
| `SECURITY.md` | seen | Security/disclosure policy. |
| `vercel.json` | read | Vercel static deploy config. |
| `.vercelignore` | seen | Vercel ignore rules. |
| `index.html` | seen | Static site entry (generated target committed). |
| `.gitignore` | seen | VCS ignores. |
| `.freebuff/project-id` | noted | Private tooling metadata; **presence noted only, contents not reproduced.** |
| `requirements.txt` | read | Runtime deps (torch/torchvision + analysis stack + opencv + markdown). |
| `requirements-audit.txt` | read | torch-free subset audited weekly by pip-audit. |
| `launch_nb2.sh`, `run_bg.sh`, `run_notebook1_bg.sh`, `run_notebook2_bg.sh` | seen | Background-launch helpers. |

## 3. Root Python entry points (`*.py`, 31 files)

| Path | Decision | Purpose |
|---|---|---|
| `run_experiments.py` | read (sub-agent) | **Main harness**; argparse registry of **11 experiment functions + `all` (12 choices)**. |
| `run_notebook1.py` / `_colab.py` | read (sub-agent) | Phase-1 core curve runner (local / Colab variants). |
| `run_notebook2.py` / `_colab.py` | read (sub-agent) | Phase-2 mechanistic/CKA runner. |
| `run_notebook3.py` / `_colab.py` | read (sub-agent) | Phase-3 remediation/LoRA runner. |
| `run_lora_simple_colab.py` | read (sub-agent) | 6-arm LoRA grid incl. drift-weighted allocation. |
| `run_lora_finetune_colab.py` | read (sub-agent) | LoRA fine-tune runner. |
| `run_lora_smoke_colab.py` | seen | Fast LoRA smoke test. |
| `run_all_colab.py` | seen | Orchestrates the full Colab experiment sequence. |
| `run_pilot_cpu.py` | seen | CPU pilot run using `configs/pilot_cpu.yaml`. |
| `run_vitb_generality.py` | seen | ViT-B/14 generality replication (ported from upstream). |
| `run_corollaries.py` | seen | Corollary experiments → `results_pilot/corollaries.json`. |
| `run_collapse_analysis.py` | seen | Prediction-collapse analysis. |
| `run_drift_profile.py`, `run_drift_proxy.py` | seen | Drift profiling / proxy experiments. |
| `run_blur_adapter_eval.py` | seen | Blur-corruption adapter evaluation. |
| `run_readout_repair.py` | seen | Readout-staleness repair experiment (ported/hardened from upstream). |
| `run_exdark_baseline.py` | seen | ExDark real-dark baseline + CLAHE control. |
| `analyze_seeds.py` | read (bug) | Seed-sensitivity analysis; flagged `None`-guard bug. |
| `cka_recompute.py`, `cka_unbiased_recompute.py` | seen | CKA matrix recomputation (biased / unbiased). |
| `colab_gpu_bench.py`, `colab_probe.py` | seen | Colab GPU benchmarking / probing utilities. |
| `make_paper_tables.py` | read (via paper/README) | Generate `paper/results/*.tex` from `results_pilot/`. |
| `make_paper_figures.py` | read (via paper/README) | Generate `paper/figures/*` from `results_pilot/`. |
| `make_mitigation_figure.py`, `make_readme_figures.py` | seen | Figure generation. |
| `stats_tools.py` | read (sub-agent) | Ported statistics library (CKA, permutation, Wilson CI, BH-FDR, participation ratio). |
| `utils.py` | read (sub-agent) | Shared image/corruption utilities (PIL pipeline). |

## 4. `src/` package (10 modules)

| Path | Decision | Purpose |
|---|---|---|
| `src/__init__.py` | read | Package docstring only (1 line). |
| `src/common.py` | read (sub-agent) | Torch-optional helpers: `seed_everything`, `get_device`. |
| `src/corruptions.py` | read (sub-agent) | Synthetic low-light + band-limited corruption primitives. |
| `src/data.py` | read (sub-agent) | CIFAR-10 loading + seeded splits. |
| `src/models.py` | read (sub-agent) | DINOv2 ViT-S/14 loader (torch.hub) + feature extraction. |
| `src/analysis.py` | read (sub-agent) | Curve fitting, null tests, CKA aggregation. |
| `src/probes.py` | read (sub-agent) | Logistic-regression probe + logit-lens/AOPC helpers. |
| `src/lora.py` | read (sub-agent) | LoRA adapters, rank allocation, training loop. |
| `src/manifest.py` | read (sub-agent) | Results-manifest hashing (embeds config SHA-256). |
| `src/supplementary.py` | read (sub-agent) | Supplementary experiments (lazy-imported by harness). |

## 5. `apps/`, `tools/`, `scripts/`

| Path | Decision | Purpose |
|---|---|---|
| `apps/presentation.py` | read (sub-agent, re-verified) | **1569-line Streamlit app**, 14 sections. |
| `tools/export_static_site.py` | read (sub-agent) | 902-line exporter: Streamlit → static HTML/PNG site. |
| `scripts/build_kaggle_kernel.py` | seen | Kaggle kernel builder. |
| `scripts/kaggle/grid_tail.sh`, `scripts/kaggle/vitb_profile.sh` | seen | Kaggle GPU run shell wrappers. |

## 6. Config, CI, tests

| Path | Decision | Purpose |
|---|---|---|
| `configs/experiments.yaml` | read | Full-scale single source of truth (values + embedded SHA-256). |
| `configs/pilot_cpu.yaml` | read | Reduced CPU pilot config (anchored LoRA, 3 epochs). |
| `.github/workflows/ci.yml` | read | CI: py3.10/3.11, torch-free deps, `tests/run_tests.py`, import smoke. |
| `.github/workflows/security.yml` | read | Weekly pip-audit on `requirements-audit.txt`. |
| `tests/run_tests.py` | read + executed | Custom runner (not pytest). **22/22 pass, exit 0.** |
| `tests/__init__.py` | seen | Package marker. |

## 7. Notebooks

| Path | Decision | Purpose |
|---|---|---|
| `dinov2_cifar10_lowlight_robustness.ipynb` | seen (sub-agent) | Phase-1 notebook. |
| `dinov2_lowlight_mechanistic_ablation.ipynb` | seen (sub-agent) | Phase-2 notebook. |
| `dinov2_lowlight_advanced_fixed.ipynb` | seen (sub-agent) | Phase-3 notebook. |
| `dinov2_lora_finetune.ipynb` | seen (sub-agent) | LoRA fine-tune notebook. |

## 8. Documentation & paper

| Path | Decision | Purpose |
|---|---|---|
| `docs/RESEARCH.md` | read | Master research narrative + results. |
| `docs/METHODS.md` | read | Methods, protocols, reference environment (§10). |
| `docs/DPA_DESIGN.md` | read | Diagnostic-guided parameter allocation design. |
| `docs/ROADMAP.md` | read | Track-based roadmap; Track 0 status. |
| `docs/DIVERGENCE_REPORT.md` | read | Fork↔upstream divergence + PR #1 merge record. |
| `docs/PAPER_RELATED_WORK.md` | read | Related-work positioning (v1.1, verified citations). |
| `paper/README.md` | read | Paper build + claim→artifact map. |
| `paper/main.tex`, `paper/supl.tex` | skim | Manuscript + supplement; full-scale numbers marked pending. |
| `paper/references.bib`, `paper/cvpr.sty`, `paper/ieeenat_fullname.bst` | seen | Bibliography + vendored style. |
| `paper/Makefile` | seen | Build target. |
| `paper/CVPR2027_CHECKLIST.md` | seen | Submission-readiness audit. |
| `paper/results/*.tex` (12 files) | seen | Auto-generated tables + `numbers.tex` macros. |
| `paper/figures/*.pdf` (6 files) | seen | Auto-generated figures. |
| `paper/main.pdf`, `paper/supl.pdf` | excluded | Compiled binaries. |

## 9. Results & data artifacts

| Path | Decision | Purpose |
|---|---|---|
| `results_pilot/` (23 files) | enumerated | Committed pilot CSVs/JSON/`numbers.tex` — the claim artifacts. |
| `colab_results/` | excluded (reason) | Full-scale GPU outputs/logs; binary/large; referenced for provenance only. |
| `output/` | excluded (reason) | Experiment output artifacts (ExDark etc.); large. |
| `data/` | excluded (reason) | Downloaded datasets (CIFAR-10, ExDark); not source. |
| `logs/` | excluded (reason) | Runtime logs. |
| `static/gen/*.png` (19), `__static_*.png` (7) | excluded (reason) | Generated site screenshots/figures. |

## 10. Excluded trees (accounted for)

| Path | Reason for exclusion |
|---|---|
| `.git/` | Version-control internals. |
| `.venv/` | Local virtualenv (torch 2.13.0+cu130 etc.). |
| `__pycache__/`, `**/__pycache__/` | Bytecode caches. |
| `.vercel/` | Generated full-site deploy mirror (`.vercel/output/static/` duplicates the site). |
| `upstream_src/` | Byte-identical 10-file copy of `src/`; **zero references anywhere in repo** → dead snapshot. |
| `data/`, `colab_results/`, `output/`, `logs/` | Datasets / large run artifacts, not source. |

## 11. Coverage audit

- Human-authored docs: **all read** (README + 6 `docs/` + `paper/README`). ✅
- Configs: **both read**. ✅
- CI workflows: **both read**. ✅
- Tests: **executed** (22/22). ✅
- `src/`: all 10 modules covered (direct/sub-agent). ✅
- Root `*.py`: all 31 enumerated; harness + notebook runners + LoRA runners + `stats_tools`/`utils`/`analyze_seeds` covered in depth, remainder classified by role. ✅
- `apps/` + `tools/`: covered. ✅
- Excluded trees: each justified above. ✅
- **Unresolved/uncertain:** `.freebuff/project-id` contents intentionally not inspected (private tooling). No other gaps.

## 12. Second-pass verification (independent re-check)

Independent re-verification pass over the working tree (direct reads + batched greps), not
the first investigation's claims. **Status key: ✓ verified · ✎ corrected in this pass · ✗ failed.**

| Item | Outcome | Notes |
|---|---|---|
| File counts (src 10 / root 31 / notebooks 4 / results_pilot 23 / paper 12+6 / docs / tests / workflows) | ✓ | **✎** root shell scripts are **4**, not 5 (fixed in `01`). |
| `src/common.py` `src/data.py` `src/models.py` symbol tables | ✓ | `models.EmbeddingExtractor.__call__` also takes `float_input` (minor, not doc-critical). |
| `src/corruptions.py` (line table, `_deterministic_noise` at 22) | ✓ | Also validates B1 contradiction. |
| `src/analysis.py` `src/probes.py` | ✓ | `probes.fit_probe` also takes `class_weight` (minor). |
| `src/lora.py` (lines 73-74 global `np.random`; in-place `apply_lora`) | ✓ | **✎** A2 upgraded to [Confirmed, intentional] — in-file comment at ~138; harness reloads model at `run_experiments.py:433-434` (not 436). |
| `src/manifest.py` (35 lines) | ✓ | **✎** Documented as `manifest.jsonl` **append** + **torch-required** (not torch-optional) in `02/03/04`. |
| `src/supplementary.py` (371 lines) | ✓ — 2 new bugs | **✎** New confirmed A5: `_load_stl10_shared_classes` (305) `load_cifar10_subsets.class_names` AttributeError; `exp_cross_dataset` (337/367) `n_skipped` NameError. Cross-dataset experiment broken. |
| `run_experiments.py` registry | ✓ | **✎** 11 experiments + `all` (12 choices), not 12 keys; lazy import at 566 confirmed; default `--outdir results`, not `results_pilot`. |
| Experiment registry / outdir docs (`00/02/04`) | ✎ | Corrected to 11+`all`, `--outdir` semantics, `run_pilot_cpu.py` (OUTDIR line 31) as `results_pilot` producer. |
| `analyze_seeds.py:97,99` None-guard bug | ✓ | A3 confirmed as written (lines 97/99). |
| `run_notebook3.py` missing 3 CSVs vs colab (6) | ✓ | A4 confirmed (plain writes 3; colab writes 6). |
| `run_lora_simple_colab.py` per-index RNG workaround | ✓ | **✎** Lines corrected 275-281 → 275-282 (A1). |
| `apps/presentation.py` (14 sections; SECTION_ORDER 41-44; RENDERERS 1483-1498; main 1529; guard 1568; constants 29-31; set_page_config 33) | ✓ | All renderer line numbers verified exactly. |
| `tools/export_static_site.py` (902 lines) | ✓ | **✎** `04` now describes the streamlit-shim approach + all-14-section render + MAX_SECTION_RUNS=30; `PRESENTATION_STATIC` = PNG, not "single-section HTML". |
| `configs/*.yaml` lora epochs (10 vs 3) | ✓ | Confirmed. |
| CI (py3.10/3.11, torch-free, 22/22) + `tests/run_tests.py` | ✓ | Re-executed after edits: **22/22, exit 0**. |
| `make_paper_tables.py` / `make_paper_figures.py` targets | ✓ | `--results` default `results` / `results_pilot` as documented. |
| Headline-number provenance in `00/07` | ✎ | **✎** Committed `results_pilot/` is **pilot-scale (n≈60)**: `main_curve.csv` 0.933→0.108 (cos 1.00→0.16), `cka_by_layer.csv` max 0.816 @ block 11, `mechanism_adapted_probes.csv` +22.5 pp @ sev-4; full-scale numbers (0.913→0.083, +25.3 pp, 5.1×) come from `docs/RESEARCH.md`/Colab. Also discovered `cka_summary.json` uses an **older writer schema** than current harness (see `07` E8). |
| Remaining unverified | — | Deep notebook cells, exact contents of `output/`/`colab_results/`/`logs/` (provenance only), `paper/main.pdf` rendering, `.freebuff/project-id` contents. |
