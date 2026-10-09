# 05 — Configuration & Integrations

## 1. Configuration files

The repo has exactly two config files, both YAML, both the **single source of truth** for
their scale. The harness appends a JSON line per experiment to `manifest.jsonl` with a
SHA-256 of the config, so a
result is only valid for the exact config that produced it.

### `configs/experiments.yaml` — full-scale (65 lines)
| Section | Keys |
|---|---|
| `data` | `dataset: cifar10`, `n_test: 1000`, `n_train: 5000`, `split_seed: 42`, `test_fraction: 0.30` |
| `corruption` | `brightness_factors: [1.0,0.75,0.55,0.38,0.25,0.15]`, `noise_std: [0,2,4,6,9,13]`, `lowpass_cutoffs`, `highpass_cutoffs` |
| `model` | `backbone: dinov2_vits14`, `input_size: 224`, ImageNet mean/std |
| `probe` | `kind: logistic_regression`, `C: 1.0`, `max_iter: 2000` |
| `stats` | `n_bootstrap: 2000`, `ci: 95`, `n_permutations: 5000`, `alpha: 0.05`, `bonferroni: true` |
| `cka` | `unbiased: true` |
| `mechanism` | `logit_lens_topk: 16`, `aopc_fractions: [0.1…0.5]` |
| `lora` | `ranks: [4,8,16]`, `targets: [attn, attn_mlp]`, `alpha_multiplier: 2`, `epochs: 10`, `batch_size: 64`, `lr_lora: 5e-5`, `lr_head: 1e-3`, `weight_decay: 0.01`, `seed: 42` |
| `mitigation` | `gamma_base: 0.35`, `clahe_clip_limit: 0.03` |
| `seed` | `42` |

### `configs/pilot_cpu.yaml` — reduced CPU pilot (62 lines)
Same schema, reduced scale: `n_test: 400`, `n_train: 800`, `n_bootstrap: 1000`,
`n_permutations: 1000`, `lora.epochs: 3`, `batch_size: 32`, plus an extra
`lora.anchored: {enabled: true, lambda_anchor: 1.0}` block. Header states results are
PILOT-SCALE (test fold n≈60).

> **Note (LoRA protocol resolution):** the epoch discrepancy (`pilot_cpu.yaml` = 3,
> `experiments.yaml` = 10) explains the historical "LoRA helps vs hurts" contradiction — it
> is a protocol-scale artifact, not a scientific disagreement. Upstream's negative claim has
> no committed artifact; the positive one is backed by
> `colab_results/lora_run_fixed/lora_full_run.log`.

## 2. Dependency files

| File | Purpose | Contents |
|---|---|---|
| `requirements.txt` | Runtime | `torch>=2.0`, `torchvision>=0.15`, `numpy`, `scikit-learn`, `scipy`, `scikit-image`, `matplotlib`, `pillow`, `opencv-python-headless>=4.8`, `markdown>=3.4`; commented exact pins + CPU torch install hint |
| `requirements-audit.txt` | torch-free audit subset (used by security CI) | `numpy>=1.24`, `scipy>=1.10`, `scikit-learn>=1.3`, `scikit-image>=0.21`, `matplotlib>=3.7`, `pyyaml>=6.0` |

- `pyyaml` appears only in the audit file (config loading needs it); `opencv` is for the
  ExDark CLAHE control; `markdown` is for static-site export.
- `docs/METHODS.md` §10 records the exact pinned reference environment that produced the
  committed results (torch 2.13.0, torchvision 0.28.0, scikit-learn 1.9.0, numpy 2.5.2, …).

## 3. External integrations

| Integration | Where | Purpose / notes |
|---|---|---|
| **PyTorch Hub** (`facebookresearch/dinov2`) | `src/models.py:19`, `utils.py:154` | Downloads DINOv2 weights (`dinov2_vits14`, ViT-B for generality). Network I/O. |
| **torchvision CIFAR-10** | `src/data.py`, `utils.py:505` | Auto-download to `data/`. |
| **OpenCV** | `src/corruptions.py:139` (CLAHE) | Enhancement control in ExDark study. |
| **Google Colab (GPU)** | `run_*_colab.py`, `run_all_colab.py`, `colab_*.py` | Full-scale GPU runs; outputs land in `colab_results/`. |
| **Kaggle** | `scripts/build_kaggle_kernel.py`, `scripts/kaggle/*.sh` | Alternative GPU execution (grid tail, ViT-B profile). |
| **Vercel** | `vercel.json`, `.vercelignore`, `.vercel/`, `index.html` | Hosts the exported static presentation site. |
| **GitHub Actions** | `.github/workflows/ci.yml`, `security.yml` | CI + weekly security audit. |
| **TeX Live / Overleaf** | `paper/Makefile`, `paper/*.sty/.bst` | Paper build (`pdfLaTeX`). |

## 4. CI configuration

### `.github/workflows/ci.yml`
- Triggers: push/PR to `main`, manual.
- Matrix: Python `3.10`, `3.11` (ubuntu-latest).
- Installs **only** `numpy scipy scikit-learn scikit-image pyyaml` (deliberately no torch).
- Runs `python tests/run_tests.py`, then a torch-free import smoke:
  `import src.analysis, src.corruptions, src.probes, src.common`.

### `.github/workflows/security.yml`
- Triggers: weekly cron (Mon 06:00 UTC), push to `main` touching `requirements.txt` or the
  workflow, manual.
- Installs `pip-audit`, runs `pip-audit -r requirements-audit.txt --strict`.
- Rationale in-file: torch/CUDA wheels excluded to avoid multi-GB downloads; torch is
  exercised on Colab instead.

## 5. Deploy configuration

- `vercel.json` — pinned static deploy config (HEAD commit `90577cb`).
- `.vercelignore` — excludes non-site paths from the deploy bundle.
- `.vercel/output/static/` — generated mirror (excluded from analysis).
- Static output committed: `index.html` + `static/gen/*.png`.

## 6. Environment variables / modes

| Variable | Effect |
|---|---|
| `PRESENTATION_SMOKE=1` | Run a headless render check of `apps/presentation.py`. |
| `PRESENTATION_STATIC=1` | Switch the app to static export mode. |
| `PRESENTATION_SECTION=<name>` | Select a single section for static export. |

No secrets are read from the environment by application code. The only credential-shaped
artifact is `.freebuff/project-id` (private tooling metadata; contents not inspected).

## 7. Configuration gaps / observations

- No `.env`, no secrets config, no runtime feature flags beyond the presentation modes.
- Config schema is **not validated** by a schema/linter; a typo in a YAML key would surface
  only at experiment runtime.
- Two divergent LoRA epoch defaults (3 vs 10) are intentional (pilot vs full) but easy to
  confuse — documented here to prevent misreading.
