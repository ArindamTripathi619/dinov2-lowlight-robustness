# 03 — Module Reference

*Signatures verified against HEAD `90577cb`. Line numbers are from that commit.*

## `src/` package

### `src/common.py` (118 lines) — torch-optional utilities
| Symbol | Line | Notes |
|---|---|---|
| `_torch()` | 17 | Lazy torch import; keeps module importable without torch. |
| `seed_everything(seed)` | 26 | Seeds python/numpy/torch. |
| `get_device()` | 36 | Returns cuda/mps/cpu string. |
| `sha256_of_file(path)` | 43 | Used for config fingerprinting. |
| `load_yaml_config(path)` | 51 | YAML → dict. |
| `ensure_dir(path)` | 58 | mkdir -p. |
| `save_json / save_csv / read_csv_rows` | 63/69/83 | I/O helpers. |
| `matplotlib_agg()` | 90 | Headless backend. |
| `format_perm_pvalue(p)` | 98 | p-value formatting. |
| `wilson_ci(k, n, z)` | 102 | Wilson CI helper. |

### `src/corruptions.py` (147 lines) — corruption physics
Two-stage model: stage 1 analog `I' = a·I + η`; stage 2 uint8 digitization.
| Symbol | Line | Notes |
|---|---|---|
| `BRIGHTNESS_FACTORS`, `NOISE_STD` | 18-19 | Severity ladders (6 values). |
| `_deterministic_noise(image, severity, sigma)` | 22 | Order-independent noise keyed on blake2b(image bytes, severity). |
| `low_light_stage1(...)` | 39 | Analog stage only, float32, no quantization. |
| `low_light(...)` | 58 | Full two-stage → uint8 (standard ladder). |
| `quantization_gap_truth(...)` | 67 | RMS stage1↔stage2 (digitization cost). |
| `_fft_filter`, `low_pass`, `high_pass` | 91/107/112 | Frequency-band controls. |
| `simple_gain`, `gamma_correct`, `clahe_enhance` | 121/127/139 | Enhancement controls. |

> `_deterministic_noise` is present in the merged tree (see Open Questions — the
> divergence report's §8.2/§9 states the upstream deterministic primitive was *not* ported).

### `src/data.py` (48 lines)
| `load_cifar10_subsets(n_test=1000, n_train=5000, seed=42, …)` | 16 | Seeded CIFAR-10 test/train subsets. |

### `src/models.py` (97 lines) — DINOv2
| Symbol | Line | Notes |
|---|---|---|
| `load_dinov2(model_name="dinov2_vits14", device="cuda")` | 19 | `torch.hub.load("facebookresearch/dinov2", …)`. |
| `build_uint8_preprocess(input_size)` | 25 | uint8 → normalized tensor pipeline. |
| `float_to_tensor(img_f32, input_size)` | 34 | float image → tensor. |
| `EmbeddingExtractor` | 41 | Hook-based per-layer feature extraction. |
| — `__init__ / _install_hooks / _remove_hooks / __call__` | 48/55/63/70 | Layer collection optional. |

### `src/analysis.py` (202 lines) — statistics
| Symbol | Line | Notes |
|---|---|---|
| `linear_cka_unbiased(X, Y)` | 25 | Unbiased linear CKA (Kornblith 2019 App. B). |
| `participation_ratio(X)` | 51 | Spectral concentration of drift covariance. |
| `curve_permutation_test(...)` | 83 | Curve-shape null test. |
| `paired_permutation_test(a, b, …)` | 131 | Paired significance. |
| `multiset_distance_test(...)` | 153 | Curve multiset comparison. |
| `bootstrap_accuracy_ci(...)` | 180 | Bootstrap CI. |
| `benjamini_hochberg(pvals)` | 191 | BH-FDR correction. |

### `src/probes.py` (70 lines)
| Symbol | Line | Notes |
|---|---|---|
| `fit_probe(X, y, C=1.0, max_iter=2000, …)` | 13 | Logistic-regression probe. |
| `fit_probe_on_severity(...)` | 23 | Probe trained at a given severity. |
| `NTrainableProbe(n_per_class, seed)` | 31 | Class-balanced probe. |
| `stratified_split(labels, test_fraction, seed)` | 62 | Split helper. |

### `src/lora.py` (247 lines) — LoRA adaptation
| Symbol | Line | Notes |
|---|---|---|
| `LoRALayer(original_layer, r, alpha)` | 30 | Low-rank wrapper (scaled by alpha/r). |
| `apply_lora(model, rank, alpha, target="attn")` | 45 | **In-place** install into `block.attn.{qkv,proj}` (+ MLP if `attn_mlp`); freezes base, unfreezes `lora_A/B`. |
| `LowLightAugCIFAR(Dataset)` | 62 | 70% degraded sev 2–4, 50% h-flip. |
| `__getitem__` | 71 | **Uses global `np.random`** (L73-74) — see Debt. |
| `BackboneClassifier(backbone, num_classes)` | 80 | Frozen-backbone + linear head. |
| `AnchoredBackboneClassifier(...)` | 93 | Adds representation-anchoring loss. |
| `info_nce_loss(z1, z2, temperature)` | 119 | Contrastive anchor loss. |
| `train_lora(pretrained_model, …, cfg, …)` | 129 | Standard LoRA training loop. |
| `train_lora_anchored(...)` | 177 | Anchored variant. |
| `extract_backbone(clf, …, batch_size=64)` | 236 | Post-training feature extraction. |

### `src/manifest.py` (35 lines)
| `write_manifest(results_dir, config_path, experiment, extra)` | 18 | **Appends** one JSON line to `results_dir/manifest.jsonl`, embedding the config SHA-256 (provenance). **Top-level `import torch` → this module is *not* torch-optional** (the rest of `src/` is). |

### `src/supplementary.py` (371 lines) — secondary experiments
Lazy-imported by the harness (`run_experiments.py:566`).
| Symbol | Line |
|---|---|
| `_count_params`, `_measure_flops`, `_measure_latency`, `_severities` | 33/37/54/76 |
| `exp_efficiency(cfg, device, outdir)` | 84 |
| `exp_failure_analysis(cfg, device, outdir)` | 163 |
| `exp_seed_sensitivity(cfg, device, outdir)` | 257 |
| `_load_stl10_shared_classes(n_test, seed)` | 305 |
| `exp_cross_dataset(cfg, device, outdir)` | 337 |

> **`src/supplementary.py` cross-dataset path is broken (see Debt A5):**
> `_load_stl10_shared_classes` (305) references `load_cifar10_subsets.class_names` — a
> function attribute that does not exist (class names live in `CIFAR10_CLASS_NAMES` in
> `src/data.py`) → `AttributeError`; and `exp_cross_dataset` (367) uses undefined
> `n_skipped` (the helper returns `n_dropped`) → `NameError`. `experiment cross_dataset`
> cannot complete.

## Root shared libraries

### `stats_tools.py` (223 lines)
Torch-free port of the statistics stack. **Duplicates much of `src/analysis.py`** plus
additions:
| Symbol | Line |
|---|---|
| `linear_cka_unbiased` | 28 |
| `participation_ratio` | 55 |
| `curve_permutation_test` | 84 |
| `paired_permutation_test` | 125 |
| `bootstrap_accuracy_ci_correct` | 148 |
| `wilson_ci` | 168 |
| `benjamini_hochberg` | 181 |
| `readout_collapse_metrics` | 196 | top1_share / entropy_ratio / dominant class (F6 "99% frog"). |

### `utils.py` (523 lines)
Legacy/shared image + plotting utilities, **duplicating** parts of `src/`:
- Corruptions: `low_light` (30), `blur` (50), `jpeg` (63), `contrast` (82),
  `get_corruption` (107), `_fft_filter` (114), `low_pass` (132), `high_pass` (138).
- Model: `load_dinov2` (154), `get_embeddings` (169), `get_embeddings_with_hooks` (192),
  `register_layer_hooks` (224).
- Probe/eval: `train_linear_probe` (241), `evaluate_across_severities` (256),
  `bootstrap_accuracy_ci` (286).
- CKA: `linear_cka` (317, biased), `compute_cka_matrix` (327).
- Plot/IO: `setup_output_dir` (346), `save_fig` (352), `plot_*` (359-463),
  `save_results_csv` (489), `load_cifar10_subset` (505).

## Entry-point map (high-value scripts)

| Script | Role | Key symbols/lines |
|---|---|---|
| `run_experiments.py` | Main harness; argparse registry | `EXPERIMENTS` registry; lazy `src.supplementary` at 566 |
| `run_notebook1{,_colab}.py` | Phase 1 curves | imports `src.data/corruptions/models` |
| `run_notebook2{,_colab}.py` | Phase 2 CKA/mechanism | — |
| `run_notebook3{,_colab}.py` | Phase 3 remediation/LoRA | plain runner omits 3 CSVs (Debt) |
| `run_lora_simple_colab.py` | 6-arm LoRA grid | aug seed fix at 275-281 |
| `run_readout_repair.py` | Readout-staleness repair | hardened paired-perm + Wilson CI |
| `run_exdark_baseline.py` | Real-dark baseline + CLAHE | uses opencv `clahe_enhance` |
| `run_vitb_generality.py` | ViT-B/14 replication | — |
| `analyze_seeds.py` | Seed-sensitivity analysis | `None`-guard bug at 97,99 |
| `make_paper_tables.py` / `make_paper_figures.py` | `results_pilot/` → `paper/` | — |
| `analyze_seeds.py`, `cka_recompute.py`, `cka_unbiased_recompute.py` | post-hoc analysis | — |
| `tools/export_static_site.py` (902) | `presentation.py` → static site | drives `PRESENTATION_STATIC` |

## `apps/presentation.py` (1569 lines) — structure

| Element | Line(s) |
|---|---|
| `PROJECT/OUTPUT/COLAB` path constants | 29-31 |
| `st.set_page_config(...)` | 33-34 |
| `SECTION_ORDER` (14 keys) | 41-44 |
| `SECTION_LABELS` | 46-61 |
| `SECTION_TITLES` | 63-78 |
| Section renderers | `_render_home` 399, `_render_methods` 504, `_render_collapse` 793, `_render_localize` 826, `_render_mechanism` 869, `_render_proxy` 915, `_render_fix` 970, `_render_grid` 1022, `_render_families` 1081, `_render_readout` 1172, `_render_exdark` 1228, `_render_vitb` 1294, `_render_bugs` 1382, `_render_gallery` 1453 |
| `RENDERERS` dispatch | 1483-1498 |
| `main()` | 1529 |
| Entry guard | 1568 |

Modes: `PRESENTATION_SMOKE=1` (render check), `PRESENTATION_STATIC=1
PRESENTATION_SECTION=<name>` (single-section HTML for export).
