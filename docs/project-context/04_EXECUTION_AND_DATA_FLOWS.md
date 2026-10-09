# 04 — Execution & Data Flows

## 1. Canonical experiment flow (config → results)

```
configs/*.yaml
   │ load_yaml_config + sha256_of_file  (src/common.py:51,43)
   ▼
run_experiments.py --experiment <key>
   │ argparse registry: main_curve | cka | frequency | mechanism |
   │ quantization | lora_ablation | mitigation | efficiency |
   │ failure_analysis | seed_sensitivity | cross_dataset | all
   ▼
src/{data,corruptions,models,probes,analysis,lora,supplementary}
   │
   ├─ write CSV/JSON ──► <--outdir>   (default `results/`; committed pilot = `results_pilot/`)
   └─ write_manifest() ──► manifest.jsonl  (append; embedded config SHA-256)
```

- `run_experiments.py:566` **lazily** imports `src.supplementary` only when a
  supplementary experiment is selected, so the base harness does not require its deps.
- Every result is bound to the exact config that produced it via the manifest hash.

## 2. Corruption pipeline (the measurement backbone)

Observation model, two stages (`src/corruptions.py`):

```
clean uint8 image I
  │  stage 1 (analog):  I' = a_s · I + η,   η ~ N(0, σ_s²)   [low_light_stage1]
  │  stage 2 (digitize):Q  = clip(round(I'), 0, 255) → uint8 [low_light]
  ▼
corrupted image, severity s ∈ {0..5}
```

- `a_s` ∈ `BRIGHTNESS_FACTORS = [1.0, 0.75, 0.55, 0.38, 0.25, 0.15]`
- `σ_s` ∈ `NOISE_STD = [0.0, 2.0, 4.0, 6.0, 9.0, 13.0]`
- Noise is **order-independent**, keyed on `blake2b(image.tobytes(), severity)`
  (`_deterministic_noise`), so identical `(image, severity)` always yields the same draw.
- **Band-limited controls** (`low_pass`, `high_pass`) isolate frequency effects; **enhancement
  controls** (`simple_gain`, `gamma_correct`, `clahe_enhance`) model remediation baselines.

## 3. Feature-extraction / probe flow (Phases 1–2)

```
CIFAR-10 subset (src/data.py, seeded split)
   │
   ├─ for each severity s: corrupt images (src/corruptions)
   ▼
EmbeddingExtractor(model, device)  ← load_dinov2("dinov2_vits14")  (torch.hub)
   │  collect_layers=False → CLS embedding (default)
   │  collect_layers=True  → per-layer features (CKA / drift)
   ▼
probes.fit_probe  →  logistic-regression readout
   ▼
analysis: bootstrap_accuracy_ci, curve_permutation_test, paired_permutation_test,
          linear_cka_unbiased, participation_ratio, benjamini_hochberg
   ▼
results_pilot/  (main_curve.csv, cka_by_layer.csv, frequency_tests.csv, …)
```

## 4. LoRA remediation flow (Phase 3)

```
frozen DINOv2 backbone
   │  apply_lora(model, rank, alpha, target)   ← installs LoRALayer into attn(/mlp)
   ▼
BackboneClassifier  (frozen backbone + linear head)   [or AnchoredBackboneClassifier]
   │  training data: LowLightAugCIFAR  (70% sev 2–4, 50% h-flip, DataLoader num_workers=2)
   ▼
train_lora / train_lora_anchored  (AdamW; lr_lora / lr_head; epochs; alpha = 2·rank)
   ▼
extract_backbone → probe eval per severity → results
```

Allocation variants compared: uniform rank, **drift-weighted** (rank 3→8 by max-normalized
CKA drop, late-heavy; 172,800 params ≈ 0.78%), late-only rank-8 (0.25%), and full
fine-tuning (22.06M) as the parity upper bound. Drift-weighted matches uniform and full-FT on
mean robustness with far fewer trainable params. See `docs/DPA_DESIGN.md` / `docs/METHODS.md`.

## 5. Results → paper flow

```
results_pilot/*.csv|json
   │ make_paper_tables.py  --results ../results_pilot --out results
   │ make_paper_figures.py --results ../results_pilot --out figures
   ▼
paper/results/*.tex  +  paper/figures/*  (AUTO-GENERATED — never hand-edit)
   │ make   (pdflatex + bibtex + 2×pdflatex)  or Overleaf
   ▼
paper/main.pdf, paper/supl.pdf
```

For full-scale numbers, `--results` points at a downloaded Colab results directory; the same
commands rebuild every table/figure.

## 6. Results → presentation → static site flow

```
results_pilot/ + output/ + colab_results/   (PROJECT/OUTPUT/COLAB path constants)
   ▼
apps/presentation.py  (site build source; 14 sections via RENDERERS)
   │  PRESENTATION_SMOKE=1                  → headless render check (all 14 sections)
   │  PRESENTATION_STATIC=1 PRESENTATION_SECTION=<name> → one section to PNG (matplotlib)
   ▼
tools/export_static_site.py  (902 lines)   → streamlit shim + import presentation.py,
                                             render all 14 sections incl. widget states
                                             (safety cap MAX_SECTION_RUNS=30)
   ▼
index.html + static/gen/*.png  ──► Vercel (vercel.json, pinned deploy)
```

## 7. Phase-runner (notebook) flow

`run_notebook{1,2,3}.py` re-implement the phase notebooks as scripts, writing CSVs into
`output/`/`results`. The `_colab.py` variants target GPU Colab; `run_all_colab.py`
orchestrates the full sequence. The `.ipynb` files themselves are **self-contained** and do
not import `src/`, so their logic is a parallel copy (see Debt).

## 8. Data sources & lifecycle

| Data | Source | Notes |
|---|---|---|
| CIFAR-10 | auto-download via torchvision | Seeded subsets (`n_test`, `n_train`); test pool disjoint from train pool. |
| DINOv2 weights | `torch.hub` (`facebookresearch/dinov2`) | `dinov2_vits14` default; ViT-B for generality run. |
| ExDark | local/`data/` | 7,363 real dark photographs for sim-to-real. |
| STL-10 | `_load_stl10_shared_classes` | Cross-dataset supplementary experiment. |

Datasets are not committed (excluded: `data/`, `colab_results/`, `output/`, `logs/`).

## 9. Determinism & reproducibility notes

- Config SHA-256 is embedded in every manifest → result provenance.
- Corruption noise is order-independent (`_deterministic_noise`).
- **Exception:** `src/lora.py:73-74` (`LowLightAugCIFAR.__getitem__`) consumes the global
  `np.random` stream inside a `num_workers=2` DataLoader → augmentation draws can correlate
  across workers. `run_lora_simple_colab.py:275-281` works around this. See Debt §.
- `docs/METHODS.md` §10 records the exact pinned reference environment that produced the
  committed results.

## 10. Full command cheat-sheet

```bash
python3 tests/run_tests.py
python -c "import src.analysis, src.corruptions, src.probes, src.common; print('OK')"
python3 run_experiments.py --experiment main_curve
python3 run_pilot_cpu.py
python3 make_paper_tables.py  --results results_pilot --out paper/results
python3 make_paper_figures.py --results results_pilot --out paper/figures
python3 tools/export_static_site.py
```
