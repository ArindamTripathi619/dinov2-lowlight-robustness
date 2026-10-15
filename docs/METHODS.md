# Methods Appendix

*Exact parameters, formulas, data protocol, and environment for every experiment in this study. All values verified against source (`utils.py`, `run_notebook1.py`, `run_notebook2.py`, `run_lora_simple_colab.py`, `cka_recompute.py`, `run_readout_repair.py`, `cka_unbiased_recompute.py`, `run_blur_adapter_eval.py`, `run_drift_proxy.py`, `run_drift_profile.py`, `stats_tools.py`, `scripts/build_kaggle_kernel.py`).*

---

## 1. Backbone & Preprocessing

| Item | Value |
|------|-------|
| Model | DINOv2 ViT-S/14 via `torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")` |
| Parameters | 22,277,760 (all frozen except Phase 3 LoRA + head) |
| Input size | 224 × 224 (CIFAR-10 32×32 upsampled) |
| Patch size | 14 → 256 tokens + CLS |
| Embedding used | CLS token, pooled (`model(tensors)` output), dim 384 |
| Normalization | ImageNet: mean (0.485, 0.456, 0.406), std (0.229, 0.224, 0.225) |
| Transform order | `ToTensor → Resize((224,224), antialias) → Normalize` |
| Device | CUDA if available, else CPU; `model.eval()` + `torch.no_grad()` throughout |

---

## 2. Data Protocol

| Experiment | Source | n | Sampling |
|------------|--------|---|----------|
| Phase 1 (notebook 1) | CIFAR-10 **test** split | 1,000 | `default_rng(42).choice(10000, 1000)`, replace=False |
| Phase 2 (notebook 2) | CIFAR-10 test split | 500 | fresh `default_rng(42).choice(10000, 500)` — independent draw; overlaps Phase 1 by ~5% (chance) |
| Phase 3 training pool | CIFAR-10 **train** split | 5,000 | `default_rng(42).choice(50000, 5000)` (same RNG instance as the test draw, consumed sequentially) |
| Phase 3 test set | CIFAR-10 test split | 1,000 | `default_rng(42).choice(10000, 1000)` — identical index set to Phase 1's images |

All probe splits stratified 70/30 (`train_test_split(..., test_size=0.3, random_state=42, stratify=labels)`); with equal `random_state` and label arrays, the original-model and LoRA probes receive **identical** test indices, so the Phase 3 comparison is same-images.

**Notes on cross-phase comparability:** Phase 3's test set equals Phase 1's image set, which is why the original-model baselines match (0.9133 at severity 0 in both). Phase 2's 500-image set is a different random sample; its severity-0 accuracy (0.920) differing slightly from Phase 1's (0.913) is sampling noise, not inconsistency. Phase 3 trains on the **train** split, so no test image is seen during LoRA training.

---

## 3. Corruption Functions (exact parameter tables)

### 3.1 `low_light(image, severity)`

```python
img = image * brightness_factors[severity] + N(0, noise_std[severity])
img = clip(img, 0, 255)
```

| Severity | Brightness factor | Gaussian noise σ (0–255 scale) |
|----------|-------------------|-------------------------------|
| 0 | 1.00 | 0 |
| 1 | 0.75 | 2 |
| 2 | 0.55 | 4 |
| 3 | 0.38 | 6 |
| 4 | 0.25 | 9 |
| 5 | 0.15 | 13 |

Noise is i.i.d. per pixel per channel, drawn **without a fixed seed** (severity-level stochasticity; accuracy aggregates over ≥500 images make this negligible).

### 3.2 `blur(image, severity)`

```python
img.filter(ImageFilter.GaussianBlur(radius=radii[severity]))
```

| Severity | 0 | 1 | 2 | 3 | 4 | 5 |
|----------|---|---|---|---|---|---|
| Gaussian radius (px) | 0.0 (identity) | 0.5 | 1.0 | 2.0 | 3.5 | 5.5 |

Deterministic (`rng` accepted for signature parity, unused).

### 3.3 `jpeg(image, severity)`

PIL JPEG re-encode at stepwise decreasing quality. **Severity 0 returns the image untouched** — JPEG is lossy even at q100 (chroma subsampling), and severity 0 must be the pristine image the linear probe trains on. Deterministic.

| Severity | 0 | 1 | 2 | 3 | 4 | 5 |
|----------|---|---|---|---|---|---|
| JPEG quality | pristine | 60 | 40 | 25 | 15 | 8 |

### 3.4 `contrast(image, severity)`

```python
img = factors[s] * img + (1 - factors[s]) * img.mean(axis=2, keepdims=True)
```

Blend toward the image's **scalar grayscale mean** (not Hendrycks & Dietterich's per-channel means — documented here so nobody cites the exact HD formula for our numbers). Deterministic.

| Severity | 0 | 1 | 2 | 3 | 4 | 5 |
|----------|---|---|---|---|---|---|
| Contrast factor | 1.0 (identity) | 0.8 | 0.6 | 0.4 | 0.25 | 0.15 |

### 3.5 Frequency filters (`low_pass` / `high_pass`)

Circular FFT mask applied per channel; cutoff expressed as a fraction of the max image-domain radius from the spectrum center.

| Severity | `low_pass` cutoff (keep ≤ r) | `high_pass` cutoff (keep > r) |
|----------|------------------------------|-------------------------------|
| 1 | 0.90 | 0.05 |
| 2 | 0.70 | 0.10 |
| 3 | 0.50 | 0.15 |
| 4 | 0.35 | 0.20 |
| 5 | 0.20 | 0.30 |

---

## 4. Linear CKA (corrected formula)

For activation matrices X, Y (n_samples × d, row-centered):

```
HSIC(X, Y) = ‖Xᵀ Y‖²_F
CKA(X, Y)  = HSIC(X, Y) / sqrt( HSIC(X, X) · HSIC(Y, Y) )
```

Implementation (`utils.py::linear_cka`):

```python
X = X - X.mean(0, keepdims=True)
Y = Y - Y.mean(0, keepdims=True)
hsic  = np.linalg.norm(X.T @ Y, "fro") ** 2
var1  = np.linalg.norm(X.T @ X, "fro") ** 2
var2  = np.linalg.norm(Y.T @ Y, "fro") ** 2
return hsic / (np.sqrt(var1 * var2) + 1e-8)
```

- Numerical epsilon 1e-8; theoretical range [0, 1], CKA(X, X) = 1.
- Applied per transformer block (0–11), comparing **CLS activations of the same images** under clean vs. degraded input.
- **History:** the original implementation omitted the `sqrt`, squaring all values (severity-5 values up to ≈ 79.9). Ordering across layers was preserved (squaring is monotonic), so the late-layer conclusion held; all reported values in this repo use the corrected formula.

**Unbiased estimator robustness check (`cka_unbiased_recompute.py`).** The biased (HSIC-ratio) estimator is known to inflate similarity for finite n. We recomputed the full layer × severity matrix with the unbiased linear CKA (Kornblith et al. 2019, App. B, via `stats_tools.cka_u`; same images, same protocol) → `output/notebook2/cka_matrix_unbiased.csv` (via `stats_tools.linear_cka_unbiased`)

- Biased allocation: `{0:6, 1:6, 2:4, 3:5, 4:5, 5:5, 6:6, 7:7, 8:7, 9:8, 10:8, 11:8}` (115,200 params)
- Unbiased allocation: `{0:5, 1:6, 2:4, 3:4, 4:5, 5:5, 6:6, 7:7, 8:8, 9:8, 10:8, 11:8}` (113,664 params)
- **Allocations differ** at blocks 0 (6→5), 3 (5→4), 8 (7→8), but the late-heavy structure is **estimator-robust**: blocks 9–11 hold maximum rank 8 under both estimators, and the unbiased estimator *strengthens* late concentration (block 10 sev-5 drop 0.815 → 0.859). The published drift-arm results remain valid; a re-run with the unbiased profile would change 3 of 12 block ranks (−1,536 params). Noted as estimator-sensitivity for the merged paper; no result depends on the biased values.

**Cross-device determinism certificate.** The Phase-2 CKA matrix computed locally (CPU) and inside the Kaggle T4 kernel agree to the 6th decimal place (~1e-6 float noise) — the embedding+CKA pipeline is deterministic across CPU/CUDA implementations, which certifies that GPU- and CPU-produced artifacts in this repo are directly comparable.

---

## 5. Bootstrap Confidence Intervals

| Parameter | Value |
|-----------|-------|
| Resamples | 1,000 per severity |
| Interval | 95th percentile method (`np.percentile` at 2.5 / 97.5) |
| Unit | per-image correctness vector resampled with replacement |
| RNG | `np.random.default_rng(severity)` — **severity-indexed seed** (bug fix: original code re-seeded with 0 inside the loop, making resamples identical across severities) |

---

## 6. Linear Probe

| Parameter | Value |
|-----------|-------|
| Classifier | `sklearn.linear_model.LogisticRegression` |
| max_iter | 2,000 |
| C (inverse reg.) | 1.0 |
| Train data | Clean (severity-0) embeddings **only** |
| Split | 70/30, stratified, seed 42 |

---

## 7. LoRA Configuration (Phase 3)

### 7.1 Adapter placement & size

| Item | Value |
|------|-------|
| Targets | `attn.qkv` and `attn.proj` in every block (24 modules: 2 × 12 blocks) |
| Rank r | 8 |
| Alpha α | 16 (scaling = α/r = 2.0) |
| Trainable params | 221,184 / 22,277,760 = **0.99%** |
| Backbone | frozen; the classifier head (384 → 10) also trains |

### 7.2 Training

| Item | Value |
|------|-------|
| Steps | 10 epochs × ⌈5000/64⌉ = 780 steps/epoch ≈ 7,800 total |
| Optimizer | AdamW |
| Param groups | adapters lr 5e-5; head lr 1e-3; weight_decay 0.01 (groups deduplicated by `id()` — see §9) |
| Loss | CrossEntropy |
| Batch | 64, shuffled, num_workers 2 |
| Augmentation (train only) | p = 0.7 → `low_light` at severity ~ Uniform{2,3,4}; p = 0.5 → horizontal flip |
| Runtime | ~10 min on Colab free-tier T4 |

### 7.3 Evaluation

Same protocol as §6: probe on clean LoRA embeddings (70/30, seed 42), evaluated at all 6 severities on the 1,000-image test set; the original model is evaluated identically for the comparison.

**Matched-noise evaluation (fixed after the first full run).** The noise in `low_light` is stochastic; in the *first* run it was drawn independently for the LoRA pass and the original-model pass, so each was evaluated on slightly different corrupted images (visible as run-to-run jitter in the original column, e.g. sev-5 baseline 0.080 vs 0.073, while the noise-free sev-0 matched exactly at 0.9133). The script of record now precomputes the corrupted test arrays **once** per severity (seeded `default_rng(1000 + severity)`) and feeds the identical arrays to both models — the comparison is image-matched. For the same reason, eval-time noise is deterministic across reruns.

**Artifact of record.** The results quoted in the docs come from the post-augmentation-fix, matched-noise-script run: `colab_results/lora_run_fixed/` (log + both plots). The earlier `colab_results/lora_run/` artifacts predate both fixes and are retained for comparison only.

### 7.4 Phase 3 v2 grid (nine arms)

All arms share the §7.2 diet (5,000 images, 70% corrupted / 30% clean, 10 epochs, AdamW) and the §7.3 matched-noise evaluation. The Phase 0 refactor made each arm a flag combination of one script; artifacts of record in `colab_results/sessionD/<arm>/` (log, plots, `lora_adapters.pt`).

| Arm | Invocation | Trainable params | Notes |
|-----|------------|------------------|-------|
| `seed{42,43,44}_all` | `--seed {42,43,44} --layers all` | 221,184 (0.99%) | uniform rank-8 error bars |
| `late` | `--seed 42 --layers late` | 55,296 (0.25%) | rank-8 on blocks 9–11 only |
| `drift` | `--seed {42,43,44} --layers drift` | 172,800 (0.78%) | per-block ranks ∝ CKA drop (below); 3-seed replicated |
| `fullft` | `--seed {42,43,44} --layers fullft` | 22,056,576 (100%) | backbone lr 1e-5, head lr 1e-3; optimizer param groups disjoint; 3-seed replicated |

**Drift-weighted rank allocation.** Ranks are proportional to the sev-5 CKA drop profile of §4's artifact (`output/notebook2/cka_matrix.csv`), embedded in the script as `DEFAULT_DRIFT = [0.58, 0.61, 0.44, 0.47, 0.53, 0.53, 0.58, 0.68, 0.76, 0.77, 0.81, 0.78]`. Allocation: block-wise rank = `round(r × drop_i / max(drop))`; blocks rounding below 1 would be dropped entirely (none at r = 8 with this profile), giving `{0:6, 1:6, 2:4, 3:5, 4:5, 5:5, 6:6, 7:7, 8:7, 9:8, 10:8, 11:8}` (block 8: 8 × 0.763/0.815 = 7.49 → 7) — all 24 modules adapted, 172,800 trainable. `--drift-csv` overrides the embedded profile with a recomputed CKA matrix.

**Cross-seed comparability caveat.** Each run draws its own 1,000-image test set seeded by `--seed` (§5 registry), so rows with different seeds differ in both init *and* test draw. Within any row, Original vs adapted columns are image- and noise-matched.

### 7.6 Corruption-family grid (blur / JPEG / contrast; sessions E + F)

Phase-3 arms for the new corruption families, all through the same §7.2/§7.3 protocol via `run_lora_simple_colab.py --corruption {blur,jpeg,contrast}`:

| Arm | Where | Before → After (mean over sev 0–5) | Artifact of record |
|-----|-------|-------------------------------------|--------------------|
| jpeg, drift-weighted ranks | Kaggle T4 | 0.586 → **0.779** (+19.3 pts; worst-case sev-5 +25.7) | `colab_results/kaggle_sessionF/jpeg_drift/` |
| contrast, uniform r8 | Kaggle T4 | 0.885 → 0.960 (+7.5) | `colab_results/kaggle_sessionF/contrast_uniform/` |
| contrast, drift-weighted ranks | Kaggle T4 | 0.885 → 0.954 (+6.9) | `colab_results/kaggle_sessionF/contrast_drift/` |
| blur, uniform r8 | Colab T4 (train) + local CPU (re-eval) | 0.526 → **0.698** (+17.2; sev-5 0.143 → 0.270) | train `colab_results/sessionE/blur_uniform/`; eval `output/blur_reeval/blur_uniform/` |
| blur, drift-weighted ranks | Colab T4 (train) + local CPU (re-eval) | 0.526 → **0.703** (+17.7; sev-5 0.143 → 0.303) | train `colab_results/sessionE/blur_drift/`; eval `output/blur_reeval/blur_drift/` |
| jpeg, uniform r8 | Colab T4 | trained; eval complete, numbers folded into the family analysis | `colab_results/sessionE/jpeg_uniform/` |

The sessionE blur arms trained to convergence on Colab T4 but their eval logs were truncated on salvage; both were re-evaluated from the retained `lora_adapters.pt` checkpoints with `run_blur_adapter_eval.py`, which rebuilds the LoRA backbone from the checkpoint's `block_ranks` config (the rebuild path proven in §7.9's transfer pass) and replays the §7.3 protocol exactly — matched-noise eval corruption (`1000 + severity`), probe on severity-0 embeddings, 70/30 split (`random_state=42`), original-model pass on the identical corrupted arrays. Comparability certificate: the re-evaluated original-model column reproduces the Phase-1 curve **exactly** (0.913/0.873/0.620/0.387/0.220/0.143). The blur drift-vs-uniform pair (+17.7 vs +17.2, sev-5 0.303 vs 0.270) repeats the low-light pattern: parity on mean within seed noise, drift arm matching or edging worst-case — on a profile derived from blur's own CKA matrix (block ranks `{0:4 … 8:8, 9:8, 10:8, 11:7}`). The drift-vs-uniform contrast comparison (+6.9 vs +7.5, overlapping given the ±1.5-pt seed noise of §6.1) is itself a result: the contrast CKA drift profile is ~flat (max drop 0.30, *falling* in late blocks), so drift-weighting has no misallocated budget to fix — and gracefully loses nothing.

### 7.7 Readout-staleness decomposition (`run_readout_repair.py`)

Decomposes the Phase-1 collapse into *feature drift* vs *readout staleness* (the severity-adapted-probe control from upstream's H4, ported and hardened): Arm A = the §6 probe trained on clean embeddings, fixed across severities; Arm B = a probe retrained per severity on degraded versions of the **same train fold** (strict same-fold control — no test-fold reuse). Paired permutation test (5,000 permutations, seed 42) on per-image correctness; Wilson 95% CIs; `readout_collapse_metrics` (top1_share / entropy_ratio / dominant class) on Arm A predictions. n = 1,000 test images (seed 42), low_light, DINOv2 ViT-S/14, local CPU.

Artifact: `output/readout_repair/` (results.csv, summary.json — the runner's default output directory).

| Severity | Arm A fixed | Arm B adapted | Δ | p (paired perm.) |
|----------|-------------|---------------|-----|------|
| 0 | 0.913 | 0.913 | +0.0 | — |
| 1 | 0.900 | 0.890 | −1.0 | 0.18 |
| 2 | 0.860 | 0.840 | −2.0 | 0.15 |
| 3 | 0.633 | 0.720 | **+8.7** | 0.0032 |
| 4 | 0.243 | 0.497 | **+25.3** | 0.0002 |
| 5 | 0.077 | 0.377 | **+30.0** | 0.0002 |

Readout collapse signature (Arm A): top1_share 0.14 (sev 2, dominant *deer*) → 0.73 (sev 5, dominant *frog*); entropy_ratio 1.00 → 0.26. **Interpretation:** at low severity, readout repair is flat-to-negative (a probe-level echo of upstream's "grace regime"); from severity 3 the readout is genuinely stale and adaptation recovers a third to half of the gap — but the majority of the loss stays in the features (sev-5 ceiling 0.377 vs 0.913 clean), confirming the LoRA result's premise that *features*, not the linear head, are the bottleneck.

### 7.8 ViT-B drift-profile sweep (`run_drift_profile.py`; kernel `vitb-drift-profile` v1)

Third-architecture replication of the drift profile (RESEARCH §5.1) plus the Track-1 proxy agreement re-check, run as one severity×corruption sweep over a single clean pass. Everything is produced by the Track-1 harness (`run_drift_proxy.py`); `run_drift_profile.py` loops it over severities and emits per-key CSVs.

| Item | Value |
|------|-------|
| Model | DINOv2 ViT-B/14 via `torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")`, 86M params, frozen, eval mode |
| Modules profiled | 37: `patch_embed` + per block k ∈ 0–11 the whole block (`blocks.k`) and its `attn` / `mlp` sublayers |
| Activation readout | forward hooks; tokens pooled with `vit_token=cls` (CLS-token activation per module); 4D tensors → channel spatial mean |
| Data | CIFAR-10 test split, n = 1,000, seed 42 (`load_cifar10_subset`) |
| Corruptions × severities | low_light, jpeg × sev 1–5 (§3 parameter tables) |
| Eval-noise convention | matched-noise rng `default_rng(1000 + severity)` per severity — the §7.3 convention, so v9 LoRA arms see identical corrupted arrays |
| Metrics per module | energy/mean/cov/cos proxies + biased linear CKA (`utils.linear_cka`) + unbiased CKA (`stats_tools`); profile of record = `cka_unbiased_drop` |
| Profiler protocol cost | two forward passes per condition (clean + degraded), no labels, no backprop |
| Compute | Kaggle script kernel `arindamtripathi/vitb-drift-profile` v1, Tesla T4, torch 2.11.0+cu128, **259.5 s** wall; built via `scripts/build_kaggle_kernel.py --variant vitb_profile` |
| Artifacts of record | `output/vitb_profile/`: `drift_profile_sev{1..5}_{low_light,jpeg}.csv`, `agreement_report.{txt,json}`, `proxy_profiles.csv`, 11 PNGs, kernel log |

**Headline numbers (sev 5, unbiased CKA drop).** Block-level: low_light 0.179 (b0) → 0.803 / 0.866 / 0.859 (b9 / b10 / b11); jpeg 0.127 (b0) → 0.705 / 0.790 / 0.785. Sublayer level: **mlp drop > attn drop in all 12 blocks under both corruptions** (low_light gaps 0.015–0.159, max at b2: 0.727 vs 0.681; jpeg gaps 0.017–0.108, max at b1: 0.456 vs 0.358). Gate columns at sev 5: best ρ jpeg 0.876 (energy_drop) / 0.882 (cos_drop) vs low_light 0.692 / 0.681, top-5 containment 0.4–0.6 — third-architecture NO-GO.

**v9 consumer note (re-indexing).** The drift CSVs key `layer` by *profiled-module* row (0–36, in the `agreement_report.json` module order: `patch_embed`, then per block `blocks.k`, `blocks.k.attn`, `blocks.k.mlp`). The Phase-3 LoRA consumer (`run_lora_simple_colab.py::load_drift_profile`) expects **dense 0..n_blocks−1 block indices** — v9 must select the 12 `blocks.k` rows (regex `^blocks\.\d+$`) and re-index them 0–11 before `--drift-csv` use (implemented in `tools/reindex_drift_for_lora.py`; v9 run 2026-10-09).

### 7.9 ExDark real-dark suite (`run_exdark_baseline.py`)

| Item | Value |
|------|-------|
| Dataset | ExDark, 7,363 images, 12 classes (Kaggle mirror `mangosata/exclusivelydarkimagedataset-from-csbdu`, per-class counts match official stats) |
| Labels | directory structure (no lighting tags survive in any public mirror) |
| Darkness axis | empirical: mean Rec.601 luminance per image (0–255 scale); dataset median 33.1, range 0.5–157.3 |
| Probe | logistic regression on frozen DINOv2 embeddings, stratified 70/30, `random_state=42` |
| Passes | (1) raw, (2) CLAHE-enhanced (`cv2.createCLAHE`, tile 8×8, clip 2.0, on L-channel of LAB), (3) adapter transfer — rebuild LoRA from `lora_adapters.pt` (adapter tensors + per-block rank config), load into a fresh pristine hub model, embed, probe on the same split |
| Feature cache | `--cache-feats` memoizes raw+CLAHE base-model features (model-keyed npz in the output dir); transfer runs share it via copy |
| Environment | local CPU, streaming batches of 16 (peak RSS ~1 GB) |

Transfer-test protocol notes: adapters were trained on synthetic CIFAR darkness only; ExDark images are used exclusively for evaluation; the probe split (`random_state=42`) is identical across all passes, so raw-vs-adapted deltas are computed on the same test indices.

### 7.10 Cross-family drift atlas (`run_drift_profile.py --dataset imagenet`; kernel `drift-atlas-v4`)

Track 3's profile atlas of record (RESEARCH §5.2, DPA_DESIGN §2). Same harness as §7.8, extended by the `--dataset imagenet` mode: `utils.load_imagenet_subset` enumerates the ImageNet-val extract in sorted (wnid, filename) order and draws a seeded subset without replacement — the §7.8 determinism convention transplanted to class-folder JPEGs. `--cache-dir` persists the decoded image list once per session (keyed by n/seed/data-root hash), so the six per-model invocations read one FUSE glob+decode instead of six.

| Item | Value |
|------|-------|
| Models (6 ImageNet legs) | torchvision ResNet-50 IMAGENET1K_V1; timm convnext_tiny (28.6M), vit_small_patch16_224 (22.1M), swin_tiny_patch4_window7_224 (28.3M), deit_small_patch16_224 (22.1M); torch.hub dinov2_vitb14 — all frozen, eval mode, `.eval().to(device)` at load (the torchvision weights= loader leaves train/CPU; fixed in `run_drift_proxy.load_model`) |
| Consistency legs (2) | ResNet-50 + ConvNeXt-T on CIFAR-10 test n = 1,000, sev 1/5 — the §7.8 image axis, anchoring atlas rows to Track 1/2 artifacts |
| Data (ImageNet legs) | `sautkin/imagenet1kvalid` (ILSVRC2012 val, 1000 folders × 50), n = 2,000, seed 42, resolved at `/kaggle/input/datasets/sautkin/imagenet1kvalid` |
| Modules profiled | 25 (ResNet-50), 15 (ConvNeXt-T), 15 (Swin-T), 38 (ViT-S/16), 38 (DeiT-S), 37 (DINOv2 ViT-B/14) — forward hooks, no backprop |
| Corruptions × severities | low_light, jpeg × sev 1–5 (§3 tables); matched-noise rng `1000 + severity` (§7.3) |
| Metrics | energy/mean/cov/cos proxies + biased & unbiased CKA; profile of record `cka_unbiased_drop`; cov_shift computed in float32 (documented in `COV_DTYPE`) |
| Harness guards (v4) | `PYTHONUNBUFFERED=1`; per-model `timeout --kill-after` (4,500 s ImageNet / 2,400 s CIFAR); 8.5 h deadline guard before starting a model; `ATLAS_MARK` audit lines mirrored to `output/atlas_marks.log` |
| Compute | Kaggle script kernel `arindamtripathi/drift-atlas-v4`, Tesla T4, torch 2.11.0+cu128, **2,768 s (46 min)** total; per-model 365–707 s (ImageNet), 51 s (CIFAR legs); built via `scripts/build_kaggle_kernel.py` (atlas v4 variant) |
| Artifacts of record | `output/profiles/<model>/`: `drift_profile_sev{1..5}_{low_light,jpeg}.csv`, `agreement_report.{txt,json}`, `proxy_profiles.csv`, heatmap + per-condition PNGs, `profile.log`; atlas roll-up `output/atlas/atlas_summary.csv` + `output/atlas/atlas_fig1.png` (`tools/build_atlas.py`); run audit `output/atlas_marks.log` |

Note on the txt reports: the `images :` header line said "CIFAR-10 test" for the ImageNet legs in the v4-run output (hardcoded label, since fixed in `write_agreement_report` with a regression test); the JSON `meta.dataset` was correct (`"imagenet"`) throughout, and the six txt files were patched post-run to match. Numbers unaffected.

### 7.11 SSR S1 smoke (`run_ssr.py`; kernel `ssr-s1`)

Track 4's first real-data session (SSR_PLAN §7 S1; verdict RESEARCH §6.7). Runner `run_ssr.py`; core math (`channel_stats`, `apply_affine`, shift distance, gate, top-k selector) is the torch-free `src/ssr.py` already covered by §CI tests (47/47 at the S2 harness landing, 2026-10-15).

| Item | Value |
|------|-------|
| Models | torchvision ResNet-50 IMAGENET1K_V1 (BN arm, 8 arms); timm `vit_small_patch16_224` pretrained (LN arm, 7 arms — no bn_stats) — both frozen, eval mode, CUDA |
| Data | `sautkin/imagenet1kvalid` mount (§7.10), one seeded draw of n_total = 640 images (`--n-est 128 --n-images 512`), prefix-split disjoint per SSR_PLAN §6: first 128 = clean gate calibration + estimation pool (B = 4 batches × 64, `--est-batches 4 --batch-size 64`), remaining 512 = eval; image-list cache `ssr_cache/imagenet_n640_seed42_*.pkl` holds the union; seed 42; matched-noise `rng 1000 + severity` (§7.3) |
| Corruptions × severities | low_light, jpeg × {1, 5} (the S1 smoke grid; S2 would extend to the full 1–5 ladder) |
| Selection input (of record) | Track 3 atlas severity-5 CSVs, `output/profiles/<model>/drift_profile_sev5_{corr}.csv`, ranked by `cka_unbiased_drop`, k = top 25% (ResNet-50: top-6/25; ViT-S: top-10/38) — the one-measurement-two-consumers link, live |
| Gate | τ = max clean RMS standardized moment distance × margin 1.0 (n_pool = 128 clean images); per-invocation, model-specific (ResNet-50 τ=0.106, ViT-S τ=0.082); `d ≤ τ` → bypass (no module touched) |
| Identity check (C2) | On clean inputs: gate must close and disarmed-hook max abs logit diff must be 0.0 → `bit_identical: true` in `ssr_meta.json`. All 4 invocations: bit-identical, max diff 0.0 |
| Arms | frozen · ssr_topk · ssr_all · ssr_gate_off · bn_stats (ResNet-50 only) · tent (BN native / LN-affine translation, SSR_PLAN §0.3) · aqr · clahe — per-arm top-1, Δ vs frozen, shift distance, gate state, modules touched, wall time in `ssr_results.csv` |
| Compute | Kaggle script kernel `arindamtripathi/ssr-s1`, Tesla T4, torch 2.11.0+cu128, **2,027 s (34 min)** total (4 invocations: 955/336/429/306 s); built via `scripts/build_kaggle_kernel.py --variant ssr_s1`, tail `scripts/kaggle/ssr.sh` |
| Artifacts of record | `output/ssr/<model>_<corr>/`: `ssr_results.csv` (16 rows = 8 arms × 2 severities, or 14 for ViT-S), `ssr_meta.json` (selection, gate, identity, per-condition shift), `ssr.log`; run audit `output/ssr_marks.log` |

Known incident (harness, not results): the kernel's exit verifier initially hardcoded `vit_small_patch16_224_*` dir names while `ssr.sh` sanitizes the timm slug to `timm_vit_small_patch16_224_*` (colon → underscore), so the completed run was misreported as `SSR_S1_FAILED` in `output/ssr_marks.log` despite all four legs rc=0 with valid artifacts. Verifier now derives names from the driver's slug list; regression test `test_ssr_kernel_verifier_matches_driver_dir_names` (44th test) locks the invariant.

### 7.12 SSR S2 attachment-point probe (`run_ssr.py --attach`; kernel `ssr-s2`)

Track 4's S2 probe (SSR_PLAN §7.2; verdict RESEARCH §6.8). Protocol = §7.11 byte-for-byte with `--attach` as the only design variable; scope logic in the torch-free `src/ssr.py` (`module_in_scope`, `map_scope_rows`), driver `scripts/kaggle/ssr_s2.sh`, builder variant `ssr_s2`.

| Item | Value |
|------|-------|
| Models | the same two: torchvision ResNet-50 IMAGENET1K_V1 (BN) and timm `vit_small_patch16_224` pretrained (LN), frozen, eval mode, CUDA |
| Data | identical draw and split to §7.11: `sautkin/imagenet1kvalid` mount, n_total = 640 (`--n-est 128 --n-images 512`), first 128 = calibration/estimation pool (B = 4 × 64), remaining 512 = eval, seed 42, matched-noise `rng 1000 + severity` (§7.3), same cached image list `ssr_cache/imagenet_n640_seed42_*.pkl` |
| Corruptions × severities | **low_light × {1, 5} only** — S1's maximal-harm family and the §2 luminance family; jpeg deferred to S3 |
| Legs (`--attach`) | 7: ResNet-50 {profiled, sublayer, stream} + ViT-S {profiled, sublayer, stream, norm}; output dir `output/ssr_s2/<slug>_<scope>_low_light/` |
| Selection (of record) | k = top 25% of the **scope's rows** of the same severity-5 atlas CSVs: profiled top-6/25 / top-10/38 (S1 parity), sublayer top-4/16 / top-6/24, stream top-1/4 / top-3/12, norm top-6/24 (norm rows borrow their sublayer's drift rank: norm1 ← attn, norm2 ← mlp — the atlas does not hook norms). Exact module lists in each `ssr_meta.json.selected_modules` (e.g. ResNet stream = [`layer4`]; ViT stream = [`blocks.7`, `blocks.8`, `blocks.11`]) |
| Gate | τ per leg (each scope forms its own moment set): ResNet 0.106 / 0.112 / 0.091 (profiled / stream / sublayer); ViT 0.082 / 0.088 / 0.068 / 0.064 (profiled / sublayer / stream / norm). Open at sev 1 and 5 on all 7 legs; closed on clean ×7 |
| Identity check (C2) | 7/7 legs `bit_identical: true`, disarmed-hook max abs logit diff 0.0 |
| Arms | frozen · ssr_topk · ssr_gate_off · tent (adaptation-only control, unaffected by attachment) — `ssr_gate_off` ≡ `ssr_topk` in every cell because the gate was open in every shifted evaluation |
| Compute | kernel `arindamtripathi/ssr-s2` v1, Tesla T4, **601 s (10 min)** — legs 162 / 95 / 65 / 80 / 69 / 63 / 67 s in driver order (ResNet profiled/sublayer/stream, then ViT profiled/sublayer/stream/norm); built via `scripts/build_kaggle_kernel.py --variant ssr_s2`, tail `scripts/kaggle/ssr_s2.sh`; exit marker `SSR_S2_COMPLETE` (verifier derives dir names from the driver's legs — `test_ssr_s2_kernel_verifier_matches_driver_dir_names`) |
| Artifacts of record | `output/ssr_s2/<slug>_<scope>_low_light/`: `ssr_results.csv` (8 rows = 4 arms × 2 severities), `ssr_meta.json` (`attach`, `selection`, `selected_modules`, `gate`, `identity`), `ssr.log`; run audit `output/ssr_s2_marks.log` |
| Known harness notes | (1) `profiled` selections contain same-tensor container/child pairs (ViT `blocks` + `blocks.11`; ResNet `layer1` + `layer1.2` and `layer4` + `layer4.2` — verified identical tensor object, so those affines compose twice); the duplication exists in S1's run as well (hence the exact profiled parity), and the non-profiled scopes are duplicate-free (containers excluded by `module_in_scope`). (2) `ssr_meta.json.per_condition_shift` keys both severities as `low_light@sev5` (leaked severity loop variable overwrites the sev1 entry) — the per-row `shift_distance` in `ssr_results.csv` is correct; meta-only defect, both noted for S3. |

---

## 8. Random Seed Registry

| Where | Seed | Purpose |
|-------|------|---------|
| `load_cifar10_subset` | 42 | test-image sampling (Phases 1–2) |
| Phase 3 train pool | 42 | 5,000-image draw |
| Phase 3 test set | 42 (independent RNG instance) | 1,000-image draw |
| All probe splits | 42 | stratified 70/30 |
| Bootstrap | severity index (0–5) | per-severity resampling independence |
| LoRA per-batch corruption / flips | per-sample `default_rng(seed=idx)` | training-time augmentation; worker-correlation fix (see §9.6) |
| Eval-time corruption (Phase 3) | 1000 + severity | matched-noise comparison; see §7.3 |
| Phase 3 v2 arms | `--seed` per arm (42/43/44) | test-set draw, torch init, shuffling — see §7.4 comparability caveat |
| ExDark suite | 42 | stratified 70/30 split, identical across all passes |
| Corruption-family GPU arms (sessions E/F) | 42 | uniform + drift arms, blur/jpeg/contrast |
| Readout-repair decomposition | 42 | test draw, probe split, permutation resampling (5,000) |
| ViT-B drift-profile sweep (v8) | 42 | n = 1,000 test draw (`load_cifar10_subset`); eval corruption rng `1000 + severity` per §7.3; harness `run_drift_profile.py`, kernel `vitb-drift-profile` v1 |
| Blur adapter re-evaluation | 42 | test draw + eval corruption (`1000 + severity`), identical to §7.3; adapters from sessionE (seed 42) |
| Track 2 v9 LoRA arms + allocation-shape ablations | 42/43/44 | per-arm test draw, init, shuffling; identical across arms within a seed for paired comparisons; kernels `vitb-lora-v9`, `vitb-lora-abl` — protocol + results in RESEARCH §6.5/§6.6 |
| Track 3 cross-family atlas | 42 | n = 2,000 ImageNet-val draw (`load_imagenet_subset`), identical across all 6 model legs (shared-image comparison); eval corruption rng `1000 + severity` per §7.3; kernel `drift-atlas-v4` — protocol §7.10, results RESEARCH §5.2 |

---

## 9. Known Implementation Notes

1. **AdamW parameter groups.** The classifier head originally appeared in both the LoRA group (`requires_grad` filter) and the head group, crashing AdamW ("some parameters appear in more than one parameter group"). Fixed by excluding head parameter `id()`s from the LoRA group.
2. **Headless matplotlib.** All scripts force `matplotlib.use("Agg")` before any figure work (centralized in `utils.py`).
3. **Notebook vs. script parity.** The `.py` runners are the canonical implementations; the `.ipynb` versions mirror them cell-by-cell. Where they diverged historically (the CKA bug existed in both), both were fixed.
4. **CKA numeric record.** `output/notebook2/cka_matrix.csv` (produced by `cka_recompute.py`, same protocol as `run_notebook2.py`) is the authoritative numeric artifact for the layer × severity CKA matrix quoted in the docs. Early console logs predate the CKA fix and are superseded by this CSV.
4b. **Phase 1 output naming.** Since the Phase 0 parameterization, `run_notebook1.py` writes `corruption_results.csv` (generic name for any `--corruption`). The tracked artifact `output/notebook1/dinov2_lowlight_results.csv` predates the rename and remains the Phase 1 record of the published run; a fresh default-mode reproduction produces the same numbers under the new filename.
4c. **Torch seeding (Phase 0 review fix).** `torch.manual_seed(--seed)` is applied *before* any model creation, so LoRA-A init, head init, dropout, and shuffling are all deterministic functions of `--seed`. Multi-seed runs on separate VMs therefore differ only where the seed intends.
5. **Seed-consumption subtlety.** `load_cifar10_subset` creates a *fresh* `default_rng(42)` per call, so Phase 1/2 draws are independent samples, not nested subsets. The Phase 3 script creates one RNG instance and draws test (1,000) then train (5,000) sequentially — the test draw therefore coincides with Phase 1's set. Train-pool images come from the CIFAR-10 **train** split, so LoRA training never sees a test image.
6. **Worker-correlated augmentation (Phase 3, fixed).** The first LoRA implementation drew augmentation randomness from NumPy's *global* RNG inside `__getitem__` with `num_workers=2`; DataLoader workers fork without re-seeding NumPy, so both workers produced correlated severity/flip/noise streams. Fixed with per-sample `default_rng(seed=idx)` — deterministic per image, statistically independent across images — and the training noise itself is now threaded through the same per-sample RNG (`low_light(..., rng=rng)`). The original published run predated this fix; the run of record (`colab_results/lora_run_fixed/`) uses it. Impact assessed as benign (marginal rates were correct; correlation reduces augmentation diversity without biasing), but dark-end numbers improved modestly with the fix (sev-5 LoRA 0.337 → 0.377).
7. **`low_light` seeding.** Severity 0 is deterministic (brightness scaling only). Severities ≥ 1 add Gaussian noise; everywhere in the run of record this comes from an explicit seeded generator (per-sample during training, `1000 + severity` during eval), never the global stream.

---

## 10. Environment (local reference run)

| Package | Version |
|---------|---------|
| Python | 3.14.3 |
| torch | 2.13.0 |
| torchvision | 0.28.0 |
| scikit-learn | 1.9.0 |
| numpy | 2.5.2 |
| scipy | 1.18.1 |
| matplotlib | 3.11.1 |

Colab reference (Phase 3): torch 2.11.0+cu128, torchvision 0.26.0+cu128, sklearn 1.6.1, Tesla T4 (15.6 GB), CUDA 12.8.

Kaggle reference (corruption-family session F): Kaggle T4 image (Python 3.12), single Tesla T4; exact torch version per the retained kernel log (`colab_results/kaggle_sessionF/kernel_run.log`).

Pinned versions for local reproduction: see `requirements.txt`.

---

## 11. Kaggle GPU channel (free-tier, session F onward)

Primary GPU channel after Colab availability became unreliable. A Kaggle **script kernel ships only the single `code_file`**, so all runtime files are packaged as base64 blobs inside one generated script.

**Build** (`scripts/build_kaggle_kernel.py` — repo-resident, /tmp-wipe-proof):

```bash
.venv/bin/python3 scripts/build_kaggle_kernel.py [--pkg /tmp/kaggle_kernel]
# packages: utils.py, run_notebook2.py, run_lora_simple_colab.py (repo root),
#           scripts/kaggle/grid_tail.sh (the arm chain),
#           colab_results/sessionE/nb2_jpeg/cka_matrix.csv (jpeg drift profile)
```

Generated: `<pkg>/kaggle_corrgrid_tail.py` (byte-identical to the kernel of record; verified by `diff` against the v6 run's artifact) + `kernel-metadata.json`. Per-file source overrides: `--source name=path`.

**Push and run.** Both GPU knobs matter — omitting either leaves the kernel on CPU:

```bash
cd /tmp/kaggle_kernel && kaggle kernels push -p .
# kernel-metadata.json: "enable_gpu": true, AND accelerator nvidiaTeslaT4
kaggle kernels status arindamtripathi/corrgrid-tail   # poll
kaggle kernels output arindamtripathi/corrgrid-tail -p /tmp/kg_out  # pull
```

Kernel runtime behavior (embedded driver): extracts packaged files to cwd (Kaggle mounts `/kaggle/working` as the capture dir), extracts CIFAR-10 from the private dataset mount `arindamtripathi/cifar10-python` (162 MB, staged once via the Kaggle datasets API; falls back to torchvision download), **hard-aborts if CUDA is unavailable** (no wasted run), then executes `grid_tail.sh` and prints `GRID_COMPLETE`/`GRID_FAILED` as the terminal marker.

**Quota:** 30 GPU-hours/week (verified phone-verified account). v6's three-arm chain used ~37 min wall on T4. Kernels persist server-side across laptop reboots — the channel's key advantage over Colab sessions and /tmp-based staging.
