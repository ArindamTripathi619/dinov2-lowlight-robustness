# Methods Appendix

*Exact parameters, formulas, data protocol, and environment for every experiment in this study. All values verified against source (`utils.py`, `run_notebook1.py`, `run_notebook2.py`, `run_lora_simple_colab.py`, `cka_recompute.py`).*

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

### 3.2 Frequency filters (`low_pass` / `high_pass`)

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

Pinned versions for local reproduction: see `requirements.txt`.
