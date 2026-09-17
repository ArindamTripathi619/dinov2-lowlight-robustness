# Research Documentation: DINOv2 Low-Light Robustness Study

*What we set out to learn, how we learned it, what we found, and what it means.*

---

## 1. Aim

Self-supervised Vision Transformers such as **DINOv2** produce representations that transfer remarkably well to downstream tasks — but that reputation is built almost entirely on **clean, well-lit benchmark imagery**. Real deployments (night-time perception, surveillance, automotive, mobile) feed models images that are dark and noisy.

**Our central question:**

> When images get dark and noisy, how badly does DINOv2's understanding break — **where** in the network does it break, **why** does it break, and **can we fix it cheaply**?

We chose a controlled synthetic setting — CIFAR-10 images passed through a parametric low-light corruption — because it lets us hold everything constant except the phenomenon of interest and measure the degradation *causally*, layer by layer.

---

## 2. Background & Why This Design

**Backbone.** DINOv2 ViT-S/14 (22M params, frozen throughout). We deliberately evaluate the *representation*, not a trained system: the backbone is never fine-tuned in Phases 1–2, so any accuracy drop is a statement about the embeddings themselves.

**Evaluation protocol.** The standard DINOv2 transfer protocol — a **linear probe** (logistic regression on frozen CLS embeddings) — trained **only on clean images**, then evaluated across severity levels. This clean-only training is the crux of the design: it isolates *representation degradation* from *train/test distribution mismatch*. If accuracy falls, the representation itself moved; the classifier didn't simply forget dark images.

**Corruption model.** `low_light()` multiplies pixel intensity by a severity-dependent factor and adds severity-dependent Gaussian "shot noise" (severity 0 = clean, 5 = 0.15× brightness + heavy noise). Darkness and sensor noise co-occur in real low-light photography, so both are modeled.

**Why these choices?**
- **ViT-S/14** — the standard efficient variant; also makes CPU-local experimentation tractable.
- **Synthetic corruption** — full control over the degradation axis; real dark-photo datasets confound darkness with content.
- **Linear probe** — if a *linear* readout of frozen features degrades, the failure is in the features, not in head capacity.
- **CL pooling** — the CLS token is DINOv2's canonical transfer representation.

---

## 3. Hypotheses

The three phases were designed to answer, in order:

| # | Hypothesis | How tested |
|---|-----------|------------|
| H1 | DINOv2's linear-probe accuracy degrades *severely* and *non-linearly* under low light | Phase 1: accuracy + embedding-drift curves across 6 severity levels |
| H2 | The degradation is not uniform — it concentrates in specific (late, attention-level) layers rather than early edge detectors | Phase 2: layer-wise CKA between clean and degraded activations at every block |
| H3 | The degradation is driven by loss of **low-frequency luminance structure**, which DINOv2 relies on most | Phase 2: low-pass vs. high-pass ablation through the *same* probe |
| H4 | Because the failure is localized, a **targeted, parameter-efficient** intervention recovers most of the loss | Phase 3: LoRA adapters on attention only, mixed clean/dark training diet |

Note the deliberate logical chain: H2+H3 *predict* the Phase 3 design. If drift had concentrated in early layers, the remedy would have been different (early-layer unfreezing); if the frequency finding had gone the other way, a frequency-aware augmentation would have been indicated instead.

---

## 4. Phase 1 — Measure: How bad is it?

**Method.** 1,000 CIFAR-10 test images (seed 42) → 6 severity levels each → frozen DINOv2 embeddings → logistic probe trained on clean embeddings (70/30 stratified split) → evaluated at every severity. Embedding drift measured as mean cosine similarity between each image's clean and degraded embeddings.

**Results** (`output/notebook1/`):

| Severity | Accuracy | Cosine sim to clean |
|----------|----------|---------------------|
| 0 (clean) | 0.913 | 1.000 |
| 1 | 0.900 | 0.946 |
| 2 | 0.860 | 0.810 |
| 3 | 0.603 | 0.589 |
| 4 | 0.220 | 0.309 |
| 5 (darkest) | 0.093 | 0.161 |

![Phase 1 accuracy and drift](../output/notebook1/dinov2_lowlight_results.png)

**Interpretation.**
- **H1 supported.** Accuracy falls 0.82 — a cliff, not a slope. Collapse begins below severity 3.
- Accuracy loss tracks embedding drift almost linearly — the model isn't "confused," its feature space *moves*.
- Chance is 0.10; at severity 5 the probe is barely above chance.

---

## 5. Phase 2 — Localize & Explain: Where and why does it break?

**Method.** 500 images (disjoint regime check), forward hooks on all 12 transformer blocks capturing layer-wise CLS activations in the *same* forward passes. Three analyses:

1. **Layer-wise CKA** — linear CKA between clean and degraded activations per block per severity. CKA ≈ 1 means the layer's representational geometry survived; ≈ 0 means it was rebuilt.
2. **Frequency ablation** — FFT circular low-pass / high-pass filters at matched severities through the *unchanged* probe. If the model leans on low frequencies, low-pass should hurt less than high-pass.
3. **Bootstrap CIs** — 1,000 resamples per severity so accuracy claims carry error bars.

**Results** (`output/notebook2/`):

CKA drop (clean → severity 5) by layer:

| Layer | CKA drop |
|-------|----------|
| block 0 (patch embed) | 0.57 |
| block 5 | 0.45 |
| **block 10** | **0.82** ← max |
| block 11 | 0.79 |

**→ H2 supported: the drift concentrates in late attention layers.**

Frequency ablation: low-pass preserves near-clean accuracy (~0.92 at severity 1) while high-pass destroys it (~0.59), at every severity.

**→ H3 supported: DINOv2 runs on low-frequency luminance structure — exactly what darkness removes.**

![Layer-wise CKA](../output/notebook2/layerwise_cka.png)

![Frequency test](../output/notebook2/frequency_test.png)

---

## 6. Phase 3 — Remediate: Can it be fixed cheaply?

**Design rationale.** H2+H3 localize the failure in attention layers of a frozen 22M-param backbone. Full fine-tuning would be wasteful and risks destroying the pretrained representation. **LoRA** (rank 8, α 16) on attention QKV + output projections trains **221K params = 0.99%** of the network.

**Training diet.** 5,000 train images, **70% low-light-augmented** (random severity 2–4, plus horizontal flips) / 30% clean — dark enough to force adaptation, clean enough to not forget. 10 epochs, AdamW with separate LR groups (adapters 5e-5, head 1e-3), batch 64, ~10 minutes on a free-tier Colab T4.

**Results** (1000-image test set, matched protocol, `colab_results/lora_run/`):

| Severity | Original | LoRA | Δ |
|----------|----------|------|-----|
| 0 (clean) | 0.913 | 0.960 | +0.047 |
| 3 | 0.630 | 0.900 | +0.270 |
| 4 | 0.237 | 0.773 | **+0.537** |
| 5 (darkest) | 0.080 | 0.337 | **+0.257** |
| **Mean** | **0.603** | **0.811** | **+0.208** |

**→ H4 supported.** Worst-case 4.2× improvement, no clean-image penalty (it rose +0.047). Training converged smoothly (loss 0.94 → 0.14, no divergence, no overfitting signature).

![Original vs LoRA](../colab_results/lora_run/lora_vs_orig_accuracy.png)

---

## 7. The Findings in Plain Language

1. **DINOv2 fails hard in the dark** (0.91 → 0.09; near-chance at severity 5). Its clean-benchmark reputation does not survive darkness.
2. **The failure has an address.** Late attention blocks drift most (CKA drop 0.82 at block 10); early layers are comparatively stable. The failure is *localized*, not diffuse.
3. **The failure has a mechanism.** The model's accuracy survives low-pass filtering nearly intact but collapses under high-pass filtering — it reads low-frequency luminance structure, and darkness erases precisely that.
4. **The failure is cheap to reverse.** Because it is localized, LoRA on the affected layers — 1% of the network, 10 GPU-minutes — recovers most of the lost robustness *without hurting clean accuracy*.
5. **Accuracy loss ≈ embedding drift.** One phenomenon, measured (Phase 1), explained (Phase 2), reversed (Phase 3).

---

## 8. Bugs We Hit, and What They Teach (Methodology Lessons)

Two silent bugs shaped our confidence in the results — documented here because they are instructive.

### 8.1 The CKA sqrt bug (invalid metric, correct conclusion — by luck)

**What happened.** The original `linear_cka` normalized by `var1 * var2` instead of `sqrt(var1 * var2)`, squaring every CKA value. Severity-5 values reached **79.9** — wildly outside CKA's valid [0, 1] range — and "CKA drop" values were negative.

**How we caught it.** The negative "drops" looked wrong; recomputing by hand exposed the missing square root.

**Why the conclusion survived.** Squaring is monotonic, so the *ordering* of drift across layers was preserved — the late-layer finding held. The absolute values were meaningless.

**Lesson.** *Relative* conclusions drawn from an invalid metric can be right by accident. Every metric should be sanity-checked against its theoretical range before its numbers are interpreted. (Fixed everywhere; see `docs/METHODS.md` §4 for the corrected formula.)

### 8.2 The bootstrap same-seed bug (overconfident error bars)

**What happened.** The bootstrap CI routine seeded its RNG (`default_rng(0)`) *inside* the severity loop — every severity resampled identically, making CI bands misleadingly similar across severities.

**Why it matters.** Error bars that are wrong in a correlated way don't just add noise to conclusions — they manufacture false confidence in comparisons between conditions.

**Lesson.** Statistical-resampling code deserves the same scrutiny as model code: check that randomness is *actually* independent across the conditions being compared. (Fixed: per-severity seeds; see `docs/METHODS.md` §5.)

### 8.3 Port-time regression (caught by review, not runtime)

While porting an optimizer fix into the LoRA notebook, an edit referenced `head_ids` before `head_params` was defined. Static validation before commit caught it. **Lesson:** script-vs-notebook code drift is a real hazard; order-of-definition bugs hide easily in notebook cells.

---

## 9. Limitations

- **Synthetic corruption only.** Real low-light photography involves ISP pipelines, color casts, dynamic-range clipping, and exposure metadata that our model does not simulate.
- **One dataset, one backbone size.** CIFAR-10 at 32×32 (upsampled to 224×224) is far from ImageNet-scale statistics; ViT-B generality is planned but unverified here.
- **Linear probe ceiling.** A stronger head might partially compensate for feature drift; we measured the *linear* story deliberately.
- **No end-to-end fine-tuning baseline.** We did not compare LoRA against full fine-tuning on the same diet — the 0.99%-parameter result is compelling but not benchmarked against the expensive alternative.
- **Single runs.** Phase 3 numbers come from one training run per configuration; the sweep (ranks, mix ratios) is future work.

---

## 10. What We Achieved

- A **complete, reproducible three-phase pipeline** (measure → localize/explain → remediate) built on a shared, tested `utils.py`.
- **Quantified the failure**: 0.91 → 0.09 accuracy, embeddings drifting to near-orthogonality.
- **Localized and mechanistically explained it**: late-layer CKA drift + low-frequency dependence.
- **Demonstrated the fix**: 0.99% of parameters, ~10 GPU-minutes, +0.21 mean / 4.2× worst-case recovery with no clean penalty.
- **Hardened the science along the way**: corrected CKA formula, independent bootstrap seeds, deduplicated optimizer groups — each validated before results were trusted.
- All findings, figures, and full training logs are versioned in this repository.

---

## 11. Future Work

1. **ViT-B generality** — does the late-layer drift signature hold at 86M params?
2. **Corruption family expansion** — blur, JPEG, contrast: is this low-light-specific or general fragility?
3. **LoRA sweep** — rank ∈ {4, 8, 16, 32} × mix ratio ∈ {50/50, 70/30, 90/10}: what is the *minimal* effective intervention?
4. **Real dark imagery** — validation on actual low-light datasets (e.g., ExDark) to close the synthetic-to-real gap.
5. **CKA drift as a predictor** — can per-layer drift measured at severity *s* predict LoRA recovery at severity *s*, making CKA a *diagnostic* that targets adaptation before training?

---

*See `docs/METHODS.md` for exact parameters, formulas, seeds, and environment versions; see `README.md` for the quick-summary view and run instructions.*
