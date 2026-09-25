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
- **H1 supported.** Accuracy falls 0.82 — a cliff, not a slope: near-flat through severity 2, collapsing steeply between severity 2 and 4.
- Accuracy loss tracks embedding drift almost linearly — the model isn't "confused," its feature space *moves*.
- Chance is 0.10; at severity 5 the probe is barely above chance.

---

## 5. Phase 2 — Localize & Explain: Where and why does it break?

**Method.** 500 CIFAR-10 test images (an independent seed-42 draw from the same test split as Phase 1 — overlap with Phase 1's 1,000 images is ~5%, i.e. chance level for two random draws), forward hooks on all 12 transformer blocks capturing layer-wise CLS activations in the *same* forward passes. Three analyses:

1. **Layer-wise CKA** — linear CKA between clean and degraded activations per block per severity. CKA ≈ 1 means the layer's representational geometry survived; ≈ 0 means it was rebuilt.
2. **Frequency ablation** — FFT circular low-pass / high-pass filters at matched severities through the *unchanged* probe. If the model leans on low frequencies, low-pass should hurt less than high-pass.
3. **Bootstrap CIs** — 1,000 resamples per severity so accuracy claims carry error bars.

**Results** (`output/notebook2/`):

CKA drop (clean → severity 5) by layer — full matrix in `output/notebook2/cka_matrix.csv`:

| Layer | CKA drop |
|-------|----------|
| block 0 (patch embed) | 0.58 |
| block 5 | 0.53 |
| **block 10** | **0.81** ← max |
| block 11 | 0.78 |

**→ H2 supported: the drift concentrates in late attention layers.**

Frequency ablation: low-pass preserves near-clean accuracy (~0.92 at severity 1) while high-pass destroys it (~0.59), at every severity.

**→ H3 supported: DINOv2 runs on low-frequency luminance structure — exactly what darkness removes.**

![Layer-wise CKA](../output/notebook2/layerwise_cka.png)

![Frequency test](../output/notebook2/frequency_test.png)

---

## 6. Phase 3 — Remediate: Can it be fixed cheaply?

**Design rationale.** H2+H3 localize the failure in attention layers of a frozen 22M-param backbone, suggesting a targeted low-parameter intervention instead of updating everything. **LoRA** (rank 8, α 16) on attention QKV + output projections trains **221K params = 0.99%** of the network. Whether that bet pays off is an empirical question — Phase 3 v2 (below) benchmarks it against full fine-tuning directly.

**Training diet.** 5,000 train images, **70% low-light-augmented** (random severity 2–4, plus horizontal flips; per-sample seeded RNG so DataLoader workers cannot produce correlated augmentation streams) / 30% clean — dark enough to force adaptation, clean enough to not forget. 10 epochs, AdamW with separate LR groups (adapters 5e-5, head 1e-3), batch 64, ~10 minutes on a free-tier Colab T4.

**Results** (1000-image test set, matched protocol, artifact of record `colab_results/lora_run_fixed/`):

| Severity | Original | LoRA | Δ |
|----------|----------|------|-----|
| 0 (clean) | 0.913 | 0.953 | +0.040 |
| 3 | 0.603 | 0.883 | +0.280 |
| 4 | 0.213 | 0.807 | **+0.593** |
| 5 (darkest) | 0.073 | 0.377 | **+0.303** |
| **Mean** | **0.594** | **0.818** | **+0.224** |

**→ H4 supported.** Worst-case 5.1× improvement (0.073 → 0.377), no clean-image penalty (it rose +0.040). Training converged smoothly (loss 0.93 → 0.10, no divergence, no overfitting signature).

![Original vs LoRA](../colab_results/lora_run_fixed/lora_vs_orig_accuracy.png)

### 6.1 Phase 3 v2 — the full grid: targeting, seeds, and the full-FT baseline

Phase 3 v2 turns the original single-run result into a defensible comparison grid. The enabler was the Phase 0 refactor of `run_lora_simple_colab.py` (argparse flags for rank/layers/seed/model/corruption, torch seeding before any model creation, adapter checkpointing). Six arms, all on the same diet/protocol as the run of record, artifacts in `colab_results/sessionD/`:

| Arm | Trainable params | Sev 0 | Sev 2 | Sev 3 | Sev 4 | Sev 5 | Mean |
|------|-----------------|-------|-------|-------|-------|-------|------|
| Original (frozen) | 0 | 0.913 | 0.860 | 0.600 | 0.223 | 0.083 | 0.598 |
| Uniform LoRA r8, seed 42 | 221,184 (0.99%) | 0.960 | 0.950 | 0.890 | 0.750 | 0.387 | 0.815 |
| Uniform LoRA r8, seed 43 | 221,184 (0.99%) | 0.980 | 0.967 | 0.917 | 0.757 | 0.370 | 0.828 |
| Uniform LoRA r8, seed 44 | 221,184 (0.99%) | 0.977 | 0.967 | 0.930 | 0.777 | 0.413 | 0.839 |
| Late-only LoRA (blocks 9–11, r8) | 55,296 (0.25%) | 0.937 | 0.903 | 0.723 | 0.460 | 0.160 | 0.686 |
| **Drift-weighted LoRA**, seed 42 (ranks ∝ CKA drop) | 172,800 (0.78%) | 0.963 | 0.937 | 0.890 | 0.750 | 0.380 | 0.811 |
| Drift-weighted LoRA, seed 43 | 172,800 (0.78%) | 0.980 | 0.977 | 0.967 | 0.930 | 0.757 | 0.317 | 0.821 |
| Drift-weighted LoRA, seed 44 | 172,800 (0.78%) | 0.977 | 0.973 | 0.973 | 0.930 | 0.780 | 0.417 | 0.842 |
| Full fine-tuning, seed 42 (backbone lr 1e-5, head lr 1e-3) | 22,056,576 (100%) | 0.947 | 0.923 | 0.890 | 0.750 | 0.393 | 0.810 |
| Full fine-tuning, seed 43 | 22,056,576 (100%) | 0.963 | 0.960 | 0.960 | 0.937 | 0.803 | 0.453 | 0.846 |
| Full fine-tuning, seed 44 | 22,056,576 (100%) | 0.970 | 0.970 | 0.943 | 0.900 | 0.783 | 0.460 | 0.838 |

*Note on cross-seed comparability:* each run draws its own 1,000-image test set (seeded per `--seed`), so rows using different seeds are not directly comparable cell-by-cell; within a row, Original vs adapted is image-matched and noise-matched. The three uniform rows therefore measure configuration-level variance (init + test draw), not pure init variance.

**Findings from the grid:**

1. **Parity at a fraction of the cost — now with 3-seed error bars on every headline arm.** Across seeds 42/43/44: drift-weighted 0.825 ± 0.016, uniform 0.827 ± 0.012, full fine-tuning 0.831 ± 0.019. All three ranges fully overlap — while drift-weighting trains **0.78%** of the parameters and full FT trains 100%. The CKA diagnostic identified where adaptation budget matters.
2. **Full FT buys nothing on mean robustness and costs clean accuracy.** Its mean (0.831 ± 0.019) is statistically indistinguishable from both LoRA arms, and its clean accuracy is the *worst of all adapted arms* (seed mean 0.960 vs 0.972–0.973 for LoRA arms) — the predicted forgetting cost, now replicating across all three seeds. Full FT shows a hint of better extreme-dark accuracy (sev-5 mean 0.436 vs 0.371–0.390), but per-severity seed spread (0.04–0.10 across arms) keeps even that within noise.
3. **Late-only is a useful negative result.** CKA says drift is late-concentrated, but rank-8 on blocks 9–11 only (0.25% of params) reaches just 0.686 — total adaptation budget matters too; the drift profile says where budget helps, not that early blocks need none.
4. **Seed spread is real and now quantified per arm.** SDs: uniform ±1.2, drift ±1.6, full-FT ±1.9 points (ranges 0.815–0.839 / 0.811–0.842 / 0.810–0.846). Any claim of one configuration *beating* another by less than ~1.5 points is unjustified — which is exactly why drift-vs-uniform is claimed as *parity*, not superiority.

### 6.2 Sim-to-real: does synthetic-dark adaptation transfer to real darkness?

The ExDark suite (7,363 real low-light photographs, 12 classes; empirical luminance axis; see §6.2 of the baseline protocol in `docs/METHODS.md` §7.5) closes the loop. First the baseline: frozen DINOv2 probes at **0.725** on real dark images; accuracy is **flat across the darkness axis** (darkest luminance quintile 0.713 vs brightest 0.704) and CLAHE enhancement buys **+0.001** — real darkness within ExDark's range does not reproduce the synthetic collapse (which is an extreme, lower-luminance regime), and errors are class-confusion (People 0.53, Table 0.47), not luminance-driven (Boat 0.97).

Then the transfer test — adapters trained *only* on synthetic CIFAR darkness, probed on real ExDark (same split, zero real-dark training):

| Evaluation | ExDark 12-class accuracy |
|------------|--------------------------|
| Frozen DINOv2 (raw) | 0.725 |
| Frozen + CLAHE | 0.726 (+0.001) |
| + drift-weighted synthetic-dark LoRA | **0.743 (+0.019)** |
| + uniform synthetic-dark LoRA (seed 42) | 0.741 (+0.016) |

**Interpretation.** Synthetic-dark adaptation **does transfer to real darkness — modestly**: +1.9 points (drift arm; uniform arm +1.6 — indistinguishable, consistent with the §6.1 parity) from 0.78–0.99% of parameters on never-seen real photographs. But that is ~9% of the +21-point gain the same adapters buy on the synthetic severity axis. Together with the flat darkness curve and the null CLAHE control, the honest conclusion is: *synthetic extreme darkening and real-world low light are different regimes; adaptation to the synthetic regime yields a small domain-agnostic benefit on real photos, while real low-light failure modes (class confusion) are largely orthogonal to luminance.* Quantifying that gap — rather than assuming synthetic results carry over — is itself a result.

---

## 7. The Findings in Plain Language

1. **DINOv2 fails hard in the dark** (0.91 → 0.09; near-chance at severity 5). Its clean-benchmark reputation does not survive darkness.
2. **The failure has an address.** Late attention blocks drift most (CKA drop 0.82 at block 10); early layers are comparatively stable. The failure is *localized*, not diffuse.
3. **The failure has a mechanism.** The model's accuracy survives low-pass filtering nearly intact but collapses under high-pass filtering — it reads low-frequency luminance structure, and darkness erases precisely that.
4. **The failure is cheap to reverse.** Because it is localized, LoRA on the affected layers — 1% of the network, 10 GPU-minutes — recovers most of the lost robustness *without hurting clean accuracy*. CKA-guided rank allocation matches uniform LoRA and full fine-tuning at 0.78% of parameters; full FT recovers the same robustness at 100× the cost plus a clean-accuracy penalty.
5. **Accuracy loss ≈ embedding drift.** One phenomenon, measured (Phase 1), explained (Phase 2), reversed (Phase 3).
6. **Synthetic darkness ≠ real darkness.** On real low-light photography (ExDark), frozen accuracy is flat across the luminance axis and CLAHE buys nothing — but synthetic-dark adapters still transfer a small (+1.9 pt) benefit. The synthetic regime is harsher than real darkness; sim-to-real gains are real but ~9% of synthetic-axis gains.

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

### 8.4 Worker-correlated augmentation + unmatched eval noise (subtle RNG hygiene)

Two related issues surfaced during a full-codebase audit and re-run. First, augmentation randomness came from NumPy's *global* RNG inside `__getitem__` while `num_workers=2` — but DataLoader workers fork without re-seeding NumPy, so both workers emitted correlated severity/flip/noise streams. Second, the evaluation-time noise in `low_light` was drawn independently for the LoRA pass and the original-model pass, so the two models were never compared on identical corrupted images (visible only as small run-to-run jitter in the baselines — the noise-free severity-0 matched exactly, which is what gave it away). Fixes: per-sample `default_rng(seed=idx)` for training, and precomputed seeded corrupted test arrays shared by both models. **Lesson:** *any* call to a global RNG inside a DataLoader worker or a comparison loop is a bug-in-waiting; and an exactly-matching baseline alongside jittering ones is a diagnostic signature worth learning to read. (See `docs/METHODS.md` §7.3, §9.6–9.7.)

---

## 9. Limitations

- **Synthetic-first design; real data used for validation only.** Our corruptions simulate darkness and frequency loss, not ISP pipelines, color casts, or exposure metadata. The ExDark suite (§6.2) quantifies the sim-to-real gap — adaptation transfers, but only ~9% of the synthetic-axis gain — rather than closing it.
- **Real-dark validation is bounded by ExDark's luminance range.** ExDark is dark but not extreme; the flat accuracy curve shows the synthetic collapse (an extreme, lower-luminance regime) simply does not manifest there. Findings about *extreme* darkness remain synthetic-only.
- **One dataset, one backbone size.** CIFAR-10 at 32×32 (upsampled to 224×224) is far from ImageNet-scale statistics; ViT-B generality is planned but unverified here.
- **Linear probe ceiling.** A stronger head might partially compensate for feature drift; we measured the *linear* story deliberately.
- **Seed noise and test-draw confound.** All three headline arms now carry 3-seed error bars (uniform ±1.2, drift ±1.6, full-FT ±1.9 points), but each seed also draws its own 1,000-image test set, so these SDs include test-draw variance, and per-severity spread is wide (0.04–0.10 at sev-5) — extreme-dark comparisons in particular remain underpowered. Rank × mix-ratio sweep remains future work.

---

## 10. What We Achieved

- A **complete, reproducible three-phase pipeline** (measure → localize/explain → remediate) built on a shared, tested `utils.py`.
- **Quantified the failure**: 0.91 → 0.09 accuracy, embeddings drifting to near-orthogonality.
- **Localized and mechanistically explained it**: late-layer CKA drift + low-frequency dependence.
- **Demonstrated the fix**: 0.99% of parameters, ~10 GPU-minutes, +0.22 mean / 5.1× worst-case recovery with no clean penalty.
- **Closed the advisor's loop with replicated evidence**: CKA-guided rank allocation matches uniform LoRA (0.825 ± 0.016 vs 0.827 ± 0.012, 3 seeds each) and full fine-tuning (0.831 ± 0.019) at 0.78% of parameters, with the late-only ablation (0.686) showing the drift profile is informative but total budget also matters.
- **Quantified the sim-to-real gap**: real-dark baseline (0.725, flat luminance curve, null CLAHE control) and adapter transfer (+1.9 pts from synthetic-only training) on 7,363 real ExDark photographs.
- **Hardened the science along the way**: corrected CKA formula, independent bootstrap seeds, deduplicated optimizer groups, per-sample augmentation RNG, matched-noise evaluation — each validated before results were trusted.
- All findings, figures, and full training logs are versioned in this repository.

---

## 11. Future Work

1. **ViT-B generality** — does the late-layer drift signature hold at 86M params?
2. **Corruption family expansion** — blur, JPEG, contrast: is this low-light-specific or general fragility? (Infrastructure shipped in the Phase 0 refactor: `--corruption blur|jpeg|contrast`, no code changes needed.)
3. **LoRA sweep** — rank ∈ {4, 8, 16, 32} × mix ratio ∈ {50/50, 70/30, 90/10}: what is the *minimal* effective intervention?
4. ~~Seed-replicate the drift and full-FT arms~~ — **done**: all three headline arms now have 3-seed error bars (uniform 0.827 ± 0.012, drift 0.825 ± 0.016, full-FT 0.831 ± 0.019; ranges fully overlap).
5. **Real-dark adaptation** — the transfer result (+1.9 pts) is a floor, not a ceiling: fine-tune the probe (not just adapters) on a *small* real-dark split, or adapt with real-dark data mixed into the diet, and measure how much of the remaining gap closes. The ExDark infrastructure (darkness axis, CLAHE control, feature cache) supports this directly.
6. ~~CKA drift as a predictor~~ — **done in Phase 3 v2**: per-layer drift measured in Phase 2 was converted directly into rank allocation, matching uniform LoRA at 22% fewer parameters.

---

*See `docs/METHODS.md` for exact parameters, formulas, seeds, and environment versions; see `README.md` for the quick-summary view and run instructions.*
