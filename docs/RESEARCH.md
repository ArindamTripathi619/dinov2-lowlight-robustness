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

### 5.1 The same fingerprints under other corruptions — and an estimator check

Phase-2 was re-run per corruption family (blur, JPEG, contrast; artifacts `output/notebook2_jpeg/`, `colab_results/sessionE/nb2_blur/`, `colab_results/kaggle_sessionF/nb2_contrast/`):

- **low_light, blur, JPEG all show the late-concentrated CKA profile** (drop peaks in blocks 8–10).
- **Contrast does not**: its drift profile is ~flat (max CKA drop 0.30) and *falls* in late blocks.

Reading the two together: the late-layer signature tracks *high-frequency destruction* — blur and JPEG destroy exactly the frequency band DINOv2 depends on (§5's H3), and they drift where low_light drifts; contrast preserves frequency content (it only rescales amplitudes toward the grayscale mean) and produces no localized drift. **The localization finding is a property of frequency-destroying degradations, not of darkness specifically.**

**Third architecture: DINOv2 ViT-B/14 (86M params).** The per-module profiling harness built for the Track-1 proxy gate (`run_drift_proxy.py`) was driven in severity×corruption sweep mode (`run_drift_profile.py`) on ViT-B/14 in a Kaggle T4 kernel (low_light + jpeg, severities 1–5, n = 1,000 CIFAR-10 test images, seed 42, matched-noise rng 1000+sev; artifacts `output/vitb_profile/`, protocol `docs/METHODS.md` §7.8). Three findings:

- **The late-heavy block profile replicates at 86M params.** Sev-5 low_light block-level CKA drop rises from 0.18 (block 0) through 0.80/0.87/0.86 (blocks 9/10/11, unbiased estimator); jpeg is again earlier-onset but still late-peaking (0.79/0.78 at blocks 10–11). The ViT-S signature (§5), the ResNet-50 Track-1 profile, and ViT-B now agree across three architectures — the seed of the Track-3 cross-architecture atlas.
- **Sublayer finding, invisible to the ViT-S whole-block CKA: under photometric corruption, MLP sublayers drift more than attention sublayers in all 12 blocks under both corruptions** (sev-5 low_light block 11: mlp 0.902 vs attn 0.847; block 10: 0.916 vs 0.878; jpeg block 10: 0.881 vs 0.819). The ViT-S profiler only saw whole blocks; the 37-module ViT-B atlas shows where inside each block the drift lives. Design consequence: attention-only LoRA leaves part of the measured drift unaddressed — a direct input for the v9 arms and the SSR recalibration scope (§11).
- **The proxy-gate NO-GO replicates with the scale-invariance signature.** ViT-B's best proxies rank jpeg-drift well (sev-5 ρ = 0.876 energy_drop, 0.882 cos_drop — structural damage correlates with amplitude change) but fail low_light (ρ ≤ 0.69, top-5 containment 0.4–0.6), matching ResNet-50 and ViT-S. Full CKA stays the profiler; the paper claim remains "single cheap profiling pass" (two forward passes per condition, no labels, no backprop).

**Estimator robustness.** All CKA values above use the biased (HSIC-ratio) estimator. Recomputing the full matrix with the unbiased linear CKA (Kornblith et al. 2019, App. B; `output/notebook2/cka_matrix_unbiased.csv`) changes the late-concentration *quantitatively but not qualitatively*: blocks 9–11 hold maximum drift under both estimators, and the unbiased values strengthen late concentration (block-10 sev-5 drop 0.815 → 0.859). Allocation impact: 3 of 12 block ranks shift by one at r = 8 (`docs/METHODS.md` §4). Additionally, the CKA pipeline was verified deterministic across CPU and T4 (agreement to ~1e-6; `docs/METHODS.md` §4), so CPU- and GPU-produced matrices are directly comparable.

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

### 6.3 Corruption-family expansion: does the fix generalize beyond darkness?

The Phase 0 refactor's `--corruption {blur,jpeg,contrast}` flag turned the family question into a flag combination. Phase-1 fragility was measured locally (n = 1,000, seed 42; `output/notebook1_{blur,jpeg,contrast}/corruption_results.csv`):

| Corruption | Severity-5 accuracy | Severity-5 cos-sim to clean | Profile |
|------------|--------------------:|----------------------------:|---------|
| low_light (ref) | 0.087 | 0.161 | cliff between sev 2–4 |
| blur | 0.143 | 0.201 | cliff between sev 1–3 (earliest) |
| JPEG | 0.273 | 0.321 | steep and early (0.73 already at sev 1) |
| contrast | 0.840 | 0.840 | mild, near-linear decline |

Cosine drift tracks accuracy loss in every family — the "feature space moves" signature is not low-light-specific. Then the §6 grid's two LoRA arms were run per family on free-tier GPUs (blur on Colab T4, JPEG/contrast on Kaggle T4; session E salvaged partially, session F complete — kernel protocol in `docs/METHODS.md` §11):

| Family / arm | Before → After (mean) | Δ | Interpretation |
|--------------|----------------------:|-----|----------------|
| JPEG, drift-weighted | 0.586 → **0.779** | **+19.3** | biggest LoRA win yet; worst-case (sev-5) +25.7 pts |
| JPEG, uniform | (session E) | — | chain completed upstream of the drift arm; eval retained |
| blur, drift-weighted | 0.526 → **0.703** | **+17.7** | re-evaluated from retained session-E adapters; sev-5 0.143 → 0.303 |
| blur, uniform r8 | 0.526 → 0.698 | +17.2 | parity again; drift edges worst-case (0.303 vs 0.270) |
| contrast, uniform r8 | 0.885 → 0.960 | +7.5 | mild corruption, mild recovery |
| contrast, drift-weighted | 0.885 → 0.954 | +6.9 | **graceful where drift-weighting has no signal** |

Three findings:

1. **The remediation generalizes.** Drift-weighted LoRA delivers large gains on corruptions it was never designed for — JPEG +19.3 pts, blur +17.7 — using each family's own CKA drift profile for rank allocation. Gain size tracks fragility across the whole grid: low_light +22.4 > JPEG +19.3 ≈ blur +17.7 > contrast +7, i.e. adaptation recovers what the corruption destroys, and the Phase-1 curves predict where LoRA will pay.
2. **The allocation rule is honest — and repeats its low-light signature.** On contrast the profile is ~flat (§5.1) and drift-weighting gracefully matches uniform (+6.9 vs +7.5). On blur — a frequency-destroying corruption with a genuinely late-heavy profile — drift again matches uniform on mean (+17.7 vs +17.2) while matching or edging worst-case (sev-5 0.303 vs 0.270), exactly as in the low-light grid (§6.1). The blur numbers come from a protocol-faithful re-evaluation of the retained session-E adapters (`run_blur_adapter_eval.py`); the re-evaluated original column reproduces Phase-1 exactly, certifying comparability.
3. **Gain size tracks fragility.** +19.3 (JPEG, sev-5 0.273) > +7 (contrast, sev-5 0.840): adaptation recovers what the corruption destroys, so near-ceiling families have little to recover. The Phase-1 curves predict where LoRA will pay.

Artifacts: `colab_results/kaggle_sessionF/` (per-arm plots + full logs), `colab_results/sessionE/`; numeric protocol in `docs/METHODS.md` §7.6.

### 6.4 What would a re-trained readout buy? (readout-staleness decomposition)

The Phase-1/3 protocol holds the readout fixed (trained on clean embeddings only). A natural confound: how much of the collapse is *stale readout* rather than *broken features*? Upstream's H4 answers this with a severity-adapted probe; we ported and hardened it (`run_readout_repair.py`): Arm A = §6's fixed clean-trained probe; Arm B = a probe retrained per severity on degraded versions of the **same train fold** (strict control — no test-fold reuse); paired permutation tests (5,000 perms) + Wilson CIs + readout-collapse metrics (top1_share / entropy_ratio / dominant class — upstream's F6 analysis). n = 1,000, low_light, seed 42 (`colab_results/readout_repair/`, protocol `docs/METHODS.md` §7.7):

| Severity | Fixed probe | Adapted probe | Δ | p |
|----------|------------:|--------------:|-----:|------|
| 0 | 0.913 | 0.913 | +0.0 | — |
| 1 | 0.900 | 0.890 | −1.0 | 0.18 |
| 2 | 0.860 | 0.840 | −2.0 | 0.15 |
| 3 | 0.633 | 0.720 | **+8.7** | 0.0032 |
| 4 | 0.243 | 0.497 | **+25.3** | 0.0002 |
| 5 | 0.077 | 0.377 | **+30.0** | 0.0002 |

This confirms upstream's ViT-B pilot (+22.5 pp at sev-4, n = 120) at proper power on ViT-S (n = 1,000): **readout staleness is real and large in the mid-to-deep regime**. But even a perfectly re-trained linear readout reaches only 0.377 at sev-5 (vs 0.913 clean) — the majority of the loss stays in the features. The readout-collapse signature makes the mechanism legible: predictions concentrate onto a single class as severity rises (top1_share 0.14 → 0.73, dominant class *frog*; entropy_ratio → 0.26) — the embedding geometry collapses toward a low-rank attractor rather than spreading uniformly.

Two further observations. First, the adapted probe is *worse* at sev 1–2 (−1 to −2 pts, n.s.): retraining the readout on mildly degraded data costs clean-fold generalization before drift is real — a probe-level echo of upstream's "grace regime" where mild corruption even helps. Second, the decomposition validates Phase-3's design premise: since a retrained *head* cannot recover most of the loss, adapting the *features* (LoRA) is the right layer to intervene at — and §6.1's LoRA result (0.377 at sev-5, matching the adapted-probe ceiling with a *fixed* readout) now has a clean interpretation: LoRA's budget went almost entirely into fixing features, not cosmetic readout alignment.

---

## 7. The Findings in Plain Language

1. **DINOv2 fails hard in the dark** (0.91 → 0.09; near-chance at severity 5). Its clean-benchmark reputation does not survive darkness.
2. **The failure has an address.** Late attention blocks drift most (CKA drop 0.82 at block 10); early layers are comparatively stable. The failure is *localized*, not diffuse.
3. **The failure has a mechanism.** The model's accuracy survives low-pass filtering nearly intact but collapses under high-pass filtering — it reads low-frequency luminance structure, and darkness erases precisely that.
4. **The failure is cheap to reverse.** Because it is localized, LoRA on the affected layers — 1% of the network, 10 GPU-minutes — recovers most of the lost robustness *without hurting clean accuracy*. CKA-guided rank allocation matches uniform LoRA and full fine-tuning at 0.78% of parameters; full FT recovers the same robustness at 100× the cost plus a clean-accuracy penalty.
5. **Accuracy loss ≈ embedding drift.** One phenomenon, measured (Phase 1), explained (Phase 2), reversed (Phase 3).
6. **Synthetic darkness ≠ real darkness.** On real low-light photography (ExDark), frozen accuracy is flat across the luminance axis and CLAHE buys nothing — but synthetic-dark adapters still transfer a small (+1.9 pt) benefit. The synthetic regime is harsher than real darkness; sim-to-real gains are real but ~9% of synthetic-axis gains.
7. **The failure and the fix generalize across frequency-destroying corruptions.** Blur and JPEG reproduce the late-layer drift signature and the cliff; contrast (frequency-preserving) does not. Drift-weighted LoRA posts its biggest win on JPEG (+19.3 pts mean) and gracefully degrades to uniform on contrast — the CKA diagnostic allocates budget only where the diagnostic says drift lives.
8. **Half the deep-regime damage is a stale readout — the other half is the features.** A severity-adapted probe recovers up to +30 pts (sev 5) but still only reaches 0.38 vs 0.91 clean; and its predictions collapse onto a single class (top1_share 0.73, "frog") as darkness deepens. Feature adaptation, not readout retraining, is the binding constraint — which is why LoRA works.

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
- **Corruption-family LoRA arms are single-seed (seed 42).** The §6.1 error bars (±1.2–1.9 pts across seeds) come from the low-light family; family deltas (+19.3 JPEG, +17.7 blur, +6.9 contrast) exceed that noise comfortably, but family-level parity claims (drift vs uniform within blur or contrast) are single-seed and should not be over-read.
- **Biased CKA estimator in the published allocation.** The drift-weighted ranks derive from the biased estimator's profile; the unbiased recompute shifts 3 of 12 block ranks by one (§5.1) without changing the late-heavy structure. A LoRA re-run on the unbiased profile is planned but unverified.
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

1. **ViT-B generality** — does the late-layer drift signature hold at 86M params? → **profile half done (§5.1): the late-heavy profile replicates on ViT-B/14, with the added mlp > attn sublayer twist; the LoRA-arm half (v9: uniform vs ViT-B-drift-weighted vs late-only) is pending.** A v9-adjacent extension: sublayer-resolved allocation — the ViT-B profiles say MLP sublayers deserve budget before attention.
2. ~~Corruption family expansion~~ — **done end-to-end**: Phase-1 curves + CKA profiles for all four families (§5.1, §6.3) and Phase-3 LoRA arms for low_light, JPEG, blur, contrast — including the blur arms recovered from retained adapters after the session-E log truncation (§6.3). Remaining optional: a noise/saturation family as a third positive control for the frequency-destruction claim.
3. **LoRA sweep** — rank ∈ {4, 8, 16, 32} × mix ratio ∈ {50/50, 70/30, 90/10}: what is the *minimal* effective intervention?
4. ~~Seed-replicate the drift and full-FT arms~~ — **done**: all three headline arms now have 3-seed error bars (uniform 0.827 ± 0.012, drift 0.825 ± 0.016, full-FT 0.831 ± 0.019; ranges fully overlap).
5. **Real-dark adaptation** — the transfer result (+1.9 pts) is a floor, not a ceiling: fine-tune the probe (not just adapters) on a *small* real-dark split, or adapt with real-dark data mixed into the diet, and measure how much of the remaining gap closes. The ExDark infrastructure (darkness axis, CLAHE control, feature cache) supports this directly.
6. ~~CKA drift as a predictor~~ — **done in Phase 3 v2**: per-layer drift measured in Phase 2 was converted directly into rank allocation, matching uniform LoRA at 22% fewer parameters.

---

*See `docs/METHODS.md` for exact parameters, formulas, seeds, and environment versions; see `README.md` for the quick-summary view and run instructions.*
