# Paper Draft — Related Work & Positioning

*Draft v1.1 (2026-09-21). All citations verified via web search; §2.2 novelty boundary re-scoped after a dedicated prior-art search.*
*Status: standalone paper section; to be merged into the manuscript skeleton.*
*Numbers cited from our results match commit `7ede489` artifacts exactly.*

---

## 2. Related Work

Our work sits at the intersection of four literatures: corruption robustness of vision
transformers, frequency-domain analyses of ViT behavior, representation comparison via
centered kernel alignment (CKA), and parameter-efficient fine-tuning (PEFT) of frozen
foundation models. We position each below, and state explicitly what is confirmatory and
what is incremental.

### 2.1 Robustness claims for self-supervised ViTs

DINOv2's authors report strong out-of-distribution performance, including on
ImageNet-A/R/C/Sketch (Oquab et al., 2023). Our Phase 1 result is a *replication and
boundary-condition* of that claim, not a contradiction: under a synthetic photometric
axis (extreme low light, severity 0–5) largely absent from standard corruption
benchmarks, frozen DINOv2 ViT-S/14 with a linear readout degrades steeply and
non-uniformly (mean accuracy 0.913 → 0.083 across severities on CIFAR-10-derived
inputs). Standard benchmark suites (Hendrycks & Dietterich, 2019) established the
severity-graded evaluation paradigm we follow; our additions are protocol hygiene —
matched-noise evaluation, per-severity bootstrap seeds, and dataset-draw controls —
which we found necessary for stable layer-wise comparisons (§5 of this paper).

### 2.2 Frequency sensitivity of ViTs

That ViTs rely preferentially on low-frequency structure is well documented: Fourier
sensitivity analyses show naturally-trained models are robust at low frequencies and
fragile at high ones (Yin et al., 2019), and direct frequency-perspective studies find
ViTs exploit high-frequency content poorly relative to CNNs (Bai et al., 2022), with
high-frequency artifact sensitivity reported for ViTs generally (Paul & Chen, 2022),
with the roles of architecture and training recipe in corruption robustness dissected
at scale by Tian et al. (2025). Our Phase 2 high-pass ablation is a *clean replication* of this pattern on frozen
DINOv2 features. The incremental part is the *linkage*: in the same model, on the same
eval set, we connect the frequency axis to the depth axis — the corrupted-vs-clean CKA
profile is flat through early/mid blocks and drops sharply in blocks 10–11, i.e. the
frequency-fragile regime coincides with late-layer representation drift. To our
knowledge this within-model, within-eval-set connection has not been previously
reported for DINOv2 under photometric corruption. A dedicated prior-art search
(2026-09-21) found adjacent but distinct work that must be acknowledged: clean-vs-
corrupted CKA has been applied *descriptively* to a 3D medical foundation model
(arXiv:2608.06613); layer-wise robustness dynamics have been traced through depth
("robustness fading", arXiv:2608.04442); and corruption-induced distributional shift
has been localized to late processing stages via batch-norm statistics (Schneider et
al., 2020). None of these connect the frequency axis to the depth axis within one
model, and none turn the measurement into an adaptation-budget prescription. The
claim is therefore stated narrowly: *within-model, within-eval-set linkage of
frequency fragility to late-layer CKA drift, used prescriptively*, for which we found
no precedent.

### 2.3 CKA as a representation diagnostic

CKA was introduced for comparing representations within and across networks
(Kornblith et al., 2019) and used to characterize what makes ViT representations
distinct from CNNs (Raghu et al., 2021). Prior CKA robustness work mostly compares
*architectures*; the nearest use of clean-vs-corrupted CKA on a foundation model
(arXiv:2608.06613) is descriptive — it measures where correspondence breaks down but
stops there. We instead use CKA *longitudinally* — same model, clean vs. corrupted
inputs — as a per-layer drift measurement whose explicit purpose is to allocate an
intervention budget. This repositions CKA from a descriptive tool to a prescriptive one.

### 2.4 Diagnostic-guided parameter-efficient adaptation

LoRA (Hu et al., 2021) is the standard PEFT substrate; adaptive variants learn ranks
during training (AdaLoRA, Zhang et al. 2023; La-LoRA, Gu et al. 2026). Closest to our approach,
RepSAM (Chu et al., 2026) measures per-layer CKA domain gaps when adapting SAM to
robotic vision, finds the gap concentrated in *shallow* layers (CKA < 0.5), and
allocates LoRA ranks *before training*: highest rank (16) to shallowest layers, lowest
(4) to deep layers, achieving near-full-fine-tuning performance at reduced trainable
parameters.

Our Phase 3 applies the same diagnostic-first principle to a different failure mode —
photometric corruption of a frozen classification backbone — and obtains the *opposite*
allocation: our measured drift concentrates in blocks 10–11, so we weight rank
(3 → 8 by max-normalized CKA drop) toward late blocks. The resulting drift-weighted
arm (172,800 trainable params, 0.78%) matches both uniform-rank LoRA (221,184 params)
and full fine-tuning (22.06M params) on mean robustness across three seeds each
(0.825 ± 0.016 vs. 0.827 ± 0.012 vs. 0.831 ± 0.019; ranges fully overlapping),
with full FT showing the *worst* clean accuracy (seed mean 0.960 vs. 0.972–0.973) — a
forgetting cost LoRA avoids. Read jointly, RepSAM and our work suggest the general recipe
*measure representation drift, then allocate rank in proportion to it*, with the
allocation direction determined by where the measured gap actually lives (input-level
domain shift → shallow; task/photometric degradation → late). A useful negative
ablation supports this: rank-8 on late blocks only (0.25% of params) is clearly
insufficient (mean 0.686, sev-5 0.160), showing the drift profile identifies *where*
budget matters but some early/mid capacity remains necessary. Seed-replication with
error bars is reported in §6 [pending: 4 replication arms].

### 2.5 Training-time robustness interventions

FastDINOv2 (Zhang et al., 2025) improves DINOv2's corruption robustness *during
pretraining* via a spectral-domain curriculum, cutting training time to 62% of baseline
with matched-or-better robustness. This is complementary to our post-hoc approach:
where FastDINOv2 changes what the foundation model learns, we leave the backbone
untouched and adapt 0.78–0.99% of a frozen model's parameters. The two approaches have
different cost profiles (full pretraining access vs. adapter-only) and could plausibly
compose.

### 2.6 Synthetic corruption vs. real darkness

Low-light enhancement and recognition have a dedicated literature with standard
benchmarks, notably ExDark (Loh & Chan, 2019). Our ExDark experiments quantify the
sim-to-real gap for synthetic-darkness claims: (i) real darkness does not reproduce the
synthetic collapse — frozen DINOv2 achieves 0.725 on 7,363 real dark photographs with
a *flat* luminance-response curve across ExDark's brightness range; (ii) the standard
enhancement control (CLAHE) buys +0.001 — there is nothing luminance-limited to fix;
(iii) adapters trained *only* on synthetic darkness transfer modestly but measurably
(+1.9 points), versus +21 points on the synthetic severity axis itself. Together these
delimit what synthetic extreme-darkening benchmarks do and do not model, and caution
against reading corruption-benchmark gains as real-low-light gains.

### 2.7 Positioning statement

This is a **well-executed synthesis study with one incremental mechanism**. We do not
claim to discover that ViTs are frequency-fragile (§2.2) or that DINOv2 degrades under
corruption (§2.1). We claim: (1) a within-model linkage between frequency fragility and
late-layer drift; (2) CKA-allocated LoRA as a prescription that achieves full-FT parity
at <1% of trainable parameters on this failure mode, consistent with — and directionally
extending — RepSAM's diagnostic-first allocation; (3) a quantified sim-to-real boundary
for synthetic low-light robustness. Single small backbone, single dataset, synthetic
axis: these scope claims are stated, not hidden, in §8 (Limitations).

---

## References (verified unless marked)

- Zhang, J., Wang, J., Sun, Z., Zou, J., Balestriero, R., 2025 — FastDINOv2: Frequency
  Based Curriculum Learning Improves Robustness and Training Speed. arXiv:2507.03779
  (NeurIPS 2025).
- Bai, J., Yuan, L., Xia, S.-T., Yan, S., Li, Z., Liu, W., 2022 — Improving Vision
  Transformers by Revisiting High-frequency Components. arXiv:2204.00993.
- Tian, R., Wu, Z., Dai, Q., Goldblum, M., Hu, H., Jiang, Y.-G., 2025 — The Role of
  ViT Design and Training in Robustness to Common Corruptions. IEEE Transactions on
  Multimedia 27:1374–1385 (DOI 10.1109/TMM.2024.3521721).
- Schneider et al., 2020 — Improving Robustness Against Common Corruptions by
  Covariate Shift Adaptation. NeurIPS 2020.
- "Robustness fading" (shallow-robust → deep-fragile layer dynamics), 2026 —
  arXiv:2608.04442 (complete author list at bibliography stage).
- Clean-vs-corrupted CKA for 3D medical foundation models, 2026 — arXiv:2608.06613
  (complete title/author list at bibliography stage).
- Zhang, Q., Chen, M., Bukharin, A., Karampatziakis, N., He, P., Cheng, Y., Zhao, T.,
  2023 — AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning.
  ICLR 2023 (arXiv:2303.10512).
- Gu, J., Yuan, J., et al., 2026 — La-LoRA: Parameter-efficient fine-tuning with
  layer-wise adaptive low-rank adaptation. Neural Networks 194:108095.
- Chu et al., 2026 — RepSAM: Bridging Foundation Models to Robotic Vision via
  Representation-Guided Adaptation. arXiv:2605.25495.
- Oquab et al., 2023 — DINOv2: Learning Robust Visual Features without Supervision.
  arXiv:2304.07193.
- Hendrycks & Dietterich, 2019 — Benchmarking Neural Network Robustness to Common
  Corruptions and Perturbations. ICLR 2019.
- Yin et al., 2019 — A Fourier Perspective on Model Robustness in Computer Vision.
  NeurIPS 2019.
- Paul & Chen, 2022 — Vision Transformers Are Robust Learners. AAAI 2022.
- Kornblith, Norouzi, Lee, Hinton, 2019 — Similarity of Neural Network Representations
  Revisited. ICML 2019.
- Raghu et al., 2021 — Do Vision Transformers See Like Convolutional Neural Networks?
  NeurIPS 2021.
- Hu et al., 2021 — LoRA: Low-Rank Adaptation of Large Language Models.
  arXiv:2106.09685.
- Loh & Chan, 2019 — Getting to know low-light images with the Exclusively Dark
  dataset. Computer Vision and Image Understanding.
