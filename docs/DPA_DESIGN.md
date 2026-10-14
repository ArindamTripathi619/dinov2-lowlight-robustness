# DPA_DESIGN — Drift-Profiled Adaptation + Selective Statistic Recalibration

*Design doc for the methods paper (docs/ROADMAP.md §1 decision). Created 2026-10-03.*
*§1 is complete: profiler selection is settled by the Track 1 gate run (commit 5e7afe4)
and the prior-art searches (§1.3, log in §6). §2 records the Track 3 atlas run
(2026-10-14). §3–§5 remain scoped stubs pending the SSR arms (Track 4) and the
atlas-family allocation runs (Track 2 closed 2026-10-11 apart from the sublayer-resolved extension).
Claims in this doc are deliberately narrow — see PAPER_RELATED_WORK.md §2.7.*

---

## 1. DPA: drift-profiled adaptation, and the profiler choice

### 1.1 Method definition

**DPA** is a two-step, label-free adaptation recipe:

1. **PROFILE.** Run the frozen model once on the same images clean and once degraded
   (forward-only, no labels, no backprop). Record per-module activations via forward
   hooks, pool each output to a per-sample feature vector (spatial mean for conv maps,
   CLS token for transformer blocks), and measure per-module representational drift

   `β_m = 1 − linear_CKA(A_clean_m, A_degraded_m)`

   (biased estimator when matching historical artifacts; bias-corrected estimator
   reported alongside — METHODS §4). The result is a **drift profile** β over modules.
2. **ALLOCATE.** Distribute the adaptation budget across modules in proportion to the
   measured drift — LoRA rank per module for DPA's training arm
   (`r_m = round(r · β_m / max β)` in the current implementation), top-k module
   selection for DPA's test-time arm (SSR, §4).

Properties that do the work: the profile is **per (model, corruption family,
severity)**, so the allocation adapts to where the failure mode actually lives in that
model — it is measured, not assumed. The profiler needs only two forward passes over a
small image set plus n×d Gram arithmetic: on CPU, the entire Track 1 profiling run
(ResNet-50, n=1000, 2 corruptions) took ~4 minutes; the forward passes are the cost,
and they shrink to seconds on GPU-class hardware.

### 1.2 Profiler selection experiment (Track 1 gate) — outcome

The roadmap (§2 Track 1) pre-registered a cheaper-than-CKA profiler search with a
go/no-go gate: **Spearman ρ(proxy profile, CKA profile) ≥ 0.8 on ≥ 2 corruptions AND
top-k modules by proxy contain the top-k by CKA.**

Setup: torchvision ResNet-50 (IMAGENET1K_V1), 25 hooked modules (4 stage containers,
16 bottlenecks, 4 residual downsample projections, avgpool readout — stage containers
duplicate their last bottleneck by construction), CIFAR-10 test n=1000 seed 42,
severity 5, corruptions blur + low_light. Proxy candidates (forward-only, higher =
more drift): activation-energy drop ‖Δa‖²/‖a_clean‖², feature-mean shift
‖Δμ‖₂/‖μ‖₂, feature-covariance shift ‖ΔΣ‖_F/‖Σ‖_F, cosine drop 1 − mean cos(a_c, a_d).
Reference: 1 − CKA_unbiased. Harness: `run_drift_proxy.py`; artifacts of record in
`output/drift_proxy/` (commit 5e7afe4).

**Outcome: NO-GO** — no candidate clears the gate on both corruptions.

| proxy | ρ vs CKA (blur) | ρ vs CKA (low_light) | top-5 containment (best) |
|---|---|---|---|
| **cos_drop** | **0.752** | **0.676** | 0.80 (blur) |
| energy_drop | 0.679 | 0.661 | 0.60 / 0.60 |
| mean_shift | −0.078 | 0.032 | 0.00 |
| cov_shift | 0.077 | 0.074 | 0.00 |

A ViT-S/14 harness check (n=300, 12 blocks + attn/mlp sub-modules, CLS pooling;
`output/drift_proxy_dinov2_check/`) reaches the same verdict (best ρ ≈ 0.70) — the
shortfall is not ResNet-specific.

Interpretation (what the failure teaches):

- **Scale invariance is the dividing line.** CKA is invariant to isotropic rescaling
  of the representation; every candidate except cos_drop is scale-sensitive. Low-light
  shrinks activation amplitude broadly, which amplitude statistics register as drift —
  but the late-stage CKA collapse (layer4.2: drop 0.64 blur / 0.79 low_light) is
  *structural* (rotation/manifold damage), which amplitude statistics cannot rank.
  cos_drop, the only scale-invariant candidate, is best under blur — consistent with
  this account, but still under the bar.
- **mean_shift is structurally blind for blur**: exactly 0.0000 at every 1×1-conv
  downsample — Gaussian blur preserves DC, so per-channel dataset means pass through
  channel-mixing convolutions unchanged. A mechanistic, not statistical, failure.
- **cov_shift is noise-dominated**: every module's covariance is almost fully rebuilt
  (relative Frobenius shift 0.5–1.7), and its n=100 low-light ρ of 0.87 collapsed to
  0.07 at n=1000 — small-sample artifact, a useful reminder that profiler validation
  needs the n that the gate will be cited at.
- **The ResNet-50 profile itself is a finding**: strongly late-heavy under both
  corruptions (vs. layer1: 0.06/0.14) — the first CNN data point for Track 3's
  cross-architecture claim, obtained on CPU at zero GPU cost.

**Decision (pre-registered fallback): keep full CKA as the profiler.** CKA is itself
forward-only and already proven on ViT-S (Phase 2), so the paper claim weakens from
"cheap proxy beats CKA" to **"a single cheap profiling pass prescribes the budget"** —
two forward passes, no labels, no backprop. The gate result is design evidence, not a
dead end: §4's SSR ablation "proxy-vs-CKA profiler" is thereby resolved (CKA profiler;
the proxy candidates are recorded here as the negative result they are).

### 1.3 Prior-art adjacency for drift-profiled allocation (searched 2026-10-03, log §6)

The allocation-signal taxonomy, from the closest work outward:

| method | allocation signal | when measured | needs backprop? | relation to DPA |
|---|---|---|---|---|
| **RepSAM** (Chu et al., IJCAI-ECAI 2026, arXiv:2605.25495) | per-layer CKA domain gap (SAM → robotic vision) | before training | no | **same diagnostic-first principle**, different shift: input-level domain gap → shallow-heavy; our photometric degradation → late-heavy. The direction is an empirical variable — DPA measures it per model × shift |
| AdaLoRA (Zhang et al., ICLR 2023) | SVD-based importance of weight updates | during training | yes | training-time, iterative; DPA allocates before any training from the failure mode itself |
| La-LoRA (Gu et al., Neural Networks 2025) | layer-wise progressive rank adjustment | during training | yes | same contrast as AdaLoRA |
| IFCLoRA (Zhang et al., arXiv:2607.22251) | task-conditioned information-flow centrality (intervention tracing) fused with gradient sensitivity | before training (calibration set + tracing) | yes (gradients) | closest *pre-training* allocator; signal = task topology, not degradation-induced drift; targets task adaptation (GSM8K), not robustness |
| HALoRA (AAAI 2026); GEM (AAAI 2026) | hierarchical budget; entropy-guided layer-wise capacity | training-side | yes | same family; training-side signals |
| Surgical fine-tuning (Lee et al., ICLR 2023, arXiv:2210.11466) | a priori rule: which depth to tune depends on shift *type* (input perturbation → first layers; label shift → last) | n/a (no measurement) | yes | proves layer-*selectivity* matters; DPA replaces the type→depth rule with a per-model, per-corruption measurement. Noted tension: their covariate-shift result favors early layers while our photometric profiles are late-heavy on both a supervised CNN and a self-supervised ViT — exactly why "measure, don't assume" is the framework's point |
| Galichin et al. (EACL 2026 Findings) | feature-drift CKA, post-hoc analysis of what fine-tuning did | after training | n/a (analysis) | measures drift from adaptation; DPA measures drift from the degradation, before adapting, and uses it prescriptively |

**Novelty statement (narrow, per PAPER_RELATED_WORK §2.2/§2.4 conventions):** no prior
work found that (a) measures clean-vs-degraded representational drift of the *actual*
failure mode with a forward-only pass and (b) uses that measurement to allocate an
adaptation budget. RepSAM is the nearest neighbor (CKA-guided, pre-training) and must
be credited as the origin of the diagnostic-first principle; DPA's increment is the
shift-conditioned, model-conditioned, measured direction + the shared-profile design
(§1.4). If manuscript assembly surfaces anything closer, the claim narrows again.

### 1.4 Consequences for DPA design

- **Profiler of record: linear CKA** on pooled activations (spatial mean for conv,
  CLS for ViT blocks), biased estimator for artifact continuity, unbiased reported.
  The `run_drift_proxy.py` harness (module selection at depth ≤ 2 + pool leaves;
  4D/3D/2D pooling; fired-module filtering) is the model-agnostic profiling harness
  Tracks 2 and 3 reuse — validated on torchvision ResNet-50 and torch.hub ViT-S/14.
- **Granularity**: per-block/stage by default; the same hooks yield attn/mlp
  sub-modules for ViT families at no extra cost (Track 3 can raise `--max-depth`).
- **One measurement, two consumers**: β_m drives LoRA rank allocation (DPA training
  arm) *and* top-k module selection for SSR recalibration (§4). This coupling —
  diagnostic reused across both arms — is what makes the framework coherent rather
  than two methods.
- **Allocation-rule ablations — DONE at n=3 (2026-10-11, kernel `vitb-lora-abl`,
  RESEARCH §6.6):** r ∝ β^τ (τ=2.0) vs hard top-k (k=9) vs RepSAM-style inverted-β
  (falsification), budget-matched to the drift rule's Σr=73. All three tie uniform and
  r ∝ β_m within seed noise — **allocation shape is inert at this budget on
  ViT-B/low_light; coverage is the discriminator** (late-only stays clearly short).
  The inverted-β falsification *failed to falsify*: putting 18 rank units on the
  least-drifted block still matches uniform (+0.0013 ± 0.0057 paired). Nulls were
  already on record for ViT-S (grid parity) and ViT-B (v9, RESEARCH §6.5).
- **Profiling-set size**: n=1000 suffices (cov_shift's n=100 instability is a
  warning); profiling images may be unlabeled and drawn from the deployment-ish
  distribution — no labels consumed anywhere.

---

## 2. Cross-family drift atlas (Track 3 — RUN 2026-10-14)

**Run.** Kernel `arindamtripathi/drift-atlas-v4` (T4, 46 min wall, `fail=0`,
`ATLAS_COMPLETE`): ResNet-50, ConvNeXt-T, ViT-S/16, Swin-T, DeiT-S, DINOv2 ViT-B/14
on a shared ImageNet-val subset (n = 2,000, seed 42, `sautkin/imagenet1kvalid`),
low_light + jpeg × severities 1–5, forward-only CKA (the §1.2 gate verdict fixed the
profiler before this run). Plus a CIFAR-10 consistency leg (ResNet-50, ConvNeXt-T,
n = 1,000, sev 1/5) anchoring the atlas rows to the Track 1/2 image axis.
Artifacts: `output/profiles/<model>/` (10 CSVs + agreement reports per model),
atlas aggregation `output/atlas/atlas_summary.csv` + Fig. 1 candidate
`output/atlas/atlas_fig1.png` (`tools/build_atlas.py`).

**Headline statistic — late-half drift mass @ sev 5** (0.5 = uniform;
>0.5 = drift concentrated in the deeper half of profiled modules):

| model | family | low_light | jpeg | worst module (low_light) |
|---|---|---|---|---|
| Swin-T | hybrid | 0.664 | **0.823** | layers.1.downsample |
| ViT-S/16 | ViT | 0.615 | 0.762 | blocks.10.mlp (drop 0.98) |
| ResNet-50 | CNN | 0.536 | 0.725 | layer4 |
| DeiT-S | ViT | 0.519 | 0.621 | blocks.7.mlp (mid-depth) |
| ConvNeXt-T | CNN | 0.587 | 0.489 | head / stages.1 |
| DINOv2 ViT-B/14 | ViT (self-sup.) | **0.380** | **0.429** | blocks.1.mlp (drop 0.82) |

**The cross-architecture answer.** "Do CNNs drift early and ViTs late?" — as posed,
**no**: late-heavy drift is the *majority pattern across all three families* (5/6
models ≥ 0.52 on low_light), and Swin-T (the hybrid) is the *most* late-heavy, not
in-between. The discriminating axis is not the architecture family but something
closer to the **pretraining objective**: the one model that flips — early-heavy under
both corruptions, patch_embed drop 0.38, blocks.0 drop 0.54, worst at blocks.1.mlp —
is the self-supervised DINOv2, whose supervised ViT cousins (ViT-S, DeiT-S) both sit
late. This also means Track 2's late-only conclusions and the §1.2 ResNet-50 profile
are *not* universal defaults: allocation and recalibration scopes must read the atlas
per model, which is exactly what DPA/SSR design (§1.4, §4) already prescribes.

**Severity monotonicity** holds on all 6 × 2 (model × corruption) series — mean CKA
drop rises strictly with severity (e.g. ResNet-50 low_light 0.013 → 0.32); the
atlas is not a noise artifact of n = 2,000.

**Gate re-check (free bonus).** The §1.2 proxy NO-GO replicates on all 6: best
low_light ρ = 0.693 (DINOv2 cov_shift), jpeg ρ reaches 0.91 (Swin cov_shift) — the
familiar jpeg-correlates / low_light-doesn't signature, now spanning six
architectures. The fallback (full CKA profiler) stands with six more data points.

**Limitations of this atlas (stated, not hidden):** n = 2,000 not 10k (cost control;
rank stability at n = 1k was already established in Track 2); no state-space/Mamba
row (timm ≥ 1.0.3 ships no vmamba without the separate `ssm` package — parked);
single-seed. Dataset axis matters — ResNet-50's CIFAR leg (0.718) is later than its
ImageNet leg (0.536) — so cross-dataset profile comparisons are qualitative.

---

## 3. Allocation rule (Tracks 2–3 — pending)

Current rule `r_m = round(r · β_m / max β)` produced grid parity with uniform and
full-FT at 0.78% params (ViT-S, 4 corruptions, 3 seeds). The first ViT-B point landed
2026-10-10 (v9, n=3: parity with uniform at 24% fewer params — RESEARCH §6.5, the same
parity-at-lower-cost picture ViT-S showed); the atlas families (Track 3) generalize it
further. The §1.4 shape ablations closed 2026-10-11 with a null result that *sharpens*
the claim (RESEARCH §6.6): at ~0.4% budget the loss surface is flat across allocation
shapes — sharpened, top-k, even inverted-β all tie uniform — so the defensible value of
the rule is the param saving at parity, and "our shape wins" is off the table; whether
shape matters at tighter budgets is the open shape×budget question for Track 3. Routed adapters (routing derived from β) stay parked
as the end-game ablation per ROADMAP §5.

---

## 4. SSR: selective statistic recalibration (Track 4 — pending)

**Method.** Store clean per-module channel statistics (μ, σ) once per model. At test
time, estimate the shift from a few unlabeled batches and recalibrate **only the top-k
drift modules** (by β_m) via per-channel affine or norm-stat replacement. The gate is
0 on clean inputs — exactly-identity guarantee: with no estimated shift, SSR is
bit-identical to the frozen model.

**Design invariants** (what makes it SSR and not generic TTA): module selection comes
from the same measured profile β as DPA's allocation (§1.4, one measurement, two
consumers); recalibration is label-free and (in the stat-replacement variant)
backprop-free; selection is per (model, corruption family), i.e. drift-profiled.

**Prior-art adjacency (searched 2026-10-03, log §6):**

| method | mechanism | selection | relation to SSR |
|---|---|---|---|
| TENT (Wang et al., ICLR 2021) | entropy minimization on BN affine params, online | adapt-all (no selection) | SSR needs no backprop, no loss, selects modules by measured drift |
| Schneider et al. (NeurIPS 2020) | replace BN stats from test batches, per corruption | adapt-all | nearest backprop-free neighbor; SSR adds drift-profiled top-k selection + clean-gate |
| TTN (Lim et al., ICLR 2023) | domain-shift-aware mixing of source/test stats | all BN, fixed ratio, post-training calibration | SSR's ratio is estimated from the unlabeled batches, modules are drift-selected, no post-training |
| Nado et al. (2020) | prediction-time BN stats | all | same contrast as Schneider |
| Channel-Selective Normalization (arXiv:2402.04958) | selects channels for label shift | channel-level, label-shift-specific | SSR selects modules for covariate-style corruption, label-free |
| AQR (Mehrbod et al., arXiv:2511.03148, Nov 2025) | piecewise-linear quantile transform of norm-layer pre-activations onto stored source percentiles; BN/GN/LN-agnostic, backprop-free | static heuristic (all layers / top half, hand-tuned per model — never measured) | **closest mechanism neighbor**: same "store source stats once, recalibrate label-free at test time" design; differs in statistic (quantiles vs moments), has no drift-based selection and no clean gate; its own App. D shows *where* you recalibrate matters enormously (top-half 63.7 vs all-blocks 36.25, ViT-B/ImageNet) yet chooses by heuristic |

| GALA, layer selection for TTA (Sahoo et al., AAAI 2025, arXiv:2404.03784) | gradient-aligned layer selection for *parameter* updates, plugged into existing TTA losses | per-sample gradient alignment (needs backprop + a TTA loss) | second published "where you adapt decides the outcome" evidence (with AQR App. D); selects by gradients at adaptation time, not by a forward-only measured drift profile, and adapts parameters instead of recalibrating statistics; no clean gate |

**No prior work found that selects which modules to recalibrate from a measured
drift profile, or that guarantees identity on clean inputs** (closest selection
neighbor: GALA — gradient-based, loss-coupled, per-sample). SSR's ablations from
ROADMAP Track 4 update accordingly: top-k vs all and gate on/off stay; the
"proxy-vs-CKA profiler" ablation is resolved by the §1.2 gate (CKA profiler).
Baselines stay: TENT, BN-adapt (Schneider-style adapt-all), enhancement front-ends
(ZeroDCE/CLAHE), augmentation-only FT.

---

## 5. Evaluation plan (Tracks 2–5 — pending)

Claims to close, in dependency order: (a) ViT-B profile + drift-weighted LoRA
(Track 2, Kaggle v8/v9 + ablations) — **closed at n=3** (RESEARCH §6.5; shape
ablations §6.6) — cross-architecture pair ViT-S/ViT-B + the ViT-B generality item;
(b) atlas across 5+ families (Track 3) — the "CNNs early vs ViTs late" question
phrased as a measurement, with ResNet-50 already late-heavy (§1.2); (c) SSR arms vs
TENT/BN-adapt on the atlas models (Track 4); (d) real-dark dense eval (Track 5).
Protocol parity rules carry over: matched-noise rng 1000+severity, probe-on-clean
severities, seed 42 default, severity-5 profiles as the allocation inputs.

---

## 6. Prior-art search log

All searches 2026-10-03, run before writing §1/§4 claims per ROADMAP §6.

**Drift/rank-allocation adjacency:**
- RepSAM — arXiv:2605.25495 + IJCAI-ECAI 2026 preprint (AIR44). Confirmed:
  "theoretically grounded CKA-guided rank allocation", ranks fixed before training,
  shallow-heavy. Credited as the diagnostic-first origin.
- IFCLoRA — arXiv:2607.22251 (Jul 2026). Pre-fine-tuning allocation via intervention
  tracing + IFC centrality + gradient sensitivity; its own abstract taxonomizes the
  family as "local gradient, activation, or matrix statistics collected before or
  during fine-tuning" — no drift-conditioned allocator in the family.
- Also screened: HALoRA (AAAI 2026), GEM (AAAI 2026), La-LoRA, LoRA-FA/layer-wise
  importance (EMNLP 2024 Findings), Surgical fine-tuning (ICLR 2023), SAFT (ECCV 2024),
  Galichin et al. (EACL 2026 Findings).

**Recalibration adjacency:**
- TENT — arXiv:2006.10726 (ICLR 2021 spotlight): "estimates normalization statistics
  and optimizes channel-wise affine transformations" online; backprop, adapt-all.
- Schneider et al. (NeurIPS 2020) — BN-stat replacement, per-corruption, adapt-all
  (already cited, PAPER_RELATED_WORK §2.2).
- TTN (ICLR 2023), Nado et al. (2020), Channel-Selective Normalization
  (arXiv:2402.04958), Buffer layers for TTA (NeurIPS 2025) — screened; none select
  modules by measured drift, none guarantee identity on clean inputs.

**Verdict:** the §1.3 and §4 novelty statements hold as written (narrow forms).
At manuscript assembly, fold IFCLoRA + Surgical fine-tuning into PAPER_RELATED_WORK
§2.4, and TTN/Nado/Buffer-layers into the SSR baseline paragraph.

**Second round — SSR/Track-4 searches (2026-10-13, run before writing
`SSR_PLAN.md` claims per this log's convention):**

*Moment/statistic recalibration adjacency:*
- AdaBN (Li et al., Pattern Recognition 2018; arXiv:1603.04779) — offline domain
  adaptation by BN-stat replacement; adapt-all, BN-only. The mechanism family's
  origin; Schneider/TTN already carry its TTA version.
- **AQR (arXiv:2511.03148, full text read 2026-10-13) — closest neighbor found.**
  Stores source percentiles once (setup phase), applies a per-channel piecewise-
  linear quantile transform at test time, backprop-free, label-free, works on
  BN/GN/LN — i.e. it already delivered the "no BatchNorm needed" property.
  Differences that preserve §4 as written: (a) selection = static heuristic
  (all / top-half layers, hand-tuned per architecture), never a measured profile;
  (b) always applies — no clean-input identity gate; (c) quantile transport, not
  moment matching. Its App. D (top-half 63.7 vs all-blocks 36.25 on ViT-B/ImageNet)
  is published evidence that *where* to recalibrate is the decisive choice —
  motivating SSR's C1 from a result we did not have to produce ourselves.
- DPTA / "Unraveling Batch Normalization" (AAAI 2024, arXiv:2312.09486) —
  inexact mini-batch BN statistics at TTA; adapt-all, BN-only.
- ESBN (IJCAI 2025) — BN shift estimation for source-free DA; adapt-all, BN-only.
- Neuron editing (Amodio et al., NeurIPS 2019) — percentile-based activation
  transforms, but a data-editing method, not TTA (surfaced via AQR's lineage).
- Also screened: SAR (norm-agnostic but entropy + backprop), MEMO (augmentation
  + per-sample backprop — the "augmentation-only" family's TTA form).

*Shift-detection / when-to-adapt gating adjacency:*
- Confidence-driven adaptation (Mummadi et al., arXiv:2106.14999) — confidence
  signals steer adaptation, but always-on; not a clean-calibrated distance gate.
- Test-Time Selective Adaptation (ICML 2025) — selection at *modality* level
  under multi-modal shift; not module-level, no identity guarantee.
- TTA surveys (Liang et al., IJCV 2025; Wang et al. 2023) — no mechanism found
  that gates adaptation on a clean-calibrated statistical distance with an
  exactly-identity guarantee on clean inputs.

*Allocation side (DPA §1.3 family; logged for completeness):*
- LS-LoRA (arXiv:2610.11620, Oct 2026) — layer-selective LoRA placement by
  input–output similarity (capability retention). Similarity, not degradation-
  induced drift; training-side. RepSAM-adjacent; does not touch §1.3 but belongs
  in PAPER_RELATED_WORK §2.4 at assembly.

**Verdict (2026-10-13):** §4 holds as written — AQR, the closest neighbor,
selects by static heuristic (never by measured drift) and applies always (no
gate). One narrowing: cross-norm-architecture applicability is no longer novel
per se (AQR demonstrated it) — that aspect becomes a *demonstration requirement*
for SSR (C4), not a claim. Track-4 consequences: (a) **AQR joins the baseline
list** (strongest published backprop-free TTA; also supplies the published
motivation for measured selection); (b) moment-vs-quantile becomes a mechanism
ablation candidate.

**Third round — SSR_PLAN §10 parallel searches (2026-10-14, run while the
Track 3 v4 kernel executed):**

*C1 adjacency (layer/module selection for TTA):*
- **GALA — "A Layer Selection Approach to Test Time Adaptation" (Sahoo et al.,
  AAAI 2025, arXiv:2404.03784).** Selects which layers to update per sample by
  *gradient alignment* (filters noisy-gradient samples; plugs into existing TTA
  losses). Needs backprop + a loss; adapts parameters, not statistics; never
  measures representational drift. Consequence: C1's motivation now has two
  published supports (AQR App. D + GALA), while §4's novelty line survives —
  selection by a forward-only measured drift profile remains unclaimed. Fold
  into PAPER_RELATED_WORK at assembly (with IFCLoRA / Surgical fine-tuning).
- Also screened: "Layerwise early stopping for TTA" (surfaced via the
  awesome-tta index); Activation-Guided Layer Selection for LoRA (Information
  2026) — training-side, activation-similarity heuristic, §1.3-family.

*Backprop-free TTA mechanism refresh:*
- **ADAPT — "Backpropagation-Free Test-Time Adaptation via Probabilistic
  Gaussian Alignment" (Zhang et al., NeurIPS 2025).** Closed-form class-conditional
  Gaussian inference over features (class means + shared covariance), CLIP priors
  + history bank. Prediction-level likelihood correction — not module-output
  recalibration; screened, not a baseline (CLIP dependency + different
  intervention point). Buffer layers (NeurIPS 2025) already screened in round 1.
- TENT re-confirmed (arXiv:2006.10726): "estimates normalization statistics and
  optimizes channel-wise affine transformations" — the wording SSR_PLAN §0.3's
  TENT-LN labeling leans on (same mechanism, applied to the norm the model has).

*C2 gate re-check (fresh vocabulary: trigger / bypass / open-world / when-to-*
*adapt):* Mummadi confidence-driven (already screened), Zhou et al. open-world
TTA (ICLR 2023), Shift-ACT class-wise thresholds (Neurocomputing 2026), TTA
surveys — still no clean-calibrated statistical-distance gate with an
exactly-identity guarantee.

**Verdict (2026-10-14):** §4 holds as written — no narrowing, no claim edits;
GALA and ADAPT enter the log (GALA also enters the §4 adjacency table).
