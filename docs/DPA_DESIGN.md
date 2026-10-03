# DPA_DESIGN — Drift-Profiled Adaptation + Selective Statistic Recalibration

*Design doc for the methods paper (docs/ROADMAP.md §1 decision). Created 2026-10-03.*
*§1 is complete: profiler selection is settled by the Track 1 gate run (commit 5e7afe4)
and the prior-art searches (§1.3, log in §6). §2–§5 are scoped stubs pending Tracks 2–4.
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
- **Allocation-rule ablations to run** (Track 2 v9 / Track 3): r ∝ β_m (current) vs
  r ∝ β^τ (sharpen) vs hard top-k vs uniform (null) — the uniform null already exists
  (grid parity result); RepSAM-style inverted-β is the natural falsification arm.
- **Profiling-set size**: n=1000 suffices (cov_shift's n=100 instability is a
  warning); profiling images may be unlabeled and drawn from the deployment-ish
  distribution — no labels consumed anywhere.

---

## 2. Cross-family drift atlas (Track 3 — pending)

Planned: per-stage/per-block β profiles for ResNet-50, ConvNeXt-T, ViT-S/16, Swin-T,
DeiT-S (+ ViT-B/14 from Track 2) on a shared ImageNet-val subset under low_light +
jpeg severity pairs. Harness = §1.4 (`run_drift_proxy.py` parameterized by model;
timm-backed loaders land with this track). Output: `output/profiles/<model>_<corr>.csv`
+ the paper's Fig. 1. The gate outcome (§1.2) fixes the measurement: CKA profiles,
not proxies. ResNet-50 late-heavy rows already exist (§1.2) as the first atlas entry.

---

## 3. Allocation rule (Tracks 2–3 — pending)

Current rule `r_m = round(r · β_m / max β)` produced grid parity with uniform and
full-FT at 0.78% params (ViT-S, 4 corruptions, 3 seeds). ViT-B (Track 2 v9) and the
atlas families (Track 3) generalize it; §1.4 lists the ablation set (β^τ, top-k,
inverted-β falsification). Routed adapters (routing derived from β) stay parked as the
end-game ablation per ROADMAP §5.

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

**No prior work found that selects which modules to recalibrate from a measured
drift profile, or that guarantees identity on clean inputs.** SSR's ablations from
ROADMAP Track 4 update accordingly: top-k vs all and gate on/off stay; the
"proxy-vs-CKA profiler" ablation is resolved by the §1.2 gate (CKA profiler).
Baselines stay: TENT, BN-adapt (Schneider-style adapt-all), enhancement front-ends
(ZeroDCE/CLAHE), augmentation-only FT.

---

## 5. Evaluation plan (Tracks 2–5 — pending)

Claims to close, in dependency order: (a) ViT-B profile + drift-weighted LoRA
(Track 2, Kaggle v8/v9) — cross-architecture pair ViT-S/ViT-B + the ViT-B generality
item; (b) atlas across 5+ families (Track 3) — the "CNNs early vs ViTs late" question
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
