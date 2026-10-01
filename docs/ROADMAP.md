# Research Roadmap — From Corruption Grid to a Methods Paper

*Created 2026-10-01. Synthesizes the advisor's three memos (candidate directions,
architecture-agnostic directions, free-tier compute plan) with the current state of
this repo. All GPU work assumes free-tier Colab/Kaggle (METHODS §11); total GPU cost
of the recommended path fits in ~2–3 weeks of Kaggle quota (30 h/week).*


---

## 0. Where we stand (assets on the table)

- **Proven diagnostic**: per-module CKA drift profiles (ViT-S, 4 corruptions; biased +
  unbiased estimator agreement; CPU↔T4 determinism to 1e-6).
- **Proven allocation**: drift-weighted LoRA = uniform at 0.78% params, 3 seeds; big
  wins on frequency-destroying corruptions (jpeg +19.3, blur +17.7, low_light +22);
  graceful degradation on contrast (the rule is honest).
- **Proven decomposition**: readout staleness is real (+25.3 pp @sev4) but features
  are the bottleneck (plateau 0.377 vs 0.913 clean).
- **The gap the advisor confirmed**: real darkness barely hurts frozen DINOv2 on
  classification (ExDark flat, CLAHE +0.001) → the *problem* needs real benchmarks
  where darkness hurts (ACDC night, Dark Zurich, DarkFace/BDD-night, detection), and
  the *scale* needs more model families (not just one ViT-S on CIFAR-10).
- In flight: **PR #1 to upstream** (arghya's review), **blur uniform re-eval** (CPU,
  running), ViT-B campaign not started.

**Decision (advisor's two rankings converge on this):** the paper is **Drift-Profiled
Adaptation (DPA) + Selective Statistic Recalibration (SSR)** — the "drift-profiled,
label-free adaptation framework". Illumination-Routed Adapters (advisor's other #1)
is absorbed as the **end-game ablation** of DPA rather than a separate paper: routing
derived from the layer-drift diagnostic *is* DPA applied to adapter routing.
Illumination Registers and the Canonicalizer are parked (Archive, §5).

---

## 1. Priorities and rationale (what order, and why)

| # | Track | Why this position |
|---|-------|-------------------|
| 0 | Close sessionE blur arms; PR #1 follow-through | hygiene: the grid must be complete and the integration recorded before new claims |
| 1 | **Drift-proxy pipeline on ResNet-50** (advisor's "suggested next step") | **the whole paper stands or falls here**, for ~1 GPU-session; validates "cheap proxy ≈ CKA" outside ViT |
| 2 | **ViT-B drift-weighted LoRA** (kernel v8/v9) | the one remaining novelty claim from the *current* paper; also the ViT-family data point for the cross-architecture claim |
| 3 | **Cross-family drift profiles** (6 models) | the reviewer-ungettable result: *shape* of drift profiles across CNN/ViT/hybrid/state-space |
| 4 | **SSR: selective test-time recalibration** | label-free, inference-only, fits free tier; combines with DPA into the framework paper |
| 5 | **Real-darkness dense-task eval** (ACDC/Dark Zurich) | answers the "synthetic problem" objection on the benchmarks where darkness actually hurts |
| 6 | Manuscript assembly | after 1–5 produce the three claim pillars (cross-family, selective, real-dark) |
| A | Archive: routed adapters (as ablation), registers, canonicalizer | revisit only after the framework paper is out |


---

## 2. Track details

### Track 0 — Hygiene (this week, mostly waiting on others)
- [x] Blur re-evals done (`run_blur_adapter_eval.py`, n=1000): drift **0.526 → 0.703 (+17.7)**,
  uniform **0.526 → 0.698 (+17.2)**; original columns reproduce Phase-1 exactly
  (protocol comparability certificate). Folded into METHODS §7.6 / RESEARCH §6.3 / README.
- [x] Blur uniform re-eval done → both arms folded into METHODS §7.6 / RESEARCH §6.3 /
  README; committed and pushed.
- [ ] PR #1: send arghya the link + DIVERGENCE_REPORT; offer the deterministic-noise
  reconciliation as a follow-up PR on the merged base.
- [ ] After merge: rebase cleanup (fast-forward `main` to the rebased lineage), adopt
  upstream `configs/` manifest.

### Track 1 — Drift-proxy feasibility (the gate; ~1 GPU-session + CPU analysis)
**Goal:** a drift proxy *much cheaper than CKA* that predicts where adaptation pays off,
agreement-checked against CKA on a model with a different architecture (ResNet-50).

- Proxy candidates (all forward-only, no backprop): per-module activation-energy drop
  (‖Δact‖² / ‖act‖² clean→degraded), per-module feature-statistics shift (mean/cov
  distance), cosine-of-activations (cheapest possible CKA relative).
- **Go/no-go gate:** rank correlation (Spearman) between proxy profile and CKA profile
  ≥ 0.8 on ≥ 2 corruptions; and top-k modules by proxy contain the top-k by CKA.
- Deliverable: `run_drift_proxy.py` (hooks on all modules, proxy + CKA + agreement
  report) + `docs/DPA_DESIGN.md` §1.
- Fallback if it fails: keep full CKA as the profiler (it is forward-only anyway and
  already proven); the paper claim weakens to "single cheap profiling pass" instead of
  "cheap proxy". **The paper does not die here.**

### Track 2 — ViT-B drift-weighted LoRA (kernels v8 + v9; ~2–4 GPU-hours)
- v8: ViT-B/14 CKA drift profile under low_light + jpeg (forward hooks on 12 blocks,
  86M params; fits T4 easily for forward passes).
- v9: LoRA arms on ViT-B: uniform r8 vs drift-weighted (ViT-B-allocated ranks) vs
  late-only, seed 42, cifar10-python dataset mount reused.
- Claims it closes: (a) the current paper's ViT-B generality item; (b) first
  cross-architecture profile pair (ViT-S vs ViT-B).
- Fold into RESEARCH §6 + the future framework paper as the ViT data point.

### Track 3 — Cross-family drift profiles (the centerpiece; ~5–10 GPU-hours)
**Models (timm, all ≤ 90M):** ResNet-50, ConvNeXt-Tiny, ViT-S/16, Swin-Tiny, DeiT-S
(+ ViT-B/14 from Track 2; + one state-space/Mamba vision model if accessible).
**Protocol:** same ImageNet-val 10k subset (cached fp16 features), synthetic
low_light/jpeg severity pairs, per-stage/per-block CKA + proxy profiles.
- **The finding reviewers cannot get elsewhere:** do CNNs drift early and ViTs late?
  Hybrid/state-space in between? This is the cross-architecture claim.
- Output: the profile atlas (`output/profiles/<model>_<corruption>.csv`) +
  `docs/DPA_DESIGN.md` §2 + the paper's Fig. 1.
- Cost control: forward-only, cached features, one Kaggle session per 2 models.

### Track 4 — SSR: selective statistic recalibration (label-free; ~5–8 GPU-hours)
- Store clean per-module channel statistics (μ, σ) once per model.
- At test time: estimate illumination shift from a few unlabeled batches → recalibrate
  **only top-k drift modules** (per-channel affine or norm-stat replacement); gate =
  0 on clean inputs (exactly-identity guarantee).
- Baselines: TENT, BN-adapt (adapt-all), enhancement front-ends (ZeroDCE/CLAHE),
  augmentation-only FT. Ablations: proxy-vs-CKA profiler, top-k vs all, gate on/off.
- This is the cheapest method arm and the one that makes the framework "label-free".

### Track 5 — Real-darkness dense eval (the credibility arm; ~3–5 GPU-hours)
- ACDC night + Dark Zurich (segmentation, eval-only with a frozen SegFormer/DeepLab
  head + SSR recalibration), DarkFace/ExDark-detection (detection, norm-layer-only).
- Datasets hosted on Kaggle where possible (no download quota pain; check licenses).
- This directly answers "your problem is synthetic" — the advisor's #1 acceptance risk.

### Track 6 — Manuscript
- Framing: **"Drift-Profiled, Label-Free Adaptation: where representation damage
  lives determines where adaptation pays"** — diagnostic (profiles) → allocation
  (LoRA/routing) → label-free repair (SSR) → cross-architecture atlas → real-dark.
- Target: CVPR/ICCV main; fallbacks: UG2+/NTIRE workshops, journal. Advisor's honest
  caveat stands: breadth + seeds + real benchmarks substitute for scale.
- Assemble from: RESEARCH.md (narrative), DPA_DESIGN.md (method), PAPER_RELATED_WORK.md
  (related work, re-scope after prior-art searches), upstream's paper/ skeleton post-merge.

---

## 3. Execution order and dependencies

```
Track 0 (blur done → uniform → docs+push)   ──┐
Track 1 (drift proxy, ResNet-50)  ── gate ────┤
Track 2 (ViT-B v8/v9, Kaggle)                 ├──→ Track 3 (atlas) ──→ Track 4 (SSR)
                                              │                            │
                                              └── PR #1 merge ──────────→ Track 5 (ACDC/DarkZurich)
                                                                               │
                                                                        Track 6 (manuscript)
```

- **Track 1 gates Track 3's proxy claim only** (not the atlas — CKA profiles can start
  immediately). Run Track 1 *concurrently* with Track 2 (proxy work is CPU/1-session).
- Tracks 2 and 3 share the profiling harness — build it once (`run_drift_profile.py`,
  model-agnostic via timm), parameterize by model.
- Track 4 needs Track 3's profiles (top-k selection). Track 5 needs Track 4's method.
- Two Kaggle sessions/week cadence: one profiling session + one method session.

---

## 4. GPU / compute budget (free tier, per advisor's table)

| Block | Cost | Sessions (30 h/wk quota) |
|---|---|---|
| ViT-B profile + LoRA arms (Track 2) | 2–4 GPU-h | 1 |
| Cross-family profiles, 6 models (Track 3) | 5–10 GPU-h | 1–2 |
| SSR arms, 6 models × 3 datasets × 3 seeds (Track 4) | 5–8 GPU-h | 2 |
| Dense-task eval (Track 5) | 3–5 GPU-h | 1 |
| **Total** | **~15–27 GPU-h** | **~2–3 weeks** |

Compute tricks (from the memo, adopt wholesale): ImageNet-val subsets (10–20k) with
cached fp16 features; `torch.autocast` + `channels_last`; adapt only top-k modules
(short backward); checkpoint to `/kaggle/working` every N steps; Kaggle for long jobs,
Colab for iteration; multi-seed jobs as separate short sessions emitting JSONs.

---

## 5. Archive (explicitly parked, with reasons)

- **Illumination-Routed Adapters** — not discarded: it is DPA's allocation applied to
  *routing*; becomes the end-game ablation ("routing vs uniform vs allocation-by-rank")
  once DPA/SSR results exist. Prior-art search required before any novelty claim
  (Mixture-of-LoRA / input-conditional adapters are adjacent).
- **Illumination Registers** — highest novelty, highest risk (needs pretraining-scale
  resources; distinctness from artifact-sink registers unproven). Revisit post-paper.
- **Illumination Canonicalizer** — deprioritized by the compute memo (backprop through
  the frozen model × many model pairs = most expensive path). The cross-model transfer
  hook is the only part worth revisiting.

---

## 6. Immediate next actions (this week)

1. Finish blur uniform re-eval → docs + commit + push (Track 0).
2. `run_drift_proxy.py` on ResNet-50 (Track 1) — the go/no-go gate; CPU-able for a
   first pass with CIFAR-10-C-style degradations, ImageNet-subset for the real pass.
3. Build v8 kernel (ViT-B CKA profile) and queue it on Kaggle (Track 2) — independent
   of #2, uses the established kernel-builder pipeline.
4. Prior-art searches (advisor offered): drift-proxy allocation (RepSAM/AdaLoRA
   adjacency), selective test-time recalibration (TENT adjacency). Do these *before*
   writing DPA_DESIGN claims.
5. Send PR #1 to arghya (his review gates the merge, not the research).
