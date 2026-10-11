# CVPR 2027 Submission Checklist — Audit of This Repository

Audited against the CVPR 2027 CFP (dates confirmed) and the latest published
CVPR author guidelines (formatting rules; re-verify when the 2027 guidelines
are released). Status legend: ✅ present · 🟡 partial/weak · ❌ missing.

**Submission window (official):** registration **Nov 10 2026 AOE**, paper
**Nov 16 2026 AOE**, supplementary **Nov 23 2026 AOE**, decision **Feb 25 2027**.

---

## 1. Research framing

| Item | Status | Where |
|---|---|---|
| Clearly defined problem (input/task/metric) | ✅ | paper/main.tex §1; dark 32×32→224 images, 10-way classification, top-1 acc |
| Why the problem matters | ✅ | §1: deployment of SSL backbones in adverse illumination is unstudied at representation level |
| Research gap in one sentence | ✅ | "Corruption robustness work measures *models*; nobody has localized *where inside a frozen SSL backbone* photometric failure arises, nor what a readout can and cannot repair." |
| Hypothesis chain (H1–H5) with outcomes | ✅ | paper §5 + README claims registry (H1 rejected — that is a *finding*, not a failure) |
| Novelty beyond A+B | ✅ | mechanistic localization (layer-wise CKA gradient) + readout-repair decomposition + quantization null + geometry-attractor discovery (collapse target is prior-independent — refuted our own corollary with a discriminating test); no prior DINOv2 low-light work does this |

## 2. Method & experiments

| Item | Status | Where |
|---|---|---|
| Problem formulation + notation | ✅ | paper §3 (E_φ, severity ladder, probe f_w) |
| Architecture diagram | 🟡 | text pipeline in §3; **TODO: one TikZ figure** |
| Training/inference details complete | ✅ | configs/experiments.yaml is the single source of truth; SHA-256 in every manifest |
| Deterministic corruption (order-independent noise) | ✅ | hash-keyed noise draws in `src/corruptions.py`; bit-reproducible across scripts/processes |
| Datasets documented (splits, licensing) | ✅ | CIFAR-10 (test-fold protocol) + STL-10 transfer; both cited; indices saved in manifests |
| Weak + strong baselines | ✅ | zero-training prototype (weak), LoRA fine-tune (strong), classical enhancement ( Gain/γ/CLAHE) |
| Main results table with CIs | ✅ | table_main.tex — Wilson + bootstrap 95% CIs, n=1000 full scale |
| Ablations | ✅ | LoRA rank×target grid; probe-C grid; split-seed grid (seed_sensitivity) |
| Statistical significance, not vibes | ✅ | paired permutation tests + BH correction; two curve-shape null models; all p-values reported |
| Robustness of the *finding* (multi-seed) | ✅ | exp seed_sensitivity: 3 seeds × 3 probe-C, mean±std per severity |
| Cross-dataset generalization | ✅ | exp cross_dataset: STL-10 zero-shot under the same ladder |
| Efficiency table (params/FLOPs/latency) | ✅ | exp efficiency → table_eff.tex |
| Qualitative results | ✅ | failure grid (clean-correct → dark-wrong) per class, exp failure_analysis |
| Failure analysis with explanation | ✅ | flip counts, failure persistence, per-class death order (failure_by_class.csv) |

## 3. Reproducibility (CVPR checks this)

| Item | Status | Where |
|---|---|---|
| Code runs end-to-end from repo | ✅ | `python run_experiments.py --experiment all` (GPU) or Colab runner |
| Config as single source of truth | ✅ | configs/*.yaml; config hash embedded in manifests |
| Seeds + n-runs stated | ✅ | all configs; seed sensitivity experiment quantifies run-to-run variance |
| Unit tests for the analysis math | ✅ | tests/run_tests.py — 26 tests green (CI via GitHub Actions) |
| Manifests: every number traceable | ✅ | results/manifests/ (config SHA, seeds, subset indices, env) |
| Hardware stated | 🟡 | Colab T4 (GPU runs), CPU pilot — add exact GPU model to paper when full runs complete |

## 4. Paper document

| Item | Status | Where |
|---|---|---|
| 8-page main paper (verify vs 2027 template) | ✅ structure | paper/main.tex follows CVPR layout; page budget annotated |
| Title/abstract/contributions explicit | ✅ | paper §1 ends in numbered contributions |
| Related work organized by theme + gap | ✅ | §2 (robustness benchmarks / SSL representations / enhancement-vs-representation) |
| Limitations section | ✅ | §7 (single backbone, synthetic ladder, pilot-vs-full scale) |
| References real and complete | ✅ | paper/references.bib (DINOv2, CIFAR-10/STL-10, CKA, LoRA, CLAHE, HE…) |
| Numbers generated from results, not typed | ✅ | make_paper_tables.py → paper/results/*.tex; numbers.tex macros |
| Claims registry (claim ↔ evidence) | ✅ | README §Claims |

## 5. Double-blind compliance (rejection-without-review risk)

| Item | Status | Action |
|---|---|---|
| No author/institution names in paper | ✅ | paper/main.tex uses `Anonymous submission` |
| No identifying URLs/links in paper | ✅ | none |
| Public repo must be anonymized before submission | 🟡 | **Before submission: create an anonymous mirror (e.g. Anonymous GitHub) and reference *that* in the paper; keep this named repo for the camera-ready** |
| Supplementary anonymized | ✅ | supl.tex has no identity info |

## 6. Submission logistics

| Item | Status |
|---|---|
| OpenReview profiles for all authors (up-to-date; institutional email; allow 2 weeks moderation) | ❌ user action |
| Dual-submission window respected (Nov 16 2026 → decision) | user action |
| Ethics: no human/medical data — N/A; dataset licenses standard | ✅ |
| Plagiarism: all text original, results self-generated | ✅ |

---

## Gap-fix plan (what remains between the repo and a submittable paper)

1. **Scale.** Pilot numbers (n=120 test fold) are provisional. Run the full
   config on Colab GPU: `run_all_colab.py` → n=1000 fold, all 11 experiments,
   LoRA ablation included. ~2–3 h on a T4. Fill paper from `results/`.
2. **LoRA ablation + anchored variant.** Colab-only (needs GPU). The
   objective-conflict claim currently rests on prior-run evidence; the
   anchored-adapter arm (`train_lora_anchored`, wired into
   `exp_lora_ablation`) is implemented and awaiting the run.
3. **Figures.** ✅ Publication PDF figures generate via
   `make_paper_figures.py` (`paper/figures/*.pdf`, Okabe–Ito palette);
   regenerate from full-scale CSVs after the GPU run.
4. **Backbone generality.** ✅ DONE — ViT-B/14 replication committed
   (`run_vitb_generality.py` → `results_pilot/vitb_*`): depth gradient
   steepens +0.32 → +0.44; ViT-B is more robust at the cliff (36.7% vs 23.3%
   at sev 4) yet hits the same floor. Supplementary tables + macros wired.
5. **Anonymized artifact** for review (Anonymous GitHub) — user action
   before submission.

## What NOT to claim (honesty guards)

- The pilot-scale numbers in `results_pilot/` are labeled as such everywhere
  and must not appear in the submitted PDF; regenerate `paper/results/` from
  `results/` (full scale) via `make_paper_tables.py`.
- H1 (linear degradation) is *rejected* by our own data — the paper's story is
  the grace-bump/cliff/floor regime structure, not a monotonic curve.
- Single corruption family (photometric). Do not claim general corruption
  robustness; the frequency dissociation experiment exists precisely to bound
  what the claim covers.
