# Divergence Report — `officialarghya29/dinov2-lowlight-robustness` vs `ArindamTripathi619/dinov2-lowlight-robustness`

*Generated 2026-09-21 from git facts (merge-base `fc9ea04`). Purpose: shared basis for
reconciling the two lines of development. Nothing has been merged; no branch created.*

---

## 1. Situation

- The fork diverged from merge-base `fc9ea04` (Feb 2026).
- GitHub shows the fork **12 commits ahead / 10 behind**. Both numbers are real content,
  not housekeeping: **neither repo is a superset of the other** — this is parallel
  evolution, not a stale fork.
- **Timelines confirm concurrency:** upstream's 10-commit push landed **Sep 14–15,
  2026**; this fork's Phase 3 v2 work landed **Sep 19–21, 2026**. The lines were
  developed simultaneously with no cross-pollination after February.

## 2. What each side built (from `git diff --name-only` against the merge-base)

| | Upstream (arghya) | This fork (Arindam) |
|---|---|---|
| **Science** | ViT-B/14 generality replication (`run_vitb_generality.py`, `results_pilot/vitb_*.csv`) — the roadmap item this fork ranked lowest and never ran; corollary experiments (`run_corollaries.py` → `corollaries.json`); collapse analysis (`run_collapse_analysis.py`); real CPU pilot (`run_pilot_cpu.py`) | ExDark real-dark suite: baseline + two sim-to-real transfer tests (`run_exdark_baseline.py`, `output/exdark_*/`); 6-arm LoRA grid incl. **drift-weighted allocation** and full-FT baseline (`colab_results/sessionD/`, adapters included) |
| **Paper** | Full CVPR-style draft, **compiled PDF**, figures, tables, supplementary (`paper/`, ~25 files) | Internal narrative record (`docs/RESEARCH.md`, `docs/METHODS.md`) + verified related-work draft (`docs/PAPER_RELATED_WORK.md`) |
| **Engineering** | `src/` harness (11 modules), test suite (`tests/`), CI + weekly security-audit workflows, `configs/`, pinned requirements | v2 runner features: `--layers {all,late,drift,fullft}`, `--corruption {low_light,blur,jpeg,contrast}`, adapter checkpointing, shared feature caches; bug fixes (worker-correlated augmentation, bootstrap RNG, CKA formula) |
| **Signals** | `run_lora_simple_colab.py` = **335 lines**, their semantics | same path = **503 lines**, ours |

## 3. Conflict surface

- **Overlap: only 6 files** touched by both sides — `README.md`, `requirements.txt`,
  `.gitignore`, `run_lora_simple_colab.py`, `run_lora_finetune_colab.py`,
  `dinov2_lora_finetune.ipynb`.
- Everything else lives in **disjoint trees**: upstream-only ≈ 73 files (`paper/`,
  `src/`, `tests/`, `.github/`, `configs/`, `results_pilot/`, analysis scripts),
  fork-only ≈ 57 files (`docs/`, `output/`, `colab_results/`, ExDark/v2 scripts).
- A merge therefore resolves 6 files, most of which are straightforward.

## 4. Scientific compatibility (checked, not assumed)

- Upstream corollary 1 ("adapted readout beats enhancement", p≈0.0004 at sev-5)
  **independently corroborates** our ExDark CLAHE-null finding.
- Upstream corollary 2 ("geometry attractor" — collapse persists under imbalanced
  prior) **matches** our class-confusion-not-darkness failure analysis.
- Both lines therefore tell one coherent story; integration strengthens rather than
  contradicts either.

## 5. Semantic audit items (must resolve at merge time)

1. **Corruption determinism** — upstream commit `0737301` ("Deterministic corruption")
   may implement the same determinism this fork fixed in Phase 0 (per-sample aug
   seeds). If the semantics differ, the merged repo must reconcile them or the two
   result sets are not comparable.
2. **`run_lora_simple_colab.py`** — keep the 503-line v2 (superset of features) but
   verify upstream's pilot results remain reproducible through it, or keep both
   entry points with clear README roles.
3. **README** — hand-merge: their quickstart/claims-registry structure + our results
   narrative and artifact tables.
4. **requirements.txt** — union of both, then re-pin.

## 6. Recommended resolution (roles)

1. **Fork (Arindam)** integrates: merge `upstream/main` into `main`, take upstream
   wholesale for `paper/`, `src/`, `tests/`, CI, configs, `results_pilot/`; take fork
   for runners, `docs/`, artifacts; hand-merge the 6 overlaps per §5.
2. **Fork** runs upstream's `tests/run_tests.py` + this fork's smoke on the merged
   tree; cross-verifies both result sets still reproduce.
3. **Fork** pushes and opens a PR to upstream: *"Integrate parallel v2 development:
   ExDark sim-to-real, 6-arm LoRA grid, drift-weighted LoRA"*.
4. **Upstream (arghya)** reviews (his `paper/`+`src/` arrive untouched — low-risk
   review), confirms the claims registry covers both result sets, merges.
5. **Both** agree in the PR that both result sets are canonical and future work
   starts from the merged base. GitHub ahead/behind returns to zero; one mature
   version exists everywhere.

*Why this direction:* the fork has the merge tooling, context, and test assets loaded;
asking upstream to merge fork-side work would put conflict resolution on the person
with less context. Remaining divergent fails the "one mature version everywhere"
requirement.

## 7. Post-merge opportunities (no action yet)

- Upstream's **ViT-B replication** + our **drift-weighted allocation** → run the
  CKA-guided allocation on ViT-B (the strongest open combination of the two lines).
- Their CVPR paper draft + our `PAPER_RELATED_WORK.md` v1.1 (verified citations,
  re-scoped novelty claim, RepSAM/FastDINOv2 positioning) → single manuscript.
- Their test suite + our corruption-axis runner → corruption expansion (blur/JPEG/
  contrast) gets CI-tested parameterizations for free.

---

*Facts in this report are reproducible via: `git fetch upstream` then
`git log --oneline main..upstream/main`, `git log --oneline upstream/main..main`,
`git diff --name-only $(git merge-base main upstream/main) upstream/main` (and `main`).*
