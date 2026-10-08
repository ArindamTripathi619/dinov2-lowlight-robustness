# Divergence Report — `officialarghya29/dinov2-lowlight-robustness` vs `ArindamTripathi619/dinov2-lowlight-robustness`

*Generated 2026-09-21 from git facts (merge-base `fc9ea04`). Purpose: shared basis for
reconciling the two lines of development. **Superseded by events: PR #1 merged
2026-10-01 — see §9.** Sections §1–§8 are kept as the record of the pre-merge state.*

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

---

## 8. Addendum (2026-09-28): upstream force-push, ports, and H4 at scale

### 8.1 Upstream rewrote its history — merge-base is gone

Upstream force-pushed a rewritten history: new root `752c8bd`, content-identical to the
old base `fc9ea04`. Consequence: **`git merge-base main upstream/main` is now empty** —
GitHub classifies the fork as unrelated history and **blocks the PR**. This supersedes
§6's "merge `upstream/main` into `main`" mechanics (a normal merge is no longer possible
without `--allow-unrelated-histories`, which would produce an unreadable history).

**Rebase plan (supersedes §6 step 1 mechanically, not in spirit):**

1. `git rebase --onto upstream/main <old-root> main` — replay our 14 commits onto
   upstream's new root.
2. Resolve the **same 6 overlapping files** of §3 (roles unchanged: ours win for
   runners/docs, upstream wins for `paper/`, `src/`, `tests/`, CI, configs).
3. Adopt upstream housekeeping landed after the rewrite (pip-audit CI workflow,
   torch-free `src/common.py`) rather than re-fighting it.
4. Re-verify: `tests/run_tests.py` + our smoke; then PR to upstream.

§6's *rationale* (fork integrates, upstream reviews its own untouched trees) is
unaffected — only the git mechanics change.

### 8.2 What was ported from upstream (and what deliberately was not)

Ported and verified locally (`stats_tools.py`, torch-free; sanity-checked: `cka_u(X,X)=1.0`,
independent data ≈ 0.014):

- Unbiased linear CKA (Kornblith et al. 2019, App. B) → robustness check §8.3 below.
- Curve/permutation null tests, paired permutation test, Wilson CI, BH-FDR.
- Participation ratio (spectral concentration of the drift covariance).
- `readout_collapse_metrics` (top1_share / entropy_ratio / dominant class) — their F6
  "99% frog" analysis, now backing our §6.4 signature.

Deliberately **not** ported: upstream's deterministic-noise corruption primitive (would
break byte-comparability with every committed result in this repo; our seeded-noise
protocol stands until the merged repo re-baselines), and their `configs/` manifest
(adopt at rebase time, step 3 above).

Audit result that matters for novelty: upstream's `src/lora.py` has **no drift
allocation** — our drift-weighted rank allocation remains the unique contribution at
the intersection of the two lines. Their ViT-B late-vs-early CKA gap pilot
(+0.32 → +0.44) supersedes our planned ViT-B curve run; the remaining open novelty is
**GPU-scale drift-weighted LoRA on ViT-B**.

### 8.3 Upstream's H4 confirmed at proper power

Upstream's H4 (readout-staleness) pilot: +22.5 pp at sev-4 from a severity-adapted
probe, n = 120, ViT-B. We ported the arm hardened (`run_readout_repair.py` — strict
same-train-fold control, paired permutation p, Wilson CIs) and ran it at n = 1,000,
ViT-S: **+25.3 pp at sev-4 (p = 0.0002), +30.0 pp at sev-5** — with the new finding that
adaptation is flat-to-negative at sev 1–2 (probe-level echo of their grace regime) and
that even the adapted readout plateaus at 0.377 vs 0.913 clean, keeping feature drift
the dominant bottleneck (`docs/RESEARCH.md` §6.4).

Relatedly, their F5 tension ("darkness ≠ generic frequency damage": dark 61.7% vs
low-pass 80.0% vs high-pass 14.2% at sev-3) is compatible with our §5.1 extension:
darkness is *one* frequency-destroying corruption, not the only one — blur/JPEG
reproduce the late-drift signature, contrast (frequency-preserving) does not. Both
claims survive at different scopes; the merged paper should state both scopes
explicitly. Their grace regime (sev-1 helps, 95.0 > 93.3 at n = 120) vs our sev-1 dip
(0.903 < 0.913 at n = 1,000) remains an open n-vs-noise question to settle on the
merged base.

### 8.4 Unbiased-CKA verdict (affects their allocator review)

Recomputing our CKA matrix with their (ported) unbiased estimator shifts the
r = 8 allocation at 3 of 12 blocks (one rank each; `docs/METHODS.md` §4) while blocks
9–11 keep maximum rank under both estimators. **The drift-allocation claim survives
estimator choice**; reviewers running their estimator will reproduce the late-heavy
structure, not the exact rank vector.

---

## 9. Addendum (2026-10-01, verified 2026-10-08): PR #1 merged — integration complete

The rebase plan of §8.1 was executed and shipped as
[PR #1](https://github.com/officialarghya29/dinov2-lowlight-robustness/pull/1),
merged 2026-10-01 (merge commit `9bc6b50`):

- The fork's 14 commits were replayed onto upstream's rewritten root, so `upstream/main`
  and this repo's `main` share one lineage again — §8.1's empty-merge-base problem is
  resolved, and §6's "nothing has been merged" status above is historic.
- Overlap files resolved per §5/§8.1 roles: this fork's runners + `docs/`, upstream's
  `paper/`, `src/`, `tests/`, CI, `configs/`. **The `configs/` manifest §8.2 declined to
  port is now in-tree** (it arrived with the merge and is used by `run_experiments.py`
  and the other `configs/`-reading runners) — that open item is closed.
- Verification on the merged tree (per the PR): upstream's `tests/run_tests.py`
  22/22 pass; no deletions relative to either side — both content sets fully preserved.
- **Still open from §8.2:** upstream's deterministic-noise corruption primitive was
  deliberately not ported (it would break byte-comparability with this repo's committed
  results); the seeded-noise protocol (`1000 + severity`) stands until a follow-up PR
  reconciles the two. Tracked as the optional residual of ROADMAP Track 0.
- §7's post-merge opportunities: (a) ViT-B × drift-weighted → v8 profile done
  (`6a9de11`), LoRA-arm half (v9) still open; (b) single manuscript → ROADMAP Track 6;
  (c) CI-tested parameterizations → `tests/` + `.github/workflows/` now in-tree.
