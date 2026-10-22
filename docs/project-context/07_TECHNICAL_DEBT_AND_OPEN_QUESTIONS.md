# 07 — Technical Debt & Open Questions

*Findings are labelled **[Confirmed]** (verified by direct read/execution),
**[Reported]** (from a sub-agent report, not independently re-verified), or
**[Question]** (needs a decision/owner).*

## A. Confirmed / reported defects

### A1. Global-RNG augmentation under multiprocessing — **[Resolved 2026-10-11]**
`src/lora.py:73-74` (`LowLightAugCIFAR.__getitem__`) draws from the global `np.random`
stream (`np.random.random()`, `np.random.randint`) inside a `DataLoader` using
`num_workers=2`. With worker processes, the global RNG is forked, so augmentation draws can
be **correlated across workers**, and reproducibility depends on worker scheduling.

- **Impact:** LoRA augmentation determinism/reproducibility.
- **Existing workaround:** `run_lora_simple_colab.py:275-282` replaces the global stream
  with a per-sample `np.random.default_rng(seed=idx)` inside `__getitem__`, so DataLoader
  worker forks never share RNG state.
- **Fix direction:** give each sample a deterministic per-index seed (as
  `src/corruptions.py:_deterministic_noise` already does for corruption) instead of the
  global stream. Do **not** change without re-baselining committed artifacts.
- **Resolved 2026-10-11:** `__getitem__` now uses `np.random.default_rng(seed=i)`
  (mirrors `run_lora_simple_colab.py`, the Phase-3 run of record). Re-baseline check:
  no committed artifact was produced by this path — the Phase-3 results of record came
  from the Colab runner, and no `lora_*.csv` from the harness `lora_ablation` experiment
  is committed — so nothing needed re-baselining. Regression test:
  `test_lowlight_augmentation_deterministic_per_index` (torch-optional; verified to fail
  on the pre-fix code: 11/12 indices changed with access order).

### A2. `apply_lora` mutates its argument in place — **[Confirmed, intentional]**
`src/lora.py:45-59` replaces submodules on the passed model and flips `requires_grad`
directly — no copy. This is **deliberate**: an in-file comment (`src/lora.py:138`,
"reused; apply_lora mutates in place") documents it, and the harness reloads a fresh
model per config (`run_experiments.py:433-434`) so adapters never accumulate. Still a
latent footgun for any caller that passes a still-needed model.

- **Fix direction:** return a wrapped copy, or guard against double-application.

### A3. `analyze_seeds.py` None-guard — **[Resolved 2026-10-11]**
`analyze_seeds.py:97,99` assumes a `severity 0:` line exists in each log; if a log lacks it,
`clean` stays `None` and later arithmetic raises `TypeError`.

- **Fix direction:** skip/guard runs with no clean baseline, or fail with a clear message.
- **Resolved 2026-10-11:** crash reproduced first (TypeError), then guarded: a missing
  `severity 0:` line now prints `-` and marks the CSV row `no_clean` instead of crashing.
  Regression test: `test_analyze_seeds_guards_missing_clean_baseline`.

### A4. `run_notebook3.py` output divergence — **[Confirmed, documented]**
The plain `run_notebook3.py` does not emit three CSVs that its `.ipynb` counterpart and
`run_notebook3_colab.py` produce: `quantization_results.csv`, `geometry_results.csv`,
`zero_shot_results.csv`.Downstream consumers must not assume the plain runner's output set matches
the Colab/notebook output set.
- **Re-verified 2026-10-11:** no committed artifacts use these three names anywhere in
  the tree, and no in-repo consumer reads them (the CSVs read downstream —
  `make_paper_tables.py`, `make_readme_figures.py`, `run_all_colab.py` — are different
  files such as `cross_dataset.csv`, and those readers tolerate absence). The caveat is
  now stated in `run_notebook3.py`'s header; no behavior change made to the unused path.

### A5. `src/supplementary.py` cross-dataset path is broken — **[Resolved 2026-10-11]**
Two laddering defects, both found by direct read on the 2nd pass:

1. `_load_stl10_shared_classes` (`src/supplementary.py:305`) does
   `load_cifar10_subsets.class_names` — but `load_cifar10_subsets` is a **function**, and
   `.class_names` is a module-level constant (`CIFAR10_CLASS_NAMES` in `src/data.py`) →
   `AttributeError`.
2. `exp_cross_dataset` (`src/supplementary.py:337`, erroring near line 367) references
   `n_skipped`, but the helper's return value was bound to `n_dropped` (line ~355) →
   `NameError` before the CSV is written.

- **Impact:** `python3 run_experiments.py --experiment cross_dataset` cannot complete; the
  cross-dataset evidence strand (CIFAR-10 vs STL-10 shared classes) is unavailable from the
  current tree. The committed `cross_dataset.csv` in `results_pilot/` must predate a
  refactor that broke the function.
- **Fix direction:** read class names from `CIFAR10_CLASS_NAMES`; rename `n_dropped` →
  `n_skipped` (or vice-versa); add a torch-optional unit test exercising the helper with a
  stubbed loader.
- **Resolved 2026-10-11:** helper reads `CIFAR10_CLASS_NAMES` (imported at module top);
  the call site binds the helper's third return to `n_skipped` (matching the JSON key).
  Two torch-optional regression tests added: `test_stl10_helper_maps_shared_classes` and
  `test_exp_cross_dataset_end_to_end_stubbed` (stubs CIFAR-10/DINOv2/STL-10 — no
  downloads). Both verified to fail on the pre-fix code (AttributeError / NameError).
- **Correction:** the claim that a committed `cross_dataset.csv` exists in
  `results_pilot/` was wrong — `git ls-files` shows **no** committed `cross_dataset.csv`
  anywhere; the experiment never produced a committed artifact. Downstream readers
  (`make_paper_tables.py:163,259`, `make_readme_figures.py:149`, `run_all_colab.py:161`)
  tolerate its absence.

### A6. Agreement-report regression test was unrunnable in CI — **[Resolved 2026-10-16]**
`tests/run_tests.py::test_agreement_report_dataset_label` (added 2026-10-14, `98c6730`)
did `import run_drift_proxy`, but `run_drift_proxy.py` imports `torch` at module scope — so
in CI's torch-free environment the test **errored** (`ModuleNotFoundError`, 47/48) while any
torch-enabled local run reported 48/48. Never caught because GitHub Actions has been
unavailable on this account since 2026-10-14 (every run dies in ~3 s: "account is locked
due to a billing issue"), i.e. the suite had not actually executed on 3.10/3.11 since.

- **Fix direction:** the report is plain text/JSON rendering needing no torch — move it to
  a torch-free module rather than soft-skipping the test (soft-skips exist elsewhere, but
  this regression test exists precisely to run in CI).
- **Resolved 2026-10-16:** renderer extracted to `agreement_report.py` (torch-free);
  `run_drift_proxy` re-exports `PROXY_KEYS` + `write_agreement_report`, so
  `run_drift_profile.py` and the runner are unchanged. The test imports the torch-free
  module and also asserts the re-export wiring, so the runner cannot drift from the tested
  writer. Verified **48/48 on Python 3.10 and 3.11 with CI's exact dependency set**
  (reproduced locally: `uv run --python 3.10 --with numpy,scipy,scikit-learn,scikit-image,pyyaml
  python tests/run_tests.py`, same for 3.11) plus 48/48 on the local torch-enabled env.

### A7. Deployed site drifted from its source; 24 gallery figures were never committed — **[Resolved 2026-10-22]**
Two coupled defects in the presentation pipeline:

1. `index.html` (deployed site) was last exported at `40fe066` (2026-10-13, 14 sections)
   and was **never regenerated** after `aa3a006` (2026-10-15) added the Track-4 SSR
   section — the live site was missing the entire SSR story while the repo docs described
   it. `PRESENTATION_SMOKE` could not catch this: it renders the *app source*, not the
   committed export.
2. The gallery section enumerates `output/**/*.png` from the working tree, so it
   referenced 24 figures in gitignored dirs (`output/notebook1_{blur,jpeg,contrast}/`,
   `output/notebook2_{blur,jpeg,contrast}/`, `output/drift_proxy_smoke/`) that were
   **never committed** — a fresh clone produced broken image links (and the exporter's
   existence self-check only passed because the files existed locally). The root cause was
   `.gitignore`: `output/**` plus per-dir negations, where the corruption-family re-run
   dirs had never been added to the negation list (their base dirs `notebook1/`/`notebook2/`
   were, which is why the base figures were tracked).

- **Resolved 2026-10-22:** all 24 figures + sibling CSVs committed; `.gitignore` negations
  added for the 7 dirs; `index.html` re-exported (15 sections, all 245 figures present);
  exporter badges (`N sections` / `N committed figures`) now derived from the build rather
  than hardcoded (they said "14 sections / 84 committed figures"); bugs section updated
  4 → 6 (exit-verifier naming, hardcoded report label). Exporter's existing self-check
  (broken-src detection) plus the committed files now make the invariant enforceable from
  a clean clone. **Process rule: re-run `python3 tools/export_static_site.py` and commit
  the refreshed `index.html` + `static/gen/` whenever `apps/presentation.py` changes** —
  smoke alone does not prove the deployed site is current.

## B. Contradiction to resolve

### B1. Deterministic-noise primitive: present, but report says it was not ported — **[Closed 2026-10-09]**
`src/corruptions.py:22` defines and **uses** `_deterministic_noise` (order-independent
noise keyed on blake2b(image bytes, severity)) in `low_light_stage1`, while the fork's
seeded protocol (`default_rng(1000 + severity)`) drives every committed artifact.

- Reconciled by agreement: the two primitives **coexist by pipeline and must not be
  merged** — verdict + consumer table in `docs/DIVERGENCE_REPORT.md` §8.5; protocol
  stands, no re-baseline.
- Which artifacts used which protocol: the seeded protocol produced everything in
  `results_pilot/`, `colab_results/`, `output/` (§8.5 consumer table); the deterministic
  primitive serves only the merged `configs/`-reading runners.

## C. Duplication / dead weight

| Item | Issue | Action |
|---|---|---|
| `upstream_src/` | Byte-identical 10-file copy of `src/`; **zero references** anywhere. | Delete after confirmation (out of scope here). |
| `src/analysis.py` vs `stats_tools.py` | Same statistics (CKA, permutation, PR, BH) implemented twice; `stats_tools.py` adds `readout_collapse_metrics`; `src/common.py` also has `wilson_ci`. | Consolidate behind one module; keep the torch-free `stats_tools` as the shared base. |
| `utils.py` vs `src/` | `utils.py` (523 lines) duplicates corruption, DINOv2 loading, probe, CKA, plotting logic. | Pick one home; `utils.py` is the legacy path. |
| `.ipynb` notebooks | Self-contained; do **not** import `src/`, so they silently drift from the library. | Either import `src/` or auto-generate from it. |
| Root `__static_*.png` (7) | Generated screenshots sitting in repo root. | Move to `static/` or gitignore. |
| `.freebuff/project-id` | Private tooling metadata committed in-tree. | Confirm intentional; otherwise ignore/remove. |

## D. Documentation staleness

- `docs/DIVERGENCE_REPORT.md` §1–§8 are explicitly superseded by §9 (PR #1 merged); §8.2/§9
  additionally conflict with §B1 above. Read the report as history, not current truth.
- `paper/README.md` `main.tex` references a `table_lora.tex` claim artifact that is **not
  present** in `paper/results/` [Reported: `main.tex:458,466` prints
  `"[pending full-scale GPU run]"`]. The LoRA "positive" result is backed by
  `colab_results/lora_run_fixed/lora_full_run.log` instead.
- Paper tables/macros currently reflect **pilot-scale** numbers; full-scale GPU numbers are
  pending. Bars/CIs are wide by design at pilot n.

## E. Open questions / risks

1. **[Closed 2026-10-09] Corruption protocol of record.** The seeded `1000+severity`
   protocol produced the committed `results_pilot/` artifacts (consumer table in
   `docs/DIVERGENCE_REPORT.md` §8.5); byte-exact re-run generator pinning stays under E8.
2. **[Closed 2026-10-10] GPU-scale drift-weighted LoRA on ViT-B.** v9 ran all three arms
   × seeds 42/43/44 (kernel `vitb-lora-v9`): parity with uniform at 24% fewer params —
   `output/v9_lora/`, RESEARCH §6.5. The allocation-shape ablations (β^τ / top-k /
   inverted-β) closed 2026-10-11 at n=3 (kernel `vitb-lora-abl`, `output/v9_ablation/`,
   RESEARCH §6.6): all shapes tie uniform/drift — shape inert, coverage discriminates;
   only sublayer-resolved allocation remains before paper claims.
3. **[Question] Full-scale paper numbers.** When the merged base has full-scale results, all
   tables/figures/macros must be regenerated and the `[pending]` placeholders removed.
4. **[Closed 2026-10-09] Optional residual of ROADMAP Track 0.** Deterministic-noise
   reconciled by agreement — verdict in `docs/DIVERGENCE_REPORT.md` §8.5/§9; protocol stands.
5. **[Risk, reduced 2026-10-11] No torch-path tests in CI.** CI cannot catch regressions
   in the model/LoRA/presentation paths (torch absent — see `06_TESTING_AND_QUALITY.md`
   §5), but three torch-optional regression tests (Debt §A1/A3/A5) now exercise
   `src/lora` + `src/supplementary` whenever the suite runs in a torch-enabled local
   environment. A dedicated torch CI job remains open (F.8).
6. **[Risk] No lint/type gate.** Style/type drift is unguarded in CI.
7. **[Risk] Presentation drift.** The README section list and the app's `SECTION_ORDER` were
   reconciled in recent commits (`6f75d67`, `8d1c1fc`); any section rename must update both.
8. **[Question] Committed pilot-artifact ↔ harness drift.** `results_pilot/cka_summary.json`
   uses an **older schema** (`early_mean_0_3` / `late_mean_8_11` / `max_drop` /
   `max_drop_block`) than the current harness writer (`early_mean_drop_blocks_0_3` /
   `late_mean_drop_blocks_8_11` / `max_drop_value` / …); the pilot `main_curve.csv`
   (0.933 → 0.108, n≈60) also differs from the full-scale `docs/RESEARCH.md` numbers
   (0.913 → 0.083, n=1000), and no `lora_*.csv` exists in `results_pilot/`. **The committed
   artifacts were not all produced by the current `run_experiments.py`** — pin the generator
   commit before promising byte-exact re-runs (ties into B1).
   **Pin (2026-10-11):** `results_pilot/` artifacts were introduced by commit `128e9cb`
   ("Run real CPU pilot, add collapse finding, supplementary suite, CVPR audit") — that
   commit's harness state is the generator of record; HEAD's writer schema differs (the
   `cka_summary.json` drift above). Byte-exact re-runs must check out `128e9cb`.
9. **[Risk, external] GitHub-hosted CI is billing-locked.** Every Actions run (push, PR,
   schedule, manual) since 2026-10-14 aborts in ~3–4 s with zero steps started and the
   annotation *"account is locked due to a billing issue"* — an account-level action, not
   a workflow/code defect. `main` has no branch protection, so merges are unblocked, but
   PR checks stay red until billing is cleared; then `gh run rerun <id>` should pass (the
   exact recipe passes locally). Mitigation: **`scripts/ci_local.sh`** reproduces the CI
   matrix (py3.10/3.11, torch-free deps via `uv`) plus the gates GitHub CI cannot run
   (native torch tests, 9 kernel builds, presentation smoke) and is the gate of record
   in the interim (2026-10-22: ALL GREEN). Related: the weekly `pip-audit` Security
   workflow fails the same way and needs no code change.

## F. Suggested triage order (if asked to harden)

1. [done 2026-10-11] Fix `src/lora.py` augmentation RNG (A1) — resolved without
   re-baseline (no committed artifact came from that path).
2. [done 2026-10-11] Add a clear guard for `analyze_seeds.py` (A3) — cheap, isolated.
3. [done 2026-10-11] Fix the `src/supplementary.py` cross-dataset bugs (A5) — small,
   isolated, unlocks one experiment.
4. [done 2026-10-09] Reconcile the deterministic-noise contradiction (B1) — verdict in DIVERGENCE §8.5.
5. Pin which harness version produced the committed `results_pilot/` artifacts (E8).
6. Consolidate stats duplication (C) behind `stats_tools.py`.
7. Delete dead `upstream_src/` after confirmation.
8. Add a minimal torch-path smoke test (model load + one forward) behind an optional CI job.
