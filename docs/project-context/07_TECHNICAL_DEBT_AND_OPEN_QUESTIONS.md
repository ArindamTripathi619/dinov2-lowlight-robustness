# 07 — Technical Debt & Open Questions

*Findings are labelled **[Confirmed]** (verified by direct read/execution),
**[Reported]** (from a sub-agent report, not independently re-verified), or
**[Question]** (needs a decision/owner).*

## A. Confirmed / reported defects

### A1. Global-RNG augmentation under multiprocessing — **[Confirmed]**
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

### A2. `apply_lora` mutates its argument in place — **[Confirmed, intentional]**
`src/lora.py:45-59` replaces submodules on the passed model and flips `requires_grad`
directly — no copy. This is **deliberate**: an in-file comment (`src/lora.py:138`,
"reused; apply_lora mutates in place") documents it, and the harness reloads a fresh
model per config (`run_experiments.py:433-434`) so adapters never accumulate. Still a
latent footgun for any caller that passes a still-needed model.

- **Fix direction:** return a wrapped copy, or guard against double-application.

### A3. `analyze_seeds.py` None-guard — **[Reported]**
`analyze_seeds.py:97,99` assumes a `severity 0:` line exists in each log; if a log lacks it,
`clean` stays `None` and later arithmetic raises `TypeError`.

- **Fix direction:** skip/guard runs with no clean baseline, or fail with a clear message.

### A4. `run_notebook3.py` output divergence — **[Reported]**
The plain `run_notebook3.py` does not emit three CSVs that its `.ipynb` counterpart and
`run_notebook3_colab.py` produce: `quantization_results.csv`, `geometry_results.csv`,
`zero_shot_results.csv`. Downstream consumers must not assume the plain runner's output set
matches the Colab/notebook output set.

### A5. `src/supplementary.py` cross-dataset path is broken — **[Confirmed]**
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

## B. Contradiction to resolve

### B1. Deterministic-noise primitive: present, but report says it was not ported — **[Confirmed]**
`src/corruptions.py:22` defines and **uses** `_deterministic_noise` (order-independent
noise keyed on blake2b(image bytes, severity)) in `low_light_stage1`. Yet
`docs/DIVERGENCE_REPORT.md` §8.2 and §9 state upstream's deterministic-noise corruption
primitive was *deliberately not ported* (to preserve byte-comparability), keeping the
seeded-noise protocol.

- Either the merged tree adopted a deterministic primitive after the report was written, or
  the two "deterministic" designs differ (upstream `0737301` vs this `_deterministic_noise`).
- **Action:** the owner should reconcile `docs/DIVERGENCE_REPORT.md` §8.2/§9 with the actual
  `src/corruptions.py` contents, and confirm which corruption protocol the committed
  `results_pilot/` artifacts were produced with. Until then, treat the report's claim as
  **stale**.

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

1. **[Question] Corruption protocol of record.** Which of {seeded `1000+severity`,
   `_deterministic_noise`} produced the committed `results_pilot/` artifacts? Blocks any
   "reproduce byte-exact" claim (see §B1).
2. **[Question] GPU-scale drift-weighted LoRA on ViT-B.** Identified in `docs/DIVERGENCE_REPORT.md`
   §7 / ROADMAP as the strongest remaining open combination; the LoRA-arm half (v9) is open.
3. **[Question] Full-scale paper numbers.** When the merged base has full-scale results, all
   tables/figures/macros must be regenerated and the `[pending]` placeholders removed.
4. **[Question] Optional residual of ROADMAP Track 0.** Reconcile deterministic-noise
   primitive in a follow-up PR (see `docs/DIVERGENCE_REPORT.md` §9).
5. **[Risk] No torch-path tests.** CI cannot catch regressions in the model/LoRA/presentation
   paths (see `06_TESTING_AND_QUALITY.md` §5).
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

## F. Suggested triage order (if asked to harden)

1. Fix `src/lora.py` augmentation RNG (A1) with re-baseline plan.
2. Add a clear guard for `analyze_seeds.py` (A3) — cheap, isolated.
3. Fix the `src/supplementary.py` cross-dataset bugs (A5) — small, isolated, unlocks one experiment.
4. Reconcile the deterministic-noise contradiction (B1) — documentation + owner decision.
5. Pin which harness version produced the committed `results_pilot/` artifacts (E8).
6. Consolidate stats duplication (C) behind `stats_tools.py`.
7. Delete dead `upstream_src/` after confirmation.
8. Add a minimal torch-path smoke test (model load + one forward) behind an optional CI job.
