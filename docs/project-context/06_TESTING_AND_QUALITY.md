# 06 — Testing & Quality

## 1. Test runner

`tests/run_tests.py` (293 lines) is a **custom, dependency-free test runner** — **not
pytest** (pytest is not a dependency and is not installed in the observed environment).

- Discovery: collects every module-global `callable` whose name starts with `test_`.
- Execution: calls each in definition order; `AssertionError` → FAIL, any other exception
  → ERROR; prints `PASS/FAIL/ERROR` per test.
- Exit code: number of failures (`sys.exit(main())`), so **non-zero on any failure**.
- Output: `"<n>/<total> passed"`.

### Run it

```bash
python3 tests/run_tests.py
```

**Observed result at HEAD `90577cb`: `22/22 passed`, exit 0.**

## 2. What is covered (22 tests)

Everything tested is in the **torch-free analysis stack** (`src/corruptions.py`,
`src/analysis.py`, `src/probes.py`, plus local `biased` math in-test). The runner does not
import torch.

| Group | Tests |
|---|---|
| Corruption physics (9) | `low_light_severity0_is_identity`, `low_light_stage1_no_uint8_snap`, `two_stage_gap_matches_quantization_truth`, `quantization_gap_grows_with_darkness`, `quantization_gap_noise_free_monotone_in_expectation`, `gain_recovers_scale_exactly_without_noise`, `gamma_brightens_shadows_more_than_highlights`, `bandpass_filters_preserve_shape`, `clahe_returns_valid_uint8` |
| CKA (4) | `cka_identical_representations_is_one`, `cka_invariant_to_orthogonal_transform_and_scale`, `unbiased_cka_less_inflated_than_biased_for_small_n`, `cka_distinguishes_related_from_unrelated` |
| Spectral / curves (4) | `participation_ratio_isotropic_and_lowrank`, `curve_test_detects_cliff_shape`, `curve_test_passes_linear_shape`, `paired_permutation_detects_and_calibrates` |
| Statistics (3) | `bh_correction_controls_fdr_monotonicity`, `bootstrap_ci_brackets_observed`, `wilson_extreme_cases` |
| Probes / splits (2) | `stratified_split_preserves_class_balance`, `ntrainable_probe_uses_exactly_n_per_class` |

## 3. Continuous integration

### `.github/workflows/ci.yml`
- Matrix: **Python 3.10 and 3.11** on `ubuntu-latest`.
- Installs only `numpy scipy scikit-learn scikit-image pyyaml` (no torch).
- Steps:
  1. `python tests/run_tests.py`
  2. Import smoke: `python -c "import src.analysis, src.corruptions, src.probes, src.common; print('analysis stack OK')"`

This proves the **torch-optional import invariant** enforced by `src/common.py:_torch`.

### `.github/workflows/security.yml`
- Weekly (Mon 06:00 UTC) + on push touching `requirements.txt` / the workflow.
- `pip-audit -r requirements-audit.txt --strict` — torch-free stack only, by design.

## 4. Quality practices observed

- **Auto-generated artifacts never hand-edited:** `paper/results/*.tex`,
  `paper/figures/*`, `results_pilot/numbers.tex`.
- **Provenance by construction:** config SHA-256 embedded in result manifests.
- **Deterministic corruption** (order-independent noise) so results are call-order independent.
- **Statistical hygiene:** bootstrap CIs, permutation nulls, paired permutation tests,
  Bonferroni/BH-FDR, Wilson CIs.
- **Documented reference environment** (`docs/METHODS.md` §10) for byte-exact reproduction.
- **Paper manuscript audit:** `paper/CVPR2027_CHECKLIST.md`.
- **Security policy:** `SECURITY.md`.

## 5. Gaps / weaknesses

- **No torch-path tests.** The DINOv2 loader, `EmbeddingExtractor`, LoRA training, and the
  presentation app are untested by CI (torch absent). Model-layer behavior is validated only
  by the research runs themselves.
- **No pytest / coverage tooling.** The custom runner has no fixtures, parametrization,
  coverage, or test isolation; tests run in one process and share global RNG state.
- **No lint/type gate.** No ruff/flake8/mypy/pyright config or CI step (some scripts do carry
  `# noqa: BLE001` markers, implying local flake8/ruff use).
- **No performance/regression tests** for the static exporter or presentation.
- **Notebook duplication** is not guarded against `src/` drift.

## 6. How to verify a change (practical checklist)

```bash
python3 tests/run_tests.py                                   # 22/22 expected
python -c "import src.analysis, src.corruptions, src.probes, src.common; print('OK')"
PRESENTATION_SMOKE=1 python3 apps/presentation.py             # if touching the site's source
git status --porcelain                                        # confirm no stray artifacts
```
