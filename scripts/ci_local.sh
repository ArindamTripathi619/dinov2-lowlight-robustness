#!/usr/bin/env bash
# ci_local.sh — local reproduction of .github/workflows/ci.yml.
#
# GitHub-hosted CI is unusable for this account right now (billing lock kills
# every run in ~4s with zero steps started, since 2026-10-14). This script
# reproduces the CI matrix exactly and adds the gates GitHub CI cannot run:
#
#   1. tests/run_tests.py + analysis-stack import smoke on Python 3.10 (CI matrix)
#   2. the same on Python 3.11 (CI matrix)
#   3. the same on the native python3 (torch-enabled: exercises the
#      torch-optional regression tests CI self-skips) + runner imports
#   4. builds all 10 Kaggle kernel variants (manifest wiring regressions)
#   5. presentation smoke: renders all 15 sections headlessly
#
# Exits non-zero on the first failure. Requires: uv (matrix steps), torch +
# pandas/matplotlib in the native python3 (steps 3-5).
set -euo pipefail
cd "$(dirname "$0")/.."

CI_DEPS="numpy,scipy,scikit-learn,scikit-image,pyyaml"
IMPORT_SMOKE="import src.analysis, src.corruptions, src.probes, src.common; print('analysis stack OK')"
PY_MATRIX=("3.10" "3.11")

command -v uv >/dev/null 2>&1 || { echo "error: uv not found (https://docs.astral.sh/uv/)" >&2; exit 1; }

for v in "${PY_MATRIX[@]}"; do
    echo "=== [CI matrix] Python $v: tests/run_tests.py ==="
    uv run --python "$v" --with "$CI_DEPS" python tests/run_tests.py
    echo "=== [CI matrix] Python $v: import smoke ==="
    uv run --python "$v" --with "$CI_DEPS" python -c "$IMPORT_SMOKE"
done

echo "=== [native] python3: tests/run_tests.py (torch-optional tests active) ==="
python3 tests/run_tests.py
echo "=== [native] import smoke (analysis stack + runners) ==="
python3 -c "$IMPORT_SMOKE"
python3 -c "import run_drift_proxy, run_drift_profile, run_ssr; print('runner imports OK')"

echo "=== [kernels] build all 10 variants ==="
for variant in grid vitb_profile vitb_lora_v9 vitb_lora_abl atlas atlas_v4 ssr_s1 ssr_s2 ssr_s3 ssr_s4; do
    pkg="$(mktemp -d)"
    python3 scripts/build_kaggle_kernel.py --variant "$variant" --pkg "$pkg" >/dev/null
    echo "  $variant OK"
done

echo "=== [presentation] smoke: render all 15 sections headlessly ==="
PRESENTATION_SMOKE=1 python3 apps/presentation.py

echo
echo "ALL GREEN: CI matrix (3.10/3.11, torch-free) + native + kernels + presentation"
git status --porcelain | sed 's/^/  dirty: /' || true
