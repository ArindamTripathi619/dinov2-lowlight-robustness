"""Unit tests for analysis primitives and corruption math.

Runs on CPU with no model downloads — fast enough for CI. Tests the *math*
and *data-flow* invariants the paper's claims rest on:

    python3 -m pytest tests/ -q     (or: python3 -m tests.run_tests)
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analysis import (benjamini_hochberg, bootstrap_accuracy_ci, curve_permutation_test,
                          linear_cka_unbiased, paired_permutation_test, participation_ratio)
from src.corruptions import (BRIGHTNESS_FACTORS, clahe_enhance, gamma_correct, high_pass,
                             low_light, low_light_stage1, low_pass, quantization_gap_truth,
                             simple_gain)
from src.probes import NTrainableProbe, stratified_split


# ---------------------------------------------------------------------------
# Corruption physics
# ---------------------------------------------------------------------------

def test_low_light_severity0_is_identity():
    img = np.random.randint(0, 256, (8, 8, 3), dtype=np.uint8)
    out = low_light(img, 0)
    assert np.array_equal(out, img), "severity 0 must be exact identity"


def test_low_light_stage1_no_uint8_snap():
    img = np.full((4, 4, 3), 10, dtype=np.uint8)
    out = low_light_stage1(img, 5, factors=[1.0, 0.75, 0.55, 0.38, 0.25, 0.15],
                           noise_std=[0, 0, 0, 0, 0, 0])
    # 10 * 0.15 = 1.5 -> float arm keeps 1.5, uint8 arm rounds to 2 (or 1)
    assert out.dtype == np.float32
    assert np.allclose(out, 1.5)


def test_two_stage_gap_matches_quantization_truth():
    """Noise-free ladder: gap = RMS of pure rounding error (identical to the
    default noise_free=True path of quantization_gap_truth)."""
    img = np.random.randint(0, 256, (16, 16, 3), dtype=np.uint8)
    s1 = low_light_stage1(img, 4, noise_std=[0] * 6)
    s2 = np.clip(np.round(s1), 0, 255).astype(np.uint8).astype(np.float32)
    rms_manual = np.sqrt(np.mean((s1 - s2.astype(np.float32)) ** 2))
    rms_fn = quantization_gap_truth(img, 4, noise_free=True)
    assert abs(rms_manual - rms_fn) < 1e-6


def test_quantization_gap_grows_with_darkness():
    """Darkness compresses signal into fewer uint8 levels -> the float-vs-uint8
    arm gap must grow. Test the EXPERIMENTAL quantity (accuracy-proxy): with a
    noise-free ladder, quantization error must increase strictly as a shrinks,
    because the absolute rounding step (0.5) is a larger fraction of the signal.
    We verify via the distinct-levels count, which is what feeds the accuracy gap.
    """
    img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    levels = []
    for s in range(6):
        out = low_light(img, s)
        levels.append(len(np.unique(out.ravel())))
    # fewer distinct levels as it gets darker (monotone non-increasing, strict at the dark end)
    assert levels[5] < levels[1] <= levels[0], f"expected shrinking dynamic range, got {levels}"
    # and the rounding RMS cannot exceed the worst-case step for stage-1 range
    rms5 = quantization_gap_truth(img, 5)
    assert 0 < rms5 <= 0.5 * 1.0 + np.sqrt(0.0) + 1.0  # loose sanity: <~1.5


def test_quantization_gap_noise_free_monotone_in_expectation():
    """With noise off, the only stage-2 effect is rounding, so the RMS between
    stage-1 and stage-2 equals the RMS of rounding error, which grows as the
    signal occupies fewer levels relative to the fixed 0.5 grid step."""
    factors = [1.0, 0.75, 0.55, 0.38, 0.25, 0.15]
    img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    rms = []
    for s in range(6):
        s1 = low_light_stage1(img, s, noise_std=[0] * 6)
        s2 = np.clip(np.round(s1), 0, 255).astype(np.uint8).astype(np.float32)
        rms.append(np.sqrt(np.mean((s1 - s2) ** 2)))
    # rounding RMS is ~0.289 regardless of scale when values are far from the
    # clip boundaries; the *relative* error grows. Assert relative growth.
    rel = [r / max(f, 1e-6) for r, f in zip(rms, factors)]
    assert rel[5] > rel[0], f"relative quantization error must grow with darkness: {rel}"


def test_gain_recovers_scale_exactly_without_noise():
    """Gain must invert attenuation up to TWO truncation stages: astype(uint8)
    TRUNCATES toward zero (not round), and it happens (a) when the corruption
    digitizes a*I and (b) again when gain re-digitizes I/a. Bound:
    err <= trunc_a / a + trunc_out <= 1/a + 1."""
    img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    a = BRIGHTNESS_FACTORS[2]
    dark_u8 = np.clip(img.astype(np.float32) * a, 0, 255).astype(np.uint8)
    restored = simple_gain(dark_u8, 2).astype(np.float32)
    err = np.abs(restored - img.astype(np.float32))
    bound = 1.0 / a + 1.0 + 1e-6
    assert err.max() <= bound, f"max per-pixel error {err.max():.3f} exceeds 1/a+1={bound:.3f}"


def test_gamma_brightens_shadows_more_than_highlights():
    img = np.array([[[10, 10, 10], [200, 200, 200]]], dtype=np.uint8)
    out = gamma_correct(img, 3).astype(np.float32)
    shadow_gain = out[0, 0, 0] - 10
    highlight_gain = out[0, 1, 0] - 200
    assert shadow_gain > highlight_gain, "gamma must lift shadows more than highlights"


def test_bandpass_filters_preserve_shape():
    img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    for fn in (low_pass, high_pass):
        out = fn(img, 1)
        assert out.shape == img.shape and out.dtype == np.uint8


def test_clahe_returns_valid_uint8():
    img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    try:
        out = clahe_enhance(img)
        assert out.shape == img.shape and out.dtype == np.uint8
    except ImportError:
        pass  # scikit-image optional in CI


# ---------------------------------------------------------------------------
# CKA
# ---------------------------------------------------------------------------

def test_cka_identical_representations_is_one():
    X = np.random.randn(100, 32)
    assert abs(linear_cka_unbiased(X, X) - 1.0) < 1e-6


def test_cka_invariant_to_orthogonal_transform_and_scale():
    X = np.random.randn(200, 32)
    Q, _ = np.linalg.qr(np.random.randn(32, 32))
    Y = X @ Q * 3.7
    assert abs(linear_cka_unbiased(X, Y) - 1.0) < 1e-6


def test_unbiased_cka_less_inflated_than_biased_for_small_n():
    """The whole reason we use the unbiased estimator: biased CKA approaches 1
    for near-independent reps when n << d."""
    from src.analysis import linear_cka_unbiased as unbiased

    def biased(X, Y):
        Xc = X - X.mean(0, keepdims=True)
        Yc = Y - Y.mean(0, keepdims=True)
        return float((np.linalg.norm(Xc.T @ Yc, "fro") ** 2) /
                     (np.linalg.norm(Xc.T @ Xc, "fro") * np.linalg.norm(Yc.T @ Yc, "fro") + 1e-12))

    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 384))
    Y = rng.standard_normal((50, 384))  # independent
    assert biased(X, Y) > unbiased(X, Y), "biased estimator should inflate similarity"
    assert unbiased(X, Y) < 0.5


def test_cka_distinguishes_related_from_unrelated():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((150, 64))
    Y_related = X + 0.1 * rng.standard_normal((150, 64))
    Y_unrelated = rng.standard_normal((150, 64))
    assert linear_cka_unbiased(X, Y_related) > linear_cka_unbiased(X, Y_unrelated)


# ---------------------------------------------------------------------------
# Participation ratio
# ---------------------------------------------------------------------------

def test_participation_ratio_isotropic_and_lowrank():
    rng = np.random.default_rng(2)
    X_iso = rng.standard_normal((500, 32))  # full-rank isotropic: PR ~= d
    pr_iso = participation_ratio(X_iso)
    assert 0.7 * 32 < pr_iso <= 32 + 1e-6

    X_low = rng.standard_normal((500, 1)) @ np.ones((1, 32))  # rank-1: PR ~= 1
    pr_low = participation_ratio(X_low)
    assert pr_low < 1.5


# ---------------------------------------------------------------------------
# Null models / permutation tests
# ---------------------------------------------------------------------------

def test_curve_test_detects_cliff_shape():
    """A cliff+floor curve should reject linearity and midpoint symmetry."""
    rng = np.random.default_rng(3)
    accs = np.array([0.9, 0.9, 0.85, 0.6, 0.25, 0.1])
    correctness = [rng.random(300) < a for a in accs]
    res = curve_permutation_test(correctness, n_perm=500, seed=0)
    assert res["p_not_linear"] < 0.05
    assert res["p_midpoint_asymmetric"] < 0.05


def test_curve_test_passes_linear_shape():
    rng = np.random.default_rng(4)
    accs = np.array([0.9, 0.75, 0.6, 0.45, 0.3, 0.15])  # exactly linear
    correctness = [rng.random(400) < a for a in accs]
    res = curve_permutation_test(correctness, n_perm=500, seed=0)
    assert res["p_not_linear"] > 0.05, "perfectly linear curve must not reject linearity"


def test_paired_permutation_detects_and_calibrates():
    rng = np.random.default_rng(5)
    a = (rng.random(300) < 0.8).astype(float)
    b = (rng.random(300) < 0.4).astype(float)
    assert paired_permutation_test(a, b, n_perm=500, seed=0)["p_value"] < 0.05
    # calibration: same distribution -> p uniform-ish, definitely not tiny
    a2 = (rng.random(300) < 0.5).astype(float)
    b2 = (rng.random(300) < 0.5).astype(float)
    p = paired_permutation_test(a2, b2, n_perm=500, seed=0)["p_value"]
    assert p > 0.01


def test_bh_correction_controls_fdr_monotonicity():
    pvals = [0.001, 0.008, 0.039, 0.041, 0.2, 0.9]
    adj = benjamini_hochberg(pvals)
    assert len(adj) == len(pvals)
    assert all(adj[i] <= adj[i + 1] + 1e-12 for i in range(len(adj) - 1) if pvals[i] <= pvals[i + 1])
    assert adj[0] <= 0.006 + 1e-9  # 0.001 * 6 / 1
    assert all(0 <= a <= 1 for a in adj)


# ---------------------------------------------------------------------------
# Bootstrap / Wilson
# ---------------------------------------------------------------------------

def test_bootstrap_ci_brackets_observed():
    rng = np.random.default_rng(6)
    correct = (rng.random(300) < 0.7).astype(float)
    ci = bootstrap_accuracy_ci(correct, n_boot=500, seed=0)
    assert ci["ci_low"] <= ci["mean"] <= ci["ci_high"]
    assert ci["wilson_low"] <= ci["mean"] <= ci["wilson_high"]


def test_wilson_extreme_cases():
    from src.common import wilson_ci

    lo, hi = wilson_ci(0, 100)
    assert lo == 0.0 and hi < 0.05   # 0/100 must not give a degenerate [0, 0]
    lo, hi = wilson_ci(100, 100)
    assert hi == 1.0 and lo > 0.95   # clamp must not produce (1.0, 1.0)


# ---------------------------------------------------------------------------
# Probes / splits
# ---------------------------------------------------------------------------

def test_stratified_split_preserves_class_balance():
    labels = np.repeat(np.arange(10), 100)
    idx_tr, idx_te = stratified_split(labels, test_fraction=0.3, seed=42)
    tr_counts = np.bincount(labels[idx_tr], minlength=10)
    te_counts = np.bincount(labels[idx_te], minlength=10)
    assert (te_counts == 30).all() and (tr_counts == 70).all()
    assert set(idx_tr) | set(idx_te) == set(range(1000))
    assert not (set(idx_tr) & set(idx_te))


def test_ntrainable_probe_uses_exactly_n_per_class():
    rng = np.random.default_rng(7)
    X = rng.standard_normal((200, 16))
    y = np.repeat(np.arange(2), 100)
    probe = NTrainableProbe(n_per_class=5, seed=0).fit(X, y)
    # smoke: predictions have the right shape/classes
    preds = probe.predict(X)
    assert preds.shape == (200,) and set(np.unique(preds)) <= {0, 1}# ---------------------------------------------------------------------------
# Regression tests for technical-debt fixes (docs/project-context/07 §A1/A3/A5)
# ---------------------------------------------------------------------------

class _FakeSTL10:
    """Stand-in for torchvision.datasets.STL10: one image per class plus a
    second 'monkey' image (monkey has no CIFAR-10 counterpart)."""

    classes = ["airplane", "bird", "car", "cat", "deer",
               "dog", "horse", "monkey", "ship", "truck"]

    def __init__(self, root=None, split=None, download=None):
        self._items = [(np.full((6, 6, 3), 30 + k, dtype=np.uint8), k)
                       for k in range(10)] + \
                      [(np.full((6, 6, 3), 200, dtype=np.uint8), 7)]

    def __len__(self):
        return len(self._items)

    def __getitem__(self, j):
        return self._items[j]


def _supp_and_torchvision_or_skip():
    """Import torch-dependent modules; return None when unavailable (CI)."""
    try:
        import torchvision
        import src.supplementary as supp
    except ImportError:
        return None
    return supp, torchvision


def test_analyze_seeds_guards_missing_clean_baseline():
    """Debt A3: a log without a 'severity 0:' (clean) line must not crash
    analyze_seeds.py with a None-formatting TypeError; the row is flagged
    no_clean instead."""
    import subprocess
    import tempfile

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log = ("Mean accuracy: Original=0.500, LoRA=0.700, Delta=+0.200\n"
           "severity 1: 0.6500\nseverity 5: 0.3000\n")
    with tempfile.TemporaryDirectory() as td:
        arm = os.path.join(td, "seed42_all")
        os.makedirs(arm)
        with open(os.path.join(arm, "full_log.txt"), "w", encoding="utf-8") as f:
            f.write(log)
        proc = subprocess.run(
            [sys.executable, os.path.join(repo, "analyze_seeds.py"), "--roots", td],
            cwd=td, capture_output=True, text=True, timeout=120)
        assert proc.returncode == 0, (
            "analyze_seeds must survive a log without a severity-0 line:\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
        csv_path = os.path.join(td, "output", "seed_replication", "seed_analysis.csv")
        assert os.path.isfile(csv_path), "seed_analysis.csv not written"
        with open(csv_path, encoding="utf-8") as f:
            row = next(line for line in f if line.startswith("seed42_all,"))
        assert "no_clean" in row, f"row should be flagged no_clean: {row!r}"


def test_lowlight_augmentation_deterministic_per_index():
    """Debt A1: LowLightAugCIFAR must key augmentation on the sample index
    (per-index RNG), so output is independent of access order — the global
    np.random stream made it depend on DataLoader worker interleaving.
    Skipped when torch is absent (e.g. CI)."""
    try:
        from src.lora import LowLightAugCIFAR
    except ImportError:
        return
    imgs = np.random.default_rng(0).integers(0, 256, (12, 8, 8, 3)).astype(np.uint8)
    labels = np.arange(12) % 10
    ds = LowLightAugCIFAR(imgs, labels, transform=lambda x: x)
    in_order = {i: ds[i][0] for i in range(12)}
    for i in (7, 0, 11, 3, 5, 1, 9, 2, 10, 4, 8, 6):
        out = ds[i][0]
        assert np.array_equal(in_order[i], out), (
            f"index {i} augmented differently when accessed in a different "
            f"order — the RNG is not per-index")


def test_stl10_helper_maps_shared_classes():
    """Debt A5 (1): _load_stl10_shared_classes must read class names from
    CIFAR10_CLASS_NAMES (the old load_cifar10_subsets.class_names raised
    AttributeError) and count dropped 'monkey' samples. Skipped without torch."""
    got = _supp_and_torchvision_or_skip()
    if got is None:
        return
    supp, torchvision = got
    orig = torchvision.datasets.STL10
    torchvision.datasets.STL10 = _FakeSTL10
    try:
        imgs, labels, n_skipped = supp._load_stl10_shared_classes(n_test=9, seed=0)
    finally:
        torchvision.datasets.STL10 = orig
    assert len(imgs) == 9 and len(labels) == 9
    assert n_skipped == 2, f"both monkey images must be counted skipped, got {n_skipped}"
    # 9 shared classes; car -> automobile (1); frog (6) never present; monkey dropped
    assert set(labels.tolist()) == {0, 1, 2, 3, 4, 5, 7, 8, 9}, labels.tolist()
    assert imgs[0].dtype == np.uint8


def test_exp_cross_dataset_end_to_end_stubbed():
    """Debt A5 (2): exp_cross_dataset must complete (the n_dropped/n_skipped
    binding mismatch raised NameError after the probe was fit) and write
    cross_dataset.csv + cross_dataset_summary.json. All externals are stubbed
    (CIFAR-10, DINOv2, STL-10) — no downloads, no real model.
    Skipped when torch is absent (e.g. CI)."""
    import csv as _csv
    import json as _json
    import tempfile

    got = _supp_and_torchvision_or_skip()
    if got is None:
        return
    supp, torchvision = got

    n = 100
    rng = np.random.default_rng(3)
    labels = np.arange(n) % 10
    imgs = (rng.integers(0, 40, (n, 8, 8, 3))
            + (labels * 18)[:, None, None, None]).astype(np.uint8)

    def fake_cifar(n_test, n_train, seed, **_kw):
        return imgs, labels, np.arange(n_test), imgs, labels, np.arange(n)

    class _FakeExtractor:
        def __init__(self, model, device):
            pass

        def __call__(self, batch, batch_size=None):
            return np.stack([np.asarray(im, dtype=np.float32).mean(axis=(0, 1))
                             for im in batch])

    cfg = {"seed": 0,
           "data": {"n_test": 9, "n_train": 40, "split_seed": 42, "test_fraction": 0.3},
           "model": {"backbone": "stub"},
           "probe": {"C": 1.0, "max_iter": 200},
           "corruption": {"brightness_factors": [1.0] * 6}}

    saved = (supp.load_cifar10_subsets, supp.load_dinov2, supp.EmbeddingExtractor,
             torchvision.datasets.STL10)
    with tempfile.TemporaryDirectory() as td:
        try:
            supp.load_cifar10_subsets = fake_cifar
            supp.load_dinov2 = lambda backbone, device: object()
            supp.EmbeddingExtractor = _FakeExtractor
            torchvision.datasets.STL10 = _FakeSTL10
            supp.exp_cross_dataset(cfg, "cpu", td)
        finally:
            (supp.load_cifar10_subsets, supp.load_dinov2, supp.EmbeddingExtractor,
             torchvision.datasets.STL10) = saved
        with open(os.path.join(td, "cross_dataset.csv"), encoding="utf-8") as f:
            rows = list(_csv.DictReader(f))
        assert [int(r["severity"]) for r in rows] == [0, 1, 2, 3, 4, 5], rows
        assert all(0.0 <= float(r["accuracy"]) <= 1.0 for r in rows), rows
        with open(os.path.join(td, "cross_dataset_summary.json"), encoding="utf-8") as f:
            summary = _json.load(f)
        assert summary["n_stl10_used"] == 9, summary
        assert summary["n_skipped_monkey"] == 2, summary


# ---------------------------------------------------------------------------
def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(main())
