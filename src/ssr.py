"""Selective statistic recalibration (SSR) — model-agnostic NumPy core.

One measurement, two consumers (DPA_DESIGN §1.4): the same drift profile that
allocates LoRA rank also selects which modules SSR recalibrates. The math here
is everything SSR does *between* hooks — clean moment bookkeeping, the
moment-matching affine, the shift gate, and drift-ranked selection. No torch:
this module must import in the torch-free CI stack (AGENTS environment notes).

Gate contract (SSR_PLAN §5): on clean inputs SSR is bit-identical to the
frozen model. That guarantee holds by construction — `recalibrate` returns the
*same array object* when the gate is closed, so no arithmetic (and no float
rounding) touches the activation path.
"""

import numpy as np

EPS = 1e-6
DEFAULT_CLIP = 4.0


def channel_stats(activations, channel_axis=1, eps=EPS):
    """Per-channel mean/std over every axis except ``channel_axis``.

    Works for conv activations (N, C, H, W), token sequences (N, L, C) and
    pooled vectors (N, C): pass the channel axis explicitly.
    """
    x = np.asarray(activations, dtype=np.float64)
    axes = tuple(i for i in range(x.ndim) if i != channel_axis)
    mean = x.mean(axis=axes)
    std = x.std(axis=axes)
    return mean, np.maximum(std, eps)


def pool_stats(pairs, weights=None, eps=EPS):
    """Pool per-batch (mean, std) pairs into one (mean, std) via the
    parallel-variance formula. ``weights`` = batch sizes (uniform if None)."""
    if not pairs:
        raise ValueError("pool_stats needs at least one batch")
    w = np.ones(len(pairs), dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    means = np.stack([p[0] for p in pairs])
    stds = np.stack([p[1] for p in pairs])
    mean = np.sum(w[:, None] * means, axis=0)
    var = np.sum(w[:, None] * (stds ** 2 + means ** 2), axis=0) - mean ** 2
    return mean, np.sqrt(np.maximum(var, 0.0) + eps)


def moment_affine(clean, test, clip=DEFAULT_CLIP):
    """Per-channel scale/shift mapping ``test`` moments onto ``clean`` ones:
    a = clip(σ_clean / σ_test), b = μ_clean − a · μ_test. ``clean``/``test``
    are (mean, std) pairs."""
    clean_mean, clean_std = clean
    test_mean, test_std = test
    a = clean_std / np.maximum(np.asarray(test_std, dtype=np.float64), EPS)
    if clip is not None:
        a = np.clip(a, 1.0 / clip, clip)
    b = clean_mean - a * test_mean
    return a, b


def apply_affine(activations, a, b, channel_axis=1):
    """y' = a · y + b, broadcast along ``channel_axis``. Output matches the
    input dtype (float32 activations stay float32)."""
    x = np.asarray(activations)
    shape = [1] * x.ndim
    shape[channel_axis] = -1
    out = x.astype(np.float64) * np.asarray(a).reshape(shape) + np.asarray(b).reshape(shape)
    return out.astype(x.dtype, copy=False)


def recalibrate(activations, a, b, open_gate, channel_axis=1):
    """Apply the affine iff the gate opened; otherwise return the input
    object untouched (bit-identity by construction, SSR_PLAN §5)."""
    if not open_gate:
        return activations
    return apply_affine(activations, a, b, channel_axis=channel_axis)


def moment_distance(clean, test, eps=EPS):
    """RMS standardized moment distance between two (mean, std) pairs:
    per-channel mean shift in units of clean σ, plus log-std ratio."""
    clean_mean, clean_std = clean
    test_mean, test_std = test
    z_mean = (np.asarray(test_mean) - np.asarray(clean_mean)) / np.maximum(np.asarray(clean_std), eps)
    z_std = np.log(np.maximum(np.asarray(test_std), eps) / np.maximum(np.asarray(clean_std), eps))
    return float(np.sqrt(np.mean(z_mean ** 2) + np.mean(z_std ** 2)))


def fit_gate(clean_distances, quantile=1.0, margin=1.0):
    """Threshold from clean-calibration distances. quantile=1.0 = max, the
    strictest no-false-positive setting."""
    d = np.asarray(clean_distances, dtype=np.float64)
    if d.size == 0:
        raise ValueError("fit_gate needs at least one clean distance")
    return float(margin * np.quantile(d, quantile))


def gate_open(distance, threshold):
    """True when the estimated shift exceeds the clean threshold."""
    return bool(distance > threshold)


def select_topk(rows, k, key="cka_unbiased_drop"):
    """Drift-ranked top-k module selection from profile-CSV rows.

    Rows keep their original (forward) order in the output; ranking only
    decides membership. Ties break toward the earlier module. k is clamped to
    [0, len(rows)]."""
    if k < 0:
        raise ValueError("k must be >= 0")
    order = sorted(range(len(rows)), key=lambda i: (-float(rows[i][key]), i))
    chosen = set(order[:min(k, len(rows))])
    return [rows[i] for i in range(len(rows)) if i in chosen]
