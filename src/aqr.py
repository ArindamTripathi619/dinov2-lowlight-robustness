"""AQR baseline — quantile-transport recalibration (torch-free).

AQR (Mehrbod et al., arXiv:2511.03148) is the strongest published backprop-free
TTA baseline and Track 4's reviewer-facing contrast (SSR_PLAN §3/§6): it stores
source-distribution statistics once and recalibrates activations at test time,
but selects *where* by a static heuristic and applies always — no measured-drift
selection, no clean gate. This module hosts only its transport math so the
baseline can run in the same torch-free CI stack as ``src.ssr``.

Fitting uses the clean (source) calibration set and the unlabeled test batch
separately, mirroring SSR's clean-moment storage; the map itself is a
per-channel piecewise-linear quantile transform with tail clamping.
"""

import numpy as np

EPS = 1e-12
DEFAULT_N_QUANTILES = 101


def fit_channel_quantiles(activations, channel_axis=1, n_quantiles=DEFAULT_N_QUANTILES):
    """Per-channel quantiles of ``activations`` over all non-channel axes.

    Returns an (n_quantiles, C) array of non-decreasing values (q=0..100)."""
    x = np.asarray(activations, dtype=np.float64)
    moved = np.moveaxis(x, channel_axis, -1)
    flat = moved.reshape(-1, moved.shape[-1])
    qs = np.linspace(0.0, 100.0, n_quantiles)
    return np.percentile(flat, qs, axis=0)


def apply_quantile_map(values, source_q, target_q):
    """Piecewise-linear quantile transport onto the source distribution.

    ``source_q``/``target_q`` are (n_quantiles, C) arrays from
    :func:`fit_channel_quantiles`; ``values`` is channel-last (..., C). For
    x in [t_j, t_{j+1}): y = s_j + (x − t_j) · (s_{j+1} − s_j)/(t_{j+1} − t_j);
    out-of-range values clamp to [s_0, s_100]. Degenerate (flat) target
    segments fall back to s_j, which then equals the clamp in the limit."""
    src = np.asarray(source_q, dtype=np.float64)
    tgt = np.asarray(target_q, dtype=np.float64)
    if src.shape != tgt.shape or src.ndim != 2:
        raise ValueError(f"quantile arrays must both be (n_q, C), got {src.shape} vs {tgt.shape}")
    x = np.asarray(values, dtype=np.float64)
    if x.shape[-1] != src.shape[1]:
        raise ValueError(f"channel mismatch: values have {x.shape[-1]}, quantiles {src.shape[1]}")
    n_q = src.shape[0]
    idx = np.sum(x[..., None] >= tgt[:-1].T, axis=-1) - 1
    idx = np.clip(idx, 0, n_q - 2)
    channels = np.arange(src.shape[1])
    t0 = tgt[idx, channels]
    t1 = tgt[idx + 1, channels]
    s0 = src[idx, channels]
    s1 = src[idx + 1, channels]
    dt = t1 - t0
    dt = np.where(np.abs(dt) > EPS, dt, 1.0)
    out = s0 + (x - t0) * (s1 - s0) / dt
    out = np.clip(out, src[0], src[-1])
    return out.astype(np.asarray(values).dtype, copy=False)


def apply_aqr(activations, source_q, target_q, channel_axis=1):
    """Recalibrate ``activations`` (any channel axis) with the AQR map."""
    x = np.asarray(activations)
    moved = np.moveaxis(x, channel_axis, -1)
    out = apply_quantile_map(moved, source_q, target_q)
    return np.moveaxis(out, -1, channel_axis)
