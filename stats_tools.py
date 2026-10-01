"""Statistical machinery ported from upstream (officialarghya29) src/analysis.py.

Self-contained: imports only numpy/scipy/sklearn — deliberately torch-free so
analysis and CI environments can use it without the ML stack. Ported and
credited per docs/DIVERGENCE_REPORT.md reconciliation plan; added here:
readout-collapse metrics (top-1 share, entropy ratio) implementing upstream's
prediction-distribution analysis (their F6/H6) on our evaluation protocol.

Contents (all upstream unless noted):
- linear_cka_unbiased: bias-corrected linear CKA (Kornblith et al. 2019, App. A)
- participation_ratio: effective dimensionality of the embedding cloud
- curve_permutation_test: linearity + midpoint-symmetry null tests
- paired_permutation_test: exact paired permutation test
- bootstrap_accuracy_ci_correct / wilson_ci: correctness-vector intervals
- benjamini_hochberg: FDR control
- readout_collapse_metrics: [OURS] top-1 share + normalized prediction entropy
  per severity — exposes the constant-class collapse that per-class accuracy
  alone hides (upstream F6: 99% "frog" at sev-5, entropy ratio 0.02).
"""

import numpy as np
from scipy.stats import entropy as _sp_entropy


# ---------------------------------------------------------------------------
# Representation similarity (upstream src/analysis.py, verbatim algorithm)
# ---------------------------------------------------------------------------
def linear_cka_unbiased(X: np.ndarray, Y: np.ndarray) -> float:
    """Unbiased estimator of linear CKA (Kornblith et al. 2019, Appendix A).

    Uses only the OFF-DIAGONAL entries of the feature-centered Gram matrices:

        CKA_u = sum_{i!=j} K_ij L_ij / sqrt( sum_{i!=j} K_ij^2 * sum_{i!=j} L_ij^2 )

    The standard (biased) estimator includes the diagonal (self-similarity)
    terms, which inflates similarity toward 1 when n is small relative to d —
    at n=300, d=384 the bias is material. Our CKA artifacts were computed at
    n=500, d=384 (bias small but nonzero); cka_unbiased_recompute.py quantifies
    the delta on the exact artifact of record.
    """
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)
    n = X.shape[0]
    Xc = X - X.mean(0, keepdims=True)
    Yc = Y - Y.mean(0, keepdims=True)
    K = Xc @ Xc.T
    L = Yc @ Yc.T
    off = ~np.eye(n, dtype=bool)
    hsic = float((K[off] * L[off]).sum())
    var_x = float((K[off] ** 2).sum())
    var_y = float((L[off] ** 2).sum())
    return float(hsic / np.sqrt(var_x * var_y + 1e-12))


def participation_ratio(X: np.ndarray) -> float:
    """PR = (sum lambda)^2 / sum lambda^2 of covariance eigenvalues.

    Effective number of dimensions carrying variance. A drop means the
    embedding cloud collapses toward a low-dim manifold; stability with low
    cosine-to-clean means rotation instead. (Upstream uses this to separate
    'collapse' from 'rotation' — exactly our Phase-1 cos-sim story.)
    """
    S = np.cov(X.astype(np.float64).T)
    eig = np.linalg.eigvalsh(S)[::-1]
    eig = np.clip(eig, 0, None)
    return float((eig.sum() ** 2) / ((eig ** 2).sum() + 1e-12))


# ---------------------------------------------------------------------------
# Null models + permutation tests (upstream, verbatim algorithms)
# ---------------------------------------------------------------------------
def _curve_ssr(acc: np.ndarray) -> float:
    s = np.arange(len(acc), dtype=np.float64)
    A = np.stack([np.ones_like(s), s], axis=1)
    coef, *_ = np.linalg.lstsq(A, acc, rcond=None)
    resid = acc - A @ coef
    return float(np.sum(resid ** 2))


def _curve_midpoint_gap(acc: np.ndarray) -> float:
    return float(abs(acc[2] - acc[4]))


def curve_permutation_test(correctness_by_severity, n_perm: int = 5000, seed: int = 42) -> dict:
    """Parametric-bootstrap null tests for the severity curve.

    Test 1 (non-linearity): under H0 ('the curve IS linear'), simulate
    correctness vectors as Bernoulli(p_lin(s)) and compute the null
    distribution of SSR. p = P(SSR_null >= SSR_obs).
    Test 2 (midpoint asymmetry): H0 = acc(2) == acc(4), pooled Bernoulli.
    """
    rng = np.random.default_rng(seed)
    corr = np.stack([np.asarray(c, dtype=float) for c in correctness_by_severity])
    n = corr.shape[1]
    accs = corr.mean(axis=1)

    obs_ssr = _curve_ssr(accs)
    s = np.arange(len(accs), dtype=np.float64)
    A = np.stack([np.ones_like(s), s], axis=1)
    coef, *_ = np.linalg.lstsq(A, accs, rcond=None)
    p_lin = np.clip(A @ coef, 0.0, 1.0)

    null_ssr = np.empty(n_perm)
    for i in range(n_perm):
        sim = np.stack([(rng.random(n) < p) for p in p_lin]).astype(float)
        null_ssr[i] = _curve_ssr(sim.mean(axis=1))

    p_pool = (corr[2].sum() + corr[4].sum()) / (2 * n)
    obs_gap = _curve_midpoint_gap(accs)
    null_gap = np.empty(n_perm)
    for i in range(n_perm):
        a = (rng.random(n) < p_pool).mean()
        b = (rng.random(n) < p_pool).mean()
        null_gap[i] = abs(a - b)

    return {
        "observed_ssr": obs_ssr,
        "p_not_linear": float((np.sum(null_ssr >= obs_ssr) + 1) / (n_perm + 1)),
        "observed_midpoint_gap": obs_gap,
        "p_midpoint_asymmetric": float((np.sum(null_gap >= obs_gap) + 1) / (n_perm + 1)),
        "n_perm": n_perm,
    }


def paired_permutation_test(scores_a: np.ndarray, scores_b: np.ndarray,
                            n_perm: int = 5000, seed: int = 42) -> dict:
    """Exact paired permutation test on per-example scores. H0: A and B are
    exchangeable per example."""
    rng = np.random.default_rng(seed)
    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)
    diff = scores_a.mean() - scores_b.mean()
    n = len(scores_a)
    count = 0
    for _ in range(n_perm):
        mask = rng.random(n) < 0.5
        d = (np.where(mask, scores_a, scores_b) - np.where(mask, scores_b, scores_a)).mean()
        if abs(d) >= abs(diff):
            count += 1
    return {
        "observed_diff": float(diff),
        "p_value": float((count + 1) / (n_perm + 1)),
        "n_perm": n_perm,
        "n": n,
    }


def bootstrap_accuracy_ci_correct(correct: np.ndarray, n_boot: int = 2000,
                                  ci: int = 95, seed: int = 42) -> dict:
    """Bootstrap CI over a correctness vector (0/1). Upstream variant — takes
    per-example correctness rather than our utils.bootstrap_accuracy_ci's
    embeddings+probe signature; used by the readout-repair runner."""
    rng = np.random.default_rng(seed)
    correct = np.asarray(correct, dtype=float)
    n = len(correct)
    means = np.empty(n_boot)
    for i in range(n_boot):
        means[i] = correct[rng.integers(0, n, n)].mean()
    alpha = (100 - ci) / 2
    return {
        "mean": float(correct.mean()),
        "ci_low": float(np.percentile(means, alpha)),
        "ci_high": float(np.percentile(means, 100 - alpha)),
        "n": n,
    }


def wilson_ci(correct: np.ndarray, ci: int = 95) -> dict:
    """Wilson score interval for a proportion — better behaved than the
    normal approximation at extreme accuracies (the sev-5 floor)."""
    correct = np.asarray(correct, dtype=float)
    n = len(correct)
    p = correct.mean()
    z = 1.959963984540054 if ci == 95 else 2.5758293035489004
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return {"mean": float(p), "ci_low": float(center - half), "ci_high": float(center + half)}


def benjamini_hochberg(pvals) -> list:
    """BH-FDR adjusted p-values (upstream)."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    return list(out)


# ---------------------------------------------------------------------------
# Readout-collapse metrics [OURS — implements upstream's F6/H6 analysis]
# ---------------------------------------------------------------------------
def readout_collapse_metrics(y_true, y_pred, n_classes: int = 10) -> dict:
    """Prediction-distribution analysis per evaluation fold.

    Returns:
        top1_share: fraction of predictions going to the single most-predicted
            class (upstream: 28% -> 72% -> 99% across sev 3/4/5 — the
            constant-class 'frog' collapse).
        n_classes_used: number of distinct classes predicted.
        entropy_ratio: normalized Shannon entropy of the prediction
            distribution (1.0 = uniform use of classes; 0.02 at upstream's
            sev-5 collapse).
        dominant_class: the class absorbing the predictions.
        accuracy: (convenience) agreement with y_true.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    counts = np.bincount(y_pred, minlength=n_classes).astype(float)
    dist = counts / counts.sum()
    top1_share = float(dist.max())
    n_used = int((counts > 0).sum())
    ent = float(_sp_entropy(dist, base=n_classes)) if n_classes > 1 else 0.0
    return {
        "top1_share": top1_share,
        "n_classes_used": n_used,
        "entropy_ratio": ent,
        "dominant_class": int(dist.argmax()),
        "accuracy": float((y_true == y_pred).mean()),
    }
