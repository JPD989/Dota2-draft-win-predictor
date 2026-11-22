"""Scalar evaluation metrics and bootstrap confidence intervals.

All bootstrap functions draw resample indices from
``numpy.random.default_rng(seed)`` in a fixed order, so the resample set is a
pure function of ``(seed, n, n_boot)``. :func:`bootstrap_auc_matrix` is the
core: for every replicate it draws ONE index set and scores every model on it,
which is what makes the paired delta-AUC comparison correlated (and tight).

Degenerate resamples (a single class drawn — AUC undefined) are dropped from
the percentile computation; if every replicate is degenerate the CI is NaN.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import rankdata

from src.config import SEED
from src.eval.calibration import _check_yp, expected_calibration_error

__all__ = [
    "auc_score",
    "bootstrap_auc_ci",
    "bootstrap_auc_matrix",
    "compute_metrics",
    "paired_bootstrap_delta_auc",
    "percentile_ci",
]

_LL_EPS: float = 1e-15


def auc_score(y: ArrayLike, p: ArrayLike) -> float:
    """ROC AUC via the Mann–Whitney rank statistic (ties get average rank).

    Matches ``sklearn.metrics.roc_auc_score`` but returns NaN instead of
    raising when only one class is present.

    Parameters
    ----------
    y, p : array-like
        Binary labels and scores, shape ``(N,)``.

    Returns
    -------
    float
        AUC in [0, 1], or NaN if ``y`` has a single class.
    """
    y_arr = np.asarray(y, dtype=np.float64)
    n_pos = float(y_arr.sum())
    n_neg = float(y_arr.size) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(np.asarray(p, dtype=np.float64))
    return float((ranks[y_arr == 1.0].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def compute_metrics(y: ArrayLike, p: ArrayLike) -> dict[str, float]:
    """Headline metrics for one set of predictions.

    Parameters
    ----------
    y : array-like
        Binary labels (radiant win), shape ``(N,)``.
    p : array-like
        Predicted P(radiant win), shape ``(N,)``.

    Returns
    -------
    dict[str, float]
        ``auc`` (NaN if single class), ``log_loss`` (p clipped to
        [1e-15, 1 - 1e-15]), ``brier``, ``ece`` (15 equal-width bins),
        ``ece_q`` (15 equal-mass bins), ``n`` (int). Both ECEs come from
        :func:`src.eval.calibration.expected_calibration_error`.
    """
    y_arr, p_arr = _check_yp(y, p)
    q = np.clip(p_arr, _LL_EPS, 1.0 - _LL_EPS)
    log_loss = -np.mean(y_arr * np.log(q) + (1.0 - y_arr) * np.log(1.0 - q))
    return {
        "auc": auc_score(y_arr, p_arr),
        "log_loss": float(log_loss),
        "brier": float(np.mean((p_arr - y_arr) ** 2)),
        "ece": expected_calibration_error(y_arr, p_arr, n_bins=15, strategy="uniform"),
        "ece_q": expected_calibration_error(y_arr, p_arr, n_bins=15, strategy="quantile"),
        "n": int(y_arr.size),
    }


def bootstrap_auc_matrix(
    y: ArrayLike, preds: Mapping[str, ArrayLike], n_boot: int = 1000, seed: int = SEED
) -> tuple[list[str], np.ndarray]:
    """Bootstrap AUCs of several models on SHARED row resamples.

    For each replicate ``b`` one index set ``idx_b`` (n draws with
    replacement) is drawn and every model is scored on ``(y[idx_b], p[idx_b])``.

    Parameters
    ----------
    y : array-like
        Binary labels, shape ``(N,)``.
    preds : Mapping[str, array-like]
        Model name -> predictions, each shape ``(N,)``.
    n_boot : int, optional
        Number of replicates, by default 1000.
    seed : int, optional
        RNG seed, by default ``SEED``.

    Returns
    -------
    tuple[list[str], np.ndarray]
        Model names (in ``preds`` order) and an ``(n_boot, n_models)`` AUC
        array; rows for degenerate (single-class) resamples are all NaN.
    """
    if n_boot < 1:
        raise ValueError(f"n_boot must be >= 1, got {n_boot}")
    names = list(preds)
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    p_mat = np.column_stack([np.asarray(preds[k], dtype=np.float64).ravel() for k in names])
    if p_mat.shape[0] != y_arr.size:
        raise ValueError("predictions and labels have different lengths")
    rng = np.random.default_rng(seed)
    n = y_arr.size
    out = np.full((n_boot, len(names)), np.nan)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y_arr[idx]
        n_pos = yb.sum()
        n_neg = n - n_pos
        if n_pos == 0 or n_neg == 0:
            continue  # degenerate resample: AUC undefined for every model alike
        ranks = rankdata(p_mat[idx], axis=0)
        out[b] = (ranks[yb == 1.0].sum(axis=0) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return names, out


def percentile_ci(samples: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    """Percentile interval of bootstrap samples, ignoring NaN (degenerate) replicates.

    Returns ``(nan, nan)`` when no replicate is valid.
    """
    s = np.asarray(samples, dtype=np.float64)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return float("nan"), float("nan")
    alpha = (1.0 - level) / 2.0 * 100.0
    lo, hi = np.percentile(s, [alpha, 100.0 - alpha])
    return float(lo), float(hi)


def bootstrap_auc_ci(
    y: ArrayLike, p: ArrayLike, n_boot: int = 1000, seed: int = SEED
) -> tuple[float, float]:
    """Percentile 95% CI on AUC by row resampling.

    Parameters
    ----------
    y, p : array-like
        Labels and predictions, shape ``(N,)``.
    n_boot : int, optional
        Number of replicates, by default 1000.
    seed : int, optional
        RNG seed, by default ``SEED``.

    Returns
    -------
    tuple[float, float]
        ``(lo, hi)``; NaN if every resample was single-class.
    """
    _, aucs = bootstrap_auc_matrix(y, {"p": p}, n_boot=n_boot, seed=seed)
    return percentile_ci(aucs[:, 0])


def paired_bootstrap_delta_auc(
    y: ArrayLike, p_a: ArrayLike, p_b: ArrayLike, n_boot: int = 1000, seed: int = SEED
) -> tuple[float, float, float]:
    """Paired bootstrap for ``AUC(a) − AUC(b)`` on the same test rows.

    ONE index resample per replicate is applied to both models, so the
    interval reflects the correlated comparison (shared test-set noise
    cancels) rather than two independent CIs.

    Parameters
    ----------
    y : array-like
        Labels, shape ``(N,)``.
    p_a, p_b : array-like
        Predictions of models a and b on the same rows.
    n_boot : int, optional
        Number of replicates, by default 1000.
    seed : int, optional
        RNG seed, by default ``SEED``.

    Returns
    -------
    tuple[float, float, float]
        ``(delta, lo, hi)``: point estimate on the full data and the
        percentile 95% interval of the replicate deltas.
    """
    _, aucs = bootstrap_auc_matrix(y, {"a": p_a, "b": p_b}, n_boot=n_boot, seed=seed)
    delta = auc_score(y, p_a) - auc_score(y, p_b)
    lo, hi = percentile_ci(aucs[:, 0] - aucs[:, 1])
    return float(delta), lo, hi
