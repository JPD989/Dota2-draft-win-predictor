"""Calibration analysis: reliability curves, ECE, and post-hoc recalibrators.

This module holds the project's single ECE implementation
(:func:`expected_calibration_error`); ``src.eval.metrics`` imports it. It must
not import anything from ``metrics.py`` (no import cycle).

Recalibrators are fit on **validation** predictions and applied to test
predictions (docs/architecture.md §3). Their outputs are clipped to
``[1e-6, 1 - 1e-6]`` so log-loss stays finite even when test predictions are
more extreme than anything seen in validation.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from src.stats.wilson import wilson_ci

__all__ = [
    "expected_calibration_error",
    "fit_isotonic",
    "fit_platt",
    "plot_reliability",
    "reliability_curve",
]

_EPS: float = 1e-6
_STRATEGIES = ("uniform", "quantile")
CURVE_COLUMNS: list[str] = ["bin_mid", "p_mean", "y_rate", "count", "ci_lo", "ci_hi"]


def _check_yp(y: ArrayLike, p: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Validate and convert labels / probabilities to 1-D float64 arrays."""
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    p_arr = np.asarray(p, dtype=np.float64).ravel()
    if y_arr.shape != p_arr.shape:
        raise ValueError(f"y and p length mismatch: {y_arr.shape[0]} vs {p_arr.shape[0]}")
    if y_arr.size == 0:
        raise ValueError("empty input")
    if not np.all(np.isfinite(p_arr)) or p_arr.min() < 0.0 or p_arr.max() > 1.0:
        raise ValueError("probabilities must be finite and in [0, 1]")
    if not np.all((y_arr == 0.0) | (y_arr == 1.0)):
        raise ValueError("labels must be 0/1")
    return y_arr, p_arr


def _bin_edges(p: np.ndarray, n_bins: int, strategy: str) -> np.ndarray:
    """Bin edges over [0, 1]; quantile edges are deduplicated (ties collapse bins)."""
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    if strategy == "uniform":
        return np.linspace(0.0, 1.0, n_bins + 1)
    if strategy == "quantile":
        edges = np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1))
        edges[0], edges[-1] = 0.0, 1.0
        return np.unique(edges)
    raise ValueError(f"strategy must be one of {_STRATEGIES}, got {strategy!r}")


def reliability_curve(
    y: ArrayLike, p: ArrayLike, n_bins: int = 15, strategy: str = "uniform"
) -> pd.DataFrame:
    """Binned reliability curve with Wilson 95% intervals on the empirical rate.

    Parameters
    ----------
    y : array-like
        Binary labels (radiant win as 0/1), shape ``(N,)``.
    p : array-like
        Predicted P(radiant win), shape ``(N,)``.
    n_bins : int, optional
        Number of bins, by default 15.
    strategy : {"uniform", "quantile"}, optional
        ``"uniform"``: equal-width bins on [0, 1]. ``"quantile"``: equal-mass
        bins from the quantiles of ``p`` (tied quantiles merge bins).

    Returns
    -------
    pd.DataFrame
        One row per **non-empty** bin, columns ``bin_mid`` (midpoint of the
        bin edges), ``p_mean`` (mean prediction), ``y_rate`` (empirical win
        rate), ``count``, ``ci_lo``, ``ci_hi`` (Wilson 95% on ``y_rate``).
    """
    y_arr, p_arr = _check_yp(y, p)
    edges = _bin_edges(p_arr, n_bins, strategy)
    k = len(edges) - 1
    # Right-closed last bin: p == 1.0 lands in bin k-1.
    idx = np.clip(np.searchsorted(edges, p_arr, side="right") - 1, 0, k - 1)
    count = np.bincount(idx, minlength=k)
    p_sum = np.bincount(idx, weights=p_arr, minlength=k)
    y_sum = np.bincount(idx, weights=y_arr, minlength=k)
    keep = count > 0
    lo, hi = wilson_ci(y_sum[keep], count[keep])
    return pd.DataFrame(
        {
            "bin_mid": ((edges[:-1] + edges[1:]) / 2.0)[keep],
            "p_mean": p_sum[keep] / count[keep],
            "y_rate": y_sum[keep] / count[keep],
            "count": count[keep].astype(np.int64),
            "ci_lo": np.asarray(lo, dtype=np.float64),
            "ci_hi": np.asarray(hi, dtype=np.float64),
        }
    )


def expected_calibration_error(
    y: ArrayLike, p: ArrayLike, n_bins: int = 15, strategy: str = "uniform"
) -> float:
    """Expected calibration error: ``Σ_b (n_b / N) · |y_rate_b − p_mean_b|``.

    The project's only ECE implementation (``metrics.compute_metrics`` calls
    it with ``strategy="uniform"`` for ``ece`` and ``"quantile"`` for ``ece_q``).

    Parameters
    ----------
    y, p : array-like
        Labels and predicted probabilities, shape ``(N,)``.
    n_bins : int, optional
        Number of bins, by default 15.
    strategy : {"uniform", "quantile"}, optional
        Binning strategy, see :func:`reliability_curve`.

    Returns
    -------
    float
        ECE in [0, 1].
    """
    curve = reliability_curve(y, p, n_bins=n_bins, strategy=strategy)
    w = curve["count"].to_numpy(dtype=np.float64)
    gap = np.abs(curve["y_rate"].to_numpy() - curve["p_mean"].to_numpy())
    return float(np.sum(w * gap) / w.sum())


def _logit(p: np.ndarray) -> np.ndarray:
    q = np.clip(p, _EPS, 1.0 - _EPS)
    return np.log(q / (1.0 - q))


def fit_platt(y_val: ArrayLike, p_val: ArrayLike) -> Callable[[np.ndarray], np.ndarray]:
    """Fit Platt scaling ``sigmoid(a · logit(p) + b)`` on validation predictions.

    Unregularized logistic regression on the logit of the (clipped) input
    probability. Logit-space input means out-of-range test predictions
    extrapolate smoothly; outputs are clipped to ``[1e-6, 1 - 1e-6]``.

    Parameters
    ----------
    y_val, p_val : array-like
        Validation labels and uncalibrated predictions.

    Returns
    -------
    Callable[[np.ndarray], np.ndarray]
        Maps raw probabilities to calibrated float64 probabilities.
    """
    y_arr, p_arr = _check_yp(y_val, p_val)
    if np.unique(y_arr).size < 2:
        raise ValueError("fit_platt needs both classes in y_val")
    lr = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)  # ~unregularized, any sklearn
    lr.fit(_logit(p_arr).reshape(-1, 1), y_arr.astype(int))
    a, b = float(lr.coef_[0, 0]), float(lr.intercept_[0])

    def _apply(p: np.ndarray) -> np.ndarray:
        z = a * _logit(np.asarray(p, dtype=np.float64)) + b
        return np.clip(1.0 / (1.0 + np.exp(-z)), _EPS, 1.0 - _EPS)

    return _apply


def fit_isotonic(y_val: ArrayLike, p_val: ArrayLike) -> Callable[[np.ndarray], np.ndarray]:
    """Fit isotonic recalibration on validation predictions.

    Inputs outside the validation range are clip-extrapolated
    (``out_of_bounds="clip"``) and outputs are clipped to
    ``[1e-6, 1 - 1e-6]`` — isotonic regression otherwise returns exact 0/1 at
    its range ends, which makes log-loss infinite.

    Parameters
    ----------
    y_val, p_val : array-like
        Validation labels and uncalibrated predictions.

    Returns
    -------
    Callable[[np.ndarray], np.ndarray]
        Maps raw probabilities to calibrated float64 probabilities.
    """
    y_arr, p_arr = _check_yp(y_val, p_val)
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
    iso.fit(p_arr, y_arr)

    def _apply(p: np.ndarray) -> np.ndarray:
        out = iso.predict(np.asarray(p, dtype=np.float64).ravel())
        return np.clip(out.astype(np.float64), _EPS, 1.0 - _EPS)

    return _apply


def plot_reliability(curves: dict[str, pd.DataFrame], out_path: Path, *,
                     min_count: int = 50) -> None:
    """Plot reliability curves (one line per model) with Wilson error bars to a PNG.

    Uses the object-oriented ``matplotlib.figure.Figure`` API (no pyplot
    state), so it is safe in headless runs.

    Parameters
    ----------
    curves : dict[str, pd.DataFrame]
        Model name -> output of :func:`reliability_curve`.
    out_path : Path
        Destination ``.png``; parent directories are created.
    min_count : int, optional
        Bins with fewer rows are not drawn (their intervals span most of the
        axis and hide the populated range). Falls back to all bins if a curve
        would otherwise be empty. Axes zoom to the drawn range. Default 50.
    """
    from matplotlib.figure import Figure

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = Figure(figsize=(6.5, 6.0))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1, label="perfect")
    lo, hi = 0.5, 0.5
    for name, curve in curves.items():
        dense = curve[curve["count"] >= min_count]
        curve = dense if len(dense) else curve
        lo = min(lo, curve["p_mean"].min(), curve["ci_lo"].min())
        hi = max(hi, curve["p_mean"].max(), curve["ci_hi"].max())
        yerr = np.vstack(
            [curve["y_rate"] - curve["ci_lo"], curve["ci_hi"] - curve["y_rate"]]
        ).clip(min=0.0)
        ax.errorbar(
            curve["p_mean"], curve["y_rate"], yerr=yerr, marker="o", markersize=3,
            linewidth=1.2, capsize=2, label=name,
        )
    ax.set_xlabel("mean predicted P(radiant win)")
    ax.set_ylabel("empirical radiant win rate")
    ax.set_title("Reliability (Wilson 95% CI per bin)")
    lo, hi = max(0.0, lo - 0.03), min(1.0, hi + 0.03)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
