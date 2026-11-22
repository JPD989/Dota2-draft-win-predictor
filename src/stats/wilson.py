"""Wilson score confidence intervals for binomial proportions."""

from __future__ import annotations

import numpy as np


def wilson_ci(
    wins: np.ndarray | int,
    n: np.ndarray | int,
    z: float = 1.96,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized Wilson score interval for a binomial proportion.

    Parameters
    ----------
    wins : np.ndarray | int
        Number of successes, elementwise ``0 <= wins <= n``.
    n : np.ndarray | int
        Number of trials. Cells with ``n == 0`` get the trivial interval
        ``(0.0, 1.0)``.
    z : float, optional
        Normal quantile for the confidence level (default 1.96, ~95%).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(lo, hi)`` float64 bounds, broadcast to the common shape of
        ``wins`` and ``n`` (0-d arrays for scalar inputs).
    """
    wins_arr, n_arr = np.broadcast_arrays(
        np.asarray(wins, dtype=np.float64), np.asarray(n, dtype=np.float64)
    )
    n_safe = np.where(n_arr > 0, n_arr, 1.0)
    p = wins_arr / n_safe
    z2 = z * z
    denom = 1.0 + z2 / n_safe
    center = (p + z2 / (2.0 * n_safe)) / denom
    half = (z / denom) * np.sqrt(p * (1.0 - p) / n_safe + z2 / (4.0 * n_safe**2))
    lo = np.where(n_arr > 0, center - half, 0.0)
    hi = np.where(n_arr > 0, center + half, 1.0)
    return lo, hi
