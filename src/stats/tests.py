"""Hypothesis tests for hero win-rate shifts (two-proportion z, BH-FDR)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.stats.synergy import hero_win_rates

if TYPE_CHECKING:
    from src.data.dataset import HeroVocab


def two_proportion_ztest(w1: int, n1: int, w2: int, n2: int) -> tuple[float, float]:
    """Two-sided two-proportion z-test with pooled variance.

    Parameters
    ----------
    w1, n1 : int
        Successes and trials in sample 1.
    w2, n2 : int
        Successes and trials in sample 2.

    Returns
    -------
    tuple[float, float]
        ``(z, p_value)``; ``z > 0`` when sample 1's proportion is higher.
        Degenerate cases (an empty sample, or pooled proportion of exactly
        0 or 1) return ``(0.0, 1.0)``.
    """
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0
    pooled = (w1 + w2) / (n1 + n2)
    se = np.sqrt(pooled * (1.0 - pooled) * (1.0 / n1 + 1.0 / n2))
    if se == 0.0:
        return 0.0, 1.0
    z = (w1 / n1 - w2 / n2) / se
    return float(z), float(2.0 * norm.sf(abs(z)))


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg step-up q-values, in the input order.

    ``q_i = min_{j: p_j >= p_i} (p_j * m / rank_j)``, clipped to [0, 1].
    """
    m = p_values.size
    order = np.argsort(p_values, kind="stable")
    ranked = p_values[order] * m / np.arange(1, m + 1)
    q_sorted = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(m, dtype=np.float64)
    q[order] = np.clip(q_sorted, 0.0, 1.0)
    return q


def hero_wr_shift_table(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    vocab: HeroVocab,
    *,
    label_a: str,
    label_b: str,
    min_games: int = 200,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Per-hero win-rate shift between two match sets with BH-FDR control.

    Filter FIRST: only heroes with at least ``min_games`` games in BOTH
    frames are tested, and the Benjamini-Hochberg correction runs over
    exactly that set (``m_tests`` = its size).

    Parameters
    ----------
    df_a, df_b : pd.DataFrame
        The two match sets to compare (e.g. two patches or rank brackets).
    vocab : HeroVocab
        Hero id <-> index mapping shared by both frames.
    label_a, label_b : str
        Human-readable names for the two sets, stored in the table.
    min_games : int, optional
        Per-frame games floor for a hero to enter the tests (default 200).
    alpha : float, optional
        FDR level for the ``significant`` flag (default 0.05).

    Returns
    -------
    pd.DataFrame
        Columns ``hero_id, label_a, label_b, games_a, games_b, wr_a, wr_b,
        delta, z, p_value, q_value, m_tests, significant`` with
        ``delta = wr_b - wr_a`` (``z`` shares its sign), sorted by
        ``|delta|`` descending.
    """
    wr_a = hero_win_rates(df_a, vocab)
    wr_b = hero_win_rates(df_b, vocab)
    keep = (wr_a["games"].to_numpy() >= min_games) & (
        wr_b["games"].to_numpy() >= min_games
    )
    a = wr_a.loc[keep].reset_index(drop=True)
    b = wr_b.loc[keep].reset_index(drop=True)
    m_tests = int(keep.sum())

    z_and_p = [
        two_proportion_ztest(int(wb), int(nb), int(wa), int(na))
        for wa, na, wb, nb in zip(a["wins"], a["games"], b["wins"], b["games"])
    ]
    z = np.array([zp[0] for zp in z_and_p], dtype=np.float64)
    p_value = np.array([zp[1] for zp in z_and_p], dtype=np.float64)
    q_value = _benjamini_hochberg(p_value)

    table = pd.DataFrame(
        {
            "hero_id": a["hero_id"],
            "label_a": label_a,
            "label_b": label_b,
            "games_a": a["games"],
            "games_b": b["games"],
            "wr_a": a["wr"],
            "wr_b": b["wr"],
            "delta": b["wr"] - a["wr"],
            "z": z,
            "p_value": p_value,
            "q_value": q_value,
            "m_tests": m_tests,
            "significant": q_value <= alpha,
        }
    )
    order = table["delta"].abs().sort_values(ascending=False).index
    return table.reindex(order).reset_index(drop=True)
