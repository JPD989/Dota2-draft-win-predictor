"""Hero win rates and pairwise synergy / counter matrices.

All pair scores live in log-odds space: ``logit(p) = log(p / (1 - p))`` with
``p`` clipped to ``[1e-3, 1 - 1e-3]`` first. Raw scores are shrunk toward a
zero-mean prior with empirical-Bayes precision weights (:func:`_eb_shrink`).

Caveat for the write-up: marginal win rates attenuate the underlying per-hero
logistic coefficients, so "synergy = interaction coefficient" is an
approximation, not an identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.config import MIN_PAIR_GAMES
from src.stats.wilson import wilson_ci

if TYPE_CHECKING:
    from src.data.dataset import HeroVocab

_CLIP: float = 1e-3
_TEAM_SIZE: int = 5


def _logit(p: np.ndarray) -> np.ndarray:
    """Log-odds with the pinned clipping of ``p`` to ``[1e-3, 1 - 1e-3]``."""
    p = np.clip(p, _CLIP, 1.0 - _CLIP)
    return np.log(p / (1.0 - p))


def _team_indices(
    df: pd.DataFrame, vocab: HeroVocab
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert team columns to vocab-index matrices.

    Assumes full drafts (5 heroes per team), which holds for all collected
    matches; partial drafts are a recommender-path concern only.

    Parameters
    ----------
    df : pd.DataFrame
        Matches with ``radiant_team``, ``dire_team``, ``radiant_win`` columns.
    vocab : HeroVocab
        Hero id <-> index mapping.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(r_idx, d_idx, y)``: two ``(N, 5)`` int64 vocab-index matrices and
        the ``(N,)`` bool radiant-win vector.

    Raises
    ------
    KeyError
        If any hero id is not present in ``vocab``.
    """
    if len(df) == 0:
        empty = np.empty((0, _TEAM_SIZE), dtype=np.int64)
        return empty, empty.copy(), np.empty(0, dtype=bool)
    r = np.asarray(df["radiant_team"].tolist(), dtype=np.int64)
    d = np.asarray(df["dire_team"].tolist(), dtype=np.int64)
    y = df["radiant_win"].to_numpy(dtype=bool)

    hero_ids = np.asarray(vocab.hero_ids, dtype=np.int64)
    lut = np.full(int(hero_ids.max()) + 1, -1, dtype=np.int64)
    lut[hero_ids] = np.arange(hero_ids.size, dtype=np.int64)

    both = np.concatenate([r, d], axis=1)
    out_of_range = (both < 0) | (both >= lut.size)
    mapped = lut[np.where(out_of_range, 0, both)]
    bad = out_of_range | (mapped < 0)
    if bad.any():
        raise KeyError(f"hero ids not in vocab: {sorted(set(both[bad].tolist()))}")
    return mapped[:, :_TEAM_SIZE], mapped[:, _TEAM_SIZE:], y


def _hero_counts(
    r_idx: np.ndarray, d_idx: np.ndarray, y: np.ndarray, size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per-hero ``(games, wins)`` int64 vectors, pooled over both sides."""
    y_f = y.astype(np.float64)
    games = np.bincount(r_idx.ravel(), minlength=size) + np.bincount(
        d_idx.ravel(), minlength=size
    )
    wins = np.bincount(
        r_idx.ravel(), weights=np.repeat(y_f, _TEAM_SIZE), minlength=size
    ) + np.bincount(
        d_idx.ravel(), weights=np.repeat(1.0 - y_f, _TEAM_SIZE), minlength=size
    )
    return games.astype(np.int64), np.rint(wins).astype(np.int64)


def _hero_logit_wr(games: np.ndarray, wins: np.ndarray) -> np.ndarray:
    """Per-hero clipped logit win rate; heroes with 0 games get logit(0.5) = 0."""
    wr = np.divide(
        wins, games, out=np.full(games.shape, 0.5), where=games > 0
    )
    return _logit(wr)


def hero_win_rates(df: pd.DataFrame, vocab: HeroVocab) -> pd.DataFrame:
    """Per-hero win rates pooled over both sides.

    Parameters
    ----------
    df : pd.DataFrame
        Matches (see :func:`_team_indices` for required columns).
    vocab : HeroVocab
        Hero id <-> index mapping; every vocab hero gets a row.

    Returns
    -------
    pd.DataFrame
        Columns ``hero_id, games, wins, wr, ci_lo, ci_hi`` (Wilson 95%),
        sorted by ``hero_id``. Heroes with 0 games have ``wr = NaN`` and the
        trivial interval ``(0, 1)``.
    """
    r_idx, d_idx, y = _team_indices(df, vocab)
    games, wins = _hero_counts(r_idx, d_idx, y, vocab.size)
    wr = np.divide(wins, games, out=np.full(vocab.size, np.nan), where=games > 0)
    ci_lo, ci_hi = wilson_ci(wins, games)
    return pd.DataFrame(
        {
            "hero_id": vocab.hero_ids,
            "games": games,
            "wins": wins,
            "wr": wr,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
        }
    )


@dataclass(frozen=True)
class PairStats:
    """Pairwise hero statistics in log-odds units.

    Attributes
    ----------
    value : np.ndarray
        ``(V, V)`` float64 EB-shrunk score; exactly 0 where the cell has
        fewer than ``min_games`` pair games.
    raw : np.ndarray
        ``(V, V)`` float64 unshrunk score; NaN where invalid.
    games : np.ndarray
        ``(V, V)`` int64 pair sample sizes (total over orientations).
    kind : str
        ``"synergy"`` or ``"counter"``.
    """

    value: np.ndarray
    raw: np.ndarray
    games: np.ndarray
    kind: str

    def save(self, path: Path) -> None:
        """Write ``value``, ``raw``, ``games`` and ``kind`` to a ``.npz`` file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            value=self.value,
            raw=self.raw,
            games=self.games,
            kind=np.asarray(self.kind),
        )

    @classmethod
    def load(cls, path: Path) -> PairStats:
        """Load a :class:`PairStats` written by :meth:`save`."""
        with np.load(path) as data:
            return cls(
                value=data["value"],
                raw=data["raw"],
                games=data["games"],
                kind=data["kind"].item(),
            )


def _eb_shrink(raw: np.ndarray, se2: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Empirical-Bayes shrinkage toward the zero-mean prior (pinned recipe).

    Over valid cells only, each unordered pair counted once, with weights
    ``w = 1/se2``::

        tau2  = max( sum(w*raw^2)/sum(w) - sum(w*se2)/sum(w), 0 )
        value = raw * tau2 / (tau2 + se2)

    Invalid cells get value 0 exactly.
    """
    iu = np.triu_indices(raw.shape[0], k=1)
    mask = valid[iu]
    tau2 = 0.0
    if mask.any():
        r = raw[iu][mask]
        s = se2[iu][mask]
        w = 1.0 / s
        sw = float(w.sum())
        tau2 = max(float((w * r * r).sum() / sw - (w * s).sum() / sw), 0.0)
    value = np.zeros_like(raw)
    value[valid] = raw[valid] * (tau2 / (tau2 + se2[valid]))
    return value


def synergy_matrix(
    df: pd.DataFrame, vocab: HeroVocab, *, min_games: int = MIN_PAIR_GAMES
) -> PairStats:
    """Same-team synergy matrix, pooled over both sides.

    For heroes ``a, b`` on the same team::

        raw[a, b] = logit(pair_wr_ab) - logit(wr_a) - logit(wr_b)

    (hero win rates are pooled over both sides, so the base-rate logit is 0
    and cancels). The matrix is symmetric with a zero diagonal. Cells with
    fewer than ``min_games`` pair games are NaN in ``raw`` and 0 in ``value``.

    Parameters
    ----------
    df : pd.DataFrame
        Matches with full 5+5 drafts.
    vocab : HeroVocab
        Hero id <-> index mapping defining the ``(V, V)`` layout.
    min_games : int, optional
        Validity floor on total pair games (default ``MIN_PAIR_GAMES``).

    Returns
    -------
    PairStats
        EB-shrunk synergy scores, ``kind="synergy"``.
    """
    size = vocab.size
    r_idx, d_idx, y = _team_indices(df, vocab)
    games_h, wins_h = _hero_counts(r_idx, d_idx, y, size)
    m = _hero_logit_wr(games_h, wins_h)

    ii, jj = np.triu_indices(_TEAM_SIZE, k=1)  # the 10 within-team pairs
    y_f = y.astype(np.float64)
    flat_parts: list[np.ndarray] = []
    win_parts: list[np.ndarray] = []
    for team, team_win in ((r_idx, y_f), (d_idx, 1.0 - y_f)):
        a = team[:, ii]
        b = team[:, jj]
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)
        flat_parts.append((lo * size + hi).ravel())
        win_parts.append(np.repeat(team_win, ii.size))
    flat = np.concatenate(flat_parts)
    win_w = np.concatenate(win_parts)
    upper_games = np.bincount(flat, minlength=size * size).reshape(size, size)
    upper_wins = np.bincount(flat, weights=win_w, minlength=size * size).reshape(
        size, size
    )
    pair_games = upper_games + upper_games.T
    pair_wins = upper_wins + upper_wins.T

    valid = pair_games >= min_games
    np.fill_diagonal(valid, False)
    pair_wr = np.divide(
        pair_wins, pair_games, out=np.full((size, size), 0.5), where=pair_games > 0
    )
    raw = np.full((size, size), np.nan)
    raw_all = _logit(pair_wr) - (m[:, None] + m[None, :])
    raw[valid] = raw_all[valid]

    p_hat = np.clip(pair_wr, _CLIP, 1.0 - _CLIP)
    se2 = 1.0 / (np.maximum(pair_games, 1) * p_hat * (1.0 - p_hat))
    value = _eb_shrink(raw, se2, valid)
    return PairStats(
        value=value, raw=raw, games=pair_games.astype(np.int64), kind="synergy"
    )


def counter_matrix(
    df: pd.DataFrame, vocab: HeroVocab, *, min_games: int = MIN_PAIR_GAMES
) -> PairStats:
    """Cross-team counter matrix; positive value means ``a`` counters ``b``.

    Each orientation is computed separately, then averaged — this cancels the
    radiant side term exactly in the additive-logit model::

        o1 = logit(P(a's team wins | a radiant, b dire))   # from n1 games
        o2 = logit(P(a's team wins | a dire, b radiant))   # from n2 games
        raw[a, b] = (o1 + o2) / 2 - (logit(wr_a) - logit(wr_b))

    A cell is valid when ``n1 >= min_games/2 and n2 >= min_games/2``;
    ``games[a, b] = n1 + n2``. Antisymmetry ``raw[b, a] = -raw[a, b]`` holds
    exactly: the upper triangle is computed and the lower is its negated
    transpose.

    Parameters
    ----------
    df : pd.DataFrame
        Matches with full 5+5 drafts.
    vocab : HeroVocab
        Hero id <-> index mapping defining the ``(V, V)`` layout.
    min_games : int, optional
        Validity floor; each orientation needs ``min_games/2`` games
        (default ``MIN_PAIR_GAMES``).

    Returns
    -------
    PairStats
        EB-shrunk counter scores, ``kind="counter"``.
    """
    size = vocab.size
    r_idx, d_idx, y = _team_indices(df, vocab)
    games_h, wins_h = _hero_counts(r_idx, d_idx, y, size)
    m = _hero_logit_wr(games_h, wins_h)

    # Directed counts: row = radiant hero, col = dire hero.
    flat = (r_idx[:, :, None] * size + d_idx[:, None, :]).ravel()
    y_w = np.repeat(y.astype(np.float64), _TEAM_SIZE * _TEAM_SIZE)
    n1 = np.bincount(flat, minlength=size * size).reshape(size, size)
    w1 = np.bincount(flat, weights=y_w, minlength=size * size).reshape(size, size)

    # L[a, b] = logit P(a's team wins | a radiant, b dire); o2 = -L.T exactly.
    p1 = np.clip(
        np.divide(w1, n1, out=np.full((size, size), 0.5), where=n1 > 0),
        _CLIP,
        1.0 - _CLIP,
    )
    L = np.log(p1 / (1.0 - p1))
    raw_all = (L - L.T) / 2.0 - (m[:, None] - m[None, :])
    upper = np.triu(raw_all, k=1)
    raw_signed = upper - upper.T  # exact antisymmetry by construction

    valid = (n1 >= min_games / 2) & (n1.T >= min_games / 2)
    np.fill_diagonal(valid, False)
    raw = np.full((size, size), np.nan)
    raw[valid] = raw_signed[valid]

    # Variance of the orientation average; p*(1-p) is invariant under p -> 1-p.
    var1 = 1.0 / (np.maximum(n1, 1) * p1 * (1.0 - p1))
    se2 = (var1 + var1.T) / 4.0
    value = _eb_shrink(raw, se2, valid)
    return PairStats(
        value=value, raw=raw, games=(n1 + n1.T).astype(np.int64), kind="counter"
    )


def radiant_side_advantage(df: pd.DataFrame) -> tuple[float, float, float]:
    """Global radiant side bias with its Wilson 95% CI.

    Parameters
    ----------
    df : pd.DataFrame
        Matches with a ``radiant_win`` column.

    Returns
    -------
    tuple[float, float, float]
        ``(radiant_wr, ci_lo, ci_hi)``; ``radiant_wr`` is NaN on empty input.
    """
    n = len(df)
    wins = int(df["radiant_win"].sum())
    lo, hi = wilson_ci(wins, n)
    wr = wins / n if n > 0 else float("nan")
    return wr, float(lo), float(hi)
