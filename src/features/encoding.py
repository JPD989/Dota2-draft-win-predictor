"""Draft feature encoding: index matrices, one-hot features, labels, mirror augmentation.

Conventions (docs/architecture.md §3, §4.4, §6): team order is always radiant first
(columns 0-4 radiant, 5-9 dire), values are contiguous vocab indices, and ``-1`` marks
an empty slot in a partial draft.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse

from src.config import RANK_NORM
from src.data.dataset import HeroVocab


def teams_to_matrix(df: pd.DataFrame, vocab: HeroVocab) -> np.ndarray:
    """Encode the two team columns as an ``(N, 10)`` matrix of vocab indices.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``radiant_team`` and ``dire_team`` columns of hero-id
        sequences with 0 to 5 entries each (partial drafts allowed).
    vocab : HeroVocab
        Hero id to contiguous index mapping.

    Returns
    -------
    np.ndarray
        ``(N, 10)`` int32; columns 0-4 radiant, 5-9 dire. Teams shorter than 5
        are right-padded with ``-1`` (empty slot).

    Raises
    ------
    KeyError
        If a team contains a hero id unknown to `vocab`.
    ValueError
        If a team has more than 5 entries.
    """
    lut = {hero_id: idx for idx, hero_id in enumerate(vocab.hero_ids)}
    out = np.full((len(df), 10), -1, dtype=np.int32)
    for i, (radiant, dire) in enumerate(zip(df["radiant_team"], df["dire_team"])):
        for offset, team in ((0, radiant), (5, dire)):
            if len(team) > 5:
                raise ValueError(f"team has {len(team)} heroes, expected at most 5")
            for j, hero_id in enumerate(team):
                out[i, offset + j] = lut[int(hero_id)]  # KeyError on unknown hero id
    return out


def one_hot_drafts(teams: np.ndarray, vocab_size: int) -> scipy.sparse.csr_matrix:
    """One-hot encode an index matrix into a sparse draft feature matrix.

    Parameters
    ----------
    teams : np.ndarray
        ``(N, 10)`` vocab-index matrix from :func:`teams_to_matrix`.
    vocab_size : int
        Vocabulary size ``V``.

    Returns
    -------
    scipy.sparse.csr_matrix
        ``(N, 2V)`` float32. Radiant hero with vocab index ``v`` sets column
        ``v``; dire hero sets column ``V + v``. ``-1`` entries (empty slots)
        contribute nothing.
    """
    row_idx, slot_idx = np.nonzero(teams >= 0)
    cols = teams[row_idx, slot_idx].astype(np.int64) + np.where(slot_idx >= 5, vocab_size, 0)
    data = np.ones(row_idx.shape[0], dtype=np.float32)
    return scipy.sparse.csr_matrix(
        (data, (row_idx, cols)), shape=(teams.shape[0], 2 * vocab_size), dtype=np.float32
    )


def labels(df: pd.DataFrame) -> np.ndarray:
    """Return the ``radiant_win`` labels as a ``(N,)`` float64 array."""
    return df["radiant_win"].to_numpy(dtype=np.float64)


def rank_feature(df: pd.DataFrame) -> np.ndarray:
    """Return the normalized rank feature ``avg_rank_tier / RANK_NORM``.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain an ``avg_rank_tier`` column.

    Returns
    -------
    np.ndarray
        ``(N, 1)`` float32.
    """
    values = df["avg_rank_tier"].to_numpy(dtype=np.float32) / np.float32(RANK_NORM)
    return values.reshape(-1, 1)


def mirror(
    teams: np.ndarray, y: np.ndarray, extra: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mirror-augment drafts: append side-swapped copies with flipped labels.

    Parameters
    ----------
    teams : np.ndarray
        ``(N, 10)`` vocab-index matrix (radiant columns 0-4, dire 5-9).
    y : np.ndarray
        ``(N,)`` labels, P(radiant win) targets.
    extra : np.ndarray or None, optional
        Side-invariant extra features of shape ``(N, k)`` (e.g. the rank
        column); tiled alongside the doubled rows when given.

    Returns
    -------
    tuple of np.ndarray
        ``(teams_out, y_out)`` of shapes ``(2N, 10)`` and ``(2N,)`` — the
        originals followed by copies with the radiant/dire blocks swapped and
        labels flipped to ``1 - y``. If `extra` is given, a third array of
        shape ``(2N, k)`` is returned.
    """
    swapped = np.concatenate([teams[:, 5:], teams[:, :5]], axis=1)
    teams_out = np.concatenate([teams, swapped], axis=0)
    y_out = np.concatenate([y, 1.0 - y], axis=0)
    if extra is None:
        return teams_out, y_out
    return teams_out, y_out, np.concatenate([extra, extra], axis=0)
