"""Abstract interface and shared helpers for draft win-prediction models.

Every model in the benchmark (logistic regression, XGBoost, transformer, GNN)
implements :class:`DraftModel` so the evaluation harness, the recommender and
the Streamlit app can treat them interchangeably. Checkpoints are
self-contained: they carry the hero vocabulary and the mirror delta, so
``load()`` needs nothing but the checkpoint file.
"""

from __future__ import annotations

import abc
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.special import expit

if TYPE_CHECKING:
    from src.data.dataset import HeroVocab

__all__ = [
    "DraftModel",
    "apply_mirror_delta",
    "compute_mirror_delta",
    "logit",
    "sigmoid",
    "validate_predict_frame",
]


class DraftModel(abc.ABC):
    """Common interface the harness, recommender and app rely on.

    Attributes
    ----------
    name : str
        Registry name: ``"logreg" | "xgboost" | "transformer" | "gnn"``.
    use_rank : bool
        Whether the model consumes the normalized ``avg_rank_tier`` feature.
        Set in ``__init__``; consumers may inspect it.
    vocab : HeroVocab | None
        Hero vocabulary used for encoding. Set by :meth:`fit` *and* by
        :meth:`load`; models are authoritative for their own encoding, so
        :meth:`predict_proba` never takes an external vocab.
    """

    name: str
    use_rank: bool
    vocab: HeroVocab | None = None

    @property
    def is_fitted(self) -> bool:
        """``True`` after :meth:`fit` or :meth:`load` has populated the model."""
        return self.vocab is not None

    @abc.abstractmethod
    def fit(self, train: pd.DataFrame, val: pd.DataFrame, vocab: HeroVocab) -> "DraftModel":
        """Train on ``train``, using ``val`` for early stopping / tuning.

        Implementations must store ``vocab`` on the model and compute the
        mirror delta from the *un-mirrored* train labels (see
        :func:`compute_mirror_delta`).

        Parameters
        ----------
        train : pd.DataFrame
            Training matches (schema of §2 in docs/architecture.md).
        val : pd.DataFrame
            Validation matches; never augmented.
        vocab : HeroVocab
            Hero vocabulary shared by the benchmark run.

        Returns
        -------
        DraftModel
            ``self``, for chaining.
        """

    @abc.abstractmethod
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict P(radiant win) for each row of ``df``.

        Accepts partial drafts (teams with fewer than 5 heroes, §6). Encodes
        internally with ``self.vocab``. If ``use_rank`` and the
        ``avg_rank_tier`` column is missing, imputes 50 silently.

        Parameters
        ----------
        df : pd.DataFrame
            Frame with ``radiant_team`` / ``dire_team`` hero-id list columns.

        Returns
        -------
        np.ndarray
            Shape ``(N,)`` float64 probabilities in [0, 1].
        """

    @abc.abstractmethod
    def save(self, path: Path) -> None:
        """Serialize the fitted model to ``path``.

        The checkpoint MUST include ``self.vocab`` (its sorted hero ids) and
        the mirror delta, so :meth:`load` is self-contained — no external
        vocab file or train split is needed to restore predictions exactly.

        Parameters
        ----------
        path : Path
            Destination file (``.pkl`` for sklearn/xgboost wrappers, ``.pt``
            for torch models; see ``src.models.checkpoint_path``).
        """

    @classmethod
    @abc.abstractmethod
    def load(cls, path: Path) -> "DraftModel":
        """Restore a fitted model from a checkpoint written by :meth:`save`.

        Parameters
        ----------
        path : Path
            Checkpoint file.

        Returns
        -------
        DraftModel
            Fitted model with ``vocab`` and mirror delta restored;
            ``is_fitted`` is ``True``.
        """


def sigmoid(x: ArrayLike) -> np.ndarray:
    """Logistic function ``1 / (1 + exp(-x))``, element-wise.

    Parameters
    ----------
    x : array-like
        Logits.

    Returns
    -------
    np.ndarray
        Float64 values in (0, 1); scalar input yields a NumPy scalar.
    """
    return expit(np.asarray(x, dtype=np.float64))


def logit(p: ArrayLike, *, eps: float = 1e-6) -> np.ndarray:
    """Log-odds ``log(p / (1 - p))`` with ``p`` clipped to ``[eps, 1 - eps]``.

    The clip guards against infinities at exactly 0 or 1. (The stats module
    uses its own, wider 1e-3 clip for win-rate formulas; this is the
    model-side helper.)

    Parameters
    ----------
    p : array-like
        Probabilities.
    eps : float, optional
        Clip bound, by default 1e-6.

    Returns
    -------
    np.ndarray
        Float64 log-odds; scalar input yields a NumPy scalar.
    """
    q = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.log(q / (1.0 - q))


def compute_mirror_delta(y_train: ArrayLike, mirror_augment: bool) -> float:
    """Fixed logit offset restoring the side bias under mirror augmentation.

    Mirror augmentation doubles the train set with (swapped teams, flipped
    label), which makes the training objective side-symmetric. The real
    radiant advantage (~53% win rate) is then re-introduced at predict time
    as a constant offset ``delta = logit(mean(y_train))`` computed on the
    **un-mirrored** train labels and stored in the checkpoint. Without
    mirroring the model learns the side bias from data and ``delta = 0``.

    Parameters
    ----------
    y_train : array-like
        Un-mirrored train labels (``radiant_win`` as 0/1).
    mirror_augment : bool
        Whether the model is trained with mirror augmentation.

    Returns
    -------
    float
        ``logit(mean(y_train))`` if ``mirror_augment``, else ``0.0``.
    """
    if not mirror_augment:
        return 0.0
    return float(logit(np.asarray(y_train, dtype=np.float64).mean()))


def apply_mirror_delta(p: ArrayLike, delta: float) -> np.ndarray:
    """Shift probabilities by a fixed logit offset: ``sigmoid(logit(p) + delta)``.

    For models whose backend outputs probabilities (sklearn, XGBoost). Torch
    models can equivalently add ``delta`` to their raw logit before the
    sigmoid. With ``delta == 0.0`` (no mirror augmentation) the input is
    returned unchanged as float64, so calling this unconditionally is safe.

    Parameters
    ----------
    p : array-like
        Predicted probabilities, shape ``(N,)``.
    delta : float
        Stored mirror delta (see :func:`compute_mirror_delta`).

    Returns
    -------
    np.ndarray
        Shifted float64 probabilities, shape ``(N,)``.
    """
    out = np.asarray(p, dtype=np.float64)
    if delta == 0.0:
        return out
    return sigmoid(logit(out) + delta)


def validate_predict_frame(df: pd.DataFrame) -> None:
    """Validate a prediction input frame, raising ``ValueError`` if malformed.

    Checks that the ``radiant_team`` / ``dire_team`` columns exist and that no
    team holds more than 5 heroes. Partial drafts — fewer than 5 picks, even
    empty teams — are allowed (recommender path, §6). Hero-id validity is left
    to the model's ``HeroVocab``, which raises ``KeyError`` on unknown ids.

    Parameters
    ----------
    df : pd.DataFrame
        Frame with per-match ``radiant_team`` / ``dire_team`` hero-id lists.

    Raises
    ------
    ValueError
        If a team column is missing or any team has more than 5 entries.
    """
    missing = [c for c in ("radiant_team", "dire_team") if c not in df.columns]
    if missing:
        raise ValueError(
            f"prediction frame is missing column(s) {missing}; expected "
            "'radiant_team' and 'dire_team' hero-id list columns"
        )
    for col in ("radiant_team", "dire_team"):
        sizes = df[col].map(len)
        if len(sizes) and int(sizes.max()) > 5:
            raise ValueError(
                f"column {col!r} has a team with {int(sizes.max())} heroes (max 5)"
            )
