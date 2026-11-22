"""Classical baselines: logistic regression and XGBoost on one-hot hero vectors.

Both models share the same features (docs/architecture.md §4.8): the ``(N, 2V)``
one-hot draft matrix from :func:`src.features.encoding.one_hot_drafts`, plus the
normalized rank column when ``use_rank``. This module must stay torch-free.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.config import RANK_NORM, SEED
from src.data.dataset import HeroVocab
from src.features.encoding import labels, mirror, one_hot_drafts, rank_feature, teams_to_matrix
from src.models.base import DraftModel, apply_mirror_delta, compute_mirror_delta, validate_predict_frame

__all__ = ["LogisticDraftModel", "XGBoostDraftModel", "rank_input"]

_DEFAULT_RANK_TIER: float = 50.0


def rank_input(df: pd.DataFrame) -> np.ndarray:
    """Normalized rank column, imputing ``avg_rank_tier = 50`` when missing.

    Parameters
    ----------
    df : pd.DataFrame
        Matches; the ``avg_rank_tier`` column is optional, NaN entries are imputed.

    Returns
    -------
    np.ndarray
        ``(N, 1)`` float32, ``avg_rank_tier / RANK_NORM``.
    """
    if "avg_rank_tier" not in df.columns:
        return np.full((len(df), 1), _DEFAULT_RANK_TIER / RANK_NORM, dtype=np.float32)
    return rank_feature(df.assign(avg_rank_tier=df["avg_rank_tier"].fillna(_DEFAULT_RANK_TIER)))


class _OneHotDraftModel(DraftModel):
    """Shared encoding, mirroring and pickling for the sklearn-style baselines."""

    def __init__(self, use_rank: bool, mirror_augment: bool, seed: int) -> None:
        self.use_rank = use_rank
        self.mirror_augment = mirror_augment
        self.seed = seed
        self.vocab: HeroVocab | None = None
        self.delta: float = 0.0
        self.estimator: Any = None

    def _features(self, teams: np.ndarray, rank: np.ndarray | None) -> scipy.sparse.csr_matrix:
        """One-hot drafts, with the rank column appended when ``use_rank``."""
        assert self.vocab is not None
        x = one_hot_drafts(teams, self.vocab.size)
        if rank is not None:
            x = scipy.sparse.hstack([x, scipy.sparse.csr_matrix(rank)], format="csr")
        return x

    def _encode(self, df: pd.DataFrame) -> scipy.sparse.csr_matrix:
        """Encode an un-augmented frame (val / test / predict path)."""
        assert self.vocab is not None
        rank = rank_input(df) if self.use_rank else None
        return self._features(teams_to_matrix(df, self.vocab), rank)

    def _train_xy(self, train: pd.DataFrame) -> tuple[scipy.sparse.csr_matrix, np.ndarray]:
        """Train features/labels, mirror-augmented if requested; sets ``self.delta``."""
        assert self.vocab is not None
        teams, y = teams_to_matrix(train, self.vocab), labels(train)
        rank = rank_input(train) if self.use_rank else None
        self.delta = compute_mirror_delta(y, self.mirror_augment)
        if self.mirror_augment:
            if rank is None:
                teams, y = mirror(teams, y)
            else:
                teams, y, rank = mirror(teams, y, rank)
        return self._features(teams, rank), y

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """P(radiant win), shape ``(N,)`` float64; partial drafts allowed (§6)."""
        if not self.is_fitted:
            raise RuntimeError(f"{type(self).__name__} is not fitted; call fit() or load()")
        validate_predict_frame(df)
        if len(df) == 0:
            return np.empty(0, dtype=np.float64)
        p = self.estimator.predict_proba(self._encode(df))[:, 1]
        return apply_mirror_delta(p, self.delta)

    def save(self, path: Path) -> None:
        """Pickle the whole wrapper (vocab, delta and estimator included)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "_OneHotDraftModel":
        """Unpickle a wrapper written by :meth:`save`."""
        with path.open("rb") as f:
            model = pickle.load(f)
        if not isinstance(model, cls):
            raise TypeError(f"{path} holds a {type(model).__name__}, expected {cls.__name__}")
        return model


class LogisticDraftModel(_OneHotDraftModel):
    """L2 logistic regression on one-hot hero vectors — the benchmark baseline.

    Parameters
    ----------
    C : float, optional
        Inverse L2 regularization strength, by default 1.0.
    use_rank : bool, optional
        Append the normalized rank feature, by default False.
    mirror_augment : bool, optional
        Train on mirror-doubled data plus the side delta (§3), by default False.
    seed : int, optional
        Random seed, by default ``SEED``.
    """

    name = "logreg"

    def __init__(self, C: float = 1.0, use_rank: bool = False,
                 mirror_augment: bool = False, seed: int = SEED) -> None:
        super().__init__(use_rank, mirror_augment, seed)
        self.C = C

    def fit(self, train: pd.DataFrame, val: pd.DataFrame, vocab: HeroVocab) -> "LogisticDraftModel":
        """Fit on ``train``; ``val`` is unused (no tuning in the baseline). Returns self."""
        self.vocab = vocab
        x, y = self._train_xy(train)
        self.estimator = LogisticRegression(
            C=self.C, solver="lbfgs", max_iter=2000, random_state=self.seed
        ).fit(x, y)
        return self


class XGBoostDraftModel(_OneHotDraftModel):
    """Gradient-boosted trees on the one-hot features, early-stopped on val log-loss.

    Parameters
    ----------
    n_estimators : int, optional
        Maximum boosting rounds, by default 400.
    max_depth : int, optional
        Tree depth, by default 8.
    learning_rate : float, optional
        Shrinkage, by default 0.1.
    use_rank : bool, optional
        Append the normalized rank feature, by default False.
    mirror_augment : bool, optional
        Train on mirror-doubled data plus the side delta (§3), by default False.
    seed : int, optional
        Random seed, by default ``SEED``.
    **xgb_kwargs
        Forwarded to ``XGBClassifier``; ``early_stopping_rounds`` defaults to 20.
    """

    name = "xgboost"

    def __init__(self, n_estimators: int = 400, max_depth: int = 8, learning_rate: float = 0.1,
                 use_rank: bool = False, mirror_augment: bool = False, seed: int = SEED,
                 **xgb_kwargs: Any) -> None:
        super().__init__(use_rank, mirror_augment, seed)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.xgb_kwargs = {"early_stopping_rounds": 20, "tree_method": "hist", **xgb_kwargs}

    def fit(self, train: pd.DataFrame, val: pd.DataFrame, vocab: HeroVocab) -> "XGBoostDraftModel":
        """Fit on (possibly mirrored) ``train``, early-stopping on un-augmented ``val``."""
        self.vocab = vocab
        x, y = self._train_xy(train)
        self.estimator = XGBClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            learning_rate=self.learning_rate, objective="binary:logistic",
            eval_metric="logloss", random_state=self.seed, **self.xgb_kwargs,
        )
        self.estimator.fit(x, y, eval_set=[(self._encode(val), labels(val))], verbose=False)
        return self
