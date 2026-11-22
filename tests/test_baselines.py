"""DraftModel tests for the classical baselines (logreg, XGBoost) on synthetic data."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src.data.dataset import HeroVocab
from src.models import checkpoint_path, get_model_class
from src.models.base import DraftModel, logit
from src.models.baselines import LogisticDraftModel, XGBoostDraftModel

# n_jobs pinned: XGBoost's all-cores default is pathologically slow on tiny data on some WSL hosts.
FACTORIES: dict[str, Callable[..., DraftModel]] = {
    "logreg": lambda **kw: LogisticDraftModel(**kw),
    "xgboost": lambda **kw: XGBoostDraftModel(n_estimators=100, max_depth=4, n_jobs=2, **kw),
}
EMPTY = pd.DataFrame({"radiant_team": [[]], "dire_team": [[]]})


@pytest.fixture(scope="module")
def data(synthetic_matches) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Independent synthetic train / val / test draws with the same planted signal."""
    return synthetic_matches(2000, 0), synthetic_matches(500, 1), synthetic_matches(1000, 2)


@pytest.fixture(scope="module", params=list(FACTORIES))
def fitted(request, data, tiny_vocab) -> DraftModel:
    train, val, _ = data
    return FACTORIES[request.param]().fit(train, val, tiny_vocab)


def test_registry_and_defaults_are_canonical() -> None:
    assert get_model_class("logreg") is LogisticDraftModel
    assert get_model_class("xgboost") is XGBoostDraftModel
    for cls in (LogisticDraftModel, XGBoostDraftModel):
        params = inspect.signature(cls.__init__).parameters
        assert params["use_rank"].default is False
        assert params["mirror_augment"].default is False


def test_auc_beats_chance_on_planted_signal(fitted, data) -> None:
    _, _, test = data
    p = fitted.predict_proba(test)
    assert p.shape == (len(test),) and p.dtype == np.float64
    assert np.all((p >= 0) & (p <= 1))
    assert roc_auc_score(test["radiant_win"], p) > 0.55


def test_save_load_roundtrip_exact(fitted, data, tmp_path: Path) -> None:
    _, _, test = data
    path = checkpoint_path(fitted.name, tmp_path, tag="t")
    fitted.save(path)
    loaded = type(fitted).load(path)
    assert loaded.is_fitted
    assert loaded.vocab is not None and loaded.vocab.hero_ids == fitted.vocab.hero_ids
    assert loaded.delta == fitted.delta
    np.testing.assert_array_equal(loaded.predict_proba(test), fitted.predict_proba(test))


def test_partial_drafts(fitted, data) -> None:
    _, _, test = data
    partial = pd.DataFrame({
        "radiant_team": [t[:k] for t, k in zip(test["radiant_team"][:5], [0, 1, 2, 3, 5])],
        "dire_team": [t[:k] for t, k in zip(test["dire_team"][:5], [5, 0, 4, 1, 2])],
    })
    p = fitted.predict_proba(partial)
    assert p.shape == (5,) and np.all(np.isfinite(p)) and np.all((p > 0) & (p < 1))
    assert fitted.predict_proba(partial.iloc[:0]).shape == (0,)


@pytest.mark.parametrize("name", [
    "logreg",
    pytest.param("xgboost", marks=pytest.mark.xfail(
        reason="all-empty drafts are out-of-distribution for trees (§4.15)", strict=False)),
])
def test_empty_draft_near_base_rate(name, data, tiny_vocab) -> None:
    train, val, _ = data
    model = FACTORIES[name]().fit(train, val, tiny_vocab)
    assert abs(model.predict_proba(EMPTY)[0] - train["radiant_win"].mean()) < 0.05


@pytest.mark.parametrize("name", list(FACTORIES))
def test_mirror_delta_restores_side_bias(name, data, tiny_vocab) -> None:
    train, val, test = data
    model = FACTORIES[name](mirror_augment=True).fit(train, val, tiny_vocab)
    base = train["radiant_win"].mean()
    assert model.delta == pytest.approx(float(logit(base)))
    assert abs(model.predict_proba(test).mean() - base) < 0.05


def test_mirrored_logreg_is_antisymmetric_around_delta(data, tiny_vocab) -> None:
    """With mirrored data the convex optimum satisfies f(x) - delta = -(f(swap x) - delta)."""
    train, val, test = data
    model = LogisticDraftModel(mirror_augment=True).fit(train, val, tiny_vocab)
    swapped = test.assign(radiant_team=test["dire_team"], dire_team=test["radiant_team"])
    z, z_sw = logit(model.predict_proba(test)), logit(model.predict_proba(swapped))
    np.testing.assert_allclose(z - model.delta, -(z_sw - model.delta), atol=1e-3)


@pytest.mark.parametrize("name", list(FACTORIES))
def test_use_rank_imputes_missing_column(name, data, tiny_vocab) -> None:
    train, val, test = data
    model = FACTORIES[name](use_rank=True).fit(train, val, tiny_vocab)
    frame = test[["radiant_team", "dire_team"]]
    np.testing.assert_array_equal(model.predict_proba(frame),
                                  model.predict_proba(frame.assign(avg_rank_tier=50)))


def test_unfitted_and_bad_inputs(tiny_vocab: HeroVocab, data) -> None:
    with pytest.raises(RuntimeError):
        LogisticDraftModel().predict_proba(EMPTY)
    train, val, _ = data
    model = LogisticDraftModel().fit(train, val, tiny_vocab)
    with pytest.raises(KeyError):
        model.predict_proba(pd.DataFrame({"radiant_team": [[999]], "dire_team": [[]]}))
    with pytest.raises(ValueError):
        model.predict_proba(pd.DataFrame({"radiant_team": [[1, 2, 3, 5, 8, 11]], "dire_team": [[]]}))
