"""Tests for src.recommend.recommender."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.dataset import chronological_split
from src.recommend.recommender import legal_picks, recommend
from tests.conftest import HERO_IDS, SYNERGY_PAIRS
from tests.test_eval import StubLogReg


@pytest.fixture(scope="module")
def pair_model(tiny_vocab):
    from tests.conftest import make_synthetic_matches

    split = chronological_split(make_synthetic_matches(4000, seed=11))
    return StubLogReg("pairs", pairs=True, C=0.5).fit(split.train, split.val, tiny_vocab)


def test_legal_picks_excludes_picked_and_banned(tiny_vocab):
    legal = legal_picks([1, 2], [3], [5, 999], tiny_vocab)
    assert legal == sorted(set(HERO_IDS) - {1, 2, 3, 5})


def test_recommend_excludes_and_single_batch_call(pair_model, tiny_vocab):
    pair_model.n_predict_calls = 0
    rec = recommend(pair_model, tiny_vocab, radiant=[1, 2], dire=[3], side="dire", banned=[5, 8])
    assert pair_model.n_predict_calls == 1
    assert list(rec.columns) == ["hero_id", "win_prob"]
    assert set(rec["hero_id"]) == set(HERO_IDS) - {1, 2, 3, 5, 8}
    assert rec["win_prob"].is_monotonic_decreasing
    frame = pair_model.last_predict_frame
    assert len(frame) == len(rec) and "avg_rank_tier" in frame.columns
    assert (frame["avg_rank_tier"] == 50).all()
    assert all(list(r) == [1, 2] for r in frame["radiant_team"])
    assert all(len(d) == 2 and d[0] == 3 for d in frame["dire_team"])


def test_recommend_dire_prob_is_dire_side(pair_model, tiny_vocab):
    rec = recommend(pair_model, tiny_vocab, radiant=[1], dire=[], side="dire", avg_rank_tier=70)
    assert (pair_model.last_predict_frame["avg_rank_tier"] == 70).all()
    df = pair_model.last_predict_frame
    p_rad = pair_model.predict_proba(df)
    expected = dict(zip([int(d[-1]) for d in df["dire_team"]], 1.0 - p_rad))
    for h, wp in zip(rec["hero_id"], rec["win_prob"]):
        assert wp == pytest.approx(expected[h])


@pytest.mark.parametrize("side", ["radiant", "dire"])
@pytest.mark.parametrize("pair", SYNERGY_PAIRS)
def test_planted_synergy_partner_in_top3(pair_model, tiny_vocab, side, pair):
    a, b = pair
    own, other = [a], [h for h in (19, 23) if h not in pair]
    kw = {"radiant": own, "dire": other} if side == "radiant" else {"radiant": other, "dire": own}
    rec = recommend(pair_model, tiny_vocab, side=side, top_k=3, **kw)
    assert len(rec) == 3
    assert b in rec["hero_id"].tolist()


def test_recommend_validation(pair_model, tiny_vocab):
    with pytest.raises(ValueError, match="side"):
        recommend(pair_model, tiny_vocab, radiant=[], dire=[], side="both")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="twice"):
        recommend(pair_model, tiny_vocab, radiant=[1], dire=[1], side="dire")
    with pytest.raises(ValueError, match="already has 5"):
        recommend(pair_model, tiny_vocab, radiant=[1, 2, 3, 5, 8], dire=[], side="radiant")
    full = recommend(pair_model, tiny_vocab, radiant=[1, 2, 3, 5, 8], dire=[11, 14, 19, 23], side="dire",
                     banned=[h for h in HERO_IDS if h not in {1, 2, 3, 5, 8, 11, 14, 19, 23}])
    assert full.empty and list(full.columns) == ["hero_id", "win_prob"]
    assert np.issubdtype(full["hero_id"].dtype, np.integer)


@pytest.mark.skip(reason="stats-mode formula sanity belongs to the app engineer (StatsDraftModel in app/)")
def test_stats_mode_formula_sanity():  # pragma: no cover
    ...
