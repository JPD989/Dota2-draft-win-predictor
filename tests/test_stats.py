"""Formula tests for src.stats: Wilson CI, synergy/counter, EB shrinkage, z-tests + BH."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.dataset import HeroVocab
from src.stats.synergy import (
    PairStats,
    _eb_shrink,
    counter_matrix,
    hero_win_rates,
    radiant_side_advantage,
    synergy_matrix,
)
from src.stats.tests import _benjamini_hochberg, hero_wr_shift_table, two_proportion_ztest
from src.stats.wilson import wilson_ci
from tests.conftest import BASE_RATE, COUNTER_PAIR, HERO_IDS, SYNERGY_PAIRS


def null_matches(n: int, seed: int, base: float = 0.5) -> pd.DataFrame:
    """Random drafts, outcome independent of heroes (planted null)."""
    rng = np.random.default_rng(seed)
    ids = np.asarray(HERO_IDS)
    picks = np.array([rng.choice(ids, 10, replace=False) for _ in range(n)])
    return pd.DataFrame({
        "radiant_team": list(picks[:, :5]),
        "dire_team": list(picks[:, 5:]),
        "radiant_win": rng.random(n) < base,
    })


# --------------------------------------------------------------------- Wilson
def test_wilson_known_value() -> None:
    lo, hi = wilson_ci(8, 10)
    assert lo == pytest.approx(0.490, abs=1e-2)
    assert hi == pytest.approx(0.943, abs=1e-2)


def test_wilson_vectorized_and_edges() -> None:
    wins, n = np.array([0, 5, 10, 0]), np.array([10, 10, 10, 0])
    lo, hi = wilson_ci(wins, n)
    assert lo.shape == hi.shape == (4,)
    assert (lo[3], hi[3]) == (0.0, 1.0)  # n = 0 -> trivial interval
    # bounds may miss [0, 1] by float rounding (~1e-17) at w = 0 or w = n
    assert np.all((lo >= -1e-12) & (lo <= wins / np.maximum(n, 1)) & (hi <= 1 + 1e-12))
    assert lo[0] == pytest.approx(0.0, abs=1e-12) and hi[2] == pytest.approx(1.0, abs=1e-12)
    # symmetry: lo(w, n) == 1 - hi(n - w, n)
    lo2, hi2 = wilson_ci(np.array([3, 7]), 10)
    assert lo2[0] == pytest.approx(1 - hi2[1])


# ------------------------------------------------------------- hero win rates
def test_hero_win_rates_invariants(synthetic_matches, tiny_vocab) -> None:
    df = synthetic_matches(2000)
    t = hero_win_rates(df, tiny_vocab)
    assert list(t.columns) == ["hero_id", "games", "wins", "wr", "ci_lo", "ci_hi"]
    assert t["hero_id"].tolist() == tiny_vocab.hero_ids
    assert t["games"].sum() == 10 * len(df)  # every match has 10 heroes
    assert t["wins"].sum() == 5 * len(df)  # ... and exactly 5 winners
    assert np.all((t["ci_lo"] <= t["wr"]) & (t["wr"] <= t["ci_hi"]))


def test_radiant_side_advantage(synthetic_matches) -> None:
    wr, lo, hi = radiant_side_advantage(synthetic_matches(2000))
    assert lo < wr < hi
    assert abs(wr - BASE_RATE) < 0.05


# -------------------------------------------------------------------- synergy
def test_synergy_planted_null(tiny_vocab) -> None:
    s = synergy_matrix(null_matches(2000, seed=0), tiny_vocab)
    iu = np.triu_indices(tiny_vocab.size, k=1)
    assert abs(np.nanmean(s.raw[iu])) < 0.05  # unbiased on average
    assert np.abs(s.value).max() < 0.05  # EB shrinks pure noise to ~0


def test_synergy_planted_pairs_and_shape(synthetic_matches, tiny_vocab) -> None:
    s = synergy_matrix(synthetic_matches(2000), tiny_vocab)
    assert s.kind == "synergy"
    np.testing.assert_array_equal(s.value, s.value.T)
    np.testing.assert_array_equal(np.isnan(s.raw), np.isnan(s.raw.T))
    assert np.all(np.diag(s.value) == 0) and np.all(np.isnan(np.diag(s.raw)))
    for a, b in SYNERGY_PAIRS:
        i, j = tiny_vocab.idx(a), tiny_vocab.idx(b)
        assert s.raw[i, j] > 0.3
        assert s.value[i, j] > 0


def test_min_games_floor(synthetic_matches, tiny_vocab) -> None:
    s = synergy_matrix(synthetic_matches(2000), tiny_vocab, min_games=10**6)
    assert np.all(np.isnan(s.raw)) and np.all(s.value == 0)


# -------------------------------------------------------------------- counter
def test_counter_antisymmetry_exact(synthetic_matches, tiny_vocab) -> None:
    c = counter_matrix(synthetic_matches(2000), tiny_vocab)
    assert c.kind == "counter"
    valid = ~np.isnan(c.raw)
    np.testing.assert_array_equal(valid, valid.T)
    assert np.array_equal(c.raw[valid], -c.raw.T[valid])  # exact, not approx
    assert np.array_equal(c.value, -c.value.T)


def test_counter_planted_pair(synthetic_matches, tiny_vocab) -> None:
    c = counter_matrix(synthetic_matches(2000), tiny_vocab)
    i, j = tiny_vocab.idx(COUNTER_PAIR[0]), tiny_vocab.idx(COUNTER_PAIR[1])
    assert c.raw[i, j] > 0.3 and c.raw[j, i] < -0.3
    # A single planted cell among 190 can be shrunk all the way to 0 by the global
    # zero-mean prior (tau2 floors at 0); it must never flip sign.
    assert c.value[i, j] >= 0


def test_counter_cancels_side_bias(tiny_vocab) -> None:
    # Strong radiant bias, no hero effects: the orientation average removes it.
    c = counter_matrix(null_matches(3000, seed=1, base=0.65), tiny_vocab)
    assert np.nanmean(np.abs(c.raw)) < 0.12


# ------------------------------------------------------------- EB shrinkage
def _three_hero_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = np.full((3, 3), np.nan)
    se2 = np.ones((3, 3))
    for (a, b), r, s in [((0, 1), 0.4, 0.04), ((0, 2), -0.2, 0.01), ((1, 2), 0.1, 0.09)]:
        raw[a, b] = raw[b, a] = r
        se2[a, b] = se2[b, a] = s
    valid = ~np.isnan(raw)
    return raw, se2, valid


def test_eb_hand_computed_three_heroes() -> None:
    """w = 1/se2 = (25, 100, 100/9); sum w = 1225/9.
    sum w*raw^2 = 4 + 4 + 1/9 = 73/9; sum w*se2 = 3 = 27/9.
    tau2 = (73 - 27) / 1225 = 46/1225.
    value = raw * tau2 / (tau2 + se2):
      (0,1): 0.4 * 46/95 ; (0,2): -0.2 * 46/58.25 ; (1,2): 0.1 * 46/156.25
    """
    raw, se2, valid = _three_hero_fixture()
    v = _eb_shrink(raw, se2, valid)
    expected = {(0, 1): 0.4 * 46 / 95, (0, 2): -0.2 * 46 / 58.25, (1, 2): 0.1 * 46 / 156.25}
    for (a, b), e in expected.items():
        assert v[a, b] == pytest.approx(e, rel=1e-12)
        assert v[b, a] == pytest.approx(e, rel=1e-12)
    assert np.all(np.diag(v) == 0)


def test_eb_invalid_cells_excluded_and_zero() -> None:
    raw, se2, valid = _three_hero_fixture()
    raw2, valid2 = raw.copy(), valid.copy()
    raw2[1, 2] = raw2[2, 1] = 50.0  # huge but invalid: must not affect tau2
    valid2[1, 2] = valid2[2, 1] = False
    v = _eb_shrink(raw2, se2, valid2)
    assert v[1, 2] == 0.0 and v[2, 1] == 0.0
    # tau2 from the two remaining pairs only: w = (25, 100)
    tau2 = (4 + 4) / 125 - 2 / 125
    assert v[0, 1] == pytest.approx(0.4 * tau2 / (tau2 + 0.04), rel=1e-12)


def test_eb_tau2_floor_zero() -> None:
    raw, se2, valid = _three_hero_fixture()
    v = _eb_shrink(raw * 0.01, se2, valid)  # signal far below noise -> tau2 = 0
    assert np.all(v == 0)


def test_shrinkage_stronger_for_low_n(synthetic_matches, tiny_vocab) -> None:
    raw = np.array([[np.nan, 0.3, 0.3], [0.3, np.nan, 0.5], [0.3, 0.5, np.nan]])
    se2 = np.array([[1, 0.01, 0.2], [0.01, 1, 0.05], [0.2, 0.05, 1]])
    v = _eb_shrink(raw, se2, ~np.isnan(raw))
    assert 0 < v[0, 2] < v[0, 1] < 0.3  # same raw, more noise -> pulled closer to 0
    # On real matrices: shrink factor value/raw increases with sample size.
    s = synergy_matrix(synthetic_matches(4000, seed=2), tiny_vocab, min_games=50)
    iu = np.triu_indices(tiny_vocab.size, k=1)
    ok = ~np.isnan(s.raw[iu]) & (np.abs(s.raw[iu]) > 1e-9)
    factor, games = (s.value[iu] / s.raw[iu])[ok], s.games[iu][ok]
    if factor.max() > 0:  # tau2 > 0 on this sample
        assert np.corrcoef(factor, games)[0, 1] > 0.5


def test_pairstats_round_trip(tmp_path, synthetic_matches, tiny_vocab) -> None:
    for ps in (synergy_matrix(synthetic_matches(1000), tiny_vocab),
               counter_matrix(synthetic_matches(1000), tiny_vocab)):
        path = tmp_path / f"{ps.kind}.npz"
        ps.save(path)
        back = PairStats.load(path)
        assert back.kind == ps.kind and isinstance(back.kind, str)
        np.testing.assert_array_equal(back.value, ps.value)
        np.testing.assert_array_equal(back.raw, ps.raw)  # NaN-aware
        np.testing.assert_array_equal(back.games, ps.games)
        assert back.games.dtype == np.int64


# --------------------------------------------------------- z-test + BH table
def test_two_proportion_ztest_known() -> None:
    z, p = two_proportion_ztest(60, 100, 40, 100)
    assert z == pytest.approx(0.2 / np.sqrt(0.25 * 0.02), rel=1e-9)  # 2.828
    assert p == pytest.approx(0.004678, abs=1e-5)
    z2, p2 = two_proportion_ztest(40, 100, 60, 100)
    assert z2 == pytest.approx(-z) and p2 == pytest.approx(p)
    assert two_proportion_ztest(0, 0, 5, 10) == (0.0, 1.0)
    assert two_proportion_ztest(10, 10, 10, 10) == (0.0, 1.0)


def test_benjamini_hochberg_known() -> None:
    q = _benjamini_hochberg(np.array([0.01, 0.04, 0.03, 0.2]))
    np.testing.assert_allclose(q, [0.04, 0.04 * 4 / 3, 0.04 * 4 / 3, 0.2])


def test_shift_table_filter_first_and_bh(synthetic_matches, tiny_vocab) -> None:
    a, b = synthetic_matches(2000, seed=0), synthetic_matches(2000, seed=1)
    hero = 3
    # plant a real shift in B: `hero` wins every game it appears in
    on_r = b["radiant_team"].map(lambda t: hero in t).to_numpy()
    on_d = b["dire_team"].map(lambda t: hero in t).to_numpy()
    b.loc[on_r, "radiant_win"] = True
    b.loc[on_d, "radiant_win"] = False

    t = hero_wr_shift_table(a, b, tiny_vocab, label_a="A", label_b="B", min_games=200)
    for col in ("wr_a", "wr_b", "delta", "z", "p_value", "q_value", "m_tests", "significant"):
        assert col in t.columns
    assert (t["m_tests"] == len(t)).all() and len(t) == tiny_vocab.size
    assert np.all(t["q_value"] >= t["p_value"] - 1e-15)
    assert (t["significant"] == (t["q_value"] <= 0.05)).all()
    assert t["delta"].abs().is_monotonic_decreasing
    assert np.all(np.sign(t["z"]) == np.sign(t["delta"]))
    top = t.iloc[0]
    assert top["hero_id"] == hero and top["significant"]

    # filter first: a min_games above some heroes' counts shrinks the BH family
    counts = np.minimum(hero_win_rates(a, tiny_vocab)["games"],
                        hero_win_rates(b, tiny_vocab)["games"])
    thr = int(np.median(counts)) + 1
    t2 = hero_wr_shift_table(a, b, tiny_vocab, label_a="A", label_b="B", min_games=thr)
    assert len(t2) == int((counts >= thr).sum()) < tiny_vocab.size
    assert (t2["m_tests"] == len(t2)).all()


def test_vocab_independent_of_hero_order() -> None:
    assert HeroVocab([129, 1, 5, 1]).hero_ids == [1, 5, 129]
