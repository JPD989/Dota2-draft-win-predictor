"""Data-pipeline tests: collector pure functions, validation, loading, splits.

All inputs are synthetic or written to ``tmp_path``; nothing reads ``data/raw``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import collect_matches as cm
from src.data.dataset import (
    HeroVocab,
    chronological_split,
    load_matches,
    patch_split,
)
from src.data.validate import validate

VALID_HEROES = set(range(1, 21))


# ----------------------------------------------------------------- parse_team
@pytest.mark.parametrize(
    "field, expected",
    [
        ("1,2,3,4,5", [1, 2, 3, 4, 5]),
        ([5, 4, 3, 2, 1], [5, 4, 3, 2, 1]),
        ((1, 2, 3, 4, 5), [1, 2, 3, 4, 5]),
        ("1,2,3,4,4", None),  # duplicate hero
        ([1, 2, 3, 4], None),  # too short
        ("1,2,3,4,5,6", None),  # too long
        ("1,2,x,4,5", None),  # unparsable
        (None, None),
        (12345, None),  # unsupported type
    ],
)
def test_parse_team(field: object, expected: list[int] | None) -> None:
    assert cm.parse_team(field) == expected


# ---------------------------------------------------------------- clean_match
TS, NAMES = [100, 200, 300], ["7.39", "7.40", "7.41"]


def _raw_match(**over: object) -> dict:
    m = {
        "match_id": 42,
        "start_time": 250,
        "duration": 1800,
        "avg_rank_tier": 45,
        "radiant_win": True,
        "radiant_team": "1,2,3,4,5",
        "dire_team": [6, 7, 8, 9, 10],
        "lobby_type": cm.RANKED_LOBBY,
        "game_mode": cm.ALL_DRAFT_MODE,
    }
    m.update(over)
    return m


def test_clean_match_valid() -> None:
    row = cm.clean_match(_raw_match(), VALID_HEROES, 0, TS, NAMES)
    assert row is not None
    assert list(row) == cm.EXPECTED_COLS
    assert row["radiant_team"] == [1, 2, 3, 4, 5]
    assert row["patch"] == "7.40"
    assert row["radiant_win"] is True


@pytest.mark.parametrize(
    "override",
    [
        {"lobby_type": 0},
        {"game_mode": 1},
        {"duration": cm.MIN_DURATION_S - 1},
        {"duration": None},
        {"avg_rank_tier": None},
        {"avg_rank_tier": 10},  # below min_rank_tier=20
        {"dire_team": [5, 7, 8, 9, 10]},  # hero on both teams
        {"dire_team": [6, 7, 8, 9, 99]},  # unknown hero
        {"radiant_team": "1,2,3"},
    ],
)
def test_clean_match_filters(override: dict) -> None:
    assert cm.clean_match(_raw_match(**override), VALID_HEROES, 20, TS, NAMES) is None


def test_clean_match_duration_boundary_kept() -> None:
    m = _raw_match(duration=cm.MIN_DURATION_S)
    assert cm.clean_match(m, VALID_HEROES, 0, TS, NAMES) is not None


# ------------------------------------------------------------------ get_patch
@pytest.mark.parametrize(
    "t, expected",
    [(50, "7.39"), (100, "7.39"), (199, "7.39"), (200, "7.40"), (299, "7.40"),
     (300, "7.41"), (10**10, "7.41")],
)
def test_get_patch_bisection_edges(t: int, expected: str) -> None:
    # before the first patch date falls back to the first patch; a patch starts AT its date
    assert cm.get_patch(t, TS, NAMES) == expected


# ----------------------------------------------------------------- checkpoint
def test_checkpoint_round_trip(tmp_path) -> None:
    assert cm.load_checkpoint(tmp_path) == {
        "last_match_id": None, "total_kept": 0, "shard_index": 0, "seen_ids": []}
    state = {"last_match_id": 999, "total_kept": 1234, "shard_index": 3}
    seen = {5, 7, 11}
    cm.save_checkpoint(tmp_path, state, seen)
    loaded = cm.load_checkpoint(tmp_path)
    assert set(loaded.pop("seen_ids")) == seen
    assert loaded == {"last_match_id": 999, "total_kept": 1234, "shard_index": 3}


def test_write_shard_schema(tmp_path) -> None:
    row = cm.clean_match(_raw_match(), VALID_HEROES, 0, TS, NAMES)
    cm.write_shard([row, {**row, "match_id": 43}], tmp_path, 7)
    df = pd.read_parquet(tmp_path / "matches_0007.parquet")
    assert list(df.columns) == cm.EXPECTED_COLS
    assert len(df) == 2


# ------------------------------------------------------------------- validate
def _good_frame(n: int = 4) -> pd.DataFrame:
    return pd.DataFrame({
        "match_id": np.arange(n, dtype=np.int64) + 1,
        "start_time": np.arange(n, dtype=np.int64) * 60,
        "duration": np.full(n, 1800, dtype=np.int64),
        "avg_rank_tier": np.full(n, 45, dtype=np.int64),
        "radiant_win": np.arange(n) % 2 == 0,
        "radiant_team": [np.array([1, 2, 3, 4, 5])] * n,
        "dire_team": [np.array([6, 7, 8, 9, 10])] * n,
        "patch": ["7.41"] * n,
    })


def test_validate_good_frame() -> None:
    assert validate(_good_frame(), VALID_HEROES) == []


def _mutations() -> list[tuple[str, callable]]:
    def dup(df):
        df.loc[1, "match_id"] = df.loc[0, "match_id"]
    def short(df):
        df.loc[0, "duration"] = 600
    def rank(df):
        df.loc[0, "avg_rank_tier"] = 0
    def null_patch(df):
        df.loc[0, "patch"] = None
    def four(df):
        df.at[0, "radiant_team"] = np.array([1, 2, 3, 4])
    def overlap(df):
        df.at[0, "dire_team"] = np.array([1, 7, 8, 9, 10])
    def unknown(df):
        df.at[0, "dire_team"] = np.array([6, 7, 8, 9, 77])
    return [("duplicate match_ids", dup), ("duration < 900", short),
            ("invalid avg_rank_tier", rank), ("null values in 'patch'", null_patch),
            ("malformed teams", four), ("malformed teams", overlap),
            ("unknown hero", unknown)]


@pytest.mark.parametrize("expected, mutate", _mutations())
def test_validate_bad_frames(expected: str, mutate) -> None:
    df = _good_frame()
    mutate(df)
    errors = validate(df, VALID_HEROES)
    assert any(expected in e for e in errors), errors


# --------------------------------------------------------------- load_matches
def test_load_matches_recursive_dedup_sorted(tmp_path) -> None:
    a = _good_frame(4)  # ids 1..4, times 0..180
    b = _good_frame(4)
    b["match_id"] += 2  # ids 3..6 -> 3, 4 duplicated
    b["start_time"] = [-60, 500, 600, 700]
    a.to_parquet(tmp_path / "matches_0000.parquet")
    (tmp_path / "patch-742").mkdir()
    b.to_parquet(tmp_path / "patch-742" / "matches_0000.parquet")
    (tmp_path / "notes.parquet").write_bytes(b"not a shard")  # ignored by the glob

    df = load_matches(tmp_path)
    assert sorted(df["match_id"]) == [1, 2, 3, 4, 5, 6]
    assert df["match_id"].is_unique
    assert df["start_time"].is_monotonic_increasing
    assert list(df.index) == list(range(len(df)))
    # keep="first": flat shard sorts before the subdir, so its id-3 row (t=120) wins
    assert int(df.loc[df["match_id"] == 3, "start_time"].iloc[0]) == 120


def test_load_matches_missing(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="collect"):
        load_matches(tmp_path)


# --------------------------------------------------------------------- splits
def test_chronological_split(synthetic_matches) -> None:
    df = synthetic_matches(1000).sample(frac=1.0, random_state=3)  # shuffled input
    s = chronological_split(df, val_frac=0.1, test_frac=0.2)
    assert (len(s.train), len(s.val), len(s.test)) == (700, 100, 200)
    assert s.name == "chrono-70-10-20"
    ids = [set(x["match_id"]) for x in (s.train, s.val, s.test)]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    assert sum(map(len, ids)) == len(df)
    assert s.train["start_time"].max() <= s.val["start_time"].min()
    assert s.val["start_time"].max() <= s.test["start_time"].min()
    s2 = chronological_split(df.sample(frac=1.0, random_state=9), 0.1, 0.2)
    pd.testing.assert_frame_equal(s.test, s2.test)  # deterministic regardless of order


def test_chronological_split_bad_fracs(synthetic_matches) -> None:
    with pytest.raises(ValueError):
        chronological_split(synthetic_matches(100), val_frac=0.5, test_frac=0.5)


def test_patch_split(synthetic_matches) -> None:
    df = synthetic_matches(1000)
    df.loc[600:, "patch"] = "7.42"
    s = patch_split(df, ["7.41"], "7.42", val_frac=0.1)
    assert (len(s.train), len(s.val), len(s.test)) == (540, 60, 400)
    assert set(s.test["patch"]) == {"7.42"}
    assert set(s.train["patch"]) | set(s.val["patch"]) == {"7.41"}
    assert s.train["start_time"].max() <= s.val["start_time"].min()
    assert s.name == "patch-7.41->7.42"
    with pytest.raises(ValueError):
        patch_split(df, ["7.40"], "7.42")


def test_vocab_round_trip(tmp_path, synthetic_matches, tiny_vocab) -> None:
    v = HeroVocab.from_matches(synthetic_matches(500))
    assert v.hero_ids == tiny_vocab.hero_ids
    v.save(tmp_path / "v.json")
    w = HeroVocab.load(tmp_path / "v.json")
    assert w.hero_ids == v.hero_ids and w.idx(129) == v.size - 1
    with pytest.raises(KeyError):
        w.idx(4)


# ---------------------------------------------------------------- explorer source
def test_fetch_explorer_page_cursor(monkeypatch) -> None:
    """A full page moves the cursor to the smallest id returned; a short page
    moves it to the bottom of the scanned id window, so no match is skipped."""
    monkeypatch.setattr(cm, "EXPLORER_LIMIT", 3)
    monkeypatch.setattr(cm, "EXPLORER_ID_WINDOW", 100)
    queries: list[str] = []

    def fake_query(session, sql):
        queries.append(sql)
        return rows

    monkeypatch.setattr(cm, "explorer_query", fake_query)

    rows = [{"match_id": i} for i in (999, 990, 950)]  # limit hit
    page, cursor = cm.fetch_explorer_page(None, 1000)
    assert len(page) == 3 and cursor == 950
    assert "match_id < 1000 and match_id >= 900" in queries[-1]
    assert f"lobby_type = {cm.RANKED_LOBBY}" in queries[-1]

    rows = [{"match_id": 990}]  # window exhausted before the limit
    assert cm.fetch_explorer_page(None, 1000)[1] == 900
    rows = []
    assert cm.fetch_explorer_page(None, 1000) == ([], 900)


def test_fetch_explorer_page_starts_at_newest(monkeypatch) -> None:
    def fake_query(session, sql):
        return [{"m": 5000}] if "max(match_id)" in sql else []

    monkeypatch.setattr(cm, "explorer_query", fake_query)
    assert cm.fetch_explorer_page(None, None)[1] == 5001 - cm.EXPLORER_ID_WINDOW
