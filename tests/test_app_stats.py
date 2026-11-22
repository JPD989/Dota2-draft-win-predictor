"""Stats-mode formula sanity tests for the app's StatsDraftModel helper.

The app file is loaded by path (``app/`` is not a package); importing it must
not pull in Streamlit or torch.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.dataset import HeroVocab
from tests.conftest import SYNERGY_PAIRS

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "streamlit_app.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses need the module registered
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def app():
    return _load(APP, "streamlit_app_under_test")


def _frame(radiant: list[list[int]], dire: list[list[int]]) -> pd.DataFrame:
    return pd.DataFrame({"radiant_team": radiant, "dire_team": dire})


@pytest.fixture()
def toy(app):
    """4 heroes (ids 10, 20, 30, 40) with hand-set matrices."""
    vocab = HeroVocab([10, 20, 30, 40])
    syn = np.zeros((4, 4))
    syn[0, 1] = syn[1, 0] = 0.2
    syn[2, 3] = syn[3, 2] = -0.1
    syn[0, 2] = syn[2, 0] = 0.05
    cnt = np.zeros((4, 4))
    cnt[0, 2], cnt[2, 0] = 0.3, -0.3
    cnt[1, 3], cnt[3, 1] = -0.15, 0.15
    return app.StatsDraftModel(vocab, syn, cnt, radiant_wr_train=0.53)


def _logit(p: float) -> float:
    return float(np.log(p / (1 - p)))


def test_hand_computed_score(toy) -> None:
    # R = {10, 20}, D = {30, 40}:
    # base + syn[10,20] - syn[30,40] + cnt[10,30] + cnt[10,40] + cnt[20,30] + cnt[20,40]
    expected = _logit(0.53) + 0.2 - (-0.1) + 0.3 + 0.0 + 0.0 + (-0.15)
    s = toy.score(_frame([[10, 20]], [[30, 40]]))
    assert s[0] == pytest.approx(expected, rel=1e-12)
    p = toy.predict_proba(_frame([[10, 20]], [[30, 40]]))
    assert p.shape == (1,) and p.dtype == np.float64
    assert p[0] == pytest.approx(1 / (1 + np.exp(-expected)))


def test_three_hero_team_counts_each_pair_once(toy) -> None:
    expected = _logit(0.53) + 0.2 + 0.05 + 0.0  # pairs (10,20), (10,30), (20,30)
    assert toy.score(_frame([[10, 20, 30]], [[]]))[0] == pytest.approx(expected)


def test_empty_draft_is_base_rate(toy) -> None:
    assert toy.predict_proba(_frame([[]], [[]]))[0] == pytest.approx(0.53)


def test_side_swap_antisymmetry(toy) -> None:
    b = _logit(0.53)
    s1 = toy.score(_frame([[10, 20]], [[30, 40]]))[0]
    s2 = toy.score(_frame([[30, 40]], [[10, 20]]))[0]
    assert (s1 - b) == pytest.approx(-(s2 - b))


def test_batch_matches_rowwise(toy) -> None:
    df = _frame([[10], [20, 40], [], [10, 20, 30]], [[30, 40], [10], [20], [40]])
    batch = toy.predict_proba(df)
    rows = np.array([toy.predict_proba(df.iloc[[i]])[0] for i in range(len(df))])
    np.testing.assert_allclose(batch, rows)
    assert np.all((batch > 0) & (batch < 1))


def test_shape_validation(app) -> None:
    with pytest.raises(ValueError):
        app.StatsDraftModel(HeroVocab([1, 2]), np.zeros((3, 3)), np.zeros((2, 2)), 0.5)


def test_fit_on_synthetic_finds_planted_synergy(app, synthetic_matches, tiny_vocab) -> None:
    df = synthetic_matches(2000)
    m = app.StatsDraftModel(tiny_vocab, np.zeros((20, 20)), np.zeros((20, 20)), 0.5)
    m.fit(df, df.iloc[:0], tiny_vocab)
    assert m.radiant_wr_train == pytest.approx(df["radiant_win"].mean())
    a, b = SYNERGY_PAIRS[0]
    together = m.predict_proba(_frame([[a, b]], [[]]))[0]
    assert together > m.radiant_wr_train


def test_save_load_round_trip(app, toy, tmp_path) -> None:
    toy.save(tmp_path / "stats.npz")
    back = app.StatsDraftModel.load(tmp_path / "stats.npz")
    df = _frame([[10, 20], [40]], [[30], [10, 20]])
    np.testing.assert_array_equal(back.predict_proba(df), toy.predict_proba(df))
    assert back.vocab.hero_ids == toy.vocab.hero_ids and back.is_fitted


def test_artifacts_round_trip(app, synthetic_matches, tmp_path) -> None:
    build = _load(ROOT / "scripts" / "build_artifacts.py", "build_artifacts_under_test")
    df = synthetic_matches(2000)
    assert app.missing_artifacts(tmp_path) == list(app.REQUIRED_ARTIFACTS)
    meta = build.build_artifacts(df, tmp_path)
    (tmp_path / "heroes.json").write_text('{"1": "Anti-Mage"}')
    assert app.missing_artifacts(tmp_path) == []
    art = app.load_artifacts(tmp_path)
    assert meta["n_train"] == 1600  # train block of chrono-80-10-10 only
    model = app.StatsDraftModel.from_artifacts(art)
    p0 = model.predict_proba(_frame([[]], [[]]))[0]
    assert p0 == pytest.approx(meta["radiant_wr_train"])
    assert meta["radiant_wr_train"] == pytest.approx(df["radiant_win"].iloc[:1600].mean())


def test_find_checkpoints(app, tmp_path) -> None:
    rel = tmp_path / "release"
    rel.mkdir()
    for f in ("logreg.pkl", "logreg_rank.pkl", "xgboost.pkl", "transformer_rec.pt",
              "gnn.pt", "notes.txt", "logregx.pkl"):
        (tmp_path / f).touch()
    (rel / "logreg.pkl").touch()
    found = app.find_checkpoints((tmp_path, rel), include_torch=False)
    assert set(found) == {"logreg", "logreg [rank]", "xgboost", "logreg (release)"}
    found_t = app.find_checkpoints((tmp_path, rel), include_torch=True)
    assert found_t["transformer [rec]"] == ("transformer", tmp_path / "transformer_rec.pt")
    assert "gnn" in found_t


def test_app_import_is_light() -> None:
    code = (f"import importlib.util, sys; s = importlib.util.spec_from_file_location('a', "
            f"{str(APP)!r}); m = importlib.util.module_from_spec(s); sys.modules['a'] = m; "
            "s.loader.exec_module(m); "
            "assert 'torch' not in sys.modules, 'torch imported'; "
            "assert 'streamlit' not in sys.modules, 'streamlit imported'")
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT)


def test_recommender_with_stats_model(app, synthetic_matches, tiny_vocab) -> None:
    rec = pytest.importorskip("src.recommend.recommender")
    df = synthetic_matches(4000, seed=3)
    m = app.StatsDraftModel(tiny_vocab, np.zeros((20, 20)), np.zeros((20, 20)), 0.5)
    m.fit(df, df.iloc[:0], tiny_vocab)
    a, b = SYNERGY_PAIRS[0]
    out = rec.recommend(m, tiny_vocab, radiant=[a], dire=[], side="radiant", top_k=3)
    assert b in out["hero_id"].tolist()
    assert a not in rec.recommend(m, tiny_vocab, radiant=[a], dire=[],
                                  side="radiant")["hero_id"].tolist()


# ----------------------------------------------------------------- UI helpers (pure)
def test_load_hero_assets_optional(app, tmp_path) -> None:
    assert app.load_hero_assets(tmp_path) == {}  # missing file -> no portraits, no crash
    (tmp_path / app.HERO_ASSETS).write_text("not json")
    assert app.load_hero_assets(tmp_path) == {}
    (tmp_path / app.HERO_ASSETS).write_text('{"53": {"name": "furion", "img": "https://x/y.png"}}')
    assert app.load_hero_assets(tmp_path) == {53: {"name": "furion", "img": "https://x/y.png"}}


def test_committed_hero_assets_cover_vocab(app) -> None:
    assets = app.load_hero_assets()
    if not assets:
        pytest.skip("hero_assets.json not built")
    vocab = HeroVocab.load(ROOT / "data" / "artifacts" / "hero_vocab.json")
    assert set(vocab.hero_ids) <= set(assets)
    cdn = "https://cdn.cloudflare.steamstatic.com/"
    assert all(a["img"].startswith(cdn) and "?" not in a["img"] for a in assets.values())
    assert assets[53]["name"] == "furion"


def test_build_hero_assets_from_dotaconstants() -> None:
    fetch = _load(ROOT / "scripts" / "fetch_hero_constants.py", "fetch_hero_constants_under_test")
    out = fetch.build_hero_assets({"53": {
        "name": "npc_dota_hero_furion", "localized_name": "Nature's Prophet",
        "img": "/apps/dota2/images/dota_react/heroes/furion.png?", "icon": None,
        "primary_attr": "int", "attack_type": "Ranged", "roles": ["Carry"]}})
    assert out["53"]["name"] == "furion" and out["53"]["icon"] is None
    assert out["53"]["img"] == ("https://cdn.cloudflare.steamstatic.com"
                                "/apps/dota2/images/dota_react/heroes/furion.png")


def test_sanitize_draft_and_add_pick(app) -> None:
    r, d, b = app.sanitize_draft([1, 2, 2, 3, 4, 5, 6], [2, 7], [7, 8])
    assert (r, d, b) == ([1, 2, 3, 4, 5], [7], [8])
    state = {"radiant": [1], "dire": [2], "bans": [3]}
    assert app.add_pick(state, "radiant", 4) and state["radiant"] == [1, 4]
    assert not app.add_pick(state, "dire", 3)  # banned
    assert not app.add_pick(state, "dire", 1)  # taken by the other side
    assert not app.add_pick({"radiant": [1, 2, 3, 4, 5]}, "radiant", 9)  # full
    assert not app.add_pick(state, "spectators", 9)


def test_labels(app) -> None:
    assert app.rank_label(53) == "Legend 3" and app.rank_label(75) == "Divine 5"
    assert app.model_title(app.STATS_LABEL) == "Stats heuristic"
    assert app.model_title("transformer [rec] (release)") == "Transformer [rec] (release)"


def test_draft_hero_stats_and_pairs(app, synthetic_matches, tmp_path) -> None:
    build = _load(ROOT / "scripts" / "build_artifacts.py", "build_artifacts_under_test")
    build.build_artifacts(synthetic_matches(2000), tmp_path)
    (tmp_path / "heroes.json").write_text("{}")
    art = app.load_artifacts(tmp_path)
    a, b, c = art.vocab.hero_ids[:3]
    stats = app.draft_hero_stats(art, [a, b], [c])
    assert [s["side"] for s in stats] == ["radiant", "radiant", "dire"]
    assert all(s["ci_lo"] <= s["wr"] <= s["ci_hi"] for s in stats)
    pairs = app.draft_pairs(art, [a, b], [c], top=10)
    assert {p["kind"] for p in pairs} <= {"synergy", "counter"}
    assert all(p["value"] != 0 for p in pairs)
    assert [abs(p["value"]) for p in pairs] == sorted((abs(p["value"]) for p in pairs), reverse=True)
    assert app.draft_pairs(art, [], []) == []


def test_components_escape_and_fallback() -> None:
    ui = _load(ROOT / "app" / "components.py", "components_under_test")
    html = ui.team("radiant", [1], {1: "<b>Evil</b>"}, {})
    assert "<b>Evil</b>" not in html and "&lt;b&gt;" in html
    assert "dd-img" not in html  # no asset -> initials only
    assert html.count("dd-empty") == 4
    assert ui.initials("Nature's Prophet") == "NP" and ui.initials("Anti-Mage") == "AM"
    assert "Radiant +" in ui.win_bar(0.6, 1, 1) and "Even" in ui.win_bar(0.5, 1, 1)


def test_is_experimental_rule(app) -> None:
    """Only neural models trained without pick masking are flagged, and only mid-draft."""
    class Fake:
        def __init__(self, name: str, pick_dropout: float | None = None) -> None:
            self.name = name
            if pick_dropout is not None:
                self.pick_dropout = pick_dropout

    for name in ("stats", "logreg", "xgboost"):
        assert not app.is_experimental(Fake(name), partial=True)
    assert app.is_experimental(Fake("gnn", 0.0), partial=True)
    assert app.is_experimental(Fake("transformer", 0.0), partial=True)
    assert not app.is_experimental(Fake("transformer", 0.15), partial=True)
    assert not app.is_experimental(Fake("gnn", 0.0), partial=False)
