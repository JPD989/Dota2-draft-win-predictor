"""Transformer and GNN tests: module-level checks plus the shared DraftModel tests."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

torch = pytest.importorskip("torch")

from src.data.dataset import HeroVocab  # noqa: E402
from src.models import checkpoint_path, get_model_class  # noqa: E402
from src.models.base import DraftModel, logit  # noqa: E402
from src.models.gnn import DraftGNN, GNNDraftModel, build_hero_graph  # noqa: E402
from src.models.transformer import DraftTransformer, TransformerDraftModel  # noqa: E402

from .conftest import HERO_IDS  # noqa: E402

SMALL = dict(batch_size=256, max_epochs=15, patience=3, device="cpu")
FACTORIES: dict[str, Callable[..., DraftModel]] = {
    "transformer": lambda **kw: TransformerDraftModel(d_model=32, n_heads=4, dim_ff=64,
                                                      lr=3e-3, **{**SMALL, **kw}),
    "gnn": lambda **kw: GNNDraftModel(d_model=32, lr=3e-3, **{**SMALL, **kw}),
}
EMPTY = pd.DataFrame({"radiant_team": [[]], "dire_team": [[]]})


@pytest.fixture(scope="module")
def data(synthetic_matches) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return synthetic_matches(2000, 0), synthetic_matches(500, 1), synthetic_matches(1000, 2)


@pytest.fixture(scope="module", params=list(FACTORIES))
def fitted(request, data, tiny_vocab) -> DraftModel:
    train, val, _ = data
    return FACTORIES[request.param]().fit(train, val, tiny_vocab)


def _batch(b: int = 8, seed: int = 0) -> torch.Tensor:
    """(b, 10) random draft index matrix with a few -1 empty slots."""
    g = np.random.default_rng(seed)
    teams = np.stack([g.permutation(len(HERO_IDS))[:10] for _ in range(b)])
    teams[0, 7:] = -1
    teams[1, :5] = -1
    return torch.as_tensor(teams, dtype=torch.long)


# --- module-level torch tests ---------------------------------------------------------
def test_transformer_forward_shape_and_tokens() -> None:
    net = DraftTransformer(len(HERO_IDS), d_model=16, n_heads=2, dim_ff=32, use_rank=True)
    tokens, team_ids = net.encode(_batch())
    assert tokens.shape == team_ids.shape == (8, 11)
    assert (tokens[:, 0] == len(HERO_IDS) + 1).all()           # CLS
    assert (tokens[0, 8:] == len(HERO_IDS)).all()              # empty slots -> MASK
    assert team_ids[0].tolist() == [0] + [1] * 5 + [2] * 5
    out = net(tokens, team_ids, torch.full((8, 1), 0.5))
    assert out.shape == (8,) and torch.isfinite(out).all()


def test_gnn_forward_shape_and_empty_team(data, tiny_vocab) -> None:
    graph = build_hero_graph(data[0], tiny_vocab)
    net = DraftGNN(tiny_vocab.size, d_model=16, use_rank=False)
    teams = _batch()
    teams[2] = -1                                               # all-empty draft
    out = net(teams, graph.a_syn, graph.a_cnt)
    assert out.shape == (8,) and torch.isfinite(out).all()


def _net_and_forward(kind: str, tiny_vocab: HeroVocab, train: pd.DataFrame):
    torch.manual_seed(0)
    if kind == "transformer":
        net = DraftTransformer(tiny_vocab.size, d_model=16, n_heads=2, dim_ff=32, dropout=0.0)
        return net, lambda t: net(*net.encode(t))
    graph = build_hero_graph(train, tiny_vocab)
    net = DraftGNN(tiny_vocab.size, d_model=16, dropout=0.0)
    return net, lambda t: net(t, graph.a_syn, graph.a_cnt)


@pytest.mark.parametrize("kind", ["transformer", "gnn"])
def test_gradients_flow_and_step_decreases_loss(kind, data, tiny_vocab) -> None:
    net, forward = _net_and_forward(kind, tiny_vocab, data[0])
    teams, y = _batch(32, seed=1), torch.as_tensor(np.arange(32) % 2, dtype=torch.float32)
    loss_fn = torch.nn.functional.binary_cross_entropy_with_logits
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    loss = loss_fn(forward(teams), y)
    before = loss.item()
    opt.zero_grad()
    loss.backward()
    grads = [p.grad for p in net.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert sum(float(g.abs().sum()) for g in grads) > 0
    opt.step()
    with torch.no_grad():
        assert loss_fn(forward(teams), y).item() < before


def test_build_hero_graph_properties(synthetic_matches) -> None:
    vocab = HeroVocab(HERO_IDS + [999])                          # hero 999 never played
    # 8k rows: at 2k the single planted counter pair is shrunk to exactly 0 by EB (tau2 = 0).
    graph = build_hero_graph(synthetic_matches(8000, 0), vocab)
    for a in (graph.a_syn, graph.a_cnt):
        a = a.numpy()
        assert a.shape == (vocab.size, vocab.size) and a.dtype == np.float32
        assert np.isfinite(a).all()
        assert np.all(np.diag(a) == 0)
        assert np.all(a[vocab.idx(999)] == 0) and np.all(a[:, vocab.idx(999)] == 0)
        assert np.all(np.abs(a).sum(axis=1) <= 1 + 1e-5)
    cnt = graph.a_cnt.numpy()
    assert (cnt > 0).any() and (cnt < 0).any()                   # signed
    np.testing.assert_array_equal(np.sign(cnt), -np.sign(cnt.T))  # antisymmetric signs kept
    assert (graph.a_syn.numpy() > 0).any()


# --- DraftModel tests -----------------------------------------------------------------
def test_registry_and_defaults_are_canonical() -> None:
    assert get_model_class("transformer") is TransformerDraftModel
    assert get_model_class("gnn") is GNNDraftModel
    for cls in (TransformerDraftModel, GNNDraftModel):
        params = inspect.signature(cls.__init__).parameters
        assert params["use_rank"].default is False
        assert params["mirror_augment"].default is False
        assert params["pick_dropout"].default == 0.0


def test_auc_beats_chance_on_planted_signal(fitted, data) -> None:
    _, _, test = data
    p = fitted.predict_proba(test)
    assert p.shape == (len(test),) and p.dtype == np.float64
    assert np.all((p >= 0) & (p <= 1))
    assert roc_auc_score(test["radiant_win"], p) > 0.55
    assert len(fitted.history) >= 1


def test_save_load_roundtrip_exact(fitted, data, tmp_path: Path) -> None:
    _, _, test = data
    path = checkpoint_path(fitted.name, tmp_path)
    fitted.save(path)
    loaded = type(fitted).load(path)
    assert loaded.is_fitted and loaded.vocab.hero_ids == fitted.vocab.hero_ids
    assert loaded.delta == fitted.delta
    np.testing.assert_array_equal(loaded.predict_proba(test), fitted.predict_proba(test))
    if isinstance(fitted, GNNDraftModel):
        torch.testing.assert_close(loaded.graph.a_syn, fitted.graph.a_syn)
        torch.testing.assert_close(loaded.graph.a_cnt, fitted.graph.a_cnt)


def test_partial_drafts(fitted, data) -> None:
    _, _, test = data
    partial = pd.DataFrame({
        "radiant_team": [t[:k] for t, k in zip(test["radiant_team"][:5], [0, 1, 2, 3, 5])],
        "dire_team": [t[:k] for t, k in zip(test["dire_team"][:5], [5, 0, 4, 1, 2])],
    })
    p = fitted.predict_proba(partial)
    assert p.shape == (5,) and np.all(np.isfinite(p)) and np.all((p > 0) & (p < 1))
    assert fitted.predict_proba(partial.iloc[:0]).shape == (0,)


@pytest.mark.parametrize("name", list(FACTORIES))
def test_empty_draft_near_base_rate(name, data, tiny_vocab) -> None:
    """Masking-trained models see partial drafts in-distribution (§4.15)."""
    train, val, _ = data
    model = FACTORIES[name](pick_dropout=0.3).fit(train, val, tiny_vocab)
    assert abs(model.predict_proba(EMPTY)[0] - train["radiant_win"].mean()) < 0.05


@pytest.mark.parametrize("name", list(FACTORIES))
def test_mirror_delta_restores_side_bias(name, data, tiny_vocab) -> None:
    train, val, test = data
    model = FACTORIES[name](mirror_augment=True).fit(train, val, tiny_vocab)
    base = train["radiant_win"].mean()
    assert model.delta == pytest.approx(float(logit(base)))
    assert abs(model.predict_proba(test).mean() - base) < 0.05


@pytest.mark.parametrize("name", list(FACTORIES))
def test_same_seed_same_predictions_and_rank_imputation(name, data, tiny_vocab) -> None:
    train, val, test = data
    a = FACTORIES[name](use_rank=True, max_epochs=2).fit(train, val, tiny_vocab)
    b = FACTORIES[name](use_rank=True, max_epochs=2).fit(train, val, tiny_vocab)
    np.testing.assert_array_equal(a.predict_proba(test), b.predict_proba(test))
    frame = test[["radiant_team", "dire_team"]]
    np.testing.assert_array_equal(a.predict_proba(frame),
                                  a.predict_proba(frame.assign(avg_rank_tier=50)))


def test_gnn_graph_built_from_train_only(data, tiny_vocab) -> None:
    train, val, _ = data
    model = FACTORIES["gnn"](max_epochs=1).fit(train, val, tiny_vocab)
    expected = build_hero_graph(train, tiny_vocab)
    torch.testing.assert_close(model.graph.a_syn, expected.a_syn)
    torch.testing.assert_close(model.graph.a_cnt, expected.a_cnt)


def test_tracker_receives_epoch_logs(data, tiny_vocab) -> None:
    class Recorder:
        def __init__(self) -> None:
            self.rows: list[dict] = []

        def log(self, d: dict, step: int | None = None) -> None:
            self.rows.append(d)

    rec = Recorder()
    train, val, _ = data
    FACTORIES["gnn"](max_epochs=2, patience=5, tracker=rec).fit(train, val, tiny_vocab)
    assert len(rec.rows) == 2 and {"epoch", "train_loss", "val_log_loss"} <= set(rec.rows[0])
