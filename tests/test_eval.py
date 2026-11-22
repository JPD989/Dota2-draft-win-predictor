"""Tests for src.eval.metrics, src.eval.calibration and src.eval.harness."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.data.dataset import HeroVocab, Split, chronological_split
from src.eval import calibration as cal
from src.eval.calibration import (
    expected_calibration_error,
    fit_isotonic,
    fit_platt,
    plot_reliability,
    reliability_curve,
)
from src.eval.harness import TABLE_COLUMNS, run_benchmark
from src.eval.metrics import (
    auc_score,
    bootstrap_auc_ci,
    bootstrap_auc_matrix,
    compute_metrics,
    paired_bootstrap_delta_auc,
)
from src.features.encoding import labels, one_hot_drafts, teams_to_matrix
from src.models.base import DraftModel


# --------------------------------------------------------------------------- stub model
class StubLogReg(DraftModel):
    """Tiny sklearn logistic regression on one-hot drafts (test double, no models/ dependency).

    ``pairs=True`` adds same-team pair indicators (radiant pair minus dire pair) so the
    planted synergy is learnable. Records what ``fit`` saw and how often it predicts.
    """

    def __init__(self, name: str = "logreg", C: float = 1.0, pairs: bool = False) -> None:
        self.name = name
        self.use_rank = False
        self.vocab: HeroVocab | None = None
        self.C, self.pairs = C, pairs
        self.fit_ids: set[int] = set()
        self.n_fit_calls = 0
        self.n_predict_calls = 0
        self.last_predict_frame: pd.DataFrame | None = None

    def _features(self, df: pd.DataFrame) -> np.ndarray:
        assert self.vocab is not None
        v = self.vocab.size
        x = one_hot_drafts(teams_to_matrix(df, self.vocab), v).toarray()
        if not self.pairs:
            return x
        r, d = x[:, :v], x[:, v:]
        iu = np.triu_indices(v, k=1)
        pair = (r[:, :, None] * r[:, None, :] - d[:, :, None] * d[:, None, :])[:, iu[0], iu[1]]
        return np.hstack([x, pair])

    def fit(self, train: pd.DataFrame, val: pd.DataFrame, vocab: HeroVocab) -> "StubLogReg":
        self.vocab = vocab
        self.n_fit_calls += 1
        self.fit_ids = set(train["match_id"]) | set(val["match_id"])
        self._clf = LogisticRegression(C=self.C, max_iter=2000).fit(self._features(train), labels(train))
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        self.n_predict_calls += 1
        self.last_predict_frame = df
        return self._clf.predict_proba(self._features(df))[:, 1].astype(np.float64)

    def save(self, path: Path) -> None:
        path.write_bytes(pickle.dumps(self))

    @classmethod
    def load(cls, path: Path) -> "StubLogReg":
        return pickle.loads(path.read_bytes())


class ConstantModel(StubLogReg):
    """Predicts the train base rate for every row (AUC 0.5)."""

    def fit(self, train, val, vocab):  # type: ignore[override]
        self.vocab = vocab
        self.n_fit_calls += 1
        self.fit_ids = set(train["match_id"]) | set(val["match_id"])
        self._rate = float(labels(train).mean())
        return self

    def predict_proba(self, df):  # type: ignore[override]
        self.n_predict_calls += 1
        return np.full(len(df), self._rate)


@pytest.fixture()
def split(synthetic_matches) -> Split:
    return chronological_split(synthetic_matches(2000, seed=0))


# --------------------------------------------------------------------------- metrics
def test_auc_matches_sklearn_and_handles_single_class():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500).astype(float)
    p = np.round(rng.random(500), 2)  # ties on purpose
    assert auc_score(y, p) == pytest.approx(roc_auc_score(y, p), abs=1e-12)
    assert np.isnan(auc_score(np.ones(10), rng.random(10)))


def test_compute_metrics_keys_and_values():
    y = np.array([0, 0, 1, 1], dtype=float)
    p = np.array([0.1, 0.4, 0.35, 0.8])
    m = compute_metrics(y, p)
    assert set(m) == {"auc", "log_loss", "brier", "ece", "ece_q", "n"}
    assert m["auc"] == pytest.approx(0.75)
    assert m["brier"] == pytest.approx(np.mean((p - y) ** 2))
    assert m["log_loss"] == pytest.approx(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    assert m["n"] == 4
    assert m["ece"] == expected_calibration_error(y, p, 15, "uniform")
    assert m["ece_q"] == expected_calibration_error(y, p, 15, "quantile")
    # log-loss stays finite on hard 0/1 predictions
    assert np.isfinite(compute_metrics(y, np.array([1.0, 0.0, 0.0, 1.0]))["log_loss"])


def test_compute_metrics_rejects_bad_input():
    with pytest.raises(ValueError):
        compute_metrics([0, 1], [0.5])
    with pytest.raises(ValueError):
        compute_metrics([0, 1], [0.5, 1.2])


def test_bootstrap_auc_ci_brackets_point_estimate():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 400).astype(float)
    p = np.clip(0.5 + 0.2 * (y - 0.5) + rng.normal(0, 0.2, 400), 0, 1)
    lo, hi = bootstrap_auc_ci(y, p, n_boot=200, seed=3)
    assert lo < auc_score(y, p) < hi
    assert (lo, hi) == bootstrap_auc_ci(y, p, n_boot=200, seed=3)  # deterministic


def test_paired_bootstrap_identical_models_zero_width():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 300).astype(float)
    p = rng.random(300)
    assert paired_bootstrap_delta_auc(y, p, p.copy(), n_boot=100) == (0.0, 0.0, 0.0)


def test_paired_bootstrap_shares_indices():
    """Paired CI is a difference of the SAME replicates and is tighter than the
    naive independent combination when the models are correlated."""
    rng = np.random.default_rng(4)
    n = 600
    y = rng.integers(0, 2, n).astype(float)
    base = 0.6 * y + rng.normal(0, 0.5, n)
    p_a = 1 / (1 + np.exp(-(base + rng.normal(0, 0.1, n))))
    p_b = 1 / (1 + np.exp(-(base + rng.normal(0, 0.1, n))))
    delta, lo, hi = paired_bootstrap_delta_auc(y, p_a, p_b, n_boot=300, seed=7)
    _, mat = bootstrap_auc_matrix(y, {"a": p_a, "b": p_b}, n_boot=300, seed=7)
    d = mat[:, 0] - mat[:, 1]
    assert (lo, hi) == pytest.approx(tuple(np.percentile(d, [2.5, 97.5])))
    assert delta == pytest.approx(auc_score(y, p_a) - auc_score(y, p_b))
    a_lo, a_hi = bootstrap_auc_ci(y, p_a, n_boot=300, seed=7)
    assert hi - lo < a_hi - a_lo


def test_bootstrap_degenerate_resamples_are_dropped():
    y = np.array([1.0] * 9 + [0.0])  # many resamples draw no negative
    p = np.linspace(0.1, 0.9, 10)
    _, mat = bootstrap_auc_matrix(y, {"m": p}, n_boot=200, seed=0)
    assert np.isnan(mat).any() and np.isfinite(mat).any()
    lo, hi = bootstrap_auc_ci(y, p, n_boot=200, seed=0)
    assert np.isfinite(lo) and np.isfinite(hi)
    assert all(np.isnan(v) for v in bootstrap_auc_ci(np.ones(5), np.full(5, 0.5), n_boot=20))


# --------------------------------------------------------------------------- calibration
def test_reliability_curve_schema_and_counts():
    rng = np.random.default_rng(5)
    p = rng.random(3000)
    y = (rng.random(3000) < p).astype(float)
    for strategy in ("uniform", "quantile"):
        c = reliability_curve(y, p, n_bins=10, strategy=strategy)
        assert list(c.columns) == cal.CURVE_COLUMNS
        assert c["count"].sum() == 3000
        assert (c["ci_lo"] <= c["y_rate"]).all() and (c["y_rate"] <= c["ci_hi"]).all()
    q = reliability_curve(y, p, n_bins=10, strategy="quantile")
    assert q["count"].min() >= 290 and q["count"].max() <= 310  # equal mass
    with pytest.raises(ValueError):
        reliability_curve(y, p, strategy="bogus")


def test_ece_hand_computed_and_calibrated_is_small():
    y = np.array([1, 0, 1, 1], dtype=float)
    p = np.array([0.1, 0.1, 0.9, 0.9])
    # bins: {0.1: y_rate 0.5, gap 0.4}, {0.9: y_rate 1.0, gap 0.1} -> (2*0.4 + 2*0.1)/4
    assert expected_calibration_error(y, p, n_bins=10) == pytest.approx(0.25)
    rng = np.random.default_rng(6)
    p = rng.random(50_000)
    y = (rng.random(50_000) < p).astype(float)
    assert expected_calibration_error(y, p) < 0.02
    assert expected_calibration_error(y, p, strategy="quantile") < 0.02
    # p == 1.0 must land in the last bin, not crash
    assert np.isfinite(expected_calibration_error([1.0, 0.0], [1.0, 0.0]))


@pytest.mark.parametrize("fitter", [fit_isotonic, fit_platt])
def test_recalibrators_clip_and_extrapolate(fitter):
    rng = np.random.default_rng(8)
    p_val = rng.uniform(0.3, 0.7, 2000)
    y_val = (rng.random(2000) < p_val).astype(float)
    f = fitter(y_val, p_val)
    p_test = np.array([0.0, 1e-9, 0.01, 0.5, 0.99, 1.0])  # far outside the val range
    out = f(p_test)
    assert out.dtype == np.float64 and out.shape == p_test.shape
    assert out.min() >= 1e-6 and out.max() <= 1 - 1e-6
    assert np.all(np.diff(out) >= 0)  # monotone
    y_test = np.array([1.0, 1.0, 0.0, 1.0, 0.0, 0.0])  # adversarial: worst-case labels at extremes
    assert np.isfinite(compute_metrics(y_test, out)["log_loss"])


def test_isotonic_extreme_val_still_finite():
    """Isotonic hits exact 0/1 when val ends are pure; clipping must keep log-loss finite."""
    p_val = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    y_val = np.array([0, 0, 0, 1, 1, 1], dtype=float)
    out = fit_isotonic(y_val, p_val)(np.array([0.05, 0.95]))
    assert np.isfinite(compute_metrics(np.array([1.0, 0.0]), out)["log_loss"])


def test_plot_reliability_writes_png(tmp_path):
    rng = np.random.default_rng(9)
    p = rng.random(500)
    y = (rng.random(500) < p).astype(float)
    out = tmp_path / "sub" / "rel.png"
    plot_reliability({"a": reliability_curve(y, p), "b": reliability_curve(y, p, strategy="quantile")}, out)
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


# --------------------------------------------------------------------------- harness
def _assert_outputs(out_dir: Path, names: list[str], split: Split) -> dict:
    for f in ("metrics.json", "predictions.parquet", "table.md", "reliability.png"):
        assert (out_dir / f).exists(), f
    text = (out_dir / "metrics.json").read_text()
    assert "NaN" not in text and "Infinity" not in text
    meta = json.loads(text)
    assert meta["split"] == split.name and meta["seed"] == 42
    assert (meta["n_train"], meta["n_val"], meta["n_test"]) == (len(split.train), len(split.val), len(split.test))
    assert list(meta["models"]) == names
    for entry in meta["models"].values():
        assert list(entry) == [
            "auc", "auc_lo", "auc_hi", "delta_auc_vs_logreg", "delta_lo", "delta_hi",
            "log_loss", "brier", "ece", "ece_q", "n", "fit_seconds",
        ]
        assert entry["n"] == len(split.test)
    pred = pd.read_parquet(out_dir / "predictions.parquet")
    assert list(pred.columns) == ["match_id", "y", *names]
    assert pred["match_id"].dtype == np.int64
    assert all(pred[c].dtype == np.float64 for c in ["y", *names])
    np.testing.assert_array_equal(pred["match_id"].to_numpy(), split.test["match_id"].to_numpy())
    return meta


def test_harness_end_to_end_stub(tmp_path, split, tiny_vocab):
    models = [StubLogReg("logreg"), StubLogReg("pairs", pairs=True, C=0.5), ConstantModel("const")]
    res = run_benchmark(models, split, tiny_vocab, tmp_path, n_boot=100)
    meta = _assert_outputs(tmp_path, ["logreg", "pairs", "const"], split)

    test_ids = set(split.test["match_id"])
    for m in models:
        assert m.n_fit_calls == 1 and not (m.fit_ids & test_ids)  # never fit on test
    t = res.table
    assert list(t.columns) == TABLE_COLUMNS
    assert t["auc"].is_monotonic_decreasing
    assert t.loc["logreg", ["delta_auc_vs_logreg", "delta_lo", "delta_hi"]].tolist() == [0.0, 0.0, 0.0]
    assert t.loc["pairs", "auc"] > t.loc["logreg", "auc"] > 0.55
    assert t.loc["pairs", "delta_lo"] <= t.loc["pairs", "delta_auc_vs_logreg"] <= t.loc["pairs", "delta_hi"]
    assert t.loc["const", "auc"] == pytest.approx(0.5)
    assert (t["n_test"] == len(split.test)).all() and (t["fit_seconds"] > 0).all()
    assert set(res.predictions) == {"logreg", "pairs", "const"}
    assert meta["models"]["pairs"]["auc"] == pytest.approx(t.loc["pairs", "auc"])
    md = (tmp_path / "table.md").read_text()
    assert md.index("| pairs ") < md.index("| logreg ") < md.index("| const ")


def test_harness_recalibrate_adds_isotonic_rows(tmp_path, split, tiny_vocab):
    res = run_benchmark([StubLogReg("logreg"), StubLogReg("other", C=0.05)], split, tiny_vocab,
                        tmp_path, recalibrate=True, n_boot=50)
    names = ["logreg", "other", "logreg+isotonic", "other+isotonic"]
    meta = _assert_outputs(tmp_path, names, split)
    assert set(res.table.index) == set(names)
    for n in ("logreg+isotonic", "other+isotonic"):
        assert np.isfinite(res.table.loc[n, "log_loss"])
        assert np.isfinite(meta["models"][n]["delta_auc_vs_logreg"])
    # isotonic on val, applied to test: the map equals a fresh fit on the model's val predictions
    m = StubLogReg("logreg").fit(split.train, split.val, tiny_vocab)
    iso = fit_isotonic(labels(split.val), m.predict_proba(split.val))
    np.testing.assert_allclose(res.predictions["logreg+isotonic"], iso(m.predict_proba(split.test)))


def test_harness_prefitted_and_no_logreg(tmp_path, split, tiny_vocab):
    pre = StubLogReg("a").fit(split.train, split.val, tiny_vocab)
    fresh = StubLogReg("b", C=0.1)
    res = run_benchmark([pre, fresh], split, tiny_vocab, tmp_path, n_boot=30)
    assert pre.n_fit_calls == 1  # not refit
    assert np.isnan(res.table.loc["a", "fit_seconds"]) and res.table.loc["b", "fit_seconds"] > 0
    assert res.table[["delta_auc_vs_logreg", "delta_lo", "delta_hi"]].isna().all().all()
    meta = json.loads((tmp_path / "metrics.json").read_text())
    assert meta["models"]["a"]["fit_seconds"] is None
    assert meta["models"]["b"]["delta_auc_vs_logreg"] is None


def test_harness_shared_bootstrap_identical_models(tmp_path, split, tiny_vocab):
    """Two identical models on shared indices: delta exactly 0, identical AUC CIs."""
    res = run_benchmark([StubLogReg("logreg"), StubLogReg("twin")], split, tiny_vocab, tmp_path, n_boot=50)
    t = res.table
    assert t.loc["twin", ["delta_auc_vs_logreg", "delta_lo", "delta_hi"]].tolist() == [0.0, 0.0, 0.0]
    assert t.loc["twin", ["auc_lo", "auc_hi"]].tolist() == t.loc["logreg", ["auc_lo", "auc_hi"]].tolist()


def test_harness_rejects_duplicate_names(tmp_path, split, tiny_vocab):
    with pytest.raises(ValueError, match="unique"):
        run_benchmark([StubLogReg("x"), StubLogReg("x")], split, tiny_vocab, tmp_path)


def test_harness_logreg_xgboost_end_to_end(tmp_path, split, tiny_vocab):
    """Real baselines through the registry; skipped until src/models/baselines.py exists."""
    from src.models import get_model_class

    try:
        classes = [get_model_class("logreg"), get_model_class("xgboost")]
    except (ImportError, ModuleNotFoundError) as exc:
        pytest.skip(f"baseline models not available yet: {exc}")
    models = [cls() for cls in classes]
    res = run_benchmark(models, split, tiny_vocab, tmp_path, recalibrate=True, n_boot=50)
    names = [m.name for m in models]
    _assert_outputs(tmp_path, [*names, *(f"{n}+isotonic" for n in names)], split)
    assert {"delta_auc_vs_logreg", "delta_lo", "delta_hi"} <= set(res.table.columns)
    assert np.isfinite(res.table.loc["xgboost", "delta_auc_vs_logreg"])
    assert np.isfinite(res.table.loc["xgboost+isotonic", "log_loss"])
