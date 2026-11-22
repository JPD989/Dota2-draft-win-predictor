"""Tests for src.eval.drift."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.eval.drift import patch_drift_study, time_drift_study
from tests.test_eval import StubLogReg

TIME_COLS = [
    "model", "window", "n", "window_start", "window_end", "days_since_train_end",
    "auc", "auc_lo", "auc_hi", "log_loss", "brier", "ece",
]


def test_time_drift_windows_strictly_after_val(synthetic_matches, tiny_vocab, tmp_path, capsys):
    df = synthetic_matches(2000, seed=0).sample(frac=1.0, random_state=0)  # shuffled input
    built: list[StubLogReg] = []

    def factory() -> StubLogReg:
        built.append(StubLogReg("logreg"))
        return built[-1]

    res = time_drift_study(df, {"logreg": factory}, tiny_vocab, n_windows=4, out_dir=tmp_path, n_boot=50)

    assert list(res.columns) == TIME_COLS
    assert len(res) == 4 and res["window"].tolist() == [0, 1, 2, 3]
    assert res["n"].sum() == 2000 - int(2000 * 0.6) and res["n"].max() - res["n"].min() <= 1

    ordered = df.sort_values(["start_time", "match_id"]).reset_index(drop=True)
    val_end = pd.Timestamp(int(ordered["start_time"].iloc[int(2000 * 0.6) - 1]), unit="s", tz="UTC")
    assert (res["window_start"] > val_end).all()
    assert (res["window_start"].iloc[1:].to_numpy() > res["window_end"].iloc[:-1].to_numpy()).all()
    assert res["days_since_train_end"].is_monotonic_increasing and (res["days_since_train_end"] > 0).all()

    # fit saw exactly the front 60% (54% train + 6% val), nothing later
    assert len(built) == 1 and built[0].fit_ids == set(ordered["match_id"].iloc[: int(2000 * 0.6)])
    assert (res["auc_lo"] <= res["auc"]).all() and (res["auc"] <= res["auc_hi"]).all()

    assert "WARNING" in capsys.readouterr().out  # 2000 min-spaced matches span ~1 day
    assert (tmp_path / "time_drift.csv").exists() and (tmp_path / "time_drift.png").exists()


def test_time_drift_no_warning_on_long_span(synthetic_matches, tiny_vocab, tmp_path, capsys):
    df = synthetic_matches(600, seed=1)
    df["start_time"] = df["start_time"].iloc[0] + np.arange(len(df)) * 7200  # 2 h apart -> 20 days in back 40%
    time_drift_study(df, {"a": lambda: StubLogReg("a")}, tiny_vocab, n_windows=2, out_dir=tmp_path, n_boot=10)
    assert "WARNING" not in capsys.readouterr().out


def _two_patch_frame(synthetic_matches) -> pd.DataFrame:
    df = synthetic_matches(2000, seed=2)
    df["patch"] = np.where(np.arange(len(df)) < 1400, "7.41", "7.42")
    return df


def test_patch_drift_requires_10k_rows(synthetic_matches, tiny_vocab, tmp_path):
    with pytest.raises(ValueError, match=r"10,000 rows per patch.*'7\.41': 1,400.*'7\.42': 600"):
        patch_drift_study(_two_patch_frame(synthetic_matches), {"a": lambda: StubLogReg("a")}, tiny_vocab,
                          train_patch="7.41", test_patch="7.42", out_dir=tmp_path)
    with pytest.raises(ValueError, match="'7.43': 0"):
        patch_drift_study(_two_patch_frame(synthetic_matches), {"a": lambda: StubLogReg("a")}, tiny_vocab,
                          train_patch="7.41", test_patch="7.43", out_dir=tmp_path, min_rows=100)


def test_patch_drift_study_splits_and_columns(synthetic_matches, tiny_vocab, tmp_path):
    df = _two_patch_frame(synthetic_matches)
    built: list[StubLogReg] = []

    def factory() -> StubLogReg:
        built.append(StubLogReg("logreg"))
        return built[-1]

    res = patch_drift_study(df, {"logreg": factory}, tiny_vocab, train_patch="7.41", test_patch="7.42",
                            out_dir=tmp_path, n_boot=50, min_rows=500)
    assert len(res) == 1
    row = res.iloc[0]
    assert (row["n_train"], row["n_within"], row["n_next"]) == (1400 - 2 * 140, 140, 600)
    for k in ("auc", "log_loss", "brier", "ece"):
        assert row[f"delta_{k}"] == pytest.approx(row[f"{k}_next"] - row[f"{k}_within"])
    assert row["delta_auc_lo"] <= row["delta_auc_hi"]
    # fit used only train-patch train+val rows (no within-test, no next-patch rows)
    train_patch = df[df["patch"] == "7.41"].sort_values(["start_time", "match_id"])
    assert built[0].fit_ids == set(train_patch["match_id"].iloc[:1260])
    assert (tmp_path / "patch_drift.csv").exists() and (tmp_path / "patch_drift.png").exists()
