"""The shared evaluation harness — the single place the fairness guarantee lives.

Protocol (docs/architecture.md §4.13), in this order:

1. **Fit** every unfitted model on ``split.train`` (+ ``split.val`` for early
   stopping). Test rows are never passed to ``fit``. Pre-fitted models are
   used as-is (``fit_seconds = NaN``).
2. **Recalibrate** (optional): an isotonic map is fit on each model's
   *validation* predictions and applied to its test predictions, giving an
   extra ``"{name}+isotonic"`` entry.
3. **Evaluate** every entry on the identical test set: point metrics, a
   bootstrap AUC CI and a paired delta-AUC vs ``"logreg"``. All entries share
   ONE set of bootstrap resample indices (one index set per replicate).
4. **Write** ``metrics.json``, ``predictions.parquet``, ``table.md`` and
   ``reliability.png`` into ``out_dir``.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import SEED
from src.data.dataset import HeroVocab, Split
from src.eval.calibration import fit_isotonic, plot_reliability, reliability_curve
from src.eval.metrics import auc_score, bootstrap_auc_matrix, compute_metrics, percentile_ci
from src.models.base import DraftModel

__all__ = ["BenchmarkResult", "TABLE_COLUMNS", "run_benchmark"]

REFERENCE_MODEL: str = "logreg"
TABLE_COLUMNS: list[str] = [
    "auc", "auc_lo", "auc_hi", "delta_auc_vs_logreg", "delta_lo", "delta_hi",
    "log_loss", "brier", "ece", "ece_q", "n_test", "fit_seconds",
]
_JSON_KEYS: list[str] = [
    "auc", "auc_lo", "auc_hi", "delta_auc_vs_logreg", "delta_lo", "delta_hi",
    "log_loss", "brier", "ece", "ece_q", "n", "fit_seconds",
]


@dataclass
class BenchmarkResult:
    """Harness output.

    Attributes
    ----------
    table : pd.DataFrame
        Index = entry name (models, plus ``"{name}+isotonic"`` when
        recalibrating), columns :data:`TABLE_COLUMNS`, sorted by AUC desc.
    predictions : dict[str, np.ndarray]
        Entry name -> test-set P(radiant win), aligned with ``split.test`` rows.
    """

    table: pd.DataFrame
    predictions: dict[str, np.ndarray]


def _checked_predict(model: DraftModel, df: pd.DataFrame) -> np.ndarray:
    """Predict and validate shape / range — a bad model must fail loudly here."""
    p = np.asarray(model.predict_proba(df), dtype=np.float64)
    if p.shape != (len(df),):
        raise ValueError(f"{model.name}: predict_proba returned shape {p.shape}, expected ({len(df)},)")
    if not np.all(np.isfinite(p)) or p.min() < 0.0 or p.max() > 1.0:
        raise ValueError(f"{model.name}: predictions must be finite and in [0, 1]")
    return p


def _json_safe(value: Any) -> Any:
    """Recursively convert NumPy scalars to Python and NaN/inf to ``None``."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        f = float(value)
        return f if math.isfinite(f) else None
    return value


def _fmt(value: Any, col: str) -> str:
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return "–"
    if col == "n_test":
        return str(int(value))
    if col == "fit_seconds":
        return f"{float(value):.2f}"
    if col.startswith("delta"):
        return f"{float(value):+.4f}"
    return f"{float(value):.4f}"


def _table_markdown(table: pd.DataFrame, split_name: str) -> str:
    """Render the result table as GitHub markdown (no tabulate dependency)."""
    header = ["model", *table.columns]
    lines = [
        f"Benchmark on split `{split_name}` (95% bootstrap CIs; delta = paired vs `{REFERENCE_MODEL}`)",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] + ["---:"] * len(table.columns)) + "|",
    ]
    for name, row in table.iterrows():
        cells = [str(name)] + [_fmt(row[c], c) for c in table.columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def run_benchmark(
    models: Sequence[DraftModel],
    split: Split,
    vocab: HeroVocab,
    out_dir: Path,
    *,
    recalibrate: bool = False,
    seed: int = SEED,
    n_boot: int = 1000,
) -> BenchmarkResult:
    """Fit (if needed) and evaluate models on one split under identical conditions.

    Parameters
    ----------
    models : Sequence[DraftModel]
        Models to compare; ``name`` must be unique. Unfitted models are fit
        on ``split.train`` / ``split.val``; fitted ones are evaluated as-is.
    split : Split
        Train/val/test partition; every entry is scored on ``split.test``.
    vocab : HeroVocab
        Vocabulary passed to ``fit``.
    out_dir : Path
        Output directory (created if missing).
    recalibrate : bool, optional
        If True, add ``"{name}+isotonic"`` entries: isotonic fit on val
        predictions, applied to test predictions. Default False.
    seed : int, optional
        Bootstrap seed, by default ``SEED``.
    n_boot : int, optional
        Bootstrap replicates for AUC CIs and delta-AUC, by default 1000
        (an addition to the §4.13 signature so tests can stay fast).

    Returns
    -------
    BenchmarkResult
        Result table and test predictions per entry.

    Raises
    ------
    ValueError
        On an empty model list, duplicate names, an empty test split, or
        malformed predictions.
    """
    if not models:
        raise ValueError("run_benchmark needs at least one model")
    names = [m.name for m in models]
    if len(set(names)) != len(names):
        raise ValueError(f"model names must be unique, got {names}")
    if len(split.test) == 0:
        raise ValueError("test split is empty")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Fit — train and val only. Test data is not touched until every fit is done.
    fit_seconds: dict[str, float] = {}
    for model in models:
        if model.is_fitted:
            fit_seconds[model.name] = float("nan")
        else:
            t0 = time.perf_counter()
            model.fit(split.train, split.val, vocab)
            fit_seconds[model.name] = time.perf_counter() - t0

    # 2. Predict on test (+ optional isotonic recalibration fit on val).
    y_test = split.test["radiant_win"].to_numpy(dtype=np.float64)
    preds: dict[str, np.ndarray] = {}
    for model in models:
        preds[model.name] = _checked_predict(model, split.test)
    if recalibrate:
        if len(split.val) == 0:
            raise ValueError("recalibrate=True needs a non-empty validation split")
        y_val = split.val["radiant_win"].to_numpy(dtype=np.float64)
        for model in models:
            iso = fit_isotonic(y_val, _checked_predict(model, split.val))
            name = f"{model.name}+isotonic"
            preds[name] = iso(preds[model.name])
            fit_seconds[name] = fit_seconds[model.name]  # cost of the base model

    # 3. Evaluate: point metrics + ONE shared bootstrap resample set for all entries.
    entry_names, boot = bootstrap_auc_matrix(y_test, preds, n_boot=n_boot, seed=seed)
    ref = entry_names.index(REFERENCE_MODEL) if REFERENCE_MODEL in preds else None
    rows: dict[str, dict[str, float]] = {}
    for j, name in enumerate(entry_names):
        m = compute_metrics(y_test, preds[name])
        auc_lo, auc_hi = percentile_ci(boot[:, j])
        if ref is None:
            delta = d_lo = d_hi = float("nan")
        else:
            delta = m["auc"] - auc_score(y_test, preds[REFERENCE_MODEL])
            d_lo, d_hi = percentile_ci(boot[:, j] - boot[:, ref])
        rows[name] = {
            "auc": m["auc"], "auc_lo": auc_lo, "auc_hi": auc_hi,
            "delta_auc_vs_logreg": delta, "delta_lo": d_lo, "delta_hi": d_hi,
            "log_loss": m["log_loss"], "brier": m["brier"], "ece": m["ece"],
            "ece_q": m["ece_q"], "n": m["n"], "fit_seconds": fit_seconds[name],
        }

    table = pd.DataFrame.from_dict(rows, orient="index").rename(columns={"n": "n_test"})
    table = table[TABLE_COLUMNS].astype({"n_test": np.int64})
    table.index.name = "model"
    table = table.sort_values("auc", ascending=False, kind="stable", na_position="last")

    # 4. Write outputs.
    metrics = {
        "split": split.name,
        "seed": int(seed),
        "n_train": len(split.train),
        "n_val": len(split.val),
        "n_test": len(split.test),
        "models": {name: {k: rows[name][k] for k in _JSON_KEYS} for name in entry_names},
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(_json_safe(metrics), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    pred_df = pd.DataFrame(
        {"match_id": split.test["match_id"].to_numpy(dtype=np.int64), "y": y_test}
        | {name: preds[name].astype(np.float64) for name in entry_names}
    )
    pred_df.to_parquet(out_dir / "predictions.parquet", index=False)
    (out_dir / "table.md").write_text(_table_markdown(table, split.name), encoding="utf-8")
    plot_reliability(
        {m.name: reliability_curve(y_test, preds[m.name]) for m in models},  # raw models only
        out_dir / "reliability.png",
    )
    return BenchmarkResult(table=table, predictions=preds)
