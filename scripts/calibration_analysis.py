"""Calibration analysis: raw vs Platt vs isotonic recalibration per saved model.

Rebuilds the deterministic default ``chronological_split(load_matches())``
(byte-identical to the harness's split), loads every model that has a
checkpoint, fits Platt and isotonic calibrators on VAL predictions and
evaluates all variants on TEST. Writes to ``--results``:

- ``reliability.png``              raw reliability curves (one per model)
- ``reliability_recalibrated.png`` isotonic- and Platt-recalibrated curves
- ``calibration_table.md``         ECE / ECE-quantile / log-loss / Brier / AUC,
                                   before and after recalibration

Usage
-----
    python scripts/calibration_analysis.py [--model-dir models] [--tag T]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # fallback if not pip-installed

import pandas as pd  # noqa: E402

from src.config import MODELS_DIR, RAW_DIR, RESULTS_DIR  # noqa: E402
from src.data.dataset import chronological_split, load_matches  # noqa: E402

ALL_MODELS = ("logreg", "xgboost", "transformer", "gnn")


def _markdown_table(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub markdown table (no tabulate dependency)."""
    def fmt(v: object) -> str:
        return f"{v:.4f}" if isinstance(v, float) else str(v)
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |"
              for row in df.itertuples(index=False)]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code."""
    ap = argparse.ArgumentParser(description="Before/after recalibration analysis.")
    ap.add_argument("--models", nargs="+", default=list(ALL_MODELS), choices=ALL_MODELS)
    ap.add_argument("--data", type=Path, default=RAW_DIR, help="raw shard directory")
    ap.add_argument("--model-dir", type=Path, default=MODELS_DIR)
    ap.add_argument("--tag", type=str, default=None, help="checkpoint tag")
    ap.add_argument("--n-bins", type=int, default=15)
    ap.add_argument("--results", type=Path, default=RESULTS_DIR / "calibration")
    args = ap.parse_args(argv)

    try:
        from src.eval.calibration import (fit_isotonic, fit_platt, plot_reliability,
                                          reliability_curve)
        from src.eval.metrics import compute_metrics
        from src.models import checkpoint_path, get_model_class
    except ImportError as e:
        print(f"Eval/model modules unavailable ({e}).")
        return 0

    models = {}
    for name in args.models:
        path = checkpoint_path(name, args.model_dir, args.tag)
        if not path.exists():
            print(f"skip {name}: no checkpoint at {path}")
            continue
        try:
            models[name] = get_model_class(name).load(path)
        except Exception as e:  # missing torch, incompatible checkpoint, ...
            print(f"skip {name}: could not load {path} ({type(e).__name__}: {e})")
    if not models:
        print("No checkpoints found. Train first, e.g. `make train-baselines`.")
        return 0

    try:
        split = chronological_split(load_matches(args.data))
    except FileNotFoundError as e:
        print(f"No match data: {e}")
        return 0
    y_val = split.val["radiant_win"].to_numpy(dtype=float)
    y_test = split.test["radiant_win"].to_numpy(dtype=float)

    rows, raw_curves, recal_curves = [], {}, {}
    for name, model in models.items():
        p_val = model.predict_proba(split.val)
        p_test = model.predict_proba(split.test)
        variants = {
            "raw": p_test,
            "platt": fit_platt(y_val, p_val)(p_test),
            "isotonic": fit_isotonic(y_val, p_val)(p_test),
        }
        for method, p in variants.items():
            m = compute_metrics(y_test, p)
            rows.append({"model": name, "method": method, **m})
            curve = reliability_curve(y_test, p, n_bins=args.n_bins)
            if method == "raw":
                raw_curves[name] = curve
            else:
                recal_curves[f"{name}+{method}"] = curve

    args.results.mkdir(parents=True, exist_ok=True)
    plot_reliability(raw_curves, args.results / "reliability.png")
    plot_reliability(recal_curves, args.results / "reliability_recalibrated.png")
    table = pd.DataFrame(rows)
    cols = ["model", "method", "ece", "ece_q", "log_loss", "brier", "auc", "n"]
    table = table[[c for c in cols if c in table.columns]]
    md = (f"# Calibration ({split.name}; calibrators fit on val, evaluated on test)\n\n"
          + _markdown_table(table) + "\n")
    (args.results / "calibration_table.md").write_text(md)
    print(md)
    print(f"wrote reliability.png, reliability_recalibrated.png, calibration_table.md "
          f"-> {args.results}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
