"""Fit the logistic-regression and XGBoost baselines and save their checkpoints.

Uses the default chronological split (identical to the harness). ``--sample N``
subsamples the TRAIN block only, so val/test stay fixed and results remain
comparable (this is also the data-size ablation). Only validation metrics are
reported here; the test set is touched exclusively by ``run_benchmark.py``.

Usage
-----
    python scripts/train_baselines.py [--sample 20000] [--use-rank] [--mirror] [--tag T]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # fallback if not pip-installed

from src.config import MODELS_DIR, RAW_DIR, RESULTS_DIR, SEED  # noqa: E402
from src.data.dataset import HeroVocab, chronological_split, load_matches  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code."""
    ap = argparse.ArgumentParser(description="Fit logreg + XGBoost baselines.")
    ap.add_argument("--data", type=Path, default=RAW_DIR, help="raw shard directory")
    ap.add_argument("--sample", type=int, default=None,
                    help="subsample N rows of the train block (val/test unchanged)")
    ap.add_argument("--models", nargs="+", default=["logreg", "xgboost"],
                    choices=["logreg", "xgboost"])
    ap.add_argument("--use-rank", action="store_true", help="add avg_rank_tier/80 feature")
    ap.add_argument("--mirror", action="store_true", help="mirror augmentation (ablation)")
    ap.add_argument("--C", type=float, default=1.0, help="logreg inverse regularization")
    ap.add_argument("--n-estimators", type=int, default=400)
    ap.add_argument("--max-depth", type=int, default=8)
    ap.add_argument("--learning-rate", type=float, default=0.1)
    ap.add_argument("--tag", type=str, default=None, help="checkpoint suffix, e.g. 'rank'")
    ap.add_argument("--out", type=Path, default=MODELS_DIR, help="checkpoint directory")
    ap.add_argument("--results", type=Path, default=RESULTS_DIR / "baselines")
    args = ap.parse_args(argv)

    try:
        df = load_matches(args.data)
    except FileNotFoundError as e:
        print(f"No match data: {e}")
        return 0

    try:
        from src.eval.metrics import compute_metrics
        from src.models import checkpoint_path, get_model_class
        classes = {name: get_model_class(name) for name in args.models}
    except (ImportError, AttributeError) as e:
        print(f"Model/eval modules unavailable ({e}); nothing to train.")
        return 0

    vocab = HeroVocab.from_matches(df)
    split = chronological_split(df)
    train = split.train
    if args.sample is not None and args.sample < len(train):
        train = train.sample(n=args.sample, random_state=SEED).sort_values(
            ["start_time", "match_id"], kind="stable").reset_index(drop=True)
    print(f"split {split.name}: train={len(train):,} val={len(split.val):,} "
          f"(test untouched: {len(split.test):,})")

    kwargs = {
        "logreg": dict(C=args.C, use_rank=args.use_rank, mirror_augment=args.mirror,
                       seed=SEED),
        "xgboost": dict(n_estimators=args.n_estimators, max_depth=args.max_depth,
                        learning_rate=args.learning_rate, use_rank=args.use_rank,
                        mirror_augment=args.mirror, seed=SEED),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    args.results.mkdir(parents=True, exist_ok=True)
    report: dict[str, dict] = {}
    for name, cls in classes.items():
        model = cls(**kwargs[name])
        t0 = time.perf_counter()
        model.fit(train, split.val, vocab)
        fit_s = time.perf_counter() - t0
        path = checkpoint_path(name, args.out, args.tag)
        model.save(path)
        y_val = split.val["radiant_win"].to_numpy(dtype=float)
        m = compute_metrics(y_val, model.predict_proba(split.val))
        m["fit_seconds"] = fit_s
        report[name] = m
        print(f"{name:<8} val auc={m['auc']:.4f} log_loss={m['log_loss']:.4f} "
              f"brier={m['brier']:.4f} ({fit_s:.1f}s) -> {path}")

    stem = "val_metrics" if args.tag is None else f"val_metrics_{args.tag}"
    out = args.results / f"{stem}.json"
    out.write_text(json.dumps({"split": split.name, "n_train": len(train),
                               "n_val": len(split.val), "args": {k: str(v) for k, v in
                                                                 vars(args).items()},
                               "models": report}, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
