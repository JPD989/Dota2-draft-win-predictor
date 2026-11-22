"""Drift study: how fast does a draft model go stale?

Two modes (docs/architecture.md §4.14):

- ``--mode time``  single-patch proxy: train on the earliest 54% (val 6%), then
  evaluate on ``--n-windows`` consecutive later windows.
- ``--mode patch`` train on ``--train-patch``, compare within-patch test vs the
  whole ``--test-patch`` (needs >= 10k rows of each; requires a second
  collection run into its own ``data/raw/<era>`` directory).

Models are fit fresh by the study (canonical policy). Torch models are
allowed but slow on CPU.

Usage
-----
    python scripts/patch_drift.py --mode time
    python scripts/patch_drift.py --mode patch --train-patch 7.41 --test-patch 7.42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # fallback if not pip-installed

import pandas as pd  # noqa: E402

from src.config import RAW_DIR, RESULTS_DIR, SEED  # noqa: E402
from src.data.dataset import HeroVocab, load_matches  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code."""
    ap = argparse.ArgumentParser(description="Time- or patch-based drift study.")
    ap.add_argument("--mode", choices=["time", "patch"], default="time")
    ap.add_argument("--train-patch", type=str, default=None)
    ap.add_argument("--test-patch", type=str, default=None)
    ap.add_argument("--n-windows", type=int, default=5, help="time mode only")
    ap.add_argument("--models", nargs="+", default=["logreg", "xgboost"],
                    choices=["logreg", "xgboost", "transformer", "gnn"])
    ap.add_argument("--data", type=Path, default=RAW_DIR, help="raw shard directory")
    ap.add_argument("--extra-data", type=Path, nargs="*", default=[],
                    help="additional shard directories (e.g. a second patch) to append")
    ap.add_argument("--sample", type=int, default=None,
                    help="random subsample of ALL rows (smoke tests only)")
    ap.add_argument("--results", type=Path, default=RESULTS_DIR / "drift")
    args = ap.parse_args(argv)

    try:
        df = load_matches(args.data)
    except FileNotFoundError as e:
        print(f"No match data: {e}")
        return 0
    for extra in args.extra_data:
        try:
            df = pd.concat([df, load_matches(extra)], ignore_index=True)
        except FileNotFoundError as e:
            print(f"No match data: {e}")
            return 0
    if args.extra_data:
        df = df.drop_duplicates("match_id").sort_values(
            ["start_time", "match_id"], kind="stable").reset_index(drop=True)
    if args.sample is not None and args.sample < len(df):
        df = df.sample(n=args.sample, random_state=SEED).sort_values(
            ["start_time", "match_id"], kind="stable").reset_index(drop=True)

    if args.mode == "patch":
        patches = sorted(df["patch"].astype(str).unique())
        if args.train_patch is None or args.test_patch is None:
            print(f"--mode patch needs --train-patch and --test-patch (available: {patches})")
            return 0
        missing = [p for p in (args.train_patch, args.test_patch) if p not in patches]
        if missing:
            print(f"patch(es) {missing} not in the data (available: {patches}). Collect a "
                  "second era first: python src/data/collect_matches.py --out "
                  "data/raw/patch-<id>")
            return 0

    try:
        from src.eval.drift import patch_drift_study, time_drift_study
        from src.models import get_model_class
        classes = {name: get_model_class(name) for name in args.models}
    except (ImportError, AttributeError) as e:
        print(f"Drift/model modules unavailable ({e}).")
        return 0

    factories = {name: (lambda cls=cls: cls()) for name, cls in classes.items()}
    vocab = HeroVocab.from_matches(df)
    args.results.mkdir(parents=True, exist_ok=True)
    try:
        if args.mode == "time":
            out = time_drift_study(df, factories, vocab, n_windows=args.n_windows,
                                   out_dir=args.results)
        else:
            out = patch_drift_study(df, factories, vocab, train_patch=args.train_patch,
                                    test_patch=args.test_patch, out_dir=args.results)
    except ValueError as e:  # e.g. a patch with < 10k rows
        print(f"Cannot run the {args.mode} drift study: {e}")
        return 0

    print(out.to_string(float_format=lambda v: f"{v:.4f}"))
    print(f"wrote results to {args.results}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
