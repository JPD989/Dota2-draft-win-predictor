"""Run the shared benchmark harness on the default chronological split.

Policy (docs/architecture.md §4.17):

- Baselines (logreg, xgboost) are always fit fresh in-process.
- Torch models (transformer, gnn) are load-only: their checkpoint comes from
  ``--checkpoint NAME=PATH`` or ``checkpoint_path(name, --model-dir, --tag)``.
  A missing checkpoint (or missing torch) prints a skip message; it never
  retrains and never stack-traces.

Usage
-----
    python scripts/run_benchmark.py --models logreg xgboost transformer gnn
    python scripts/run_benchmark.py --models logreg xgboost --sample 20000 --results /tmp/b
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # fallback if not pip-installed

from src.config import MODELS_DIR, RAW_DIR, RESULTS_DIR, SEED  # noqa: E402
from src.data.dataset import HeroVocab, Split, chronological_split, load_matches  # noqa: E402

BASELINES = ("logreg", "xgboost")
ALL_MODELS = ("logreg", "xgboost", "transformer", "gnn")


def _parse_checkpoints(items: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for item in items:
        name, sep, path = item.partition("=")
        if not sep or name not in ALL_MODELS:
            raise SystemExit(f"--checkpoint expects NAME=PATH with NAME in {ALL_MODELS}, "
                             f"got {item!r}")
        out[name] = Path(path)
    return out


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code."""
    ap = argparse.ArgumentParser(description="Benchmark draft models on one fixed split.")
    ap.add_argument("--models", nargs="+", default=list(ALL_MODELS), choices=ALL_MODELS)
    ap.add_argument("--data", type=Path, default=RAW_DIR, help="raw shard directory")
    ap.add_argument("--model-dir", type=Path, default=MODELS_DIR,
                    help="where torch checkpoints are looked up")
    ap.add_argument("--tag", type=str, default=None,
                    help="checkpoint tag for torch models; results go to RESULTS/<tag>")
    ap.add_argument("--checkpoint", nargs="*", default=[], metavar="NAME=PATH",
                    help="explicit checkpoint paths, overriding --model-dir/--tag")
    ap.add_argument("--sample", type=int, default=None,
                    help="subsample N train rows for the fresh baselines (val/test unchanged)")
    ap.add_argument("--use-rank", action="store_true", help="baselines use the rank feature")
    ap.add_argument("--mirror", action="store_true", help="baselines use mirror augmentation")
    ap.add_argument("--recalibrate", action="store_true",
                    help="add '+isotonic' rows (isotonic fit on val)")
    ap.add_argument("--results", type=Path, default=RESULTS_DIR / "benchmark")
    args = ap.parse_args(argv)
    explicit = _parse_checkpoints(args.checkpoint)

    try:
        df = load_matches(args.data)
    except FileNotFoundError as e:
        print(f"No match data: {e}")
        return 0
    try:
        from src.eval.harness import run_benchmark
        from src.models import checkpoint_path, get_model_class
    except ImportError as e:
        print(f"Harness/model modules unavailable ({e}); cannot benchmark yet.")
        return 0

    vocab = HeroVocab.from_matches(df)
    split = chronological_split(df)
    if args.sample is not None and args.sample < len(split.train):
        train = split.train.sample(n=args.sample, random_state=SEED).sort_values(
            ["start_time", "match_id"], kind="stable").reset_index(drop=True)
        split = Split(train=train, val=split.val, test=split.test,
                      name=f"{split.name}-train{args.sample}")

    models = []
    for name in args.models:
        try:
            cls = get_model_class(name)
        except (ImportError, AttributeError) as e:
            print(f"skip {name}: implementation unavailable ({e})")
            continue
        if name in BASELINES:
            models.append(cls(use_rank=args.use_rank, mirror_augment=args.mirror))
            continue
        path = explicit.get(name, checkpoint_path(name, args.model_dir, args.tag))
        if not path.exists():
            print(f"skip {name}: no checkpoint at {path} "
                  f"(train it with scripts/train_{name}.py)")
            continue
        try:
            models.append(cls.load(path))
            print(f"loaded {name} from {path}")
        except Exception as e:  # corrupt/incompatible checkpoint: skip, don't crash
            print(f"skip {name}: could not load {path} ({type(e).__name__}: {e})")

    if not models:
        print("No models to benchmark.")
        return 0

    out_dir = args.results / args.tag if args.tag else args.results
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"split {split.name}: train={len(split.train):,} val={len(split.val):,} "
          f"test={len(split.test):,}")
    result = run_benchmark(models, split, vocab, out_dir, recalibrate=args.recalibrate,
                           seed=SEED)
    cols = [c for c in ("auc", "auc_lo", "auc_hi", "delta_auc_vs_logreg", "log_loss",
                        "brier", "ece") if c in result.table.columns]
    print(result.table[cols].to_string(float_format=lambda v: f"{v:.4f}"))
    print(f"wrote metrics.json, predictions.parquet, table.md, reliability.png -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
