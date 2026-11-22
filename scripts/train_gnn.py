"""Train the hero-graph GNN (synergy + counter adjacency) and save its checkpoint.

Same split policy as ``train_baselines.py``: default chronological split,
``--sample N`` subsamples the TRAIN block only, early stopping on the
un-augmented val set, and only val metrics are printed (test belongs to the
harness). The synergy/counter adjacency is built inside ``fit`` from the train
block only. Use ``--pick-dropout 0.15 --tag rec`` for a recommender checkpoint.

Usage
-----
    python scripts/train_gnn.py [--epochs 30] [--sample N] [--pick-dropout P]
                                [--mirror] [--use-rank] [--tag T] [--wandb]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # fallback if not pip-installed

from src.config import MODELS_DIR, RAW_DIR, SEED  # noqa: E402
from src.data.dataset import HeroVocab, chronological_split, load_matches  # noqa: E402
from src.utils.tracking import init_tracker  # noqa: E402

MODEL_NAME = "gnn"


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code."""
    ap = argparse.ArgumentParser(
        description="Train the draft GNN (hero graph built from the train split only).")
    ap.add_argument("--data", type=Path, default=RAW_DIR, help="raw shard directory")
    ap.add_argument("--sample", type=int, default=None,
                    help="subsample N rows of the train block (val/test unchanged)")
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=30, help="max epochs (early stopping)")
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--pick-dropout", type=float, default=0.0,
                    help="train-time prob. of masking each hero token")
    ap.add_argument("--mirror", action="store_true", help="mirror augmentation (ablation)")
    ap.add_argument("--use-rank", action="store_true", help="add avg_rank_tier/80 feature")
    ap.add_argument("--device", type=str, default=None, help="cpu | cuda (default: auto)")
    ap.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False,
                    help="log to Weights & Biases (default: off)")
    ap.add_argument("--tag", type=str, default=None, help="checkpoint suffix, e.g. 'rec'")
    ap.add_argument("--out", type=Path, default=MODELS_DIR, help="checkpoint directory")
    args = ap.parse_args(argv)

    try:
        df = load_matches(args.data)
    except FileNotFoundError as e:
        print(f"No match data: {e}")
        return 0
    try:
        from src.eval.metrics import compute_metrics
        from src.models import checkpoint_path, get_model_class
        cls = get_model_class(MODEL_NAME)
    except (ImportError, AttributeError) as e:
        print(f"{MODEL_NAME} unavailable ({e}). Install torch: pip install -e '.[torch]'.")
        return 0

    vocab = HeroVocab.from_matches(df)
    split = chronological_split(df)
    train = split.train
    if args.sample is not None and args.sample < len(train):
        train = train.sample(n=args.sample, random_state=SEED).sort_values(
            ["start_time", "match_id"], kind="stable").reset_index(drop=True)
    print(f"split {split.name}: train={len(train):,} val={len(split.val):,}")

    config = dict(d_model=args.d_model, n_layers=args.n_layers,
                  dropout=args.dropout, use_rank=args.use_rank,
                  mirror_augment=args.mirror, pick_dropout=args.pick_dropout, lr=args.lr,
                  weight_decay=args.weight_decay, batch_size=args.batch_size,
                  max_epochs=args.epochs, patience=args.patience, seed=SEED)
    tracker = init_tracker("dota2-draft-predictor",
                           {**config, "model": MODEL_NAME, "tag": args.tag,
                            "n_train": len(train)}, enabled=args.wandb)
    model = cls(**config, device=args.device, tracker=tracker)
    model.fit(train, split.val, vocab)

    path = checkpoint_path(MODEL_NAME, args.out, args.tag)
    args.out.mkdir(parents=True, exist_ok=True)
    model.save(path)
    m = compute_metrics(split.val["radiant_win"].to_numpy(dtype=float),
                        model.predict_proba(split.val))
    tracker.log({f"final_val/{k}": v for k, v in m.items()})
    tracker.finish()
    print(f"{MODEL_NAME} val auc={m['auc']:.4f} log_loss={m['log_loss']:.4f} -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
