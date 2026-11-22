"""Build the small, committable stats artifacts the Streamlit app runs on.

Loads all matches, takes the TRAIN block of the default chronological split
only (leakage rule, docs/architecture.md §3), and writes to ``--out``:

- ``hero_vocab.json``          hero vocabulary (all heroes seen in the data)
- ``synergy.npz``              PairStats (EB-shrunk same-team synergy)
- ``counter.npz``              PairStats (EB-shrunk cross-team counter)
- ``hero_win_rates.parquet``   per-hero win rates with Wilson 95% CIs
- ``meta.json``                train rows, radiant_wr_train, split name, dates

Usage
-----
    python scripts/build_artifacts.py [--data data/raw] [--out data/artifacts]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # fallback if not pip-installed

import pandas as pd  # noqa: E402

from src.config import ARTIFACTS_DIR, MIN_PAIR_GAMES, RAW_DIR  # noqa: E402
from src.data.dataset import HeroVocab, chronological_split, load_matches  # noqa: E402
from src.stats.synergy import (  # noqa: E402
    counter_matrix,
    hero_win_rates,
    radiant_side_advantage,
    synergy_matrix,
)


def _iso(ts: int) -> str:
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat()


def build_artifacts(df: pd.DataFrame, out_dir: Path, *, val_frac: float = 0.1,
                    test_frac: float = 0.1) -> dict[str, Any]:
    """Compute train-split statistics from ``df`` and write them to ``out_dir``.

    Parameters
    ----------
    df : pd.DataFrame
        All matches (raw schema).
    out_dir : Path
        Destination directory (created if needed).
    val_frac, test_frac : float
        Fractions passed to :func:`chronological_split`; must match the
        harness defaults so the artifacts never see val/test rows.

    Returns
    -------
    dict
        The ``meta.json`` contents.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    # Vocabulary = every hero id in the data (ids only, no outcome information),
    # so the app can offer every hero; statistics below use the train block only.
    vocab = HeroVocab.from_matches(df)
    split = chronological_split(df, val_frac=val_frac, test_frac=test_frac)
    train = split.train

    vocab.save(out_dir / "hero_vocab.json")
    synergy_matrix(train, vocab).save(out_dir / "synergy.npz")
    counter_matrix(train, vocab).save(out_dir / "counter.npz")
    hero_win_rates(train, vocab).to_parquet(out_dir / "hero_win_rates.parquet", index=False)

    wr, lo, hi = radiant_side_advantage(train)
    meta: dict[str, Any] = {
        "split": split.name,
        "val_frac": val_frac,
        "test_frac": test_frac,
        "n_train": int(len(train)),
        "n_total": int(len(df)),
        "radiant_wr_train": wr,
        "radiant_wr_train_ci": [lo, hi],
        "n_heroes": vocab.size,
        "min_pair_games": MIN_PAIR_GAMES,
        "patches_train": sorted(train["patch"].astype(str).unique().tolist()),
        "train_start": _iso(train["start_time"].min()),
        "train_end": _iso(train["start_time"].max()),
        "built_at": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code."""
    ap = argparse.ArgumentParser(description="Build train-split stats artifacts for the app.")
    ap.add_argument("--data", type=Path, default=RAW_DIR, help="raw shard directory")
    ap.add_argument("--out", type=Path, default=ARTIFACTS_DIR, help="artifact directory")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    args = ap.parse_args(argv)

    try:
        df = load_matches(args.data)
    except FileNotFoundError as e:
        print(f"No match data: {e}")
        return 0
    meta = build_artifacts(df, args.out, val_frac=args.val_frac, test_frac=args.test_frac)
    print(f"built artifacts from {meta['n_train']:,} train rows ({meta['split']}) -> {args.out}")
    for p in sorted(args.out.iterdir()):
        print(f"  {p.name:<24} {p.stat().st_size / 1024:8.1f} KB")
    if not (args.out / "heroes.json").exists():
        print("note: heroes.json missing — run `python scripts/fetch_hero_constants.py`.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
