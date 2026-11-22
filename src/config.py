"""Project-wide path constants and fixed hyperparameters.

This module only *defines* paths — it never creates directories. Any function
or script that writes output is responsible for calling
``out_dir.mkdir(parents=True, exist_ok=True)`` first (see docs/architecture.md §3).
"""

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
"""Repository root, resolved from this file's location."""

RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
"""Raw parquet shards (never committed); era subdirectories allowed."""

ARTIFACTS_DIR: Path = PROJECT_ROOT / "data" / "artifacts"
"""Small committable artifacts (hero names, vocab, synergy/counter matrices)."""

MODELS_DIR: Path = PROJECT_ROOT / "models"
"""Trained model checkpoints (gitignored, except models/release/)."""

RESULTS_DIR: Path = PROJECT_ROOT / "results"
"""Harness outputs: metrics, predictions, tables, plots (gitignored)."""

FIGURES_DIR: Path = PROJECT_ROOT / "docs" / "figures"
"""Committable headline figures, copied from results/ by `make figures`."""

SEED: int = 42
"""Global random seed; every stochastic function defaults to it."""

RANK_NORM: float = 80.0
"""Fixed divisor for the rank feature: avg_rank_tier / RANK_NORM (no fitted scaler)."""

MIN_PAIR_GAMES: int = 100
"""Hard floor for synergy/counter cells (total games across both orientations)."""
