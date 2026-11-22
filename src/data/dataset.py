"""Match loading, hero vocabulary, and deterministic train/val/test splits.

Conventions (docs/architecture.md §2–§3):

- Hero ids are sparse in [1, ~150]; arrays are never indexed by raw hero id —
  all encodings go through :class:`HeroVocab`.
- Splits are chronological, never shuffled, and byte-deterministic: rows are
  stable-sorted by ``start_time`` with ``match_id`` as tiebreaker, so the same
  DataFrame always produces the identical split.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import RAW_DIR

_TIME_SORT_COLS = ["start_time", "match_id"]


class HeroVocab:
    """Bidirectional hero_id <-> contiguous index [0, V) mapping, sorted by hero_id.

    Parameters
    ----------
    hero_ids : Sequence[int]
        Hero ids to include. Duplicates are dropped; the vocabulary is always
        stored sorted ascending so index assignment is deterministic.
    """

    def __init__(self, hero_ids: Sequence[int]) -> None:
        self._hero_ids: list[int] = sorted({int(h) for h in hero_ids})
        self._id_to_idx: dict[int, int] = {h: i for i, h in enumerate(self._hero_ids)}

    @property
    def size(self) -> int:
        """Number of heroes V in the vocabulary."""
        return len(self._hero_ids)

    @property
    def hero_ids(self) -> list[int]:
        """Hero ids sorted ascending (a copy; mutating it does not affect the vocab)."""
        return list(self._hero_ids)

    def idx(self, hero_id: int) -> int:
        """Return the contiguous index of ``hero_id``.

        Raises
        ------
        KeyError
            If ``hero_id`` is not in the vocabulary.
        """
        try:
            return self._id_to_idx[int(hero_id)]
        except KeyError:
            raise KeyError(
                f"Unknown hero id {hero_id} (vocabulary has {self.size} heroes)."
            ) from None

    def hero_id(self, idx: int) -> int:
        """Return the hero id at vocabulary index ``idx``.

        Raises
        ------
        IndexError
            If ``idx`` is outside [0, V).
        """
        if not 0 <= idx < self.size:
            raise IndexError(f"Vocab index {idx} out of range [0, {self.size}).")
        return self._hero_ids[idx]

    def save(self, path: Path) -> None:
        """Write the vocabulary to ``path`` as JSON ``{"hero_ids": [...]}``."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump({"hero_ids": self._hero_ids}, f)

    @classmethod
    def load(cls, path: Path) -> "HeroVocab":
        """Load a vocabulary previously written by :meth:`save`."""
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(data["hero_ids"])

    @classmethod
    def from_matches(cls, df: pd.DataFrame) -> "HeroVocab":
        """Build the vocabulary from the union of both team columns of ``df``.

        Parameters
        ----------
        df : pd.DataFrame
            Matches frame with ``radiant_team`` and ``dire_team`` list columns.
        """
        ids: set[int] = set()
        for col in ("radiant_team", "dire_team"):
            for team in df[col]:
                ids.update(int(h) for h in team)
        return cls(sorted(ids))


def load_matches(raw_dir: Path = RAW_DIR, *, sort_by_time: bool = True) -> pd.DataFrame:
    """Load all match shards under ``raw_dir`` into a single deduplicated frame.

    Globs ``raw_dir/**/matches_*.parquet`` recursively (era subdirectories and
    legacy flat shards coexist), concatenates in sorted shard-path order, drops
    duplicate ``match_id`` rows keeping the first occurrence, and optionally
    sorts chronologically (by ``start_time``, then ``match_id``).

    Parameters
    ----------
    raw_dir : Path, optional
        Directory to search recursively. Defaults to :data:`src.config.RAW_DIR`.
    sort_by_time : bool, optional
        If True (default), return rows in chronological order.

    Returns
    -------
    pd.DataFrame
        Matches with a fresh RangeIndex.

    Raises
    ------
    FileNotFoundError
        If no shard files are found under ``raw_dir``.
    """
    shards = sorted(raw_dir.glob("**/matches_*.parquet"))
    if not shards:
        raise FileNotFoundError(
            f"No matches_*.parquet shards found under {raw_dir}. "
            "Run the collector first, e.g. "
            "`python src/data/collect_matches.py --out data/raw` (or `make data`)."
        )
    df = pd.concat([pd.read_parquet(p) for p in shards], ignore_index=True)
    df = df.drop_duplicates(subset="match_id", keep="first")
    if sort_by_time:
        df = df.sort_values(_TIME_SORT_COLS, kind="stable")
    return df.reset_index(drop=True)


@dataclass(frozen=True)
class Split:
    """A named train/val/test partition of the matches frame."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    name: str  # e.g. "chrono-80-10-10" or "patch-7.41->7.42"


def chronological_split(
    df: pd.DataFrame, val_frac: float = 0.1, test_frac: float = 0.1
) -> Split:
    """Split ``df`` chronologically into contiguous [train | val | test] blocks.

    Rows are stable-sorted by ``start_time`` with ``match_id`` as tiebreaker,
    then cut so that val gets ``int(n * val_frac)`` rows, test gets
    ``int(n * test_frac)`` rows, and train gets the rest. Deterministic: the
    same input frame always yields a byte-identical split (all consumers —
    harness, calibration_analysis, build_artifacts — rely on this).

    Parameters
    ----------
    df : pd.DataFrame
        Matches frame with ``start_time`` and ``match_id`` columns.
    val_frac, test_frac : float, optional
        Fractions of rows for the validation and test blocks (defaults 0.1).

    Returns
    -------
    Split
        Named e.g. ``"chrono-80-10-10"`` (percentages derived from the fracs).

    Raises
    ------
    ValueError
        If the fractions are negative or leave no room for a train block.
    """
    if val_frac < 0 or test_frac < 0 or val_frac + test_frac >= 1:
        raise ValueError(
            f"Invalid fractions val_frac={val_frac}, test_frac={test_frac}: "
            "both must be >= 0 and sum to < 1."
        )
    ordered = df.sort_values(_TIME_SORT_COLS, kind="stable").reset_index(drop=True)
    n = len(ordered)
    n_val = int(n * val_frac)
    n_test = int(n * test_frac)
    n_train = n - n_val - n_test
    name = (
        f"chrono-{round((1 - val_frac - test_frac) * 100)}"
        f"-{round(val_frac * 100)}-{round(test_frac * 100)}"
    )
    return Split(
        train=ordered.iloc[:n_train].reset_index(drop=True),
        val=ordered.iloc[n_train : n_train + n_val].reset_index(drop=True),
        test=ordered.iloc[n_train + n_val :].reset_index(drop=True),
        name=name,
    )


def patch_split(
    df: pd.DataFrame,
    train_patches: Sequence[str],
    test_patch: str,
    val_frac: float = 0.1,
) -> Split:
    """Train/val from ``train_patches`` rows, test = all ``test_patch`` rows.

    Train and val are the chronological split (no test carve-out) of the rows
    whose ``patch`` is in ``train_patches``; the test block is every row of
    ``test_patch``, in chronological order. This is the benchmark path —
    ``src.eval.drift`` builds its own splits (§4.14).

    Parameters
    ----------
    df : pd.DataFrame
        Matches frame with a ``patch`` string column.
    train_patches : Sequence[str]
        Patches used for training and validation.
    test_patch : str
        Patch held out entirely for testing.
    val_frac : float, optional
        Validation fraction of the train-patches rows (default 0.1).

    Returns
    -------
    Split
        Named e.g. ``"patch-7.41->7.42"``.

    Raises
    ------
    ValueError
        If either patch selection matches no rows.
    """
    train_rows = df[df["patch"].isin(train_patches)]
    test_rows = df[df["patch"] == test_patch]
    if train_rows.empty:
        raise ValueError(f"No rows found for train_patches={list(train_patches)}.")
    if test_rows.empty:
        raise ValueError(f"No rows found for test_patch={test_patch!r}.")
    inner = chronological_split(train_rows, val_frac=val_frac, test_frac=0.0)
    return Split(
        train=inner.train,
        val=inner.val,
        test=test_rows.sort_values(_TIME_SORT_COLS, kind="stable").reset_index(drop=True),
        name=f"patch-{'+'.join(train_patches)}->{test_patch}",
    )
