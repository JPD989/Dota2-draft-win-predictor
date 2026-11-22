"""Shared synthetic fixtures. No test reads ``data/raw``.

The generator plants a known signal so model and statistics tests have
something real to find:

* per-hero strengths (small),
* same-team synergy for ``SYNERGY_PAIRS``,
* ``COUNTER_PAIR[0]`` counters ``COUNTER_PAIR[1]`` when on opposite teams,
* a radiant base rate of ``BASE_RATE``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from src.data.dataset import HeroVocab

#: 20 heroes with deliberately non-contiguous ids (never index by raw id).
HERO_IDS: list[int] = [1, 2, 3, 5, 8, 11, 14, 19, 23, 26, 31, 35, 42, 48, 53, 67, 74, 88, 102, 129]
BASE_RATE: float = 0.53
#: Same-team pairs with planted positive synergy (log-odds bonus).
SYNERGY_PAIRS: list[tuple[int, int]] = [(1, 2), (5, 8), (42, 129)]
SYNERGY_LOGIT: float = 1.5
#: (a, b): a's team gets a log-odds bonus when b is on the other team.
COUNTER_PAIR: tuple[int, int] = (11, 14)
COUNTER_LOGIT: float = 1.5
T0: int = 1_750_000_000  # unix seconds of the first synthetic match


def make_synthetic_matches(n: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Generate ``n`` matches following the raw-data schema with planted signal.

    Rows are in chronological order (60 s apart) with unique ``match_id``.
    """
    rng = np.random.default_rng(seed)
    ids = np.asarray(HERO_IDS)
    strength = dict(zip(HERO_IDS, np.random.default_rng(1234).normal(0.0, 0.15, len(ids))))
    base = np.log(BASE_RATE / (1.0 - BASE_RATE))

    radiant, dire, wins = [], [], []
    for _ in range(n):
        picks = rng.choice(ids, size=10, replace=False)
        r, d = picks[:5], picks[5:]
        rs, ds = set(r.tolist()), set(d.tolist())
        z = base + sum(strength[h] for h in rs) - sum(strength[h] for h in ds)
        for a, b in SYNERGY_PAIRS:
            if a in rs and b in rs:
                z += SYNERGY_LOGIT
            if a in ds and b in ds:
                z -= SYNERGY_LOGIT
        a, b = COUNTER_PAIR
        if a in rs and b in ds:
            z += COUNTER_LOGIT
        if a in ds and b in rs:
            z -= COUNTER_LOGIT
        radiant.append(r.astype(np.int64))
        dire.append(d.astype(np.int64))
        wins.append(bool(rng.random() < 1.0 / (1.0 + np.exp(-z))))

    return pd.DataFrame(
        {
            "match_id": np.arange(1, n + 1, dtype=np.int64) + 8_000_000_000,
            "start_time": T0 + 60 * np.arange(n, dtype=np.int64),
            "duration": rng.integers(900, 4000, n).astype(np.int64),
            "avg_rank_tier": rng.choice([15, 25, 35, 45, 55, 65, 75], n).astype(np.int64),
            "radiant_win": np.asarray(wins, dtype=bool),
            "radiant_team": radiant,
            "dire_team": dire,
            "patch": "7.41",
        }
    )


@pytest.fixture(scope="session")
def tiny_vocab() -> HeroVocab:
    """20-hero vocabulary with non-contiguous ids."""
    return HeroVocab(HERO_IDS)


@pytest.fixture(scope="session")
def synthetic_matches() -> Callable[..., pd.DataFrame]:
    """Factory: ``synthetic_matches(n=2000, seed=0) -> DataFrame`` (memoized, copy returned)."""
    cache: dict[tuple[int, int], pd.DataFrame] = {}

    def _make(n: int = 2000, seed: int = 0) -> pd.DataFrame:
        key = (n, seed)
        if key not in cache:
            cache[key] = make_synthetic_matches(n, seed)
        return cache[key].copy()

    return _make
