"""Pick recommender: score every legal next pick with a fitted DraftModel.

Partial-draft caveat (docs/architecture.md §4.15, §6): only masking-trained
models (transformer / GNN with ``pick_dropout > 0``) see partial drafts
in-distribution. Logistic regression degrades gracefully (it sums the weights
of the heroes present). XGBoost on partial drafts is out-of-distribution and
its rankings are unreliable — the app labels non-masking models
"experimental" for mid-draft use.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd

from src.data.dataset import HeroVocab
from src.models.base import DraftModel

__all__ = ["DEFAULT_RANK_TIER", "legal_picks", "recommend"]

DEFAULT_RANK_TIER: int = 50
_TEAM_SIZE: int = 5


def legal_picks(
    radiant: Sequence[int], dire: Sequence[int], banned: Sequence[int], vocab: HeroVocab
) -> list[int]:
    """All vocabulary hero ids not already picked (either side) or banned.

    Parameters
    ----------
    radiant, dire : Sequence[int]
        Hero ids picked so far by each side.
    banned : Sequence[int]
        Banned hero ids (ids outside the vocab are ignored).
    vocab : HeroVocab
        The model's vocabulary — defines the candidate pool.

    Returns
    -------
    list[int]
        Legal hero ids, sorted ascending.
    """
    taken = {int(h) for h in (*radiant, *dire, *banned)}
    return [h for h in vocab.hero_ids if h not in taken]


def _validate_draft(radiant: Sequence[int], dire: Sequence[int], side: str) -> None:
    if side not in ("radiant", "dire"):
        raise ValueError(f"side must be 'radiant' or 'dire', got {side!r}")
    picked = [int(h) for h in (*radiant, *dire)]
    if len(set(picked)) != len(picked):
        raise ValueError(f"a hero is picked twice in radiant={list(radiant)} dire={list(dire)}")
    for name, team in (("radiant", radiant), ("dire", dire)):
        if len(team) > _TEAM_SIZE:
            raise ValueError(f"{name} has {len(team)} picks (max {_TEAM_SIZE})")
    if len(radiant if side == "radiant" else dire) >= _TEAM_SIZE:
        raise ValueError(f"{side} already has {_TEAM_SIZE} picks; nothing to recommend")


def recommend(
    model: DraftModel,
    vocab: HeroVocab,
    *,
    radiant: Sequence[int],
    dire: Sequence[int],
    side: Literal["radiant", "dire"],
    banned: Sequence[int] = (),
    avg_rank_tier: int | None = None,
    top_k: int | None = None,
) -> pd.DataFrame:
    """Rank every legal pick for ``side`` by that side's predicted win probability.

    For each legal hero ``h`` the draft with ``h`` added to ``side`` becomes
    one row of a batch frame (``radiant_team``, ``dire_team``, and always an
    ``avg_rank_tier`` column — default 50 when ``None``); the whole batch goes
    through ONE ``model.predict_proba`` call. See the module docstring for the
    partial-draft caveat: rankings from non-masking models (especially
    XGBoost) mid-draft are experimental.

    Parameters
    ----------
    model : DraftModel
        Fitted model.
    vocab : HeroVocab
        Candidate pool (normally ``model.vocab``).
    radiant, dire : Sequence[int]
        Current picks (0–5 each; the picking side has at most 4).
    side : {"radiant", "dire"}
        The side about to pick.
    banned : Sequence[int], optional
        Banned hero ids, excluded from candidates.
    avg_rank_tier : int | None, optional
        Lobby rank tier fed to rank-aware models; 50 if ``None``.
    top_k : int | None, optional
        Keep only the best ``top_k`` rows; all if ``None``.

    Returns
    -------
    pd.DataFrame
        Columns ``hero_id`` (int64) and ``win_prob`` (float64, the PICKING
        side's win probability), sorted by ``win_prob`` desc (ties by
        ``hero_id``), fresh RangeIndex.

    Raises
    ------
    ValueError
        On an invalid ``side``, duplicate picks, oversized teams, or a
        malformed prediction array.
    """
    _validate_draft(radiant, dire, side)
    candidates = legal_picks(radiant, dire, banned, vocab)
    if not candidates:
        return pd.DataFrame({"hero_id": pd.Series(dtype=np.int64), "win_prob": pd.Series(dtype=np.float64)})

    base_r = np.asarray(radiant, dtype=np.int64)
    base_d = np.asarray(dire, dtype=np.int64)
    add = [np.append(base_r if side == "radiant" else base_d, h) for h in candidates]
    batch = pd.DataFrame(
        {
            "radiant_team": add if side == "radiant" else [base_r.copy() for _ in candidates],
            "dire_team": add if side == "dire" else [base_d.copy() for _ in candidates],
            "avg_rank_tier": np.full(
                len(candidates), DEFAULT_RANK_TIER if avg_rank_tier is None else int(avg_rank_tier),
                dtype=np.int64,
            ),
        }
    )
    p_radiant = np.asarray(model.predict_proba(batch), dtype=np.float64)
    if p_radiant.shape != (len(candidates),):
        raise ValueError(f"predict_proba returned shape {p_radiant.shape}, expected ({len(candidates)},)")
    win_prob = p_radiant if side == "radiant" else 1.0 - p_radiant

    out = pd.DataFrame({"hero_id": np.asarray(candidates, dtype=np.int64), "win_prob": win_prob})
    out = out.sort_values(["win_prob", "hero_id"], ascending=[False, True], kind="stable")
    if top_k is not None:
        out = out.head(top_k)
    return out.reset_index(drop=True)
