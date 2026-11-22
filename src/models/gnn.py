"""Graph neural network over a hero synergy / counter graph (pure PyTorch, dense ops).

Nodes are the ``V`` heroes. Two signed adjacency matrices — EB-shrunk synergy and
counter scores from :mod:`src.stats.synergy`, computed on the TRAIN split only —
drive message passing; each match is scored from the masked mean of its picked
heroes' node vectors per team (docs/architecture.md §4.10). Training is shared
with the transformer via :class:`src.models.transformer.TorchDraftTrainer`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.config import SEED
from src.data.dataset import HeroVocab
from src.models.transformer import TorchDraftTrainer
from src.stats.synergy import counter_matrix, synergy_matrix
from src.utils.tracking import Tracker

__all__ = ["DraftGNN", "GNNDraftModel", "HeroGraph", "build_hero_graph"]


@dataclass
class HeroGraph:
    """Row-normalized signed hero adjacency matrices.

    Attributes
    ----------
    a_syn : torch.Tensor
        ``(V, V)`` float32 same-team synergy adjacency.
    a_cnt : torch.Tensor
        ``(V, V)`` float32 cross-team counter adjacency (``a`` counters ``b`` > 0).
    """

    a_syn: torch.Tensor
    a_cnt: torch.Tensor


def _normalize(a: np.ndarray) -> torch.Tensor:
    """L1-of-absolute row normalization; signs kept, zero rows stay zero, zero diagonal."""
    out = a / np.maximum(np.abs(a).sum(axis=1, keepdims=True), 1e-6)
    np.fill_diagonal(out, 0.0)
    return torch.as_tensor(out, dtype=torch.float32)


def build_hero_graph(train_df: pd.DataFrame, vocab: HeroVocab) -> HeroGraph:
    """Build the hero graph from the TRAIN split only (leakage rule, §3).

    Uses ``PairStats.value`` (EB-shrunk, zero where too few games) of
    :func:`synergy_matrix` and :func:`counter_matrix`, then
    ``A / max(|A|.sum(axis=1), 1e-6)`` row normalization with the diagonal forced to 0.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training matches with full 5+5 drafts.
    vocab : HeroVocab
        Hero vocabulary defining the node order.

    Returns
    -------
    HeroGraph
        Normalized ``(V, V)`` float32 adjacencies.
    """
    return HeroGraph(a_syn=_normalize(synergy_matrix(train_df, vocab).value),
                     a_cnt=_normalize(counter_matrix(train_df, vocab).value))


class _GNNLayer(nn.Module):
    """``H' = LN(H + drop(act(H W_self + A_syn H W_syn + A_cnt H W_cnt)))``."""

    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.w_self = nn.Linear(d_model, d_model)
        self.w_syn = nn.Linear(d_model, d_model, bias=False)
        self.w_cnt = nn.Linear(d_model, d_model, bias=False)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h: Tensor, a_syn: Tensor, a_cnt: Tensor) -> Tensor:
        m = self.act(self.w_self(h) + a_syn @ self.w_syn(h) + a_cnt @ self.w_cnt(h))
        return self.norm(h + self.drop(m))


class DraftGNN(nn.Module):
    """Message passing over heroes, then a per-match MLP on team mean vectors.

    Parameters
    ----------
    vocab_size : int
        Number of heroes (graph nodes) ``V``.
    d_model : int, optional
        Node embedding width, by default 64.
    n_layers : int, optional
        Message-passing layers, by default 2.
    dropout : float, optional
        Dropout rate, by default 0.1.
    use_rank : bool, optional
        Append the rank scalar to the MLP input, by default False.
    """

    def __init__(self, vocab_size: int, d_model: int = 64, n_layers: int = 2,
                 dropout: float = 0.1, use_rank: bool = False) -> None:
        super().__init__()
        self.use_rank = use_rank
        self.node = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(_GNNLayer(d_model, dropout) for _ in range(n_layers))
        self.head = nn.Sequential(nn.Linear(3 * d_model + int(use_rank), d_model), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(d_model, 1))

    def forward(self, teams: Tensor, a_syn: Tensor, a_cnt: Tensor,
                rank: Tensor | None = None) -> Tensor:
        """teams (B, 10) long (``-1`` = empty), adjacencies (V, V), rank (B, 1) -> (B,) logits."""
        h = self.node.weight
        for layer in self.layers:
            h = layer(h, a_syn, a_cnt)
        # Masked team means as (B, V) membership weights @ H: a matmul, whose backward is
        # deterministic on CPU (unlike the scatter-add backward of h[teams]).
        onehot = F.one_hot(teams.clamp(min=0), h.shape[0]).to(h.dtype)
        onehot = onehot * (teams >= 0).unsqueeze(-1).to(h.dtype)    # (B, 10, V), empties zeroed
        feats = []
        for side in (onehot[:, :5].sum(1), onehot[:, 5:].sum(1)):  # (B, V) pick counts
            weights = side / side.sum(1, keepdim=True).clamp(min=1.0)  # all-empty -> zeros
            feats.append(weights @ h)
        feats.append(feats[0] - feats[1])
        if self.use_rank:
            if rank is None:
                raise ValueError("use_rank=True requires a rank tensor")
            feats.append(rank)
        return self.head(torch.cat(feats, dim=1)).squeeze(-1)


class GNNDraftModel(TorchDraftTrainer):
    """:class:`DraftModel` wrapper around :class:`DraftGNN`.

    The graph is built inside :meth:`fit` from the train split only and stored
    in the checkpoint. Trainer options match
    :class:`src.models.transformer.TransformerDraftModel`.
    """

    name = "gnn"

    def __init__(self, *, d_model: int = 64, n_layers: int = 2, dropout: float = 0.1,
                 use_rank: bool = False, mirror_augment: bool = False,
                 pick_dropout: float = 0.0, lr: float = 3e-4, weight_decay: float = 1e-5,
                 batch_size: int = 1024, max_epochs: int = 30, patience: int = 4,
                 seed: int = SEED, device: str | None = None,
                 tracker: Tracker | None = None) -> None:
        super().__init__(
            {"d_model": d_model, "n_layers": n_layers, "dropout": dropout},
            use_rank=use_rank, mirror_augment=mirror_augment, pick_dropout=pick_dropout, lr=lr,
            weight_decay=weight_decay, batch_size=batch_size, max_epochs=max_epochs,
            patience=patience, seed=seed, device=device, tracker=tracker)
        self.graph: HeroGraph | None = None

    def _prepare(self, train: pd.DataFrame, vocab: HeroVocab) -> None:
        g = build_hero_graph(train, vocab)
        self._set_extra_state({"a_syn": g.a_syn, "a_cnt": g.a_cnt})

    def _extra_state(self) -> dict[str, Tensor]:
        assert self.graph is not None
        return {"a_syn": self.graph.a_syn, "a_cnt": self.graph.a_cnt}

    def _set_extra_state(self, state: dict[str, Tensor]) -> None:
        self.graph = HeroGraph(a_syn=state["a_syn"].to(self.device),
                               a_cnt=state["a_cnt"].to(self.device))

    def _build_network(self, vocab_size: int) -> DraftGNN:
        return DraftGNN(vocab_size, use_rank=self.use_rank, **self.arch)

    def _logits(self, teams: Tensor, rank: Tensor | None) -> Tensor:
        assert self.net is not None and self.graph is not None
        return self.net(teams, self.graph.a_syn, self.graph.a_cnt, rank)
