"""Hero-embedding Transformer over the 10 picks, plus the shared torch trainer.

Token layout (docs/architecture.md §4.9): embedding table of size ``V + 2``
(heroes ``0..V-1``, ``V`` = MASK for empty slots, ``V + 1`` = CLS); the input is
``[CLS, r1..r5, d1..d5]`` plus a team embedding (CLS=0, radiant=1, dire=2) and no
positional encoding, since drafts are sets.

:class:`TorchDraftTrainer` holds the training loop shared with the GNN
(``src.models.gnn``): mirror augmentation + side delta, pick dropout, early
stopping on un-augmented val log-loss with best weights restored, seeding,
tracker logging and self-contained ``.pt`` checkpoints.
"""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.config import SEED
from src.data.dataset import HeroVocab
from src.features.encoding import labels, mirror, teams_to_matrix
from src.models.base import DraftModel, compute_mirror_delta, validate_predict_frame
from src.models.baselines import rank_input
from src.utils.tracking import Tracker

__all__ = ["DraftTransformer", "TorchDraftTrainer", "TransformerDraftModel"]

_PREDICT_BATCH: int = 8192


class TorchDraftTrainer(DraftModel):
    """Base class implementing fit / predict / save / load for torch draft models.

    Subclasses provide :meth:`_build_network` and :meth:`_logits`; models whose
    inputs depend on the train split (the GNN graph) also override
    :meth:`_prepare`, :meth:`_extra_state` and :meth:`_set_extra_state`.

    Parameters
    ----------
    arch : dict
        Architecture hyperparameters, saved in the checkpoint config.
    use_rank, mirror_augment, pick_dropout, lr, weight_decay, batch_size, max_epochs,
    patience, seed, device, tracker
        Trainer options, see :class:`TransformerDraftModel`.
    """

    def __init__(self, arch: dict[str, Any], *, use_rank: bool, mirror_augment: bool,
                 pick_dropout: float, lr: float, weight_decay: float, batch_size: int,
                 max_epochs: int, patience: int, seed: int, device: str | None,
                 tracker: Tracker | None) -> None:
        if not 0.0 <= pick_dropout < 1.0:
            raise ValueError(f"pick_dropout must be in [0, 1), got {pick_dropout}")
        self.arch = dict(arch)
        self.use_rank = use_rank
        self.mirror_augment = mirror_augment
        self.pick_dropout = pick_dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.seed = seed
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tracker = tracker or Tracker()
        self.vocab: HeroVocab | None = None
        self.delta: float = 0.0
        self.net: nn.Module | None = None
        self.history: list[dict[str, float]] = []

    @property
    def config(self) -> dict[str, Any]:
        """Constructor kwargs needed to rebuild the model (no device / tracker)."""
        return {**self.arch, "use_rank": self.use_rank, "mirror_augment": self.mirror_augment,
                "pick_dropout": self.pick_dropout, "lr": self.lr,
                "weight_decay": self.weight_decay, "batch_size": self.batch_size,
                "max_epochs": self.max_epochs, "patience": self.patience, "seed": self.seed}

    # --- subclass hooks --------------------------------------------------------------
    @abc.abstractmethod
    def _build_network(self, vocab_size: int) -> nn.Module:
        """Instantiate the (untrained) network."""

    @abc.abstractmethod
    def _logits(self, teams: Tensor, rank: Tensor | None) -> Tensor:
        """``(B, 10)`` long vocab indices (``-1`` = empty) -> ``(B,)`` logits, no delta."""

    def _prepare(self, train: pd.DataFrame, vocab: HeroVocab) -> None:
        """Compute train-split-derived inputs before training (default: nothing)."""

    def _extra_state(self) -> dict[str, Tensor]:
        """Extra tensors to store in the checkpoint (default: none)."""
        return {}

    def _set_extra_state(self, state: dict[str, Tensor]) -> None:
        """Restore what :meth:`_extra_state` saved (default: nothing)."""

    # --- training --------------------------------------------------------------------
    def _rank(self, df: pd.DataFrame) -> np.ndarray | None:
        return rank_input(df) if self.use_rank else None

    def _batch_logits(self, teams: np.ndarray, rank: np.ndarray | None) -> Tensor:
        t = torch.as_tensor(teams, dtype=torch.long, device=self.device)
        r = None if rank is None else torch.as_tensor(rank, dtype=torch.float32, device=self.device)
        return self._logits(t, r)

    @torch.no_grad()
    def _predict_logits(self, teams: np.ndarray, rank: np.ndarray | None) -> np.ndarray:
        """Eval-mode logits including the mirror delta, float64 ``(N,)``."""
        assert self.net is not None
        self.net.eval()
        out = [
            self._batch_logits(teams[i:i + _PREDICT_BATCH],
                               None if rank is None else rank[i:i + _PREDICT_BATCH])
            .double().cpu().numpy()
            for i in range(0, len(teams), _PREDICT_BATCH)
        ]
        return np.concatenate(out) + self.delta if out else np.empty(0)

    def fit(self, train: pd.DataFrame, val: pd.DataFrame, vocab: HeroVocab) -> "TorchDraftTrainer":
        """Train with AdamW + BCE, early-stopping on un-augmented val log-loss.

        Augmentation (mirror, pick dropout) touches train only; the best-epoch
        weights are restored at the end. Returns self.
        """
        if len(val) == 0:
            raise ValueError("val split is empty; it is required for early stopping")
        self.vocab = vocab
        teams, y, rank = teams_to_matrix(train, vocab), labels(train), self._rank(train)
        self.delta = compute_mirror_delta(y, self.mirror_augment)
        if self.mirror_augment:
            if rank is None:
                teams, y = mirror(teams, y)
            else:
                teams, y, rank = mirror(teams, y, rank)
        val_teams, val_y, val_rank = teams_to_matrix(val, vocab), labels(val), self._rank(val)

        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self._prepare(train, vocab)
        self.net = self._build_network(vocab.size).to(self.device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        y_t = torch.as_tensor(y, dtype=torch.float32, device=self.device)

        best_loss, best_state, bad, self.history = float("inf"), None, 0, []
        for epoch in range(self.max_epochs):
            self.net.train()
            perm, total = rng.permutation(len(teams)), 0.0
            for start in range(0, len(perm), self.batch_size):
                idx = perm[start:start + self.batch_size]
                bt = teams[idx]
                if self.pick_dropout > 0.0:
                    bt = np.where(rng.random(bt.shape) < self.pick_dropout, -1, bt)
                logits = self._batch_logits(bt, None if rank is None else rank[idx])
                # No delta here: the mirrored objective is side-symmetric by design and
                # delta is re-applied at predict time (val loss below includes it).
                loss = F.binary_cross_entropy_with_logits(logits, y_t[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.detach().item() * len(idx)
            z = self._predict_logits(val_teams, val_rank)
            val_loss = float(np.mean(np.logaddexp(0.0, z) - val_y * z))  # BCE with logits
            record = {"epoch": epoch, "train_loss": total / len(perm), "val_log_loss": val_loss}
            self.history.append(record)
            self.tracker.log(record, step=epoch)
            if val_loss < best_loss - 1e-6:
                best_loss, bad = val_loss, 0
                best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        assert best_state is not None
        self.net.load_state_dict(best_state)
        self.net.eval()
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """P(radiant win), shape ``(N,)`` float64; partial drafts allowed (§6)."""
        if not self.is_fitted:
            raise RuntimeError(f"{type(self).__name__} is not fitted; call fit() or load()")
        validate_predict_frame(df)
        assert self.vocab is not None
        z = self._predict_logits(teams_to_matrix(df, self.vocab), self._rank(df))
        return 1.0 / (1.0 + np.exp(-z))

    # --- checkpoints -----------------------------------------------------------------
    def save(self, path: Path) -> None:
        """Single ``.pt``: state_dict, config, vocab hero ids, delta, extra tensors."""
        if not self.is_fitted or self.net is None:
            raise RuntimeError("cannot save an unfitted model")
        assert self.vocab is not None
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model": self.name,
            "config": self.config,
            "hero_ids": self.vocab.hero_ids,
            "delta": self.delta,
            "state_dict": {k: v.cpu() for k, v in self.net.state_dict().items()},
            "extra": {k: v.cpu() for k, v in self._extra_state().items()},
        }, path)

    @classmethod
    def load(cls, path: Path, device: str | None = None) -> "TorchDraftTrainer":
        """Restore a checkpoint written by :meth:`save` (needs nothing but the file)."""
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        if ckpt["model"] != cls.name:
            raise TypeError(f"{path} is a {ckpt['model']!r} checkpoint, expected {cls.name!r}")
        model = cls(**ckpt["config"], device=device)
        model.vocab = HeroVocab(ckpt["hero_ids"])
        model.delta = float(ckpt["delta"])
        model._set_extra_state(ckpt["extra"])
        model.net = model._build_network(model.vocab.size)
        model.net.load_state_dict(ckpt["state_dict"])
        model.net.to(model.device).eval()
        return model


class DraftTransformer(nn.Module):
    """Transformer encoder over ``[CLS, r1..r5, d1..d5]`` -> one win logit.

    Parameters
    ----------
    vocab_size : int
        Number of heroes ``V``; the table has ``V + 2`` rows (MASK, CLS).
    d_model, n_heads, n_layers, dim_ff, dropout
        Encoder hyperparameters.
    use_rank : bool, optional
        Concatenate the rank scalar to the normalized CLS vector before the MLP.
    """

    def __init__(self, vocab_size: int, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 2, dim_ff: int = 128, dropout: float = 0.1,
                 use_rank: bool = False) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.use_rank = use_rank
        self.tok = nn.Embedding(vocab_size + 2, d_model)
        self.team = nn.Embedding(3, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_ff, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model + int(use_rank), d_model), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(d_model, 1))
        self.register_buffer("team_layout", torch.tensor([0] + [1] * 5 + [2] * 5), persistent=False)

    def encode(self, teams: Tensor) -> tuple[Tensor, Tensor]:
        """``(B, 10)`` vocab indices with ``-1`` empties -> ``(tokens, team_ids)``, ``(B, 11)``."""
        hero = torch.where(teams < 0, torch.full_like(teams, self.vocab_size), teams)
        cls = torch.full_like(hero[:, :1], self.vocab_size + 1)
        tokens = torch.cat([cls, hero], dim=1)
        return tokens, self.team_layout.expand(tokens.shape[0], -1)

    def forward(self, tokens: Tensor, team_ids: Tensor, rank: Tensor | None = None) -> Tensor:
        """tokens (B, 11) long, team_ids (B, 11) long, rank (B, 1) float -> (B,) logits."""
        h = self.encoder(self.tok(tokens) + self.team(team_ids))
        x = self.norm(h[:, 0])
        if self.use_rank:
            if rank is None:
                raise ValueError("use_rank=True requires a rank tensor")
            x = torch.cat([x, rank], dim=1)
        return self.head(x).squeeze(-1)


class TransformerDraftModel(TorchDraftTrainer):
    """:class:`DraftModel` wrapper around :class:`DraftTransformer`.

    Parameters
    ----------
    d_model, n_heads, n_layers, dim_ff, dropout
        Architecture, see :class:`DraftTransformer`.
    use_rank : bool, optional
        Use the normalized rank feature, by default False.
    mirror_augment : bool, optional
        Mirror-double the train set and add the side delta (§3), by default False.
    pick_dropout : float, optional
        Train-time probability of replacing each hero with MASK, by default 0.0.
    lr, weight_decay : float, optional
        AdamW settings.
    batch_size, max_epochs, patience : int, optional
        Training loop settings; early stopping on un-augmented val log-loss.
    seed : int, optional
        Seed for init, shuffling and dropout masks (deterministic on CPU).
    device : str or None, optional
        Torch device; defaults to CUDA when available.
    tracker : Tracker or None, optional
        Per-epoch metric logger (no-op by default); not saved in checkpoints.
    """

    name = "transformer"

    def __init__(self, *, d_model: int = 64, n_heads: int = 4, n_layers: int = 2,
                 dim_ff: int = 128, dropout: float = 0.1, use_rank: bool = False,
                 mirror_augment: bool = False, pick_dropout: float = 0.0, lr: float = 3e-4,
                 weight_decay: float = 1e-5, batch_size: int = 1024, max_epochs: int = 30,
                 patience: int = 4, seed: int = SEED, device: str | None = None,
                 tracker: Tracker | None = None) -> None:
        super().__init__(
            {"d_model": d_model, "n_heads": n_heads, "n_layers": n_layers, "dim_ff": dim_ff,
             "dropout": dropout},
            use_rank=use_rank, mirror_augment=mirror_augment, pick_dropout=pick_dropout, lr=lr,
            weight_decay=weight_decay, batch_size=batch_size, max_epochs=max_epochs,
            patience=patience, seed=seed, device=device, tracker=tracker)

    def _build_network(self, vocab_size: int) -> DraftTransformer:
        return DraftTransformer(vocab_size, use_rank=self.use_rank, **self.arch)

    def _logits(self, teams: Tensor, rank: Tensor | None) -> Tensor:
        assert isinstance(self.net, DraftTransformer)
        tokens, team_ids = self.net.encode(teams)
        return self.net(tokens, team_ids, rank)
