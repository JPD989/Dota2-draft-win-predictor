"""Model registry and checkpoint conventions.

This package must stay importable in baseline-only environments (Streamlit
Community Cloud): neither torch nor any model module is imported at load
time. :func:`get_model_class` resolves :data:`MODEL_REGISTRY` entries lazily
via ``importlib``; app and scripts locate checkpoints exclusively through
:func:`get_model_class` + :func:`checkpoint_path`.
"""

from __future__ import annotations

import importlib
from pathlib import Path

from src.config import MODELS_DIR
from src.models.base import DraftModel

__all__ = ["DraftModel", "MODEL_REGISTRY", "checkpoint_path", "get_model_class"]

MODEL_REGISTRY: dict[str, tuple[str, str]] = {
    "logreg": ("src.models.baselines", "LogisticDraftModel"),
    "xgboost": ("src.models.baselines", "XGBoostDraftModel"),
    "transformer": ("src.models.transformer", "TransformerDraftModel"),
    "gnn": ("src.models.gnn", "GNNDraftModel"),
}
"""Model name -> (module, class name). Imports stay lazy — resolve via
:func:`get_model_class`, never by importing the module directly."""

_CHECKPOINT_SUFFIX: dict[str, str] = {
    "logreg": ".pkl",
    "xgboost": ".pkl",
    "transformer": ".pt",
    "gnn": ".pt",
}


def get_model_class(name: str) -> type[DraftModel]:
    """Resolve a registry name to its :class:`DraftModel` subclass.

    Imports the implementing module on demand, so torch-backed models cost
    nothing until actually requested.

    Parameters
    ----------
    name : str
        Registry key, e.g. ``"logreg"``.

    Returns
    -------
    type[DraftModel]
        The model class.

    Raises
    ------
    KeyError
        On unknown ``name``.
    ModuleNotFoundError
        If the implementing module (or a dependency such as torch) is not
        installed in this environment.
    """
    try:
        module_name, class_name = MODEL_REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown model name {name!r}; known: {sorted(MODEL_REGISTRY)}"
        ) from None
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def checkpoint_path(name: str, model_dir: Path = MODELS_DIR, tag: str | None = None) -> Path:
    """Canonical checkpoint location: ``model_dir/{name}[_{tag}]{suffix}``.

    Suffix is ``.pkl`` for logreg/xgboost (pickled wrappers) and ``.pt`` for
    transformer/gnn (torch checkpoints).

    Parameters
    ----------
    name : str
        Registry key, e.g. ``"transformer"``.
    model_dir : Path, optional
        Checkpoint directory, by default :data:`src.config.MODELS_DIR`.
    tag : str | None, optional
        Ablation tag appended as ``_{tag}`` (e.g. ``"rec"`` ->
        ``transformer_rec.pt``), by default ``None``.

    Returns
    -------
    Path
        e.g. ``models/transformer_rec.pt``.

    Raises
    ------
    KeyError
        On unknown ``name``.
    """
    try:
        suffix = _CHECKPOINT_SUFFIX[name]
    except KeyError:
        raise KeyError(
            f"unknown model name {name!r}; known: {sorted(_CHECKPOINT_SUFFIX)}"
        ) from None
    stem = name if tag is None else f"{name}_{tag}"
    return model_dir / f"{stem}{suffix}"
