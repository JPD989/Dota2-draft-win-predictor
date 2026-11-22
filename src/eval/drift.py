"""Drift studies: how fast does a draft model go stale?

Two designs (docs/architecture.md §4.14):

- :func:`time_drift_study` — single-patch proxy. Train on the oldest 54% of
  matches, validate on the next 6%, then score on ``n_windows`` consecutive
  windows covering the newest 40%.
- :func:`patch_drift_study` — the real thing: train within patch N, compare the
  within-patch test score against the score on patch N+1.

Both fit only on their own train/val blocks and write a CSV table and a PNG
plot into ``out_dir``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import RESULTS_DIR, SEED
from src.data.dataset import HeroVocab, chronological_split
from src.eval.metrics import bootstrap_auc_matrix, compute_metrics, percentile_ci
from src.models.base import DraftModel

__all__ = ["patch_drift_study", "time_drift_study"]

TRAIN_FRAC: float = 0.54
FRONT_FRAC: float = 0.60  # train + val
MIN_HONEST_SPAN_DAYS: float = 14.0
MIN_PATCH_ROWS: int = 10_000
_TIME_SORT_COLS = ["start_time", "match_id"]
_DAY = 86_400.0


def _to_utc(ts: float) -> pd.Timestamp:
    return pd.Timestamp(int(ts), unit="s", tz="UTC")


def _labels(df: pd.DataFrame) -> np.ndarray:
    return df["radiant_win"].to_numpy(dtype=np.float64)


def time_drift_study(
    df: pd.DataFrame,
    model_factories: dict[str, Callable[[], DraftModel]],
    vocab: HeroVocab,
    n_windows: int = 5,
    out_dir: Path = RESULTS_DIR / "drift",
    *,
    n_boot: int = 1000,
    seed: int = SEED,
) -> pd.DataFrame:
    """Single-patch drift proxy: score a frozen model on successively later windows.

    Rows are stable-sorted by ``(start_time, match_id)``. Train = first
    ``int(0.54 n)`` rows, val = rows up to ``int(0.60 n)`` (both inside the
    front 60%), and the remaining 40% is cut into ``n_windows`` consecutive,
    near-equal windows (``np.array_split``). Each factory builds a fresh model,
    fit once on train/val, then evaluated on every window.

    Prints an honesty warning when the evaluated span (first to last match of
    the back 40%) is shorter than 14 days: drift may then be
    indistinguishable from noise.

    Parameters
    ----------
    df : pd.DataFrame
        Matches (§2 schema).
    model_factories : dict[str, Callable[[], DraftModel]]
        Label -> zero-argument constructor of an unfitted model.
    vocab : HeroVocab
        Vocabulary passed to ``fit``.
    n_windows : int, optional
        Number of evaluation windows, by default 5.
    out_dir : Path, optional
        Writes ``time_drift.csv`` and ``time_drift.png`` here.
    n_boot : int, optional
        Bootstrap replicates for the per-window AUC CI, by default 1000.
    seed : int, optional
        Bootstrap seed, by default ``SEED``.

    Returns
    -------
    pd.DataFrame
        One row per (model, window): ``model, window, n, window_start,
        window_end`` (UTC timestamps of the first/last match),
        ``days_since_train_end`` (window midpoint minus the last train match,
        in days), ``auc, auc_lo, auc_hi, log_loss, brier, ece``.
    """
    if n_windows < 1:
        raise ValueError(f"n_windows must be >= 1, got {n_windows}")
    ordered = df.sort_values(_TIME_SORT_COLS, kind="stable").reset_index(drop=True)
    n = len(ordered)
    n_train, n_front = int(n * TRAIN_FRAC), int(n * FRONT_FRAC)
    if n_train == 0 or n_front == n_train or n - n_front < n_windows:
        raise ValueError(f"too few rows ({n}) for a 54/6/40 split with {n_windows} windows")
    train, val = ordered.iloc[:n_train], ordered.iloc[n_train:n_front]
    back = ordered.iloc[n_front:]
    windows = [back.iloc[ix] for ix in np.array_split(np.arange(len(back)), n_windows)]

    times = back["start_time"].to_numpy()
    span_days = (times.max() - times.min()) / _DAY
    if span_days < MIN_HONEST_SPAN_DAYS:
        print(
            f"WARNING: drift evaluation span is only {span_days:.1f} days "
            f"(< {MIN_HONEST_SPAN_DAYS:.0f}); any degradation may be indistinguishable "
            "from noise — treat these results as a smoke test, not a drift measurement."
        )
    train_end = float(train["start_time"].max())

    rows: list[dict] = []
    for label, factory in model_factories.items():
        model = factory()
        model.fit(train, val, vocab)
        for w, win in enumerate(windows):
            y = _labels(win)
            p = np.asarray(model.predict_proba(win), dtype=np.float64)
            m = compute_metrics(y, p)
            _, boot = bootstrap_auc_matrix(y, {label: p}, n_boot=n_boot, seed=seed)
            lo, hi = percentile_ci(boot[:, 0])
            t0, t1 = float(win["start_time"].min()), float(win["start_time"].max())
            rows.append(
                {
                    "model": label, "window": w, "n": m["n"],
                    "window_start": _to_utc(t0), "window_end": _to_utc(t1),
                    "days_since_train_end": ((t0 + t1) / 2.0 - train_end) / _DAY,
                    "auc": m["auc"], "auc_lo": lo, "auc_hi": hi,
                    "log_loss": m["log_loss"], "brier": m["brier"], "ece": m["ece"],
                }
            )
    result = pd.DataFrame(rows)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_dir / "time_drift.csv", index=False)
    _plot_time_drift(result, out_dir / "time_drift.png", span_days)
    return result


def _plot_time_drift(result: pd.DataFrame, out_path: Path, span_days: float) -> None:
    from matplotlib.figure import Figure

    fig = Figure(figsize=(7.0, 4.5))
    ax = fig.add_subplot(1, 1, 1)
    for label, g in result.groupby("model", sort=False):
        x = g["days_since_train_end"].to_numpy()
        ax.plot(x, g["auc"], marker="o", label=str(label))
        ax.fill_between(x, g["auc_lo"], g["auc_hi"], alpha=0.15)
    ax.set_xlabel("days since end of train block (window midpoint)")
    ax.set_ylabel("test AUC (95% bootstrap CI)")
    ax.set_title(f"Time drift (evaluated span {span_days:.1f} days)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)


def patch_drift_study(
    df: pd.DataFrame,
    model_factories: dict[str, Callable[[], DraftModel]],
    vocab: HeroVocab,
    *,
    train_patch: str,
    test_patch: str,
    out_dir: Path = RESULTS_DIR / "drift",
    n_boot: int = 1000,
    seed: int = SEED,
    min_rows: int = MIN_PATCH_ROWS,
) -> pd.DataFrame:
    """Train within ``train_patch``; compare within-patch vs next-patch test scores.

    Builds its own splits (not ``patch_split``):
    ``chronological_split(df[patch == train_patch])`` (80/10/10) gives
    train / val / within-patch test; ``df[patch == test_patch]`` is the drift
    test set. Models never see any ``test_patch`` row during fitting.

    Degradation columns are ``metric(next) − metric(within)``; the
    ``delta_auc`` CI combines independent bootstraps of the two (disjoint)
    test sets.

    Parameters
    ----------
    df : pd.DataFrame
        Matches with a ``patch`` column.
    model_factories : dict[str, Callable[[], DraftModel]]
        Label -> zero-argument constructor of an unfitted model.
    vocab : HeroVocab
        Vocabulary passed to ``fit`` (must cover both patches' heroes).
    train_patch, test_patch : str
        Source patch and the later patch to test drift on.
    out_dir : Path, optional
        Writes ``patch_drift.csv`` and ``patch_drift.png`` here.
    n_boot : int, optional
        Bootstrap replicates, by default 1000.
    seed : int, optional
        Bootstrap seed, by default ``SEED``.
    min_rows : int, optional
        Minimum rows per patch, by default 10,000 (lower only in tests).

    Returns
    -------
    pd.DataFrame
        One row per model: ``model, n_train, n_within, n_next``; for each of
        ``auc, log_loss, brier, ece``: ``{metric}_within``, ``{metric}_next``,
        ``delta_{metric}``; plus ``auc_within_lo/hi``, ``auc_next_lo/hi``,
        ``delta_auc_lo/hi``.

    Raises
    ------
    ValueError
        If either patch has fewer than ``min_rows`` rows.
    """
    counts = {p: int((df["patch"] == p).sum()) for p in (train_patch, test_patch)}
    small = {p: c for p, c in counts.items() if c < min_rows}
    if small:
        raise ValueError(
            f"patch_drift_study needs >= {min_rows:,} rows per patch; got "
            + ", ".join(f"{p!r}: {c:,}" for p, c in small.items())
            + f" (available patches: {sorted(df['patch'].unique().tolist())}). Collect the "
            "newer patch into its own data/raw/<era> directory first (architecture §2)."
        )
    split = chronological_split(df[df["patch"] == train_patch])
    nxt = df[df["patch"] == test_patch].sort_values(_TIME_SORT_COLS, kind="stable")
    y_within, y_next = _labels(split.test), _labels(nxt)

    rows: list[dict] = []
    for label, factory in model_factories.items():
        model = factory()
        model.fit(split.train, split.val, vocab)
        p_within = np.asarray(model.predict_proba(split.test), dtype=np.float64)
        p_next = np.asarray(model.predict_proba(nxt), dtype=np.float64)
        m_w, m_n = compute_metrics(y_within, p_within), compute_metrics(y_next, p_next)
        _, b_w = bootstrap_auc_matrix(y_within, {label: p_within}, n_boot=n_boot, seed=seed)
        _, b_n = bootstrap_auc_matrix(y_next, {label: p_next}, n_boot=n_boot, seed=seed + 1)
        row: dict = {
            "model": label, "n_train": len(split.train),
            "n_within": m_w["n"], "n_next": m_n["n"],
        }
        for k in ("auc", "log_loss", "brier", "ece"):
            row[f"{k}_within"], row[f"{k}_next"] = m_w[k], m_n[k]
            row[f"delta_{k}"] = m_n[k] - m_w[k]
        row["auc_within_lo"], row["auc_within_hi"] = percentile_ci(b_w[:, 0])
        row["auc_next_lo"], row["auc_next_hi"] = percentile_ci(b_n[:, 0])
        row["delta_auc_lo"], row["delta_auc_hi"] = percentile_ci(b_n[:, 0] - b_w[:, 0])
        rows.append(row)
    result = pd.DataFrame(rows)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_dir / "patch_drift.csv", index=False)
    _plot_patch_drift(result, out_dir / "patch_drift.png", train_patch, test_patch)
    return result


def _plot_patch_drift(result: pd.DataFrame, out_path: Path, train_patch: str, test_patch: str) -> None:
    from matplotlib.figure import Figure

    fig = Figure(figsize=(7.0, 4.5))
    ax = fig.add_subplot(1, 1, 1)
    x = np.arange(len(result))
    width = 0.38
    for offset, key, lab in ((-width / 2, "within", f"{train_patch} (within)"),
                             (width / 2, "next", f"{test_patch} (next)")):
        auc = result[f"auc_{key}"].to_numpy()
        err = np.vstack([auc - result[f"auc_{key}_lo"], result[f"auc_{key}_hi"] - auc]).clip(min=0)
        ax.bar(x + offset, auc, width, yerr=err, capsize=3, label=lab)
    ax.set_xticks(x, result["model"].tolist())
    lo = np.nanmin(result[["auc_within_lo", "auc_next_lo"]].to_numpy())
    ax.set_ylim(max(0.0, (lo if np.isfinite(lo) else 0.5) - 0.02), None)
    ax.set_ylabel("test AUC (95% bootstrap CI)")
    ax.set_title(f"Patch drift: train {train_patch} -> test {test_patch}")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
