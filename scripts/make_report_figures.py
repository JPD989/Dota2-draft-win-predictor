"""Regenerate every figure used in ``docs/report.md``.

All model figures are drawn from the harness outputs in ``results/`` (no model is
loaded or trained). Two figures also need the raw match shards: the hero win-rate
figure (``--data``) and the side-advantage panel of the patch-drift figure
(``--drift-data``); without them those parts are skipped with a message.

Figures written to ``--out`` (default ``docs/figures``):

- ``report_headline_auc.png``      AUC with 95% CIs + paired difference vs logistic regression
- ``report_prediction_spread.png`` distribution of predictions; accuracy vs confidence
- ``report_training_size.png``     test AUC vs number of training matches (log scale)
- ``report_calibration.png``       reliability diagrams on the test set, one panel per model
- ``report_patch_drift.png``       patch 7.41 -> 7.42: AUC, calibration error, side advantage
- ``report_time_drift.png``        AUC over consecutive windows within patch 7.41
- ``report_hero_win_rates.png``    ten strongest / weakest heroes with Wilson intervals

Usage
-----
    python scripts/make_report_figures.py
    python scripts/make_report_figures.py --results results --out docs/figures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # fallback if not pip-installed

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import ARTIFACTS_DIR, FIGURES_DIR, RAW_DIR, RESULTS_DIR  # noqa: E402
from src.eval.calibration import reliability_curve  # noqa: E402
from src.stats.wilson import wilson_ci  # noqa: E402

MODELS = ("logreg", "xgboost", "transformer", "gnn")
LABELS = {"logreg": "Logistic regression", "xgboost": "XGBoost",
          "transformer": "Transformer", "gnn": "GNN"}
# Validated colour-blind-safe set (all pairs separated under deuteranopia, protanopia,
# tritanopia); markers give every model a second, colour-free identity.
COLORS = {"logreg": "#8a5a00", "xgboost": "#eb6834", "transformer": "#1baf7a", "gnn": "#2a78d6"}
MARKERS = {"logreg": "o", "xgboost": "s", "transformer": "D", "gnn": "^"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3de"
TRAIN_SIZES = {"n20000": 20_000, "n80000": 80_000, "n320000": 320_000, "": 805_857}

STYLE = {
    "figure.dpi": 100, "savefig.dpi": 150, "savefig.bbox": "tight",
    "figure.facecolor": "white", "axes.facecolor": "white",
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.labelsize": 10,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "text.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "legend.frameon": False, "lines.linewidth": 2,
}


def _save(fig: plt.Figure, out: Path, name: str) -> None:
    """Write ``fig`` as PNG into ``out`` and close it."""
    path = out / name
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


def _metrics(path: Path) -> dict[str, dict[str, float]]:
    """``models`` block of a harness ``metrics.json``."""
    return json.loads(path.read_text())["models"]


def headline_auc(results: Path, out: Path) -> None:
    """AUC with bootstrap CI per model and paired ΔAUC vs logistic regression."""
    m = _metrics(results / "benchmark" / "metrics.json")
    order = sorted(MODELS, key=lambda k: m[k]["auc"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True,
                                 gridspec_kw={"wspace": 0.08})
    for i, k in enumerate(order):
        r = m[k]
        for ax, v, lo, hi in ((a1, r["auc"], r["auc_lo"], r["auc_hi"]),
                              (a2, r["delta_auc_vs_logreg"], r["delta_lo"], r["delta_hi"])):
            ax.errorbar(v, i, xerr=[[v - lo], [hi - v]], fmt=MARKERS[k], color=COLORS[k],
                        ms=8, capsize=4, elinewidth=2)
            ax.annotate(f"{v:+.3f}" if ax is a2 else f"{v:.3f}", (hi, i),
                        xytext=(6, 0), textcoords="offset points", va="center",
                        fontsize=9, color=INK)
    a1.set_yticks(range(len(order)), [LABELS[k] for k in order])
    a1.set_xlim(0.570, 0.611)
    a1.set_xlabel("Test AUC (95% bootstrap interval)")
    a1.set_title("Test AUC")
    a2.axvline(0, color=MUTED, lw=1, ls="--")
    a2.set_xlim(-0.004, 0.031)
    a2.set_xlabel("AUC minus logistic regression (paired 95% interval)")
    a2.set_title("Gain over logistic regression")
    a1.grid(axis="y", visible=False)
    a2.grid(axis="y", visible=False)
    _save(fig, out, "report_headline_auc.png")


def prediction_spread(results: Path, out: Path) -> None:
    """Histogram of GNN / logreg predictions and accuracy among confident calls."""
    pr = pd.read_parquet(results / "benchmark" / "predictions.parquet")
    y = pr["y"].to_numpy()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.4), gridspec_kw={"wspace": 0.28})
    bins = np.linspace(0.2, 0.9, 71)
    for k in ("logreg", "gnn"):
        a1.hist(pr[k], bins=bins, histtype="step", lw=2, color=COLORS[k],
                label=LABELS[k], density=True)
    a1.set_ylim(0, a1.get_ylim()[1] * 1.25)  # headroom for the legend
    a1.axvline(y.mean(), color=MUTED, lw=1, ls="--")
    a1.annotate(f"Radiant win rate\nin test set: {y.mean():.1%}", (y.mean(), a1.get_ylim()[1]),
                xytext=(40, -30), textcoords="offset points", fontsize=8.5, color=MUTED,
                arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8})
    a1.set_xlabel("Predicted P(Radiant wins)")
    a1.set_ylabel("Density of test matches")
    a1.set_title("Most predictions sit close to a coin flip")
    a1.legend(loc="upper left", fontsize=8.5)
    a1.grid(axis="x", visible=False)

    thresholds = np.arange(0.50, 0.701, 0.01)
    for k in MODELS:
        p = pr[k].to_numpy()
        conf = np.maximum(p, 1 - p)
        correct = (p >= 0.5) == (y == 1)
        acc = [correct[conf >= t].mean() if (conf >= t).sum() >= 500 else np.nan
               for t in thresholds]
        a2.plot(thresholds, acc, color=COLORS[k], marker=MARKERS[k], ms=5, markevery=5,
                label=LABELS[k])
        if k == "gnn":  # how many matches each threshold keeps, for the strongest model
            for t in (0.55, 0.60, 0.65):
                share = (conf >= t - 1e-9).mean()
                a2.annotate(f"{share:.0%} of\nmatches", (t, correct[conf >= t - 1e-9].mean()),
                            xytext=(-4, 14), textcoords="offset points", ha="center",
                            fontsize=8, color=MUTED)
    a2.set_xlabel("Only calls with confidence max(p, 1 - p) at least")
    a2.set_ylabel("Share of those calls that were right")
    a2.set_title("Accuracy rises with the model's confidence")
    a2.set_ylim(0.54, 0.76)
    a2.legend(loc="lower right", fontsize=8.5)
    _save(fig, out, "report_prediction_spread.png")


def training_size(results: Path, out: Path) -> None:
    """Test AUC vs number of training matches, same test set at every size."""
    rows = []
    for tag, n in TRAIN_SIZES.items():
        path = results / "benchmark" / tag / "metrics.json"
        if not path.exists():
            print(f"missing {path} (training-size point skipped)")
            continue
        for k, r in _metrics(path).items():
            if k in MODELS:
                rows.append((k, n, r["auc"], r["auc_lo"], r["auc_hi"]))
    t = pd.DataFrame(rows, columns=["model", "n", "auc", "lo", "hi"])
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    offsets = {"logreg": 0.97, "xgboost": 0.99, "transformer": 1.01, "gnn": 1.03}
    for k in MODELS:
        s = t[t.model == k].sort_values("n")
        x = s.n * offsets[k]  # small horizontal dodge so intervals do not overlap
        ax.errorbar(x, s.auc, yerr=[s.auc - s.lo, s.hi - s.auc], color=COLORS[k],
                    marker=MARKERS[k], ms=7, capsize=3, label=LABELS[k])
        dy = {"transformer": 6, "xgboost": -6}.get(k, 0)
        ax.annotate(LABELS[k], (x.iloc[-1], s.auc.iloc[-1]), xytext=(10, dy),
                    textcoords="offset points", va="center", fontsize=9, color=INK)
    ax.set_xscale("log")
    ax.set_xticks(list(TRAIN_SIZES.values()), ["20k", "80k", "320k", "806k"])
    ax.minorticks_off()
    ax.set_xlim(15_000, 1_500_000)
    ax.set_xlabel("Training matches (log scale)")
    ax.set_ylabel("Test AUC (95% bootstrap interval)")
    ax.set_title("The interaction models keep improving with more data")
    ax.legend(loc="upper left", fontsize=9)
    _save(fig, out, "report_training_size.png")


def calibration(results: Path, out: Path) -> None:
    """Reliability diagram per model (15 equal-mass bins, Wilson intervals)."""
    pr = pd.read_parquet(results / "benchmark" / "predictions.parquet")
    m = _metrics(results / "benchmark" / "metrics.json")
    y = pr["y"].to_numpy()
    fig, axes = plt.subplots(1, 4, figsize=(10, 3.1), sharex=True, sharey=True,
                             gridspec_kw={"wspace": 0.08})
    for ax, k in zip(axes, MODELS):
        c = reliability_curve(y, pr[k].to_numpy(), n_bins=15, strategy="quantile")
        ax.plot([0.35, 0.75], [0.35, 0.75], color=MUTED, lw=1, ls="--")
        ax.errorbar(c.p_mean, c.y_rate, yerr=[c.y_rate - c.ci_lo, c.ci_hi - c.y_rate],
                    fmt=MARKERS[k], color=COLORS[k], ms=5, capsize=2, lw=1.5)
        ax.set_title(LABELS[k], fontsize=10)
        ax.text(0.37, 0.71, f"ECE {m[k]['ece']:.4f}", fontsize=8.5, color=MUTED)
        ax.set_xlim(0.35, 0.75)
        ax.set_ylim(0.35, 0.75)
        ax.set_aspect("equal")
        ax.set_xlabel("Predicted")
    axes[0].set_ylabel("Observed Radiant win rate")
    fig.suptitle("Calibration on the test set: each point is 1/15 of the matches",
                 x=0.125, ha="left", fontsize=11, fontweight="bold", y=0.98)
    _save(fig, out, "report_calibration.png")


def _side_rates(results: Path, drift_data: Path) -> list[tuple[str, float, float, float]] | None:
    """Radiant win rate (Wilson CI) of the 7.41 test split and of the 7.42 set."""
    shards = sorted(drift_data.glob("**/matches_*.parquet")) if drift_data.exists() else []
    if not shards:
        print(f"no shards in {drift_data}: side-advantage panel skipped")
        return None
    y41 = pd.read_parquet(results / "benchmark" / "predictions.parquet")["y"].to_numpy()
    d = pd.concat([pd.read_parquet(s, columns=["match_id", "radiant_win"]) for s in shards])
    y42 = d.drop_duplicates("match_id")["radiant_win"].to_numpy(dtype=float)
    out = []
    for label, v in (("7.41 test", y41), ("7.42", y42)):
        lo, hi = wilson_ci(int(v.sum()), v.size)
        out.append((label, float(v.mean()), float(lo), float(hi)))
    return out


def patch_drift(results: Path, out: Path, drift_data: Path) -> None:
    """Within-patch vs next-patch AUC and ECE per model (+ side advantage if data)."""
    t = pd.read_csv(results / "drift" / "patch_drift.csv").set_index("model")
    models = [k for k in MODELS if k in t.index]
    side = _side_rates(results, drift_data)
    n = 3 if side else 2
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 3.4), gridspec_kw={"wspace": 0.45})
    a1, a2 = axes[0], axes[1]
    y = np.arange(len(models))
    for i, k in enumerate(models):
        r = t.loc[k]
        a1.plot([r.auc_within, r.auc_next], [i, i], color=COLORS[k], lw=2)
        a1.errorbar(r.auc_within, i, xerr=[[r.auc_within - r.auc_within_lo],
                    [r.auc_within_hi - r.auc_within]], fmt=MARKERS[k], color=COLORS[k],
                    ms=8, capsize=3)
        a1.errorbar(r.auc_next, i, xerr=[[r.auc_next - r.auc_next_lo],
                    [r.auc_next_hi - r.auc_next]], fmt=MARKERS[k], color=COLORS[k],
                    mfc="white", ms=8, capsize=3)
        a1.annotate(f"{r.delta_auc:+.3f}", ((r.auc_within + r.auc_next) / 2, i),
                    xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8.5)
        a2.plot([r.ece_within, r.ece_next], [i, i], color=COLORS[k], lw=2)
        a2.plot(r.ece_within, i, MARKERS[k], color=COLORS[k], ms=8)
        a2.plot(r.ece_next, i, MARKERS[k], color=COLORS[k], mfc="white", ms=8)
    for ax in (a1, a2):
        ax.set_yticks(y, [LABELS[k] for k in models])
        ax.set_ylim(-0.6, len(models) - 0.4)
        ax.grid(axis="y", visible=False)
    a1.set_xlabel("Test AUC  (filled: 7.41, open: 7.42)")
    a1.set_title("Ranking quality drops")
    a2.set_xlim(0, 0.036)
    a2.set_xlabel("Calibration error, ECE  (filled: 7.41, open: 7.42)")
    a2.set_title("Calibration error grows")
    a2.set_yticklabels([])
    if side:
        a3 = axes[2]
        for i, (label, p, lo, hi) in enumerate(side):
            a3.errorbar(i, p, yerr=[[p - lo], [hi - p]], fmt="o", color=INK, ms=7, capsize=4)
            a3.annotate(f"{p:.1%}", (i, hi), xytext=(0, 5), textcoords="offset points",
                        ha="center", fontsize=9)
        a3.axhline(0.5, color=MUTED, lw=1, ls="--")
        a3.set_xticks([0, 1], [s[0] for s in side])
        a3.set_xlim(-0.6, 1.6)
        a3.set_ylim(0.495, 0.54)
        a3.set_ylabel("Radiant win rate (95% Wilson CI)")
        a3.set_title("Radiant's edge shrank")
        a3.grid(axis="x", visible=False)
    _save(fig, out, "report_patch_drift.png")


def time_drift(results: Path, out: Path) -> None:
    """AUC over consecutive later windows inside patch 7.41."""
    t = pd.read_csv(results / "drift" / "time_drift.csv")
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for k in MODELS:
        s = t[t.model == k].sort_values("window")
        if s.empty:
            continue
        x = s.window + {"logreg": -0.12, "xgboost": -0.04, "transformer": 0.04, "gnn": 0.12}[k]
        ax.errorbar(x, s.auc, yerr=[s.auc - s.auc_lo, s.auc_hi - s.auc], color=COLORS[k],
                    marker=MARKERS[k], ms=6, capsize=2, lw=1.5, label=LABELS[k])
    w = t.drop_duplicates("window").sort_values("window")
    ticks = [f"{pd.Timestamp(a):%b %d}–{pd.Timestamp(b):%b %d}"
             for a, b in zip(w.window_start, w.window_end)]
    ax.set_xticks(w.window, ticks, fontsize=8.5)
    ax.set_xlabel("Evaluation window (2026, ~80k matches each)")
    ax.set_ylabel("AUC (95% bootstrap interval)")
    ax.set_title("Within patch 7.41, ranking quality holds steady")
    ax.legend(loc="lower left", ncol=4, fontsize=8.5)
    ax.grid(axis="x", visible=False)
    _save(fig, out, "report_time_drift.png")


def hero_win_rates(data: Path, out: Path) -> None:
    """Ten strongest and ten weakest heroes (pooled over sides, Wilson 95% CI)."""
    try:
        from src.data.dataset import HeroVocab, load_matches
        from src.data.heroes import load_hero_names
        from src.stats.synergy import hero_win_rates as _hero_wr
        df = load_matches(data)
        names = load_hero_names(ARTIFACTS_DIR / "heroes.json")
    except FileNotFoundError as e:
        print(f"hero win-rate figure skipped: {e}")
        return
    wr = _hero_wr(df, HeroVocab.from_matches(df)).sort_values("wr")
    sel = pd.concat([wr.head(10), wr.tail(10)])
    fig, ax = plt.subplots(figsize=(7.5, 5.6))
    yy = np.r_[np.arange(10), np.arange(11, 21)]
    above = sel.wr.to_numpy() >= 0.5
    ax.errorbar(sel.wr, yy, xerr=[sel.wr - sel.ci_lo, sel.ci_hi - sel.wr], fmt="none",
                ecolor=MUTED, capsize=2, lw=1.2)
    ax.scatter(sel.wr[above], yy[above], color=COLORS["gnn"], marker="^", s=40, zorder=3,
               label="above 50%")
    ax.scatter(sel.wr[~above], yy[~above], color=COLORS["xgboost"], marker="v", s=40,
               zorder=3, label="below 50%")
    ax.set_yticks(yy, [f"{names.get(int(h), h)}  ({g / 1000:.0f}k games)"
                       for h, g in zip(sel.hero_id, sel.games)], fontsize=8.5)
    ax.axvline(0.5, color=INK, lw=1, ls="--")
    ax.set_xlabel("Win rate, both sides pooled (95% Wilson interval)")
    ax.set_title(f"Strongest and weakest heroes, patch 7.41 ({len(df):,} matches)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", visible=False)
    _save(fig, out, "report_hero_win_rates.png")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code."""
    ap = argparse.ArgumentParser(description="Regenerate the figures of docs/report.md.")
    ap.add_argument("--results", type=Path, default=RESULTS_DIR, help="harness output root")
    ap.add_argument("--out", type=Path, default=FIGURES_DIR, help="figure directory")
    ap.add_argument("--data", type=Path, default=RAW_DIR,
                    help="patch-7.41 shards (hero win-rate figure)")
    ap.add_argument("--drift-data", type=Path, default=RAW_DIR.parent / "patch-742",
                    help="patch-7.42 shards (side-advantage panel)")
    args = ap.parse_args(argv)

    required = [args.results / "benchmark" / "metrics.json",
                args.results / "benchmark" / "predictions.parquet",
                args.results / "drift" / "patch_drift.csv",
                args.results / "drift" / "time_drift.csv"]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        print("Missing inputs: " + ", ".join(missing) + ". Run `make benchmark`, "
              "`make drift` and `make drift-patch` first.")
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(STYLE)
    headline_auc(args.results, args.out)
    prediction_spread(args.results, args.out)
    training_size(args.results, args.out)
    calibration(args.results, args.out)
    patch_drift(args.results, args.out, args.drift_data)
    time_drift(args.results, args.out)
    hero_win_rates(args.data, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
