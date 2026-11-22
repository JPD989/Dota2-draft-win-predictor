"""Streamlit demo: live draft win probability + pick recommendations.

Run with ``streamlit run app/streamlit_app.py``. The module is split in two:

- Pure logic (importable and testable without Streamlit or torch):
  :func:`load_artifacts`, :class:`StatsDraftModel`, :func:`find_checkpoints`.
- UI, guarded under :func:`main` (Streamlit is imported only there).

With no trained checkpoint the app runs in **stats mode**: a heuristic
additive log-odds model built from the committed train-split artifacts in
``data/artifacts/`` (docs/architecture.md §4.18)::

    score = logit(radiant_wr_train)
            + sum_{a in R} logit(wr_a) - sum_{b in D} logit(wr_b)
            + sum_{a<b in R} syn[a,b] - sum_{a<b in D} syn[a,b]
            + sum_{a in R, b in D} cnt[a,b]
    p = sigmoid(score)
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import MutableMapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root (Streamlit Cloud)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import ARTIFACTS_DIR, MODELS_DIR  # noqa: E402
from src.data.dataset import HeroVocab  # noqa: E402
from src.features.encoding import teams_to_matrix  # noqa: E402
from src.models import MODEL_REGISTRY, checkpoint_path  # noqa: E402
from src.models.base import DraftModel, logit, sigmoid  # noqa: E402
from src.stats.synergy import PairStats  # noqa: E402

REQUIRED_ARTIFACTS = ("heroes.json", "hero_vocab.json", "synergy.npz", "counter.npz",
                      "hero_win_rates.parquet", "meta.json")
HERO_ASSETS = "hero_assets.json"  # optional: portraits/metadata (scripts/fetch_hero_constants.py)
TORCH_MODELS = ("transformer", "gnn")
STATS_LABEL = "stats (heuristic, no training)"
MODEL_TITLES = {"stats": "Stats heuristic", "logreg": "Logistic regression",
                "xgboost": "XGBoost", "transformer": "Transformer", "gnn": "GNN"}
MEDALS = ("Herald", "Guardian", "Crusader", "Archon", "Legend", "Ancient", "Divine")
RANK_TIERS = tuple(10 * m + s for m in range(1, 8) for s in range(1, 6))  # 11 .. 75
DEFAULT_RANK = 53


# --------------------------------------------------------------------------- logic
@dataclass
class Artifacts:
    """Everything the app needs from ``data/artifacts/``."""

    hero_names: dict[int, str]
    vocab: HeroVocab
    synergy: PairStats
    counter: PairStats
    hero_win_rates: pd.DataFrame
    meta: dict[str, Any]


def missing_artifacts(artifacts_dir: Path = ARTIFACTS_DIR) -> list[str]:
    """Names of required artifact files absent from ``artifacts_dir``."""
    return [f for f in REQUIRED_ARTIFACTS if not (artifacts_dir / f).exists()]


def load_artifacts(artifacts_dir: Path = ARTIFACTS_DIR) -> Artifacts:
    """Load all app artifacts.

    Raises
    ------
    FileNotFoundError
        If any required file is missing (see :func:`missing_artifacts`).
    """
    missing = missing_artifacts(artifacts_dir)
    if missing:
        raise FileNotFoundError(f"missing artifacts in {artifacts_dir}: {missing}")
    names = {int(k): str(v) for k, v in
             json.loads((artifacts_dir / "heroes.json").read_text(encoding="utf-8")).items()}
    return Artifacts(
        hero_names=names,
        vocab=HeroVocab.load(artifacts_dir / "hero_vocab.json"),
        synergy=PairStats.load(artifacts_dir / "synergy.npz"),
        counter=PairStats.load(artifacts_dir / "counter.npz"),
        hero_win_rates=pd.read_parquet(artifacts_dir / "hero_win_rates.parquet"),
        meta=json.loads((artifacts_dir / "meta.json").read_text()),
    )


class StatsDraftModel(DraftModel):
    """Heuristic additive log-odds model over EB-shrunk synergy/counter matrices.

    Implements the pinned stats-mode formula (module docstring) so the
    recommender works with zero trained models. Not a benchmark model: the
    terms are marginal statistics added together, not fit by likelihood.

    Parameters
    ----------
    vocab : HeroVocab
        Index layout of the matrices.
    syn, cnt : np.ndarray
        ``(V, V)`` synergy (symmetric, zero diagonal) and counter
        (antisymmetric) matrices, e.g. ``PairStats.value``.
    radiant_wr_train : float
        Train-split radiant win rate (the intercept).
    hero_logit : np.ndarray or None, optional
        ``(V,)`` per-hero strength ``logit(wr)`` (side-pooled train win rates);
        ``None`` means zeros (pair effects only).
    """

    name = "stats"

    def __init__(self, vocab: HeroVocab, syn: np.ndarray, cnt: np.ndarray,
                 radiant_wr_train: float, hero_logit: np.ndarray | None = None) -> None:
        v = vocab.size
        if syn.shape != (v, v) or cnt.shape != (v, v):
            raise ValueError(f"matrices must be ({v}, {v}); got {syn.shape}, {cnt.shape}")
        self.vocab = vocab
        self.use_rank = False
        self.syn = np.asarray(syn, dtype=np.float64)
        self.cnt = np.asarray(cnt, dtype=np.float64)
        self.radiant_wr_train = float(radiant_wr_train)
        self.hero_logit = (np.zeros(v) if hero_logit is None
                           else np.asarray(hero_logit, dtype=np.float64))
        if self.hero_logit.shape != (v,):
            raise ValueError(f"hero_logit must be ({v},); got {self.hero_logit.shape}")

    @classmethod
    def from_artifacts(cls, art: Artifacts) -> StatsDraftModel:
        """Build from loaded artifacts (uses the EB-shrunk ``value`` matrices)."""
        wr = art.hero_win_rates.set_index("hero_id")["wr"].reindex(art.vocab.hero_ids)
        return cls(art.vocab, art.synergy.value, art.counter.value,
                   art.meta["radiant_wr_train"], logit(wr.fillna(0.5).to_numpy()))

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """``(N,)`` radiant log-odds score (pinned formula); partial drafts allowed."""
        t = teams_to_matrix(df, self.vocab).astype(np.int64)
        r, d = t[:, :5], t[:, 5:]
        r_ok, d_ok = r >= 0, d >= 0
        r0, d0 = np.where(r_ok, r, 0), np.where(d_ok, d, 0)

        off_diag = ~np.eye(5, dtype=bool)

        def within(idx: np.ndarray, ok: np.ndarray) -> np.ndarray:
            # each unordered pair a<b counted once: masked off-diagonal sum / 2
            m = self.syn[idx[:, :, None], idx[:, None, :]]
            mask = ok[:, :, None] & ok[:, None, :] & off_diag
            return (m * mask).sum(axis=(1, 2)) / 2.0

        cross = self.cnt[r0[:, :, None], d0[:, None, :]] * (r_ok[:, :, None] & d_ok[:, None, :])
        solo = (self.hero_logit[r0] * r_ok).sum(axis=1) - (self.hero_logit[d0] * d_ok).sum(axis=1)
        return (logit(self.radiant_wr_train) + solo + within(r0, r_ok) - within(d0, d_ok)
                + cross.sum(axis=(1, 2)))

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """``(N,)`` P(radiant win) = sigmoid(score)."""
        return np.asarray(sigmoid(self.score(df)), dtype=np.float64).reshape(-1)

    def fit(self, train: pd.DataFrame, val: pd.DataFrame, vocab: HeroVocab) -> StatsDraftModel:
        """Recompute the matrices from ``train`` (val unused). Returns self."""
        from src.stats.synergy import counter_matrix, synergy_matrix
        from src.stats.synergy import hero_win_rates
        wr = hero_win_rates(train, vocab).set_index("hero_id")["wr"].reindex(vocab.hero_ids)
        self.__init__(vocab, synergy_matrix(train, vocab).value,
                      counter_matrix(train, vocab).value,
                      float(train["radiant_win"].mean()), logit(wr.fillna(0.5).to_numpy()))
        return self

    def save(self, path: Path) -> None:
        """Write a self-contained ``.npz`` (vocab, matrices, intercept)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, hero_ids=np.asarray(self.vocab.hero_ids), syn=self.syn,
                            cnt=self.cnt, radiant_wr_train=self.radiant_wr_train,
                            hero_logit=self.hero_logit)

    @classmethod
    def load(cls, path: Path) -> StatsDraftModel:
        """Inverse of :meth:`save`."""
        with np.load(path) as z:
            return cls(HeroVocab(z["hero_ids"].tolist()), z["syn"], z["cnt"],
                       float(z["radiant_wr_train"]), z["hero_logit"])


def torch_available() -> bool:
    """True if torch is importable — checked without importing it."""
    return importlib.util.find_spec("torch") is not None


def find_checkpoints(dirs: tuple[Path, ...] = (MODELS_DIR, MODELS_DIR / "release"),
                     include_torch: bool | None = None) -> dict[str, tuple[str, Path]]:
    """Scan checkpoint dirs for registry models.

    Returns ``{label: (model_name, path)}``, label like ``"transformer [rec] (release)"``.
    Torch checkpoints are only listed when torch is installed.
    """
    if include_torch is None:
        include_torch = torch_available()
    found: dict[str, tuple[str, Path]] = {}
    for d in map(Path, dirs):
        if not d.is_dir():
            continue
        where = " (release)" if d.name == "release" else ""
        for name in MODEL_REGISTRY:
            if name in TORCH_MODELS and not include_torch:
                continue
            suffix = checkpoint_path(name, d).suffix
            for p in sorted(d.glob(f"{name}*{suffix}")):
                stem_tag = p.stem[len(name):]
                if stem_tag and not stem_tag.startswith("_"):
                    continue  # a different model whose name shares the prefix
                tag = f" [{stem_tag[1:]}]" if stem_tag else ""
                found[f"{name}{tag}{where}"] = (name, p)
    return found


def load_checkpoint(name: str, path: str) -> DraftModel:
    """Load a registry model checkpoint (imports torch only for torch models)."""
    from src.models import get_model_class
    return get_model_class(name).load(Path(path))


def is_experimental(model: DraftModel, partial: bool) -> bool:
    """§4.15: neural models trained without pick masking are overconfident on partial drafts.

    One-hot models (logistic regression, XGBoost) are exempt: a missing hero simply
    contributes nothing, and they stay calibrated on unfinished drafts.
    """
    if not partial or model.name in ("stats", "logreg", "xgboost"):
        return False
    return float(getattr(model, "pick_dropout", 0.0) or 0.0) <= 0.0


def load_hero_assets(artifacts_dir: Path = ARTIFACTS_DIR) -> dict[int, dict[str, Any]]:
    """Optional hero display metadata (image URLs, attribute, roles) keyed by hero id.

    Returns ``{}`` when ``hero_assets.json`` is missing or unreadable; the UI then
    shows hero names without portraits.
    """
    try:
        raw = json.loads((artifacts_dir / HERO_ASSETS).read_text(encoding="utf-8"))
        return {int(k): dict(v) for k, v in raw.items() if isinstance(v, dict)}
    except (OSError, ValueError, AttributeError):
        return {}


def model_title(label: str) -> str:
    """Display name for a model-picker label (``"transformer [rec]"`` -> ``"Transformer [rec]"``)."""
    if label == STATS_LABEL:
        return MODEL_TITLES["stats"]
    head, _, rest = label.partition(" ")
    return f"{MODEL_TITLES.get(head, head)} {rest}".strip()


def rank_label(tier: int) -> str:
    """``53`` -> ``"Legend 3"`` (tens digit = medal, units = stars)."""
    m, s = divmod(int(tier), 10)
    return f"{MEDALS[m - 1]} {s}" if 1 <= m <= len(MEDALS) else str(tier)


def sanitize_draft(radiant: Sequence[int], dire: Sequence[int],
                   banned: Sequence[int]) -> tuple[list[int], list[int], list[int]]:
    """Drop duplicates and conflicts (radiant > dire > bans priority); cap teams at 5."""
    seen: set[int] = set()

    def keep(heroes: Sequence[int], cap: int | None) -> list[int]:
        out = []
        for h in map(int, heroes):
            if h not in seen and (cap is None or len(out) < cap):
                out.append(h)
                seen.add(h)
        return out

    return keep(radiant, 5), keep(dire, 5), keep(banned, None)


def add_pick(state: MutableMapping[str, Any], side: str, hero: int) -> bool:
    """Append ``hero`` to ``state[side]`` if legal (team not full, hero not taken).

    Used as a button callback on ``st.session_state``; returns whether it was added.
    """
    taken = {int(h) for k in ("radiant", "dire", "bans") for h in state.get(k, [])}
    team = list(state.get(side, []))
    if side not in ("radiant", "dire") or len(team) >= 5 or int(hero) in taken:
        return False
    state[side] = [*team, int(hero)]
    return True


def draft_hero_stats(art: Artifacts, radiant: Sequence[int],
                     dire: Sequence[int]) -> list[dict[str, Any]]:
    """Train-split win rate + Wilson 95% CI for each picked hero, in pick order."""
    wr = art.hero_win_rates.set_index("hero_id")
    rows = []
    for side, team in (("radiant", radiant), ("dire", dire)):
        for h in team:
            if h in wr.index:
                r = wr.loc[h]
                rows.append({"hero_id": int(h), "side": side, "wr": float(r["wr"]),
                             "ci_lo": float(r["ci_lo"]), "ci_hi": float(r["ci_hi"]),
                             "games": int(r["games"])})
    return rows


def draft_pairs(art: Artifacts, radiant: Sequence[int], dire: Sequence[int],
                top: int = 5) -> list[dict[str, Any]]:
    """Strongest EB-shrunk pair effects present in the draft, by ``|value|``.

    Synergy rows (``kind="synergy"``, same ``team``) hold that team's log-odds lift;
    counter rows (``kind="counter"``, ``a`` Radiant vs ``b`` Dire) hold Radiant's.
    Cells shrunk to exactly 0 (too few games) are skipped.
    """
    ix = {h: art.vocab.idx(h) for h in (*radiant, *dire) if h in set(art.vocab.hero_ids)}
    rows = []
    for team_name, team in (("radiant", radiant), ("dire", dire)):
        t = [h for h in team if h in ix]
        for i, a in enumerate(t):
            for b in t[i + 1:]:
                rows.append({"kind": "synergy", "team": team_name, "a": a, "b": b,
                             "value": float(art.synergy.value[ix[a], ix[b]]),
                             "games": int(art.synergy.games[ix[a], ix[b]])})
    for a in (h for h in radiant if h in ix):
        for b in (h for h in dire if h in ix):
            rows.append({"kind": "counter", "team": "radiant", "a": a, "b": b,
                         "value": float(art.counter.value[ix[a], ix[b]]),
                         "games": int(art.counter.games[ix[a], ix[b]])})
    rows = [r for r in rows if r["value"] != 0.0]
    return sorted(rows, key=lambda r: -abs(r["value"]))[:top]


# ------------------------------------------------------------------------------ UI
METHOD_MD = """
**What the number is.** P(Radiant win) given only the ten picks (plus rank tier for
rank-aware checkpoints). Player skill, roles and the game itself are invisible to the
model, so the honest ceiling is low: about **0.59 test AUC** on this data. Most drafts land
within a few points of 50%; a 54% draft is a meaningful edge, not a sure win.

**Calibration.** A predicted 60% should mean ~60% of such drafts are won. That holds
only as well as the model's test-set reliability curve shows (calibration results in the
repo); the stats heuristic is not calibrated at all.

**Recommendations.** Every legal hero is added to the picking side and the whole batch is
scored in one call; cards show the picking side's predicted win probability and the change
versus the current draft. Neural checkpoints trained without pick masking never saw
partial drafts and are labelled *experimental* mid-draft.

**Stats mode.** `logit(radiant base rate) + Σ hero logit(win rate) + Σ same-team synergy
+ Σ cross-team counter`, all from train-split statistics shrunk with empirical Bayes. It is a
readable baseline, not a fitted model.

**Hero table.** Side-pooled train-split win rates with Wilson 95% intervals; pair effects are
EB-shrunk log-odds (×25 ≈ points near 50%). Code, benchmark and write-up:
[github.com/JPD989/Dota2-draft-win-predictor](https://github.com/JPD989/Dota2-draft-win-predictor).
"""


def _instructions_screen(st: Any, missing: list[str]) -> None:
    st.title("Dota 2 Draft Win Predictor")
    st.warning(f"Missing artifacts in `data/artifacts/`: {', '.join(missing)}")
    st.markdown(
        "Build them first (no training needed for stats mode):\n\n"
        "```bash\nmake data        # collect + validate matches (or copy shards into data/raw/)\n"
        "python scripts/fetch_hero_constants.py   # heroes.json + hero_assets.json\n"
        "make artifacts   # vocab, synergy/counter matrices, win rates, meta\n```\n"
        "Then reload this page."
    )


def _clear_draft(state: MutableMapping[str, Any]) -> None:
    for k in ("radiant", "dire", "bans"):
        state[k] = []


def _swap_sides(state: MutableMapping[str, Any]) -> None:
    state["radiant"], state["dire"] = list(state.get("dire", [])), list(state.get("radiant", []))


def _sidebar(st: Any, checkpoints: dict[str, tuple[str, Path]]) -> str:
    """Model picker in the sidebar; returns the chosen label."""
    with st.sidebar:
        st.markdown("### Model")
        choice = st.selectbox("Predictor", [STATS_LABEL, *checkpoints], format_func=model_title,
                              key="model", help="Checkpoints found in models/ and models/release/.")
        if not torch_available():
            st.caption("torch is not installed here, so transformer/GNN checkpoints are hidden.")
    return choice


def _load_model(st: Any, art: Artifacts, choice: str,
                checkpoints: dict[str, tuple[str, Path]]) -> tuple[DraftModel, str]:
    model: DraftModel = StatsDraftModel.from_artifacts(art)  # cheap: wraps loaded arrays
    if choice == STATS_LABEL:
        return model, choice
    name, path = checkpoints[choice]
    try:
        return st.cache_resource(load_checkpoint)(name, str(path)), choice
    except Exception as e:  # missing dependency / incompatible checkpoint
        st.sidebar.error(f"Could not load {path.name} ({type(e).__name__}: {e}). "
                         "Falling back to stats mode.")
        return model, STATS_LABEL


def _team_panel(st: Any, ui: Any, side: str, heroes: list[int], options: list[int],
                names: dict[int, str], assets: dict[int, dict[str, Any]]) -> None:
    with st.container(border=True, key=f"{side}_panel"):
        st.html(ui.team(side, heroes, names, assets))
        st.multiselect(f"{side.capitalize()} picks", options, format_func=names.get,
                       max_selections=5, key=side, label_visibility="collapsed",
                       placeholder=f"Search {side.capitalize()} heroes...")


def _recommendations(st: Any, ui: Any, model: DraftModel, radiant: list[int], dire: list[int],
                     banned: list[int], rank: int, p_rad: float, names: dict[int, str],
                     assets: dict[int, dict[str, Any]]) -> None:
    head, ctl = st.columns([3, 1], vertical_alignment="bottom")
    side = ctl.segmented_control("Picking side", ["radiant", "dire"], format_func=str.capitalize,
                                 default="radiant", required=True, key="side",
                                 width="stretch") or "radiant"
    own = radiant if side == "radiant" else dire
    head.html(ui.section(f"Best next picks for {side.capitalize()}",
                         "Picking side's win probability with this hero added, and the change from now."))
    if len(own) >= 5:
        other = "dire" if side == "radiant" else "radiant"
        done = len(radiant if other == "radiant" else dire) >= 5
        st.html(ui.note(f"<b>{side.capitalize()} line-up is complete.</b> " + (
            "Both line-ups are complete, so the prediction above is final."
            if done else f"Switch the picking side to {other.capitalize()} for its next pick.")))
        return
    try:
        from src.recommend.recommender import recommend
        recs = recommend(model, model.vocab, radiant=radiant, dire=dire, side=side,
                         banned=banned, avg_rank_tier=rank, top_k=10)
    except (ImportError, ValueError) as e:
        st.error(f"Recommender unavailable ({e}).")
        return
    p_side = p_rad if side == "radiant" else 1 - p_rad
    rows = list(recs.itertuples(index=False))
    for start in range(0, len(rows), 5):
        for col, (i, r) in zip(st.columns(5), enumerate(rows[start:start + 5], start + 1),
                               strict=False):
            h = int(r.hero_id)
            with col:
                st.html(ui.rec_card(i, names.get(h, f"hero {h}"), assets.get(h, {}),
                                    float(r.win_prob), float(r.win_prob) - p_side))
                st.button(f"Pick for {side.capitalize()}", key=f"pick_{h}", icon=":material/add:",
                          on_click=add_pick, args=(st.session_state, side, h), width="stretch")


def main() -> None:
    """Streamlit UI entry point."""
    import streamlit as st

    from app import components as ui

    st.set_page_config(page_title="Dota 2 Draft Win Predictor", page_icon=":crossed_swords:",
                       layout="wide")
    missing = missing_artifacts()
    if missing:
        _instructions_screen(st, missing)
        return

    art = st.cache_resource(load_artifacts)()
    assets = st.cache_resource(load_hero_assets)()
    names = {h: art.hero_names.get(h, f"hero {h}") for h in art.vocab.hero_ids}
    heroes = sorted(art.vocab.hero_ids, key=lambda h: names[h])
    st.html(ui.CSS)

    checkpoints = find_checkpoints()
    choice = _sidebar(st, checkpoints)
    model, choice = _load_model(st, art, choice, checkpoints)
    with st.sidebar:
        rank = st.select_slider("Average rank tier", RANK_TIERS, value=DEFAULT_RANK,
                                format_func=rank_label, key="rank", disabled=not model.use_rank,
                                help="Only used by checkpoints trained with --use-rank.")
        if not model.use_rank:
            st.caption("This model ignores rank tier.")
        st.divider()
        st.caption(f"Train split: {art.meta['n_train']:,} ranked All Pick matches, patch "
                   f"{', '.join(art.meta.get('patches_train', [])) or 'n/a'}.")

    # ---- draft state: sanitise before the widgets are created (keys: radiant/dire/bans)
    ss = st.session_state
    ss.radiant, ss.dire, ss.bans = sanitize_draft(ss.get("radiant", []), ss.get("dire", []),
                                                  ss.get("bans", []))
    radiant, dire, banned = list(ss.radiant), list(ss.dire), list(ss.bans)

    frame = pd.DataFrame({"radiant_team": [radiant], "dire_team": [dire],
                          "avg_rank_tier": [rank]})
    p_rad = float(model.predict_proba(frame)[0])

    patch = ", ".join(art.meta.get("patches_train", [])) or "n/a"
    st.html(ui.header([("Model", model_title(choice)), ("Patch", patch),
                       ("Train matches", f"{art.meta['n_train']:,}")]))
    if choice == STATS_LABEL:
        st.html(ui.note("<b>Heuristic stats model.</b> Train a model for real predictions. "
                        "Score = side bias + hero win rates + EB-shrunk synergy and counter terms "
                        f"from {art.meta['n_train']:,} train-split matches."))
    if is_experimental(model, len(radiant) < 5 or len(dire) < 5):
        st.html(ui.note(f"<b>Experimental mid-draft.</b> {model_title(choice)} was not trained "
                        "with pick masking, so its probabilities are overconfident on unfinished "
                        "drafts. Prefer a <code>--pick-dropout</code> checkpoint.",
                        warn=True))

    st.html(ui.win_bar(p_rad, len(radiant), len(dire)))

    left, right = st.columns(2, gap="medium")
    with left:
        _team_panel(st, ui, "radiant", radiant, [h for h in heroes if h not in {*dire, *banned}],
                    names, assets)
    with right:
        _team_panel(st, ui, "dire", dire, [h for h in heroes if h not in {*radiant, *banned}],
                    names, assets)

    with st.container(border=True, key="bans_panel"):
        c_sel, c_strip, c_act = st.columns([2, 3, 1.2], vertical_alignment="center")
        c_sel.multiselect("Bans", [h for h in heroes if h not in {*radiant, *dire}],
                          format_func=names.get, key="bans", placeholder="Search bans...")
        c_strip.html(ui.bans(banned, names, assets))
        c_act.button("Swap sides", key="swap", icon=":material/swap_horiz:", on_click=_swap_sides,
                     args=(ss,), width="stretch")
        c_act.button("Clear draft", key="clear", icon=":material/restart_alt:", on_click=_clear_draft,
                     args=(ss,), width="stretch", type="tertiary")

    st.space("medium")
    _recommendations(st, ui, model, radiant, dire, banned, rank, p_rad, names, assets)

    st.space("medium")
    c_heroes, c_pairs = st.columns(2, gap="large")
    with c_heroes:
        st.html(ui.section("Heroes in this draft", "Train-split win rate with a Wilson 95% interval."))
        stats = draft_hero_stats(art, radiant, dire)
        st.html(ui.hero_stats_table(stats, names, assets) if stats
                else ui.note("Pick heroes to see their individual win rates."))
    with c_pairs:
        st.html(ui.section("Strongest pair effects", "Synergy within a team, counters across teams."))
        pairs = draft_pairs(art, radiant, dire)
        st.html(ui.pairs_table(pairs, names, assets) if pairs
                else ui.note("Needs at least two heroes; pairs with too few games are omitted."))

    with st.expander("How to read this", icon=":material/info:"):
        st.markdown(METHOD_MD)
    st.html(ui.footer())


if __name__ == "__main__":
    main()
