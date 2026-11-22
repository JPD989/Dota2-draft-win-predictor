# Architecture & Module Contract

This document specifies the codebase: module interfaces, the statistical formulas, the
leakage rules and the output schemas. Every module is written against these interfaces, and
the tests pin the behaviour described here.

Project goal: a rigorous *benchmark* of draft-only win prediction, with statistical rigor,
a fair model comparison, calibration analysis, a patch-drift study and clean engineering.

## 1. Repository layout

```
├── README.md                     # positioning, results, quickstart, methodology
├── Makefile                      # see §8
├── pyproject.toml                # package `src`, pip install -e .[dev]  (already written)
├── app/
│   └── streamlit_app.py          # demo (draft picker + recommender)
├── data/
│   ├── raw/                      # parquet shards of the benchmark patch (never committed)
│   ├── patch-<id>/               # a later patch, read only by the drift study (never committed)
│   └── artifacts/                # committable small artifacts (see §4.17 build_artifacts.py)
├── docs/
│   ├── architecture.md           # this file
│   ├── report.md                 # plain-language report of the method and findings
│   └── figures/                  # README and report figures (make figures)
├── models/                       # trained checkpoints (gitignored, except models/release/)
├── notebooks/
│   └── eda.ipynb                 # exploratory analysis (uses src/ modules; no formula logic inline)
├── results/                      # harness outputs (gitignored)
├── scripts/                      # CLI entry points (thin argparse wrappers around src/)
│   ├── fetch_hero_constants.py
│   ├── build_artifacts.py
│   ├── train_baselines.py
│   ├── train_transformer.py
│   ├── train_gnn.py
│   ├── run_benchmark.py
│   ├── calibration_analysis.py
│   ├── patch_drift.py
│   └── make_report_figures.py
├── src/
│   ├── config.py
│   ├── data/                     # collect_matches.py + validate.py (existing), dataset.py, heroes.py
│   ├── features/encoding.py
│   ├── stats/                    # wilson.py, synergy.py, tests.py
│   ├── models/                   # base.py, baselines.py, transformer.py, gnn.py; __init__.py exports MODEL_REGISTRY
│   ├── eval/                     # metrics.py, calibration.py, harness.py, drift.py
│   ├── recommend/recommender.py
│   └── utils/tracking.py         # optional-W&B wrapper
└── tests/                        # pytest; synthetic fixtures only (no data/raw dependency)
```

Python 3.12, type hints everywhere, NumPy-style docstrings. Imports look like
`from src.stats.synergy import synergy_matrix`.

## 2. Data schema (observed, fixed)

`data/raw/**/matches_*.parquet`, 1,007,319 rows, all patch `7.41`:

| column        | dtype            | notes                                   |
|---------------|------------------|-----------------------------------------|
| match_id      | int64            | unique                                  |
| start_time    | int64            | unix seconds                            |
| duration      | int64            | ≥ 900                                   |
| avg_rank_tier | int64            | 11–75 (tens digit = medal)              |
| radiant_win   | bool             | mean ≈ 0.531 (side bias is real)        |
| radiant_team  | list<int64> (5)  | numpy array after pandas read           |
| dire_team     | list<int64> (5)  | numpy array after pandas read           |
| patch         | string           | e.g. "7.41"                             |

127 distinct hero ids, sparse in [1, ~150]. Never index arrays by raw hero id — always go
through `HeroVocab`.

**Second patch (drift study):** a later patch is collected into its own directory outside
`data/raw`, e.g. `python src/data/collect_matches.py --out data/patch-742 --source explorer
--patch 7.42`. It is kept out of `data/raw` on purpose: splits are chronological, so newer
matches in the same directory would silently become the benchmark's test set. Only
`scripts/patch_drift.py --extra-data` reads it. `load_matches` globs recursively and dedups
by match_id.

## 3. Global conventions

- `SEED = 42` everywhere; every stochastic function takes `seed: int = 42` or a `Generator`.
- Predictions are always **P(radiant win)**, shape `(N,)`, float64 in [0, 1].
- Team encoding order is always **radiant first**: index matrices are `(N, 10)` with columns
  0–4 radiant, 5–9 dire, values = vocab indices; `-1` = empty slot (partial drafts).
- Rank feature: `avg_rank_tier / 80.0` (fixed constant — no fitted scaler, no leakage).
- Leakage rules (hard):
  - Any statistic used as a model input (synergy/counter matrices for the GNN graph,
    win-rate priors, artifacts) is computed on the **train split only**.
  - Splits are chronological; no shuffling before splitting.
  - Augmentation (mirror, pick_dropout) applies to **train only** — never val or test.
  - Recalibration (Platt/isotonic) is fit on **val**, applied to test.
- **Canonical benchmark policy (fairness):** the headline table compares all four models with
  the *same* augmentation policy: `mirror_augment=False`, `pick_dropout=0.0`, `use_rank=False`.
  Mirror, pick-dropout, and rank variants are reported as ablations (via `--tag`), never mixed
  into the headline rows. Constructor defaults equal the canonical policy.
- **Mirror semantics:** `mirror_augment=True` doubles the train set with (swapped teams,
  flipped label), which makes the objective side-symmetric. To keep the 53.1% side advantage
  learnable, every model trained with mirroring adds a fixed logit offset
  `delta = logit(mean(train.radiant_win))` (computed on the un-mirrored train split, stored in
  the checkpoint) to its output logit at predict time. Without mirroring, delta = 0 (models
  learn the side bias from data).
- ECE has exactly one implementation: `src.eval.calibration.expected_calibration_error`;
  `metrics.py` imports it (no cycle: calibration.py imports nothing from metrics.py).
- Every function/script that writes output calls `out_dir.mkdir(parents=True, exist_ok=True)`
  first. `config.py` defines Path constants but creates nothing.
- All scripts must run without W&B installed/configured via `src/utils/tracking.py`.
- Plots: matplotlib only, saved as PNG, no seaborn.

## 4. Module contracts

### 4.1 `src/config.py`

```python
PROJECT_ROOT: Path      # resolved from this file's location (parents[1])
RAW_DIR: Path           # data/raw
ARTIFACTS_DIR: Path     # data/artifacts
MODELS_DIR: Path        # models
RESULTS_DIR: Path       # results
FIGURES_DIR: Path       # docs/figures
SEED: int = 42
RANK_NORM: float = 80.0
MIN_PAIR_GAMES: int = 100      # hard floor for synergy/counter cells (total across orientations)
```

### 4.2 `src/data/dataset.py`

```python
class HeroVocab:
    """Bidirectional hero_id <-> contiguous index [0, V) mapping. Sorted by hero_id."""
    def __init__(self, hero_ids: Sequence[int]) -> None: ...
    @property
    def size(self) -> int: ...
    @property
    def hero_ids(self) -> list[int]: ...           # sorted
    def idx(self, hero_id: int) -> int: ...        # KeyError on unknown
    def hero_id(self, idx: int) -> int: ...
    def save(self, path: Path) -> None: ...        # JSON {"hero_ids": [...]}
    @classmethod
    def load(cls, path: Path) -> "HeroVocab": ...
    @classmethod
    def from_matches(cls, df: pd.DataFrame) -> "HeroVocab": ...  # union of both team columns

def load_matches(raw_dir: Path = RAW_DIR, *, sort_by_time: bool = True) -> pd.DataFrame:
    """Recursive glob raw_dir/**/matches_*.parquet; concat; drop duplicate match_ids
    (keep first); optional chronological sort. FileNotFoundError with a helpful message
    if no shards found."""

@dataclass(frozen=True)
class Split:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    name: str  # e.g. "chrono-80-10-10" or "patch-7.41->7.42"

def chronological_split(df: pd.DataFrame, val_frac: float = 0.1, test_frac: float = 0.1) -> Split:
    """Sort by start_time, cut into [train | val | test]. Deterministic: same df in,
    byte-identical split out (all consumers rely on this)."""

def patch_split(df: pd.DataFrame, train_patches: Sequence[str], test_patch: str,
                val_frac: float = 0.1) -> Split:
    """Train/val = chronological split of train_patches rows; test = all test_patch rows.
    (For the benchmark path; drift.py builds its own splits, see §4.14.)"""
```

### 4.3 `src/data/heroes.py`

```python
def fetch_hero_names(timeout: int = 30) -> dict[int, str]:
    """GET https://api.opendota.com/api/heroes -> {id: localized_name}."""
def save_hero_names(names: dict[int, str], path: Path = ARTIFACTS_DIR / "heroes.json") -> None: ...
def load_hero_names(path: Path = ARTIFACTS_DIR / "heroes.json") -> dict[int, str]:
    """Load cached names (JSON string keys -> int); FileNotFoundError with a hint to run
    scripts/fetch_hero_constants.py."""
```

### 4.4 `src/features/encoding.py`

```python
def teams_to_matrix(df: pd.DataFrame, vocab: HeroVocab) -> np.ndarray:
    """(N, 10) int32 of vocab indices; cols 0-4 radiant, 5-9 dire.
    Team lists shorter than 5 are right-padded with -1 (empty slot).
    Unknown hero id -> KeyError."""

def one_hot_drafts(teams: np.ndarray, vocab_size: int) -> scipy.sparse.csr_matrix:
    """(N, 2V) float32. Radiant hero v -> column v; dire hero v -> column V + v.
    -1 entries contribute nothing."""

def labels(df: pd.DataFrame) -> np.ndarray:
    """(N,) float64 of radiant_win."""

def rank_feature(df: pd.DataFrame) -> np.ndarray:
    """(N, 1) float32, avg_rank_tier / RANK_NORM."""

def mirror(teams: np.ndarray, y: np.ndarray, extra: np.ndarray | None = None
           ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate original + (blocks swapped, labels flipped). Doubles N. If extra is
    given (e.g. rank column, shape (N, k)), it is tiled alongside and returned third."""
```

### 4.5 `src/stats/wilson.py`

```python
def wilson_ci(wins: np.ndarray | int, n: np.ndarray | int, z: float = 1.96
              ) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized Wilson score interval (lo, hi). n=0 -> (0.0, 1.0)."""
```

### 4.6 `src/stats/synergy.py`

All formulas in log-odds space. `logit(p) = log(p/(1-p))` with p clipped to
[1e-3, 1 - 1e-3] first. Caveat to carry into the write-up: marginal win rates attenuate
the underlying per-hero logistic coefficients, so "synergy = interaction coefficient" is an
approximation, not an identity.

```python
def hero_win_rates(df: pd.DataFrame, vocab: HeroVocab) -> pd.DataFrame:
    """Per hero, pooled over both sides.
    Columns: hero_id, games, wins, wr, ci_lo, ci_hi (Wilson 95%). Sorted by hero_id."""

@dataclass(frozen=True)
class PairStats:
    value: np.ndarray    # (V, V) float64 EB-shrunk score, log-odds units; 0 where n < min_games
    raw: np.ndarray      # (V, V) float64 unshrunk score; NaN where n < min_games
    games: np.ndarray    # (V, V) int64 pair sample sizes (total over orientations)
    kind: str            # "synergy" | "counter"
    def save(self, path: Path) -> None: ...      # np.savez_compressed: value, raw, games + kind
    @classmethod
    def load(cls, path: Path) -> "PairStats": ...
```

**Synergy** (same team, pooled over both sides; symmetric, zero diagonal):

```
raw[a,b] = logit(pair_wr_ab) - logit(wr_a) - logit(wr_b)
```

(hero WRs pooled over both sides → base rate 0.5 → logit(base) = 0 cancels).

**Counter** (opposite teams; antisymmetric by construction):

Compute each orientation separately, then average — this cancels the radiant side term
exactly in the additive-logit model:

```
o1 = logit(P(a's team wins | a radiant, b dire))     # from n1 games
o2 = logit(P(a's team wins | a dire, b radiant))     # from n2 games
raw[a,b] = (o1 + o2) / 2 - (logit(wr_a) - logit(wr_b))
```

Cell is valid when `n1 >= min_games/2 and n2 >= min_games/2`; `games[a,b] = n1 + n2`.
Positive value ⇒ a counters b. `raw[b,a] = -raw[a,b]` must hold exactly.

**Empirical-Bayes shrinkage** (identical recipe for both kinds, zero-mean prior):

- `se2[a,b]`: synergy `1/(n·p̂·(1-p̂))` with p̂ = clipped pair WR; counter
  `(1/(n1·p̂1·(1-p̂1)) + 1/(n2·p̂2·(1-p̂2))) / 4` (variance of the orientation average).
- Over valid cells only, **each unordered pair counted once**, with weights `w = 1/se2`:
  `tau2 = max( Σ(w·raw²)/Σw − Σ(w·se2)/Σw , 0 )`  (second moment about 0, matching the
  shrink-to-zero prior).
- `value = raw · tau2 / (tau2 + se2)`; invalid cells get value 0 exactly.
- A hand-computed 3-hero fixture pins this recipe in tests.

```python
def synergy_matrix(df, vocab, *, min_games: int = MIN_PAIR_GAMES) -> PairStats: ...
def counter_matrix(df, vocab, *, min_games: int = MIN_PAIR_GAMES) -> PairStats: ...
def radiant_side_advantage(df) -> tuple[float, float, float]:
    """(radiant_wr, ci_lo, ci_hi) — global side bias with Wilson CI."""
```

### 4.6b `src/stats/tests.py`

```python
def two_proportion_ztest(w1: int, n1: int, w2: int, n2: int) -> tuple[float, float]:
    """(z, p_value), two-sided, pooled variance."""

def hero_wr_shift_table(df_a, df_b, vocab, *, label_a: str, label_b: str,
                        min_games: int = 200, alpha: float = 0.05) -> pd.DataFrame:
    """Filter FIRST (heroes with >= min_games in BOTH A and B), then run the z-tests and
    Benjamini-Hochberg FDR over exactly that set. Columns include wr_a, wr_b, delta, z,
    p_value, q_value, m_tests, significant. Sorted by |delta| desc."""
```

### 4.7 `src/models/base.py`

```python
class DraftModel(abc.ABC):
    """Common interface the harness, recommender and app rely on."""
    name: str            # "logreg" | "xgboost" | "transformer" | "gnn"
    use_rank: bool       # set in __init__; consumers may inspect it
    vocab: HeroVocab | None  # set by fit() AND by load(); models are authoritative for
                             # their own encoding — predict never takes an external vocab

    @property
    def is_fitted(self) -> bool: ...   # True after fit() or load()

    @abc.abstractmethod
    def fit(self, train: pd.DataFrame, val: pd.DataFrame, vocab: HeroVocab) -> "DraftModel":
        """Train on train, use val for early stopping / tuning. Returns self."""

    @abc.abstractmethod
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """(N,) P(radiant win). Accepts partial drafts (§6). Encodes internally with
        self.vocab. If use_rank and 'avg_rank_tier' missing -> impute 50 silently."""

    @abc.abstractmethod
    def save(self, path: Path) -> None:
        """MUST serialize self.vocab (sorted hero_ids) and the mirror delta inside the
        checkpoint so load() is self-contained."""
    @classmethod
    @abc.abstractmethod
    def load(cls, path: Path) -> "DraftModel": ...
```

**Checkpoint & registry conventions** (`src/models/__init__.py`):

```python
MODEL_REGISTRY: dict[str, tuple[str, str]] = {   # name -> (module, class); import stays lazy
    "logreg": ("src.models.baselines", "LogisticDraftModel"),
    "xgboost": ("src.models.baselines", "XGBoostDraftModel"),
    "transformer": ("src.models.transformer", "TransformerDraftModel"),
    "gnn": ("src.models.gnn", "GNNDraftModel"),
}
def get_model_class(name: str) -> type[DraftModel]:
    """importlib-resolve MODEL_REGISTRY[name]; KeyError on unknown name."""
def checkpoint_path(name: str, model_dir: Path = MODELS_DIR, tag: str | None = None) -> Path:
    """models/{name}[_{tag}].pkl for logreg/xgboost, .pt for transformer/gnn."""
```

App and scripts resolve checkpoints exclusively through `get_model_class` +
`checkpoint_path`. `src/models/__init__.py` must not import torch (or the model modules)
at module load — baseline-only environments (Streamlit Cloud) must work.

### 4.8 `src/models/baselines.py`

```python
class LogisticDraftModel(DraftModel):
    def __init__(self, C: float = 1.0, use_rank: bool = False,
                 mirror_augment: bool = False, seed: int = SEED): ...
    # sklearn LogisticRegression(solver="lbfgs", max_iter=2000) on one_hot_drafts (+ rank col)

class XGBoostDraftModel(DraftModel):
    def __init__(self, n_estimators: int = 400, max_depth: int = 8, learning_rate: float = 0.1,
                 use_rank: bool = False, mirror_augment: bool = False, seed: int = SEED,
                 **xgb_kwargs): ...
    # xgboost.XGBClassifier on the same one-hot features; early stopping on val logloss
```

Save = pickle of the wrapper (vocab + delta included), path suffix `.pkl`.

### 4.9 `src/models/transformer.py`

Token layout (drafts are sets — no pick order in the data):

- Embedding table size `V + 2`: `0..V-1` heroes, `V` = MASK (empty slot), `V+1` = CLS.
- Input: `[CLS, r1..r5, d1..d5]` (11 tokens) + team embedding (CLS=0, radiant=1, dire=2).
  No positional encoding within teams.
- Head: CLS vector (+ optional rank scalar) → LayerNorm → MLP → 1 logit (+ mirror delta).

```python
class DraftTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 2, dim_ff: int = 128, dropout: float = 0.1,
                 use_rank: bool = False): ...
    def forward(self, tokens: Tensor, team_ids: Tensor, rank: Tensor | None = None) -> Tensor:
        """tokens (B, 11) long, team_ids (B, 11) long, rank (B, 1) float -> (B,) logits."""

class TransformerDraftModel(DraftModel):
    def __init__(self, *, d_model=64, n_heads=4, n_layers=2, dim_ff=128, dropout=0.1,
                 use_rank=False, mirror_augment=False, pick_dropout: float = 0.0,
                 lr=3e-4, weight_decay=1e-5, batch_size=1024, max_epochs=30,
                 patience=4, seed=SEED, device: str | None = None,
                 tracker: Tracker | None = None): ...
```

`pick_dropout`: during training only, each hero token independently → MASK with prob p
(makes partial-draft inference in-distribution; canonical benchmark uses 0.0, the
recommender-oriented checkpoint uses ~0.15 under a `--tag rec` run). Early stopping on
**un-augmented** val log-loss; best-epoch weights restored. Save: single `.pt` with
state_dict + full config + vocab hero_ids + delta; `load()` is self-contained.

### 4.10 `src/models/gnn.py`

Pure PyTorch (no torch-geometric — ~127 nodes, dense ops are fine).

```python
@dataclass
class HeroGraph:
    a_syn: torch.Tensor   # (V, V) float32
    a_cnt: torch.Tensor   # (V, V) float32

def build_hero_graph(train_df: pd.DataFrame, vocab: HeroVocab) -> HeroGraph:
    """From src.stats matrices on the TRAIN SPLIT ONLY. Normalization (pinned):
    A_norm = A / np.maximum(np.abs(A).sum(axis=1, keepdims=True), 1e-6)
    — L1-of-absolute row normalization: signs preserved, all-zero rows stay zero,
    diagonal forced to 0 afterwards. Both matrices use PairStats.value (EB-shrunk)."""

class DraftGNN(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 64, n_layers: int = 2,
                 dropout: float = 0.1, use_rank: bool = False): ...
    # layer: H' = act(H @ W_self + A_syn @ H @ W_syn + A_cnt @ H @ W_cnt); residual + LayerNorm
    # per match: gather picked heroes' node vectors, masked-mean per team (empty slots
    # excluded; all-empty team -> zeros) ->
    # MLP([r_mean; d_mean; r_mean - d_mean] (+ rank)) -> logit (+ mirror delta)

class GNNDraftModel(DraftModel):
    # same trainer options as TransformerDraftModel (mirror_augment=False, pick_dropout=0.0
    # defaults); checkpoint stores the HeroGraph tensors + vocab + config + delta
```

### 4.11 `src/eval/metrics.py`

```python
def compute_metrics(y, p) -> dict[str, float]:
    """{"auc", "log_loss", "brier", "ece", "ece_q", "n"}. ece = 15 equal-width bins,
    ece_q = 15 equal-mass bins; both delegate to src.eval.calibration."""

def bootstrap_auc_ci(y, p, n_boot: int = 1000, seed: int = SEED) -> tuple[float, float]:
    """Percentile 95% CI on AUC by row resampling."""

def paired_bootstrap_delta_auc(y, p_a, p_b, n_boot: int = 1000, seed: int = SEED
                               ) -> tuple[float, float, float]:
    """(delta, lo, hi) for AUC(a) - AUC(b): ONE index resample per replicate applied to
    both models — the correlated comparison the benchmark's claims rest on."""
```

### 4.12 `src/eval/calibration.py`

```python
def reliability_curve(y, p, n_bins: int = 15, strategy: str = "uniform"
                      ) -> pd.DataFrame:  # bin_mid, p_mean, y_rate, count, wilson lo/hi; strategy in {"uniform","quantile"}
def expected_calibration_error(y, p, n_bins: int = 15, strategy: str = "uniform") -> float
def fit_platt(y_val, p_val) -> Callable[[np.ndarray], np.ndarray]
def fit_isotonic(y_val, p_val) -> Callable[[np.ndarray], np.ndarray]
def plot_reliability(curves: dict[str, pd.DataFrame], out_path: Path) -> None
```

Both calibrator functions clip outputs to [1e-6, 1 - 1e-6] (isotonic hits exact 0/1 at its
range ends otherwise ⇒ infinite log-loss) and handle out-of-range inputs by clip
extrapolation. Test pins: recalibrated log-loss finite even when test predictions are more
extreme than anything in val.

### 4.13 `src/eval/harness.py`

```python
@dataclass
class BenchmarkResult:
    table: pd.DataFrame          # rows = models; cols: auc, auc_lo, auc_hi, delta_auc_vs_logreg,
                                 # delta_lo, delta_hi, log_loss, brier, ece, ece_q, n_test, fit_seconds
    predictions: dict[str, np.ndarray]

def run_benchmark(models: Sequence[DraftModel], split: Split, vocab: HeroVocab,
                  out_dir: Path, *, recalibrate: bool = False, seed: int = SEED
                  ) -> BenchmarkResult:
```

Behavior (pinned):
- For each model: `if not model.is_fitted: model.fit(train, val, vocab)` (fit_seconds = NaN
  for pre-fitted models). Evaluate ALL on the identical test set.
- Paired bootstrap delta-AUC vs the `"logreg"` row when present (shared resample indices);
  delta columns NaN if no logreg among the models.
- `recalibrate=True`: isotonic fit on val predictions per model; calibrated results appear
  as additional entries named `"{name}+isotonic"` (extra rows/keys/columns everywhere).
- Outputs in out_dir:
  - `metrics.json`: `{"split": name, "seed": s, "n_train/n_val/n_test": ints,
    "models": {name: {auc, auc_lo, auc_hi, delta_auc_vs_logreg, delta_lo, delta_hi,
    log_loss, brier, ece, ece_q, n, fit_seconds}}}`
  - `predictions.parquet`: columns `match_id` (int64), `y` (float64), then one float64
    column per model name.
  - `table.md` (sorted by AUC desc) and `reliability.png`.

The harness never touches test data during fitting — it is the single place the fairness
guarantee lives; keep it boring and auditable.

### 4.14 `src/eval/drift.py`

```python
def time_drift_study(df, model_factories: dict[str, Callable[[], DraftModel]], vocab,
                     n_windows: int = 5, out_dir: Path = RESULTS_DIR / "drift") -> pd.DataFrame:
    """Single-patch proxy. Chronological: train = first 54%, val = next 6% (both inside
    the front 60%), then n_windows equal consecutive windows over the remaining 40%.
    Returns one row per (model, window): auc, auc_lo, auc_hi (bootstrap), log_loss,
    window_start/end (dates), days_since_train_end. Prints an honesty warning when the
    evaluated span is < 14 days (drift may be indistinguishable from noise)."""

def patch_drift_study(df, model_factories, vocab, *, train_patch: str, test_patch: str,
                      out_dir: Path = RESULTS_DIR / "drift") -> pd.DataFrame:
    """Builds its own splits (NOT patch_split): chronological_split(df[patch==train_patch])
    gives train/val/test-within; df[patch==test_patch] is the drift test set.
    Degradation = metric(next patch) - metric(within-patch test). Raises with a clear
    message if either patch has < 10k rows."""
```

### 4.15 `src/recommend/recommender.py`

```python
def legal_picks(radiant, dire, banned, vocab) -> list[int]:
    """All hero ids not already picked or banned."""

def recommend(model: DraftModel, vocab: HeroVocab, *, radiant: Sequence[int],
              dire: Sequence[int], side: Literal["radiant", "dire"],
              banned: Sequence[int] = (), avg_rank_tier: int | None = None,
              top_k: int | None = None) -> pd.DataFrame:
    """For each legal hero h: add h to `side`, build the batch DataFrame (always includes
    an avg_rank_tier column — default 50 when None), one model.predict_proba call on the
    whole batch, return (hero_id, win_prob) sorted desc by the PICKING side's win prob."""
```

Partial-draft caveat (documented in the docstring + README): only masking-trained neural
models (transformer/GNN with pick_dropout > 0) see partial drafts in-distribution; without
masking they are overconfident on unfinished drafts, and the app labels them "experimental"
for mid-draft use. The one-hot models (logreg, XGBoost) stay calibrated, since a missing
hero contributes nothing (measured in docs/report.md). Sanity test: all-empty draft ⇒ prediction ≈ train base rate for every model.

### 4.16 `src/utils/tracking.py`

```python
class Tracker:
    """No-op unless wandb importable AND enabled. API: log(dict, step=None), finish(),
    config property. Never raises because of wandb problems."""
def init_tracker(project: str, config: dict, enabled: bool) -> Tracker
```

### 4.17 Scripts (thin argparse wrappers; every one supports `--help` cleanly)

| script                      | contract                                                            |
|-----------------------------|---------------------------------------------------------------------|
| fetch_hero_constants.py     | `--out data/artifacts` → heroes.json                                |
| build_artifacts.py          | `--data data/raw --out data/artifacts [--val-frac .1 --test-frac .1]`. Loads matches, takes `chronological_split(...).train` ONLY, writes: `hero_vocab.json`, `synergy.npz`, `counter.npz` (PairStats.save), `hero_win_rates.parquet`, `meta.json` (train rows, radiant_wr_train, split name, date). These are small and committable — the app runs off them without data/raw. |
| train_baselines.py          | `--data --sample N --use-rank --mirror --tag T --out models/ --results results/baselines` — fits logreg + xgboost, saves via checkpoint_path(name, tag) |
| train_transformer.py        | hparams + `--epochs --sample --pick-dropout --mirror --use-rank --tag T --wandb/--no-wandb --out models/` |
| train_gnn.py                | same shape as train_transformer.py                                  |
| run_benchmark.py            | `--models logreg xgboost transformer gnn --model-dir models/ --tag T --checkpoint NAME=PATH ... --results results/benchmark`. Baselines: always fit fresh in-process. Torch models: load-only — if the checkpoint is missing, print a skip message and continue with the rest (never stack-trace, never retrain). |
| calibration_analysis.py     | `--model-dir --tag --results results/calibration`. Loads models via registry, rebuilds the deterministic default `chronological_split(load_matches())`, predicts on val to fit Platt+isotonic, evaluates on test; writes reliability.png + calibration_table.md (before/after ECE, log-loss, Brier). Split is byte-identical to the harness's because splitting is deterministic. |
| patch_drift.py              | `--mode time|patch --train-patch --test-patch --results results/drift` |

Scripts print a clear message and exit 0 when prerequisites are missing (no data, no
checkpoints); they must not stack-trace in the demo path.

### 4.18 `app/streamlit_app.py`

- Loads from `data/artifacts/`: heroes.json, hero_vocab.json, synergy.npz, counter.npz,
  hero_win_rates.parquet, meta.json. Missing artifacts ⇒ friendly instructions screen
  (which make targets to run), not a crash.
- Hero multiselects (radiant, dire, bans) with names from heroes.json; side selector; rank
  slider; model picker scanning `models/` + `models/release/` via MODEL_REGISTRY /
  checkpoint_path (torch checkpoints only listed if torch import succeeds).
- **Stats mode** (default when no checkpoint exists — the app must demo with zero
  training). Pinned scoring formula, from artifacts only:
  `score = logit(radiant_wr_train) + Σ_{a ∈ R} logit(wr_a) − Σ_{b ∈ D} logit(wr_b) + Σ_{a<b ∈ R} syn[a,b] − Σ_{a<b ∈ D} syn[a,b]
  + Σ_{a ∈ R, b ∈ D} cnt[a,b]`, `p = sigmoid(score)`; shown with a visible
  "heuristic stats model — train a model for real predictions" banner. The same formula
  backs a `StatsDraftModel` helper inside the app file (implements predict_proba over
  artifacts so the recommender works in stats mode).
- Shows: current P(radiant win), top-10 recommendations for the picking side (via
  `src.recommend.recommend`), a calibration disclaimer, and the experimental-model warning
  from §4.15 when applicable.
- **Hosting (optional)**: the app can be hosted on Streamlit Community Cloud, which deploys from GitHub. Committed artifacts
  (data/artifacts/, few hundred KB) power stats mode; a small released checkpoint may be
  committed under `models/release/` (`.gitignore` carve-out) after training is confirmed.
  requirements: the app path must not import torch unless a torch checkpoint is selected.

### 4.19 `notebooks/eda.ipynb`

All statistics come from `src.stats` imports — no formula logic inline. Pinned choices:
1. Synergy = log-odds + EB shrinkage (§4.6).
2. Counter = orientation-averaged logit formula (§4.6), which cancels the side bias.
3. Patch-stability section: guarded single-patch message + time-window stability using
   `hero_wr_shift_table` between early/late halves.
4. No `iterrows()` — vectorized explode/groupby only.
Sections: 0 load, 1 overview (+ side bias CI), 2 hero WR + Wilson CI (top/bottom 20 plot),
3 synergy (heatmap + top/bottom-10 pairs table, shrunk values, n shown), 4 counters
(top-10 counter pairs), 5 stability across time (+patch guard), 6 stability across rank
brackets (z-tests + BH), 7 takeaways (markdown).
First cell: `import sys; sys.path.insert(0, "..")` fallback + editable-install note.
Uses ARTIFACTS heroes.json for names (fetch cell guarded by file existence).

## 5. Testing strategy (`tests/`)

- All tests run on **synthetic fixtures** (`conftest.py`): `tiny_vocab` (20 heroes,
  non-contiguous ids), `synthetic_matches(n=2000, seed)` with planted signal: a few hero
  pairs get real synergy, one hero counters another, radiant base rate 0.53. No test reads
  `data/raw`.
- `test_data_pipeline.py`: collect_matches pure functions (parse_team str/list/dupes,
  clean_match filters, get_patch bisection edges, checkpoint round-trip), validate.validate()
  on good/bad frames, load_matches dedup + recursive glob (tmp_path parquet fixtures),
  chronological_split no-overlap + boundary (max train time ≤ min val time), patch_split.
- Formula tests: wilson_ci against known values (8/10 → (0.490, 0.943) ± 1e-2); synergy of
  independent heroes |raw| < 0.05 at n=2000 planted-null; planted pair > 0; counter
  antisymmetry exact; EB pinned on the hand-computed 3-hero fixture; shrinkage pulls low-n
  cells toward 0 more than high-n; PairStats save/load round-trip.
- Model tests (per DraftModel): fits ≤2k synthetic rows in seconds; AUC > 0.55 on planted
  signal; save/load round-trips predictions exactly (incl. vocab + delta); partial-draft
  predict works; all-empty draft ≈ train base rate (±0.05); mirror_augment=True keeps mean
  prediction near base rate (delta mechanism works).
- Torch tests: forward shapes, gradients flow, one optimizer step decreases batch loss;
  build_hero_graph adjacency finite, signed, zero-diag, zero rows preserved.
- Harness: end-to-end on synthetic with logreg + xgboost writes all four outputs; paired
  delta columns present; recalibrate=True adds "+isotonic" rows with finite log-loss.
- Calibration: isotonic/platt clipping (finite log-loss on extreme inputs).
- Drift: time_drift_study window boundaries strictly after val; returns expected columns.
- Recommender: excludes picked/banned; planted-synergy partner in top-3 for the right side;
  stats-mode formula sanity (app helper).
- Marker `@pytest.mark.slow` for anything > ~5 s; default run < 60 s.

These smoke-fits on tiny synthetic data verify the code; they are not experiments.

## 6. Partial-draft semantics (recommender path)

- DataFrames: `radiant_team`/`dire_team` lists may have < 5 entries.
- Index matrices: empty slots are `-1` (teams_to_matrix pads).
- one-hot: empty slot contributes nothing. transformer: empty → MASK. GNN: excluded from
  team mean. XGBoost: fewer active columns.

## 7. Non-goals / deferred

- No bans/pick-order modeling (publicMatches doesn't provide them).
- No hyperparameter search; ablations are tagged runs (`--tag`).

## 8. Makefile & README contracts

Makefile targets beyond data collection:
`install` (pip install -e ".[dev]"), `test` (pytest -m "not slow"), `test-all`,
`artifacts` (build_artifacts.py), `eda` (jupyter nbconvert --execute), `train-baselines`,
`train-transformer`, `train-gnn`, `benchmark`, `calibrate`, `drift`, `app`
(streamlit run), `figures` (scripts/make_report_figures.py: regenerate the README/report figures).
The old `clean` is renamed `clean-data` and requires `CONFIRM=1` (it deletes the collected matches).

README.md: positioning as a benchmark (not novel research), prior-work citations
(dota2-predictor, Draftnet, academic draft-prediction papers), the ~63–65% AUC information
ceiling framing, quickstart (make targets), repo map, methodology summary (splits, leakage
rules, calibration, drift), a results section sourced from `results/benchmark/table.md`,
and limitations.
