"""Hero id -> localized name mapping, fetched once from OpenDota and cached as JSON.

The cached file lives at ``data/artifacts/heroes.json`` (committable) so the
EDA notebook and the Streamlit app never need network access at run time.
"""

import json
from pathlib import Path

import requests

from src.config import ARTIFACTS_DIR

OPENDOTA_HEROES_URL = "https://api.opendota.com/api/heroes"

_DEFAULT_PATH = ARTIFACTS_DIR / "heroes.json"


def fetch_hero_names(timeout: int = 30) -> dict[int, str]:
    """Fetch the hero id -> localized name mapping from the OpenDota API.

    Parameters
    ----------
    timeout : int, optional
        Request timeout in seconds (default 30).

    Returns
    -------
    dict[int, str]
        Mapping ``{hero_id: localized_name}``.

    Raises
    ------
    requests.HTTPError
        If the API responds with an error status.
    """
    resp = requests.get(OPENDOTA_HEROES_URL, timeout=timeout)
    resp.raise_for_status()
    return {int(h["id"]): str(h["localized_name"]) for h in resp.json()}


def save_hero_names(names: dict[int, str], path: Path = _DEFAULT_PATH) -> None:
    """Write hero names to ``path`` as JSON with string keys, sorted by hero id.

    Parameters
    ----------
    names : dict[int, str]
        Mapping ``{hero_id: localized_name}``.
    path : Path, optional
        Output file (default ``data/artifacts/heroes.json``). Parent
        directories are created as needed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump({str(k): names[k] for k in sorted(names)}, f, ensure_ascii=False, indent=2)


def load_hero_names(path: Path = _DEFAULT_PATH) -> dict[int, str]:
    """Load the cached hero names, converting JSON string keys back to ints.

    Parameters
    ----------
    path : Path, optional
        Cache file written by :func:`save_hero_names`
        (default ``data/artifacts/heroes.json``).

    Returns
    -------
    dict[int, str]
        Mapping ``{hero_id: localized_name}``.

    Raises
    ------
    FileNotFoundError
        If the cache file does not exist, with a hint to run
        ``scripts/fetch_hero_constants.py``.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Hero names cache not found at {path}. "
            "Run `python scripts/fetch_hero_constants.py` to download it from OpenDota."
        )
    with path.open("r", encoding="utf-8") as f:
        return {int(k): str(v) for k, v in json.load(f).items()}
