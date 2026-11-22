"""Cache hero constants into ``data/artifacts/``: names (heroes.json) and display assets.

- ``heroes.json``: ``{id: localized_name}`` from the OpenDota API, falling back to
  odota/dotaconstants on GitHub when the API is unreachable.
- ``hero_assets.json``: ``{id: {name, localized_name, img, icon, primary_attr,
  attack_type, roles}}`` from odota/dotaconstants, with absolute Steam CDN image
  URLs. The Streamlit app reads it offline; only the browser fetches the images.

Usage
-----
    python scripts/fetch_hero_constants.py [--out data/artifacts] [--assets-only]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # fallback if not pip-installed

from src.config import ARTIFACTS_DIR  # noqa: E402

DOTACONSTANTS_URL = "https://raw.githubusercontent.com/odota/dotaconstants/master/build/heroes.json"
STEAM_CDN = "https://cdn.cloudflare.steamstatic.com"
NPC_PREFIX = "npc_dota_hero_"


def _cdn_url(path: str) -> str:
    """Absolute CDN URL for a dotaconstants image path (drops the trailing ``?``)."""
    path = path.split("?", 1)[0]
    return path if path.startswith("http") else f"{STEAM_CDN}{path}"


def build_hero_assets(constants: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Reduce a dotaconstants ``heroes.json`` payload to the fields the app displays.

    Parameters
    ----------
    constants : dict
        ``{str(id): hero}`` as published by odota/dotaconstants.

    Returns
    -------
    dict
        ``{str(id): {name, localized_name, img, icon, primary_attr, attack_type,
        roles}}`` sorted by id; ``name`` is the short internal name (``"furion"``).
    """
    out: dict[str, dict[str, Any]] = {}
    for key in sorted(constants, key=int):
        h = constants[key]
        out[str(int(key))] = {
            "name": str(h.get("name", "")).removeprefix(NPC_PREFIX),
            "localized_name": str(h.get("localized_name", "")),
            "img": _cdn_url(h["img"]) if h.get("img") else None,
            "icon": _cdn_url(h["icon"]) if h.get("icon") else None,
            "primary_attr": h.get("primary_attr"),
            "attack_type": h.get("attack_type"),
            "roles": list(h.get("roles", [])),
        }
    return out


def fetch_dotaconstants(timeout: int = 30) -> dict[str, dict[str, Any]]:
    """Download the dotaconstants hero table (raises ``requests.RequestException``)."""
    import requests

    resp = requests.get(DOTACONSTANTS_URL, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def main(argv: list[str] | None = None) -> int:
    """Fetch hero names and display assets as JSON. Returns the process exit code."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=ARTIFACTS_DIR,
                    help="output directory (default: data/artifacts)")
    ap.add_argument("--timeout", type=int, default=30, help="HTTP timeout in seconds")
    ap.add_argument("--assets-only", action="store_true",
                    help="only (re)write hero_assets.json; leave heroes.json untouched")
    args = ap.parse_args(argv)

    import requests

    from src.data.heroes import fetch_hero_names, save_hero_names

    args.out.mkdir(parents=True, exist_ok=True)
    constants: dict[str, dict[str, Any]] | None = None
    try:
        constants = fetch_dotaconstants(timeout=args.timeout)
    except requests.RequestException as e:
        print(f"Could not reach dotaconstants ({e}); hero_assets.json not written.")

    if not args.assets_only:
        try:
            names = fetch_hero_names(timeout=args.timeout)
        except requests.RequestException as e:
            if constants is None:
                print(f"Could not reach the OpenDota API ({e}). Check your connection and retry.")
                return 1
            print(f"OpenDota API unreachable ({e}); using dotaconstants names instead.")
            names = {int(k): str(h["localized_name"]) for k, h in constants.items()}
        path = args.out / "heroes.json"
        save_hero_names(names, path)
        print(f"wrote {len(names)} hero names -> {path}")

    if constants is None:
        return 1
    assets = build_hero_assets(constants)
    path = args.out / "hero_assets.json"
    path.write_text(json.dumps(assets, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {len(assets)} hero assets -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
