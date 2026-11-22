"""
OpenDota data collection for the Dota 2 draft win predictor.

Pulls ranked public matches from OpenDota, validates them, and stores them as
parquet shards with resume support. Two sources read the same underlying
``public_matches`` table and yield identical rows:

* ``api``      -- the /publicMatches endpoint, ~100 matches per call.
* ``explorer`` -- the /explorer SQL endpoint, up to 10,000 filtered matches per
  call, paged by match_id window. Far fewer calls for large collections.

Free tier limits: 60 calls/min, 50,000 calls/month (no key needed).

Usage:
    python collect_matches.py --target 200000 --out data/raw
    python collect_matches.py --target 1000000 --out data/raw --source explorer --id-stride 8
    python collect_matches.py --target 100000 --out data/raw/patch-742 --source explorer --patch 7.42

Resume is automatic: the script stores the last seen match_id in
<out>/checkpoint.json and continues from there on the next run.
"""

import argparse
import bisect
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

API_BASE = "https://api.opendota.com/api"
PUBLIC_MATCHES = f"{API_BASE}/publicMatches"
HEROES = f"{API_BASE}/heroes"
PATCHES = f"{API_BASE}/constants/patch"
EXPLORER = f"{API_BASE}/explorer"

EXPLORER_LIMIT = 10_000     # rows per explorer call
EXPLORER_ID_WINDOW = 60_000  # match_id span scanned per call (keeps the query on the pkey index)
EXPLORER_SLEEP = 2.0        # seconds between explorer calls
CONSTANTS_CACHE = Path(__file__).resolve().parents[2] / "data" / "artifacts"

SLEEP_BETWEEN_CALLS = 1.1   # stay under 60 req/min
SHARD_SIZE = 20_000         # matches per parquet shard
RANKED_LOBBY = 7            # lobby_type 7 = ranked matchmaking
ALL_DRAFT_MODE = 22         # game_mode 22 = All Pick (ranked)
MIN_DURATION_S = 900        # drop likely abandons (< 15 min)

EXPECTED_COLS = [
    "match_id", "start_time", "duration", "avg_rank_tier",
    "radiant_win", "radiant_team", "dire_team", "patch",
]


def parse_team(field) -> list[int] | None:
    """OpenDota returns hero ids as '1,2,3,4,5' (older) or [1,2,3,4,5] (newer)."""
    if field is None:
        return None
    if isinstance(field, str):
        try:
            ids = [int(x) for x in field.split(",")]
        except ValueError:
            return None
    elif isinstance(field, (list, tuple)):
        ids = [int(x) for x in field]
    else:
        return None
    if len(ids) != 5 or len(set(ids)) != 5:
        return None
    return ids


def fetch_with_retry(session: requests.Session, url: str) -> dict | list:
    for attempt in range(8):
        r = session.get(url, timeout=30)
        if r.status_code == 429:
            wait = 60 * (attempt + 1)
            print(f"  rate limited on {url.split('/')[-1]}, sleeping {wait}s")
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"could not fetch {url} after 8 attempts")


def fetch_hero_ids(session: requests.Session) -> set[int]:
    """Valid hero ids from the API, falling back to the cached heroes.json."""
    try:
        return {h["id"] for h in fetch_with_retry(session, HEROES)}
    except (requests.RequestException, RuntimeError) as e:
        cache = CONSTANTS_CACHE / "heroes.json"
        if not cache.exists():
            raise
        print(f"  heroes endpoint unavailable ({type(e).__name__}); using {cache.name}")
        return {int(k) for k in json.loads(cache.read_text())}


def _parse_date(val) -> int:
    """Convert a patch date (ISO string or unix int) to a unix timestamp."""
    if isinstance(val, (int, float)):
        return int(val)
    return int(datetime.fromisoformat(val.replace("Z", "+00:00")).timestamp())


def fetch_patches(session: requests.Session) -> tuple[list[int], list[str]]:
    """Return (sorted timestamps, patch names) for binary search.

    The response is cached in data/artifacts/patches.json and reused when the
    endpoint is unavailable.
    """
    cache = CONSTANTS_CACHE / "patches.json"
    try:
        raw = fetch_with_retry(session, PATCHES)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(raw))
    except (requests.RequestException, RuntimeError) as e:
        if not cache.exists():
            raise
        print(f"  patch endpoint unavailable ({type(e).__name__}); using {cache.name}")
        raw = json.loads(cache.read_text())
    if isinstance(raw, list):
        data = sorted(raw, key=lambda p: _parse_date(p["date"]))
        timestamps = [_parse_date(p["date"]) for p in data]
        names = [str(p["name"]) for p in data]
    else:
        pairs = sorted(raw.items(), key=lambda kv: _parse_date(kv[1]["date"] if isinstance(kv[1], dict) else kv[1]))
        timestamps = [_parse_date(v["date"] if isinstance(v, dict) else v) for _, v in pairs]
        names = [k for k, _ in pairs]
    print(f"  {len(names)} patches loaded ({names[0]} → {names[-1]})")
    return timestamps, names


def get_patch(start_time: int, patch_timestamps: list[int], patch_names: list[str]) -> str:
    """Map a match start_time to the patch active at that time."""
    idx = bisect.bisect_right(patch_timestamps, start_time) - 1
    if idx < 0:
        return patch_names[0]
    return patch_names[idx]


def fetch_page(session: requests.Session, less_than_match_id: int | None):
    params = {}
    if less_than_match_id is not None:
        params["less_than_match_id"] = less_than_match_id
    for attempt in range(5):
        try:
            r = session.get(PUBLIC_MATCHES, params=params, timeout=30)
            if r.status_code == 429:
                wait = 60 * (attempt + 1)
                print(f"  rate limited, sleeping {wait}s")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            wait = 2 ** attempt
            print(f"  request failed ({e}), retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError("5 consecutive failures, aborting")


def explorer_query(session: requests.Session, sql: str) -> list[dict]:
    """Run one read-only SQL query on the OpenDota explorer and return its rows."""
    for attempt in range(6):
        try:
            r = session.get(EXPLORER, params={"sql": sql}, timeout=90)
            if r.status_code == 429:
                time.sleep(60 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json().get("rows") or []
        except requests.RequestException as e:
            wait = 5 * 2 ** attempt
            print(f"  explorer request failed ({type(e).__name__}), retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError("6 consecutive explorer failures, aborting")


def fetch_explorer_page(session: requests.Session, less_than_match_id: int | None
                        ) -> tuple[list[dict], int]:
    """Fetch ranked All Pick matches with match_id in one window below the cursor.

    Returns ``(rows, next_cursor)``. ``next_cursor`` is the smallest id returned
    when the row limit was hit, otherwise the bottom of the scanned window, so
    paging never skips matches.
    """
    if less_than_match_id is None:
        less_than_match_id = int(explorer_query(
            session, "select max(match_id) as m from public_matches")[0]["m"]) + 1
    low = less_than_match_id - EXPLORER_ID_WINDOW
    rows = explorer_query(session, (
        "select match_id, start_time, duration, lobby_type, game_mode, avg_rank_tier, "
        "radiant_win, radiant_team, dire_team from public_matches "
        f"where match_id < {less_than_match_id} and match_id >= {low} "
        f"and lobby_type = {RANKED_LOBBY} and game_mode = {ALL_DRAFT_MODE} "
        f"and duration >= {MIN_DURATION_S} order by match_id desc limit {EXPLORER_LIMIT}"))
    full = len(rows) >= EXPLORER_LIMIT
    return rows, (min(int(m["match_id"]) for m in rows) if full else low)


def clean_match(
    m: dict,
    valid_heroes: set[int],
    min_rank_tier: int,
    patch_timestamps: list[int],
    patch_names: list[str],
) -> dict | None:
    if m.get("lobby_type") != RANKED_LOBBY:
        return None
    if m.get("game_mode") != ALL_DRAFT_MODE:
        return None
    if (m.get("duration") or 0) < MIN_DURATION_S:
        return None
    rank = m.get("avg_rank_tier")
    if rank is None or rank < min_rank_tier:
        return None

    radiant = parse_team(m.get("radiant_team"))
    dire = parse_team(m.get("dire_team"))
    if radiant is None or dire is None:
        return None
    picks = radiant + dire
    if len(set(picks)) != 10 or any(h not in valid_heroes for h in picks):
        return None

    start_time = int(m["start_time"])
    return {
        "match_id": int(m["match_id"]),
        "start_time": start_time,
        "duration": int(m["duration"]),
        "avg_rank_tier": int(rank),
        "radiant_win": bool(m["radiant_win"]),
        "radiant_team": radiant,
        "dire_team": dire,
        "patch": get_patch(start_time, patch_timestamps, patch_names),
    }


def load_checkpoint(out_dir: Path) -> dict:
    cp = out_dir / "checkpoint.json"
    if cp.exists():
        return json.loads(cp.read_text())
    return {"last_match_id": None, "total_kept": 0, "shard_index": 0, "seen_ids": []}


def save_checkpoint(out_dir: Path, state: dict, seen_ids: set[int]):
    # persist seen_ids as a list so we can restore cross-shard dedup on resume
    state["seen_ids"] = list(seen_ids)
    (out_dir / "checkpoint.json").write_text(json.dumps(state))


def write_shard(rows: list[dict], out_dir: Path, shard_index: int):
    df = pd.DataFrame(rows)
    df = df[EXPECTED_COLS]
    path = out_dir / f"matches_{shard_index:04d}.parquet"
    df.to_parquet(path, index=False)
    print(f"  wrote {len(df):,} matches -> {path.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=200_000,
                    help="total matches to collect")
    ap.add_argument("--out", type=str, default="data/raw")
    ap.add_argument("--min-rank-tier", type=int, default=0,
                    help="e.g. 50 = Ancient+, 60 = Divine+, 0 = all ranks")
    ap.add_argument("--source", choices=["api", "explorer"], default="api",
                    help="api = /publicMatches (100/call); explorer = SQL endpoint (10k/call)")
    ap.add_argument("--patch", type=str, default=None,
                    help="keep only this patch (e.g. 7.42) and stop once paging leaves it")
    ap.add_argument("--id-stride", type=int, default=1,
                    help="explorer only: after each page skip (stride-1) id windows, so the "
                         "collection samples a longer time span (1 = contiguous)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers["User-Agent"] = "dota2-draft-predictor"

    print("fetching hero constants...")
    valid_heroes = fetch_hero_ids(session)
    print(f"  {len(valid_heroes)} heroes known")

    print("fetching patch constants...")
    patch_timestamps, patch_names = fetch_patches(session)

    state = load_checkpoint(out_dir)
    seen_ids: set[int] = set(state.pop("seen_ids", []))
    buffer: list[dict] = []
    calls = 0
    empty_pages = 0
    kept_at_start = state["total_kept"]
    t0 = time.time()

    print(f"collecting (target={args.target:,}, already have {state['total_kept']:,})")
    try:
        while state["total_kept"] < args.target:
            if args.source == "explorer":
                page, cursor = fetch_explorer_page(session, state["last_match_id"])
                cursor -= (args.id_stride - 1) * EXPLORER_ID_WINDOW
                empty_pages = 0 if page else empty_pages + 1
                if empty_pages >= 20:
                    print("20 empty id windows in a row, stopping")
                    break
            else:
                page = fetch_page(session, state["last_match_id"])
                if not page:
                    print("empty page, stopping")
                    break
                cursor = min(m["match_id"] for m in page)
            calls += 1
            state["last_match_id"] = cursor

            if args.patch is not None and page:
                newest = max(int(m["start_time"]) for m in page)
                if get_patch(newest, patch_timestamps, patch_names) != args.patch \
                        and newest < patch_timestamps[patch_names.index(args.patch)]:
                    print(f"paged past the start of patch {args.patch}, stopping")
                    break
            kept_before = len(buffer)
            for m in page:
                mid = int(m["match_id"])
                if mid in seen_ids:
                    continue
                row = clean_match(m, valid_heroes, args.min_rank_tier, patch_timestamps, patch_names)
                if row is not None and args.patch is not None and row["patch"] != args.patch:
                    row = None
                if row is not None:
                    buffer.append(row)
                    seen_ids.add(mid)
            state["total_kept"] += len(buffer) - kept_before

            if len(buffer) >= SHARD_SIZE:
                write_shard(buffer, out_dir, state["shard_index"])
                state["shard_index"] += 1
                buffer = []
                save_checkpoint(out_dir, state, seen_ids)

            if calls % (10 if args.source == "explorer" else 50) == 0:
                rate = (state["total_kept"] - kept_at_start) / max(time.time() - t0, 1)
                eta_h = (args.target - state["total_kept"]) / max(rate, 1) / 3600
                print(f"  calls={calls:,} kept={state['total_kept']:,} "
                      f"({rate:.0f} matches/s, ~{eta_h:.1f}h left)")

            time.sleep(EXPLORER_SLEEP if args.source == "explorer" else SLEEP_BETWEEN_CALLS)
    except KeyboardInterrupt:
        print("\ninterrupted, flushing buffer...")
    finally:
        if buffer:
            write_shard(buffer, out_dir, state["shard_index"])
            state["shard_index"] += 1
        save_checkpoint(out_dir, state, seen_ids)
        print(f"done: {state['total_kept']:,} matches in "
              f"{state['shard_index']} shards, {calls:,} API calls this run")

    if calls > 0:
        print(f"monthly budget note: ~{calls:,} calls used this run "
              f"(free tier = 50,000/month)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
