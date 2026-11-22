"""
Validation script for collected match data.

Checks:
  - No duplicate match_ids across shards
  - All hero IDs are valid (fetched from OpenDota constants)
  - radiant_team and dire_team each have exactly 5 unique heroes
  - No hero overlap between the two teams
  - duration >= 900s
  - avg_rank_tier is non-null and positive
  - radiant_win is boolean
  - patch column is populated

Usage:
    python src/data/validate.py --data data/raw
    make validate
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import requests


def fetch_valid_hero_ids() -> set[int]:
    r = requests.get("https://api.opendota.com/api/heroes", timeout=30)
    r.raise_for_status()
    return {h["id"] for h in r.json()}


def load_all_shards(data_dir: Path) -> pd.DataFrame:
    shards = sorted(data_dir.glob("matches_*.parquet"))
    if not shards:
        print(f"ERROR: no parquet shards found in {data_dir}")
        sys.exit(1)
    print(f"loading {len(shards)} shard(s)...")
    return pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)


def validate(df: pd.DataFrame, valid_heroes: set[int]) -> list[str]:
    errors = []
    n = len(df)

    # duplicate match_ids
    dupes = df["match_id"].duplicated().sum()
    if dupes:
        errors.append(f"duplicate match_ids: {dupes:,}")

    # null checks
    for col in ["match_id", "start_time", "duration", "avg_rank_tier", "radiant_win", "patch"]:
        nulls = df[col].isna().sum()
        if nulls:
            errors.append(f"null values in '{col}': {nulls:,}")

    # duration
    short = (df["duration"] < 900).sum()
    if short:
        errors.append(f"matches with duration < 900s: {short:,}")

    # rank tier
    bad_rank = (df["avg_rank_tier"] <= 0).sum()
    if bad_rank:
        errors.append(f"invalid avg_rank_tier (<= 0): {bad_rank:,}")

    # team validation per row
    bad_teams = 0
    bad_heroes = 0
    for _, row in df.iterrows():
        r_team = row["radiant_team"]
        d_team = row["dire_team"]

        try:
            r_team = list(r_team)
            d_team = list(d_team)
        except TypeError:
            bad_teams += 1
            continue
        if len(r_team) != 5 or len(d_team) != 5:
            bad_teams += 1
            continue
        picks = r_team + d_team
        if len(set(picks)) != 10:
            bad_teams += 1
            continue
        if any(h not in valid_heroes for h in picks):
            bad_heroes += 1

    if bad_teams:
        errors.append(f"rows with malformed teams: {bad_teams:,}")
    if bad_heroes:
        errors.append(f"rows with unknown hero IDs: {bad_heroes:,}")

    return errors


def print_summary(df: pd.DataFrame):
    print(f"\n--- summary ---")
    print(f"total matches : {len(df):,}")
    print(f"patches       : {sorted(df['patch'].unique())}")
    print(f"radiant win % : {df['radiant_win'].mean():.3f}")
    print(f"duration mean : {df['duration'].mean():.0f}s  "
          f"min={df['duration'].min()}s  max={df['duration'].max()}s")
    print(f"rank tier dist:")
    print(df["avg_rank_tier"].describe().to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/raw")
    ap.add_argument("--no-hero-fetch", action="store_true",
                    help="skip fetching hero constants (faster, skips hero ID checks)")
    args = ap.parse_args()

    data_dir = Path(args.data)
    df = load_all_shards(data_dir)

    if args.no_hero_fetch:
        valid_heroes = set(range(1, 300))  # permissive fallback
        print("skipping hero fetch, using permissive ID range 1-299")
    else:
        print("fetching hero constants...")
        valid_heroes = fetch_valid_hero_ids()
        print(f"  {len(valid_heroes)} heroes known")

    print(f"validating {len(df):,} matches...")
    errors = validate(df, valid_heroes)

    if errors:
        print("\nERRORS:")
        for e in errors:
            print(f"  ✗ {e}")
        print_summary(df)
        sys.exit(1)
    else:
        print("all checks passed")
        print_summary(df)
        sys.exit(0)


if __name__ == "__main__":
    main()
