from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DATA_URL = (
    "https://phl.carto.com/api/v2/sql?q=SELECT+*,+ST_Y(the_geom)+AS+lat,"
    "+ST_X(the_geom)+AS+lng+FROM+complaints&filename=complaints&format=csv"
    "&skipfields=cartodb_id"
)


def download_data(force: bool = False) -> pd.DataFrame:
    project_root = Path(__file__).resolve().parents[1]
    raw_path = project_root / "data" / "raw" / "complaints.csv"
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    if raw_path.exists() and not force:
        print(f"Raw data already exists at {raw_path}. Use --force to re-download.")
        df = pd.read_csv(raw_path)
    else:
        print("Downloading data...")
        df = pd.read_csv(DATA_URL)
        df.to_csv(raw_path, index=False)
        print(f"Saved raw data to {raw_path}.")

    print(f"Dataset shape: {df.shape}")
    print("Columns:")
    for col in df.columns:
        print(f"- {col}")

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Download complaints CSV data.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download data even if the file already exists.",
    )
    args = parser.parse_args()
    download_data(force=args.force)


if __name__ == "__main__":
    main()
