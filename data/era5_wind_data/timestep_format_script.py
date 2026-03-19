from pathlib import Path
import pandas as pd

# ----------------------------
# Absolute Paths
# ----------------------------

RAW_DIR = Path(
    r"C:\Users\jamie\Documents\GitHub\irish-onshore-wind-farm-feasibility-study\data\era5_wind_data\raw"
)

PROCESSED_DIR = Path(
    r"C:\Users\jamie\Documents\GitHub\irish-onshore-wind-farm-feasibility-study\data\era5_wind_data\processed"
)

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

REQUIRED_COLUMNS = {
    "timestep", "u100", "v100", "u10", "v10", "t2m", "sp", "latitude", "longitude"
}


def standardise_timestamp_column(df: pd.DataFrame, csv_name: str) -> pd.DataFrame:
    """
    Rename valid_time -> timestep if needed, then parse mixed datetime formats robustly.
    """
    if "timestep" not in df.columns and "valid_time" in df.columns:
        df = df.rename(columns={"valid_time": "timestep"})

    if "timestep" not in df.columns:
        raise ValueError(
            f"{csv_name} does not contain 'timestep' or 'valid_time' column."
        )

    # Parse datetime values robustly and convert to UTC-aware timestamps
    df["timestep"] = pd.to_datetime(
        df["timestep"],
        errors="raise",
        utc=True,
        format="mixed",
        dayfirst=True,
    )

    return df


def clean_era5_file(csv_path: Path) -> None:
    print(f"Processing: {csv_path.name}")
    df = pd.read_csv(csv_path)

    # Standardise timestamp column first
    df = standardise_timestamp_column(df, csv_path.name)

    # Validate required columns after timestamp rename
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"{csv_path.name} is missing required columns: {sorted(missing)}"
        )

    # Sort by timestep
    df = df.sort_values("timestep").reset_index(drop=True)

    # Standardise output timestamp format to ISO UTC string
    df["timestep"] = df["timestep"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Save using same filename in processed folder (overwrite if exists)
    out_path = PROCESSED_DIR / csv_path.name
    df.to_csv(out_path, index=False)

    # Validation printout
    print(f"  Saved to: {out_path}")
    print(f"  Rows: {len(df)}")
    print(f"  Start: {df['timestep'].iloc[0]}")
    print(f"  End:   {df['timestep'].iloc[-1]}")
    print(f"  Lat/Lon: {df['latitude'].iloc[0]}, {df['longitude'].iloc[0]}")
    print()


def main() -> None:
    csv_files = sorted(RAW_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {RAW_DIR}")

    print(f"Found {len(csv_files)} raw ERA5 files.\n")

    for csv_file in csv_files:
        try:
            clean_era5_file(csv_file)
        except Exception as e:
            print(f"  ERROR processing {csv_file.name}: {e}\n")

    print("ERA5 ingestion and timestamp standardisation complete.")


if __name__ == "__main__":
    main()