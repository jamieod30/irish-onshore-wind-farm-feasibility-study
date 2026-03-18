from pathlib import Path
import pandas as pd

# --- Paths ---
REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data"/ "era5_downloads" / "raw" 
PROCESSED_DIR = REPO_ROOT / "data" / "era5_wind_data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

REQUIRED_COLUMNS = {"timestep", "u100", "v100", "u10", "v10", "t2m", "sp", "latitude", "longitude"}

def clean_era5_file(csv_path: Path) -> None:
    print(f"Processing: {csv_path.name}")
    df = pd.read_csv(csv_path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path.name} is missing required columns: {sorted(missing)}")

    # Parse timestamp properly
    df["timestep"] = pd.to_datetime(df["timestep"], errors="raise", utc=False)

    # Sort by timestep just in case
    df = df.sort_values("timestep").reset_index(drop=True)

    # Choose output filename
    out_name = FILE_RENAME_MAP.get(csv_path.name, csv_path.name)
    out_path = PROCESSED_DIR / out_name

    # Save cleaned output
    df.to_csv(out_path, index=False)

    # Print quick validation summary
    print(f"  Saved to: {out_path.name}")
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
        clean_era5_file(csv_file)

    print("ERA5 ingestion and cleaning complete.")


if __name__ == "__main__":
    main()