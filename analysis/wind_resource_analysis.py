from pathlib import Path
import numpy as np
import pandas as pd

# ----------------------------
# Configuration
# ----------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = REPO_ROOT / "data" / "era5_wind_data" / "processed"
CLEAN_OUTPUT_DIR = INPUT_DIR / "cleaned"

ANALYSIS_DIR = REPO_ROOT / "analysis"
SITE_COMPARISON_PATH = ANALYSIS_DIR / "site_comparison.csv"
QA_SUMMARY_PATH = ANALYSIS_DIR / "qa_summary.csv"
SEASONAL_COMPARISON_PATH = ANALYSIS_DIR / "seasonal_wind_comparison.csv"

CLEAN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

# Extreme wind speed threshold for QA flagging at 100 m
EXTREME_WS_THRESHOLD = 40.0  # m/s

# Specific gas constant for dry air
R_DRY_AIR = 287.05  # J/(kg·K)

# Expected timestep frequency
EXPECTED_FREQ = "1h"

# Seasonal definitions
WINTER_MONTHS = [12, 1, 2]
SUMMER_MONTHS = [6, 7, 8]

# Required columns
REQUIRED_COLUMNS = {
    "timestep", "u100", "v100", "u10", "v10", "t2m", "sp", "latitude", "longitude"
}

# ----------------------------
# Helper Functions
# ----------------------------

def derive_site_name(file_path: Path) -> str:
    """
    Derive a clean site name from filenames like:
    era5_mayo.csv -> mayo
    """
    stem = file_path.stem.lower()
    if stem.startswith("era5_"):
        return stem.replace("era5_", "", 1)
    return stem


def compute_wind_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute wind speed, wind direction, air density, and wind power density.
    Assumes:
    - u/v in m/s
    - t2m in Kelvin
    - sp in Pascals
    """
    df["ws_100"] = np.sqrt(df["u100"] ** 2 + df["v100"] ** 2)
    df["ws_10"] = np.sqrt(df["u10"] ** 2 + df["v10"] ** 2)

    # Wind directions (meteorological convention)
    df["wd_100"] = (180 + np.degrees(np.arctan2(df["u100"], df["v100"]))) % 360
    df["wd_10"] = (180 + np.degrees(np.arctan2(df["u10"], df["v10"]))) % 360

    # Air density from ideal gas law
    df["rho"] = df["sp"] / (R_DRY_AIR * df["t2m"])

    # Wind power density at 100 m
    df["power_density"] = 0.5 * df["rho"] * (df["ws_100"] ** 3)

    return df


def compute_seasonal_metrics(df: pd.DataFrame) -> dict:
    """
    Compute seasonal mean wind speeds at 100 m.
    """
    month = df["timestep"].dt.month

    winter_mean_ws_100 = df.loc[month.isin(WINTER_MONTHS), "ws_100"].mean()
    summer_mean_ws_100 = df.loc[month.isin(SUMMER_MONTHS), "ws_100"].mean()

    return {
        "winter_mean_ws_100": winter_mean_ws_100,
        "summer_mean_ws_100": summer_mean_ws_100,
    }


def run_qa_checks(df: pd.DataFrame, site_name: str) -> dict:
    """
    Run QA checks:
    - total records
    - total missing values
    - duplicate timesteps
    - missing timesteps
    - extreme wind speed counts
    - contextual statistics
    - seasonal wind speed metrics
    """
    if df.empty:
        raise ValueError(f"{site_name} contains no rows.")

    df = df.sort_values("timestep").reset_index(drop=True)

    duplicate_timestamps = int(df["timestep"].duplicated().sum())

    full_range = pd.date_range(
        start=df["timestep"].min(),
        end=df["timestep"].max(),
        freq=EXPECTED_FREQ,
        tz="UTC",
    )
    missing_timestamps_count = int(len(full_range.difference(df["timestep"])))

    missing_values_total = int(df.isna().sum().sum())
    extreme_ws_100_count = int((df["ws_100"] > EXTREME_WS_THRESHOLD).sum())

    seasonal_metrics = compute_seasonal_metrics(df)

    qa = {
        "site": site_name,
        "records": len(df),
        "missing_values_total": missing_values_total,
        "duplicate_timestamps": duplicate_timestamps,
        "missing_timestamps": missing_timestamps_count,
        "mean_ws_100": df["ws_100"].mean(),
        "median_ws_100": df["ws_100"].median(),
        "p90_ws_100": df["ws_100"].quantile(0.90),
        "std_ws_100": df["ws_100"].std(),
        "min_ws_100": df["ws_100"].min(),
        "max_ws_100": df["ws_100"].max(),
        "extreme_ws_100_count": extreme_ws_100_count,
        "mean_ws_10": df["ws_10"].mean(),
        "mean_rho": df["rho"].mean(),
        "mean_power_density": df["power_density"].mean(),
        "latitude": df["latitude"].iloc[0],
        "longitude": df["longitude"].iloc[0],
        "start_timestep": df["timestep"].min(),
        "end_timestep": df["timestep"].max(),
        **seasonal_metrics,
    }

    return qa


def export_clean_dataset(df: pd.DataFrame, input_file: Path) -> Path:
    """
    Save cleaned/enhanced dataset to processed/cleaned using same filename.
    Overwrites existing file if present.
    """
    output_path = CLEAN_OUTPUT_DIR / input_file.name
    df.to_csv(output_path, index=False)
    return output_path


def process_single_file(file_path: Path) -> tuple[pd.DataFrame, dict]:
    """
    Load, parse timestamps, compute wind metrics, run QA, export cleaned file.
    """
    site_name = derive_site_name(file_path)
    print(f"Processing site: {site_name} ({file_path.name})")

    df = pd.read_csv(file_path)

    if df.empty:
        raise ValueError(f"{file_path.name} is empty.")

    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"{file_path.name} is missing required columns: {sorted(missing_cols)}"
        )

    df["timestep"] = pd.to_datetime(df["timestep"], utc=True, errors="raise")

    df = df.sort_values("timestep").reset_index(drop=True)
    df = compute_wind_metrics(df)

    qa_result = run_qa_checks(df, site_name)

    output_path = export_clean_dataset(df, file_path)
    print(f"  Clean dataset saved to: {output_path}")
    print(f"  Mean ws_100: {qa_result['mean_ws_100']:.2f} m/s")
    print(f"  Winter mean ws_100: {qa_result['winter_mean_ws_100']:.2f} m/s")
    print(f"  Summer mean ws_100: {qa_result['summer_mean_ws_100']:.2f} m/s")
    print(f"  Mean power density: {qa_result['mean_power_density']:.2f} W/m²")
    print()

    return df, qa_result


# ----------------------------
# Main Workflow
# ----------------------------

def main() -> None:
    csv_files = sorted(INPUT_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {INPUT_DIR}")

    print(f"Found {len(csv_files)} processed ERA5 files.\n")

    qa_results = []

    for csv_file in csv_files:
        _, qa = process_single_file(csv_file)
        qa_results.append(qa)

    qa_df = pd.DataFrame(qa_results).sort_values(
        by="mean_ws_100", ascending=False
    ).reset_index(drop=True)

    # Full QA summary
    qa_df.to_csv(QA_SUMMARY_PATH, index=False)

    # Site comparison file
    comparison_cols = [
        "site",
        "latitude",
        "longitude",
        "records",
        "mean_ws_100",
        "median_ws_100",
        "p90_ws_100",
        "std_ws_100",
        "mean_rho",
        "mean_power_density",
        "mean_ws_10",
        "winter_mean_ws_100",
        "summer_mean_ws_100",
        "duplicate_timestamps",
        "missing_timestamps",
        "extreme_ws_100_count",
    ]
    comparison_df = qa_df[comparison_cols]
    comparison_df.to_csv(SITE_COMPARISON_PATH, index=False)

    # Dedicated seasonal comparison file
    seasonal_df = qa_df[
        ["site", "winter_mean_ws_100", "summer_mean_ws_100"]
    ].copy()

    seasonal_df["winter_mean_ws_100"] = seasonal_df["winter_mean_ws_100"].round(2)
    seasonal_df["summer_mean_ws_100"] = seasonal_df["summer_mean_ws_100"].round(2)

    seasonal_df.to_csv(SEASONAL_COMPARISON_PATH, index=False)

    print("Wind resource analysis complete.")
    print(f"QA summary saved to: {QA_SUMMARY_PATH}")
    print(f"Site comparison saved to: {SITE_COMPARISON_PATH}")
    print(f"Seasonal comparison saved to: {SEASONAL_COMPARISON_PATH}")


if __name__ == "__main__":
    main()