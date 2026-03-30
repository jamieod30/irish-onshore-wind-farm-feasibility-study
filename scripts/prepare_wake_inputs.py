from pathlib import Path
import json
import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = REPO_ROOT / "data" / "era5" / "derived"
OUTPUT_DIR = REPO_ROOT / "data" / "wake_inputs" / "selected_site"

DIRECTIONAL_INPUT_FILE = INPUT_DIR / "directional_frequency.csv"
WEIBULL_INPUT_FILE = INPUT_DIR / "weibull_parameters.csv"
SITE_CONFIG_FILE = OUTPUT_DIR / "site_config.json"

SECTOR_OUTPUT_FILE = OUTPUT_DIR / "sector_frequency.csv"
WEIBULL_OUTPUT_FILE = OUTPUT_DIR / "weibull_global.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DIR_TO_DEG = {
    "N": 0.0,
    "NNE": 22.5,
    "NE": 45.0,
    "ENE": 67.5,
    "E": 90.0,
    "ESE": 112.5,
    "SE": 135.0,
    "SSE": 157.5,
    "S": 180.0,
    "SSW": 202.5,
    "SW": 225.0,
    "WSW": 247.5,
    "W": 270.0,
    "WNW": 292.5,
    "NW": 315.0,
    "NNW": 337.5,
}

EXPECTED_DIRECTIONS = list(DIR_TO_DEG.keys())


def load_site_name(config_path: Path) -> str:
    if not config_path.exists():
        raise FileNotFoundError(
            f"Missing site configuration file: {config_path}\n"
            f"Create site_config.json before running this script."
        )

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    if "site_name" not in config:
        raise KeyError(f"'site_name' not found in {config_path}")

    site_name = str(config["site_name"]).strip().lower()
    if not site_name:
        raise ValueError(f"'site_name' in {config_path} is empty")

    return site_name


def validate_input_files() -> None:
    missing = [
        path for path in [DIRECTIONAL_INPUT_FILE, WEIBULL_INPUT_FILE, SITE_CONFIG_FILE]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing required input file(s):\n" + "\n".join(str(p) for p in missing)
        )


def prepare_sector_frequency(site_name: str) -> pd.DataFrame:
    df = pd.read_csv(DIRECTIONAL_INPUT_FILE)

    required_columns = {"site", "direction", "count", "frequency_pct"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"{DIRECTIONAL_INPUT_FILE.name} is missing required columns: {sorted(missing)}"
        )

    site_df = df[df["site"].str.lower() == site_name].copy()
    if site_df.empty:
        available_sites = sorted(df["site"].dropna().astype(str).str.lower().unique().tolist())
        raise ValueError(
            f"Site '{site_name}' not found in {DIRECTIONAL_INPUT_FILE.name}. "
            f"Available sites: {available_sites}"
        )

    missing_dirs = sorted(set(EXPECTED_DIRECTIONS) - set(site_df["direction"]))
    if missing_dirs:
        raise ValueError(
            f"Directional frequency table is missing expected direction sectors: {missing_dirs}"
        )

    site_df["wd_deg"] = site_df["direction"].map(DIR_TO_DEG)
    if site_df["wd_deg"].isna().any():
        bad_dirs = site_df.loc[site_df["wd_deg"].isna(), "direction"].unique().tolist()
        raise ValueError(f"Unmapped direction labels found: {bad_dirs}")

    site_df["probability"] = pd.to_numeric(site_df["frequency_pct"], errors="coerce") / 100.0

    if site_df["probability"].isna().any():
        raise ValueError("Could not convert one or more frequency_pct values to numeric.")

    if (site_df["probability"] < 0).any():
        raise ValueError("Directional probabilities contain negative values.")

    total_probability = float(site_df["probability"].sum())
    if not np.isclose(total_probability, 1.0, atol=1e-3):
        raise ValueError(
            f"Directional probabilities sum to {total_probability:.6f}, not approximately 1.0."
        )

    output_df = (
        site_df[["wd_deg", "probability"]]
        .sort_values("wd_deg")
        .reset_index(drop=True)
    )

    return output_df


def prepare_weibull_global(site_name: str) -> pd.DataFrame:
    df = pd.read_csv(WEIBULL_INPUT_FILE)

    required_columns = {"site", "weibull_k", "weibull_c"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"{WEIBULL_INPUT_FILE.name} is missing required columns: {sorted(missing)}"
        )

    site_df = df[df["site"].str.lower() == site_name].copy()
    if site_df.empty:
        available_sites = sorted(df["site"].dropna().astype(str).str.lower().unique().tolist())
        raise ValueError(
            f"Site '{site_name}' not found in {WEIBULL_INPUT_FILE.name}. "
            f"Available sites: {available_sites}"
        )

    if len(site_df) != 1:
        raise ValueError(
            f"Expected exactly one Weibull row for site '{site_name}', found {len(site_df)}."
        )

    row = site_df.iloc[0]

    k = float(row["weibull_k"])
    c = float(row["weibull_c"])

    if k <= 0 or c <= 0:
        raise ValueError(
            f"Weibull parameters must be positive. Got k={k}, c={c}"
        )

    output_df = pd.DataFrame({
        "k": [k],
        "c": [c],
    })

    return output_df


def print_summary(site_name: str, sector_df: pd.DataFrame, weibull_df: pd.DataFrame) -> None:
    total_probability = sector_df["probability"].sum()
    dominant = sector_df.sort_values("probability", ascending=False).head(3).copy()

    print("Wake input preparation complete.")
    print(f"Selected site: {site_name}")
    print()
    print(f"Sector frequency file: {SECTOR_OUTPUT_FILE}")
    print(f"Weibull file: {WEIBULL_OUTPUT_FILE}")
    print()
    print(f"Directional probability sum: {total_probability:.6f}")
    print(f"Weibull k: {weibull_df.loc[0, 'k']:.3f}")
    print(f"Weibull c: {weibull_df.loc[0, 'c']:.3f}")
    print()
    print("Top 3 directional sectors:")
    print(dominant.to_string(index=False))


def main() -> None:
    validate_input_files()

    site_name = load_site_name(SITE_CONFIG_FILE)

    sector_df = prepare_sector_frequency(site_name)
    weibull_df = prepare_weibull_global(site_name)

    sector_df.to_csv(SECTOR_OUTPUT_FILE, index=False)
    weibull_df.to_csv(WEIBULL_OUTPUT_FILE, index=False)

    print_summary(site_name, sector_df, weibull_df)


if __name__ == "__main__":
    main()