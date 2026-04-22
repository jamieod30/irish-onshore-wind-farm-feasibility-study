from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import weibull_min
from windrose import WindroseAxes

REPO_ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = REPO_ROOT / "data" / "era5" / "processed"
CLEAN_OUTPUT_DIR = REPO_ROOT / "data" / "era5" / "cleaned"
DERIVED_DIR = REPO_ROOT / "data" / "era5" / "derived"

FIGURES_DIR = REPO_ROOT / "outputs" / "figures" / "resource"
TABLES_DIR = REPO_ROOT / "outputs" / "tables" / "resource"

QA_SUMMARY_PATH = DERIVED_DIR / "qa_summary.csv"
SITE_COMPARISON_PATH = DERIVED_DIR / "site_comparison.csv"
SEASONAL_COMPARISON_PATH = DERIVED_DIR / "seasonal_wind_comparison.csv"
WEIBULL_PARAMETERS_PATH = DERIVED_DIR / "weibull_parameters.csv"
MONTHLY_CLIMATOLOGY_PATH = DERIVED_DIR / "monthly_wind_climatology.csv"
DIRECTIONAL_FREQUENCY_PATH = DERIVED_DIR / "directional_frequency.csv"

CLEAN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DERIVED_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

EXTREME_WS_THRESHOLD = 40.0
R_DRY_AIR = 287.05
EXPECTED_FREQ = "1h"

WINTER_MONTHS = [12, 1, 2]
SUMMER_MONTHS = [6, 7, 8]

MONTH_NAME_MAP = {
    1: "Jan",
    2: "Feb",
    3: "Mar",
    4: "Apr",
    5: "May",
    6: "Jun",
    7: "Jul",
    8: "Aug",
    9: "Sep",
    10: "Oct",
    11: "Nov",
    12: "Dec",
}

DIRECTION_SECTORS = [
    "N", "NNE", "NE", "ENE",
    "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW",
    "W", "WNW", "NW", "NNW",
]

REQUIRED_COLUMNS = {
    "timestep", "u100", "v100", "u10", "v10", "t2m", "sp", "latitude", "longitude"
}


def derive_site_name(file_path: Path) -> str:
    stem = file_path.stem.lower()
    if stem.startswith("era5_"):
        return stem.replace("era5_", "", 1)
    return stem


def compute_wind_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["ws_100"] = np.sqrt(df["u100"] ** 2 + df["v100"] ** 2)
    df["ws_10"] = np.sqrt(df["u10"] ** 2 + df["v10"] ** 2)

    df["wd_100"] = (180 + np.degrees(np.arctan2(df["u100"], df["v100"]))) % 360
    df["wd_10"] = (180 + np.degrees(np.arctan2(df["u10"], df["v10"]))) % 360

    df["rho"] = df["sp"] / (R_DRY_AIR * df["t2m"])
    df["power_density"] = 0.5 * df["rho"] * (df["ws_100"] ** 3)

    return df


def compute_seasonal_metrics(df: pd.DataFrame) -> dict:
    month = df["timestep"].dt.month

    winter_mean_ws_100 = df.loc[month.isin(WINTER_MONTHS), "ws_100"].mean()
    summer_mean_ws_100 = df.loc[month.isin(SUMMER_MONTHS), "ws_100"].mean()

    return {
        "winter_mean_ws_100": winter_mean_ws_100,
        "summer_mean_ws_100": summer_mean_ws_100,
    }


def fit_weibull(df: pd.DataFrame) -> dict:
    ws = df["ws_100"].dropna()
    if ws.empty:
        raise ValueError("ws_100 contains no valid values for Weibull fitting.")

    shape, loc, scale = weibull_min.fit(ws, floc=0)

    return {
        "weibull_k": shape,
        "weibull_c": scale,
    }


def compute_shear(df: pd.DataFrame) -> dict:
    valid = (df["ws_100"] > 0) & (df["ws_10"] > 0)
    if not valid.any():
        return {"shear_exponent": np.nan}

    alpha = np.log(df.loc[valid, "ws_100"] / df.loc[valid, "ws_10"]) / np.log(100 / 10)
    alpha = alpha.replace([np.inf, -np.inf], np.nan).dropna()

    if alpha.empty:
        return {"shear_exponent": np.nan}

    return {"shear_exponent": alpha.mean()}


def compute_turbulence_proxy(df: pd.DataFrame) -> dict:
    ws = df["ws_100"].dropna()
    if ws.empty:
        return {"turbulence_proxy": np.nan}

    mean_ws = ws.mean()
    if mean_ws <= 0:
        return {"turbulence_proxy": np.nan}

    return {"turbulence_proxy": ws.std() / mean_ws}


def direction_from_degrees(degrees: pd.Series) -> pd.Series:
    idx = (((degrees % 360) + 11.25) // 22.5).astype(int) % 16
    return pd.Series([DIRECTION_SECTORS[i] for i in idx], index=degrees.index)


def compute_directional_frequency(df: pd.DataFrame, site_name: str) -> tuple[dict, pd.DataFrame]:
    wd = df["wd_100"].dropna()
    if wd.empty:
        summary = {
            "dominant_dir_1": np.nan,
            "dominant_dir_1_pct": np.nan,
            "dominant_dir_2": np.nan,
            "dominant_dir_2_pct": np.nan,
        }
        empty_table = pd.DataFrame(columns=["site", "direction", "count", "frequency_pct"])
        return summary, empty_table

    directions = direction_from_degrees(wd)
    freq = (
        directions.value_counts(normalize=True)
        .reindex(DIRECTION_SECTORS, fill_value=0.0)
        .reset_index()
    )
    freq.columns = ["direction", "frequency_frac"]
    freq["frequency_pct"] = freq["frequency_frac"] * 100
    freq["count"] = directions.value_counts().reindex(DIRECTION_SECTORS, fill_value=0).values
    freq["site"] = site_name

    freq = freq[["site", "direction", "count", "frequency_pct"]]

    top2 = freq.sort_values("frequency_pct", ascending=False).head(2).reset_index(drop=True)

    summary = {
        "dominant_dir_1": top2.loc[0, "direction"] if len(top2) > 0 else np.nan,
        "dominant_dir_1_pct": top2.loc[0, "frequency_pct"] if len(top2) > 0 else np.nan,
        "dominant_dir_2": top2.loc[1, "direction"] if len(top2) > 1 else np.nan,
        "dominant_dir_2_pct": top2.loc[1, "frequency_pct"] if len(top2) > 1 else np.nan,
    }

    return summary, freq


def compute_monthly_climatology(df: pd.DataFrame, site_name: str) -> pd.DataFrame:
    monthly = (
        df.assign(month=df["timestep"].dt.month)
        .groupby("month", as_index=False)
        .agg(
            mean_ws_100=("ws_100", "mean"),
            mean_ws_10=("ws_10", "mean"),
            mean_power_density=("power_density", "mean"),
            mean_rho=("rho", "mean"),
            records=("ws_100", "size"),
        )
    )

    monthly["site"] = site_name
    monthly["month_name"] = monthly["month"].map(MONTH_NAME_MAP)

    monthly = monthly[
        ["site", "month", "month_name", "records", "mean_ws_100", "mean_ws_10", "mean_power_density", "mean_rho"]
    ].sort_values("month").reset_index(drop=True)

    return monthly


def plot_histogram(df: pd.DataFrame, site: str) -> Path:
    output_path = FIGURES_DIR / f"{site}_histogram.png"

    plt.figure()
    plt.hist(df["ws_100"].dropna(), bins=50, density=True)
    plt.xlabel("Wind Speed at 100 m (m/s)")
    plt.ylabel("Probability Density")
    plt.title(f"{site.capitalize()} Wind Speed Distribution")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return output_path


def plot_weibull(df: pd.DataFrame, site: str, k: float, c: float) -> Path:
    output_path = FIGURES_DIR / f"{site}_weibull.png"

    ws = df["ws_100"].dropna()

    plt.figure()
    plt.hist(ws, bins=50, density=True, alpha=0.5, label="Data")

    x = np.linspace(0, ws.max(), 200)
    y = weibull_min.pdf(x, k, loc=0, scale=c)

    plt.plot(x, y, label=f"Weibull (k={k:.2f}, c={c:.2f})")
    plt.xlabel("Wind Speed at 100 m (m/s)")
    plt.ylabel("Probability Density")
    plt.title(f"{site.capitalize()} Weibull Fit")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return output_path


def plot_wind_rose(df: pd.DataFrame, site: str) -> Path:
    output_path = FIGURES_DIR / f"{site}_wind_rose.png"

    ax = WindroseAxes.from_ax()
    ax.bar(df["wd_100"], df["ws_100"], normed=True, opening=0.8)
    ax.set_title(f"{site.capitalize()} Wind Rose")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return output_path


def plot_monthly_climatology(monthly_df: pd.DataFrame, site: str) -> Path:
    output_path = FIGURES_DIR / f"{site}_monthly_climatology.png"

    plot_df = monthly_df.sort_values("month")

    plt.figure()
    plt.plot(plot_df["month_name"], plot_df["mean_ws_100"], marker="o")
    plt.xlabel("Month")
    plt.ylabel("Mean Wind Speed at 100 m (m/s)")
    plt.title(f"{site.capitalize()} Monthly Wind Climatology")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return output_path


def compute_advanced_metrics(df: pd.DataFrame, site_name: str) -> tuple[dict, pd.DataFrame]:
    weibull_metrics = fit_weibull(df)
    shear_metrics = compute_shear(df)
    turbulence_metrics = compute_turbulence_proxy(df)
    directional_summary, directional_frequency_df = compute_directional_frequency(df, site_name)

    plot_histogram(df, site_name)
    plot_weibull(df, site_name, weibull_metrics["weibull_k"], weibull_metrics["weibull_c"])
    plot_wind_rose(df, site_name)

    return {
        **weibull_metrics,
        **shear_metrics,
        **turbulence_metrics,
        **directional_summary,
    }, directional_frequency_df


def run_qa_checks(df: pd.DataFrame, site_name: str) -> tuple[dict, pd.DataFrame]:
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
    advanced_metrics, directional_frequency_df = compute_advanced_metrics(df, site_name)

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
        **advanced_metrics,
    }

    return qa, directional_frequency_df


def export_clean_dataset(df: pd.DataFrame, input_file: Path) -> Path:
    output_path = CLEAN_OUTPUT_DIR / input_file.name
    df.to_csv(output_path, index=False)
    return output_path


def process_single_file(file_path: Path) -> tuple[pd.DataFrame, dict, pd.DataFrame, pd.DataFrame]:
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

    qa_result, directional_frequency_df = run_qa_checks(df, site_name)
    monthly_climatology_df = compute_monthly_climatology(df, site_name)
    plot_monthly_climatology(monthly_climatology_df, site_name)

    output_path = export_clean_dataset(df, file_path)
    print(f"  Clean dataset saved to: {output_path}")
    print(f"  Mean ws_100: {qa_result['mean_ws_100']:.2f} m/s")
    print(f"  Winter mean ws_100: {qa_result['winter_mean_ws_100']:.2f} m/s")
    print(f"  Summer mean ws_100: {qa_result['summer_mean_ws_100']:.2f} m/s")
    print(f"  Weibull k: {qa_result['weibull_k']:.3f}")
    print(f"  Weibull c: {qa_result['weibull_c']:.3f}")
    print(f"  Shear exponent: {qa_result['shear_exponent']:.3f}")
    print(f"  Turbulence proxy: {qa_result['turbulence_proxy']:.3f}")
    print(
        f"  Dominant directions: {qa_result['dominant_dir_1']} ({qa_result['dominant_dir_1_pct']:.2f}%), "
        f"{qa_result['dominant_dir_2']} ({qa_result['dominant_dir_2_pct']:.2f}%)"
    )
    print(f"  Mean power density: {qa_result['mean_power_density']:.2f} W/m²")
    print()

    return df, qa_result, monthly_climatology_df, directional_frequency_df


def main() -> int:
    try:
        csv_files = sorted(INPUT_DIR.glob("*.csv"))

        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {INPUT_DIR}")

        print(f"Found {len(csv_files)} processed ERA5 files.\n")

        qa_results = []
        monthly_climatology_results = []
        directional_frequency_results = []

        for csv_file in csv_files:
            _, qa, monthly_climatology_df, directional_frequency_df = process_single_file(csv_file)
            qa_results.append(qa)
            monthly_climatology_results.append(monthly_climatology_df)
            directional_frequency_results.append(directional_frequency_df)

        qa_df = pd.DataFrame(qa_results).sort_values(
            by="mean_ws_100", ascending=False
        ).reset_index(drop=True)

        qa_df.to_csv(QA_SUMMARY_PATH, index=False)

        comparison_cols = [
            "site",
            "latitude",
            "longitude",
            "records",
            "mean_ws_100",
            "median_ws_100",
            "p90_ws_100",
            "std_ws_100",
            "turbulence_proxy",
            "mean_rho",
            "mean_power_density",
            "mean_ws_10",
            "winter_mean_ws_100",
            "summer_mean_ws_100",
            "weibull_k",
            "weibull_c",
            "shear_exponent",
            "dominant_dir_1",
            "dominant_dir_1_pct",
            "dominant_dir_2",
            "dominant_dir_2_pct",
            "duplicate_timestamps",
            "missing_timestamps",
            "extreme_ws_100_count",
        ]
        comparison_df = qa_df[comparison_cols]
        comparison_df.to_csv(SITE_COMPARISON_PATH, index=False)

        seasonal_df = qa_df[
            ["site", "winter_mean_ws_100", "summer_mean_ws_100"]
        ].copy()
        seasonal_df["winter_mean_ws_100"] = seasonal_df["winter_mean_ws_100"].round(2)
        seasonal_df["summer_mean_ws_100"] = seasonal_df["summer_mean_ws_100"].round(2)
        seasonal_df.to_csv(SEASONAL_COMPARISON_PATH, index=False)

        weibull_df = qa_df[
            ["site", "weibull_k", "weibull_c", "shear_exponent", "turbulence_proxy"]
        ].copy()
        weibull_df.to_csv(WEIBULL_PARAMETERS_PATH, index=False)

        monthly_climatology_df = (
            pd.concat(monthly_climatology_results, ignore_index=True)
            .sort_values(["site", "month"])
            .reset_index(drop=True)
        )
        monthly_climatology_df.to_csv(MONTHLY_CLIMATOLOGY_PATH, index=False)

        directional_frequency_df = (
            pd.concat(directional_frequency_results, ignore_index=True)
            .sort_values(["site", "direction"])
            .reset_index(drop=True)
        )
        directional_frequency_df.to_csv(DIRECTIONAL_FREQUENCY_PATH, index=False)

        qa_df.to_csv(TABLES_DIR / "qa_summary.csv", index=False)
        comparison_df.to_csv(TABLES_DIR / "site_comparison.csv", index=False)
        seasonal_df.to_csv(TABLES_DIR / "seasonal_wind_comparison.csv", index=False)
        weibull_df.to_csv(TABLES_DIR / "weibull_parameters.csv", index=False)
        monthly_climatology_df.to_csv(TABLES_DIR / "monthly_wind_climatology.csv", index=False)
        directional_frequency_df.to_csv(TABLES_DIR / "directional_frequency.csv", index=False)

        print("Wind resource analysis complete.")
        print(f"QA summary saved to: {QA_SUMMARY_PATH}")
        print(f"Site comparison saved to: {SITE_COMPARISON_PATH}")
        print(f"Seasonal comparison saved to: {SEASONAL_COMPARISON_PATH}")
        print(f"Weibull parameters saved to: {WEIBULL_PARAMETERS_PATH}")
        print(f"Monthly climatology saved to: {MONTHLY_CLIMATOLOGY_PATH}")
        print(f"Directional frequency saved to: {DIRECTIONAL_FREQUENCY_PATH}")
        print(f"Figures saved to: {FIGURES_DIR}")
        return 0

    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())