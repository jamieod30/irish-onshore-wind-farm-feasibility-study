from pathlib import Path
import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

LAYOUT_DIR = REPO_ROOT / "data" / "layouts" / "baseline_aligned"
LAYOUT_FILE = LAYOUT_DIR / "layout_baseline_aligned.csv"
OUTPUT_DIR = REPO_ROOT / "outputs" / "tables" / "layouts"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REQUIRED_COLUMNS = {
    "turbine_id",
    "easting_m",
    "northing_m",
    "hub_height_m",
    "rotor_diameter_m",
    "rated_power_mw",
    "turbine_model",
    "layout_name",
    "site",
    "crs",
}


def validate_layout_columns(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Layout file is missing required columns: {sorted(missing)}")


def validate_no_missing_values(df: pd.DataFrame) -> None:
    if df[list(REQUIRED_COLUMNS)].isna().any().any():
        missing_counts = df[list(REQUIRED_COLUMNS)].isna().sum()
        missing_counts = missing_counts[missing_counts > 0]
        raise ValueError(f"Layout contains missing values:\n{missing_counts.to_string()}")


def validate_unique_turbine_ids(df: pd.DataFrame) -> None:
    if df["turbine_id"].duplicated().any():
        dupes = df.loc[df["turbine_id"].duplicated(), "turbine_id"].tolist()
        raise ValueError(f"Duplicate turbine IDs found: {dupes}")


def validate_unique_coordinates(df: pd.DataFrame) -> None:
    if df.duplicated(subset=["easting_m", "northing_m"]).any():
        dupes = df.loc[
            df.duplicated(subset=["easting_m", "northing_m"]),
            ["easting_m", "northing_m"],
        ]
        raise ValueError(f"Duplicate turbine coordinates found:\n{dupes.to_string(index=False)}")


def validate_single_value_fields(df: pd.DataFrame, fields: list[str]) -> None:
    for field in fields:
        unique_values = df[field].dropna().unique()
        if len(unique_values) != 1:
            raise ValueError(
                f"Field '{field}' should contain exactly one unique value for this baseline layout, "
                f"but found {len(unique_values)} values: {unique_values.tolist()}"
            )


def compute_pairwise_spacing(df: pd.DataFrame) -> pd.DataFrame:
    coords = df[["turbine_id", "easting_m", "northing_m"]].reset_index(drop=True)
    rows = []

    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            dx = coords.loc[j, "easting_m"] - coords.loc[i, "easting_m"]
            dy = coords.loc[j, "northing_m"] - coords.loc[i, "northing_m"]
            spacing_m = float(np.sqrt(dx**2 + dy**2))

            rows.append({
                "turbine_a": coords.loc[i, "turbine_id"],
                "turbine_b": coords.loc[j, "turbine_id"],
                "dx_m": dx,
                "dy_m": dy,
                "spacing_m": spacing_m,
            })

    spacing_df = pd.DataFrame(rows)
    return spacing_df.sort_values("spacing_m").reset_index(drop=True)


def compute_nearest_neighbour_spacing(df: pd.DataFrame) -> pd.DataFrame:
    spacing_df = compute_pairwise_spacing(df)
    turbine_ids = df["turbine_id"].tolist()
    nearest_rows = []

    for tid in turbine_ids:
        candidate_rows = spacing_df[
            (spacing_df["turbine_a"] == tid) | (spacing_df["turbine_b"] == tid)
        ].copy()

        nearest = candidate_rows.nsmallest(1, "spacing_m").iloc[0]

        neighbour = (
            nearest["turbine_b"] if nearest["turbine_a"] == tid else nearest["turbine_a"]
        )

        nearest_rows.append({
            "turbine_id": tid,
            "nearest_neighbour": neighbour,
            "nearest_spacing_m": nearest["spacing_m"],
        })

    return pd.DataFrame(nearest_rows).sort_values("turbine_id").reset_index(drop=True)


def extract_spacing_tiers(
    spacing_df: pd.DataFrame,
    tolerance_m: float = 1.0,
) -> tuple[float | None, float | None]:
    """
    Identify the first two distinct spacing tiers from the pairwise spacing table.

    For a regular baseline layout this should capture:
    - short spacing ~ 5D
    - long spacing ~ 7D
    """
    unique_spacings = []

    for value in spacing_df["spacing_m"].sort_values().tolist():
        if not unique_spacings:
            unique_spacings.append(value)
            continue

        if all(abs(value - existing) > tolerance_m for existing in unique_spacings):
            unique_spacings.append(value)

        if len(unique_spacings) >= 2:
            break

    short_spacing = unique_spacings[0] if len(unique_spacings) >= 1 else None
    long_spacing = unique_spacings[1] if len(unique_spacings) >= 2 else None

    return short_spacing, long_spacing


def build_layout_summary(
    df: pd.DataFrame,
    nearest_df: pd.DataFrame,
    spacing_df: pd.DataFrame,
) -> pd.DataFrame:
    rotor_diameter = float(df["rotor_diameter_m"].iloc[0])
    rated_power = float(df["rated_power_mw"].iloc[0])

    min_spacing_m = float(nearest_df["nearest_spacing_m"].min())
    mean_spacing_m = float(nearest_df["nearest_spacing_m"].mean())

    baseline_short_spacing_m, baseline_long_spacing_m = extract_spacing_tiers(spacing_df)

    summary = {
        "layout_name": df["layout_name"].iloc[0],
        "site": df["site"].iloc[0],
        "crs": df["crs"].iloc[0],
        "n_turbines": int(len(df)),
        "turbine_model": df["turbine_model"].iloc[0],
        "hub_height_m": float(df["hub_height_m"].iloc[0]),
        "rotor_diameter_m": rotor_diameter,
        "rated_power_mw": rated_power,
        "installed_capacity_mw": float(len(df) * rated_power),
        "min_easting_m": float(df["easting_m"].min()),
        "max_easting_m": float(df["easting_m"].max()),
        "min_northing_m": float(df["northing_m"].min()),
        "max_northing_m": float(df["northing_m"].max()),
        "centroid_easting_m": float(df["easting_m"].mean()),
        "centroid_northing_m": float(df["northing_m"].mean()),
        "min_nearest_spacing_m": min_spacing_m,
        "mean_nearest_spacing_m": mean_spacing_m,
        "min_nearest_spacing_D": min_spacing_m / rotor_diameter,
        "mean_nearest_spacing_D": mean_spacing_m / rotor_diameter,
        "baseline_short_spacing_m": baseline_short_spacing_m,
        "baseline_long_spacing_m": baseline_long_spacing_m,
        "baseline_short_spacing_D": (
            baseline_short_spacing_m / rotor_diameter
            if baseline_short_spacing_m is not None
            else np.nan
        ),
        "baseline_long_spacing_D": (
            baseline_long_spacing_m / rotor_diameter
            if baseline_long_spacing_m is not None
            else np.nan
        ),
    }

    return pd.DataFrame([summary])


def run_layout_qc(layout_file: Path = LAYOUT_FILE) -> None:
    if not layout_file.exists():
        raise FileNotFoundError(f"Layout file not found: {layout_file}")

    df = pd.read_csv(layout_file)

    validate_layout_columns(df)
    validate_no_missing_values(df)
    validate_unique_turbine_ids(df)
    validate_unique_coordinates(df)
    validate_single_value_fields(
        df,
        [
            "hub_height_m",
            "rotor_diameter_m",
            "rated_power_mw",
            "turbine_model",
            "layout_name",
            "site",
            "crs",
        ],
    )

    spacing_df = compute_pairwise_spacing(df)
    nearest_df = compute_nearest_neighbour_spacing(df)
    summary_df = build_layout_summary(df, nearest_df, spacing_df)

    min_spacing_d = float(summary_df["min_nearest_spacing_D"].iloc[0])

    if min_spacing_d < 5.0:
        raise ValueError(
            f"Minimum nearest-neighbour spacing is {min_spacing_d:.2f}D, below the 5D baseline threshold."
        )

    summary_path = OUTPUT_DIR / "layout_summary.csv"
    nearest_path = OUTPUT_DIR / "layout_nearest_neighbour_spacing.csv"
    all_spacing_path = OUTPUT_DIR / "layout_all_pairwise_spacing.csv"

    summary_df.to_csv(summary_path, index=False)
    nearest_df.to_csv(nearest_path, index=False)
    spacing_df.to_csv(all_spacing_path, index=False)

    print("Layout QA complete.")
    print(f"Layout file: {layout_file}")
    print(f"Summary saved to: {summary_path}")
    print(f"Nearest-neighbour spacing saved to: {nearest_path}")
    print(f"All pairwise spacing saved to: {all_spacing_path}")
    print()
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    run_layout_qc()