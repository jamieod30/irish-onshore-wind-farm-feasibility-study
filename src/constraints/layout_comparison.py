from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_BASELINE_LAYOUT = REPO_ROOT / "data" / "layouts" / "baseline_aligned" / "layout_baseline_aligned.csv"
DEFAULT_REFINED_LAYOUT = REPO_ROOT / "data" / "layouts" / "refined_constraints" / "layout_refined_constraints.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "tables" / "constraints"
DEFAULT_REFINED_METADATA = REPO_ROOT / "data" / "layouts" / "refined_constraints" / "layout_metadata.json"

REQUIRED_LAYOUT_COLUMNS = [
    "turbine_id",
    "easting_m",
    "northing_m",
    "hub_height_m",
    "rotor_diameter_m",
    "rated_power_mw",
    "turbine_model",
]


@dataclass(frozen=True)
class LayoutComparisonResult:
    turbine_changes_df: pd.DataFrame
    summary_df: pd.DataFrame
    refined_layout_summary_df: pd.DataFrame
    metadata: dict[str, Any]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_layout(layout_csv: Path) -> pd.DataFrame:
    if not layout_csv.exists():
        raise FileNotFoundError(f"Layout file not found: {layout_csv}")

    df = pd.read_csv(layout_csv)
    missing = [c for c in REQUIRED_LAYOUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Layout CSV missing required columns: {missing}")

    df = df.copy()

    numeric_cols = [
        "easting_m",
        "northing_m",
        "hub_height_m",
        "rotor_diameter_m",
        "rated_power_mw",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df[REQUIRED_LAYOUT_COLUMNS].isna().any().any():
        raise ValueError("Layout contains missing or non-numeric required values.")

    if df["turbine_id"].duplicated().any():
        dupes = df.loc[df["turbine_id"].duplicated(), "turbine_id"].tolist()
        raise ValueError(f"Duplicate turbine IDs found: {dupes}")

    return df.reset_index(drop=True)


def compute_nearest_spacing(layout_df: pd.DataFrame) -> tuple[float, float]:
    coords = layout_df[["easting_m", "northing_m"]].to_numpy(dtype=float)
    if len(coords) < 2:
        return np.nan, np.nan

    distances = []
    for i in range(len(coords)):
        d = np.sqrt(((coords - coords[i]) ** 2).sum(axis=1))
        d[i] = np.inf
        distances.append(d.min())

    distances = np.asarray(distances, dtype=float)
    return float(np.nanmin(distances)), float(np.nanmean(distances))


def build_turbine_changes(baseline_df: pd.DataFrame, refined_df: pd.DataFrame) -> pd.DataFrame:
    baseline_cols = ["turbine_id", "easting_m", "northing_m", "rated_power_mw", "rotor_diameter_m"]
    refined_cols = ["turbine_id", "easting_m", "northing_m", "rated_power_mw", "rotor_diameter_m"]

    merged = baseline_df[baseline_cols].merge(
        refined_df[refined_cols],
        on="turbine_id",
        how="outer",
        suffixes=("_baseline", "_refined"),
        indicator=True,
    )

    def classify(row: pd.Series) -> str:
        if row["_merge"] == "left_only":
            return "removed"
        if row["_merge"] == "right_only":
            return "added"
        dx = row["easting_m_refined"] - row["easting_m_baseline"]
        dy = row["northing_m_refined"] - row["northing_m_baseline"]
        move_dist = float(np.sqrt(dx**2 + dy**2))
        return "moved" if move_dist > 1.0 else "retained"

    merged["change_type"] = merged.apply(classify, axis=1)

    merged["delta_easting_m"] = merged["easting_m_refined"] - merged["easting_m_baseline"]
    merged["delta_northing_m"] = merged["northing_m_refined"] - merged["northing_m_baseline"]
    merged["movement_distance_m"] = np.sqrt(
        (merged["delta_easting_m"].fillna(0.0) ** 2) +
        (merged["delta_northing_m"].fillna(0.0) ** 2)
    )

    cols = [
        "turbine_id",
        "change_type",
        "easting_m_baseline",
        "northing_m_baseline",
        "easting_m_refined",
        "northing_m_refined",
        "delta_easting_m",
        "delta_northing_m",
        "movement_distance_m",
        "rated_power_mw_baseline",
        "rated_power_mw_refined",
        "rotor_diameter_m_baseline",
        "rotor_diameter_m_refined",
    ]
    return merged.loc[:, cols].sort_values(["change_type", "turbine_id"]).reset_index(drop=True)


def build_refined_layout_summary(refined_df: pd.DataFrame) -> pd.DataFrame:
    min_spacing_m, mean_spacing_m = compute_nearest_spacing(refined_df)
    rotor_diameter_m = float(refined_df["rotor_diameter_m"].iloc[0])
    installed_capacity_mw = float(refined_df["rated_power_mw"].sum())

    summary = pd.DataFrame(
        [
            {
                "layout_name": str(refined_df["layout_name"].iloc[0]) if "layout_name" in refined_df.columns else "refined_constraints",
                "site": str(refined_df["site"].iloc[0]) if "site" in refined_df.columns else "unknown",
                "crs": str(refined_df["crs"].iloc[0]) if "crs" in refined_df.columns else "unknown",
                "n_turbines": int(len(refined_df)),
                "installed_capacity_mw": installed_capacity_mw,
                "hub_height_m": float(refined_df["hub_height_m"].iloc[0]),
                "rotor_diameter_m": rotor_diameter_m,
                "rated_power_mw": float(refined_df["rated_power_mw"].iloc[0]),
                "min_nearest_spacing_m": min_spacing_m,
                "mean_nearest_spacing_m": mean_spacing_m,
                "min_nearest_spacing_D": min_spacing_m / rotor_diameter_m if pd.notna(min_spacing_m) else np.nan,
                "mean_nearest_spacing_D": mean_spacing_m / rotor_diameter_m if pd.notna(mean_spacing_m) else np.nan,
                "min_easting_m": float(refined_df["easting_m"].min()),
                "max_easting_m": float(refined_df["easting_m"].max()),
                "min_northing_m": float(refined_df["northing_m"].min()),
                "max_northing_m": float(refined_df["northing_m"].max()),
                "centroid_easting_m": float(refined_df["easting_m"].mean()),
                "centroid_northing_m": float(refined_df["northing_m"].mean()),
            }
        ]
    )
    return summary


def build_summary(
    baseline_df: pd.DataFrame,
    refined_df: pd.DataFrame,
    turbine_changes_df: pd.DataFrame,
) -> pd.DataFrame:
    baseline_capacity = float(baseline_df["rated_power_mw"].sum())
    refined_capacity = float(refined_df["rated_power_mw"].sum())

    baseline_min_spacing_m, baseline_mean_spacing_m = compute_nearest_spacing(baseline_df)
    refined_min_spacing_m, refined_mean_spacing_m = compute_nearest_spacing(refined_df)

    summary = pd.DataFrame(
        [
            {
                "baseline_layout_name": str(baseline_df["layout_name"].iloc[0]) if "layout_name" in baseline_df.columns else "baseline",
                "refined_layout_name": str(refined_df["layout_name"].iloc[0]) if "layout_name" in refined_df.columns else "refined_constraints",
                "baseline_n_turbines": int(len(baseline_df)),
                "refined_n_turbines": int(len(refined_df)),
                "delta_n_turbines": int(len(refined_df) - len(baseline_df)),
                "baseline_installed_capacity_mw": baseline_capacity,
                "refined_installed_capacity_mw": refined_capacity,
                "delta_installed_capacity_mw": refined_capacity - baseline_capacity,
                "n_retained": int((turbine_changes_df["change_type"] == "retained").sum()),
                "n_moved": int((turbine_changes_df["change_type"] == "moved").sum()),
                "n_removed": int((turbine_changes_df["change_type"] == "removed").sum()),
                "n_added": int((turbine_changes_df["change_type"] == "added").sum()),
                "max_movement_distance_m": float(turbine_changes_df["movement_distance_m"].max()),
                "mean_movement_distance_m_for_moved": float(
                    turbine_changes_df.loc[
                        turbine_changes_df["change_type"] == "moved",
                        "movement_distance_m"
                    ].mean()
                ) if (turbine_changes_df["change_type"] == "moved").any() else 0.0,
                "baseline_min_nearest_spacing_m": baseline_min_spacing_m,
                "refined_min_nearest_spacing_m": refined_min_spacing_m,
                "baseline_mean_nearest_spacing_m": baseline_mean_spacing_m,
                "refined_mean_nearest_spacing_m": refined_mean_spacing_m,
            }
        ]
    )

    return summary


def build_metadata(
    baseline_layout_csv: Path,
    refined_layout_csv: Path,
    summary_df: pd.DataFrame,
) -> dict[str, Any]:
    row = summary_df.iloc[0]
    return {
        "baseline_layout_file": str(baseline_layout_csv.relative_to(REPO_ROOT)).replace("\\", "/"),
        "refined_layout_file": str(refined_layout_csv.relative_to(REPO_ROOT)).replace("\\", "/"),
        "comparison_type": "baseline_vs_refined_constraints",
        "summary": row.to_dict(),
        "notes": [
            "Comparison is geometric and capacity-based only",
            "No wake model rerun included in this step",
            "Refined layout should be re-checked with layout QA before PyWake rerun",
        ],
    }


def write_refined_layout_metadata(
    refined_df: pd.DataFrame,
    refined_layout_csv: Path,
    refined_metadata_json: Path,
) -> None:
    refined_metadata_json.parent.mkdir(parents=True, exist_ok=True)

    min_spacing_m, mean_spacing_m = compute_nearest_spacing(refined_df)
    rotor_diameter_m = float(refined_df["rotor_diameter_m"].iloc[0])

    metadata = {
        "layout_name": str(refined_df["layout_name"].iloc[0]) if "layout_name" in refined_df.columns else "refined_constraints",
        "site": str(refined_df["site"].iloc[0]) if "site" in refined_df.columns else "unknown",
        "status": "constraint-led refined layout",
        "description": (
            "Refined layout following feasibility-stage spatial constraint screening. "
            "This case is intended for rerun in PyWake as the constrained layout case."
        ),
        "coordinate_system": str(refined_df["crs"].iloc[0]) if "crs" in refined_df.columns else "unknown",
        "layout_file": str(refined_layout_csv.relative_to(REPO_ROOT)).replace("\\", "/"),
        "n_turbines": int(len(refined_df)),
        "installed_capacity_mw": float(refined_df["rated_power_mw"].sum()),
        "turbine_model": str(refined_df["turbine_model"].iloc[0]),
        "hub_height_m": float(refined_df["hub_height_m"].iloc[0]),
        "rotor_diameter_m": rotor_diameter_m,
        "rated_power_mw": float(refined_df["rated_power_mw"].iloc[0]),
        "spacing_summary": {
            "min_nearest_spacing_m": min_spacing_m,
            "mean_nearest_spacing_m": mean_spacing_m,
            "min_nearest_spacing_D": min_spacing_m / rotor_diameter_m if pd.notna(min_spacing_m) else None,
            "mean_nearest_spacing_D": mean_spacing_m / rotor_diameter_m if pd.notna(mean_spacing_m) else None,
        },
        "limitations": [
            "Feasibility-stage constraint-led revision only",
            "No terrain-optimised micro-siting applied",
            "No electrical design optimisation applied",
            "Wake performance must be reassessed separately",
        ],
    }

    with open(refined_metadata_json, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def run_layout_comparison(
    baseline_layout_csv: Path = DEFAULT_BASELINE_LAYOUT,
    refined_layout_csv: Path = DEFAULT_REFINED_LAYOUT,
) -> LayoutComparisonResult:
    baseline_df = load_layout(baseline_layout_csv)
    refined_df = load_layout(refined_layout_csv)

    turbine_changes_df = build_turbine_changes(baseline_df, refined_df)
    refined_layout_summary_df = build_refined_layout_summary(refined_df)
    summary_df = build_summary(baseline_df, refined_df, turbine_changes_df)
    metadata = build_metadata(baseline_layout_csv, refined_layout_csv, summary_df)

    return LayoutComparisonResult(
        turbine_changes_df=turbine_changes_df,
        summary_df=summary_df,
        refined_layout_summary_df=refined_layout_summary_df,
        metadata=metadata,
    )


def export_layout_comparison(
    result: LayoutComparisonResult,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> None:
    _ensure_dir(output_dir)
    result.turbine_changes_df.to_csv(output_dir / "baseline_vs_refined_turbine_changes.csv", index=False)
    result.summary_df.to_csv(output_dir / "baseline_vs_refined_summary.csv", index=False)
    result.refined_layout_summary_df.to_csv(output_dir / "refined_layout_summary.csv", index=False)

    with open(output_dir / "baseline_vs_refined_metadata.json", "w", encoding="utf-8") as f:
        json.dump(result.metadata, f, indent=2)


def print_layout_comparison_summary(result: LayoutComparisonResult) -> None:
    row = result.summary_df.iloc[0]
    print("Baseline vs refined layout comparison complete.")
    print(f"Baseline turbines: {int(row['baseline_n_turbines'])}")
    print(f"Refined turbines: {int(row['refined_n_turbines'])}")
    print(f"Delta turbines: {int(row['delta_n_turbines'])}")
    print(f"Baseline capacity: {row['baseline_installed_capacity_mw']:.1f} MW")
    print(f"Refined capacity: {row['refined_installed_capacity_mw']:.1f} MW")
    print(f"Moved turbines: {int(row['n_moved'])}")
    print(f"Removed turbines: {int(row['n_removed'])}")
    print(f"Added turbines: {int(row['n_added'])}")
    print(f"Refined minimum nearest spacing: {row['refined_min_nearest_spacing_m']:.2f} m")