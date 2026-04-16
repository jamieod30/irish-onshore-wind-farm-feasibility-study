from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_LAYOUT_CSV = REPO_ROOT / "data" / "layouts" / "baseline_aligned" / "layout_baseline_aligned.csv"
DEFAULT_HITS_CSV = REPO_ROOT / "data" / "constraints" / "derived" / "turbine_constraint_hits.csv"
DEFAULT_STATUS_CSV = REPO_ROOT / "data" / "constraints" / "derived" / "turbine_constraint_status.csv"
DEFAULT_SCREENED_LAYOUT_CSV = REPO_ROOT / "data" / "layouts" / "constraint_screened" / "layout_constraint_screened.csv"
DEFAULT_SCREENED_METADATA_JSON = REPO_ROOT / "data" / "layouts" / "constraint_screened" / "layout_metadata.json"
DEFAULT_TABLES_DIR = REPO_ROOT / "outputs" / "tables" / "constraints"

REQUIRED_LAYOUT_COLUMNS = [
    "turbine_id",
    "easting_m",
    "northing_m",
    "hub_height_m",
    "rotor_diameter_m",
    "rated_power_mw",
    "turbine_model",
]

REQUIRED_HITS_COLUMNS = [
    "turbine_id",
    "constraint_name",
    "constraint_group",
]


@dataclass(frozen=True)
class ConstraintScreeningResult:
    status_df: pd.DataFrame
    hits_df: pd.DataFrame
    screened_layout_df: pd.DataFrame
    summary_df: pd.DataFrame
    metadata: dict[str, Any]


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_layout(layout_csv: Path) -> pd.DataFrame:
    if not layout_csv.exists():
        raise FileNotFoundError(f"Layout CSV not found: {layout_csv}")

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
        raise ValueError("Layout CSV contains missing or non-numeric required values.")

    if df["turbine_id"].duplicated().any():
        dupes = df.loc[df["turbine_id"].duplicated(), "turbine_id"].tolist()
        raise ValueError(f"Duplicate turbine_id values found in layout: {dupes}")

    return df.reset_index(drop=True)


def load_constraint_hits(hits_csv: Path) -> pd.DataFrame:
    if not hits_csv.exists():
        raise FileNotFoundError(
            f"Constraint hits CSV not found: {hits_csv}\n"
            "Expected a QGIS-exported turbine constraint hit table."
        )

    df = pd.read_csv(hits_csv)
    missing = [c for c in REQUIRED_HITS_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Constraint hits CSV missing required columns: {missing}")

    df = df.copy()
    for col in REQUIRED_HITS_COLUMNS:
        df[col] = df[col].astype(str).str.strip()

    if df[REQUIRED_HITS_COLUMNS].replace("", np.nan).isna().any().any():
        raise ValueError("Constraint hits CSV contains blank required values.")

    optional_columns = ["source_layer", "buffer_m", "severity", "notes"]
    for col in optional_columns:
        if col not in df.columns:
            df[col] = pd.NA

    if "buffer_m" in df.columns:
        df["buffer_m"] = pd.to_numeric(df["buffer_m"], errors="coerce")

    return df.reset_index(drop=True)


def validate_hits_against_layout(layout_df: pd.DataFrame, hits_df: pd.DataFrame) -> None:
    layout_ids = set(layout_df["turbine_id"].astype(str))
    hit_ids = set(hits_df["turbine_id"].astype(str))

    unknown_ids = sorted(hit_ids - layout_ids)
    if unknown_ids:
        raise ValueError(
            f"Constraint hits contain turbine IDs not present in layout: {unknown_ids}"
        )


def build_constraint_status(layout_df: pd.DataFrame, hits_df: pd.DataFrame) -> pd.DataFrame:
    if hits_df.empty:
        grouped = pd.DataFrame(columns=[
            "turbine_id",
            "n_constraint_hits",
            "constraint_names",
            "constraint_groups",
            "source_layers",
            "max_buffer_m",
            "severity_list",
            "notes_combined",
        ])
    else:
        grouped = (
            hits_df.groupby("turbine_id", dropna=False)
            .agg(
                n_constraint_hits=("constraint_name", "size"),
                constraint_names=("constraint_name", lambda s: "; ".join(sorted(set(map(str, s))))),
                constraint_groups=("constraint_group", lambda s: "; ".join(sorted(set(map(str, s))))),
                source_layers=("source_layer", lambda s: "; ".join(sorted(set(map(str, s.dropna()))))),
                max_buffer_m=("buffer_m", "max"),
                severity_list=("severity", lambda s: "; ".join(sorted(set(map(str, s.dropna()))))),
                notes_combined=("notes", lambda s: " | ".join([str(v) for v in s.dropna() if str(v).strip()])),
            )
            .reset_index()
        )

    status_df = layout_df.merge(grouped, on="turbine_id", how="left")

    status_df["n_constraint_hits"] = status_df["n_constraint_hits"].fillna(0).astype(int)
    status_df["constraint_names"] = status_df["constraint_names"].fillna("")
    status_df["constraint_groups"] = status_df["constraint_groups"].fillna("")
    status_df["source_layers"] = status_df["source_layers"].fillna("")
    status_df["severity_list"] = status_df["severity_list"].fillna("")
    status_df["notes_combined"] = status_df["notes_combined"].fillna("")

    status_df["constraint_status"] = np.where(
        status_df["n_constraint_hits"] > 0,
        "infringed",
        "compliant",
    )
    status_df["recommended_action"] = np.where(
        status_df["constraint_status"] == "infringed",
        "move_or_remove",
        "retain",
    )

    cols = [
        "turbine_id",
        "constraint_status",
        "recommended_action",
        "n_constraint_hits",
        "constraint_names",
        "constraint_groups",
        "source_layers",
        "max_buffer_m",
        "severity_list",
        "notes_combined",
        "easting_m",
        "northing_m",
        "hub_height_m",
        "rotor_diameter_m",
        "rated_power_mw",
        "turbine_model",
    ]

    extra_cols = [c for c in ["layout_name", "site", "crs"] if c in status_df.columns]
    ordered_cols = cols + extra_cols

    return status_df.loc[:, ordered_cols].sort_values("turbine_id").reset_index(drop=True)


def build_screened_layout(status_df: pd.DataFrame) -> pd.DataFrame:
    screened_df = status_df.loc[status_df["constraint_status"] == "compliant"].copy()

    keep_cols = REQUIRED_LAYOUT_COLUMNS + [c for c in ["layout_name", "site", "crs"] if c in screened_df.columns]
    screened_df = screened_df.loc[:, keep_cols].reset_index(drop=True)

    if "layout_name" in screened_df.columns:
        screened_df["layout_name"] = "constraint_screened"

    return screened_df


def build_summary(status_df: pd.DataFrame) -> pd.DataFrame:
    n_total = int(len(status_df))
    n_infringed = int((status_df["constraint_status"] == "infringed").sum())
    n_compliant = int((status_df["constraint_status"] == "compliant").sum())
    installed_capacity_mw = float(status_df["rated_power_mw"].sum())
    screened_capacity_mw = float(
        status_df.loc[status_df["constraint_status"] == "compliant", "rated_power_mw"].sum()
    )

    summary = pd.DataFrame(
        [
            {
                "case_name": "baseline_constraint_screening",
                "n_turbines_total": n_total,
                "n_turbines_compliant": n_compliant,
                "n_turbines_infringed": n_infringed,
                "installed_capacity_mw": installed_capacity_mw,
                "screened_capacity_mw": screened_capacity_mw,
                "capacity_removed_mw_if_excluded": installed_capacity_mw - screened_capacity_mw,
                "screening_note": (
                    "Constraint-screened layout is an interim compliant subset only. "
                    "It is not the final refined layout unless turbines are intentionally removed."
                ),
            }
        ]
    )
    return summary


def build_metadata(
    layout_df: pd.DataFrame,
    status_df: pd.DataFrame,
    hits_csv: Path,
) -> dict[str, Any]:
    first = layout_df.iloc[0]
    n_total = int(len(layout_df))
    n_infringed = int((status_df["constraint_status"] == "infringed").sum())
    n_compliant = int((status_df["constraint_status"] == "compliant").sum())

    metadata = {
        "layout_name": "constraint_screened",
        "site": str(first["site"]) if "site" in layout_df.columns else "unknown",
        "status": "interim constraint-screened layout",
        "description": (
            "Automatic export of turbines not flagged by the feasibility-stage "
            "constraint screening process. This file is intended as an interim "
            "engineering output, not a replacement for a manually refined layout."
        ),
        "coordinate_system": str(first["crs"]) if "crs" in layout_df.columns else "unknown",
        "source_layout_name": str(first["layout_name"]) if "layout_name" in layout_df.columns else "unknown",
        "constraint_hits_file": str(hits_csv.relative_to(REPO_ROOT)).replace("\\", "/"),
        "n_turbines_baseline": n_total,
        "n_turbines_compliant": n_compliant,
        "n_turbines_infringed": n_infringed,
        "installed_capacity_mw_baseline": float(layout_df["rated_power_mw"].sum()),
        "installed_capacity_mw_constraint_screened": float(
            status_df.loc[status_df["constraint_status"] == "compliant", "rated_power_mw"].sum()
        ),
        "assumptions": [
            "Feasibility-stage screening only",
            "Constraint screening based on GIS exclusion overlaps",
            "No wake optimisation included",
            "No terrain micro-siting optimisation included",
        ],
    }
    return metadata


def run_constraint_screening(
    layout_csv: Path = DEFAULT_LAYOUT_CSV,
    hits_csv: Path = DEFAULT_HITS_CSV,
) -> ConstraintScreeningResult:
    layout_df = load_layout(layout_csv)
    hits_df = load_constraint_hits(hits_csv)

    validate_hits_against_layout(layout_df, hits_df)

    status_df = build_constraint_status(layout_df, hits_df)
    screened_layout_df = build_screened_layout(status_df)
    summary_df = build_summary(status_df)
    metadata = build_metadata(layout_df, status_df, hits_csv)

    return ConstraintScreeningResult(
        status_df=status_df,
        hits_df=hits_df,
        screened_layout_df=screened_layout_df,
        summary_df=summary_df,
        metadata=metadata,
    )


def export_constraint_screening(
    result: ConstraintScreeningResult,
    status_csv: Path = DEFAULT_STATUS_CSV,
    screened_layout_csv: Path = DEFAULT_SCREENED_LAYOUT_CSV,
    screened_metadata_json: Path = DEFAULT_SCREENED_METADATA_JSON,
    tables_dir: Path = DEFAULT_TABLES_DIR,
) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)
    _ensure_parent_dir(status_csv)
    _ensure_parent_dir(screened_layout_csv)
    _ensure_parent_dir(screened_metadata_json)

    result.hits_df.to_csv(tables_dir / "baseline_constraint_hits.csv", index=False)
    result.status_df.to_csv(status_csv, index=False)
    result.status_df.to_csv(tables_dir / "baseline_constraint_status.csv", index=False)
    result.screened_layout_df.to_csv(screened_layout_csv, index=False)
    result.summary_df.to_csv(tables_dir / "constraint_screening_summary.csv", index=False)

    with open(screened_metadata_json, "w", encoding="utf-8") as f:
        json.dump(result.metadata, f, indent=2)


def print_constraint_screening_summary(result: ConstraintScreeningResult) -> None:
    row = result.summary_df.iloc[0]
    print("Constraint screening complete.")
    print(f"Total turbines: {int(row['n_turbines_total'])}")
    print(f"Compliant turbines: {int(row['n_turbines_compliant'])}")
    print(f"Infringed turbines: {int(row['n_turbines_infringed'])}")
    print(f"Baseline installed capacity: {row['installed_capacity_mw']:.1f} MW")
    print(f"Constraint-screened capacity: {row['screened_capacity_mw']:.1f} MW")
    print(f"Capacity removed if infringed turbines are excluded: {row['capacity_removed_mw_if_excluded']:.1f} MW")

    infringed = result.status_df.loc[
        result.status_df["constraint_status"] == "infringed",
        ["turbine_id", "constraint_groups", "constraint_names", "recommended_action"],
    ]
    if infringed.empty:
        print("No baseline turbines were flagged by the supplied constraint hit table.")
    else:
        print("\nFlagged turbines:")
        print(infringed.to_string(index=False))