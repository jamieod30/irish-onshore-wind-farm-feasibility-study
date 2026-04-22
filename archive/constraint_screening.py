from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_LAYOUT_CSV = (
    REPO_ROOT / "data" / "optimisation" / "baseline" / "layout" / "layout_baseline_aligned.csv"
)

DEFAULT_HITS_CSV = (
    REPO_ROOT / "data" / "constraints" / "derived" / "turbine_constraint_hits.csv"
)

DEFAULT_STATUS_CSV = (
    REPO_ROOT / "outputs" / "tables" / "constraints" / "baseline_constraint_status.csv"
)

DEFAULT_SCREENED_LAYOUT_CSV = (
    REPO_ROOT
    / "data"
    / "optimisation"
    / "baseline"
    / "constraint_screened"
    / "layout_constraint_screened.csv"
)

DEFAULT_SCREENED_METADATA_JSON = (
    REPO_ROOT
    / "data"
    / "optimisation"
    / "baseline"
    / "constraint_screened"
    / "layout_metadata.json"
)

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

    for col in [
        "easting_m",
        "northing_m",
        "hub_height_m",
        "rotor_diameter_m",
        "rated_power_mw",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df[REQUIRED_LAYOUT_COLUMNS].isna().any().any():
        raise ValueError("Layout CSV contains missing or invalid required values.")

    if df["turbine_id"].duplicated().any():
        dupes = df.loc[df["turbine_id"].duplicated(), "turbine_id"].tolist()
        raise ValueError(f"Duplicate turbine IDs found: {dupes}")

    return df.reset_index(drop=True)


def load_constraint_hits(hits_csv: Path) -> pd.DataFrame:
    if not hits_csv.exists():
        raise FileNotFoundError(f"Constraint hits CSV not found: {hits_csv}")

    df = pd.read_csv(hits_csv)

    missing = [c for c in REQUIRED_HITS_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Constraint hits CSV missing required columns: {missing}")

    for col in REQUIRED_HITS_COLUMNS:
        df[col] = df[col].astype(str).str.strip()

    if df[REQUIRED_HITS_COLUMNS].replace("", np.nan).isna().any().any():
        raise ValueError("Constraint hits CSV contains blank required values.")

    for col in ["source_layer", "buffer_m", "severity", "notes"]:
        if col not in df.columns:
            df[col] = pd.NA

    df["buffer_m"] = pd.to_numeric(df["buffer_m"], errors="coerce")

    return df.reset_index(drop=True)


def validate_hits_against_layout(layout_df: pd.DataFrame, hits_df: pd.DataFrame) -> None:
    layout_ids = set(layout_df["turbine_id"].astype(str))
    hit_ids = set(hits_df["turbine_id"].astype(str))

    unknown = sorted(hit_ids - layout_ids)
    if unknown:
        raise ValueError(f"Constraint hits reference unknown turbine IDs: {unknown}")


def build_constraint_status(layout_df: pd.DataFrame, hits_df: pd.DataFrame) -> pd.DataFrame:
    if hits_df.empty:
        grouped = pd.DataFrame(columns=["turbine_id", "n_constraint_hits"])
    else:
        grouped = (
            hits_df.groupby("turbine_id", dropna=False)
            .agg(
                n_constraint_hits=("constraint_name", "size"),
                constraint_names=("constraint_name", lambda s: "; ".join(sorted(set(s)))),
                constraint_groups=("constraint_group", lambda s: "; ".join(sorted(set(s)))),
                source_layers=("source_layer", lambda s: "; ".join(sorted(set(s.dropna().astype(str))))),
                max_buffer_m=("buffer_m", "max"),
                severity_list=("severity", lambda s: "; ".join(sorted(set(s.dropna().astype(str))))),
                notes_combined=("notes", lambda s: " | ".join([str(v) for v in s.dropna()])),
            )
            .reset_index()
        )

    df = layout_df.merge(grouped, on="turbine_id", how="left")

    df["n_constraint_hits"] = df["n_constraint_hits"].fillna(0).astype(int)

    for col in [
        "constraint_names",
        "constraint_groups",
        "source_layers",
        "severity_list",
        "notes_combined",
    ]:
        if col in df.columns:
            df[col] = df[col].fillna("")
        else:
            df[col] = ""

    df["constraint_status"] = np.where(df["n_constraint_hits"] > 0, "infringed", "compliant")
    df["recommended_action"] = np.where(df["n_constraint_hits"] > 0, "move_or_remove", "retain")

    return df.sort_values("turbine_id").reset_index(drop=True)


def build_screened_layout(status_df: pd.DataFrame) -> pd.DataFrame:
    keep = REQUIRED_LAYOUT_COLUMNS + [c for c in ["layout_name", "site", "crs"] if c in status_df.columns]

    out = status_df.loc[status_df["constraint_status"] == "compliant", keep].copy()

    if "layout_name" in out.columns:
        out["layout_name"] = "constraint_screened"

    return out.reset_index(drop=True)


def build_summary(status_df: pd.DataFrame) -> pd.DataFrame:
    total = len(status_df)
    compliant = int((status_df["constraint_status"] == "compliant").sum())
    infringed = int((status_df["constraint_status"] == "infringed").sum())

    installed = float(status_df["rated_power_mw"].sum())
    screened = float(
        status_df.loc[
            status_df["constraint_status"] == "compliant",
            "rated_power_mw",
        ].sum()
    )

    return pd.DataFrame(
        [
            {
                "case_name": "baseline_constraint_screening",
                "n_turbines_total": total,
                "n_turbines_compliant": compliant,
                "n_turbines_infringed": infringed,
                "installed_capacity_mw": installed,
                "screened_capacity_mw": screened,
                "capacity_removed_mw_if_excluded": installed - screened,
            }
        ]
    )


def build_metadata(layout_df: pd.DataFrame, status_df: pd.DataFrame, hits_csv: Path) -> dict[str, Any]:
    first = layout_df.iloc[0]

    return {
        "layout_name": "constraint_screened",
        "site": str(first["site"]) if "site" in layout_df.columns else "unknown",
        "source_layout_name": str(first["layout_name"]) if "layout_name" in layout_df.columns else "unknown",
        "constraint_hits_file": str(hits_csv.relative_to(REPO_ROOT)).replace("\\", "/"),
        "n_turbines_baseline": int(len(layout_df)),
        "n_turbines_compliant": int((status_df["constraint_status"] == "compliant").sum()),
        "n_turbines_infringed": int((status_df["constraint_status"] == "infringed").sum()),
    }


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
    print(f"Baseline capacity: {row['installed_capacity_mw']:.1f} MW")
    print(f"Screened capacity: {row['screened_capacity_mw']:.1f} MW")