from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_LAYOUT_GLOB_ROOT = REPO_ROOT / "data" / "optimisation"
DEFAULT_TURBINE_METADATA_JSON = (
    REPO_ROOT / "data" / "turbines" / "vestas_v136_4p2mw" / "turbine_metadata.json"
)
DEFAULT_TABLES_DIR = REPO_ROOT / "outputs" / "tables" / "layouts"

DEFAULT_SUMMARY_CSV = DEFAULT_TABLES_DIR / "candidate_layout_spacing_summary.csv"
DEFAULT_NEAREST_CSV = DEFAULT_TABLES_DIR / "candidate_layout_nearest_neighbour_spacing.csv"
DEFAULT_PAIRWISE_CSV = DEFAULT_TABLES_DIR / "candidate_layout_all_pairwise_spacing.csv"

REQUIRED_LAYOUT_COLUMNS = [
    "turbine_id",
    "easting_m",
    "northing_m",
]


@dataclass(frozen=True)
class LayoutSpacingResult:
    summary_df: pd.DataFrame
    nearest_df: pd.DataFrame
    pairwise_df: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Assess pairwise and nearest-neighbour spacing for optimisation candidate layout CSVs. "
            "By default, scans data/optimisation/*/candidate_layouts/*.csv."
        )
    )
    parser.add_argument(
        "--layout-csv",
        type=Path,
        nargs="*",
        default=None,
        help=(
            "Optional one or more explicit layout CSV paths. "
            "If omitted, all candidate layout CSVs under data/optimisation/*/candidate_layouts/ are used."
        ),
    )
    parser.add_argument(
        "--turbine-metadata-json",
        type=Path,
        default=DEFAULT_TURBINE_METADATA_JSON,
        help="Path to turbine metadata JSON used as the sole source of turbine technical assumptions.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=DEFAULT_SUMMARY_CSV,
        help="Path to write layout spacing summary CSV.",
    )
    parser.add_argument(
        "--nearest-csv",
        type=Path,
        default=DEFAULT_NEAREST_CSV,
        help="Path to write nearest-neighbour spacing CSV.",
    )
    parser.add_argument(
        "--pairwise-csv",
        type=Path,
        default=DEFAULT_PAIRWISE_CSV,
        help="Path to write all-pairwise spacing CSV.",
    )
    return parser.parse_args()


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def discover_candidate_layout_csvs(
    optimisation_root: Path = DEFAULT_LAYOUT_GLOB_ROOT,
) -> list[Path]:
    candidate_layout_csvs = sorted(
        p for p in optimisation_root.glob("*/candidate_layouts/*.csv") if p.is_file()
    )

    if not candidate_layout_csvs:
        raise FileNotFoundError(
            "No candidate layout CSVs found under data/optimisation/*/candidate_layouts/."
        )

    return candidate_layout_csvs


def load_turbine_metadata(turbine_metadata_json: Path) -> dict[str, Any]:
    if not turbine_metadata_json.exists():
        raise FileNotFoundError(f"Turbine metadata JSON not found: {turbine_metadata_json}")

    with turbine_metadata_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    required = ["turbine_model", "rated_power_mw", "rotor_diameter_m", "hub_height_m"]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"Turbine metadata JSON missing required keys: {missing}")

    return data


def load_layout(layout_csv: Path) -> pd.DataFrame:
    if not layout_csv.exists():
        raise FileNotFoundError(f"Layout CSV not found: {layout_csv}")

    df = pd.read_csv(layout_csv)

    missing = [c for c in REQUIRED_LAYOUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Layout CSV missing required columns: {missing}")

    df = df.copy()

    for col in ["easting_m", "northing_m"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df[REQUIRED_LAYOUT_COLUMNS].isna().any().any():
        raise ValueError("Layout CSV contains missing or non-numeric required values.")

    if df["turbine_id"].duplicated().any():
        dupes = df.loc[df["turbine_id"].duplicated(), "turbine_id"].tolist()
        raise ValueError(f"Duplicate turbine_id values found in layout: {dupes}")

    return df.reset_index(drop=True)


def _infer_layout_name(layout_csv: Path) -> str:
    return layout_csv.stem


def _infer_case_id(layout_csv: Path) -> str:
    return layout_csv.parent.parent.name


def _build_pairwise_spacing(
    layout_df: pd.DataFrame,
    layout_csv: Path,
    turbine_meta: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    rotor_diameter = float(turbine_meta["rotor_diameter_m"])
    layout_name = _infer_layout_name(layout_csv)
    case_id = _infer_case_id(layout_csv)

    coords = layout_df.set_index("turbine_id")[["easting_m", "northing_m"]].to_dict("index")

    for turbine_id_1, turbine_id_2 in combinations(layout_df["turbine_id"].tolist(), 2):
        x1 = coords[turbine_id_1]["easting_m"]
        y1 = coords[turbine_id_1]["northing_m"]
        x2 = coords[turbine_id_2]["easting_m"]
        y2 = coords[turbine_id_2]["northing_m"]

        distance_m = float(np.hypot(x2 - x1, y2 - y1))

        rows.append(
            {
                "source_layout_csv": str(layout_csv.relative_to(REPO_ROOT)).replace("\\", "/"),
                "case_id": case_id,
                "layout_name": layout_name,
                "turbine_model": str(turbine_meta["turbine_model"]),
                "rotor_diameter_m": rotor_diameter,
                "turbine_id_1": turbine_id_1,
                "turbine_id_2": turbine_id_2,
                "distance_m": distance_m,
                "distance_D": distance_m / rotor_diameter,
            }
        )

    pairwise_df = pd.DataFrame(rows).sort_values(
        by=["case_id", "layout_name", "distance_m", "turbine_id_1", "turbine_id_2"]
    ).reset_index(drop=True)

    return pairwise_df


def _build_nearest_neighbour(
    layout_df: pd.DataFrame,
    pairwise_df: pd.DataFrame,
    layout_csv: Path,
    turbine_meta: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    layout_name = _infer_layout_name(layout_csv)
    case_id = _infer_case_id(layout_csv)

    for turbine_id in layout_df["turbine_id"]:
        candidate_rows = pairwise_df[
            (pairwise_df["turbine_id_1"] == turbine_id) | (pairwise_df["turbine_id_2"] == turbine_id)
        ].copy()

        nearest_row = candidate_rows.sort_values("distance_m", ascending=True).iloc[0]

        nearest_turbine_id = (
            nearest_row["turbine_id_2"]
            if nearest_row["turbine_id_1"] == turbine_id
            else nearest_row["turbine_id_1"]
        )

        rows.append(
            {
                "source_layout_csv": str(layout_csv.relative_to(REPO_ROOT)).replace("\\", "/"),
                "case_id": case_id,
                "layout_name": layout_name,
                "turbine_model": str(turbine_meta["turbine_model"]),
                "rotor_diameter_m": float(turbine_meta["rotor_diameter_m"]),
                "turbine_id": turbine_id,
                "nearest_turbine_id": nearest_turbine_id,
                "nearest_distance_m": float(nearest_row["distance_m"]),
                "nearest_distance_D": float(nearest_row["distance_D"]),
            }
        )

    nearest_df = pd.DataFrame(rows).sort_values(
        by=["case_id", "layout_name", "nearest_distance_m", "turbine_id"]
    ).reset_index(drop=True)

    return nearest_df


def _build_summary(
    layout_df: pd.DataFrame,
    pairwise_df: pd.DataFrame,
    nearest_df: pd.DataFrame,
    layout_csv: Path,
    turbine_meta: dict[str, Any],
) -> pd.DataFrame:
    layout_name = _infer_layout_name(layout_csv)
    case_id = _infer_case_id(layout_csv)

    rotor_diameter = float(turbine_meta["rotor_diameter_m"])
    rated_power_mw = float(turbine_meta["rated_power_mw"])

    summary_row: dict[str, Any] = {
        "source_layout_csv": str(layout_csv.relative_to(REPO_ROOT)).replace("\\", "/"),
        "case_id": case_id,
        "layout_name": layout_name,
        "turbine_model": str(turbine_meta["turbine_model"]),
        "hub_height_m": float(turbine_meta["hub_height_m"]),
        "rotor_diameter_m": rotor_diameter,
        "rated_power_mw": rated_power_mw,
        "n_turbines": int(len(layout_df)),
        "installed_capacity_mw": float(len(layout_df) * rated_power_mw),
        "min_pairwise_spacing_m": float(pairwise_df["distance_m"].min()),
        "mean_pairwise_spacing_m": float(pairwise_df["distance_m"].mean()),
        "max_pairwise_spacing_m": float(pairwise_df["distance_m"].max()),
        "min_nearest_spacing_m": float(nearest_df["nearest_distance_m"].min()),
        "mean_nearest_spacing_m": float(nearest_df["nearest_distance_m"].mean()),
        "max_nearest_spacing_m": float(nearest_df["nearest_distance_m"].max()),
        "min_pairwise_spacing_D": float(pairwise_df["distance_D"].min()),
        "mean_pairwise_spacing_D": float(pairwise_df["distance_D"].mean()),
        "max_pairwise_spacing_D": float(pairwise_df["distance_D"].max()),
        "min_nearest_spacing_D": float(nearest_df["nearest_distance_D"].min()),
        "mean_nearest_spacing_D": float(nearest_df["nearest_distance_D"].mean()),
        "max_nearest_spacing_D": float(nearest_df["nearest_distance_D"].max()),
    }

    return pd.DataFrame([summary_row])


def run_layout_spacing(
    layout_csvs: list[Path] | None = None,
    turbine_metadata_json: Path = DEFAULT_TURBINE_METADATA_JSON,
) -> LayoutSpacingResult:
    if layout_csvs is None:
        layout_csvs = discover_candidate_layout_csvs()

    turbine_meta = load_turbine_metadata(turbine_metadata_json)

    summary_frames: list[pd.DataFrame] = []
    nearest_frames: list[pd.DataFrame] = []
    pairwise_frames: list[pd.DataFrame] = []

    for layout_csv in layout_csvs:
        layout_df = load_layout(layout_csv)
        pairwise_df = _build_pairwise_spacing(layout_df, layout_csv, turbine_meta)
        nearest_df = _build_nearest_neighbour(layout_df, pairwise_df, layout_csv, turbine_meta)
        summary_df = _build_summary(layout_df, pairwise_df, nearest_df, layout_csv, turbine_meta)

        pairwise_frames.append(pairwise_df)
        nearest_frames.append(nearest_df)
        summary_frames.append(summary_df)

    return LayoutSpacingResult(
        summary_df=pd.concat(summary_frames, ignore_index=True),
        nearest_df=pd.concat(nearest_frames, ignore_index=True),
        pairwise_df=pd.concat(pairwise_frames, ignore_index=True),
    )


def export_layout_spacing(
    result: LayoutSpacingResult,
    summary_csv: Path = DEFAULT_SUMMARY_CSV,
    nearest_csv: Path = DEFAULT_NEAREST_CSV,
    pairwise_csv: Path = DEFAULT_PAIRWISE_CSV,
) -> None:
    _ensure_parent_dir(summary_csv)
    _ensure_parent_dir(nearest_csv)
    _ensure_parent_dir(pairwise_csv)

    result.summary_df.to_csv(summary_csv, index=False)
    result.nearest_df.to_csv(nearest_csv, index=False)
    result.pairwise_df.to_csv(pairwise_csv, index=False)


def print_layout_spacing_summary(result: LayoutSpacingResult) -> None:
    print("Layout spacing assessment complete.")
    print(f"Layouts assessed: {len(result.summary_df)}")

    display_cols = [
        "case_id",
        "layout_name",
        "n_turbines",
        "min_nearest_spacing_m",
        "mean_nearest_spacing_m",
        "min_nearest_spacing_D",
        "mean_nearest_spacing_D",
    ]

    print("\nLayout spacing summary:")
    print(result.summary_df.loc[:, display_cols].to_string(index=False))


def main() -> int:
    try:
        args = parse_args()

        if args.layout_csv:
            layout_csvs = [p.resolve() for p in args.layout_csv]
        else:
            layout_csvs = [p.resolve() for p in discover_candidate_layout_csvs()]

        result = run_layout_spacing(
            layout_csvs=layout_csvs,
            turbine_metadata_json=args.turbine_metadata_json.resolve(),
        )

        export_layout_spacing(
            result=result,
            summary_csv=args.summary_csv.resolve(),
            nearest_csv=args.nearest_csv.resolve(),
            pairwise_csv=args.pairwise_csv.resolve(),
        )

        print_layout_spacing_summary(result)
        return 0

    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())