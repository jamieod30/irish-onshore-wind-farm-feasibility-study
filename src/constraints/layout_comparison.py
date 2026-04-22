from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

OPTIMISATION_DIR = REPO_ROOT / "data" / "optimisation"
DEFAULT_BASELINE_LAYOUT = (
    OPTIMISATION_DIR / "baseline" / "layout" / "layout_baseline_aligned.csv"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "tables" / "optimisation"
DEFAULT_TURBINE_METADATA_JSON = (
    REPO_ROOT / "data" / "turbines" / "vestas_v136_4p2mw" / "turbine_metadata.json"
)

REQUIRED_BASE_LAYOUT_COLUMNS = [
    "turbine_id",
    "easting_m",
    "northing_m",
]


@dataclass(frozen=True)
class LayoutComparisonResult:
    case_id: str
    turbine_changes_df: pd.DataFrame
    summary_df: pd.DataFrame
    refined_layout_summary_df: pd.DataFrame
    metadata: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare one or more optimisation case layouts against the baseline layout. "
            "Use --case <case_id>, --case ?, or --all."
        )
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--case",
        dest="cases",
        action="append",
        help="Case folder name under data/optimisation/ (repeatable). Use '?' to list available cases.",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Run layout comparison for all available optimisation cases with selected layouts.",
    )
    parser.add_argument(
        "--baseline-layout-csv",
        type=Path,
        default=DEFAULT_BASELINE_LAYOUT,
        help="Path to baseline layout CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for optimisation comparison outputs.",
    )
    parser.add_argument(
        "--turbine-metadata-json",
        type=Path,
        default=DEFAULT_TURBINE_METADATA_JSON,
        help="Path to turbine metadata JSON used to enrich layouts when needed.",
    )
    return parser.parse_args()


def list_available_cases() -> list[str]:
    case_ids: list[str] = []
    if not OPTIMISATION_DIR.exists():
        return case_ids

    for path in sorted(OPTIMISATION_DIR.iterdir()):
        if not path.is_dir():
            continue
        if path.name in {"baseline", "comparisons"}:
            continue
        selected_layout_dir = path / "selected_layout"
        if resolve_case_layout_csv(path.name, must_exist=False) is not None:
            case_ids.append(path.name)

    return case_ids


def resolve_case_layout_csv(case_id: str, must_exist: bool = True) -> Path | None:
    case_dir = OPTIMISATION_DIR / case_id

    candidates = [
        case_dir / "selected_layout" / f"layout_{case_id}_aligned.csv",
        case_dir / "selected_layout" / f"layout_{case_id}.csv",
        case_dir / "selected_layout" / f"layout_{case_id}_selected.csv",
        case_dir / "selected_layout" / f"layout_{case_id}_refined.csv",
    ]

    existing = [p for p in candidates if p.exists()]
    if len(existing) == 1:
        return existing[0]
    if len(existing) > 1:
        raise ValueError(
            f"Multiple candidate layout CSVs found for case '{case_id}': "
            f"{[str(p.relative_to(REPO_ROOT)).replace(chr(92), '/') for p in existing]}"
        )

    selected_layout_dir = case_dir / "selected_layout"
    discovered = sorted(selected_layout_dir.glob("*.csv")) if selected_layout_dir.exists() else []
    if len(discovered) == 1:
        return discovered[0]

    if must_exist:
        raise FileNotFoundError(
            f"Could not resolve layout CSV for case '{case_id}'. "
            f"Checked standard locations under {case_dir / 'selected_layout'}."
        )

    return None


def refined_metadata_path_for_case(case_id: str) -> Path:
    return OPTIMISATION_DIR / case_id / "metadata" / "layout_metadata.json"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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


def infer_case_id_from_layout_path(layout_csv: Path) -> str:
    parts = layout_csv.resolve().parts
    try:
        idx = parts.index("optimisation")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    except ValueError:
        pass
    return "unknown"


def infer_site_name(layout_csv: Path) -> str:
    case_id = infer_case_id_from_layout_path(layout_csv)
    return "baseline" if case_id == "baseline" else case_id


def load_layout(layout_csv: Path, turbine_metadata_json: Path) -> pd.DataFrame:
    if not layout_csv.exists():
        raise FileNotFoundError(f"Layout file not found: {layout_csv}")

    df = pd.read_csv(layout_csv)
    missing = [c for c in REQUIRED_BASE_LAYOUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Layout CSV missing required columns: {missing}")

    df = df.copy()

    for col in ["easting_m", "northing_m"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df[REQUIRED_BASE_LAYOUT_COLUMNS].isna().any().any():
        raise ValueError("Layout contains missing or non-numeric required geometry values.")

    if df["turbine_id"].duplicated().any():
        dupes = df.loc[df["turbine_id"].duplicated(), "turbine_id"].tolist()
        raise ValueError(f"Duplicate turbine IDs found: {dupes}")

    turbine_meta = load_turbine_metadata(turbine_metadata_json)

    defaults: dict[str, Any] = {
        "hub_height_m": float(turbine_meta["hub_height_m"]),
        "rotor_diameter_m": float(turbine_meta["rotor_diameter_m"]),
        "rated_power_mw": float(turbine_meta["rated_power_mw"]),
        "turbine_model": str(turbine_meta["turbine_model"]),
        "site": infer_site_name(layout_csv),
        "layout_name": layout_csv.stem,
    }

    for col, default_value in defaults.items():
        if col not in df.columns:
            df[col] = default_value
        else:
            if pd.api.types.is_numeric_dtype(pd.Series([default_value])):
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default_value)
            else:
                df[col] = df[col].astype("object").where(
                    df[col].notna() & (df[col].astype(str).str.strip() != ""),
                    default_value,
                )

    for col in ["hub_height_m", "rotor_diameter_m", "rated_power_mw"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    required_after_enrichment = REQUIRED_BASE_LAYOUT_COLUMNS + [
        "hub_height_m",
        "rotor_diameter_m",
        "rated_power_mw",
        "turbine_model",
    ]
    if df[required_after_enrichment].isna().any().any():
        raise ValueError("Layout contains missing values after enrichment from turbine metadata.")

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
        (merged["delta_easting_m"].fillna(0.0) ** 2)
        + (merged["delta_northing_m"].fillna(0.0) ** 2)
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


def build_refined_layout_summary(refined_df: pd.DataFrame, case_id: str) -> pd.DataFrame:
    min_spacing_m, mean_spacing_m = compute_nearest_spacing(refined_df)
    rotor_diameter_m = float(refined_df["rotor_diameter_m"].iloc[0])
    installed_capacity_mw = float(refined_df["rated_power_mw"].sum())

    summary = pd.DataFrame(
        [
            {
                "case_id": case_id,
                "layout_name": (
                    str(refined_df["layout_name"].iloc[0])
                    if "layout_name" in refined_df.columns
                    else case_id
                ),
                "site": str(refined_df["site"].iloc[0]) if "site" in refined_df.columns else case_id,
                "n_turbines": int(len(refined_df)),
                "installed_capacity_mw": installed_capacity_mw,
                "hub_height_m": float(refined_df["hub_height_m"].iloc[0]),
                "rotor_diameter_m": rotor_diameter_m,
                "rated_power_mw": float(refined_df["rated_power_mw"].iloc[0]),
                "min_nearest_spacing_m": min_spacing_m,
                "mean_nearest_spacing_m": mean_spacing_m,
                "min_nearest_spacing_D": (
                    min_spacing_m / rotor_diameter_m if pd.notna(min_spacing_m) else np.nan
                ),
                "mean_nearest_spacing_D": (
                    mean_spacing_m / rotor_diameter_m if pd.notna(mean_spacing_m) else np.nan
                ),
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
    case_id: str,
) -> pd.DataFrame:
    baseline_capacity = float(baseline_df["rated_power_mw"].sum())
    refined_capacity = float(refined_df["rated_power_mw"].sum())

    baseline_min_spacing_m, baseline_mean_spacing_m = compute_nearest_spacing(baseline_df)
    refined_min_spacing_m, refined_mean_spacing_m = compute_nearest_spacing(refined_df)

    summary = pd.DataFrame(
        [
            {
                "comparison_id": f"baseline_vs_{case_id}",
                "case_id": case_id,
                "baseline_layout_name": (
                    str(baseline_df["layout_name"].iloc[0])
                    if "layout_name" in baseline_df.columns
                    else "baseline_aligned"
                ),
                "refined_layout_name": (
                    str(refined_df["layout_name"].iloc[0])
                    if "layout_name" in refined_df.columns
                    else case_id
                ),
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
                        "movement_distance_m",
                    ].mean()
                )
                if (turbine_changes_df["change_type"] == "moved").any()
                else 0.0,
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
    case_id: str,
) -> dict[str, Any]:
    row = summary_df.iloc[0]
    return {
        "comparison_id": f"baseline_vs_{case_id}",
        "case_id": case_id,
        "baseline_layout_file": str(baseline_layout_csv.relative_to(REPO_ROOT)).replace("\\", "/"),
        "refined_layout_file": str(refined_layout_csv.relative_to(REPO_ROOT)).replace("\\", "/"),
        "comparison_type": "baseline_vs_selected_case",
        "summary": row.to_dict(),
        "notes": [
            "Comparison is geometric and capacity-based only",
            "No wake model rerun included in this step",
            "Selected layout should be re-checked with layout QA before PyWake rerun",
        ],
    }


def write_refined_layout_metadata(
    refined_df: pd.DataFrame,
    refined_layout_csv: Path,
    refined_metadata_json: Path,
    case_id: str,
) -> None:
    refined_metadata_json.parent.mkdir(parents=True, exist_ok=True)

    min_spacing_m, mean_spacing_m = compute_nearest_spacing(refined_df)
    rotor_diameter_m = float(refined_df["rotor_diameter_m"].iloc[0])

    metadata = {
        "case_id": case_id,
        "layout_name": (
            str(refined_df["layout_name"].iloc[0])
            if "layout_name" in refined_df.columns
            else case_id
        ),
        "site": str(refined_df["site"].iloc[0]) if "site" in refined_df.columns else case_id,
        "status": "selected optimisation case layout",
        "description": (
            "Selected layout for optimisation case comparison against the fixed baseline benchmark. "
            "This case is intended for downstream QA and PyWake rerun."
        ),
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
            "min_nearest_spacing_D": (
                min_spacing_m / rotor_diameter_m if pd.notna(min_spacing_m) else None
            ),
            "mean_nearest_spacing_D": (
                mean_spacing_m / rotor_diameter_m if pd.notna(mean_spacing_m) else None
            ),
        },
        "limitations": [
            "Geometric/selection-stage case only",
            "No terrain-optimised micro-siting applied unless explicitly documented elsewhere",
            "No electrical design optimisation applied here",
            "Wake performance must be reassessed separately",
        ],
    }

    with open(refined_metadata_json, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def run_layout_comparison(
    baseline_layout_csv: Path,
    refined_layout_csv: Path,
    case_id: str,
    turbine_metadata_json: Path,
) -> LayoutComparisonResult:
    baseline_df = load_layout(baseline_layout_csv, turbine_metadata_json)
    refined_df = load_layout(refined_layout_csv, turbine_metadata_json)

    turbine_changes_df = build_turbine_changes(baseline_df, refined_df)
    refined_layout_summary_df = build_refined_layout_summary(refined_df, case_id)
    summary_df = build_summary(baseline_df, refined_df, turbine_changes_df, case_id)
    metadata = build_metadata(baseline_layout_csv, refined_layout_csv, summary_df, case_id)

    return LayoutComparisonResult(
        case_id=case_id,
        turbine_changes_df=turbine_changes_df,
        summary_df=summary_df,
        refined_layout_summary_df=refined_layout_summary_df,
        metadata=metadata,
    )


def comparison_bundle_stem(case_ids: list[str]) -> str:
    ordered = sorted(case_ids)
    return "layout_comparison_" + "_".join(["baseline", *ordered])


def export_layout_comparison(
    results: list[LayoutComparisonResult],
    output_dir: Path,
) -> list[Path]:
    _ensure_dir(output_dir)

    bundle_stem = comparison_bundle_stem([r.case_id for r in results])

    summary_df = pd.concat([r.summary_df for r in results], ignore_index=True)
    refined_layout_summary_df = pd.concat([r.refined_layout_summary_df for r in results], ignore_index=True)
    turbine_changes_df = pd.concat(
        [
            r.turbine_changes_df.assign(case_id=r.case_id, comparison_id=f"baseline_vs_{r.case_id}")
            for r in results
        ],
        ignore_index=True,
    )

    metadata = {
        "comparison_bundle": bundle_stem,
        "case_ids": sorted([r.case_id for r in results]),
        "comparisons": [r.metadata for r in results],
    }

    out_paths = [
        output_dir / f"{bundle_stem}_summary.csv",
        output_dir / f"{bundle_stem}_turbine_changes.csv",
        output_dir / f"{bundle_stem}_refined_layout_summary.csv",
        output_dir / f"{bundle_stem}_metadata.json",
    ]

    summary_df.to_csv(out_paths[0], index=False)
    turbine_changes_df.to_csv(out_paths[1], index=False)
    refined_layout_summary_df.to_csv(out_paths[2], index=False)

    with open(out_paths[3], "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return out_paths


def print_layout_comparison_summary(
    results: list[LayoutComparisonResult],
    output_paths: list[Path],
) -> None:
    print("Baseline vs selected case layout comparison complete.")
    print("Outputs:")
    for path in output_paths:
        print(f" - {path}")

    print()
    for result in results:
        row = result.summary_df.iloc[0]
        print(f"Case: {result.case_id}")
        print(f"  Baseline turbines: {int(row['baseline_n_turbines'])}")
        print(f"  Refined turbines: {int(row['refined_n_turbines'])}")
        print(f"  Delta turbines: {int(row['delta_n_turbines'])}")
        print(f"  Baseline capacity: {row['baseline_installed_capacity_mw']:.1f} MW")
        print(f"  Refined capacity: {row['refined_installed_capacity_mw']:.1f} MW")
        print(f"  Moved turbines: {int(row['n_moved'])}")
        print(f"  Removed turbines: {int(row['n_removed'])}")
        print(f"  Added turbines: {int(row['n_added'])}")
        print(f"  Refined minimum nearest spacing: {row['refined_min_nearest_spacing_m']:.2f} m")
        print()


def main() -> int:
    try:
        args = parse_args()
        available = list_available_cases()

        if args.cases == ["?"]:
            if not available:
                print("No optimisation cases found.")
                return 0
            print("Available cases:")
            for case_id in available:
                print(f" - {case_id}")
            return 0

        if args.all:
            case_ids = available
            if not case_ids:
                print("No optimisation cases found.")
                return 0
        else:
            if not args.cases:
                print("ERROR: No case provided. Use --case <case_id>, --case ?, or --all.")
                return 1

            case_ids = sorted(set(args.cases))
            unknown = [case_id for case_id in case_ids if case_id not in set(available)]
            if unknown:
                raise ValueError(
                    f"Unknown case(s): {unknown}. Use --case ? to list available cases."
                )

        baseline_layout_csv = args.baseline_layout_csv.resolve()
        turbine_metadata_json = args.turbine_metadata_json.resolve()
        results: list[LayoutComparisonResult] = []

        for case_id in case_ids:
            refined_layout_csv = resolve_case_layout_csv(case_id, must_exist=True).resolve()
            result = run_layout_comparison(
                baseline_layout_csv=baseline_layout_csv,
                refined_layout_csv=refined_layout_csv,
                case_id=case_id,
                turbine_metadata_json=turbine_metadata_json,
            )
            results.append(result)

            refined_df = load_layout(refined_layout_csv, turbine_metadata_json)
            write_refined_layout_metadata(
                refined_df=refined_df,
                refined_layout_csv=refined_layout_csv,
                refined_metadata_json=refined_metadata_path_for_case(case_id),
                case_id=case_id,
            )

        output_paths = export_layout_comparison(
            results=results,
            output_dir=args.output_dir.resolve(),
        )

        print_layout_comparison_summary(results, output_paths)
        return 0

    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())