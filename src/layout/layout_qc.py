from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_BASELINE_LAYOUT = (
    REPO_ROOT / "data" / "optimisation" / "baseline" / "layout" / "layout_baseline_aligned.csv"
)
DEFAULT_OPTIMISATION_ROOT = REPO_ROOT / "data" / "optimisation"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "tables" / "layouts"
DEFAULT_TURBINE_METADATA_JSON = (
    REPO_ROOT / "data" / "turbines" / "vestas_v136_4p2mw" / "turbine_metadata.json"
)

SUMMARY_OUTPUT = DEFAULT_OUTPUT_DIR / "layout_qc_summary.csv"
FAILURES_OUTPUT = DEFAULT_OUTPUT_DIR / "layout_qc_failures.csv"

BASELINE_SUMMARY_OUTPUT = DEFAULT_OUTPUT_DIR / "layout_summary.csv"
BASELINE_NEAREST_OUTPUT = DEFAULT_OUTPUT_DIR / "layout_nearest_neighbour_spacing.csv"
BASELINE_PAIRWISE_OUTPUT = DEFAULT_OUTPUT_DIR / "layout_all_pairwise_spacing.csv"

REQUIRED_GEOMETRY_COLUMNS = {
    "turbine_id",
    "easting_m",
    "northing_m",
}

ENRICHED_REQUIRED_COLUMNS = {
    "turbine_id",
    "easting_m",
    "northing_m",
    "hub_height_m",
    "rotor_diameter_m",
    "rated_power_mw",
    "turbine_model",
    "layout_name",
    "site",
}


@dataclass(frozen=True)
class LayoutQcArtifacts:
    summary_row: dict[str, Any]
    failures: list[dict[str, Any]]
    spacing_df: pd.DataFrame | None
    nearest_df: pd.DataFrame | None
    summary_df: pd.DataFrame | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run QC checks on one or more layout CSVs. "
            "Default behaviour is to check the baseline benchmark layout only."
        )
    )
    parser.add_argument(
        "--layout-csv",
        type=Path,
        nargs="*",
        default=None,
        help="Optional one or more explicit layout CSV paths to validate.",
    )
    parser.add_argument(
        "--case",
        action="append",
        default=None,
        help=(
            "Optimisation case folder name under data/optimisation/. "
            "Repeatable. Use --case ? to list available cases."
        ),
    )
    parser.add_argument(
        "--all-cases",
        action="store_true",
        help="Validate selected-layout CSVs for all discovered optimisation cases.",
    )
    parser.add_argument(
        "--all-candidates",
        action="store_true",
        help="Validate all candidate layout CSVs for all discovered optimisation cases.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help=(
            "Validate the full layout set: baseline layout, all selected-case layouts, "
            "and all candidate layouts."
        ),
    )
    parser.add_argument(
        "--turbine-metadata-json",
        type=Path,
        default=DEFAULT_TURBINE_METADATA_JSON,
        help="Path to turbine metadata JSON used to enrich layouts when needed.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for QC summary and troubleshooting outputs.",
    )
    return parser.parse_args()


def discover_case_ids(optimisation_root: Path = DEFAULT_OPTIMISATION_ROOT) -> list[str]:
    case_ids: list[str] = []

    if not optimisation_root.exists():
        return case_ids

    for p in sorted(optimisation_root.iterdir()):
        if not p.is_dir():
            continue
        if p.name in {"baseline", "comparisons"}:
            continue
        case_ids.append(p.name)

    return case_ids


def find_selected_layout_csvs_for_case(case_id: str) -> list[Path]:
    selected_dir = DEFAULT_OPTIMISATION_ROOT / case_id / "selected_layout"
    if not selected_dir.exists():
        return []

    return sorted(p for p in selected_dir.glob("*.csv") if p.is_file())


def find_candidate_layout_csvs_for_case(case_id: str) -> list[Path]:
    candidate_dir = DEFAULT_OPTIMISATION_ROOT / case_id / "candidate_layouts"
    if not candidate_dir.exists():
        return []

    return sorted(p for p in candidate_dir.glob("*.csv") if p.is_file())


def resolve_layouts_from_args(args: argparse.Namespace) -> list[Path]:
    case_args = args.case or []

    if "?" in case_args:
        case_ids = discover_case_ids()
        print("Available optimisation cases:")
        if case_ids:
            for case_id in case_ids:
                print(f" - {case_id}")
        else:
            print(" - none found")
        sys.exit(0)

    layout_paths: list[Path] = []

    if args.layout_csv:
        layout_paths.extend([p.resolve() for p in args.layout_csv])

    if case_args:
        available_case_ids = set(discover_case_ids())
        requested_case_ids = sorted(set(case_args))

        missing_case_ids = [c for c in requested_case_ids if c not in available_case_ids]
        if missing_case_ids:
            raise FileNotFoundError(
                "Requested case folder(s) not found under data/optimisation/: "
                + ", ".join(missing_case_ids)
            )

        for case_id in requested_case_ids:
            selected_csvs = find_selected_layout_csvs_for_case(case_id)
            if not selected_csvs:
                raise FileNotFoundError(
                    f"No selected-layout CSVs found for case '{case_id}' under "
                    f"data/optimisation/{case_id}/selected_layout/"
                )
            layout_paths.extend([p.resolve() for p in selected_csvs])

    if args.all_cases or args.all or args.all_candidates:
        case_ids = discover_case_ids()
    else:
        case_ids = []

    if args.all_cases:
        for case_id in case_ids:
            layout_paths.extend([p.resolve() for p in find_selected_layout_csvs_for_case(case_id)])

    if args.all_candidates:
        for case_id in case_ids:
            layout_paths.extend([p.resolve() for p in find_candidate_layout_csvs_for_case(case_id)])

    if args.all:
        layout_paths.append(DEFAULT_BASELINE_LAYOUT.resolve())
        for case_id in case_ids:
            layout_paths.extend([p.resolve() for p in find_selected_layout_csvs_for_case(case_id)])
            layout_paths.extend([p.resolve() for p in find_candidate_layout_csvs_for_case(case_id)])

    if not layout_paths:
        layout_paths = [DEFAULT_BASELINE_LAYOUT.resolve()]

    seen: set[str] = set()
    deduped: list[Path] = []
    for path in layout_paths:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            deduped.append(path.resolve())

    return deduped


def path_relative_to_repo(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def infer_case_id(layout_file: Path) -> str:
    rel = path_relative_to_repo(layout_file)

    parts = Path(rel).parts
    if len(parts) >= 3 and parts[0] == "data" and parts[1] == "optimisation":
        if parts[2] == "baseline":
            return "baseline"
        if parts[2] != "comparisons":
            return parts[2]

    return "unknown"


def infer_site_name(layout_file: Path) -> str:
    case_id = infer_case_id(layout_file)
    return "baseline" if case_id == "baseline" else case_id


def infer_layout_name(layout_file: Path, df: pd.DataFrame) -> str:
    if "layout_name" in df.columns and not df["layout_name"].dropna().empty:
        return str(df["layout_name"].dropna().iloc[0]).strip()
    return layout_file.stem


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


def enrich_layout_dataframe(
    df: pd.DataFrame,
    layout_file: Path,
    turbine_metadata_json: Path,
) -> pd.DataFrame:
    df = df.copy()
    turbine_meta = load_turbine_metadata(turbine_metadata_json)

    defaults: dict[str, Any] = {
        "hub_height_m": float(turbine_meta["hub_height_m"]),
        "rotor_diameter_m": float(turbine_meta["rotor_diameter_m"]),
        "rated_power_mw": float(turbine_meta["rated_power_mw"]),
        "turbine_model": str(turbine_meta["turbine_model"]),
        "site": infer_site_name(layout_file),
        "layout_name": layout_file.stem,
    }

    for col, default_value in defaults.items():
        if col not in df.columns:
            df[col] = default_value
        else:
            if isinstance(default_value, (int, float)):
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default_value)
            else:
                df[col] = df[col].astype("object").where(
                    df[col].notna() & (df[col].astype(str).str.strip() != ""),
                    default_value,
                )

    for col in ["hub_height_m", "rotor_diameter_m", "rated_power_mw", "easting_m", "northing_m"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def validate_geometry_columns(df: pd.DataFrame) -> None:
    missing = REQUIRED_GEOMETRY_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Layout file is missing required geometry columns: {sorted(missing)}")


def validate_enriched_required_columns(df: pd.DataFrame) -> None:
    missing = ENRICHED_REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Layout file is missing required columns after enrichment: {sorted(missing)}")


def validate_no_missing_values(df: pd.DataFrame) -> None:
    required_cols = sorted(ENRICHED_REQUIRED_COLUMNS)
    if df[required_cols].isna().any().any():
        missing_counts = df[required_cols].isna().sum()
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
        raise ValueError(
            f"Duplicate turbine coordinates found:\n{dupes.to_string(index=False)}"
        )


def validate_single_value_fields(df: pd.DataFrame, fields: list[str]) -> None:
    for field in fields:
        unique_values = df[field].dropna().astype(str).unique()
        if len(unique_values) != 1:
            raise ValueError(
                f"Field '{field}' should contain exactly one unique value, "
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

            rows.append(
                {
                    "turbine_a": coords.loc[i, "turbine_id"],
                    "turbine_b": coords.loc[j, "turbine_id"],
                    "dx_m": dx,
                    "dy_m": dy,
                    "spacing_m": spacing_m,
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=["turbine_a", "turbine_b", "dx_m", "dy_m", "spacing_m"]
        )

    spacing_df = pd.DataFrame(rows)
    return spacing_df.sort_values("spacing_m").reset_index(drop=True)


def compute_nearest_neighbour_spacing(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < 2:
        return pd.DataFrame(
            columns=["turbine_id", "nearest_neighbour", "nearest_spacing_m"]
        )

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

        nearest_rows.append(
            {
                "turbine_id": tid,
                "nearest_neighbour": neighbour,
                "nearest_spacing_m": nearest["spacing_m"],
            }
        )

    return pd.DataFrame(nearest_rows).sort_values("turbine_id").reset_index(drop=True)


def extract_spacing_tiers(
    spacing_df: pd.DataFrame,
    tolerance_m: float = 1.0,
) -> tuple[float | None, float | None]:
    unique_spacings: list[float] = []

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

    min_spacing_m = (
        float(nearest_df["nearest_spacing_m"].min()) if not nearest_df.empty else np.nan
    )
    mean_spacing_m = (
        float(nearest_df["nearest_spacing_m"].mean()) if not nearest_df.empty else np.nan
    )

    short_spacing_m, long_spacing_m = extract_spacing_tiers(spacing_df) if not spacing_df.empty else (None, None)

    summary = {
        "layout_name": df["layout_name"].iloc[0],
        "site": df["site"].iloc[0],
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
        "min_nearest_spacing_D": min_spacing_m / rotor_diameter if pd.notna(min_spacing_m) else np.nan,
        "mean_nearest_spacing_D": mean_spacing_m / rotor_diameter if pd.notna(mean_spacing_m) else np.nan,
        "short_spacing_m": short_spacing_m,
        "long_spacing_m": long_spacing_m,
        "short_spacing_D": (
            short_spacing_m / rotor_diameter if short_spacing_m is not None else np.nan
        ),
        "long_spacing_D": (
            long_spacing_m / rotor_diameter if long_spacing_m is not None else np.nan
        ),
    }

    return pd.DataFrame([summary])


def run_single_layout_qc(layout_file: Path, turbine_metadata_json: Path) -> LayoutQcArtifacts:
    failures: list[dict[str, Any]] = []

    source_layout_csv = path_relative_to_repo(layout_file)
    case_id = infer_case_id(layout_file)
    file_stem = layout_file.stem

    def add_failure(check_name: str, detail: str) -> None:
        failures.append(
            {
                "source_layout_csv": source_layout_csv,
                "case_id": case_id,
                "layout_file_name": layout_file.name,
                "layout_stem": file_stem,
                "check_name": check_name,
                "detail": detail,
            }
        )

    spacing_df: pd.DataFrame | None = None
    nearest_df: pd.DataFrame | None = None
    summary_df: pd.DataFrame | None = None

    try:
        if not layout_file.exists():
            raise FileNotFoundError(f"Layout file not found: {layout_file}")

        raw_df = pd.read_csv(layout_file)

        checks = [
            ("required_geometry_columns", lambda: validate_geometry_columns(raw_df)),
        ]

        for check_name, fn in checks:
            try:
                fn()
            except Exception as exc:
                add_failure(check_name, str(exc))

        if not failures:
            df = enrich_layout_dataframe(
                df=raw_df,
                layout_file=layout_file,
                turbine_metadata_json=turbine_metadata_json,
            )

            enriched_checks = [
                ("required_columns_after_enrichment", lambda: validate_enriched_required_columns(df)),
                ("no_missing_values", lambda: validate_no_missing_values(df)),
                ("unique_turbine_ids", lambda: validate_unique_turbine_ids(df)),
                ("unique_coordinates", lambda: validate_unique_coordinates(df)),
                (
                    "single_value_fields",
                    lambda: validate_single_value_fields(
                        df,
                        [
                            "hub_height_m",
                            "rotor_diameter_m",
                            "rated_power_mw",
                            "turbine_model",
                            "layout_name",
                            "site",
                        ],
                    ),
                ),
            ]

            for check_name, fn in enriched_checks:
                try:
                    fn()
                except Exception as exc:
                    add_failure(check_name, str(exc))

            if not failures:
                spacing_df = compute_pairwise_spacing(df)
                nearest_df = compute_nearest_neighbour_spacing(df)
                summary_df = build_layout_summary(df, nearest_df, spacing_df)

                min_spacing_d = float(summary_df["min_nearest_spacing_D"].iloc[0])

                if pd.notna(min_spacing_d) and min_spacing_d < 2.0:
                    add_failure(
                        "minimum_spacing_threshold",
                        (
                            f"Minimum nearest-neighbour spacing is {min_spacing_d:.2f}D, "
                            "below the 2D threshold."
                        ),
                    )

        if "df" in locals() and not df.empty:
            layout_name = str(df["layout_name"].iloc[0])
            site = str(df["site"].iloc[0])
            n_turbines = int(len(df))
        else:
            layout_name = file_stem
            site = "unknown"
            n_turbines = 0

    except Exception as exc:
        add_failure("load_or_parse", str(exc))
        layout_name = file_stem
        site = "unknown"
        n_turbines = 0

    pass_flag = len(failures) == 0
    failed_checks = "; ".join(f["check_name"] for f in failures) if failures else ""
    failure_details = " | ".join(
        f"{f['check_name']}: {f['detail'].replace(chr(10), ' // ')}" for f in failures
    ) if failures else ""

    summary_row: dict[str, Any] = {
        "source_layout_csv": source_layout_csv,
        "case_id": case_id,
        "layout_file_name": layout_file.name,
        "layout_stem": file_stem,
        "layout_name": layout_name,
        "site": site,
        "n_turbines": n_turbines,
        "pass_flag": pass_flag,
        "n_failures": len(failures),
        "failed_checks": failed_checks,
        "failure_details": failure_details,
    }

    if summary_df is not None and not summary_df.empty:
        summary_row.update(summary_df.iloc[0].to_dict())

    return LayoutQcArtifacts(
        summary_row=summary_row,
        failures=failures,
        spacing_df=spacing_df,
        nearest_df=nearest_df,
        summary_df=summary_df,
    )


def export_qc_outputs(
    artifacts: list[LayoutQcArtifacts],
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = [a.summary_row for a in artifacts]
    failures_rows = [row for a in artifacts for row in a.failures]

    summary_df = pd.DataFrame(summary_rows).sort_values(
        by=["pass_flag", "case_id", "layout_name", "source_layout_csv"],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)

    failures_df = pd.DataFrame(failures_rows)
    if not failures_df.empty:
        failures_df = failures_df.sort_values(
            by=["case_id", "layout_stem", "check_name"]
        ).reset_index(drop=True)

    summary_path = output_dir / "layout_qc_summary.csv"
    failures_path = output_dir / "layout_qc_failures.csv"

    summary_df.to_csv(summary_path, index=False)

    if failures_df.empty:
        pd.DataFrame(
            columns=[
                "source_layout_csv",
                "case_id",
                "layout_file_name",
                "layout_stem",
                "check_name",
                "detail",
            ]
        ).to_csv(failures_path, index=False)
    else:
        failures_df.to_csv(failures_path, index=False)

    baseline_artifacts = [
        a for a in artifacts if a.summary_row.get("source_layout_csv") == path_relative_to_repo(DEFAULT_BASELINE_LAYOUT)
    ]
    if len(baseline_artifacts) == 1:
        artifact = baseline_artifacts[0]
        passed = bool(artifact.summary_row["pass_flag"])

        if passed and artifact.summary_df is not None:
            artifact.summary_df.to_csv(BASELINE_SUMMARY_OUTPUT, index=False)
        if passed and artifact.nearest_df is not None:
            artifact.nearest_df.to_csv(BASELINE_NEAREST_OUTPUT, index=False)
        if passed and artifact.spacing_df is not None:
            artifact.spacing_df.to_csv(BASELINE_PAIRWISE_OUTPUT, index=False)

    return summary_path, failures_path


def print_qc_summary(artifacts: list[LayoutQcArtifacts], summary_path: Path, failures_path: Path) -> None:
    n_total = len(artifacts)
    n_pass = sum(1 for a in artifacts if a.summary_row["pass_flag"])
    n_fail = n_total - n_pass

    print("Layout QC complete.")
    print(f"Layouts checked: {n_total}")
    print(f"Passed: {n_pass}")
    print(f"Failed: {n_fail}")
    print(f"QC summary saved to: {summary_path}")
    print(f"QC failures saved to: {failures_path}")

    if n_fail > 0:
        print("\nFailed layouts:")
        failed_rows = [
            a.summary_row
            for a in artifacts
            if not a.summary_row["pass_flag"]
        ]
        failed_df = pd.DataFrame(failed_rows)[
            ["case_id", "layout_name", "source_layout_csv", "n_failures", "failed_checks"]
        ]
        print(failed_df.to_string(index=False))


def main() -> None:
    args = parse_args()

    try:
        layout_files = resolve_layouts_from_args(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    turbine_metadata_json = args.turbine_metadata_json.resolve()
    artifacts = [run_single_layout_qc(layout_file, turbine_metadata_json) for layout_file in layout_files]

    summary_path, failures_path = export_qc_outputs(
        artifacts=artifacts,
        output_dir=args.output_dir.resolve(),
    )

    print_qc_summary(artifacts, summary_path, failures_path)

    if any(not a.summary_row["pass_flag"] for a in artifacts):
        raise SystemExit(1)

    raise SystemExit(0)


if __name__ == "__main__":
    main()