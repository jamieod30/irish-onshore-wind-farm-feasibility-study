from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

BASELINE_REFERENCE_CSV = (
    REPO_ROOT / "data" / "optimisation" / "baseline" / "metadata" / "baseline_performance_reference.csv"
)
SELECTED_CASES_OUTPUT_ROOT = REPO_ROOT / "outputs" / "wake" / "selected_cases"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "tables" / "optimisation"


@dataclass(frozen=True)
class PerformanceComparisonResult:
    case_id: str
    summary_df: pd.DataFrame
    metadata: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare baseline benchmark performance against one or more selected-case "
            "PyWake runs. Use --case <case_id>, --case ?, or --all."
        )
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--case",
        dest="case_id",
        help="Case folder name under outputs/wake/selected_cases/, or '?' to list available cases.",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Compare all selected-case PyWake runs.",
    )
    parser.add_argument(
        "--baseline-reference-csv",
        type=Path,
        default=BASELINE_REFERENCE_CSV,
        help="Path to baseline performance reference CSV.",
    )
    parser.add_argument(
        "--selected-cases-output-root",
        type=Path,
        default=SELECTED_CASES_OUTPUT_ROOT,
        help="Root directory containing selected-case PyWake outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for optimisation comparison outputs.",
    )
    return parser.parse_args()


def list_available_cases(selected_cases_output_root: Path) -> list[str]:
    if not selected_cases_output_root.exists():
        return []

    case_ids: list[str] = []
    for path in sorted(selected_cases_output_root.iterdir()):
        if not path.is_dir():
            continue
        if (path / "farm_summary.csv").exists():
            case_ids.append(path.name)
    return case_ids


def load_single_row_csv(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required CSV not found: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"CSV contains no rows: {path}")
    if len(df) != 1:
        raise ValueError(f"Expected exactly one row in {path}, found {len(df)}")

    return df.iloc[0].to_dict()


def load_baseline_reference(path: Path) -> dict[str, Any]:
    row = load_single_row_csv(path)

    required = [
        "reference_case_id",
        "model",
        "site_name",
        "layout_name",
        "num_turbines",
        "installed_capacity_mw",
        "gross_aep_gwh",
        "net_aep_gwh",
        "wake_loss_gwh",
        "wake_loss_pct",
        "gross_capacity_factor",
        "net_capacity_factor",
    ]
    missing = [k for k in required if k not in row]
    if missing:
        raise ValueError(f"Baseline reference CSV missing required columns: {missing}")

    return row


def load_case_farm_summary(case_dir: Path) -> dict[str, Any]:
    return load_single_row_csv(case_dir / "farm_summary.csv")


def load_case_metadata(case_dir: Path) -> dict[str, Any]:
    metadata_path = case_dir / "run_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Case run metadata not found: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def path_relative_to_repo(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT)).replace("\\", "/")


def pct_change(new: float, old: float) -> float | None:
    if old == 0:
        return None
    return 100.0 * (new - old) / old


def build_case_comparison_summary(
    *,
    case_id: str,
    baseline_reference: dict[str, Any],
    case_farm: dict[str, Any],
    case_metadata: dict[str, Any],
    case_dir: Path,
) -> pd.DataFrame:
    baseline_installed_capacity_mw = float(baseline_reference["installed_capacity_mw"])
    baseline_gross_aep_gwh = float(baseline_reference["gross_aep_gwh"])
    baseline_net_aep_gwh = float(baseline_reference["net_aep_gwh"])
    baseline_wake_loss_gwh = float(baseline_reference["wake_loss_gwh"])
    baseline_wake_loss_pct = float(baseline_reference["wake_loss_pct"])
    baseline_gross_cf = float(baseline_reference["gross_capacity_factor"])
    baseline_net_cf = float(baseline_reference["net_capacity_factor"])
    baseline_num_turbines = int(baseline_reference["num_turbines"])

    case_installed_capacity_mw = float(case_farm["installed_capacity_mw"])
    case_gross_aep_gwh = float(case_farm["gross_aep_gwh"])
    case_net_aep_gwh = float(case_farm["net_aep_gwh"])
    case_wake_loss_gwh = float(case_farm["wake_loss_gwh"])
    case_wake_loss_pct = float(case_farm["wake_loss_pct"])
    case_gross_cf = float(case_farm["gross_capacity_factor"])
    case_net_cf = float(case_farm["net_capacity_factor"])
    case_num_turbines = int(case_farm["num_turbines"])

    baseline_net_aep_per_mw = baseline_net_aep_gwh / baseline_installed_capacity_mw
    case_net_aep_per_mw = case_net_aep_gwh / case_installed_capacity_mw

    baseline_net_aep_per_turbine = baseline_net_aep_gwh / baseline_num_turbines
    case_net_aep_per_turbine = case_net_aep_gwh / case_num_turbines

    row = {
        "comparison_id": f"baseline_vs_{case_id}_performance",
        "case_id": case_id,
        "baseline_reference_case_id": str(baseline_reference["reference_case_id"]),
        "baseline_layout_name": str(baseline_reference["layout_name"]),
        "case_layout_name": str(case_farm["layout_name"]),
        "resource_site_name": str(case_metadata.get("resource_site_name", "")),
        "baseline_num_turbines": baseline_num_turbines,
        "case_num_turbines": case_num_turbines,
        "delta_num_turbines": case_num_turbines - baseline_num_turbines,
        "baseline_installed_capacity_mw": baseline_installed_capacity_mw,
        "case_installed_capacity_mw": case_installed_capacity_mw,
        "delta_installed_capacity_mw": case_installed_capacity_mw - baseline_installed_capacity_mw,
        "pct_change_installed_capacity_mw": pct_change(
            case_installed_capacity_mw, baseline_installed_capacity_mw
        ),
        "baseline_gross_aep_gwh": baseline_gross_aep_gwh,
        "case_gross_aep_gwh": case_gross_aep_gwh,
        "delta_gross_aep_gwh": case_gross_aep_gwh - baseline_gross_aep_gwh,
        "pct_change_gross_aep_gwh": pct_change(case_gross_aep_gwh, baseline_gross_aep_gwh),
        "baseline_net_aep_gwh": baseline_net_aep_gwh,
        "case_net_aep_gwh": case_net_aep_gwh,
        "delta_net_aep_gwh": case_net_aep_gwh - baseline_net_aep_gwh,
        "pct_change_net_aep_gwh": pct_change(case_net_aep_gwh, baseline_net_aep_gwh),
        "baseline_wake_loss_gwh": baseline_wake_loss_gwh,
        "case_wake_loss_gwh": case_wake_loss_gwh,
        "delta_wake_loss_gwh": case_wake_loss_gwh - baseline_wake_loss_gwh,
        "pct_change_wake_loss_gwh": pct_change(case_wake_loss_gwh, baseline_wake_loss_gwh),
        "baseline_wake_loss_pct": baseline_wake_loss_pct,
        "case_wake_loss_pct": case_wake_loss_pct,
        "delta_wake_loss_pct": case_wake_loss_pct - baseline_wake_loss_pct,
        "abs_delta_wake_loss_pct": abs(case_wake_loss_pct - baseline_wake_loss_pct),
        "baseline_gross_capacity_factor": baseline_gross_cf,
        "case_gross_capacity_factor": case_gross_cf,
        "delta_gross_capacity_factor": case_gross_cf - baseline_gross_cf,
        "baseline_net_capacity_factor": baseline_net_cf,
        "case_net_capacity_factor": case_net_cf,
        "delta_net_capacity_factor": case_net_cf - baseline_net_cf,
        "baseline_net_aep_per_mw": baseline_net_aep_per_mw,
        "case_net_aep_per_mw": case_net_aep_per_mw,
        "delta_net_aep_per_mw": case_net_aep_per_mw - baseline_net_aep_per_mw,
        "baseline_net_aep_per_turbine": baseline_net_aep_per_turbine,
        "case_net_aep_per_turbine": case_net_aep_per_turbine,
        "delta_net_aep_per_turbine": case_net_aep_per_turbine - baseline_net_aep_per_turbine,
        "source_case_farm_summary": path_relative_to_repo(case_dir / "farm_summary.csv"),
        "source_case_run_metadata": path_relative_to_repo(case_dir / "run_metadata.json"),
    }

    return pd.DataFrame([row])


def build_case_metadata(
    *,
    case_id: str,
    summary_df: pd.DataFrame,
    baseline_reference_csv: Path,
    case_dir: Path,
) -> dict[str, Any]:
    return {
        "comparison_id": f"baseline_vs_{case_id}_performance",
        "case_id": case_id,
        "comparison_type": "baseline_vs_selected_case_performance",
        "baseline_reference_csv": path_relative_to_repo(baseline_reference_csv),
        "case_farm_summary_csv": path_relative_to_repo(case_dir / "farm_summary.csv"),
        "case_run_metadata_json": path_relative_to_repo(case_dir / "run_metadata.json"),
        "summary": summary_df.iloc[0].to_dict(),
        "notes": [
            "Comparison is performance-based using completed PyWake runs.",
            "Baseline reference is treated as the fixed project benchmark.",
            "Selected-case run is compared directly against the benchmark on energy and wake metrics.",
        ],
    }


def run_case_performance_comparison(
    *,
    case_id: str,
    baseline_reference_csv: Path,
    selected_cases_output_root: Path,
) -> PerformanceComparisonResult:
    baseline_reference = load_baseline_reference(baseline_reference_csv)
    case_dir = (selected_cases_output_root / case_id).resolve()

    case_farm = load_case_farm_summary(case_dir)
    case_metadata = load_case_metadata(case_dir)

    summary_df = build_case_comparison_summary(
        case_id=case_id,
        baseline_reference=baseline_reference,
        case_farm=case_farm,
        case_metadata=case_metadata,
        case_dir=case_dir,
    )
    metadata = build_case_metadata(
        case_id=case_id,
        summary_df=summary_df,
        baseline_reference_csv=baseline_reference_csv.resolve(),
        case_dir=case_dir,
    )

    return PerformanceComparisonResult(
        case_id=case_id,
        summary_df=summary_df,
        metadata=metadata,
    )


def export_results(
    results: list[PerformanceComparisonResult],
    output_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    case_ids = sorted([r.case_id for r in results])
    bundle_stem = "performance_comparison_" + "_".join(["baseline", *case_ids])

    summary_df = pd.concat([r.summary_df for r in results], ignore_index=True)
    metadata = {
        "comparison_bundle": bundle_stem,
        "case_ids": case_ids,
        "comparisons": [r.metadata for r in results],
    }

    summary_path = output_dir / f"{bundle_stem}_summary.csv"
    metadata_path = output_dir / f"{bundle_stem}_metadata.json"
    ranking_path = output_dir / f"{bundle_stem}_ranking.csv"

    ranking_df = summary_df[
        [
            "case_id",
            "case_layout_name",
            "case_num_turbines",
            "case_installed_capacity_mw",
            "case_net_aep_gwh",
            "case_wake_loss_pct",
            "case_net_capacity_factor",
            "delta_net_aep_gwh",
            "delta_wake_loss_pct",
            "pct_change_net_aep_gwh",
        ]
    ].copy()

    ranking_df = ranking_df.sort_values(
        by=["case_net_aep_gwh", "case_wake_loss_pct", "case_id"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    ranking_df.insert(0, "rank_by_case_net_aep", np.arange(1, len(ranking_df) + 1))

    summary_df.to_csv(summary_path, index=False)
    ranking_df.to_csv(ranking_path, index=False)

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return [summary_path, ranking_path, metadata_path]


def print_results(
    results: list[PerformanceComparisonResult],
    output_paths: list[Path],
) -> None:
    print("Baseline vs selected-case performance comparison complete.")
    print("Outputs:")
    for path in output_paths:
        print(f" - {path}")

    print()
    for result in results:
        row = result.summary_df.iloc[0]
        print(f"Case: {result.case_id}")
        print(f"  Baseline net AEP: {float(row['baseline_net_aep_gwh']):.3f} GWh")
        print(f"  Case net AEP: {float(row['case_net_aep_gwh']):.3f} GWh")
        print(f"  Delta net AEP: {float(row['delta_net_aep_gwh']):.3f} GWh")
        print(f"  Baseline wake loss: {float(row['baseline_wake_loss_pct']):.3f}%")
        print(f"  Case wake loss: {float(row['case_wake_loss_pct']):.3f}%")
        print(f"  Delta wake loss: {float(row['delta_wake_loss_pct']):.3f} percentage points")
        print(f"  Baseline net CF: {float(row['baseline_net_capacity_factor']):.6f}")
        print(f"  Case net CF: {float(row['case_net_capacity_factor']):.6f}")
        print()


def main() -> int:
    try:
        args = parse_args()
        selected_cases_output_root = args.selected_cases_output_root.resolve()
        available = list_available_cases(selected_cases_output_root)

        if args.case_id == "?":
            if not available:
                print("No selected-case PyWake runs found.")
                return 0

            print("Available cases:")
            for case_id in available:
                print(f" - {case_id}")
            return 0

        if args.all:
            case_ids = available
            if not case_ids:
                print("No selected-case PyWake runs found.")
                return 0
        else:
            if args.case_id not in available:
                raise ValueError(
                    f"Unknown case '{args.case_id}'. Use --case ? to list available cases."
                )
            case_ids = [args.case_id]

        results = [
            run_case_performance_comparison(
                case_id=case_id,
                baseline_reference_csv=args.baseline_reference_csv.resolve(),
                selected_cases_output_root=selected_cases_output_root,
            )
            for case_id in case_ids
        ]

        output_paths = export_results(results, args.output_dir.resolve())
        print_results(results, output_paths)
        return 0

    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())