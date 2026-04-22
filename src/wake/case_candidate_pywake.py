from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from src.wake.pywake_runner import (
    WakeRunResult,
    export_results,
    load_layout,
    run_pywake_noj,
    validate_layout_against_turbine_definition,
)
from src.wake.turbine_definition import DEFAULT_TURBINE

OPTIMISATION_DIR = REPO_ROOT / "data" / "optimisation"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "wake" / "optimisation"
BASELINE_REFERENCE_CSV = (
    REPO_ROOT
    / "data"
    / "optimisation"
    / "baseline"
    / "metadata"
    / "baseline_performance_reference.csv"
)


@dataclass(frozen=True)
class CandidateRunArtifacts:
    case_id: str
    layout_csv: Path
    output_dir: Path
    result: WakeRunResult


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run PyWake NOJ for candidate layouts in one or more optimisation cases. "
            "Use --case <case_id>, --case ?, or --all."
        )
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--case",
        dest="case_id",
        help="Case folder name under data/optimisation/, or '?' to list available cases.",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Run all optimisation cases with candidate_layouts.",
    )
    parser.add_argument(
        "--turbine-name",
        type=str,
        default=DEFAULT_TURBINE,
        help="Turbine folder name under data/turbines/.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for optimisation wake outputs.",
    )
    parser.add_argument(
        "--baseline-reference-csv",
        type=Path,
        default=BASELINE_REFERENCE_CSV,
        help="Baseline performance reference CSV used to derive the benchmark wake-loss target.",
    )
    parser.add_argument(
        "--underutilised-threshold-pct",
        type=float,
        default=4.0,
        help="Wake loss percentage below which a layout is flagged as underutilised.",
    )
    parser.add_argument(
        "--acceptable-threshold-pct",
        type=float,
        default=6.0,
        help="Upper wake loss percentage for the acceptable band.",
    )
    parser.add_argument(
        "--skip-wake-map",
        action="store_true",
        help="Accepted for interface consistency. Candidate batch runs skip wake maps by default.",
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
        if (path / "candidate_layouts").exists():
            case_ids.append(path.name)

    return case_ids


def discover_candidate_layout_csvs(case_id: str) -> list[Path]:
    candidate_dir = OPTIMISATION_DIR / case_id / "candidate_layouts"
    if not candidate_dir.exists():
        raise FileNotFoundError(f"Candidate layout directory not found: {candidate_dir}")

    csvs = sorted(p for p in candidate_dir.glob("*.csv") if p.is_file())
    if not csvs:
        raise FileNotFoundError(f"No candidate layout CSVs found in: {candidate_dir}")

    return csvs


def path_relative_to_repo(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT)).replace("\\", "/")


def candidate_output_dir(output_root: Path, case_id: str, layout_csv: Path) -> Path:
    return output_root / case_id / layout_csv.stem


def relabel_wake_result(
    result: WakeRunResult,
    *,
    case_id: str,
    layout_csv: Path,
    output_dir: Path,
) -> WakeRunResult:
    farm_summary = result.farm_summary.copy()
    turbine_summary = result.turbine_summary.copy()
    flow_case_summary = result.flow_case_summary.copy()
    metadata = dict(result.metadata)

    farm_summary["site_name"] = case_id

    if "site" in turbine_summary.columns:
        turbine_summary["site"] = case_id

    metadata["run_name"] = output_dir.name
    metadata["benchmark_case"] = False
    metadata["site_name"] = case_id
    metadata["layout_file"] = path_relative_to_repo(layout_csv.resolve())

    return WakeRunResult(
        farm_summary=farm_summary,
        turbine_summary=turbine_summary,
        flow_case_summary=flow_case_summary,
        metadata=metadata,
    )


def load_baseline_reference(baseline_reference_csv: Path) -> dict[str, Any]:
    if not baseline_reference_csv.exists():
        raise FileNotFoundError(
            f"Baseline performance reference CSV not found: {baseline_reference_csv}"
        )

    df = pd.read_csv(baseline_reference_csv)
    if df.empty:
        raise ValueError(
            f"Baseline performance reference CSV contains no rows: {baseline_reference_csv}"
        )
    if len(df) != 1:
        raise ValueError(
            f"Expected exactly one row in baseline performance reference CSV, found {len(df)}."
        )

    row = df.iloc[0]

    required_columns = [
        "reference_case_id",
        "wake_loss_pct",
        "net_aep_gwh",
        "installed_capacity_mw",
        "net_capacity_factor",
    ]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"Baseline performance reference CSV missing required columns: {missing}"
        )

    return {
        "reference_case_id": str(row["reference_case_id"]),
        "wake_loss_pct": float(row["wake_loss_pct"]),
        "net_aep_gwh": float(row["net_aep_gwh"]),
        "installed_capacity_mw": float(row["installed_capacity_mw"]),
        "net_capacity_factor": float(row["net_capacity_factor"]),
    }


def classify_wake_loss_band(
    wake_loss_pct: float,
    underutilised_threshold_pct: float,
    acceptable_threshold_pct: float,
) -> str:
    if wake_loss_pct < underutilised_threshold_pct:
        return "underutilised"
    if wake_loss_pct <= acceptable_threshold_pct:
        return "acceptable"
    return "overcrowded"


def choose_best_candidate(
    summary_df: pd.DataFrame,
    underutilised_threshold_pct: float,
    acceptable_threshold_pct: float,
) -> pd.DataFrame:
    if summary_df.empty:
        raise ValueError("Candidate summary table is empty.")

    acceptable_df = summary_df.loc[
        summary_df["wake_loss_band"] == "acceptable"
    ].copy()

    if not acceptable_df.empty:
        selected_df = acceptable_df.sort_values(
            by=[
                "net_aep_gwh",
                "net_capacity_factor",
                "wake_loss_pct",
                "num_turbines",
                "layout_name",
            ],
            ascending=[False, False, True, False, True],
        ).reset_index(drop=True)
        selection_basis = "highest net AEP within acceptable wake-loss band"
        winner_layout_name = str(selected_df.loc[0, "layout_name"])
    else:
        overcrowded_df = summary_df.loc[
            summary_df["wake_loss_band"] == "overcrowded"
        ].copy()
        underutilised_df = summary_df.loc[
            summary_df["wake_loss_band"] == "underutilised"
        ].copy()

        if not overcrowded_df.empty and underutilised_df.empty:
            selected_df = overcrowded_df.sort_values(
                by=[
                    "wake_loss_pct",
                    "net_aep_gwh",
                    "net_capacity_factor",
                    "num_turbines",
                    "layout_name",
                ],
                ascending=[True, False, False, False, True],
            ).reset_index(drop=True)
            selection_basis = (
                "no acceptable layouts; selected lowest wake-loss overcrowded layout"
            )
            winner_layout_name = str(selected_df.loc[0, "layout_name"])
        elif not underutilised_df.empty and overcrowded_df.empty:
            selected_df = underutilised_df.sort_values(
                by=[
                    "net_aep_gwh",
                    "net_capacity_factor",
                    "wake_loss_pct",
                    "num_turbines",
                    "layout_name",
                ],
                ascending=[False, False, False, False, True],
            ).reset_index(drop=True)
            selection_basis = (
                "no acceptable layouts; selected highest net AEP underutilised layout"
            )
            winner_layout_name = str(selected_df.loc[0, "layout_name"])
        else:
            selected_df = summary_df.sort_values(
                by=[
                    "abs_delta_vs_benchmark_wake_loss_pct",
                    "net_aep_gwh",
                    "net_capacity_factor",
                    "layout_name",
                ],
                ascending=[True, False, False, True],
            ).reset_index(drop=True)
            selection_basis = (
                "no acceptable layouts; selected layout closest to benchmark wake loss"
            )
            winner_layout_name = str(selected_df.loc[0, "layout_name"])

    out = summary_df.copy()
    out["selection_basis"] = selection_basis
    out["selected_layout_flag"] = out["layout_name"] == winner_layout_name

    out = out.sort_values(
        by=[
            "selected_layout_flag",
            "wake_loss_band_priority",
            "net_aep_gwh",
            "net_capacity_factor",
            "wake_loss_pct",
            "layout_name",
        ],
        ascending=[False, True, False, False, True, True],
    ).reset_index(drop=True)

    out.insert(0, "rank_overall", np.arange(1, len(out) + 1))
    return out


def build_case_summary_df(
    artifacts: list[CandidateRunArtifacts],
    baseline_reference: dict[str, Any],
    underutilised_threshold_pct: float,
    acceptable_threshold_pct: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    benchmark_wake_loss_pct = float(baseline_reference["wake_loss_pct"])

    for artifact in artifacts:
        farm_row = artifact.result.farm_summary.iloc[0]
        wake_loss_pct = float(farm_row["wake_loss_pct"])
        wake_loss_band = classify_wake_loss_band(
            wake_loss_pct=wake_loss_pct,
            underutilised_threshold_pct=underutilised_threshold_pct,
            acceptable_threshold_pct=acceptable_threshold_pct,
        )

        wake_loss_band_priority = {
            "acceptable": 1,
            "underutilised": 2,
            "overcrowded": 3,
        }[wake_loss_band]

        rows.append(
            {
                "case_id": artifact.case_id,
                "layout_name": str(farm_row["layout_name"]),
                "source_layout_csv": path_relative_to_repo(artifact.layout_csv),
                "output_dir": path_relative_to_repo(artifact.output_dir),
                "num_turbines": int(farm_row["num_turbines"]),
                "installed_capacity_mw": float(farm_row["installed_capacity_mw"]),
                "gross_aep_gwh": float(farm_row["gross_aep_gwh"]),
                "net_aep_gwh": float(farm_row["net_aep_gwh"]),
                "wake_loss_gwh": float(farm_row["wake_loss_gwh"]),
                "wake_loss_pct": wake_loss_pct,
                "gross_capacity_factor": float(farm_row["gross_capacity_factor"]),
                "net_capacity_factor": float(farm_row["net_capacity_factor"]),
                "benchmark_wake_loss_pct": benchmark_wake_loss_pct,
                "delta_vs_benchmark_wake_loss_pct": wake_loss_pct - benchmark_wake_loss_pct,
                "abs_delta_vs_benchmark_wake_loss_pct": abs(
                    wake_loss_pct - benchmark_wake_loss_pct
                ),
                "wake_loss_band": wake_loss_band,
                "wake_loss_band_priority": wake_loss_band_priority,
            }
        )

    summary_df = pd.DataFrame(rows)
    summary_df = choose_best_candidate(
        summary_df=summary_df,
        underutilised_threshold_pct=underutilised_threshold_pct,
        acceptable_threshold_pct=acceptable_threshold_pct,
    )

    return summary_df


def build_case_metadata(
    case_id: str,
    artifacts: list[CandidateRunArtifacts],
    case_summary_df: pd.DataFrame,
    turbine_name: str,
    baseline_reference: dict[str, Any],
    underutilised_threshold_pct: float,
    acceptable_threshold_pct: float,
) -> dict[str, Any]:
    best = case_summary_df.loc[case_summary_df["selected_layout_flag"]].iloc[0].to_dict()

    return {
        "case_id": case_id,
        "comparison_basis": "candidate_layout_internal_screening",
        "turbine_name": turbine_name,
        "n_candidate_layouts": int(len(artifacts)),
        "baseline_reference": baseline_reference,
        "wake_loss_thresholds_pct": {
            "underutilised_lt": underutilised_threshold_pct,
            "acceptable_upper": acceptable_threshold_pct,
        },
        "candidate_layouts": [
            {
                "layout_name": artifact.layout_csv.stem,
                "source_layout_csv": path_relative_to_repo(artifact.layout_csv),
                "output_dir": path_relative_to_repo(artifact.output_dir),
            }
            for artifact in artifacts
        ],
        "selection_rule": (
            "Choose highest net_aep_gwh within acceptable wake-loss band; "
            "if none acceptable, prefer lowest wake-loss overcrowded layout or "
            "highest net AEP underutilised layout as applicable."
        ),
        "best_candidate": best,
        "notes": [
            "This step compares candidate layouts within one optimisation case.",
            "This is separate from baseline-vs-case comparison.",
            "Wake-loss bands are benchmark-informed and intended for feasibility-stage screening.",
            "Wake maps are not batch-exported here.",
        ],
    }


def export_case_outputs(
    case_id: str,
    case_summary_df: pd.DataFrame,
    metadata: dict[str, Any],
    output_root: Path,
) -> tuple[Path, Path]:
    case_dir = output_root / case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    summary_path = case_dir / "candidate_pywake_summary.csv"
    metadata_path = case_dir / "candidate_pywake_metadata.json"

    export_df = case_summary_df.drop(columns=["wake_loss_band_priority"]).copy()

    export_df.to_csv(summary_path, index=False)
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return summary_path, metadata_path


def print_case_summary(
    case_id: str,
    case_summary_df: pd.DataFrame,
    summary_path: Path,
    metadata_path: Path,
) -> None:
    print("Candidate PyWake assessment complete.")
    print(f"Case: {case_id}")
    print(f"Summary CSV: {summary_path}")
    print(f"Metadata JSON: {metadata_path}")
    print()

    display_cols = [
        "rank_overall",
        "layout_name",
        "num_turbines",
        "net_aep_gwh",
        "wake_loss_pct",
        "wake_loss_band",
        "net_capacity_factor",
        "selected_layout_flag",
    ]
    print(case_summary_df.loc[:, display_cols].to_string(index=False))


def run_case(
    case_id: str,
    *,
    turbine_name: str,
    output_root: Path,
    baseline_reference: dict[str, Any],
    underutilised_threshold_pct: float,
    acceptable_threshold_pct: float,
) -> None:
    layout_csvs = discover_candidate_layout_csvs(case_id)
    artifacts: list[CandidateRunArtifacts] = []

    for layout_csv in layout_csvs:
        layout_csv = layout_csv.resolve()
        output_dir = candidate_output_dir(
            output_root=output_root.resolve(),
            case_id=case_id,
            layout_csv=layout_csv,
        ).resolve()

        layout_df = load_layout(
            layout_csv=layout_csv,
            turbine_name=turbine_name,
        )
        layout_df.attrs["source_file"] = str(layout_csv)

        validate_layout_against_turbine_definition(
            layout_df=layout_df,
            turbine_name=turbine_name,
        )

        base_result = run_pywake_noj(
            layout_df=layout_df,
            turbine_name=turbine_name,
            output_dir=output_dir,
        )

        result = relabel_wake_result(
            base_result,
            case_id=case_id,
            layout_csv=layout_csv,
            output_dir=output_dir,
        )

        export_results(result, output_dir)

        artifacts.append(
            CandidateRunArtifacts(
                case_id=case_id,
                layout_csv=layout_csv,
                output_dir=output_dir,
                result=result,
            )
        )

    case_summary_df = build_case_summary_df(
        artifacts=artifacts,
        baseline_reference=baseline_reference,
        underutilised_threshold_pct=underutilised_threshold_pct,
        acceptable_threshold_pct=acceptable_threshold_pct,
    )

    metadata = build_case_metadata(
        case_id=case_id,
        artifacts=artifacts,
        case_summary_df=case_summary_df,
        turbine_name=turbine_name,
        baseline_reference=baseline_reference,
        underutilised_threshold_pct=underutilised_threshold_pct,
        acceptable_threshold_pct=acceptable_threshold_pct,
    )

    summary_path, metadata_path = export_case_outputs(
        case_id=case_id,
        case_summary_df=case_summary_df,
        metadata=metadata,
        output_root=output_root.resolve(),
    )

    print_case_summary(
        case_id=case_id,
        case_summary_df=case_summary_df,
        summary_path=summary_path,
        metadata_path=metadata_path,
    )


def main() -> int:
    try:
        args = parse_args()

        if args.acceptable_threshold_pct <= args.underutilised_threshold_pct:
            raise ValueError(
                "--acceptable-threshold-pct must be greater than --underutilised-threshold-pct."
            )

        available = list_available_cases()

        if args.case_id == "?":
            if not available:
                print("No optimisation cases with candidate_layouts found.")
                return 0

            print("Available cases:")
            for case_id in available:
                print(f" - {case_id}")
            return 0

        if args.all:
            case_ids = available
            if not case_ids:
                print("No optimisation cases with candidate_layouts found.")
                return 0
        else:
            if args.case_id not in available:
                raise ValueError(
                    f"Unknown case '{args.case_id}'. Use --case ? to list available cases."
                )
            case_ids = [args.case_id]

        baseline_reference = load_baseline_reference(args.baseline_reference_csv.resolve())

        for case_id in case_ids:
            run_case(
                case_id=case_id,
                turbine_name=args.turbine_name,
                output_root=args.output_root.resolve(),
                baseline_reference=baseline_reference,
                underutilised_threshold_pct=args.underutilised_threshold_pct,
                acceptable_threshold_pct=args.acceptable_threshold_pct,
            )

        return 0

    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())