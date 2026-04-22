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

import pandas as pd

from src.wake.pywake_runner import (
    WakeRunResult,
    export_results,
    export_wake_map,
    load_layout,
    run_pywake_noj,
    validate_layout_against_turbine_definition,
)
from src.wake.turbine_definition import DEFAULT_TURBINE

OPTIMISATION_DIR = REPO_ROOT / "data" / "optimisation"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "wake" / "selected_cases"


@dataclass(frozen=True)
class SelectedCaseRunArtifacts:
    case_id: str
    layout_csv: Path
    output_dir: Path
    result: WakeRunResult


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run PyWake NOJ for one or more selected optimisation case layouts. "
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
        help="Run all optimisation cases with selected layouts.",
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
        help="Root directory for selected-case wake outputs.",
    )
    parser.add_argument(
        "--skip-wake-map",
        action="store_true",
        help="Skip wake map generation.",
    )
    parser.add_argument(
        "--wake-map-wd",
        type=float,
        default=None,
        help="Wind direction in degrees for wake map. Defaults to dominant sector.",
    )
    parser.add_argument(
        "--wake-map-ws",
        type=float,
        default=10.0,
        help="Wind speed in m/s for wake map. Defaults to 10.0 m/s.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Wake map export resolution.",
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
        if resolve_selected_layout_csv(path.name, must_exist=False) is not None:
            case_ids.append(path.name)

    return case_ids


def resolve_selected_layout_csv(case_id: str, must_exist: bool = True) -> Path | None:
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
            f"Multiple selected layout CSVs found for case '{case_id}': "
            f"{[str(p.relative_to(REPO_ROOT)).replace(chr(92), '/') for p in existing]}"
        )

    selected_dir = case_dir / "selected_layout"
    discovered = sorted(selected_dir.glob("*.csv")) if selected_dir.exists() else []
    if len(discovered) == 1:
        return discovered[0]

    if must_exist:
        raise FileNotFoundError(
            f"Could not resolve selected layout CSV for case '{case_id}'. "
            f"Checked standard locations under {case_dir / 'selected_layout'}."
        )

    return None


def path_relative_to_repo(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT)).replace("\\", "/")


def selected_case_output_dir(output_root: Path, case_id: str) -> Path:
    return output_root / case_id


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

    layout_name = (
        str(turbine_summary["layout_name"].iloc[0])
        if "layout_name" in turbine_summary.columns and not turbine_summary.empty
        else layout_csv.stem
    )

    farm_summary["site_name"] = case_id
    farm_summary["layout_name"] = layout_name

    if "site" in turbine_summary.columns:
        turbine_summary["site"] = case_id
    else:
        turbine_summary["site"] = case_id

    if "layout_name" in turbine_summary.columns:
        turbine_summary["layout_name"] = layout_name
    else:
        turbine_summary["layout_name"] = layout_name

    metadata["run_name"] = output_dir.name
    metadata["benchmark_case"] = False
    metadata["site_name"] = case_id
    metadata["layout_name"] = layout_name
    metadata["layout_file"] = path_relative_to_repo(layout_csv.resolve())

    return WakeRunResult(
        farm_summary=farm_summary,
        turbine_summary=turbine_summary,
        flow_case_summary=flow_case_summary,
        metadata=metadata,
    )


def build_batch_summary_df(artifacts: list[SelectedCaseRunArtifacts]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for artifact in artifacts:
        farm_row = artifact.result.farm_summary.iloc[0]
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
                "wake_loss_pct": float(farm_row["wake_loss_pct"]),
                "gross_capacity_factor": float(farm_row["gross_capacity_factor"]),
                "net_capacity_factor": float(farm_row["net_capacity_factor"]),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(by=["case_id", "layout_name"], ascending=[True, True])
        .reset_index(drop=True)
    )


def export_batch_summary(
    artifacts: list[SelectedCaseRunArtifacts],
    output_root: Path,
) -> Path | None:
    if not artifacts:
        return None

    summary_df = build_batch_summary_df(artifacts)
    summary_path = output_root / "selected_cases_pywake_summary.csv"
    output_root.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)
    return summary_path


def print_results(
    artifacts: list[SelectedCaseRunArtifacts],
    summary_path: Path | None,
) -> None:
    print("Selected-case PyWake assessment complete.")
    print()

    for artifact in artifacts:
        row = artifact.result.farm_summary.iloc[0]
        print(f"Case: {artifact.case_id}")
        print(f"  Layout CSV: {artifact.layout_csv}")
        print(f"  Output directory: {artifact.output_dir}")
        print(f"  Turbines: {int(row['num_turbines'])}")
        print(f"  Gross AEP: {float(row['gross_aep_gwh']):.3f} GWh")
        print(f"  Net AEP: {float(row['net_aep_gwh']):.3f} GWh")
        print(f"  Wake loss: {float(row['wake_loss_pct']):.3f}%")
        print(f"  Net capacity factor: {float(row['net_capacity_factor']):.6f}")
        print()

    if summary_path is not None:
        print(f"Batch summary CSV: {summary_path}")


def run_case(
    case_id: str,
    *,
    turbine_name: str,
    output_root: Path,
    skip_wake_map: bool,
    wake_map_wd: float | None,
    wake_map_ws: float | None,
    dpi: int,
) -> SelectedCaseRunArtifacts:
    layout_csv = resolve_selected_layout_csv(case_id, must_exist=True).resolve()
    output_dir = selected_case_output_dir(output_root, case_id).resolve()

    layout_df = load_layout(layout_csv=layout_csv, turbine_name=turbine_name)
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

    if not skip_wake_map:
        export_wake_map(
            layout_df=layout_df,
            output_dir=output_dir,
            turbine_name=turbine_name,
            wake_map_wd=wake_map_wd,
            wake_map_ws=wake_map_ws,
            dpi=dpi,
        )
        if "wake_map.png" not in result.metadata["outputs"]:
            result.metadata["outputs"].append("wake_map.png")
            with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as f:
                json.dump(result.metadata, f, indent=2)

    return SelectedCaseRunArtifacts(
        case_id=case_id,
        layout_csv=layout_csv,
        output_dir=output_dir,
        result=result,
    )


def main() -> int:
    try:
        args = parse_args()
        available = list_available_cases()

        if args.case_id == "?":
            if not available:
                print("No optimisation cases with selected layouts found.")
                return 0

            print("Available cases:")
            for case_id in available:
                print(f" - {case_id}")
            return 0

        if args.all:
            case_ids = available
            if not case_ids:
                print("No optimisation cases with selected layouts found.")
                return 0
        else:
            if args.case_id not in available:
                raise ValueError(
                    f"Unknown case '{args.case_id}'. Use --case ? to list available cases."
                )
            case_ids = [args.case_id]

        artifacts: list[SelectedCaseRunArtifacts] = []
        for case_id in case_ids:
            artifacts.append(
                run_case(
                    case_id=case_id,
                    turbine_name=args.turbine_name,
                    output_root=args.output_root.resolve(),
                    skip_wake_map=args.skip_wake_map,
                    wake_map_wd=args.wake_map_wd,
                    wake_map_ws=args.wake_map_ws,
                    dpi=args.dpi,
                )
            )

        summary_path = export_batch_summary(artifacts, args.output_root.resolve())
        print_results(artifacts, summary_path)
        return 0

    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())