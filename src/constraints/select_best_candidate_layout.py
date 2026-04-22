from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

OPTIMISATION_DIR = REPO_ROOT / "data" / "optimisation"
DEFAULT_WAKE_OUTPUT_ROOT = REPO_ROOT / "outputs" / "wake" / "optimisation"


@dataclass(frozen=True)
class PromotionResult:
    case_id: str
    status: str
    message: str
    selected_layout_name: str | None = None
    promoted_layout_csv: Path | None = None
    metadata_json: Path | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Promote selected candidate layouts into each case's selected_layout folder "
            "using candidate PyWake screening outputs."
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
        help="Promote selected layouts for all available optimisation cases.",
    )
    parser.add_argument(
        "--wake-output-root",
        type=Path,
        default=DEFAULT_WAKE_OUTPUT_ROOT,
        help="Root directory containing candidate PyWake outputs.",
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


def path_relative_to_repo(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT)).replace("\\", "/")


def candidate_summary_path(case_id: str, wake_output_root: Path) -> Path:
    return wake_output_root / case_id / "candidate_pywake_summary.csv"


def candidate_metadata_path(case_id: str, wake_output_root: Path) -> Path:
    return wake_output_root / case_id / "candidate_pywake_metadata.json"


def selected_layout_dir(case_id: str) -> Path:
    return OPTIMISATION_DIR / case_id / "selected_layout"


def selected_layout_csv_path(case_id: str) -> Path:
    return selected_layout_dir(case_id) / f"layout_{case_id}_aligned.csv"


def selected_layout_metadata_path(case_id: str) -> Path:
    return OPTIMISATION_DIR / case_id / "metadata" / "layout_metadata.json"


def load_candidate_summary(summary_csv: Path) -> pd.DataFrame:
    if not summary_csv.exists():
        raise FileNotFoundError(f"Candidate PyWake summary not found: {summary_csv}")

    df = pd.read_csv(summary_csv)
    if df.empty:
        raise ValueError(f"Candidate PyWake summary contains no rows: {summary_csv}")

    required_columns = [
        "layout_name",
        "source_layout_csv",
        "num_turbines",
        "installed_capacity_mw",
        "gross_aep_gwh",
        "net_aep_gwh",
        "wake_loss_gwh",
        "wake_loss_pct",
        "gross_capacity_factor",
        "net_capacity_factor",
        "wake_loss_band",
        "selected_layout_flag",
    ]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"Candidate PyWake summary missing required columns: {missing}"
        )

    return df


def load_candidate_metadata(metadata_json: Path) -> dict[str, Any]:
    if not metadata_json.exists():
        raise FileNotFoundError(f"Candidate PyWake metadata not found: {metadata_json}")

    with metadata_json.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_selected_row(summary_df: pd.DataFrame) -> pd.Series:
    selected = summary_df.loc[summary_df["selected_layout_flag"] == True].copy()
    if len(selected) != 1:
        raise ValueError(
            f"Expected exactly one selected layout in candidate summary, found {len(selected)}."
        )
    return selected.iloc[0]


def load_layout_csv(layout_csv: Path) -> pd.DataFrame:
    if not layout_csv.exists():
        raise FileNotFoundError(f"Selected source layout CSV not found: {layout_csv}")

    df = pd.read_csv(layout_csv)
    if df.empty:
        raise ValueError(f"Selected source layout CSV is empty: {layout_csv}")

    return df


def ensure_selected_layout_columns(
    layout_df: pd.DataFrame,
    case_id: str,
    selected_layout_name: str,
) -> pd.DataFrame:
    df = layout_df.copy()

    if "site" not in df.columns:
        df["site"] = case_id
    else:
        df["site"] = df["site"].fillna(case_id).replace("", case_id)

    if "layout_name" not in df.columns:
        df["layout_name"] = selected_layout_name
    else:
        df["layout_name"] = (
            df["layout_name"].fillna(selected_layout_name).replace("", selected_layout_name)
        )

    preferred_order = [
        "turbine_id",
        "easting_m",
        "northing_m",
        "hub_height_m",
        "rotor_diameter_m",
        "rated_power_mw",
        "turbine_model",
        "site",
        "layout_name",
    ]

    ordered = [c for c in preferred_order if c in df.columns]
    remainder = [c for c in df.columns if c not in ordered]
    return df.loc[:, ordered + remainder]


def build_selected_layout_metadata(
    *,
    case_id: str,
    selected_row: pd.Series,
    selected_layout_csv: Path,
    candidate_metadata: dict[str, Any],
    wake_output_root: Path,
) -> dict[str, Any]:
    best_candidate = candidate_metadata.get("best_candidate", {})
    baseline_reference = candidate_metadata.get("baseline_reference", {})
    thresholds = candidate_metadata.get("wake_loss_thresholds_pct", {})

    return {
        "case_id": case_id,
        "layout_name": str(selected_row["layout_name"]),
        "site": case_id,
        "status": "selected optimisation case layout",
        "description": (
            "Selected layout promoted from candidate layout screening using PyWake-based "
            "feasibility-stage comparison against the benchmark wake-loss band."
        ),
        "layout_file": path_relative_to_repo(selected_layout_csv),
        "selection_source": {
            "candidate_pywake_summary": path_relative_to_repo(
                candidate_summary_path(case_id, wake_output_root)
            ),
            "candidate_pywake_metadata": path_relative_to_repo(
                candidate_metadata_path(case_id, wake_output_root)
            ),
            "source_layout_csv": str(selected_row["source_layout_csv"]),
        },
        "selection_result": {
            "selected_layout_name": str(selected_row["layout_name"]),
            "num_turbines": int(selected_row["num_turbines"]),
            "installed_capacity_mw": float(selected_row["installed_capacity_mw"]),
            "gross_aep_gwh": float(selected_row["gross_aep_gwh"]),
            "net_aep_gwh": float(selected_row["net_aep_gwh"]),
            "wake_loss_gwh": float(selected_row["wake_loss_gwh"]),
            "wake_loss_pct": float(selected_row["wake_loss_pct"]),
            "gross_capacity_factor": float(selected_row["gross_capacity_factor"]),
            "net_capacity_factor": float(selected_row["net_capacity_factor"]),
            "wake_loss_band": str(selected_row["wake_loss_band"]),
        },
        "benchmark_reference": baseline_reference,
        "wake_loss_thresholds_pct": thresholds,
        "best_candidate_snapshot": best_candidate,
        "limitations": [
            "Feasibility-stage candidate selection only",
            "Selection rule is based on acceptable wake-loss band then highest net AEP",
            "Detailed micro-siting, electrical design, and planning review still required",
        ],
    }


def promote_case(case_id: str, wake_output_root: Path) -> PromotionResult:
    summary_csv = candidate_summary_path(case_id, wake_output_root)
    metadata_json = candidate_metadata_path(case_id, wake_output_root)

    if not summary_csv.exists() or not metadata_json.exists():
        return PromotionResult(
            case_id=case_id,
            status="skipped",
            message="Candidate PyWake outputs not found.",
        )

    summary_df = load_candidate_summary(summary_csv)
    candidate_metadata = load_candidate_metadata(metadata_json)
    selected_row = resolve_selected_row(summary_df)

    source_layout_csv = (REPO_ROOT / str(selected_row["source_layout_csv"])).resolve()
    layout_df = load_layout_csv(source_layout_csv)
    layout_df = ensure_selected_layout_columns(
        layout_df=layout_df,
        case_id=case_id,
        selected_layout_name=str(selected_row["layout_name"]),
    )

    out_dir = selected_layout_dir(case_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = selected_layout_csv_path(case_id)
    layout_df.to_csv(out_csv, index=False)

    out_metadata = selected_layout_metadata_path(case_id)
    out_metadata.parent.mkdir(parents=True, exist_ok=True)

    metadata = build_selected_layout_metadata(
        case_id=case_id,
        selected_row=selected_row,
        selected_layout_csv=out_csv,
        candidate_metadata=candidate_metadata,
        wake_output_root=wake_output_root,
    )

    with out_metadata.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return PromotionResult(
        case_id=case_id,
        status="success",
        message=(
            f"Selected {selected_row['layout_name']} "
            f"({float(selected_row['net_aep_gwh']):.3f} GWh, "
            f"{float(selected_row['wake_loss_pct']):.3f}% wake loss)"
        ),
        selected_layout_name=str(selected_row["layout_name"]),
        promoted_layout_csv=out_csv,
        metadata_json=out_metadata,
    )


def print_results(results: list[PromotionResult]) -> None:
    print("Selected candidate layout promotion complete.")
    print()

    for result in results:
        print(f"{result.case_id}: {result.status.upper()} - {result.message}")
        if result.promoted_layout_csv is not None:
            print(f"  Layout CSV: {result.promoted_layout_csv}")
        if result.metadata_json is not None:
            print(f"  Metadata JSON: {result.metadata_json}")

    print()
    print(f"Processed: {len(results)}")
    print(f"Succeeded: {sum(r.status == 'success' for r in results)}")
    print(f"Skipped: {sum(r.status == 'skipped' for r in results)}")
    print(f"Failed: {sum(r.status == 'failed' for r in results)}")


def main() -> int:
    args = parse_args()

    try:
        available_cases = list_available_cases()

        if args.case_id == "?":
            if not available_cases:
                print("No optimisation cases with candidate_layouts found.")
                return 0

            print("Available cases:")
            for case_id in available_cases:
                print(f" - {case_id}")
            return 0

        if args.all:
            case_ids = available_cases
            if not case_ids:
                print("No optimisation cases with candidate_layouts found.")
                return 0
        else:
            if args.case_id not in available_cases:
                raise ValueError(
                    f"Unknown case '{args.case_id}'. Use --case ? to list available cases."
                )
            case_ids = [args.case_id]

        results: list[PromotionResult] = []
        had_failure = False

        for case_id in case_ids:
            try:
                result = promote_case(case_id, args.wake_output_root.resolve())
            except Exception as exc:
                had_failure = True
                result = PromotionResult(
                    case_id=case_id,
                    status="failed",
                    message=str(exc),
                )
            results.append(result)

        print_results(results)
        return 1 if had_failure else 0

    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())