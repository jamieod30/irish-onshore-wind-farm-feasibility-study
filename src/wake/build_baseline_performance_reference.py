from __future__ import annotations

import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

FARM_SUMMARY_PATH = REPO_ROOT / "outputs" / "wake" / "baseline_noj" / "farm_summary.csv"
FLOW_CASE_SUMMARY_PATH = REPO_ROOT / "outputs" / "wake" / "baseline_noj" / "flow_case_summary.csv"
TURBINE_SUMMARY_PATH = REPO_ROOT / "outputs" / "wake" / "baseline_noj" / "turbine_summary.csv"

OUTPUT_DIR = REPO_ROOT / "data" / "optimisation" / "baseline" / "metadata"
OUTPUT_PATH = OUTPUT_DIR / "baseline_performance_reference.csv"


def read_single_row_csv(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"No rows found in {path}")
    if len(rows) > 1:
        raise ValueError(f"Expected one row in {path}, found {len(rows)}")

    return rows[0]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def to_float(value: str) -> float:
    return float(value.strip())


def build_reference_row() -> dict[str, str]:
    farm = read_single_row_csv(FARM_SUMMARY_PATH)
    flow_cases = read_csv_rows(FLOW_CASE_SUMMARY_PATH)
    turbines = read_csv_rows(TURBINE_SUMMARY_PATH)

    wake_loss_pct_by_direction = [to_float(r["wake_loss_pct"]) for r in flow_cases]
    turbine_wake_loss_pct = [to_float(r["wake_loss_pct"]) for r in turbines]
    turbine_net_cf = [to_float(r["capacity_factor_net"]) for r in turbines]

    peak_direction_row = max(flow_cases, key=lambda r: to_float(r["wake_loss_pct"]))
    worst_turbine_row = max(turbines, key=lambda r: to_float(r["wake_loss_pct"]))
    best_turbine_row = min(turbines, key=lambda r: to_float(r["wake_loss_pct"]))

    row = {
        "reference_case_id": "baseline_noj",
        "model": farm["model"],
        "site_name": farm["site_name"],
        "layout_name": farm["layout_name"],
        "num_turbines": farm["num_turbines"],
        "installed_capacity_mw": farm["installed_capacity_mw"],
        "gross_aep_gwh": farm["gross_aep_gwh"],
        "net_aep_gwh": farm["net_aep_gwh"],
        "wake_loss_gwh": farm["wake_loss_gwh"],
        "wake_loss_pct": farm["wake_loss_pct"],
        "gross_capacity_factor": farm["gross_capacity_factor"],
        "net_capacity_factor": farm["net_capacity_factor"],
        "availability_loss_applied": farm["availability_loss_applied"],
        "electrical_loss_applied": farm["electrical_loss_applied"],
        "source_farm_summary": str(FARM_SUMMARY_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
        "source_flow_case_summary": str(FLOW_CASE_SUMMARY_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
        "source_turbine_summary": str(TURBINE_SUMMARY_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
        "peak_direction_wake_loss_pct": f"{max(wake_loss_pct_by_direction):.6f}",
        "peak_direction_wake_loss_wd_deg": peak_direction_row["wd_deg"],
        "min_direction_wake_loss_pct": f"{min(wake_loss_pct_by_direction):.6f}",
        "mean_turbine_wake_loss_pct": f"{sum(turbine_wake_loss_pct) / len(turbine_wake_loss_pct):.6f}",
        "max_turbine_wake_loss_pct": f"{max(turbine_wake_loss_pct):.6f}",
        "worst_turbine_id": worst_turbine_row["turbine_id"],
        "min_turbine_wake_loss_pct": f"{min(turbine_wake_loss_pct):.6f}",
        "best_turbine_id": best_turbine_row["turbine_id"],
        "mean_turbine_net_capacity_factor": f"{sum(turbine_net_cf) / len(turbine_net_cf):.6f}",
        "notes": (
            "Benchmark resource-led 12-turbine baseline retained as fixed comparison case "
            "for later parcel-constrained layout assessment."
        ),
    }
    return row


def write_reference_csv(row: dict[str, str]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def main() -> int:
    try:
        row = build_reference_row()
        write_reference_csv(row)
        print(f"Wrote baseline performance reference to: {OUTPUT_PATH}")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())