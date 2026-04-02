from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from py_wake import NOJ

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.wake.site_definition import (
    build_site,
    load_sector_frequency,
    load_site_config,
)
from src.wake.turbine_definition import (
    DEFAULT_TURBINE,
    build_wind_turbine,
    load_turbine_metadata,
)

DEFAULT_LAYOUT = REPO_ROOT / "data" / "layouts" / "baseline_aligned" / "layout_baseline_aligned.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "wake" / "baseline_noj"

REQUIRED_LAYOUT_COLUMNS = [
    "turbine_id",
    "easting_m",
    "northing_m",
    "hub_height_m",
    "rotor_diameter_m",
    "rated_power_mw",
    "turbine_model",
]


@dataclass(frozen=True)
class WakeRunResult:
    farm_summary: pd.DataFrame
    turbine_summary: pd.DataFrame
    flow_case_summary: pd.DataFrame
    metadata: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run baseline PyWake NOJ wake model and export AEP results."
    )
    parser.add_argument(
        "--layout-csv",
        type=Path,
        default=DEFAULT_LAYOUT,
        help="Path to layout CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for CSV/JSON/PNG outputs.",
    )
    parser.add_argument(
        "--turbine-name",
        type=str,
        default=DEFAULT_TURBINE,
        help="Turbine folder name under data/turbines/.",
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


def path_relative_to_repo(path: Path | str) -> str:
    """Return a portable repository-relative path string where possible."""
    path = Path(path).resolve()
    try:
        return str(path.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def load_layout(layout_csv: Path) -> pd.DataFrame:
    if not layout_csv.exists():
        raise FileNotFoundError(f"Layout file not found: {layout_csv}")

    df = pd.read_csv(layout_csv)
    missing = [c for c in REQUIRED_LAYOUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Layout CSV is missing required columns: {missing}")

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
        raise ValueError("Layout CSV contains missing or non-numeric values in required fields.")

    if df["turbine_id"].duplicated().any():
        dupes = df.loc[df["turbine_id"].duplicated(), "turbine_id"].tolist()
        raise ValueError(f"Duplicate turbine_id values found: {dupes}")

    return df.reset_index(drop=True)


def validate_layout_against_turbine_definition(
    layout_df: pd.DataFrame,
    turbine_name: str,
    tolerance_m: float = 0.5,
    tolerance_mw: float = 0.05,
) -> None:
    metadata, _ = load_turbine_metadata(turbine_name)

    expected_hub_height_m = float(metadata["hub_height_m"])
    expected_rotor_diameter_m = float(metadata["rotor_diameter_m"])
    expected_rated_power_mw = float(metadata["rated_power_mw"])

    hub_height_unique = np.unique(layout_df["hub_height_m"].round(6))
    rotor_unique = np.unique(layout_df["rotor_diameter_m"].round(6))
    rated_power_unique = np.unique(layout_df["rated_power_mw"].round(6))
    model_unique = layout_df["turbine_model"].dropna().astype(str).unique()

    if len(hub_height_unique) != 1:
        raise ValueError(f"Layout contains multiple hub heights: {hub_height_unique.tolist()}")
    if len(rotor_unique) != 1:
        raise ValueError(f"Layout contains multiple rotor diameters: {rotor_unique.tolist()}")
    if len(rated_power_unique) != 1:
        raise ValueError(f"Layout contains multiple rated powers: {rated_power_unique.tolist()}")
    if len(model_unique) != 1:
        raise ValueError(f"Layout contains multiple turbine_model values: {model_unique.tolist()}")

    if abs(float(hub_height_unique[0]) - expected_hub_height_m) > tolerance_m:
        raise ValueError(
            f"Layout hub height {hub_height_unique[0]} m does not match turbine definition "
            f"{expected_hub_height_m} m."
        )
    if abs(float(rotor_unique[0]) - expected_rotor_diameter_m) > tolerance_m:
        raise ValueError(
            f"Layout rotor diameter {rotor_unique[0]} m does not match turbine definition "
            f"{expected_rotor_diameter_m} m."
        )
    if abs(float(rated_power_unique[0]) - expected_rated_power_mw) > tolerance_mw:
        raise ValueError(
            f"Layout rated power {rated_power_unique[0]} MW does not match turbine definition "
            f"{expected_rated_power_mw} MW."
        )


def get_default_wake_map_case() -> tuple[float, float]:
    sector_df = load_sector_frequency()
    dominant_idx = sector_df["probability"].idxmax()
    dominant_wd = float(sector_df.loc[dominant_idx, "wd_deg"])
    representative_ws = 10.0
    return dominant_wd, representative_ws


def run_pywake_noj(
    layout_df: pd.DataFrame,
    turbine_name: str = DEFAULT_TURBINE,
) -> WakeRunResult:
    site = build_site()
    turbine = build_wind_turbine(turbine_name)
    wind_farm_model = NOJ(site, turbine)

    x = layout_df["easting_m"].to_numpy(dtype=float)
    y = layout_df["northing_m"].to_numpy(dtype=float)
    h = layout_df["hub_height_m"].to_numpy(dtype=float)

    sector_df = load_sector_frequency()
    wd = sector_df["wd_deg"].to_numpy(dtype=float)

    # Explicit wind-speed grid to avoid hidden PyWake defaults.
    # 0-30 m/s at 1 m/s spacing is adequate for this feasibility-stage baseline.
    ws = np.arange(0.0, 31.0, 1.0, dtype=float)

    simulation_result = wind_farm_model(x=x, y=y, h=h, wd=wd, ws=ws)

    aep_net_ilk = np.asarray(simulation_result.aep_ilk(with_wake_loss=True), dtype=float)
    aep_gross_ilk = np.asarray(simulation_result.aep_ilk(with_wake_loss=False), dtype=float)

    if aep_net_ilk.ndim != 3:
        raise ValueError(
            f"Unexpected aep_ilk dimensions for net case: {aep_net_ilk.shape}. "
            "Expected (wt, wd, ws)."
        )
    if aep_gross_ilk.ndim != 3:
        raise ValueError(
            f"Unexpected aep_ilk dimensions for gross case: {aep_gross_ilk.shape}. "
            "Expected (wt, wd, ws)."
        )

    n_wt = len(layout_df)
    if aep_net_ilk.shape[0] != n_wt:
        raise ValueError(
            f"Mismatch between layout turbine count ({n_wt}) and AEP result shape "
            f"{aep_net_ilk.shape}."
        )
    if aep_gross_ilk.shape != aep_net_ilk.shape:
        raise ValueError(
            f"Net and gross AEP arrays have different shapes: "
            f"{aep_net_ilk.shape} vs {aep_gross_ilk.shape}."
        )
    if aep_net_ilk.shape[1] != len(wd):
        raise ValueError(
            f"Wind direction dimension mismatch: result has {aep_net_ilk.shape[1]}, "
            f"input wd has {len(wd)}."
        )
    if aep_net_ilk.shape[2] != len(ws):
        raise ValueError(
            f"Wind speed dimension mismatch: result has {aep_net_ilk.shape[2]}, "
            f"input ws has {len(ws)}."
        )

    turbine_net_gwh = aep_net_ilk.sum(axis=(1, 2))
    turbine_gross_gwh = aep_gross_ilk.sum(axis=(1, 2))

    wd_net_gwh = aep_net_ilk.sum(axis=(0, 2))
    wd_gross_gwh = aep_gross_ilk.sum(axis=(0, 2))

    farm_net_gwh = float(aep_net_ilk.sum())
    farm_gross_gwh = float(aep_gross_ilk.sum())
    farm_wake_loss_gwh = farm_gross_gwh - farm_net_gwh
    farm_wake_loss_pct = (
        100.0 * farm_wake_loss_gwh / farm_gross_gwh if farm_gross_gwh > 0 else np.nan
    )

    installed_capacity_mw = float(layout_df["rated_power_mw"].sum())
    hours_per_year = 8760.0
    gross_capacity_factor = farm_gross_gwh / (installed_capacity_mw * hours_per_year / 1000.0)
    net_capacity_factor = farm_net_gwh / (installed_capacity_mw * hours_per_year / 1000.0)

    turbine_summary = layout_df.copy()
    turbine_summary["gross_aep_gwh"] = turbine_gross_gwh
    turbine_summary["net_aep_gwh"] = turbine_net_gwh
    turbine_summary["wake_loss_gwh"] = turbine_summary["gross_aep_gwh"] - turbine_summary["net_aep_gwh"]
    turbine_summary["wake_loss_pct"] = np.where(
        turbine_summary["gross_aep_gwh"] > 0,
        100.0 * turbine_summary["wake_loss_gwh"] / turbine_summary["gross_aep_gwh"],
        np.nan,
    )
    turbine_summary["capacity_factor_net"] = turbine_summary["net_aep_gwh"] / (
        turbine_summary["rated_power_mw"] * hours_per_year / 1000.0
    )
    turbine_summary = turbine_summary.sort_values("net_aep_gwh", ascending=False).reset_index(drop=True)

    flow_case_summary = (
        pd.DataFrame(
            {
                "wd_deg": wd,
                "net_aep_gwh": wd_net_gwh,
                "gross_aep_gwh": wd_gross_gwh,
            }
        )
        .assign(
            wake_loss_gwh=lambda df: df["gross_aep_gwh"] - df["net_aep_gwh"],
            wake_loss_pct=lambda df: np.where(
                df["gross_aep_gwh"] > 0,
                100.0 * (df["gross_aep_gwh"] - df["net_aep_gwh"]) / df["gross_aep_gwh"],
                np.nan,
            ),
        )
        .sort_values("wd_deg")
        .reset_index(drop=True)
    )

    site_config = load_site_config()
    layout_name = "unknown"
    if "layout_name" in layout_df.columns and len(layout_df["layout_name"].dropna()) > 0:
        layout_name = str(layout_df["layout_name"].dropna().iloc[0])

    farm_summary = pd.DataFrame(
        [
            {
                "model": "NOJ",
                "site_name": site_config["site_name"],
                "layout_name": layout_name,
                "num_turbines": int(len(layout_df)),
                "installed_capacity_mw": installed_capacity_mw,
                "gross_aep_gwh": farm_gross_gwh,
                "net_aep_gwh": farm_net_gwh,
                "wake_loss_gwh": farm_wake_loss_gwh,
                "wake_loss_pct": farm_wake_loss_pct,
                "gross_capacity_factor": gross_capacity_factor,
                "net_capacity_factor": net_capacity_factor,
                "availability_loss_applied": False,
                "electrical_loss_applied": False,
            }
        ]
    )

    metadata = {
        "model": "NOJ",
        "turbine_name": turbine_name,
        "layout_file": path_relative_to_repo(layout_df.attrs.get("source_file", "")),
        "simulation_grid": {
            "wd_deg": wd.tolist(),
            "ws_ms_min": float(ws.min()),
            "ws_ms_max": float(ws.max()),
            "ws_step_ms": 1.0,
            "n_wd": int(len(wd)),
            "n_ws": int(len(ws)),
        },
        "assumptions": {
            "wake_model": "N. O. Jensen (NOJ) engineering wake model",
            "resource_model": (
                "UniformWeibullSite using 16 directional sectors with globally "
                "uniform Weibull parameters"
            ),
            "availability_losses_applied": False,
            "electrical_losses_applied": False,
            "result_basis": "gross AEP = no wake losses only; net AEP = wake losses only",
        },
    }

    return WakeRunResult(
        farm_summary=farm_summary,
        turbine_summary=turbine_summary,
        flow_case_summary=flow_case_summary,
        metadata=metadata,
    )


def export_results(result: WakeRunResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    result.farm_summary.to_csv(output_dir / "farm_summary.csv", index=False)
    result.turbine_summary.to_csv(output_dir / "turbine_summary.csv", index=False)
    result.flow_case_summary.to_csv(output_dir / "flow_case_summary.csv", index=False)

    with open(output_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(result.metadata, f, indent=2)


def export_wake_map(
    layout_df: pd.DataFrame,
    output_dir: Path,
    turbine_name: str,
    wake_map_wd: float | None,
    wake_map_ws: float | None,
    dpi: int,
) -> Path:
    site = build_site()
    turbine = build_wind_turbine(turbine_name)
    wind_farm_model = NOJ(site, turbine)

    x = layout_df["easting_m"].to_numpy(dtype=float)
    y = layout_df["northing_m"].to_numpy(dtype=float)
    h = layout_df["hub_height_m"].to_numpy(dtype=float)

    default_wd, default_ws = get_default_wake_map_case()
    wd = float(default_wd if wake_map_wd is None else wake_map_wd)
    ws = float(default_ws if wake_map_ws is None else wake_map_ws)

    sim_res = wind_farm_model(x=x, y=y, h=h, wd=[wd], ws=[ws])

    fig, ax = plt.subplots(figsize=(10, 8))
    flow_map = sim_res.flow_map(wd=wd, ws=ws)
    flow_map.plot_wake_map(ax=ax)

    # Remove default PyWake turbine index labels if present.
    for text in list(ax.texts):
        label = text.get_text().strip()
        if label.isdigit():
            text.remove()

    # Overlay project turbine IDs.
    rotor_diameter_m = float(layout_df["rotor_diameter_m"].iloc[0])
    x_offset = 0.06 * rotor_diameter_m
    y_offset = 0.06 * rotor_diameter_m

    for _, row in layout_df.iterrows():
        ax.text(
            float(row["easting_m"]) + x_offset,
            float(row["northing_m"]) + y_offset,
            str(row["turbine_id"]),
            fontsize=8,
            ha="left",
            va="bottom",
            color="black",
            bbox={
                "boxstyle": "round,pad=0.15",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.7,
            },
        )

    ax.set_xlabel("Easting [m]")
    ax.set_ylabel("Northing [m]")
    ax.set_title(f"NOJ wake map at {ws:.1f} m/s and {wd:.1f}°")
    fig.tight_layout()

    output_path = output_dir / "wake_map.png"
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def print_summary(result: WakeRunResult, output_dir: Path) -> None:
    row = result.farm_summary.iloc[0]
    print("Baseline NOJ wake model complete.")
    print(f"Output directory: {path_relative_to_repo(output_dir)}")
    print(f"Installed capacity: {row['installed_capacity_mw']:.1f} MW")
    print(f"Gross AEP: {row['gross_aep_gwh']:.3f} GWh")
    print(f"Net AEP (wake only): {row['net_aep_gwh']:.3f} GWh")
    print(f"Wake loss: {row['wake_loss_gwh']:.3f} GWh ({row['wake_loss_pct']:.2f}%)")
    print(f"Gross capacity factor: {row['gross_capacity_factor']:.3%}")
    print(f"Net capacity factor: {row['net_capacity_factor']:.3%}")
    print("No availability or electrical losses have been applied.")
    print("Top 5 turbines by net AEP:")
    cols = ["turbine_id", "gross_aep_gwh", "net_aep_gwh", "wake_loss_pct"]
    print(result.turbine_summary.loc[:, cols].head(5).to_string(index=False))


def main() -> None:
    args = parse_args()

    layout_csv = args.layout_csv.resolve()
    output_dir = args.output_dir.resolve()

    layout_df = load_layout(layout_csv)
    layout_df.attrs["source_file"] = str(layout_csv)

    validate_layout_against_turbine_definition(
        layout_df,
        turbine_name=args.turbine_name,
    )

    result = run_pywake_noj(
        layout_df=layout_df,
        turbine_name=args.turbine_name,
    )
    export_results(result, output_dir)

    if not args.skip_wake_map:
        export_wake_map(
            layout_df=layout_df,
            output_dir=output_dir,
            turbine_name=args.turbine_name,
            wake_map_wd=args.wake_map_wd,
            wake_map_ws=args.wake_map_ws,
            dpi=args.dpi,
        )

    print_summary(result, output_dir)


if __name__ == "__main__":
    main()