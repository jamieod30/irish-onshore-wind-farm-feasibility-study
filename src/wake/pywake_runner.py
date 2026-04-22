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

DEFAULT_LAYOUT = (
    REPO_ROOT
    / "data"
    / "optimisation"
    / "baseline"
    / "layout"
    / "layout_baseline_aligned.csv"
)

DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "wake" / "baseline_noj"

REQUIRED_BASE_LAYOUT_COLUMNS = [
    "turbine_id",
    "easting_m",
    "northing_m",
]

OPTIONAL_ENRICHED_COLUMNS = [
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
        description="Run PyWake NOJ wake model for a supplied layout."
    )
    parser.add_argument("--layout-csv", type=Path, default=DEFAULT_LAYOUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--turbine-name", type=str, default=DEFAULT_TURBINE)
    parser.add_argument("--skip-wake-map", action="store_true")
    parser.add_argument("--wake-map-wd", type=float, default=None)
    parser.add_argument("--wake-map-ws", type=float, default=10.0)
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def path_relative_to_repo(path: Path | str) -> str:
    path = Path(path).resolve()
    try:
        return str(path.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


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


def infer_layout_name(layout_csv: Path, df: pd.DataFrame) -> str:
    if "layout_name" in df.columns and not df["layout_name"].dropna().empty:
        return str(df["layout_name"].dropna().iloc[0])
    return layout_csv.stem


def load_layout(
    layout_csv: Path,
    turbine_name: str = DEFAULT_TURBINE,
) -> pd.DataFrame:
    if not layout_csv.exists():
        raise FileNotFoundError(f"Layout file not found: {layout_csv}")

    df = pd.read_csv(layout_csv)

    missing_base = [c for c in REQUIRED_BASE_LAYOUT_COLUMNS if c not in df.columns]
    if missing_base:
        raise ValueError(f"Layout CSV missing required columns: {missing_base}")

    df = df.copy()

    for col in ["easting_m", "northing_m"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df[REQUIRED_BASE_LAYOUT_COLUMNS].isna().any().any():
        raise ValueError("Layout CSV contains invalid required geometry values.")

    if df["turbine_id"].duplicated().any():
        dupes = df.loc[df["turbine_id"].duplicated(), "turbine_id"].tolist()
        raise ValueError(f"Duplicate turbine IDs found: {dupes}")

    turbine_metadata, _ = load_turbine_metadata(turbine_name)

    default_values: dict[str, Any] = {
        "hub_height_m": float(turbine_metadata["hub_height_m"]),
        "rotor_diameter_m": float(turbine_metadata["rotor_diameter_m"]),
        "rated_power_mw": float(turbine_metadata["rated_power_mw"]),
        "turbine_model": str(turbine_metadata["turbine_model"]),
        "site": infer_site_name(layout_csv),
        "layout_name": layout_csv.stem,
    }

    for col, default_value in default_values.items():
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

    required_after_enrichment = REQUIRED_BASE_LAYOUT_COLUMNS + OPTIONAL_ENRICHED_COLUMNS
    if df[required_after_enrichment].isna().any().any():
        raise ValueError(
            "Layout CSV contains missing values after enrichment from turbine metadata."
        )

    return df.reset_index(drop=True)


def validate_layout_against_turbine_definition(
    layout_df: pd.DataFrame,
    turbine_name: str,
) -> None:
    metadata, _ = load_turbine_metadata(turbine_name)

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

    if abs(float(hub_height_unique[0]) - float(metadata["hub_height_m"])) > 0.5:
        raise ValueError("Hub height mismatch with turbine definition.")

    if abs(float(rotor_unique[0]) - float(metadata["rotor_diameter_m"])) > 0.5:
        raise ValueError("Rotor diameter mismatch with turbine definition.")

    if abs(float(rated_power_unique[0]) - float(metadata["rated_power_mw"])) > 0.05:
        raise ValueError("Rated power mismatch with turbine definition.")


def get_default_wake_map_case() -> tuple[float, float]:
    sector_df = load_sector_frequency()
    dominant_idx = sector_df["probability"].idxmax()
    dominant_wd = float(sector_df.loc[dominant_idx, "wd_deg"])
    representative_ws = 10.0
    return dominant_wd, representative_ws


def build_run_metadata(
    layout_df: pd.DataFrame,
    turbine_name: str,
    output_dir: Path,
    wd: np.ndarray,
    ws: np.ndarray,
) -> dict[str, Any]:
    site_cfg = load_site_config()
    source_file = Path(layout_df.attrs["source_file"]).resolve()
    site_name = infer_site_name(source_file)
    layout_name = infer_layout_name(source_file, layout_df)
    benchmark_case = site_name == "baseline"

    return {
        "run_name": output_dir.name,
        "model": "NOJ",
        "benchmark_case": benchmark_case,
        "site_name": site_name,
        "resource_site_name": site_cfg["site_name"],
        "turbine_name": turbine_name,
        "layout_name": layout_name,
        "layout_file": path_relative_to_repo(source_file),
        "resource_inputs": {
            "site_config": path_relative_to_repo(
                REPO_ROOT / "data" / "wake_inputs" / "selected_site" / "site_config.json"
            ),
            "sector_frequency": path_relative_to_repo(
                REPO_ROOT / "data" / "wake_inputs" / "selected_site" / "sector_frequency.csv"
            ),
            "weibull_global": path_relative_to_repo(
                REPO_ROOT / "data" / "wake_inputs" / "selected_site" / "weibull_global.csv"
            ),
        },
        "simulation_grid": {
            "wd_deg": wd.tolist(),
            "ws_ms_min": float(ws.min()),
            "ws_ms_max": float(ws.max()),
            "ws_step_ms": float(ws[1] - ws[0]) if len(ws) > 1 else None,
            "n_wd": int(len(wd)),
            "n_ws": int(len(ws)),
        },
        "losses_applied": {
            "wake": True,
            "availability": False,
            "electrical": False,
        },
        "outputs": [
            "farm_summary.csv",
            "turbine_summary.csv",
            "flow_case_summary.csv",
            "run_metadata.json",
        ],
        "assumptions": {
            "wake_model": "N. O. Jensen (NOJ) engineering wake model",
            "resource_model": (
                "UniformWeibullSite using 16 directional sectors with globally "
                "uniform Weibull parameters"
            ),
            "result_basis": "gross AEP = no wake losses only; net AEP = wake losses only",
        },
    }


def run_pywake_noj(
    layout_df: pd.DataFrame,
    turbine_name: str,
    output_dir: Path,
) -> WakeRunResult:
    site = build_site()
    turbine = build_wind_turbine(turbine_name)
    wf_model = NOJ(site, turbine)

    x = layout_df["easting_m"].to_numpy(float)
    y = layout_df["northing_m"].to_numpy(float)
    h = layout_df["hub_height_m"].to_numpy(float)

    sector_df = load_sector_frequency()
    wd = sector_df["wd_deg"].to_numpy(float)
    ws = np.arange(0.0, 31.0, 1.0)

    sim = wf_model(x=x, y=y, h=h, wd=wd, ws=ws)

    net = np.asarray(sim.aep_ilk(with_wake_loss=True), dtype=float)
    gross = np.asarray(sim.aep_ilk(with_wake_loss=False), dtype=float)

    turbine_net = net.sum(axis=(1, 2))
    turbine_gross = gross.sum(axis=(1, 2))
    wd_net = net.sum(axis=(0, 2))
    wd_gross = gross.sum(axis=(0, 2))

    farm_net = float(net.sum())
    farm_gross = float(gross.sum())

    wake_loss = farm_gross - farm_net
    wake_loss_pct = 100 * wake_loss / farm_gross if farm_gross > 0 else np.nan

    capacity = float(layout_df["rated_power_mw"].sum())
    hours = 8760.0

    turbine_summary = layout_df.copy()
    turbine_summary["gross_aep_gwh"] = turbine_gross
    turbine_summary["net_aep_gwh"] = turbine_net
    turbine_summary["wake_loss_gwh"] = turbine_gross - turbine_net
    turbine_summary["wake_loss_pct"] = np.where(
        turbine_summary["gross_aep_gwh"] > 0,
        100 * turbine_summary["wake_loss_gwh"] / turbine_summary["gross_aep_gwh"],
        np.nan,
    )
    turbine_summary["capacity_factor_net"] = (
        turbine_summary["net_aep_gwh"]
        / (turbine_summary["rated_power_mw"] * hours / 1000)
    )
    turbine_summary = turbine_summary.sort_values(
        "net_aep_gwh", ascending=False
    ).reset_index(drop=True)

    flow_case_summary = pd.DataFrame(
        {
            "wd_deg": wd,
            "gross_aep_gwh": wd_gross,
            "net_aep_gwh": wd_net,
        }
    )
    flow_case_summary["wake_loss_gwh"] = (
        flow_case_summary["gross_aep_gwh"] - flow_case_summary["net_aep_gwh"]
    )
    flow_case_summary["wake_loss_pct"] = np.where(
        flow_case_summary["gross_aep_gwh"] > 0,
        100 * flow_case_summary["wake_loss_gwh"] / flow_case_summary["gross_aep_gwh"],
        np.nan,
    )

    source_file = Path(layout_df.attrs["source_file"]).resolve()
    farm_summary = pd.DataFrame(
        [
            {
                "model": "NOJ",
                "site_name": infer_site_name(source_file),
                "layout_name": infer_layout_name(source_file, layout_df),
                "num_turbines": len(layout_df),
                "installed_capacity_mw": capacity,
                "gross_aep_gwh": farm_gross,
                "net_aep_gwh": farm_net,
                "wake_loss_gwh": wake_loss,
                "wake_loss_pct": wake_loss_pct,
                "gross_capacity_factor": farm_gross / (capacity * hours / 1000),
                "net_capacity_factor": farm_net / (capacity * hours / 1000),
                "availability_loss_applied": False,
                "electrical_loss_applied": False,
            }
        ]
    )

    metadata = build_run_metadata(
        layout_df=layout_df,
        turbine_name=turbine_name,
        output_dir=output_dir,
        wd=wd,
        ws=ws,
    )

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
    wf_model = NOJ(site, turbine)

    x = layout_df["easting_m"].to_numpy(float)
    y = layout_df["northing_m"].to_numpy(float)
    h = layout_df["hub_height_m"].to_numpy(float)

    default_wd, default_ws = get_default_wake_map_case()
    wd = float(default_wd if wake_map_wd is None else wake_map_wd)
    ws = float(default_ws if wake_map_ws is None else wake_map_ws)

    sim = wf_model(x=x, y=y, h=h, wd=[wd], ws=[ws])

    fig, ax = plt.subplots(figsize=(10, 8))
    flow_map = sim.flow_map(wd=wd, ws=ws)
    flow_map.plot_wake_map(ax=ax)

    for text in list(ax.texts):
        if text.get_text().strip().isdigit():
            text.remove()

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
    print("PyWake run complete.")
    print(f"Output directory: {path_relative_to_repo(output_dir)}")
    print(f"Gross AEP: {row['gross_aep_gwh']:.3f} GWh")
    print(f"Net AEP: {row['net_aep_gwh']:.3f} GWh")
    print(f"Wake loss: {row['wake_loss_pct']:.2f}%")
    print(f"Gross capacity factor: {row['gross_capacity_factor']:.3%}")
    print(f"Net capacity factor: {row['net_capacity_factor']:.3%}")


def main() -> int:
    args = parse_args()

    try:
        layout_csv = args.layout_csv.resolve()
        output_dir = args.output_dir.resolve()

        layout_df = load_layout(layout_csv, turbine_name=args.turbine_name)
        layout_df.attrs["source_file"] = str(layout_csv)

        validate_layout_against_turbine_definition(
            layout_df,
            args.turbine_name,
        )

        result = run_pywake_noj(layout_df, args.turbine_name, output_dir)
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

            if "wake_map.png" not in result.metadata["outputs"]:
                result.metadata["outputs"].append("wake_map.png")
                with open(output_dir / "run_metadata.json", "w", encoding="utf-8") as f:
                    json.dump(result.metadata, f, indent=2)

        print_summary(result, output_dir)
        return 0

    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())