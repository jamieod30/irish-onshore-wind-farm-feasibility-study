from pathlib import Path
import json
import pandas as pd
import numpy as np

from py_wake.wind_turbines import WindTurbine
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_TURBINE = "vestas_v136_4p2mw"
TURBINE_BASE_DIR = REPO_ROOT / "data" / "turbines"


def load_turbine_metadata(turbine_name: str = DEFAULT_TURBINE) -> tuple[dict, Path]:
    turbine_dir = TURBINE_BASE_DIR / turbine_name
    metadata_file = turbine_dir / "turbine_metadata.json"

    if not turbine_dir.exists():
        raise FileNotFoundError(f"Turbine folder not found: {turbine_dir}")

    if not metadata_file.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_file}")

    with open(metadata_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    required_fields = [
        "turbine_model",
        "manufacturer",
        "rated_power_mw",
        "rotor_diameter_m",
        "hub_height_m",
    ]
    missing = [field for field in required_fields if field not in metadata]
    if missing:
        raise KeyError(f"Missing required metadata fields: {missing}")

    return metadata, turbine_dir


def load_curve_csv(curve_file: Path, required_columns: list[str]) -> pd.DataFrame:
    if not curve_file.exists():
        raise FileNotFoundError(f"Missing curve file: {curve_file}")

    df = pd.read_csv(curve_file)

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"{curve_file.name} is missing required columns: {missing}")

    df = df.copy()
    for col in required_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df[required_columns].isna().any().any():
        raise ValueError(f"{curve_file.name} contains non-numeric or missing values.")

    return df.sort_values(required_columns[0]).reset_index(drop=True)


def validate_curves(
    power_df: pd.DataFrame,
    ct_df: pd.DataFrame,
    rated_power_mw: float,
) -> None:
    power_ws = power_df["wind_speed_ms"].to_numpy()
    ct_ws = ct_df["wind_speed_ms"].to_numpy()

    if len(power_df) != len(ct_df):
        raise ValueError(
            f"Power and CT curves have different lengths: {len(power_df)} vs {len(ct_df)}"
        )

    if not np.allclose(power_ws, ct_ws):
        raise ValueError("Power and CT curve wind speed columns do not align.")

    if not np.all(np.diff(power_ws) > 0):
        raise ValueError("Wind speed values must be strictly increasing.")

    if (power_df["power_kw"] < 0).any():
        raise ValueError("Power curve contains negative values.")

    if (ct_df["ct"] < 0).any() or (ct_df["ct"] > 1).any():
        raise ValueError("CT curve contains values outside physical bounds [0, 1].")

    rated_power_kw = rated_power_mw * 1000.0
    max_power = float(power_df["power_kw"].max())

    if max_power < 0.95 * rated_power_kw:
        raise ValueError(
            f"Maximum power {max_power:.1f} kW is too low relative to rated power "
            f"{rated_power_kw:.1f} kW."
        )


def build_wind_turbine(turbine_name: str = DEFAULT_TURBINE) -> WindTurbine:
    metadata, turbine_dir = load_turbine_metadata(turbine_name)

    power_file = turbine_dir / "power_curve.csv"
    ct_file = turbine_dir / "ct_curve.csv"

    power_df = load_curve_csv(power_file, ["wind_speed_ms", "power_kw"])
    ct_df = load_curve_csv(ct_file, ["wind_speed_ms", "ct"])

    validate_curves(
        power_df=power_df,
        ct_df=ct_df,
        rated_power_mw=float(metadata["rated_power_mw"]),
    )

    ws = power_df["wind_speed_ms"].to_numpy(dtype=float)
    power_w = power_df["power_kw"].to_numpy(dtype=float) * 1000.0
    ct = ct_df["ct"].to_numpy(dtype=float)

    power_ct_function = PowerCtTabular(
        ws=ws,
        power=power_w,
        power_unit="W",
        ct=ct,
    )

    wind_turbine = WindTurbine(
        name=str(metadata["turbine_model"]),
        diameter=float(metadata["rotor_diameter_m"]),
        hub_height=float(metadata["hub_height_m"]),
        powerCtFunction=power_ct_function,
    )

    return wind_turbine


def summarize_turbine(turbine_name: str = DEFAULT_TURBINE) -> None:
    metadata, turbine_dir = load_turbine_metadata(turbine_name)

    power_df = load_curve_csv(
        turbine_dir / "power_curve.csv",
        ["wind_speed_ms", "power_kw"],
    )
    ct_df = load_curve_csv(
        turbine_dir / "ct_curve.csv",
        ["wind_speed_ms", "ct"],
    )

    validate_curves(
        power_df=power_df,
        ct_df=ct_df,
        rated_power_mw=float(metadata["rated_power_mw"]),
    )

    print("Turbine definition loaded successfully.")
    print(f"Turbine folder: {turbine_dir}")
    print(f"Model: {metadata['turbine_model']}")
    print(f"Manufacturer: {metadata['manufacturer']}")
    print(f"Rated power: {metadata['rated_power_mw']} MW")
    print(f"Rotor diameter: {metadata['rotor_diameter_m']} m")
    print(f"Hub height: {metadata['hub_height_m']} m")
    print(f"Power curve points: {len(power_df)}")
    print(f"CT curve points: {len(ct_df)}")
    print(f"Max power: {power_df['power_kw'].max():.1f} kW")
    print(f"Max CT: {ct_df['ct'].max():.3f}")


if __name__ == "__main__":
    summarize_turbine()