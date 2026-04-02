from pathlib import Path
import json
import numpy as np
import pandas as pd
import argparse

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_TURBINE = "vestas_v136_4p2mw"
TURBINE_BASE_DIR = REPO_ROOT / "data" / "turbines"


def load_metadata(turbine_dir: Path) -> dict:
    metadata_file = turbine_dir / "turbine_metadata.json"

    if not metadata_file.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_file}")

    with open(metadata_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    required = [
        "rated_power_mw",
        "cut_in_ms",
        "rated_ws_ms",
        "cut_out_ms"
    ]

    missing = [k for k in required if k not in metadata]
    if missing:
        raise KeyError(f"Missing required metadata fields: {missing}")

    return metadata


def generate_power_curve(ws, cut_in, rated_ws, cut_out, rated_power_kw):
    power = []

    for v in ws:
        if v < cut_in:
            p = 0.0

        elif cut_in <= v < rated_ws:
            p = rated_power_kw * ((v - cut_in) / (rated_ws - cut_in)) ** 3

        elif rated_ws <= v < cut_out:
            p = rated_power_kw

        else:
            p = 0.0  # includes v >= cut_out

        power.append(p)

    return np.round(power, 2)


def generate_ct_curve(ws, cut_in, rated_ws, cut_out, ct_max=0.8):
    ct = []

    for v in ws:
        if v < cut_in:
            val = 0.0

        elif cut_in <= v < 6:
            val = ct_max

        elif 6 <= v < 10:
            val = ct_max - 0.05 * (v - 6)

        elif 10 <= v < rated_ws:
            val = 0.6

        elif rated_ws <= v < 20:
            val = 0.6 - 0.03 * (v - rated_ws)

        elif 20 <= v < cut_out:
            val = 0.3

        else:
            val = 0.0  # includes v >= cut_out

        ct.append(val)

    return np.round(ct, 3)


def main():
    parser = argparse.ArgumentParser(description="Generate turbine curves")
    parser.add_argument(
        "--turbine",
        type=str,
        default=DEFAULT_TURBINE,
        help="Turbine folder name (default: vestas_v136_4p2mw)"
    )

    args = parser.parse_args()

    turbine_dir = TURBINE_BASE_DIR / args.turbine

    if not turbine_dir.exists():
        raise FileNotFoundError(f"Turbine folder not found: {turbine_dir}")

    metadata = load_metadata(turbine_dir)

    cut_in = metadata["cut_in_ms"]
    rated_ws = metadata["rated_ws_ms"]
    cut_out = metadata["cut_out_ms"]
    rated_power_kw = metadata["rated_power_mw"] * 1000

    # Optional metadata fields with defaults
    ws_step = metadata.get("wind_speed_step_ms", 1.0)
    ct_max = metadata.get("ct_max", 0.8)

    ws = np.arange(0, cut_out + ws_step, ws_step)

    power = generate_power_curve(ws, cut_in, rated_ws, cut_out, rated_power_kw)
    ct = generate_ct_curve(ws, cut_in, rated_ws, cut_out, ct_max)

    power_df = pd.DataFrame({
        "wind_speed_ms": ws,
        "power_kw": power
    })

    ct_df = pd.DataFrame({
        "wind_speed_ms": ws,
        "ct": ct
    })

    # QA checks
    if power_df["power_kw"].max() < 0.95 * rated_power_kw:
        raise ValueError("Power curve peak too low relative to rated power")

    if (ct_df["ct"] > 1.0).any() or (ct_df["ct"] < 0.0).any():
        raise ValueError("CT values outside physical bounds")

    power_file = turbine_dir / "power_curve.csv"
    ct_file = turbine_dir / "ct_curve.csv"

    power_df.to_csv(power_file, index=False)
    ct_df.to_csv(ct_file, index=False)

    print("Turbine curves generated")
    print(f"Turbine: {args.turbine}")
    print(f"Power curve: {power_file}")
    print(f"CT curve: {ct_file}")
    print()
    print(f"Rated power: {rated_power_kw} kW")
    print(f"Cut-in: {cut_in} m/s | Rated: {rated_ws} m/s | Cut-out: {cut_out} m/s")
    print(f"Wind speed step: {ws_step} m/s")
    print(f"Max CT: {ct_df['ct'].max()}")


if __name__ == "__main__":
    main()