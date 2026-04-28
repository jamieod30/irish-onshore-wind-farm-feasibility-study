from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]

INPUT_CSV = REPO_ROOT / "outputs" / "tables" / "resource" / "site_selection_scoring.csv"
OUTPUT_PNG = (
    REPO_ROOT
    / "outputs"
    / "figures"
    / "resource"
    / "mean_wind_speed_power_density_comparison.png"
)


def main() -> int:
    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)

    required = {"site", "rank", "mean_ws_100", "mean_power_density"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {INPUT_CSV}: {sorted(missing)}")

    df = df.sort_values("rank").reset_index(drop=True)
    df["region"] = df["site"].str.title()

    x = np.arange(len(df))
    width = 0.38

    fig, ax1 = plt.subplots(figsize=(8, 4.8))

    ax2 = ax1.twinx()

    bars1 = ax1.bar(
        x - width / 2,
        df["mean_ws_100"],
        width,
        label="Mean wind speed (m/s)",
    )

    bars2 = ax2.bar(
        x + width / 2,
        df["mean_power_density"],
        width,
        label="Mean power density (W/m²)",
        color='orange',
    )

    ax1.set_title("Mean Wind Speed and Power Density by Region")
    ax1.set_xlabel("Candidate region")
    ax1.set_ylabel("Mean wind speed at 100 m (m/s)")
    ax2.set_ylabel("Mean power density (W/m²)")

    ax1.set_xticks(x)
    ax1.set_xticklabels(df["region"])

    ax1.grid(axis="y", linewidth=0.4, alpha=0.4)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper right")

    ax1.bar_label(bars1, fmt="%.1f", padding=3, fontsize=7)
    ax2.bar_label(bars2, fmt="%.1f", padding=3, fontsize=7)

    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure to: {OUTPUT_PNG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())