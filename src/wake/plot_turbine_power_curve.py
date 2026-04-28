from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]

power_curve_path = REPO_ROOT / "data" / "turbines" / "vestas_v136_4p2mw" / "power_curve.csv"
output_path = REPO_ROOT / "outputs" / "figures" / "wake" / "vestas_v136_power_curve.png"


output_path.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(power_curve_path)

plt.figure(figsize=(7, 4.5))
plt.plot(df["wind_speed_ms"], df["power_kw"], linewidth=2)
for x in [3,12,25]:
    plt.axvline(x=x, linestyle="--", linewidth=0.8, alpha=0.5)
plt.text(3, 150, "Cut-in", ha="center", fontsize=9)
plt.text(12, 150, "Rated", ha="center", fontsize=9)
plt.text(25, 150, "Cut-out", ha="center", fontsize=9)
plt.xlabel("Wind speed (m/s)")
plt.ylabel("Power output (kW)")
plt.title("Vestas V136-4.2 MW Power Curve")
plt.grid(True, linewidth=0.4, alpha=0.5)
plt.tight_layout()
plt.savefig(output_path, dpi=300)
plt.close()

print(f"Saved figure to: {output_path}")