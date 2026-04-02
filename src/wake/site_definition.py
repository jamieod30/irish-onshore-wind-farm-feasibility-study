from pathlib import Path
import json
import numpy as np
import pandas as pd

from py_wake.site import UniformWeibullSite

REPO_ROOT = Path(__file__).resolve().parents[2]

SITE_DIR = REPO_ROOT / "data" / "wake_inputs" / "selected_site"

SITE_CONFIG_FILE = SITE_DIR / "site_config.json"
SECTOR_FREQUENCY_FILE = SITE_DIR / "sector_frequency.csv"
WEIBULL_FILE = SITE_DIR / "weibull_global.csv"

EXPECTED_WD = np.arange(0.0, 360.0, 22.5)


def load_site_config() -> dict:
    if not SITE_CONFIG_FILE.exists():
        raise FileNotFoundError(f"Missing site config file: {SITE_CONFIG_FILE}")

    with open(SITE_CONFIG_FILE, "r", encoding="utf-8") as f:
        config = json.load(f)

    required = ["site_key", "site_name"]
    missing = [k for k in required if k not in config]
    if missing:
        raise KeyError(f"Missing required site config fields: {missing}")

    return config


def load_sector_frequency() -> pd.DataFrame:
    if not SECTOR_FREQUENCY_FILE.exists():
        raise FileNotFoundError(f"Missing sector frequency file: {SECTOR_FREQUENCY_FILE}")

    df = pd.read_csv(SECTOR_FREQUENCY_FILE)

    required_columns = {"wd_deg", "probability"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"{SECTOR_FREQUENCY_FILE.name} is missing required columns: {sorted(missing)}"
        )

    df = df.copy()
    df["wd_deg"] = pd.to_numeric(df["wd_deg"], errors="coerce")
    df["probability"] = pd.to_numeric(df["probability"], errors="coerce")

    if df[["wd_deg", "probability"]].isna().any().any():
        raise ValueError("Sector frequency file contains non-numeric or missing values.")

    df = df.sort_values("wd_deg").reset_index(drop=True)

    if len(df) != 16:
        raise ValueError(f"Expected 16 wind direction sectors, found {len(df)}.")

    if not np.allclose(df["wd_deg"].to_numpy(), EXPECTED_WD):
        raise ValueError(
            "Wind direction sectors do not match expected 16-sector format "
            "(0, 22.5, ..., 337.5)."
        )

    prob_sum = float(df["probability"].sum())
    if not np.isclose(prob_sum, 1.0, atol=1e-3):
        raise ValueError(
            f"Directional probabilities sum to {prob_sum:.6f}, not approximately 1.0."
        )

    if (df["probability"] < 0).any():
        raise ValueError("Directional probabilities contain negative values.")

    return df


def load_weibull_global() -> pd.DataFrame:
    if not WEIBULL_FILE.exists():
        raise FileNotFoundError(f"Missing Weibull file: {WEIBULL_FILE}")

    df = pd.read_csv(WEIBULL_FILE)

    required_columns = {"k", "c"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"{WEIBULL_FILE.name} is missing required columns: {sorted(missing)}"
        )

    if len(df) != 1:
        raise ValueError(f"Expected exactly one row in {WEIBULL_FILE.name}, found {len(df)}.")

    df = df.copy()
    df["k"] = pd.to_numeric(df["k"], errors="coerce")
    df["c"] = pd.to_numeric(df["c"], errors="coerce")

    if df[["k", "c"]].isna().any().any():
        raise ValueError("Weibull file contains non-numeric or missing values.")

    k = float(df.loc[0, "k"])
    c = float(df.loc[0, "c"])

    if k <= 0 or c <= 0:
        raise ValueError(f"Weibull parameters must be positive. Got k={k}, c={c}")

    return df


def build_site() -> UniformWeibullSite:
    sector_df = load_sector_frequency()
    weibull_df = load_weibull_global()

    wd = sector_df["wd_deg"].to_numpy(dtype=float)
    p_wd = sector_df["probability"].to_numpy(dtype=float)
    k = float(weibull_df.loc[0, "k"])
    a = float(weibull_df.loc[0, "c"])

    # UniformWeibullSite expects one Weibull A and k value per direction sector
    a_arr = np.full_like(wd, a, dtype=float)
    k_arr = np.full_like(wd, k, dtype=float)

    site = UniformWeibullSite(
        p_wd=p_wd,
        a=a_arr,
        k=k_arr,
        wd=wd,
        ti=0.1,
    )

    return site


def summarize_site() -> None:
    config = load_site_config()
    sector_df = load_sector_frequency()
    weibull_df = load_weibull_global()

    prob_sum = float(sector_df["probability"].sum())
    dominant = sector_df.sort_values("probability", ascending=False).head(3)

    print("Site definition loaded successfully.")
    print(f"Site key: {config['site_key']}")
    print(f"Site name: {config['site_name']}")
    print(f"Sector count: {len(sector_df)}")
    print(f"Directional probability sum: {prob_sum:.6f}")
    print(f"Weibull k: {float(weibull_df.loc[0, 'k']):.3f}")
    print(f"Weibull c: {float(weibull_df.loc[0, 'c']):.3f}")
    print("Top 3 directional sectors:")
    print(dominant.to_string(index=False))


if __name__ == "__main__":
    summarize_site()