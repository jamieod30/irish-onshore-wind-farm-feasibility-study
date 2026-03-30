from pathlib import Path
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]

DERIVED_DIR = REPO_ROOT / "data" / "era5" / "derived"
TABLES_DIR = REPO_ROOT / "outputs" / "tables" / "resource"

SITE_COMPARISON_FILE = DERIVED_DIR / "site_comparison.csv"

SCORING_OUTPUT_FILE = DERIVED_DIR / "site_selection_scoring.csv"
RANKING_OUTPUT_FILE = DERIVED_DIR / "site_selection_ranking.csv"

TABLES_SCORING_OUTPUT_FILE = TABLES_DIR / "site_selection_scoring.csv"
TABLES_RANKING_OUTPUT_FILE = TABLES_DIR / "site_selection_ranking.csv"

DERIVED_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

WEIGHTS = {
    "resource_score": 0.45,
    "weibull_k_score": 0.10,
    "directional_consistency_score": 0.10,
    "winter_score": 0.10,
    "shear_score": 0.10,
    "turbulence_score": 0.15,
}

REQUIRED_SITE_COLUMNS = {
    "site",
    "latitude",
    "longitude",
    "mean_ws_100",
    "mean_power_density",
    "weibull_c",
    "weibull_k",
    "winter_mean_ws_100",
    "shear_exponent",
    "turbulence_proxy",
    "dominant_dir_1",
    "dominant_dir_2",
    "dominant_dir_1_pct",
    "dominant_dir_2_pct",
}


def validate_weights(weights: dict[str, float]) -> None:
    total = sum(weights.values())
    if not np.isclose(total, 1.0):
        raise ValueError(f"WEIGHTS must sum to 1.0, got {total:.6f}")


def validate_columns(df: pd.DataFrame, required: set[str], df_name: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{df_name} is missing required columns: {sorted(missing)}")


def min_max_score(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").copy()

    if s.isna().any():
        raise ValueError(f"Series contains non-numeric or missing values: {series.name}")

    s_min = s.min()
    s_max = s.max()

    if np.isclose(s_max, s_min):
        return pd.Series(3.0, index=s.index)

    if higher_is_better:
        scaled = 1.0 + 4.0 * (s - s_min) / (s_max - s_min)
    else:
        scaled = 1.0 + 4.0 * (s_max - s) / (s_max - s_min)

    return scaled


def build_resource_index(df: pd.DataFrame) -> pd.Series:
    resource_cols = ["mean_ws_100", "mean_power_density", "weibull_c"]

    normalised_parts = []
    for col in resource_cols:
        s = pd.to_numeric(df[col], errors="coerce")

        if s.isna().any():
            raise ValueError(f"Column {col} contains non-numeric or missing values.")

        s_min = s.min()
        s_max = s.max()

        if np.isclose(s_max, s_min):
            normalised = pd.Series(0.5, index=s.index)
        else:
            normalised = (s - s_min) / (s_max - s_min)

        normalised_parts.append(normalised.rename(col))

    resource_index = pd.concat(normalised_parts, axis=1).mean(axis=1)
    return resource_index


def shear_band_score(alpha: float) -> float:
    if pd.isna(alpha):
        return np.nan

    if alpha < 0.08:
        return 2.0
    elif alpha < 0.14:
        return 4.0
    elif alpha <= 0.20:
        return 5.0
    elif alpha <= 0.25:
        return 4.0
    elif alpha <= 0.30:
        return 3.0
    elif alpha <= 0.40:
        return 2.0
    else:
        return 1.0


def main() -> None:
    validate_weights(WEIGHTS)

    df = pd.read_csv(SITE_COMPARISON_FILE)
    validate_columns(df, REQUIRED_SITE_COLUMNS, "site_comparison.csv")

    if df["site"].duplicated().any():
        dupes = df.loc[df["site"].duplicated(), "site"].tolist()
        raise ValueError(f"Duplicate site names found in site_comparison.csv: {dupes}")

    df["top2_directional_share_pct"] = (
        pd.to_numeric(df["dominant_dir_1_pct"], errors="coerce")
        + pd.to_numeric(df["dominant_dir_2_pct"], errors="coerce")
    )

    if df["top2_directional_share_pct"].isna().any():
        raise ValueError(
            "Could not compute top2_directional_share_pct from dominant direction percentages."
        )

    df["resource_index"] = build_resource_index(df)

    df["resource_score"] = min_max_score(df["resource_index"], higher_is_better=True)
    df["weibull_k_score"] = min_max_score(df["weibull_k"], higher_is_better=True)
    df["directional_consistency_score"] = min_max_score(
        df["top2_directional_share_pct"], higher_is_better=True
    )
    df["winter_score"] = min_max_score(df["winter_mean_ws_100"], higher_is_better=True)
    df["shear_score"] = df["shear_exponent"].apply(shear_band_score)
    df["turbulence_score"] = min_max_score(df["turbulence_proxy"], higher_is_better=False)

    scoring_columns = [
        "resource_score",
        "weibull_k_score",
        "directional_consistency_score",
        "winter_score",
        "shear_score",
        "turbulence_score",
    ]

    if df[scoring_columns].isna().any().any():
        raise ValueError("One or more scoring columns contain missing values after scoring.")

    df["weighted_score"] = (
        df["resource_score"] * WEIGHTS["resource_score"]
        + df["weibull_k_score"] * WEIGHTS["weibull_k_score"]
        + df["directional_consistency_score"] * WEIGHTS["directional_consistency_score"]
        + df["winter_score"] * WEIGHTS["winter_score"]
        + df["shear_score"] * WEIGHTS["shear_score"]
        + df["turbulence_score"] * WEIGHTS["turbulence_score"]
    )

    df = df.sort_values("weighted_score", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)

    scoring_cols = [
        "rank",
        "site",
        "latitude",
        "longitude",
        "mean_ws_100",
        "mean_power_density",
        "weibull_c",
        "weibull_k",
        "winter_mean_ws_100",
        "shear_exponent",
        "turbulence_proxy",
        "dominant_dir_1",
        "dominant_dir_1_pct",
        "dominant_dir_2",
        "dominant_dir_2_pct",
        "top2_directional_share_pct",
        "resource_index",
        "resource_score",
        "weibull_k_score",
        "directional_consistency_score",
        "winter_score",
        "shear_score",
        "turbulence_score",
        "weighted_score",
    ]

    ranking_cols = [
        "rank",
        "site",
        "weighted_score",
    ]

    scoring_df = df[scoring_cols].copy()
    ranking_df = df[ranking_cols].copy()

    round_cols_scoring = [
        "latitude",
        "longitude",
        "mean_ws_100",
        "mean_power_density",
        "weibull_c",
        "weibull_k",
        "winter_mean_ws_100",
        "shear_exponent",
        "turbulence_proxy",
        "dominant_dir_1_pct",
        "dominant_dir_2_pct",
        "top2_directional_share_pct",
        "resource_index",
        "resource_score",
        "weibull_k_score",
        "directional_consistency_score",
        "winter_score",
        "shear_score",
        "turbulence_score",
        "weighted_score",
    ]

    scoring_df[round_cols_scoring] = scoring_df[round_cols_scoring].round(3)
    ranking_df["weighted_score"] = ranking_df["weighted_score"].round(3)

    scoring_df.to_csv(SCORING_OUTPUT_FILE, index=False)
    ranking_df.to_csv(RANKING_OUTPUT_FILE, index=False)

    scoring_df.to_csv(TABLES_SCORING_OUTPUT_FILE, index=False)
    ranking_df.to_csv(TABLES_RANKING_OUTPUT_FILE, index=False)

    print(f"Site selection scoring saved to: {SCORING_OUTPUT_FILE}")
    print(f"Site selection ranking saved to: {RANKING_OUTPUT_FILE}")
    print()
    print(ranking_df.to_string(index=False))


if __name__ == "__main__":
    main()