from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.constraints.layout_comparison import (
    DEFAULT_BASELINE_LAYOUT,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_REFINED_LAYOUT,
    DEFAULT_REFINED_METADATA,
    export_layout_comparison,
    print_layout_comparison_summary,
    run_layout_comparison,
    write_refined_layout_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare baseline and refined constraint-led layouts."
    )
    parser.add_argument(
        "--baseline-layout-csv",
        type=Path,
        default=DEFAULT_BASELINE_LAYOUT,
        help="Path to baseline layout CSV.",
    )
    parser.add_argument(
        "--refined-layout-csv",
        type=Path,
        default=DEFAULT_REFINED_LAYOUT,
        help="Path to refined layout CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for comparison outputs.",
    )
    parser.add_argument(
        "--refined-metadata-json",
        type=Path,
        default=DEFAULT_REFINED_METADATA,
        help="Path to refined layout metadata JSON output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    result = run_layout_comparison(
        baseline_layout_csv=args.baseline_layout_csv.resolve(),
        refined_layout_csv=args.refined_layout_csv.resolve(),
    )

    export_layout_comparison(
        result=result,
        output_dir=args.output_dir.resolve(),
    )

    # Also write refined layout metadata for later PyWake rerun.
    import pandas as pd
    refined_df = pd.read_csv(args.refined_layout_csv.resolve())
    write_refined_layout_metadata(
        refined_df=refined_df,
        refined_layout_csv=args.refined_layout_csv.resolve(),
        refined_metadata_json=args.refined_metadata_json.resolve(),
    )

    print_layout_comparison_summary(result)


if __name__ == "__main__":
    main()