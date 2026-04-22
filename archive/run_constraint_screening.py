from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.constraints.constraint_screening import (
    DEFAULT_HITS_CSV,
    DEFAULT_LAYOUT_CSV,
    DEFAULT_SCREENED_LAYOUT_CSV,
    DEFAULT_SCREENED_METADATA_JSON,
    DEFAULT_STATUS_CSV,
    DEFAULT_TABLES_DIR,
    export_constraint_screening,
    print_constraint_screening_summary,
    run_constraint_screening,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run feasibility-stage constraint screening for the baseline layout."
    )
    parser.add_argument("--layout-csv", type=Path, default=DEFAULT_LAYOUT_CSV)
    parser.add_argument("--hits-csv", type=Path, default=DEFAULT_HITS_CSV)
    parser.add_argument("--status-csv", type=Path, default=DEFAULT_STATUS_CSV)
    parser.add_argument("--screened-layout-csv", type=Path, default=DEFAULT_SCREENED_LAYOUT_CSV)
    parser.add_argument("--screened-metadata-json", type=Path, default=DEFAULT_SCREENED_METADATA_JSON)
    parser.add_argument("--tables-dir", type=Path, default=DEFAULT_TABLES_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        result = run_constraint_screening(
            layout_csv=args.layout_csv.resolve(),
            hits_csv=args.hits_csv.resolve(),
        )

        export_constraint_screening(
            result=result,
            status_csv=args.status_csv.resolve(),
            screened_layout_csv=args.screened_layout_csv.resolve(),
            screened_metadata_json=args.screened_metadata_json.resolve(),
            tables_dir=args.tables_dir.resolve(),
        )

        print_constraint_screening_summary(result)
        return 0

    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())