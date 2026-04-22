from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "src" / "wake" / "pywake_runner.py"


def main() -> int:
    cmd = [
        sys.executable,
        str(RUNNER),
        "--layout-csv",
        str(
            REPO_ROOT
            / "data"
            / "optimisation"
            / "baseline"
            / "layout"
            / "layout_baseline_aligned.csv"
        ),
        "--output-dir",
        str(REPO_ROOT / "outputs" / "wake" / "baseline_noj"),
    ]

    result = subprocess.run(cmd, check=False)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())