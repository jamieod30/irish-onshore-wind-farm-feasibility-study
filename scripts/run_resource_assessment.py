from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

TIMESTEP_SCRIPT = REPO_ROOT / "src" / "data_processing" / "timestep_format.py"
RESOURCE_SCRIPT = REPO_ROOT / "src" / "resource_assessment" / "wind_resource_analysis.py"


def run_script(script_path: Path) -> int:
    if not script_path.exists():
        print(f"ERROR: Script not found: {script_path}")
        return 1

    print(f"\nRunning: {script_path}")
    result = subprocess.run([sys.executable, str(script_path)], check=False)
    return result.returncode


def main() -> int:
    timestep_rc = run_script(TIMESTEP_SCRIPT)
    if timestep_rc != 0:
        return timestep_rc

    resource_rc = run_script(RESOURCE_SCRIPT)
    if resource_rc != 0:
        return resource_rc

    print("\nResource assessment pipeline complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())