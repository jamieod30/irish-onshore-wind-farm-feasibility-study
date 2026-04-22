from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

SCRIPTS = [
    REPO_ROOT / "scripts" / "run_resource_assessment.py",
    REPO_ROOT / "scripts" / "run_site_selection.py",
    REPO_ROOT / "scripts" / "run_wake_inputs.py",
    REPO_ROOT / "scripts" / "run_generate_turbine_curves.py",
]


def run_script(script_path: Path) -> None:
    print(f"\nRunning: {script_path.name}")
    result = subprocess.run([sys.executable, str(script_path)], check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Script failed: {script_path}")


def main() -> int:
    try:
        for script in SCRIPTS:
            run_script(script)
        print("\nPipeline sequence 1 complete.")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())