from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]

TIMESTEP_SCRIPT = REPO_ROOT / "src" / "data_processing" / "timestep_format.py"
RESOURCE_SCRIPT = REPO_ROOT / "src" / "resource_assessment" / "wind_resource_analysis.py"


def run_script(script_path: Path) -> None:
    print(f"\nRunning: {script_path}")
    result = subprocess.run([sys.executable, str(script_path)], check=False)

    if result.returncode != 0:
        raise RuntimeError(f"Script failed: {script_path}")


def main() -> None:
    run_script(TIMESTEP_SCRIPT)
    run_script(RESOURCE_SCRIPT)
    print("\nResource assessment pipeline complete.")


if __name__ == "__main__":
    main()