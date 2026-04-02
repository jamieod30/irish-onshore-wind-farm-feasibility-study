from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "src" / "wake" / "generate_turbine_curves.py"


def main():
    if not SCRIPT.exists():
        raise FileNotFoundError(f"Script not found: {SCRIPT}")

    # Pass through CLI args if provided
    args = [sys.executable, str(SCRIPT)] + sys.argv[1:]

    print(f"Running: {SCRIPT}")
    result = subprocess.run(args, check=False)

    if result.returncode != 0:
        raise RuntimeError("Curve generation failed")

    print("\nTurbine curves generated successfully.")


if __name__ == "__main__":
    main()