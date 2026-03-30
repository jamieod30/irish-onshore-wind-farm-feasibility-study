from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
PREP_SCRIPT = REPO_ROOT / "src" / "wake" / "prepare_wake_inputs.py"


def main() -> None:
    if not PREP_SCRIPT.exists():
        raise FileNotFoundError(f"Script not found: {PREP_SCRIPT}")

    print(f"Running: {PREP_SCRIPT}")
    result = subprocess.run([sys.executable, str(PREP_SCRIPT)], check=False)

    if result.returncode != 0:
        raise RuntimeError(f"Script failed: {PREP_SCRIPT}")

    print("\nWake input preparation complete.")


if __name__ == "__main__":
    main()