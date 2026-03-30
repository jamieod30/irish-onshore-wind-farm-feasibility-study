from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]

SITE_SELECTION_SCRIPT = REPO_ROOT / "src" / "resource_assessment" / "site_selection.py"


def main() -> None:
    print(f"Running: {SITE_SELECTION_SCRIPT}")
    result = subprocess.run([sys.executable, str(SITE_SELECTION_SCRIPT)], check=False)

    if result.returncode != 0:
        raise RuntimeError(f"Script failed: {SITE_SELECTION_SCRIPT}")

    print("\nSite selection pipeline complete.")


if __name__ == "__main__":
    main()