from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
LAYOUT_QC_SCRIPT = REPO_ROOT / "src" / "layout" / "layout_qc.py"


def main() -> int:
    if not LAYOUT_QC_SCRIPT.exists():
        print(f"ERROR: Script not found: {LAYOUT_QC_SCRIPT}")
        return 1

    cmd = [sys.executable, str(LAYOUT_QC_SCRIPT), *sys.argv[1:]]

    print(f"Running: {LAYOUT_QC_SCRIPT}")
    result = subprocess.run(cmd, check=False)

    if result.returncode == 0:
        print("\nLayout QA complete.")
    elif result.returncode == 1:
        print("\nLayout QA complete with one or more QC failures.")
    else:
        print(
            f"\nERROR: Layout QC failed with exit code {result.returncode}: {LAYOUT_QC_SCRIPT}"
        )

    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())