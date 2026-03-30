from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]

LAYOUT_QC_SCRIPT = REPO_ROOT / "src" / "layout" / "layout_qc.py"


def main() -> None:
    print(f"Running: {LAYOUT_QC_SCRIPT}")
    result = subprocess.run([sys.executable, str(LAYOUT_QC_SCRIPT)], check=False)

    if result.returncode != 0:
        raise RuntimeError(f"Script failed: {LAYOUT_QC_SCRIPT}")

    print("\nLayout QA complete.")


if __name__ == "__main__":
    main()