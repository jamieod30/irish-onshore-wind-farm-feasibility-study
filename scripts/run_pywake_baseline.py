from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "src" / "wake" / "pywake_runner.py"


def main() -> None:
    cmd = [
        sys.executable,
        str(RUNNER),
        "--layout-csv",
        str(REPO_ROOT / "data" / "layouts" / "baseline_aligned" / "layout_baseline_aligned.csv"),
        "--output-dir",
        str(REPO_ROOT / "outputs" / "wake" / "baseline_noj"),
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
