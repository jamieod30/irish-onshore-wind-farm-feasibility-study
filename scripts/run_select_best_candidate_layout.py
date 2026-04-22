from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "src" / "constraints" / "select_best_candidate_layout.py"


def main() -> int:
    if not SCRIPT.exists():
        print(f"ERROR: Script not found: {SCRIPT}")
        return 1

    result = subprocess.run([sys.executable, str(SCRIPT), *sys.argv[1:]], check=False)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())