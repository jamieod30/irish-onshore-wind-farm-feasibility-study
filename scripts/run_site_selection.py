from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SITE_SELECTION_SCRIPT = REPO_ROOT / "src" / "resource_assessment" / "site_selection.py"


def main() -> int:
    if not SITE_SELECTION_SCRIPT.exists():
        print(f"ERROR: Script not found: {SITE_SELECTION_SCRIPT}")
        return 1

    print(f"Running: {SITE_SELECTION_SCRIPT}")
    result = subprocess.run([sys.executable, str(SITE_SELECTION_SCRIPT)], check=False)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())