from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PREP_SCRIPT = REPO_ROOT / "src" / "wake" / "prepare_wake_inputs.py"


def main() -> int:
    if not PREP_SCRIPT.exists():
        print(f"ERROR: Script not found: {PREP_SCRIPT}")
        return 1

    result = subprocess.run([sys.executable, str(PREP_SCRIPT)], check=False)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())