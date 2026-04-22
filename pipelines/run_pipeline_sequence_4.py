from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

COMMANDS = [
    [sys.executable, str(REPO_ROOT / "scripts" / "run_case_candidate_pywake.py"), "--all"],
    [sys.executable, str(REPO_ROOT / "scripts" / "run_select_best_candidate_layout.py"), "--all"],
]


def run_command(cmd: list[str]) -> None:
    print(f"\nRunning: {' '.join(cmd[1:])}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def main() -> int:
    try:
        for cmd in COMMANDS:
            run_command(cmd)
        print("\nPipeline sequence 4 complete.")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())