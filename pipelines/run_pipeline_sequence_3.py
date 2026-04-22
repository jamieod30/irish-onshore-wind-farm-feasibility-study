from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

COMMANDS = [
    [sys.executable, str(REPO_ROOT / "scripts" / "run_layout_qc.py"), "--all"],
    [sys.executable, str(REPO_ROOT / "scripts" / "run_layout_spacing.py")],
]


def run_command(cmd: list[str]) -> int:
    print(f"\nRunning: {' '.join(cmd[1:])}")
    result = subprocess.run(cmd, check=False)
    return result.returncode


def main() -> int:
    try:
        qc_cmd = COMMANDS[0]
        qc_returncode = run_command(qc_cmd)

        if qc_returncode == 0:
            pass
        elif qc_returncode == 1:
            print(
                "\nLayout QC reported one or more layout failures. "
                "Continuing pipeline sequence 3 so spacing outputs are still generated."
            )
        else:
            raise RuntimeError(f"Command failed: {' '.join(qc_cmd)}")

        for cmd in COMMANDS[1:]:
            returncode = run_command(cmd)
            if returncode != 0:
                raise RuntimeError(f"Command failed: {' '.join(cmd)}")

        print("\nPipeline sequence 3 complete.")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())