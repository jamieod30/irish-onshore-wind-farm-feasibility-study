from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

PIPELINES = [
    REPO_ROOT / "pipelines" / "run_pipeline_sequence_1.py",
    REPO_ROOT / "pipelines" / "run_pipeline_sequence_2.py",
    REPO_ROOT / "pipelines" / "run_pipeline_sequence_3.py",
    REPO_ROOT / "pipelines" / "run_pipeline_sequence_4.py",
    REPO_ROOT / "pipelines" / "run_pipeline_sequence_5.py",
]


def run_pipeline(script_path: Path) -> None:
    print(f"\nRunning pipeline: {script_path.name}")
    result = subprocess.run([sys.executable, str(script_path)], check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Pipeline failed: {script_path}")


def main() -> int:
    try:
        for pipeline in PIPELINES:
            run_pipeline(pipeline)

        print("\nFull project pipeline complete.")
        print("Note: layout comparison per selected case is still run separately unless --all support is added.")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())