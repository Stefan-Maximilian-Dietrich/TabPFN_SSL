# scripts/smoke_local.py
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
MODEL_RUNNER = REPO / "scripts" / "model_runner.py"


def main() -> int:
    os.environ.setdefault("RESULTS_DIR", str(REPO / "results_smoke"))

    cmd = [
        sys.executable,
        str(MODEL_RUNNER),
        "--task", "tasks/local_smoke.py",
        "--num-seeds", "1",
        "--base-seed", "0",
        "--exp-num", "0",
    ]
    print("[SMOKE] Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("[SMOKE] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())