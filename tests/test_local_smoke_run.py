# tests/test_local_smoke_run.py
import os
import subprocess
import sys
from pathlib import Path


def test_local_smoke_run_creates_outputs(tmp_path):
    env = dict(os.environ)
    env["RESULTS_DIR"] = str(tmp_path / "results")

    cmd = [sys.executable, "scripts/smoke_local.py"]
    subprocess.run(cmd, check=True, env=env)

    csv_files = list(Path(env["RESULTS_DIR"]).rglob("*.csv"))
    assert len(csv_files) >= 1, "Smoke run produced no CSV output files."