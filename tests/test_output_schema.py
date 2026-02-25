import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_smoke_output_csv_schema(tmp_path):
    """
    Runs the local smoke pipeline and checks that at least one CSV exists
    and contains expected columns from save_confusion_matrices_long.
    """
    env = dict(os.environ)
    env["RESULTS_DIR"] = str(tmp_path / "results")

    subprocess.run([sys.executable, "scripts/smoke_local.py"], check=True, env=env)

    csv_files = list(Path(env["RESULTS_DIR"]).rglob("*.csv"))
    assert csv_files, "No CSV output produced by smoke run."

    df = pd.read_csv(csv_files[0])

    # Minimal expected schema for long confusion-matrix CSV
    expected_cols = {"jobid", "rank", "seed", "cm_index", "row", "col", "value"}
    missing = expected_cols - set(df.columns)
    assert not missing, f"Missing columns in output CSV: {missing}"

    # Basic sanity checks
    assert len(df) > 0
    assert df["value"].notna().all()