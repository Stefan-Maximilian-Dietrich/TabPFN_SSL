import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_smoke_output_csv_schema(tmp_path):
    """
    Runs the local smoke pipeline and checks that at least one CSV exists
    and contains the expected schema for the confusion-matrix long format.

    In this repo, the long format uses:
      jobid, rank, seed, cm_index, true, pred, count
    """
    env = dict(os.environ)
    env["RESULTS_DIR"] = str(tmp_path / "results")

    subprocess.run([sys.executable, "scripts/smoke_local.py"], check=True, env=env)

    csv_files = list(Path(env["RESULTS_DIR"]).rglob("*.csv"))
    assert csv_files, "No CSV output produced by smoke run."

    df = pd.read_csv(csv_files[0])

    expected_cols = {"jobid", "rank", "seed", "cm_index", "true", "pred", "count"}
    missing = expected_cols - set(df.columns)
    assert not missing, f"Missing columns in output CSV: {missing}"

    # Basic sanity checks
    assert len(df) > 0
    assert df["count"].notna().all()

    # Confusion-matrix coordinates should be 0/1 for binary classification
    assert set(df["true"].unique()).issubset({0, 1})
    assert set(df["pred"].unique()).issubset({0, 1})

    # Counts should be non-negative
    assert (df["count"] >= 0).all()