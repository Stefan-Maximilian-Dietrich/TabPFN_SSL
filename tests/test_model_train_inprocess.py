from pathlib import Path

import pandas as pd

from scripts.model_train import run_one_seed


def test_run_one_seed_writes_two_csvs(tmp_path):
    """
    Runs the training entrypoint in-process (not via subprocess),
    so coverage includes scripts/model_train.py and parts of scripts/functions.py.
    """
    results_dir = str(tmp_path / "results")

    # Uses the smoke task we added earlier
    run_one_seed(
        task_path="tasks/local_smoke.py",
        seed=0,
        jobid="noid",
        rank=0,
        results_dir=results_dir,
        experiment_nr=0,
    )

    # Expect: one CSV for SSL decision method and one for supervised baseline
    csv_files = list(Path(results_dir).rglob("*.csv"))
    assert len(csv_files) >= 2

    # Quick sanity check: schema exists
    df = pd.read_csv(csv_files[0])
    expected_cols = {"jobid", "rank", "seed", "cm_index", "true", "pred", "count"}
    assert expected_cols.issubset(df.columns)