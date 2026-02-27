import pytest
from scripts.model_train import run_one_seed


def test_run_one_seed_invalid_task_path(tmp_path):
    with pytest.raises(Exception):
        run_one_seed(
            task_path="tasks/DOES_NOT_EXIST.py",
            seed=0,
            jobid="noid",
            rank=0,
            results_dir=str(tmp_path),
            experiment_nr=0,
        )