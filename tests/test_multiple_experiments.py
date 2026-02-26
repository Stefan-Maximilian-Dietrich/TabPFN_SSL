import pytest

from scripts.model_train import run_one_seed


def test_run_one_seed_experiment_index_out_of_range(tmp_path):
    """
    tasks/local_smoke.py contains only one experiment (index 0).
    Requesting experiment_nr=1 must raise IndexError.
    This tests the expected error behaviour and increases coverage.
    """
    with pytest.raises(IndexError):
        run_one_seed(
            task_path="tasks/local_smoke.py",
            seed=1,
            jobid="noid",
            rank=0,
            results_dir=str(tmp_path / "results"),
            experiment_nr=1,
        )