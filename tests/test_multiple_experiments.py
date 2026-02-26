from pathlib import Path

from scripts.model_train import run_one_seed


def test_two_experiments_by_calling_run_one_seed_twice(tmp_path):
    """
    run_one_seed runs exactly ONE experiment: Experiments[experiment_nr].
    To cover multiple experiments, we call it twice with different experiment_nr.
    """

    # We reuse an existing task that has Experiments (e.g. tasks/local_smoke.py).
    # If your local_smoke only has one experiment, use a task file that has two.
    task_path = "tasks/local_smoke.py"

    results_dir = str(tmp_path / "results")

    # experiment 0
    run_one_seed(
        task_path=task_path,
        seed=0,
        jobid="noid",
        rank=0,
        results_dir=results_dir,
        experiment_nr=0,
    )

    # experiment 1 (must exist!)
    run_one_seed(
        task_path=task_path,
        seed=1,
        jobid="noid",
        rank=0,
        results_dir=results_dir,
        experiment_nr=1,
    )

    csv_files = list(Path(results_dir).rglob("*.csv"))
    # each run writes >=2 CSVs (ssl + supervised)
    assert len(csv_files) >= 4