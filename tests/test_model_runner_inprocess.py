import os
import sys
from pathlib import Path

import pytest


def test_model_runner_assigns_correct_seed_range(monkeypatch, tmp_path):
    """
    Tests scripts/model_runner.py in-process by patching run_one_seed.
    This covers argument parsing and seed partition logic.
    """
    # Import in a way that works with our conftest.py sys.path adjustments
    import scripts.model_runner as mr

    calls = []

    def fake_run_one_seed(task_path, seed_id, jobid, rank, results_dir, experiment_nr):
        calls.append((task_path, seed_id, jobid, rank, results_dir, experiment_nr))

    # Patch the heavy function
    monkeypatch.setattr(mr, "run_one_seed", fake_run_one_seed)

    # Force CPU in runner
    monkeypatch.setattr(mr.torch.cuda, "is_available", lambda: False)

    # Simulate distributed environment: world_size=2, rank=1
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("SLURM_JOB_ID", "noid")
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path / "results"))

    # total seeds=5 => seeds_per_rank=ceil(5/2)=3
    # rank 0 gets seeds indices 0,1,2 => seed_id 100,101,102
    # rank 1 gets seeds indices 3,4   => seed_id 103,104
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "model_runner",
            "--task",
            "tasks/local_smoke.py",
            "--num-seeds",
            "5",
            "--base-seed",
            "100",
            "--exp-num",
            "0",
        ],
    )

    mr.main()

    got_seed_ids = [c[1] for c in calls]
    assert got_seed_ids == [103, 104]
    assert all(c[0] == "tasks/local_smoke.py" for c in calls)


def test_model_runner_no_seeds_assigned(monkeypatch, tmp_path, capsys):
    """
    Covers the early-return branch when a rank gets no seeds assigned.
    """
    import scripts.model_runner as mr

    monkeypatch.setattr(mr.torch.cuda, "is_available", lambda: False)

    # world_size=4, total_seeds=2 => seeds_per_rank=ceil(2/4)=1
    # rank=3 => start_index=3*1=3 >= total_seeds => should print "keine Seeds"
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path / "results"))

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "model_runner",
            "--task",
            "tasks/local_smoke.py",
            "--num-seeds",
            "2",
            "--base-seed",
            "0",
            "--exp-num",
            "0",
        ],
    )

    mr.main()
    out = capsys.readouterr().out
    assert "keine Seeds zugewiesen" in out