import os
import math
import argparse

import torch
from scripts.model_train import run_one_seed

import importlib.util
from pathlib import Path

def load_task_module(task_path: str):
    repo_root = Path(__file__).resolve().parent
    full_path = (repo_root / task_path).resolve()
    spec = importlib.util.spec_from_file_location("task_module", str(full_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore
    return module

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="tasks/experiments.py",
                    help="Pfad zum Task-Sheet unterhalb des Repo-Roots (z.B. tasks/experiments.py)")

    parser.add_argument("--num-seeds", type=int, required=True)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--exp-num", type=int, default=0)
    args = parser.parse_args()

    jobid = os.getenv("SLURM_JOB_ID", "noid")

    # Multi-GPU / Multi-Node (torchrun)
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))

    # Ergebnisse-Basisordner vom Slurm-Skript
    results_dir = os.getenv("RESULTS_DIR", "results")

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    print(
        f"Job {jobid} | WORLD_SIZE={world_size} | "
        f"RANK={rank} | LOCAL_RANK={local_rank} | Device={device}"
    )
    print(f"RESULTS_DIR = {results_dir}")
    task_path = args.task
    experiment_nr = args.exp_num
    total_seeds = args.num_seeds
    seeds_per_rank = math.ceil(total_seeds / world_size)
    start_index = rank * seeds_per_rank
    end_index = min(total_seeds, (rank + 1) * seeds_per_rank)

    if start_index >= total_seeds:
        print(f"Rank {rank}: keine Seeds zugewiesen.")
        return

    for seed_index in range(start_index, end_index):
        seed_id = args.base_seed + seed_index
        print(f"\n=== Rank {rank}: starte Seed {seed_id} ===")
        run_one_seed(task_path, seed_id, jobid, rank, results_dir, experiment_nr)

if __name__ == "__main__":
    main()
