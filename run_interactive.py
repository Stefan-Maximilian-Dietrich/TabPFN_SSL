#!/usr/bin/env python3
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

REPO = Path(__file__).resolve().parent
TASKS_DIR = REPO / "tasks"
MODEL_RUNNER = REPO / "scripts/model_runner.py"
SLURM_SCRIPT = REPO / "scripts/train_seeds_multinode.slurm"  # bei dir im Repo-Root

def have_cmd(cmd):
    from shutil import which
    return which(cmd) is not None

def is_lrz_like() -> bool:
    return ("SLURM_CLUSTER_NAME" in os.environ) or have_cmd("sbatch")

def list_task_files():
    if not TASKS_DIR.exists():
        return []
    return sorted([p for p in TASKS_DIR.glob("*.py") if p.is_file() and p.name != "__init__.py"])

def ask_choice(title: str, options: list[str], default_index: int = 0) -> str:
    print(f"\n{title}")
    for i, opt in enumerate(options, 1):
        prefix = "->" if (i - 1) == default_index else "  "
        print(f"{prefix} [{i}] {opt}")
    while True:
        raw = input(f"Selection (1-{len(options)}) [default {default_index+1}]: ").strip()
        if raw == "":
            return options[default_index]
        if raw.isdigit():
            k = int(raw)
            if 1 <= k <= len(options):
                return options[k - 1]
        print("Please enter a valid number.")

def ask_int(prompt: str, default: int) -> int:
    while True:
        raw = input(f"{prompt} [default {default}]: ").strip()
        if raw == "":
            return default
        try:
            return int(raw)
        except ValueError:
            print("Please enter an integer.")

def ask_exp_list():
    raw = input("Choose experiment: number (e.g. 3), range (e.g. 0-10) or 'all': ").strip().lower()
    if raw in ("all", ""):
        return list(range(0, 11))
    if re.fullmatch(r"\d+", raw):
        return [int(raw)]
    m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", raw)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if a > b:
            a, b = b, a
        return list(range(a, b + 1))
    print("Invalid input. Using 'all' (0..10).")
    return list(range(0, 11))


def run_local(task_file: Path, exp_nums: list[int], num_seeds: int, base_seed: int):
    print("\n[LOCAL] Starting local execution (sequential, without Slurm).")
    for exp in exp_nums:
        cmd = [
            sys.executable,
            "-m", "scripts.model_runner",
            "--task", str(task_file.relative_to(REPO)),
            "--num-seeds", str(num_seeds),
            "--base-seed", str(base_seed),
            "--exp-num", str(exp),
        ]
        print(f"\n[LOCAL] EXP_NUM={exp} -> {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

def submit_lrz(task_file: Path, exp_nums: list[int], num_seeds: int, base_seed: int):
    print("\n[LRZ] Submit via sbatch (each EXP_NUM is a separate job).")
    logs = REPO / "logs"
    err = REPO / "error"
    logs.mkdir(exist_ok=True)
    err.mkdir(exist_ok=True)

    for exp in exp_nums:
        jobname = f"TabPFN_{task_file.stem}_EXP{exp}"
        export = f"ALL,EXP_NUM={exp},NUM_SEEDS={num_seeds},BASE_SEED={base_seed},TASK_FILE={task_file.relative_to(REPO)}"
        cmd = [
            "sbatch",
            f"--job-name={jobname}",
            f"--export={export}",
            str(SLURM_SCRIPT),
        ]
        print(f"\n[LRZ] submit EXP_NUM={exp} -> {' '.join(cmd)}")
        out = subprocess.check_output(cmd, text=True).strip()
        print(out)

    print("\n[LRZ] Status:")
    try:
        subprocess.run(
            [
                "squeue",
                "-u", os.getenv("USER", ""),
                "-o", "%.18i %.12P %.40j %.8u %.2t %.10M %.6D %R"
            ],
            check=False
        )
    except FileNotFoundError:
        print("squeue not found (unexpected on LRZ).")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["auto", "local", "lrz"], default="auto",
                    help="auto detects sbatch; local forces local execution; lrz forces LRZ submit.")
    args = ap.parse_args()

    # choosing mode 
    auto_lrz = is_lrz_like()
    if args.mode == "auto":
        mode = "lrz" if auto_lrz else "local"
    else:
        mode = args.mode

    print("======================================")
    print(" TabPFN_SSL - Interactive Runner")
    print("======================================")
    print(f"Repo:  {REPO}")
    print(f"Mode:  {mode} (auto_detect_lrz={auto_lrz})")

    # choose sheet 
    task_files = list_task_files()
    if not task_files:
        print(f"ERROR: No task files found in {TASKS_DIR}.")
        sys.exit(1)

    task_choice = ask_choice(
        "Which task sheet would you like to run?",
        [p.name for p in task_files],
        default_index=0
    )
    task_file = TASKS_DIR / task_choice
    print(task_file)
    # choose exp
    exp_nums = ask_exp_list()

    # seeds
    num_seeds = ask_int("NUM_SEEDS", default=5)
    base_seed = ask_int("BASE_SEED", default=0)

    # run
    if mode == "local":
        run_local(task_file, exp_nums, num_seeds, base_seed)
    else:
        submit_lrz(task_file, exp_nums, num_seeds, base_seed)

    print("\nDone.")

if __name__ == "__main__":
    main()