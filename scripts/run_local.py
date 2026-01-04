#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
RUNNER = REPO / "model_runner.py"

def run_one(exp_num: int, num_seeds: int, base_seed: int):
    cmd = [
        sys.executable, str(RUNNER),
        "--num-seeds", str(num_seeds),
        "--base-seed", str(base_seed),
        "--exp-num", str(exp_num),
    ]
    print(f"\n[LOCAL] Running EXP_NUM={exp_num}: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="all", help="exp number or 'all' or '0-10'")
    ap.add_argument("--num-seeds", type=int, default=5)
    ap.add_argument("--base-seed", type=int, default=0)
    args = ap.parse_args()

    if args.exp == "all":
        exp_list = list(range(0, 11))
    elif "-" in args.exp:
        a, b = args.exp.split("-", 1)
        exp_list = list(range(int(a), int(b) + 1))
    else:
        exp_list = [int(args.exp)]

    for exp_num in exp_list:
        run_one(exp_num, args.num_seeds, args.base_seed)

if __name__ == "__main__":
    main()
