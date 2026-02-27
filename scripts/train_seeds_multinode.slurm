#!/bin/bash

#SBATCH --output=logs/%x-%j.out
#SBATCH --error=error/%x-%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --qos=gpu
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --export=ALL

set -euo pipefail

source ~/.secrets/hf.env

# Fail if token missing
if [ -z "$HUGGINGFACE_HUB_TOKEN" ]; then
  echo "ERROR: HUGGINGFACE_HUB_TOKEN is not set. Create ~/.secrets/hf.env"
  exit 1
fi


# configuration 
EXP_DIR=/dss/dsshome1/03/di35lox/SSH/TabPFN_SSL
IMG=/dss/dsshome1/03/di35lox/containers/nvidia+pytorch+23.10-py3.sqsh

NUM_SEEDS="${NUM_SEEDS:-5}"
BASE_SEED="${BASE_SEED:-0}"
EXP_NUM="${EXP_NUM:-0}"
TASK_FILE="${TASK_FILE:-tasks/experiments.py}"


cd "$EXP_DIR"

mkdir -p logs error

# results directory 
export RESULTS_DIR="$EXP_DIR/Results/"
mkdir -p "$RESULTS_DIR"

echo "=================================================="
echo "SLURM JOB START"
echo "JobName         = ${SLURM_JOB_NAME}"
echo "JobID           = ${SLURM_JOB_ID}"
echo "Partition       = ${SLURM_JOB_PARTITION:-}"
echo "NodeList        = ${SLURM_NODELIST:-}"
echo "SubmitDir       = ${SLURM_SUBMIT_DIR:-}"
echo "WorkingDir      = $(pwd)"
echo "RESULTS_DIR     = $RESULTS_DIR"
echo "NUM_SEEDS       = $NUM_SEEDS"
echo "BASE_SEED       = $BASE_SEED"
echo "EXP_NUM         = $EXP_NUM"
echo "=================================================="

# run setup
export NCCL_DEBUG=WARN

GPUS_ON_NODE="${SLURM_GPUS_ON_NODE:-1}"

export WORLD_SIZE=$(( SLURM_NNODES * GPUS_ON_NODE ))
export MASTER_ADDR
MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
export MASTER_PORT=29353

echo "WORLD_SIZE      = $WORLD_SIZE"
echo "SLURM_NNODES    = $SLURM_NNODES"
echo "GPUs per node   = $GPUS_ON_NODE"
echo "MASTER_ADDR     = $MASTER_ADDR"
echo "MASTER_PORT     = $MASTER_PORT"

# runtime measurment
START_TS=$(date +%s)

# Container + torchrun 
srun -N"$SLURM_NNODES" \
  --ntasks-per-node=1 \
  --container-mounts="$EXP_DIR:/workspace" \
  --container-image="$IMG" \
  torchrun \
    --nproc_per_node="$GPUS_ON_NODE" \
    --nnodes="$SLURM_NNODES" \
    --node_rank="$SLURM_NODEID" \
    --rdzv_id="$SLURM_JOB_ID" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    /workspace/scripts/model_runner.py \
        --task="$TASK_FILE" \
        --num-seeds="$NUM_SEEDS" \
        --base-seed="$BASE_SEED" \
        --exp-num="$EXP_NUM"


# return runtime and summary 
END_TS=$(date +%s)
RUNTIME_SEC=$((END_TS - START_TS))

GPUS_TOTAL=$(( SLURM_NNODES * GPUS_ON_NODE ))

echo "======================================================"
echo " JOB-ZUSAMMENFASSUNG"
echo "------------------------------------------------------"
echo "  Job-ID           : $SLURM_JOB_ID"
echo "  Job-Name         : $SLURM_JOB_NAME"
echo "  Partition        : ${SLURM_JOB_PARTITION:-}"
echo "  Nodes            : $SLURM_NNODES"
echo "  GPUs pro Node    : $GPUS_ON_NODE"
echo "  GPUs gesamt      : $GPUS_TOTAL"
echo "  Container-Image  : $IMG"
echo "  Results-Ordner   : $RESULTS_DIR"
echo "  Seeds gesamt     : $NUM_SEEDS"
echo "  Base-Seed        : $BASE_SEED"
echo "  Experiment NR.   : $EXP_NUM"
echo "  Laufzeit (Sek)   : $RUNTIME_SEC"
echo "======================================================"

