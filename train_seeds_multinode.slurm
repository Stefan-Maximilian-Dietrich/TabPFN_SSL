#!/bin/bash
#
# Submit-Beispiel:
# sbatch --job-name="TabPFN_EXP3" --export=ALL,EXP_NUM=3 train_seeds_multinode.slurm
#
# EXP_NUM kommt aus der Umgebung (via --export). Default = 0.

#SBATCH --output=logs/%x-%j.out
#SBATCH --error=error/%x-%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --qos=gpu
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --export=ALL

set -euo pipefail

# >>> HIER deinen Token einfügen (den aus deinem Projekt):
export HUGGINGFACE_HUB_TOKEN="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"

############################
# Konfiguration des Experiments
############################
EXP_DIR=/dss/dsshome1/03/di35lox/SSH/SystemTest/Experiment08
IMG=/dss/dsshome1/03/di35lox/containers/nvidia+pytorch+23.10-py3.sqsh

NUM_SEEDS=100
BASE_SEED=0

# EXP_NUM wird von run.sh/sbatch gesetzt: --export=ALL,EXP_NUM=...
EXP_NUM="${EXP_NUM:-0}"

cd "$EXP_DIR"

# Wichtig: SLURM öffnet Output/Error sehr früh. Daher sollten die Ordner existieren.
mkdir -p logs error

# Ergebnisordner
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

############################
# Distributed / torchrun Setup
############################
export NCCL_DEBUG=WARN

# Fallback, falls SLURM_GPUS_ON_NODE nicht gesetzt ist
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

############################
# Laufzeitmessung starten
############################
START_TS=$(date +%s)

# ====== Container + torchrun ======
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
    /workspace/model_runner.py \
      --num-seeds="$NUM_SEEDS" \
      --base-seed="$BASE_SEED" \
      --exp-num="$EXP_NUM"

############################
# Laufzeit ausgeben + Zusammenfassung
############################
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

