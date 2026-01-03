#!/bin/bash
set -euo pipefail

cd ~/SSH/SystemTest/Experiment08

# optional: nur nötig, wenn du die Dateien wirklich direkt ausführst (./file)
chmod +x train_seeds_multinode.slurm || true
chmod +x decision.py classifier.py data.py experiments.py functions.py model_runner.py model_train.py || true

echo "Starting jobs..."

# >>> HIER deine Experiment-Nummern eintragen:
EXP_LIST=(103)

# optional: Partition oder andere sbatch-Optionen global setzen:
# SBATCH_EXTRA=(--partition=gpu --time=02:00:00)
SBATCH_EXTRA=()

for EXP_NUM in "${EXP_LIST[@]}"; do
  JOB_NAME="EXP${EXP_NUM}"

  echo "Submitting: JOB_NAME=${JOB_NAME}, EXP_NUM=${EXP_NUM}"

  # Job abschicken:
  # - Jobname wird beim Submit gesetzt
  # - EXP_NUM wird als Umgebungsvariable an den Job exportiert
  OUT=$(sbatch "${SBATCH_EXTRA[@]}" \
        --job-name="${JOB_NAME}" \
        --export=ALL,EXP_NUM="${EXP_NUM}" \
        train_seeds_multinode.slurm)

  # JobID aus "Submitted batch job <id>" ziehen
  JOBID=$(echo "$OUT" | awk '{print $4}')
  echo "  -> Submitted JobID: ${JOBID}"
done

echo
echo "Queue (user di35lox):"
squeue -u di35lox -o "%.18i %.20P %.20j %.20u %.2t %.10M %.6D %R"
