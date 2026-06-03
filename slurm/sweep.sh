#!/bin/bash
# Parallel hyperparameter sweep via SLURM job array + Optuna shared storage.
#
# Each array element runs exactly ONE Optuna trial. All elements coordinate
# through a shared SQLite file — Optuna's TPE sampler handles concurrency
# safely. No hydra-submitit-launcher needed.
#
# Usage:
#   # 50 trials, 4 simultaneous jobs:
#   sbatch --array=0-49%4 slurm/sweep.sh
#
#   # 100 trials, all parallel (watch your cluster limits):
#   sbatch --array=0-99 slurm/sweep.sh
#
#   # Override domain:
#   sbatch --array=0-49%4 --export=ALL,DOMAIN=bacteria slurm/sweep.sh
#
# After all jobs finish, inspect results:
#   singularity exec micro_ir.sif python -c "
#     import optuna
#     study = optuna.load_study(
#         study_name='bacteria_augmentation_sweep',
#         storage='sqlite:////scratch/${USER}/sweeps/bacteria_sweep.db'
#     )
#     print(study.best_trial)
#   "

#SBATCH --job-name=micro_ir_sweep
#SBATCH --output=logs/sweep_%A_%a.out
#SBATCH --error=logs/sweep_%A_%a.err
#SBATCH --time=04:00:00           # per-trial wall time — adjust to your domain
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
##SBATCH --partition=gpu           # uncomment and set your partition
##SBATCH --account=myproject       # uncomment if required

set -euo pipefail

# ── paths (edit these) ────────────────────────────────────────────────────────
SIF=/scratch/${USER}/micro_ir.sif
DATA_DIR=/mnt/ssd3
SWEEP_DIR=/scratch/${USER}/sweeps          # shared across all array elements
OUTPUT_DIR=/scratch/${USER}/outputs

# ── sweep parameters (can be overridden via --export) ─────────────────────────
DOMAIN=${DOMAIN:-bacteria}

mkdir -p "${SWEEP_DIR}" "${OUTPUT_DIR}" logs

STORAGE="sqlite:////scratch/${USER}/sweeps/${DOMAIN}_sweep.db"

echo "Job ${SLURM_ARRAY_JOB_ID}[${SLURM_ARRAY_TASK_ID}] | domain=${DOMAIN}"
echo "Node: $(hostname) | Storage: ${STORAGE}"

singularity exec --nv \
    --bind "${DATA_DIR}:${DATA_DIR}" \
    --bind "${SWEEP_DIR}:/scratch/${USER}/sweeps" \
    --bind "${OUTPUT_DIR}:/app/outputs" \
    "${SIF}" \
    python main.py \
        --multirun \
        domain="${DOMAIN}" \
        "hydra/sweeper=optuna_${DOMAIN}" \
        mode=train \
        hydra.sweeper.n_trials=1 \
        "hydra.sweeper.storage=${STORAGE}" \
        mlflow.tracking_uri="sqlite:////app/outputs/mlflow.db" \
        mlflow.experiment_name="${DOMAIN}_sweep_${SLURM_ARRAY_JOB_ID}"
