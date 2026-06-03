#!/bin/bash
# Single training run inside a Singularity container.
#
# Usage:
#   sbatch slurm/train.sh                          # uses defaults below
#   sbatch --export=DOMAIN=mlrod slurm/train.sh    # override domain
#   sbatch --export=ALL,DOMAIN=bacteria,SEED=123 slurm/train.sh
#
# Adjust the #SBATCH lines to match your cluster's partition names and limits.

#SBATCH --job-name=micro_ir_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
##SBATCH --partition=gpu        # uncomment and set your partition
##SBATCH --account=myproject    # uncomment if required

set -euo pipefail

# ── paths (edit these) ────────────────────────────────────────────────────────
SIF=/scratch/${USER}/micro_ir.sif          # path to your .sif file
DATA_DIR=/mnt/ssd3                         # bind-mount for zarr data
OUTPUT_DIR=/scratch/${USER}/outputs        # mlflow DB, checkpoints, hydra outputs

# ── sweep parameters (can be overridden via --export) ─────────────────────────
DOMAIN=${DOMAIN:-microplastic}
SEED=${SEED:-42}

mkdir -p "${OUTPUT_DIR}" logs

echo "Job ${SLURM_JOB_ID} | domain=${DOMAIN} | seed=${SEED}"
echo "Node: $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

singularity exec --nv \
    --bind "${DATA_DIR}:${DATA_DIR}" \
    --bind "${OUTPUT_DIR}:/app/outputs" \
    "${SIF}" \
    python main.py \
        domain="${DOMAIN}" \
        seed="${SEED}" \
        mode=train \
        mlflow.tracking_uri="sqlite:////app/outputs/mlflow.db" \
        mlflow.experiment_name="${DOMAIN}_training"
