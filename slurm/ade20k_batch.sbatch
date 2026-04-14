#!/bin/bash
#SBATCH --job-name=canvit-ade20k
#SBATCH --account=def-skrishna_gpu
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/ade20k-%A_%a.out
#SBATCH --error=logs/ade20k-%A_%a.err

# Runs the FULL batch eval matrix sequentially.
# On SLURM: submit as a single job (safe — skips existing files).
# On crockett: run directly without sbatch.
#
# Usage:
#   sbatch slurm/ade20k_batch.sbatch --n-runs 5          # Nibi (single long job)
#   bash slurm/ade20k_batch.sbatch --n-runs 1             # crockett (direct)
#
# For parallel SLURM jobs, use --dry-run to print individual commands:
#   uv run python -m canvit_eval.batch --n-runs 5 --dry-run | \
#       xargs -I{} sbatch --wrap="{}" slurm/ade20k_seg.sbatch

source slurm/prologue.sh
exec uv run python -m canvit_eval.batch "$@"
