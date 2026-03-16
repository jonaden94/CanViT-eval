#!/bin/bash
#SBATCH --job-name=canvit-ade20k-batch
#SBATCH --account=rrg-skrishna_gpu
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/ade20k-batch-%j.out
#SBATCH --error=logs/ade20k-batch-%j.err

# Full batch eval: all policies × resolutions × runs.
# Usage: sbatch slurm/ade20k_batch.sbatch --n-runs 5

source slurm/prologue.sh
exec uv run python -m canvit_eval.batch "$@"
