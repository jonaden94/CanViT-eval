#!/bin/bash
#SBATCH --job-name=canvit-in1k-clf
#SBATCH --account=def-skrishna_gpu
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/in1k-clf-%j.out
#SBATCH --error=logs/in1k-clf-%j.err

# IN1K classification eval.
# Usage: sbatch slurm/in1k_clf.sbatch

source slurm/prologue.sh
exec uv run python -m canvit_eval in1k-clf "$@"
