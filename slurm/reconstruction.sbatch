#!/bin/bash
#SBATCH --job-name=canvit-recon
#SBATCH --account=rrg-skrishna_gpu
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/recon-%j.out
#SBATCH --error=logs/recon-%j.err

# Reconstruction quality eval (ablation checkpoints).
# Usage: sbatch slurm/reconstruction.sbatch --model-repo canvit/canvitb16-abl-baseline-2026-03-15

source slurm/prologue.sh
exec uv run python -m canvit_eval reconstruction "$@"
