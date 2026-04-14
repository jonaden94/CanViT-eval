#!/bin/bash
#SBATCH --job-name=canvit-ade20k-seg
#SBATCH --account=def-skrishna_gpu
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/ade20k-seg-%j.out
#SBATCH --error=logs/ade20k-seg-%j.err

# ADE20K segmentation eval (single config).
# Usage: sbatch slurm/ade20k_seg.sbatch --probe-repo canvit/probe-ade20k-40k-s512-c32-in21k

source slurm/prologue.sh
exec uv run python -m canvit_eval ade20k-seg "$@"
