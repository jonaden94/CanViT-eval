#!/bin/bash
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=0:04:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source bench/pt/slurm/env.sh
uv run python bench/pt/run.py "$@"
