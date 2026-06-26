#!/bin/bash
# Submit both exp21 eval arrays (IN1k frozen-probe + teacher-reconstruction),
# 32 (run x override_scale) configs each, on grete:shared.
#   bash slurm_nhr/jon_exp21_modulation/submit.sh
set -eu
cd /user/henrich1/u25995/jonathan/repos/CanViT-eval
D=slurm_nhr/jon_exp21_modulation

mkdir -p logs/jon_exp21_modulation/in1k_clf logs/jon_exp21_modulation/reconstruction
mkdir -p results/jon_exp21_modulation

# Prereq: HF checkpoints must exist (run once):
#   bash /user/henrich1/u25995/jonathan/repos/CanViT-pretrain/slurm_nhr/runs/jon_exp21_modulation/_eval_convert_checkpoints.sh

sbatch "$D/in1k_clf.sbatch"
sbatch "$D/reconstruction.sbatch"
echo "submitted in1k_clf + reconstruction (array 0-31%4 each)"
