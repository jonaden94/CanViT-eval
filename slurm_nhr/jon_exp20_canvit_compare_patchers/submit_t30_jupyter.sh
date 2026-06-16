#!/bin/bash
# Launch the n_timesteps=30 eval on the jupyter/V100 partition.
# The jupyter QOS ('interactive') forbids arrays and allows only 2 jobs/user, so
# this submits exactly 2 LOOPING jobs (in1k + recon), each iterating over all 17
# runs internally and resumable (skips *_t30 outputs that already exist).
#   bash slurm_nhr/jon_exp20_canvit_compare_patchers/submit_t30_jupyter.sh
set -eu
cd /user/henrich1/u25995/jonathan/repos/CanViT-eval
D=slurm_nhr/jon_exp20_canvit_compare_patchers
mkdir -p logs/jon_exp20_canvit_compare_patchers/in1k_clf_t30 \
         logs/jon_exp20_canvit_compare_patchers/reconstruction_t30
sbatch "$D/in1k_clf_t30_jupyter.sbatch"
sbatch "$D/reconstruction_t30_jupyter.sbatch"
