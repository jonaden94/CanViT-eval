#!/bin/bash
#SBATCH -p jupyter
#SBATCH --delay-boot=1
#SBATCH -G V100:1
#SBATCH -t 1-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH -A nib00021
#SBATCH -C inet                  # HuggingFace Hub access (probe download)
#SBATCH --mem=128G
#SBATCH -o logs/jon_exp20_canvit_compare_patchers/in1k_clf_t30/job-%J.log
#SBATCH -e logs/jon_exp20_canvit_compare_patchers/in1k_clf_t30/job-%J.log

# IN1k frozen-probe eval @ n_timesteps=30 on the jupyter/V100 partition.
# The jupyter QOS ('interactive') forbids job arrays and allows only 2 jobs per
# user, so this is ONE job that LOOPS over all runs (not an array). It is
# RESUMABLE: runs whose *_in1k_t30.pt already exists are skipped, and a per-run
# failure is logged without aborting the loop -- so if the job is ever killed,
# just resubmit and it continues. Outputs to *_t30 files so the original n=10
# results are NOT overwritten.
#
# Submit: sbatch in1k_clf_t30_jupyter.sbatch
# Optionally pass a subset of runs:  sbatch in1k_clf_t30_jupyter.sbatch RUN1 RUN2 ...

cd /user/henrich1/u25995/jonathan/repos/CanViT-eval
source /user/henrich1/u25995/.bashrc   # before set -u: /etc/bashrc reads an unbound var
set -u
source slurm_nhr/env.sh

log() { echo "[$(date '+%H:%M:%S')] $*"; }

VENV=".venv-cu126"          # V100 build for the jupyter partition
[ -d "$VENV" ] || { log "ERROR: $VENV not found"; exit 1; }
[ -n "${IMAGENET_VAL:-}" ] && [ -d "$IMAGENET_VAL" ] || { log "ERROR: IMAGENET_VAL unset/missing: ${IMAGENET_VAL:-<unset>}"; exit 1; }

RUNS=(
    exp20-base-foveated exp20-cond1-fovi exp20-cond1-reg exp20-cond1-square
    exp20-cond2-fovi exp20-cond2-reg exp20-cond2-square
    exp20-cond3-fovi exp20-cond3-reg exp20-cond3-square
    exp20-cond4-fovi exp20-cond4-reg exp20-cond4-square
    exp20-cond5-strided exp20-cond6-strided exp20-uniform16 exp20-uniform8
)
[ "$#" -gt 0 ] && RUNS=("$@")

log "=== exp20 IN1k t30 loop :: ${#RUNS[@]} runs :: host $(hostname) ==="
for RUN in "${RUNS[@]}"; do
    OUT="results/jon_exp20_canvit_compare_patchers/${RUN}_in1k_t30.pt"
    CKPT_HF="/user/henrich1/u25995/jonathan/repos/CanViT-pretrain/logs/patcher_compare/$RUN/checkpoints/latest-hf"
    if [ -f "$OUT" ]; then log "skip $RUN (output exists)"; continue; fi
    if [ ! -d "$CKPT_HF" ]; then log "skip $RUN (no checkpoint: $CKPT_HF)"; continue; fi
    log "--- $RUN -> $OUT ---"
    "$VENV/bin/python" -m canvit_eval in1k-clf \
        --mode frozen \
        --episode.model-repo "$CKPT_HF" \
        --episode.n-timesteps 30 \
        --episode.policy coarse_to_fine \
        --output "$OUT" \
        --batch-size 64 \
        --num-workers 8 || log "FAILED $RUN (rc=$?) -- continuing"
done
log "=== in1k t30 loop done ==="
