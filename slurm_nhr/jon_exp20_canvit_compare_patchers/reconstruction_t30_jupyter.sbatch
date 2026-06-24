#!/bin/bash
#SBATCH -p jupyter
#SBATCH --delay-boot=1
#SBATCH -G V100:1
#SBATCH -t 2-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH -A nib00021
#SBATCH -C inet                  # HuggingFace Hub access (teacher download)
#SBATCH --mem=128G
#SBATCH -o logs/jon_exp20_canvit_compare_patchers/reconstruction_t30/job-%J.log
#SBATCH -e logs/jon_exp20_canvit_compare_patchers/reconstruction_t30/job-%J.log

# Teacher-reconstruction eval @ n_timesteps=30 on the jupyter/V100 partition.
# The jupyter QOS ('interactive') forbids job arrays and allows only 2 jobs per
# user, so this is ONE job that LOOPS over all runs (not an array). It is
# RESUMABLE: runs whose *_recon_t30.pt already exists are skipped, and a per-run
# failure is logged without aborting the loop -- so if the job is ever killed
# (this is the long one), just resubmit and it continues. Canvas-grid pinned to
# 32 (only standardized grid). Outputs to *_t30 files so the original n=10
# results are NOT overwritten.
#
# Submit: sbatch reconstruction_t30_jupyter.sbatch
# Optionally pass a subset of runs:  sbatch reconstruction_t30_jupyter.sbatch RUN1 RUN2 ...

cd /user/henrich1/u25995/jonathan/repos/CanViT-eval
source /user/henrich1/u25995/.bashrc   # before set -u: /etc/bashrc reads an unbound var
set -u
source slurm_nhr/env.sh

# V100 has 32 GB (vs the 40 GB A100 these were tuned on); the DINOv3 teacher +
# 30-glimpse episode OOMs at batch 64. expandable_segments reclaims fragmented
# reserve; batch size only affects throughput, not results.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

log() { echo "[$(date '+%H:%M:%S')] $*"; }

VENV=".venv-cu126"          # V100 build for the jupyter partition
FLAT_VAL=/user/henrich1/u25995/jonathan/datasets/imagenet1k-val-flattened
CACHE="results/jon_exp20_canvit_compare_patchers/_cache/dinov3b16_in1k_val_512.pt"
[ -d "$VENV" ] || { log "ERROR: $VENV not found"; exit 1; }
[ -d "$FLAT_VAL" ] || { log "ERROR: flattened val dir not found: $FLAT_VAL"; exit 1; }

RUNS=(
    exp20-base-foveated exp20-cond1-fovi exp20-cond1-reg exp20-cond1-square
    exp20-cond2-fovi exp20-cond2-reg exp20-cond2-square
    exp20-cond3-fovi exp20-cond3-reg exp20-cond3-square
    exp20-cond4-fovi exp20-cond4-reg exp20-cond4-square
    exp20-cond5-strided exp20-cond6-strided exp20-uniform16 exp20-uniform8
)
[ "$#" -gt 0 ] && RUNS=("$@")

log "=== exp20 reconstruction t30 loop :: ${#RUNS[@]} runs :: host $(hostname) ==="
for RUN in "${RUNS[@]}"; do
    OUT="results/jon_exp20_canvit_compare_patchers/${RUN}_recon_t30.pt"
    CKPT_HF="/user/henrich1/u25995/jonathan/repos/CanViT-pretrain/logs/patcher_compare/$RUN/checkpoints/latest-hf"
    if [ -f "$OUT" ]; then log "skip $RUN (output exists)"; continue; fi
    if [ ! -d "$CKPT_HF" ]; then log "skip $RUN (no checkpoint: $CKPT_HF)"; continue; fi
    log "--- $RUN -> $OUT ---"
    "$VENV/bin/python" -m canvit_eval reconstruction \
        --model-repo "$CKPT_HF" \
        --episode.n-timesteps 30 \
        --episode.policy coarse_to_fine \
        --episode.canvas-grid 32 \
        --image-dir "$FLAT_VAL" \
        --teacher-cache "$CACHE" \
        --output "$OUT" \
        --batch-size 16 \
        --num-workers 8 || log "FAILED $RUN (rc=$?) -- continuing"
done
log "=== reconstruction t30 loop done ==="
