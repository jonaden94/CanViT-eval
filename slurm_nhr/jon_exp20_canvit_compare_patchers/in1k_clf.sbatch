#!/bin/bash
#SBATCH -p grete:shared
#SBATCH --nodes=1
#SBATCH --gpus-per-node=A100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -t 0-01:30:00
#SBATCH -A nib00021
#SBATCH -C inet                  # required for HuggingFace Hub access (probe download)
#SBATCH --mem=64G
#SBATCH --exclude=ggpu129,ggpu202
#SBATCH --array=0-16%1
#SBATCH -o logs/jon_exp20_canvit_compare_patchers/in1k_clf/job-%A_%a.log
#SBATCH -e logs/jon_exp20_canvit_compare_patchers/in1k_clf/job-%A_%a.log

# IN1k frozen-probe eval of the 17 exp20 jon_exp20_canvit_compare_patchers runs.
# One task per run (sequential array, %1). Each task:
#   - loads <run>/checkpoints/latest-hf/ (HF-format checkpoint @ step 200,704)
#   - runs coarse_to_fine policy for 10 timesteps (glimpses 0-9)
#   - writes results/jon_exp20_canvit_compare_patchers/<run>_in1k.pt
#
# Submit:
#   sbatch slurm_nhr/jon_exp20_canvit_compare_patchers/in1k_clf.sbatch

cd /user/henrich1/u25995/jonathan/repos/CanViT-eval
source /user/henrich1/u25995/.bashrc

set -eu

log() { echo "[$(date '+%H:%M:%S')] $*"; }

source slurm_nhr/env.sh

VENV=".venv"
if [ ! -d "$VENV" ]; then
    log "ERROR: $VENV not found. Run: uv sync"
    exit 1
fi
if [ -z "${IMAGENET_VAL:-}" ] || [ ! -d "$IMAGENET_VAL" ]; then
    log "ERROR: IMAGENET_VAL is unset or missing: ${IMAGENET_VAL:-<unset>}"
    exit 1
fi

# 20 stable runs (the 2 still-running tails — film-pos-sigma16-mlp768,
# pad-learned — are excluded; submit a follow-up array for them once they
# finish and their checkpoints are HF-converted).
RUNS=(
    exp20-base-foveated
    exp20-cond1-fovi
    exp20-cond1-reg
    exp20-cond1-square
    exp20-cond2-fovi
    exp20-cond2-reg
    exp20-cond2-square
    exp20-cond3-fovi
    exp20-cond3-reg
    exp20-cond3-square
    exp20-cond4-fovi
    exp20-cond4-reg
    exp20-cond4-square
    exp20-cond5-strided
    exp20-cond6-strided
    exp20-uniform16
    exp20-uniform8
)

RUN="${RUNS[$SLURM_ARRAY_TASK_ID]}"
CKPT_HF="/user/henrich1/u25995/jonathan/repos/CanViT-pretrain/logs/patcher_compare/$RUN/checkpoints/latest-hf"
OUT="results/jon_exp20_canvit_compare_patchers/${RUN}_in1k.pt"

if [ ! -d "$CKPT_HF" ]; then
    log "ERROR: HF-format checkpoint not found: $CKPT_HF"
    log "Convert with: CanViT-specialize/scripts/pretrain_ckpt_to_hf_format.py"
    exit 1
fi

log "=== exp20 IN1k frozen-probe eval :: task $SLURM_ARRAY_TASK_ID/16 :: $RUN ==="
log "Job: ${SLURM_ARRAY_JOB_ID:-none}_${SLURM_ARRAY_TASK_ID:-none}, Host: $(hostname)"
log "  ckpt:         $CKPT_HF"
log "  IMAGENET_VAL: $IMAGENET_VAL"
log "  output:       $OUT"
log "  policy:       coarse_to_fine, n_timesteps=10"

exec "$VENV/bin/python" -m canvit_eval in1k-clf \
    --mode frozen \
    --episode.model-repo "$CKPT_HF" \
    --episode.n-timesteps 10 \
    --episode.policy coarse_to_fine \
    --output "$OUT" \
    --batch-size 64 \
    --num-workers 8
