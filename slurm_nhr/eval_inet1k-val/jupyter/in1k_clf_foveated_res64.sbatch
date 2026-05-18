#!/bin/bash
#SBATCH -p jupyter
#SBATCH -G V100:1
#SBATCH -t 0-02:00:00
#SBATCH -A nib00021
#SBATCH -C inet                  # required for HuggingFace Hub access (probe download)
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --delay-boot=1
#SBATCH -o logs/eval_inet1k-val/job-%j.log
#SBATCH -e logs/eval_inet1k-val/job-%j.log
#SBATCH --job-name=in1k-foveated-res64-v100
##SBATCH --mail-type=BEGIN,END,FAIL
##SBATCH --mail-user=jonathan.henrich@uni-goettingen.de

# IN1k frozen-probe eval of the foveated-fullimg-res64 exp17 checkpoint on V100
# (jupyter partition, sm_70 — requires cu126 torch via .venv-v100).
# Same as in1k_clf_foveated_fullimg.sbatch but with the res64 variant
# (foveated_patcher: resolution=64, cart_patch_size=8).
#
# Requires .venv-v100:
#   bash _setup_v100/setup.sh
#
# Submit:
#   sbatch slurm_nhr/eval_inet1k-val/jupyter/in1k_clf_foveated_res64.sbatch

cd /user/henrich1/u25995/jonathan/repos/CanViT-eval
source /user/henrich1/u25995/.bashrc

set -eu

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "=== exp17 foveated-fullimg-res64 IN1k frozen-probe eval (1× V100, jupyter) ==="
log "Job: ${SLURM_JOB_ID:-none}, Host: $(hostname)"
log "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"

source slurm_nhr/env.sh

VENV=".venv-v100"
if [ ! -d "$VENV" ]; then
    log "ERROR: $VENV not found. Run: bash _setup_v100/setup.sh"
    exit 1
fi

CKPT_HF="/user/henrich1/u25995/jonathan/repos/CanViT-pretrain/logs/foveated/exp17-1gpu-foveated-fullimg-res64/checkpoints/latest-hf"
OUT="results/exp17_foveated_fullimg_res64_in1k.pt"

if [ ! -d "$CKPT_HF" ]; then
    log "ERROR: HF-format checkpoint not found: $CKPT_HF"
    exit 1
fi
if [ -z "${IMAGENET_VAL:-}" ] || [ ! -d "$IMAGENET_VAL" ]; then
    log "ERROR: IMAGENET_VAL is unset or missing: ${IMAGENET_VAL:-<unset>}"
    exit 1
fi

mkdir -p "$(dirname "$OUT")"

log "  ckpt:         $CKPT_HF"
log "  IMAGENET_VAL: $IMAGENET_VAL"
log "  output:       $OUT"

exec "$VENV/bin/python" -m canvit_eval in1k-clf \
    --mode frozen \
    --episode.model-repo "$CKPT_HF" \
    --output "$OUT" \
    --batch-size 64 \
    --num-workers 8 \
    "$@"
