#!/bin/bash
#SBATCH -p grete:shared
#SBATCH --nodes=1
#SBATCH --gpus-per-node=A100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -t 0-01:00:00
#SBATCH -A nib00021
#SBATCH -C inet                  # required for HuggingFace Hub access (probe download)
#SBATCH --mem=64G
#SBATCH -o logs/eval_inet1k-val/job-%j.log
#SBATCH -e logs/eval_inet1k-val/job-%j.log
##SBATCH --mail-type=BEGIN,END,FAIL
##SBATCH --mail-user=jonathan.henrich@uni-goettingen.de

# IN1k frozen-probe eval of exp17 step-462848.
# Fuses the local CanViT-pretrain checkpoint (already converted to HF format)
# with the default DINOv3 IN1k linear probe and reports per-timestep top-1/top-5.
#
# Submit:
#   sbatch slurm_nhr/eval_inet1k-val/in1k_clf.sbatch

cd /user/henrich1/u25995/jonathan/repos/CanViT-eval
source /user/henrich1/u25995/.bashrc

set -eu

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "=== exp17 IN1k frozen-probe eval (1× A100, grete:shared) ==="
log "Job: ${SLURM_JOB_ID:-none}, Host: $(hostname)"
log "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"

source slurm_nhr/env.sh

VENV=".venv"
if [ ! -d "$VENV" ]; then
    log "ERROR: $VENV not found. Run: uv sync"
    exit 1
fi

CKPT_HF="/user/henrich1/u25995/jonathan/repos/CanViT-pretrain/checkpoints/exp17-1gpu/step-462848-hf"
OUT="results/exp17_step462848_in1k.pt"

if [ ! -d "$CKPT_HF" ]; then
    log "ERROR: HF-format checkpoint not found: $CKPT_HF"
    log "Convert with: scripts/pretrain_ckpt_to_hf_format.py (in CanViT-specialize)"
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
