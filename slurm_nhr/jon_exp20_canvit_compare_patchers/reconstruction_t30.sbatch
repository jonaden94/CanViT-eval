#!/bin/bash
#SBATCH -p grete:shared
#SBATCH --nodes=1
#SBATCH --gpus-per-node=A100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -t 0-01:30:00
#SBATCH -A nib00021
#SBATCH -C inet                  # required for HuggingFace Hub access (teacher download)
#SBATCH --mem=64G
#SBATCH --exclude=ggpu129,ggpu202
#SBATCH --array=0-16%1
#SBATCH -o logs/jon_exp20_canvit_compare_patchers/reconstruction_t30/job-%A_%a.log
#SBATCH -e logs/jon_exp20_canvit_compare_patchers/reconstruction_t30/job-%A_%a.log

# Teacher-reconstruction (cosine-vs-teacher) eval of the 17 exp20 runs.
# Same metric as pretraining-time validation, dumped per-timestep.
#
# Uses the FLATTENED ImageNet-1k val (50k images, no class subdirs) so the
# reconstruction dataset's flat-iterdir scan finds all images. Same 50k images
# as the in1k-clf eval ⇒ cosine and accuracy line up image-for-image in the
# analysis notebook.
#
# Teacher features for these 50k images are identical across runs. Task 0
# computes & caches them; the remaining 19 tasks read the cache. Saves ~4 h
# of redundant teacher inference. Cache path is shared across the array.
#
# Submit:
#   sbatch slurm_nhr/jon_exp20_canvit_compare_patchers/reconstruction.sbatch

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

FLAT_VAL=/user/henrich1/u25995/jonathan/datasets/imagenet1k-val-flattened
if [ ! -d "$FLAT_VAL" ]; then
    log "ERROR: flattened val dir not found: $FLAT_VAL"
    exit 1
fi

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
OUT="results/jon_exp20_canvit_compare_patchers/${RUN}_recon_t30.pt"
CACHE="results/jon_exp20_canvit_compare_patchers/_cache/dinov3b16_in1k_val_512.pt"

if [ ! -d "$CKPT_HF" ]; then
    log "ERROR: HF-format checkpoint not found: $CKPT_HF"
    exit 1
fi

log "=== exp20 reconstruction eval :: task $SLURM_ARRAY_TASK_ID/16 :: $RUN ==="
log "Job: ${SLURM_ARRAY_JOB_ID:-none}_${SLURM_ARRAY_TASK_ID:-none}, Host: $(hostname)"
log "  ckpt:          $CKPT_HF"
log "  image_dir:     $FLAT_VAL"
log "  teacher_cache: $CACHE"
log "  output:        $OUT"
log "  policy:        coarse_to_fine, n_timesteps=30"

# All exp20 models were standardized at canvas_grid=32 ONLY (scene_std is
# initialized for c=32 and undefined elsewhere -- same as in1k-clf's
# FUSION_CANVAS_GRID). Pin it explicitly: the 16px-patch runs already derive 32
# from scene_size//patch_size, but uniform8 (8px patches) would derive 64 and
# trip `assert scene_std.initialized`. Grid 32 keeps every run comparable.
exec "$VENV/bin/python" -m canvit_eval reconstruction \
    --model-repo "$CKPT_HF" \
    --episode.n-timesteps 30 \
    --episode.policy coarse_to_fine \
    --episode.canvas-grid 32 \
    --image-dir "$FLAT_VAL" \
    --teacher-cache "$CACHE" \
    --output "$OUT" \
    --batch-size 64 \
    --num-workers 8
