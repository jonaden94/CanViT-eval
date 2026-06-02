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
#SBATCH --array=0-19%1
#SBATCH -o logs/foveated_add_ons/reconstruction/job-%A_%a.log
#SBATCH -e logs/foveated_add_ons/reconstruction/job-%A_%a.log

# Teacher-reconstruction (cosine-vs-teacher) eval of the 20 stable exp19 runs.
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
#   sbatch slurm_nhr/foveated_add_ons/reconstruction.sbatch

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
    exp19-backbone-registers-19
    exp19-baseline-res64-doubleres
    exp19-cond-bias
    exp19-coordconv
    exp19-coordconv-mlp768
    exp19-film-learned512
    exp19-film-learned512-mlp768
    exp19-film-pos-sigma1
    exp19-film-pos-sigma1-mlp768
    exp19-film-pos-sigma16
    exp19-film-pos-sigma4
    exp19-film-pos-sigma4-mlp768
    exp19-fixsize722
    exp19-mlp768
    exp19-mlp768x2
    exp19-pad-learned-shared
    exp19-per-ring
    exp19-res36-cart6-doubleres
    exp19-res49-cart6-visual-noforcematch
    exp19-visual-noforcematch
)

RUN="${RUNS[$SLURM_ARRAY_TASK_ID]}"
CKPT_HF="/user/henrich1/u25995/jonathan/repos/CanViT-pretrain/logs/foveated_add_ons/$RUN/checkpoints/latest-hf"
OUT="results/foveated_add_ons/${RUN}_recon.pt"
CACHE="results/foveated_add_ons/_cache/dinov3b16_in1k_val_512.pt"

if [ ! -d "$CKPT_HF" ]; then
    log "ERROR: HF-format checkpoint not found: $CKPT_HF"
    exit 1
fi

log "=== exp19 reconstruction eval :: task $SLURM_ARRAY_TASK_ID/19 :: $RUN ==="
log "Job: ${SLURM_ARRAY_JOB_ID:-none}_${SLURM_ARRAY_TASK_ID:-none}, Host: $(hostname)"
log "  ckpt:          $CKPT_HF"
log "  image_dir:     $FLAT_VAL"
log "  teacher_cache: $CACHE"
log "  output:        $OUT"
log "  policy:        coarse_to_fine, n_timesteps=10"

exec "$VENV/bin/python" -m canvit_eval reconstruction \
    --episode.model-repo "$CKPT_HF" \
    --episode.n-timesteps 10 \
    --episode.policy coarse_to_fine \
    --image-dir "$FLAT_VAL" \
    --teacher-cache "$CACHE" \
    --output "$OUT" \
    --batch-size 64 \
    --num-workers 8
