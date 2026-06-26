#!/bin/bash
#SBATCH -p grete:shared
#SBATCH --nodes=1
#SBATCH --gpus-per-node=A100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -t 0-02:00:00
#SBATCH -A nib00021
#SBATCH -C inet                  # HuggingFace Hub access (teacher + checkpoint deps)
#SBATCH --mem=64G
#SBATCH --array=0-31%4
#SBATCH -o logs/jon_exp21_modulation/reconstruction/job-%A_%a.log
#SBATCH -e logs/jon_exp21_modulation/reconstruction/job-%A_%a.log

# Teacher-reconstruction eval (cosine vs the run's DINOv3 teacher, auto-selected
# ViT-B/ViT-L from the checkpoint's teacher_name) of the 32 exp21
# (run x override_scale) configs on the nested ImageNet-1k val (rglob, no flat
# copy). One task per config (see configs.sh). Each task:
#   - loads <run>/checkpoints/latest-hf  (HF checkpoint @ step 200,704)
#   - runs coarse_to_fine, 21 timesteps (3 quadtree levels), canvas_grid=32
#   - applies --episode.override-scale (None = policy scales; else fixed zoom)
#   - streams teacher features per-batch; writes per-timestep cosine to
#     results/jon_exp21_modulation/<run>__<tag>__recon.pt
#
# Submit:  sbatch slurm_nhr/jon_exp21_modulation/reconstruction.sbatch

cd /user/henrich1/u25995/jonathan/repos/CanViT-eval
source /user/henrich1/u25995/.bashrc
set -eu
log() { echo "[$(date '+%H:%M:%S')] $*"; }

source slurm_nhr/env.sh
D=slurm_nhr/jon_exp21_modulation
source "$D/configs.sh"

VENV=".venv"
[ -d "$VENV" ] || { log "ERROR: $VENV not found (run: uv sync)"; exit 1; }
if [ -z "${IMAGENET_VAL:-}" ] || [ ! -d "$IMAGENET_VAL" ]; then
    log "ERROR: IMAGENET_VAL unset/missing: ${IMAGENET_VAL:-<unset>}"; exit 1
fi

parse_eval_config "$SLURM_ARRAY_TASK_ID"   # -> RUN, OVR_CLI, TAG
CKPT_HF="/user/henrich1/u25995/jonathan/repos/CanViT-pretrain/logs/jon_exp21_modulation/$RUN/checkpoints/latest-hf"
OUT="results/jon_exp21_modulation/${RUN}__${TAG}__recon.pt"
[ -d "$CKPT_HF" ] || { log "ERROR: HF checkpoint missing: $CKPT_HF (run _eval_convert_checkpoints.sh)"; exit 1; }
mkdir -p "$(dirname "$OUT")"

log "=== exp21 reconstruction :: task $SLURM_ARRAY_TASK_ID :: $RUN  override_scale=$OVR_CLI ==="
log "  ckpt:         $CKPT_HF"
log "  IMAGENET_VAL: $IMAGENET_VAL"
log "  out:          $OUT"

exec "$VENV/bin/python" -m canvit_eval reconstruction \
    --model-repo "$CKPT_HF" \
    --episode.policy coarse_to_fine \
    --episode.n-timesteps 21 \
    --episode.canvas-grid 32 \
    --episode.override-scale "$OVR_CLI" \
    --image-dir "$IMAGENET_VAL" \
    --output "$OUT" \
    --batch-size 64 \
    --num-workers 8
