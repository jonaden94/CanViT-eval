#!/bin/bash
#SBATCH -p grete:shared
#SBATCH --nodes=1
#SBATCH --gpus-per-node=A100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -t 0-02:00:00
#SBATCH -A nib00021
#SBATCH -C inet                  # required for HuggingFace Hub access (teacher download)
#SBATCH --mem=64G
#SBATCH --exclude=ggpu129,ggpu202

# Single-run teacher-reconstruction (cosine-vs-teacher) eval. Submit one per
# run via:
#   sbatch --export=ALL,RUN=<run_name> \
#          --output=logs/jon_exp20_canvit_compare_patchers/reconstruction/job-%j_<run_name>.log \
#          slurm_nhr/jon_exp20_canvit_compare_patchers/reconstruction_single.sbatch
#
# NOTE: no --teacher-cache here. Parallel jobs would race on a shared cache
# file (torch.save isn't atomic). Each job computes teacher features locally;
# the extra ~10-15 min per job is acceptable for the speed gain from running
# all 20 in parallel.

cd /user/henrich1/u25995/jonathan/repos/CanViT-eval
source /user/henrich1/u25995/.bashrc

set -eu
log() { echo "[$(date '+%H:%M:%S')] $*"; }
source slurm_nhr/env.sh

VENV=".venv"
: "${RUN:?RUN must be exported (e.g. --export=ALL,RUN=exp20-per-ring)}"
[ -d "$VENV" ] || { log "ERROR: $VENV not found"; exit 1; }

FLAT_VAL=/user/henrich1/u25995/jonathan/datasets/imagenet1k-val-flattened
[ -d "$FLAT_VAL" ] || { log "ERROR: flattened val dir not found: $FLAT_VAL"; exit 1; }

CKPT_HF="/user/henrich1/u25995/jonathan/repos/CanViT-pretrain/logs/patcher_compare/$RUN/checkpoints/latest-hf"
OUT="results/jon_exp20_canvit_compare_patchers/${RUN}_recon.pt"
[ -d "$CKPT_HF" ] || { log "ERROR: HF checkpoint not found: $CKPT_HF"; exit 1; }

log "=== exp20 reconstruction :: $RUN ==="
log "Job: ${SLURM_JOB_ID:-none}, Host: $(hostname)"
log "  ckpt:      $CKPT_HF"
log "  image_dir: $FLAT_VAL"
log "  output:    $OUT"

exec "$VENV/bin/python" -m canvit_eval reconstruction \
    --model-repo "$CKPT_HF" \
    --episode.n-timesteps 10 \
    --episode.policy coarse_to_fine \
    --image-dir "$FLAT_VAL" \
    --output "$OUT" \
    --batch-size 64 \
    --num-workers 8
