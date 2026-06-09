#!/bin/bash
#SBATCH -p grete:shared
#SBATCH --nodes=1
#SBATCH --gpus-per-node=A100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -t 0-02:00:00
#SBATCH -A nib00021
#SBATCH -C inet                  # required for HuggingFace Hub access (probe download)
#SBATCH --mem=64G
#SBATCH --exclude=ggpu129,ggpu202

# Single-run IN1k frozen-probe eval. Submit one per run via:
#   sbatch --export=ALL,RUN=<run_name> \
#          --output=logs/jon_exp19_canvit_fovi_add_ons/in1k_clf/job-%j_<run_name>.log \
#          slurm_nhr/jon_exp19_canvit_fovi_add_ons/in1k_clf_single.sbatch

cd /user/henrich1/u25995/jonathan/repos/CanViT-eval
source /user/henrich1/u25995/.bashrc

set -eu
log() { echo "[$(date '+%H:%M:%S')] $*"; }
source slurm_nhr/env.sh

VENV=".venv"
: "${RUN:?RUN must be exported (e.g. --export=ALL,RUN=exp19-per-ring)}"
[ -d "$VENV" ] || { log "ERROR: $VENV not found"; exit 1; }
[ -n "${IMAGENET_VAL:-}" ] && [ -d "$IMAGENET_VAL" ] || { log "ERROR: IMAGENET_VAL unset/missing"; exit 1; }

CKPT_HF="/user/henrich1/u25995/jonathan/repos/CanViT-pretrain/logs/foveated_add_ons/$RUN/checkpoints/latest-hf"
OUT="results/jon_exp19_canvit_fovi_add_ons/${RUN}_in1k.pt"
[ -d "$CKPT_HF" ] || { log "ERROR: HF checkpoint not found: $CKPT_HF"; exit 1; }

log "=== exp19 IN1k frozen-probe :: $RUN ==="
log "Job: ${SLURM_JOB_ID:-none}, Host: $(hostname)"
log "  ckpt:   $CKPT_HF"
log "  output: $OUT"

exec "$VENV/bin/python" -m canvit_eval in1k-clf \
    --mode frozen \
    --episode.model-repo "$CKPT_HF" \
    --episode.n-timesteps 10 \
    --episode.policy coarse_to_fine \
    --output "$OUT" \
    --batch-size 64 \
    --num-workers 8
