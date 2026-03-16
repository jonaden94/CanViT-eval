#!/bin/bash
# Shared SLURM prologue for all CanViT eval jobs.
# Source this from sbatch scripts: source slurm/prologue.sh

set -eu  # NOT -x: would trace secret exports (HF_TOKEN, COMET_API_KEY) into logs

log() { echo "[$(date '+%H:%M:%S')] $*"; }
log "Job: ${SLURM_JOB_ID:-local} on $(hostname)"
log "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"

# Source env if it exists (HF tokens, ADE20K_ROOT, etc.)
if [ -f slurm/env.sh ]; then
    source slurm/env.sh
fi

mkdir -p logs
