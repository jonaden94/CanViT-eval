# Nibi cluster environment for CanViT-bench
# Source this: source slurm/env.sh

echo "[env] Setting up environment..."

export PATH=$HOME/.local/bin:$PATH

if [ -n "$SLURM_TMPDIR" ]; then
    export UV_CACHE_DIR="$SLURM_TMPDIR/.uv-cache"
    export UV_PROJECT_ENVIRONMENT="$SLURM_TMPDIR/.venv"
    echo "[env] Using SLURM_TMPDIR for uv cache/venv"
else
    echo "[env] No SLURM_TMPDIR (interactive session)"
fi

export HF_HOME="$SCRATCH/.huggingface"
export TORCH_HOME="$SCRATCH/.torch"

uv sync --no-dev
echo "[env] Done"
