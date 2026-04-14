#!/usr/bin/env bash
# Inference benchmark matrix — run on crockett (RTX 4090 + Ryzen 9 7950X).
#
# Runs CPU benchmarks first (no GPU conflict), then waits for any running
# eval batch to finish before starting GPU benchmarks.
#
# Usage (from canvit-eval repo root):
#   nohup bash bench/pt/run_matrix.sh > /tmp/canvit_bench.log 2>&1 &
#
# Override UV project location if not at default ($HOME/projects/canvit-eval):
#   UV_PROJECT=/path/to/canvit-eval bash bench/pt/run_matrix.sh
#
# Results: bench/pt/results/*.jsonl (relative to canvit-eval repo root).

set -euo pipefail

UV_PROJECT="${UV_PROJECT:-$HOME/projects/canvit-eval}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN="$SCRIPT_DIR/run.py"

# ── Configuration ─────────────────────────────────────────────────────
CUDA_BUDGET=60           # seconds per CUDA config (measurement only, excludes warmup)
CPU_BUDGET=30            # seconds per CPU config
MAX_ITERS=500
WARMUP=3                 # warmup iterations (triggers compile, primes caches)
CPU_THREADS=1            # single-threaded for reproducible CPU latency
CUDA_RESOLUTIONS=(128 256 512 1024)
CPU_RESOLUTIONS=(128 256 512)

run() {
    local label="$1"; shift
    echo "=== $(date +%H:%M:%S) $label ==="
    cd "$UV_PROJECT" && uv run python "$RUN" "$@"
    echo ""
}

echo "============================================"
echo "Benchmark matrix — $(date -Iseconds)"
echo "Script: $RUN"
echo "UV project: $UV_PROJECT"
echo "CUDA budget: ${CUDA_BUDGET}s, CPU budget: ${CPU_BUDGET}s"
echo "Max iters: $MAX_ITERS, Warmup: $WARMUP"
echo "============================================"
echo ""

# ── Phase 1: CPU benchmarks (safe to run alongside GPU eval) ──────────
echo ">>> Phase 1: CPU benchmarks (threads=$CPU_THREADS)"
for px in "${CPU_RESOLUTIONS[@]}"; do
    run "CanViT CPU fp32 ${px}px" \
        --model canvit --device cpu --scene-px "$px" --dtype fp32 --batch-size 1 \
        --num-threads "$CPU_THREADS" --time-budget-s "$CPU_BUDGET" --max-iters "$MAX_ITERS" --warmup-iters "$WARMUP"

    run "DINOv3-ViTB CPU fp32 ${px}px" \
        --model dinov3-vitb16 --device cpu --scene-px "$px" --dtype fp32 --batch-size 1 \
        --num-threads "$CPU_THREADS" --time-budget-s "$CPU_BUDGET" --max-iters "$MAX_ITERS" --warmup-iters "$WARMUP"
done

# ── Phase 2: Wait for GPU to be free ──────────────────────────────────
echo ">>> Waiting for GPU to be free..."
while nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q .; do
    echo "  GPU in use, waiting 30s... ($(date +%H:%M:%S))"
    sleep 30
done
echo "  GPU is free. Starting CUDA benchmarks."

# ── Phase 3: CUDA benchmarks (exclusive GPU access) ──────────────────
echo ">>> Phase 3: CUDA benchmarks (compiled, amp-bf16)"
for px in "${CUDA_RESOLUTIONS[@]}"; do
    run "CanViT CUDA compiled bf16 ${px}px" \
        --model canvit --device cuda --scene-px "$px" --compiled --dtype amp-bf16 --batch-size 1 \
        --time-budget-s "$CUDA_BUDGET" --max-iters "$MAX_ITERS" --warmup-iters "$WARMUP"

    run "DINOv3-ViTB CUDA compiled bf16 ${px}px" \
        --model dinov3-vitb16 --device cuda --scene-px "$px" --compiled --dtype amp-bf16 --batch-size 1 \
        --time-budget-s "$CUDA_BUDGET" --max-iters "$MAX_ITERS" --warmup-iters "$WARMUP"
done

# DINOv3 at 2048px — will it fit in 24GB?
run "DINOv3-ViTB CUDA compiled bf16 2048px" \
    --model dinov3-vitb16 --device cuda --scene-px 2048 --compiled --dtype amp-bf16 --batch-size 1 \
    --time-budget-s "$CUDA_BUDGET" --max-iters "$MAX_ITERS" --warmup-iters "$WARMUP" || true

echo "============================================"
echo "All done — $(date -Iseconds)"
echo "Results in: $SCRIPT_DIR/results/"
echo "============================================"
