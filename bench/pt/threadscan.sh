#!/usr/bin/env bash
# Thread-count scaling scan for bench/pt/run.py.
#
# Runs one (model, device, scene_px, dtype) config across a list of CPU
# thread counts, shuffling the order across multiple passes. This isolates
# the thread-count effect while averaging any time-local contention across
# thread counts (each thread count is visited once per pass, at different
# wall times).
#
# The script refuses to start if the GPU or CPU is busy. Rationale:
#   - GPU busy → another CUDA job is on the box; bench is CPU but the
#     exclusive-GPU rule extends to any bench discipline.
#   - CPU load too high → measurements will reflect contention.
#
# Usage (from canvit-eval repo root, crockett or any Linux host):
#
#   bash bench/pt/threadscan.sh                      # defaults below
#   THREADS="4 8 16 32" PASSES=3 bash bench/pt/threadscan.sh
#   SCENE_PX=1024 MODEL=dinov3-vitb16 bash bench/pt/threadscan.sh
#
# Results: bench/pt/results/bench_<model>_e_fp32_<scene>px_[cgN_]cpu_tN_<ts>.jsonl
# Log:     /tmp/threadscan_<ts>.log

set -euo pipefail

# ── Defaults (overridable via env) ───────────────────────────────────────
MODEL="${MODEL:-canvit}"
SCENE_PX="${SCENE_PX:-512}"
DTYPE="${DTYPE:-fp32}"
THREADS="${THREADS:-4 8 16 32}"
PASSES="${PASSES:-3}"
TIME_BUDGET_S="${TIME_BUDGET_S:-60}"
MAX_ITERS="${MAX_ITERS:-300}"
WARMUP="${WARMUP:-3}"

# Idle thresholds
MAX_GPU_UTIL="${MAX_GPU_UTIL:-5}"       # percent
MAX_LOAD_1MIN="${MAX_LOAD_1MIN:-2.0}"
MAX_TOTAL_CPU="${MAX_TOTAL_CPU:-30.0}"  # sum over processes

# Derive paths from script location — no hardcoded repo paths.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN_PY="$SCRIPT_DIR/run.py"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
UV_PROJECT="${UV_PROJECT:-$REPO_ROOT}"

TS=$(date +%Y%m%d_%H%M%S)
LOG="/tmp/threadscan_${TS}.log"

echo "=== threadscan $TS ===" | tee -a "$LOG"
echo "repo:        $REPO_ROOT" | tee -a "$LOG"
echo "uv project:  $UV_PROJECT" | tee -a "$LOG"
echo "run.py:      $RUN_PY" | tee -a "$LOG"
echo "model:       $MODEL" | tee -a "$LOG"
echo "scene_px:    $SCENE_PX" | tee -a "$LOG"
echo "dtype:       $DTYPE" | tee -a "$LOG"
echo "threads:     $THREADS" | tee -a "$LOG"
echo "passes:      $PASSES" | tee -a "$LOG"
echo "max_iters:   $MAX_ITERS" | tee -a "$LOG"
echo "warmup:      $WARMUP" | tee -a "$LOG"
echo "log:         $LOG" | tee -a "$LOG"

# ── Pre-flight: refuse to start if GPU or CPU is busy ────────────────────
echo "" | tee -a "$LOG"
echo "=== PRE-FLIGHT ===" | tee -a "$LOG"

if command -v nvidia-smi >/dev/null 2>&1; then
    gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
    gpu_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    gpu_procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)
    echo "GPU: util=${gpu_util}% mem=${gpu_mem}MiB procs=${gpu_procs}" | tee -a "$LOG"
    if [[ "$gpu_util" -gt "$MAX_GPU_UTIL" ]] || [[ "$gpu_procs" -gt 0 ]]; then
        echo "ABORT: GPU busy (util>${MAX_GPU_UTIL}% or procs>0)." | tee -a "$LOG"
        exit 2
    fi
else
    echo "GPU: nvidia-smi not found (skipping)" | tee -a "$LOG"
fi

load1=$(uptime | awk -F'load average:' '{print $2}' | awk -F',' '{print $1}' | tr -d ' ')
echo "CPU load (1 min): $load1" | tee -a "$LOG"
if awk "BEGIN {exit !($load1 > $MAX_LOAD_1MIN)}"; then
    echo "ABORT: CPU load average > ${MAX_LOAD_1MIN}." | tee -a "$LOG"
    exit 3
fi

total_cpu=$(ps -eo pcpu --no-headers 2>/dev/null | awk '{s+=$1} END {printf "%.1f", s}')
echo "Total CPU%: ${total_cpu}" | tee -a "$LOG"
if awk "BEGIN {exit !(${total_cpu} > $MAX_TOTAL_CPU)}"; then
    echo "ABORT: total CPU usage > ${MAX_TOTAL_CPU}%." | tee -a "$LOG"
    ps -eo pid,user,pcpu,pmem,comm --sort=-pcpu --no-headers 2>/dev/null | head -10 | tee -a "$LOG"
    exit 4
fi

echo "Idle check passed." | tee -a "$LOG"

# ── Deterministic per-pass shuffle of the thread list ────────────────────
# Python is the single source of truth for the shuffle; we pass a seed per
# pass so the order is reproducible. Writes "pass_idx thread_count" lines.
plan_file=$(mktemp)
python3 -c "
import random, sys
threads = '$THREADS'.split()
passes = int('$PASSES')
for p in range(passes):
    rng = random.Random(p * 1000 + 42)
    order = list(threads)
    rng.shuffle(order)
    for t in order:
        print(p, t)
" > "$plan_file"

echo "" | tee -a "$LOG"
echo "=== PLAN ===" | tee -a "$LOG"
cat "$plan_file" | tee -a "$LOG"

# ── Run ──────────────────────────────────────────────────────────────────
cd "$UV_PROJECT"

while read -r pass_idx t; do
    echo "" | tee -a "$LOG"
    echo "[$(date +%H:%M:%S)] >>> pass=$pass_idx threads=$t" | tee -a "$LOG"
    uv run python "$RUN_PY" \
        --model "$MODEL" --device cpu \
        --scene-px "$SCENE_PX" --dtype "$DTYPE" --batch-size 1 \
        --num-threads "$t" \
        --time-budget-s "$TIME_BUDGET_S" \
        --max-iters "$MAX_ITERS" \
        --warmup-iters "$WARMUP" \
        2>&1 | tee -a "$LOG"
done < "$plan_file"

rm -f "$plan_file"

# ── Post-flight ──────────────────────────────────────────────────────────
echo "" | tee -a "$LOG"
echo "=== POST-FLIGHT $(date -Iseconds) ===" | tee -a "$LOG"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tee -a "$LOG"
fi
uptime | tee -a "$LOG"
echo "Next: uv run python bench/pt/analyze.py --glob 'bench/pt/results/*${TS:0:8}*.jsonl'" | tee -a "$LOG"
echo "DONE." | tee -a "$LOG"
