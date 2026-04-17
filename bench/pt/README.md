# Bench (PyTorch) — inference-latency measurement

Measures per-forward-pass latency at `batch_size=1` with explicit device
sync before/after each timed iteration. Intended for the paper's
latency claim; NOT a throughput harness.

## Files

- `run.py` — single-config runner. Produces one JSONL per invocation
  under `bench/pt/results/`. Streams raw per-iteration timings + warmup
  rows + a peak-memory row (CUDA only).
- `matrix.py` — matrix driver. Generates one subprocess per cell in
  `(model × device × scene × dtype × threads)`, pre-flight gated, with
  shuffled order per pass across multiple passes. Replaces the old
  `run_matrix.sh` + `threadscan.sh` shell scripts.
- `analyze.py` — distributional stats over the JSONLs: median /
  p5 / p95 / p99 / std, bootstrap 95% CI on the median, pairwise
  CI-disjoint test across adjacent thread counts, per-run Spearman
  time-drift detection.

## Flow

```
matrix.py  --->  N × run.py  --->  bench/pt/results/*.jsonl
                                         |
                                         v
                    [ sync to paper repo: data/latency_bench/ ]
                                         |
                                         v
                  [ paper-repo: export/bench.py ---> hw_bench.json ]
                                         |
                                         v
            [ paper-repo: hw_latency.py + hw_bench_table.typ ]
```

## Typical invocations

```bash
# Preview the full paper matrix (no execution).
uv run python bench/pt/matrix.py --dry-run

# Paper matrix. CUDA requires exclusive GPU per the project rule.
uv run python bench/pt/matrix.py --passes 3

# CPU thread-count scan at scene=512, no GPU needed:
uv run python bench/pt/matrix.py \
    --models canvit --cpu-scenes 512 --cuda-scenes \
    --cpu-threads 1 4 8 16 --passes 3 --skip-gpu-for-cpu-jobs

# Pre-flight gates: GPU util > 5%, any GPU process, CPU load > 2.0,
# or total CPU% > 30% aborts before any measurement.
# Override via IdleThresholds in matrix.py, or pass --skip-preflight.

# Analyze the resulting JSONLs:
uv run python bench/pt/analyze.py --pattern 'bench/pt/results/*.jsonl'
```

## Raw-data location

Results land in `bench/pt/results/` (in-repo, gitignored). On crockett
this is `/home/yberreby/projects/canvit-eval/bench/pt/results/`.

Old 2026-03-14 data that produced the currently-tracked `hw_bench.json`
is preserved as a tarball in `bench/pt/archive/canvit-bench-raw-20260314.tgz`.
The 2.8 GB `~/projects/canvit-bench/` sibling clone that previously
held this data was retired 2026-04-17.

## Load-bearing constraints

- **CPU fp32 only.** bf16 is usually slower on CPU (Ryzen 7950X has no
  native bf16 instructions). CUDA gets fp32 + bf16.
- **Exclusive GPU during CUDA bench.** Contention invalidates
  latency data irrecoverably. Pre-flight enforces this.
- **`max_iters=500` default** in `run.py` (was 100, silent footgun —
  fixed 2026-04-17).

## Known gap

`run.py`'s compile path is asymmetric (CanViT uses `model.compile()`;
DINOv3 uses `torch.compile(teacher.model)` with attribute replacement)
and neither passes `fullgraph=True`, despite the paper appendix claiming
fullgraph. Pending unified compile API + fullgraph audit (will the
DINOv3 HF path actually fullgraph-compile?).

## Compile cache

`matrix.py` sets `TORCHINDUCTOR_CACHE_DIR=~/.cache/torch/inductor`
on every spawned subprocess. The default is `/tmp/torchinductor_$USER`,
which DOES persist across processes (per the
[torch.compile caching docs](https://docs.pytorch.org/tutorials/recipes/torch_compile_caching_configuration_tutorial.html)) —
but `/tmp` is subject to systemd-tmpfiles cleanup by file age on Linux,
so long-gap re-runs can still miss cache and pay the full ~15 s cold
compile. Moving to `~/.cache` gives user-scoped, non-cleaned storage.

Verified empirically on crockett 2026-04-17: pass 0 cold compile 14.5 s,
pass 1 same-shape 2.1 s (~7× speedup).
