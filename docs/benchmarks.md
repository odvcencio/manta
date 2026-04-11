# Benchmarks

Barracuda has two benchmark layers:

- Go microbenchmarks for isolated runtime kernels and trainer math.
- End-to-end training smokes for the native default embedding-model path.

Run the default microbenchmarks with Ferrous Wheel:

```bash
BARR_BENCH_ROOT=$PWD ferrous-wheel run scripts/bench.fw
```

Run CUDA microbenchmarks:

```bash
BARR_BENCH_ROOT=$PWD BARR_BENCH_CUDA=1 ferrous-wheel run scripts/bench.fw
```

Run the default-model training smoke from a local asset package:

```bash
BARR_BENCH_ROOT=$PWD BARR_BENCH_MODEL_ASSETS=/path/to/assets/barracuda-embed-v0 ferrous-wheel run scripts/bench.fw
```

The model smoke copies the package into a temporary directory before training. It does not mutate the source asset directory.

If you want a binary runner instead of `run` mode:

```bash
ferrous-wheel build scripts/bench.fw -o bin/barr-bench
bin/barr-bench
```

## Current Default Model Smoke

The current reference smoke uses:

- model package: `barracuda-embed-v0`
- encoder repeats: `2`
- tokenizer vocab: `2454`
- max sequence length: `256`
- train set: `4096` contrastive examples
- eval set: `512` contrastive examples
- batch size: `256`
- loss: InfoNCE
- temperature: `0.05`
- learning rate: `0.005`
- eval cadence: every `4` steps

Latest local CUDA result:

```text
throughput: elapsed=25.236s pairs/s=51937.87 train_pairs/s=42594.29 eval_pairs/s=423845.92
accelerators: forward=cuda optimizer=cuda activation=host contrastive=cuda
profile delta: matmul_bind_calls=131174 matmul_runs=107552 matmul_run_upload_mb=7930.39 matmul_run_download_mb=5159.03 optimizer_updates=112 activation_calls=0 contrastive_calls=16
```

This is the promoted default path. It includes CUDA matmul scratch-buffer reuse, grouped batched backward, exact-length grouped contrastive forward for variable-length text, and strided-batched cuBLAS for grouped attention matmuls.

## Recent Perf Delta

The training hot path moved as follows on the same mini smoke:

| Path | Train pairs/s | Matmul runs |
| --- | ---: | ---: |
| Instrumented baseline | `21975.83` | `409600` |
| CUDA scratch reuse | `22933.16` | `409600` |
| Grouped batched backward default | `33328.94` | `237568` |
| Exact-length grouped forward default | `35856.58` | `140056` |
| Strided-batched grouped attention | `42594.29` | `107552` |

The main wins came from grouping real text batches by sequence length during backward, coalescing parameter-gradient matmuls into taller `X^T*dY` operations, grouping contrastive forward sequences by exact token length inside each original batch, and promoting rank-3 x rank-3 CUDA matmul to `cublasSgemmStridedBatched`. The forward grouping keeps the full in-batch negative set intact and avoids padding, so attention math does not change.

## How Much Faster Can It Get?

The current profile still shows the next bottleneck is backend transfer/orchestration, not raw math:

```text
MatMulRuns ~= 107552 per 4096-example mini smoke
RunUploadedBytes ~= 7.93 GiB
RunDownloadedBytes ~= 5.16 GiB
```

Reasonable next targets:

- `45k-55k train pairs/s`: reduce forward/backward activation upload/download churn and keep same-length grouped intermediates device-resident.
- `55k-75k train pairs/s`: fuse attention and FFN backward kernels so small tensor products stop round-tripping through host-owned orchestration.
- `75k-120k train pairs/s`: persistent device-resident training steps with fewer per-layer dispatches and larger effective batches.

The practical ceiling for this tiny model is dominated by orchestration and host-device transfer, not raw GEMM throughput. Larger models will shift more time into actual math, but the same device-residency work is still required to get high GPU utilization.

## Relevant Flags

```bash
BARR_TRAIN_DISABLE_BATCHED_BACKWARD=1
```

Disables the promoted grouped batched-backward path and returns to per-sequence backward.

```bash
BARR_TRAIN_DISABLE_BATCHED_FORWARD=1
```

Disables the promoted batched forward path and returns to per-sequence forward encoding. Batched forward is enabled by default because the larger default-model run underfeeds the GPU unless forward work is coalesced aggressively.

```bash
BARR_TRAIN_ENABLE_ACTIVATION_ACCEL=1
```

Enables CUDA/Metal activation backward acceleration. It is currently opt-in because the standalone activation kernels are upload/download-bound without broader activation residency.
