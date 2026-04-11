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
- batch size: `512`
- loss: InfoNCE
- temperature: `0.05`
- learning rate: `0.005`
- eval cadence: every `4` steps

Latest local CUDA result:

```text
throughput: elapsed=12.414s pairs/s=190053.33 train_pairs/s=173560.62 eval_pairs/s=792624.86
accelerators: forward=cuda optimizer=cuda activation=host contrastive=cuda
profile delta: matmul_bind_calls=131126 matmul_runs=28848 matmul_run_upload_mb=4140.19 matmul_run_download_mb=2272.91 optimizer_updates=56 activation_calls=0 contrastive_calls=8
```

This is the promoted default benchmark path. It includes CUDA matmul scratch-buffer reuse, grouped batched backward, exact-length grouped contrastive forward for variable-length text, strided-batched cuBLAS for grouped attention matmuls, rank-3 transpose support for grouped attention backward, and batch-512 contrastive training. The larger batch keeps the full in-batch negative set intact, improves contrastive signal, halves optimizer/contrastive calls on this smoke, and reduces per-pair orchestration overhead. The CLI/runtime default remains conservative until Barracuda has an adaptive CPU/GPU batch policy.

## Recent Perf Delta

The training hot path moved as follows on the same mini smoke:

| Path | Train pairs/s | Matmul runs |
| --- | ---: | ---: |
| Instrumented baseline | `21975.83` | `409600` |
| CUDA scratch reuse | `22933.16` | `409600` |
| Grouped batched backward default | `33328.94` | `237568` |
| Exact-length grouped forward default | `35856.58` | `140056` |
| Strided-batched grouped attention | `42594.29` | `107552` |
| Rank-3 transpose batched attention backward | `82262.50` | `50208` |
| Batch-512 default benchmark smoke | `173560.62` | `28848` |

The main wins came from grouping real text batches by sequence length during backward, coalescing parameter-gradient matmuls into taller `X^T*dY` operations, grouping contrastive forward sequences by exact token length inside each original batch, promoting rank-3 x rank-3 CUDA matmul to `cublasSgemmStridedBatched`, allowing strided-batched matmul to handle transpose flags directly, and increasing the effective contrastive batch. The forward grouping keeps the full in-batch negative set intact and avoids padding, so attention math does not change.

## How Much Faster Can It Get?

The current profile still shows the next bottleneck is backend transfer/orchestration, not raw math:

```text
MatMulRuns ~= 28848 per 4096-example mini smoke at batch 512
RunUploadedBytes ~= 4.14 GiB
RunDownloadedBytes ~= 2.27 GiB
```

Reasonable next targets:

- `180k-240k train pairs/s`: keep the batch-512 path stable while reducing host materialization inside same-length groups.
- `240k-320k train pairs/s`: keep forward/backward intermediates device-resident across full layer groups and move optimizer/activation state through the same device-resident path.
- `320k+ train pairs/s`: persistent device-resident training steps with fused attention/FFN backward and fewer per-layer dispatches.

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
