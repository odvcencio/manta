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
throughput: elapsed=33.426s pairs/s=39212.02 train_pairs/s=33328.94 eval_pairs/s=133403.79
accelerators: forward=cuda optimizer=cuda activation=host contrastive=cuda
profile delta: matmul_bind_calls=131174 matmul_runs=237568 matmul_run_upload_mb=3856.50 matmul_run_download_mb=2384.16 optimizer_updates=112 activation_calls=0 contrastive_calls=16
```

This is the promoted default path. It includes CUDA matmul scratch-buffer reuse and grouped batched backward.

## Recent Perf Delta

The training hot path moved as follows on the same mini smoke:

| Path | Train pairs/s | Matmul runs |
| --- | ---: | ---: |
| Instrumented baseline | `21975.83` | `409600` |
| CUDA scratch reuse | `22933.16` | `409600` |
| Grouped batched backward default | `33328.94` | `237568` |

The main win came from grouping real text batches by sequence length during backward and coalescing parameter-gradient matmuls into taller `X^T*dY` operations. That cut backend matmul dispatch pressure by about `42%` and moved train throughput by about `52%` over the instrumented baseline.

## How Much Faster Can It Get?

The current profile still shows most wall time in backend matmul dispatch:

```text
MatMul RunNanos ~= 28.2s of 33.4s trainer elapsed
RunUploadedBytes ~= 3.86 GiB
RunDownloadedBytes ~= 2.38 GiB
```

Reasonable next targets:

- `45k-55k train pairs/s`: batch packing by sequence length before training plus more backward coalescing.
- `55k-75k train pairs/s`: keep forward activations and backward intermediates device-resident across forward to backward.
- `75k-120k train pairs/s`: fused attention/FFN backward kernels that stop routing every small tensor product through host-owned orchestration.

The practical ceiling for this tiny model is dominated by orchestration and host-device transfer, not raw GEMM throughput. Larger models will shift more time into actual math, but the same device-residency work is still required to get high GPU utilization.

## Relevant Flags

```bash
BARR_TRAIN_DISABLE_BATCHED_BACKWARD=1
```

Disables the promoted grouped batched-backward path and returns to per-sequence backward.

```bash
BARR_TRAIN_BATCHED_FORWARD=1
```

Enables the experimental batched forward path. It is currently gated because it is effectively tied with the default path on the reference smoke.

```bash
BARR_TRAIN_ENABLE_ACTIVATION_ACCEL=1
```

Enables CUDA/Metal activation backward acceleration. It is currently opt-in because the standalone activation kernels are upload/download-bound without broader activation residency.
