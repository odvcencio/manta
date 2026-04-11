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

Capture CPU or heap profiles for any `barr` command with:

```bash
BARR_CPU_PROFILE=/tmp/barr.cpu.pprof go run ./cmd/barr train-embed ...
BARR_MEM_PROFILE=/tmp/barr.mem.pprof go run ./cmd/barr train-embed ...
```

Then inspect with `go tool pprof -top /tmp/barr.cpu.pprof`.

## Current Default Model Smoke

The current reference smoke uses:

- model package: `barracuda-embed-v0`
- encoder repeats: `2`
- tokenizer vocab: `2454`
- max sequence length: `256`
- train set: `4096` contrastive examples
- eval set: `512` contrastive examples
- batch size: `1024`
- loss: InfoNCE
- temperature: `0.05`
- learning rate: `0.005`
- eval cadence: every `4` steps

Latest local CUDA result:

```text
throughput: elapsed=5.139s examples/s=896.74 pairs/s=867250.60 train_examples/s=845.15 train_pairs/s=865437.87 eval_examples/s=1752.73 eval_pairs/s=897395.54 optimizer_steps/s=0.83
accelerators: forward=cuda optimizer=cuda activation=host contrastive=cuda
profile delta: matmul_bind_calls=30 matmul_runs=13644 matmul_run_upload_mb=3727.56 matmul_run_download_mb=2000.00 optimizer_updates=28 activation_calls=0 contrastive_calls=4
```

This is the promoted default benchmark path. It includes CUDA matmul scratch-buffer reuse, grouped batched backward, exact-length grouped contrastive forward for variable-length text, strided-batched cuBLAS for grouped attention matmuls, rank-3 transpose support for grouped attention backward, batch-1024 contrastive training, sequence matmul bindings disabled by default, Q/K/V forward projection coalescing through a multi-bound-right CUDA path, Q/K/V attention-gradient accumulation through one concatenated shared-left GEMM, combined V/K attention backward gradients in one doubled-batch GEMM, Q/K/V input-gradient accumulation into one resident-right output download with one CUDA sync per accumulated group, and grouped activation-backward helpers kept behind the activation accelerator flag. The larger batch keeps the full in-batch negative set intact, improves contrastive signal, cuts optimizer/contrastive calls on this smoke, and reduces per-pair orchestration overhead. Disabling per-sequence matmul bindings trades a small upload increase for a large reduction in backend binding churn. Q/K/V coalescing preserves per-weight residency and quantization while uploading each shared left-hand activation once across query/key/value projections. Concatenated shared-left attention-gradient coalescing computes `input^T*[dQ|dK|dV]` as one standard GEMM, then splits the result back into the Q/K/V weight gradients. V/K gradient coalescing computes `scores^T*dMixed` and `dPreSoftmax^T*Q` in a single strided-batched dispatch because both use the same transpose shape. Input-gradient coalescing computes `dQ*Wq^T + dK*Wk^T + dV*Wv^T` against resident Q/K/V weights, synchronizes once after the accumulated cuBLAS calls, and downloads the accumulated result once. The CLI/runtime default remains conservative until Barracuda has an adaptive CPU/GPU batch policy.

Read the throughput line with both lenses:

- `train_examples/s` measures actual encoder training throughput.
- `train_pairs/s` measures in-batch contrastive work and grows roughly with `batch^2`.
- `optimizer_steps/s` protects the benchmark from hiding that very large batches reduce update count on small datasets.

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
| Batch-512 benchmark smoke | `177192.16` | `28848` |
| Batch-1024 default benchmark smoke | `407407.98` | `16464` |
| Disable sequence matmul bindings by default | `707265.87` | `16464` |
| Return grouped bound-right outputs as views | `742477.76` | `16464` |
| Concatenate shared-left Q/K/V gradient matmul | `762372.72` | `15336` |
| Combine attention V/K gradient matmuls | `803746.93` | `14772` |
| Accumulate Q/K/V input gradients with resident weights | `825766.14` | `13644` |
| Single-sync accumulated Q/K/V input gradients | `865437.87` | `13644` |

The main wins came from grouping real text batches by sequence length during backward, coalescing parameter-gradient matmuls into taller `X^T*dY` operations, grouping contrastive forward sequences by exact token length inside each original batch, promoting rank-3 x rank-3 CUDA matmul to `cublasSgemmStridedBatched`, allowing strided-batched matmul to handle transpose flags directly, and increasing the effective contrastive batch. The forward grouping keeps the full in-batch negative set intact and avoids padding, so attention math does not change.

The Q/K/V multi-bound-right path is transfer progress: it reduces matmul run uploads from `4173.15 MB` to `3727.56 MB` while preserving each weight's resident quantized form. The concatenated shared-left gradient path and combined V/K gradient path are dispatch-count wins on top of that. Accumulated input gradients keep the Q/K/V weights resident and reduce matmul run downloads from `2208.41 MB` to `2000.00 MB`. Single-sync accumulation removes the redundant per-term CUDA syncs inside that primitive while keeping the same transfer profile. Batch-1024 A/B measured `845.15 train_examples/s`, `865437.87 train_pairs/s`, and `13644` matmul runs by default versus `748.52 train_examples/s`, `766485.96 train_pairs/s`, and `13644` matmul runs with `BARR_CUDA_DISABLE_ACCUMULATED_MATMUL_SINGLE_SYNC=1`.

Ranked BPE tokenization removed a startup/data-ingest bottleneck before longer training runs. A direct batch-2048 `train-embed` CPU profile moved tokenizer encode time from `2.13s` cumulative (`BPETokenizer.Encode` -> `bpeMerge` -> `applyMerge`) to `0.25s` cumulative (`bpeMergeRanked`). End-to-end throughput remains dominated by training transfer/orchestration after tokenization, but corpus ingestion no longer burns a large fraction of host CPU.

Trainer-owned scratch buffers now reuse transient float32 flattening inputs for batched matmul dispatches. On the same direct batch-2048 CPU profile, `flattenFixedFloat32Matrices` moved from roughly `0.72s` cumulative to `0.09s` cumulative. The remaining host-side allocation/copy profile is mostly broader activation/state materialization and unavoidable host-device transfer until full device residency lands.

## Batch Sweep

Batch size is now the largest exposed training knob. On the same 4096-example mini smoke:

| Batch | Run steps | Train examples/s | Train pairs/s | Matmul runs | Max RSS |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `512` | `8` | `558.19` | `285791.12` | `28848` | `1.02 GB` |
| `1024` | `4` | `845.15` | `865437.87` | `13644` | `1.52 GB` |
| `2048` | `2` | `827.20` | `1694107.56` | `9408` | `2.50 GB` |
| `4096` | `1` | `741.49` | `3037152.08` | `5616` | `4.47 GB` |

Batch 2048 is a useful ceiling or large-corpus setting, but it halves optimizer steps on this mini smoke. Batch 4096 is one update and regresses example throughput despite very high pair throughput. Batch 1024 is the benchmark default because it keeps multiple updates in the smoke and captures most of the real throughput win.

## How Much Faster Can It Get?

The current profile still shows the next bottleneck is backend transfer/orchestration, not raw math:

```text
MatMulRuns ~= 13644 per 4096-example mini smoke at batch 1024
RunUploadedBytes ~= 3.73 GiB
RunDownloadedBytes ~= 2.00 GiB
```

Reasonable next targets:

- `850-1000 train_examples/s`: reduce host materialization and launch overhead inside same-length groups.
- `1000+ train_examples/s`: keep forward/backward intermediates device-resident across full layer groups and move optimizer/activation state through the same device-resident path.
- `1200+ train_examples/s`: persistent device-resident training steps with fused attention/FFN backward and fewer per-layer dispatches.

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
BARR_TRAIN_ENABLE_SEQUENCE_MATMUL_BINDINGS=1
```

Re-enables per-sequence matmul bindings. These are disabled by default because batch-1024 grouped training spends more time binding and unbinding short-lived sequence tensors than it saves in uploads. Keep this for small-batch experiments and regression checks.

```bash
BARR_TRAIN_DISABLE_QKV_MULTI_BOUND=1
```

Disables Q/K/V forward projection coalescing. By default CUDA uses one uploaded left-hand activation with three resident right-hand Q/K/V matrices for same-length forward groups. This cuts transfer bytes while preserving each right-hand weight's own quantization state.

```bash
BARR_TRAIN_DISABLE_SHARED_LEFT_MATMUL=1
```

Disables shared-left matmul coalescing. By default Barracuda first tries the concatenated shared-left gradient path for attention backward Q/K/V weight-gradient matmuls, then falls back to the backend shared-left interface when concatenation is disabled or unavailable.

```bash
BARR_TRAIN_DISABLE_CONCAT_SHARED_LEFT_MATMUL=1
```

Disables only the concatenated shared-left path. This keeps the older backend shared-left fallback enabled for A/B testing.

```bash
BARR_TRAIN_DISABLE_COMBINED_ATTENTION_VK_GRAD=1
```

Disables only the combined V/K attention backward gradient path. This returns to separate strided-batched GEMMs for `scores^T*dMixed` and `dPreSoftmax^T*Q`.

```bash
BARR_TRAIN_DISABLE_ACCUMULATED_ATTENTION_INPUT_GRAD=1
```

Disables only the accumulated Q/K/V attention input-gradient path. This returns to three resident right-hand matmuls for `dQ*Wq^T`, `dK*Wk^T`, and `dV*Wv^T`, followed by host accumulation.

```bash
BARR_CUDA_DISABLE_ACCUMULATED_MATMUL_SINGLE_SYNC=1
```

Disables CUDA single-sync accumulation inside the backend accumulated resident-right matmul primitive. This keeps the same logical training path but synchronizes after each accumulated cuBLAS term, which is useful for A/B testing backend launch overhead.

```bash
BARR_TRAIN_ENABLE_ACTIVATION_ACCEL=1
```

Enables CUDA/Metal activation backward acceleration. Activation backward now batches GELU, softmax, and layernorm across same-length groups before dispatching. On the batch-1024 mini smoke this reduces activation launches to `2568`, but it is still opt-in because standalone activation kernels are upload/download-bound without broader activation residency. A local CUDA opt-in run measured:

```text
throughput: elapsed=8.108s examples/s=568.33 pairs/s=549633.49 train_examples/s=522.53 train_pairs/s=535070.28 eval_examples/s=1901.79 eval_pairs/s=973718.32 optimizer_steps/s=0.51
accelerators: forward=cuda optimizer=cuda activation=cuda contrastive=cuda
profile delta: matmul_bind_calls=30 matmul_runs=16464 matmul_run_upload_mb=4173.15 matmul_run_download_mb=2208.41 optimizer_updates=28 activation_calls=2568 contrastive_calls=4
```

```bash
BARR_TRAIN_ENABLE_SOFTMAX_BACKWARD_ACCEL=1
```

Enables only the attention softmax backward activation path while keeping GELU and layernorm backward on the host. This is a profiling seam, not a promoted default. On the same batch-1024 mini smoke it reduced activation dispatches to `642`, but still regressed because each call uploads grad/prob tensors and downloads the result without broader activation residency:

```text
throughput: elapsed=6.244s examples/s=737.99 pairs/s=713722.96 train_examples/s=689.45 train_pairs/s=706001.30 eval_examples/s=1689.85 eval_pairs/s=865203.52 optimizer_steps/s=0.67
accelerators: forward=cuda optimizer=cuda activation=cuda contrastive=cuda
profile delta: matmul_bind_calls=30 matmul_runs=13644 matmul_run_upload_mb=3727.56 matmul_run_download_mb=2000.00 optimizer_updates=28 activation_calls=642 contrastive_calls=4
```

```bash
BARR_TRAIN_DISABLE_SOFTMAX_BACKWARD_ACCEL=1
```

Disables the softmax backward activation path even when the broader activation accelerator is enabled. Use it to isolate GELU/layernorm activation experiments from attention softmax experiments.
