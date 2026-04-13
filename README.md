# Manta

Manta is an inference-first GPU language and runtime stack. It compiles `.manta` source into backend-neutral `.mll` execution plans for GPU-accelerated embedding, reranking, retrieval-time scoring, and decode-time inference. Write the model-facing compute once, then run it through CUDA, Metal, Vulkan, DirectML, or WebGPU with the same artifact and the same entrypoint contract.

- **Inference-first product surface** with `kernel` and `pipeline` abstractions
- **Three-level IR pipeline**: HIR (typed) -> MIR (semantic) -> LIR (scheduled)
- **Portable artifact format** (`.mll`): compile once, deploy anywhere
- **Portable backend surface**: CUDA, Metal, Vulkan, DirectML, and WebGPU variants from the same source
- **First-class KV cache**: `kv_cache` type with `kv_read`/`kv_write` for autoregressive decoding
- **TurboQuant-native direction**: Manta is designed to consume and emit quantized tensors and quantized vectors without repacking through a separate framework
- **Schedule hints**: `tile`, `vector_width`, `subgroup`, memory classes -- backend-neutral, lowered late
- **Hybrid runtime**: backend-native execution where promoted kernels exist, host reference execution where they do not yet
- **Pure Go toolchain**: no Python, no C++ build dependencies

## Product direction

The near-term target is an inference-first product surface for embedding, reranking, retrieval-time scoring, decode, and CorkScrewDB integration. Manta also owns the native training path needed to produce those default artifacts, so model authors do not need to train in Python, export through another format, and deploy through a separate runtime.

That means the language and runtime should bias toward:

- quantized weights and quantized vectors as first-class runtime values
- low-friction query-time embedding and scoring entrypoints
- rerank/select entrypoints that can return ids directly
- inference kernels that can consume TurboQuant-native layouts directly
- portable `.mll` artifacts that CorkScrewDB can load without rewriting host code
- sealed MLL package exports that carry model definition, weights, tokenizer, memory plan, and metadata together

## Install

```bash
go install github.com/odvcencio/manta/cmd/manta@latest
```

## Quick Start

Write a `.manta` source file:

```manta
param token_embedding: f16[V, D] @weight("weights/token_embedding")
param projection: f16[D, E] @weight("weights/projection")

kernel l2_normalize(x: f16[T, E]) -> f16[T, E] {
    return normalize(x)
}

pipeline embed(tokens: i32[T]) -> f16[T, E] {
    let hidden = gather(token_embedding, tokens)
    let projected = @matmul(hidden, projection)
    return l2_normalize(projected)
}
```

Compile and run:

```bash
manta compile embed.manta embed.mll
manta run embed.mll embed
```

Or use the built-in demo:

```bash
manta demo tiny_embed
manta demo tiny_decode
manta demo tiny_score
manta demo tiny_rerank
manta demo tiny_select
```

## Language

### Declarations

**Parameters** bind external weights with shape and dtype:

```manta
param wq: f16[D, D] @weight("weights/wq")
```

**Kernels** define fused compute regions:

```manta
kernel l2_normalize(x: f16[T, E]) -> f16[T, E] {
    return normalize(x)
}
```

**Pipelines** orchestrate steps including intrinsics, kernel calls, and KV cache operations:

```manta
pipeline decode_step(x: f16[T, D], cache: kv_cache) -> f16[T, D] {
    let q = @matmul(x, wq)
    let q2 = rope(q)
    let past = kv_read(cache)
    kv_write(cache, q2)
    return softmax(q2 + past)
}
```

### Types

| Type | Description |
|------|-------------|
| `f16[D1, D2, ...]` | Half-precision tensor with symbolic shape |
| `f32[D1, D2, ...]` | Single-precision tensor |
| `i32[D1, ...]` | Integer tensor |
| `q4[D1, D2, ...]` | 4-bit quantized tensor/value buffer |
| `q8[D1, D2, ...]` | 8-bit quantized tensor/value buffer |
| `kv_cache` | Mutable key-value cache for autoregressive decoding |

Dimensions are symbolic (`T`, `D`, `V`, `E`) and resolved at load time.

Quantized inference dtypes such as `q4[...]` and `q8[...]` are part of the active Manta surface. The current bootstrap runtime can consume them directly for quantized scoring paths, and the next step is richer TurboQuant block-format coverage.

### Operations

**Intrinsics** (prefixed with `@`) dispatch to backend libraries:

| Intrinsic | Signature | Backend dispatch |
|-----------|-----------|-----------------|
| `@matmul(a, b)` | `f16[M,K] x f16[K,N] -> f16[M,N]` | cuBLAS / MPS |

**Built-in functions**:

| Function | Description |
|----------|-------------|
| `gather(table, indices)` | Lookup rows by index |
| `softmax(x)` | Row-wise softmax |
| `normalize(x)` | L2 normalization per row |
| `rmsnorm(x)` | RMS normalization per row |
| `rope(x)` | Rotary position embeddings |
| `dequant(x)` | Dequantization conversion |
| `dot(query, docs)` | Row-wise dot product scoring |
| `cosine(query, docs)` | Row-wise cosine scoring |
| `l2_distance(query, docs)` | Row-wise distance scoring |
| `topk(scores, k)` | Return top-`k` indices from a score vector |
| `kv_read(cache)` | Read from KV cache |
| `kv_write(cache, value)` | Write to KV cache |

**Binary operators**: `+`, `-`, `*`, `/` (element-wise on tensors).

### Statements

```manta
let result = @matmul(x, w)    // local binding
return softmax(result)         // return value
kv_write(cache, value)         // expression statement (side effect)
```

## Compilation Pipeline

```
.manta source
  |  Parse (recursive descent)
  v
Syntax AST
  |  Semantic analysis (type check, scope, constraints)
  v
HIR -- source-oriented, fully typed, symbolic shapes
  |
  v
MIR -- tensor-semantic operations (gather, matmul, softmax, ...)
  |
  v
LIR -- scheduled plan: buffers, kernels, schedule hints, storage classes
  |
  v
.mll artifact -- backend-neutral plan with CUDA, Metal, Vulkan, DirectML, and WebGPU kernel variants
```

### Intermediate representations

**HIR** preserves source structure with full type information. Parameters carry binding paths, entry points distinguish kernels from pipelines, types include symbolic shape dimensions.

**MIR** classifies operations semantically: `gather`, `matmul`, `softmax`, `rope`, `kv_read`, `kv_write`, `kernel_call`, `pointwise`, `reduce`, `dequant`. Backend-neutral.

**LIR** is a scheduled execution plan. Buffers have storage classes (`device_local`, `host_visible`, `unified`). Kernels have schedule hints (`tile`, `vector_width`, `subgroup`, `memory`). All concepts are backend-neutral -- `tile` not `blockDim`, `subgroup` not `warp`.

### Schedule hints

Hints guide backend-specific lowering without leaking backend concepts into the IR:

| Operation | Tile | Vector Width | Subgroup | Memory |
|-----------|------|-------------|----------|--------|
| softmax | [64] | 1 | yes | workgroup_local |
| normalize | [128] | 4 | yes | workgroup_local |
| rope | [128] | 2 | yes | -- |
| binary ops | [128] | 4 | -- | -- |

These map to thread blocks, threadgroups, workgroups, or backend graph dispatches at the backend level.

## Artifact Format

The `.mll` artifact carries a Manta execution plan:

```json
{
  "version": "manta/v0alpha1",
  "name": "tiny_embed",
  "params": [{"name": "token_embedding", "binding": "weights/token_embedding", ...}],
  "entry_points": [{"name": "embed", "kind": "pipeline", ...}],
  "requirements": {"supported_backends": ["cuda", "metal", "vulkan", "directml", "webgpu"]},
  "buffers": [{"name": "hidden", "dtype": "f16", "storage_class": "device_local", ...}],
  "kernels": [{"name": "l2_normalize", "variants": [
    {"backend": "cuda", "entry": "l2_normalize_cuda", "source": "..."},
    {"backend": "metal", "entry": "l2_normalize_metal", "source": "..."},
    {"backend": "vulkan", "entry": "l2_normalize_vulkan", "source": "..."},
    {"backend": "directml", "entry": "l2_normalize_directml", "source": "..."},
    {"backend": "webgpu", "entry": "l2_normalize_webgpu", "source": "..."}
  ], ...}],
  "steps": [
    {"kind": "gather", "inputs": ["token_embedding", "tokens"], "outputs": ["hidden"]},
    {"kind": "matmul", "inputs": ["hidden", "projection"], "outputs": ["projected"]},
    {"kind": "launch_kernel", "kernel": "l2_normalize", "inputs": ["projected"], "outputs": ["result"]},
    {"kind": "return", "outputs": ["embeddings"]}
  ]
}
```

Artifacts are validated on load: all referenced buffers, kernels, and entry points must exist, I/O flows must be consistent, and kernel variants must be present for all declared backends.

## Runtime

### Loading and executing

```go
import (
    mantaartifact "github.com/odvcencio/manta/artifact/manta"
    "github.com/odvcencio/manta/runtime"
    "github.com/odvcencio/manta/runtime/backend"
    "github.com/odvcencio/manta/runtime/backends/cuda"
    "github.com/odvcencio/manta/runtime/backends/directml"
    "github.com/odvcencio/manta/runtime/backends/metal"
    "github.com/odvcencio/manta/runtime/backends/vulkan"
    "github.com/odvcencio/manta/runtime/backends/webgpu"
)

rt := runtime.New(cuda.New(), metal.New(), vulkan.New(), directml.New(), webgpu.New())

prog, err := rt.LoadFile(ctx, "model.mll",
    runtime.WithWeight("token_embedding", embeddingData),
    runtime.WithWeight("projection", projectionData),
)

result, err := prog.Run(ctx, backend.Request{
    Entry:  "embed",
    Inputs: map[string]any{"tokens": []int32{1, 42, 7}},
})

output := result.Outputs["embeddings"]
```

The runtime tries each backend in registration order. The first backend that can load the module is selected. Weight bindings are validated against the module's parameter declarations.

For promoted kernel classes, the runtime can compile and launch backend-native kernels. Where a kernel shape has not been promoted yet, the backend still owns execution but may fall back to the host reference path. This keeps the runtime honest while Manta grows.

### Backend interface

```go
type Backend interface {
    Kind() mantaartifact.BackendKind
    CanLoad(mod *mantaartifact.Module) bool
    Load(ctx context.Context, mod *mantaartifact.Module, weights map[string]WeightBinding) (Executor, error)
}

type Executor interface {
    Backend() mantaartifact.BackendKind
    Run(ctx context.Context, req Request) (Result, error)
}
```

### Execution trace

Every execution returns a trace of steps with kernel variant information:

```go
for _, step := range result.Trace {
    fmt.Printf("%s: %s (variant: %s)\n", step.Kind, step.Name, step.Variant)
}
// gather: gather (variant: )
// matmul: matmul (variant: )
// launch_kernel: l2_normalize (variant: l2_normalize_cuda)
// return: return (variant: )
```

Outputs also carry backend launch metadata such as the selected kernel entry, launch API, tile-derived launch shape, and whether execution ran on device or through backend-owned host fallback.

## Deployment targets

Manta is being shaped around two concrete deployment targets:

1. standalone inference binaries written in Go
2. CorkScrewDB as a runtime host for embedding, reranking, and quantized vector-aware scoring

That means `.mll` artifacts should be good at:

- `embed(tokens) -> embeddings`
- `score(query, docs) -> scores`
- `rerank(query, docs) -> top_ids`
- `decode_step(x, cache) -> logits`

and eventually:

- consuming quantized vector collections directly
- producing TurboQuant-native outputs without post-hoc repacking

## CLI

```
manta compile <source.manta> [output.mll]             Compile .manta source to a Manta artifact
manta init-model [flags] <artifact.mll>             Create the default quantized embedding training package
manta train-corpus [flags] <artifact.mll> <corpus>  Train tokenizer, mine pairs, and fit the embedder
manta tokenize-embed <artifact.mll> <text> <tokens> Convert text JSONL to reusable token JSONL
manta train-embed [flags] <artifact.mll> <train>    Fit an initialized package on token or text JSONL
manta train-embed --eval-only <artifact.mll> <eval> Evaluate a package without optimizer steps
manta train-embed --no-tokenizer <artifact.mll> <tokens> Force token JSONL beside a tokenizer
manta compare-train-metrics <current> [baseline]    Summarize training metrics JSON and deltas
manta diagnose-train-metrics <metrics.json>        Explain backend use and transfer pressure
manta gate-train-metrics [flags] <metrics.json>    Enforce quality and efficiency thresholds
manta export-mll <artifact.mll> [output.mll]        Seal an artifact package into a weight-carrying MLL file
manta inspect <artifact.mll>                        Inspect and verify an artifact package
manta run <artifact.mll> [entry]                    Load and execute an artifact entry point
manta demo [tiny_embed|tiny_decode|tiny_score]      Run a built-in preset module
manta version                                      Print version
```

Before a candidate run, use `ferrous-wheel run scripts/verify_manta_production.fw` to preflight the local `.mll` training, eval-only, sealed export, and inspect path.

For production-grade `manta-embed-v1` candidate training, use `scripts/acquire_manta_embed_v1_datasets.fw` followed by `scripts/train_manta_embed_v1_candidate.fw`; they record dataset hashes, repo provenance, eval-only gates, sealed export, and artifact hashes. See `docs/production-embedding.md`.

## Design Constraints

- **No CUDA-only concepts leak upward.** The MIR and LIR use `tile`, `subgroup`, `vector`, and abstract memory classes. Backend-specific mapping happens only during kernel source emission.
- **Same source, same artifact, same runtime contract** across all backends.

## Status

The compiler, IR pipeline, artifact format, semantic analysis, runtime, and CLI are functional. CUDA executes promoted kernel classes on device on Linux, Metal has the matching Apple device path, and Vulkan, DirectML, and WebGPU now have artifact/compiler/runtime surfaces that execute through backend-owned host fallback while device runtimes land. The retrieval surface includes direct quantized scoring plus `topk`-based reranking.

## Development

```bash
CGO_ENABLED=0 go test ./artifact/manta ./cmd/manta ./compiler ./models ./runtime/backend ./runtime/backends/metal ./runtime/backends/vulkan ./runtime/backends/directml ./runtime/backends/webgpu ./syntax
go build ./cmd/manta/
```

CUDA-backed runtime tests require a working CUDA device and should be run separately from the no-cgo public gate.

## Benchmarks

Manta keeps reproducible perf checks as Ferrous Wheel workflows.

```bash
MANTA_BENCH_ROOT=$PWD ferrous-wheel run scripts/bench.fw
MANTA_BENCH_ROOT=$PWD MANTA_BENCH_CUDA=1 ferrous-wheel run scripts/bench.fw
MANTA_BENCH_ROOT=$PWD MANTA_BENCH_MODEL_ASSETS=/path/to/assets/manta-embed-v1 ferrous-wheel run scripts/bench.fw
```

Current `manta-embed-v1` CUDA smoke: `845.15` train examples/s and `865437.87` train pairs/s on batch `1024`, with the promoted grouped CUDA training path enabled by default. See `docs/benchmarks.md` for the full profile and the next perf targets.

## License

Manta is open source under the Apache License, Version 2.0. See `LICENSE` and `NOTICE`.
