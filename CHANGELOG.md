# Changelog

## v0.1.0-alpha — 2026-04-09

### Added

- **Inference-only product direction**: Barracuda is explicitly scoped as an inference language, not a training language. Backward pass, optimizer state, and training loops are non-goals for the core Barracuda surface.
- **Embedding/reranker focus**: the repo spec now positions Barracuda around OSS embedding models, rerankers, retrieval-time scoring, quantized inference, and CorkScrewDB integration.
- **TurboQuant-native direction**: the spec now calls out quantized tensor and quantized vector support as a first-class product target rather than an afterthought.
- **Source language** (`.bar`): `param`, `kernel`, `pipeline` declarations with typed tensor shapes, symbolic dimensions, `kv_cache` type, `@weight` bindings, intrinsic calls (`@matmul`), built-in functions (gather, softmax, normalize, rmsnorm, rope, dequant, kv_read, kv_write), binary operators, `let` bindings, `return` statements
- **Recursive descent parser** with lexer, span tracking, and diagnostic collection
- **Semantic analysis**: type checking, scope validation, duplicate detection, constraint enforcement (matmul rank, pipeline-only ops, no numeric literals yet)
- **Three-level IR pipeline**: HIR (source-oriented, typed) -> MIR (tensor-semantic operations) -> LIR (scheduled plan with buffers, kernels, schedule hints)
- **Artifact format** (`barr/v0alpha1`): JSON-serialized execution plan with dual-backend kernel variants, comprehensive validation, file I/O
- **Kernel synthesis**: auto-generates kernels for binary ops and builtins with CUDA and Metal source emission
- **Schedule hints**: tile, vector_width, subgroup, memory class -- backend-neutral scheduling guidance
- **Hybrid executor**: host-based reference execution for the bootstrap surface, plus backend-native dispatch for promoted kernel classes
- **Tensor runtime**: `Tensor` type with f16/f32/i32 support, `KVCache` mutable state, value materialization with symbolic dimension binding
- **Runtime**: backend registry, module loading with weight binding validation, backend selection (CUDA-first, Metal fallback)
- **CUDA backend**: variant compilation with SHA256 source hashing, backend-owned kernel dispatch, and real Linux/CUDA device execution for promoted row-wise and element-wise kernel classes
- **Metal backend**: same public interface and launch-model contract as CUDA, with backend-owned execution path and darwin-specific device runtime scaffold
- **CLI** (`barr`): `compile`, `run`, `demo`, `version` subcommands
- **Presets**: `tiny_embed` (embedding + projection + L2 normalize) and `tiny_decode` (matmul + rope + KV cache + softmax attention)
- **Quantized-native retrieval**: `tiny_score` now scores directly against `q4` docs without a fake dequant hop
- **Reranking surface**: `topk(scores, k)` plus `tiny_rerank` make it possible to return ranked ids directly from `.bar`
