# Changelog

## Unreleased

### Added

- Added a staged `manta-embed-v1` shipping pipeline that trains a mixed pretrain/BEIR candidate, mines model-hard negatives, runs a FiQA-weighted fine-tune, evaluates BEIR retrieval metrics, and can install gated assets into CorkScrewDB.
- Added `retrievaldump` for per-query BEIR retrieval diagnostics.

### Changed

- Manta source parsing now uses a Go-authored gotreesitter grammar and lowers the CST into the existing compiler AST.
- Default embedding package initialization now exposes `encoder_repeats` through the Go config and `manta init-model --encoder-repeats`.

## v0.1.0-alpha — 2026-04-09

### Added

- **Inference-first product direction**: Manta ships the narrowest useful product surface first: embedding, reranking, scoring, decode, quantized inference, and CorkScrewDB integration. Native training support is part of the stack so default models can be trained and shipped through the same artifact family.
- **Embedding/reranker focus**: the repo spec now positions Manta around OSS embedding models, rerankers, retrieval-time scoring, quantized inference, and CorkScrewDB integration.
- **TurboQuant-native direction**: the spec now calls out quantized tensor and quantized vector support as a first-class product target rather than an afterthought.
- **Source language** (`.manta`): `param`, `kernel`, `pipeline` declarations with typed tensor shapes, symbolic dimensions, `kv_cache` type, `@weight` bindings, intrinsic calls (`@matmul`), built-in functions (gather, softmax, normalize, rmsnorm, rope, dequant, kv_read, kv_write), binary operators, `let` bindings, `return` statements
- **Recursive descent parser** with lexer, span tracking, and diagnostic collection
- **Semantic analysis**: type checking, scope validation, duplicate detection, constraint enforcement (matmul rank, pipeline-only ops, no numeric literals yet)
- **Three-level IR pipeline**: HIR (source-oriented, typed) -> MIR (tensor-semantic operations) -> LIR (scheduled plan with buffers, kernels, schedule hints)
- **Artifact format** (`manta/v0alpha1`): JSON-serialized execution plan with dual-backend kernel variants, comprehensive validation, file I/O
- **Kernel synthesis**: auto-generates kernels for binary ops and builtins with CUDA and Metal source emission
- **Schedule hints**: tile, vector_width, subgroup, memory class -- backend-neutral scheduling guidance
- **Hybrid executor**: host-based reference execution for the bootstrap surface, plus backend-native dispatch for promoted kernel classes
- **Tensor runtime**: `Tensor` type with f16/f32/i32 support, `KVCache` mutable state, value materialization with symbolic dimension binding
- **Runtime**: backend registry, module loading with weight binding validation, backend selection (CUDA-first, Metal fallback)
- **CUDA backend**: variant compilation with SHA256 source hashing, backend-owned kernel dispatch, and real Linux/CUDA device execution for promoted row-wise and element-wise kernel classes
- **Metal backend**: same public interface and launch-model contract as CUDA, with backend-owned execution path and darwin-specific device runtime scaffold
- **CLI** (`manta`): `compile`, `run`, `demo`, `version` subcommands
- **Presets**: `tiny_embed` (embedding + projection + L2 normalize) and `tiny_decode` (matmul + rope + KV cache + softmax attention)
- **Quantized-native retrieval**: `tiny_score` now scores directly against `q4` docs without a fake dequant hop
- **Reranking surface**: `topk(scores, k)` plus `tiny_rerank` make it possible to return ranked ids directly from `.manta`
