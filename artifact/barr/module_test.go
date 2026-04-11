package barr

import (
	"os"
	"path/filepath"
	"testing"
)

func TestEncodeDecodeJSONRoundTrip(t *testing.T) {
	module := NewModule("demo")
	module.Params = []Param{
		{
			Name:    "token_embedding",
			Binding: "weights/token_embedding",
			Type: ValueType{
				Kind:   ValueTensor,
				Tensor: &TensorType{DType: "f16", Shape: []string{"V", "D"}},
			},
		},
	}
	module.EntryPoints = []EntryPoint{
		{
			Name: "embed",
			Kind: EntryPointPipeline,
			Inputs: []ValueBinding{
				{Name: "tokens", Type: ValueType{Kind: ValueTensor, Tensor: &TensorType{DType: "i32", Shape: []string{"T"}}}},
			},
			Outputs: []ValueBinding{
				{Name: "embeddings", Type: ValueType{Kind: ValueTensor, Tensor: &TensorType{DType: "f16", Shape: []string{"T", "D"}}}},
			},
		},
	}
	module.Buffers = []Buffer{
		{Name: "x", DType: "f16", Shape: []string{"N", "D"}, StorageClass: "device_local"},
	}
	module.Kernels = []Kernel{
		{
			Name: "rmsnorm",
			Inputs: []ValueBinding{
				{Name: "x", Type: ValueType{Kind: ValueTensor, Tensor: &TensorType{DType: "f16", Shape: []string{"T", "D"}}}},
			},
			Outputs: []ValueBinding{
				{Name: "y", Type: ValueType{Kind: ValueTensor, Tensor: &TensorType{DType: "f16", Shape: []string{"T", "D"}}}},
			},
			Hints: ScheduleHints{Tile: []int{128}, VectorWidth: 4, Subgroup: true, Memory: "device_local"},
			Body: []KernelOp{
				{Kind: KernelOpPointwise, Name: "normalize", Op: "normalize", Inputs: []string{"x"}, Outputs: []string{"y"}},
				{Kind: KernelOpReturn, Op: "return", Outputs: []string{"y"}},
			},
			Variants: []KernelVariant{
				{Backend: BackendCUDA, Entry: "rmsnorm_cuda", Source: "extern \"C\" __global__ void rmsnorm_cuda() {}"},
				{Backend: BackendMetal, Entry: "rmsnorm_metal", Source: "kernel void rmsnorm_metal() {}"},
			},
		},
	}
	module.Steps = []Step{
		{Entry: "embed", Kind: StepGather, Name: "embed_lookup", Inputs: []string{"token_embedding", "tokens"}, Outputs: []string{"x"}},
		{Entry: "embed", Kind: StepReturn, Outputs: []string{"embeddings"}},
	}

	data, err := EncodeJSON(module)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}

	got, err := DecodeJSON(data)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}

	if got.Name != module.Name {
		t.Fatalf("name mismatch: got %q want %q", got.Name, module.Name)
	}
	if len(got.Params) != 1 || got.Params[0].Binding != "weights/token_embedding" {
		t.Fatalf("param binding mismatch: %+v", got.Params)
	}
	if !got.SupportsBackend(BackendCUDA) || !got.SupportsBackend(BackendMetal) {
		t.Fatalf("expected both CUDA and Metal support: %+v", got.Requirements.SupportedBackends)
	}
	if len(got.Steps) != len(module.Steps) {
		t.Fatalf("step count mismatch: got %d want %d", len(got.Steps), len(module.Steps))
	}
	if len(got.Kernels) != 1 || len(got.Kernels[0].Body) != 2 {
		t.Fatalf("kernel body mismatch: %+v", got.Kernels)
	}

	path := filepath.Join(t.TempDir(), "demo.mll")
	if err := WriteFile(path, module); err != nil {
		t.Fatalf("write file: %v", err)
	}
	data, err = os.ReadFile(path)
	if err != nil {
		t.Fatalf("read raw file: %v", err)
	}
	if !IsMLLBytes(data) {
		t.Fatal("expected MLL file bytes")
	}
	fromFile, err := ReadFile(path)
	if err != nil {
		t.Fatalf("read file: %v", err)
	}
	if fromFile.Name != module.Name || len(fromFile.Kernels) != 1 {
		t.Fatalf("file roundtrip mismatch: %+v", fromFile)
	}

	jsonPath := filepath.Join(t.TempDir(), "legacy-demo.barr")
	if err := WriteJSONFile(jsonPath, module); err != nil {
		t.Fatalf("write legacy json file: %v", err)
	}
	fromJSON, err := ReadFile(jsonPath)
	if err != nil {
		t.Fatalf("read legacy json file: %v", err)
	}
	if fromJSON.Name != module.Name || len(fromJSON.EntryPoints) != len(module.EntryPoints) {
		t.Fatalf("legacy json roundtrip mismatch: %+v", fromJSON)
	}
}

func TestValidateRejectsUnknownStepEntry(t *testing.T) {
	module := NewModule("bad")
	module.EntryPoints = []EntryPoint{
		{
			Name: "embed",
			Kind: EntryPointPipeline,
			Inputs: []ValueBinding{
				{Name: "tokens", Type: ValueType{Kind: ValueTensor, Tensor: &TensorType{DType: "i32", Shape: []string{"T"}}}},
			},
			Outputs: []ValueBinding{
				{Name: "embeddings", Type: ValueType{Kind: ValueTensor, Tensor: &TensorType{DType: "f16", Shape: []string{"T", "D"}}}},
			},
		},
	}
	module.Buffers = []Buffer{{Name: "x", DType: "f16", Shape: []string{"T", "D"}}}
	module.Steps = []Step{
		{Entry: "missing", Kind: StepGather, Inputs: []string{"tokens"}, Outputs: []string{"x"}},
	}

	if err := module.Validate(); err == nil {
		t.Fatal("expected validate error")
	}
}

func TestValidateRejectsMissingKernelVariantSource(t *testing.T) {
	module := NewModule("bad-kernel")
	module.EntryPoints = []EntryPoint{
		{
			Name: "embed",
			Kind: EntryPointPipeline,
			Inputs: []ValueBinding{
				{Name: "tokens", Type: ValueType{Kind: ValueTensor, Tensor: &TensorType{DType: "i32", Shape: []string{"T"}}}},
			},
			Outputs: []ValueBinding{
				{Name: "embeddings", Type: ValueType{Kind: ValueTensor, Tensor: &TensorType{DType: "f16", Shape: []string{"T", "D"}}}},
			},
		},
	}
	module.Buffers = []Buffer{{Name: "embeddings", DType: "f16", Shape: []string{"T", "D"}}}
	module.Kernels = []Kernel{
		{
			Name: "embed_kernel",
			Inputs: []ValueBinding{
				{Name: "x", Type: ValueType{Kind: ValueTensor, Tensor: &TensorType{DType: "f16", Shape: []string{"T", "D"}}}},
			},
			Outputs: []ValueBinding{
				{Name: "result", Type: ValueType{Kind: ValueTensor, Tensor: &TensorType{DType: "f16", Shape: []string{"T", "D"}}}},
			},
			Body: []KernelOp{
				{Kind: KernelOpPointwise, Op: "identity", Inputs: []string{"x"}, Outputs: []string{"result"}},
				{Kind: KernelOpReturn, Op: "return", Outputs: []string{"result"}},
			},
			Variants: []KernelVariant{
				{Backend: BackendCUDA, Entry: "embed_kernel_cuda", Source: "extern \"C\" __global__ void embed_kernel_cuda() {}"},
				{Backend: BackendMetal, Entry: "embed_kernel_metal"},
			},
		},
	}
	module.Steps = []Step{
		{Entry: "embed", Kind: StepLaunchKernel, Name: "embed_kernel", Kernel: "embed_kernel", Inputs: []string{"tokens"}, Outputs: []string{"embeddings"}},
		{Entry: "embed", Kind: StepReturn, Outputs: []string{"embeddings"}},
	}

	if err := module.Validate(); err == nil {
		t.Fatal("expected validate error")
	}
}

func TestValidateCandidatePackEntryPoint(t *testing.T) {
	module := NewModule("packed")
	module.EntryPoints = []EntryPoint{
		{
			Name: "rerank_candidates_packed",
			Kind: EntryPointPipeline,
			Inputs: []ValueBinding{
				{Name: "query", Type: ValueType{Kind: ValueTensor, Tensor: &TensorType{DType: "f16", Shape: []string{"D"}}}},
				{Name: "docs", Type: ValueType{Kind: ValueTensor, Tensor: &TensorType{DType: "q4", Shape: []string{"N", "D"}}}},
				{Name: "candidate_ids", Type: ValueType{Kind: ValueTensor, Tensor: &TensorType{DType: "i64", Shape: []string{"N"}}}},
			},
			Outputs: []ValueBinding{
				{Name: "candidates", Type: ValueType{Kind: ValueCandidatePack, CandidatePack: &CandidatePackType{Shape: []string{"K", "D"}}}},
			},
		},
	}
	module.Buffers = []Buffer{
		{Name: "scores", DType: "f32", Shape: []string{"N"}},
		{Name: "top_indices", DType: "i32", Shape: []string{"K"}},
		{Name: "top_candidate_ids", DType: "i64", Shape: []string{"K"}},
		{Name: "top_scores", DType: "f32", Shape: []string{"K"}},
		{Name: "top_docs", DType: "q4", Shape: []string{"K", "D"}},
		{Name: "candidates", DType: "candidate_pack", Shape: []string{"K", "D"}},
	}
	module.Kernels = []Kernel{
		{
			Name: "cosine",
			Inputs: []ValueBinding{
				{Name: "query", Type: ValueType{Kind: ValueTensor, Tensor: &TensorType{DType: "f16", Shape: []string{"D"}}}},
				{Name: "docs", Type: ValueType{Kind: ValueTensor, Tensor: &TensorType{DType: "q4", Shape: []string{"N", "D"}}}},
			},
			Outputs: []ValueBinding{
				{Name: "scores", Type: ValueType{Kind: ValueTensor, Tensor: &TensorType{DType: "f32", Shape: []string{"N"}}}},
			},
			Body: []KernelOp{
				{Kind: KernelOpBuiltin, Op: "cosine", Inputs: []string{"query", "docs"}, Outputs: []string{"scores"}},
				{Kind: KernelOpReturn, Op: "return", Outputs: []string{"scores"}},
			},
			Variants: []KernelVariant{
				{Backend: BackendCUDA, Entry: "cosine_cuda", Source: "extern \"C\" __global__ void cosine_cuda() {}"},
				{Backend: BackendMetal, Entry: "cosine_metal", Source: "kernel void cosine_metal() {}"},
			},
		},
	}
	module.Steps = []Step{
		{Entry: "rerank_candidates_packed", Kind: StepLaunchKernel, Name: "cosine", Kernel: "cosine", Inputs: []string{"query", "docs"}, Outputs: []string{"scores"}},
		{Entry: "rerank_candidates_packed", Kind: StepTopK, Name: "topk", Inputs: []string{"scores"}, Outputs: []string{"top_indices"}, Attributes: map[string]string{"k": "2"}},
		{Entry: "rerank_candidates_packed", Kind: StepGather, Name: "gather_ids", Inputs: []string{"candidate_ids", "top_indices"}, Outputs: []string{"top_candidate_ids"}},
		{Entry: "rerank_candidates_packed", Kind: StepGather, Name: "gather_scores", Inputs: []string{"scores", "top_indices"}, Outputs: []string{"top_scores"}},
		{Entry: "rerank_candidates_packed", Kind: StepGather, Name: "gather_docs", Inputs: []string{"docs", "top_indices"}, Outputs: []string{"top_docs"}},
		{Entry: "rerank_candidates_packed", Kind: StepPack, Name: "pack_candidates", Inputs: []string{"top_candidate_ids", "top_scores", "top_docs"}, Outputs: []string{"candidates"}},
		{Entry: "rerank_candidates_packed", Kind: StepReturn, Outputs: []string{"candidates"}},
	}

	if err := module.Validate(); err != nil {
		t.Fatalf("validate candidate pack: %v", err)
	}
}
