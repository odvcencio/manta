package mantaruntime

import (
	"context"
	"fmt"
	"path/filepath"
	"strings"
	"testing"

	mantaartifact "github.com/odvcencio/manta/artifact/manta"
	"github.com/odvcencio/manta/compiler"
	"github.com/odvcencio/manta/runtime/backend"
	"github.com/odvcencio/manta/runtime/backends/cuda"
	"github.com/odvcencio/manta/runtime/backends/directml"
	"github.com/odvcencio/manta/runtime/backends/metal"
	"github.com/odvcencio/manta/runtime/backends/vulkan"
	"github.com/odvcencio/manta/runtime/backends/webgpu"
)

type stubBackend struct {
	kind         mantaartifact.BackendKind
	capabilities []string
	loads        int
}

func (b *stubBackend) Kind() mantaartifact.BackendKind { return b.kind }

func (b *stubBackend) Capabilities() []string {
	return append([]string(nil), b.capabilities...)
}

func (b *stubBackend) CanLoad(mod *mantaartifact.Module) bool {
	return mod != nil && mod.SupportsBackend(b.kind)
}

func (b *stubBackend) Load(_ context.Context, _ *mantaartifact.Module, _ map[string]backend.WeightBinding) (backend.Executor, error) {
	b.loads++
	return stubExecutor{kind: b.kind}, nil
}

type stubExecutor struct {
	kind mantaartifact.BackendKind
}

func (e stubExecutor) Backend() mantaartifact.BackendKind { return e.kind }

func (e stubExecutor) Run(_ context.Context, _ backend.Request) (backend.Result, error) {
	return backend.Result{}, fmt.Errorf("stub executor does not run")
}

type cacheKeyStubBackend struct {
	stubBackend
	cacheKeys []string
}

func (b *cacheKeyStubBackend) Load(_ context.Context, _ *mantaartifact.Module, _ map[string]backend.WeightBinding) (backend.Executor, error) {
	return nil, fmt.Errorf("unexpected uncached load")
}

func (b *cacheKeyStubBackend) LoadWithCacheKey(_ context.Context, _ *mantaartifact.Module, _ map[string]backend.WeightBinding, cacheKey string) (backend.Executor, error) {
	b.loads++
	b.cacheKeys = append(b.cacheKeys, cacheKey)
	return stubExecutor{kind: b.kind}, nil
}

func TestLoadRejectsMissingWeightBindings(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_embed", Preset: compiler.PresetTinyEmbed})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	rt := New(cuda.New(), metal.New())
	_, err = rt.Load(context.Background(), bundle.Artifact, WithWeight("token_embedding", backend.NewTensorF16([]int{3, 2}, []float32{
		1, 0,
		0, 1,
		1, 1,
	})))
	if err == nil {
		t.Fatal("expected missing weight binding error")
	}
	if !strings.Contains(err.Error(), "projection") {
		t.Fatalf("expected missing projection binding, got %v", err)
	}
}

func TestLoadAcceptsTinyEmbedBindings(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_embed", Preset: compiler.PresetTinyEmbed})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	rt := New(cuda.New(), metal.New())
	prog, err := rt.Load(
		context.Background(),
		bundle.Artifact,
		tinyEmbedWeights()...,
	)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if got := prog.Backend(); got == "" {
		t.Fatal("expected selected backend")
	}
}

func TestLoadFallsBackWhenBackendMissingCapabilities(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_candidates", Preset: compiler.PresetTinyPackedCandidates})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	cudaOnly := &stubBackend{kind: mantaartifact.BackendCUDA}
	metalCapable := &stubBackend{
		kind:         mantaartifact.BackendMetal,
		capabilities: []string{mantaartifact.CapabilityCandidatePack},
	}
	rt := New(cudaOnly, metalCapable)
	prog, err := rt.Load(context.Background(), bundle.Artifact, tinyRerankWeights()...)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if got := prog.Backend(); got != mantaartifact.BackendMetal {
		t.Fatalf("backend = %q, want %q", got, mantaartifact.BackendMetal)
	}
	if cudaOnly.loads != 0 {
		t.Fatalf("unexpected CUDA load attempts = %d", cudaOnly.loads)
	}
	if metalCapable.loads != 1 {
		t.Fatalf("metal load attempts = %d, want 1", metalCapable.loads)
	}
}

func TestLoadReportsMissingBackendCapabilities(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_candidates", Preset: compiler.PresetTinyPackedCandidates})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	rt := New(&stubBackend{kind: mantaartifact.BackendCUDA}, &stubBackend{kind: mantaartifact.BackendMetal})
	_, err = rt.Load(context.Background(), bundle.Artifact, tinyRerankWeights()...)
	if err == nil {
		t.Fatal("expected missing capability error")
	}
	if !strings.Contains(err.Error(), mantaartifact.CapabilityCandidatePack) {
		t.Fatalf("expected candidate_pack in error, got %v", err)
	}
}

func TestLoadFileAcceptsSerializedArtifact(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_embed", Preset: compiler.PresetTinyEmbed})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	path := filepath.Join(t.TempDir(), "tiny_embed.mll")
	if err := mantaartifact.WriteFile(path, bundle.Artifact); err != nil {
		t.Fatalf("write artifact: %v", err)
	}

	rt := New(cuda.New(), metal.New())
	prog, err := rt.LoadFile(context.Background(), path, tinyEmbedWeights()...)
	if err != nil {
		t.Fatalf("load file: %v", err)
	}
	if got := prog.Backend(); got == "" {
		t.Fatal("expected selected backend from file load")
	}
}

func TestLoadFileUsesSiblingPackageManifestCacheKey(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_embed", Preset: compiler.PresetTinyEmbed})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	dir := t.TempDir()
	path := filepath.Join(dir, "tiny_embed.mll")
	if err := mantaartifact.WriteFile(path, bundle.Artifact); err != nil {
		t.Fatalf("write artifact: %v", err)
	}

	manifest, err := BuildPackageManifest(PackageEmbedding, bundle.Artifact, map[string]string{
		"artifact": path,
	})
	if err != nil {
		t.Fatalf("build package manifest: %v", err)
	}
	manifestPath := DefaultPackageManifestPath(path)
	if err := manifest.WriteFile(manifestPath); err != nil {
		t.Fatalf("write package manifest: %v", err)
	}

	backend := &cacheKeyStubBackend{stubBackend: stubBackend{kind: mantaartifact.BackendCUDA}}
	rt := New(backend)
	loadOnce := func() error {
		_, err := rt.LoadFile(context.Background(), path, tinyEmbedWeights()...)
		return err
	}
	if err := loadOnce(); err != nil {
		t.Fatalf("first load file: %v", err)
	}
	if err := loadOnce(); err != nil {
		t.Fatalf("second load file: %v", err)
	}

	wantKey := manifest.CacheKey()
	if wantKey == "" {
		t.Fatal("expected non-empty cache key")
	}
	if backend.loads != 2 {
		t.Fatalf("cache-key loads = %d, want 2", backend.loads)
	}
	if len(backend.cacheKeys) != 2 {
		t.Fatalf("cache key calls = %d, want 2", len(backend.cacheKeys))
	}
	for i, got := range backend.cacheKeys {
		if got != wantKey {
			t.Fatalf("cache key call %d = %q, want %q", i, got, wantKey)
		}
	}
}

func TestRunHonorsLazyStagedMemoryPlan(t *testing.T) {
	mod := newLazyStagedParamModule()
	weights := map[string]*backend.Tensor{
		"used":   backend.NewTensorF16([]int{2, 2}, []float32{1, 2, 3, 4}),
		"unused": backend.NewTensorF16([]int{2, 2}, []float32{9, 8, 7, 6}),
	}
	plan := NewMemoryPlan(mod, weights, MemoryPlanOptions{DeviceBudgetBytes: 1})

	rt := New(cuda.New(), metal.New())
	prog, err := rt.Load(
		context.Background(),
		mod,
		WithWeight("used", weights["used"]),
		WithWeight("unused", weights["unused"]),
		WithMemoryPlan(plan),
	)
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	result, err := prog.Run(context.Background(), backend.Request{Entry: "serve", Inputs: map[string]any{}})
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	output, ok := result.Outputs["result"]
	if !ok {
		t.Fatalf("missing result output: %+v", result.Outputs)
	}
	tensor, ok := output.Data.(*backend.Tensor)
	if !ok {
		t.Fatalf("output data type = %T, want *backend.Tensor", output.Data)
	}
	assertTensorClose(t, tensor, []int{2, 2}, []float32{1, 2, 3, 4})
	if got := result.Metadata["params_eager_materialized"]; got != "0" {
		t.Fatalf("params_eager_materialized = %q, want 0", got)
	}
	if got := result.Metadata["params_lazy_materialized"]; got != "1" {
		t.Fatalf("params_lazy_materialized = %q, want 1", got)
	}
	if got := result.Metadata["params_released"]; got != "1" {
		t.Fatalf("params_released = %q, want 1", got)
	}
	if got := result.Metadata["params_unused_for_entry"]; got != "1" {
		t.Fatalf("params_unused_for_entry = %q, want 1", got)
	}
	if got := result.Metadata["param_materialization"]; got != "lazy_on_demand" {
		t.Fatalf("param_materialization = %q, want lazy_on_demand", got)
	}
}

func TestRunTinyEmbedEntryPoint(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_embed", Preset: compiler.PresetTinyEmbed})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	rt := New(cuda.New(), metal.New())
	prog, err := rt.Load(context.Background(), bundle.Artifact, tinyEmbedWeights()...)
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	result, err := prog.Run(context.Background(), backend.Request{
		Entry:  "embed",
		Inputs: map[string]any{"tokens": backend.NewTensorI32([]int{2}, []int32{0, 2})},
	})
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	output, ok := result.Outputs["embeddings"]
	if !ok {
		t.Fatalf("missing embeddings output: %+v", result.Outputs)
	}
	if output.Type.Kind != mantaartifact.ValueTensor || output.Type.Tensor == nil {
		t.Fatalf("unexpected output type: %+v", output.Type)
	}
	if output.Type.Tensor.DType != "f16" {
		t.Fatalf("output dtype = %q, want f16", output.Type.Tensor.DType)
	}
	if got := strings.Join(output.Type.Tensor.Shape, ","); got != "2,2" {
		t.Fatalf("output shape = %q, want 2,2", got)
	}
	if output.Producer != "kernel:l2_normalize" {
		t.Fatalf("output producer = %q, want kernel:l2_normalize", output.Producer)
	}
	tensor, ok := output.Data.(*backend.Tensor)
	if !ok {
		t.Fatalf("output data type = %T, want *backend.Tensor", output.Data)
	}
	assertTensorClose(t, tensor, []int{2, 2}, []float32{
		1, 0,
		0.70710677, 0.70710677,
	})
	if output.Metadata["variant_entry"] != "l2_normalize_cuda" {
		t.Fatalf("variant_entry = %v, want l2_normalize_cuda", output.Metadata["variant_entry"])
	}
	if output.Metadata["dispatch_mode"] != "backend_native" {
		t.Fatalf("dispatch_mode = %v, want backend_native", output.Metadata["dispatch_mode"])
	}
	if output.Metadata["launch_api"] != "cuLaunchKernel" {
		t.Fatalf("launch_api = %v, want cuLaunchKernel", output.Metadata["launch_api"])
	}
	if output.Metadata["launch_block_size"] != 128 {
		t.Fatalf("launch_block_size = %v, want 128", output.Metadata["launch_block_size"])
	}
	if got := result.Metadata["compiled_kernels"]; got != "1" {
		t.Fatalf("compiled_kernels = %q, want 1", got)
	}
	if got := result.Metadata["kernel_dispatch"]; got != "backend_native" {
		t.Fatalf("kernel_dispatch = %q, want backend_native", got)
	}
	if got := result.Metadata["entrypoint"]; got != "embed" {
		t.Fatalf("entrypoint metadata = %q, want embed", got)
	}
	if got := len(result.Trace); got != len(bundle.Artifact.Steps) {
		t.Fatalf("trace len = %d, want %d", got, len(bundle.Artifact.Steps))
	}
	if result.Trace[len(result.Trace)-2].Variant != "l2_normalize_cuda" {
		t.Fatalf("trace variant = %q, want l2_normalize_cuda", result.Trace[len(result.Trace)-2].Variant)
	}
}

func TestPortableGPUBackendsLoadMLLArtifactsWithHostFallback(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_embed", Preset: compiler.PresetTinyEmbed})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	cases := []struct {
		name      string
		rt        *Runtime
		backend   mantaartifact.BackendKind
		entry     string
		launchAPI string
	}{
		{name: "vulkan", rt: New(vulkan.New()), backend: mantaartifact.BackendVulkan, entry: "l2_normalize_vulkan", launchAPI: "vkCmdDispatch"},
		{name: "directml", rt: New(directml.New()), backend: mantaartifact.BackendDirectML, entry: "l2_normalize_directml", launchAPI: "IDMLCommandRecorder::RecordDispatch"},
		{name: "webgpu", rt: New(webgpu.New()), backend: mantaartifact.BackendWebGPU, entry: "l2_normalize_webgpu", launchAPI: "GPUComputePassEncoder.dispatchWorkgroups"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			prog, err := tc.rt.Load(context.Background(), bundle.Artifact, tinyEmbedWeights()...)
			if err != nil {
				t.Fatalf("load: %v", err)
			}
			if got := prog.Backend(); got != tc.backend {
				t.Fatalf("backend = %q, want %q", got, tc.backend)
			}
			result, err := prog.Run(context.Background(), backend.Request{
				Entry:  "embed",
				Inputs: map[string]any{"tokens": backend.NewTensorI32([]int{2}, []int32{0, 2})},
			})
			if err != nil {
				t.Fatalf("run: %v", err)
			}
			output := result.Outputs["embeddings"]
			if output.Metadata["variant_entry"] != tc.entry {
				t.Fatalf("variant_entry = %v, want %s", output.Metadata["variant_entry"], tc.entry)
			}
			if output.Metadata["launch_api"] != tc.launchAPI {
				t.Fatalf("launch_api = %v, want %s", output.Metadata["launch_api"], tc.launchAPI)
			}
			if output.Metadata["execution_mode"] != "host_fallback" {
				t.Fatalf("execution_mode = %v, want host_fallback", output.Metadata["execution_mode"])
			}
			tensor := output.Data.(*backend.Tensor)
			assertTensorClose(t, tensor, []int{2, 2}, []float32{
				1, 0,
				0.70710677, 0.70710677,
			})
		})
	}
}

func TestRunTinyDecodeEntryPoint(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_decode", Preset: compiler.PresetTinyDecode})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	rt := New(cuda.New(), metal.New())
	prog, err := rt.Load(context.Background(), bundle.Artifact, WithWeight("wq", backend.NewTensorF16([]int{2, 2}, []float32{
		1, 0,
		0, 1,
	})))
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	cache := backend.NewKVCache(backend.NewTensorF16([]int{2, 2}, []float32{
		0, 0,
		0, 0,
	}))
	result, err := prog.Run(context.Background(), backend.Request{
		Entry: "decode_step",
		Inputs: map[string]any{
			"x": backend.NewTensorF16([]int{2, 2}, []float32{
				1, 0,
				0, 1,
			}),
			"cache": cache,
		},
	})
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	output, ok := result.Outputs["logits"]
	if !ok {
		t.Fatalf("missing logits output: %+v", result.Outputs)
	}
	if output.Producer != "kernel:softmax" {
		t.Fatalf("output producer = %q, want kernel:softmax", output.Producer)
	}
	tensor, ok := output.Data.(*backend.Tensor)
	if !ok {
		t.Fatalf("output data type = %T, want *backend.Tensor", output.Data)
	}
	if output.Metadata["variant_entry"] != "softmax_cuda" {
		t.Fatalf("variant_entry = %v, want softmax_cuda", output.Metadata["variant_entry"])
	}
	if output.Metadata["launch_api"] != "cuLaunchKernel" {
		t.Fatalf("launch_api = %v, want cuLaunchKernel", output.Metadata["launch_api"])
	}
	if output.Metadata["launch_block_size"] != 64 {
		t.Fatalf("launch_block_size = %v, want 64", output.Metadata["launch_block_size"])
	}
	if output.Metadata["launch_memory"] != "workgroup_local" {
		t.Fatalf("launch_memory = %v, want workgroup_local", output.Metadata["launch_memory"])
	}
	for r := 0; r < tensor.Shape[0]; r++ {
		rowSum := float32(0)
		for c := 0; c < tensor.Shape[1]; c++ {
			rowSum += tensor.F32[r*tensor.Shape[1]+c]
		}
		assertClose(t, rowSum, 1, 0.0005)
	}
	if got := result.Metadata["backend"]; got != "cuda" {
		t.Fatalf("backend metadata = %q, want cuda", got)
	}
	if got := result.Metadata["status"]; got != "hybrid" {
		t.Fatalf("status metadata = %q, want hybrid", got)
	}
	if got := result.Metadata["step_count"]; got != "7" {
		t.Fatalf("step_count metadata = %q, want 7", got)
	}
	if got := result.Metadata["compiled_kernels"]; got != "3" {
		t.Fatalf("compiled_kernels = %q, want 3", got)
	}
	if got := result.Trace[len(result.Trace)-2].Variant; got != "softmax_cuda" {
		t.Fatalf("softmax trace variant = %q, want softmax_cuda", got)
	}
	if cache.Value == nil {
		t.Fatal("expected kv cache mutation")
	}
	assertTensorClose(t, cache.Value, []int{2, 2}, []float32{
		1, 0,
		-0.84147096, 0.5403023,
	})
}

func TestRunTinyScoreEntryPoint(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_score", Preset: compiler.PresetTinyScore})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	rt := New(cuda.New(), metal.New())
	prog, err := rt.Load(context.Background(), bundle.Artifact, tinyScoreWeights()...)
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	result, err := prog.Run(context.Background(), backend.Request{
		Entry:  "score",
		Inputs: map[string]any{"query": backend.NewTensorF16([]int{2}, []float32{1, 0})},
	})
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	output, ok := result.Outputs["scores"]
	if !ok {
		t.Fatalf("missing scores output: %+v", result.Outputs)
	}
	if output.Producer != "kernel:cosine" {
		t.Fatalf("output producer = %q, want kernel:cosine", output.Producer)
	}
	if output.Type.Kind != mantaartifact.ValueTensor || output.Type.Tensor == nil || output.Type.Tensor.DType != "f32" {
		t.Fatalf("unexpected output type: %+v", output.Type)
	}
	tensor, ok := output.Data.(*backend.Tensor)
	if !ok {
		t.Fatalf("output data type = %T, want *backend.Tensor", output.Data)
	}
	assertTensorClose(t, tensor, []int{2}, []float32{1, 0})
	if output.Metadata["variant_entry"] != "cosine_cuda" {
		t.Fatalf("variant_entry = %v, want cosine_cuda", output.Metadata["variant_entry"])
	}
	if output.Metadata["device_execution"] != true {
		t.Fatalf("device_execution = %v, want true", output.Metadata["device_execution"])
	}
	if got := result.Metadata["compiled_kernels"]; got != "1" {
		t.Fatalf("compiled_kernels = %q, want 1", got)
	}
	if got := result.Metadata["entrypoint"]; got != "score" {
		t.Fatalf("entrypoint metadata = %q, want score", got)
	}
}

func TestRunTinyRerankEntryPoint(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_rerank", Preset: compiler.PresetTinyRerank})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	rt := New(cuda.New(), metal.New())
	prog, err := rt.Load(context.Background(), bundle.Artifact, tinyRerankWeights()...)
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	result, err := prog.Run(context.Background(), backend.Request{
		Entry:  "rerank",
		Inputs: map[string]any{"query": backend.NewTensorF16([]int{2}, []float32{1, 0})},
	})
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	topIDs, ok := result.Outputs["top_ids"]
	if !ok {
		t.Fatalf("missing top_ids output: %+v", result.Outputs)
	}
	if topIDs.Producer != "topk:topk" {
		t.Fatalf("top_ids producer = %q, want topk:topk", topIDs.Producer)
	}
	tensor, ok := topIDs.Data.(*backend.Tensor)
	if !ok {
		t.Fatalf("top_ids data type = %T, want *backend.Tensor", topIDs.Data)
	}
	assertTensorI32(t, tensor, []int{2}, []int32{0, 2})
	topScores, ok := result.Outputs["top_scores"]
	if !ok {
		t.Fatalf("missing top_scores output: %+v", result.Outputs)
	}
	if topScores.Producer != "gather:gather" {
		t.Fatalf("top_scores producer = %q, want gather:gather", topScores.Producer)
	}
	scoreTensor, ok := topScores.Data.(*backend.Tensor)
	if !ok {
		t.Fatalf("top_scores data type = %T, want *backend.Tensor", topScores.Data)
	}
	assertTensorClose(t, scoreTensor, []int{2}, []float32{1, 0.70710677})
	if got := result.Metadata["entrypoint"]; got != "rerank" {
		t.Fatalf("entrypoint metadata = %q, want rerank", got)
	}
	if got := len(result.Trace); got != 4 {
		t.Fatalf("trace len = %d, want 4", got)
	}
	if result.Trace[1].Kind != mantaartifact.StepTopK {
		t.Fatalf("trace[1].kind = %q, want topk", result.Trace[1].Kind)
	}
	if result.Trace[2].Kind != mantaartifact.StepGather {
		t.Fatalf("trace[2].kind = %q, want gather", result.Trace[2].Kind)
	}
}

func TestRunTinySelectEntryPoint(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_select", Preset: compiler.PresetTinySelect})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	rt := New(cuda.New(), metal.New())
	prog, err := rt.Load(context.Background(), bundle.Artifact, tinyRerankWeights()...)
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	result, err := prog.Run(context.Background(), backend.Request{
		Entry:  "select_scores",
		Inputs: map[string]any{"query": backend.NewTensorF16([]int{2}, []float32{1, 0})},
	})
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	output, ok := result.Outputs["top_scores"]
	if !ok {
		t.Fatalf("missing top_scores output: %+v", result.Outputs)
	}
	if output.Producer != "gather:gather" {
		t.Fatalf("output producer = %q, want gather:gather", output.Producer)
	}
	tensor, ok := output.Data.(*backend.Tensor)
	if !ok {
		t.Fatalf("output data type = %T, want *backend.Tensor", output.Data)
	}
	assertTensorClose(t, tensor, []int{2}, []float32{1, 0.70710677})
	if got := len(result.Trace); got != 4 {
		t.Fatalf("trace len = %d, want 4", got)
	}
	if result.Trace[2].Kind != mantaartifact.StepGather {
		t.Fatalf("trace[2].kind = %q, want gather", result.Trace[2].Kind)
	}
}

func TestRunTinyRetrieveEntryPoint(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_retrieve", Preset: compiler.PresetTinyRetrieve})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	rt := New(cuda.New(), metal.New())
	prog, err := rt.Load(context.Background(), bundle.Artifact, tinyRerankWeights()...)
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	result, err := prog.Run(context.Background(), backend.Request{
		Entry:  "retrieve",
		Inputs: map[string]any{"query": backend.NewTensorF16([]int{2}, []float32{1, 0})},
	})
	if err != nil {
		t.Fatalf("run: %v", err)
	}

	topIDs, ok := result.Outputs["top_ids"]
	if !ok {
		t.Fatalf("missing top_ids output: %+v", result.Outputs)
	}
	topScores, ok := result.Outputs["top_scores"]
	if !ok {
		t.Fatalf("missing top_scores output: %+v", result.Outputs)
	}
	topDocs, ok := result.Outputs["top_docs"]
	if !ok {
		t.Fatalf("missing top_docs output: %+v", result.Outputs)
	}

	idTensor := topIDs.Data.(*backend.Tensor)
	scoreTensor := topScores.Data.(*backend.Tensor)
	docTensor := topDocs.Data.(*backend.Tensor)
	assertTensorI32(t, idTensor, []int{2}, []int32{0, 2})
	assertTensorClose(t, scoreTensor, []int{2}, []float32{1, 0.70710677})
	assertTensorClose(t, docTensor, []int{2, 2}, []float32{
		1, 0,
		1, 1,
	})
	if docTensor.DType != "q4" {
		t.Fatalf("top_docs dtype = %q, want q4", docTensor.DType)
	}
	if got := len(result.Trace); got != 5 {
		t.Fatalf("trace len = %d, want 5", got)
	}
	if result.Trace[3].Kind != mantaartifact.StepGather {
		t.Fatalf("trace[3].kind = %q, want gather", result.Trace[3].Kind)
	}
}

func TestRunTinyEmbedEntryPointOnMetal(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_embed", Preset: compiler.PresetTinyEmbed})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	rt := New(metal.New())
	prog, err := rt.Load(context.Background(), bundle.Artifact, tinyEmbedWeights()...)
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	result, err := prog.Run(context.Background(), backend.Request{
		Entry:  "embed",
		Inputs: map[string]any{"tokens": backend.NewTensorI32([]int{2}, []int32{0, 2})},
	})
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if got := prog.Backend(); got != mantaartifact.BackendMetal {
		t.Fatalf("backend = %q, want metal", got)
	}
	output := result.Outputs["embeddings"]
	if output.Metadata["variant_entry"] != "l2_normalize_metal" {
		t.Fatalf("variant_entry = %v, want l2_normalize_metal", output.Metadata["variant_entry"])
	}
	if output.Metadata["dispatch_mode"] != "backend_native" {
		t.Fatalf("dispatch_mode = %v, want backend_native", output.Metadata["dispatch_mode"])
	}
	if output.Metadata["launch_api"] != "dispatchThreadgroups" {
		t.Fatalf("launch_api = %v, want dispatchThreadgroups", output.Metadata["launch_api"])
	}
	if output.Metadata["launch_threadgroup_size"] != 128 {
		t.Fatalf("launch_threadgroup_size = %v, want 128", output.Metadata["launch_threadgroup_size"])
	}
	if result.Trace[len(result.Trace)-2].Variant != "l2_normalize_metal" {
		t.Fatalf("trace variant = %q, want l2_normalize_metal", result.Trace[len(result.Trace)-2].Variant)
	}
	tensor, ok := output.Data.(*backend.Tensor)
	if !ok {
		t.Fatalf("metal output data type = %T, want *backend.Tensor", output.Data)
	}
	assertTensorClose(t, tensor, []int{2, 2}, []float32{
		1, 0,
		0.70710677, 0.70710677,
	})
}

func TestRunTinyEmbedParityAcrossBackends(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_embed", Preset: compiler.PresetTinyEmbed})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	cudaRuntime := New(cuda.New())
	cudaProg, err := cudaRuntime.Load(context.Background(), bundle.Artifact, tinyEmbedWeights()...)
	if err != nil {
		t.Fatalf("load cuda: %v", err)
	}
	cudaResult, err := cudaProg.Run(context.Background(), backend.Request{
		Entry:  "embed",
		Inputs: map[string]any{"tokens": backend.NewTensorI32([]int{2}, []int32{0, 2})},
	})
	if err != nil {
		t.Fatalf("run cuda: %v", err)
	}

	metalRuntime := New(metal.New())
	metalProg, err := metalRuntime.Load(context.Background(), bundle.Artifact, tinyEmbedWeights()...)
	if err != nil {
		t.Fatalf("load metal: %v", err)
	}
	metalResult, err := metalProg.Run(context.Background(), backend.Request{
		Entry:  "embed",
		Inputs: map[string]any{"tokens": backend.NewTensorI32([]int{2}, []int32{0, 2})},
	})
	if err != nil {
		t.Fatalf("run metal: %v", err)
	}

	cudaTensor := cudaResult.Outputs["embeddings"].Data.(*backend.Tensor)
	metalTensor := metalResult.Outputs["embeddings"].Data.(*backend.Tensor)
	assertTensorClose(t, cudaTensor, metalTensor.Shape, metalTensor.F32)
}

func TestRunTinyScoreParityAcrossBackends(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_score", Preset: compiler.PresetTinyScore})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	cudaRuntime := New(cuda.New())
	cudaProg, err := cudaRuntime.Load(context.Background(), bundle.Artifact, tinyScoreWeights()...)
	if err != nil {
		t.Fatalf("load cuda: %v", err)
	}
	cudaResult, err := cudaProg.Run(context.Background(), backend.Request{
		Entry:  "score",
		Inputs: map[string]any{"query": backend.NewTensorF16([]int{2}, []float32{1, 0})},
	})
	if err != nil {
		t.Fatalf("run cuda: %v", err)
	}

	metalRuntime := New(metal.New())
	metalProg, err := metalRuntime.Load(context.Background(), bundle.Artifact, tinyScoreWeights()...)
	if err != nil {
		t.Fatalf("load metal: %v", err)
	}
	metalResult, err := metalProg.Run(context.Background(), backend.Request{
		Entry:  "score",
		Inputs: map[string]any{"query": backend.NewTensorF16([]int{2}, []float32{1, 0})},
	})
	if err != nil {
		t.Fatalf("run metal: %v", err)
	}

	cudaTensor := cudaResult.Outputs["scores"].Data.(*backend.Tensor)
	metalTensor := metalResult.Outputs["scores"].Data.(*backend.Tensor)
	assertTensorClose(t, cudaTensor, metalTensor.Shape, metalTensor.F32)
	if metalResult.Outputs["scores"].Metadata["variant_entry"] != "cosine_metal" {
		t.Fatalf("metal variant_entry = %v, want cosine_metal", metalResult.Outputs["scores"].Metadata["variant_entry"])
	}
}

func TestRunTinyRerankParityAcrossBackends(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_rerank", Preset: compiler.PresetTinyRerank})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	cudaRuntime := New(cuda.New())
	cudaProg, err := cudaRuntime.Load(context.Background(), bundle.Artifact, tinyRerankWeights()...)
	if err != nil {
		t.Fatalf("load cuda: %v", err)
	}
	cudaResult, err := cudaProg.Run(context.Background(), backend.Request{
		Entry:  "rerank",
		Inputs: map[string]any{"query": backend.NewTensorF16([]int{2}, []float32{1, 0})},
	})
	if err != nil {
		t.Fatalf("run cuda: %v", err)
	}

	metalRuntime := New(metal.New())
	metalProg, err := metalRuntime.Load(context.Background(), bundle.Artifact, tinyRerankWeights()...)
	if err != nil {
		t.Fatalf("load metal: %v", err)
	}
	metalResult, err := metalProg.Run(context.Background(), backend.Request{
		Entry:  "rerank",
		Inputs: map[string]any{"query": backend.NewTensorF16([]int{2}, []float32{1, 0})},
	})
	if err != nil {
		t.Fatalf("run metal: %v", err)
	}

	cudaTensor := cudaResult.Outputs["top_ids"].Data.(*backend.Tensor)
	metalTensor := metalResult.Outputs["top_ids"].Data.(*backend.Tensor)
	assertTensorI32(t, cudaTensor, metalTensor.Shape, metalTensor.I32)
	cudaScores := cudaResult.Outputs["top_scores"].Data.(*backend.Tensor)
	metalScores := metalResult.Outputs["top_scores"].Data.(*backend.Tensor)
	assertTensorClose(t, cudaScores, metalScores.Shape, metalScores.F32)
}

func TestRunTinySelectParityAcrossBackends(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_select", Preset: compiler.PresetTinySelect})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	cudaRuntime := New(cuda.New())
	cudaProg, err := cudaRuntime.Load(context.Background(), bundle.Artifact, tinyRerankWeights()...)
	if err != nil {
		t.Fatalf("load cuda: %v", err)
	}
	cudaResult, err := cudaProg.Run(context.Background(), backend.Request{
		Entry:  "select_scores",
		Inputs: map[string]any{"query": backend.NewTensorF16([]int{2}, []float32{1, 0})},
	})
	if err != nil {
		t.Fatalf("run cuda: %v", err)
	}

	metalRuntime := New(metal.New())
	metalProg, err := metalRuntime.Load(context.Background(), bundle.Artifact, tinyRerankWeights()...)
	if err != nil {
		t.Fatalf("load metal: %v", err)
	}
	metalResult, err := metalProg.Run(context.Background(), backend.Request{
		Entry:  "select_scores",
		Inputs: map[string]any{"query": backend.NewTensorF16([]int{2}, []float32{1, 0})},
	})
	if err != nil {
		t.Fatalf("run metal: %v", err)
	}

	cudaTensor := cudaResult.Outputs["top_scores"].Data.(*backend.Tensor)
	metalTensor := metalResult.Outputs["top_scores"].Data.(*backend.Tensor)
	assertTensorClose(t, cudaTensor, metalTensor.Shape, metalTensor.F32)
}

func TestRunTinyRetrieveParityAcrossBackends(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_retrieve", Preset: compiler.PresetTinyRetrieve})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	cudaRuntime := New(cuda.New())
	cudaProg, err := cudaRuntime.Load(context.Background(), bundle.Artifact, tinyRerankWeights()...)
	if err != nil {
		t.Fatalf("load cuda: %v", err)
	}
	cudaResult, err := cudaProg.Run(context.Background(), backend.Request{
		Entry:  "retrieve",
		Inputs: map[string]any{"query": backend.NewTensorF16([]int{2}, []float32{1, 0})},
	})
	if err != nil {
		t.Fatalf("run cuda: %v", err)
	}

	metalRuntime := New(metal.New())
	metalProg, err := metalRuntime.Load(context.Background(), bundle.Artifact, tinyRerankWeights()...)
	if err != nil {
		t.Fatalf("load metal: %v", err)
	}
	metalResult, err := metalProg.Run(context.Background(), backend.Request{
		Entry:  "retrieve",
		Inputs: map[string]any{"query": backend.NewTensorF16([]int{2}, []float32{1, 0})},
	})
	if err != nil {
		t.Fatalf("run metal: %v", err)
	}

	cudaIDs := cudaResult.Outputs["top_ids"].Data.(*backend.Tensor)
	metalIDs := metalResult.Outputs["top_ids"].Data.(*backend.Tensor)
	assertTensorI32(t, cudaIDs, metalIDs.Shape, metalIDs.I32)
	cudaScores := cudaResult.Outputs["top_scores"].Data.(*backend.Tensor)
	metalScores := metalResult.Outputs["top_scores"].Data.(*backend.Tensor)
	assertTensorClose(t, cudaScores, metalScores.Shape, metalScores.F32)
	cudaDocs := cudaResult.Outputs["top_docs"].Data.(*backend.Tensor)
	metalDocs := metalResult.Outputs["top_docs"].Data.(*backend.Tensor)
	assertTensorClose(t, cudaDocs, metalDocs.Shape, metalDocs.F32)
	if metalDocs.DType != "q4" {
		t.Fatalf("metal top_docs dtype = %q, want q4", metalDocs.DType)
	}
}

func TestRunTinyCandidatesEntryPoint(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_candidates", Preset: compiler.PresetTinyCandidates})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	rt := New(cuda.New(), metal.New())
	prog, err := rt.Load(context.Background(), bundle.Artifact)
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	result, err := prog.Run(context.Background(), backend.Request{
		Entry: "rerank_candidates",
		Inputs: map[string]any{
			"query":         backend.NewTensorF16([]int{2}, []float32{1, 0}),
			"docs":          backend.NewTensorQ4([]int{3, 2}, []float32{1, 0, 0, 1, 1, 1}),
			"candidate_ids": backend.NewTensorI64([]int{3}, []int64{1001, 2002, 3003}),
		},
	})
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	topCandidateIDs := result.Outputs["top_candidate_ids"].Data.(*backend.Tensor)
	topScores := result.Outputs["top_scores"].Data.(*backend.Tensor)
	topDocs := result.Outputs["top_docs"].Data.(*backend.Tensor)
	assertTensorI64(t, topCandidateIDs, []int{2}, []int64{1001, 3003})
	assertTensorClose(t, topScores, []int{2}, []float32{1, 0.70710677})
	assertTensorClose(t, topDocs, []int{2, 2}, []float32{
		1, 0,
		1, 1,
	})
	if topDocs.DType != "q4" {
		t.Fatalf("top_docs dtype = %q, want q4", topDocs.DType)
	}
}

func TestRunTinyCandidatesParityAcrossBackends(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_candidates", Preset: compiler.PresetTinyCandidates})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	cudaRuntime := New(cuda.New())
	cudaProg, err := cudaRuntime.Load(context.Background(), bundle.Artifact)
	if err != nil {
		t.Fatalf("load cuda: %v", err)
	}
	cudaResult, err := cudaProg.Run(context.Background(), backend.Request{
		Entry: "rerank_candidates",
		Inputs: map[string]any{
			"query":         backend.NewTensorF16([]int{2}, []float32{1, 0}),
			"docs":          backend.NewTensorQ4([]int{3, 2}, []float32{1, 0, 0, 1, 1, 1}),
			"candidate_ids": backend.NewTensorI64([]int{3}, []int64{1001, 2002, 3003}),
		},
	})
	if err != nil {
		t.Fatalf("run cuda: %v", err)
	}

	metalRuntime := New(metal.New())
	metalProg, err := metalRuntime.Load(context.Background(), bundle.Artifact)
	if err != nil {
		t.Fatalf("load metal: %v", err)
	}
	metalResult, err := metalProg.Run(context.Background(), backend.Request{
		Entry: "rerank_candidates",
		Inputs: map[string]any{
			"query":         backend.NewTensorF16([]int{2}, []float32{1, 0}),
			"docs":          backend.NewTensorQ4([]int{3, 2}, []float32{1, 0, 0, 1, 1, 1}),
			"candidate_ids": backend.NewTensorI64([]int{3}, []int64{1001, 2002, 3003}),
		},
	})
	if err != nil {
		t.Fatalf("run metal: %v", err)
	}

	cudaIDs := cudaResult.Outputs["top_candidate_ids"].Data.(*backend.Tensor)
	metalIDs := metalResult.Outputs["top_candidate_ids"].Data.(*backend.Tensor)
	assertTensorI64(t, cudaIDs, metalIDs.Shape, metalIDs.I64)
	cudaScores := cudaResult.Outputs["top_scores"].Data.(*backend.Tensor)
	metalScores := metalResult.Outputs["top_scores"].Data.(*backend.Tensor)
	assertTensorClose(t, cudaScores, metalScores.Shape, metalScores.F32)
	cudaDocs := cudaResult.Outputs["top_docs"].Data.(*backend.Tensor)
	metalDocs := metalResult.Outputs["top_docs"].Data.(*backend.Tensor)
	assertTensorClose(t, cudaDocs, metalDocs.Shape, metalDocs.F32)
	if metalDocs.DType != "q4" {
		t.Fatalf("metal top_docs dtype = %q, want q4", metalDocs.DType)
	}
}

func TestRunTinyBatchCandidatesEntryPoint(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_batch_candidates", Preset: compiler.PresetTinyBatchCandidates})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	rt := New(cuda.New(), metal.New())
	prog, err := rt.Load(context.Background(), bundle.Artifact)
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	result, err := prog.Run(context.Background(), backend.Request{
		Entry: "rerank_candidates_batch",
		Inputs: map[string]any{
			"queries": backend.NewTensorF16([]int{2, 2}, []float32{
				1, 0,
				0, 1,
			}),
			"docs": backend.NewTensorQ4([]int{2, 3, 2}, []float32{
				1, 0,
				0, 1,
				1, 1,
				0, 1,
				1, 0,
				1, 1,
			}),
			"candidate_ids": backend.NewTensorI64([]int{2, 3}, []int64{
				1001, 2002, 3003,
				4004, 5005, 6006,
			}),
		},
	})
	if err != nil {
		t.Fatalf("run: %v", err)
	}

	topCandidateIDs := result.Outputs["top_candidate_ids"].Data.(*backend.Tensor)
	topScores := result.Outputs["top_scores"].Data.(*backend.Tensor)
	topDocs := result.Outputs["top_docs"].Data.(*backend.Tensor)
	assertTensorI64(t, topCandidateIDs, []int{2, 2}, []int64{
		1001, 3003,
		4004, 6006,
	})
	assertTensorClose(t, topScores, []int{2, 2}, []float32{
		1, 0.70710677,
		1, 0.70710677,
	})
	assertTensorClose(t, topDocs, []int{2, 2, 2}, []float32{
		1, 0,
		1, 1,
		0, 1,
		1, 1,
	})
	if topDocs.DType != "q4" {
		t.Fatalf("top_docs dtype = %q, want q4", topDocs.DType)
	}
}

func TestRunTinyBatchCandidatesParityAcrossBackends(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_batch_candidates", Preset: compiler.PresetTinyBatchCandidates})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	inputs := map[string]any{
		"queries": backend.NewTensorF16([]int{2, 2}, []float32{
			1, 0,
			0, 1,
		}),
		"docs": backend.NewTensorQ4([]int{2, 3, 2}, []float32{
			1, 0,
			0, 1,
			1, 1,
			0, 1,
			1, 0,
			1, 1,
		}),
		"candidate_ids": backend.NewTensorI64([]int{2, 3}, []int64{
			1001, 2002, 3003,
			4004, 5005, 6006,
		}),
	}

	cudaRuntime := New(cuda.New())
	cudaProg, err := cudaRuntime.Load(context.Background(), bundle.Artifact)
	if err != nil {
		t.Fatalf("load cuda: %v", err)
	}
	cudaResult, err := cudaProg.Run(context.Background(), backend.Request{
		Entry:  "rerank_candidates_batch",
		Inputs: inputs,
	})
	if err != nil {
		t.Fatalf("run cuda: %v", err)
	}

	metalRuntime := New(metal.New())
	metalProg, err := metalRuntime.Load(context.Background(), bundle.Artifact)
	if err != nil {
		t.Fatalf("load metal: %v", err)
	}
	metalResult, err := metalProg.Run(context.Background(), backend.Request{
		Entry:  "rerank_candidates_batch",
		Inputs: inputs,
	})
	if err != nil {
		t.Fatalf("run metal: %v", err)
	}

	cudaIDs := cudaResult.Outputs["top_candidate_ids"].Data.(*backend.Tensor)
	metalIDs := metalResult.Outputs["top_candidate_ids"].Data.(*backend.Tensor)
	assertTensorI64(t, cudaIDs, metalIDs.Shape, metalIDs.I64)
	cudaScores := cudaResult.Outputs["top_scores"].Data.(*backend.Tensor)
	metalScores := metalResult.Outputs["top_scores"].Data.(*backend.Tensor)
	assertTensorClose(t, cudaScores, metalScores.Shape, metalScores.F32)
	cudaDocs := cudaResult.Outputs["top_docs"].Data.(*backend.Tensor)
	metalDocs := metalResult.Outputs["top_docs"].Data.(*backend.Tensor)
	assertTensorClose(t, cudaDocs, metalDocs.Shape, metalDocs.F32)
	if metalDocs.DType != "q4" {
		t.Fatalf("metal top_docs dtype = %q, want q4", metalDocs.DType)
	}
}

func TestRunTinyPackedCandidatesEntryPoint(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_packed_candidates", Preset: compiler.PresetTinyPackedCandidates})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	rt := New(cuda.New(), metal.New())
	prog, err := rt.Load(context.Background(), bundle.Artifact)
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	result, err := prog.Run(context.Background(), backend.Request{
		Entry: "rerank_candidates_packed",
		Inputs: map[string]any{
			"query":         backend.NewTensorF16([]int{2}, []float32{1, 0}),
			"docs":          backend.NewTensorQ4([]int{3, 2}, []float32{1, 0, 0, 1, 1, 1}),
			"candidate_ids": backend.NewTensorI64([]int{3}, []int64{1001, 2002, 3003}),
		},
	})
	if err != nil {
		t.Fatalf("run: %v", err)
	}

	output := result.Outputs["candidates"]
	if output.Type.Kind != mantaartifact.ValueCandidatePack || output.Type.CandidatePack == nil {
		t.Fatalf("output kind = %+v, want candidate_pack", output.Type)
	}
	pack, ok := output.Data.(*backend.CandidatePack)
	if !ok {
		t.Fatalf("output data type = %T, want *backend.CandidatePack", output.Data)
	}
	assertTensorI64(t, pack.IDs, []int{2}, []int64{1001, 3003})
	assertTensorClose(t, pack.Scores, []int{2}, []float32{1, 0.70710677})
	assertTensorClose(t, pack.Docs, []int{2, 2}, []float32{
		1, 0,
		1, 1,
	})
	if pack.Docs.DType != "q4" {
		t.Fatalf("packed docs dtype = %q, want q4", pack.Docs.DType)
	}
}

func TestRunTinyPackedCandidatesParityAcrossBackends(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_packed_candidates", Preset: compiler.PresetTinyPackedCandidates})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	inputs := map[string]any{
		"query":         backend.NewTensorF16([]int{2}, []float32{1, 0}),
		"docs":          backend.NewTensorQ4([]int{3, 2}, []float32{1, 0, 0, 1, 1, 1}),
		"candidate_ids": backend.NewTensorI64([]int{3}, []int64{1001, 2002, 3003}),
	}

	cudaRuntime := New(cuda.New())
	cudaProg, err := cudaRuntime.Load(context.Background(), bundle.Artifact)
	if err != nil {
		t.Fatalf("load cuda: %v", err)
	}
	cudaResult, err := cudaProg.Run(context.Background(), backend.Request{
		Entry:  "rerank_candidates_packed",
		Inputs: inputs,
	})
	if err != nil {
		t.Fatalf("run cuda: %v", err)
	}

	metalRuntime := New(metal.New())
	metalProg, err := metalRuntime.Load(context.Background(), bundle.Artifact)
	if err != nil {
		t.Fatalf("load metal: %v", err)
	}
	metalResult, err := metalProg.Run(context.Background(), backend.Request{
		Entry:  "rerank_candidates_packed",
		Inputs: inputs,
	})
	if err != nil {
		t.Fatalf("run metal: %v", err)
	}

	cudaPack := cudaResult.Outputs["candidates"].Data.(*backend.CandidatePack)
	metalPack := metalResult.Outputs["candidates"].Data.(*backend.CandidatePack)
	assertTensorI64(t, cudaPack.IDs, metalPack.IDs.Shape, metalPack.IDs.I64)
	assertTensorClose(t, cudaPack.Scores, metalPack.Scores.Shape, metalPack.Scores.F32)
	assertTensorClose(t, cudaPack.Docs, metalPack.Docs.Shape, metalPack.Docs.F32)
}

func TestRunBatchedPackedCandidatesEntryPoint(t *testing.T) {
	src := []byte(`
pipeline rerank_candidates_packed_batch(queries: f16[Q, D], docs: q4[Q, N, D], candidate_ids: i64[Q, N]) -> candidate_pack[Q, 2, D] {
    let scores = cosine(queries, docs)
    let top_indices = topk(scores, 2)
    let top_candidate_ids = gather(candidate_ids, top_indices)
    let top_scores = gather(scores, top_indices)
    let top_docs = gather(docs, top_indices)
    return pack_candidates(top_candidate_ids, top_scores, top_docs)
}
`)

	bundle, err := compiler.Build(src, compiler.Options{ModuleName: "packed_batch"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	rt := New(cuda.New(), metal.New())
	prog, err := rt.Load(context.Background(), bundle.Artifact)
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	result, err := prog.Run(context.Background(), backend.Request{
		Entry: "rerank_candidates_packed_batch",
		Inputs: map[string]any{
			"queries": backend.NewTensorF16([]int{2, 2}, []float32{
				1, 0,
				0, 1,
			}),
			"docs": backend.NewTensorQ4([]int{2, 3, 2}, []float32{
				1, 0,
				0, 1,
				1, 1,
				0, 1,
				1, 0,
				1, 1,
			}),
			"candidate_ids": backend.NewTensorI64([]int{2, 3}, []int64{
				1001, 2002, 3003,
				4004, 5005, 6006,
			}),
		},
	})
	if err != nil {
		t.Fatalf("run: %v", err)
	}

	output := result.Outputs["result"]
	if output.Type.Kind != mantaartifact.ValueCandidatePack || output.Type.CandidatePack == nil {
		t.Fatalf("output kind = %+v, want candidate_pack", output.Type)
	}
	pack := output.Data.(*backend.CandidatePack)
	assertTensorI64(t, pack.IDs, []int{2, 2}, []int64{
		1001, 3003,
		4004, 6006,
	})
	assertTensorClose(t, pack.Scores, []int{2, 2}, []float32{
		1, 0.70710677,
		1, 0.70710677,
	})
	assertTensorClose(t, pack.Docs, []int{2, 2, 2}, []float32{
		1, 0,
		1, 1,
		0, 1,
		1, 1,
	})
}

func TestRunBatchedScoreFallsBackToHost(t *testing.T) {
	src := []byte(`
pipeline score_batch(queries: f16[Q, D], docs: q4[Q, N, D]) -> (scores: f32[Q, N]) {
    return cosine(queries, docs)
}
`)

	bundle, err := compiler.Build(src, compiler.Options{ModuleName: "score_batch"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	rt := New(cuda.New(), metal.New())
	prog, err := rt.Load(context.Background(), bundle.Artifact)
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	result, err := prog.Run(context.Background(), backend.Request{
		Entry: "score_batch",
		Inputs: map[string]any{
			"queries": backend.NewTensorF16([]int{2, 2}, []float32{
				1, 0,
				0, 1,
			}),
			"docs": backend.NewTensorQ4([]int{2, 3, 2}, []float32{
				1, 0,
				0, 1,
				1, 1,
				0, 1,
				1, 0,
				1, 1,
			}),
		},
	})
	if err != nil {
		t.Fatalf("run: %v", err)
	}

	output := result.Outputs["scores"]
	tensor := output.Data.(*backend.Tensor)
	assertTensorClose(t, tensor, []int{2, 3}, []float32{
		1, 0, 0.70710677,
		1, 0, 0.70710677,
	})
	if output.Metadata["device_execution"] != false {
		t.Fatalf("device_execution = %v, want false", output.Metadata["device_execution"])
	}
	if output.Metadata["execution_mode"] != "host_fallback" {
		t.Fatalf("execution_mode = %v, want host_fallback", output.Metadata["execution_mode"])
	}
	if output.Metadata["launch_api"] != "host_reference" {
		t.Fatalf("launch_api = %v, want host_reference", output.Metadata["launch_api"])
	}
}

func TestRunDirectQuantizedBuiltinScoreOps(t *testing.T) {
	cases := []struct {
		name     string
		op       string
		want     []float32
		producer string
	}{
		{name: "dot", op: "dot", want: []float32{1, 0, 1}, producer: "kernel:dot"},
		{name: "cosine", op: "cosine", want: []float32{1, 0, 0.70710677}, producer: "kernel:cosine"},
		{name: "l2_distance", op: "l2_distance", want: []float32{0, 1.4142135, 1}, producer: "kernel:l2_distance"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			src := []byte(`
pipeline score(query: f16[D], docs: q4[N, D]) -> f32[N] {
    return ` + tc.op + `(query, docs)
}
`)

			bundle, err := compiler.Build(src, compiler.Options{ModuleName: tc.name})
			if err != nil {
				t.Fatalf("build: %v", err)
			}

			rt := New(cuda.New())
			prog, err := rt.Load(context.Background(), bundle.Artifact)
			if err != nil {
				t.Fatalf("load: %v", err)
			}

			result, err := prog.Run(context.Background(), backend.Request{
				Entry: "score",
				Inputs: map[string]any{
					"query": backend.NewTensorF16([]int{2}, []float32{1, 0}),
					"docs": backend.NewTensorQ4([]int{3, 2}, []float32{
						1, 0,
						0, 1,
						1, 1,
					}),
				},
			})
			if err != nil {
				t.Fatalf("run: %v", err)
			}

			output, ok := result.Outputs["scores"]
			if !ok {
				t.Fatalf("missing scores output: %+v", result.Outputs)
			}
			if output.Producer != tc.producer {
				t.Fatalf("output producer = %q, want %q", output.Producer, tc.producer)
			}
			tensor, ok := output.Data.(*backend.Tensor)
			if !ok {
				t.Fatalf("output data type = %T, want *backend.Tensor", output.Data)
			}
			assertTensorClose(t, tensor, []int{3}, tc.want)
			if output.Metadata["device_execution"] != true {
				t.Fatalf("device_execution = %v, want true", output.Metadata["device_execution"])
			}
		})
	}
}

func TestRunRejectsMissingEntryInput(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_embed", Preset: compiler.PresetTinyEmbed})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	rt := New(cuda.New(), metal.New())
	prog, err := rt.Load(context.Background(), bundle.Artifact, tinyEmbedWeights()...)
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	_, err = prog.Run(context.Background(), backend.Request{Entry: "embed", Inputs: map[string]any{}})
	if err == nil {
		t.Fatal("expected missing input error")
	}
	if !strings.Contains(err.Error(), "tokens") {
		t.Fatalf("expected tokens missing error, got %v", err)
	}
}

func TestLoadRejectsInconsistentParamShapes(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_embed", Preset: compiler.PresetTinyEmbed})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	rt := New(cuda.New())
	_, err = rt.Load(
		context.Background(),
		bundle.Artifact,
		WithWeight("token_embedding", backend.NewTensorF16([]int{3, 2}, []float32{
			1, 0,
			0, 1,
			1, 1,
		})),
		WithWeight("projection", backend.NewTensorF16([]int{3, 2}, []float32{
			1, 0,
			0, 1,
			1, 1,
		})),
	)
	if err == nil {
		t.Fatal("expected param shape mismatch")
	}
	if !strings.Contains(err.Error(), "symbol \"D\" mismatch") {
		t.Fatalf("expected D mismatch, got %v", err)
	}
}

func TestRunRejectsInputShapeMismatch(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_decode", Preset: compiler.PresetTinyDecode})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	rt := New(cuda.New())
	prog, err := rt.Load(
		context.Background(),
		bundle.Artifact,
		WithWeight("wq", backend.NewTensorF16([]int{2, 2}, []float32{
			1, 0,
			0, 1,
		})),
	)
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	_, err = prog.Run(context.Background(), backend.Request{
		Entry: "decode_step",
		Inputs: map[string]any{
			"x": backend.NewTensorF16([]int{2, 3}, []float32{
				1, 0, 0,
				0, 1, 0,
			}),
			"cache": backend.NewKVCache(nil),
		},
	})
	if err == nil {
		t.Fatal("expected input shape mismatch")
	}
	if !strings.Contains(err.Error(), "symbol \"D\" mismatch") {
		t.Fatalf("expected D mismatch, got %v", err)
	}
}

func TestRunIdentityPipelineUsesAliasSteps(t *testing.T) {
	src := []byte(`
pipeline identity(x: f16[T, D]) -> f16[T, D] {
    let forwarded = x
    return forwarded
}
`)

	bundle, err := compiler.Build(src, compiler.Options{ModuleName: "identity"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	rt := New(cuda.New())
	prog, err := rt.Load(context.Background(), bundle.Artifact)
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	input := backend.NewTensorF16([]int{2, 2}, []float32{
		1, 2,
		3, 4,
	})
	result, err := prog.Run(context.Background(), backend.Request{
		Entry:  "identity",
		Inputs: map[string]any{"x": input},
	})
	if err != nil {
		t.Fatalf("run: %v", err)
	}

	output, ok := result.Outputs["result"]
	if !ok {
		t.Fatalf("missing result output: %+v", result.Outputs)
	}
	tensor, ok := output.Data.(*backend.Tensor)
	if !ok {
		t.Fatalf("output data type = %T, want *backend.Tensor", output.Data)
	}
	assertTensorClose(t, tensor, []int{2, 2}, []float32{
		1, 2,
		3, 4,
	})
	if output.Producer != "input:x" {
		t.Fatalf("output producer = %q, want input:x", output.Producer)
	}
	if got := len(result.Trace); got != 3 {
		t.Fatalf("trace len = %d, want 3", got)
	}
	if result.Trace[0].Kind != mantaartifact.StepAlias || result.Trace[1].Kind != mantaartifact.StepAlias {
		t.Fatalf("expected alias trace steps, got %+v", result.Trace)
	}
}

func tinyEmbedWeights() []LoadOption {
	return []LoadOption{
		WithWeight("token_embedding", backend.NewTensorF16([]int{3, 2}, []float32{
			1, 0,
			0, 1,
			1, 1,
		})),
		WithWeight("projection", backend.NewTensorF16([]int{2, 2}, []float32{
			1, 0,
			0, 1,
		})),
	}
}

func tinyScoreWeights() []LoadOption {
	return []LoadOption{
		WithWeight("docs", backend.NewTensorQ4([]int{2, 2}, []float32{
			1, 0,
			0, 1,
		})),
	}
}

func tinyRerankWeights() []LoadOption {
	return []LoadOption{
		WithWeight("docs", backend.NewTensorQ4([]int{3, 2}, []float32{
			1, 0,
			0, 1,
			1, 1,
		})),
	}
}

func newLazyStagedParamModule() *mantaartifact.Module {
	mod := mantaartifact.NewModule("lazy_params")
	valueType := mantaartifact.ValueType{
		Kind: mantaartifact.ValueTensor,
		Tensor: &mantaartifact.TensorType{
			DType: "f16",
			Shape: []string{"T", "D"},
		},
	}
	mod.Params = []mantaartifact.Param{
		{Name: "used", Type: valueType, Binding: "used"},
		{Name: "unused", Type: valueType, Binding: "unused"},
	}
	mod.EntryPoints = []mantaartifact.EntryPoint{
		{
			Name: "serve",
			Kind: mantaartifact.EntryPointPipeline,
			Outputs: []mantaartifact.ValueBinding{
				{Name: "result", Type: valueType},
			},
		},
	}
	mod.Buffers = []mantaartifact.Buffer{
		{Name: "result", DType: "f16", Shape: []string{"T", "D"}},
	}
	mod.Steps = []mantaartifact.Step{
		{
			Entry:   "serve",
			Kind:    mantaartifact.StepAlias,
			Name:    "forward_used",
			Inputs:  []string{"used"},
			Outputs: []string{"result"},
		},
		{
			Entry:   "serve",
			Kind:    mantaartifact.StepReturn,
			Name:    "return_result",
			Outputs: []string{"result"},
		},
	}
	return mod
}

func assertTensorClose(t *testing.T, tensor *backend.Tensor, wantShape []int, want []float32) {
	t.Helper()
	if tensor == nil {
		t.Fatal("tensor is nil")
	}
	if len(tensor.Shape) != len(wantShape) {
		t.Fatalf("tensor rank = %d, want %d", len(tensor.Shape), len(wantShape))
	}
	for i := range wantShape {
		if tensor.Shape[i] != wantShape[i] {
			t.Fatalf("tensor shape[%d] = %d, want %d", i, tensor.Shape[i], wantShape[i])
		}
	}
	if len(tensor.F32) != len(want) {
		t.Fatalf("tensor values len = %d, want %d", len(tensor.F32), len(want))
	}
	for i, got := range tensor.F32 {
		assertClose(t, got, want[i], 0.0005)
	}
}

func assertTensorI32(t *testing.T, tensor *backend.Tensor, wantShape []int, want []int32) {
	t.Helper()
	if tensor == nil {
		t.Fatal("tensor is nil")
	}
	if len(tensor.Shape) != len(wantShape) {
		t.Fatalf("tensor rank = %d, want %d", len(tensor.Shape), len(wantShape))
	}
	for i := range wantShape {
		if tensor.Shape[i] != wantShape[i] {
			t.Fatalf("tensor shape[%d] = %d, want %d", i, tensor.Shape[i], wantShape[i])
		}
	}
	if len(tensor.I32) != len(want) {
		t.Fatalf("tensor values len = %d, want %d", len(tensor.I32), len(want))
	}
	for i, got := range tensor.I32 {
		if got != want[i] {
			t.Fatalf("tensor[%d] = %d, want %d", i, got, want[i])
		}
	}
}

func assertTensorI64(t *testing.T, tensor *backend.Tensor, wantShape []int, want []int64) {
	t.Helper()
	if tensor == nil {
		t.Fatal("tensor is nil")
	}
	if len(tensor.Shape) != len(wantShape) {
		t.Fatalf("tensor rank = %d, want %d", len(tensor.Shape), len(wantShape))
	}
	for i := range wantShape {
		if tensor.Shape[i] != wantShape[i] {
			t.Fatalf("tensor shape[%d] = %d, want %d", i, tensor.Shape[i], wantShape[i])
		}
	}
	if len(tensor.I64) != len(want) {
		t.Fatalf("tensor values len = %d, want %d", len(tensor.I64), len(want))
	}
	for i, got := range tensor.I64 {
		if got != want[i] {
			t.Fatalf("tensor[%d] = %d, want %d", i, got, want[i])
		}
	}
}

func assertClose(t *testing.T, got, want, tol float32) {
	t.Helper()
	diff := got - want
	if diff < -tol || diff > tol {
		t.Fatalf("value = %f, want %f (tol=%f)", got, want, tol)
	}
}
