package mantaruntime

import (
	"context"
	"math"
	"testing"

	"github.com/odvcencio/manta/compiler"
	"github.com/odvcencio/manta/runtime/backend"
	"github.com/odvcencio/manta/runtime/backends/cuda"
)

func TestRunAttentionEmbedUsesCUDADeviceDispatch(t *testing.T) {
	src := []byte(`
pipeline embed_pooled(tokens: i32[T], attention_mask: i32[T], token_embedding: f16[V, D], wq: f16[D, D], wk: f16[D, D], wv: f16[D, D], wo: f16[D, D], projection: f16[D, E]) -> f16[E] {
    let hidden = gather(token_embedding, tokens)
    let q = @matmul(hidden, wq)
    let k = @matmul(hidden, wk)
    let v = @matmul(hidden, wv)
    let kt = transpose(k)
    let scores = @matmul(q, kt)
    let probs = softmax(scores)
    let mixed = @matmul(probs, v)
    let attended = @matmul(mixed, wo)
    let projected = @matmul(attended, projection)
    let normalized = normalize(projected)
    return mean_pool(normalized, attention_mask)
}
`)

	bundle, err := compiler.Build(src, compiler.Options{ModuleName: "tiny_attention_runtime"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	rt := New(cuda.New())
	prog, err := rt.Load(context.Background(), bundle.Artifact)
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	result, err := prog.Run(context.Background(), backend.Request{
		Entry: "embed_pooled",
		Inputs: map[string]any{
			"tokens":         backend.NewTensorI32([]int{2}, []int32{0, 1}),
			"attention_mask": backend.NewTensorI32([]int{2}, []int32{1, 1}),
			"token_embedding": backend.NewTensorF16([]int{2, 2}, []float32{
				1, 0,
				0, 1,
			}),
			"wq": backend.NewTensorF16([]int{2, 2}, []float32{
				1, 0,
				0, 1,
			}),
			"wk": backend.NewTensorF16([]int{2, 2}, []float32{
				1, 0,
				0, 1,
			}),
			"wv": backend.NewTensorF16([]int{2, 2}, []float32{
				1, 0,
				0, 1,
			}),
			"wo": backend.NewTensorF16([]int{2, 2}, []float32{
				1, 0,
				0, 1,
			}),
			"projection": backend.NewTensorF16([]int{2, 2}, []float32{
				1, 0,
				0, 1,
			}),
		},
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
	want := identityAttentionExpected()
	assertTensorClose(t, tensor, []int{2}, want)

	matmulSteps := 0
	transposeSteps := 0
	softmaxSteps := 0
	for _, step := range result.Trace {
		switch step.Kind {
		case "matmul":
			matmulSteps++
			if step.Variant != "cublas_sgemm" {
				t.Fatalf("matmul variant = %q, want cublas_sgemm", step.Variant)
			}
		case "transpose":
			transposeSteps++
		case "launch_kernel":
			if step.Kernel == "softmax" {
				softmaxSteps++
			}
		}
	}
	if matmulSteps != 7 {
		t.Fatalf("matmul step count = %d, want 7", matmulSteps)
	}
	if transposeSteps != 1 {
		t.Fatalf("transpose step count = %d, want 1", transposeSteps)
	}
	if softmaxSteps != 1 {
		t.Fatalf("softmax step count = %d, want 1", softmaxSteps)
	}
}

func TestRunAttentionEmbedBatchUsesCUDADeviceDispatch(t *testing.T) {
	src := []byte(`
pipeline embed_pooled_batch(tokens: i32[B, T], attention_mask: i32[B, T], token_embedding: f16[V, D], wq: f16[D, D], wk: f16[D, D], wv: f16[D, D], wo: f16[D, D], projection: f16[D, E]) -> f16[B, E] {
    let hidden = gather(token_embedding, tokens)
    let q = @matmul(hidden, wq)
    let k = @matmul(hidden, wk)
    let v = @matmul(hidden, wv)
    let kt = transpose(k)
    let scores = @matmul(q, kt)
    let probs = softmax(scores)
    let mixed = @matmul(probs, v)
    let attended = @matmul(mixed, wo)
    let projected = @matmul(attended, projection)
    let normalized = normalize(projected)
    return mean_pool(normalized, attention_mask)
}
`)

	bundle, err := compiler.Build(src, compiler.Options{ModuleName: "tiny_attention_runtime_batch"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	rt := New(cuda.New())
	prog, err := rt.Load(context.Background(), bundle.Artifact)
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	result, err := prog.Run(context.Background(), backend.Request{
		Entry: "embed_pooled_batch",
		Inputs: map[string]any{
			"tokens":         backend.NewTensorI32([]int{2, 2}, []int32{0, 1, 1, 0}),
			"attention_mask": backend.NewTensorI32([]int{2, 2}, []int32{1, 1, 1, 1}),
			"token_embedding": backend.NewTensorF16([]int{2, 2}, []float32{
				1, 0,
				0, 1,
			}),
			"wq": backend.NewTensorF16([]int{2, 2}, []float32{
				1, 0,
				0, 1,
			}),
			"wk": backend.NewTensorF16([]int{2, 2}, []float32{
				1, 0,
				0, 1,
			}),
			"wv": backend.NewTensorF16([]int{2, 2}, []float32{
				1, 0,
				0, 1,
			}),
			"wo": backend.NewTensorF16([]int{2, 2}, []float32{
				1, 0,
				0, 1,
			}),
			"projection": backend.NewTensorF16([]int{2, 2}, []float32{
				1, 0,
				0, 1,
			}),
		},
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
	want := identityAttentionExpected()
	assertTensorClose(t, tensor, []int{2, 2}, []float32{want[0], want[1], want[0], want[1]})

	matmulSteps := 0
	for _, step := range result.Trace {
		if step.Kind != "matmul" {
			continue
		}
		matmulSteps++
		if step.Variant != "cublas_sgemm" {
			t.Fatalf("batched attention matmul variant = %q, want cublas_sgemm", step.Variant)
		}
	}
	if matmulSteps != 7 {
		t.Fatalf("batched attention matmul count = %d, want 7", matmulSteps)
	}
}

func identityAttentionExpected() []float32 {
	e := float32(math.Exp(1))
	a := e / (1 + e)
	b := float32(1) / (1 + e)
	norm := float32(math.Sqrt(float64(a*a + b*b)))
	value := float32(0.5) / norm
	return []float32{value, value}
}
