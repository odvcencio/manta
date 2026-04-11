package barruntime

import (
	"context"
	"testing"

	"github.com/odvcencio/barracuda/compiler"
	"github.com/odvcencio/barracuda/runtime/backend"
	"github.com/odvcencio/barracuda/runtime/backends/cuda"
)

func TestRunMatMulStepBatchedUsesCUDADeviceDispatch(t *testing.T) {
	src := []byte(`
pipeline project_batch(x: f16[B, T, D], w: f16[D, E]) -> f16[B, T, E] {
    return @matmul(x, w)
}
`)

	bundle, err := compiler.Build(src, compiler.Options{ModuleName: "project_batch"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	rt := New(cuda.New())
	prog, err := rt.Load(context.Background(), bundle.Artifact)
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	result, err := prog.Run(context.Background(), backend.Request{
		Entry: "project_batch",
		Inputs: map[string]any{
			"x": backend.NewTensorF16([]int{2, 2, 2}, []float32{
				1, 2,
				3, 4,
				5, 6,
				7, 8,
			}),
			"w": backend.NewTensorF16([]int{2, 2}, []float32{
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
	if output.Producer != "matmul:matmul" {
		t.Fatalf("output producer = %q, want matmul:matmul", output.Producer)
	}
	if output.Metadata["dispatch_mode"] != "backend_native" {
		t.Fatalf("dispatch_mode = %v, want backend_native", output.Metadata["dispatch_mode"])
	}
	if output.Metadata["launch_api"] != "cublasSgemm" {
		t.Fatalf("launch_api = %v, want cublasSgemm", output.Metadata["launch_api"])
	}
	if output.Metadata["variant_entry"] != "cublas_sgemm" {
		t.Fatalf("variant_entry = %v, want cublas_sgemm", output.Metadata["variant_entry"])
	}
	tensor, ok := output.Data.(*backend.Tensor)
	if !ok {
		t.Fatalf("output data type = %T, want *backend.Tensor", output.Data)
	}
	assertTensorClose(t, tensor, []int{2, 2, 2}, []float32{
		1, 2,
		3, 4,
		5, 6,
		7, 8,
	})
	if len(result.Trace) != 2 {
		t.Fatalf("trace len = %d, want 2", len(result.Trace))
	}
	if result.Trace[0].Kind != "matmul" {
		t.Fatalf("trace[0].kind = %q, want matmul", result.Trace[0].Kind)
	}
	if result.Trace[0].Variant != "cublas_sgemm" {
		t.Fatalf("trace[0].variant = %q, want cublas_sgemm", result.Trace[0].Variant)
	}
}
