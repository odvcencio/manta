package barruntime

import (
	"context"
	"math"
	"testing"

	"github.com/odvcencio/barracuda/compiler"
	"github.com/odvcencio/barracuda/runtime/backend"
	"github.com/odvcencio/barracuda/runtime/backends/cuda"
)

func TestRunGELUFFNUsesCUDADeviceDispatch(t *testing.T) {
	src := []byte(`
pipeline ffn(x: f16[T, D], w: f16[D, E]) -> f16[T, E] {
    let projected = @matmul(x, w)
    return gelu(projected)
}
`)

	bundle, err := compiler.Build(src, compiler.Options{ModuleName: "tiny_gelu"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	rt := New(cuda.New())
	prog, err := rt.Load(context.Background(), bundle.Artifact)
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	result, err := prog.Run(context.Background(), backend.Request{
		Entry: "ffn",
		Inputs: map[string]any{
			"x": backend.NewTensorF16([]int{2, 2}, []float32{
				-1, 0,
				1, 2,
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
	if output.Producer != "kernel:gelu" {
		t.Fatalf("output producer = %q, want kernel:gelu", output.Producer)
	}
	if output.Metadata["dispatch_mode"] != "backend_native" {
		t.Fatalf("dispatch_mode = %v, want backend_native", output.Metadata["dispatch_mode"])
	}
	if output.Metadata["launch_api"] != "cuLaunchKernel" {
		t.Fatalf("launch_api = %v, want cuLaunchKernel", output.Metadata["launch_api"])
	}
	if output.Metadata["variant_entry"] != "gelu_cuda" {
		t.Fatalf("variant_entry = %v, want gelu_cuda", output.Metadata["variant_entry"])
	}
	tensor, ok := output.Data.(*backend.Tensor)
	if !ok {
		t.Fatalf("output data type = %T, want *backend.Tensor", output.Data)
	}
	want := []float32{
		approxRuntimeGELU(-1), approxRuntimeGELU(0),
		approxRuntimeGELU(1), approxRuntimeGELU(2),
	}
	assertTensorClose(t, tensor, []int{2, 2}, want)
	if len(result.Trace) != 3 {
		t.Fatalf("trace len = %d, want 3", len(result.Trace))
	}
	if result.Trace[0].Kind != "matmul" || result.Trace[0].Variant != "cublas_sgemm" {
		t.Fatalf("trace[0] = %+v, want matmul via cublas_sgemm", result.Trace[0])
	}
	if result.Trace[1].Kind != "launch_kernel" || result.Trace[1].Variant != "gelu_cuda" {
		t.Fatalf("trace[1] = %+v, want gelu via gelu_cuda", result.Trace[1])
	}
	if result.Trace[2].Kind != "return" {
		t.Fatalf("trace[2] = %+v, want return step", result.Trace[2])
	}
}

func approxRuntimeGELU(x float32) float32 {
	cubic := x * x * x
	inner := float32(0.7978845608) * (x + float32(0.044715)*cubic)
	return 0.5 * x * (1 + float32(math.Tanh(float64(inner))))
}
