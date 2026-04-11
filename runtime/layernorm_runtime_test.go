package barruntime

import (
	"context"
	"testing"

	"github.com/odvcencio/manta/compiler"
	"github.com/odvcencio/manta/runtime/backend"
	"github.com/odvcencio/manta/runtime/backends/cuda"
)

func TestRunResidualLayerNormUsesCUDADeviceDispatch(t *testing.T) {
	src := []byte(`
pipeline encoder_block(x: f16[T, D], w: f16[D, D]) -> f16[T, D] {
    let projected = @matmul(x, w)
    let residual = projected + x
    return layernorm(residual)
}
`)

	bundle, err := compiler.Build(src, compiler.Options{ModuleName: "tiny_layernorm_runtime"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	rt := New(cuda.New())
	prog, err := rt.Load(context.Background(), bundle.Artifact)
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	result, err := prog.Run(context.Background(), backend.Request{
		Entry: "encoder_block",
		Inputs: map[string]any{
			"x": backend.NewTensorF16([]int{2, 2}, []float32{
				1, 2,
				3, 4,
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
	if output.Producer != "kernel:layernorm" {
		t.Fatalf("output producer = %q, want kernel:layernorm", output.Producer)
	}
	if output.Metadata["dispatch_mode"] != "backend_native" {
		t.Fatalf("dispatch_mode = %v, want backend_native", output.Metadata["dispatch_mode"])
	}
	if output.Metadata["launch_api"] != "cuLaunchKernel" {
		t.Fatalf("launch_api = %v, want cuLaunchKernel", output.Metadata["launch_api"])
	}
	if output.Metadata["variant_entry"] != "layernorm_cuda" {
		t.Fatalf("variant_entry = %v, want layernorm_cuda", output.Metadata["variant_entry"])
	}
	tensor, ok := output.Data.(*backend.Tensor)
	if !ok {
		t.Fatalf("output data type = %T, want *backend.Tensor", output.Data)
	}
	assertTensorClose(t, tensor, []int{2, 2}, []float32{
		-0.999995, 0.999995,
		-0.999995, 0.999995,
	})
	if len(result.Trace) != 4 {
		t.Fatalf("trace len = %d, want 4", len(result.Trace))
	}
	if result.Trace[0].Kind != "matmul" || result.Trace[0].Variant != "cublas_sgemm" {
		t.Fatalf("trace[0] = %+v, want matmul via cublas_sgemm", result.Trace[0])
	}
	if result.Trace[1].Kind != "launch_kernel" || result.Trace[1].Kernel != "binary_add" {
		t.Fatalf("trace[1] = %+v, want binary_add kernel", result.Trace[1])
	}
	if result.Trace[2].Kind != "launch_kernel" || result.Trace[2].Variant != "layernorm_cuda" {
		t.Fatalf("trace[2] = %+v, want layernorm via layernorm_cuda", result.Trace[2])
	}
}
