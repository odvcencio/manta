package webgpu

import (
	"context"
	"strings"
	"testing"

	mantaartifact "github.com/odvcencio/manta/artifact/manta"
	mantaruntime "github.com/odvcencio/manta/runtime"
	"github.com/odvcencio/manta/runtime/backend"
)

func TestMirageDecodeBuiltinWGSLKernels(t *testing.T) {
	kernels := MirageDecodeKernels()
	if len(kernels) != 5 {
		t.Fatalf("kernel count = %d want 5", len(kernels))
	}
	seen := map[mantaartifact.StepKind]bool{}
	for _, kernel := range kernels {
		seen[kernel.StepKind] = true
		if kernel.Entry == "" {
			t.Fatalf("empty entry for %+v", kernel)
		}
		if !strings.Contains(kernel.Source, "@compute") {
			t.Fatalf("%s source missing @compute", kernel.Entry)
		}
		if !strings.Contains(kernel.Source, "fn "+kernel.Entry+"(") {
			t.Fatalf("%s source missing entry function", kernel.Entry)
		}
		if !strings.Contains(kernel.Source, "@group(0) @binding(0)") {
			t.Fatalf("%s source missing storage bindings", kernel.Entry)
		}
		if strings.Contains(strings.ToLower(kernel.Source), "todo") {
			t.Fatalf("%s source contains todo marker", kernel.Entry)
		}
		if kernel.SourceHash() == "" {
			t.Fatalf("%s source hash is empty", kernel.Entry)
		}
		meta := kernel.Metadata()
		if meta["launch_api"] != "GPUComputePassEncoder.dispatchWorkgroups" {
			t.Fatalf("%s launch_api = %v", kernel.Entry, meta["launch_api"])
		}
	}
	for _, kind := range []mantaartifact.StepKind{
		mantaartifact.StepConv2D,
		mantaartifact.StepConv2DTrans,
		mantaartifact.StepGDN,
		mantaartifact.StepIGDN,
		mantaartifact.StepTurboQDecode,
	} {
		if !seen[kind] {
			t.Fatalf("missing builtin for %s", kind)
		}
	}
}

func TestWebGPUDispatchesMirageBuiltinWithWGSLMetadata(t *testing.T) {
	mod := mantaartifact.NewModule("webgpu_conv")
	mod.Params = []mantaartifact.Param{
		{Name: "w", Binding: "weights/w", Type: webgpuTensorType("f16", []string{"1", "1", "2", "2"})},
		{Name: "b", Binding: "weights/b", Type: webgpuTensorType("f16", []string{"1"})},
	}
	mod.EntryPoints = []mantaartifact.EntryPoint{{
		Name: "conv",
		Kind: mantaartifact.EntryPointPipeline,
		Inputs: []mantaartifact.ValueBinding{
			{Name: "x", Type: webgpuTensorType("f16", []string{"1", "1", "2", "2"})},
		},
		Outputs: []mantaartifact.ValueBinding{{Name: "y", Type: webgpuTensorType("f16", []string{"1", "1", "1", "1"})}},
	}}
	mod.Buffers = []mantaartifact.Buffer{{Name: "y", DType: "f16", Shape: []string{"1", "1", "1", "1"}}}
	mod.Steps = []mantaartifact.Step{
		{Entry: "conv", Kind: mantaartifact.StepConv2D, Name: "conv", Inputs: []string{"x", "w", "b"}, Outputs: []string{"y"}},
		{Entry: "conv", Kind: mantaartifact.StepReturn, Name: "return", Outputs: []string{"y"}},
	}

	prog, err := mantaruntime.New(New()).Load(context.Background(), mod,
		mantaruntime.WithWeight("w", backend.NewTensorF16([]int{1, 1, 2, 2}, []float32{1, 0, 0, 1})),
		mantaruntime.WithWeight("b", backend.NewTensorF16([]int{1}, []float32{0.5})),
	)
	if err != nil {
		t.Fatal(err)
	}
	result, err := prog.Run(context.Background(), backend.Request{
		Entry:  "conv",
		Inputs: map[string]any{"x": backend.NewTensorF16([]int{1, 1, 2, 2}, []float32{1, 2, 3, 4})},
	})
	if err != nil {
		t.Fatal(err)
	}
	out := result.Outputs["y"]
	tensor := out.Data.(*backend.Tensor)
	if tensor.F32[0] != 5.5 {
		t.Fatalf("conv output = %v want 5.5", tensor.F32[0])
	}
	kernel, ok := BuiltinForStep(mantaartifact.StepConv2D)
	if !ok {
		t.Fatal("missing conv2d builtin")
	}
	if out.Metadata["variant_entry"] != kernel.Entry {
		t.Fatalf("variant_entry = %v want %s", out.Metadata["variant_entry"], kernel.Entry)
	}
	if out.Metadata["source_hash"] != kernel.SourceHash() {
		t.Fatalf("source_hash = %v want %s", out.Metadata["source_hash"], kernel.SourceHash())
	}
	if out.Metadata["execution_mode"] != "wgsl_host_reference" {
		t.Fatalf("execution_mode = %v", out.Metadata["execution_mode"])
	}
	if out.Metadata["device_execution"] != false {
		t.Fatalf("device_execution = %v", out.Metadata["device_execution"])
	}
	if out.Metadata["launch_api"] != "GPUComputePassEncoder.dispatchWorkgroups" {
		t.Fatalf("launch_api = %v", out.Metadata["launch_api"])
	}
	if out.Metadata["wgsl_source_hash"] != kernel.SourceHash() {
		t.Fatalf("wgsl_source_hash = %v want %s", out.Metadata["wgsl_source_hash"], kernel.SourceHash())
	}
}

func webgpuTensorType(dtype string, shape []string) mantaartifact.ValueType {
	return mantaartifact.ValueType{
		Kind:   mantaartifact.ValueTensor,
		Tensor: &mantaartifact.TensorType{DType: dtype, Shape: append([]string(nil), shape...)},
	}
}
