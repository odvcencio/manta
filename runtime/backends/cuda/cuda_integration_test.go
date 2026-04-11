//go:build linux && cgo

package cuda

import (
	"context"
	"math"
	"testing"

	"github.com/odvcencio/barracuda/artifact/barr"
	"github.com/odvcencio/barracuda/compiler"
	barruntime "github.com/odvcencio/barracuda/runtime"
	"github.com/odvcencio/barracuda/runtime/backend"
)

func TestCUDADeviceExecutionTinyEmbed(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_embed", Preset: compiler.PresetTinyEmbed})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	rt := barruntime.New(New())
	prog, err := rt.Load(
		context.Background(),
		bundle.Artifact,
		barruntime.WithWeight("token_embedding", backend.NewTensorF16([]int{3, 2}, []float32{
			1, 0,
			0, 1,
			1, 1,
		})),
		barruntime.WithWeight("projection", backend.NewTensorF16([]int{2, 2}, []float32{
			1, 0,
			0, 1,
		})),
	)
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
	output := result.Outputs["embeddings"]
	if output.Metadata["device_execution"] != true {
		t.Fatalf("device_execution = %v, want true", output.Metadata["device_execution"])
	}
	if output.Metadata["launch_api"] != "cuLaunchKernel" {
		t.Fatalf("launch_api = %v, want cuLaunchKernel", output.Metadata["launch_api"])
	}
	tensor, ok := output.Data.(*backend.Tensor)
	if !ok {
		t.Fatalf("output type = %T, want *backend.Tensor", output.Data)
	}
	assertTensorClose(t, tensor, []int{2, 2}, []float32{
		1, 0,
		0.70710677, 0.70710677,
	})
}

func TestCUDADeviceExecutionTinyScore(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_score", Preset: compiler.PresetTinyScore})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	rt := barruntime.New(New())
	prog, err := rt.Load(
		context.Background(),
		bundle.Artifact,
		barruntime.WithWeight("docs", backend.NewTensorQ4([]int{2, 2}, []float32{
			1, 0,
			0, 1,
		})),
	)
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
	output := result.Outputs["scores"]
	if output.Metadata["device_execution"] != true {
		t.Fatalf("device_execution = %v, want true", output.Metadata["device_execution"])
	}
	if output.Metadata["launch_api"] != "cuLaunchKernel" {
		t.Fatalf("launch_api = %v, want cuLaunchKernel", output.Metadata["launch_api"])
	}
	if output.Metadata["variant_entry"] != "cosine_cuda" {
		t.Fatalf("variant_entry = %v, want cosine_cuda", output.Metadata["variant_entry"])
	}
	tensor, ok := output.Data.(*backend.Tensor)
	if !ok {
		t.Fatalf("output type = %T, want *backend.Tensor", output.Data)
	}
	assertTensorClose(t, tensor, []int{2}, []float32{1, 0})
}

func TestCUDABoundQuantizedMatrixMatchesHostFakeQuantization(t *testing.T) {
	accelAny, err := NewMatMulAccelerator()
	if err != nil {
		t.Fatalf("new matmul accelerator: %v", err)
	}
	if accelAny == nil {
		t.Skip("no cuda matmul accelerator available")
	}
	defer accelAny.Close()

	lhs := backend.NewTensorF32([]int{2, 2}, []float32{
		1.0, -0.5,
		0.25, 0.75,
	})
	rhs := &backend.Tensor{
		DType: "q8",
		Shape: []int{2, 2},
		F32: []float32{
			0.9, -0.35,
			0.2, 0.7,
		},
	}
	if err := accelAny.BindMatrix("quant_rhs", rhs); err != nil {
		t.Fatalf("bind quantized rhs: %v", err)
	}
	result, err := accelAny.RunMatMulWithBoundRight(lhs, "quant_rhs", barr.ValueType{
		Kind: barr.ValueTensor,
		Tensor: &barr.TensorType{
			DType: "f32",
		},
	}, false, false)
	if err != nil {
		t.Fatalf("run bound quantized matmul: %v", err)
	}
	if len(result.Outputs) != 1 || result.Outputs[0] == nil {
		t.Fatalf("output count = %d, want 1", len(result.Outputs))
	}

	qrhs := fakeQuantizeCopy(rhs.F32, 8)
	want := make([]float32, 4)
	fillHostMatMul(lhs.F32, 2, 2, qrhs, 2, want)
	assertTensorClose(t, result.Outputs[0], []int{2, 2}, want)

	stats := accelAny.Stats()
	if stats.BindCalls < 1 {
		t.Fatalf("bind calls = %d, want at least 1", stats.BindCalls)
	}
	if stats.UploadedBytes < 16 {
		t.Fatalf("uploaded bytes = %d, want at least 16", stats.UploadedBytes)
	}
	if stats.QuantizePasses < 1 {
		t.Fatalf("quantize passes = %d, want at least 1", stats.QuantizePasses)
	}
	if stats.QuantizedBytes < 16 {
		t.Fatalf("quantized bytes = %d, want at least 16", stats.QuantizedBytes)
	}
	if stats.BoundMatrices < 1 {
		t.Fatalf("bound matrices = %d, want at least 1", stats.BoundMatrices)
	}
}

func TestCUDAStridedBatchedMatMulMatchesHost(t *testing.T) {
	accelAny, err := NewMatMulAccelerator()
	if err != nil {
		t.Fatalf("new matmul accelerator: %v", err)
	}
	if accelAny == nil {
		t.Skip("no cuda matmul accelerator available")
	}
	defer accelAny.Close()

	lhs := backend.NewTensorF32([]int{2, 2, 3}, []float32{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
		10, 11, 12,
	})
	rhs := backend.NewTensorF32([]int{2, 3, 2}, []float32{
		1, 0,
		0, 1,
		1, 1,
		2, 0,
		0, 2,
		1, -1,
	})
	result, err := accelAny.RunMatMul([]*backend.Tensor{lhs, rhs}, barr.ValueType{
		Kind: barr.ValueTensor,
		Tensor: &barr.TensorType{
			DType: "f32",
		},
	})
	if err != nil {
		t.Fatalf("run strided batched matmul: %v", err)
	}
	if len(result.Outputs) != 1 || result.Outputs[0] == nil {
		t.Fatalf("output count = %d, want 1", len(result.Outputs))
	}
	if result.Metadata["launch_api"] != "cublasSgemmStridedBatched" {
		t.Fatalf("launch_api = %v, want cublasSgemmStridedBatched", result.Metadata["launch_api"])
	}
	assertTensorClose(t, result.Outputs[0], []int{2, 2, 2}, []float32{
		4, 5,
		10, 11,
		23, 7,
		32, 10,
	})
}

func assertTensorClose(t *testing.T, tensor *backend.Tensor, wantShape []int, want []float32) {
	t.Helper()
	if tensor == nil {
		t.Fatal("tensor is nil")
	}
	if len(tensor.Shape) != len(wantShape) {
		t.Fatalf("rank = %d, want %d", len(tensor.Shape), len(wantShape))
	}
	for i := range wantShape {
		if tensor.Shape[i] != wantShape[i] {
			t.Fatalf("shape[%d] = %d, want %d", i, tensor.Shape[i], wantShape[i])
		}
	}
	for i, got := range tensor.F32 {
		diff := got - want[i]
		if diff < -0.0005 || diff > 0.0005 {
			t.Fatalf("tensor[%d] = %f, want %f", i, got, want[i])
		}
	}
}

func fillHostMatMul(lhs []float32, rows, inner int, rhs []float32, cols int, out []float32) {
	for row := 0; row < rows; row++ {
		outBase := row * cols
		lhsBase := row * inner
		for col := 0; col < cols; col++ {
			sum := float32(0)
			for kk := 0; kk < inner; kk++ {
				sum += lhs[lhsBase+kk] * rhs[kk*cols+col]
			}
			out[outBase+col] = sum
		}
	}
}

func fakeQuantizeCopy(data []float32, bits int) []float32 {
	out := append([]float32(nil), data...)
	if bits <= 0 || len(out) == 0 {
		return out
	}
	maxAbs := float32(0)
	for _, v := range out {
		abs := float32(math.Abs(float64(v)))
		if abs > maxAbs {
			maxAbs = abs
		}
	}
	if maxAbs == 0 {
		return out
	}
	levelsInt := (1 << uint(bits-1)) - 1
	levels := float32(levelsInt)
	if levels <= 0 {
		return out
	}
	scale := maxAbs / levels
	if scale == 0 {
		return out
	}
	for i, v := range out {
		q := float32(math.Round(float64(v / scale)))
		if q > levels {
			q = levels
		}
		if q < -levels {
			q = -levels
		}
		out[i] = q * scale
	}
	return out
}
