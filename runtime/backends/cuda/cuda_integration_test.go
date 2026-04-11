//go:build linux && cgo

package cuda

import (
	"context"
	"math"
	"testing"

	"github.com/odvcencio/manta/artifact/barr"
	"github.com/odvcencio/manta/compiler"
	barruntime "github.com/odvcencio/manta/runtime"
	"github.com/odvcencio/manta/runtime/backend"
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

func TestCUDAMultiBoundRightMatMulUploadsLHSOnce(t *testing.T) {
	accelAny, err := NewMatMulAccelerator()
	if err != nil {
		t.Fatalf("new matmul accelerator: %v", err)
	}
	if accelAny == nil {
		t.Skip("no cuda matmul accelerator available")
	}
	defer accelAny.Close()
	multi, ok := accelAny.(backend.MultiBoundRightMatMulAccelerator)
	if !ok {
		t.Fatal("cuda matmul accelerator does not implement multi-bound-right matmul")
	}

	lhs := backend.NewTensorF32([]int{2, 2}, []float32{
		1.0, -0.5,
		0.25, 0.75,
	})
	rhsA := &backend.Tensor{
		DType: "q8",
		Shape: []int{2, 2},
		F32: []float32{
			0.9, -0.35,
			0.2, 0.7,
		},
	}
	rhsB := &backend.Tensor{
		DType: "q8",
		Shape: []int{2, 2},
		F32: []float32{
			0.1, 1.2,
			-0.8, 0.4,
		},
	}
	if err := accelAny.BindMatrix("rhs_a", rhsA); err != nil {
		t.Fatalf("bind rhs_a: %v", err)
	}
	if err := accelAny.BindMatrix("rhs_b", rhsB); err != nil {
		t.Fatalf("bind rhs_b: %v", err)
	}
	results, err := multi.RunMatMulWithBoundRights(lhs, []string{"rhs_a", "rhs_b"}, barr.ValueType{
		Kind: barr.ValueTensor,
		Tensor: &barr.TensorType{
			DType: "f32",
		},
	}, false, false)
	if err != nil {
		t.Fatalf("run multi bound-right matmul: %v", err)
	}
	if len(results) != 2 {
		t.Fatalf("result count = %d, want 2", len(results))
	}

	for i, rhs := range []*backend.Tensor{rhsA, rhsB} {
		if len(results[i].Outputs) != 1 || results[i].Outputs[0] == nil {
			t.Fatalf("result %d output count = %d, want 1", i, len(results[i].Outputs))
		}
		qrhs := fakeQuantizeCopy(rhs.F32, 8)
		want := make([]float32, 4)
		fillHostMatMul(lhs.F32, 2, 2, qrhs, 2, want)
		assertTensorClose(t, results[i].Outputs[0], []int{2, 2}, want)
		if results[i].Metadata["coalesced_lhs"] != true {
			t.Fatalf("result %d coalesced_lhs = %v, want true", i, results[i].Metadata["coalesced_lhs"])
		}
	}
	stats := accelAny.Stats()
	if stats.RunUploadedBytes != int64(len(lhs.F32)*4) {
		t.Fatalf("run uploaded bytes = %d, want one lhs upload %d", stats.RunUploadedBytes, len(lhs.F32)*4)
	}
	if stats.RunCalls != 2 || stats.BoundRightCalls != 2 {
		t.Fatalf("run calls=%d bound-right calls=%d, want 2/2", stats.RunCalls, stats.BoundRightCalls)
	}
}

func TestCUDAAccumulatedBoundRightMatMulDownloadsOnce(t *testing.T) {
	accelAny, err := NewMatMulAccelerator()
	if err != nil {
		t.Fatalf("new matmul accelerator: %v", err)
	}
	if accelAny == nil {
		t.Skip("no cuda matmul accelerator available")
	}
	defer accelAny.Close()
	accumulated, ok := accelAny.(backend.AccumulatedBoundRightMatMulAccelerator)
	if !ok {
		t.Fatal("cuda matmul accelerator does not implement accumulated bound-right matmul")
	}

	lhsA := backend.NewTensorF32([]int{2, 2}, []float32{
		1.0, -0.5,
		0.25, 0.75,
	})
	lhsB := backend.NewTensorF32([]int{2, 2}, []float32{
		-1.0, 0.5,
		2.0, -0.25,
	})
	rhsA := &backend.Tensor{
		DType: "q8",
		Shape: []int{2, 2},
		F32: []float32{
			0.9, -0.35,
			0.2, 0.7,
		},
	}
	rhsB := &backend.Tensor{
		DType: "q8",
		Shape: []int{2, 2},
		F32: []float32{
			0.1, 1.2,
			-0.8, 0.4,
		},
	}
	if err := accelAny.BindMatrix("rhs_a", rhsA); err != nil {
		t.Fatalf("bind rhs_a: %v", err)
	}
	if err := accelAny.BindMatrix("rhs_b", rhsB); err != nil {
		t.Fatalf("bind rhs_b: %v", err)
	}
	result, err := accumulated.RunAccumulatedMatMulsWithBoundRights([]*backend.Tensor{lhsA, lhsB}, []string{"rhs_a", "rhs_b"}, barr.ValueType{
		Kind: barr.ValueTensor,
		Tensor: &barr.TensorType{
			DType: "f32",
		},
	}, false, true)
	if err != nil {
		t.Fatalf("run accumulated bound-right matmul: %v", err)
	}
	if len(result.Outputs) != 1 || result.Outputs[0] == nil {
		t.Fatalf("output count = %d, want 1", len(result.Outputs))
	}

	transpose2 := func(in []float32) []float32 {
		return []float32{
			in[0], in[2],
			in[1], in[3],
		}
	}
	want := make([]float32, 4)
	step := make([]float32, 4)
	fillHostMatMul(lhsA.F32, 2, 2, transpose2(fakeQuantizeCopy(rhsA.F32, 8)), 2, step)
	for i := range want {
		want[i] += step[i]
		step[i] = 0
	}
	fillHostMatMul(lhsB.F32, 2, 2, transpose2(fakeQuantizeCopy(rhsB.F32, 8)), 2, step)
	for i := range want {
		want[i] += step[i]
	}
	assertTensorClose(t, result.Outputs[0], []int{2, 2}, want)
	if result.Metadata["accumulated_bound_rights"] != true {
		t.Fatalf("accumulated_bound_rights = %v, want true", result.Metadata["accumulated_bound_rights"])
	}
	if result.Metadata["accumulated_download_once"] != true {
		t.Fatalf("accumulated_download_once = %v, want true", result.Metadata["accumulated_download_once"])
	}
	stats := accelAny.Stats()
	wantUpload := int64((len(lhsA.F32) + len(lhsB.F32)) * 4)
	if stats.RunUploadedBytes != wantUpload {
		t.Fatalf("run uploaded bytes = %d, want lhs uploads %d", stats.RunUploadedBytes, wantUpload)
	}
	if stats.RunDownloadedBytes != int64(len(want)*4) {
		t.Fatalf("run downloaded bytes = %d, want one output download %d", stats.RunDownloadedBytes, len(want)*4)
	}
	if stats.RunCalls != 1 || stats.BoundRightCalls != 1 {
		t.Fatalf("run calls=%d bound-right calls=%d, want 1/1 logical accumulated dispatch", stats.RunCalls, stats.BoundRightCalls)
	}
}

func TestCUDASharedLeftMatMulUploadsLHSOnce(t *testing.T) {
	accelAny, err := NewMatMulAccelerator()
	if err != nil {
		t.Fatalf("new matmul accelerator: %v", err)
	}
	if accelAny == nil {
		t.Skip("no cuda matmul accelerator available")
	}
	defer accelAny.Close()
	shared, ok := accelAny.(backend.SharedLeftMatMulAccelerator)
	if !ok {
		t.Fatal("cuda matmul accelerator does not implement shared-left matmul")
	}

	lhs := backend.NewTensorF32([]int{3, 2}, []float32{
		1, 2,
		3, 4,
		5, 6,
	})
	rhsA := backend.NewTensorF32([]int{3, 2}, []float32{
		1, 0,
		0, 1,
		1, 1,
	})
	rhsB := backend.NewTensorF32([]int{3, 2}, []float32{
		2, 1,
		1, 0,
		0, 2,
	})
	results, err := shared.RunMatMulsWithSharedLeft(lhs, []*backend.Tensor{rhsA, rhsB}, barr.ValueType{
		Kind: barr.ValueTensor,
		Tensor: &barr.TensorType{
			DType: "f32",
		},
	}, true, false)
	if err != nil {
		t.Fatalf("run shared-left matmul: %v", err)
	}
	if len(results) != 2 {
		t.Fatalf("result count = %d, want 2", len(results))
	}
	lhsT := []float32{
		1, 3, 5,
		2, 4, 6,
	}
	for i, rhs := range []*backend.Tensor{rhsA, rhsB} {
		if len(results[i].Outputs) != 1 || results[i].Outputs[0] == nil {
			t.Fatalf("result %d output count = %d, want 1", i, len(results[i].Outputs))
		}
		want := make([]float32, 4)
		fillHostMatMul(lhsT, 2, 3, rhs.F32, 2, want)
		assertTensorClose(t, results[i].Outputs[0], []int{2, 2}, want)
		if results[i].Metadata["shared_lhs"] != true {
			t.Fatalf("result %d shared_lhs = %v, want true", i, results[i].Metadata["shared_lhs"])
		}
	}
	stats := accelAny.Stats()
	wantUpload := int64((len(lhs.F32) + len(rhsA.F32) + len(rhsB.F32)) * 4)
	if stats.RunUploadedBytes != wantUpload {
		t.Fatalf("run uploaded bytes = %d, want one lhs upload plus rhs uploads %d", stats.RunUploadedBytes, wantUpload)
	}
	if stats.RunCalls != 2 {
		t.Fatalf("run calls = %d, want 2", stats.RunCalls)
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

func TestCUDAStridedBatchedMatMulTransposeMatchesHost(t *testing.T) {
	accelAny, err := NewMatMulAccelerator()
	if err != nil {
		t.Fatalf("new matmul accelerator: %v", err)
	}
	if accelAny == nil {
		t.Skip("no cuda matmul accelerator available")
	}
	defer accelAny.Close()

	outputType := barr.ValueType{
		Kind: barr.ValueTensor,
		Tensor: &barr.TensorType{
			DType: "f32",
		},
	}
	t.Run("transpose right", func(t *testing.T) {
		lhs := backend.NewTensorF32([]int{2, 2, 3}, []float32{
			1, 2, 3,
			4, 5, 6,
			7, 8, 9,
			10, 11, 12,
		})
		rhs := backend.NewTensorF32([]int{2, 2, 3}, []float32{
			1, 0, 1,
			0, 1, 1,
			2, 0, 1,
			0, 2, -1,
		})
		result, err := accelAny.RunMatMulWithTranspose([]*backend.Tensor{lhs, rhs}, outputType, false, true)
		if err != nil {
			t.Fatalf("run transposed strided batched matmul: %v", err)
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
	})
	t.Run("transpose left", func(t *testing.T) {
		lhs := backend.NewTensorF32([]int{2, 3, 2}, []float32{
			1, 4,
			2, 5,
			3, 6,
			7, 10,
			8, 11,
			9, 12,
		})
		rhs := backend.NewTensorF32([]int{2, 3, 2}, []float32{
			1, 0,
			0, 1,
			1, 1,
			2, 0,
			0, 2,
			1, -1,
		})
		result, err := accelAny.RunMatMulWithTranspose([]*backend.Tensor{lhs, rhs}, outputType, true, false)
		if err != nil {
			t.Fatalf("run transposed strided batched matmul: %v", err)
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
