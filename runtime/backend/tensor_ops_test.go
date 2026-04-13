package backend

import (
	"context"
	"math"
	"testing"

	mantaartifact "github.com/odvcencio/manta/artifact/manta"
)

func TestMaterializeValueTensor(t *testing.T) {
	value, err := MaterializeValue(
		mantaartifact.ValueType{Kind: mantaartifact.ValueTensor, Tensor: &mantaartifact.TensorType{DType: "f16", Shape: []string{"T", "D"}}},
		NewTensorF16([]int{2, 2}, []float32{1, 2, 3, 4}),
	)
	if err != nil {
		t.Fatalf("materialize: %v", err)
	}
	tensor, ok := value.(*Tensor)
	if !ok {
		t.Fatalf("value type = %T, want *Tensor", value)
	}
	if tensor.DType != "f16" || tensor.Shape[0] != 2 || tensor.Shape[1] != 2 {
		t.Fatalf("unexpected tensor: %+v", tensor)
	}
}

func TestMaterializeValueTensorI64(t *testing.T) {
	value, err := MaterializeValue(
		mantaartifact.ValueType{Kind: mantaartifact.ValueTensor, Tensor: &mantaartifact.TensorType{DType: "i64", Shape: []string{"T"}}},
		NewTensorI64([]int{3}, []int64{101, 202, 303}),
	)
	if err != nil {
		t.Fatalf("materialize i64: %v", err)
	}
	tensor, ok := value.(*Tensor)
	if !ok {
		t.Fatalf("value type = %T, want *Tensor", value)
	}
	if tensor.DType != "i64" || len(tensor.I64) != 3 {
		t.Fatalf("unexpected i64 tensor: %+v", tensor)
	}
}

func TestMaterializeValueCandidatePack(t *testing.T) {
	value, err := MaterializeValue(
		mantaartifact.ValueType{Kind: mantaartifact.ValueCandidatePack, CandidatePack: &mantaartifact.CandidatePackType{Shape: []string{"K", "D"}}},
		NewCandidatePack(
			NewTensorI64([]int{2}, []int64{1001, 3003}),
			NewTensorF32([]int{2}, []float32{1, 0.70710677}),
			NewTensorQ4([]int{2, 2}, []float32{1, 0, 1, 1}),
		),
	)
	if err != nil {
		t.Fatalf("materialize candidate pack: %v", err)
	}
	pack, ok := value.(*CandidatePack)
	if !ok {
		t.Fatalf("value type = %T, want *CandidatePack", value)
	}
	assertTensorI64(t, pack.IDs, []int{2}, []int64{1001, 3003})
	assertTensorClose(t, pack.Scores, []int{2}, []float32{1, 0.70710677})
	assertTensorClose(t, pack.Docs, []int{2, 2}, []float32{1, 0, 1, 1})
}

func TestMaterializeValueWithBindingsBindsSymbols(t *testing.T) {
	bindings := map[string]int{}
	_, concrete, err := MaterializeValueWithBindings(
		mantaartifact.ValueType{Kind: mantaartifact.ValueTensor, Tensor: &mantaartifact.TensorType{DType: "f16", Shape: []string{"T", "D"}}},
		NewTensorF16([]int{2, 4}, []float32{1, 2, 3, 4, 5, 6, 7, 8}),
		bindings,
	)
	if err != nil {
		t.Fatalf("materialize with bindings: %v", err)
	}
	if bindings["T"] != 2 || bindings["D"] != 4 {
		t.Fatalf("unexpected bindings: %+v", bindings)
	}
	if got := concrete.Tensor.Shape[0] + "," + concrete.Tensor.Shape[1]; got != "2,4" {
		t.Fatalf("unexpected concrete shape: %v", concrete.Tensor.Shape)
	}
}

func TestMaterializeValueWithBindingsRejectsMismatch(t *testing.T) {
	bindings := map[string]int{"D": 2}
	_, _, err := MaterializeValueWithBindings(
		mantaartifact.ValueType{Kind: mantaartifact.ValueTensor, Tensor: &mantaartifact.TensorType{DType: "f16", Shape: []string{"T", "D"}}},
		NewTensorF16([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6}),
		bindings,
	)
	if err == nil {
		t.Fatal("expected mismatch error")
	}
}

func TestPreviewValueWithBindingsReusesTensorPointer(t *testing.T) {
	bindings := map[string]int{}
	input := NewTensorF16([]int{2, 2}, []float32{1, 2, 3, 4})
	value, concrete, err := PreviewValueWithBindings(
		mantaartifact.ValueType{Kind: mantaartifact.ValueTensor, Tensor: &mantaartifact.TensorType{DType: "f16", Shape: []string{"T", "D"}}},
		input,
		bindings,
	)
	if err != nil {
		t.Fatalf("preview with bindings: %v", err)
	}
	tensor, ok := value.(*Tensor)
	if !ok {
		t.Fatalf("preview type = %T, want *Tensor", value)
	}
	if tensor != input {
		t.Fatal("preview cloned tensor pointer")
	}
	if bindings["T"] != 2 || bindings["D"] != 2 {
		t.Fatalf("unexpected bindings after preview: %+v", bindings)
	}
	if got := concrete.Tensor.Shape[0] + "," + concrete.Tensor.Shape[1]; got != "2,2" {
		t.Fatalf("unexpected concrete shape: %v", concrete.Tensor.Shape)
	}
}

func TestGatherTensor(t *testing.T) {
	table := NewTensorF16([]int{3, 2}, []float32{
		1, 0,
		0, 1,
		1, 1,
	})
	indices := NewTensorI32([]int{2}, []int32{2, 0})
	out, err := gatherTensor(table, indices)
	if err != nil {
		t.Fatalf("gather: %v", err)
	}
	assertTensorClose(t, out, []int{2, 2}, []float32{
		1, 1,
		1, 0,
	})
}

func TestMatmulTensor(t *testing.T) {
	lhs := NewTensorF16([]int{2, 2}, []float32{
		1, 2,
		3, 4,
	})
	rhs := NewTensorF16([]int{2, 2}, []float32{
		5, 6,
		7, 8,
	})
	out, err := matmulTensor(lhs, rhs)
	if err != nil {
		t.Fatalf("matmul: %v", err)
	}
	assertTensorClose(t, out, []int{2, 2}, []float32{
		19, 22,
		43, 50,
	})
}

func TestMatmulTensorBatched(t *testing.T) {
	lhs := NewTensorF16([]int{2, 2, 2}, []float32{
		1, 2,
		3, 4,
		5, 6,
		7, 8,
	})
	rhs := NewTensorF16([]int{2, 2}, []float32{
		1, 0,
		0, 1,
	})
	out, err := matmulTensor(lhs, rhs)
	if err != nil {
		t.Fatalf("batched matmul: %v", err)
	}
	assertTensorClose(t, out, []int{2, 2, 2}, []float32{
		1, 2,
		3, 4,
		5, 6,
		7, 8,
	})
}

func TestMatmulTensorBatchedRHS(t *testing.T) {
	lhs := NewTensorF16([]int{2, 2, 2}, []float32{
		1, 0,
		0, 1,
		1, 2,
		3, 4,
	})
	rhs := NewTensorF16([]int{2, 2, 2}, []float32{
		1, 2,
		3, 4,
		2, 0,
		1, 2,
	})
	out, err := matmulTensor(lhs, rhs)
	if err != nil {
		t.Fatalf("batched rhs matmul: %v", err)
	}
	assertTensorClose(t, out, []int{2, 2, 2}, []float32{
		1, 2,
		3, 4,
		4, 4,
		10, 8,
	})
}

func TestExecuteSymbolicDispatchesImageStep(t *testing.T) {
	mod := mantaartifact.NewModule("image_dispatch")
	mod.EntryPoints = []mantaartifact.EntryPoint{{
		Name: "conv",
		Kind: mantaartifact.EntryPointPipeline,
		Inputs: []mantaartifact.ValueBinding{
			{Name: "x", Type: tensorValueType("f16", []string{"1", "1", "2", "2"})},
			{Name: "w", Type: tensorValueType("f16", []string{"1", "1", "2", "2"})},
			{Name: "b", Type: tensorValueType("f16", []string{"1"})},
		},
		Outputs: []mantaartifact.ValueBinding{
			{Name: "y", Type: tensorValueType("f16", []string{"1", "1", "1", "1"})},
		},
	}}
	mod.Buffers = []mantaartifact.Buffer{{Name: "y", DType: "f16", Shape: []string{"1", "1", "1", "1"}}}
	mod.Steps = []mantaartifact.Step{
		{Entry: "conv", Kind: mantaartifact.StepConv2D, Name: "conv2d", Inputs: []string{"x", "w", "b"}, Outputs: []string{"y"}},
		{Entry: "conv", Kind: mantaartifact.StepReturn, Name: "return", Outputs: []string{"y"}},
	}
	if err := mod.Validate(); err != nil {
		t.Fatal(err)
	}
	dispatch := func(_ context.Context, step mantaartifact.Step, outputType mantaartifact.ValueType, inputs []*Tensor) (StepDispatchResult, bool, error) {
		if step.Kind != mantaartifact.StepConv2D {
			return StepDispatchResult{}, false, nil
		}
		if len(inputs) != 3 {
			t.Fatalf("dispatch inputs = %d want 3", len(inputs))
		}
		if outputType.Tensor == nil || outputType.Tensor.DType != "f16" {
			t.Fatalf("unexpected output type: %+v", outputType)
		}
		return StepDispatchResult{
			Outputs:      []*Tensor{NewTensorF16([]int{1, 1, 1, 1}, []float32{42})},
			VariantEntry: "cuda_conv2d_test",
			Metadata:     map[string]any{"dispatch_mode": "test_device"},
		}, true, nil
	}
	result, err := ExecuteSymbolic(context.Background(), mod, nil, nil, nil, dispatch, mantaartifact.BackendCUDA, Request{
		Entry: "conv",
		Inputs: map[string]any{
			"x": NewTensorF16([]int{1, 1, 2, 2}, []float32{1, 2, 3, 4}),
			"w": NewTensorF16([]int{1, 1, 2, 2}, []float32{1, 0, 0, 1}),
			"b": NewTensorF16([]int{1}, []float32{0}),
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	out := result.Outputs["y"].Data.(*Tensor)
	if out.F32[0] != 42 {
		t.Fatalf("dispatched output = %v", out.F32)
	}
	if got := result.Outputs["y"].Metadata["dispatch_mode"]; got != "test_device" {
		t.Fatalf("dispatch metadata = %v", got)
	}
	if len(result.Trace) == 0 || result.Trace[0].Variant != "cuda_conv2d_test" {
		t.Fatalf("trace variant not recorded: %+v", result.Trace)
	}
}

func TestExecuteSymbolicDispatchesMultiOutputImageStep(t *testing.T) {
	mod := mantaartifact.NewModule("turboquant_dispatch")
	mod.EntryPoints = []mantaartifact.EntryPoint{{
		Name: "quantize",
		Kind: mantaartifact.EntryPointPipeline,
		Inputs: []mantaartifact.ValueBinding{
			{Name: "y", Type: tensorValueType("f16", []string{"1", "2", "1", "1"})},
		},
		Outputs: []mantaartifact.ValueBinding{
			{Name: "coords", Type: tensorValueType("q2", []string{"1", "2", "1", "1"})},
			{Name: "norms", Type: tensorValueType("q_norm", []string{"1", "1", "1"})},
		},
	}}
	mod.Buffers = []mantaartifact.Buffer{
		{Name: "coords", DType: "q2", Shape: []string{"1", "2", "1", "1"}},
		{Name: "norms", DType: "q_norm", Shape: []string{"1", "1", "1"}},
	}
	mod.Steps = []mantaartifact.Step{
		{Entry: "quantize", Kind: mantaartifact.StepTurboQEncode, Name: "quantize", Inputs: []string{"y"}, Outputs: []string{"coords", "norms"}, Attributes: map[string]string{"bits": "2"}},
		{Entry: "quantize", Kind: mantaartifact.StepReturn, Name: "return", Outputs: []string{"coords", "norms"}},
	}
	if err := mod.Validate(); err != nil {
		t.Fatal(err)
	}
	dispatch := func(_ context.Context, step mantaartifact.Step, _ mantaartifact.ValueType, inputs []*Tensor) (StepDispatchResult, bool, error) {
		if step.Kind != mantaartifact.StepTurboQEncode {
			return StepDispatchResult{}, false, nil
		}
		if len(inputs) != 1 {
			t.Fatalf("dispatch inputs = %d want 1", len(inputs))
		}
		return StepDispatchResult{
			Outputs: []*Tensor{
				NewTensorQ2([]int{1, 2, 1, 1}, []float32{1, 2}),
				NewTensorQNorm([]int{1, 1, 1}, []float32{128}),
			},
			VariantEntry: "cuda_turboquant_encode_test",
			Metadata:     map[string]any{"dispatch_mode": "test_device"},
		}, true, nil
	}
	result, err := ExecuteSymbolic(context.Background(), mod, nil, nil, nil, dispatch, mantaartifact.BackendCUDA, Request{
		Entry:  "quantize",
		Inputs: map[string]any{"y": NewTensorF16([]int{1, 2, 1, 1}, []float32{1, 2})},
	})
	if err != nil {
		t.Fatal(err)
	}
	if got := result.Outputs["coords"].Data.(*Tensor).F32; got[0] != 1 || got[1] != 2 {
		t.Fatalf("coords = %v", got)
	}
	if got := result.Outputs["norms"].Data.(*Tensor).F32[0]; got != 128 {
		t.Fatalf("norm = %v", got)
	}
}

func TestTransposeTensor(t *testing.T) {
	in := NewTensorF16([]int{2, 3}, []float32{
		1, 2, 3,
		4, 5, 6,
	})
	out, err := transposeTensor(in)
	if err != nil {
		t.Fatalf("transpose: %v", err)
	}
	assertTensorClose(t, out, []int{3, 2}, []float32{
		1, 4,
		2, 5,
		3, 6,
	})
}

func TestTransposeTensorBatched(t *testing.T) {
	in := NewTensorF16([]int{2, 2, 3}, []float32{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
		10, 11, 12,
	})
	out, err := transposeTensor(in)
	if err != nil {
		t.Fatalf("batched transpose: %v", err)
	}
	assertTensorClose(t, out, []int{2, 3, 2}, []float32{
		1, 4,
		2, 5,
		3, 6,
		7, 10,
		8, 11,
		9, 12,
	})
}

func TestNormalizeRows(t *testing.T) {
	in := NewTensorF16([]int{2, 2}, []float32{
		3, 4,
		1, 0,
	})
	out := normalizeRows(in)
	assertTensorClose(t, out, []int{2, 2}, []float32{
		0.6, 0.8,
		1, 0,
	})
}

func TestNormalizeRowsBatched(t *testing.T) {
	in := NewTensorF16([]int{2, 2, 2}, []float32{
		3, 4,
		1, 0,
		0, 5,
		1, 1,
	})
	out := normalizeRows(in)
	assertTensorClose(t, out, []int{2, 2, 2}, []float32{
		0.6, 0.8,
		1, 0,
		0, 1,
		0.70710677, 0.70710677,
	})
}

func TestLayerNormRows(t *testing.T) {
	in := NewTensorF16([]int{2, 2}, []float32{
		2, 4,
		6, 8,
	})
	out := layerNormRows(in)
	assertTensorClose(t, out, []int{2, 2}, []float32{
		-0.999995, 0.999995,
		-0.999995, 0.999995,
	})
}

func TestLayerNormRowsBatched(t *testing.T) {
	in := NewTensorF16([]int{2, 2, 2}, []float32{
		2, 4,
		6, 8,
		1, 3,
		5, 7,
	})
	out := layerNormRows(in)
	assertTensorClose(t, out, []int{2, 2, 2}, []float32{
		-0.999995, 0.999995,
		-0.999995, 0.999995,
		-0.999995, 0.999995,
		-0.999995, 0.999995,
	})
}

func TestSoftmaxRowsBatched(t *testing.T) {
	in := NewTensorF16([]int{2, 2, 2}, []float32{
		0, 1,
		1, 0,
		2, 0,
		0, 2,
	})
	out := softmaxRows(in)
	e := float32(math.Exp(1))
	e2 := float32(math.Exp(2))
	assertTensorClose(t, out, []int{2, 2, 2}, []float32{
		1 / (1 + e), e / (1 + e),
		e / (1 + e), 1 / (1 + e),
		e2 / (1 + e2), 1 / (1 + e2),
		1 / (1 + e2), e2 / (1 + e2),
	})
}

func TestMeanPoolTensor(t *testing.T) {
	in := NewTensorF16([]int{2, 2}, []float32{
		1, 0,
		0.70710677, 0.70710677,
	})
	out, err := meanPoolTensor(in)
	if err != nil {
		t.Fatalf("mean_pool: %v", err)
	}
	assertTensorClose(t, out, []int{2}, []float32{
		0.8535534, 0.35355338,
	})
}

func TestMeanPoolTensorBatched(t *testing.T) {
	in := NewTensorF16([]int{2, 2, 2}, []float32{
		1, 0,
		0.70710677, 0.70710677,
		0, 1,
		1, 0,
	})
	out, err := meanPoolTensor(in)
	if err != nil {
		t.Fatalf("batched mean_pool: %v", err)
	}
	assertTensorClose(t, out, []int{2, 2}, []float32{
		0.8535534, 0.35355338,
		0.5, 0.5,
	})
}

func TestMeanPoolMaskedTensor(t *testing.T) {
	in := NewTensorF16([]int{2, 2}, []float32{
		1, 0,
		0.70710677, 0.70710677,
	})
	mask := NewTensorI32([]int{2}, []int32{1, 0})
	out, err := meanPoolMaskedTensor(in, mask)
	if err != nil {
		t.Fatalf("masked mean_pool: %v", err)
	}
	assertTensorClose(t, out, []int{2}, []float32{
		1, 0,
	})
}

func TestMeanPoolMaskedTensorBatched(t *testing.T) {
	in := NewTensorF16([]int{2, 2, 2}, []float32{
		1, 0,
		0.70710677, 0.70710677,
		0, 1,
		1, 0,
	})
	mask := NewTensorI32([]int{2, 2}, []int32{
		1, 1,
		1, 0,
	})
	out, err := meanPoolMaskedTensor(in, mask)
	if err != nil {
		t.Fatalf("batched masked mean_pool: %v", err)
	}
	assertTensorClose(t, out, []int{2, 2}, []float32{
		0.8535534, 0.35355338,
		0, 1,
	})
}

func TestSoftmaxRows(t *testing.T) {
	in := NewTensorF16([]int{2, 2}, []float32{
		0, 0,
		1, 0,
	})
	out := softmaxRows(in)
	assertTensorClose(t, out, []int{2, 2}, []float32{
		0.5, 0.5,
		0.7310586, 0.26894143,
	})
}

func TestGELUTensor(t *testing.T) {
	in := NewTensorF16([]int{2, 2}, []float32{
		-1, 0,
		1, 2,
	})
	out := geluTensor(in)
	want := make([]float32, len(in.F32))
	for i, x := range in.F32 {
		want[i] = approxGELU(x)
	}
	assertTensorClose(t, out, []int{2, 2}, want)
}

func TestRoPERows(t *testing.T) {
	in := NewTensorF16([]int{2, 2}, []float32{
		1, 0,
		0, 1,
	})
	out := ropeRows(in)
	assertTensorClose(t, out, []int{2, 2}, []float32{
		1, 0,
		-0.84147096, 0.5403023,
	})
}

func TestDotRows(t *testing.T) {
	query := NewTensorF16([]int{2}, []float32{1, 0})
	docs := NewTensorF16([]int{3, 2}, []float32{
		1, 0,
		0, 1,
		1, 1,
	})
	out, err := dotRows(query, docs)
	if err != nil {
		t.Fatalf("dot: %v", err)
	}
	assertTensorClose(t, out, []int{3}, []float32{1, 0, 1})
}

func TestCosineRows(t *testing.T) {
	query := NewTensorF16([]int{2}, []float32{1, 0})
	docs := NewTensorF16([]int{3, 2}, []float32{
		1, 0,
		0, 1,
		1, 1,
	})
	out, err := cosineRows(query, docs)
	if err != nil {
		t.Fatalf("cosine: %v", err)
	}
	assertTensorClose(t, out, []int{3}, []float32{1, 0, 0.70710677})
}

func TestCosineRowsBatched(t *testing.T) {
	query := NewTensorF16([]int{2, 2}, []float32{
		1, 0,
		0, 1,
	})
	docs := NewTensorQ4([]int{2, 3, 2}, []float32{
		1, 0,
		0, 1,
		1, 1,
		0, 1,
		1, 0,
		1, 1,
	})
	out, err := cosineRows(query, docs)
	if err != nil {
		t.Fatalf("batched cosine: %v", err)
	}
	assertTensorClose(t, out, []int{2, 3}, []float32{
		1, 0, 0.70710677,
		1, 0, 0.70710677,
	})
}

func TestL2DistanceRows(t *testing.T) {
	query := NewTensorF16([]int{2}, []float32{1, 0})
	docs := NewTensorF16([]int{3, 2}, []float32{
		1, 0,
		0, 1,
		1, 1,
	})
	out, err := l2DistanceRows(query, docs)
	if err != nil {
		t.Fatalf("l2_distance: %v", err)
	}
	assertTensorClose(t, out, []int{3}, []float32{0, 1.4142135, 1})
}

func TestNewTensorQ4(t *testing.T) {
	tensor := NewTensorQ4([]int{2, 2}, []float32{1, 0, 0, 1})
	if tensor.DType != "q4" {
		t.Fatalf("dtype = %q, want q4", tensor.DType)
	}
	if len(tensor.F32) != 4 {
		t.Fatalf("len(F32) = %d, want 4", len(tensor.F32))
	}
}

func TestTopKTensor(t *testing.T) {
	in := NewTensorF32([]int{4}, []float32{0.25, 0.9, 0.5, 0.9})
	out, err := topKTensor(in, 3)
	if err != nil {
		t.Fatalf("topk: %v", err)
	}
	assertTensorI32(t, out, []int{3}, []int32{1, 3, 2})
}

func TestTopKTensorRank2(t *testing.T) {
	in := NewTensorF32([]int{2, 3}, []float32{
		0.25, 0.9, 0.5,
		0.9, 0.1, 0.9,
	})
	out, err := topKTensor(in, 2)
	if err != nil {
		t.Fatalf("topk rank2: %v", err)
	}
	assertTensorI32(t, out, []int{2, 2}, []int32{
		1, 2,
		0, 2,
	})
}

func TestGatherTensorRank1(t *testing.T) {
	table := NewTensorF32([]int{4}, []float32{0.25, 0.9, 0.5, 0.9})
	indices := NewTensorI32([]int{2}, []int32{3, 1})
	out, err := gatherTensor(table, indices)
	if err != nil {
		t.Fatalf("gather rank1: %v", err)
	}
	assertTensorClose(t, out, []int{2}, []float32{0.9, 0.9})
}

func TestGatherTensorRank1I32(t *testing.T) {
	table := NewTensorI32([]int{4}, []int32{101, 202, 303, 404})
	indices := NewTensorI32([]int{2}, []int32{2, 0})
	out, err := gatherTensor(table, indices)
	if err != nil {
		t.Fatalf("gather rank1 i32: %v", err)
	}
	assertTensorI32(t, out, []int{2}, []int32{303, 101})
}

func TestGatherTensorRank1I64(t *testing.T) {
	table := NewTensorI64([]int{4}, []int64{1001, 2002, 3003, 4004})
	indices := NewTensorI32([]int{2}, []int32{2, 0})
	out, err := gatherTensor(table, indices)
	if err != nil {
		t.Fatalf("gather rank1 i64: %v", err)
	}
	assertTensorI64(t, out, []int{2}, []int64{3003, 1001})
}

func TestGatherTensorRank2Q4(t *testing.T) {
	table := NewTensorQ4([]int{3, 2}, []float32{
		1, 0,
		0, 1,
		1, 1,
	})
	indices := NewTensorI32([]int{2}, []int32{2, 0})
	out, err := gatherTensor(table, indices)
	if err != nil {
		t.Fatalf("gather rank2 q4: %v", err)
	}
	if out.DType != "q4" {
		t.Fatalf("dtype = %q, want q4", out.DType)
	}
	assertTensorClose(t, out, []int{2, 2}, []float32{
		1, 1,
		1, 0,
	})
}

func TestGatherTensorRank2WithRank2Indices(t *testing.T) {
	table := NewTensorI64([]int{2, 3}, []int64{
		1001, 2002, 3003,
		4004, 5005, 6006,
	})
	indices := NewTensorI32([]int{2, 2}, []int32{
		0, 2,
		0, 2,
	})
	out, err := gatherTensor(table, indices)
	if err != nil {
		t.Fatalf("gather rank2/rank2: %v", err)
	}
	assertTensorI64(t, out, []int{2, 2}, []int64{
		1001, 3003,
		4004, 6006,
	})
}

func TestGatherTensorRank2SharedWithRank2Indices(t *testing.T) {
	table := NewTensorF16([]int{3, 2}, []float32{
		1, 0,
		0, 1,
		1, 1,
	})
	indices := NewTensorI32([]int{2, 2}, []int32{
		0, 2,
		1, 0,
	})
	out, err := gatherTensor(table, indices, mantaartifact.ValueType{
		Kind: mantaartifact.ValueTensor,
		Tensor: &mantaartifact.TensorType{
			DType: "f16",
			Shape: []string{"B", "T", "D"},
		},
	})
	if err != nil {
		t.Fatalf("gather shared rank2/rank2: %v", err)
	}
	assertTensorClose(t, out, []int{2, 2, 2}, []float32{
		1, 0,
		1, 1,
		0, 1,
		1, 0,
	})
}

func TestGatherTensorRank3Q4WithRank2Indices(t *testing.T) {
	table := NewTensorQ4([]int{2, 3, 2}, []float32{
		1, 0,
		0, 1,
		1, 1,
		0, 1,
		1, 0,
		1, 1,
	})
	indices := NewTensorI32([]int{2, 2}, []int32{
		0, 2,
		0, 2,
	})
	out, err := gatherTensor(table, indices)
	if err != nil {
		t.Fatalf("gather rank3/rank2 q4: %v", err)
	}
	if out.DType != "q4" {
		t.Fatalf("dtype = %q, want q4", out.DType)
	}
	assertTensorClose(t, out, []int{2, 2, 2}, []float32{
		1, 0,
		1, 1,
		0, 1,
		1, 1,
	})
}

func assertTensorClose(t *testing.T, tensor *Tensor, wantShape []int, want []float32) {
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
		diff := got - want[i]
		if diff < -0.0005 || diff > 0.0005 {
			t.Fatalf("tensor[%d] = %f, want %f", i, got, want[i])
		}
	}
}

func assertTensorI32(t *testing.T, tensor *Tensor, wantShape []int, want []int32) {
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

func assertTensorI64(t *testing.T, tensor *Tensor, wantShape []int, want []int64) {
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

func tensorValueType(dtype string, shape []string) mantaartifact.ValueType {
	return mantaartifact.ValueType{
		Kind:   mantaartifact.ValueTensor,
		Tensor: &mantaartifact.TensorType{DType: dtype, Shape: append([]string(nil), shape...)},
	}
}

func approxGELU(x float32) float32 {
	cubic := x * x * x
	inner := float32(0.7978845608) * (x + float32(0.044715)*cubic)
	return 0.5 * x * (1 + float32(math.Tanh(float64(inner))))
}
