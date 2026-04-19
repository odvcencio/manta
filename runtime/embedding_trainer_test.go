package mantaruntime

import (
	"context"
	"math"
	"path/filepath"
	"strings"
	"testing"

	mantaartifact "github.com/odvcencio/manta/artifact/manta"
	"github.com/odvcencio/manta/compiler"
	"github.com/odvcencio/manta/runtime/backend"
	"github.com/odvcencio/manta/runtime/backends/cuda"
	"github.com/odvcencio/manta/runtime/backends/metal"
)

func TestFlattenFixedFloat32MatricesScratchReusesContiguousViews(t *testing.T) {
	trainer := &EmbeddingTrainer{}
	base := []float32{1, 2, 3, 4, 5, 6}
	matrices := [][]float32{
		base[0:2],
		base[2:4],
		base[4:6],
	}
	out, ok := trainer.flattenFixedFloat32MatricesScratch(0, matrices, 2)
	if !ok {
		t.Fatal("flatten contiguous matrices failed")
	}
	if len(out) != len(base) {
		t.Fatalf("flattened len = %d, want %d", len(out), len(base))
	}
	if &out[0] != &base[0] {
		t.Fatal("contiguous flatten copied instead of reusing the backing array")
	}
	if len(trainer.scratchF32) != 0 {
		t.Fatalf("scratch buffer count = %d, want 0", len(trainer.scratchF32))
	}
}

func TestFlattenFixedFloat32MatricesScratchCopiesNonContiguousViews(t *testing.T) {
	trainer := &EmbeddingTrainer{}
	left := []float32{1, 2}
	right := []float32{3, 4}
	out, ok := trainer.flattenFixedFloat32MatricesScratch(0, [][]float32{left, right}, 2)
	if !ok {
		t.Fatal("flatten non-contiguous matrices failed")
	}
	if len(out) != 4 {
		t.Fatalf("flattened len = %d, want 4", len(out))
	}
	if &out[0] == &left[0] {
		t.Fatal("non-contiguous flatten reused left backing array")
	}
	for i, want := range []float32{1, 2, 3, 4} {
		if out[i] != want {
			t.Fatalf("out[%d] = %v, want %v", i, out[i], want)
		}
	}
	if len(trainer.scratchF32) != 1 {
		t.Fatalf("scratch buffer count = %d, want 1", len(trainer.scratchF32))
	}
}

type countingMatMulAccelerator struct {
	bindCalls         int
	runCalls          int
	maxRunBatches     int
	boundRightRuns    int
	multiBoundRuns    int
	sharedLeftRuns    int
	accumulatedRuns   int
	maxBoundRightRows int
	maxSharedLeftRHS  int
	maxAccumTerms     int
	maxRunOutputCols  int
	bound             map[string]*backend.Tensor
}

func (a *countingMatMulAccelerator) Backend() mantaartifact.BackendKind {
	return mantaartifact.BackendCUDA
}
func (a *countingMatMulAccelerator) RunMatMul(inputs []*backend.Tensor, outputType mantaartifact.ValueType) (backend.StepDispatchResult, error) {
	a.runCalls++
	if len(inputs) == 2 && len(inputs[0].Shape) == 3 && len(inputs[1].Shape) == 3 {
		lhs := inputs[0]
		rhs := inputs[1]
		if lhs.Shape[0] == rhs.Shape[0] && lhs.Shape[2] == rhs.Shape[1] {
			if lhs.Shape[0] > a.maxRunBatches {
				a.maxRunBatches = lhs.Shape[0]
			}
			if rhs.Shape[2] > a.maxRunOutputCols {
				a.maxRunOutputCols = rhs.Shape[2]
			}
			out := make([]float32, lhs.Shape[0]*lhs.Shape[1]*rhs.Shape[2])
			for batch := 0; batch < lhs.Shape[0]; batch++ {
				lhsBase := batch * lhs.Shape[1] * lhs.Shape[2]
				rhsBase := batch * rhs.Shape[1] * rhs.Shape[2]
				outBase := batch * lhs.Shape[1] * rhs.Shape[2]
				fillHostMatMul(
					lhs.F32[lhsBase:lhsBase+lhs.Shape[1]*lhs.Shape[2]],
					lhs.Shape[1],
					lhs.Shape[2],
					rhs.F32[rhsBase:rhsBase+rhs.Shape[1]*rhs.Shape[2]],
					rhs.Shape[2],
					out[outBase:outBase+lhs.Shape[1]*rhs.Shape[2]],
				)
			}
			return backend.StepDispatchResult{Outputs: []*backend.Tensor{
				backend.NewTensorF32([]int{lhs.Shape[0], lhs.Shape[1], rhs.Shape[2]}, out),
			}}, nil
		}
	}
	return backend.StepDispatchResult{}, nil
}
func (a *countingMatMulAccelerator) RunMatMulWithTranspose(inputs []*backend.Tensor, outputType mantaartifact.ValueType, transposeLeft, transposeRight bool) (backend.StepDispatchResult, error) {
	a.runCalls++
	if len(inputs) == 2 && len(inputs[0].Shape) == 3 && len(inputs[1].Shape) == 3 {
		lhs := inputs[0]
		rhs := inputs[1]
		lhsRows, lhsCols := lhs.Shape[1], lhs.Shape[2]
		rhsRows, rhsCols := rhs.Shape[1], rhs.Shape[2]
		outRows, outCols, ok := trainerMatMulShape(lhsRows, lhsCols, rhsRows, rhsCols, transposeLeft, transposeRight)
		if lhs.Shape[0] == rhs.Shape[0] && ok {
			if lhs.Shape[0] > a.maxRunBatches {
				a.maxRunBatches = lhs.Shape[0]
			}
			if outCols > a.maxRunOutputCols {
				a.maxRunOutputCols = outCols
			}
			out := make([]float32, lhs.Shape[0]*outRows*outCols)
			for batch := 0; batch < lhs.Shape[0]; batch++ {
				lhsBase := batch * lhsRows * lhsCols
				rhsBase := batch * rhsRows * rhsCols
				outBase := batch * outRows * outCols
				fillHostMatMulTranspose(
					lhs.F32[lhsBase:lhsBase+lhsRows*lhsCols],
					lhsRows,
					lhsCols,
					rhs.F32[rhsBase:rhsBase+rhsRows*rhsCols],
					rhsRows,
					rhsCols,
					transposeLeft,
					transposeRight,
					out[outBase:outBase+outRows*outCols],
				)
			}
			return backend.StepDispatchResult{Outputs: []*backend.Tensor{
				backend.NewTensorF32([]int{lhs.Shape[0], outRows, outCols}, out),
			}}, nil
		}
	}
	if len(inputs) == 2 && len(inputs[0].Shape) == 2 && len(inputs[1].Shape) == 2 {
		lhs := inputs[0]
		rhs := inputs[1]
		lhsRows, lhsCols := lhs.Shape[0], lhs.Shape[1]
		rhsRows, rhsCols := rhs.Shape[0], rhs.Shape[1]
		outRows, outCols, ok := trainerMatMulShape(lhsRows, lhsCols, rhsRows, rhsCols, transposeLeft, transposeRight)
		if ok {
			if outCols > a.maxRunOutputCols {
				a.maxRunOutputCols = outCols
			}
			out := make([]float32, outRows*outCols)
			fillHostMatMulTranspose(lhs.F32, lhsRows, lhsCols, rhs.F32, rhsRows, rhsCols, transposeLeft, transposeRight, out)
			return backend.StepDispatchResult{Outputs: []*backend.Tensor{
				backend.NewTensorF32([]int{outRows, outCols}, out),
			}}, nil
		}
	}
	return backend.StepDispatchResult{}, nil
}
func (a *countingMatMulAccelerator) BindMatrix(name string, tensor *backend.Tensor) error {
	if a.bound == nil {
		a.bound = map[string]*backend.Tensor{}
	}
	a.bindCalls++
	a.bound[name] = tensor
	return nil
}
func (a *countingMatMulAccelerator) UnbindMatrix(name string) error {
	delete(a.bound, name)
	return nil
}
func (a *countingMatMulAccelerator) RunMatMulWithBoundLeft(leftName string, rhs *backend.Tensor, outputType mantaartifact.ValueType, transposeLeft, transposeRight bool) (backend.StepDispatchResult, error) {
	return backend.StepDispatchResult{}, nil
}
func (a *countingMatMulAccelerator) RunMatMulWithBoundRight(lhs *backend.Tensor, rightName string, outputType mantaartifact.ValueType, transposeLeft, transposeRight bool) (backend.StepDispatchResult, error) {
	a.boundRightRuns++
	if lhs != nil && len(lhs.Shape) > 0 && lhs.Shape[0] > a.maxBoundRightRows {
		a.maxBoundRightRows = lhs.Shape[0]
	}
	return backend.StepDispatchResult{}, nil
}
func (a *countingMatMulAccelerator) RunMatMulWithBoundRights(lhs *backend.Tensor, rightNames []string, outputType mantaartifact.ValueType, transposeLeft, transposeRight bool) ([]backend.StepDispatchResult, error) {
	a.multiBoundRuns++
	a.boundRightRuns += len(rightNames)
	if lhs != nil && len(lhs.Shape) > 0 && lhs.Shape[0] > a.maxBoundRightRows {
		a.maxBoundRightRows = lhs.Shape[0]
	}
	results := make([]backend.StepDispatchResult, len(rightNames))
	rows := 0
	inner := 0
	if lhs != nil && len(lhs.Shape) == 2 {
		rows = lhs.Shape[0]
		inner = lhs.Shape[1]
	}
	for i, name := range rightNames {
		cols := inner
		if a.bound != nil {
			if rhs := a.bound[name]; rhs != nil && len(rhs.Shape) == 2 {
				cols = rhs.Shape[1]
			}
		}
		results[i] = backend.StepDispatchResult{Outputs: []*backend.Tensor{
			backend.NewTensorF32([]int{rows, cols}, make([]float32, rows*cols)),
		}}
	}
	return results, nil
}
func (a *countingMatMulAccelerator) RunAccumulatedMatMulsWithBoundRights(lhsInputs []*backend.Tensor, rightNames []string, outputType mantaartifact.ValueType, transposeLeft, transposeRight bool) (backend.StepDispatchResult, error) {
	a.accumulatedRuns++
	a.boundRightRuns += len(rightNames)
	if len(rightNames) > a.maxAccumTerms {
		a.maxAccumTerms = len(rightNames)
	}
	if len(lhsInputs) != len(rightNames) || len(lhsInputs) == 0 {
		return backend.StepDispatchResult{}, nil
	}
	var out []float32
	var outShape []int
	for i, lhs := range lhsInputs {
		if lhs == nil || len(lhs.Shape) != 2 {
			return backend.StepDispatchResult{}, nil
		}
		if lhs.Shape[0] > a.maxBoundRightRows {
			a.maxBoundRightRows = lhs.Shape[0]
		}
		rhs := (*backend.Tensor)(nil)
		if a.bound != nil {
			rhs = a.bound[rightNames[i]]
		}
		if rhs == nil || len(rhs.Shape) != 2 {
			return backend.StepDispatchResult{}, nil
		}
		outRows, outCols, ok := trainerMatMulShape(lhs.Shape[0], lhs.Shape[1], rhs.Shape[0], rhs.Shape[1], transposeLeft, transposeRight)
		if !ok {
			return backend.StepDispatchResult{}, nil
		}
		if i == 0 {
			outShape = []int{outRows, outCols}
			out = make([]float32, outRows*outCols)
		} else if len(outShape) != 2 || outShape[0] != outRows || outShape[1] != outCols {
			return backend.StepDispatchResult{}, nil
		}
		step := make([]float32, outRows*outCols)
		fillHostMatMulTranspose(lhs.F32, lhs.Shape[0], lhs.Shape[1], rhs.F32, rhs.Shape[0], rhs.Shape[1], transposeLeft, transposeRight, step)
		addFloat32Slice(out, step)
	}
	return backend.StepDispatchResult{Outputs: []*backend.Tensor{
		backend.NewTensorF32(outShape, out),
	}}, nil
}
func (a *countingMatMulAccelerator) RunMatMulsWithSharedLeft(lhs *backend.Tensor, rhs []*backend.Tensor, outputType mantaartifact.ValueType, transposeLeft, transposeRight bool) ([]backend.StepDispatchResult, error) {
	a.sharedLeftRuns++
	if len(rhs) > a.maxSharedLeftRHS {
		a.maxSharedLeftRHS = len(rhs)
	}
	results := make([]backend.StepDispatchResult, len(rhs))
	lhsRows := 0
	lhsCols := 0
	if lhs != nil && len(lhs.Shape) == 2 {
		lhsRows = lhs.Shape[0]
		lhsCols = lhs.Shape[1]
	}
	for i, right := range rhs {
		rhsRows := 0
		rhsCols := 0
		if right != nil && len(right.Shape) == 2 {
			rhsRows = right.Shape[0]
			rhsCols = right.Shape[1]
		}
		rows, cols, ok := trainerMatMulShape(lhsRows, lhsCols, rhsRows, rhsCols, transposeLeft, transposeRight)
		if !ok {
			rows, cols = lhsCols, rhsCols
		}
		results[i] = backend.StepDispatchResult{Outputs: []*backend.Tensor{
			backend.NewTensorF32([]int{rows, cols}, make([]float32, rows*cols)),
		}}
	}
	return results, nil
}
func (a *countingMatMulAccelerator) Stats() backend.MatMulAcceleratorStats {
	return backend.MatMulAcceleratorStats{
		BindCalls:       int64(a.bindCalls),
		BoundMatrices:   int64(len(a.bound)),
		RunCalls:        int64(a.boundRightRuns + a.runCalls),
		BoundRightCalls: int64(a.boundRightRuns),
	}
}
func (a *countingMatMulAccelerator) Close() {}

type countingActivationAccelerator struct {
	bindCalls              int
	unbindCalls            int
	geluBackwardCalls      int
	softmaxBackwardCalls   int
	layerNormBackwardCalls int
	bound                  map[string]*backend.Tensor
}

func (a *countingActivationAccelerator) Backend() mantaartifact.BackendKind {
	return mantaartifact.BackendCUDA
}

func (a *countingActivationAccelerator) BindTensor(name string, tensor *backend.Tensor) error {
	if a.bound == nil {
		a.bound = map[string]*backend.Tensor{}
	}
	a.bindCalls++
	a.bound[name] = tensor
	return nil
}

func (a *countingActivationAccelerator) UnbindTensor(name string) error {
	a.unbindCalls++
	delete(a.bound, name)
	return nil
}

func (a *countingActivationAccelerator) RunGELUBackwardMul(gradOut, preAct *backend.Tensor) (*backend.Tensor, error) {
	a.geluBackwardCalls++
	if gradOut == nil || preAct == nil {
		return backend.NewTensorF32(nil, nil), nil
	}
	out := make([]float32, len(gradOut.F32))
	for i := range out {
		out[i] = gradOut.F32[i] * geluBackward(preAct.F32[i])
	}
	return backend.NewTensorF32(append([]int(nil), gradOut.Shape...), out), nil
}

func (a *countingActivationAccelerator) RunGELUBackwardMulWithBoundPreAct(gradOut *backend.Tensor, preActName string) (*backend.Tensor, error) {
	preAct := a.bound[preActName]
	return a.RunGELUBackwardMul(gradOut, preAct)
}

func (a *countingActivationAccelerator) RunSoftmaxBackwardRows(gradOut, probs *backend.Tensor) (*backend.Tensor, error) {
	a.softmaxBackwardCalls++
	if gradOut == nil || probs == nil || len(gradOut.Shape) != 2 {
		return backend.NewTensorF32(nil, nil), nil
	}
	out := make([]float32, len(gradOut.F32))
	rows, cols := gradOut.Shape[0], gradOut.Shape[1]
	for row := 0; row < rows; row++ {
		backwardSoftmaxRow(out[row*cols:(row+1)*cols], gradOut.F32[row*cols:(row+1)*cols], probs.F32[row*cols:(row+1)*cols])
	}
	return backend.NewTensorF32(append([]int(nil), gradOut.Shape...), out), nil
}

func (a *countingActivationAccelerator) RunSoftmaxBackwardRowsWithBoundProbs(gradOut *backend.Tensor, probsName string) (*backend.Tensor, error) {
	probs := a.bound[probsName]
	return a.RunSoftmaxBackwardRows(gradOut, probs)
}

func (a *countingActivationAccelerator) RunLayerNormBackwardRows(gradOut, normalized, pre *backend.Tensor) (*backend.Tensor, error) {
	a.layerNormBackwardCalls++
	if gradOut == nil || normalized == nil || pre == nil || len(gradOut.Shape) != 2 {
		return backend.NewTensorF32(nil, nil), nil
	}
	out := make([]float32, len(gradOut.F32))
	rows, cols := gradOut.Shape[0], gradOut.Shape[1]
	for row := 0; row < rows; row++ {
		backwardLayerNormRow(
			out[row*cols:(row+1)*cols],
			gradOut.F32[row*cols:(row+1)*cols],
			normalized.F32[row*cols:(row+1)*cols],
			pre.F32[row*cols:(row+1)*cols],
		)
	}
	return backend.NewTensorF32(append([]int(nil), gradOut.Shape...), out), nil
}

func (a *countingActivationAccelerator) RunLayerNormBackwardRowsWithBoundInputs(gradOut *backend.Tensor, normalizedName, preName string) (*backend.Tensor, error) {
	return a.RunLayerNormBackwardRows(gradOut, a.bound[normalizedName], a.bound[preName])
}

func (a *countingActivationAccelerator) Stats() backend.ActivationAcceleratorStats {
	return backend.ActivationAcceleratorStats{
		BindCalls:              int64(a.bindCalls),
		GELUBackwardCalls:      int64(a.geluBackwardCalls),
		SoftmaxBackwardCalls:   int64(a.softmaxBackwardCalls),
		LayerNormBackwardCalls: int64(a.layerNormBackwardCalls),
		BoundTensors:           int64(len(a.bound)),
	}
}

func (a *countingActivationAccelerator) Close() {}

func TestEmbeddingTrainerRejectsNonTrainableParams(t *testing.T) {
	src := []byte(`
param token_embedding: q8[V, D] @weight("weights/token_embedding")
param projection: q8[D, E] @weight("weights/projection")

pipeline embed_pooled(tokens: i32[T], attention_mask: i32[T]) -> f16[E] {
    let hidden_q = gather(token_embedding, tokens)
    let hidden = dequant(hidden_q)
    let projection_f = dequant(projection)
    let projected = @matmul(hidden, projection_f)
    let normalized = normalize(projected)
    return mean_pool(normalized, attention_mask)
}

pipeline embed_pooled_batch(tokens: i32[B, T], attention_mask: i32[B, T]) -> f16[B, E] {
    let hidden_q = gather(token_embedding, tokens)
    let hidden = dequant(hidden_q)
    let projection_f = dequant(projection)
    let projected = @matmul(hidden, projection_f)
    let normalized = normalize(projected)
    return mean_pool(normalized, attention_mask)
}
`)

	bundle, err := compiler.Build(src, compiler.Options{ModuleName: "frozen_train_embed_q8"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	_, err = NewEmbeddingTrainer(bundle.Artifact, tinyMaskedEmbeddingManifest(), map[string]*backend.Tensor{
		"token_embedding": backend.NewTensorF32([]int{3, 2}, []float32{
			0.9, 0.1,
			0.3, 0.7,
			0.6, 0.4,
		}),
		"projection": backend.NewTensorF32([]int{2, 2}, []float32{
			1, 0,
			0, 1,
		}),
	}, EmbeddingTrainConfig{})
	if err == nil {
		t.Fatal("expected non-trainable param error")
	}
	if !strings.Contains(err.Error(), `not marked @trainable`) {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestEmbeddingTrainerTrainStepReducesLossAndExportsQuantizedWeights(t *testing.T) {
	src := []byte(`
param token_embedding: q8[V, D] @weight("weights/token_embedding") @trainable
param projection: q8[D, E] @weight("weights/projection") @trainable

pipeline embed_pooled(tokens: i32[T], attention_mask: i32[T]) -> f16[E] {
    let hidden_q = gather(token_embedding, tokens)
    let hidden = dequant(hidden_q)
    let projection_f = dequant(projection)
    let projected = @matmul(hidden, projection_f)
    let normalized = normalize(projected)
    return mean_pool(normalized, attention_mask)
}

pipeline embed_pooled_batch(tokens: i32[B, T], attention_mask: i32[B, T]) -> f16[B, E] {
    let hidden_q = gather(token_embedding, tokens)
    let hidden = dequant(hidden_q)
    let projection_f = dequant(projection)
    let projected = @matmul(hidden, projection_f)
    let normalized = normalize(projected)
    return mean_pool(normalized, attention_mask)
}
`)

	bundle, err := compiler.Build(src, compiler.Options{ModuleName: "tiny_train_embed_q8"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	manifest := tinyMaskedEmbeddingManifest()
	manifest.Name = "tiny_train_embed_q8"
	trainer, err := NewEmbeddingTrainer(bundle.Artifact, manifest, map[string]*backend.Tensor{
		"token_embedding": backend.NewTensorF32([]int{3, 2}, []float32{
			1, 0,
			0, 1,
			1, 1,
		}),
		"projection": backend.NewTensorF32([]int{2, 2}, []float32{
			1, 0,
			0, 1,
		}),
	}, EmbeddingTrainConfig{LearningRate: 0.2})
	if err != nil {
		t.Fatalf("new trainer: %v", err)
	}

	batch := []EmbeddingPairExample{
		{LeftTokens: []int32{0}, RightTokens: []int32{0}, LeftMask: []int32{1}, RightMask: []int32{1}, Target: 1},
		{LeftTokens: []int32{1}, RightTokens: []int32{1}, LeftMask: []int32{1}, RightMask: []int32{1}, Target: 1},
		{LeftTokens: []int32{0}, RightTokens: []int32{1}, LeftMask: []int32{1}, RightMask: []int32{1}, Target: -1},
		{LeftTokens: []int32{1}, RightTokens: []int32{0}, LeftMask: []int32{1}, RightMask: []int32{1}, Target: -1},
	}

	before, err := trainer.EvalBatch(batch)
	if err != nil {
		t.Fatalf("eval before: %v", err)
	}
	for i := 0; i < 32; i++ {
		if _, err := trainer.TrainStep(batch); err != nil {
			t.Fatalf("train step %d: %v", i, err)
		}
	}
	after, err := trainer.EvalBatch(batch)
	if err != nil {
		t.Fatalf("eval after: %v", err)
	}
	if after.Loss >= before.Loss {
		t.Fatalf("loss did not decrease: before=%f after=%f", before.Loss, after.Loss)
	}

	exported, err := trainer.ExportInferenceWeights()
	if err != nil {
		t.Fatalf("export weights: %v", err)
	}
	if got := exported["token_embedding"].DType; got != "q8" {
		t.Fatalf("token_embedding export dtype = %q, want q8", got)
	}
	if got := exported["projection"].DType; got != "q8" {
		t.Fatalf("projection export dtype = %q, want q8", got)
	}

	loadOpts, err := trainer.ExportLoadOptions()
	if err != nil {
		t.Fatalf("export load options: %v", err)
	}
	rt := New(cuda.New(), metal.New())
	model, err := rt.LoadEmbedding(context.Background(), bundle.Artifact, manifest, loadOpts...)
	if err != nil {
		t.Fatalf("load trained model: %v", err)
	}
	result, err := model.Embed(context.Background(), []int32{0})
	if err != nil {
		t.Fatalf("embed trained model: %v", err)
	}
	if result.Embeddings == nil {
		t.Fatal("expected embedding output")
	}
	if got := result.Embeddings.DType; got != "f16" {
		t.Fatalf("embedding dtype = %q, want f16", got)
	}
}

func TestEmbeddingTrainerTrainStepSupportsFFNGELUAndExportsQuantizedWeights(t *testing.T) {
	src := []byte(`
param token_embedding: q8[V, D] @weight("weights/token_embedding") @trainable
param ffn_up: q8[D, H] @weight("weights/ffn_up") @trainable
param projection: q8[H, E] @weight("weights/projection") @trainable

pipeline embed_pooled(tokens: i32[T], attention_mask: i32[T]) -> f16[E] {
    let hidden_q = gather(token_embedding, tokens)
    let hidden = dequant(hidden_q)
    let ffn_up_f = dequant(ffn_up)
    let projection_f = dequant(projection)
    let ffn_hidden = @matmul(hidden, ffn_up_f)
    let activated = gelu(ffn_hidden)
    let projected = @matmul(activated, projection_f)
    let normalized = normalize(projected)
    return mean_pool(normalized, attention_mask)
}

pipeline embed_pooled_batch(tokens: i32[B, T], attention_mask: i32[B, T]) -> f16[B, E] {
    let hidden_q = gather(token_embedding, tokens)
    let hidden = dequant(hidden_q)
    let ffn_up_f = dequant(ffn_up)
    let projection_f = dequant(projection)
    let ffn_hidden = @matmul(hidden, ffn_up_f)
    let activated = gelu(ffn_hidden)
    let projected = @matmul(activated, projection_f)
    let normalized = normalize(projected)
    return mean_pool(normalized, attention_mask)
}
`)

	bundle, err := compiler.Build(src, compiler.Options{ModuleName: "tiny_train_embed_ffn_q8"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	manifest := tinyMaskedEmbeddingManifest()
	manifest.Name = "tiny_train_embed_ffn_q8"
	manifest.HiddenProjectionParam = "ffn_up"
	trainer, err := NewEmbeddingTrainer(bundle.Artifact, manifest, map[string]*backend.Tensor{
		"token_embedding": backend.NewTensorF32([]int{3, 2}, []float32{
			1, 0,
			0, 1,
			1, 1,
		}),
		"ffn_up": backend.NewTensorF32([]int{2, 3}, []float32{
			1, 0, 1,
			0, 1, 1,
		}),
		"projection": backend.NewTensorF32([]int{3, 2}, []float32{
			1, 0,
			0, 1,
			0.5, 0.5,
		}),
	}, EmbeddingTrainConfig{LearningRate: 0.05})
	if err != nil {
		t.Fatalf("new trainer: %v", err)
	}

	batch := []EmbeddingPairExample{
		{LeftTokens: []int32{0}, RightTokens: []int32{0}, LeftMask: []int32{1}, RightMask: []int32{1}, Target: 1},
		{LeftTokens: []int32{1}, RightTokens: []int32{1}, LeftMask: []int32{1}, RightMask: []int32{1}, Target: 1},
		{LeftTokens: []int32{0}, RightTokens: []int32{1}, LeftMask: []int32{1}, RightMask: []int32{1}, Target: -1},
		{LeftTokens: []int32{1}, RightTokens: []int32{0}, LeftMask: []int32{1}, RightMask: []int32{1}, Target: -1},
	}

	before, err := trainer.EvalBatch(batch)
	if err != nil {
		t.Fatalf("eval before: %v", err)
	}
	for i := 0; i < 32; i++ {
		if _, err := trainer.TrainStep(batch); err != nil {
			t.Fatalf("train step %d: %v", i, err)
		}
	}
	after, err := trainer.EvalBatch(batch)
	if err != nil {
		t.Fatalf("eval after: %v", err)
	}
	if after.Loss >= before.Loss {
		t.Fatalf("ffn loss did not decrease: before=%f after=%f", before.Loss, after.Loss)
	}

	exported, err := trainer.ExportInferenceWeights()
	if err != nil {
		t.Fatalf("export weights: %v", err)
	}
	if got := exported["token_embedding"].DType; got != "q8" {
		t.Fatalf("token_embedding export dtype = %q, want q8", got)
	}
	if got := exported["ffn_up"].DType; got != "q8" {
		t.Fatalf("ffn_up export dtype = %q, want q8", got)
	}
	if got := exported["projection"].DType; got != "q8" {
		t.Fatalf("projection export dtype = %q, want q8", got)
	}

	loadOpts, err := trainer.ExportLoadOptions()
	if err != nil {
		t.Fatalf("export load options: %v", err)
	}
	rt := New(cuda.New(), metal.New())
	model, err := rt.LoadEmbedding(context.Background(), bundle.Artifact, manifest, loadOpts...)
	if err != nil {
		t.Fatalf("load trained ffn model: %v", err)
	}
	result, err := model.Embed(context.Background(), []int32{0})
	if err != nil {
		t.Fatalf("embed trained ffn model: %v", err)
	}
	if result.Embeddings == nil {
		t.Fatal("expected embedding output")
	}
	if got := result.Embeddings.DType; got != "f16" {
		t.Fatalf("embedding dtype = %q, want f16", got)
	}
}

func TestEmbeddingTrainerTrainStepSupportsAttentionAndExportsQuantizedWeights(t *testing.T) {
	trainer := newTinyTrainableAttentionEmbeddingTrainer(t, 0.05)

	batch := []EmbeddingPairExample{
		{LeftTokens: []int32{0}, RightTokens: []int32{0}, LeftMask: []int32{1}, RightMask: []int32{1}, Target: 1},
		{LeftTokens: []int32{1}, RightTokens: []int32{1}, LeftMask: []int32{1}, RightMask: []int32{1}, Target: 1},
		{LeftTokens: []int32{0}, RightTokens: []int32{1}, LeftMask: []int32{1}, RightMask: []int32{1}, Target: -1},
		{LeftTokens: []int32{1}, RightTokens: []int32{0}, LeftMask: []int32{1}, RightMask: []int32{1}, Target: -1},
	}

	before, err := trainer.EvalBatch(batch)
	if err != nil {
		t.Fatalf("eval before: %v", err)
	}
	for i := 0; i < 32; i++ {
		if _, err := trainer.TrainStep(batch); err != nil {
			t.Fatalf("train step %d: %v", i, err)
		}
	}
	after, err := trainer.EvalBatch(batch)
	if err != nil {
		t.Fatalf("eval after: %v", err)
	}
	if after.Loss >= before.Loss {
		t.Fatalf("attention loss did not decrease: before=%f after=%f", before.Loss, after.Loss)
	}

	exported, err := trainer.ExportInferenceWeights()
	if err != nil {
		t.Fatalf("export weights: %v", err)
	}
	for _, name := range []string{"token_embedding", "attn_q", "attn_k", "attn_v", "attn_o", "projection"} {
		if got := exported[name].DType; got != "q8" {
			t.Fatalf("%s export dtype = %q, want q8", name, got)
		}
	}
}

func TestEmbeddingTrainerEvaluatePairsImprovesSeparation(t *testing.T) {
	src := []byte(`
param token_embedding: q8[V, D] @weight("weights/token_embedding") @trainable
param projection: q8[D, E] @weight("weights/projection") @trainable

pipeline embed_pooled(tokens: i32[T], attention_mask: i32[T]) -> f16[E] {
    let hidden_q = gather(token_embedding, tokens)
    let hidden = dequant(hidden_q)
    let projection_f = dequant(projection)
    let projected = @matmul(hidden, projection_f)
    let normalized = normalize(projected)
    return mean_pool(normalized, attention_mask)
}

pipeline embed_pooled_batch(tokens: i32[B, T], attention_mask: i32[B, T]) -> f16[B, E] {
    let hidden_q = gather(token_embedding, tokens)
    let hidden = dequant(hidden_q)
    let projection_f = dequant(projection)
    let projected = @matmul(hidden, projection_f)
    let normalized = normalize(projected)
    return mean_pool(normalized, attention_mask)
}
`)

	bundle, err := compiler.Build(src, compiler.Options{ModuleName: "tiny_train_embed_q8"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	manifest := tinyMaskedEmbeddingManifest()
	manifest.Name = "tiny_train_embed_q8"
	trainer, err := NewEmbeddingTrainer(bundle.Artifact, manifest, map[string]*backend.Tensor{
		"token_embedding": backend.NewTensorF32([]int{3, 2}, []float32{
			1, 0,
			0, 1,
			1, 1,
		}),
		"projection": backend.NewTensorF32([]int{2, 2}, []float32{
			1, 0,
			0, 1,
		}),
	}, EmbeddingTrainConfig{LearningRate: 0.05})
	if err != nil {
		t.Fatalf("new trainer: %v", err)
	}

	batch := []EmbeddingPairExample{
		{LeftTokens: []int32{0}, RightTokens: []int32{0}, LeftMask: []int32{1}, RightMask: []int32{1}, Target: 1},
		{LeftTokens: []int32{1}, RightTokens: []int32{1}, LeftMask: []int32{1}, RightMask: []int32{1}, Target: 1},
		{LeftTokens: []int32{0}, RightTokens: []int32{1}, LeftMask: []int32{1}, RightMask: []int32{1}, Target: -1},
		{LeftTokens: []int32{1}, RightTokens: []int32{0}, LeftMask: []int32{1}, RightMask: []int32{1}, Target: -1},
	}

	before, err := trainer.EvaluatePairs(batch)
	if err != nil {
		t.Fatalf("eval before: %v", err)
	}
	for i := 0; i < 24; i++ {
		if _, err := trainer.TrainStep(batch); err != nil {
			t.Fatalf("train step %d: %v", i, err)
		}
	}
	after, err := trainer.EvaluatePairs(batch)
	if err != nil {
		t.Fatalf("eval after: %v", err)
	}
	if after.ScoreMargin <= before.ScoreMargin {
		t.Fatalf("score margin did not improve: before=%f after=%f", before.ScoreMargin, after.ScoreMargin)
	}
	if after.PairAccuracy < before.PairAccuracy {
		t.Fatalf("pair accuracy regressed: before=%f after=%f", before.PairAccuracy, after.PairAccuracy)
	}
}

func TestEvalScoreMetricsCalibratePositiveScoreShift(t *testing.T) {
	metrics := EmbeddingEvalMetrics{
		PairAccuracy:  0.5,
		PositiveCount: 2,
		NegativeCount: 2,
	}
	finalizeEvalScoreMetrics(&metrics, []embeddingEvalScore{
		{Score: 0.9, Positive: true},
		{Score: 0.8, Positive: true},
		{Score: 0.7, Positive: false},
		{Score: 0.6, Positive: false},
	})

	assertClose(t, metrics.PairAccuracy, 0.5, 0.000001)
	assertClose(t, metrics.ThresholdAccuracy, 1, 0.000001)
	assertClose(t, metrics.ScoreThreshold, 0.8, 0.000001)
	assertClose(t, metrics.ROCAUC, 1, 0.000001)
}

func TestEvalRankMetricsTrackGroupedPairwiseRetrieval(t *testing.T) {
	metrics := EmbeddingEvalMetrics{}
	finalizeEvalRankMetrics(&metrics, []embeddingEvalRankScore{
		{QueryKey: "q1", Score: 0.8, Positive: true},
		{QueryKey: "q1", Score: 0.2, Positive: false},
		{QueryKey: "q1", Score: 0.1, Positive: false},
		{QueryKey: "q2", Score: 0.3, Positive: true},
		{QueryKey: "q2", Score: 0.9, Positive: false},
		{QueryKey: "q2", Score: 0.4, Positive: false},
		{QueryKey: "q3", Score: 0.9, Positive: false},
	})

	assertClose(t, metrics.Top1Accuracy, 0.5, 0.000001)
	assertClose(t, metrics.Top5Accuracy, 1, 0.000001)
	assertClose(t, metrics.Top10Accuracy, 1, 0.000001)
	assertClose(t, metrics.MeanReciprocalRank, 2.0/3.0, 0.000001)
	assertClose(t, metrics.MeanPositiveRank, 2, 0.000001)
}

func TestEmbeddingTrainerEvaluatePairsTracksGroupedRankingMetrics(t *testing.T) {
	src := []byte(`
param token_embedding: q8[V, D] @weight("weights/token_embedding") @trainable
param projection: q8[D, E] @weight("weights/projection") @trainable

pipeline embed_pooled(tokens: i32[T], attention_mask: i32[T]) -> f16[E] {
    let hidden_q = gather(token_embedding, tokens)
    let hidden = dequant(hidden_q)
    let projection_f = dequant(projection)
    let projected = @matmul(hidden, projection_f)
    let normalized = normalize(projected)
    return mean_pool(normalized, attention_mask)
}

pipeline embed_pooled_batch(tokens: i32[B, T], attention_mask: i32[B, T]) -> f16[B, E] {
    let hidden_q = gather(token_embedding, tokens)
    let hidden = dequant(hidden_q)
    let projection_f = dequant(projection)
    let projected = @matmul(hidden, projection_f)
    let normalized = normalize(projected)
    return mean_pool(normalized, attention_mask)
}
`)

	bundle, err := compiler.Build(src, compiler.Options{ModuleName: "tiny_ranked_eval_embed_q8"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	manifest := tinyMaskedEmbeddingManifest()
	manifest.Name = "tiny_ranked_eval_embed_q8"
	trainer, err := NewEmbeddingTrainer(bundle.Artifact, manifest, map[string]*backend.Tensor{
		"token_embedding": backend.NewTensorF32([]int{3, 2}, []float32{
			1, 0,
			0, 1,
			1, 0,
		}),
		"projection": backend.NewTensorF32([]int{2, 2}, []float32{
			1, 0,
			0, 1,
		}),
	}, EmbeddingTrainConfig{LearningRate: 0.05})
	if err != nil {
		t.Fatalf("new trainer: %v", err)
	}

	got, err := trainer.EvaluatePairs([]EmbeddingPairExample{
		{LeftTokens: []int32{0}, RightTokens: []int32{0}, LeftMask: []int32{1}, RightMask: []int32{1}, Target: 1},
		{LeftTokens: []int32{0}, RightTokens: []int32{1}, LeftMask: []int32{1}, RightMask: []int32{1}, Target: -1},
		{LeftTokens: []int32{1}, RightTokens: []int32{0}, LeftMask: []int32{1}, RightMask: []int32{1}, Target: -1},
		{LeftTokens: []int32{1}, RightTokens: []int32{1}, LeftMask: []int32{1}, RightMask: []int32{1}, Target: 1},
	})
	if err != nil {
		t.Fatalf("evaluate pairs: %v", err)
	}

	assertClose(t, got.Top1Accuracy, 1, 0.000001)
	assertClose(t, got.Top5Accuracy, 1, 0.000001)
	assertClose(t, got.Top10Accuracy, 1, 0.000001)
	assertClose(t, got.MeanReciprocalRank, 1, 0.000001)
	assertClose(t, got.MeanPositiveRank, 1, 0.000001)
}

func TestDefaultEmbeddingCheckpointPath(t *testing.T) {
	got := DefaultEmbeddingCheckpointPath("/tmp/tiny_train_embed_q8.mll")
	if want := "/tmp/tiny_train_embed_q8.embed-train.mll"; got != want {
		t.Fatalf("checkpoint path = %q, want %q", got, want)
	}
}

func TestEmbeddingTrainerCheckpointRoundTrip(t *testing.T) {
	src := []byte(`
param token_embedding: q8[V, D] @weight("weights/token_embedding") @trainable
param projection: q8[D, E] @weight("weights/projection") @trainable

pipeline embed_pooled(tokens: i32[T], attention_mask: i32[T]) -> f16[E] {
    let hidden_q = gather(token_embedding, tokens)
    let hidden = dequant(hidden_q)
    let projection_f = dequant(projection)
    let projected = @matmul(hidden, projection_f)
    let normalized = normalize(projected)
    return mean_pool(normalized, attention_mask)
}

pipeline embed_pooled_batch(tokens: i32[B, T], attention_mask: i32[B, T]) -> f16[B, E] {
    let hidden_q = gather(token_embedding, tokens)
    let hidden = dequant(hidden_q)
    let projection_f = dequant(projection)
    let projected = @matmul(hidden, projection_f)
    let normalized = normalize(projected)
    return mean_pool(normalized, attention_mask)
}
`)

	bundle, err := compiler.Build(src, compiler.Options{ModuleName: "tiny_train_embed_q8"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	manifest := tinyMaskedEmbeddingManifest()
	manifest.Name = "tiny_train_embed_q8"
	cfg := EmbeddingTrainConfig{LearningRate: 0.05}
	trainer, err := NewEmbeddingTrainer(bundle.Artifact, manifest, map[string]*backend.Tensor{
		"token_embedding": backend.NewTensorF32([]int{3, 2}, []float32{
			1, 0,
			0, 1,
			1, 1,
		}),
		"projection": backend.NewTensorF32([]int{2, 2}, []float32{
			1, 0,
			0, 1,
		}),
	}, cfg)
	if err != nil {
		t.Fatalf("new trainer: %v", err)
	}

	batch := []EmbeddingPairExample{
		{LeftTokens: []int32{0}, RightTokens: []int32{0}, LeftMask: []int32{1}, RightMask: []int32{1}, Target: 1},
		{LeftTokens: []int32{1}, RightTokens: []int32{1}, LeftMask: []int32{1}, RightMask: []int32{1}, Target: 1},
		{LeftTokens: []int32{0}, RightTokens: []int32{1}, LeftMask: []int32{1}, RightMask: []int32{1}, Target: -1},
		{LeftTokens: []int32{1}, RightTokens: []int32{0}, LeftMask: []int32{1}, RightMask: []int32{1}, Target: -1},
	}

	for i := 0; i < 4; i++ {
		if _, err := trainer.TrainStep(batch); err != nil {
			t.Fatalf("train step %d: %v", i, err)
		}
	}
	checkpoint, err := trainer.Checkpoint()
	if err != nil {
		t.Fatalf("checkpoint: %v", err)
	}
	if checkpoint.Step != 4 {
		t.Fatalf("checkpoint step = %d, want 4", checkpoint.Step)
	}
	if checkpoint.TokenMoment1 == nil || checkpoint.TokenMoment2 == nil || checkpoint.ProjMoment1 == nil || checkpoint.ProjMoment2 == nil {
		t.Fatal("expected optimizer moments in checkpoint")
	}

	path := filepath.Join(t.TempDir(), "tiny_train_embed_q8.embed-train.mll")
	if err := checkpoint.WriteFile(path); err != nil {
		t.Fatalf("write checkpoint: %v", err)
	}
	loaded, err := ReadEmbeddingTrainCheckpointFile(path)
	if err != nil {
		t.Fatalf("read checkpoint: %v", err)
	}
	restored, err := NewEmbeddingTrainerFromCheckpoint(bundle.Artifact, loaded)
	if err != nil {
		t.Fatalf("restore trainer: %v", err)
	}

	beforeA, err := trainer.EvalBatch(batch)
	if err != nil {
		t.Fatalf("eval original: %v", err)
	}
	beforeB, err := restored.EvalBatch(batch)
	if err != nil {
		t.Fatalf("eval restored: %v", err)
	}
	assertClose(t, beforeA.Loss, beforeB.Loss, 0.000001)
	assertClose(t, beforeA.AverageScore, beforeB.AverageScore, 0.000001)

	if _, err := trainer.TrainStep(batch); err != nil {
		t.Fatalf("train original after restore: %v", err)
	}
	if _, err := restored.TrainStep(batch); err != nil {
		t.Fatalf("train restored after restore: %v", err)
	}
	exportA, err := trainer.ExportInferenceWeights()
	if err != nil {
		t.Fatalf("export original: %v", err)
	}
	exportB, err := restored.ExportInferenceWeights()
	if err != nil {
		t.Fatalf("export restored: %v", err)
	}
	assertTensorClose(t, exportA["token_embedding"], exportB["token_embedding"].Shape, exportB["token_embedding"].F32)
	assertTensorClose(t, exportA["projection"], exportB["projection"].Shape, exportB["projection"].F32)
}

func TestEmbeddingTrainerFFNCheckpointRoundTrip(t *testing.T) {
	trainer := newTinyTrainableFFNEmbeddingTrainer(t, 0.05)
	batch := tinyEmbeddingPairDataset()

	for i := 0; i < 4; i++ {
		if _, err := trainer.TrainStep(batch); err != nil {
			t.Fatalf("train step %d: %v", i, err)
		}
	}

	checkpoint, err := trainer.Checkpoint()
	if err != nil {
		t.Fatalf("checkpoint: %v", err)
	}
	restored, err := NewEmbeddingTrainerFromCheckpoint(trainer.module, checkpoint)
	if err != nil {
		t.Fatalf("restore checkpoint: %v", err)
	}

	if _, err := trainer.TrainStep(batch); err != nil {
		t.Fatalf("continue original trainer: %v", err)
	}
	if _, err := restored.TrainStep(batch); err != nil {
		t.Fatalf("continue restored trainer: %v", err)
	}

	exportA, err := trainer.ExportInferenceWeights()
	if err != nil {
		t.Fatalf("export original: %v", err)
	}
	exportB, err := restored.ExportInferenceWeights()
	if err != nil {
		t.Fatalf("export restored: %v", err)
	}
	assertTensorClose(t, exportA["token_embedding"], exportB["token_embedding"].Shape, exportB["token_embedding"].F32)
	assertTensorClose(t, exportA["ffn_up"], exportB["ffn_up"].Shape, exportB["ffn_up"].F32)
	assertTensorClose(t, exportA["projection"], exportB["projection"].Shape, exportB["projection"].F32)
}

func TestEmbeddingTrainerAttentionCheckpointRoundTrip(t *testing.T) {
	trainer := newTinyTrainableAttentionEmbeddingTrainer(t, 0.05)
	batch := tinyEmbeddingPairDataset()

	for i := 0; i < 4; i++ {
		if _, err := trainer.TrainStep(batch); err != nil {
			t.Fatalf("train step %d: %v", i, err)
		}
	}

	checkpoint, err := trainer.Checkpoint()
	if err != nil {
		t.Fatalf("checkpoint: %v", err)
	}
	restored, err := NewEmbeddingTrainerFromCheckpoint(trainer.module, checkpoint)
	if err != nil {
		t.Fatalf("restore checkpoint: %v", err)
	}
	t.Cleanup(restored.Close)

	if _, err := trainer.TrainStep(batch); err != nil {
		t.Fatalf("continue original trainer: %v", err)
	}
	if _, err := restored.TrainStep(batch); err != nil {
		t.Fatalf("continue restored trainer: %v", err)
	}

	exportA, err := trainer.ExportInferenceWeights()
	if err != nil {
		t.Fatalf("export original: %v", err)
	}
	exportB, err := restored.ExportInferenceWeights()
	if err != nil {
		t.Fatalf("export restored: %v", err)
	}
	for _, name := range []string{"token_embedding", "attn_q", "attn_k", "attn_v", "attn_o", "projection"} {
		assertTensorClose(t, exportA[name], exportB[name].Shape, exportB[name].F32)
	}
}

func TestEmbeddingTrainerTrainStepSupportsEncoderResidualLayerNormAndExportsQuantizedWeights(t *testing.T) {
	trainer := newTinyTrainableEncoderEmbeddingTrainer(t, 0.02)

	batch := []EmbeddingPairExample{
		{LeftTokens: []int32{0, 2}, RightTokens: []int32{1, 2}, LeftMask: []int32{1, 1}, RightMask: []int32{1, 1}, Target: 0},
		{LeftTokens: []int32{0, 0}, RightTokens: []int32{0, 2}, LeftMask: []int32{1, 1}, RightMask: []int32{1, 1}, Target: 0.5},
		{LeftTokens: []int32{1, 1}, RightTokens: []int32{1, 2}, LeftMask: []int32{1, 1}, RightMask: []int32{1, 1}, Target: 0.5},
		{LeftTokens: []int32{0, 1}, RightTokens: []int32{1, 0}, LeftMask: []int32{1, 1}, RightMask: []int32{1, 1}, Target: 0},
	}
	beforeMaster := map[string][]float32{
		"token_embedding": append([]float32(nil), trainer.tokenEmbed.F32...),
		"attn_q":          append([]float32(nil), trainer.attentionQuery.F32...),
		"attn_k":          append([]float32(nil), trainer.attentionKey.F32...),
		"attn_v":          append([]float32(nil), trainer.attentionValue.F32...),
		"attn_o":          append([]float32(nil), trainer.attentionOutput.F32...),
		"ffn_up":          append([]float32(nil), trainer.hiddenProjection.F32...),
		"projection":      append([]float32(nil), trainer.projection.F32...),
	}

	before, err := trainer.EvalBatch(batch)
	if err != nil {
		t.Fatalf("eval before: %v", err)
	}
	for i := 0; i < 32; i++ {
		if _, err := trainer.TrainStep(batch); err != nil {
			t.Fatalf("train step %d: %v", i, err)
		}
	}
	after, err := trainer.EvalBatch(batch)
	if err != nil {
		t.Fatalf("eval after: %v", err)
	}
	if after.Loss > before.Loss+0.000001 {
		t.Fatalf("encoder loss regressed: before=%f after=%f", before.Loss, after.Loss)
	}

	exported, err := trainer.ExportInferenceWeights()
	if err != nil {
		t.Fatalf("export weights: %v", err)
	}
	for _, name := range []string{"token_embedding", "attn_q", "attn_k", "attn_v", "attn_o", "ffn_up", "projection"} {
		if got := exported[name].DType; got != "q8" {
			t.Fatalf("%s export dtype = %q, want q8", name, got)
		}
	}
	changed := false
	for _, name := range []string{"token_embedding", "attn_q", "attn_k", "attn_v", "attn_o", "ffn_up", "projection"} {
		master := trainerMasterTensorByName(trainer, name)
		for i, value := range master.F32 {
			if abs32(value-beforeMaster[name][i]) > 1e-6 {
				changed = true
				break
			}
		}
		if changed {
			break
		}
	}
	if !changed {
		t.Fatal("expected encoder train step to update at least one exported weight tensor")
	}
}

func TestEmbeddingTrainerTrainStepSupportsRepeatedEncoderAndExportsQuantizedWeights(t *testing.T) {
	trainer := newTinyTrainableRepeatedEncoderEmbeddingTrainer(t, 0.02)
	if got := trainer.encoderRepeats(); got != 2 {
		t.Fatalf("encoder repeats = %d, want 2", got)
	}

	batch := []EmbeddingPairExample{
		{LeftTokens: []int32{0, 2}, RightTokens: []int32{1, 2}, LeftMask: []int32{1, 1}, RightMask: []int32{1, 1}, Target: 0},
		{LeftTokens: []int32{0, 0}, RightTokens: []int32{0, 2}, LeftMask: []int32{1, 1}, RightMask: []int32{1, 1}, Target: 0.5},
		{LeftTokens: []int32{1, 1}, RightTokens: []int32{1, 2}, LeftMask: []int32{1, 1}, RightMask: []int32{1, 1}, Target: 0.5},
		{LeftTokens: []int32{0, 1}, RightTokens: []int32{1, 0}, LeftMask: []int32{1, 1}, RightMask: []int32{1, 1}, Target: 0},
	}
	before, err := trainer.EvalBatch(batch)
	if err != nil {
		t.Fatalf("eval before: %v", err)
	}
	beforeProjection := append([]float32(nil), trainer.projection.F32...)
	for i := 0; i < 24; i++ {
		if _, err := trainer.TrainStep(batch); err != nil {
			t.Fatalf("train step %d: %v", i, err)
		}
	}
	after, err := trainer.EvalBatch(batch)
	if err != nil {
		t.Fatalf("eval after: %v", err)
	}
	if after.Loss > before.Loss+0.000001 {
		t.Fatalf("repeated encoder loss regressed: before=%f after=%f", before.Loss, after.Loss)
	}
	exported, err := trainer.ExportInferenceWeights()
	if err != nil {
		t.Fatalf("export weights: %v", err)
	}
	for _, name := range []string{"token_embedding", "attn_q", "attn_k", "attn_v", "attn_o", "ffn_up", "projection"} {
		if got := exported[name].DType; got != "q8" {
			t.Fatalf("%s export dtype = %q, want q8", name, got)
		}
	}
	changed := false
	for i, value := range trainer.projection.F32 {
		if abs32(value-beforeProjection[i]) > 1e-6 {
			changed = true
			break
		}
	}
	if !changed {
		t.Fatal("expected repeated encoder train step to update projection weights")
	}
}

func TestEmbeddingTrainerEncoderCheckpointRoundTrip(t *testing.T) {
	trainer := newTinyTrainableEncoderEmbeddingTrainer(t, 0.02)
	batch := tinyEncoderPairDataset()

	for i := 0; i < 4; i++ {
		if _, err := trainer.TrainStep(batch); err != nil {
			t.Fatalf("train step %d: %v", i, err)
		}
	}

	checkpoint, err := trainer.Checkpoint()
	if err != nil {
		t.Fatalf("checkpoint: %v", err)
	}
	restored, err := NewEmbeddingTrainerFromCheckpoint(trainer.module, checkpoint)
	if err != nil {
		t.Fatalf("restore checkpoint: %v", err)
	}
	t.Cleanup(restored.Close)

	if _, err := trainer.TrainStep(batch); err != nil {
		t.Fatalf("continue original trainer: %v", err)
	}
	if _, err := restored.TrainStep(batch); err != nil {
		t.Fatalf("continue restored trainer: %v", err)
	}

	exportA, err := trainer.ExportInferenceWeights()
	if err != nil {
		t.Fatalf("export original: %v", err)
	}
	exportB, err := restored.ExportInferenceWeights()
	if err != nil {
		t.Fatalf("export restored: %v", err)
	}
	for _, name := range []string{"token_embedding", "attn_q", "attn_k", "attn_v", "attn_o", "ffn_up", "projection"} {
		assertTensorClose(t, exportA[name], exportB[name].Shape, exportB[name].F32)
	}
}

func TestEmbeddingTrainerForwardMatMulAcceleratorMatchesHost(t *testing.T) {
	trainer := newTinyTrainableFFNEmbeddingTrainer(t, 0.05)
	if trainer.forwardMatMul == nil {
		t.Skip("no trainer matmul accelerator available")
	}
	if trainer.forwardBackend != mantaartifact.BackendCUDA && trainer.forwardBackend != mantaartifact.BackendMetal {
		t.Fatalf("forward backend = %q, want cuda or metal", trainer.forwardBackend)
	}
	rhs := backend.NewTensorF32([]int{2, 3}, []float32{
		1, 2, 3,
		4, 5, 6,
	})
	got, ok := trainer.tryForwardMatMul([]float32{
		1, 2,
		3, 4,
	}, 2, 2, rhs, 3)
	if !ok {
		t.Fatal("accelerated matmul was not used")
	}
	want := make([]float32, 6)
	fillHostMatMul([]float32{
		1, 2,
		3, 4,
	}, 2, 2, rhs.F32, 3, want)
	assertTensorClose(t, backend.NewTensorF32([]int{2, 3}, got), []int{2, 3}, want)
}

func TestEmbeddingTrainerTransposedMatMulAcceleratorMatchesHost(t *testing.T) {
	trainer := newTinyTrainableFFNEmbeddingTrainer(t, 0.05)
	if trainer.forwardMatMul == nil {
		t.Skip("no trainer matmul accelerator available")
	}
	lhs := []float32{
		1, 2, 3,
		4, 5, 6,
	}
	rhs := []float32{
		1, 0, 2, 1,
		0, 1, 3, 2,
	}
	got, ok := trainer.tryTrainerMatMul(lhs, 2, 3, rhs, 2, 4, true, false)
	if !ok {
		t.Fatal("accelerated transposed matmul was not used")
	}
	want := make([]float32, 12)
	fillHostMatMulTranspose(lhs, 2, 3, rhs, 2, 4, true, false, want)
	assertTensorClose(t, backend.NewTensorF32([]int{3, 4}, got), []int{3, 4}, want)
}

func TestEmbeddingTrainerBoundRightMatMulMatchesHostAndRefreshes(t *testing.T) {
	trainer := newTinyTrainableFFNEmbeddingTrainer(t, 0.05)
	if trainer.forwardMatMul == nil {
		t.Skip("no trainer matmul accelerator available")
	}
	lhs := backend.NewTensorF32([]int{2, 2}, []float32{
		1, 2,
		3, 4,
	})
	rhsA := backend.NewTensorF32([]int{2, 3}, []float32{
		1, 2, 3,
		4, 5, 6,
	})
	if err := trainer.forwardMatMul.BindMatrix(trainer.projParam.Name, rhsA); err != nil {
		t.Fatalf("bind rhsA: %v", err)
	}
	resultA, err := trainer.forwardMatMul.RunMatMulWithBoundRight(lhs, trainer.projParam.Name, mantaartifact.ValueType{
		Kind: mantaartifact.ValueTensor,
		Tensor: &mantaartifact.TensorType{
			DType: "f32",
		},
	}, false, false)
	if err != nil {
		t.Fatalf("run bound rhsA: %v", err)
	}
	wantA := make([]float32, 6)
	fillHostMatMul([]float32{
		1, 2,
		3, 4,
	}, 2, 2, rhsA.F32, 3, wantA)
	assertTensorClose(t, resultA.Outputs[0], []int{2, 3}, wantA)
	if got := resultA.Metadata["rhs_binding"]; got != trainer.projParam.Name {
		t.Fatalf("rhs_binding = %v, want %q", got, trainer.projParam.Name)
	}

	rhsB := backend.NewTensorF32([]int{2, 3}, []float32{
		2, 0, 1,
		1, 3, 2,
	})
	if err := trainer.forwardMatMul.BindMatrix(trainer.projParam.Name, rhsB); err != nil {
		t.Fatalf("bind rhsB: %v", err)
	}
	resultB, err := trainer.forwardMatMul.RunMatMulWithBoundRight(lhs, trainer.projParam.Name, mantaartifact.ValueType{
		Kind: mantaartifact.ValueTensor,
		Tensor: &mantaartifact.TensorType{
			DType: "f32",
		},
	}, false, false)
	if err != nil {
		t.Fatalf("run bound rhsB: %v", err)
	}
	wantB := make([]float32, 6)
	fillHostMatMul([]float32{
		1, 2,
		3, 4,
	}, 2, 2, rhsB.F32, 3, wantB)
	assertTensorClose(t, resultB.Outputs[0], []int{2, 3}, wantB)
}

func TestEmbeddingTrainerBoundLeftMatMulMatchesHostAndRefreshes(t *testing.T) {
	trainer := newTinyTrainableFFNEmbeddingTrainer(t, 0.05)
	if trainer.forwardMatMul == nil {
		t.Skip("no trainer matmul accelerator available")
	}
	lhsA := backend.NewTensorF32([]int{2, 3}, []float32{
		1, 2, 3,
		4, 5, 6,
	})
	rhs := backend.NewTensorF32([]int{2, 2}, []float32{
		1, 0,
		2, 1,
	})
	if err := trainer.forwardMatMul.BindMatrix(trainer.hiddenParam.Name, lhsA); err != nil {
		t.Fatalf("bind lhsA: %v", err)
	}
	resultA, err := trainer.forwardMatMul.RunMatMulWithBoundLeft(trainer.hiddenParam.Name, rhs, mantaartifact.ValueType{
		Kind: mantaartifact.ValueTensor,
		Tensor: &mantaartifact.TensorType{
			DType: "f32",
		},
	}, true, false)
	if err != nil {
		t.Fatalf("run bound lhsA: %v", err)
	}
	wantA := make([]float32, 6)
	fillHostMatMulTranspose(lhsA.F32, 2, 3, rhs.F32, 2, 2, true, false, wantA)
	assertTensorClose(t, resultA.Outputs[0], []int{3, 2}, wantA)
	if got := resultA.Metadata["lhs_binding"]; got != trainer.hiddenParam.Name {
		t.Fatalf("lhs_binding = %v, want %q", got, trainer.hiddenParam.Name)
	}

	lhsB := backend.NewTensorF32([]int{2, 3}, []float32{
		2, 1, 0,
		3, 2, 1,
	})
	if err := trainer.forwardMatMul.BindMatrix(trainer.hiddenParam.Name, lhsB); err != nil {
		t.Fatalf("bind lhsB: %v", err)
	}
	resultB, err := trainer.forwardMatMul.RunMatMulWithBoundLeft(trainer.hiddenParam.Name, rhs, mantaartifact.ValueType{
		Kind: mantaartifact.ValueTensor,
		Tensor: &mantaartifact.TensorType{
			DType: "f32",
		},
	}, true, false)
	if err != nil {
		t.Fatalf("run bound lhsB: %v", err)
	}
	wantB := make([]float32, 6)
	fillHostMatMulTranspose(lhsB.F32, 2, 3, rhs.F32, 2, 2, true, false, wantB)
	assertTensorClose(t, resultB.Outputs[0], []int{3, 2}, wantB)
}

func TestEmbeddingTrainerOptimizerAcceleratorMatchesHostAdamW(t *testing.T) {
	trainer := newTinyTrainableFFNEmbeddingTrainer(t, 0.05)
	if trainer.optimizerAccel == nil {
		t.Skip("no trainer optimizer accelerator available")
	}
	paramA := backend.NewTensorF32([]int{2, 2}, []float32{
		0.5, -0.25,
		1.0, -0.75,
	})
	mom1A := backend.NewTensorF32([]int{2, 2}, []float32{
		0.1, -0.05,
		0.2, -0.1,
	})
	mom2A := backend.NewTensorF32([]int{2, 2}, []float32{
		0.01, 0.02,
		0.03, 0.04,
	})
	grad := []float32{
		0.2, -0.1,
		0.05, -0.15,
	}
	paramB := paramA.Clone()
	mom1B := mom1A.Clone()
	mom2B := mom2A.Clone()

	cfg := trainer.optimizerUpdateConfig(0.5)
	cfg.Step = 3
	if err := trainer.optimizerAccel.ApplyUpdate(trainer.projParam.Name, cfg, paramA, mom1A, mom2A, backend.NewTensorF32([]int{2, 2}, grad)); err != nil {
		t.Fatalf("accelerated optimizer update: %v", err)
	}
	if err := trainer.optimizerAccel.SyncState(trainer.projParam.Name, paramA, mom1A, mom2A, true); err != nil {
		t.Fatalf("sync accelerated optimizer state: %v", err)
	}
	applyOptimizerUpdate(trainer.config, cfg.Step, paramB, mom1B, mom2B, grad, cfg.Scale)

	assertTensorClose(t, paramA, paramB.Shape, paramB.F32)
	assertTensorClose(t, mom1A, mom1B.Shape, mom1B.F32)
	assertTensorClose(t, mom2A, mom2B.Shape, mom2B.F32)
}

func TestEmbeddingTrainerCheckpointSyncsResidentOptimizerMoments(t *testing.T) {
	trainer := newTinyTrainableFFNEmbeddingTrainer(t, 0.05)
	if trainer.optimizerAccel == nil {
		t.Skip("no trainer optimizer accelerator available")
	}
	batch := tinyEmbeddingPairDataset()

	if _, err := trainer.TrainStep(batch); err != nil {
		t.Fatalf("train step: %v", err)
	}
	allZero := true
	for _, v := range trainer.projMom1.F32 {
		if abs32(v) > 1e-9 {
			allZero = false
			break
		}
	}
	if !allZero {
		t.Fatal("expected resident optimizer path to defer host moment sync until checkpoint")
	}

	checkpoint, err := trainer.Checkpoint()
	if err != nil {
		t.Fatalf("checkpoint: %v", err)
	}
	if checkpoint.ProjMoment1 == nil || checkpoint.ProjMoment2 == nil {
		t.Fatal("expected checkpoint to include projection moments")
	}
	nonZero := false
	for i := range checkpoint.ProjMoment1.F32 {
		if abs32(checkpoint.ProjMoment1.F32[i]) > 1e-9 || abs32(checkpoint.ProjMoment2.F32[i]) > 1e-9 {
			nonZero = true
			break
		}
	}
	if !nonZero {
		t.Fatal("expected checkpoint to sync resident optimizer moments")
	}
}

func TestTrainerActivationAccelModeFromEnv(t *testing.T) {
	for _, tc := range []struct {
		name    string
		env     map[string]string
		full    bool
		softmax bool
	}{
		{
			name: "default disabled",
		},
		{
			name: "full enables all activation backward",
			env: map[string]string{
				"MANTA_TRAIN_ENABLE_ACTIVATION_ACCEL": "1",
			},
			full:    true,
			softmax: true,
		},
		{
			name: "softmax only",
			env: map[string]string{
				"MANTA_TRAIN_ENABLE_SOFTMAX_BACKWARD_ACCEL": "1",
			},
			softmax: true,
		},
		{
			name: "global disable wins",
			env: map[string]string{
				"MANTA_TRAIN_DISABLE_ACTIVATION_ACCEL":      "1",
				"MANTA_TRAIN_ENABLE_ACTIVATION_ACCEL":       "1",
				"MANTA_TRAIN_ENABLE_SOFTMAX_BACKWARD_ACCEL": "1",
			},
		},
		{
			name: "softmax disable can narrow full mode",
			env: map[string]string{
				"MANTA_TRAIN_ENABLE_ACTIVATION_ACCEL":        "1",
				"MANTA_TRAIN_DISABLE_SOFTMAX_BACKWARD_ACCEL": "1",
			},
			full: true,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv("MANTA_TRAIN_DISABLE_ACTIVATION_ACCEL", "")
			t.Setenv("MANTA_TRAIN_ENABLE_ACTIVATION_ACCEL", "")
			t.Setenv("MANTA_TRAIN_ENABLE_SOFTMAX_BACKWARD_ACCEL", "")
			t.Setenv("MANTA_TRAIN_DISABLE_SOFTMAX_BACKWARD_ACCEL", "")
			for name, value := range tc.env {
				t.Setenv(name, value)
			}
			got := trainerActivationAccelModeFromEnv()
			if got.fullBackward != tc.full {
				t.Fatalf("full backward = %v, want %v", got.fullBackward, tc.full)
			}
			if got.softmaxBackward != tc.softmax {
				t.Fatalf("softmax backward = %v, want %v", got.softmaxBackward, tc.softmax)
			}
		})
	}
}

func TestFastGELUApproximationIsOptIn(t *testing.T) {
	t.Setenv("MANTA_TRAIN_ENABLE_FAST_GELU", "")
	if fastGELUEnabled() {
		t.Fatal("fast GELU enabled by default")
	}
	t.Setenv("MANTA_TRAIN_ENABLE_FAST_GELU", "1")
	if !fastGELUEnabled() {
		t.Fatal("fast GELU env did not enable approximation")
	}
	x := float32(1.25)
	if geluForwardMode(x, true) == geluForwardMode(x, false) {
		t.Fatal("fast GELU forward unexpectedly matched precise path exactly")
	}
	if geluBackwardMode(x, true) == geluBackwardMode(x, false) {
		t.Fatal("fast GELU backward unexpectedly matched precise path exactly")
	}
	if fastTanh(4) != 1 || fastTanh(-4) != -1 {
		t.Fatal("fast tanh did not clamp outside approximation range")
	}
}

func TestEmbeddingTrainerGELUBackwardAcceleratorMatchesHost(t *testing.T) {
	t.Setenv("MANTA_TRAIN_ENABLE_ACTIVATION_ACCEL", "1")
	trainer := newTinyTrainableFFNEmbeddingTrainer(t, 0.05)
	if trainer.activationAccel == nil {
		t.Skip("no trainer activation accelerator available")
	}
	gradOut := []float32{
		0.2, -0.1, 0.05,
		-0.25, 0.4, -0.3,
	}
	preAct := []float32{
		-1.0, -0.5, 0.0,
		0.5, 1.0, 1.5,
	}
	got, ok := trainer.tryGELUBackwardMul(gradOut, preAct, 2, 3, "")
	if !ok {
		t.Fatal("accelerated gelu backward was not used")
	}
	want := make([]float32, len(gradOut))
	for i := range want {
		want[i] = gradOut[i] * geluBackward(preAct[i])
	}
	assertTensorClose(t, backend.NewTensorF32([]int{2, 3}, got), []int{2, 3}, want)
}

func TestEmbeddingTrainerActivationAccelShapeLimitSkipsLargeUnboundCalls(t *testing.T) {
	t.Setenv("MANTA_TRAIN_ACTIVATION_ACCEL_MAX_ELEMENTS", "4")
	activation := &countingActivationAccelerator{}
	trainer := &EmbeddingTrainer{
		activationAccel:      activation,
		activationAccelFull:  true,
		softmaxBackwardAccel: true,
	}
	grad := [][]float32{
		{0.2, -0.1, 0.05},
		{-0.25, 0.4, -0.3},
	}
	pre := [][]float32{
		{-1.0, -0.5, 0.0},
		{0.5, 1.0, 1.5},
	}
	if _, ok := trainer.tryBatchedGELUBackwardMul(grad, pre, 1, 3); ok {
		t.Fatal("expected large unbound activation call to fall back to host")
	}
	if activation.geluBackwardCalls != 0 {
		t.Fatalf("gelu backward calls = %d, want 0", activation.geluBackwardCalls)
	}

	t.Setenv("MANTA_TRAIN_ACTIVATION_ACCEL_MAX_ELEMENTS", "0")
	if _, ok := trainer.tryBatchedGELUBackwardMul(grad, pre, 1, 3); !ok {
		t.Fatal("expected unlimited activation shape limit to allow accelerator")
	}
	if activation.geluBackwardCalls != 1 {
		t.Fatalf("gelu backward calls = %d, want 1", activation.geluBackwardCalls)
	}
}

func TestEmbeddingTrainerActivationAccelShapeLimitAllowsBoundInputs(t *testing.T) {
	t.Setenv("MANTA_TRAIN_ACTIVATION_ACCEL_MAX_ELEMENTS", "4")
	activation := &countingActivationAccelerator{}
	trainer := &EmbeddingTrainer{
		activationAccel:     activation,
		activationAccelFull: true,
	}
	pre := []float32{-1.0, -0.5, 0.0, 0.5, 1.0, 1.5}
	if err := activation.BindTensor("pre", backend.NewTensorF32([]int{2, 3}, pre)); err != nil {
		t.Fatalf("bind pre: %v", err)
	}
	grad := []float32{0.2, -0.1, 0.05, -0.25, 0.4, -0.3}
	if _, ok := trainer.tryGELUBackwardMul(grad, nil, 2, 3, "pre"); !ok {
		t.Fatal("expected bound activation input to bypass unbound shape limit")
	}
	if activation.geluBackwardCalls != 1 {
		t.Fatalf("gelu backward calls = %d, want 1", activation.geluBackwardCalls)
	}
}

func TestEmbeddingTrainerSoftmaxBackwardAcceleratorMatchesHost(t *testing.T) {
	t.Setenv("MANTA_TRAIN_ENABLE_SOFTMAX_BACKWARD_ACCEL", "1")
	trainer := newTinyTrainableAttentionEmbeddingTrainer(t, 0.05)
	if trainer.activationAccel == nil {
		t.Skip("no trainer activation accelerator available")
	}
	if trainer.activationAccelFull {
		t.Fatal("softmax-only env unexpectedly enabled full activation backward")
	}
	if !trainer.softmaxBackwardAccel {
		t.Fatal("softmax-only env did not enable softmax backward acceleration")
	}
	gradOut := []float32{
		0.3, -0.1,
		-0.2, 0.4,
	}
	probs := []float32{
		0.7, 0.3,
		0.25, 0.75,
	}
	got, ok := trainer.trySoftmaxBackwardRows(gradOut, probs, 2, 2, "")
	if !ok {
		t.Fatal("accelerated softmax backward was not used")
	}
	want := make([]float32, len(gradOut))
	for row := 0; row < 2; row++ {
		backwardSoftmaxRow(want[row*2:(row+1)*2], gradOut[row*2:(row+1)*2], probs[row*2:(row+1)*2])
	}
	assertTensorClose(t, backend.NewTensorF32([]int{2, 2}, got), []int{2, 2}, want)
	if _, ok := trainer.tryGELUBackwardMul(gradOut, probs, 2, 2, ""); ok {
		t.Fatal("softmax-only env unexpectedly enabled gelu backward acceleration")
	}
}

func TestEmbeddingTrainerLayerNormBackwardAcceleratorMatchesHost(t *testing.T) {
	t.Setenv("MANTA_TRAIN_ENABLE_ACTIVATION_ACCEL", "1")
	trainer := newTinyTrainableEncoderEmbeddingTrainer(t, 0.05)
	if trainer.activationAccel == nil {
		t.Skip("no trainer activation accelerator available")
	}
	gradOut := []float32{
		0.2, -0.1, 0.3,
		-0.4, 0.25, 0.15,
	}
	pre := []float32{
		1.2, -0.4, 0.1,
		0.5, 1.0, -0.5,
	}
	normalized := make([]float32, len(pre))
	for row := 0; row < 2; row++ {
		layerNormRow(normalized[row*3:(row+1)*3], pre[row*3:(row+1)*3])
	}
	got, ok := trainer.tryLayerNormBackwardRows(gradOut, normalized, pre, 2, 3, "", "")
	if !ok {
		t.Fatal("accelerated layernorm backward was not used")
	}
	want := make([]float32, len(gradOut))
	for row := 0; row < 2; row++ {
		backwardLayerNormRow(
			want[row*3:(row+1)*3],
			gradOut[row*3:(row+1)*3],
			normalized[row*3:(row+1)*3],
			pre[row*3:(row+1)*3],
		)
	}
	assertTensorClose(t, backend.NewTensorF32([]int{2, 3}, got), []int{2, 3}, want)
}

func TestEmbeddingTrainerBatchedGELUBackwardAcceleratorMatchesHost(t *testing.T) {
	t.Setenv("MANTA_TRAIN_ENABLE_ACTIVATION_ACCEL", "1")
	trainer := newTinyTrainableFFNEmbeddingTrainer(t, 0.05)
	if trainer.activationAccel == nil {
		t.Skip("no trainer activation accelerator available")
	}
	gradOut := [][]float32{
		{
			0.2, -0.1, 0.05,
			-0.25, 0.4, -0.3,
		},
		{
			-0.15, 0.35, -0.05,
			0.45, -0.2, 0.1,
		},
	}
	preAct := [][]float32{
		{
			-1.0, -0.5, 0.0,
			0.5, 1.0, 1.5,
		},
		{
			1.25, -1.25, 0.75,
			-0.75, 0.25, -0.25,
		},
	}
	got, ok := trainer.tryBatchedGELUBackwardMul(gradOut, preAct, 2, 3)
	if !ok {
		t.Fatal("accelerated batched gelu backward was not used")
	}
	if len(got) != len(gradOut) {
		t.Fatalf("batched gelu outputs = %d, want %d", len(got), len(gradOut))
	}
	for batch := range got {
		want := make([]float32, len(gradOut[batch]))
		for i := range want {
			want[i] = gradOut[batch][i] * geluBackward(preAct[batch][i])
		}
		assertTensorClose(t, backend.NewTensorF32([]int{2, 3}, got[batch]), []int{2, 3}, want)
	}
}

func TestEmbeddingTrainerBatchedSoftmaxBackwardAcceleratorMatchesHost(t *testing.T) {
	t.Setenv("MANTA_TRAIN_ENABLE_ACTIVATION_ACCEL", "1")
	trainer := newTinyTrainableAttentionEmbeddingTrainer(t, 0.05)
	if trainer.activationAccel == nil {
		t.Skip("no trainer activation accelerator available")
	}
	gradOut := [][]float32{
		{
			0.3, -0.1,
			-0.2, 0.4,
		},
		{
			0.1, -0.25,
			0.35, -0.15,
		},
	}
	probs := [][]float32{
		{
			0.7, 0.3,
			0.25, 0.75,
		},
		{
			0.6, 0.4,
			0.1, 0.9,
		},
	}
	got, ok := trainer.tryBatchedSoftmaxBackwardRows(gradOut, probs, 2, 2)
	if !ok {
		t.Fatal("accelerated batched softmax backward was not used")
	}
	if len(got) != len(gradOut) {
		t.Fatalf("batched softmax outputs = %d, want %d", len(got), len(gradOut))
	}
	for batch := range got {
		want := make([]float32, len(gradOut[batch]))
		for row := 0; row < 2; row++ {
			backwardSoftmaxRow(want[row*2:(row+1)*2], gradOut[batch][row*2:(row+1)*2], probs[batch][row*2:(row+1)*2])
		}
		assertTensorClose(t, backend.NewTensorF32([]int{2, 2}, got[batch]), []int{2, 2}, want)
	}
}

func TestEmbeddingTrainerBatchedLayerNormBackwardAcceleratorMatchesHost(t *testing.T) {
	t.Setenv("MANTA_TRAIN_ENABLE_ACTIVATION_ACCEL", "1")
	trainer := newTinyTrainableEncoderEmbeddingTrainer(t, 0.05)
	if trainer.activationAccel == nil {
		t.Skip("no trainer activation accelerator available")
	}
	gradOut := [][]float32{
		{
			0.2, -0.1, 0.3,
			-0.4, 0.25, 0.15,
		},
		{
			-0.35, 0.05, 0.45,
			0.1, -0.3, 0.2,
		},
	}
	pre := [][]float32{
		{
			1.2, -0.4, 0.1,
			0.5, 1.0, -0.5,
		},
		{
			-1.0, 0.75, 0.25,
			1.5, -0.25, 0.0,
		},
	}
	normalized := make([][]float32, len(pre))
	for batch := range pre {
		normalized[batch] = make([]float32, len(pre[batch]))
		for row := 0; row < 2; row++ {
			layerNormRow(normalized[batch][row*3:(row+1)*3], pre[batch][row*3:(row+1)*3])
		}
	}
	got, ok := trainer.tryBatchedLayerNormBackwardRows(gradOut, normalized, pre, 2, 3)
	if !ok {
		t.Fatal("accelerated batched layernorm backward was not used")
	}
	if len(got) != len(gradOut) {
		t.Fatalf("batched layernorm outputs = %d, want %d", len(got), len(gradOut))
	}
	for batch := range got {
		want := make([]float32, len(gradOut[batch]))
		for row := 0; row < 2; row++ {
			backwardLayerNormRow(
				want[row*3:(row+1)*3],
				gradOut[batch][row*3:(row+1)*3],
				normalized[batch][row*3:(row+1)*3],
				pre[batch][row*3:(row+1)*3],
			)
		}
		assertTensorClose(t, backend.NewTensorF32([]int{2, 3}, got[batch]), []int{2, 3}, want)
	}
}

func TestEmbeddingTrainerBatchedForwardSkipsUnusedActivationBindings(t *testing.T) {
	trainer := newTinyTrainableEncoderEmbeddingTrainer(t, 0.05)
	if trainer.forwardMatMul != nil {
		trainer.forwardMatMul.Close()
	}
	matmul := &countingMatMulAccelerator{}
	activation := &countingActivationAccelerator{}
	trainer.forwardMatMul = matmul
	trainer.activationAccel = activation
	trainer.activationAccelFull = true
	trainer.softmaxBackwardAccel = true

	forward := trainer.prepareForwardWeights()
	trainer.primeForwardWeightResidency(forward.attnQ, forward.attnK, forward.attnV, forward.attnO, forward.hidden, forward.proj)
	batch := []EmbeddingContrastiveExample{
		{QueryTokens: []int32{0, 2}, PositiveTokens: []int32{0, 0}, QueryMask: []int32{1, 1}, PositiveMask: []int32{1, 1}},
		{QueryTokens: []int32{1, 2}, PositiveTokens: []int32{1, 1}, QueryMask: []int32{1, 1}, PositiveMask: []int32{1, 1}},
	}
	queries, positives, err := trainer.encodeContrastiveBatch(batch, forward, true)
	if err != nil {
		t.Fatalf("encode contrastive batch: %v", err)
	}
	defer trainer.releaseEncodedSequences(queries)
	defer trainer.releaseEncodedSequences(positives)
	if activation.bindCalls != 0 {
		t.Fatalf("activation bind calls = %d, want 0 unused per-sequence activation binds in batched forward", activation.bindCalls)
	}
	if matmul.boundRightRuns == 0 {
		t.Fatal("expected batched forward to keep using matmul acceleration")
	}
}

func TestEmbeddingTrainerBatchedForwardSkipsSingletonActivationBindings(t *testing.T) {
	trainer := newTinyTrainableEncoderEmbeddingTrainer(t, 0.05)
	if trainer.forwardMatMul != nil {
		trainer.forwardMatMul.Close()
	}
	trainer.forwardMatMul = &countingMatMulAccelerator{}
	activation := &countingActivationAccelerator{}
	trainer.activationAccel = activation
	trainer.activationAccelFull = true
	trainer.softmaxBackwardAccel = true

	forward := trainer.prepareForwardWeights()
	trainer.primeForwardWeightResidency(forward.attnQ, forward.attnK, forward.attnV, forward.attnO, forward.hidden, forward.proj)
	batch := []EmbeddingContrastiveExample{
		{QueryTokens: []int32{0}, PositiveTokens: []int32{0, 0}, QueryMask: []int32{1}, PositiveMask: []int32{1, 1}},
	}
	queries, positives, err := trainer.encodeContrastiveBatch(batch, forward, true)
	if err != nil {
		t.Fatalf("encode contrastive batch: %v", err)
	}
	defer trainer.releaseEncodedSequences(queries)
	defer trainer.releaseEncodedSequences(positives)
	if activation.bindCalls != 0 {
		t.Fatalf("activation bind calls = %d, want 0 unused singleton activation binds in batched forward", activation.bindCalls)
	}
}

func TestEmbeddingTrainerBatchedForwardKeepsActivationBindingsWhenBatchedBackwardDisabled(t *testing.T) {
	t.Setenv("MANTA_TRAIN_DISABLE_BATCHED_BACKWARD", "1")
	trainer := newTinyTrainableEncoderEmbeddingTrainer(t, 0.05)
	if trainer.forwardMatMul != nil {
		trainer.forwardMatMul.Close()
	}
	trainer.forwardMatMul = &countingMatMulAccelerator{}
	activation := &countingActivationAccelerator{}
	trainer.activationAccel = activation
	trainer.activationAccelFull = true
	trainer.softmaxBackwardAccel = true

	forward := trainer.prepareForwardWeights()
	trainer.primeForwardWeightResidency(forward.attnQ, forward.attnK, forward.attnV, forward.attnO, forward.hidden, forward.proj)
	batch := []EmbeddingContrastiveExample{
		{QueryTokens: []int32{0, 2}, PositiveTokens: []int32{0, 0}, QueryMask: []int32{1, 1}, PositiveMask: []int32{1, 1}},
		{QueryTokens: []int32{1, 2}, PositiveTokens: []int32{1, 1}, QueryMask: []int32{1, 1}, PositiveMask: []int32{1, 1}},
	}
	queries, positives, err := trainer.encodeContrastiveBatch(batch, forward, true)
	if err != nil {
		t.Fatalf("encode contrastive batch: %v", err)
	}
	defer trainer.releaseEncodedSequences(queries)
	defer trainer.releaseEncodedSequences(positives)
	if activation.bindCalls == 0 {
		t.Fatal("expected activation bindings when batched backward is disabled")
	}
}

func TestEmbeddingTrainerBatchedBackwardSkipsSingletonUnboundActivationKernels(t *testing.T) {
	trainer := newTinyTrainableEncoderEmbeddingTrainer(t, 0.05)
	if trainer.forwardMatMul != nil {
		trainer.forwardMatMul.Close()
	}
	trainer.forwardMatMul = &countingMatMulAccelerator{}
	activation := &countingActivationAccelerator{}
	trainer.activationAccel = activation
	trainer.activationAccelFull = true
	trainer.softmaxBackwardAccel = true

	batch := []EmbeddingContrastiveExample{
		{QueryTokens: []int32{0}, PositiveTokens: []int32{0}, QueryMask: []int32{1}, PositiveMask: []int32{1}},
		{QueryTokens: []int32{1}, PositiveTokens: []int32{1, 2}, QueryMask: []int32{1}, PositiveMask: []int32{1, 1}},
	}
	if _, err := trainer.TrainContrastiveStep(batch); err != nil {
		t.Fatalf("train contrastive step: %v", err)
	}
	if activation.geluBackwardCalls != 1 {
		t.Fatalf("gelu backward calls = %d, want 1 grouped batched call", activation.geluBackwardCalls)
	}
	if activation.softmaxBackwardCalls != 1 {
		t.Fatalf("softmax backward calls = %d, want 1 grouped batched call", activation.softmaxBackwardCalls)
	}
	if activation.layerNormBackwardCalls != 2 {
		t.Fatalf("layernorm backward calls = %d, want 2 grouped batched calls", activation.layerNormBackwardCalls)
	}
}

func TestEmbeddingTrainerSingleForwardKeepsActivationBindings(t *testing.T) {
	trainer := newTinyTrainableEncoderEmbeddingTrainer(t, 0.05)
	activation := &countingActivationAccelerator{}
	trainer.activationAccel = activation
	trainer.activationAccelFull = true
	trainer.softmaxBackwardAccel = true

	forward := trainer.prepareForwardWeights()
	mask, err := trainer.prepareMask([]int32{0, 2}, nil)
	if err != nil {
		t.Fatalf("prepare mask: %v", err)
	}
	seq, err := trainer.encodeSequence([]int32{0, 2}, mask, forward.token, forward.attnQ, forward.attnK, forward.attnV, forward.attnO, forward.hidden, forward.proj, true)
	if err != nil {
		t.Fatalf("encode sequence: %v", err)
	}
	defer trainer.releaseEncodedSequenceBindings(seq)
	if activation.bindCalls == 0 {
		t.Fatal("expected single-sequence forward to keep activation bindings for bound backward paths")
	}
}

func TestEmbeddingTrainerAttentionActivationsBindAndRelease(t *testing.T) {
	t.Setenv("MANTA_TRAIN_ENABLE_SEQUENCE_MATMUL_BINDINGS", "1")
	trainer := newTinyTrainableAttentionEmbeddingTrainer(t, 0.05)
	if trainer.forwardMatMul == nil {
		t.Skip("no trainer matmul accelerator available")
	}
	tokenForward := forwardTensorForParam(trainer.tokenParam, trainer.tokenEmbed, trainer.config.WeightBits)
	attnQForward := forwardTensorForParam(trainer.attnQParam, trainer.attentionQuery, trainer.config.WeightBits)
	attnKForward := forwardTensorForParam(trainer.attnKParam, trainer.attentionKey, trainer.config.WeightBits)
	attnVForward := forwardTensorForParam(trainer.attnVParam, trainer.attentionValue, trainer.config.WeightBits)
	attnOForward := forwardTensorForParam(trainer.attnOParam, trainer.attentionOutput, trainer.config.WeightBits)
	projForward := forwardTensorForParam(trainer.projParam, trainer.projection, trainer.config.WeightBits)
	trainer.primeForwardWeightResidency(attnQForward, attnKForward, attnVForward, attnOForward, nil, projForward)

	mask, err := trainer.prepareMask([]int32{0, 2}, nil)
	if err != nil {
		t.Fatalf("prepare mask: %v", err)
	}
	state, err := trainer.encodeSequence([]int32{0, 2}, mask, tokenForward, attnQForward, attnKForward, attnVForward, attnOForward, nil, projForward, true)
	if err != nil {
		t.Fatalf("encode sequence: %v", err)
	}
	layer := state.finalLayer()
	if layer == nil {
		t.Fatal("expected final attention layer")
	}
	if layer.inputBinding == "" || layer.hiddenBinding == "" || layer.attnQBinding == "" || layer.attnKBinding == "" || layer.attnVBinding == "" || layer.attnScoresBinding == "" || layer.attnMixedBinding == "" {
		t.Fatalf("expected attention activation bindings, got input=%q hidden=%q q=%q k=%q v=%q scores=%q mixed=%q", layer.inputBinding, layer.hiddenBinding, layer.attnQBinding, layer.attnKBinding, layer.attnVBinding, layer.attnScoresBinding, layer.attnMixedBinding)
	}
	result, err := trainer.forwardMatMul.RunMatMulWithBoundRight(
		backend.NewTensorF32([]int{2, 2}, layer.attnQ),
		layer.attnKBinding,
		mantaartifact.ValueType{Kind: mantaartifact.ValueTensor, Tensor: &mantaartifact.TensorType{DType: "f32"}},
		false,
		true,
	)
	if err != nil {
		t.Fatalf("run with bound attention key: %v", err)
	}
	wantScores := make([]float32, 4)
	fillHostMatMulTranspose(layer.attnQ, 2, 2, layer.attnK, 2, 2, false, true, wantScores)
	assertTensorClose(t, result.Outputs[0], []int{2, 2}, wantScores)
	leftResult, err := trainer.forwardMatMul.RunMatMulWithBoundLeft(
		layer.inputBinding,
		backend.NewTensorF32([]int{2, 2}, layer.attnQ),
		mantaartifact.ValueType{Kind: mantaartifact.ValueTensor, Tensor: &mantaartifact.TensorType{DType: "f32"}},
		true,
		false,
	)
	if err != nil {
		t.Fatalf("run with bound input activation: %v", err)
	}
	wantGrad := make([]float32, 4)
	fillHostMatMulTranspose(layer.input, 2, 2, layer.attnQ, 2, 2, true, false, wantGrad)
	assertTensorClose(t, leftResult.Outputs[0], []int{2, 2}, wantGrad)
	trainer.releaseEncodedSequenceBindings(state)
	if layer.inputBinding != "" || layer.hiddenBinding != "" || layer.attnQBinding != "" || layer.attnKBinding != "" || layer.attnVBinding != "" || layer.attnScoresBinding != "" || layer.attnMixedBinding != "" {
		t.Fatalf("expected bindings released, got input=%q hidden=%q q=%q k=%q v=%q scores=%q mixed=%q", layer.inputBinding, layer.hiddenBinding, layer.attnQBinding, layer.attnKBinding, layer.attnVBinding, layer.attnScoresBinding, layer.attnMixedBinding)
	}
	if _, err := trainer.forwardMatMul.RunMatMulWithBoundRight(
		backend.NewTensorF32([]int{2, 2}, layer.attnQ),
		"seq_missing_k",
		mantaartifact.ValueType{Kind: mantaartifact.ValueTensor, Tensor: &mantaartifact.TensorType{DType: "f32"}},
		false,
		true,
	); err == nil {
		t.Fatal("expected missing bound-right activation error")
	}
	if _, err := trainer.forwardMatMul.RunMatMulWithBoundLeft(
		"seq_missing_input",
		backend.NewTensorF32([]int{2, 2}, layer.attnQ),
		mantaartifact.ValueType{Kind: mantaartifact.ValueTensor, Tensor: &mantaartifact.TensorType{DType: "f32"}},
		true,
		false,
	); err == nil {
		t.Fatal("expected missing bound-left activation error")
	}
}

func TestEmbeddingTrainerFFNActivationsBindAndRelease(t *testing.T) {
	t.Setenv("MANTA_TRAIN_ENABLE_SEQUENCE_MATMUL_BINDINGS", "1")
	trainer := newTinyTrainableFFNEmbeddingTrainer(t, 0.05)
	if trainer.forwardMatMul == nil {
		t.Skip("no trainer matmul accelerator available")
	}
	tokenForward := forwardTensorForParam(trainer.tokenParam, trainer.tokenEmbed, trainer.config.WeightBits)
	hiddenForward := forwardTensorForParam(trainer.hiddenParam, trainer.hiddenProjection, trainer.config.WeightBits)
	projForward := forwardTensorForParam(trainer.projParam, trainer.projection, trainer.config.WeightBits)
	trainer.primeForwardWeightResidency(nil, nil, nil, nil, hiddenForward, projForward)

	mask, err := trainer.prepareMask([]int32{0, 2}, nil)
	if err != nil {
		t.Fatalf("prepare mask: %v", err)
	}
	state, err := trainer.encodeSequence([]int32{0, 2}, mask, tokenForward, nil, nil, nil, nil, hiddenForward, projForward, true)
	if err != nil {
		t.Fatalf("encode sequence: %v", err)
	}
	layer := state.finalLayer()
	if layer == nil {
		t.Fatal("expected final ffn layer")
	}
	if layer.inputBinding == "" || layer.hiddenBinding == "" || layer.activatedBinding == "" {
		t.Fatalf("expected ffn activation bindings, got input=%q hidden=%q activated=%q", layer.inputBinding, layer.hiddenBinding, layer.activatedBinding)
	}
	result, err := trainer.forwardMatMul.RunMatMulWithBoundLeft(
		layer.activatedBinding,
		backend.NewTensorF32([]int{2, 2}, []float32{
			1, 0,
			0, 1,
		}),
		mantaartifact.ValueType{Kind: mantaartifact.ValueTensor, Tensor: &mantaartifact.TensorType{DType: "f32"}},
		true,
		false,
	)
	if err != nil {
		t.Fatalf("run with bound activated state: %v", err)
	}
	want := make([]float32, 6)
	fillHostMatMulTranspose(layer.activated, 2, 3, []float32{
		1, 0,
		0, 1,
	}, 2, 2, true, false, want)
	assertTensorClose(t, result.Outputs[0], []int{3, 2}, want)
	trainer.releaseEncodedSequenceBindings(state)
	if layer.inputBinding != "" || layer.hiddenBinding != "" || layer.activatedBinding != "" {
		t.Fatalf("expected ffn bindings released, got input=%q hidden=%q activated=%q", layer.inputBinding, layer.hiddenBinding, layer.activatedBinding)
	}
}

func TestEmbeddingTrainerForwardWeightCacheReusesTensorsUntilUpdate(t *testing.T) {
	trainer := newTinyTrainableAttentionEmbeddingTrainer(t, 0.05)
	first := trainer.prepareForwardWeights()
	second := trainer.prepareForwardWeights()
	if first == nil || second == nil {
		t.Fatal("expected cached forward weights")
	}
	if first != second {
		t.Fatal("expected prepareForwardWeights to reuse cached forward weight bundle")
	}
	if first.token != second.token || first.attnQ != second.attnQ || first.attnK != second.attnK || first.attnV != second.attnV || first.attnO != second.attnO || first.proj != second.proj {
		t.Fatal("expected cached forward tensors to be reused")
	}

	trainer.applyOptimizerUpdate(trainer.projParam.Name, trainer.projection, trainer.projMom1, trainer.projMom2, make([]float32, len(trainer.projection.F32)), 1)

	third := trainer.prepareForwardWeights()
	if third == nil {
		t.Fatal("expected forward weights after update")
	}
	if third != first {
		t.Fatal("expected optimizer update to refresh cached forward weight bundle in place")
	}
	if third.proj != first.proj {
		t.Fatal("expected projection forward tensor to be refreshed in place")
	}
}

func TestEmbeddingTrainerPrimeForwardWeightResidencySkipsRedundantBinds(t *testing.T) {
	trainer := newTinyTrainableAttentionEmbeddingTrainer(t, 0.05)
	if trainer.forwardMatMul != nil {
		trainer.forwardMatMul.Close()
	}
	fake := &countingMatMulAccelerator{}
	trainer.forwardMatMul = fake

	forward := trainer.prepareForwardWeights()
	trainer.primeForwardWeightResidency(forward.attnQ, forward.attnK, forward.attnV, forward.attnO, forward.hidden, forward.proj)
	firstCalls := fake.bindCalls
	if firstCalls != 5 {
		t.Fatalf("initial bind calls = %d, want 5", firstCalls)
	}

	trainer.primeForwardWeightResidency(forward.attnQ, forward.attnK, forward.attnV, forward.attnO, forward.hidden, forward.proj)
	if fake.bindCalls != firstCalls {
		t.Fatalf("redundant bind calls = %d, want %d", fake.bindCalls, firstCalls)
	}
	stats := trainer.ForwardResidencyStats()
	if stats.BindSkips != 1 {
		t.Fatalf("bind skips = %d, want 1", stats.BindSkips)
	}
	if stats.MatMul.BindCalls != int64(firstCalls) {
		t.Fatalf("backend bind calls = %d, want %d", stats.MatMul.BindCalls, firstCalls)
	}

	trainer.applyOptimizerUpdate(trainer.projParam.Name, trainer.projection, trainer.projMom1, trainer.projMom2, make([]float32, len(trainer.projection.F32)), 1)
	forward = trainer.prepareForwardWeights()
	trainer.primeForwardWeightResidency(forward.attnQ, forward.attnK, forward.attnV, forward.attnO, forward.hidden, forward.proj)
	if fake.bindCalls != firstCalls+5 {
		t.Fatalf("rebind calls after invalidation = %d, want %d", fake.bindCalls, firstCalls+5)
	}
	stats = trainer.ForwardResidencyStats()
	if stats.BindSkips != 1 {
		t.Fatalf("bind skips after invalidation = %d, want 1", stats.BindSkips)
	}
	if stats.MatMul.BindCalls != int64(firstCalls+5) {
		t.Fatalf("backend bind calls after invalidation = %d, want %d", stats.MatMul.BindCalls, firstCalls+5)
	}
}

func TestEmbeddingTrainerEvaluatePairsSkipsSequenceBindingChurn(t *testing.T) {
	trainer := newTinyTrainableAttentionEmbeddingTrainer(t, 0.05)
	if trainer.forwardMatMul != nil {
		trainer.forwardMatMul.Close()
	}
	fake := &countingMatMulAccelerator{}
	trainer.forwardMatMul = fake

	batch := []EmbeddingPairExample{
		{LeftTokens: []int32{0, 2}, RightTokens: []int32{0, 2}, LeftMask: []int32{1, 1}, RightMask: []int32{1, 1}, Target: 1},
	}
	if _, err := trainer.EvaluatePairs(batch); err != nil {
		t.Fatalf("evaluate pairs: %v", err)
	}
	if fake.bindCalls != 5 {
		t.Fatalf("bind calls = %d, want only 5 forward-weight binds", fake.bindCalls)
	}
	stats := trainer.ForwardResidencyStats()
	if stats.MatMul.BindCalls != 5 {
		t.Fatalf("forward residency bind calls = %d, want 5", stats.MatMul.BindCalls)
	}
}

func TestEmbeddingTrainerEvaluatePairsUsesBatchedForwardChunks(t *testing.T) {
	t.Setenv("MANTA_TRAIN_PAIR_EVAL_BATCH_SIZE", "2")
	trainer := newTinyTrainableAttentionEmbeddingTrainer(t, 0.05)
	if trainer.forwardMatMul != nil {
		trainer.forwardMatMul.Close()
	}
	fake := &countingMatMulAccelerator{}
	trainer.forwardMatMul = fake

	batch := []EmbeddingPairExample{
		{LeftTokens: []int32{0, 2}, RightTokens: []int32{1, 2}, LeftMask: []int32{1, 1}, RightMask: []int32{1, 1}, Target: 1},
		{LeftTokens: []int32{2, 0}, RightTokens: []int32{2, 1}, LeftMask: []int32{1, 1}, RightMask: []int32{1, 1}, Target: 1},
		{LeftTokens: []int32{0, 2}, RightTokens: []int32{1, 2}, LeftMask: []int32{1, 1}, RightMask: []int32{1, 1}, Target: -1},
	}
	metrics, err := trainer.EvaluatePairs(batch)
	if err != nil {
		t.Fatalf("evaluate pairs: %v", err)
	}
	if metrics.PairCount != len(batch) {
		t.Fatalf("pair count = %d, want %d", metrics.PairCount, len(batch))
	}
	if trainer.sequenceBindingID != 0 {
		t.Fatalf("sequence binding count = %d, want default sequence matmul bindings disabled", trainer.sequenceBindingID)
	}
	if fake.bindCalls != 5 {
		t.Fatalf("bind calls = %d, want only 5 forward-weight binds", fake.bindCalls)
	}
	if fake.multiBoundRuns == 0 {
		t.Fatal("expected pairwise eval to coalesce q/k/v bound-right matmuls")
	}
	if fake.maxBoundRightRows < 8 {
		t.Fatalf("max bound-right lhs rows = %d, want chunked pairwise eval rows", fake.maxBoundRightRows)
	}
}

func TestPairwiseEvalBatchSizeDefaultAndEnv(t *testing.T) {
	t.Setenv("MANTA_TRAIN_PAIR_EVAL_BATCH_SIZE", "")
	if got := pairwiseEvalBatchSize(1024); got != 512 {
		t.Fatalf("default pairwise eval batch size = %d, want 512", got)
	}
	if got := pairwiseEvalBatchSize(128); got != 128 {
		t.Fatalf("capped pairwise eval batch size = %d, want total size", got)
	}
	t.Setenv("MANTA_TRAIN_PAIR_EVAL_BATCH_SIZE", "64")
	if got := pairwiseEvalBatchSize(1024); got != 64 {
		t.Fatalf("env pairwise eval batch size = %d, want 64", got)
	}
}

func TestEmbeddingTrainerTrainContrastiveStepEncodesEachSequenceOncePerBatch(t *testing.T) {
	trainer := newTinyTrainableAttentionEmbeddingTrainer(t, 0.05)
	if trainer.forwardMatMul != nil {
		trainer.forwardMatMul.Close()
	}
	fake := &countingMatMulAccelerator{}
	trainer.forwardMatMul = fake
	batch := []EmbeddingContrastiveExample{
		{QueryTokens: []int32{0, 2}, PositiveTokens: []int32{0, 0}, QueryMask: []int32{1, 1}, PositiveMask: []int32{1, 1}},
		{QueryTokens: []int32{1, 2}, PositiveTokens: []int32{1, 1}, QueryMask: []int32{1, 1}, PositiveMask: []int32{1, 1}},
		{QueryTokens: []int32{2, 0}, PositiveTokens: []int32{2, 2}, QueryMask: []int32{1, 1}, PositiveMask: []int32{1, 1}},
	}

	forward := trainer.prepareForwardWeights()
	trainer.primeForwardWeightResidency(forward.attnQ, forward.attnK, forward.attnV, forward.attnO, forward.hidden, forward.proj)
	queries, positives, err := trainer.encodeContrastiveBatch(batch, forward, true)
	if err != nil {
		t.Fatalf("encode contrastive batch: %v", err)
	}
	defer trainer.releaseEncodedSequences(queries)
	defer trainer.releaseEncodedSequences(positives)
	if trainer.sequenceBindingID != 0 {
		t.Fatalf("sequence binding count = %d, want default sequence matmul bindings disabled", trainer.sequenceBindingID)
	}
	if fake.boundRightRuns == 0 {
		t.Fatalf("batched forward path did not attempt bound-right matmul")
	}
	if fake.multiBoundRuns == 0 {
		t.Fatalf("batched forward path did not coalesce q/k/v bound-right matmuls")
	}
	if fake.maxBoundRightRows < 6 {
		t.Fatalf("max bound-right lhs rows = %d, want batched rows", fake.maxBoundRightRows)
	}
}

func TestEmbeddingTrainerTrainStepUsesBatchedPairwiseForward(t *testing.T) {
	trainer := newTinyTrainableAttentionEmbeddingTrainer(t, 0.05)
	if trainer.forwardMatMul != nil {
		trainer.forwardMatMul.Close()
	}
	fake := &countingMatMulAccelerator{}
	trainer.forwardMatMul = fake
	batch := []EmbeddingPairExample{
		{LeftTokens: []int32{0, 2}, RightTokens: []int32{0, 0}, LeftMask: []int32{1, 1}, RightMask: []int32{1, 1}, Target: 1},
		{LeftTokens: []int32{1, 2}, RightTokens: []int32{1, 1}, LeftMask: []int32{1, 1}, RightMask: []int32{1, 1}, Target: 1},
		{LeftTokens: []int32{2, 0}, RightTokens: []int32{2, 2}, LeftMask: []int32{1, 1}, RightMask: []int32{1, 1}, Target: 0},
	}

	if _, err := trainer.TrainStep(batch); err != nil {
		t.Fatalf("train pairwise step: %v", err)
	}
	if trainer.step != 1 {
		t.Fatalf("step = %d, want 1", trainer.step)
	}
	if fake.boundRightRuns == 0 {
		t.Fatalf("pairwise train path did not attempt bound-right matmul")
	}
	if fake.multiBoundRuns == 0 {
		t.Fatalf("pairwise train path did not coalesce q/k/v bound-right matmuls")
	}
	if fake.maxBoundRightRows < 6 {
		t.Fatalf("max bound-right lhs rows = %d, want batched pair rows", fake.maxBoundRightRows)
	}
}

func TestEmbeddingTrainerEvalBatchUsesPairwiseEvalGate(t *testing.T) {
	t.Setenv("MANTA_TRAIN_DISABLE_BATCHED_PAIR_TRAIN", "1")
	trainer := newTinyTrainableAttentionEmbeddingTrainer(t, 0.05)
	if trainer.forwardMatMul != nil {
		trainer.forwardMatMul.Close()
	}
	fake := &countingMatMulAccelerator{}
	trainer.forwardMatMul = fake
	batch := []EmbeddingPairExample{
		{LeftTokens: []int32{0, 2}, RightTokens: []int32{0, 0}, LeftMask: []int32{1, 1}, RightMask: []int32{1, 1}, Target: 1},
		{LeftTokens: []int32{1, 2}, RightTokens: []int32{1, 1}, LeftMask: []int32{1, 1}, RightMask: []int32{1, 1}, Target: 1},
		{LeftTokens: []int32{2, 0}, RightTokens: []int32{2, 2}, LeftMask: []int32{1, 1}, RightMask: []int32{1, 1}, Target: 0},
	}

	if _, err := trainer.EvalBatch(batch); err != nil {
		t.Fatalf("eval pairwise batch: %v", err)
	}
	if trainer.step != 0 {
		t.Fatalf("step = %d, want eval-only batch to leave step unchanged", trainer.step)
	}
	if fake.multiBoundRuns == 0 {
		t.Fatalf("pairwise eval path did not coalesce q/k/v bound-right matmuls")
	}
	if fake.maxBoundRightRows < 6 {
		t.Fatalf("max bound-right lhs rows = %d, want batched eval pair rows", fake.maxBoundRightRows)
	}
}

func TestEmbeddingTrainerBatchedForwardReusesDuplicateSequences(t *testing.T) {
	trainer := newTinyTrainableAttentionEmbeddingTrainer(t, 0.05)
	if trainer.forwardMatMul != nil {
		trainer.forwardMatMul.Close()
	}
	fake := &countingMatMulAccelerator{}
	trainer.forwardMatMul = fake
	batch := []EmbeddingContrastiveExample{
		{QueryTokens: []int32{0, 2}, PositiveTokens: []int32{1, 2}, QueryMask: []int32{1, 1}, PositiveMask: []int32{1, 1}},
		{QueryTokens: []int32{0, 2}, PositiveTokens: []int32{1, 2}, QueryMask: []int32{1, 1}, PositiveMask: []int32{1, 1}},
		{QueryTokens: []int32{0, 2}, PositiveTokens: []int32{2, 2}, QueryMask: []int32{1, 1}, PositiveMask: []int32{1, 1}},
	}

	forward := trainer.prepareForwardWeights()
	trainer.primeForwardWeightResidency(forward.attnQ, forward.attnK, forward.attnV, forward.attnO, forward.hidden, forward.proj)
	queries, positives, err := trainer.encodeContrastiveBatch(batch, forward, true)
	if err != nil {
		t.Fatalf("encode contrastive batch: %v", err)
	}
	defer trainer.releaseEncodedSequences(queries)
	defer trainer.releaseEncodedSequences(positives)
	if queries[0] != queries[1] || queries[1] != queries[2] {
		t.Fatalf("duplicate query token sequences were not reused")
	}
	if positives[0] != positives[1] {
		t.Fatalf("duplicate positive token sequences were not reused")
	}
	if positives[2] == positives[0] {
		t.Fatalf("distinct positive token sequence reused duplicate encoding")
	}
	if fake.maxBoundRightRows != 6 {
		t.Fatalf("max bound-right lhs rows = %d, want only 3 unique sequences of length 2", fake.maxBoundRightRows)
	}
}

func TestEmbeddingTrainerBatchedForwardGroupsVariableSequenceLengths(t *testing.T) {
	trainer := newTinyTrainableAttentionEmbeddingTrainer(t, 0.05)
	if trainer.forwardMatMul != nil {
		trainer.forwardMatMul.Close()
	}
	fake := &countingMatMulAccelerator{}
	trainer.forwardMatMul = fake
	batch := []EmbeddingContrastiveExample{
		{QueryTokens: []int32{0}, PositiveTokens: []int32{0, 2}, QueryMask: []int32{1}, PositiveMask: []int32{1, 1}},
		{QueryTokens: []int32{1}, PositiveTokens: []int32{1, 2}, QueryMask: []int32{1}, PositiveMask: []int32{1, 1}},
		{QueryTokens: []int32{2, 0}, PositiveTokens: []int32{2}, QueryMask: []int32{1, 1}, PositiveMask: []int32{1}},
	}

	forward := trainer.prepareForwardWeights()
	trainer.primeForwardWeightResidency(forward.attnQ, forward.attnK, forward.attnV, forward.attnO, forward.hidden, forward.proj)
	queries, positives, err := trainer.encodeContrastiveBatch(batch, forward, true)
	if err != nil {
		t.Fatalf("encode contrastive batch: %v", err)
	}
	defer trainer.releaseEncodedSequences(queries)
	defer trainer.releaseEncodedSequences(positives)
	if trainer.sequenceBindingID != 0 {
		t.Fatalf("sequence binding count = %d, want default sequence matmul bindings disabled", trainer.sequenceBindingID)
	}
	if fake.boundRightRuns == 0 {
		t.Fatalf("batched forward path did not attempt bound-right matmul")
	}
	if fake.multiBoundRuns == 0 {
		t.Fatalf("batched forward path did not coalesce q/k/v bound-right matmuls")
	}
	if fake.maxBoundRightRows <= 2 {
		t.Fatalf("max bound-right lhs rows = %d, want length-grouped rows above any single sequence", fake.maxBoundRightRows)
	}
	if fake.runCalls == 0 || fake.maxRunBatches < 2 {
		t.Fatalf("attention matmul run calls=%d max batches=%d, want batched attention dispatch", fake.runCalls, fake.maxRunBatches)
	}
}

func TestEmbeddingTrainerQKVMultiBoundCanBeDisabled(t *testing.T) {
	t.Setenv("MANTA_TRAIN_DISABLE_QKV_MULTI_BOUND", "1")
	trainer := newTinyTrainableAttentionEmbeddingTrainer(t, 0.05)
	if trainer.forwardMatMul != nil {
		trainer.forwardMatMul.Close()
	}
	fake := &countingMatMulAccelerator{}
	trainer.forwardMatMul = fake
	batch := []EmbeddingContrastiveExample{
		{QueryTokens: []int32{0, 2}, PositiveTokens: []int32{0, 0}, QueryMask: []int32{1, 1}, PositiveMask: []int32{1, 1}},
		{QueryTokens: []int32{1, 2}, PositiveTokens: []int32{1, 1}, QueryMask: []int32{1, 1}, PositiveMask: []int32{1, 1}},
	}

	forward := trainer.prepareForwardWeights()
	trainer.primeForwardWeightResidency(forward.attnQ, forward.attnK, forward.attnV, forward.attnO, forward.hidden, forward.proj)
	queries, positives, err := trainer.encodeContrastiveBatch(batch, forward, true)
	if err != nil {
		t.Fatalf("encode contrastive batch: %v", err)
	}
	defer trainer.releaseEncodedSequences(queries)
	defer trainer.releaseEncodedSequences(positives)
	if fake.multiBoundRuns != 0 {
		t.Fatalf("multi-bound q/k/v runs = %d, want disabled", fake.multiBoundRuns)
	}
	if fake.boundRightRuns == 0 {
		t.Fatal("expected fallback bound-right matmuls when q/k/v coalescing is disabled")
	}
}

func TestEmbeddingTrainerAttentionBackwardUsesConcatenatedSharedLeftQKVGradMatMul(t *testing.T) {
	trainer := newTinyTrainableEncoderEmbeddingTrainer(t, 0.005)
	if trainer.forwardMatMul != nil {
		trainer.forwardMatMul.Close()
	}
	fake := &countingMatMulAccelerator{}
	trainer.forwardMatMul = fake
	trainer.config.ContrastiveLoss = "infonce"
	trainer.config.Temperature = 0.05

	if _, err := trainer.TrainContrastiveStep(tinyEncoderContrastiveDataset()); err != nil {
		t.Fatalf("train contrastive step: %v", err)
	}
	if fake.sharedLeftRuns != 0 {
		t.Fatalf("shared-left backend runs = %d, want concatenated standard matmul path", fake.sharedLeftRuns)
	}
	if fake.maxRunOutputCols < 6 {
		t.Fatalf("max standard matmul output cols = %d, want concatenated q/k/v width", fake.maxRunOutputCols)
	}
}

func TestEmbeddingTrainerConcatenatedSharedLeftQKVGradMatchesSeparateMatMuls(t *testing.T) {
	trainer := newTinyTrainableEncoderEmbeddingTrainer(t, 0.005)
	if trainer.forwardMatMul != nil {
		trainer.forwardMatMul.Close()
	}
	trainer.forwardMatMul = &countingMatMulAccelerator{}

	lhsMatrices := [][]float32{
		{
			1, 2,
			3, 4,
		},
		{
			-1, 2,
			0.5, -3,
		},
	}
	rhsA := [][]float32{
		{
			0.5, -1,
			2, 3,
		},
		{
			1, 4,
			-2, 0.25,
		},
	}
	rhsB := [][]float32{
		{
			1.5, 0.25,
			-0.5, 2,
		},
		{
			3, -1,
			0.75, 2,
		},
	}
	rhsC := [][]float32{
		{
			-1, 2,
			4, 0.5,
		},
		{
			2, 1,
			-1.5, 3,
		},
	}

	got, ok := trainer.tryConcatenatedSharedLeftAccumulatedTransposeMatMuls(lhsMatrices, [][][]float32{rhsA, rhsB, rhsC}, 2, 2, 2)
	if !ok {
		t.Fatal("expected concatenated shared-left matmul to run")
	}
	if len(got) != 3 {
		t.Fatalf("result count = %d, want 3", len(got))
	}
	for setIndex, rhsSet := range [][][]float32{rhsA, rhsB, rhsC} {
		want := make([]float32, 4)
		for i := range lhsMatrices {
			step := make([]float32, 4)
			fillHostMatMulTranspose(lhsMatrices[i], 2, 2, rhsSet[i], 2, 2, true, false, step)
			addFloat32Slice(want, step)
		}
		assertTensorClose(t, backend.NewTensorF32([]int{2, 2}, got[setIndex]), []int{2, 2}, want)
	}
}

func TestEmbeddingTrainerConcatenatedSharedLeftQKVGradCanBeDisabled(t *testing.T) {
	t.Setenv("MANTA_TRAIN_DISABLE_CONCAT_SHARED_LEFT_MATMUL", "1")
	trainer := newTinyTrainableEncoderEmbeddingTrainer(t, 0.005)
	if trainer.forwardMatMul != nil {
		trainer.forwardMatMul.Close()
	}
	fake := &countingMatMulAccelerator{}
	trainer.forwardMatMul = fake
	trainer.config.ContrastiveLoss = "infonce"
	trainer.config.Temperature = 0.05

	if _, err := trainer.TrainContrastiveStep(tinyEncoderContrastiveDataset()); err != nil {
		t.Fatalf("train contrastive step: %v", err)
	}
	if fake.sharedLeftRuns == 0 {
		t.Fatal("expected shared-left backend fallback when concatenated path is disabled")
	}
	if fake.maxSharedLeftRHS < 3 {
		t.Fatalf("max shared-left rhs count = %d, want at least 3", fake.maxSharedLeftRHS)
	}
}

func TestEmbeddingTrainerCombinedAttentionVKGradMatchesSeparateMatMuls(t *testing.T) {
	trainer := newTinyTrainableEncoderEmbeddingTrainer(t, 0.005)
	if trainer.forwardMatMul != nil {
		trainer.forwardMatMul.Close()
	}
	fake := &countingMatMulAccelerator{}
	trainer.forwardMatMul = fake

	attnScores := [][]float32{
		{
			0.2, 0.8,
			0.6, 0.4,
		},
		{
			0.9, 0.1,
			0.3, 0.7,
		},
	}
	gradMixed := [][]float32{
		{
			1, -2,
			3, 4,
		},
		{
			-1, 0.5,
			2, -3,
		},
	}
	gradPreSoftmax := [][]float32{
		{
			0.5, -1,
			2, 0.25,
		},
		{
			1.5, 0.75,
			-0.5, 3,
		},
	}
	attnQ := [][]float32{
		{
			2, 1,
			-1, 0.5,
		},
		{
			0.25, -2,
			1, 4,
		},
	}

	gotV, gotK, ok := trainer.tryCombinedAttentionValueKeyGradMatMul(attnScores, gradMixed, gradPreSoftmax, attnQ, 2, 2)
	if !ok {
		t.Fatal("expected combined V/K gradient matmul to run")
	}
	if fake.maxRunBatches != 4 {
		t.Fatalf("max run batches = %d, want 4 combined batches", fake.maxRunBatches)
	}
	for i := range attnScores {
		wantV := make([]float32, 4)
		fillHostMatMulTranspose(attnScores[i], 2, 2, gradMixed[i], 2, 2, true, false, wantV)
		assertTensorClose(t, backend.NewTensorF32([]int{2, 2}, gotV[i]), []int{2, 2}, wantV)

		wantK := make([]float32, 4)
		fillHostMatMulTranspose(gradPreSoftmax[i], 2, 2, attnQ[i], 2, 2, true, false, wantK)
		assertTensorClose(t, backend.NewTensorF32([]int{2, 2}, gotK[i]), []int{2, 2}, wantK)
	}
}

func TestEmbeddingTrainerCombinedAttentionVKGradCanBeDisabled(t *testing.T) {
	t.Setenv("MANTA_TRAIN_DISABLE_COMBINED_ATTENTION_VK_GRAD", "1")
	trainer := newTinyTrainableEncoderEmbeddingTrainer(t, 0.005)
	if trainer.forwardMatMul != nil {
		trainer.forwardMatMul.Close()
	}
	trainer.forwardMatMul = &countingMatMulAccelerator{}

	_, _, ok := trainer.tryCombinedAttentionValueKeyGradMatMul(
		[][]float32{{1, 0, 0, 1}},
		[][]float32{{1, 2, 3, 4}},
		[][]float32{{0.5, 0, 0, 0.5}},
		[][]float32{{2, 0, 0, 2}},
		2,
		2,
	)
	if ok {
		t.Fatal("expected combined V/K gradient matmul to be disabled")
	}
}

func TestEmbeddingTrainerAttentionBackwardUsesAccumulatedInputGradMatMul(t *testing.T) {
	trainer := newTinyTrainableEncoderEmbeddingTrainer(t, 0.005)
	if trainer.forwardMatMul != nil {
		trainer.forwardMatMul.Close()
	}
	fake := &countingMatMulAccelerator{}
	trainer.forwardMatMul = fake
	trainer.config.ContrastiveLoss = "infonce"
	trainer.config.Temperature = 0.05

	if _, err := trainer.TrainContrastiveStep(tinyEncoderContrastiveDataset()); err != nil {
		t.Fatalf("train contrastive step: %v", err)
	}
	if fake.accumulatedRuns == 0 {
		t.Fatal("expected accumulated input-gradient bound-right matmul")
	}
	if fake.maxAccumTerms < 3 {
		t.Fatalf("max accumulated terms = %d, want q/k/v terms", fake.maxAccumTerms)
	}
}

func TestEmbeddingTrainerAccumulatedAttentionInputGradMatchesSeparateMatMuls(t *testing.T) {
	trainer := newTinyTrainableEncoderEmbeddingTrainer(t, 0.005)
	if trainer.forwardMatMul != nil {
		trainer.forwardMatMul.Close()
	}
	fake := &countingMatMulAccelerator{}
	trainer.forwardMatMul = fake

	attentionQuery := backend.NewTensorF32([]int{2, 2}, []float32{
		1, 2,
		3, 4,
	})
	attentionKey := backend.NewTensorF32([]int{2, 2}, []float32{
		-1, 0.5,
		2, -0.25,
	})
	attentionValue := backend.NewTensorF32([]int{2, 2}, []float32{
		0.25, 1.5,
		-2, 3,
	})
	if err := fake.BindMatrix(trainer.attnQParam.Name, attentionQuery); err != nil {
		t.Fatalf("bind q: %v", err)
	}
	if err := fake.BindMatrix(trainer.attnKParam.Name, attentionKey); err != nil {
		t.Fatalf("bind k: %v", err)
	}
	if err := fake.BindMatrix(trainer.attnVParam.Name, attentionValue); err != nil {
		t.Fatalf("bind v: %v", err)
	}

	gradQ := [][]float32{
		{
			1, -1,
			0.5, 2,
		},
		{
			-0.25, 3,
			1.5, -2,
		},
	}
	gradK := [][]float32{
		{
			2, 0.25,
			-1.5, 1,
		},
		{
			0.5, -0.5,
			3, 1,
		},
	}
	gradV := [][]float32{
		{
			-1, 4,
			2, -0.75,
		},
		{
			1, 2,
			-3, 0.25,
		},
	}

	got, ok := trainer.tryAccumulatedAttentionInputGradMatMul(gradQ, gradK, gradV, 2, 2, attentionQuery, attentionKey, attentionValue)
	if !ok {
		t.Fatal("expected accumulated attention input-gradient matmul to run")
	}
	if fake.accumulatedRuns != 1 {
		t.Fatalf("accumulated runs = %d, want 1", fake.accumulatedRuns)
	}
	for i := range gradQ {
		want := make([]float32, 4)
		step := make([]float32, 4)
		fillHostMatMulTranspose(gradQ[i], 2, 2, attentionQuery.F32, 2, 2, false, true, step)
		addFloat32Slice(want, step)
		for j := range step {
			step[j] = 0
		}
		fillHostMatMulTranspose(gradK[i], 2, 2, attentionKey.F32, 2, 2, false, true, step)
		addFloat32Slice(want, step)
		for j := range step {
			step[j] = 0
		}
		fillHostMatMulTranspose(gradV[i], 2, 2, attentionValue.F32, 2, 2, false, true, step)
		addFloat32Slice(want, step)
		assertTensorClose(t, backend.NewTensorF32([]int{2, 2}, got[i]), []int{2, 2}, want)
	}
}

func TestEmbeddingTrainerAccumulatedAttentionInputGradCanBeDisabled(t *testing.T) {
	t.Setenv("MANTA_TRAIN_DISABLE_ACCUMULATED_ATTENTION_INPUT_GRAD", "1")
	trainer := newTinyTrainableEncoderEmbeddingTrainer(t, 0.005)
	if trainer.forwardMatMul != nil {
		trainer.forwardMatMul.Close()
	}
	fake := &countingMatMulAccelerator{}
	trainer.forwardMatMul = fake

	attentionQuery := backend.NewTensorF32([]int{2, 2}, []float32{1, 0, 0, 1})
	if err := fake.BindMatrix(trainer.attnQParam.Name, attentionQuery); err != nil {
		t.Fatalf("bind q: %v", err)
	}
	if err := fake.BindMatrix(trainer.attnKParam.Name, attentionQuery); err != nil {
		t.Fatalf("bind k: %v", err)
	}
	if err := fake.BindMatrix(trainer.attnVParam.Name, attentionQuery); err != nil {
		t.Fatalf("bind v: %v", err)
	}
	if _, ok := trainer.tryAccumulatedAttentionInputGradMatMul(
		[][]float32{{1, 2, 3, 4}},
		[][]float32{{1, 2, 3, 4}},
		[][]float32{{1, 2, 3, 4}},
		2,
		2,
		attentionQuery,
		attentionQuery,
		attentionQuery,
	); ok {
		t.Fatal("expected accumulated attention input-gradient matmul to be disabled")
	}
	if fake.accumulatedRuns != 0 {
		t.Fatalf("accumulated runs = %d, want disabled", fake.accumulatedRuns)
	}
}

func TestEmbeddingTrainerSharedLeftQKVGradMatMulCanBeDisabled(t *testing.T) {
	t.Setenv("MANTA_TRAIN_DISABLE_SHARED_LEFT_MATMUL", "1")
	trainer := newTinyTrainableEncoderEmbeddingTrainer(t, 0.005)
	if trainer.forwardMatMul != nil {
		trainer.forwardMatMul.Close()
	}
	fake := &countingMatMulAccelerator{}
	trainer.forwardMatMul = fake
	trainer.config.ContrastiveLoss = "infonce"
	trainer.config.Temperature = 0.05

	if _, err := trainer.TrainContrastiveStep(tinyEncoderContrastiveDataset()); err != nil {
		t.Fatalf("train contrastive step: %v", err)
	}
	if fake.sharedLeftRuns != 0 {
		t.Fatalf("shared-left matmul runs = %d, want disabled", fake.sharedLeftRuns)
	}
}

func TestEmbeddingTrainerBatchedForwardCanBeDisabled(t *testing.T) {
	t.Setenv("MANTA_TRAIN_DISABLE_BATCHED_FORWARD", "1")
	trainer := newTinyTrainableAttentionEmbeddingTrainer(t, 0.05)
	if trainer.forwardMatMul != nil {
		trainer.forwardMatMul.Close()
	}
	fake := &countingMatMulAccelerator{}
	trainer.forwardMatMul = fake
	batch := []EmbeddingContrastiveExample{
		{QueryTokens: []int32{0, 2}, PositiveTokens: []int32{0, 0}, QueryMask: []int32{1, 1}, PositiveMask: []int32{1, 1}},
		{QueryTokens: []int32{1, 2}, PositiveTokens: []int32{1, 1}, QueryMask: []int32{1, 1}, PositiveMask: []int32{1, 1}},
		{QueryTokens: []int32{2, 0}, PositiveTokens: []int32{2, 2}, QueryMask: []int32{1, 1}, PositiveMask: []int32{1, 1}},
	}

	forward := trainer.prepareForwardWeights()
	trainer.primeForwardWeightResidency(forward.attnQ, forward.attnK, forward.attnV, forward.attnO, forward.hidden, forward.proj)
	queries, positives, err := trainer.encodeContrastiveBatch(batch, forward, true)
	if err != nil {
		t.Fatalf("encode contrastive batch: %v", err)
	}
	defer trainer.releaseEncodedSequences(queries)
	defer trainer.releaseEncodedSequences(positives)
	if fake.boundRightRuns == 0 {
		t.Fatal("expected regular per-sequence bound-right matmul calls")
	}
	if fake.maxBoundRightRows > 2 {
		t.Fatalf("max bound-right lhs rows = %d, want per-sequence rows", fake.maxBoundRightRows)
	}
}

func TestEmbeddingTrainerSequenceMatMulBindingsCanBeEnabled(t *testing.T) {
	t.Setenv("MANTA_TRAIN_ENABLE_SEQUENCE_MATMUL_BINDINGS", "1")
	trainer := newTinyTrainableAttentionEmbeddingTrainer(t, 0.05)
	if trainer.forwardMatMul != nil {
		trainer.forwardMatMul.Close()
	}
	fake := &countingMatMulAccelerator{}
	trainer.forwardMatMul = fake
	batch := []EmbeddingContrastiveExample{
		{QueryTokens: []int32{0, 2}, PositiveTokens: []int32{0, 0}, QueryMask: []int32{1, 1}, PositiveMask: []int32{1, 1}},
		{QueryTokens: []int32{1, 2}, PositiveTokens: []int32{1, 1}, QueryMask: []int32{1, 1}, PositiveMask: []int32{1, 1}},
		{QueryTokens: []int32{2, 0}, PositiveTokens: []int32{2, 2}, QueryMask: []int32{1, 1}, PositiveMask: []int32{1, 1}},
	}

	forward := trainer.prepareForwardWeights()
	trainer.primeForwardWeightResidency(forward.attnQ, forward.attnK, forward.attnV, forward.attnO, forward.hidden, forward.proj)
	queries, positives, err := trainer.encodeContrastiveBatch(batch, forward, true)
	if err != nil {
		t.Fatalf("encode contrastive batch: %v", err)
	}
	defer trainer.releaseEncodedSequences(queries)
	defer trainer.releaseEncodedSequences(positives)
	if trainer.sequenceBindingID != 6 {
		t.Fatalf("sequence binding count = %d, want 6 encoded sequences", trainer.sequenceBindingID)
	}
	if fake.bindCalls <= 5 {
		t.Fatalf("bind calls = %d, want sequence matmul bindings beyond forward weights", fake.bindCalls)
	}
}

func TestEmbeddingTrainerWriteEmbeddingPackage(t *testing.T) {
	src := []byte(`
param token_embedding: q8[V, D] @weight("weights/token_embedding") @trainable
param projection: q8[D, E] @weight("weights/projection") @trainable

pipeline embed_pooled(tokens: i32[T], attention_mask: i32[T]) -> f16[E] {
    let hidden_q = gather(token_embedding, tokens)
    let hidden = dequant(hidden_q)
    let projection_f = dequant(projection)
    let projected = @matmul(hidden, projection_f)
    let normalized = normalize(projected)
    return mean_pool(normalized, attention_mask)
}

pipeline embed_pooled_batch(tokens: i32[B, T], attention_mask: i32[B, T]) -> f16[B, E] {
    let hidden_q = gather(token_embedding, tokens)
    let hidden = dequant(hidden_q)
    let projection_f = dequant(projection)
    let projected = @matmul(hidden, projection_f)
    let normalized = normalize(projected)
    return mean_pool(normalized, attention_mask)
}
`)

	bundle, err := compiler.Build(src, compiler.Options{ModuleName: "tiny_train_embed_q8"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	manifest := tinyMaskedEmbeddingManifest()
	manifest.Name = "tiny_train_embed_q8"
	trainer, err := NewEmbeddingTrainer(bundle.Artifact, manifest, map[string]*backend.Tensor{
		"token_embedding": backend.NewTensorF32([]int{3, 2}, []float32{
			1, 0,
			0, 1,
			1, 1,
		}),
		"projection": backend.NewTensorF32([]int{2, 2}, []float32{
			1, 0,
			0, 1,
		}),
	}, EmbeddingTrainConfig{LearningRate: 0.05})
	if err != nil {
		t.Fatalf("new trainer: %v", err)
	}
	for i := 0; i < 8; i++ {
		if _, err := trainer.TrainStep([]EmbeddingPairExample{
			{LeftTokens: []int32{0}, RightTokens: []int32{0}, LeftMask: []int32{1}, RightMask: []int32{1}, Target: 1},
			{LeftTokens: []int32{0}, RightTokens: []int32{1}, LeftMask: []int32{1}, RightMask: []int32{1}, Target: -1},
		}); err != nil {
			t.Fatalf("train step %d: %v", i, err)
		}
	}

	packagePath := filepath.Join(t.TempDir(), "tiny_train_embed_q8.mll")
	paths, err := trainer.WriteEmbeddingPackage(packagePath)
	if err != nil {
		t.Fatalf("write embedding package: %v", err)
	}
	if paths.ArtifactPath != packagePath {
		t.Fatalf("artifact path = %q, want %q", paths.ArtifactPath, packagePath)
	}
	if paths.MemoryPlanPath != DefaultMemoryPlanPath(packagePath) {
		t.Fatalf("memory plan path = %q, want %q", paths.MemoryPlanPath, DefaultMemoryPlanPath(packagePath))
	}
	if paths.PackageManifestPath != DefaultPackageManifestPath(packagePath) {
		t.Fatalf("package manifest path = %q, want %q", paths.PackageManifestPath, DefaultPackageManifestPath(packagePath))
	}

	rt := New(cuda.New(), metal.New())
	model, err := rt.LoadEmbeddingPackage(context.Background(), packagePath)
	if err != nil {
		t.Fatalf("load embedding package: %v", err)
	}
	result, err := model.Embed(context.Background(), []int32{0})
	if err != nil {
		t.Fatalf("embed from written package: %v", err)
	}
	if result.Embeddings == nil {
		t.Fatal("expected embeddings from written package")
	}
	if got := result.Embeddings.DType; got != "f16" {
		t.Fatalf("embedding dtype = %q, want f16", got)
	}
	if model.MemoryPlan() == nil {
		t.Fatal("expected memory plan on loaded model")
	}
}

func TestEmbeddingTrainerWriteTrainingPackageAndReload(t *testing.T) {
	src := []byte(`
param token_embedding: q8[V, D] @weight("weights/token_embedding") @trainable
param projection: q8[D, E] @weight("weights/projection") @trainable

pipeline embed_pooled(tokens: i32[T], attention_mask: i32[T]) -> f16[E] {
    let hidden_q = gather(token_embedding, tokens)
    let hidden = dequant(hidden_q)
    let projection_f = dequant(projection)
    let projected = @matmul(hidden, projection_f)
    let normalized = normalize(projected)
    return mean_pool(normalized, attention_mask)
}

pipeline embed_pooled_batch(tokens: i32[B, T], attention_mask: i32[B, T]) -> f16[B, E] {
    let hidden_q = gather(token_embedding, tokens)
    let hidden = dequant(hidden_q)
    let projection_f = dequant(projection)
    let projected = @matmul(hidden, projection_f)
    let normalized = normalize(projected)
    return mean_pool(normalized, attention_mask)
}
`)

	bundle, err := compiler.Build(src, compiler.Options{ModuleName: "tiny_train_embed_q8"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	manifest := tinyMaskedEmbeddingManifest()
	manifest.Name = "tiny_train_embed_q8"
	trainer, err := NewEmbeddingTrainer(bundle.Artifact, manifest, map[string]*backend.Tensor{
		"token_embedding": backend.NewTensorF32([]int{3, 2}, []float32{
			1, 0,
			0, 1,
			1, 1,
		}),
		"projection": backend.NewTensorF32([]int{2, 2}, []float32{
			1, 0,
			0, 1,
		}),
	}, EmbeddingTrainConfig{LearningRate: 0.05})
	if err != nil {
		t.Fatalf("new trainer: %v", err)
	}

	batch := []EmbeddingPairExample{
		{LeftTokens: []int32{0}, RightTokens: []int32{0}, LeftMask: []int32{1}, RightMask: []int32{1}, Target: 1},
		{LeftTokens: []int32{0}, RightTokens: []int32{1}, LeftMask: []int32{1}, RightMask: []int32{1}, Target: -1},
	}
	for i := 0; i < 6; i++ {
		if _, err := trainer.TrainStep(batch); err != nil {
			t.Fatalf("train step %d: %v", i, err)
		}
	}

	path := filepath.Join(t.TempDir(), "tiny_train_embed_q8.mll")
	paths, err := trainer.WriteTrainingPackage(path)
	if err != nil {
		t.Fatalf("write training package: %v", err)
	}
	restored, err := LoadEmbeddingTrainerPackage(path)
	if err != nil {
		t.Fatalf("load training package: %v", err)
	}
	beforeA, err := trainer.EvaluatePairs(batch)
	if err != nil {
		t.Fatalf("eval original: %v", err)
	}
	beforeB, err := restored.EvaluatePairs(batch)
	if err != nil {
		t.Fatalf("eval restored: %v", err)
	}
	assertClose(t, beforeA.Loss, beforeB.Loss, 0.000001)
	assertClose(t, beforeA.ScoreMargin, beforeB.ScoreMargin, 0.000001)
	if paths.TrainManifestPath != DefaultEmbeddingTrainManifestPath(path) {
		t.Fatalf("train manifest path = %q, want %q", paths.TrainManifestPath, DefaultEmbeddingTrainManifestPath(path))
	}
	if paths.CheckpointPath != DefaultEmbeddingCheckpointPath(path) {
		t.Fatalf("checkpoint path = %q, want %q", paths.CheckpointPath, DefaultEmbeddingCheckpointPath(path))
	}
	if paths.MemoryPlanPath != DefaultMemoryPlanPath(path) {
		t.Fatalf("memory plan path = %q, want %q", paths.MemoryPlanPath, DefaultMemoryPlanPath(path))
	}
	if paths.TrainProfilePath != DefaultEmbeddingTrainProfilePath(path) {
		t.Fatalf("training profile path = %q, want %q", paths.TrainProfilePath, DefaultEmbeddingTrainProfilePath(path))
	}
	if paths.PackageManifestPath != DefaultPackageManifestPath(path) {
		t.Fatalf("package manifest path = %q, want %q", paths.PackageManifestPath, DefaultPackageManifestPath(path))
	}
	profile, err := ReadEmbeddingTrainProfileFile(paths.TrainProfilePath)
	if err != nil {
		t.Fatalf("read training profile: %v", err)
	}
	if profile.Step != trainer.step {
		t.Fatalf("training profile step = %d, want %d", profile.Step, trainer.step)
	}
	if restored.MemoryPlan() == nil {
		t.Fatal("expected restored trainer memory plan")
	}
}

func newTinyTrainableFFNEmbeddingTrainer(t *testing.T, learningRate float32) *EmbeddingTrainer {
	t.Helper()
	src := []byte(`
param token_embedding: q8[V, D] @weight("weights/token_embedding") @trainable
param ffn_up: q8[D, H] @weight("weights/ffn_up") @trainable
param projection: q8[H, E] @weight("weights/projection") @trainable

pipeline embed_pooled(tokens: i32[T], attention_mask: i32[T]) -> f16[E] {
    let hidden_q = gather(token_embedding, tokens)
    let hidden = dequant(hidden_q)
    let ffn_up_f = dequant(ffn_up)
    let projection_f = dequant(projection)
    let ffn_hidden = @matmul(hidden, ffn_up_f)
    let activated = gelu(ffn_hidden)
    let projected = @matmul(activated, projection_f)
    let normalized = normalize(projected)
    return mean_pool(normalized, attention_mask)
}

pipeline embed_pooled_batch(tokens: i32[B, T], attention_mask: i32[B, T]) -> f16[B, E] {
    let hidden_q = gather(token_embedding, tokens)
    let hidden = dequant(hidden_q)
    let ffn_up_f = dequant(ffn_up)
    let projection_f = dequant(projection)
    let ffn_hidden = @matmul(hidden, ffn_up_f)
    let activated = gelu(ffn_hidden)
    let projected = @matmul(activated, projection_f)
    let normalized = normalize(projected)
    return mean_pool(normalized, attention_mask)
}
`)

	bundle, err := compiler.Build(src, compiler.Options{ModuleName: "tiny_train_embed_ffn_q8"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	manifest := tinyMaskedEmbeddingManifest()
	manifest.Name = "tiny_train_embed_ffn_q8"
	manifest.HiddenProjectionParam = "ffn_up"
	trainer, err := NewEmbeddingTrainer(bundle.Artifact, manifest, map[string]*backend.Tensor{
		"token_embedding": backend.NewTensorF32([]int{3, 2}, []float32{
			1, 0,
			0, 1,
			1, 1,
		}),
		"ffn_up": backend.NewTensorF32([]int{2, 3}, []float32{
			1, 0, 1,
			0, 1, 1,
		}),
		"projection": backend.NewTensorF32([]int{3, 2}, []float32{
			1, 0,
			0, 1,
			0.5, 0.5,
		}),
	}, EmbeddingTrainConfig{LearningRate: learningRate})
	if err != nil {
		t.Fatalf("new trainer: %v", err)
	}
	t.Cleanup(trainer.Close)
	return trainer
}

func newTinyTrainableAttentionEmbeddingTrainer(t *testing.T, learningRate float32) *EmbeddingTrainer {
	t.Helper()
	src := []byte(`
param token_embedding: q8[V, D] @weight("weights/token_embedding") @trainable
param attn_q: q8[D, D] @weight("weights/attn_q") @trainable
param attn_k: q8[D, D] @weight("weights/attn_k") @trainable
param attn_v: q8[D, D] @weight("weights/attn_v") @trainable
param attn_o: q8[D, D] @weight("weights/attn_o") @trainable
param projection: q8[D, E] @weight("weights/projection") @trainable

pipeline embed_pooled(tokens: i32[T], attention_mask: i32[T]) -> f16[E] {
    let hidden_q = gather(token_embedding, tokens)
    let hidden = dequant(hidden_q)
    let wq_f = dequant(attn_q)
    let wk_f = dequant(attn_k)
    let wv_f = dequant(attn_v)
    let wo_f = dequant(attn_o)
    let projection_f = dequant(projection)
    let q = @matmul(hidden, wq_f)
    let k = @matmul(hidden, wk_f)
    let v = @matmul(hidden, wv_f)
    let kt = transpose(k)
    let scores = @matmul(q, kt)
    let probs = softmax(scores)
    let mixed = @matmul(probs, v)
    let attended = @matmul(mixed, wo_f)
    let projected = @matmul(attended, projection_f)
    let normalized = normalize(projected)
    return mean_pool(normalized, attention_mask)
}

pipeline embed_pooled_batch(tokens: i32[B, T], attention_mask: i32[B, T]) -> f16[B, E] {
    let hidden_q = gather(token_embedding, tokens)
    let hidden = dequant(hidden_q)
    let wq_f = dequant(attn_q)
    let wk_f = dequant(attn_k)
    let wv_f = dequant(attn_v)
    let wo_f = dequant(attn_o)
    let projection_f = dequant(projection)
    let q = @matmul(hidden, wq_f)
    let k = @matmul(hidden, wk_f)
    let v = @matmul(hidden, wv_f)
    let kt = transpose(k)
    let scores = @matmul(q, kt)
    let probs = softmax(scores)
    let mixed = @matmul(probs, v)
    let attended = @matmul(mixed, wo_f)
    let projected = @matmul(attended, projection_f)
    let normalized = normalize(projected)
    return mean_pool(normalized, attention_mask)
}
`)

	bundle, err := compiler.Build(src, compiler.Options{ModuleName: "tiny_train_embed_attn_q8"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	manifest := tinyMaskedEmbeddingManifest()
	manifest.Name = "tiny_train_embed_attn_q8"
	manifest.AttentionQueryParam = "attn_q"
	manifest.AttentionKeyParam = "attn_k"
	manifest.AttentionValueParam = "attn_v"
	manifest.AttentionOutputParam = "attn_o"
	trainer, err := NewEmbeddingTrainer(bundle.Artifact, manifest, map[string]*backend.Tensor{
		"token_embedding": backend.NewTensorF32([]int{3, 2}, []float32{
			1, 0,
			0, 1,
			1, 1,
		}),
		"attn_q": backend.NewTensorF32([]int{2, 2}, []float32{
			1, 0,
			0, 1,
		}),
		"attn_k": backend.NewTensorF32([]int{2, 2}, []float32{
			1, 0,
			0, 1,
		}),
		"attn_v": backend.NewTensorF32([]int{2, 2}, []float32{
			1, 0,
			0, 1,
		}),
		"attn_o": backend.NewTensorF32([]int{2, 2}, []float32{
			1, 0,
			0, 1,
		}),
		"projection": backend.NewTensorF32([]int{2, 2}, []float32{
			1, 0,
			0, 1,
		}),
	}, EmbeddingTrainConfig{LearningRate: learningRate})
	if err != nil {
		t.Fatalf("new trainer: %v", err)
	}
	t.Cleanup(trainer.Close)
	return trainer
}

func newTinyTrainableEncoderEmbeddingTrainer(t *testing.T, learningRate float32) *EmbeddingTrainer {
	t.Helper()
	src := []byte(`
param token_embedding: q8[V, D] @weight("weights/token_embedding") @trainable
param attn_q: q8[D, D] @weight("weights/attn_q") @trainable
param attn_k: q8[D, D] @weight("weights/attn_k") @trainable
param attn_v: q8[D, D] @weight("weights/attn_v") @trainable
param attn_o: q8[D, D] @weight("weights/attn_o") @trainable
param ffn_up: q8[D, H] @weight("weights/ffn_up") @trainable
param projection: q8[H, D] @weight("weights/projection") @trainable

pipeline embed_pooled(tokens: i32[T], attention_mask: i32[T]) -> f16[D] {
    let hidden_q = gather(token_embedding, tokens)
    let hidden = dequant(hidden_q)
    let wq_f = dequant(attn_q)
    let wk_f = dequant(attn_k)
    let wv_f = dequant(attn_v)
    let wo_f = dequant(attn_o)
    let ffn_up_f = dequant(ffn_up)
    let projection_f = dequant(projection)
    let q = @matmul(hidden, wq_f)
    let k = @matmul(hidden, wk_f)
    let v = @matmul(hidden, wv_f)
    let kt = transpose(k)
    let scores = @matmul(q, kt)
    let probs = softmax(scores)
    let mixed = @matmul(probs, v)
    let attended = @matmul(mixed, wo_f)
    let attn_hidden = layernorm(attended + hidden)
    let ffn_hidden = @matmul(attn_hidden, ffn_up_f)
    let activated = gelu(ffn_hidden)
    let projected = @matmul(activated, projection_f)
    let encoded = layernorm(projected + attn_hidden)
    let normalized = normalize(encoded)
    return mean_pool(normalized, attention_mask)
}

pipeline embed_pooled_batch(tokens: i32[B, T], attention_mask: i32[B, T]) -> f16[B, D] {
    let hidden_q = gather(token_embedding, tokens)
    let hidden = dequant(hidden_q)
    let wq_f = dequant(attn_q)
    let wk_f = dequant(attn_k)
    let wv_f = dequant(attn_v)
    let wo_f = dequant(attn_o)
    let ffn_up_f = dequant(ffn_up)
    let projection_f = dequant(projection)
    let q = @matmul(hidden, wq_f)
    let k = @matmul(hidden, wk_f)
    let v = @matmul(hidden, wv_f)
    let kt = transpose(k)
    let scores = @matmul(q, kt)
    let probs = softmax(scores)
    let mixed = @matmul(probs, v)
    let attended = @matmul(mixed, wo_f)
    let attn_hidden = layernorm(attended + hidden)
    let ffn_hidden = @matmul(attn_hidden, ffn_up_f)
    let activated = gelu(ffn_hidden)
    let projected = @matmul(activated, projection_f)
    let encoded = layernorm(projected + attn_hidden)
    let normalized = normalize(encoded)
    return mean_pool(normalized, attention_mask)
}
`)

	bundle, err := compiler.Build(src, compiler.Options{ModuleName: "tiny_train_embed_encoder_q8"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	manifest := tinyMaskedEmbeddingManifest()
	manifest.Name = "tiny_train_embed_encoder_q8"
	manifest.AttentionQueryParam = "attn_q"
	manifest.AttentionKeyParam = "attn_k"
	manifest.AttentionValueParam = "attn_v"
	manifest.AttentionOutputParam = "attn_o"
	manifest.AttentionResidual = true
	manifest.AttentionLayerNorm = true
	manifest.HiddenProjectionParam = "ffn_up"
	manifest.FFNResidual = true
	manifest.FFNLayerNorm = true
	trainer, err := NewEmbeddingTrainer(bundle.Artifact, manifest, map[string]*backend.Tensor{
		"token_embedding": backend.NewTensorF32([]int{3, 3}, []float32{
			0.9, 0.1, 0.2,
			0.2, 0.8, 0.3,
			0.6, 0.4, 0.7,
		}),
		"attn_q": backend.NewTensorF32([]int{3, 3}, []float32{
			0.9, 0.1, 0.0,
			0.1, 0.8, 0.1,
			0.0, 0.2, 0.9,
		}),
		"attn_k": backend.NewTensorF32([]int{3, 3}, []float32{
			0.8, 0.2, 0.1,
			0.0, 0.9, 0.1,
			0.1, 0.1, 0.8,
		}),
		"attn_v": backend.NewTensorF32([]int{3, 3}, []float32{
			0.7, 0.2, 0.1,
			0.2, 0.7, 0.2,
			0.1, 0.3, 0.8,
		}),
		"attn_o": backend.NewTensorF32([]int{3, 3}, []float32{
			0.6, 0.2, 0.2,
			0.2, 0.7, 0.1,
			0.1, 0.2, 0.7,
		}),
		"ffn_up": backend.NewTensorF32([]int{3, 4}, []float32{
			0.7, 0.1, 0.4, 0.2,
			0.2, 0.8, 0.5, 0.3,
			0.4, 0.3, 0.7, 0.6,
		}),
		"projection": backend.NewTensorF32([]int{4, 3}, []float32{
			0.6, 0.2, 0.2,
			0.3, 0.5, 0.2,
			0.2, 0.3, 0.5,
			0.4, 0.1, 0.5,
		}),
	}, EmbeddingTrainConfig{LearningRate: learningRate})
	if err != nil {
		t.Fatalf("new trainer: %v", err)
	}
	t.Cleanup(trainer.Close)
	return trainer
}

func newTinyTrainableRepeatedEncoderEmbeddingTrainer(t *testing.T, learningRate float32) *EmbeddingTrainer {
	t.Helper()
	src := []byte(`
param token_embedding: q8[V, D] @weight("weights/token_embedding") @trainable
param attn_q: q8[D, D] @weight("weights/attn_q") @trainable
param attn_k: q8[D, D] @weight("weights/attn_k") @trainable
param attn_v: q8[D, D] @weight("weights/attn_v") @trainable
param attn_o: q8[D, D] @weight("weights/attn_o") @trainable
param ffn_up: q8[D, H] @weight("weights/ffn_up") @trainable
param projection: q8[H, D] @weight("weights/projection") @trainable

pipeline embed_pooled(tokens: i32[T], attention_mask: i32[T]) -> f16[D] {
    let hidden_q = gather(token_embedding, tokens)
    let hidden = dequant(hidden_q)
    let wq_f = dequant(attn_q)
    let wk_f = dequant(attn_k)
    let wv_f = dequant(attn_v)
    let wo_f = dequant(attn_o)
    let ffn_up_f = dequant(ffn_up)
    let projection_f = dequant(projection)

    let q1 = @matmul(hidden, wq_f)
    let k1 = @matmul(hidden, wk_f)
    let v1 = @matmul(hidden, wv_f)
    let kt1 = transpose(k1)
    let scores1 = @matmul(q1, kt1)
    let probs1 = softmax(scores1)
    let mixed1 = @matmul(probs1, v1)
    let attended1 = @matmul(mixed1, wo_f)
    let attn_hidden1 = layernorm(attended1 + hidden)
    let ffn_hidden1 = @matmul(attn_hidden1, ffn_up_f)
    let activated1 = gelu(ffn_hidden1)
    let projected1 = @matmul(activated1, projection_f)
    let encoded1 = layernorm(projected1 + attn_hidden1)

    let q2 = @matmul(encoded1, wq_f)
    let k2 = @matmul(encoded1, wk_f)
    let v2 = @matmul(encoded1, wv_f)
    let kt2 = transpose(k2)
    let scores2 = @matmul(q2, kt2)
    let probs2 = softmax(scores2)
    let mixed2 = @matmul(probs2, v2)
    let attended2 = @matmul(mixed2, wo_f)
    let attn_hidden2 = layernorm(attended2 + encoded1)
    let ffn_hidden2 = @matmul(attn_hidden2, ffn_up_f)
    let activated2 = gelu(ffn_hidden2)
    let projected2 = @matmul(activated2, projection_f)
    let encoded2 = layernorm(projected2 + attn_hidden2)
    let normalized = normalize(encoded2)
    return mean_pool(normalized, attention_mask)
}

pipeline embed_pooled_batch(tokens: i32[B, T], attention_mask: i32[B, T]) -> f16[B, D] {
    let hidden_q = gather(token_embedding, tokens)
    let hidden = dequant(hidden_q)
    let wq_f = dequant(attn_q)
    let wk_f = dequant(attn_k)
    let wv_f = dequant(attn_v)
    let wo_f = dequant(attn_o)
    let ffn_up_f = dequant(ffn_up)
    let projection_f = dequant(projection)

    let q1 = @matmul(hidden, wq_f)
    let k1 = @matmul(hidden, wk_f)
    let v1 = @matmul(hidden, wv_f)
    let kt1 = transpose(k1)
    let scores1 = @matmul(q1, kt1)
    let probs1 = softmax(scores1)
    let mixed1 = @matmul(probs1, v1)
    let attended1 = @matmul(mixed1, wo_f)
    let attn_hidden1 = layernorm(attended1 + hidden)
    let ffn_hidden1 = @matmul(attn_hidden1, ffn_up_f)
    let activated1 = gelu(ffn_hidden1)
    let projected1 = @matmul(activated1, projection_f)
    let encoded1 = layernorm(projected1 + attn_hidden1)

    let q2 = @matmul(encoded1, wq_f)
    let k2 = @matmul(encoded1, wk_f)
    let v2 = @matmul(encoded1, wv_f)
    let kt2 = transpose(k2)
    let scores2 = @matmul(q2, kt2)
    let probs2 = softmax(scores2)
    let mixed2 = @matmul(probs2, v2)
    let attended2 = @matmul(mixed2, wo_f)
    let attn_hidden2 = layernorm(attended2 + encoded1)
    let ffn_hidden2 = @matmul(attn_hidden2, ffn_up_f)
    let activated2 = gelu(ffn_hidden2)
    let projected2 = @matmul(activated2, projection_f)
    let encoded2 = layernorm(projected2 + attn_hidden2)
    let normalized = normalize(encoded2)
    return mean_pool(normalized, attention_mask)
}
`)

	bundle, err := compiler.Build(src, compiler.Options{ModuleName: "tiny_train_embed_encoder_q8x2"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	manifest := tinyMaskedEmbeddingManifest()
	manifest.Name = "tiny_train_embed_encoder_q8x2"
	manifest.AttentionQueryParam = "attn_q"
	manifest.AttentionKeyParam = "attn_k"
	manifest.AttentionValueParam = "attn_v"
	manifest.AttentionOutputParam = "attn_o"
	manifest.AttentionResidual = true
	manifest.AttentionLayerNorm = true
	manifest.HiddenProjectionParam = "ffn_up"
	manifest.FFNResidual = true
	manifest.FFNLayerNorm = true
	manifest.EncoderRepeats = 2
	trainer, err := NewEmbeddingTrainer(bundle.Artifact, manifest, map[string]*backend.Tensor{
		"token_embedding": backend.NewTensorF32([]int{3, 3}, []float32{
			0.9, 0.1, 0.2,
			0.2, 0.8, 0.3,
			0.6, 0.4, 0.7,
		}),
		"attn_q": backend.NewTensorF32([]int{3, 3}, []float32{
			0.9, 0.1, 0.0,
			0.1, 0.8, 0.1,
			0.0, 0.2, 0.9,
		}),
		"attn_k": backend.NewTensorF32([]int{3, 3}, []float32{
			0.8, 0.2, 0.1,
			0.0, 0.9, 0.1,
			0.1, 0.2, 0.8,
		}),
		"attn_v": backend.NewTensorF32([]int{3, 3}, []float32{
			0.7, 0.1, 0.2,
			0.2, 0.7, 0.1,
			0.1, 0.3, 0.8,
		}),
		"attn_o": backend.NewTensorF32([]int{3, 3}, []float32{
			0.8, 0.0, 0.2,
			0.1, 0.9, 0.0,
			0.2, 0.1, 0.8,
		}),
		"ffn_up": backend.NewTensorF32([]int{3, 4}, []float32{
			0.6, 0.2, 0.1, 0.3,
			0.1, 0.7, 0.2, 0.2,
			0.2, 0.1, 0.8, 0.4,
		}),
		"projection": backend.NewTensorF32([]int{4, 3}, []float32{
			0.7, 0.1, 0.2,
			0.2, 0.6, 0.1,
			0.1, 0.2, 0.8,
			0.3, 0.1, 0.5,
		}),
	}, EmbeddingTrainConfig{LearningRate: learningRate})
	if err != nil {
		t.Fatalf("new repeated encoder trainer: %v", err)
	}
	t.Cleanup(trainer.Close)
	return trainer
}

func abs32(v float32) float32 {
	if v < 0 {
		return -v
	}
	return v
}

func trainerMasterTensorByName(trainer *EmbeddingTrainer, name string) *backend.Tensor {
	switch name {
	case "token_embedding":
		return trainer.tokenEmbed
	case "attn_q":
		return trainer.attentionQuery
	case "attn_k":
		return trainer.attentionKey
	case "attn_v":
		return trainer.attentionValue
	case "attn_o":
		return trainer.attentionOutput
	case "ffn_up":
		return trainer.hiddenProjection
	case "projection":
		return trainer.projection
	default:
		return nil
	}
}

func TestInfoNCEMultiPositiveK1MatchesSinglePositive(t *testing.T) {
	scores := []float32{0.9, 0.3, -0.1, 0.5, 0.0}
	temperature := float32(0.05)

	singleProbs := make([]float32, len(scores))
	singleLoss := infoNCERowProbsAndLossInto(scores, 0, temperature, singleProbs)

	multiProbs := make([]float32, len(scores))
	multiLoss, mass := infoNCEMultiPositiveRowProbsAndLossInto(scores, []int{0}, temperature, multiProbs)

	if diff := abs32(singleLoss - multiLoss); diff > 1e-5 {
		t.Fatalf("K=1 loss mismatch: single=%v multi=%v diff=%v", singleLoss, multiLoss, diff)
	}
	for i := range singleProbs {
		if diff := abs32(singleProbs[i] - multiProbs[i]); diff > 1e-6 {
			t.Fatalf("K=1 prob[%d] mismatch: single=%v multi=%v", i, singleProbs[i], multiProbs[i])
		}
	}
	// mass should equal probs[0] for K=1.
	if diff := abs32(mass - multiProbs[0]); diff > 1e-6 {
		t.Fatalf("K=1 positiveMass=%v want probs[0]=%v", mass, multiProbs[0])
	}
}

func TestInfoNCEMultiPositiveMatchesClosedForm(t *testing.T) {
	scores := []float32{0.8, 0.6, 0.4, 0.1, -0.2}
	temperature := float32(0.1)
	targets := []int{0, 1, 2}

	probs := make([]float32, len(scores))
	loss, mass := infoNCEMultiPositiveRowProbsAndLossInto(scores, targets, temperature, probs)

	// Closed-form reference (float64 for precision):
	var zmax float64
	for _, s := range scores {
		l := float64(s) / float64(temperature)
		if l > zmax || zmax == 0 {
			zmax = l
		}
	}
	var z float64
	exps := make([]float64, len(scores))
	for i, s := range scores {
		exps[i] = math.Exp(float64(s)/float64(temperature) - zmax)
		z += exps[i]
	}
	var q []float64
	for _, e := range exps {
		q = append(q, e/z)
	}
	want := float64(0)
	for _, t := range targets {
		want += q[t]
	}
	wantLoss := -math.Log(want)

	if diff := math.Abs(float64(loss) - wantLoss); diff > 1e-5 {
		t.Fatalf("loss=%v want %v (diff=%v)", loss, wantLoss, diff)
	}
	if diff := math.Abs(float64(mass) - want); diff > 1e-6 {
		t.Fatalf("mass=%v want %v", mass, want)
	}
	for i, p := range probs {
		if diff := math.Abs(float64(p) - q[i]); diff > 1e-6 {
			t.Fatalf("prob[%d]=%v want %v", i, p, q[i])
		}
	}
}

func TestInfoNCEMultiPositiveDedupesDuplicateTargets(t *testing.T) {
	scores := []float32{0.5, 0.2, -0.1}
	temperature := float32(0.05)

	probs1 := make([]float32, len(scores))
	loss1, mass1 := infoNCEMultiPositiveRowProbsAndLossInto(scores, []int{0, 1}, temperature, probs1)

	probs2 := make([]float32, len(scores))
	loss2, mass2 := infoNCEMultiPositiveRowProbsAndLossInto(scores, []int{0, 1, 0, 1, 1}, temperature, probs2)

	if diff := abs32(loss1 - loss2); diff > 1e-6 {
		t.Fatalf("dup loss differs: %v vs %v", loss1, loss2)
	}
	if diff := abs32(mass1 - mass2); diff > 1e-6 {
		t.Fatalf("dup mass differs: %v vs %v", mass1, mass2)
	}
}

func TestInfoNCEMultiPositiveEmptyTargetsSetsZero(t *testing.T) {
	scores := []float32{1, 2, 3}
	probs := []float32{9, 9, 9}
	loss, mass := infoNCEMultiPositiveRowProbsAndLossInto(scores, nil, 0.05, probs)
	if loss != 0 || mass != 0 {
		t.Fatalf("empty targets: loss=%v mass=%v", loss, mass)
	}
	for i, p := range probs {
		if p != 0 {
			t.Fatalf("empty targets: probs[%d]=%v want 0", i, p)
		}
	}
}

type fakeIntnRng struct{ seed int }

func (f *fakeIntnRng) Intn(n int) int {
	f.seed = (f.seed*1103515245 + 12345) & 0x7fffffff
	if n <= 0 {
		return 0
	}
	return f.seed % n
}

func TestMergeHardNegativesByGroupCollapsesByGroupID(t *testing.T) {
	rows := []EmbeddingHardNegativeExample{
		{
			QueryTokens:    []int32{1, 2, 3},
			PositiveTokens: []int32{10, 11},
			NegativeTokens: [][]int32{{20, 21}, {30, 31}},
			GroupID:        "q1",
			Source:         "nfcorpus",
		},
		{
			QueryTokens:    []int32{1, 2, 3},
			PositiveTokens: []int32{12, 13},
			NegativeTokens: [][]int32{{20, 21}, {40, 41}},
			GroupID:        "q1",
			Source:         "nfcorpus",
		},
		{
			QueryTokens:    []int32{7, 8, 9},
			PositiveTokens: []int32{50, 51},
			NegativeTokens: [][]int32{{60, 61}},
			GroupID:        "q2",
			Source:         "fiqa",
		},
	}
	out := MergeHardNegativesByGroup(rows, 0, nil)
	if len(out) != 2 {
		t.Fatalf("got %d groups, want 2", len(out))
	}
	if len(out[0].PositiveTokens) != 2 {
		t.Fatalf("group 0 positives=%d want 2", len(out[0].PositiveTokens))
	}
	if len(out[0].NegativeTokens) != 3 { // {20,21} is deduped
		t.Fatalf("group 0 negatives=%d want 3 (after dedup)", len(out[0].NegativeTokens))
	}
	if out[0].Source != "nfcorpus" || out[0].GroupID != "q1" {
		t.Fatalf("group 0 metadata: source=%s group=%s", out[0].Source, out[0].GroupID)
	}
	if len(out[1].PositiveTokens) != 1 || out[1].GroupID != "q2" {
		t.Fatalf("group 1: positives=%d group=%s", len(out[1].PositiveTokens), out[1].GroupID)
	}
}

func TestMergeHardNegativesByGroupCapsPositives(t *testing.T) {
	rows := make([]EmbeddingHardNegativeExample, 0, 5)
	for i := 0; i < 5; i++ {
		rows = append(rows, EmbeddingHardNegativeExample{
			QueryTokens:    []int32{1},
			PositiveTokens: []int32{int32(100 + i)},
			NegativeTokens: [][]int32{{200}},
			GroupID:        "q",
		})
	}
	rng := &fakeIntnRng{seed: 1}
	out := MergeHardNegativesByGroup(rows, 3, rng)
	if len(out) != 1 {
		t.Fatalf("want 1 group, got %d", len(out))
	}
	if len(out[0].PositiveTokens) != 3 {
		t.Fatalf("positives capped=%d want 3", len(out[0].PositiveTokens))
	}
}

func TestMergeHardNegativesByGroupFallsBackToQueryHash(t *testing.T) {
	rows := []EmbeddingHardNegativeExample{
		{QueryTokens: []int32{1, 2, 3}, PositiveTokens: []int32{10}, NegativeTokens: [][]int32{{20}}},
		{QueryTokens: []int32{1, 2, 3}, PositiveTokens: []int32{11}, NegativeTokens: [][]int32{{21}}},
		{QueryTokens: []int32{4, 5, 6}, PositiveTokens: []int32{12}, NegativeTokens: [][]int32{{22}}},
	}
	out := MergeHardNegativesByGroup(rows, 0, nil)
	if len(out) != 2 {
		t.Fatalf("want 2 groups via query hash fallback, got %d", len(out))
	}
}
