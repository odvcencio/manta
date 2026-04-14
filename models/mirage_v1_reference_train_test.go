package models

import (
	"math"
	"testing"

	mantaartifact "github.com/odvcencio/manta/artifact/manta"
	"github.com/odvcencio/manta/runtime/backend"
)

func TestMirageV1ReferenceTrainSingleImageConverges(t *testing.T) {
	mod, err := DefaultMirageV1Module(MirageV1Config{
		ImageHeight:    16,
		ImageWidth:     16,
		LatentChannels: 4,
		HyperChannels:  4,
		BitWidth:       2,
		Lambda:         0.001,
	})
	if err != nil {
		t.Fatal(err)
	}
	weights, err := InitMirageV1ReferenceWeights(mod, 7)
	if err != nil {
		t.Fatal(err)
	}
	image := mirageReferenceTrainImage()
	history, err := TrainMirageV1Reference(mod, weights, []*backend.Tensor{image}, MirageV1ReferenceTrainConfig{
		Steps:        24,
		LearningRate: 0.02,
		GradientClip: 0.5,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("loss %.6f -> %.6f, mse %.6f -> %.6f, rate %.6f -> %.6f",
		history.InitialLoss,
		history.FinalLoss,
		history.MSEs[0],
		history.MSEs[len(history.MSEs)-1],
		history.Rates[0],
		history.Rates[len(history.Rates)-1],
	)
	if len(history.Losses) != 25 {
		t.Fatalf("loss history length = %d want 25", len(history.Losses))
	}
	if len(history.GradientNorms) != 24 {
		t.Fatalf("gradient norm history length = %d want 24", len(history.GradientNorms))
	}
	if history.GradientNorms[0].Total <= 0 || history.GradientNorms[0].Analysis <= 0 {
		t.Fatalf("expected non-zero total and analysis gradient norms: %+v", history.GradientNorms[0])
	}
	if history.FinalLoss >= history.InitialLoss {
		t.Fatalf("loss did not decrease: initial=%.6f final=%.6f losses=%v", history.InitialLoss, history.FinalLoss, history.Losses)
	}
	minImprovement := float32(0.03)
	if history.InitialLoss-history.FinalLoss < minImprovement {
		t.Fatalf("loss improvement %.6f below %.6f; losses=%v", history.InitialLoss-history.FinalLoss, minImprovement, history.Losses)
	}
	if history.MSEs[len(history.MSEs)-1] >= history.MSEs[0] {
		t.Fatalf("mse did not decrease: initial=%.6f final=%.6f", history.MSEs[0], history.MSEs[len(history.MSEs)-1])
	}
}

func TestMirageV1ReferenceTrainAdamRuns(t *testing.T) {
	mod, err := DefaultMirageV1Module(MirageV1Config{
		ImageHeight:    16,
		ImageWidth:     16,
		LatentChannels: 4,
		HyperChannels:  4,
		BitWidth:       2,
		Lambda:         0,
		LambdaSet:      true,
	})
	if err != nil {
		t.Fatal(err)
	}
	weights, err := InitMirageV1ReferenceWeights(mod, 7)
	if err != nil {
		t.Fatal(err)
	}
	history, err := TrainMirageV1Reference(mod, weights, []*backend.Tensor{mirageReferenceTrainImage()}, MirageV1ReferenceTrainConfig{
		Steps:        4,
		LearningRate: 0.001,
		Optimizer:    "adam",
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(history.Losses) != 5 {
		t.Fatalf("loss history length = %d want 5", len(history.Losses))
	}
	if history.FinalLoss >= history.InitialLoss {
		t.Fatalf("adam loss did not decrease: initial=%.6f final=%.6f", history.InitialLoss, history.FinalLoss)
	}
}

func TestMirageV1ReferenceTrainCosineScheduleAndCheckpoints(t *testing.T) {
	mod, err := DefaultMirageV1Module(MirageV1Config{
		ImageHeight:    16,
		ImageWidth:     16,
		LatentChannels: 4,
		HyperChannels:  4,
		BitWidth:       2,
		Lambda:         0,
		LambdaSet:      true,
	})
	if err != nil {
		t.Fatal(err)
	}
	weights, err := InitMirageV1ReferenceWeights(mod, 7)
	if err != nil {
		t.Fatal(err)
	}
	var callbacks []MirageV1ReferenceCheckpoint
	history, err := TrainMirageV1Reference(mod, weights, []*backend.Tensor{mirageReferenceTrainImage()}, MirageV1ReferenceTrainConfig{
		Steps:                4,
		LearningRate:         0.01,
		FinalLearningRate:    0.001,
		LearningRateSchedule: "cosine",
		Optimizer:            "adam",
		CheckpointEvery:      2,
		CheckpointFunc: func(checkpoint MirageV1ReferenceCheckpoint, weights map[string]*backend.Tensor) error {
			callbacks = append(callbacks, checkpoint)
			if len(weights) == 0 {
				t.Fatalf("checkpoint weights are empty")
			}
			return nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(history.LearningRates) != 4 {
		t.Fatalf("learning rate history length = %d want 4", len(history.LearningRates))
	}
	if len(history.Lambdas) != 4 {
		t.Fatalf("lambda history length = %d want 4", len(history.Lambdas))
	}
	if !nearFloat32(history.LearningRates[0], 0.01, 1e-7) {
		t.Fatalf("first learning rate = %.8f want 0.01", history.LearningRates[0])
	}
	if !nearFloat32(history.LearningRates[3], 0.001, 1e-7) {
		t.Fatalf("last learning rate = %.8f want 0.001", history.LearningRates[3])
	}
	if len(history.Checkpoints) != 2 || len(callbacks) != 2 {
		t.Fatalf("checkpoints=%d callbacks=%d want 2", len(history.Checkpoints), len(callbacks))
	}
	if history.Checkpoints[0].Step != 2 || history.Checkpoints[1].Step != 4 {
		t.Fatalf("checkpoint steps = %+v", history.Checkpoints)
	}
	if history.Checkpoints[1].LearningRate != history.LearningRates[3] {
		t.Fatalf("checkpoint learning rate = %.8f want %.8f", history.Checkpoints[1].LearningRate, history.LearningRates[3])
	}
}

func TestMirageV1ReferenceLambdaSchedule(t *testing.T) {
	cfg := normalizeMirageReferenceTrainConfig(MirageV1ReferenceTrainConfig{
		LambdaSchedule:   "linear",
		InitialLambda:    0,
		LambdaDelaySteps: 1,
		LambdaRampSteps:  2,
	})
	target := float32(0.01)
	want := []float32{0, 0, 0.005, 0.01, 0.01}
	for step, expected := range want {
		if got := mirageReferenceLambda(cfg, target, step); !nearFloat32(got, expected, 1e-7) {
			t.Fatalf("lambda at step %d = %.8f want %.8f", step, got, expected)
		}
	}
}

func TestMirageV1ReferenceFreezeAnalysisSteps(t *testing.T) {
	mod := &mantaartifact.Module{
		Params: []mantaartifact.Param{
			{Name: "ga0_weight", Trainable: true},
			{Name: "gdn0_beta", Trainable: true},
			{Name: "hs_logits_weight", Trainable: true},
			{Name: "gs0_weight", Trainable: true},
		},
	}
	weights := map[string]*backend.Tensor{
		"ga0_weight":       backend.NewTensorF16([]int{1}, []float32{1}),
		"gdn0_beta":        backend.NewTensorF16([]int{1}, []float32{1}),
		"hs_logits_weight": backend.NewTensorF16([]int{1}, []float32{1}),
		"gs0_weight":       backend.NewTensorF16([]int{1}, []float32{1}),
	}
	grads := map[string]*backend.Tensor{
		"ga0_weight":       backend.NewTensorF16([]int{1}, []float32{1}),
		"gdn0_beta":        backend.NewTensorF16([]int{1}, []float32{1}),
		"hs_logits_weight": backend.NewTensorF16([]int{1}, []float32{1}),
		"gs0_weight":       backend.NewTensorF16([]int{1}, []float32{1}),
	}
	cfg := normalizeMirageReferenceTrainConfig(MirageV1ReferenceTrainConfig{
		LearningRate:        0.1,
		Optimizer:           "sgd",
		FreezeAnalysisSteps: 1,
	})
	if err := applyMirageReferenceUpdate(mod, weights, grads, cfg, nil, 0); err != nil {
		t.Fatal(err)
	}
	if weights["ga0_weight"].F32[0] != 1 || weights["gdn0_beta"].F32[0] != 1 {
		t.Fatalf("analysis params changed during freeze: ga=%.4f gdn=%.4f", weights["ga0_weight"].F32[0], weights["gdn0_beta"].F32[0])
	}
	if !nearFloat32(weights["hs_logits_weight"].F32[0], 0.9, 1e-7) || !nearFloat32(weights["gs0_weight"].F32[0], 0.9, 1e-7) {
		t.Fatalf("non-analysis params were not updated: hs=%.4f gs=%.4f", weights["hs_logits_weight"].F32[0], weights["gs0_weight"].F32[0])
	}
	if err := applyMirageReferenceUpdate(mod, weights, grads, cfg, nil, 1); err != nil {
		t.Fatal(err)
	}
	if !nearFloat32(weights["ga0_weight"].F32[0], 0.9, 1e-7) || !nearFloat32(weights["gdn0_beta"].F32[0], 0.9, 1e-7) {
		t.Fatalf("analysis params did not update after freeze: ga=%.4f gdn=%.4f", weights["ga0_weight"].F32[0], weights["gdn0_beta"].F32[0])
	}
}

func TestInitMirageV1ReferenceWeightsAreValid(t *testing.T) {
	mod, err := DefaultMirageV1Module(MirageV1Config{})
	if err != nil {
		t.Fatal(err)
	}
	weights, err := InitMirageV1ReferenceWeights(mod, 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(weights) != len(mod.Params) {
		t.Fatalf("weight count = %d want %d", len(weights), len(mod.Params))
	}
	for _, param := range mod.Params {
		weight := weights[param.Name]
		if weight == nil {
			t.Fatalf("missing weight %s", param.Name)
		}
		shape := concreteShape(t, param.Type.Tensor.Shape)
		if !sameShape(weight.Shape, shape) {
			t.Fatalf("%s shape = %v want %v", param.Name, weight.Shape, shape)
		}
		if len(weight.F32) != weight.Elements() {
			t.Fatalf("%s storage = %d want %d", param.Name, len(weight.F32), weight.Elements())
		}
	}
}

func nearFloat32(got, want, tol float32) bool {
	return math.Abs(float64(got-want)) <= float64(tol)
}

func mirageReferenceTrainImage() *backend.Tensor {
	const height = 16
	const width = 16
	values := make([]float32, 1*3*height*width)
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			values[(0*height+y)*width+x] = float32(x) / float32(width-1)
			values[(1*height+y)*width+x] = float32(y) / float32(height-1)
			values[(2*height+y)*width+x] = 0.25 + 0.5*float32((x+y)%7)/6
		}
	}
	return backend.NewTensorF16([]int{1, 3, height, width}, values)
}
