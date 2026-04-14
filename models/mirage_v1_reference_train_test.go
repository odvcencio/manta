package models

import (
	"testing"

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
