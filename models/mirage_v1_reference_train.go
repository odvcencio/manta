package models

import (
	"fmt"
	"math"
	"math/rand"
	"strconv"
	"strings"

	mantaartifact "github.com/odvcencio/manta/artifact/manta"
	"github.com/odvcencio/manta/runtime/backend"
)

// MirageV1ReferenceTrainConfig controls the tiny CPU reference training loop
// used to de-risk the Mirage graph before backend backward kernels exist.
type MirageV1ReferenceTrainConfig struct {
	Steps        int
	LearningRate float32
	GradientClip float32
	WeightDecay  float32
	MinGDNBeta   float32
}

// MirageV1ReferenceTrainHistory records loss movement through a reference run.
type MirageV1ReferenceTrainHistory struct {
	InitialLoss   float32
	FinalLoss     float32
	Losses        []float32
	MSEs          []float32
	Rates         []float32
	GradientNorms []MirageV1ReferenceGradientNorms
}

// MirageV1ReferenceGradientNorms records raw gradient L2 norms by graph region
// before clipping and SGD updates are applied.
type MirageV1ReferenceGradientNorms struct {
	Total          float32
	Analysis       float32
	HyperAnalysis  float32
	HyperSynthesis float32
	Synthesis      float32
	Prior          float32
	Other          float32
}

// InitMirageV1ReferenceWeights creates deterministic trainable weights for the
// reference Mirage v1 training loop.
func InitMirageV1ReferenceWeights(mod *mantaartifact.Module, seed int64) (map[string]*backend.Tensor, error) {
	if mod == nil {
		return nil, fmt.Errorf("nil module")
	}
	rng := rand.New(rand.NewSource(seed))
	weights := make(map[string]*backend.Tensor, len(mod.Params))
	for _, param := range mod.Params {
		if param.Type.Tensor == nil {
			return nil, fmt.Errorf("param %q is not a tensor", param.Name)
		}
		shape, err := concreteParamShape(param.Type.Tensor.Shape)
		if err != nil {
			return nil, fmt.Errorf("param %q: %w", param.Name, err)
		}
		n := tensorElementCount(shape)
		values := make([]float32, n)
		switch {
		case strings.Contains(param.Name, "beta"):
			fillFloat32(values, 1)
		case strings.Contains(param.Name, "gamma"):
			fillGDNGamma(values, shape, 0.01)
		case strings.Contains(param.Name, "bias"):
			fillFloat32(values, 0.02)
		case strings.Contains(param.Name, "logits"):
			fillFloat32(values, 0)
		default:
			scale := xavierScale(shape)
			for i := range values {
				values[i] = float32((rng.Float64()*2 - 1) * float64(scale))
			}
		}
		weights[param.Name] = tensorForDType(param.Type.Tensor.DType, shape, values)
	}
	return weights, nil
}

// TrainMirageV1Reference fits Mirage v1 on a tiny in-memory image set using the
// backend reference autograd path and plain clipped SGD.
func TrainMirageV1Reference(mod *mantaartifact.Module, weights map[string]*backend.Tensor, images []*backend.Tensor, cfg MirageV1ReferenceTrainConfig) (MirageV1ReferenceTrainHistory, error) {
	if mod == nil {
		return MirageV1ReferenceTrainHistory{}, fmt.Errorf("nil module")
	}
	if len(images) == 0 {
		return MirageV1ReferenceTrainHistory{}, fmt.Errorf("at least one training image is required")
	}
	cfg = normalizeMirageReferenceTrainConfig(cfg)
	initial, err := MirageV1ReferenceEval(mod, weights, images)
	if err != nil {
		return MirageV1ReferenceTrainHistory{}, err
	}
	history := MirageV1ReferenceTrainHistory{
		InitialLoss:   initial.Loss,
		Losses:        make([]float32, 0, cfg.Steps+1),
		MSEs:          make([]float32, 0, cfg.Steps+1),
		Rates:         make([]float32, 0, cfg.Steps+1),
		GradientNorms: make([]MirageV1ReferenceGradientNorms, 0, cfg.Steps),
	}
	for step := 0; step < cfg.Steps; step++ {
		image := images[step%len(images)]
		result, err := backend.ExecuteAutograd(mod, backend.GradRequest{
			Entry:   "train_step",
			Inputs:  map[string]*backend.Tensor{"x": image},
			Weights: weights,
		})
		if err != nil {
			return MirageV1ReferenceTrainHistory{}, err
		}
		metrics, err := mirageTrainMetrics(result)
		if err != nil {
			return MirageV1ReferenceTrainHistory{}, err
		}
		history.Losses = append(history.Losses, metrics.Loss)
		history.MSEs = append(history.MSEs, metrics.MSE)
		history.Rates = append(history.Rates, metrics.Rate)
		history.GradientNorms = append(history.GradientNorms, mirageReferenceGradientNorms(result.Gradients))
		if err := applyMirageReferenceSGD(mod, weights, result.Gradients, cfg); err != nil {
			return MirageV1ReferenceTrainHistory{}, err
		}
	}
	final, err := MirageV1ReferenceEval(mod, weights, images)
	if err != nil {
		return MirageV1ReferenceTrainHistory{}, err
	}
	history.FinalLoss = final.Loss
	history.Losses = append(history.Losses, final.Loss)
	history.MSEs = append(history.MSEs, final.MSE)
	history.Rates = append(history.Rates, final.Rate)
	return history, nil
}

func mirageReferenceGradientNorms(grads map[string]*backend.Tensor) MirageV1ReferenceGradientNorms {
	var total, analysis, hyperAnalysis, hyperSynthesis, synthesis, prior, other float64
	for name, grad := range grads {
		if grad == nil {
			continue
		}
		sum := gradSquaredNorm(grad)
		total += sum
		switch {
		case strings.HasPrefix(name, "ga") || strings.HasPrefix(name, "gdn"):
			analysis += sum
		case strings.HasPrefix(name, "ha_"):
			hyperAnalysis += sum
		case strings.HasPrefix(name, "hs_"):
			hyperSynthesis += sum
		case strings.HasPrefix(name, "gs") || strings.HasPrefix(name, "igdn"):
			synthesis += sum
		case name == "prior_z_logits":
			prior += sum
		default:
			other += sum
		}
	}
	return MirageV1ReferenceGradientNorms{
		Total:          float32(math.Sqrt(total)),
		Analysis:       float32(math.Sqrt(analysis)),
		HyperAnalysis:  float32(math.Sqrt(hyperAnalysis)),
		HyperSynthesis: float32(math.Sqrt(hyperSynthesis)),
		Synthesis:      float32(math.Sqrt(synthesis)),
		Prior:          float32(math.Sqrt(prior)),
		Other:          float32(math.Sqrt(other)),
	}
}

func gradSquaredNorm(t *backend.Tensor) float64 {
	sum := 0.0
	for _, v := range t.F32 {
		sum += float64(v) * float64(v)
	}
	return sum
}

// MirageV1ReferenceMetrics summarizes one average reference eval pass.
type MirageV1ReferenceMetrics struct {
	Loss float32
	MSE  float32
	Rate float32
}

// MirageV1ReferenceEval evaluates the current weights using ExecuteAutograd.
func MirageV1ReferenceEval(mod *mantaartifact.Module, weights map[string]*backend.Tensor, images []*backend.Tensor) (MirageV1ReferenceMetrics, error) {
	if len(images) == 0 {
		return MirageV1ReferenceMetrics{}, fmt.Errorf("at least one image is required")
	}
	total := MirageV1ReferenceMetrics{}
	for _, image := range images {
		result, err := backend.ExecuteAutograd(mod, backend.GradRequest{
			Entry:   "train_step",
			Inputs:  map[string]*backend.Tensor{"x": image},
			Weights: weights,
		})
		if err != nil {
			return MirageV1ReferenceMetrics{}, err
		}
		metrics, err := mirageTrainMetrics(result)
		if err != nil {
			return MirageV1ReferenceMetrics{}, err
		}
		total.Loss += metrics.Loss
		total.MSE += metrics.MSE
		total.Rate += metrics.Rate
	}
	scale := float32(1 / float64(len(images)))
	total.Loss *= scale
	total.MSE *= scale
	total.Rate *= scale
	return total, nil
}

func normalizeMirageReferenceTrainConfig(cfg MirageV1ReferenceTrainConfig) MirageV1ReferenceTrainConfig {
	if cfg.Steps <= 0 {
		cfg.Steps = 16
	}
	if cfg.LearningRate == 0 {
		cfg.LearningRate = 0.01
	}
	if cfg.GradientClip == 0 {
		cfg.GradientClip = 1
	}
	if cfg.MinGDNBeta == 0 {
		cfg.MinGDNBeta = 1e-4
	}
	return cfg
}

func applyMirageReferenceSGD(mod *mantaartifact.Module, weights, grads map[string]*backend.Tensor, cfg MirageV1ReferenceTrainConfig) error {
	for _, param := range mod.Params {
		if !param.Trainable {
			continue
		}
		weight := weights[param.Name]
		grad := grads[param.Name]
		if weight == nil || grad == nil {
			continue
		}
		if len(weight.F32) != len(grad.F32) {
			return fmt.Errorf("gradient shape mismatch for %q", param.Name)
		}
		for i := range weight.F32 {
			g := grad.F32[i]
			if cfg.GradientClip > 0 {
				g = clampFloat32(g, -cfg.GradientClip, cfg.GradientClip)
			}
			if cfg.WeightDecay != 0 {
				g += cfg.WeightDecay * weight.F32[i]
			}
			weight.F32[i] -= cfg.LearningRate * g
		}
		constrainMirageParam(param.Name, weight, cfg)
	}
	return nil
}

func constrainMirageParam(name string, weight *backend.Tensor, cfg MirageV1ReferenceTrainConfig) {
	if weight == nil {
		return
	}
	switch {
	case strings.Contains(name, "beta"):
		for i := range weight.F32 {
			if weight.F32[i] < cfg.MinGDNBeta {
				weight.F32[i] = cfg.MinGDNBeta
			}
		}
	case strings.Contains(name, "gamma"):
		for i := range weight.F32 {
			if weight.F32[i] < 0 {
				weight.F32[i] = 0
			}
		}
	}
}

func mirageTrainMetrics(result backend.GradResult) (MirageV1ReferenceMetrics, error) {
	loss, err := scalarOutput(result, "loss")
	if err != nil {
		return MirageV1ReferenceMetrics{}, err
	}
	mse, err := scalarOutput(result, "mse")
	if err != nil {
		return MirageV1ReferenceMetrics{}, err
	}
	rate, err := scalarOutput(result, "rate")
	if err != nil {
		return MirageV1ReferenceMetrics{}, err
	}
	return MirageV1ReferenceMetrics{Loss: loss, MSE: mse, Rate: rate}, nil
}

func scalarOutput(result backend.GradResult, name string) (float32, error) {
	tensor := result.Outputs[name]
	if tensor == nil || len(tensor.F32) != 1 {
		return 0, fmt.Errorf("missing scalar output %q", name)
	}
	value := tensor.F32[0]
	if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
		return 0, fmt.Errorf("non-finite scalar output %q: %v", name, value)
	}
	return value, nil
}

func concreteParamShape(shape []string) ([]int, error) {
	out := make([]int, len(shape))
	for i, dim := range shape {
		n, err := strconv.Atoi(dim)
		if err != nil {
			return nil, fmt.Errorf("non-concrete shape dim %q", dim)
		}
		if n <= 0 {
			return nil, fmt.Errorf("non-positive shape dim %d", n)
		}
		out[i] = n
	}
	return out, nil
}

func tensorElementCount(shape []int) int {
	n := 1
	for _, dim := range shape {
		n *= dim
	}
	return n
}

func fillFloat32(values []float32, value float32) {
	for i := range values {
		values[i] = value
	}
}

func fillGDNGamma(values []float32, shape []int, diagonal float32) {
	if len(shape) == 2 && shape[0] == shape[1] {
		for i := 0; i < shape[0]; i++ {
			values[i*shape[1]+i] = diagonal
		}
		return
	}
	fillFloat32(values, diagonal)
}

func xavierScale(shape []int) float32 {
	if len(shape) < 2 {
		return 0.02
	}
	fanIn := 1
	for _, dim := range shape[1:] {
		fanIn *= dim
	}
	if fanIn <= 0 {
		return 0.02
	}
	return float32(math.Sqrt(2/float64(fanIn)) * 0.25)
}

func tensorForDType(dtype string, shape []int, values []float32) *backend.Tensor {
	switch dtype {
	case "f16":
		return backend.NewTensorF16(shape, values)
	case "q2":
		return backend.NewTensorQ2(shape, values)
	case "q4":
		return backend.NewTensorQ4(shape, values)
	case "q8":
		return backend.NewTensorQ8(shape, values)
	case "q_norm":
		return backend.NewTensorQNorm(shape, values)
	default:
		return backend.NewTensorF32(shape, values)
	}
}

func clampFloat32(value, lo, hi float32) float32 {
	if value < lo {
		return lo
	}
	if value > hi {
		return hi
	}
	return value
}
