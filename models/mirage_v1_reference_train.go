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
	Steps                int
	InitialStep          int
	ScheduleSteps        int
	LearningRate         float32
	LearningRateSchedule string
	FinalLearningRate    float32
	LambdaSchedule       string
	InitialLambda        float32
	LambdaDelaySteps     int
	LambdaRampSteps      int
	GradientClip         float32
	WeightDecay          float32
	MinGDNBeta           float32
	Optimizer            string
	AdamBeta1            float32
	AdamBeta2            float32
	AdamEpsilon          float32
	FreezeAnalysisSteps  int
	CheckpointEvery      int
	CheckpointFunc       func(MirageV1ReferenceCheckpoint, map[string]*backend.Tensor) error
	CheckpointStateFunc  func(MirageV1ReferenceCheckpoint, map[string]*backend.Tensor, *MirageV1ReferenceOptimizerState) error
	OptimizerState       *MirageV1ReferenceOptimizerState
	ImageGradAccelerator backend.ImageGradAccelerator
}

// MirageV1ReferenceTrainHistory records loss movement through a reference run.
type MirageV1ReferenceTrainHistory struct {
	InitialLoss    float32
	InitialMSE     float32
	InitialRate    float32
	FinalLoss      float32
	FinalMSE       float32
	FinalRate      float32
	Losses         []float32
	MSEs           []float32
	Rates          []float32
	LearningRates  []float32
	Lambdas        []float32
	GradientNorms  []MirageV1ReferenceGradientNorms
	Checkpoints    []MirageV1ReferenceCheckpoint
	OptimizerState *MirageV1ReferenceOptimizerState
}

// MirageV1ReferenceCheckpoint records full-dataset reference metrics at a
// checkpoint boundary after the optimizer update for Step has been applied.
type MirageV1ReferenceCheckpoint struct {
	Step         int
	Loss         float32
	MSE          float32
	Rate         float32
	LearningRate float32
	Lambda       float32
}

// MirageV1ReferenceOptimizerState snapshots Adam optimizer moments so a
// reference training run can resume without resetting Adam's adaptive state.
type MirageV1ReferenceOptimizerState struct {
	Step int
	M    map[string]*backend.Tensor
	V    map[string]*backend.Tensor
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
	if err := validateMirageReferenceTrainConfig(cfg); err != nil {
		return MirageV1ReferenceTrainHistory{}, err
	}
	targetLambda, err := mirageReferenceTrainStepLambda(mod)
	if err != nil {
		return MirageV1ReferenceTrainHistory{}, err
	}
	defer func() {
		_ = setMirageReferenceTrainStepLambda(mod, targetLambda)
	}()
	if err := setMirageReferenceTrainStepLambda(mod, mirageReferenceLambda(cfg, targetLambda, cfg.InitialStep)); err != nil {
		return MirageV1ReferenceTrainHistory{}, err
	}
	initial, err := MirageV1ReferenceEvalWithAccelerator(mod, weights, images, cfg.ImageGradAccelerator)
	if err != nil {
		return MirageV1ReferenceTrainHistory{}, err
	}
	history := MirageV1ReferenceTrainHistory{
		InitialLoss:   initial.Loss,
		InitialMSE:    initial.MSE,
		InitialRate:   initial.Rate,
		Losses:        make([]float32, 0, cfg.Steps+1),
		MSEs:          make([]float32, 0, cfg.Steps+1),
		Rates:         make([]float32, 0, cfg.Steps+1),
		LearningRates: make([]float32, 0, cfg.Steps),
		Lambdas:       make([]float32, 0, cfg.Steps),
		GradientNorms: make([]MirageV1ReferenceGradientNorms, 0, cfg.Steps),
		Checkpoints:   make([]MirageV1ReferenceCheckpoint, 0, checkpointCapacity(cfg)),
	}
	opt, err := newMirageReferenceOptimizerState(mod, weights, cfg)
	if err != nil {
		return MirageV1ReferenceTrainHistory{}, err
	}
	for step := 0; step < cfg.Steps; step++ {
		globalStep := cfg.InitialStep + step
		learningRate := mirageReferenceLearningRate(cfg, globalStep)
		activeLambda := mirageReferenceLambda(cfg, targetLambda, globalStep)
		if err := setMirageReferenceTrainStepLambda(mod, activeLambda); err != nil {
			return MirageV1ReferenceTrainHistory{}, err
		}
		image := images[step%len(images)]
		result, err := backend.ExecuteAutograd(mod, backend.GradRequest{
			Entry:                "train_step",
			Inputs:               map[string]*backend.Tensor{"x": image},
			Weights:              weights,
			ImageGradAccelerator: cfg.ImageGradAccelerator,
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
		history.LearningRates = append(history.LearningRates, learningRate)
		history.Lambdas = append(history.Lambdas, activeLambda)
		history.GradientNorms = append(history.GradientNorms, mirageReferenceGradientNorms(result.Gradients))
		stepCfg := cfg
		stepCfg.LearningRate = learningRate
		if err := applyMirageReferenceUpdate(mod, weights, result.Gradients, stepCfg, opt, globalStep); err != nil {
			return MirageV1ReferenceTrainHistory{}, err
		}
		stepNumber := globalStep + 1
		if cfg.CheckpointEvery > 0 && stepNumber%cfg.CheckpointEvery == 0 {
			checkpointMetrics, err := MirageV1ReferenceEvalWithAccelerator(mod, weights, images, cfg.ImageGradAccelerator)
			if err != nil {
				return MirageV1ReferenceTrainHistory{}, err
			}
			checkpoint := MirageV1ReferenceCheckpoint{
				Step:         stepNumber,
				Loss:         checkpointMetrics.Loss,
				MSE:          checkpointMetrics.MSE,
				Rate:         checkpointMetrics.Rate,
				LearningRate: learningRate,
				Lambda:       activeLambda,
			}
			history.Checkpoints = append(history.Checkpoints, checkpoint)
			if cfg.CheckpointFunc != nil {
				if err := cfg.CheckpointFunc(checkpoint, weights); err != nil {
					return MirageV1ReferenceTrainHistory{}, err
				}
			}
			if cfg.CheckpointStateFunc != nil {
				if err := cfg.CheckpointStateFunc(checkpoint, weights, snapshotMirageReferenceOptimizerState(opt, weights)); err != nil {
					return MirageV1ReferenceTrainHistory{}, err
				}
			}
		}
	}
	if err := setMirageReferenceTrainStepLambda(mod, targetLambda); err != nil {
		return MirageV1ReferenceTrainHistory{}, err
	}
	final, err := MirageV1ReferenceEvalWithAccelerator(mod, weights, images, cfg.ImageGradAccelerator)
	if err != nil {
		return MirageV1ReferenceTrainHistory{}, err
	}
	history.FinalLoss = final.Loss
	history.FinalMSE = final.MSE
	history.FinalRate = final.Rate
	history.Losses = append(history.Losses, final.Loss)
	history.MSEs = append(history.MSEs, final.MSE)
	history.Rates = append(history.Rates, final.Rate)
	history.OptimizerState = snapshotMirageReferenceOptimizerState(opt, weights)
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
	return MirageV1ReferenceEvalWithAccelerator(mod, weights, images, nil)
}

// MirageV1ReferenceEvalWithAccelerator evaluates the current weights using
// ExecuteAutograd, optionally reusing image kernels for forward/backward image
// primitives.
func MirageV1ReferenceEvalWithAccelerator(mod *mantaartifact.Module, weights map[string]*backend.Tensor, images []*backend.Tensor, imageGrad backend.ImageGradAccelerator) (MirageV1ReferenceMetrics, error) {
	if len(images) == 0 {
		return MirageV1ReferenceMetrics{}, fmt.Errorf("at least one image is required")
	}
	total := MirageV1ReferenceMetrics{}
	for _, image := range images {
		result, err := backend.ExecuteAutograd(mod, backend.GradRequest{
			Entry:                "train_step",
			Inputs:               map[string]*backend.Tensor{"x": image},
			Weights:              weights,
			ImageGradAccelerator: imageGrad,
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
	if cfg.InitialStep == 0 && cfg.OptimizerState != nil && cfg.OptimizerState.Step > 0 {
		cfg.InitialStep = cfg.OptimizerState.Step
	}
	if cfg.ScheduleSteps <= 0 {
		cfg.ScheduleSteps = cfg.InitialStep + cfg.Steps
	}
	if cfg.LearningRate == 0 {
		cfg.LearningRate = 0.01
	}
	cfg.LearningRateSchedule = strings.ToLower(strings.TrimSpace(cfg.LearningRateSchedule))
	if cfg.LearningRateSchedule == "" {
		cfg.LearningRateSchedule = "constant"
	}
	cfg.LambdaSchedule = strings.ToLower(strings.TrimSpace(cfg.LambdaSchedule))
	if cfg.LambdaSchedule == "" {
		cfg.LambdaSchedule = "constant"
	}
	if cfg.FinalLearningRate == 0 {
		cfg.FinalLearningRate = cfg.LearningRate
	}
	if cfg.GradientClip == 0 {
		cfg.GradientClip = 1
	}
	if cfg.MinGDNBeta == 0 {
		cfg.MinGDNBeta = 1e-4
	}
	cfg.Optimizer = strings.ToLower(strings.TrimSpace(cfg.Optimizer))
	if cfg.Optimizer == "" {
		cfg.Optimizer = "sgd"
	}
	if cfg.AdamBeta1 == 0 {
		cfg.AdamBeta1 = 0.9
	}
	if cfg.AdamBeta2 == 0 {
		cfg.AdamBeta2 = 0.999
	}
	if cfg.AdamEpsilon == 0 {
		cfg.AdamEpsilon = 1e-8
	}
	return cfg
}

func validateMirageReferenceTrainConfig(cfg MirageV1ReferenceTrainConfig) error {
	if cfg.InitialStep < 0 {
		return fmt.Errorf("initial step must be non-negative")
	}
	if cfg.ScheduleSteps < 0 {
		return fmt.Errorf("schedule steps must be non-negative")
	}
	switch cfg.LearningRateSchedule {
	case "constant", "cosine":
	default:
		return fmt.Errorf("unknown learning rate schedule %q", cfg.LearningRateSchedule)
	}
	switch cfg.LambdaSchedule {
	case "constant", "linear":
	default:
		return fmt.Errorf("unknown lambda schedule %q", cfg.LambdaSchedule)
	}
	if cfg.LambdaDelaySteps < 0 {
		return fmt.Errorf("lambda delay steps must be non-negative")
	}
	if cfg.LambdaRampSteps < 0 {
		return fmt.Errorf("lambda ramp steps must be non-negative")
	}
	if cfg.FreezeAnalysisSteps < 0 {
		return fmt.Errorf("analysis freeze steps must be non-negative")
	}
	if cfg.CheckpointEvery < 0 {
		return fmt.Errorf("checkpoint interval must be non-negative")
	}
	if cfg.OptimizerState != nil && cfg.Optimizer != "adam" {
		return fmt.Errorf("optimizer state resume requires adam optimizer")
	}
	return nil
}

func mirageReferenceLearningRate(cfg MirageV1ReferenceTrainConfig, globalStep int) float32 {
	switch cfg.LearningRateSchedule {
	case "cosine":
		if cfg.ScheduleSteps <= 1 {
			return cfg.LearningRate
		}
		progress := float64(globalStep) / float64(cfg.ScheduleSteps-1)
		if progress < 0 {
			progress = 0
		}
		if progress > 1 {
			progress = 1
		}
		cosine := 0.5 * (1 + math.Cos(math.Pi*progress))
		return cfg.FinalLearningRate + float32(cosine)*float32(cfg.LearningRate-cfg.FinalLearningRate)
	default:
		return cfg.LearningRate
	}
}

func mirageReferenceLambda(cfg MirageV1ReferenceTrainConfig, target float32, step int) float32 {
	if cfg.LambdaSchedule != "linear" {
		return target
	}
	if step < cfg.LambdaDelaySteps {
		return cfg.InitialLambda
	}
	if cfg.LambdaRampSteps <= 0 {
		return target
	}
	progress := float64(step-cfg.LambdaDelaySteps) / float64(cfg.LambdaRampSteps)
	if progress <= 0 {
		return cfg.InitialLambda
	}
	if progress >= 1 {
		return target
	}
	return cfg.InitialLambda + float32(progress)*(target-cfg.InitialLambda)
}

func checkpointCapacity(cfg MirageV1ReferenceTrainConfig) int {
	if cfg.CheckpointEvery <= 0 {
		return 0
	}
	return cfg.Steps / cfg.CheckpointEvery
}

type mirageReferenceOptimizerState struct {
	step int
	m    map[string][]float32
	v    map[string][]float32
}

func newMirageReferenceOptimizerState(mod *mantaartifact.Module, weights map[string]*backend.Tensor, cfg MirageV1ReferenceTrainConfig) (*mirageReferenceOptimizerState, error) {
	if cfg.Optimizer != "adam" {
		return nil, nil
	}
	state := &mirageReferenceOptimizerState{
		step: cfg.InitialStep,
		m:    make(map[string][]float32, len(mod.Params)),
		v:    make(map[string][]float32, len(mod.Params)),
	}
	for _, param := range mod.Params {
		if !param.Trainable {
			continue
		}
		weight := weights[param.Name]
		if weight == nil {
			continue
		}
		state.m[param.Name] = make([]float32, len(weight.F32))
		state.v[param.Name] = make([]float32, len(weight.F32))
	}
	if cfg.OptimizerState == nil {
		return state, nil
	}
	if cfg.OptimizerState.Step < 0 {
		return nil, fmt.Errorf("adam optimizer state step must be non-negative")
	}
	state.step = cfg.OptimizerState.Step
	for name, tensor := range cfg.OptimizerState.M {
		if tensor == nil {
			return nil, fmt.Errorf("adam first moment %q is nil", name)
		}
		moment := state.m[name]
		weight := weights[name]
		if weight == nil || moment == nil {
			return nil, fmt.Errorf("adam first moment %q has no matching trainable weight", name)
		}
		if len(tensor.F32) != len(moment) {
			return nil, fmt.Errorf("adam first moment %q length = %d want %d", name, len(tensor.F32), len(moment))
		}
		if !sameIntShape(tensor.Shape, weight.Shape) {
			return nil, fmt.Errorf("adam first moment %q shape = %v want %v", name, tensor.Shape, weight.Shape)
		}
		copy(moment, tensor.F32)
	}
	for name, tensor := range cfg.OptimizerState.V {
		if tensor == nil {
			return nil, fmt.Errorf("adam second moment %q is nil", name)
		}
		moment := state.v[name]
		weight := weights[name]
		if weight == nil || moment == nil {
			return nil, fmt.Errorf("adam second moment %q has no matching trainable weight", name)
		}
		if len(tensor.F32) != len(moment) {
			return nil, fmt.Errorf("adam second moment %q length = %d want %d", name, len(tensor.F32), len(moment))
		}
		if !sameIntShape(tensor.Shape, weight.Shape) {
			return nil, fmt.Errorf("adam second moment %q shape = %v want %v", name, tensor.Shape, weight.Shape)
		}
		copy(moment, tensor.F32)
	}
	for name := range state.m {
		if cfg.OptimizerState.M[name] == nil {
			return nil, fmt.Errorf("adam first moment missing %q", name)
		}
		if cfg.OptimizerState.V[name] == nil {
			return nil, fmt.Errorf("adam second moment missing %q", name)
		}
	}
	return state, nil
}

func snapshotMirageReferenceOptimizerState(opt *mirageReferenceOptimizerState, weights map[string]*backend.Tensor) *MirageV1ReferenceOptimizerState {
	if opt == nil {
		return nil
	}
	out := &MirageV1ReferenceOptimizerState{
		Step: opt.step,
		M:    make(map[string]*backend.Tensor, len(opt.m)),
		V:    make(map[string]*backend.Tensor, len(opt.v)),
	}
	for name, values := range opt.m {
		out.M[name] = backend.NewTensorF32(momentShape(name, values, weights), values)
	}
	for name, values := range opt.v {
		out.V[name] = backend.NewTensorF32(momentShape(name, values, weights), values)
	}
	return out
}

func momentShape(name string, values []float32, weights map[string]*backend.Tensor) []int {
	if weight := weights[name]; weight != nil && len(weight.Shape) > 0 && weight.Elements() == len(values) {
		return weight.Shape
	}
	return []int{len(values)}
}

func sameIntShape(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func applyMirageReferenceUpdate(mod *mantaartifact.Module, weights, grads map[string]*backend.Tensor, cfg MirageV1ReferenceTrainConfig, opt *mirageReferenceOptimizerState, step int) error {
	switch cfg.Optimizer {
	case "sgd":
		return applyMirageReferenceSGD(mod, weights, grads, cfg, step)
	case "adam":
		return applyMirageReferenceAdam(mod, weights, grads, cfg, opt, step)
	default:
		return fmt.Errorf("unknown optimizer %q", cfg.Optimizer)
	}
}

func applyMirageReferenceSGD(mod *mantaartifact.Module, weights, grads map[string]*backend.Tensor, cfg MirageV1ReferenceTrainConfig, step int) error {
	for _, param := range mod.Params {
		if !param.Trainable {
			continue
		}
		if mirageReferenceParamFrozen(param.Name, cfg, step) {
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
			g := prepareMirageReferenceGrad(grad.F32[i], weight.F32[i], cfg)
			weight.F32[i] -= cfg.LearningRate * g
		}
		constrainMirageParam(param.Name, weight, cfg)
	}
	return nil
}

func applyMirageReferenceAdam(mod *mantaartifact.Module, weights, grads map[string]*backend.Tensor, cfg MirageV1ReferenceTrainConfig, opt *mirageReferenceOptimizerState, step int) error {
	if opt == nil {
		return fmt.Errorf("adam optimizer state is nil")
	}
	opt.step++
	beta1, beta2 := float64(cfg.AdamBeta1), float64(cfg.AdamBeta2)
	bias1 := 1 - math.Pow(beta1, float64(opt.step))
	bias2 := 1 - math.Pow(beta2, float64(opt.step))
	for _, param := range mod.Params {
		if !param.Trainable {
			continue
		}
		if mirageReferenceParamFrozen(param.Name, cfg, step) {
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
		m := opt.m[param.Name]
		v := opt.v[param.Name]
		if len(m) != len(weight.F32) || len(v) != len(weight.F32) {
			return fmt.Errorf("adam state shape mismatch for %q", param.Name)
		}
		for i := range weight.F32 {
			g := prepareMirageReferenceGrad(grad.F32[i], weight.F32[i], cfg)
			m[i] = cfg.AdamBeta1*m[i] + (1-cfg.AdamBeta1)*g
			v[i] = cfg.AdamBeta2*v[i] + (1-cfg.AdamBeta2)*g*g
			mHat := float64(m[i]) / bias1
			vHat := float64(v[i]) / bias2
			update := float64(cfg.LearningRate) * mHat / (math.Sqrt(vHat) + float64(cfg.AdamEpsilon))
			weight.F32[i] -= float32(update)
		}
		constrainMirageParam(param.Name, weight, cfg)
	}
	return nil
}

func mirageReferenceParamFrozen(name string, cfg MirageV1ReferenceTrainConfig, step int) bool {
	if cfg.FreezeAnalysisSteps <= 0 || step >= cfg.FreezeAnalysisSteps {
		return false
	}
	return mirageReferenceAnalysisParam(name)
}

func mirageReferenceAnalysisParam(name string) bool {
	return strings.HasPrefix(name, "ga") || strings.HasPrefix(name, "gdn")
}

func prepareMirageReferenceGrad(grad, weight float32, cfg MirageV1ReferenceTrainConfig) float32 {
	g := grad
	if cfg.GradientClip > 0 {
		g = clampFloat32(g, -cfg.GradientClip, cfg.GradientClip)
	}
	if cfg.WeightDecay != 0 {
		g += cfg.WeightDecay * weight
	}
	return g
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

func mirageReferenceTrainStepLambda(mod *mantaartifact.Module) (float32, error) {
	step, err := mirageReferenceTrainStepLoss(mod)
	if err != nil {
		return 0, err
	}
	value, err := strconv.ParseFloat(step.Attributes["lambda"], 32)
	if err != nil {
		return 0, fmt.Errorf("parse train_step lambda: %w", err)
	}
	return float32(value), nil
}

func setMirageReferenceTrainStepLambda(mod *mantaartifact.Module, lambda float32) error {
	step, err := mirageReferenceTrainStepLoss(mod)
	if err != nil {
		return err
	}
	if step.Attributes == nil {
		step.Attributes = map[string]string{}
	}
	step.Attributes["lambda"] = strconv.FormatFloat(float64(lambda), 'g', 8, 32)
	if mod.Metadata != nil {
		mod.Metadata["lambda"] = float64(lambda)
	}
	return nil
}

func mirageReferenceTrainStepLoss(mod *mantaartifact.Module) (*mantaartifact.Step, error) {
	if mod == nil {
		return nil, fmt.Errorf("nil module")
	}
	for i := range mod.Steps {
		step := &mod.Steps[i]
		if step.Entry == "train_step" && step.Kind == mantaartifact.StepRDLoss {
			return step, nil
		}
	}
	return nil, fmt.Errorf("train_step rate-distortion loss not found")
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
	return float32(math.Sqrt(2 / float64(fanIn)))
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
