package backend

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"

	"github.com/odvcencio/barracuda/artifact/barr"
)

// Request is an execution request for a loaded program.
type Request struct {
	Entry  string
	Inputs map[string]any
}

// Value is a typed runtime value flowing through the symbolic executor.
type Value struct {
	Type     barr.ValueType
	Data     any
	Producer string
	Inputs   []string
	Metadata map[string]any
}

// TraceStep records one executed plan step.
type TraceStep struct {
	Entry   string
	Kind    barr.StepKind
	Name    string
	Kernel  string
	Variant string
	Inputs  []string
	Outputs []string
}

// WeightBinding attaches an external param binding to runtime-managed weight data.
type WeightBinding struct {
	Name        string
	Data        any
	Residency   string
	AccessCount int
}

// CompiledKernel is a backend-selected variant cached at load time.
type CompiledKernel struct {
	Name       string
	Backend    barr.BackendKind
	Entry      string
	Source     string
	SourceHash string
	Meta       map[string]string
}

// NativeKernelProgram is a backend-owned compiled kernel program.
type NativeKernelProgram struct {
	Compiled     CompiledKernel
	LaunchConfig map[string]any
	Fallback     func(inputs []*Tensor) ([]*Tensor, error)
	Run          func(inputs []*Tensor) ([]*Tensor, error)
}

// KernelDispatchResult is the result of dispatching a compiled backend kernel.
type KernelDispatchResult struct {
	Outputs      []*Tensor
	VariantEntry string
	SourceHash   string
	Metadata     map[string]any
}

// StepDispatchResult is the result of dispatching a backend-owned non-kernel step.
type StepDispatchResult struct {
	Outputs      []*Tensor
	VariantEntry string
	SourceHash   string
	Metadata     map[string]any
}

// OptimizerUpdateConfig describes one parameter update.
type OptimizerUpdateConfig struct {
	Optimizer    string
	Step         int
	LearningRate float32
	WeightDecay  float32
	Beta1        float32
	Beta2        float32
	Epsilon      float32
	Scale        float32
}

// MatMulAcceleratorStats summarizes backend-owned matmul prep, residency, and run activity.
type MatMulAcceleratorStats struct {
	BindCalls          int64
	UploadedBytes      int64
	QuantizePasses     int64
	QuantizedBytes     int64
	BindNanos          int64
	QuantizeNanos      int64
	BoundMatrices      int64
	RunCalls           int64
	BoundLeftCalls     int64
	BoundRightCalls    int64
	RunUploadedBytes   int64
	RunDownloadedBytes int64
	RunNanos           int64
}

// OptimizerAcceleratorStats summarizes backend-owned optimizer update activity.
type OptimizerAcceleratorStats struct {
	UpdateCalls     int64
	SyncCalls       int64
	UploadedBytes   int64
	DownloadedBytes int64
	UpdateNanos     int64
	SyncNanos       int64
	ResidentParams  int64
}

// ActivationAcceleratorStats summarizes backend-owned activation backward activity.
type ActivationAcceleratorStats struct {
	BindCalls              int64
	GELUBackwardCalls      int64
	SoftmaxBackwardCalls   int64
	LayerNormBackwardCalls int64
	UploadedBytes          int64
	DownloadedBytes        int64
	RunNanos               int64
	BoundTensors           int64
}

// ContrastiveLossConfig describes one backend-owned contrastive loss/gradient run.
type ContrastiveLossConfig struct {
	Temperature float32
}

// ContrastiveAcceleratorStats summarizes backend-owned contrastive loss activity.
type ContrastiveAcceleratorStats struct {
	RunCalls        int64
	UploadedBytes   int64
	DownloadedBytes int64
	RunNanos        int64
}

// ContrastiveGradResult contains pooled embedding gradients and unnormalized row metrics.
type ContrastiveGradResult struct {
	QueryGrads    *Tensor
	PositiveGrads *Tensor
	LossSum       float32
	ScoreSum      float32
}

// MatMulAccelerator exposes a backend-owned matmul fast path for non-plan callers.
type MatMulAccelerator interface {
	Backend() barr.BackendKind
	RunMatMul(inputs []*Tensor, outputType barr.ValueType) (StepDispatchResult, error)
	RunMatMulWithTranspose(inputs []*Tensor, outputType barr.ValueType, transposeLeft, transposeRight bool) (StepDispatchResult, error)
	BindMatrix(name string, tensor *Tensor) error
	UnbindMatrix(name string) error
	RunMatMulWithBoundLeft(leftName string, rhs *Tensor, outputType barr.ValueType, transposeLeft, transposeRight bool) (StepDispatchResult, error)
	RunMatMulWithBoundRight(lhs *Tensor, rightName string, outputType barr.ValueType, transposeLeft, transposeRight bool) (StepDispatchResult, error)
	Stats() MatMulAcceleratorStats
	Close()
}

// MultiBoundRightMatMulAccelerator optionally coalesces several resident-right matmuls
// that share the same left input. It preserves each bound right-hand tensor's own
// residency and quantization state while avoiding repeated LHS uploads.
type MultiBoundRightMatMulAccelerator interface {
	RunMatMulWithBoundRights(lhs *Tensor, rightNames []string, outputType barr.ValueType, transposeLeft, transposeRight bool) ([]StepDispatchResult, error)
}

// SharedLeftMatMulAccelerator optionally coalesces several matmuls that share
// the same left input but use different non-resident right inputs.
type SharedLeftMatMulAccelerator interface {
	RunMatMulsWithSharedLeft(lhs *Tensor, rhs []*Tensor, outputType barr.ValueType, transposeLeft, transposeRight bool) ([]StepDispatchResult, error)
}

// AccumulatedBoundRightMatMulAccelerator optionally coalesces several
// resident-right matmuls with distinct left inputs into one accumulated output.
type AccumulatedBoundRightMatMulAccelerator interface {
	RunAccumulatedMatMulsWithBoundRights(lhs []*Tensor, rightNames []string, outputType barr.ValueType, transposeLeft, transposeRight bool) (StepDispatchResult, error)
}

// OptimizerAccelerator exposes a backend-owned optimizer update fast path.
type OptimizerAccelerator interface {
	Backend() barr.BackendKind
	ApplyUpdate(name string, cfg OptimizerUpdateConfig, tensor, mom1, mom2, grad *Tensor) error
	SyncState(name string, tensor, mom1, mom2 *Tensor, includeMoments bool) error
	Stats() OptimizerAcceleratorStats
	Close()
}

// ActivationAccelerator exposes backend-owned elementwise training ops.
type ActivationAccelerator interface {
	Backend() barr.BackendKind
	BindTensor(name string, tensor *Tensor) error
	UnbindTensor(name string) error
	RunGELUBackwardMul(gradOut, preAct *Tensor) (*Tensor, error)
	RunGELUBackwardMulWithBoundPreAct(gradOut *Tensor, preActName string) (*Tensor, error)
	RunSoftmaxBackwardRows(gradOut, probs *Tensor) (*Tensor, error)
	RunSoftmaxBackwardRowsWithBoundProbs(gradOut *Tensor, probsName string) (*Tensor, error)
	RunLayerNormBackwardRows(gradOut, normalized, pre *Tensor) (*Tensor, error)
	RunLayerNormBackwardRowsWithBoundInputs(gradOut *Tensor, normalizedName, preName string) (*Tensor, error)
	Stats() ActivationAcceleratorStats
	Close()
}

// ContrastiveAccelerator exposes backend-owned contrastive loss and pooled-gradient ops.
type ContrastiveAccelerator interface {
	Backend() barr.BackendKind
	RunInfoNCE(query, positive *Tensor, cfg ContrastiveLossConfig) (ContrastiveGradResult, error)
	Stats() ContrastiveAcceleratorStats
	Close()
}

var matMulAcceleratorFactories []matMulAcceleratorFactory
var optimizerAcceleratorFactories []optimizerAcceleratorFactory
var activationAcceleratorFactories []activationAcceleratorFactory
var contrastiveAcceleratorFactories []contrastiveAcceleratorFactory

type matMulAcceleratorFactory struct {
	kind    barr.BackendKind
	factory func() (MatMulAccelerator, error)
}

type optimizerAcceleratorFactory struct {
	kind    barr.BackendKind
	factory func() (OptimizerAccelerator, error)
}

type activationAcceleratorFactory struct {
	kind    barr.BackendKind
	factory func() (ActivationAccelerator, error)
}

type contrastiveAcceleratorFactory struct {
	kind    barr.BackendKind
	factory func() (ContrastiveAccelerator, error)
}

// KernelDispatcher executes a launch_kernel step through a backend-owned path.
type KernelDispatcher func(ctx context.Context, kernel barr.Kernel, inputs []*Tensor) (KernelDispatchResult, error)

// StepDispatcher executes a backend-owned plan step such as library-backed matmul.
type StepDispatcher func(ctx context.Context, step barr.Step, outputType barr.ValueType, inputs []*Tensor) (StepDispatchResult, bool, error)

// Result is the execution response from a backend.
type Result struct {
	Outputs  map[string]Value
	Metadata map[string]string
	Trace    []TraceStep
}

// Executor runs a previously loaded Barracuda module.
type Executor interface {
	Backend() barr.BackendKind
	Run(ctx context.Context, req Request) (Result, error)
}

// Backend loads Barracuda modules and returns executors.
type Backend interface {
	Kind() barr.BackendKind
	CanLoad(mod *barr.Module) bool
	Load(ctx context.Context, mod *barr.Module, weights map[string]WeightBinding) (Executor, error)
}

// CapabilityProvider reports runtime features a backend can satisfy.
type CapabilityProvider interface {
	Capabilities() []string
}

// CompileVariants resolves backend-specific kernel variants at load time.
func CompileVariants(mod *barr.Module, kind barr.BackendKind) (map[string]CompiledKernel, error) {
	compiled := map[string]CompiledKernel{}
	if mod == nil {
		return compiled, nil
	}
	for _, kernel := range mod.Kernels {
		variant, ok := kernelVariantForBackend(kernel, kind)
		if !ok {
			return nil, fmt.Errorf("kernel %q missing variant for backend %q", kernel.Name, kind)
		}
		sum := sha256.Sum256([]byte(variant.Source))
		compiled[kernel.Name] = CompiledKernel{
			Name:       kernel.Name,
			Backend:    kind,
			Entry:      variant.Entry,
			Source:     variant.Source,
			SourceHash: hex.EncodeToString(sum[:]),
			Meta:       cloneStringMap(variant.Meta),
		}
	}
	return compiled, nil
}

func cloneStringMap(in map[string]string) map[string]string {
	if len(in) == 0 {
		return nil
	}
	out := make(map[string]string, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}

// RegisterMatMulAccelerator registers an optional backend-owned matmul fast path.
func RegisterMatMulAccelerator(kind barr.BackendKind, factory func() (MatMulAccelerator, error)) {
	if factory == nil {
		return
	}
	matMulAcceleratorFactories = append(matMulAcceleratorFactories, matMulAcceleratorFactory{
		kind:    kind,
		factory: factory,
	})
}

// RegisterOptimizerAccelerator registers an optional backend-owned optimizer fast path.
func RegisterOptimizerAccelerator(kind barr.BackendKind, factory func() (OptimizerAccelerator, error)) {
	if factory == nil {
		return
	}
	optimizerAcceleratorFactories = append(optimizerAcceleratorFactories, optimizerAcceleratorFactory{
		kind:    kind,
		factory: factory,
	})
}

// RegisterActivationAccelerator registers an optional backend-owned activation fast path.
func RegisterActivationAccelerator(kind barr.BackendKind, factory func() (ActivationAccelerator, error)) {
	if factory == nil {
		return
	}
	activationAcceleratorFactories = append(activationAcceleratorFactories, activationAcceleratorFactory{
		kind:    kind,
		factory: factory,
	})
}

// RegisterContrastiveAccelerator registers an optional backend-owned contrastive fast path.
func RegisterContrastiveAccelerator(kind barr.BackendKind, factory func() (ContrastiveAccelerator, error)) {
	if factory == nil {
		return
	}
	contrastiveAcceleratorFactories = append(contrastiveAcceleratorFactories, contrastiveAcceleratorFactory{
		kind:    kind,
		factory: factory,
	})
}

// NewPreferredMatMulAccelerator returns the first available registered accelerator.
func NewPreferredMatMulAccelerator(preferred ...barr.BackendKind) (MatMulAccelerator, barr.BackendKind, error) {
	for _, kind := range preferred {
		for _, candidate := range matMulAcceleratorFactories {
			if candidate.kind != kind {
				continue
			}
			accel, err := candidate.factory()
			if err != nil {
				continue
			}
			if accel != nil {
				return accel, kind, nil
			}
		}
	}
	return nil, "", nil
}

// NewPreferredContrastiveAccelerator returns the first available registered contrastive accelerator.
func NewPreferredContrastiveAccelerator(preferred ...barr.BackendKind) (ContrastiveAccelerator, barr.BackendKind, error) {
	for _, kind := range preferred {
		for _, candidate := range contrastiveAcceleratorFactories {
			if candidate.kind != kind {
				continue
			}
			accel, err := candidate.factory()
			if err != nil {
				continue
			}
			if accel != nil {
				return accel, kind, nil
			}
		}
	}
	return nil, "", nil
}

// NewPreferredOptimizerAccelerator returns the first available registered optimizer accelerator.
func NewPreferredOptimizerAccelerator(preferred ...barr.BackendKind) (OptimizerAccelerator, barr.BackendKind, error) {
	for _, kind := range preferred {
		for _, candidate := range optimizerAcceleratorFactories {
			if candidate.kind != kind {
				continue
			}
			accel, err := candidate.factory()
			if err != nil {
				continue
			}
			if accel != nil {
				return accel, kind, nil
			}
		}
	}
	return nil, "", nil
}

// NewPreferredActivationAccelerator returns the first available registered activation accelerator.
func NewPreferredActivationAccelerator(preferred ...barr.BackendKind) (ActivationAccelerator, barr.BackendKind, error) {
	for _, kind := range preferred {
		for _, candidate := range activationAcceleratorFactories {
			if candidate.kind != kind {
				continue
			}
			accel, err := candidate.factory()
			if err != nil {
				continue
			}
			if accel != nil {
				return accel, kind, nil
			}
		}
	}
	return nil, "", nil
}
