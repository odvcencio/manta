package webgpu

import (
	"context"
	"fmt"
	"sync"

	mantaartifact "github.com/odvcencio/manta/artifact/manta"
	"github.com/odvcencio/manta/runtime/backend"
)

type cachedLoad struct {
	compiled map[string]backend.CompiledKernel
	native   map[string]backend.NativeKernelProgram
}

// Backend is the WebGPU backend surface. Kernel sources and launch plans are
// selected here; until a browser/device binding is attached, numerical
// execution uses the shared reference implementation and records that fact in
// step metadata.
type Backend struct {
	mu          sync.Mutex
	loadCache   map[string]cachedLoad
	cacheHits   int
	cacheMisses int
}

// New returns the WebGPU backend surface.
func New() *Backend {
	return &Backend{loadCache: map[string]cachedLoad{}}
}

func (b *Backend) Kind() mantaartifact.BackendKind {
	return mantaartifact.BackendWebGPU
}

func (b *Backend) Capabilities() []string {
	return []string{
		mantaartifact.CapabilityCandidatePack,
		mantaartifact.CapabilityKVCache,
		mantaartifact.CapabilityMaskedMeanPool,
		mantaartifact.CapabilityHostFallback,
		mantaartifact.CapabilityImageOps,
		mantaartifact.CapabilityTrainingLosses,
		mantaartifact.CapabilityTurboQuant,
	}
}

func (b *Backend) CanLoad(mod *mantaartifact.Module) bool {
	return b != nil && mod != nil && mod.SupportsBackend(mantaartifact.BackendWebGPU)
}

func (b *Backend) Load(ctx context.Context, mod *mantaartifact.Module, weights map[string]backend.WeightBinding) (backend.Executor, error) {
	return b.load(ctx, mod, weights, "")
}

func (b *Backend) LoadWithCacheKey(ctx context.Context, mod *mantaartifact.Module, weights map[string]backend.WeightBinding, cacheKey string) (backend.Executor, error) {
	return b.load(ctx, mod, weights, cacheKey)
}

func (b *Backend) load(_ context.Context, mod *mantaartifact.Module, weights map[string]backend.WeightBinding, cacheKey string) (backend.Executor, error) {
	if b == nil {
		return nil, fmt.Errorf("nil WebGPU backend")
	}
	if cacheKey != "" {
		if cached, ok := b.cachedLoad(cacheKey); ok {
			return &executor{module: mod, weights: weights, compiled: cached.compiled, native: cached.native}, nil
		}
	}
	compiled, err := backend.CompileVariants(mod, mantaartifact.BackendWebGPU)
	if err != nil {
		return nil, err
	}
	native := map[string]backend.NativeKernelProgram{}
	for _, kernel := range mod.Kernels {
		prog, err := backend.CompileNativeKernelProgram(mantaartifact.BackendWebGPU, kernel, compiled[kernel.Name])
		if err != nil {
			return nil, err
		}
		if prog.LaunchConfig == nil {
			prog.LaunchConfig = map[string]any{}
		}
		prog.LaunchConfig["device_execution"] = false
		prog.LaunchConfig["execution_mode"] = "host_fallback"
		prog.LaunchConfig["fallback_reason"] = "webgpu_device_runtime_not_bound"
		native[kernel.Name] = prog
	}
	if cacheKey != "" {
		b.storeCachedLoad(cacheKey, cachedLoad{compiled: compiled, native: native})
	}
	return &executor{module: mod, weights: weights, compiled: compiled, native: native}, nil
}

func (b *Backend) cachedLoad(cacheKey string) (cachedLoad, bool) {
	b.mu.Lock()
	defer b.mu.Unlock()
	cached, ok := b.loadCache[cacheKey]
	if ok {
		b.cacheHits++
	} else {
		b.cacheMisses++
	}
	return cached, ok
}

func (b *Backend) storeCachedLoad(cacheKey string, cached cachedLoad) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.loadCache[cacheKey] = cached
}

type executor struct {
	module   *mantaartifact.Module
	weights  map[string]backend.WeightBinding
	compiled map[string]backend.CompiledKernel
	native   map[string]backend.NativeKernelProgram
}

func (e *executor) Backend() mantaartifact.BackendKind {
	return mantaartifact.BackendWebGPU
}

func (e *executor) Run(ctx context.Context, req backend.Request) (backend.Result, error) {
	return backend.ExecuteSymbolic(ctx, e.module, e.weights, e.compiled, e.dispatchKernel, e.dispatchStep, mantaartifact.BackendWebGPU, req)
}

func (e *executor) dispatchKernel(_ context.Context, kernel mantaartifact.Kernel, inputs []*backend.Tensor) (backend.KernelDispatchResult, error) {
	prog, ok := e.native[kernel.Name]
	if !ok {
		return backend.KernelDispatchResult{}, fmt.Errorf("WebGPU kernel %q is not compiled", kernel.Name)
	}
	meta := cloneLaunchConfig(prog.LaunchConfig)
	meta["device_execution"] = false
	meta["execution_mode"] = "host_fallback"
	meta["fallback_reason"] = "webgpu_device_runtime_not_bound"
	out, err := prog.Fallback(inputs)
	if err != nil {
		return backend.KernelDispatchResult{}, err
	}
	return backend.KernelDispatchResult{
		Outputs:      out,
		VariantEntry: prog.Compiled.Entry,
		SourceHash:   prog.Compiled.SourceHash,
		Metadata:     meta,
	}, nil
}

func (e *executor) dispatchStep(_ context.Context, step mantaartifact.Step, _ mantaartifact.ValueType, inputs []*backend.Tensor) (backend.StepDispatchResult, bool, error) {
	kernel, ok := BuiltinForStep(step.Kind)
	if !ok {
		return backend.StepDispatchResult{}, false, nil
	}
	out, ok, err := runBuiltinReference(step, inputs)
	if err != nil || !ok {
		return backend.StepDispatchResult{}, ok, err
	}
	return backend.StepDispatchResult{
		Outputs:      out,
		VariantEntry: kernel.Entry,
		SourceHash:   kernel.SourceHash(),
		Metadata:     kernel.Metadata(),
	}, true, nil
}

func runBuiltinReference(step mantaartifact.Step, inputs []*backend.Tensor) ([]*backend.Tensor, bool, error) {
	switch step.Kind {
	case mantaartifact.StepConv2D:
		if len(inputs) < 2 || len(inputs) > 3 || inputs[0] == nil || inputs[1] == nil {
			return nil, false, nil
		}
		var bias *backend.Tensor
		if len(inputs) == 3 {
			bias = inputs[2]
		}
		out, err := backend.Conv2DReference(inputs[0], inputs[1], bias, step.Attributes)
		return []*backend.Tensor{out}, true, err
	case mantaartifact.StepConv2DTrans:
		if len(inputs) < 2 || len(inputs) > 3 || inputs[0] == nil || inputs[1] == nil {
			return nil, false, nil
		}
		var bias *backend.Tensor
		if len(inputs) == 3 {
			bias = inputs[2]
		}
		out, err := backend.Conv2DTransposeReference(inputs[0], inputs[1], bias, step.Attributes)
		return []*backend.Tensor{out}, true, err
	case mantaartifact.StepGDN, mantaartifact.StepIGDN:
		if len(inputs) < 1 || len(inputs) > 3 || inputs[0] == nil {
			return nil, false, nil
		}
		var beta, gamma *backend.Tensor
		if len(inputs) > 1 {
			beta = inputs[1]
		}
		if len(inputs) > 2 {
			gamma = inputs[2]
		}
		out, err := backend.GDNReference(inputs[0], beta, gamma, step.Kind == mantaartifact.StepIGDN)
		return []*backend.Tensor{out}, true, err
	case mantaartifact.StepTurboQDecode:
		if len(inputs) != 2 || inputs[0] == nil || inputs[1] == nil {
			return nil, false, nil
		}
		out, err := backend.TurboQuantDecodeReference(inputs[0], inputs[1], step.Attributes)
		return []*backend.Tensor{out}, true, err
	default:
		return nil, false, nil
	}
}

func cloneLaunchConfig(in map[string]any) map[string]any {
	if len(in) == 0 {
		return map[string]any{}
	}
	out := make(map[string]any, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}
