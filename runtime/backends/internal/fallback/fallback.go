package fallback

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

// Backend is a backend-owned host fallback executor for newly added GPU APIs
// while their device runtimes are still being implemented.
type Backend struct {
	kind        mantaartifact.BackendKind
	label       string
	mu          sync.Mutex
	loadCache   map[string]cachedLoad
	cacheHits   int
	cacheMisses int
}

func New(kind mantaartifact.BackendKind, label string) *Backend {
	return &Backend{kind: kind, label: label, loadCache: map[string]cachedLoad{}}
}

func (b *Backend) Kind() mantaartifact.BackendKind {
	if b == nil {
		return ""
	}
	return b.kind
}

func (b *Backend) Capabilities() []string {
	return []string{
		mantaartifact.CapabilityCandidatePack,
		mantaartifact.CapabilityKVCache,
		mantaartifact.CapabilityMaskedMeanPool,
		mantaartifact.CapabilityHostFallback,
	}
}

func (b *Backend) CanLoad(mod *mantaartifact.Module) bool {
	return b != nil && mod != nil && mod.SupportsBackend(b.kind)
}

func (b *Backend) Load(ctx context.Context, mod *mantaartifact.Module, weights map[string]backend.WeightBinding) (backend.Executor, error) {
	return b.load(ctx, mod, weights, "")
}

func (b *Backend) LoadWithCacheKey(ctx context.Context, mod *mantaartifact.Module, weights map[string]backend.WeightBinding, cacheKey string) (backend.Executor, error) {
	return b.load(ctx, mod, weights, cacheKey)
}

func (b *Backend) load(_ context.Context, mod *mantaartifact.Module, weights map[string]backend.WeightBinding, cacheKey string) (backend.Executor, error) {
	if b == nil {
		return nil, fmt.Errorf("nil fallback backend")
	}
	if cacheKey != "" {
		if cached, ok := b.cachedLoad(cacheKey); ok {
			return &executor{kind: b.kind, label: b.label, module: mod, weights: weights, compiled: cached.compiled, native: cached.native}, nil
		}
	}
	compiled, err := backend.CompileVariants(mod, b.kind)
	if err != nil {
		return nil, err
	}
	native := map[string]backend.NativeKernelProgram{}
	for _, kernel := range mod.Kernels {
		prog, err := backend.CompileNativeKernelProgram(b.kind, kernel, compiled[kernel.Name])
		if err != nil {
			return nil, err
		}
		if prog.LaunchConfig == nil {
			prog.LaunchConfig = map[string]any{}
		}
		prog.LaunchConfig["device_execution"] = false
		prog.LaunchConfig["execution_mode"] = "host_fallback"
		prog.LaunchConfig["fallback_reason"] = "device_runtime_not_implemented"
		native[kernel.Name] = prog
	}
	if cacheKey != "" {
		b.storeCachedLoad(cacheKey, cachedLoad{compiled: compiled, native: native})
	}
	return &executor{kind: b.kind, label: b.label, module: mod, weights: weights, compiled: compiled, native: native}, nil
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
	kind     mantaartifact.BackendKind
	label    string
	module   *mantaartifact.Module
	weights  map[string]backend.WeightBinding
	compiled map[string]backend.CompiledKernel
	native   map[string]backend.NativeKernelProgram
}

func (e *executor) Backend() mantaartifact.BackendKind {
	return e.kind
}

func (e *executor) Run(ctx context.Context, req backend.Request) (backend.Result, error) {
	return backend.ExecuteSymbolic(ctx, e.module, e.weights, e.compiled, e.dispatchKernel, e.dispatchStep, e.kind, req)
}

func (e *executor) dispatchKernel(_ context.Context, kernel mantaartifact.Kernel, inputs []*backend.Tensor) (backend.KernelDispatchResult, error) {
	prog, ok := e.native[kernel.Name]
	if !ok {
		return backend.KernelDispatchResult{}, fmt.Errorf("%s kernel %q is not compiled", e.label, kernel.Name)
	}
	meta := cloneLaunchConfig(prog.LaunchConfig)
	meta["device_execution"] = false
	meta["execution_mode"] = "host_fallback"
	meta["fallback_reason"] = "device_runtime_not_implemented"
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

func (e *executor) dispatchStep(context.Context, mantaartifact.Step, mantaartifact.ValueType, []*backend.Tensor) (backend.StepDispatchResult, bool, error) {
	return backend.StepDispatchResult{}, false, nil
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
