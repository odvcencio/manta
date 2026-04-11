package barruntime

import (
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/odvcencio/manta/artifact/barr"
	"github.com/odvcencio/manta/runtime/backend"
)

// Runtime owns backend selection and module loading.
type Runtime struct {
	backends []backend.Backend
}

// LoadOption mutates runtime loading behavior.
type LoadOption func(*loadConfig)

type loadConfig struct {
	weights           map[string]backend.WeightBinding
	candidateMetadata map[int64]map[string]string
	memoryPlan        *MemoryPlan
	packageManifest   *PackageManifest
}

// Program is a loaded executable module.
type Program struct {
	module            *barr.Module
	executor          backend.Executor
	candidateMetadata map[int64]map[string]string
	memoryPlan        *MemoryPlan
}

// New constructs a runtime from registered backends.
func New(backends ...backend.Backend) *Runtime {
	cp := append([]backend.Backend(nil), backends...)
	return &Runtime{backends: cp}
}

// Load selects a compatible backend and prepares the program.
func (rt *Runtime) Load(ctx context.Context, mod *barr.Module, opts ...LoadOption) (*Program, error) {
	if mod == nil {
		return nil, fmt.Errorf("nil module")
	}
	if err := mod.Validate(); err != nil {
		return nil, err
	}
	cfg := loadConfig{weights: map[string]backend.WeightBinding{}}
	for _, opt := range opts {
		opt(&cfg)
	}
	if err := applyMemoryPlanToWeights(mod, cfg.weights, cfg.memoryPlan); err != nil {
		return nil, err
	}
	if err := validateParamBindings(mod, cfg.weights); err != nil {
		return nil, err
	}
	var reasons []string
	for _, candidate := range rt.backends {
		if !candidate.CanLoad(mod) {
			reasons = append(reasons, fmt.Sprintf("%s: unsupported backend", candidate.Kind()))
			continue
		}
		if missing := missingBackendCapabilities(candidate, mod); len(missing) > 0 {
			reasons = append(reasons, fmt.Sprintf("%s: missing capabilities [%s]", candidate.Kind(), strings.Join(missing, ", ")))
			continue
		}
		exec, err := loadBackendExecutor(ctx, candidate, mod, cfg.weights, cfg.packageManifest)
		if err != nil {
			reasons = append(reasons, fmt.Sprintf("%s: %v", candidate.Kind(), err))
			continue
		}
		return &Program{
			module:            mod,
			executor:          exec,
			candidateMetadata: cloneCandidateMetadataLookup(cfg.candidateMetadata),
			memoryPlan:        cloneMemoryPlan(cfg.memoryPlan),
		}, nil
	}
	if len(reasons) == 0 {
		return nil, fmt.Errorf("no compatible backend for module %q", mod.Name)
	}
	return nil, fmt.Errorf("no compatible backend for module %q: %s", mod.Name, strings.Join(reasons, "; "))
}

// LoadFile reads a serialized .barr artifact and loads it.
func (rt *Runtime) LoadFile(ctx context.Context, path string, opts ...LoadOption) (*Program, error) {
	mod, err := barr.ReadFile(path)
	if err != nil {
		return nil, err
	}
	manifestPath := ResolvePackageManifestPath(path)
	if _, err := os.Stat(manifestPath); err == nil {
		manifest, err := ReadPackageManifestFile(manifestPath)
		if err != nil {
			return nil, err
		}
		opts = append(opts, WithPackageManifest(manifest))
	}
	return rt.Load(ctx, mod, opts...)
}

// Backend reports the selected backend.
func (p *Program) Backend() barr.BackendKind {
	if p == nil || p.executor == nil {
		return ""
	}
	return p.executor.Backend()
}

// Run executes the loaded program.
func (p *Program) Run(ctx context.Context, req backend.Request) (backend.Result, error) {
	if p == nil || p.executor == nil {
		return backend.Result{}, fmt.Errorf("program is not loaded")
	}
	return p.executor.Run(ctx, req)
}

func (p *Program) MemoryPlan() *MemoryPlan {
	if p == nil {
		return nil
	}
	return cloneMemoryPlan(p.memoryPlan)
}

// WithWeight binds a Manta param to runtime-managed data.
func WithWeight(name string, data any) LoadOption {
	return func(cfg *loadConfig) {
		if cfg.weights == nil {
			cfg.weights = map[string]backend.WeightBinding{}
		}
		cfg.weights[name] = backend.WeightBinding{Name: name, Data: data}
	}
}

// WithCandidateMetadata binds external candidate metadata by candidate id.
func WithCandidateMetadata(metadata map[int64]map[string]string) LoadOption {
	return func(cfg *loadConfig) {
		cfg.candidateMetadata = cloneCandidateMetadataLookup(metadata)
	}
}

func WithMemoryPlan(plan MemoryPlan) LoadOption {
	return func(cfg *loadConfig) {
		cfg.memoryPlan = cloneMemoryPlan(&plan)
	}
}

func WithPackageManifest(manifest PackageManifest) LoadOption {
	return func(cfg *loadConfig) {
		cp := manifest
		cfg.packageManifest = &cp
	}
}

func validateParamBindings(mod *barr.Module, weights map[string]backend.WeightBinding) error {
	bindings := map[string]int{}
	for _, param := range mod.Params {
		weight, ok := weights[param.Name]
		if !ok {
			return fmt.Errorf("missing weight binding for param %q", param.Name)
		}
		if _, _, err := backend.PreviewValueWithBindings(param.Type, weight.Data, bindings); err != nil {
			return fmt.Errorf("param %q: %w", param.Name, err)
		}
	}
	return nil
}

func applyMemoryPlanToWeights(mod *barr.Module, weights map[string]backend.WeightBinding, plan *MemoryPlan) error {
	if mod == nil || plan == nil {
		return nil
	}
	if plan.ModuleName != "" && plan.ModuleName != mod.Name {
		return fmt.Errorf("memory plan module %q does not match module %q", plan.ModuleName, mod.Name)
	}
	planned := make(map[string]WeightMemoryPlan, len(plan.Weights))
	for _, item := range plan.Weights {
		planned[item.Name] = item
	}
	for _, param := range mod.Params {
		weight, ok := weights[param.Name]
		if !ok {
			continue
		}
		item, ok := planned[param.Name]
		if !ok {
			return fmt.Errorf("memory plan missing weight %q", param.Name)
		}
		weight.Residency = string(item.Residency)
		weight.AccessCount = item.AccessCount
		weights[param.Name] = weight
	}
	return nil
}

func cloneCandidateMetadataLookup(in map[int64]map[string]string) map[int64]map[string]string {
	if len(in) == 0 {
		return nil
	}
	out := make(map[int64]map[string]string, len(in))
	for id, meta := range in {
		out[id] = cloneCandidateMetadata(meta)
	}
	return out
}

func cloneCandidateMetadata(in map[string]string) map[string]string {
	if len(in) == 0 {
		return nil
	}
	out := make(map[string]string, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}

func missingBackendCapabilities(candidate backend.Backend, mod *barr.Module) []string {
	if candidate == nil || mod == nil {
		return nil
	}
	provider, ok := candidate.(backend.CapabilityProvider)
	if !ok {
		return barr.MissingCapabilities(mod.Requirements.Capabilities, nil)
	}
	return barr.MissingCapabilities(mod.Requirements.Capabilities, provider.Capabilities())
}

type cacheKeyLoader interface {
	LoadWithCacheKey(ctx context.Context, mod *barr.Module, weights map[string]backend.WeightBinding, cacheKey string) (backend.Executor, error)
}

func loadBackendExecutor(ctx context.Context, candidate backend.Backend, mod *barr.Module, weights map[string]backend.WeightBinding, manifest *PackageManifest) (backend.Executor, error) {
	cacheKey := ""
	if manifest != nil {
		cacheKey = manifest.CacheKey()
	}
	if loader, ok := candidate.(cacheKeyLoader); ok {
		return loader.LoadWithCacheKey(ctx, mod, weights, cacheKey)
	}
	return candidate.Load(ctx, mod, weights)
}
