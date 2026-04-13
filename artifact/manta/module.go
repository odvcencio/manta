package manta

import (
	"encoding/json"
	"fmt"
)

// Version is the current artifact schema version.
const Version = "manta/v0alpha1"

// BackendKind names an execution backend.
type BackendKind string

const (
	BackendCUDA     BackendKind = "cuda"
	BackendMetal    BackendKind = "metal"
	BackendVulkan   BackendKind = "vulkan"
	BackendDirectML BackendKind = "directml"
	BackendWebGPU   BackendKind = "webgpu"
)

// Capability names backend/runtime features required by a module.
const (
	CapabilityCandidatePack   = "candidate_pack"
	CapabilityKVCache         = "kv_cache"
	CapabilityMaskedMeanPool  = "masked_mean_pool"
	CapabilityHostFallback    = "host_fallback"
	CapabilityDeviceExecution = "device_execution"
	CapabilityImageOps        = "image_ops"
	CapabilityTrainingLosses  = "training_losses"
	CapabilityTurboQuant      = "turboquant"
)

// EntryPointKind identifies a top-level entrypoint in the artifact.
type EntryPointKind string

const (
	EntryPointKernel   EntryPointKind = "kernel"
	EntryPointPipeline EntryPointKind = "pipeline"
)

// StepKind identifies a plan step in the executable artifact.
type StepKind string

const (
	StepAlloc        StepKind = "alloc"
	StepAlias        StepKind = "alias"
	StepGather       StepKind = "gather"
	StepPack         StepKind = "pack_candidates"
	StepReshape      StepKind = "reshape"
	StepTranspose    StepKind = "transpose"
	StepDequant      StepKind = "dequant"
	StepMatMul       StepKind = "matmul"
	StepTopK         StepKind = "topk"
	StepLaunchKernel StepKind = "launch_kernel"
	StepKVRead       StepKind = "kv_read"
	StepKVWrite      StepKind = "kv_write"
	StepCopy         StepKind = "copy"
	StepReturn       StepKind = "return"
	StepConv2D       StepKind = "conv2d"
	StepConv2DTrans  StepKind = "conv2d_transpose"
	StepGDN          StepKind = "gdn"
	StepIGDN         StepKind = "igdn"
	StepTurboQEncode StepKind = "turboquant_encode"
	StepTurboQDecode StepKind = "turboquant_decode"
	StepCrossEntropy StepKind = "cross_entropy_factorized"
	StepMSELoss      StepKind = "mse_loss"
	StepMSSSIMLoss   StepKind = "ms_ssim_loss"
	StepScalarAdd    StepKind = "scalar_add"
	StepRDLoss       StepKind = "rate_distortion_loss"
)

// ValueKind classifies the values used by params and entrypoints.
type ValueKind string

const (
	ValueTensor        ValueKind = "tensor"
	ValueKVCache       ValueKind = "kv_cache"
	ValueCandidatePack ValueKind = "candidate_pack"
)

// TensorType is the v0 data shape contract carried into the artifact.
type TensorType struct {
	DType string   `json:"dtype"`
	Shape []string `json:"shape,omitempty"`
}

// CandidatePackType is the retrieval-oriented packed output contract.
type CandidatePackType struct {
	Shape []string `json:"shape,omitempty"`
}

// ValueType describes a runtime-visible value.
type ValueType struct {
	Kind          ValueKind          `json:"kind"`
	Tensor        *TensorType        `json:"tensor,omitempty"`
	CandidatePack *CandidatePackType `json:"candidate_pack,omitempty"`
}

// Param describes an external binding such as a model weight.
type Param struct {
	Name      string    `json:"name"`
	Type      ValueType `json:"type"`
	Binding   string    `json:"binding"`
	Trainable bool      `json:"trainable,omitempty"`
}

// Module is the compiled Manta artifact.
type Module struct {
	Version      string         `json:"version"`
	Name         string         `json:"name"`
	Params       []Param        `json:"params,omitempty"`
	EntryPoints  []EntryPoint   `json:"entry_points"`
	Requirements Requirements   `json:"requirements"`
	Buffers      []Buffer       `json:"buffers,omitempty"`
	Kernels      []Kernel       `json:"kernels,omitempty"`
	Steps        []Step         `json:"steps,omitempty"`
	Metadata     map[string]any `json:"metadata,omitempty"`
}

// Requirements captures backend and capability needs.
type Requirements struct {
	SupportedBackends []BackendKind `json:"supported_backends"`
	Capabilities      []string      `json:"capabilities,omitempty"`
}

// EntryPoint is a kernel or pipeline entry exposed by the module.
type EntryPoint struct {
	Name    string         `json:"name"`
	Kind    EntryPointKind `json:"kind"`
	Inputs  []ValueBinding `json:"inputs,omitempty"`
	Outputs []ValueBinding `json:"outputs,omitempty"`
}

// ValueBinding names a value and its type in an entrypoint signature.
type ValueBinding struct {
	Name string    `json:"name"`
	Type ValueType `json:"type"`
}

// Buffer describes a runtime storage object.
type Buffer struct {
	Name         string   `json:"name"`
	DType        string   `json:"dtype"`
	Shape        []string `json:"shape,omitempty"`
	StorageClass string   `json:"storage_class,omitempty"`
}

// Kernel is a fused region with backend variants.
type Kernel struct {
	Name     string          `json:"name"`
	Inputs   []ValueBinding  `json:"inputs,omitempty"`
	Outputs  []ValueBinding  `json:"outputs,omitempty"`
	Hints    ScheduleHints   `json:"hints,omitempty"`
	Body     []KernelOp      `json:"body,omitempty"`
	Variants []KernelVariant `json:"variants,omitempty"`
}

// KernelVariant is backend-specific executable content.
type KernelVariant struct {
	Backend BackendKind       `json:"backend"`
	Entry   string            `json:"entry"`
	Source  string            `json:"source,omitempty"`
	Meta    map[string]string `json:"meta,omitempty"`
}

// ScheduleHints describe backend-neutral scheduling intent for a kernel body.
type ScheduleHints struct {
	Tile        []int  `json:"tile,omitempty"`
	Tile2D      []int  `json:"tile_2d,omitempty"`
	VectorWidth int    `json:"vector_width,omitempty"`
	Subgroup    bool   `json:"subgroup,omitempty"`
	Subgroup2D  []int  `json:"subgroup_2d,omitempty"`
	Halo        []int  `json:"halo,omitempty"`
	Memory      string `json:"memory,omitempty"`
}

// KernelOpKind identifies a kernel body operation.
type KernelOpKind string

const (
	KernelOpPointwise KernelOpKind = "pointwise"
	KernelOpReduce    KernelOpKind = "reduce"
	KernelOpBuiltin   KernelOpKind = "builtin"
	KernelOpReturn    KernelOpKind = "return"
)

// KernelOp is one backend-neutral operation inside a scheduled kernel body.
type KernelOp struct {
	Kind       KernelOpKind      `json:"kind"`
	Name       string            `json:"name,omitempty"`
	Op         string            `json:"op"`
	Inputs     []string          `json:"inputs,omitempty"`
	Outputs    []string          `json:"outputs,omitempty"`
	Attributes map[string]string `json:"attributes,omitempty"`
}

// Step is one high-level executable plan action.
type Step struct {
	Entry      string            `json:"entry,omitempty"`
	Kind       StepKind          `json:"kind"`
	Name       string            `json:"name,omitempty"`
	Kernel     string            `json:"kernel,omitempty"`
	Inputs     []string          `json:"inputs,omitempty"`
	Outputs    []string          `json:"outputs,omitempty"`
	Attributes map[string]string `json:"attributes,omitempty"`
}

// NewModule returns a minimal module skeleton.
func NewModule(name string) *Module {
	return &Module{
		Version: Version,
		Name:    name,
		Requirements: Requirements{
			SupportedBackends: []BackendKind{BackendCUDA, BackendMetal, BackendVulkan, BackendDirectML, BackendWebGPU},
		},
	}
}

// SupportsBackend reports whether the module may execute on the backend.
func (m *Module) SupportsBackend(kind BackendKind) bool {
	if m == nil {
		return false
	}
	for _, candidate := range m.Requirements.SupportedBackends {
		if candidate == kind {
			return true
		}
	}
	return false
}

// MissingCapabilities returns the required capabilities that are absent from available.
func MissingCapabilities(required, available []string) []string {
	if len(required) == 0 {
		return nil
	}
	have := map[string]bool{}
	for _, capability := range available {
		if capability == "" {
			continue
		}
		have[capability] = true
	}
	var missing []string
	for _, capability := range required {
		if capability == "" || have[capability] {
			continue
		}
		missing = append(missing, capability)
	}
	return missing
}

// Validate checks basic artifact invariants.
func (m *Module) Validate() error {
	if m == nil {
		return fmt.Errorf("nil module")
	}
	if m.Name == "" {
		return fmt.Errorf("module name is required")
	}
	if m.Version == "" {
		return fmt.Errorf("module version is required")
	}
	if m.Version != Version {
		return fmt.Errorf("module version %q is not supported, want %q", m.Version, Version)
	}
	if len(m.Requirements.SupportedBackends) == 0 {
		return fmt.Errorf("at least one supported backend is required")
	}
	seenBackends := map[BackendKind]bool{}
	for _, kind := range m.Requirements.SupportedBackends {
		if kind == "" {
			return fmt.Errorf("supported backend is required")
		}
		if seenBackends[kind] {
			return fmt.Errorf("duplicate supported backend %q", kind)
		}
		seenBackends[kind] = true
	}
	seenCapabilities := map[string]bool{}
	for _, capability := range m.Requirements.Capabilities {
		if capability == "" {
			return fmt.Errorf("capability name is required")
		}
		if seenCapabilities[capability] {
			return fmt.Errorf("duplicate capability %q", capability)
		}
		seenCapabilities[capability] = true
	}
	for _, param := range m.Params {
		if param.Name == "" {
			return fmt.Errorf("param name is required")
		}
		if param.Binding == "" {
			return fmt.Errorf("param %q binding is required", param.Name)
		}
		if err := validateValueType(param.Type); err != nil {
			return fmt.Errorf("param %q: %w", param.Name, err)
		}
	}
	entryByName := map[string]EntryPoint{}
	for _, entry := range m.EntryPoints {
		if entry.Name == "" {
			return fmt.Errorf("entrypoint name is required")
		}
		entryByName[entry.Name] = entry
		for _, input := range entry.Inputs {
			if input.Name == "" {
				return fmt.Errorf("entrypoint %q input name is required", entry.Name)
			}
			if err := validateValueType(input.Type); err != nil {
				return fmt.Errorf("entrypoint %q input %q: %w", entry.Name, input.Name, err)
			}
		}
		for _, output := range entry.Outputs {
			if output.Name == "" {
				return fmt.Errorf("entrypoint %q output name is required", entry.Name)
			}
			if err := validateValueType(output.Type); err != nil {
				return fmt.Errorf("entrypoint %q output %q: %w", entry.Name, output.Name, err)
			}
		}
	}
	bufferByName := map[string]Buffer{}
	for _, buf := range m.Buffers {
		if buf.Name == "" {
			return fmt.Errorf("buffer name is required")
		}
		if buf.DType == "" {
			return fmt.Errorf("buffer %q dtype is required", buf.Name)
		}
		bufferByName[buf.Name] = buf
	}
	kernelByName := map[string]Kernel{}
	for _, kernel := range m.Kernels {
		if kernel.Name == "" {
			return fmt.Errorf("kernel name is required")
		}
		kernelByName[kernel.Name] = kernel
		for _, input := range kernel.Inputs {
			if input.Name == "" {
				return fmt.Errorf("kernel %q input name is required", kernel.Name)
			}
			if err := validateValueType(input.Type); err != nil {
				return fmt.Errorf("kernel %q input %q: %w", kernel.Name, input.Name, err)
			}
		}
		for _, output := range kernel.Outputs {
			if output.Name == "" {
				return fmt.Errorf("kernel %q output name is required", kernel.Name)
			}
			if err := validateValueType(output.Type); err != nil {
				return fmt.Errorf("kernel %q output %q: %w", kernel.Name, output.Name, err)
			}
		}
		for _, op := range kernel.Body {
			if op.Op == "" {
				return fmt.Errorf("kernel %q has body op with empty op name", kernel.Name)
			}
		}
		if err := validateKernelVariants(m.Requirements.SupportedBackends, kernel); err != nil {
			return err
		}
	}
	paramByName := map[string]Param{}
	for _, param := range m.Params {
		paramByName[param.Name] = param
	}
	knownByEntry := map[string]map[string]bool{}
	for name, entry := range entryByName {
		known := map[string]bool{}
		for paramName := range paramByName {
			known[paramName] = true
		}
		for _, input := range entry.Inputs {
			known[input.Name] = true
		}
		knownByEntry[name] = known
	}
	for _, step := range m.Steps {
		if step.Entry == "" {
			return fmt.Errorf("step %q entry is required", step.Name)
		}
		entry, ok := entryByName[step.Entry]
		if !ok {
			return fmt.Errorf("step %q references unknown entrypoint %q", step.Name, step.Entry)
		}
		if step.Kind == StepLaunchKernel {
			if step.Kernel == "" {
				return fmt.Errorf("launch_kernel step %q is missing kernel name", step.Name)
			}
			if _, ok := kernelByName[step.Kernel]; !ok {
				return fmt.Errorf("step %q references unknown kernel %q", step.Name, step.Kernel)
			}
		}
		known := knownByEntry[step.Entry]
		for _, input := range step.Inputs {
			if input == "" {
				return fmt.Errorf("step %q has empty input name", step.Name)
			}
			if !known[input] {
				return fmt.Errorf("step %q references unknown input %q", step.Name, input)
			}
		}
		for _, output := range step.Outputs {
			if output == "" {
				return fmt.Errorf("step %q has empty output name", step.Name)
			}
			if step.Kind == StepReturn {
				if !entryHasOutput(entry, output) {
					return fmt.Errorf("return step %q output %q is not declared on entrypoint %q", step.Name, output, step.Entry)
				}
				continue
			}
			if _, ok := bufferByName[output]; !ok {
				return fmt.Errorf("step %q output %q is missing a declared buffer", step.Name, output)
			}
			known[output] = true
		}
	}
	return nil
}

func validateKernelVariants(required []BackendKind, kernel Kernel) error {
	seen := map[BackendKind]bool{}
	for _, variant := range kernel.Variants {
		if variant.Backend == "" {
			return fmt.Errorf("kernel %q variant backend is required", kernel.Name)
		}
		if seen[variant.Backend] {
			return fmt.Errorf("kernel %q has duplicate variant for backend %q", kernel.Name, variant.Backend)
		}
		seen[variant.Backend] = true
		if variant.Entry == "" {
			return fmt.Errorf("kernel %q variant entry is required for backend %q", kernel.Name, variant.Backend)
		}
		if variant.Source == "" {
			return fmt.Errorf("kernel %q variant source is required for backend %q", kernel.Name, variant.Backend)
		}
	}
	for _, backend := range required {
		if !seen[backend] {
			return fmt.Errorf("kernel %q is missing variant for backend %q", kernel.Name, backend)
		}
	}
	return nil
}

func entryHasOutput(entry EntryPoint, name string) bool {
	for _, output := range entry.Outputs {
		if output.Name == name {
			return true
		}
	}
	return false
}

func validateValueType(v ValueType) error {
	switch v.Kind {
	case ValueTensor:
		if v.Tensor == nil {
			return fmt.Errorf("tensor metadata is required")
		}
		if v.Tensor.DType == "" {
			return fmt.Errorf("tensor dtype is required")
		}
		return nil
	case ValueKVCache:
		return nil
	case ValueCandidatePack:
		if v.CandidatePack == nil {
			return fmt.Errorf("candidate pack metadata is required")
		}
		if rank := len(v.CandidatePack.Shape); rank != 2 && rank != 3 {
			return fmt.Errorf("candidate pack shape rank must be 2 or 3")
		}
		return nil
	case "":
		return fmt.Errorf("value kind is required")
	default:
		return fmt.Errorf("unsupported value kind %q", v.Kind)
	}
}

// EncodeJSON serializes the module.
func EncodeJSON(m *Module) ([]byte, error) {
	if err := m.Validate(); err != nil {
		return nil, err
	}
	return json.MarshalIndent(m, "", "  ")
}

// DecodeJSON deserializes a module.
func DecodeJSON(data []byte) (*Module, error) {
	var m Module
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, err
	}
	if err := m.Validate(); err != nil {
		return nil, err
	}
	return &m, nil
}
