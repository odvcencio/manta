package manta

import legacy "github.com/odvcencio/manta/artifact/barr"

const (
	Version            = legacy.Version
	MLLMetadataVersion = legacy.MLLMetadataVersion

	BackendCUDA  = legacy.BackendCUDA
	BackendMetal = legacy.BackendMetal

	CapabilityCandidatePack   = legacy.CapabilityCandidatePack
	CapabilityKVCache         = legacy.CapabilityKVCache
	CapabilityMaskedMeanPool  = legacy.CapabilityMaskedMeanPool
	CapabilityHostFallback    = legacy.CapabilityHostFallback
	CapabilityDeviceExecution = legacy.CapabilityDeviceExecution

	EntryPointKernel   = legacy.EntryPointKernel
	EntryPointPipeline = legacy.EntryPointPipeline

	StepAlloc        = legacy.StepAlloc
	StepAlias        = legacy.StepAlias
	StepGather       = legacy.StepGather
	StepPack         = legacy.StepPack
	StepReshape      = legacy.StepReshape
	StepTranspose    = legacy.StepTranspose
	StepDequant      = legacy.StepDequant
	StepMatMul       = legacy.StepMatMul
	StepTopK         = legacy.StepTopK
	StepLaunchKernel = legacy.StepLaunchKernel
	StepKVRead       = legacy.StepKVRead
	StepKVWrite      = legacy.StepKVWrite
	StepCopy         = legacy.StepCopy
	StepReturn       = legacy.StepReturn

	ValueTensor        = legacy.ValueTensor
	ValueKVCache       = legacy.ValueKVCache
	ValueCandidatePack = legacy.ValueCandidatePack

	KernelOpPointwise = legacy.KernelOpPointwise
	KernelOpReduce    = legacy.KernelOpReduce
	KernelOpBuiltin   = legacy.KernelOpBuiltin
	KernelOpReturn    = legacy.KernelOpReturn
)

var MLLTagXBAR = legacy.MLLTagXBAR

type (
	BackendKind       = legacy.BackendKind
	EntryPointKind    = legacy.EntryPointKind
	StepKind          = legacy.StepKind
	ValueKind         = legacy.ValueKind
	TensorType        = legacy.TensorType
	CandidatePackType = legacy.CandidatePackType
	ValueType         = legacy.ValueType
	Param             = legacy.Param
	Module            = legacy.Module
	Requirements      = legacy.Requirements
	EntryPoint        = legacy.EntryPoint
	ValueBinding      = legacy.ValueBinding
	Buffer            = legacy.Buffer
	Kernel            = legacy.Kernel
	KernelVariant     = legacy.KernelVariant
	ScheduleHints     = legacy.ScheduleHints
	KernelOpKind      = legacy.KernelOpKind
	KernelOp          = legacy.KernelOp
	Step              = legacy.Step
	MLLMetadata       = legacy.MLLMetadata
)

func NewModule(name string) *Module {
	return legacy.NewModule(name)
}

func MissingCapabilities(required, available []string) []string {
	return legacy.MissingCapabilities(required, available)
}

func EncodeJSON(m *Module) ([]byte, error) {
	return legacy.EncodeJSON(m)
}

func DecodeJSON(data []byte) (*Module, error) {
	return legacy.DecodeJSON(data)
}

func EncodeMLLMetadata(meta MLLMetadata) ([]byte, error) {
	return legacy.EncodeMLLMetadata(meta)
}

func DecodeMLLMetadata(data []byte) (MLLMetadata, error) {
	return legacy.DecodeMLLMetadata(data)
}

func IsMLLBytes(data []byte) bool {
	return legacy.IsMLLBytes(data)
}

func EncodeMLL(mod *Module) ([]byte, error) {
	return legacy.EncodeMLL(mod)
}

func DecodeMLL(data []byte) (*Module, error) {
	return legacy.DecodeMLL(data)
}

func ReadFile(path string) (*Module, error) {
	return legacy.ReadFile(path)
}

func WriteFile(path string, m *Module) error {
	return legacy.WriteFile(path, m)
}
