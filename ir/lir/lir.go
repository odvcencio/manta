package lir

import mantaartifact "github.com/odvcencio/manta/artifact/manta"

// Plan is the scheduled executable IR.
type Plan struct {
	Name    string
	Buffers []Buffer
	Kernels []Kernel
	Params  []Param
	Steps   []mantaartifact.Step
}

// Buffer is a scheduled storage object.
type Buffer struct {
	Name         string
	DType        string
	Shape        []string
	StorageClass string
}

// Kernel is a fused region eligible for backend-specific variants.
type Kernel struct {
	Name    string
	Inputs  []Value
	Outputs []Value
	Hints   ScheduleHints
	Body    []KernelOp
}

// Param is a scheduled external binding.
type Param struct {
	Name      string
	Binding   string
	DType     string
	Shape     []string
	Trainable bool
}

// ValueKind identifies a scheduled runtime value.
type ValueKind string

const (
	ValueTensor  ValueKind = "tensor"
	ValueKVCache ValueKind = "kv_cache"
)

// Value is a typed kernel input or output.
type Value struct {
	Name  string
	Kind  ValueKind
	DType string
	Shape []string
}

// ScheduleHints describe backend-neutral kernel scheduling intent.
type ScheduleHints struct {
	Tile        []int
	Tile2D      []int
	VectorWidth int
	Subgroup    bool
	Subgroup2D  []int
	Halo        []int
	Memory      string
}

// KernelOpKind identifies a lowered kernel body operation.
type KernelOpKind string

const (
	KernelOpPointwise KernelOpKind = "pointwise"
	KernelOpReduce    KernelOpKind = "reduce"
	KernelOpBuiltin   KernelOpKind = "builtin"
	KernelOpReturn    KernelOpKind = "return"
)

// KernelOp is one backend-neutral operation inside a scheduled kernel body.
type KernelOp struct {
	Kind       KernelOpKind
	Name       string
	Op         string
	Inputs     []string
	Outputs    []string
	Attributes map[string]string
}
