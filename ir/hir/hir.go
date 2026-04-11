package hir

import "github.com/odvcencio/manta/syntax"

// Module is the source-oriented typed IR.
type Module struct {
	Name        string
	Params      []Param
	EntryPoints []EntryPoint
	Diagnostics []syntax.Diagnostic
}

// EntryPointKind identifies a top-level executable unit.
type EntryPointKind string

const (
	EntryPointKernel   EntryPointKind = "kernel"
	EntryPointPipeline EntryPointKind = "pipeline"
)

// EntryPoint is a kernel or pipeline with typed tensor inputs and outputs.
type EntryPoint struct {
	Name    string
	Kind    EntryPointKind
	Inputs  []Value
	Outputs []Value
}

// Param is an externally bound model parameter.
type Param struct {
	Name      string
	Binding   string
	Type      TensorType
	Trainable bool
}

// Value is a named typed input or output.
type Value struct {
	Name string
	Type Type
}

// TypeKind identifies HIR value kinds.
type TypeKind string

const (
	TypeTensor        TypeKind = "tensor"
	TypeKVCache       TypeKind = "kv_cache"
	TypeCandidatePack TypeKind = "candidate_pack"
)

// Type is the HIR-level value type.
type Type struct {
	Kind          TypeKind
	Tensor        *TensorType
	CandidatePack *CandidatePackType
}

// TensorType is the v0 tensor type shape.
type TensorType struct {
	DType string
	Shape []DimExpr
}

// CandidatePackType is the HIR retrieval-pack shape.
type CandidatePackType struct {
	Shape []DimExpr
}

// DimExpr is a symbolic dimension name in v0.
type DimExpr struct {
	Name string
}
