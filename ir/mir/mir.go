package mir

// Module is the tensor semantics IR.
type Module struct {
	Name string
	Ops  []Op
}

// OpKind classifies a tensor operation in MIR.
type OpKind string

const (
	OpGather    OpKind = "gather"
	OpDequant   OpKind = "dequant"
	OpMatMul    OpKind = "matmul"
	OpTopK      OpKind = "topk"
	OpPointwise OpKind = "pointwise"
	OpReduce    OpKind = "reduce"
	OpSoftmax   OpKind = "softmax"
	OpRoPE      OpKind = "rope"
	OpScore     OpKind = "score"
	OpKVRead    OpKind = "kv_read"
	OpKVWrite   OpKind = "kv_write"
	OpKernel    OpKind = "kernel_call"
)

// Op is a backend-neutral tensor operation.
type Op struct {
	Name    string
	Kind    OpKind
	Callee  string
	Inputs  []string
	Outputs []string
}
