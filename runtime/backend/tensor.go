package backend

import (
	"fmt"
	"slices"
	"strconv"

	mantaartifact "github.com/odvcencio/manta/artifact/manta"
)

// Tensor is the bootstrap dense runtime tensor representation.
// f16 values are stored in F32 with DType=f16 until a real device path exists.
type Tensor struct {
	DType string
	Shape []int
	F32   []float32
	I32   []int32
	I64   []int64
}

// KVCache is the mutable runtime cache object used by kv_read/kv_write.
type KVCache struct {
	Value *Tensor
}

// CandidatePack is the retrieval-native packed result object.
type CandidatePack struct {
	IDs    *Tensor
	Scores *Tensor
	Docs   *Tensor
}

func NewTensorF32(shape []int, data []float32) *Tensor {
	return &Tensor{
		DType: "f32",
		Shape: append([]int(nil), shape...),
		F32:   append([]float32(nil), data...),
	}
}

func NewTensorF16(shape []int, data []float32) *Tensor {
	return &Tensor{
		DType: "f16",
		Shape: append([]int(nil), shape...),
		F32:   append([]float32(nil), data...),
	}
}

func NewTensorQ4(shape []int, data []float32) *Tensor {
	return &Tensor{
		DType: "q4",
		Shape: append([]int(nil), shape...),
		F32:   append([]float32(nil), data...),
	}
}

func NewTensorQ8(shape []int, data []float32) *Tensor {
	return &Tensor{
		DType: "q8",
		Shape: append([]int(nil), shape...),
		F32:   append([]float32(nil), data...),
	}
}

func NewTensorI32(shape []int, data []int32) *Tensor {
	return &Tensor{
		DType: "i32",
		Shape: append([]int(nil), shape...),
		I32:   append([]int32(nil), data...),
	}
}

func NewTensorI64(shape []int, data []int64) *Tensor {
	return &Tensor{
		DType: "i64",
		Shape: append([]int(nil), shape...),
		I64:   append([]int64(nil), data...),
	}
}

func NewKVCache(value *Tensor) *KVCache {
	cache := &KVCache{}
	if value != nil {
		cache.Value = value.Clone()
	}
	return cache
}

func NewCandidatePack(ids, scores, docs *Tensor) *CandidatePack {
	pack := &CandidatePack{}
	if ids != nil {
		pack.IDs = ids.Clone()
	}
	if scores != nil {
		pack.Scores = scores.Clone()
	}
	if docs != nil {
		pack.Docs = docs.Clone()
	}
	return pack
}

func (t *Tensor) Clone() *Tensor {
	if t == nil {
		return nil
	}
	return &Tensor{
		DType: t.DType,
		Shape: append([]int(nil), t.Shape...),
		F32:   append([]float32(nil), t.F32...),
		I32:   append([]int32(nil), t.I32...),
		I64:   append([]int64(nil), t.I64...),
	}
}

func (p *CandidatePack) Clone() *CandidatePack {
	if p == nil {
		return nil
	}
	return &CandidatePack{
		IDs:    p.IDs.Clone(),
		Scores: p.Scores.Clone(),
		Docs:   p.Docs.Clone(),
	}
}

func (t *Tensor) Rank() int {
	if t == nil {
		return 0
	}
	return len(t.Shape)
}

func (t *Tensor) Elements() int {
	if t == nil {
		return 0
	}
	if len(t.Shape) == 0 {
		return 1
	}
	n := 1
	for _, dim := range t.Shape {
		n *= dim
	}
	return n
}

func (t *Tensor) EqualShape(other *Tensor) bool {
	if t == nil || other == nil {
		return false
	}
	return slices.Equal(t.Shape, other.Shape)
}

func MaterializeValue(typ mantaartifact.ValueType, data any) (any, error) {
	switch typ.Kind {
	case mantaartifact.ValueTensor:
		return materializeTensor(typ, data)
	case mantaartifact.ValueKVCache:
		return materializeKVCache(data)
	case mantaartifact.ValueCandidatePack:
		return materializeCandidatePack(typ, data)
	default:
		return nil, fmt.Errorf("unsupported runtime value kind %q", typ.Kind)
	}
}

func PreviewValue(typ mantaartifact.ValueType, data any) (any, error) {
	switch typ.Kind {
	case mantaartifact.ValueTensor:
		return previewTensor(typ, data)
	case mantaartifact.ValueKVCache:
		return previewKVCache(data)
	case mantaartifact.ValueCandidatePack:
		return previewCandidatePack(typ, data)
	default:
		return nil, fmt.Errorf("unsupported runtime value kind %q", typ.Kind)
	}
}

func MaterializeValueWithBindings(typ mantaartifact.ValueType, data any, bindings map[string]int) (any, mantaartifact.ValueType, error) {
	value, err := MaterializeValue(typ, data)
	if err != nil {
		return nil, mantaartifact.ValueType{}, err
	}
	concrete, err := concretizeValueType(typ, value, bindings)
	if err != nil {
		return nil, mantaartifact.ValueType{}, err
	}
	return value, concrete, nil
}

func PreviewValueWithBindings(typ mantaartifact.ValueType, data any, bindings map[string]int) (any, mantaartifact.ValueType, error) {
	value, err := PreviewValue(typ, data)
	if err != nil {
		return nil, mantaartifact.ValueType{}, err
	}
	concrete, err := concretizeValueType(typ, value, bindings)
	if err != nil {
		return nil, mantaartifact.ValueType{}, err
	}
	return value, concrete, nil
}

func materializeTensor(typ mantaartifact.ValueType, data any) (*Tensor, error) {
	switch v := data.(type) {
	case *Tensor:
		if err := validateTensorType(typ, v); err != nil {
			return nil, err
		}
		return v.Clone(), nil
	case Tensor:
		cp := v.Clone()
		if err := validateTensorType(typ, cp); err != nil {
			return nil, err
		}
		return cp, nil
	case []int32:
		if typ.Tensor != nil && typ.Tensor.DType == "i64" {
			out := make([]int64, len(v))
			for i, n := range v {
				out[i] = int64(n)
			}
			t := NewTensorI64([]int{len(out)}, out)
			if err := validateTensorType(typ, t); err != nil {
				return nil, err
			}
			return t, nil
		}
		t := NewTensorI32([]int{len(v)}, v)
		if err := validateTensorType(typ, t); err != nil {
			return nil, err
		}
		return t, nil
	case []int64:
		t := NewTensorI64([]int{len(v)}, v)
		if err := validateTensorType(typ, t); err != nil {
			return nil, err
		}
		return t, nil
	case []int:
		if typ.Tensor != nil && typ.Tensor.DType == "i64" {
			out := make([]int64, len(v))
			for i, n := range v {
				out[i] = int64(n)
			}
			t := NewTensorI64([]int{len(out)}, out)
			if err := validateTensorType(typ, t); err != nil {
				return nil, err
			}
			return t, nil
		}
		out := make([]int32, len(v))
		for i, n := range v {
			out[i] = int32(n)
		}
		t := NewTensorI32([]int{len(out)}, out)
		if err := validateTensorType(typ, t); err != nil {
			return nil, err
		}
		return t, nil
	case []float32:
		dtype := "f32"
		if typ.Tensor != nil && typ.Tensor.DType != "" {
			dtype = typ.Tensor.DType
		}
		t := &Tensor{
			DType: dtype,
			Shape: []int{len(v)},
			F32:   append([]float32(nil), v...),
		}
		if err := validateTensorType(typ, t); err != nil {
			return nil, err
		}
		return t, nil
	default:
		return nil, fmt.Errorf("expected tensor input, got %T", data)
	}
}

func previewTensor(typ mantaartifact.ValueType, data any) (*Tensor, error) {
	switch v := data.(type) {
	case *Tensor:
		if err := validateTensorType(typ, v); err != nil {
			return nil, err
		}
		return v, nil
	case Tensor:
		cp := v.Clone()
		if err := validateTensorType(typ, cp); err != nil {
			return nil, err
		}
		return cp, nil
	case []int32, []int64, []int, []float32:
		return materializeTensor(typ, data)
	default:
		return nil, fmt.Errorf("expected tensor input, got %T", data)
	}
}

func materializeKVCache(data any) (*KVCache, error) {
	switch v := data.(type) {
	case *KVCache:
		if v == nil {
			return &KVCache{}, nil
		}
		return v, nil
	case KVCache:
		return NewKVCache(v.Value), nil
	case *Tensor:
		return NewKVCache(v), nil
	case nil:
		return &KVCache{}, nil
	default:
		return nil, fmt.Errorf("expected kv_cache input, got %T", data)
	}
}

func previewKVCache(data any) (*KVCache, error) {
	switch v := data.(type) {
	case *KVCache:
		if v == nil {
			return &KVCache{}, nil
		}
		return v, nil
	case KVCache, *Tensor, nil:
		return materializeKVCache(data)
	default:
		return nil, fmt.Errorf("expected kv_cache input, got %T", data)
	}
}

func materializeCandidatePack(typ mantaartifact.ValueType, data any) (*CandidatePack, error) {
	switch v := data.(type) {
	case *CandidatePack:
		if err := validateCandidatePackType(typ, v); err != nil {
			return nil, err
		}
		return v.Clone(), nil
	case CandidatePack:
		cp := v.Clone()
		if err := validateCandidatePackType(typ, cp); err != nil {
			return nil, err
		}
		return cp, nil
	default:
		return nil, fmt.Errorf("expected candidate_pack input, got %T", data)
	}
}

func previewCandidatePack(typ mantaartifact.ValueType, data any) (*CandidatePack, error) {
	switch v := data.(type) {
	case *CandidatePack:
		if err := validateCandidatePackType(typ, v); err != nil {
			return nil, err
		}
		return v, nil
	case CandidatePack:
		cp := v.Clone()
		if err := validateCandidatePackType(typ, cp); err != nil {
			return nil, err
		}
		return cp, nil
	default:
		return nil, fmt.Errorf("expected candidate_pack input, got %T", data)
	}
}

func validateTensorType(typ mantaartifact.ValueType, t *Tensor) error {
	if t == nil {
		return fmt.Errorf("tensor is nil")
	}
	if typ.Kind != mantaartifact.ValueTensor || typ.Tensor == nil {
		return fmt.Errorf("expected tensor type metadata")
	}
	expected := typ.Tensor.DType
	if expected != "" && t.DType != expected && !(expected == "f16" && t.DType == "f32") {
		return fmt.Errorf("tensor dtype %q does not match expected %q", t.DType, expected)
	}
	if len(typ.Tensor.Shape) > 0 && len(typ.Tensor.Shape) != len(t.Shape) {
		return fmt.Errorf("tensor rank %d does not match expected rank %d", len(t.Shape), len(typ.Tensor.Shape))
	}
	if elems := t.Elements(); elems != len(t.F32) && elems != len(t.I32) && elems != len(t.I64) {
		return fmt.Errorf("tensor element count %d does not match backing data", elems)
	}
	return nil
}

func validateCandidatePackType(typ mantaartifact.ValueType, pack *CandidatePack) error {
	if pack == nil {
		return fmt.Errorf("candidate pack is nil")
	}
	if typ.Kind != mantaartifact.ValueCandidatePack || typ.CandidatePack == nil {
		return fmt.Errorf("expected candidate pack type metadata")
	}
	if pack.IDs == nil || pack.Scores == nil || pack.Docs == nil {
		return fmt.Errorf("candidate pack requires ids, scores, and docs")
	}
	if pack.IDs.DType != "i64" {
		return fmt.Errorf("candidate pack ids dtype %q does not match expected %q", pack.IDs.DType, "i64")
	}
	if pack.Scores.DType != "f32" {
		return fmt.Errorf("candidate pack scores dtype %q does not match expected %q", pack.Scores.DType, "f32")
	}
	if pack.Docs.DType != "q4" {
		return fmt.Errorf("candidate pack docs dtype %q does not match expected %q", pack.Docs.DType, "q4")
	}
	if len(typ.CandidatePack.Shape) != len(pack.Docs.Shape) {
		return fmt.Errorf("candidate pack rank %d does not match expected rank %d", len(pack.Docs.Shape), len(typ.CandidatePack.Shape))
	}
	if len(pack.Docs.Shape) == 2 {
		if len(pack.IDs.Shape) != 1 || len(pack.Scores.Shape) != 1 {
			return fmt.Errorf("candidate pack rank-2 docs require rank-1 ids and scores")
		}
		if pack.IDs.Shape[0] != pack.Docs.Shape[0] || pack.Scores.Shape[0] != pack.Docs.Shape[0] {
			return fmt.Errorf("candidate pack leading dimension mismatch")
		}
	}
	if len(pack.Docs.Shape) == 3 {
		if len(pack.IDs.Shape) != 2 || len(pack.Scores.Shape) != 2 {
			return fmt.Errorf("candidate pack rank-3 docs require rank-2 ids and scores")
		}
		if pack.IDs.Shape[0] != pack.Docs.Shape[0] || pack.Scores.Shape[0] != pack.Docs.Shape[0] || pack.IDs.Shape[1] != pack.Docs.Shape[1] || pack.Scores.Shape[1] != pack.Docs.Shape[1] {
			return fmt.Errorf("candidate pack leading dimensions mismatch")
		}
	}
	return nil
}

func concretizeValueType(typ mantaartifact.ValueType, value any, bindings map[string]int) (mantaartifact.ValueType, error) {
	switch typ.Kind {
	case mantaartifact.ValueTensor:
		t, ok := value.(*Tensor)
		if !ok || t == nil {
			return mantaartifact.ValueType{}, fmt.Errorf("expected tensor value, got %T", value)
		}
		if typ.Tensor == nil {
			return mantaartifact.ValueType{}, fmt.Errorf("tensor type metadata missing")
		}
		if err := bindShape(typ.Tensor.Shape, t.Shape, bindings); err != nil {
			return mantaartifact.ValueType{}, err
		}
		return mantaartifact.ValueType{
			Kind: mantaartifact.ValueTensor,
			Tensor: &mantaartifact.TensorType{
				DType: typ.Tensor.DType,
				Shape: concreteShapeStrings(typ.Tensor.Shape, t.Shape),
			},
		}, nil
	case mantaartifact.ValueKVCache:
		return mantaartifact.ValueType{Kind: mantaartifact.ValueKVCache}, nil
	case mantaartifact.ValueCandidatePack:
		pack, ok := value.(*CandidatePack)
		if !ok || pack == nil {
			return mantaartifact.ValueType{}, fmt.Errorf("expected candidate pack value, got %T", value)
		}
		if typ.CandidatePack == nil {
			return mantaartifact.ValueType{}, fmt.Errorf("candidate pack type metadata missing")
		}
		if err := validateCandidatePackType(typ, pack); err != nil {
			return mantaartifact.ValueType{}, err
		}
		if err := bindShape(typ.CandidatePack.Shape, pack.Docs.Shape, bindings); err != nil {
			return mantaartifact.ValueType{}, err
		}
		return mantaartifact.ValueType{
			Kind:          mantaartifact.ValueCandidatePack,
			CandidatePack: &mantaartifact.CandidatePackType{Shape: concreteShapeStrings(typ.CandidatePack.Shape, pack.Docs.Shape)},
		}, nil
	default:
		return mantaartifact.ValueType{}, fmt.Errorf("unsupported runtime value kind %q", typ.Kind)
	}
}

func bindShape(symbols []string, actual []int, bindings map[string]int) error {
	if len(symbols) == 0 {
		return nil
	}
	if len(symbols) != len(actual) {
		return fmt.Errorf("rank mismatch: expected %d dims, got %d", len(symbols), len(actual))
	}
	for i, symbol := range symbols {
		if symbol == "" {
			continue
		}
		if literal, err := strconv.Atoi(symbol); err == nil {
			if literal != actual[i] {
				return fmt.Errorf("dimension %d mismatch: expected %d, got %d", i, literal, actual[i])
			}
			continue
		}
		if prev, ok := bindings[symbol]; ok {
			if prev != actual[i] {
				return fmt.Errorf("symbol %q mismatch: expected %d, got %d", symbol, prev, actual[i])
			}
			continue
		}
		bindings[symbol] = actual[i]
	}
	return nil
}

func concreteShapeStrings(symbols []string, actual []int) []string {
	if len(actual) == 0 {
		return nil
	}
	out := make([]string, len(actual))
	for i, dim := range actual {
		out[i] = strconv.Itoa(dim)
	}
	return out
}

func requireTensor(value Value, name string) (*Tensor, error) {
	t, ok := value.Data.(*Tensor)
	if !ok || t == nil {
		return nil, fmt.Errorf("%s is not a tensor", name)
	}
	return t, nil
}

func requireKVCache(value Value, name string) (*KVCache, error) {
	cache, ok := value.Data.(*KVCache)
	if !ok || cache == nil {
		return nil, fmt.Errorf("%s is not a kv_cache", name)
	}
	return cache, nil
}

func requireCandidatePack(value Value, name string) (*CandidatePack, error) {
	pack, ok := value.Data.(*CandidatePack)
	if !ok || pack == nil {
		return nil, fmt.Errorf("%s is not a candidate_pack", name)
	}
	return pack, nil
}
