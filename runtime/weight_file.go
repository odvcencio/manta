package barruntime

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"

	"github.com/odvcencio/manta/artifact/barr"
	"github.com/odvcencio/manta/runtime/backend"
	mll "github.com/odvcencio/mll"
)

// WeightFileVersion identifies the serialized Manta weight file schema.
const WeightFileVersion = "manta/weights/v0alpha1"

var tagXWGT = [4]byte{'X', 'W', 'G', 'T'}

// WeightFile stores named runtime tensors for package loading.
type WeightFile struct {
	Version string                     `json:"version"`
	Weights map[string]*backend.Tensor `json:"weights"`
}

type weightFileMLLMetadata struct {
	Version            string            `json:"version"`
	LogicalTensorDType map[string]string `json:"logical_tensor_dtypes,omitempty"`
}

// DefaultWeightFilePath returns the conventional sibling weight path for an artifact.
func DefaultWeightFilePath(barrPath string) string {
	return defaultManifestPath(barrPath, ".weights.mll")
}

// NewWeightFile clones the provided named tensors into a serializable weight file.
func NewWeightFile(weights map[string]*backend.Tensor) WeightFile {
	out := WeightFile{
		Version: WeightFileVersion,
		Weights: make(map[string]*backend.Tensor, len(weights)),
	}
	for name, tensor := range weights {
		if tensor == nil {
			continue
		}
		out.Weights[name] = tensor.Clone()
	}
	return out
}

// Validate checks the serialized weight file contract.
func (w WeightFile) Validate() error {
	if w.Version == "" {
		return fmt.Errorf("weight file version is required")
	}
	if w.Version != WeightFileVersion {
		return fmt.Errorf("weight file version %q is not supported, want %q", w.Version, WeightFileVersion)
	}
	if len(w.Weights) == 0 {
		return fmt.Errorf("weight file has no weights")
	}
	for name, tensor := range w.Weights {
		if name == "" {
			return fmt.Errorf("weight name is required")
		}
		if tensor == nil {
			return fmt.Errorf("weight %q is nil", name)
		}
		if tensor.DType == "" {
			return fmt.Errorf("weight %q dtype is required", name)
		}
		if len(tensor.Shape) == 0 {
			return fmt.Errorf("weight %q shape is required", name)
		}
	}
	return nil
}

// WriteFile writes the weight file as an MLL weights-only container.
func (w WeightFile) WriteFile(path string) error {
	if err := w.Validate(); err != nil {
		return err
	}
	data, err := encodeWeightFileMLL(w)
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}

// ReadWeightFile reads a serialized MLL weight file.
func ReadWeightFile(path string) (WeightFile, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return WeightFile{}, err
	}
	if !barr.IsMLLBytes(data) {
		return WeightFile{}, fmt.Errorf("weight file %q is not an MLL file", path)
	}
	return decodeWeightFileMLL(data)
}

// LoadOptions converts a weight file into runtime load options.
func (w WeightFile) LoadOptions() []LoadOption {
	if len(w.Weights) == 0 {
		return nil
	}
	names := make([]string, 0, len(w.Weights))
	for name := range w.Weights {
		names = append(names, name)
	}
	sort.Strings(names)
	opts := make([]LoadOption, 0, len(names))
	for _, name := range names {
		opts = append(opts, WithWeight(name, w.Weights[name].Clone()))
	}
	return opts
}

func encodeWeightFileMLL(weights WeightFile) ([]byte, error) {
	strg := mll.NewStringTableBuilder()
	strg.Intern("")
	typeBuilder := mll.NewTypeBuilder()
	parmBuilder := mll.NewParmBuilder()
	tnsrBuilder := mll.NewTnsrBuilder()
	logicalDTypes := map[string]string{}

	names := make([]string, 0, len(weights.Weights))
	for name := range weights.Weights {
		names = append(names, name)
	}
	sort.Strings(names)

	for _, name := range names {
		tensor := weights.Weights[name]
		if tensor == nil {
			continue
		}
		shape, err := intShapeToMLLShape(strg, tensor.Shape)
		if err != nil {
			return nil, fmt.Errorf("weight %q shape: %w", name, err)
		}
		storageDType, raw, widened, err := encodeTensorStorage(tensor)
		if err != nil {
			return nil, fmt.Errorf("weight %q encode: %w", name, err)
		}
		if widened {
			logicalDTypes[name] = tensor.DType
		}
		nameIdx := strg.Intern(name)
		typeIndex := typeBuilder.AddTensorType(nameIdx, storageDType, shape)
		parmBuilder.Add(mll.ParmDecl{
			NameIdx: nameIdx,
			TypeRef: mll.Ref{Tag: mll.TagTYPE, Index: typeIndex},
		})
		shape64 := make([]uint64, len(tensor.Shape))
		for i, dim := range tensor.Shape {
			if dim < 0 {
				return nil, fmt.Errorf("weight %q dim %d is negative", name, dim)
			}
			shape64[i] = uint64(dim)
		}
		tnsrBuilder.Add(mll.TensorEntry{
			NameIdx: nameIdx,
			DType:   storageDType,
			Shape:   shape64,
			Data:    raw,
		})
	}

	head := mll.HeadSection{
		Name:        strg.Intern("manta-weights"),
		Description: strg.Intern("Manta weight container"),
		Metadata: []mll.HeadMetadataEntry{
			headStringMeta(strg, "weight_file_version", weights.Version),
			headIntMeta(strg, "weight_count", int64(len(names))),
		},
	}
	metaBody, err := json.Marshal(weightFileMLLMetadata{
		Version:            weights.Version,
		LogicalTensorDType: logicalDTypes,
	})
	if err != nil {
		return nil, err
	}

	sections := make([]mll.SectionInput, 0, 6)
	if body, digestBody, err := encodeHeadSection(head, mll.ProfileWeightsOnly); err != nil {
		return nil, err
	} else {
		sections = append(sections, mll.SectionInput{
			Tag:           mll.TagHEAD,
			Body:          body,
			DigestBody:    digestBody,
			Flags:         mll.SectionFlagRequired,
			SchemaVersion: 1,
		})
	}
	if body, err := encodeSection(strg.Write); err != nil {
		return nil, err
	} else {
		sections = append(sections, mll.SectionInput{Tag: mll.TagSTRG, Body: body, Flags: mll.SectionFlagRequired, SchemaVersion: 1})
	}
	if body, err := encodeSection(typeBuilder.Write); err != nil {
		return nil, err
	} else {
		sections = append(sections, mll.SectionInput{Tag: mll.TagTYPE, Body: body, SchemaVersion: 1})
	}
	if body, err := encodeSection(parmBuilder.Write); err != nil {
		return nil, err
	} else {
		sections = append(sections, mll.SectionInput{Tag: mll.TagPARM, Body: body, Flags: mll.SectionFlagRequired, SchemaVersion: 1})
	}
	if body, err := encodeSection(tnsrBuilder.Write); err != nil {
		return nil, err
	} else {
		sections = append(sections, mll.SectionInput{Tag: mll.TagTNSR, Body: body, Flags: mll.SectionFlagRequired | mll.SectionFlagAligned, SchemaVersion: 1})
	}
	sections = append(sections, mll.SectionInput{
		Tag:           tagXWGT,
		Body:          metaBody,
		Flags:         mll.SectionFlagSkippable | mll.SectionFlagSchemaless,
		SchemaVersion: 1,
	})

	return mll.WriteToBytes(mll.ProfileWeightsOnly, mll.V1_0, sections)
}

func decodeWeightFileMLL(data []byte) (WeightFile, error) {
	reader, err := mll.ReadBytes(data, mll.WithDigestVerification())
	if err != nil {
		return WeightFile{}, err
	}
	if reader.Profile() != mll.ProfileWeightsOnly {
		return WeightFile{}, fmt.Errorf("weight file profile = %d, want %d", reader.Profile(), mll.ProfileWeightsOnly)
	}
	meta := weightFileMLLMetadata{Version: WeightFileVersion}
	if body, ok := reader.Section(tagXWGT); ok {
		if err := json.Unmarshal(body, &meta); err != nil {
			return WeightFile{}, err
		}
	}
	return decodeWeightFileFromMLLReader(reader, meta.Version, meta.LogicalTensorDType)
}

func decodeWeightFileFromMLLReader(reader *mll.Reader, version string, logicalTensorDTypes map[string]string) (WeightFile, error) {
	if reader == nil {
		return WeightFile{}, fmt.Errorf("nil MLL reader")
	}
	strgBody, ok := reader.Section(mll.TagSTRG)
	if !ok {
		return WeightFile{}, fmt.Errorf("weight file missing STRG section")
	}
	strg, err := mll.ReadStringTable(strgBody)
	if err != nil {
		return WeightFile{}, err
	}
	tnsrBody, ok := reader.Section(mll.TagTNSR)
	if !ok {
		return WeightFile{}, fmt.Errorf("weight file missing TNSR section")
	}
	tnsr, err := mll.ReadTnsrSection(tnsrBody)
	if err != nil {
		return WeightFile{}, err
	}
	if version == "" {
		version = WeightFileVersion
	}
	weights := WeightFile{
		Version: version,
		Weights: make(map[string]*backend.Tensor, len(tnsr.Tensors)),
	}
	for _, entry := range tnsr.Tensors {
		name := strg.At(entry.NameIdx)
		if name == "" {
			return WeightFile{}, fmt.Errorf("weight tensor missing name for index %d", entry.NameIdx)
		}
		logical := logicalTensorDTypes[name]
		tensor, err := decodeTensorEntry(entry, logical)
		if err != nil {
			return WeightFile{}, fmt.Errorf("decode weight %q: %w", name, err)
		}
		weights.Weights[name] = tensor
	}
	if err := weights.Validate(); err != nil {
		return WeightFile{}, err
	}
	return weights, nil
}

func intShapeToMLLShape(strg *mll.StringTable, shape []int) ([]mll.Dimension, error) {
	out := make([]mll.Dimension, len(shape))
	for i, dim := range shape {
		if dim <= 0 {
			return nil, fmt.Errorf("shape dim %d must be positive", dim)
		}
		out[i] = mll.DimLiteral(int64(dim))
		_ = strg
	}
	return out, nil
}

func decodeTensorEntry(entry mll.TensorEntry, logicalDType string) (*backend.Tensor, error) {
	shape := make([]int, len(entry.Shape))
	for i, dim := range entry.Shape {
		if dim > math.MaxInt {
			return nil, fmt.Errorf("shape dim %d overflows int", dim)
		}
		shape[i] = int(dim)
	}
	dtype := logicalDType
	if dtype == "" {
		var err error
		dtype, err = mllDTypeToBarr(entry.DType)
		if err != nil {
			return nil, err
		}
	}
	switch dtype {
	case "i32":
		if len(entry.Data)%4 != 0 {
			return nil, fmt.Errorf("i32 tensor data length %d is not aligned", len(entry.Data))
		}
		data := make([]int32, len(entry.Data)/4)
		for i := range data {
			data[i] = int32(binary.LittleEndian.Uint32(entry.Data[i*4:]))
		}
		return backend.NewTensorI32(shape, data), nil
	case "i64":
		if len(entry.Data)%8 != 0 {
			return nil, fmt.Errorf("i64 tensor data length %d is not aligned", len(entry.Data))
		}
		data := make([]int64, len(entry.Data)/8)
		for i := range data {
			data[i] = int64(binary.LittleEndian.Uint64(entry.Data[i*8:]))
		}
		return backend.NewTensorI64(shape, data), nil
	case "f16":
		if len(entry.Data)%2 != 0 {
			return nil, fmt.Errorf("f16 tensor data length %d is not aligned", len(entry.Data))
		}
		data := make([]float32, len(entry.Data)/2)
		for i := range data {
			data[i] = halfToFloat32(binary.LittleEndian.Uint16(entry.Data[i*2:]))
		}
		return backend.NewTensorF16(shape, data), nil
	case "f32", "q4", "q8":
		if len(entry.Data)%4 != 0 {
			return nil, fmt.Errorf("%s tensor data length %d is not aligned", dtype, len(entry.Data))
		}
		data := make([]float32, len(entry.Data)/4)
		for i := range data {
			data[i] = math.Float32frombits(binary.LittleEndian.Uint32(entry.Data[i*4:]))
		}
		switch dtype {
		case "q4":
			return backend.NewTensorQ4(shape, data), nil
		case "q8":
			return backend.NewTensorQ8(shape, data), nil
		default:
			return backend.NewTensorF32(shape, data), nil
		}
	default:
		return nil, fmt.Errorf("unsupported tensor dtype %q", dtype)
	}
}

func mllDTypeToBarr(dtype mll.DType) (string, error) {
	switch dtype {
	case mll.DTypeI32:
		return "i32", nil
	case mll.DTypeI64:
		return "i64", nil
	case mll.DTypeF16:
		return "f16", nil
	case mll.DTypeF32:
		return "f32", nil
	case mll.DTypeQ4:
		return "q4", nil
	case mll.DTypeQ8:
		return "q8", nil
	default:
		return "", fmt.Errorf("unsupported MLL dtype %d", dtype)
	}
}

func halfToFloat32(bits uint16) float32 {
	sign := uint32(bits&0x8000) << 16
	exp := (bits >> 10) & 0x1f
	mant := uint32(bits & 0x03ff)
	switch exp {
	case 0:
		if mant == 0 {
			return math.Float32frombits(sign)
		}
		exp32 := uint32(127 - 15 + 1)
		for mant&0x0400 == 0 {
			mant <<= 1
			exp32--
		}
		mant &= 0x03ff
		return math.Float32frombits(sign | (exp32 << 23) | (mant << 13))
	case 0x1f:
		return math.Float32frombits(sign | 0x7f800000 | (mant << 13))
	default:
		exp32 := uint32(exp) + (127 - 15)
		return math.Float32frombits(sign | (exp32 << 23) | (mant << 13))
	}
}
