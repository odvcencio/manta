package barr

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"sort"
	"strconv"
	"strings"

	mll "github.com/odvcencio/mll"
)

const MLLMetadataVersion = "manta/mll/v0alpha1"

var MLLTagXMTA = [4]byte{'X', 'M', 'T', 'A'}

// MLLMetadata preserves the Manta module inside an MLL container while
// the native MLL section model catches up with Manta-specific semantics.
type MLLMetadata struct {
	Version            string                     `json:"version"`
	ModuleName         string                     `json:"module_name"`
	ModuleVersion      string                     `json:"module_version"`
	Artifact           json.RawMessage            `json:"artifact"`
	JSONFiles          map[string]json.RawMessage `json:"json_files,omitempty"`
	LogicalTensorDType map[string]string          `json:"logical_tensor_dtypes,omitempty"`
}

type mllState struct {
	strings *mll.StringTable
	types   *mll.TypeBuilder
	dims    map[string]bool
}

func EncodeMLLMetadata(meta MLLMetadata) ([]byte, error) {
	if err := meta.Validate(); err != nil {
		return nil, err
	}
	return json.Marshal(meta)
}

func DecodeMLLMetadata(data []byte) (MLLMetadata, error) {
	var meta MLLMetadata
	if err := json.Unmarshal(data, &meta); err != nil {
		return MLLMetadata{}, err
	}
	if err := meta.Validate(); err != nil {
		return MLLMetadata{}, err
	}
	return meta, nil
}

func (m MLLMetadata) Validate() error {
	if m.Version == "" {
		return fmt.Errorf("MLL metadata version is required")
	}
	if m.Version != MLLMetadataVersion {
		return fmt.Errorf("MLL metadata version %q is not supported, want %q", m.Version, MLLMetadataVersion)
	}
	if m.ModuleName == "" {
		return fmt.Errorf("MLL metadata module_name is required")
	}
	if m.ModuleVersion == "" {
		return fmt.Errorf("MLL metadata module_version is required")
	}
	if len(m.Artifact) == 0 || !json.Valid(m.Artifact) {
		return fmt.Errorf("MLL metadata artifact must be valid JSON")
	}
	for role, body := range m.JSONFiles {
		if role == "" {
			return fmt.Errorf("MLL metadata json_files role is required")
		}
		if len(body) == 0 || !json.Valid(body) {
			return fmt.Errorf("MLL metadata json_files[%q] must be valid JSON", role)
		}
	}
	for name, dtype := range m.LogicalTensorDType {
		if name == "" {
			return fmt.Errorf("MLL metadata logical tensor name is required")
		}
		if dtype == "" {
			return fmt.Errorf("MLL metadata logical tensor dtype for %q is required", name)
		}
	}
	return nil
}

func IsMLLBytes(data []byte) bool {
	return len(data) >= len(mll.Magic) && bytes.Equal(data[:len(mll.Magic)], mll.Magic[:])
}

func EncodeMLL(mod *Module) ([]byte, error) {
	if mod == nil {
		return nil, fmt.Errorf("nil module")
	}
	if err := mod.Validate(); err != nil {
		return nil, err
	}

	artifactJSON, err := EncodeJSON(mod)
	if err != nil {
		return nil, err
	}
	metaBytes, err := EncodeMLLMetadata(MLLMetadata{
		Version:       MLLMetadataVersion,
		ModuleName:    mod.Name,
		ModuleVersion: mod.Version,
		Artifact:      json.RawMessage(bytes.TrimSpace(artifactJSON)),
	})
	if err != nil {
		return nil, err
	}

	state := &mllState{
		strings: mll.NewStringTableBuilder(),
		types:   mll.NewTypeBuilder(),
		dims:    map[string]bool{},
	}
	state.strings.Intern("")

	var (
		parmBuilder mll.ParmBuilder
		entrBuilder mll.EntrBuilder
		buffBuilder mll.BuffBuilder
		krnlBuilder mll.KrnlBuilder
		dimsBuilder mll.DimsBuilder
		tnsrBuilder mll.TnsrBuilder
	)

	for _, param := range mod.Params {
		typeRef, err := state.addValueTypeRef("param:"+param.Name, param.Type)
		if err != nil {
			return nil, fmt.Errorf("param %q type: %w", param.Name, err)
		}
		parmBuilder.Add(mll.ParmDecl{
			NameIdx:    state.strings.Intern(param.Name),
			TypeRef:    typeRef,
			BindingIdx: state.internOptional(param.Binding),
			Trainable:  param.Trainable,
		})
	}

	for _, entry := range mod.EntryPoints {
		inputs := make([]mll.ValueBinding, 0, len(entry.Inputs))
		for _, input := range entry.Inputs {
			typeRef, err := state.addValueTypeRef("entry:"+entry.Name+":input:"+input.Name, input.Type)
			if err != nil {
				return nil, fmt.Errorf("entry %q input %q type: %w", entry.Name, input.Name, err)
			}
			inputs = append(inputs, mll.ValueBinding{NameIdx: state.strings.Intern(input.Name), TypeRef: typeRef})
		}
		outputs := make([]mll.ValueBinding, 0, len(entry.Outputs))
		for _, output := range entry.Outputs {
			typeRef, err := state.addValueTypeRef("entry:"+entry.Name+":output:"+output.Name, output.Type)
			if err != nil {
				return nil, fmt.Errorf("entry %q output %q type: %w", entry.Name, output.Name, err)
			}
			outputs = append(outputs, mll.ValueBinding{NameIdx: state.strings.Intern(output.Name), TypeRef: typeRef})
		}
		entrBuilder.Add(mll.EntryPoint{
			NameIdx: state.strings.Intern(entry.Name),
			Kind:    entryPointKindToMLL(entry.Kind),
			Inputs:  inputs,
			Outputs: outputs,
		})
	}

	for _, buffer := range mod.Buffers {
		typeRef, err := state.addBufferTypeRef("buffer:"+buffer.Name, buffer.DType, buffer.Shape)
		if err != nil {
			return nil, fmt.Errorf("buffer %q type: %w", buffer.Name, err)
		}
		buffBuilder.Add(mll.BuffDecl{
			NameIdx:      state.strings.Intern(buffer.Name),
			TypeRef:      typeRef,
			StorageClass: bufferStorageClassToMLL(buffer.StorageClass),
		})
	}

	for _, kernel := range mod.Kernels {
		body, err := json.Marshal(kernel)
		if err != nil {
			return nil, fmt.Errorf("marshal kernel %q: %w", kernel.Name, err)
		}
		krnlBuilder.Add(mll.KernelDecl{
			NameIdx: state.strings.Intern(kernel.Name),
			Body:    body,
		})
	}

	if len(state.dims) > 0 {
		names := make([]string, 0, len(state.dims))
		for name := range state.dims {
			names = append(names, name)
		}
		sort.Strings(names)
		for _, name := range names {
			dimsBuilder.Add(mll.DimDecl{
				NameIdx: state.strings.Intern(name),
				Bound:   mll.DimBoundDynamic,
			})
		}
	}

	head := mll.HeadSection{
		Name:        state.strings.Intern(mod.Name),
		Description: state.strings.Intern("Manta module artifact"),
		Metadata: []mll.HeadMetadataEntry{
			headStringMeta(state.strings, "artifact_version", mod.Version),
			headStringMeta(state.strings, "supported_backends", joinBackends(mod.Requirements.SupportedBackends)),
			headStringMeta(state.strings, "capabilities", strings.Join(mod.Requirements.Capabilities, ",")),
			headIntMeta(state.strings, "param_count", int64(len(mod.Params))),
			headIntMeta(state.strings, "entry_count", int64(len(mod.EntryPoints))),
			headIntMeta(state.strings, "kernel_count", int64(len(mod.Kernels))),
		},
	}

	sections := make([]mll.SectionInput, 0, 9)
	if body, digestBody, err := encodeHeadSection(head); err != nil {
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
	if body, err := encodeSection(state.strings.Write); err != nil {
		return nil, err
	} else {
		sections = append(sections, mll.SectionInput{Tag: mll.TagSTRG, Body: body, Flags: mll.SectionFlagRequired, SchemaVersion: 1})
	}
	if body, err := encodeSection(dimsBuilder.Write); err != nil {
		return nil, err
	} else {
		sections = append(sections, mll.SectionInput{Tag: mll.TagDIMS, Body: body, Flags: mll.SectionFlagRequired, SchemaVersion: 1})
	}
	if body, err := encodeSection(state.types.Write); err != nil {
		return nil, err
	} else {
		sections = append(sections, mll.SectionInput{Tag: mll.TagTYPE, Body: body, SchemaVersion: 1})
	}
	if body, err := encodeSection(parmBuilder.Write); err != nil {
		return nil, err
	} else {
		sections = append(sections, mll.SectionInput{Tag: mll.TagPARM, Body: body, Flags: mll.SectionFlagRequired, SchemaVersion: 1})
	}
	if body, err := encodeSection(entrBuilder.Write); err != nil {
		return nil, err
	} else {
		sections = append(sections, mll.SectionInput{Tag: mll.TagENTR, Body: body, Flags: mll.SectionFlagRequired, SchemaVersion: 1})
	}
	if len(mod.Buffers) > 0 {
		if body, err := encodeSection(buffBuilder.Write); err != nil {
			return nil, err
		} else {
			sections = append(sections, mll.SectionInput{Tag: mll.TagBUFF, Body: body, SchemaVersion: 1})
		}
	}
	if len(mod.Kernels) > 0 {
		if body, err := encodeSection(krnlBuilder.Write); err != nil {
			return nil, err
		} else {
			sections = append(sections, mll.SectionInput{Tag: mll.TagKRNL, Body: body, SchemaVersion: 1})
		}
	}
	if body, err := encodeSection(tnsrBuilder.Write); err != nil {
		return nil, err
	} else {
		sections = append(sections, mll.SectionInput{Tag: mll.TagTNSR, Body: body, Flags: mll.SectionFlagRequired | mll.SectionFlagAligned, SchemaVersion: 1})
	}
	if body, err := encodeSection((mll.SchmSection{}).Write); err != nil {
		return nil, err
	} else {
		sections = append(sections, mll.SectionInput{Tag: mll.TagSCHM, Body: body, SchemaVersion: 1})
	}
	sections = append(sections, mll.SectionInput{
		Tag:           MLLTagXMTA,
		Body:          metaBytes,
		Flags:         mll.SectionFlagSkippable | mll.SectionFlagSchemaless,
		SchemaVersion: 1,
	})

	return mll.WriteToBytes(mll.ProfileSealed, mll.V1_0, sections)
}

func DecodeMLL(data []byte) (*Module, error) {
	reader, err := mll.ReadBytes(data, mll.WithDigestVerification())
	if err != nil {
		return nil, err
	}
	body, ok := reader.Section(MLLTagXMTA)
	if !ok {
		return nil, fmt.Errorf("mll artifact missing XMTA metadata section")
	}
	meta, err := DecodeMLLMetadata(body)
	if err != nil {
		return nil, err
	}
	return DecodeJSON(meta.Artifact)
}

func ReadFile(path string) (*Module, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if !IsMLLBytes(data) {
		return nil, fmt.Errorf("artifact %q is not an MLL file", path)
	}
	return DecodeMLL(data)
}

func WriteFile(path string, m *Module) error {
	data, err := EncodeMLL(m)
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}

func encodeHeadSection(head mll.HeadSection) ([]byte, []byte, error) {
	var body bytes.Buffer
	if err := head.Write(&body); err != nil {
		return nil, nil, err
	}
	return body.Bytes(), head.DigestBody(mll.ProfileSealed), nil
}

func encodeSection(write func(io.Writer) error) ([]byte, error) {
	var body bytes.Buffer
	if err := write(&body); err != nil {
		return nil, err
	}
	return body.Bytes(), nil
}

func headStringMeta(strg *mll.StringTable, key, value string) mll.HeadMetadataEntry {
	return mll.HeadMetadataEntry{Key: strg.Intern(key), Kind: mll.HeadValueString, StringIdx: strg.Intern(value)}
}

func headIntMeta(strg *mll.StringTable, key string, value int64) mll.HeadMetadataEntry {
	return mll.HeadMetadataEntry{Key: strg.Intern(key), Kind: mll.HeadValueI64, I64: value}
}

func joinBackends(backends []BackendKind) string {
	if len(backends) == 0 {
		return ""
	}
	parts := make([]string, len(backends))
	for i, backend := range backends {
		parts[i] = string(backend)
	}
	return strings.Join(parts, ",")
}

func (s *mllState) internOptional(value string) uint32 {
	if value == "" {
		return 0
	}
	return s.strings.Intern(value)
}

func (s *mllState) addValueTypeRef(name string, typ ValueType) (mll.Ref, error) {
	idx := uint32(len(s.types.Decls()))
	nameIdx := s.strings.Intern(name)
	switch typ.Kind {
	case ValueTensor:
		if typ.Tensor == nil {
			return mll.Ref{}, fmt.Errorf("tensor payload is required")
		}
		dtype, err := barrDTypeToMLL(typ.Tensor.DType)
		if err != nil {
			return mll.Ref{}, err
		}
		shape, err := s.shape(typ.Tensor.Shape)
		if err != nil {
			return mll.Ref{}, err
		}
		s.types.AddTensorType(nameIdx, dtype, shape)
	case ValueKVCache:
		s.types.AddKVCacheType(nameIdx, 0, 0, 0)
	case ValueCandidatePack:
		rank := uint32(0)
		if typ.CandidatePack != nil {
			rank = uint32(len(typ.CandidatePack.Shape))
		}
		s.types.AddCandidatePackType(nameIdx, rank)
	default:
		return mll.Ref{}, fmt.Errorf("unsupported value kind %q", typ.Kind)
	}
	return mll.Ref{Tag: mll.TagTYPE, Index: idx}, nil
}

func (s *mllState) addTensorTypeRef(name, dtype string, shape []string) (mll.Ref, error) {
	idx := uint32(len(s.types.Decls()))
	nameIdx := s.strings.Intern(name)
	mllDType, err := barrDTypeToMLL(dtype)
	if err != nil {
		return mll.Ref{}, err
	}
	mllShape, err := s.shape(shape)
	if err != nil {
		return mll.Ref{}, err
	}
	s.types.AddTensorType(nameIdx, mllDType, mllShape)
	return mll.Ref{Tag: mll.TagTYPE, Index: idx}, nil
}

func (s *mllState) addBufferTypeRef(name, dtype string, shape []string) (mll.Ref, error) {
	idx := uint32(len(s.types.Decls()))
	nameIdx := s.strings.Intern(name)
	switch dtype {
	case "candidate_pack":
		s.types.AddCandidatePackType(nameIdx, uint32(len(shape)))
	case "kv_cache":
		s.types.AddKVCacheType(nameIdx, 0, 0, 0)
	default:
		mllDType, err := barrDTypeToMLL(dtype)
		if err != nil {
			return mll.Ref{}, err
		}
		mllShape, err := s.shape(shape)
		if err != nil {
			return mll.Ref{}, err
		}
		s.types.AddTensorType(nameIdx, mllDType, mllShape)
	}
	return mll.Ref{Tag: mll.TagTYPE, Index: idx}, nil
}

func (s *mllState) shape(shape []string) ([]mll.Dimension, error) {
	out := make([]mll.Dimension, 0, len(shape))
	for _, part := range shape {
		if part == "" {
			return nil, fmt.Errorf("empty shape dimension")
		}
		if value, err := strconv.ParseInt(part, 10, 64); err == nil {
			out = append(out, mll.DimLiteral(value))
			continue
		}
		s.dims[part] = true
		out = append(out, mll.Dimension{
			Kind:      mll.DimKindSymbol,
			Symbol:    part,
			SymbolIdx: s.strings.Intern(part),
		})
	}
	return out, nil
}

func entryPointKindToMLL(kind EntryPointKind) uint8 {
	switch kind {
	case EntryPointKernel:
		return mll.EntryKindKernel
	case EntryPointPipeline:
		return mll.EntryKindPipeline
	default:
		return mll.EntryKindFunction
	}
}

func bufferStorageClassToMLL(storage string) uint8 {
	switch storage {
	case "host_visible", "unified":
		return mll.StorageClassIO
	case "workgroup_local":
		return mll.StorageClassWorkspace
	default:
		return mll.StorageClassActivation
	}
}

func barrDTypeToMLL(dtype string) (mll.DType, error) {
	switch dtype {
	case "i32":
		return mll.DTypeI32, nil
	case "i64":
		return mll.DTypeI64, nil
	case "f16":
		return mll.DTypeF16, nil
	case "f32":
		return mll.DTypeF32, nil
	case "q4":
		return mll.DTypeQ4, nil
	case "q8":
		return mll.DTypeQ8, nil
	default:
		return mll.DTypeInvalid, fmt.Errorf("unsupported Manta dtype %q", dtype)
	}
}
