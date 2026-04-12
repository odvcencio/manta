package barruntime

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"sort"
	"strconv"

	"github.com/odvcencio/manta/artifact/barr"
	"github.com/odvcencio/manta/runtime/backend"
	mll "github.com/odvcencio/mll"
)

type mllExportState struct {
	strings *mll.StringTable
	types   *mll.TypeBuilder
	dims    map[string]bool
}

func DefaultMLLPath(barrPath string) string {
	return defaultManifestPath(barrPath, ".mll")
}

// ExportPackageToMLL exports a Manta artifact plus its sibling package
// files into a sealed MLL container. The resulting file keeps the current
// Manta module JSON in a schemaless XMTA section while populating the
// closest matching MLL core sections.
func ExportPackageToMLL(barrPath, outPath string) (string, error) {
	if barrPath == "" {
		return "", fmt.Errorf("artifact path is required")
	}
	if outPath == "" {
		outPath = DefaultMLLPath(barrPath)
	}

	mod, err := barr.ReadFile(barrPath)
	if err != nil {
		return "", err
	}

	packageKind, err := verifyPackageManifestForMLL(barrPath)
	if err != nil {
		return "", err
	}

	weights, err := loadWeightsForMLLExport(barrPath, mod)
	if err != nil {
		return "", err
	}

	plan, err := loadMemoryPlanForMLLExport(barrPath, mod, weights)
	if err != nil {
		return "", err
	}

	artifactJSON, err := barr.EncodeJSON(mod)
	if err != nil {
		return "", err
	}
	jsonFiles, err := loadOptionalMLLJSONFiles(barrPath)
	if err != nil {
		return "", err
	}

	file, err := buildMLLExport(mod, artifactJSON, jsonFiles, packageKind, weights, plan)
	if err != nil {
		return "", err
	}
	if err := os.WriteFile(outPath, file, 0o644); err != nil {
		return "", err
	}
	return outPath, nil
}

func buildMLLExport(mod *barr.Module, artifactJSON []byte, jsonFiles map[string]json.RawMessage, packageKind PackageKind, weights map[string]*backend.Tensor, plan *MemoryPlan) ([]byte, error) {
	if mod == nil {
		return nil, fmt.Errorf("nil module")
	}

	state := &mllExportState{
		strings: mll.NewStringTableBuilder(),
		types:   mll.NewTypeBuilder(),
		dims:    map[string]bool{},
	}
	// Reserve index 0 so Manta optional fields can safely use 0 as "absent".
	state.strings.Intern("")

	var (
		parmBuilder mll.ParmBuilder
		entrBuilder mll.EntrBuilder
		buffBuilder mll.BuffBuilder
		krnlBuilder mll.KrnlBuilder
		mempBuilder mll.MempBuilder
		tnsrBuilder mll.TnsrBuilder
	)

	paramRefs := make(map[string]mll.Ref, len(mod.Params))
	for i, param := range mod.Params {
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
		paramRefs[param.Name] = mll.Ref{Tag: mll.TagPARM, Index: uint32(i)}
	}

	entryRefs := make(map[string]mll.Ref, len(mod.EntryPoints))
	for i, entry := range mod.EntryPoints {
		inputs := make([]mll.ValueBinding, 0, len(entry.Inputs))
		for _, input := range entry.Inputs {
			typeRef, err := state.addValueTypeRef("entry:"+entry.Name+":input:"+input.Name, input.Type)
			if err != nil {
				return nil, fmt.Errorf("entry %q input %q type: %w", entry.Name, input.Name, err)
			}
			inputs = append(inputs, mll.ValueBinding{
				NameIdx: state.strings.Intern(input.Name),
				TypeRef: typeRef,
			})
		}
		outputs := make([]mll.ValueBinding, 0, len(entry.Outputs))
		for _, output := range entry.Outputs {
			typeRef, err := state.addValueTypeRef("entry:"+entry.Name+":output:"+output.Name, output.Type)
			if err != nil {
				return nil, fmt.Errorf("entry %q output %q type: %w", entry.Name, output.Name, err)
			}
			outputs = append(outputs, mll.ValueBinding{
				NameIdx: state.strings.Intern(output.Name),
				TypeRef: typeRef,
			})
		}
		entrBuilder.Add(mll.EntryPoint{
			NameIdx: state.strings.Intern(entry.Name),
			Kind:    entryPointKindToMLL(entry.Kind),
			Inputs:  inputs,
			Outputs: outputs,
		})
		entryRefs[entry.Name] = mll.Ref{Tag: mll.TagENTR, Index: uint32(i)}
	}

	for _, buffer := range mod.Buffers {
		typeRef, err := state.addTensorTypeRef("buffer:"+buffer.Name, buffer.DType, buffer.Shape)
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

	logicalTensorDTypes := map[string]string{}
	if len(weights) > 0 {
		names := make([]string, 0, len(weights))
		for name := range weights {
			names = append(names, name)
		}
		sort.Strings(names)
		for _, name := range names {
			tensor := weights[name]
			if tensor == nil {
				continue
			}
			storageDType, raw, widened, err := encodeTensorStorage(tensor)
			if err != nil {
				return nil, fmt.Errorf("encode tensor %q: %w", name, err)
			}
			if widened {
				logicalTensorDTypes[name] = tensor.DType
			}
			shape := make([]uint64, len(tensor.Shape))
			for i, dim := range tensor.Shape {
				if dim < 0 {
					return nil, fmt.Errorf("tensor %q has negative dim %d", name, dim)
				}
				shape[i] = uint64(dim)
			}
			tnsrBuilder.Add(mll.TensorEntry{
				NameIdx: state.strings.Intern(name),
				DType:   storageDType,
				Shape:   shape,
				Data:    raw,
			})
		}
	}

	if plan != nil {
		for _, item := range plan.Weights {
			paramRef, ok := paramRefs[item.Name]
			if !ok {
				return nil, fmt.Errorf("memory plan references unknown param %q", item.Name)
			}
			mempBuilder.Add(mll.MempEntry{
				ParamRef:    paramRef,
				Residency:   residencyToMLL(item.Residency),
				AccessCount: uint32(item.AccessCount),
			})
		}
	}

	var dimsBuilder mll.DimsBuilder
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
				Value:   0,
			})
		}
	}

	head := mll.HeadSection{
		Name:        state.strings.Intern(mod.Name),
		Description: state.strings.Intern("Manta sealed export"),
		Metadata:    buildMLLHeadMetadata(state.strings, mod, packageKind, weights),
	}

	xbar, err := buildMLLExportMetadata(mod, artifactJSON, jsonFiles, packageKind, logicalTensorDTypes)
	if err != nil {
		return nil, err
	}

	sections := make([]mll.SectionInput, 0, 10)
	if body, digestBody, err := encodeHeadSection(head, mll.ProfileSealed); err != nil {
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
		sections = append(sections, mll.SectionInput{
			Tag:           mll.TagSTRG,
			Body:          body,
			Flags:         mll.SectionFlagRequired,
			SchemaVersion: 1,
		})
	}
	if body, err := encodeSection(dimsBuilder.Write); err != nil {
		return nil, err
	} else {
		sections = append(sections, mll.SectionInput{
			Tag:           mll.TagDIMS,
			Body:          body,
			Flags:         mll.SectionFlagRequired,
			SchemaVersion: 1,
		})
	}
	if body, err := encodeSection(state.types.Write); err != nil {
		return nil, err
	} else {
		sections = append(sections, mll.SectionInput{
			Tag:           mll.TagTYPE,
			Body:          body,
			SchemaVersion: 1,
		})
	}
	if body, err := encodeSection(parmBuilder.Write); err != nil {
		return nil, err
	} else {
		sections = append(sections, mll.SectionInput{
			Tag:           mll.TagPARM,
			Body:          body,
			Flags:         mll.SectionFlagRequired,
			SchemaVersion: 1,
		})
	}
	if body, err := encodeSection(entrBuilder.Write); err != nil {
		return nil, err
	} else {
		sections = append(sections, mll.SectionInput{
			Tag:           mll.TagENTR,
			Body:          body,
			Flags:         mll.SectionFlagRequired,
			SchemaVersion: 1,
		})
	}
	if len(mod.Buffers) > 0 {
		if body, err := encodeSection(buffBuilder.Write); err != nil {
			return nil, err
		} else {
			sections = append(sections, mll.SectionInput{
				Tag:           mll.TagBUFF,
				Body:          body,
				SchemaVersion: 1,
			})
		}
	}
	if len(mod.Kernels) > 0 {
		if body, err := encodeSection(krnlBuilder.Write); err != nil {
			return nil, err
		} else {
			sections = append(sections, mll.SectionInput{
				Tag:           mll.TagKRNL,
				Body:          body,
				SchemaVersion: 1,
			})
		}
	}
	if plan != nil {
		if body, err := encodeSection(mempBuilder.Write); err != nil {
			return nil, err
		} else {
			sections = append(sections, mll.SectionInput{
				Tag:           mll.TagMEMP,
				Body:          body,
				SchemaVersion: 1,
			})
		}
	}
	if body, err := encodeSection(tnsrBuilder.Write); err != nil {
		return nil, err
	} else {
		sections = append(sections, mll.SectionInput{
			Tag:           mll.TagTNSR,
			Body:          body,
			Flags:         mll.SectionFlagRequired | mll.SectionFlagAligned,
			SchemaVersion: 1,
		})
	}
	if body, err := encodeSection((mll.SchmSection{}).Write); err != nil {
		return nil, err
	} else {
		sections = append(sections, mll.SectionInput{
			Tag:           mll.TagSCHM,
			Body:          body,
			SchemaVersion: 1,
		})
	}
	sections = append(sections, mll.SectionInput{
		Tag:           barr.MLLTagXMTA,
		Body:          xbar,
		Flags:         mll.SectionFlagSkippable | mll.SectionFlagSchemaless,
		SchemaVersion: 1,
	})

	return mll.WriteToBytes(mll.ProfileSealed, mll.V1_0, sections)
}

func verifyPackageManifestForMLL(barrPath string) (PackageKind, error) {
	manifestPath := ResolvePackageManifestPath(barrPath)
	if _, err := os.Stat(manifestPath); err != nil {
		if os.IsNotExist(err) {
			return "", nil
		}
		return "", err
	}
	manifest, err := ReadPackageManifestFile(manifestPath)
	if err != nil {
		return "", err
	}
	if err := manifest.VerifyFiles(map[string]string{
		"artifact":           barrPath,
		"embedding_manifest": ResolveEmbeddingManifestPath(barrPath),
		"score_manifest":     ResolveScoreManifestPath(barrPath),
		"retrieval_manifest": ResolveRetrievalManifestPath(barrPath),
		"tokenizer":          DefaultTokenizerPath(barrPath),
		"weights":            DefaultWeightFilePath(barrPath),
		"memory_plan":        DefaultMemoryPlanPath(barrPath),
		"train_manifest":     ResolveEmbeddingTrainManifestPath(barrPath),
		"checkpoint":         DefaultEmbeddingCheckpointPath(barrPath),
		"train_profile":      DefaultEmbeddingTrainProfilePath(barrPath),
	}); err != nil {
		return "", err
	}
	return manifest.Kind, nil
}

func loadWeightsForMLLExport(barrPath string, mod *barr.Module) (map[string]*backend.Tensor, error) {
	weightPath := DefaultWeightFilePath(barrPath)
	if _, err := os.Stat(weightPath); err != nil {
		if os.IsNotExist(err) {
			if mod != nil && len(mod.Params) > 0 {
				return nil, fmt.Errorf("weight file %q is required to export params", weightPath)
			}
			return nil, nil
		}
		return nil, err
	}
	weightFile, err := ReadWeightFile(weightPath)
	if err != nil {
		return nil, err
	}
	return weightFile.Weights, nil
}

func loadMemoryPlanForMLLExport(barrPath string, mod *barr.Module, weights map[string]*backend.Tensor) (*MemoryPlan, error) {
	if len(weights) == 0 {
		return nil, nil
	}
	planPath := DefaultMemoryPlanPath(barrPath)
	if _, err := os.Stat(planPath); err == nil {
		plan, err := ReadMemoryPlanFile(planPath)
		if err != nil {
			return nil, err
		}
		return &plan, nil
	} else if !os.IsNotExist(err) {
		return nil, err
	}
	plan := NewMemoryPlan(mod, weights, MemoryPlanOptions{})
	return &plan, nil
}

func loadOptionalMLLJSONFiles(barrPath string) (map[string]json.RawMessage, error) {
	out := map[string]json.RawMessage{}
	if data, ok, err := readOptionalPackageManifestJSON(ResolvePackageManifestPath(barrPath)); err != nil {
		return nil, fmt.Errorf("read package_manifest: %w", err)
	} else if ok {
		out["package_manifest"] = data
	}
	if data, ok, err := readOptionalTokenizerJSON(DefaultTokenizerPath(barrPath)); err != nil {
		return nil, fmt.Errorf("read tokenizer: %w", err)
	} else if ok {
		out["tokenizer"] = data
	}
	if data, ok, err := readOptionalEmbeddingManifestJSON(ResolveEmbeddingManifestPath(barrPath)); err != nil {
		return nil, fmt.Errorf("read embedding_manifest: %w", err)
	} else if ok {
		out["embedding_manifest"] = data
	}
	if data, ok, err := readOptionalScoreManifestJSON(ResolveScoreManifestPath(barrPath)); err != nil {
		return nil, fmt.Errorf("read score_manifest: %w", err)
	} else if ok {
		out["score_manifest"] = data
	}
	if data, ok, err := readOptionalRetrievalManifestJSON(ResolveRetrievalManifestPath(barrPath)); err != nil {
		return nil, fmt.Errorf("read retrieval_manifest: %w", err)
	} else if ok {
		out["retrieval_manifest"] = data
	}
	if data, ok, err := readOptionalTrainManifestJSON(ResolveEmbeddingTrainManifestPath(barrPath)); err != nil {
		return nil, fmt.Errorf("read train_manifest: %w", err)
	} else if ok {
		out["train_manifest"] = data
	}
	memoryPlanPath := DefaultMemoryPlanPath(barrPath)
	if data, ok, err := readOptionalMemoryPlanJSON(memoryPlanPath); err != nil {
		return nil, fmt.Errorf("read memory_plan: %w", err)
	} else if ok {
		out["memory_plan"] = data
	}
	trainProfilePath := DefaultEmbeddingTrainProfilePath(barrPath)
	if data, ok, err := readOptionalTrainProfileJSON(trainProfilePath); err != nil {
		return nil, fmt.Errorf("read train_profile: %w", err)
	} else if ok {
		out["train_profile"] = data
	}
	if len(out) == 0 {
		return nil, nil
	}
	return out, nil
}

func readOptionalPackageManifestJSON(path string) (json.RawMessage, bool, error) {
	if _, err := os.Stat(path); err != nil {
		if os.IsNotExist(err) {
			return nil, false, nil
		}
		return nil, false, err
	}
	manifest, err := ReadPackageManifestFile(path)
	if err != nil {
		return nil, false, err
	}
	body, err := json.Marshal(manifest)
	if err != nil {
		return nil, false, err
	}
	return json.RawMessage(body), true, nil
}

func readOptionalTrainProfileJSON(path string) (json.RawMessage, bool, error) {
	if _, err := os.Stat(path); err != nil {
		if os.IsNotExist(err) {
			return nil, false, nil
		}
		return nil, false, err
	}
	profile, err := ReadEmbeddingTrainProfileFile(path)
	if err != nil {
		return nil, false, err
	}
	body, err := json.Marshal(profile.normalized())
	if err != nil {
		return nil, false, err
	}
	return json.RawMessage(body), true, nil
}

func readOptionalEmbeddingManifestJSON(path string) (json.RawMessage, bool, error) {
	if _, err := os.Stat(path); err != nil {
		if os.IsNotExist(err) {
			return nil, false, nil
		}
		return nil, false, err
	}
	manifest, err := ReadEmbeddingManifestFile(path)
	if err != nil {
		return nil, false, err
	}
	body, err := json.Marshal(manifest)
	if err != nil {
		return nil, false, err
	}
	return json.RawMessage(body), true, nil
}

func readOptionalScoreManifestJSON(path string) (json.RawMessage, bool, error) {
	if _, err := os.Stat(path); err != nil {
		if os.IsNotExist(err) {
			return nil, false, nil
		}
		return nil, false, err
	}
	manifest, err := ReadScoreManifestFile(path)
	if err != nil {
		return nil, false, err
	}
	body, err := json.Marshal(manifest)
	if err != nil {
		return nil, false, err
	}
	return json.RawMessage(body), true, nil
}

func readOptionalRetrievalManifestJSON(path string) (json.RawMessage, bool, error) {
	if _, err := os.Stat(path); err != nil {
		if os.IsNotExist(err) {
			return nil, false, nil
		}
		return nil, false, err
	}
	manifest, err := ReadRetrievalManifestFile(path)
	if err != nil {
		return nil, false, err
	}
	body, err := json.Marshal(manifest)
	if err != nil {
		return nil, false, err
	}
	return json.RawMessage(body), true, nil
}

func readOptionalTrainManifestJSON(path string) (json.RawMessage, bool, error) {
	if _, err := os.Stat(path); err != nil {
		if os.IsNotExist(err) {
			return nil, false, nil
		}
		return nil, false, err
	}
	manifest, err := ReadEmbeddingTrainManifestFile(path)
	if err != nil {
		return nil, false, err
	}
	body, err := json.Marshal(manifest)
	if err != nil {
		return nil, false, err
	}
	return json.RawMessage(body), true, nil
}

func readOptionalTokenizerJSON(path string) (json.RawMessage, bool, error) {
	if _, err := os.Stat(path); err != nil {
		if os.IsNotExist(err) {
			return nil, false, nil
		}
		return nil, false, err
	}
	tokenizer, err := ReadTokenizerFile(path)
	if err != nil {
		return nil, false, err
	}
	body, err := json.Marshal(tokenizer)
	if err != nil {
		return nil, false, err
	}
	return json.RawMessage(body), true, nil
}

func readOptionalMemoryPlanJSON(path string) (json.RawMessage, bool, error) {
	if _, err := os.Stat(path); err != nil {
		if os.IsNotExist(err) {
			return nil, false, nil
		}
		return nil, false, err
	}
	plan, err := ReadMemoryPlanFile(path)
	if err != nil {
		return nil, false, err
	}
	body, err := json.Marshal(plan)
	if err != nil {
		return nil, false, err
	}
	return json.RawMessage(body), true, nil
}

func buildMLLExportMetadata(mod *barr.Module, artifactJSON []byte, jsonFiles map[string]json.RawMessage, packageKind PackageKind, logicalTensorDTypes map[string]string) ([]byte, error) {
	meta := barr.MLLMetadata{
		Version:       barr.MLLMetadataVersion,
		ModuleName:    mod.Name,
		ModuleVersion: mod.Version,
		Artifact:      json.RawMessage(bytes.TrimSpace(artifactJSON)),
	}
	if len(jsonFiles) > 0 {
		meta.JSONFiles = make(map[string]json.RawMessage, len(jsonFiles)+1)
		for role, body := range jsonFiles {
			meta.JSONFiles[role] = body
		}
	}
	if packageKind != "" {
		if meta.JSONFiles == nil {
			meta.JSONFiles = map[string]json.RawMessage{}
		}
		body, err := json.Marshal(map[string]string{"package_kind": string(packageKind)})
		if err != nil {
			return nil, err
		}
		meta.JSONFiles["manta_package"] = body
	}
	if len(logicalTensorDTypes) > 0 {
		meta.LogicalTensorDType = logicalTensorDTypes
	}
	return barr.EncodeMLLMetadata(meta)
}

func buildMLLHeadMetadata(strg *mll.StringTable, mod *barr.Module, packageKind PackageKind, weights map[string]*backend.Tensor) []mll.HeadMetadataEntry {
	items := []mll.HeadMetadataEntry{
		headStringMeta(strg, "artifact_version", mod.Version),
		headIntMeta(strg, "param_count", int64(len(mod.Params))),
		headIntMeta(strg, "entry_count", int64(len(mod.EntryPoints))),
		headIntMeta(strg, "kernel_count", int64(len(mod.Kernels))),
		headIntMeta(strg, "weight_count", int64(len(weights))),
	}
	if packageKind != "" {
		items = append(items, headStringMeta(strg, "package_kind", string(packageKind)))
	}
	return items
}

func headStringMeta(strg *mll.StringTable, key, value string) mll.HeadMetadataEntry {
	return mll.HeadMetadataEntry{
		Key:       strg.Intern(key),
		Kind:      mll.HeadValueString,
		StringIdx: strg.Intern(value),
	}
}

func headIntMeta(strg *mll.StringTable, key string, value int64) mll.HeadMetadataEntry {
	return mll.HeadMetadataEntry{
		Key:  strg.Intern(key),
		Kind: mll.HeadValueI64,
		I64:  value,
	}
}

func encodeHeadSection(head mll.HeadSection, profile mll.Profile) ([]byte, []byte, error) {
	var body bytes.Buffer
	if err := head.Write(&body); err != nil {
		return nil, nil, err
	}
	return body.Bytes(), head.DigestBody(profile), nil
}

func encodeSection(write func(io.Writer) error) ([]byte, error) {
	var body bytes.Buffer
	if err := write(&body); err != nil {
		return nil, err
	}
	return body.Bytes(), nil
}

func (s *mllExportState) internOptional(value string) uint32 {
	if value == "" {
		return 0
	}
	return s.strings.Intern(value)
}

func (s *mllExportState) addValueTypeRef(name string, typ barr.ValueType) (mll.Ref, error) {
	idx := uint32(len(s.types.Decls()))
	nameIdx := s.strings.Intern(name)
	switch typ.Kind {
	case barr.ValueTensor:
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
	case barr.ValueKVCache:
		s.types.AddKVCacheType(nameIdx, 0, 0, 0)
	case barr.ValueCandidatePack:
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

func (s *mllExportState) addTensorTypeRef(name, dtype string, shape []string) (mll.Ref, error) {
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

func (s *mllExportState) shape(shape []string) ([]mll.Dimension, error) {
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

func entryPointKindToMLL(kind barr.EntryPointKind) uint8 {
	switch kind {
	case barr.EntryPointKernel:
		return mll.EntryKindKernel
	case barr.EntryPointPipeline:
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

func residencyToMLL(residency MemoryResidency) uint8 {
	switch residency {
	case ResidencyHostPinned:
		return mll.ResidencyHostPinned
	case ResidencyHostShared:
		return mll.ResidencyHostShared
	case ResidencyLazyStaged:
		return mll.ResidencyLazyStaged
	default:
		return mll.ResidencyDeviceResident
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

func encodeTensorStorage(t *backend.Tensor) (mll.DType, []byte, bool, error) {
	if t == nil {
		return mll.DTypeInvalid, nil, false, fmt.Errorf("nil tensor")
	}
	switch t.DType {
	case "i32":
		data := make([]byte, 4*len(t.I32))
		for i, value := range t.I32 {
			binary.LittleEndian.PutUint32(data[i*4:], uint32(value))
		}
		return mll.DTypeI32, data, false, nil
	case "i64":
		data := make([]byte, 8*len(t.I64))
		for i, value := range t.I64 {
			binary.LittleEndian.PutUint64(data[i*8:], uint64(value))
		}
		return mll.DTypeI64, data, false, nil
	case "f16":
		data := make([]byte, 2*len(t.F32))
		for i, value := range t.F32 {
			binary.LittleEndian.PutUint16(data[i*2:], float32ToHalf(value))
		}
		return mll.DTypeF16, data, false, nil
	case "f32":
		return mll.DTypeF32, encodeFloat32Bytes(t.F32), false, nil
	case "q4", "q8":
		// Manta currently stores fake-quantized q4/q8 tensors as float32
		// values, so the first MLL export keeps the raw bytes honest and records
		// the logical dtype in XMTA metadata.
		return mll.DTypeF32, encodeFloat32Bytes(t.F32), true, nil
	default:
		return mll.DTypeInvalid, nil, false, fmt.Errorf("unsupported tensor dtype %q", t.DType)
	}
}

func encodeFloat32Bytes(values []float32) []byte {
	data := make([]byte, 4*len(values))
	for i, value := range values {
		binary.LittleEndian.PutUint32(data[i*4:], math.Float32bits(value))
	}
	return data
}

func float32ToHalf(value float32) uint16 {
	bits := math.Float32bits(value)
	sign := uint16((bits >> 16) & 0x8000)
	exp := int((bits>>23)&0xff) - 127 + 15
	mant := bits & 0x7fffff

	switch {
	case exp <= 0:
		if exp < -10 {
			return sign
		}
		mant |= 0x800000
		shift := uint32(14 - exp)
		half := uint16(mant >> shift)
		if (mant>>(shift-1))&1 != 0 {
			half++
		}
		return sign | half
	case exp >= 0x1f:
		if mant == 0 {
			return sign | 0x7c00
		}
		return sign | 0x7c00 | uint16(mant>>13)
	default:
		halfExp := uint16(exp) << 10
		halfMant := uint16(mant >> 13)
		if mant&0x00001000 != 0 {
			halfMant++
			if halfMant&0x0400 != 0 {
				halfMant = 0
				halfExp += 0x0400
				if halfExp >= 0x7c00 {
					halfExp = 0x7c00
				}
			}
		}
		return sign | halfExp | halfMant
	}
}
