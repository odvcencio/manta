package barruntime

import (
	"fmt"
	"os"
	"sort"

	"github.com/odvcencio/barracuda/artifact/barr"
	mll "github.com/odvcencio/mll"
)

const (
	manifestKindKey    = "manifest_kind"
	manifestVersionKey = "manifest_version"
)

type authoredManifestValue struct {
	kind    mll.HeadValueKind
	boolV   bool
	i64V    int64
	f64V    float64
	stringV string
}

type authoredManifestDoc struct {
	name        string
	description string
	values      map[string]authoredManifestValue
}

func authoredString(value string) authoredManifestValue {
	return authoredManifestValue{kind: mll.HeadValueString, stringV: value}
}

func authoredBool(value bool) authoredManifestValue {
	return authoredManifestValue{kind: mll.HeadValueBool, boolV: value}
}

func authoredInt(value int64) authoredManifestValue {
	return authoredManifestValue{kind: mll.HeadValueI64, i64V: value}
}

func authoredFloat(value float64) authoredManifestValue {
	return authoredManifestValue{kind: mll.HeadValueF64, f64V: value}
}

func writeAuthoredManifestMLL(path, kind, version, name, description string, values map[string]authoredManifestValue) error {
	data, err := encodeAuthoredManifestMLL(kind, version, name, description, values)
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}

func encodeAuthoredManifestMLL(kind, version, name, description string, values map[string]authoredManifestValue) ([]byte, error) {
	if kind == "" {
		return nil, fmt.Errorf("manifest kind is required")
	}
	if version == "" {
		return nil, fmt.Errorf("manifest version is required")
	}
	strg := mll.NewStringTableBuilder()
	strg.Intern("")
	if name == "" {
		name = kind
	}
	if description == "" {
		description = "Barracuda authored manifest"
	}

	keys := make([]string, 0, len(values)+2)
	keys = append(keys, manifestKindKey, manifestVersionKey)
	for key := range values {
		keys = append(keys, key)
	}
	sort.Strings(keys)

	metadata := make([]mll.HeadMetadataEntry, 0, len(keys))
	for _, key := range keys {
		value := authoredManifestValue{}
		switch key {
		case manifestKindKey:
			value = authoredString(kind)
		case manifestVersionKey:
			value = authoredString(version)
		default:
			value = values[key]
		}
		entry := mll.HeadMetadataEntry{
			Key:  strg.Intern(key),
			Kind: value.kind,
		}
		switch value.kind {
		case mll.HeadValueBool:
			entry.Bool = value.boolV
		case mll.HeadValueI64:
			entry.I64 = value.i64V
		case mll.HeadValueF64:
			entry.F64 = value.f64V
		case mll.HeadValueString:
			entry.StringIdx = strg.Intern(value.stringV)
		default:
			return nil, fmt.Errorf("manifest key %q has unsupported metadata kind %d", key, value.kind)
		}
		metadata = append(metadata, entry)
	}

	head := mll.HeadSection{
		Name:        strg.Intern(name),
		Description: strg.Intern(description),
		Metadata:    metadata,
	}

	var (
		dimsBuilder mll.DimsBuilder
		typeBuilder mll.TypeBuilder
		parmBuilder mll.ParmBuilder
		entrBuilder mll.EntrBuilder
		tnsrBuilder mll.TnsrBuilder
	)

	sections := make([]mll.SectionInput, 0, 8)
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
	if body, err := encodeSection(strg.Write); err != nil {
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
	if body, err := encodeSection(typeBuilder.Write); err != nil {
		return nil, err
	} else {
		sections = append(sections, mll.SectionInput{Tag: mll.TagTYPE, Body: body, SchemaVersion: 1})
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
		sections = append(sections, mll.SectionInput{Tag: mll.TagSCHM, Body: body, SchemaVersion: 1})
	}
	return mll.WriteToBytes(mll.ProfileSealed, mll.V1_0, sections)
}

func readAuthoredManifestMLL(path, expectedKind, expectedVersion string) (authoredManifestDoc, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return authoredManifestDoc{}, err
	}
	if !barr.IsMLLBytes(data) {
		return authoredManifestDoc{}, fmt.Errorf("not an MLL manifest")
	}
	reader, err := mll.ReadBytes(data, mll.WithDigestVerification())
	if err != nil {
		return authoredManifestDoc{}, err
	}
	if reader.Profile() != mll.ProfileSealed {
		return authoredManifestDoc{}, fmt.Errorf("manifest profile = %d, want %d", reader.Profile(), mll.ProfileSealed)
	}
	headBody, ok := reader.Section(mll.TagHEAD)
	if !ok {
		return authoredManifestDoc{}, fmt.Errorf("manifest missing HEAD section")
	}
	strgBody, ok := reader.Section(mll.TagSTRG)
	if !ok {
		return authoredManifestDoc{}, fmt.Errorf("manifest missing STRG section")
	}
	head, err := mll.ReadHeadSection(headBody)
	if err != nil {
		return authoredManifestDoc{}, err
	}
	strg, err := mll.ReadStringTable(strgBody)
	if err != nil {
		return authoredManifestDoc{}, err
	}
	doc := authoredManifestDoc{
		name:        strg.At(head.Name),
		description: strg.At(head.Description),
		values:      make(map[string]authoredManifestValue, len(head.Metadata)),
	}
	for _, entry := range head.Metadata {
		key := strg.At(entry.Key)
		if key == "" {
			continue
		}
		value := authoredManifestValue{kind: entry.Kind}
		switch entry.Kind {
		case mll.HeadValueBool:
			value.boolV = entry.Bool
		case mll.HeadValueI64:
			value.i64V = entry.I64
		case mll.HeadValueF64:
			value.f64V = entry.F64
		case mll.HeadValueString:
			value.stringV = strg.At(entry.StringIdx)
		}
		doc.values[key] = value
	}
	if expectedKind != "" {
		kind, ok, err := doc.string(manifestKindKey)
		if err != nil {
			return authoredManifestDoc{}, err
		}
		if !ok {
			return authoredManifestDoc{}, fmt.Errorf("manifest missing %q", manifestKindKey)
		}
		if kind != expectedKind {
			return authoredManifestDoc{}, fmt.Errorf("manifest kind %q, want %q", kind, expectedKind)
		}
	}
	if expectedVersion != "" {
		version, ok, err := doc.string(manifestVersionKey)
		if err != nil {
			return authoredManifestDoc{}, err
		}
		if !ok {
			return authoredManifestDoc{}, fmt.Errorf("manifest missing %q", manifestVersionKey)
		}
		if version != expectedVersion {
			return authoredManifestDoc{}, fmt.Errorf("manifest version %q, want %q", version, expectedVersion)
		}
	}
	return doc, nil
}

func (d authoredManifestDoc) string(key string) (string, bool, error) {
	value, ok := d.values[key]
	if !ok {
		return "", false, nil
	}
	if value.kind != mll.HeadValueString {
		return "", false, fmt.Errorf("manifest key %q kind = %d, want string", key, value.kind)
	}
	return value.stringV, true, nil
}

func (d authoredManifestDoc) bool(key string) (bool, bool, error) {
	value, ok := d.values[key]
	if !ok {
		return false, false, nil
	}
	if value.kind != mll.HeadValueBool {
		return false, false, fmt.Errorf("manifest key %q kind = %d, want bool", key, value.kind)
	}
	return value.boolV, true, nil
}

func (d authoredManifestDoc) int(key string) (int64, bool, error) {
	value, ok := d.values[key]
	if !ok {
		return 0, false, nil
	}
	if value.kind != mll.HeadValueI64 {
		return 0, false, fmt.Errorf("manifest key %q kind = %d, want int", key, value.kind)
	}
	return value.i64V, true, nil
}

func (d authoredManifestDoc) float(key string) (float64, bool, error) {
	value, ok := d.values[key]
	if !ok {
		return 0, false, nil
	}
	if value.kind != mll.HeadValueF64 {
		return 0, false, fmt.Errorf("manifest key %q kind = %d, want float", key, value.kind)
	}
	return value.f64V, true, nil
}

func resolveSiblingPath(artifactPath string, preferredSuffix string, legacySuffixes ...string) string {
	candidates := make([]string, 0, 1+len(legacySuffixes))
	candidates = append(candidates, preferredSuffix)
	candidates = append(candidates, legacySuffixes...)
	for _, suffix := range candidates {
		path := defaultManifestPath(artifactPath, suffix)
		if _, err := os.Stat(path); err == nil {
			return path
		}
	}
	return defaultManifestPath(artifactPath, preferredSuffix)
}
