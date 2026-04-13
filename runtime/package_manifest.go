package mantaruntime

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"

	mantaartifact "github.com/odvcencio/manta/artifact/manta"
	mll "github.com/odvcencio/mll"
)

const PackageManifestVersion = "manta/package/v0alpha1"

var tagXPKG = [4]byte{'X', 'P', 'K', 'G'}

type PackageKind string

const (
	PackageEmbedding PackageKind = "embedding"
	PackageTraining  PackageKind = "training"
)

type PackageManifestFile struct {
	Role   string `json:"role"`
	Path   string `json:"path"`
	SHA256 string `json:"sha256"`
	Bytes  int64  `json:"bytes"`
}

type PackageManifest struct {
	Version         string                `json:"version"`
	Kind            PackageKind           `json:"kind"`
	ModuleName      string                `json:"module_name"`
	ArtifactVersion string                `json:"artifact_version"`
	Files           []PackageManifestFile `json:"files"`
}

func DefaultPackageManifestPath(artifactPath string) string {
	return defaultManifestPath(artifactPath, ".package.mll")
}

func ResolvePackageManifestPath(artifactPath string) string {
	return DefaultPackageManifestPath(artifactPath)
}

func (m PackageManifest) Validate() error {
	if m.Version == "" {
		return fmt.Errorf("package manifest version is required")
	}
	if m.Version != PackageManifestVersion {
		return fmt.Errorf("package manifest version %q is not supported, want %q", m.Version, PackageManifestVersion)
	}
	if m.Kind == "" {
		return fmt.Errorf("package manifest kind is required")
	}
	if m.ModuleName == "" {
		return fmt.Errorf("package manifest module_name is required")
	}
	if m.ArtifactVersion == "" {
		return fmt.Errorf("package manifest artifact_version is required")
	}
	if len(m.Files) == 0 {
		return fmt.Errorf("package manifest must list at least one file")
	}
	seen := map[string]bool{}
	for _, item := range m.Files {
		if item.Role == "" {
			return fmt.Errorf("package manifest file role is required")
		}
		if seen[item.Role] {
			return fmt.Errorf("duplicate package manifest file role %q", item.Role)
		}
		seen[item.Role] = true
		if item.Path == "" {
			return fmt.Errorf("package manifest file %q path is required", item.Role)
		}
		if item.SHA256 == "" {
			return fmt.Errorf("package manifest file %q sha256 is required", item.Role)
		}
		if len(item.SHA256) != sha256.Size*2 {
			return fmt.Errorf("package manifest file %q sha256 must be %d hex chars", item.Role, sha256.Size*2)
		}
		if _, err := hex.DecodeString(item.SHA256); err != nil {
			return fmt.Errorf("package manifest file %q sha256 is invalid: %w", item.Role, err)
		}
		if item.Bytes < 0 {
			return fmt.Errorf("package manifest file %q size must be non-negative", item.Role)
		}
	}
	return nil
}

func (m PackageManifest) WriteFile(path string) error {
	if err := m.Validate(); err != nil {
		return err
	}
	data, err := encodePackageManifestMLL(m)
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}

func ReadPackageManifestFile(path string) (PackageManifest, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return PackageManifest{}, err
	}
	if !mantaartifact.IsMLLBytes(data) {
		return PackageManifest{}, fmt.Errorf("package manifest %q is not an MLL file", path)
	}
	return decodePackageManifestMLL(data)
}

func encodePackageManifestMLL(manifest PackageManifest) ([]byte, error) {
	strg := mll.NewStringTableBuilder()
	strg.Intern("")
	for _, item := range manifest.Files {
		strg.Intern(item.Role)
		strg.Intern(item.Path)
	}

	head := mll.HeadSection{
		Name:        strg.Intern("manta-package-manifest"),
		Description: strg.Intern("Manta package manifest"),
		Metadata: []mll.HeadMetadataEntry{
			headStringMeta(strg, manifestVersionKey, manifest.Version),
			headStringMeta(strg, "package_kind", string(manifest.Kind)),
			headStringMeta(strg, "module_name", manifest.ModuleName),
			headStringMeta(strg, "artifact_version", manifest.ArtifactVersion),
			headIntMeta(strg, "file_count", int64(len(manifest.Files))),
		},
	}

	var xbody bytes.Buffer
	writeU32 := func(v uint32) error {
		return binary.Write(&xbody, binary.LittleEndian, v)
	}
	writeI64 := func(v int64) error {
		return binary.Write(&xbody, binary.LittleEndian, v)
	}
	if err := writeU32(uint32(len(manifest.Files))); err != nil {
		return nil, err
	}
	for _, item := range manifest.Files {
		sum, err := hex.DecodeString(item.SHA256)
		if err != nil {
			return nil, fmt.Errorf("decode sha256 for %q: %w", item.Role, err)
		}
		if len(sum) != sha256.Size {
			return nil, fmt.Errorf("package manifest file %q sha256 bytes = %d, want %d", item.Role, len(sum), sha256.Size)
		}
		if err := writeU32(strg.Intern(item.Role)); err != nil {
			return nil, err
		}
		if err := writeU32(strg.Intern(item.Path)); err != nil {
			return nil, err
		}
		if err := writeI64(item.Bytes); err != nil {
			return nil, err
		}
		if _, err := xbody.Write(sum); err != nil {
			return nil, err
		}
	}

	var (
		dimsBuilder mll.DimsBuilder
		typeBuilder mll.TypeBuilder
		parmBuilder mll.ParmBuilder
		entrBuilder mll.EntrBuilder
		tnsrBuilder mll.TnsrBuilder
	)

	sections := make([]mll.SectionInput, 0, 9)
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
	sections = append(sections, mll.SectionInput{
		Tag:           tagXPKG,
		Body:          xbody.Bytes(),
		Flags:         mll.SectionFlagSkippable | mll.SectionFlagSchemaless,
		SchemaVersion: 1,
	})
	return mll.WriteToBytes(mll.ProfileSealed, mll.V1_0, sections)
}

func decodePackageManifestMLL(data []byte) (PackageManifest, error) {
	reader, err := mll.ReadBytes(data, mll.WithDigestVerification())
	if err != nil {
		return PackageManifest{}, err
	}
	if reader.Profile() != mll.ProfileSealed {
		return PackageManifest{}, fmt.Errorf("package manifest profile = %d, want %d", reader.Profile(), mll.ProfileSealed)
	}
	headBody, ok := reader.Section(mll.TagHEAD)
	if !ok {
		return PackageManifest{}, fmt.Errorf("package manifest missing HEAD section")
	}
	strgBody, ok := reader.Section(mll.TagSTRG)
	if !ok {
		return PackageManifest{}, fmt.Errorf("package manifest missing STRG section")
	}
	xpkgBody, ok := reader.Section(tagXPKG)
	if !ok {
		return PackageManifest{}, fmt.Errorf("package manifest missing XPKG section")
	}
	head, err := mll.ReadHeadSection(headBody)
	if err != nil {
		return PackageManifest{}, err
	}
	strg, err := mll.ReadStringTable(strgBody)
	if err != nil {
		return PackageManifest{}, err
	}
	meta := make(map[string]mll.HeadMetadataEntry, len(head.Metadata))
	for _, entry := range head.Metadata {
		meta[strg.At(entry.Key)] = entry
	}
	readString := func(key string) (string, error) {
		entry, ok := meta[key]
		if !ok {
			return "", fmt.Errorf("package manifest missing %q", key)
		}
		if entry.Kind != mll.HeadValueString {
			return "", fmt.Errorf("package manifest key %q kind = %d, want string", key, entry.Kind)
		}
		return strg.At(entry.StringIdx), nil
	}
	readInt := func(key string) (int64, error) {
		entry, ok := meta[key]
		if !ok {
			return 0, fmt.Errorf("package manifest missing %q", key)
		}
		if entry.Kind != mll.HeadValueI64 {
			return 0, fmt.Errorf("package manifest key %q kind = %d, want int", key, entry.Kind)
		}
		return entry.I64, nil
	}

	version, err := readString(manifestVersionKey)
	if err != nil {
		return PackageManifest{}, err
	}
	kind, err := readString("package_kind")
	if err != nil {
		return PackageManifest{}, err
	}
	moduleName, err := readString("module_name")
	if err != nil {
		return PackageManifest{}, err
	}
	artifactVersion, err := readString("artifact_version")
	if err != nil {
		return PackageManifest{}, err
	}
	expectedFiles, err := readInt("file_count")
	if err != nil {
		return PackageManifest{}, err
	}

	r := bytes.NewReader(xpkgBody)
	readU32 := func() (uint32, error) {
		var v uint32
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	}
	readI64 := func() (int64, error) {
		var v int64
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	}

	count, err := readU32()
	if err != nil {
		return PackageManifest{}, err
	}
	if int64(count) != expectedFiles {
		return PackageManifest{}, fmt.Errorf("package manifest file_count = %d, XPKG count = %d", expectedFiles, count)
	}
	files := make([]PackageManifestFile, 0, count)
	for i := uint32(0); i < count; i++ {
		roleIdx, err := readU32()
		if err != nil {
			return PackageManifest{}, err
		}
		pathIdx, err := readU32()
		if err != nil {
			return PackageManifest{}, err
		}
		size, err := readI64()
		if err != nil {
			return PackageManifest{}, err
		}
		sum := make([]byte, sha256.Size)
		if _, err := io.ReadFull(r, sum); err != nil {
			return PackageManifest{}, err
		}
		files = append(files, PackageManifestFile{
			Role:   strg.At(roleIdx),
			Path:   strg.At(pathIdx),
			SHA256: hex.EncodeToString(sum),
			Bytes:  size,
		})
	}
	if r.Len() != 0 {
		return PackageManifest{}, fmt.Errorf("package manifest XPKG has %d trailing bytes", r.Len())
	}
	manifest := PackageManifest{
		Version:         version,
		Kind:            PackageKind(kind),
		ModuleName:      moduleName,
		ArtifactVersion: artifactVersion,
		Files:           files,
	}
	if err := manifest.Validate(); err != nil {
		return PackageManifest{}, err
	}
	return manifest, nil
}

func BuildPackageManifest(kind PackageKind, mod *mantaartifact.Module, files map[string]string) (PackageManifest, error) {
	if mod == nil {
		return PackageManifest{}, fmt.Errorf("nil module")
	}
	roles := make([]string, 0, len(files))
	for role := range files {
		roles = append(roles, role)
	}
	sort.Strings(roles)
	items := make([]PackageManifestFile, 0, len(roles))
	for _, role := range roles {
		path := files[role]
		if path == "" {
			return PackageManifest{}, fmt.Errorf("package file %q path is required", role)
		}
		sum, size, err := fileHash(path)
		if err != nil {
			return PackageManifest{}, fmt.Errorf("hash %q: %w", role, err)
		}
		items = append(items, PackageManifestFile{
			Role:   role,
			Path:   filepath.Base(path),
			SHA256: sum,
			Bytes:  size,
		})
	}
	manifest := PackageManifest{
		Version:         PackageManifestVersion,
		Kind:            kind,
		ModuleName:      mod.Name,
		ArtifactVersion: mod.Version,
		Files:           items,
	}
	return manifest, manifest.Validate()
}

func RebuildSiblingPackageManifest(artifactPath string) (PackageManifest, string, error) {
	manifestPath := ResolvePackageManifestPath(artifactPath)
	current, err := ReadPackageManifestFile(manifestPath)
	if err != nil {
		return PackageManifest{}, "", err
	}
	mod, err := mantaartifact.ReadFile(artifactPath)
	if err != nil {
		return PackageManifest{}, "", err
	}
	files, err := siblingPackageFilesForKind(artifactPath, current.Kind)
	if err != nil {
		return PackageManifest{}, "", err
	}
	rebuilt, err := BuildPackageManifest(current.Kind, mod, files)
	if err != nil {
		return PackageManifest{}, "", err
	}
	outPath := DefaultPackageManifestPath(artifactPath)
	if err := rebuilt.WriteFile(outPath); err != nil {
		return PackageManifest{}, "", err
	}
	return rebuilt, outPath, nil
}

func siblingPackageFilesForKind(artifactPath string, kind PackageKind) (map[string]string, error) {
	paths := map[string]string{
		"artifact": artifactPath,
	}
	addRequired := func(role, path string) error {
		if path == "" {
			return fmt.Errorf("package file %q path is required", role)
		}
		if _, err := os.Stat(path); err != nil {
			return fmt.Errorf("package file %q: %w", role, err)
		}
		paths[role] = path
		return nil
	}
	addOptional := func(role, path string) {
		if path == "" {
			return
		}
		if _, err := os.Stat(path); err == nil {
			paths[role] = path
		}
	}
	if err := addRequired("embedding_manifest", ResolveEmbeddingManifestPath(artifactPath)); err != nil {
		return nil, err
	}
	if err := addRequired("weights", DefaultWeightFilePath(artifactPath)); err != nil {
		return nil, err
	}
	if err := addRequired("memory_plan", DefaultMemoryPlanPath(artifactPath)); err != nil {
		return nil, err
	}
	addOptional("tokenizer", DefaultTokenizerPath(artifactPath))
	switch kind {
	case PackageEmbedding:
		return paths, nil
	case PackageTraining:
		if err := addRequired("train_manifest", ResolveEmbeddingTrainManifestPath(artifactPath)); err != nil {
			return nil, err
		}
		if err := addRequired("checkpoint", DefaultEmbeddingCheckpointPath(artifactPath)); err != nil {
			return nil, err
		}
		if err := addRequired("train_profile", DefaultEmbeddingTrainProfilePath(artifactPath)); err != nil {
			return nil, err
		}
		return paths, nil
	default:
		return nil, fmt.Errorf("unsupported package kind %q", kind)
	}
}

func (m PackageManifest) CacheKey() string {
	if err := m.Validate(); err != nil {
		return ""
	}
	h := sha256.New()
	write := func(s string) {
		_, _ = h.Write([]byte(s))
		_, _ = h.Write([]byte{'\n'})
	}
	write(m.Version)
	write(string(m.Kind))
	write(m.ModuleName)
	write(m.ArtifactVersion)
	for _, item := range m.Files {
		write(item.Role)
		write(item.Path)
		write(item.SHA256)
		write(fmt.Sprintf("%d", item.Bytes))
	}
	return hex.EncodeToString(h.Sum(nil))
}

func (m PackageManifest) VerifyFiles(paths map[string]string) error {
	if err := m.Validate(); err != nil {
		return err
	}
	for _, item := range m.Files {
		path, ok := paths[item.Role]
		if !ok || path == "" {
			return fmt.Errorf("package manifest expects file role %q", item.Role)
		}
		if filepath.Base(path) != item.Path {
			return fmt.Errorf("package file %q path = %q, want %q", item.Role, filepath.Base(path), item.Path)
		}
		sum, size, err := fileHash(path)
		if err != nil {
			return fmt.Errorf("verify %q: %w", item.Role, err)
		}
		if sum != item.SHA256 {
			return fmt.Errorf("package file %q sha256 mismatch", item.Role)
		}
		if size != item.Bytes {
			return fmt.Errorf("package file %q size mismatch: got %d want %d", item.Role, size, item.Bytes)
		}
	}
	return nil
}

func fileHash(path string) (string, int64, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", 0, err
	}
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:]), int64(len(data)), nil
}
