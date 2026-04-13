package mantaruntime

import (
	"context"
	"fmt"

	mantaartifact "github.com/odvcencio/manta/artifact/manta"
	"github.com/odvcencio/manta/runtime/backend"
)

const RetrievalManifestVersion = "manta/retrieval-manifest/v0alpha1"

// RetrievalManifest describes the serving contract for a candidate-pack retrieval module.
type RetrievalManifest struct {
	Name              string `json:"name,omitempty"`
	Entry             string `json:"entry,omitempty"`
	BatchEntry        string `json:"batch_entry,omitempty"`
	QueryInput        string `json:"query_input,omitempty"`
	BatchQueryInput   string `json:"batch_query_input,omitempty"`
	DocsInput         string `json:"docs_input,omitempty"`
	CandidateIDsInput string `json:"candidate_ids_input,omitempty"`
	OutputName        string `json:"output_name,omitempty"`
	BatchOutputName   string `json:"batch_output_name,omitempty"`
	QueryDType        string `json:"query_dtype,omitempty"`
	DocsDType         string `json:"docs_dtype,omitempty"`
	CandidateIDsDType string `json:"candidate_ids_dtype,omitempty"`
}

// RetrievalModel is a manifest-backed candidate retrieval serving handle.
type RetrievalModel struct {
	program  *Program
	manifest RetrievalManifest
}

// ReadRetrievalManifestFile decodes an authored MLL retrieval manifest.
func ReadRetrievalManifestFile(path string) (RetrievalManifest, error) {
	doc, err := readAuthoredManifestMLL(path, "retrieval_manifest", RetrievalManifestVersion)
	if err != nil {
		return RetrievalManifest{}, err
	}
	return retrievalManifestFromDoc(doc)
}

// DefaultRetrievalManifestPath returns the conventional sibling manifest path for an .mll artifact.
func DefaultRetrievalManifestPath(artifactPath string) string {
	return defaultManifestPath(artifactPath, ".retrieval.mll")
}

func ResolveRetrievalManifestPath(artifactPath string) string {
	return DefaultRetrievalManifestPath(artifactPath)
}

// WriteFile writes the retrieval manifest as an authored MLL container.
func (m RetrievalManifest) WriteFile(path string) error {
	return writeAuthoredManifestMLL(path, "retrieval_manifest", RetrievalManifestVersion, m.nameOrDefault(), "Manta retrieval manifest", m.mllValues())
}

// LoadRetrieval loads a retrieval module with a validated serving manifest.
func (rt *Runtime) LoadRetrieval(ctx context.Context, mod *mantaartifact.Module, manifest RetrievalManifest, opts ...LoadOption) (*RetrievalModel, error) {
	manifest = manifest.normalized()
	if err := manifest.ValidateModule(mod); err != nil {
		return nil, err
	}
	prog, err := rt.Load(ctx, mod, opts...)
	if err != nil {
		return nil, err
	}
	return &RetrievalModel{program: prog, manifest: manifest}, nil
}

// LoadRetrievalFile reads a .mll artifact and loads it as a retrieval model.
func (rt *Runtime) LoadRetrievalFile(ctx context.Context, artifactPath string, manifest RetrievalManifest, opts ...LoadOption) (*RetrievalModel, error) {
	mod, err := mantaartifact.ReadFile(artifactPath)
	if err != nil {
		return nil, err
	}
	return rt.LoadRetrieval(ctx, mod, manifest, opts...)
}

// LoadRetrievalBundle reads a .mll artifact plus its sibling retrieval manifest.
func (rt *Runtime) LoadRetrievalBundle(ctx context.Context, artifactPath string, opts ...LoadOption) (*RetrievalModel, error) {
	return rt.LoadRetrievalBundleWithManifest(ctx, artifactPath, ResolveRetrievalManifestPath(artifactPath), opts...)
}

// LoadRetrievalBundleWithManifest reads a .mll artifact plus an explicit retrieval manifest path.
func (rt *Runtime) LoadRetrievalBundleWithManifest(ctx context.Context, artifactPath, manifestPath string, opts ...LoadOption) (*RetrievalModel, error) {
	manifest, err := ReadRetrievalManifestFile(manifestPath)
	if err != nil {
		return nil, err
	}
	return rt.LoadRetrievalFile(ctx, artifactPath, manifest, opts...)
}

func (m RetrievalManifest) nameOrDefault() string {
	if m.Name != "" {
		return m.Name
	}
	return "retrieval_manifest"
}

func (m RetrievalManifest) mllValues() map[string]authoredManifestValue {
	return map[string]authoredManifestValue{
		"name":                authoredString(m.Name),
		"entry":               authoredString(m.Entry),
		"batch_entry":         authoredString(m.BatchEntry),
		"query_input":         authoredString(m.QueryInput),
		"batch_query_input":   authoredString(m.BatchQueryInput),
		"docs_input":          authoredString(m.DocsInput),
		"candidate_ids_input": authoredString(m.CandidateIDsInput),
		"output_name":         authoredString(m.OutputName),
		"batch_output_name":   authoredString(m.BatchOutputName),
		"query_dtype":         authoredString(m.QueryDType),
		"docs_dtype":          authoredString(m.DocsDType),
		"candidate_ids_dtype": authoredString(m.CandidateIDsDType),
	}
}

func retrievalManifestFromDoc(doc authoredManifestDoc) (RetrievalManifest, error) {
	var manifest RetrievalManifest
	var err error
	if manifest.Name, _, err = doc.string("name"); err != nil {
		return RetrievalManifest{}, err
	}
	if manifest.Entry, _, err = doc.string("entry"); err != nil {
		return RetrievalManifest{}, err
	}
	if manifest.BatchEntry, _, err = doc.string("batch_entry"); err != nil {
		return RetrievalManifest{}, err
	}
	if manifest.QueryInput, _, err = doc.string("query_input"); err != nil {
		return RetrievalManifest{}, err
	}
	if manifest.BatchQueryInput, _, err = doc.string("batch_query_input"); err != nil {
		return RetrievalManifest{}, err
	}
	if manifest.DocsInput, _, err = doc.string("docs_input"); err != nil {
		return RetrievalManifest{}, err
	}
	if manifest.CandidateIDsInput, _, err = doc.string("candidate_ids_input"); err != nil {
		return RetrievalManifest{}, err
	}
	if manifest.OutputName, _, err = doc.string("output_name"); err != nil {
		return RetrievalManifest{}, err
	}
	if manifest.BatchOutputName, _, err = doc.string("batch_output_name"); err != nil {
		return RetrievalManifest{}, err
	}
	if manifest.QueryDType, _, err = doc.string("query_dtype"); err != nil {
		return RetrievalManifest{}, err
	}
	if manifest.DocsDType, _, err = doc.string("docs_dtype"); err != nil {
		return RetrievalManifest{}, err
	}
	if manifest.CandidateIDsDType, _, err = doc.string("candidate_ids_dtype"); err != nil {
		return RetrievalManifest{}, err
	}
	return manifest, nil
}

// Manifest reports the normalized retrieval manifest.
func (m *RetrievalModel) Manifest() RetrievalManifest {
	if m == nil {
		return RetrievalManifest{}
	}
	return m.manifest
}

// Backend reports the selected backend.
func (m *RetrievalModel) Backend() mantaartifact.BackendKind {
	if m == nil || m.program == nil {
		return ""
	}
	return m.program.Backend()
}

// Program exposes the underlying loaded program.
func (m *RetrievalModel) Program() *Program {
	if m == nil {
		return nil
	}
	return m.program
}

// Retrieve executes the unbatched retrieval entrypoint against Go-native candidate rows.
func (m *RetrievalModel) Retrieve(ctx context.Context, query *backend.Tensor, candidates []CandidateInput) (CandidateResult, error) {
	if m == nil || m.program == nil {
		return CandidateResult{}, fmt.Errorf("retrieval model is not loaded")
	}
	result, err := m.program.RunCandidateTable(ctx, m.manifest.Entry, query, candidates)
	if err != nil {
		return CandidateResult{}, err
	}
	if err := m.validateCandidateResult(result, false); err != nil {
		return CandidateResult{}, err
	}
	return result, nil
}

// RetrieveBatch executes the batched retrieval entrypoint against Go-native candidate rows.
func (m *RetrievalModel) RetrieveBatch(ctx context.Context, queries *backend.Tensor, batches [][]CandidateInput) (CandidateResult, error) {
	if m == nil || m.program == nil {
		return CandidateResult{}, fmt.Errorf("retrieval model is not loaded")
	}
	if m.manifest.BatchEntry == "" {
		return CandidateResult{}, fmt.Errorf("retrieval model does not declare a batch entrypoint")
	}
	result, err := m.program.RunCandidateTableBatched(ctx, m.manifest.BatchEntry, queries, batches)
	if err != nil {
		return CandidateResult{}, err
	}
	if err := m.validateCandidateResult(result, true); err != nil {
		return CandidateResult{}, err
	}
	return result, nil
}

func (m RetrievalManifest) normalized() RetrievalManifest {
	if m.Entry == "" {
		m.Entry = "rerank_candidates_packed"
	}
	if m.QueryInput == "" {
		m.QueryInput = "query"
	}
	if m.BatchQueryInput == "" {
		m.BatchQueryInput = "queries"
	}
	if m.DocsInput == "" {
		m.DocsInput = "docs"
	}
	if m.CandidateIDsInput == "" {
		m.CandidateIDsInput = "candidate_ids"
	}
	if m.QueryDType == "" {
		m.QueryDType = "f16"
	}
	if m.DocsDType == "" {
		m.DocsDType = "q4"
	}
	if m.CandidateIDsDType == "" {
		m.CandidateIDsDType = "i64"
	}
	return m
}

// ValidateModule checks that a module satisfies the retrieval serving contract.
func (m RetrievalManifest) ValidateModule(mod *mantaartifact.Module) error {
	if mod == nil {
		return fmt.Errorf("nil module")
	}
	if err := validateRetrievalEntry(mod, m.Entry, m.QueryInput, 1, m.DocsInput, 2, m.CandidateIDsInput, 1, m.QueryDType, m.DocsDType, m.CandidateIDsDType, m.OutputName, 2); err != nil {
		return err
	}
	if m.BatchEntry != "" {
		if err := validateRetrievalEntry(mod, m.BatchEntry, m.BatchQueryInput, 2, m.DocsInput, 3, m.CandidateIDsInput, 2, m.QueryDType, m.DocsDType, m.CandidateIDsDType, m.BatchOutputName, 3); err != nil {
			return err
		}
	}
	return nil
}

func (m *RetrievalModel) validateCandidateResult(result CandidateResult, batched bool) error {
	if want := m.manifest.OutputName; !batched && want != "" && result.OutputName != want {
		return fmt.Errorf("retrieval output name %q does not match manifest %q", result.OutputName, want)
	}
	if want := m.manifest.BatchOutputName; batched && want != "" && result.OutputName != want {
		return fmt.Errorf("retrieval batch output name %q does not match manifest %q", result.OutputName, want)
	}
	if len(result.Batches) == 0 {
		return fmt.Errorf("retrieval output has no candidate batches")
	}
	return nil
}

func validateRetrievalEntry(mod *mantaartifact.Module, entryName, queryInput string, queryRank int, docsInput string, docsRank int, idsInput string, idsRank int, queryDType, docsDType, idsDType, outputName string, outputRank int) error {
	entry, err := findEntryPoint(mod, entryName)
	if err != nil {
		return err
	}
	query, err := requireEntryInput(entry, queryInput)
	if err != nil {
		return err
	}
	if query.Type.Kind != mantaartifact.ValueTensor || query.Type.Tensor == nil {
		return fmt.Errorf("entrypoint %q input %q is not a tensor", entryName, queryInput)
	}
	if queryDType != "" && query.Type.Tensor.DType != queryDType {
		return fmt.Errorf("entrypoint %q input %q dtype = %q, want %q", entryName, queryInput, query.Type.Tensor.DType, queryDType)
	}
	if got := len(query.Type.Tensor.Shape); got != queryRank {
		return fmt.Errorf("entrypoint %q input %q rank = %d, want %d", entryName, queryInput, got, queryRank)
	}
	docs, err := requireEntryInput(entry, docsInput)
	if err != nil {
		return err
	}
	if docs.Type.Kind != mantaartifact.ValueTensor || docs.Type.Tensor == nil {
		return fmt.Errorf("entrypoint %q input %q is not a tensor", entryName, docsInput)
	}
	if docsDType != "" && docs.Type.Tensor.DType != docsDType {
		return fmt.Errorf("entrypoint %q input %q dtype = %q, want %q", entryName, docsInput, docs.Type.Tensor.DType, docsDType)
	}
	if got := len(docs.Type.Tensor.Shape); got != docsRank {
		return fmt.Errorf("entrypoint %q input %q rank = %d, want %d", entryName, docsInput, got, docsRank)
	}
	ids, err := requireEntryInput(entry, idsInput)
	if err != nil {
		return err
	}
	if ids.Type.Kind != mantaartifact.ValueTensor || ids.Type.Tensor == nil {
		return fmt.Errorf("entrypoint %q input %q is not a tensor", entryName, idsInput)
	}
	if idsDType != "" && ids.Type.Tensor.DType != idsDType {
		return fmt.Errorf("entrypoint %q input %q dtype = %q, want %q", entryName, idsInput, ids.Type.Tensor.DType, idsDType)
	}
	if got := len(ids.Type.Tensor.Shape); got != idsRank {
		return fmt.Errorf("entrypoint %q input %q rank = %d, want %d", entryName, idsInput, got, idsRank)
	}
	if len(entry.Outputs) != 1 {
		return fmt.Errorf("entrypoint %q output count = %d, want 1", entryName, len(entry.Outputs))
	}
	output := entry.Outputs[0]
	if outputName != "" && output.Name != outputName {
		return fmt.Errorf("entrypoint %q output %q does not match manifest %q", entryName, output.Name, outputName)
	}
	if output.Type.Kind != mantaartifact.ValueCandidatePack || output.Type.CandidatePack == nil {
		return fmt.Errorf("entrypoint %q output %q is not a candidate_pack", entryName, output.Name)
	}
	if got := len(output.Type.CandidatePack.Shape); got != outputRank {
		return fmt.Errorf("entrypoint %q candidate_pack rank = %d, want %d", entryName, got, outputRank)
	}
	return nil
}
