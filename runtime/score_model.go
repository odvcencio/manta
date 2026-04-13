package mantaruntime

import (
	"context"
	"fmt"

	mantaartifact "github.com/odvcencio/manta/artifact/manta"
	"github.com/odvcencio/manta/runtime/backend"
)

const ScoreManifestVersion = "manta/score-manifest/v0alpha1"

// ScoreManifest describes the serving contract for a score/rerank module.
type ScoreManifest struct {
	Name            string `json:"name,omitempty"`
	Entry           string `json:"entry,omitempty"`
	BatchEntry      string `json:"batch_entry,omitempty"`
	QueryInput      string `json:"query_input,omitempty"`
	BatchQueryInput string `json:"batch_query_input,omitempty"`
	DocsInput       string `json:"docs_input,omitempty"`
	OutputName      string `json:"output_name,omitempty"`
	BatchOutputName string `json:"batch_output_name,omitempty"`
	QueryDType      string `json:"query_dtype,omitempty"`
	DocsDType       string `json:"docs_dtype,omitempty"`
	OutputDType     string `json:"output_dtype,omitempty"`
}

// ScoreModel is a manifest-backed score/rerank serving handle.
type ScoreModel struct {
	program  *Program
	manifest ScoreManifest
}

// ReadScoreManifestFile decodes an authored MLL score manifest.
func ReadScoreManifestFile(path string) (ScoreManifest, error) {
	doc, err := readAuthoredManifestMLL(path, "score_manifest", ScoreManifestVersion)
	if err != nil {
		return ScoreManifest{}, err
	}
	return scoreManifestFromDoc(doc)
}

// DefaultScoreManifestPath returns the conventional sibling manifest path for an .mll artifact.
func DefaultScoreManifestPath(artifactPath string) string {
	return defaultManifestPath(artifactPath, ".score.mll")
}

func ResolveScoreManifestPath(artifactPath string) string {
	return DefaultScoreManifestPath(artifactPath)
}

// WriteFile writes the score manifest as an authored MLL container.
func (m ScoreManifest) WriteFile(path string) error {
	return writeAuthoredManifestMLL(path, "score_manifest", ScoreManifestVersion, m.nameOrDefault(), "Manta score manifest", m.mllValues())
}

// LoadScore loads a score module with a validated serving manifest.
func (rt *Runtime) LoadScore(ctx context.Context, mod *mantaartifact.Module, manifest ScoreManifest, opts ...LoadOption) (*ScoreModel, error) {
	manifest = manifest.normalized()
	if err := manifest.ValidateModule(mod); err != nil {
		return nil, err
	}
	prog, err := rt.Load(ctx, mod, opts...)
	if err != nil {
		return nil, err
	}
	return &ScoreModel{program: prog, manifest: manifest}, nil
}

// LoadScoreFile reads a .mll artifact and loads it as a score model.
func (rt *Runtime) LoadScoreFile(ctx context.Context, artifactPath string, manifest ScoreManifest, opts ...LoadOption) (*ScoreModel, error) {
	mod, err := mantaartifact.ReadFile(artifactPath)
	if err != nil {
		return nil, err
	}
	return rt.LoadScore(ctx, mod, manifest, opts...)
}

// LoadScoreBundle reads a .mll artifact plus its sibling score manifest.
func (rt *Runtime) LoadScoreBundle(ctx context.Context, artifactPath string, opts ...LoadOption) (*ScoreModel, error) {
	return rt.LoadScoreBundleWithManifest(ctx, artifactPath, ResolveScoreManifestPath(artifactPath), opts...)
}

// LoadScoreBundleWithManifest reads a .mll artifact plus an explicit score manifest path.
func (rt *Runtime) LoadScoreBundleWithManifest(ctx context.Context, artifactPath, manifestPath string, opts ...LoadOption) (*ScoreModel, error) {
	manifest, err := ReadScoreManifestFile(manifestPath)
	if err != nil {
		return nil, err
	}
	return rt.LoadScoreFile(ctx, artifactPath, manifest, opts...)
}

func (m ScoreManifest) nameOrDefault() string {
	if m.Name != "" {
		return m.Name
	}
	return "score_manifest"
}

func (m ScoreManifest) mllValues() map[string]authoredManifestValue {
	return map[string]authoredManifestValue{
		"name":              authoredString(m.Name),
		"entry":             authoredString(m.Entry),
		"batch_entry":       authoredString(m.BatchEntry),
		"query_input":       authoredString(m.QueryInput),
		"batch_query_input": authoredString(m.BatchQueryInput),
		"docs_input":        authoredString(m.DocsInput),
		"output_name":       authoredString(m.OutputName),
		"batch_output_name": authoredString(m.BatchOutputName),
		"query_dtype":       authoredString(m.QueryDType),
		"docs_dtype":        authoredString(m.DocsDType),
		"output_dtype":      authoredString(m.OutputDType),
	}
}

func scoreManifestFromDoc(doc authoredManifestDoc) (ScoreManifest, error) {
	var manifest ScoreManifest
	var err error
	if manifest.Name, _, err = doc.string("name"); err != nil {
		return ScoreManifest{}, err
	}
	if manifest.Entry, _, err = doc.string("entry"); err != nil {
		return ScoreManifest{}, err
	}
	if manifest.BatchEntry, _, err = doc.string("batch_entry"); err != nil {
		return ScoreManifest{}, err
	}
	if manifest.QueryInput, _, err = doc.string("query_input"); err != nil {
		return ScoreManifest{}, err
	}
	if manifest.BatchQueryInput, _, err = doc.string("batch_query_input"); err != nil {
		return ScoreManifest{}, err
	}
	if manifest.DocsInput, _, err = doc.string("docs_input"); err != nil {
		return ScoreManifest{}, err
	}
	if manifest.OutputName, _, err = doc.string("output_name"); err != nil {
		return ScoreManifest{}, err
	}
	if manifest.BatchOutputName, _, err = doc.string("batch_output_name"); err != nil {
		return ScoreManifest{}, err
	}
	if manifest.QueryDType, _, err = doc.string("query_dtype"); err != nil {
		return ScoreManifest{}, err
	}
	if manifest.DocsDType, _, err = doc.string("docs_dtype"); err != nil {
		return ScoreManifest{}, err
	}
	if manifest.OutputDType, _, err = doc.string("output_dtype"); err != nil {
		return ScoreManifest{}, err
	}
	return manifest, nil
}

// Manifest reports the normalized score manifest.
func (m *ScoreModel) Manifest() ScoreManifest {
	if m == nil {
		return ScoreManifest{}
	}
	return m.manifest
}

// Backend reports the selected backend.
func (m *ScoreModel) Backend() mantaartifact.BackendKind {
	if m == nil || m.program == nil {
		return ""
	}
	return m.program.Backend()
}

// Program exposes the underlying loaded program.
func (m *ScoreModel) Program() *Program {
	if m == nil {
		return nil
	}
	return m.program
}

// Score executes the unbatched score entrypoint against Go-native doc rows.
func (m *ScoreModel) Score(ctx context.Context, query *backend.Tensor, docs []*backend.Tensor) (ScoreResult, error) {
	if m == nil || m.program == nil {
		return ScoreResult{}, fmt.Errorf("score model is not loaded")
	}
	result, err := m.program.RunScoreTable(ctx, m.manifest.Entry, query, docs)
	if err != nil {
		return ScoreResult{}, err
	}
	if err := m.validateScoreResult(result, false); err != nil {
		return ScoreResult{}, err
	}
	return result, nil
}

// ScoreBatch executes the batched score entrypoint against Go-native doc rows.
func (m *ScoreModel) ScoreBatch(ctx context.Context, queries *backend.Tensor, batches [][]*backend.Tensor) (ScoreResult, error) {
	if m == nil || m.program == nil {
		return ScoreResult{}, fmt.Errorf("score model is not loaded")
	}
	if m.manifest.BatchEntry == "" {
		return ScoreResult{}, fmt.Errorf("score model does not declare a batch entrypoint")
	}
	result, err := m.program.RunScoreTableBatched(ctx, m.manifest.BatchEntry, queries, batches)
	if err != nil {
		return ScoreResult{}, err
	}
	if err := m.validateScoreResult(result, true); err != nil {
		return ScoreResult{}, err
	}
	return result, nil
}

func (m ScoreManifest) normalized() ScoreManifest {
	if m.Entry == "" {
		m.Entry = "score"
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
	if m.QueryDType == "" {
		m.QueryDType = "f16"
	}
	if m.DocsDType == "" {
		m.DocsDType = "q4"
	}
	if m.OutputDType == "" {
		m.OutputDType = "f32"
	}
	return m
}

// ValidateModule checks that a module satisfies the score serving contract.
func (m ScoreManifest) ValidateModule(mod *mantaartifact.Module) error {
	if mod == nil {
		return fmt.Errorf("nil module")
	}
	if err := validateScoreEntry(mod, m.Entry, m.QueryInput, 1, m.DocsInput, 2, m.QueryDType, m.DocsDType, m.OutputName, 1, m.OutputDType); err != nil {
		return err
	}
	if m.BatchEntry != "" {
		if err := validateScoreEntry(mod, m.BatchEntry, m.BatchQueryInput, 2, m.DocsInput, 3, m.QueryDType, m.DocsDType, m.BatchOutputName, 2, m.OutputDType); err != nil {
			return err
		}
	}
	return nil
}

func (m *ScoreModel) validateScoreResult(result ScoreResult, batched bool) error {
	if want := m.manifest.OutputName; !batched && want != "" && result.OutputName != want {
		return fmt.Errorf("score output name %q does not match manifest %q", result.OutputName, want)
	}
	if want := m.manifest.BatchOutputName; batched && want != "" && result.OutputName != want {
		return fmt.Errorf("score batch output name %q does not match manifest %q", result.OutputName, want)
	}
	if result.Scores == nil {
		return fmt.Errorf("score output tensor is nil")
	}
	if want := m.manifest.OutputDType; want != "" && result.Scores.DType != want {
		return fmt.Errorf("score output dtype %q does not match manifest %q", result.Scores.DType, want)
	}
	wantRank := 1
	if batched {
		wantRank = 2
	}
	if got := len(result.Scores.Shape); got != wantRank {
		return fmt.Errorf("score output rank %d does not match expected %d", got, wantRank)
	}
	return nil
}

func validateScoreEntry(mod *mantaartifact.Module, entryName, queryInput string, queryRank int, docsInput string, docsRank int, queryDType, docsDType, outputName string, outputRank int, outputDType string) error {
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
	if len(entry.Outputs) != 1 {
		return fmt.Errorf("entrypoint %q output count = %d, want 1", entryName, len(entry.Outputs))
	}
	output := entry.Outputs[0]
	if outputName != "" && output.Name != outputName {
		return fmt.Errorf("entrypoint %q output %q does not match manifest %q", entryName, output.Name, outputName)
	}
	if output.Type.Kind != mantaartifact.ValueTensor || output.Type.Tensor == nil {
		return fmt.Errorf("entrypoint %q output %q is not a tensor", entryName, output.Name)
	}
	if outputDType != "" && output.Type.Tensor.DType != outputDType {
		return fmt.Errorf("entrypoint %q output dtype = %q, want %q", entryName, output.Type.Tensor.DType, outputDType)
	}
	if got := len(output.Type.Tensor.Shape); got != outputRank {
		return fmt.Errorf("entrypoint %q output rank = %d, want %d", entryName, got, outputRank)
	}
	return nil
}
