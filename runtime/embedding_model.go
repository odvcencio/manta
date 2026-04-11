package barruntime

import (
	"context"
	"encoding/json"
	"fmt"
	"os"

	"github.com/odvcencio/manta/artifact/barr"
	"github.com/odvcencio/manta/runtime/backend"
)

const EmbeddingManifestVersion = "barr/embedding-manifest/v0alpha1"

// TokenizerManifest carries embedding-model tokenization limits and ids.
type TokenizerManifest struct {
	VocabSize   int   `json:"vocab_size,omitempty"`
	MaxSequence int   `json:"max_sequence,omitempty"`
	PadID       int32 `json:"pad_id,omitempty"`
	BOSID       int32 `json:"bos_id,omitempty"`
	EOSID       int32 `json:"eos_id,omitempty"`
	UnknownID   int32 `json:"unknown_id,omitempty"`
}

// EmbeddingManifest describes the serving contract for an embedding module.
type EmbeddingManifest struct {
	Name                  string            `json:"name,omitempty"`
	PooledEntry           string            `json:"pooled_entry,omitempty"`
	BatchEntry            string            `json:"batch_entry,omitempty"`
	EncoderRepeats        int               `json:"encoder_repeats,omitempty"`
	TokenInput            string            `json:"token_input,omitempty"`
	MaskInput             string            `json:"mask_input,omitempty"`
	OutputName            string            `json:"output_name,omitempty"`
	OutputDType           string            `json:"output_dtype,omitempty"`
	TokenEmbeddingParam   string            `json:"token_embedding_param,omitempty"`
	AttentionQueryParam   string            `json:"attention_query_param,omitempty"`
	AttentionKeyParam     string            `json:"attention_key_param,omitempty"`
	AttentionValueParam   string            `json:"attention_value_param,omitempty"`
	AttentionOutputParam  string            `json:"attention_output_param,omitempty"`
	AttentionResidual     bool              `json:"attention_residual,omitempty"`
	AttentionLayerNorm    bool              `json:"attention_layernorm,omitempty"`
	HiddenProjectionParam string            `json:"hidden_projection_param,omitempty"`
	FFNResidual           bool              `json:"ffn_residual,omitempty"`
	FFNLayerNorm          bool              `json:"ffn_layernorm,omitempty"`
	ProjectionParam       string            `json:"projection_param,omitempty"`
	Tokenizer             TokenizerManifest `json:"tokenizer,omitempty"`
}

// EmbeddingModel is a manifest-backed embedding serving handle.
type EmbeddingModel struct {
	program  *Program
	manifest EmbeddingManifest
}

// ReadEmbeddingManifestFile decodes a JSON embedding manifest.
func ReadEmbeddingManifestFile(path string) (EmbeddingManifest, error) {
	if doc, err := readAuthoredManifestMLL(path, "embedding_manifest", EmbeddingManifestVersion); err == nil {
		return embeddingManifestFromDoc(doc)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return EmbeddingManifest{}, err
	}
	var manifest EmbeddingManifest
	if err := json.Unmarshal(data, &manifest); err != nil {
		return EmbeddingManifest{}, err
	}
	return manifest, nil
}

// DefaultEmbeddingManifestPath returns the conventional sibling manifest path for a .barr artifact.
func DefaultEmbeddingManifestPath(barrPath string) string {
	return defaultManifestPath(barrPath, ".embedding.mll")
}

// ResolveEmbeddingManifestPath returns the preferred sibling embedding manifest path, falling back to legacy JSON.
func ResolveEmbeddingManifestPath(barrPath string) string {
	return resolveSiblingPath(barrPath, ".embedding.mll", ".embedding.json")
}

// WriteFile writes the embedding manifest as an authored MLL container.
func (m EmbeddingManifest) WriteFile(path string) error {
	return writeAuthoredManifestMLL(path, "embedding_manifest", EmbeddingManifestVersion, m.nameOrDefault(), "Manta embedding manifest", m.mllValues())
}

// LoadEmbedding loads an embedding module with a validated serving manifest.
func (rt *Runtime) LoadEmbedding(ctx context.Context, mod *barr.Module, manifest EmbeddingManifest, opts ...LoadOption) (*EmbeddingModel, error) {
	manifest = manifest.normalized()
	if err := manifest.ValidateModule(mod); err != nil {
		return nil, err
	}
	prog, err := rt.Load(ctx, mod, opts...)
	if err != nil {
		return nil, err
	}
	return &EmbeddingModel{program: prog, manifest: manifest}, nil
}

// LoadEmbeddingFile reads a .barr artifact and loads it as an embedding model.
func (rt *Runtime) LoadEmbeddingFile(ctx context.Context, barrPath string, manifest EmbeddingManifest, opts ...LoadOption) (*EmbeddingModel, error) {
	mod, err := barr.ReadFile(barrPath)
	if err != nil {
		return nil, err
	}
	return rt.LoadEmbedding(ctx, mod, manifest, opts...)
}

// LoadEmbeddingBundle reads a .barr artifact plus its sibling embedding manifest.
func (rt *Runtime) LoadEmbeddingBundle(ctx context.Context, barrPath string, opts ...LoadOption) (*EmbeddingModel, error) {
	return rt.LoadEmbeddingBundleWithManifest(ctx, barrPath, ResolveEmbeddingManifestPath(barrPath), opts...)
}

// LoadEmbeddingBundleWithManifest reads a .barr artifact plus an explicit embedding manifest path.
func (rt *Runtime) LoadEmbeddingBundleWithManifest(ctx context.Context, barrPath, manifestPath string, opts ...LoadOption) (*EmbeddingModel, error) {
	manifest, err := ReadEmbeddingManifestFile(manifestPath)
	if err != nil {
		return nil, err
	}
	return rt.LoadEmbeddingFile(ctx, barrPath, manifest, opts...)
}

func (m EmbeddingManifest) nameOrDefault() string {
	if m.Name != "" {
		return m.Name
	}
	return "embedding_manifest"
}

func (m EmbeddingManifest) mllValues() map[string]authoredManifestValue {
	return map[string]authoredManifestValue{
		"name":                    authoredString(m.Name),
		"pooled_entry":            authoredString(m.PooledEntry),
		"batch_entry":             authoredString(m.BatchEntry),
		"encoder_repeats":         authoredInt(int64(m.EncoderRepeats)),
		"token_input":             authoredString(m.TokenInput),
		"mask_input":              authoredString(m.MaskInput),
		"output_name":             authoredString(m.OutputName),
		"output_dtype":            authoredString(m.OutputDType),
		"token_embedding_param":   authoredString(m.TokenEmbeddingParam),
		"attention_query_param":   authoredString(m.AttentionQueryParam),
		"attention_key_param":     authoredString(m.AttentionKeyParam),
		"attention_value_param":   authoredString(m.AttentionValueParam),
		"attention_output_param":  authoredString(m.AttentionOutputParam),
		"attention_residual":      authoredBool(m.AttentionResidual),
		"attention_layernorm":     authoredBool(m.AttentionLayerNorm),
		"hidden_projection_param": authoredString(m.HiddenProjectionParam),
		"ffn_residual":            authoredBool(m.FFNResidual),
		"ffn_layernorm":           authoredBool(m.FFNLayerNorm),
		"projection_param":        authoredString(m.ProjectionParam),
		"tokenizer.vocab_size":    authoredInt(int64(m.Tokenizer.VocabSize)),
		"tokenizer.max_sequence":  authoredInt(int64(m.Tokenizer.MaxSequence)),
		"tokenizer.pad_id":        authoredInt(int64(m.Tokenizer.PadID)),
		"tokenizer.bos_id":        authoredInt(int64(m.Tokenizer.BOSID)),
		"tokenizer.eos_id":        authoredInt(int64(m.Tokenizer.EOSID)),
		"tokenizer.unknown_id":    authoredInt(int64(m.Tokenizer.UnknownID)),
	}
}

func embeddingManifestFromDoc(doc authoredManifestDoc) (EmbeddingManifest, error) {
	var manifest EmbeddingManifest
	if value, _, err := doc.string("name"); err != nil {
		return EmbeddingManifest{}, err
	} else {
		manifest.Name = value
	}
	if value, _, err := doc.string("pooled_entry"); err != nil {
		return EmbeddingManifest{}, err
	} else {
		manifest.PooledEntry = value
	}
	if value, _, err := doc.string("batch_entry"); err != nil {
		return EmbeddingManifest{}, err
	} else {
		manifest.BatchEntry = value
	}
	if value, _, err := doc.int("encoder_repeats"); err != nil {
		return EmbeddingManifest{}, err
	} else {
		manifest.EncoderRepeats = int(value)
	}
	if value, _, err := doc.string("token_input"); err != nil {
		return EmbeddingManifest{}, err
	} else {
		manifest.TokenInput = value
	}
	if value, _, err := doc.string("mask_input"); err != nil {
		return EmbeddingManifest{}, err
	} else {
		manifest.MaskInput = value
	}
	if value, _, err := doc.string("output_name"); err != nil {
		return EmbeddingManifest{}, err
	} else {
		manifest.OutputName = value
	}
	if value, _, err := doc.string("output_dtype"); err != nil {
		return EmbeddingManifest{}, err
	} else {
		manifest.OutputDType = value
	}
	if value, _, err := doc.string("token_embedding_param"); err != nil {
		return EmbeddingManifest{}, err
	} else {
		manifest.TokenEmbeddingParam = value
	}
	if value, _, err := doc.string("attention_query_param"); err != nil {
		return EmbeddingManifest{}, err
	} else {
		manifest.AttentionQueryParam = value
	}
	if value, _, err := doc.string("attention_key_param"); err != nil {
		return EmbeddingManifest{}, err
	} else {
		manifest.AttentionKeyParam = value
	}
	if value, _, err := doc.string("attention_value_param"); err != nil {
		return EmbeddingManifest{}, err
	} else {
		manifest.AttentionValueParam = value
	}
	if value, _, err := doc.string("attention_output_param"); err != nil {
		return EmbeddingManifest{}, err
	} else {
		manifest.AttentionOutputParam = value
	}
	if value, _, err := doc.bool("attention_residual"); err != nil {
		return EmbeddingManifest{}, err
	} else {
		manifest.AttentionResidual = value
	}
	if value, _, err := doc.bool("attention_layernorm"); err != nil {
		return EmbeddingManifest{}, err
	} else {
		manifest.AttentionLayerNorm = value
	}
	if value, _, err := doc.string("hidden_projection_param"); err != nil {
		return EmbeddingManifest{}, err
	} else {
		manifest.HiddenProjectionParam = value
	}
	if value, _, err := doc.bool("ffn_residual"); err != nil {
		return EmbeddingManifest{}, err
	} else {
		manifest.FFNResidual = value
	}
	if value, _, err := doc.bool("ffn_layernorm"); err != nil {
		return EmbeddingManifest{}, err
	} else {
		manifest.FFNLayerNorm = value
	}
	if value, _, err := doc.string("projection_param"); err != nil {
		return EmbeddingManifest{}, err
	} else {
		manifest.ProjectionParam = value
	}
	if value, _, err := doc.int("tokenizer.vocab_size"); err != nil {
		return EmbeddingManifest{}, err
	} else {
		manifest.Tokenizer.VocabSize = int(value)
	}
	if value, _, err := doc.int("tokenizer.max_sequence"); err != nil {
		return EmbeddingManifest{}, err
	} else {
		manifest.Tokenizer.MaxSequence = int(value)
	}
	if value, _, err := doc.int("tokenizer.pad_id"); err != nil {
		return EmbeddingManifest{}, err
	} else {
		manifest.Tokenizer.PadID = int32(value)
	}
	if value, _, err := doc.int("tokenizer.bos_id"); err != nil {
		return EmbeddingManifest{}, err
	} else {
		manifest.Tokenizer.BOSID = int32(value)
	}
	if value, _, err := doc.int("tokenizer.eos_id"); err != nil {
		return EmbeddingManifest{}, err
	} else {
		manifest.Tokenizer.EOSID = int32(value)
	}
	if value, _, err := doc.int("tokenizer.unknown_id"); err != nil {
		return EmbeddingManifest{}, err
	} else {
		manifest.Tokenizer.UnknownID = int32(value)
	}
	return manifest, nil
}

// Manifest reports the normalized embedding manifest.
func (m *EmbeddingModel) Manifest() EmbeddingManifest {
	if m == nil {
		return EmbeddingManifest{}
	}
	return m.manifest
}

// Backend reports the selected backend.
func (m *EmbeddingModel) Backend() barr.BackendKind {
	if m == nil || m.program == nil {
		return ""
	}
	return m.program.Backend()
}

// Program exposes the underlying loaded program.
func (m *EmbeddingModel) Program() *Program {
	if m == nil {
		return nil
	}
	return m.program
}

func (m *EmbeddingModel) MemoryPlan() *MemoryPlan {
	if m == nil || m.program == nil {
		return nil
	}
	return m.program.MemoryPlan()
}

// Embed executes the pooled embedding entrypoint for one token sequence.
func (m *EmbeddingModel) Embed(ctx context.Context, tokens []int32) (EmbeddingResult, error) {
	if m == nil || m.program == nil {
		return EmbeddingResult{}, fmt.Errorf("embedding model is not loaded")
	}
	if err := m.validateTokenSequence(tokens); err != nil {
		return EmbeddingResult{}, err
	}
	if m.manifest.MaskInput == "" {
		result, err := m.program.RunEmbed(ctx, m.manifest.PooledEntry, tokens)
		if err != nil {
			return EmbeddingResult{}, err
		}
		if err := m.validateEmbeddingResult(result, false); err != nil {
			return EmbeddingResult{}, err
		}
		return result, nil
	}
	entry, err := findEntryPoint(m.program.module, m.manifest.PooledEntry)
	if err != nil {
		return EmbeddingResult{}, err
	}
	tokenInput, err := requireEntryInput(entry, m.manifest.TokenInput)
	if err != nil {
		return EmbeddingResult{}, err
	}
	maskInput, err := requireEntryInput(entry, m.manifest.MaskInput)
	if err != nil {
		return EmbeddingResult{}, err
	}
	tokenTensor, maskTensor, err := buildMaskedTokenInputs([][]int32{tokens}, m.manifest.Tokenizer.PadID, false)
	if err != nil {
		return EmbeddingResult{}, err
	}
	raw, err := m.program.Run(ctx, backend.Request{
		Entry: m.manifest.PooledEntry,
		Inputs: map[string]any{
			tokenInput.Name: tokenTensor,
			maskInput.Name:  maskTensor,
		},
	})
	if err != nil {
		return EmbeddingResult{}, err
	}
	result, err := decodeEmbeddingResult(raw)
	if err != nil {
		return EmbeddingResult{}, err
	}
	if err := m.validateEmbeddingResult(result, false); err != nil {
		return EmbeddingResult{}, err
	}
	return result, nil
}

// EmbedBatch executes the batched pooled embedding entrypoint.
func (m *EmbeddingModel) EmbedBatch(ctx context.Context, batches [][]int32) (EmbeddingResult, error) {
	if m == nil || m.program == nil {
		return EmbeddingResult{}, fmt.Errorf("embedding model is not loaded")
	}
	if len(batches) == 0 {
		return EmbeddingResult{}, fmt.Errorf("token batches are empty")
	}
	for i, batch := range batches {
		if err := m.validateTokenSequence(batch); err != nil {
			return EmbeddingResult{}, fmt.Errorf("batch %d: %w", i, err)
		}
	}
	if m.manifest.MaskInput == "" {
		result, err := m.program.RunEmbedBatch(ctx, m.manifest.BatchEntry, batches)
		if err != nil {
			return EmbeddingResult{}, err
		}
		if err := m.validateEmbeddingResult(result, true); err != nil {
			return EmbeddingResult{}, err
		}
		return result, nil
	}
	entry, err := findEntryPoint(m.program.module, m.manifest.BatchEntry)
	if err != nil {
		return EmbeddingResult{}, err
	}
	tokenInput, err := requireEntryInput(entry, m.manifest.TokenInput)
	if err != nil {
		return EmbeddingResult{}, err
	}
	maskInput, err := requireEntryInput(entry, m.manifest.MaskInput)
	if err != nil {
		return EmbeddingResult{}, err
	}
	tokenTensor, maskTensor, err := buildMaskedTokenInputs(batches, m.manifest.Tokenizer.PadID, true)
	if err != nil {
		return EmbeddingResult{}, err
	}
	raw, err := m.program.Run(ctx, backend.Request{
		Entry: m.manifest.BatchEntry,
		Inputs: map[string]any{
			tokenInput.Name: tokenTensor,
			maskInput.Name:  maskTensor,
		},
	})
	if err != nil {
		return EmbeddingResult{}, err
	}
	result, err := decodeEmbeddingResult(raw)
	if err != nil {
		return EmbeddingResult{}, err
	}
	if err := m.validateEmbeddingResult(result, true); err != nil {
		return EmbeddingResult{}, err
	}
	return result, nil
}

func (m EmbeddingManifest) normalized() EmbeddingManifest {
	if m.PooledEntry == "" {
		m.PooledEntry = "embed_pooled"
	}
	if m.BatchEntry == "" {
		m.BatchEntry = "embed_pooled_batch"
	}
	if m.EncoderRepeats <= 0 {
		m.EncoderRepeats = 1
	}
	if m.TokenInput == "" {
		m.TokenInput = "tokens"
	}
	if m.OutputDType == "" {
		m.OutputDType = "f16"
	}
	if m.TokenEmbeddingParam == "" {
		m.TokenEmbeddingParam = "token_embedding"
	}
	if m.ProjectionParam == "" {
		m.ProjectionParam = "projection"
	}
	return m
}

// ValidateModule checks that a module satisfies the embedding serving contract.
func (m EmbeddingManifest) ValidateModule(mod *barr.Module) error {
	if mod == nil {
		return fmt.Errorf("nil module")
	}
	if (m.AttentionResidual || m.AttentionLayerNorm) && m.AttentionQueryParam == "" {
		return fmt.Errorf("attention residual/layernorm requires attention params")
	}
	if (m.FFNResidual || m.FFNLayerNorm) && m.HiddenProjectionParam == "" {
		return fmt.Errorf("ffn residual/layernorm requires hidden_projection_param")
	}
	if err := validateEmbeddingEntry(mod, m.PooledEntry, m.TokenInput, m.MaskInput, 1, 1, m.OutputDType); err != nil {
		return err
	}
	if err := validateEmbeddingEntry(mod, m.BatchEntry, m.TokenInput, m.MaskInput, 2, 2, m.OutputDType); err != nil {
		return err
	}
	if err := validateEmbeddingParam(mod, m.TokenEmbeddingParam); err != nil {
		return err
	}
	if err := validateAttentionParams(mod, m); err != nil {
		return err
	}
	if m.HiddenProjectionParam != "" {
		if err := validateEmbeddingParam(mod, m.HiddenProjectionParam); err != nil {
			return err
		}
	}
	if err := validateEmbeddingParam(mod, m.ProjectionParam); err != nil {
		return err
	}
	return nil
}

func (m *EmbeddingModel) validateTokenSequence(tokens []int32) error {
	if len(tokens) == 0 {
		return fmt.Errorf("tokens are empty")
	}
	if limit := m.manifest.Tokenizer.MaxSequence; limit > 0 && len(tokens) > limit {
		return fmt.Errorf("token sequence length %d exceeds max_sequence %d", len(tokens), limit)
	}
	if vocab := m.manifest.Tokenizer.VocabSize; vocab > 0 {
		for i, tok := range tokens {
			if tok < 0 || int(tok) >= vocab {
				return fmt.Errorf("token %d value %d is outside vocab_size %d", i, tok, vocab)
			}
		}
	}
	return nil
}

func (m *EmbeddingModel) validateEmbeddingResult(result EmbeddingResult, batched bool) error {
	if m.manifest.OutputName != "" && result.OutputName != m.manifest.OutputName {
		return fmt.Errorf("embedding output name %q does not match manifest %q", result.OutputName, m.manifest.OutputName)
	}
	if result.Embeddings == nil {
		return fmt.Errorf("embedding output tensor is nil")
	}
	if want := m.manifest.OutputDType; want != "" && result.Embeddings.DType != want {
		return fmt.Errorf("embedding output dtype %q does not match manifest %q", result.Embeddings.DType, want)
	}
	wantRank := 1
	if batched {
		wantRank = 2
	}
	if got := len(result.Embeddings.Shape); got != wantRank {
		return fmt.Errorf("embedding output rank %d does not match expected %d", got, wantRank)
	}
	return nil
}

func validateEmbeddingEntry(mod *barr.Module, entryName, tokenInput, maskInput string, tokenRank, outputRank int, outputDType string) error {
	entry, err := findEntryPoint(mod, entryName)
	if err != nil {
		return err
	}
	input, err := requireEntryInput(entry, tokenInput)
	if err != nil {
		return err
	}
	if input.Type.Kind != barr.ValueTensor || input.Type.Tensor == nil {
		return fmt.Errorf("entrypoint %q input %q is not a tensor", entryName, tokenInput)
	}
	if input.Type.Tensor.DType != "i32" {
		return fmt.Errorf("entrypoint %q input %q dtype = %q, want i32", entryName, tokenInput, input.Type.Tensor.DType)
	}
	if got := len(input.Type.Tensor.Shape); got != tokenRank {
		return fmt.Errorf("entrypoint %q input %q rank = %d, want %d", entryName, tokenInput, got, tokenRank)
	}
	if maskInput != "" {
		mask, err := requireEntryInput(entry, maskInput)
		if err != nil {
			return err
		}
		if mask.Type.Kind != barr.ValueTensor || mask.Type.Tensor == nil {
			return fmt.Errorf("entrypoint %q input %q is not a tensor", entryName, maskInput)
		}
		if mask.Type.Tensor.DType != "i32" {
			return fmt.Errorf("entrypoint %q input %q dtype = %q, want i32", entryName, maskInput, mask.Type.Tensor.DType)
		}
		if got := len(mask.Type.Tensor.Shape); got != tokenRank {
			return fmt.Errorf("entrypoint %q input %q rank = %d, want %d", entryName, maskInput, got, tokenRank)
		}
	}
	if len(entry.Outputs) != 1 {
		return fmt.Errorf("entrypoint %q output count = %d, want 1", entryName, len(entry.Outputs))
	}
	output := entry.Outputs[0]
	if output.Type.Kind != barr.ValueTensor || output.Type.Tensor == nil {
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

func validateEmbeddingParam(mod *barr.Module, name string) error {
	if name == "" {
		return nil
	}
	for _, param := range mod.Params {
		if param.Name == name {
			if param.Type.Kind != barr.ValueTensor || param.Type.Tensor == nil {
				return fmt.Errorf("param %q is not a tensor", name)
			}
			if got := len(param.Type.Tensor.Shape); got != 2 {
				return fmt.Errorf("param %q rank = %d, want 2", name, got)
			}
			return nil
		}
	}
	return fmt.Errorf("missing param %q", name)
}

func validateAttentionParams(mod *barr.Module, manifest EmbeddingManifest) error {
	names := []string{
		manifest.AttentionQueryParam,
		manifest.AttentionKeyParam,
		manifest.AttentionValueParam,
		manifest.AttentionOutputParam,
	}
	set := 0
	for _, name := range names {
		if name != "" {
			set++
		}
	}
	if set == 0 {
		return nil
	}
	if set != len(names) {
		return fmt.Errorf("attention params must declare query, key, value, and output together")
	}
	for _, name := range names {
		if err := validateEmbeddingParam(mod, name); err != nil {
			return err
		}
	}
	return nil
}

func findEntryPoint(mod *barr.Module, name string) (barr.EntryPoint, error) {
	if mod == nil {
		return barr.EntryPoint{}, fmt.Errorf("nil module")
	}
	for _, entry := range mod.EntryPoints {
		if entry.Name == name {
			return entry, nil
		}
	}
	return barr.EntryPoint{}, fmt.Errorf("unknown entrypoint %q", name)
}

func requireEntryInput(entry barr.EntryPoint, name string) (barr.ValueBinding, error) {
	for _, input := range entry.Inputs {
		if input.Name == name {
			return input, nil
		}
	}
	return barr.ValueBinding{}, fmt.Errorf("entrypoint %q does not declare input %q", entry.Name, name)
}

func buildMaskedTokenInputs(batches [][]int32, padID int32, batched bool) (*backend.Tensor, *backend.Tensor, error) {
	if len(batches) == 0 {
		return nil, nil, fmt.Errorf("token batches are empty")
	}
	maxLen := 0
	for i, batch := range batches {
		if len(batch) == 0 {
			return nil, nil, fmt.Errorf("token batch %d is empty", i)
		}
		if len(batch) > maxLen {
			maxLen = len(batch)
		}
	}
	tokenData := make([]int32, 0, len(batches)*maxLen)
	maskData := make([]int32, 0, len(batches)*maxLen)
	for _, batch := range batches {
		tokenData = append(tokenData, batch...)
		for range batch {
			maskData = append(maskData, 1)
		}
		for i := len(batch); i < maxLen; i++ {
			tokenData = append(tokenData, padID)
			maskData = append(maskData, 0)
		}
	}
	if !batched {
		return backend.NewTensorI32([]int{maxLen}, tokenData[:maxLen]), backend.NewTensorI32([]int{maxLen}, maskData[:maxLen]), nil
	}
	shape := []int{len(batches), maxLen}
	return backend.NewTensorI32(shape, tokenData), backend.NewTensorI32(shape, maskData), nil
}
