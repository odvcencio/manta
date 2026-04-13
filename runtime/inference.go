package mantaruntime

import (
	"context"
	"fmt"

	mantaartifact "github.com/odvcencio/manta/artifact/manta"
	"github.com/odvcencio/manta/runtime/backend"
)

// EmbeddingResult is the Go-facing embedding output.
type EmbeddingResult struct {
	OutputName string
	Embeddings *backend.Tensor
	Raw        backend.Result
}

// ScoreResult is the Go-facing score output.
type ScoreResult struct {
	OutputName string
	Scores     *backend.Tensor
	Raw        backend.Result
}

// RunEmbed executes an embedding entrypoint from token ids.
func (p *Program) RunEmbed(ctx context.Context, entry string, tokens []int32) (EmbeddingResult, error) {
	if p == nil {
		return EmbeddingResult{}, fmt.Errorf("program is not loaded")
	}
	entryPoint, err := p.requireNamedEntryInput(entry, "tokens")
	if err != nil {
		return EmbeddingResult{}, err
	}
	if len(tokens) == 0 {
		return EmbeddingResult{}, fmt.Errorf("tokens are empty")
	}
	tokenName := "tokens"
	for _, input := range entryPoint.Inputs {
		if input.Name == "tokens" {
			tokenName = input.Name
			break
		}
	}
	raw, err := p.Run(ctx, backend.Request{
		Entry: entry,
		Inputs: map[string]any{
			tokenName: backend.NewTensorI32([]int{len(tokens)}, tokens),
		},
	})
	if err != nil {
		return EmbeddingResult{}, err
	}
	return decodeEmbeddingResult(raw)
}

// RunEmbedBatch executes an embedding entrypoint from rectangular token batches.
func (p *Program) RunEmbedBatch(ctx context.Context, entry string, tokenBatches [][]int32) (EmbeddingResult, error) {
	if p == nil {
		return EmbeddingResult{}, fmt.Errorf("program is not loaded")
	}
	entryPoint, err := p.requireNamedEntryInput(entry, "tokens")
	if err != nil {
		return EmbeddingResult{}, err
	}
	tokenTensor, err := buildTokenBatch(tokenBatches)
	if err != nil {
		return EmbeddingResult{}, err
	}
	tokenName := "tokens"
	for _, input := range entryPoint.Inputs {
		if input.Name == "tokens" {
			tokenName = input.Name
			break
		}
	}
	raw, err := p.Run(ctx, backend.Request{
		Entry: entry,
		Inputs: map[string]any{
			tokenName: tokenTensor,
		},
	})
	if err != nil {
		return EmbeddingResult{}, err
	}
	return decodeEmbeddingResult(raw)
}

// RunScore executes an entrypoint and decodes its score tensor output.
func (p *Program) RunScore(ctx context.Context, req backend.Request) (ScoreResult, error) {
	if p == nil {
		return ScoreResult{}, fmt.Errorf("program is not loaded")
	}
	raw, err := p.Run(ctx, req)
	if err != nil {
		return ScoreResult{}, err
	}
	return decodeScoreResult(raw)
}

// RunScoreTable executes a score entrypoint from a query tensor and Go-native doc rows.
func (p *Program) RunScoreTable(ctx context.Context, entry string, query *backend.Tensor, docs []*backend.Tensor) (ScoreResult, error) {
	if p == nil {
		return ScoreResult{}, fmt.Errorf("program is not loaded")
	}
	entryPoint, err := p.requireScoringEntry(entry, false)
	if err != nil {
		return ScoreResult{}, err
	}
	docTable, err := buildDocsTable(docs)
	if err != nil {
		return ScoreResult{}, err
	}
	queryName, docsName, err := scoreEntryInputNames(entryPoint)
	if err != nil {
		return ScoreResult{}, err
	}
	return p.RunScore(ctx, backend.Request{
		Entry: entry,
		Inputs: map[string]any{
			queryName: query,
			docsName:  docTable,
		},
	})
}

// RunScoreTableBatched executes a batched score entrypoint from Go-native doc rows.
func (p *Program) RunScoreTableBatched(ctx context.Context, entry string, queries *backend.Tensor, batches [][]*backend.Tensor) (ScoreResult, error) {
	if p == nil {
		return ScoreResult{}, fmt.Errorf("program is not loaded")
	}
	entryPoint, err := p.requireScoringEntry(entry, true)
	if err != nil {
		return ScoreResult{}, err
	}
	docTable, err := buildDocsTableBatched(batches)
	if err != nil {
		return ScoreResult{}, err
	}
	queryName, docsName, err := scoreEntryInputNames(entryPoint)
	if err != nil {
		return ScoreResult{}, err
	}
	return p.RunScore(ctx, backend.Request{
		Entry: entry,
		Inputs: map[string]any{
			queryName: queries,
			docsName:  docTable,
		},
	})
}

func decodeEmbeddingResult(raw backend.Result) (EmbeddingResult, error) {
	name, tensor, err := decodeSingleTensorOutput(raw, "embeddings")
	if err != nil {
		return EmbeddingResult{}, err
	}
	return EmbeddingResult{
		OutputName: name,
		Embeddings: tensor,
		Raw:        raw,
	}, nil
}

func decodeScoreResult(raw backend.Result) (ScoreResult, error) {
	name, tensor, err := decodeSingleTensorOutput(raw, "scores")
	if err != nil {
		return ScoreResult{}, err
	}
	return ScoreResult{
		OutputName: name,
		Scores:     tensor,
		Raw:        raw,
	}, nil
}

func decodeSingleTensorOutput(raw backend.Result, preferred string) (string, *backend.Tensor, error) {
	if preferred != "" {
		if value, ok := raw.Outputs[preferred]; ok {
			tensor, ok := value.Data.(*backend.Tensor)
			if !ok || tensor == nil {
				return "", nil, fmt.Errorf("output %q has invalid tensor data type %T", preferred, value.Data)
			}
			return preferred, tensor, nil
		}
	}
	outputName := ""
	var output backend.Value
	for name, value := range raw.Outputs {
		if value.Type.Kind != mantaartifact.ValueTensor {
			continue
		}
		if outputName != "" {
			return "", nil, fmt.Errorf("multiple tensor outputs found: %q and %q", outputName, name)
		}
		outputName = name
		output = value
	}
	if outputName == "" {
		return "", nil, fmt.Errorf("no tensor output found")
	}
	tensor, ok := output.Data.(*backend.Tensor)
	if !ok || tensor == nil {
		return "", nil, fmt.Errorf("output %q has invalid tensor data type %T", outputName, output.Data)
	}
	return outputName, tensor, nil
}

func (p *Program) requireNamedEntryInput(name, inputName string) (mantaartifact.EntryPoint, error) {
	if p == nil || p.module == nil {
		return mantaartifact.EntryPoint{}, fmt.Errorf("program is not loaded")
	}
	for _, entry := range p.module.EntryPoints {
		if entry.Name != name {
			continue
		}
		for _, input := range entry.Inputs {
			if input.Name == inputName {
				return entry, nil
			}
		}
		return mantaartifact.EntryPoint{}, fmt.Errorf("entrypoint %q does not declare input %q", name, inputName)
	}
	return mantaartifact.EntryPoint{}, fmt.Errorf("unknown entrypoint %q", name)
}

func (p *Program) requireScoringEntry(name string, batched bool) (mantaartifact.EntryPoint, error) {
	entry, err := p.requireNamedEntryInput(name, "docs")
	if err != nil {
		return mantaartifact.EntryPoint{}, err
	}
	queryName, _, err := scoreEntryInputNames(entry)
	if err != nil {
		return mantaartifact.EntryPoint{}, err
	}
	if batched && queryName != "queries" {
		return mantaartifact.EntryPoint{}, fmt.Errorf("entrypoint %q is not batched", name)
	}
	if !batched && queryName != "query" {
		return mantaartifact.EntryPoint{}, fmt.Errorf("entrypoint %q is not unbatched", name)
	}
	return entry, nil
}

func scoreEntryInputNames(entry mantaartifact.EntryPoint) (queryName, docsName string, err error) {
	for _, input := range entry.Inputs {
		switch input.Name {
		case "query", "queries":
			queryName = input.Name
		case "docs":
			docsName = input.Name
		}
	}
	if queryName == "" || docsName == "" {
		return "", "", fmt.Errorf("entrypoint %q does not declare query/docs inputs", entry.Name)
	}
	return queryName, docsName, nil
}

func buildDocsTable(docs []*backend.Tensor) (*backend.Tensor, error) {
	if len(docs) == 0 {
		return nil, fmt.Errorf("docs table is empty")
	}
	first := docs[0]
	if first == nil {
		return nil, fmt.Errorf("doc 0 is nil")
	}
	if len(first.Shape) != 1 {
		return nil, fmt.Errorf("docs must be rank-1 tensors")
	}
	n, d := len(docs), first.Shape[0]
	data := make([]float32, 0, n*d)
	for i, doc := range docs {
		if doc == nil {
			return nil, fmt.Errorf("doc %d is nil", i)
		}
		if doc.DType != first.DType {
			return nil, fmt.Errorf("doc %d dtype %q does not match %q", i, doc.DType, first.DType)
		}
		if len(doc.Shape) != 1 || doc.Shape[0] != d {
			return nil, fmt.Errorf("doc %d shape %v does not match [%d]", i, doc.Shape, d)
		}
		data = append(data, doc.F32...)
	}
	return newTensorForDType(first.DType, []int{n, d}, data), nil
}

func buildDocsTableBatched(batches [][]*backend.Tensor) (*backend.Tensor, error) {
	if len(batches) == 0 {
		return nil, fmt.Errorf("docs batches are empty")
	}
	if len(batches[0]) == 0 {
		return nil, fmt.Errorf("docs batch 0 is empty")
	}
	first := batches[0][0]
	if first == nil {
		return nil, fmt.Errorf("docs batch 0 row 0 is nil")
	}
	if len(first.Shape) != 1 {
		return nil, fmt.Errorf("docs must be rank-1 tensors")
	}
	q, n, d := len(batches), len(batches[0]), first.Shape[0]
	data := make([]float32, 0, q*n*d)
	for batchIdx, batch := range batches {
		if len(batch) != n {
			return nil, fmt.Errorf("docs batch %d size %d does not match %d", batchIdx, len(batch), n)
		}
		for rowIdx, doc := range batch {
			if doc == nil {
				return nil, fmt.Errorf("docs batch %d row %d is nil", batchIdx, rowIdx)
			}
			if doc.DType != first.DType {
				return nil, fmt.Errorf("docs batch %d row %d dtype %q does not match %q", batchIdx, rowIdx, doc.DType, first.DType)
			}
			if len(doc.Shape) != 1 || doc.Shape[0] != d {
				return nil, fmt.Errorf("docs batch %d row %d shape %v does not match [%d]", batchIdx, rowIdx, doc.Shape, d)
			}
			data = append(data, doc.F32...)
		}
	}
	return newTensorForDType(first.DType, []int{q, n, d}, data), nil
}

func buildTokenBatch(batches [][]int32) (*backend.Tensor, error) {
	if len(batches) == 0 {
		return nil, fmt.Errorf("token batches are empty")
	}
	if len(batches[0]) == 0 {
		return nil, fmt.Errorf("token batch 0 is empty")
	}
	b, t := len(batches), len(batches[0])
	data := make([]int32, 0, b*t)
	for batchIdx, batch := range batches {
		if len(batch) != t {
			return nil, fmt.Errorf("token batch %d size %d does not match %d", batchIdx, len(batch), t)
		}
		data = append(data, batch...)
	}
	return backend.NewTensorI32([]int{b, t}, data), nil
}
