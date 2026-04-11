package barruntime

import (
	"context"
	"fmt"

	"github.com/odvcencio/manta/artifact/barr"
	"github.com/odvcencio/manta/runtime/backend"
)

// Candidate is one scored retrieval result.
type Candidate struct {
	ID       int64
	Score    float32
	Doc      *backend.Tensor
	Metadata map[string]string
}

// CandidateInput is one candidate row passed into a retrieval entrypoint.
type CandidateInput struct {
	ID       int64
	Doc      *backend.Tensor
	Metadata map[string]string
}

// CandidateBatch is one query worth of retrieved candidates.
type CandidateBatch struct {
	Candidates []Candidate
}

// CandidateResult is the Go-facing retrieval output for candidate_pack entrypoints.
type CandidateResult struct {
	OutputName string
	Batches    []CandidateBatch
	Raw        backend.Result
}

// RunCandidates executes an entrypoint and decodes its candidate_pack output.
func (p *Program) RunCandidates(ctx context.Context, req backend.Request) (CandidateResult, error) {
	if p == nil {
		return CandidateResult{}, fmt.Errorf("program is not loaded")
	}
	raw, err := p.Run(ctx, req)
	if err != nil {
		return CandidateResult{}, err
	}
	return decodeCandidateResult(raw, p.candidateMetadata)
}

// RunCandidateTable executes a packed-candidate retrieval entrypoint from Go-native candidate rows.
func (p *Program) RunCandidateTable(ctx context.Context, entry string, query *backend.Tensor, candidates []CandidateInput) (CandidateResult, error) {
	if p == nil {
		return CandidateResult{}, fmt.Errorf("program is not loaded")
	}
	entryPoint, err := p.requireCandidateEntry(entry, false)
	if err != nil {
		return CandidateResult{}, err
	}
	docs, ids, metadata, err := buildCandidateTable(candidates)
	if err != nil {
		return CandidateResult{}, err
	}
	queryName, docsName, idsName, err := candidateEntryInputNames(entryPoint)
	if err != nil {
		return CandidateResult{}, err
	}
	req := backend.Request{
		Entry: entry,
		Inputs: map[string]any{
			queryName: query,
			docsName:  docs,
			idsName:   ids,
		},
	}
	raw, err := p.Run(ctx, req)
	if err != nil {
		return CandidateResult{}, err
	}
	return decodeCandidateResult(raw, mergeCandidateMetadata(p.candidateMetadata, metadata))
}

// RunCandidateTableBatched executes a batched packed-candidate retrieval entrypoint from Go-native candidate rows.
func (p *Program) RunCandidateTableBatched(ctx context.Context, entry string, queries *backend.Tensor, batches [][]CandidateInput) (CandidateResult, error) {
	if p == nil {
		return CandidateResult{}, fmt.Errorf("program is not loaded")
	}
	entryPoint, err := p.requireCandidateEntry(entry, true)
	if err != nil {
		return CandidateResult{}, err
	}
	docs, ids, metadata, err := buildCandidateTableBatched(batches)
	if err != nil {
		return CandidateResult{}, err
	}
	queryName, docsName, idsName, err := candidateEntryInputNames(entryPoint)
	if err != nil {
		return CandidateResult{}, err
	}
	req := backend.Request{
		Entry: entry,
		Inputs: map[string]any{
			queryName: queries,
			docsName:  docs,
			idsName:   ids,
		},
	}
	raw, err := p.Run(ctx, req)
	if err != nil {
		return CandidateResult{}, err
	}
	return decodeCandidateResult(raw, mergeCandidateMetadata(p.candidateMetadata, metadata))
}

func decodeCandidateResult(raw backend.Result, lookup map[int64]map[string]string) (CandidateResult, error) {
	outputName := ""
	var output backend.Value
	for name, value := range raw.Outputs {
		if value.Type.Kind != barr.ValueCandidatePack {
			continue
		}
		if outputName != "" {
			return CandidateResult{}, fmt.Errorf("multiple candidate_pack outputs found: %q and %q", outputName, name)
		}
		outputName = name
		output = value
	}
	if outputName == "" {
		return CandidateResult{}, fmt.Errorf("no candidate_pack output found")
	}
	pack, ok := output.Data.(*backend.CandidatePack)
	if !ok || pack == nil {
		return CandidateResult{}, fmt.Errorf("candidate_pack output %q has invalid data type %T", outputName, output.Data)
	}
	batches, err := unpackCandidatePack(pack, lookup)
	if err != nil {
		return CandidateResult{}, fmt.Errorf("decode candidate_pack %q: %w", outputName, err)
	}
	return CandidateResult{
		OutputName: outputName,
		Batches:    batches,
		Raw:        raw,
	}, nil
}

func (p *Program) requireCandidateEntry(name string, batched bool) (barr.EntryPoint, error) {
	if p == nil || p.module == nil {
		return barr.EntryPoint{}, fmt.Errorf("program is not loaded")
	}
	for _, entry := range p.module.EntryPoints {
		if entry.Name != name {
			continue
		}
		if len(entry.Inputs) < 3 {
			return barr.EntryPoint{}, fmt.Errorf("entrypoint %q does not declare query/docs/candidate_ids inputs", name)
		}
		queryName, _, _, err := candidateEntryInputNames(entry)
		if err != nil {
			return barr.EntryPoint{}, err
		}
		if batched && queryName != "queries" {
			return barr.EntryPoint{}, fmt.Errorf("entrypoint %q is not batched", name)
		}
		if !batched && queryName != "query" {
			return barr.EntryPoint{}, fmt.Errorf("entrypoint %q is not unbatched", name)
		}
		return entry, nil
	}
	return barr.EntryPoint{}, fmt.Errorf("unknown entrypoint %q", name)
}

func candidateEntryInputNames(entry barr.EntryPoint) (queryName, docsName, idsName string, err error) {
	for _, input := range entry.Inputs {
		switch input.Name {
		case "query", "queries":
			queryName = input.Name
		case "docs":
			docsName = input.Name
		case "candidate_ids":
			idsName = input.Name
		}
	}
	if queryName == "" || docsName == "" || idsName == "" {
		return "", "", "", fmt.Errorf("entrypoint %q does not declare query/docs/candidate_ids inputs", entry.Name)
	}
	return queryName, docsName, idsName, nil
}

func unpackCandidatePack(pack *backend.CandidatePack, lookup map[int64]map[string]string) ([]CandidateBatch, error) {
	if pack == nil || pack.IDs == nil || pack.Scores == nil || pack.Docs == nil {
		return nil, fmt.Errorf("candidate pack is incomplete")
	}
	switch len(pack.Docs.Shape) {
	case 2:
		k := pack.Docs.Shape[0]
		if len(pack.IDs.Shape) != 1 || len(pack.Scores.Shape) != 1 || pack.IDs.Shape[0] != k || pack.Scores.Shape[0] != k {
			return nil, fmt.Errorf("candidate pack shape mismatch")
		}
		batch := CandidateBatch{Candidates: make([]Candidate, 0, k)}
		for i := 0; i < k; i++ {
			doc, err := sliceCandidateDoc(pack.Docs, 0, i)
			if err != nil {
				return nil, err
			}
			id := pack.IDs.I64[i]
			batch.Candidates = append(batch.Candidates, Candidate{
				ID:       id,
				Score:    pack.Scores.F32[i],
				Doc:      doc,
				Metadata: cloneCandidateMetadata(lookup[id]),
			})
		}
		return []CandidateBatch{batch}, nil
	case 3:
		q, k := pack.Docs.Shape[0], pack.Docs.Shape[1]
		if len(pack.IDs.Shape) != 2 || len(pack.Scores.Shape) != 2 || pack.IDs.Shape[0] != q || pack.Scores.Shape[0] != q || pack.IDs.Shape[1] != k || pack.Scores.Shape[1] != k {
			return nil, fmt.Errorf("candidate pack batched shape mismatch")
		}
		batches := make([]CandidateBatch, 0, q)
		for batchIdx := 0; batchIdx < q; batchIdx++ {
			batch := CandidateBatch{Candidates: make([]Candidate, 0, k)}
			for i := 0; i < k; i++ {
				doc, err := sliceCandidateDoc(pack.Docs, batchIdx, i)
				if err != nil {
					return nil, err
				}
				offset := batchIdx*k + i
				id := pack.IDs.I64[offset]
				batch.Candidates = append(batch.Candidates, Candidate{
					ID:       id,
					Score:    pack.Scores.F32[offset],
					Doc:      doc,
					Metadata: cloneCandidateMetadata(lookup[id]),
				})
			}
			batches = append(batches, batch)
		}
		return batches, nil
	default:
		return nil, fmt.Errorf("candidate pack docs rank %d is unsupported", len(pack.Docs.Shape))
	}
}

func buildCandidateTable(candidates []CandidateInput) (*backend.Tensor, *backend.Tensor, map[int64]map[string]string, error) {
	if len(candidates) == 0 {
		return nil, nil, nil, fmt.Errorf("candidate table is empty")
	}
	first := candidates[0].Doc
	if first == nil {
		return nil, nil, nil, fmt.Errorf("candidate 0 doc is nil")
	}
	if len(first.Shape) != 1 {
		return nil, nil, nil, fmt.Errorf("candidate docs must be rank-1 tensors")
	}
	n, d := len(candidates), first.Shape[0]
	ids := make([]int64, n)
	docs := make([]float32, 0, n*d)
	metadata := map[int64]map[string]string{}
	for i, candidate := range candidates {
		if candidate.Doc == nil {
			return nil, nil, nil, fmt.Errorf("candidate %d doc is nil", i)
		}
		if candidate.Doc.DType != first.DType {
			return nil, nil, nil, fmt.Errorf("candidate %d doc dtype %q does not match %q", i, candidate.Doc.DType, first.DType)
		}
		if len(candidate.Doc.Shape) != 1 || candidate.Doc.Shape[0] != d {
			return nil, nil, nil, fmt.Errorf("candidate %d doc shape %v does not match [%d]", i, candidate.Doc.Shape, d)
		}
		ids[i] = candidate.ID
		docs = append(docs, candidate.Doc.F32...)
		if len(candidate.Metadata) > 0 {
			metadata[candidate.ID] = cloneCandidateMetadata(candidate.Metadata)
		}
	}
	return newTensorForDType(first.DType, []int{n, d}, docs), backend.NewTensorI64([]int{n}, ids), metadata, nil
}

func buildCandidateTableBatched(batches [][]CandidateInput) (*backend.Tensor, *backend.Tensor, map[int64]map[string]string, error) {
	if len(batches) == 0 {
		return nil, nil, nil, fmt.Errorf("candidate batches are empty")
	}
	if len(batches[0]) == 0 {
		return nil, nil, nil, fmt.Errorf("candidate batch 0 is empty")
	}
	first := batches[0][0].Doc
	if first == nil {
		return nil, nil, nil, fmt.Errorf("candidate batch 0 row 0 doc is nil")
	}
	if len(first.Shape) != 1 {
		return nil, nil, nil, fmt.Errorf("candidate docs must be rank-1 tensors")
	}
	q, n, d := len(batches), len(batches[0]), first.Shape[0]
	ids := make([]int64, 0, q*n)
	docs := make([]float32, 0, q*n*d)
	metadata := map[int64]map[string]string{}
	for batchIdx, batch := range batches {
		if len(batch) != n {
			return nil, nil, nil, fmt.Errorf("candidate batch %d size %d does not match %d", batchIdx, len(batch), n)
		}
		for rowIdx, candidate := range batch {
			if candidate.Doc == nil {
				return nil, nil, nil, fmt.Errorf("candidate batch %d row %d doc is nil", batchIdx, rowIdx)
			}
			if candidate.Doc.DType != first.DType {
				return nil, nil, nil, fmt.Errorf("candidate batch %d row %d doc dtype %q does not match %q", batchIdx, rowIdx, candidate.Doc.DType, first.DType)
			}
			if len(candidate.Doc.Shape) != 1 || candidate.Doc.Shape[0] != d {
				return nil, nil, nil, fmt.Errorf("candidate batch %d row %d doc shape %v does not match [%d]", batchIdx, rowIdx, candidate.Doc.Shape, d)
			}
			ids = append(ids, candidate.ID)
			docs = append(docs, candidate.Doc.F32...)
			if len(candidate.Metadata) > 0 {
				metadata[candidate.ID] = cloneCandidateMetadata(candidate.Metadata)
			}
		}
	}
	return newTensorForDType(first.DType, []int{q, n, d}, docs), backend.NewTensorI64([]int{q, n}, ids), metadata, nil
}

func newTensorForDType(dtype string, shape []int, data []float32) *backend.Tensor {
	switch dtype {
	case "f32":
		return backend.NewTensorF32(shape, data)
	case "f16":
		return backend.NewTensorF16(shape, data)
	case "q4":
		return backend.NewTensorQ4(shape, data)
	case "q8":
		return backend.NewTensorQ8(shape, data)
	default:
		return &backend.Tensor{DType: dtype, Shape: append([]int(nil), shape...), F32: append([]float32(nil), data...)}
	}
}

func mergeCandidateMetadata(base, extra map[int64]map[string]string) map[int64]map[string]string {
	if len(base) == 0 && len(extra) == 0 {
		return nil
	}
	out := cloneCandidateMetadataLookup(base)
	if out == nil {
		out = map[int64]map[string]string{}
	}
	for id, meta := range extra {
		merged := cloneCandidateMetadata(out[id])
		if merged == nil {
			merged = map[string]string{}
		}
		for k, v := range meta {
			merged[k] = v
		}
		out[id] = merged
	}
	return out
}

func sliceCandidateDoc(docs *backend.Tensor, batch, row int) (*backend.Tensor, error) {
	if docs == nil {
		return nil, fmt.Errorf("docs tensor is nil")
	}
	switch len(docs.Shape) {
	case 2:
		k, d := docs.Shape[0], docs.Shape[1]
		if row < 0 || row >= k {
			return nil, fmt.Errorf("candidate row %d out of range", row)
		}
		return cloneTensorRow(docs, row*d, d)
	case 3:
		q, k, d := docs.Shape[0], docs.Shape[1], docs.Shape[2]
		if batch < 0 || batch >= q || row < 0 || row >= k {
			return nil, fmt.Errorf("candidate batch/row %d/%d out of range", batch, row)
		}
		return cloneTensorRow(docs, (batch*k+row)*d, d)
	default:
		return nil, fmt.Errorf("docs tensor rank %d is unsupported", len(docs.Shape))
	}
}

func cloneTensorRow(t *backend.Tensor, start, width int) (*backend.Tensor, error) {
	switch t.DType {
	case "f32":
		return backend.NewTensorF32([]int{width}, append([]float32(nil), t.F32[start:start+width]...)), nil
	case "f16":
		return backend.NewTensorF16([]int{width}, append([]float32(nil), t.F32[start:start+width]...)), nil
	case "q4":
		return backend.NewTensorQ4([]int{width}, append([]float32(nil), t.F32[start:start+width]...)), nil
	case "q8":
		return backend.NewTensorQ8([]int{width}, append([]float32(nil), t.F32[start:start+width]...)), nil
	case "i32":
		return backend.NewTensorI32([]int{width}, append([]int32(nil), t.I32[start:start+width]...)), nil
	case "i64":
		return backend.NewTensorI64([]int{width}, append([]int64(nil), t.I64[start:start+width]...)), nil
	default:
		return nil, fmt.Errorf("unsupported doc dtype %q", t.DType)
	}
}
