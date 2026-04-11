package barruntime

import (
	"context"
	"strings"
	"testing"

	"github.com/odvcencio/barracuda/compiler"
	"github.com/odvcencio/barracuda/runtime/backend"
	"github.com/odvcencio/barracuda/runtime/backends/cuda"
	"github.com/odvcencio/barracuda/runtime/backends/metal"
)

func TestRunCandidatesDecodesPackedOutput(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_packed_candidates", Preset: compiler.PresetTinyPackedCandidates})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	rt := New(cuda.New(), metal.New())
	prog, err := rt.Load(
		context.Background(),
		bundle.Artifact,
		WithCandidateMetadata(map[int64]map[string]string{
			1001: {"label": "alpha"},
			3003: {"label": "gamma"},
		}),
	)
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	result, err := prog.RunCandidates(context.Background(), backend.Request{
		Entry: "rerank_candidates_packed",
		Inputs: map[string]any{
			"query":         backend.NewTensorF16([]int{2}, []float32{1, 0}),
			"docs":          backend.NewTensorQ4([]int{3, 2}, []float32{1, 0, 0, 1, 1, 1}),
			"candidate_ids": backend.NewTensorI64([]int{3}, []int64{1001, 2002, 3003}),
		},
	})
	if err != nil {
		t.Fatalf("run candidates: %v", err)
	}
	if result.OutputName != "candidates" {
		t.Fatalf("output name = %q, want candidates", result.OutputName)
	}
	if len(result.Batches) != 1 {
		t.Fatalf("batch count = %d, want 1", len(result.Batches))
	}
	if len(result.Batches[0].Candidates) != 2 {
		t.Fatalf("candidate count = %d, want 2", len(result.Batches[0].Candidates))
	}
	first := result.Batches[0].Candidates[0]
	second := result.Batches[0].Candidates[1]
	if first.ID != 1001 || second.ID != 3003 {
		t.Fatalf("candidate ids = %d,%d, want 1001,3003", first.ID, second.ID)
	}
	assertClose(t, first.Score, 1, 0.0005)
	assertClose(t, second.Score, 0.70710677, 0.0005)
	assertTensorClose(t, first.Doc, []int{2}, []float32{1, 0})
	assertTensorClose(t, second.Doc, []int{2}, []float32{1, 1})
	if first.Metadata["label"] != "alpha" {
		t.Fatalf("first candidate metadata = %+v, want alpha", first.Metadata)
	}
	if second.Metadata["label"] != "gamma" {
		t.Fatalf("second candidate metadata = %+v, want gamma", second.Metadata)
	}
}

func TestRunCandidatesDecodesBatchedPackedOutput(t *testing.T) {
	src := []byte(`
pipeline rerank_candidates_packed_batch(queries: f16[Q, D], docs: q4[Q, N, D], candidate_ids: i64[Q, N]) -> candidate_pack[Q, 2, D] {
    let scores = cosine(queries, docs)
    let top_indices = topk(scores, 2)
    let top_candidate_ids = gather(candidate_ids, top_indices)
    let top_scores = gather(scores, top_indices)
    let top_docs = gather(docs, top_indices)
    return pack_candidates(top_candidate_ids, top_scores, top_docs)
}
`)

	bundle, err := compiler.Build(src, compiler.Options{ModuleName: "packed_batch"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	rt := New(cuda.New(), metal.New())
	prog, err := rt.Load(
		context.Background(),
		bundle.Artifact,
		WithCandidateMetadata(map[int64]map[string]string{
			1001: {"label": "a0"},
			3003: {"label": "a2"},
			4004: {"label": "b0"},
			6006: {"label": "b2"},
		}),
	)
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	result, err := prog.RunCandidates(context.Background(), backend.Request{
		Entry: "rerank_candidates_packed_batch",
		Inputs: map[string]any{
			"queries": backend.NewTensorF16([]int{2, 2}, []float32{
				1, 0,
				0, 1,
			}),
			"docs": backend.NewTensorQ4([]int{2, 3, 2}, []float32{
				1, 0,
				0, 1,
				1, 1,
				0, 1,
				1, 0,
				1, 1,
			}),
			"candidate_ids": backend.NewTensorI64([]int{2, 3}, []int64{
				1001, 2002, 3003,
				4004, 5005, 6006,
			}),
		},
	})
	if err != nil {
		t.Fatalf("run candidates: %v", err)
	}
	if result.OutputName != "result" {
		t.Fatalf("output name = %q, want result", result.OutputName)
	}
	if len(result.Batches) != 2 {
		t.Fatalf("batch count = %d, want 2", len(result.Batches))
	}
	if result.Batches[0].Candidates[0].Metadata["label"] != "a0" || result.Batches[1].Candidates[1].Metadata["label"] != "b2" {
		t.Fatalf("unexpected metadata decode: %+v", result.Batches)
	}
	assertTensorClose(t, result.Batches[0].Candidates[0].Doc, []int{2}, []float32{1, 0})
	assertTensorClose(t, result.Batches[1].Candidates[0].Doc, []int{2}, []float32{0, 1})
}

func TestRunCandidatesRejectsNonCandidateOutput(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_retrieve", Preset: compiler.PresetTinyRetrieve})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	rt := New(cuda.New(), metal.New())
	prog, err := rt.Load(context.Background(), bundle.Artifact, tinyRerankWeights()...)
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	_, err = prog.RunCandidates(context.Background(), backend.Request{
		Entry:  "retrieve",
		Inputs: map[string]any{"query": backend.NewTensorF16([]int{2}, []float32{1, 0})},
	})
	if err == nil {
		t.Fatal("expected missing candidate_pack error")
	}
	if !strings.Contains(err.Error(), "no candidate_pack output found") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestRunCandidateTableBuildsInputsAndDecodesOutput(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_packed_candidates", Preset: compiler.PresetTinyPackedCandidates})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	rt := New(cuda.New(), metal.New())
	prog, err := rt.Load(context.Background(), bundle.Artifact, WithCandidateMetadata(map[int64]map[string]string{
		1001: {"source": "runtime"},
	}))
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	result, err := prog.RunCandidateTable(context.Background(), "rerank_candidates_packed", backend.NewTensorF16([]int{2}, []float32{1, 0}), []CandidateInput{
		{ID: 1001, Doc: backend.NewTensorQ4([]int{2}, []float32{1, 0}), Metadata: map[string]string{"label": "alpha"}},
		{ID: 2002, Doc: backend.NewTensorQ4([]int{2}, []float32{0, 1})},
		{ID: 3003, Doc: backend.NewTensorQ4([]int{2}, []float32{1, 1}), Metadata: map[string]string{"label": "gamma"}},
	})
	if err != nil {
		t.Fatalf("run candidate table: %v", err)
	}

	if len(result.Batches) != 1 || len(result.Batches[0].Candidates) != 2 {
		t.Fatalf("unexpected candidate batches: %+v", result.Batches)
	}
	first := result.Batches[0].Candidates[0]
	if first.Metadata["source"] != "runtime" || first.Metadata["label"] != "alpha" {
		t.Fatalf("merged metadata = %+v", first.Metadata)
	}
}

func TestRunCandidateTableBatchedBuildsInputsAndDecodesOutput(t *testing.T) {
	src := []byte(`
pipeline rerank_candidates_packed_batch(queries: f16[Q, D], docs: q4[Q, N, D], candidate_ids: i64[Q, N]) -> candidate_pack[Q, 2, D] {
    let scores = cosine(queries, docs)
    let top_indices = topk(scores, 2)
    let top_candidate_ids = gather(candidate_ids, top_indices)
    let top_scores = gather(scores, top_indices)
    let top_docs = gather(docs, top_indices)
    return pack_candidates(top_candidate_ids, top_scores, top_docs)
}
`)

	bundle, err := compiler.Build(src, compiler.Options{ModuleName: "packed_batch"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	rt := New(cuda.New(), metal.New())
	prog, err := rt.Load(context.Background(), bundle.Artifact)
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	result, err := prog.RunCandidateTableBatched(context.Background(), "rerank_candidates_packed_batch", backend.NewTensorF16([]int{2, 2}, []float32{
		1, 0,
		0, 1,
	}), [][]CandidateInput{
		{
			{ID: 1001, Doc: backend.NewTensorQ4([]int{2}, []float32{1, 0})},
			{ID: 2002, Doc: backend.NewTensorQ4([]int{2}, []float32{0, 1})},
			{ID: 3003, Doc: backend.NewTensorQ4([]int{2}, []float32{1, 1})},
		},
		{
			{ID: 4004, Doc: backend.NewTensorQ4([]int{2}, []float32{0, 1})},
			{ID: 5005, Doc: backend.NewTensorQ4([]int{2}, []float32{1, 0})},
			{ID: 6006, Doc: backend.NewTensorQ4([]int{2}, []float32{1, 1})},
		},
	})
	if err != nil {
		t.Fatalf("run batched candidate table: %v", err)
	}

	if len(result.Batches) != 2 {
		t.Fatalf("batch count = %d, want 2", len(result.Batches))
	}
	if got := result.Batches[1].Candidates[0].ID; got != 4004 {
		t.Fatalf("batch 1 candidate 0 id = %d, want 4004", got)
	}
}

func TestRunCandidateTableRejectsMismatchedDocShape(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_packed_candidates", Preset: compiler.PresetTinyPackedCandidates})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	rt := New(cuda.New(), metal.New())
	prog, err := rt.Load(context.Background(), bundle.Artifact)
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	_, err = prog.RunCandidateTable(context.Background(), "rerank_candidates_packed", backend.NewTensorF16([]int{2}, []float32{1, 0}), []CandidateInput{
		{ID: 1001, Doc: backend.NewTensorQ4([]int{2}, []float32{1, 0})},
		{ID: 2002, Doc: backend.NewTensorQ4([]int{3}, []float32{0, 1, 2})},
	})
	if err == nil {
		t.Fatal("expected mismatched doc shape error")
	}
	if !strings.Contains(err.Error(), "does not match") {
		t.Fatalf("unexpected error: %v", err)
	}
}
