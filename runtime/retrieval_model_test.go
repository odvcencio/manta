package mantaruntime

import (
	"context"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	mantaartifact "github.com/odvcencio/manta/artifact/manta"
	"github.com/odvcencio/manta/compiler"
	"github.com/odvcencio/manta/runtime/backend"
	"github.com/odvcencio/manta/runtime/backends/cuda"
	"github.com/odvcencio/manta/runtime/backends/metal"
)

func TestRetrievalManifestRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "tiny_retrieval.retrieval.mll")
	want := tinyRetrievalManifest()
	if err := want.WriteFile(path); err != nil {
		t.Fatalf("write manifest: %v", err)
	}
	got, err := ReadRetrievalManifestFile(path)
	if err != nil {
		t.Fatalf("read manifest: %v", err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("manifest mismatch:\nwant: %+v\ngot:  %+v", want, got)
	}
}

func TestDefaultRetrievalManifestPath(t *testing.T) {
	got := DefaultRetrievalManifestPath("/tmp/tiny_retrieval.mll")
	if want := "/tmp/tiny_retrieval.retrieval.mll"; got != want {
		t.Fatalf("manifest path = %q, want %q", got, want)
	}
}

func TestLoadRetrievalValidatesManifest(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_score", Preset: compiler.PresetTinyScore})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	rt := New(cuda.New(), metal.New())
	_, err = rt.LoadRetrieval(context.Background(), bundle.Artifact, tinyRetrievalManifest(), tinyScoreWeights()...)
	if err == nil {
		t.Fatal("expected retrieval manifest validation error")
	}
	if !strings.Contains(err.Error(), `unknown entrypoint "rerank_candidates_packed"`) {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestRetrievalModelRetrieve(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_packed_candidates", Preset: compiler.PresetTinyPackedCandidates})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	rt := New(cuda.New(), metal.New())
	model, err := rt.LoadRetrieval(
		context.Background(),
		bundle.Artifact,
		tinyRetrievalManifest(),
		WithCandidateMetadata(map[int64]map[string]string{
			1001: {"source": "runtime"},
		}),
	)
	if err != nil {
		t.Fatalf("load retrieval: %v", err)
	}

	result, err := model.Retrieve(context.Background(), backend.NewTensorF16([]int{2}, []float32{1, 0}), []CandidateInput{
		{ID: 1001, Doc: backend.NewTensorQ4([]int{2}, []float32{1, 0}), Metadata: map[string]string{"label": "alpha"}},
		{ID: 2002, Doc: backend.NewTensorQ4([]int{2}, []float32{0, 1})},
		{ID: 3003, Doc: backend.NewTensorQ4([]int{2}, []float32{1, 1}), Metadata: map[string]string{"label": "gamma"}},
	})
	if err != nil {
		t.Fatalf("retrieve: %v", err)
	}
	if got := model.Backend(); got == "" {
		t.Fatal("expected selected backend")
	}
	if len(result.Batches) != 1 || len(result.Batches[0].Candidates) != 2 {
		t.Fatalf("unexpected retrieval result: %+v", result.Batches)
	}
	first := result.Batches[0].Candidates[0]
	if first.Metadata["source"] != "runtime" || first.Metadata["label"] != "alpha" {
		t.Fatalf("merged metadata = %+v", first.Metadata)
	}
}

func TestRetrievalModelRetrieveBatch(t *testing.T) {
	src := []byte(`
pipeline rerank_candidates_packed(query: f16[D], docs: q4[N, D], candidate_ids: i64[N]) -> candidate_pack[2, D] {
    let scores = cosine(query, docs)
    let top_indices = topk(scores, 2)
    let top_candidate_ids = gather(candidate_ids, top_indices)
    let top_scores = gather(scores, top_indices)
    let top_docs = gather(docs, top_indices)
    return pack_candidates(top_candidate_ids, top_scores, top_docs)
}

pipeline rerank_candidates_packed_batch(queries: f16[Q, D], docs: q4[Q, N, D], candidate_ids: i64[Q, N]) -> candidate_pack[Q, 2, D] {
    let scores = cosine(queries, docs)
    let top_indices = topk(scores, 2)
    let top_candidate_ids = gather(candidate_ids, top_indices)
    let top_scores = gather(scores, top_indices)
    let top_docs = gather(docs, top_indices)
    return pack_candidates(top_candidate_ids, top_scores, top_docs)
}
`)

	bundle, err := compiler.Build(src, compiler.Options{ModuleName: "retrieval_bundle"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	rt := New(cuda.New(), metal.New())
	model, err := rt.LoadRetrieval(context.Background(), bundle.Artifact, tinyRetrievalBatchManifest())
	if err != nil {
		t.Fatalf("load retrieval: %v", err)
	}

	result, err := model.RetrieveBatch(context.Background(), backend.NewTensorF16([]int{2, 2}, []float32{
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
		t.Fatalf("retrieve batch: %v", err)
	}
	if len(result.Batches) != 2 {
		t.Fatalf("batch count = %d, want 2", len(result.Batches))
	}
	if got := result.Batches[1].Candidates[0].ID; got != 4004 {
		t.Fatalf("batch 1 candidate 0 id = %d, want 4004", got)
	}
}

func TestLoadRetrievalBundleUsesSiblingManifest(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_packed_candidates", Preset: compiler.PresetTinyPackedCandidates})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	dir := t.TempDir()
	artifactPath := filepath.Join(dir, "tiny_packed_candidates.mll")
	if err := mantaartifact.WriteFile(artifactPath, bundle.Artifact); err != nil {
		t.Fatalf("write artifact: %v", err)
	}
	if err := tinyRetrievalManifest().WriteFile(DefaultRetrievalManifestPath(artifactPath)); err != nil {
		t.Fatalf("write manifest: %v", err)
	}

	rt := New(cuda.New(), metal.New())
	model, err := rt.LoadRetrievalBundle(context.Background(), artifactPath)
	if err != nil {
		t.Fatalf("load retrieval bundle: %v", err)
	}
	result, err := model.Retrieve(context.Background(), backend.NewTensorF16([]int{2}, []float32{1, 0}), []CandidateInput{
		{ID: 1001, Doc: backend.NewTensorQ4([]int{2}, []float32{1, 0})},
		{ID: 2002, Doc: backend.NewTensorQ4([]int{2}, []float32{0, 1})},
		{ID: 3003, Doc: backend.NewTensorQ4([]int{2}, []float32{1, 1})},
	})
	if err != nil {
		t.Fatalf("retrieve from bundle: %v", err)
	}
	if len(result.Batches) != 1 || len(result.Batches[0].Candidates) != 2 {
		t.Fatalf("unexpected retrieval result: %+v", result.Batches)
	}
}

func tinyRetrievalManifest() RetrievalManifest {
	return RetrievalManifest{
		Name:              "tiny_packed_candidates",
		Entry:             "rerank_candidates_packed",
		QueryInput:        "query",
		DocsInput:         "docs",
		CandidateIDsInput: "candidate_ids",
		OutputName:        "candidates",
		QueryDType:        "f16",
		DocsDType:         "q4",
		CandidateIDsDType: "i64",
	}
}

func tinyRetrievalBatchManifest() RetrievalManifest {
	manifest := tinyRetrievalManifest()
	manifest.Name = "retrieval_bundle"
	manifest.BatchEntry = "rerank_candidates_packed_batch"
	manifest.BatchOutputName = "result"
	return manifest
}
