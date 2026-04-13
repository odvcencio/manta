package mantaruntime

import (
	"context"
	"strings"
	"testing"

	"github.com/odvcencio/manta/compiler"
	"github.com/odvcencio/manta/runtime/backend"
	"github.com/odvcencio/manta/runtime/backends/cuda"
	"github.com/odvcencio/manta/runtime/backends/metal"
)

func TestRunEmbedDecodesOutput(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_embed", Preset: compiler.PresetTinyEmbed})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	rt := New(cuda.New(), metal.New())
	prog, err := rt.Load(context.Background(), bundle.Artifact, tinyEmbedWeights()...)
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	result, err := prog.RunEmbed(context.Background(), "embed", []int32{0, 2})
	if err != nil {
		t.Fatalf("run embed: %v", err)
	}
	if result.OutputName != "embeddings" {
		t.Fatalf("output name = %q, want embeddings", result.OutputName)
	}
	assertTensorClose(t, result.Embeddings, []int{2, 2}, []float32{
		1, 0,
		0.70710677, 0.70710677,
	})
	if got := result.Raw.Metadata["entrypoint"]; got != "embed" {
		t.Fatalf("entrypoint metadata = %q, want embed", got)
	}
}

func TestRunEmbedBatchDecodesOutput(t *testing.T) {
	src := []byte(`
param token_embedding: f16[V, D] @weight("weights/token_embedding")
param projection: f16[D, E] @weight("weights/projection")

kernel l2_normalize(x: f16[B, T, E]) -> f16[B, T, E] {
    return normalize(x)
}

pipeline embed_batch(tokens: i32[B, T]) -> f16[B, T, E] {
    let hidden = gather(token_embedding, tokens)
    let projected = @matmul(hidden, projection)
    return l2_normalize(projected)
}
`)

	bundle, err := compiler.Build(src, compiler.Options{ModuleName: "batch_embed"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	rt := New(cuda.New(), metal.New())
	prog, err := rt.Load(context.Background(), bundle.Artifact, tinyEmbedWeights()...)
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	result, err := prog.RunEmbedBatch(context.Background(), "embed_batch", [][]int32{
		{0, 2},
		{1, 0},
	})
	if err != nil {
		t.Fatalf("run embed batch: %v", err)
	}
	if result.OutputName != "result" {
		t.Fatalf("output name = %q, want result", result.OutputName)
	}
	assertTensorClose(t, result.Embeddings, []int{2, 2, 2}, []float32{
		1, 0,
		0.70710677, 0.70710677,
		0, 1,
		1, 0,
	})
}

func TestRunEmbedBatchRejectsRaggedTokens(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_embed", Preset: compiler.PresetTinyEmbed})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	rt := New(cuda.New(), metal.New())
	prog, err := rt.Load(context.Background(), bundle.Artifact, tinyEmbedWeights()...)
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	_, err = prog.RunEmbedBatch(context.Background(), "embed", [][]int32{
		{0, 2},
		{1},
	})
	if err == nil {
		t.Fatal("expected ragged token batch error")
	}
	if !strings.Contains(err.Error(), "token batch 1 size 1 does not match 2") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestRunEmbedPooledDecodesOutput(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_embed_pooled", Preset: compiler.PresetTinyEmbedPooled})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	rt := New(cuda.New(), metal.New())
	prog, err := rt.Load(context.Background(), bundle.Artifact, tinyEmbedWeights()...)
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	result, err := prog.RunEmbed(context.Background(), "embed_pooled", []int32{0, 2})
	if err != nil {
		t.Fatalf("run embed pooled: %v", err)
	}
	if result.OutputName != "result" {
		t.Fatalf("output name = %q, want result", result.OutputName)
	}
	assertTensorClose(t, result.Embeddings, []int{2}, []float32{
		0.8535534, 0.35355338,
	})
}

func TestRunEmbedBatchPooledDecodesOutput(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_embed_pooled", Preset: compiler.PresetTinyEmbedPooled})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	rt := New(cuda.New(), metal.New())
	prog, err := rt.Load(context.Background(), bundle.Artifact, tinyEmbedWeights()...)
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	result, err := prog.RunEmbedBatch(context.Background(), "embed_pooled_batch", [][]int32{
		{0, 2},
		{1, 0},
	})
	if err != nil {
		t.Fatalf("run embed pooled batch: %v", err)
	}
	if result.OutputName != "result" {
		t.Fatalf("output name = %q, want result", result.OutputName)
	}
	assertTensorClose(t, result.Embeddings, []int{2, 2}, []float32{
		0.8535534, 0.35355338,
		0.5, 0.5,
	})
}

func TestRunScoreDecodesOutput(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_score", Preset: compiler.PresetTinyScore})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	rt := New(cuda.New(), metal.New())
	prog, err := rt.Load(context.Background(), bundle.Artifact, tinyScoreWeights()...)
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	result, err := prog.RunScore(context.Background(), backend.Request{
		Entry:  "score",
		Inputs: map[string]any{"query": backend.NewTensorF16([]int{2}, []float32{1, 0})},
	})
	if err != nil {
		t.Fatalf("run score: %v", err)
	}
	if result.OutputName != "scores" {
		t.Fatalf("output name = %q, want scores", result.OutputName)
	}
	assertTensorClose(t, result.Scores, []int{2}, []float32{1, 0})
	if got := result.Raw.Metadata["entrypoint"]; got != "score" {
		t.Fatalf("entrypoint metadata = %q, want score", got)
	}
}

func TestRunScoreTableBuildsDocsInput(t *testing.T) {
	src := []byte(`
pipeline score(query: f16[D], docs: q4[N, D]) -> f32[N] {
    return cosine(query, docs)
}
`)

	bundle, err := compiler.Build(src, compiler.Options{ModuleName: "score_table"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	rt := New(cuda.New(), metal.New())
	prog, err := rt.Load(context.Background(), bundle.Artifact)
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	result, err := prog.RunScoreTable(context.Background(), "score", backend.NewTensorF16([]int{2}, []float32{1, 0}), []*backend.Tensor{
		backend.NewTensorQ4([]int{2}, []float32{1, 0}),
		backend.NewTensorQ4([]int{2}, []float32{0, 1}),
		backend.NewTensorQ4([]int{2}, []float32{1, 1}),
	})
	if err != nil {
		t.Fatalf("run score table: %v", err)
	}

	if result.OutputName != "scores" {
		t.Fatalf("output name = %q, want scores", result.OutputName)
	}
	assertTensorClose(t, result.Scores, []int{3}, []float32{1, 0, 0.70710677})
}

func TestRunScoreTableBatchedBuildsDocsInput(t *testing.T) {
	src := []byte(`
pipeline score_batch(queries: f16[Q, D], docs: q4[Q, N, D]) -> f32[Q, N] {
    return cosine(queries, docs)
}
`)

	bundle, err := compiler.Build(src, compiler.Options{ModuleName: "score_batch"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	rt := New(cuda.New(), metal.New())
	prog, err := rt.Load(context.Background(), bundle.Artifact)
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	result, err := prog.RunScoreTableBatched(context.Background(), "score_batch", backend.NewTensorF16([]int{2, 2}, []float32{
		1, 0,
		0, 1,
	}), [][]*backend.Tensor{
		{
			backend.NewTensorQ4([]int{2}, []float32{1, 0}),
			backend.NewTensorQ4([]int{2}, []float32{0, 1}),
			backend.NewTensorQ4([]int{2}, []float32{1, 1}),
		},
		{
			backend.NewTensorQ4([]int{2}, []float32{0, 1}),
			backend.NewTensorQ4([]int{2}, []float32{1, 0}),
			backend.NewTensorQ4([]int{2}, []float32{1, 1}),
		},
	})
	if err != nil {
		t.Fatalf("run batched score table: %v", err)
	}

	if result.OutputName != "result" {
		t.Fatalf("output name = %q, want result", result.OutputName)
	}
	assertTensorClose(t, result.Scores, []int{2, 3}, []float32{
		1, 0, 0.70710677,
		1, 0, 0.70710677,
	})
}

func TestRunScoreRejectsNonTensorOutput(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_packed_candidates", Preset: compiler.PresetTinyPackedCandidates})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	rt := New(cuda.New(), metal.New())
	prog, err := rt.Load(context.Background(), bundle.Artifact)
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	_, err = prog.RunScore(context.Background(), backend.Request{
		Entry: "rerank_candidates_packed",
		Inputs: map[string]any{
			"query":         backend.NewTensorF16([]int{2}, []float32{1, 0}),
			"docs":          backend.NewTensorQ4([]int{3, 2}, []float32{1, 0, 0, 1, 1, 1}),
			"candidate_ids": backend.NewTensorI64([]int{3}, []int64{1001, 2002, 3003}),
		},
	})
	if err == nil {
		t.Fatal("expected non-tensor output error")
	}
	if !strings.Contains(err.Error(), "no tensor output found") {
		t.Fatalf("unexpected error: %v", err)
	}
}
