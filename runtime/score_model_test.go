package barruntime

import (
	"context"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/odvcencio/manta/artifact/barr"
	"github.com/odvcencio/manta/compiler"
	"github.com/odvcencio/manta/runtime/backend"
	"github.com/odvcencio/manta/runtime/backends/cuda"
	"github.com/odvcencio/manta/runtime/backends/metal"
)

func TestScoreManifestRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "tiny_score.score.mll")
	want := scoreBundleManifest()
	if err := want.WriteFile(path); err != nil {
		t.Fatalf("write manifest: %v", err)
	}
	got, err := ReadScoreManifestFile(path)
	if err != nil {
		t.Fatalf("read manifest: %v", err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("manifest mismatch:\nwant: %+v\ngot:  %+v", want, got)
	}
}

func TestDefaultScoreManifestPath(t *testing.T) {
	got := DefaultScoreManifestPath("/tmp/tiny_score.mll")
	if want := "/tmp/tiny_score.score.mll"; got != want {
		t.Fatalf("manifest path = %q, want %q", got, want)
	}
}

func TestLoadScoreValidatesManifest(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_embed", Preset: compiler.PresetTinyEmbed})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	rt := New(cuda.New(), metal.New())
	_, err = rt.LoadScore(context.Background(), bundle.Artifact, scoreBundleManifest(), tinyEmbedWeights()...)
	if err == nil {
		t.Fatal("expected score manifest validation error")
	}
	if !strings.Contains(err.Error(), `unknown entrypoint "score"`) {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestScoreModelScore(t *testing.T) {
	src := []byte(`
pipeline score(query: f16[D], docs: q4[N, D]) -> f32[N] {
    return cosine(query, docs)
}
`)

	bundle, err := compiler.Build(src, compiler.Options{ModuleName: "score_bundle"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	rt := New(cuda.New(), metal.New())
	model, err := rt.LoadScore(context.Background(), bundle.Artifact, scoreBundleManifest())
	if err != nil {
		t.Fatalf("load score: %v", err)
	}

	result, err := model.Score(context.Background(), backend.NewTensorF16([]int{2}, []float32{1, 0}), []*backend.Tensor{
		backend.NewTensorQ4([]int{2}, []float32{1, 0}),
		backend.NewTensorQ4([]int{2}, []float32{0, 1}),
	})
	if err != nil {
		t.Fatalf("score: %v", err)
	}
	if got := model.Backend(); got == "" {
		t.Fatal("expected selected backend")
	}
	assertTensorClose(t, result.Scores, []int{2}, []float32{1, 0})
}

func TestScoreModelScoreBatch(t *testing.T) {
	src := []byte(`
pipeline score(query: f16[D], docs: q4[N, D]) -> f32[N] {
    return cosine(query, docs)
}

pipeline score_batch(queries: f16[Q, D], docs: q4[Q, N, D]) -> f32[Q, N] {
    return cosine(queries, docs)
}
`)

	bundle, err := compiler.Build(src, compiler.Options{ModuleName: "score_bundle"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	rt := New(cuda.New(), metal.New())
	model, err := rt.LoadScore(context.Background(), bundle.Artifact, scoreBundleBatchManifest())
	if err != nil {
		t.Fatalf("load score: %v", err)
	}

	result, err := model.ScoreBatch(context.Background(), backend.NewTensorF16([]int{2, 2}, []float32{
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
		t.Fatalf("score batch: %v", err)
	}
	assertTensorClose(t, result.Scores, []int{2, 3}, []float32{
		1, 0, 0.70710677,
		1, 0, 0.70710677,
	})
}

func TestLoadScoreBundleUsesSiblingManifest(t *testing.T) {
	src := []byte(`
pipeline score(query: f16[D], docs: q4[N, D]) -> f32[N] {
    return cosine(query, docs)
}
`)

	bundle, err := compiler.Build(src, compiler.Options{ModuleName: "score_bundle"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	dir := t.TempDir()
	barrPath := filepath.Join(dir, "score_bundle.mll")
	if err := barr.WriteFile(barrPath, bundle.Artifact); err != nil {
		t.Fatalf("write artifact: %v", err)
	}
	if err := scoreBundleManifest().WriteFile(DefaultScoreManifestPath(barrPath)); err != nil {
		t.Fatalf("write manifest: %v", err)
	}

	rt := New(cuda.New(), metal.New())
	model, err := rt.LoadScoreBundle(context.Background(), barrPath)
	if err != nil {
		t.Fatalf("load score bundle: %v", err)
	}
	result, err := model.Score(context.Background(), backend.NewTensorF16([]int{2}, []float32{1, 0}), []*backend.Tensor{
		backend.NewTensorQ4([]int{2}, []float32{1, 0}),
		backend.NewTensorQ4([]int{2}, []float32{0, 1}),
	})
	if err != nil {
		t.Fatalf("score from bundle: %v", err)
	}
	assertTensorClose(t, result.Scores, []int{2}, []float32{1, 0})
}

func scoreBundleManifest() ScoreManifest {
	return ScoreManifest{
		Name:        "score_bundle",
		Entry:       "score",
		QueryInput:  "query",
		DocsInput:   "docs",
		OutputName:  "scores",
		QueryDType:  "f16",
		DocsDType:   "q4",
		OutputDType: "f32",
	}
}

func scoreBundleBatchManifest() ScoreManifest {
	manifest := scoreBundleManifest()
	manifest.Name = "score_bundle"
	manifest.BatchEntry = "score_batch"
	manifest.BatchOutputName = "result"
	return manifest
}
