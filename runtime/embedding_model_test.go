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

func TestEmbeddingManifestRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "embed_pooled.embedding.mll")
	want := tinyEmbeddingManifest()
	if err := want.WriteFile(path); err != nil {
		t.Fatalf("write manifest: %v", err)
	}
	got, err := ReadEmbeddingManifestFile(path)
	if err != nil {
		t.Fatalf("read manifest: %v", err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("manifest mismatch:\nwant: %+v\ngot:  %+v", want, got)
	}
}

func TestDefaultEmbeddingManifestPath(t *testing.T) {
	got := DefaultEmbeddingManifestPath("/tmp/tiny_embed_pooled.mll")
	if want := "/tmp/tiny_embed_pooled.embedding.mll"; got != want {
		t.Fatalf("manifest path = %q, want %q", got, want)
	}
}

func TestLoadEmbeddingValidatesManifest(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_score", Preset: compiler.PresetTinyScore})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	rt := New(cuda.New(), metal.New())
	_, err = rt.LoadEmbedding(context.Background(), bundle.Artifact, tinyEmbeddingManifest(), tinyScoreWeights()...)
	if err == nil {
		t.Fatal("expected embedding manifest validation error")
	}
	if !strings.Contains(err.Error(), `unknown entrypoint "embed_pooled"`) {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestEmbeddingModelEmbed(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_embed_pooled", Preset: compiler.PresetTinyEmbedPooled})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	rt := New(cuda.New(), metal.New())
	model, err := rt.LoadEmbedding(context.Background(), bundle.Artifact, tinyEmbeddingManifest(), tinyEmbedWeights()...)
	if err != nil {
		t.Fatalf("load embedding: %v", err)
	}

	result, err := model.Embed(context.Background(), []int32{0, 2})
	if err != nil {
		t.Fatalf("embed: %v", err)
	}
	if got := model.Backend(); got == "" {
		t.Fatal("expected selected backend")
	}
	assertTensorClose(t, result.Embeddings, []int{2}, []float32{
		0.8535534, 0.35355338,
	})
}

func TestEmbeddingModelEmbedBatch(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_embed_pooled", Preset: compiler.PresetTinyEmbedPooled})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	rt := New(cuda.New(), metal.New())
	model, err := rt.LoadEmbedding(context.Background(), bundle.Artifact, tinyEmbeddingManifest(), tinyEmbedWeights()...)
	if err != nil {
		t.Fatalf("load embedding: %v", err)
	}

	result, err := model.EmbedBatch(context.Background(), [][]int32{
		{0, 2},
		{1, 0},
	})
	if err != nil {
		t.Fatalf("embed batch: %v", err)
	}
	assertTensorClose(t, result.Embeddings, []int{2, 2}, []float32{
		0.8535534, 0.35355338,
		0.5, 0.5,
	})
}

func TestEmbeddingModelRejectsTokenOutOfRange(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_embed_pooled", Preset: compiler.PresetTinyEmbedPooled})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	rt := New(cuda.New(), metal.New())
	model, err := rt.LoadEmbedding(context.Background(), bundle.Artifact, tinyEmbeddingManifest(), tinyEmbedWeights()...)
	if err != nil {
		t.Fatalf("load embedding: %v", err)
	}

	_, err = model.Embed(context.Background(), []int32{0, 3})
	if err == nil {
		t.Fatal("expected token range error")
	}
	if !strings.Contains(err.Error(), "outside vocab_size 3") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestEmbeddingModelRejectsSequenceTooLong(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_embed_pooled", Preset: compiler.PresetTinyEmbedPooled})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	rt := New(cuda.New(), metal.New())
	model, err := rt.LoadEmbedding(context.Background(), bundle.Artifact, tinyEmbeddingManifest(), tinyEmbedWeights()...)
	if err != nil {
		t.Fatalf("load embedding: %v", err)
	}

	_, err = model.Embed(context.Background(), []int32{0, 1, 2})
	if err == nil {
		t.Fatal("expected max sequence error")
	}
	if !strings.Contains(err.Error(), "exceeds max_sequence 2") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestLoadEmbeddingFileUsesManifest(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_embed_pooled", Preset: compiler.PresetTinyEmbedPooled})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	dir := t.TempDir()
	barrPath := filepath.Join(dir, "tiny_embed_pooled.mll")
	manifestPath := filepath.Join(dir, "tiny_embed_pooled.embedding.mll")
	if err := barr.WriteFile(barrPath, bundle.Artifact); err != nil {
		t.Fatalf("write artifact: %v", err)
	}
	if err := tinyEmbeddingManifest().WriteFile(manifestPath); err != nil {
		t.Fatalf("write manifest: %v", err)
	}
	manifest, err := ReadEmbeddingManifestFile(manifestPath)
	if err != nil {
		t.Fatalf("read manifest: %v", err)
	}

	rt := New(cuda.New(), metal.New())
	model, err := rt.LoadEmbeddingFile(context.Background(), barrPath, manifest, tinyEmbedWeights()...)
	if err != nil {
		t.Fatalf("load embedding file: %v", err)
	}
	result, err := model.Embed(context.Background(), []int32{0, 2})
	if err != nil {
		t.Fatalf("embed from file: %v", err)
	}
	assertTensorClose(t, result.Embeddings, []int{2}, []float32{
		0.8535534, 0.35355338,
	})
}

func TestLoadEmbeddingBundleUsesSiblingManifest(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_embed_pooled", Preset: compiler.PresetTinyEmbedPooled})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	dir := t.TempDir()
	barrPath := filepath.Join(dir, "tiny_embed_pooled.mll")
	if err := barr.WriteFile(barrPath, bundle.Artifact); err != nil {
		t.Fatalf("write artifact: %v", err)
	}
	if err := tinyEmbeddingManifest().WriteFile(DefaultEmbeddingManifestPath(barrPath)); err != nil {
		t.Fatalf("write manifest: %v", err)
	}

	rt := New(cuda.New(), metal.New())
	model, err := rt.LoadEmbeddingBundle(context.Background(), barrPath, tinyEmbedWeights()...)
	if err != nil {
		t.Fatalf("load embedding bundle: %v", err)
	}
	result, err := model.Embed(context.Background(), []int32{0, 2})
	if err != nil {
		t.Fatalf("embed from bundle: %v", err)
	}
	assertTensorClose(t, result.Embeddings, []int{2}, []float32{
		0.8535534, 0.35355338,
	})
}

func TestLoadEmbeddingPackageUsesSiblingWeights(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_embed_pooled", Preset: compiler.PresetTinyEmbedPooled})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	dir := t.TempDir()
	barrPath := filepath.Join(dir, "tiny_embed_pooled.mll")
	if err := barr.WriteFile(barrPath, bundle.Artifact); err != nil {
		t.Fatalf("write artifact: %v", err)
	}
	if err := tinyEmbeddingManifest().WriteFile(DefaultEmbeddingManifestPath(barrPath)); err != nil {
		t.Fatalf("write manifest: %v", err)
	}
	weights := NewWeightFile(map[string]*backend.Tensor{
		"token_embedding": backend.NewTensorF16([]int{3, 2}, []float32{
			1, 0,
			0, 1,
			1, 1,
		}),
		"projection": backend.NewTensorF16([]int{2, 2}, []float32{
			1, 0,
			0, 1,
		}),
	})
	if err := weights.WriteFile(DefaultWeightFilePath(barrPath)); err != nil {
		t.Fatalf("write weights: %v", err)
	}
	plan := NewMemoryPlan(bundle.Artifact, weights.Weights, MemoryPlanOptions{
		DeviceBudgetBytes: 6,
		SharedHostWeights: true,
	})
	if err := plan.WriteFile(DefaultMemoryPlanPath(barrPath)); err != nil {
		t.Fatalf("write memory plan: %v", err)
	}

	rt := New(cuda.New(), metal.New())
	model, err := rt.LoadEmbeddingPackage(context.Background(), barrPath)
	if err != nil {
		t.Fatalf("load embedding package: %v", err)
	}
	result, err := model.Embed(context.Background(), []int32{0, 2})
	if err != nil {
		t.Fatalf("embed from package: %v", err)
	}
	assertTensorClose(t, result.Embeddings, []int{2}, []float32{
		0.8535534, 0.35355338,
	})
	if got := model.MemoryPlan(); got == nil {
		t.Fatal("expected memory plan on loaded embedding package")
	} else if got.ModuleName != bundle.Artifact.Name {
		t.Fatalf("memory plan module name = %q, want %q", got.ModuleName, bundle.Artifact.Name)
	}
}

func TestLoadEmbeddingPackageAcceptsTrainingPackageManifest(t *testing.T) {
	trainer := newTinyTrainableEmbeddingTrainer(t, 0.02)
	path := filepath.Join(t.TempDir(), "tiny_train_embed_q8.mll")
	if _, err := trainer.WriteTrainingPackage(path); err != nil {
		t.Fatalf("write training package: %v", err)
	}

	rt := New(cuda.New(), metal.New())
	model, err := rt.LoadEmbeddingPackage(context.Background(), path)
	if err != nil {
		t.Fatalf("load embedding package from training bundle: %v", err)
	}
	result, err := model.Embed(context.Background(), []int32{0})
	if err != nil {
		t.Fatalf("embed from training bundle: %v", err)
	}
	if result.Embeddings == nil || len(result.Embeddings.Shape) != 1 {
		t.Fatalf("embedding shape = %v, want rank-1 tensor", result.Embeddings.Shape)
	}
}

func TestEmbeddingModelEmbedMasked(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_embed_masked_pooled", Preset: compiler.PresetTinyEmbedMaskedPooled})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	rt := New(cuda.New(), metal.New())
	model, err := rt.LoadEmbedding(context.Background(), bundle.Artifact, tinyMaskedEmbeddingManifest(), tinyEmbedWeights()...)
	if err != nil {
		t.Fatalf("load embedding: %v", err)
	}

	result, err := model.Embed(context.Background(), []int32{1})
	if err != nil {
		t.Fatalf("embed: %v", err)
	}
	assertTensorClose(t, result.Embeddings, []int{2}, []float32{
		0, 1,
	})
}

func TestEmbeddingModelEmbedBatchPadsRaggedWithMask(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_embed_masked_pooled", Preset: compiler.PresetTinyEmbedMaskedPooled})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	rt := New(cuda.New(), metal.New())
	model, err := rt.LoadEmbedding(context.Background(), bundle.Artifact, tinyMaskedEmbeddingManifest(), tinyEmbedWeights()...)
	if err != nil {
		t.Fatalf("load embedding: %v", err)
	}

	result, err := model.EmbedBatch(context.Background(), [][]int32{
		{0, 2},
		{1},
	})
	if err != nil {
		t.Fatalf("embed batch: %v", err)
	}
	assertTensorClose(t, result.Embeddings, []int{2, 2}, []float32{
		0.8535534, 0.35355338,
		0, 1,
	})
}

func tinyEmbeddingManifest() EmbeddingManifest {
	return EmbeddingManifest{
		Name:                "tiny_embed_pooled",
		PooledEntry:         "embed_pooled",
		BatchEntry:          "embed_pooled_batch",
		TokenInput:          "tokens",
		OutputName:          "result",
		OutputDType:         "f16",
		TokenEmbeddingParam: "token_embedding",
		ProjectionParam:     "projection",
		Tokenizer: TokenizerManifest{
			VocabSize:   3,
			MaxSequence: 2,
			PadID:       0,
		},
	}
}

func tinyMaskedEmbeddingManifest() EmbeddingManifest {
	manifest := tinyEmbeddingManifest()
	manifest.Name = "tiny_embed_masked_pooled"
	manifest.MaskInput = "attention_mask"
	return manifest
}
