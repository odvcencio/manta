package barruntime

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/odvcencio/barracuda/artifact/barr"
	"github.com/odvcencio/barracuda/compiler"
)

func TestInitializeEmbeddingTrainerPackageWithManifestCreatesPackage(t *testing.T) {
	source := []byte(`
param token_embedding: q8[V, D] @weight("weights/token_embedding") @trainable
param projection: q8[D, E] @weight("weights/projection") @trainable

pipeline embed_pooled(tokens: i32[T]) -> q8[E] {
    let embeddings = gather(token_embedding, tokens)
    let projected = @matmul(embeddings, projection)
    return mean_pool(projected)
}

pipeline embed_pooled_batch(tokens: i32[B, T]) -> q8[B, E] {
    let embeddings = gather(token_embedding, tokens)
    let projected = @matmul(embeddings, projection)
    return mean_pool(projected)
}
`)
	bundle, err := compiler.Build(source, compiler.Options{ModuleName: "tiny_train_embed"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	path := filepath.Join(t.TempDir(), "tiny_train_embed.barr")
	if err := barr.WriteFile(path, bundle.Artifact); err != nil {
		t.Fatalf("write artifact: %v", err)
	}
	manifest := EmbeddingManifest{
		Name:                "tiny-train-embed",
		PooledEntry:         "embed_pooled",
		BatchEntry:          "embed_pooled_batch",
		TokenInput:          "tokens",
		OutputName:          "result",
		OutputDType:         "q8",
		TokenEmbeddingParam: "token_embedding",
		ProjectionParam:     "projection",
		Tokenizer: TokenizerManifest{
			VocabSize:   8,
			MaxSequence: 8,
			PadID:       0,
		},
	}
	if err := manifest.WriteFile(DefaultEmbeddingManifestPath(path)); err != nil {
		t.Fatalf("write manifest: %v", err)
	}
	paths, err := InitializeEmbeddingTrainerPackageWithManifest(path, manifest, EmbeddingTrainConfig{LearningRate: 0.02}, EmbeddingTrainInitOptions{
		Seed:       7,
		ShapeSizes: map[string]int{"D": 4, "E": 3},
	})
	if err != nil {
		t.Fatalf("initialize training package: %v", err)
	}
	for _, candidate := range []string{
		paths.EmbeddingManifestPath,
		paths.WeightFilePath,
		paths.MemoryPlanPath,
		paths.TrainManifestPath,
		paths.CheckpointPath,
		paths.TrainProfilePath,
	} {
		if _, err := os.Stat(candidate); err != nil {
			t.Fatalf("expected package file %q: %v", candidate, err)
		}
	}
	trainer, err := LoadEmbeddingTrainerPackage(path)
	if err != nil {
		t.Fatalf("load training package: %v", err)
	}
	if got := trainer.tokenEmbed.Shape; len(got) != 2 || got[0] != 8 || got[1] != 4 {
		t.Fatalf("token embedding shape = %v, want [8 4]", got)
	}
	if got := trainer.projection.Shape; len(got) != 2 || got[0] != 4 || got[1] != 3 {
		t.Fatalf("projection shape = %v, want [4 3]", got)
	}
}

func TestInitializeEmbeddingTrainerPackageRejectsUnresolvedShapes(t *testing.T) {
	source := []byte(`
param token_embedding: q8[V, D] @weight("weights/token_embedding") @trainable
param projection: q8[D, E] @weight("weights/projection") @trainable

pipeline embed_pooled(tokens: i32[T]) -> q8[E] {
    let embeddings = gather(token_embedding, tokens)
    let projected = @matmul(embeddings, projection)
    return mean_pool(projected)
}

pipeline embed_pooled_batch(tokens: i32[B, T]) -> q8[B, E] {
    let embeddings = gather(token_embedding, tokens)
    let projected = @matmul(embeddings, projection)
    return mean_pool(projected)
}
`)
	bundle, err := compiler.Build(source, compiler.Options{ModuleName: "tiny_train_embed"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	path := filepath.Join(t.TempDir(), "tiny_train_embed.barr")
	if err := barr.WriteFile(path, bundle.Artifact); err != nil {
		t.Fatalf("write artifact: %v", err)
	}
	manifest := EmbeddingManifest{
		Name:                "tiny-train-embed",
		PooledEntry:         "embed_pooled",
		BatchEntry:          "embed_pooled_batch",
		TokenInput:          "tokens",
		OutputName:          "result",
		OutputDType:         "q8",
		TokenEmbeddingParam: "token_embedding",
		ProjectionParam:     "projection",
		Tokenizer:           TokenizerManifest{VocabSize: 8},
	}
	if _, err := InitializeEmbeddingTrainerPackageWithManifest(path, manifest, EmbeddingTrainConfig{}, EmbeddingTrainInitOptions{}); err == nil {
		t.Fatal("expected unresolved symbolic dim error")
	}
}
