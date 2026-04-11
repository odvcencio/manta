package barruntime

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/odvcencio/barracuda/artifact/barr"
	"github.com/odvcencio/barracuda/compiler"
	"github.com/odvcencio/barracuda/runtime/backend"
	"github.com/odvcencio/barracuda/runtime/backends/cuda"
	"github.com/odvcencio/barracuda/runtime/backends/metal"
)

func TestDefaultPackageManifestPath(t *testing.T) {
	got := DefaultPackageManifestPath("/tmp/tiny_embed.barr")
	want := "/tmp/tiny_embed.package.mll"
	if got != want {
		t.Fatalf("package manifest path = %q, want %q", got, want)
	}
}

func TestResolvePackageManifestPathFallsBackToLegacyJSON(t *testing.T) {
	dir := t.TempDir()
	artifactPath := filepath.Join(dir, "tiny_embed.mll")
	legacyPath := filepath.Join(dir, "tiny_embed.package.json")
	if err := os.WriteFile(legacyPath, []byte("{}\n"), 0o644); err != nil {
		t.Fatalf("write legacy manifest stub: %v", err)
	}
	got := ResolvePackageManifestPath(artifactPath)
	if got != legacyPath {
		t.Fatalf("ResolvePackageManifestPath = %q, want %q", got, legacyPath)
	}
}

func TestPackageManifestRoundTripAndVerify(t *testing.T) {
	dir := t.TempDir()
	artifactPath := filepath.Join(dir, "tiny_embed.barr")
	manifestPath := filepath.Join(dir, "tiny_embed.embedding.mll")
	weightPath := filepath.Join(dir, "tiny_embed.weights.mll")
	memoryPlanPath := filepath.Join(dir, "tiny_embed.memory.mll")

	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_embed", Preset: compiler.PresetTinyEmbed})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	if err := barr.WriteFile(artifactPath, bundle.Artifact); err != nil {
		t.Fatalf("write artifact: %v", err)
	}
	if err := tinyEmbeddingManifest().WriteFile(manifestPath); err != nil {
		t.Fatalf("write manifest: %v", err)
	}
	weights := NewWeightFile(map[string]*backend.Tensor{
		"token_embedding": backend.NewTensorF16([]int{3, 2}, []float32{1, 0, 0, 1, 1, 1}),
		"projection":      backend.NewTensorF16([]int{2, 2}, []float32{1, 0, 0, 1}),
	})
	if err := weights.WriteFile(weightPath); err != nil {
		t.Fatalf("write weights: %v", err)
	}
	plan := NewMemoryPlan(bundle.Artifact, weights.Weights, MemoryPlanOptions{})
	if err := plan.WriteFile(memoryPlanPath); err != nil {
		t.Fatalf("write memory plan: %v", err)
	}
	manifest, err := BuildPackageManifest(PackageEmbedding, bundle.Artifact, map[string]string{
		"artifact":           artifactPath,
		"embedding_manifest": manifestPath,
		"weights":            weightPath,
		"memory_plan":        memoryPlanPath,
	})
	if err != nil {
		t.Fatalf("build package manifest: %v", err)
	}
	path := filepath.Join(dir, "tiny_embed.package.mll")
	if err := manifest.WriteFile(path); err != nil {
		t.Fatalf("write package manifest: %v", err)
	}
	loaded, err := ReadPackageManifestFile(path)
	if err != nil {
		t.Fatalf("read package manifest: %v", err)
	}
	if err := loaded.VerifyFiles(map[string]string{
		"artifact":           artifactPath,
		"embedding_manifest": manifestPath,
		"weights":            weightPath,
		"memory_plan":        memoryPlanPath,
	}); err != nil {
		t.Fatalf("verify package manifest: %v", err)
	}
	if err := os.WriteFile(weightPath, []byte("tampered\n"), 0o644); err != nil {
		t.Fatalf("tamper weights: %v", err)
	}
	if err := loaded.VerifyFiles(map[string]string{
		"artifact":           artifactPath,
		"embedding_manifest": manifestPath,
		"weights":            weightPath,
		"memory_plan":        memoryPlanPath,
	}); err == nil {
		t.Fatal("expected verify failure after tampering")
	}
}

func TestPackageManifestRoundTripAndVerifyWithTokenizer(t *testing.T) {
	dir := t.TempDir()
	artifactPath := filepath.Join(dir, "tiny_embed.barr")
	manifestPath := filepath.Join(dir, "tiny_embed.embedding.mll")
	tokenizerPath := filepath.Join(dir, "tiny_embed.tokenizer.mll")
	weightPath := filepath.Join(dir, "tiny_embed.weights.mll")
	memoryPlanPath := filepath.Join(dir, "tiny_embed.memory.mll")

	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_embed", Preset: compiler.PresetTinyEmbed})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	if err := barr.WriteFile(artifactPath, bundle.Artifact); err != nil {
		t.Fatalf("write artifact: %v", err)
	}
	if err := tinyEmbeddingManifest().WriteFile(manifestPath); err != nil {
		t.Fatalf("write manifest: %v", err)
	}
	if err := (TokenizerFile{
		Version:      TokenizerFileVersion,
		Tokens:       []string{"[PAD]", "[CLS]", "[SEP]", "[UNK]", "[MASK]", "a", "b"},
		PadToken:     "[PAD]",
		BOSToken:     "[CLS]",
		EOSToken:     "[SEP]",
		UnknownToken: "[UNK]",
	}).WriteFile(tokenizerPath); err != nil {
		t.Fatalf("write tokenizer: %v", err)
	}
	weights := NewWeightFile(map[string]*backend.Tensor{
		"token_embedding": backend.NewTensorF16([]int{3, 2}, []float32{1, 0, 0, 1, 1, 1}),
		"projection":      backend.NewTensorF16([]int{2, 2}, []float32{1, 0, 0, 1}),
	})
	if err := weights.WriteFile(weightPath); err != nil {
		t.Fatalf("write weights: %v", err)
	}
	plan := NewMemoryPlan(bundle.Artifact, weights.Weights, MemoryPlanOptions{})
	if err := plan.WriteFile(memoryPlanPath); err != nil {
		t.Fatalf("write memory plan: %v", err)
	}
	manifest, err := BuildPackageManifest(PackageEmbedding, bundle.Artifact, map[string]string{
		"artifact":           artifactPath,
		"embedding_manifest": manifestPath,
		"tokenizer":          tokenizerPath,
		"weights":            weightPath,
		"memory_plan":        memoryPlanPath,
	})
	if err != nil {
		t.Fatalf("build package manifest: %v", err)
	}
	if err := manifest.VerifyFiles(map[string]string{
		"artifact":           artifactPath,
		"embedding_manifest": manifestPath,
		"tokenizer":          tokenizerPath,
		"weights":            weightPath,
		"memory_plan":        memoryPlanPath,
	}); err != nil {
		t.Fatalf("verify package manifest with tokenizer: %v", err)
	}
}

func TestRebuildSiblingPackageManifestAddsTokenizer(t *testing.T) {
	trainer := newTinyTrainableFFNEmbeddingTrainer(t, 0.05)
	artifactPath := filepath.Join(t.TempDir(), "tiny_train_embed_q8.mll")
	paths, err := trainer.WriteTrainingPackage(artifactPath)
	if err != nil {
		t.Fatalf("write training package: %v", err)
	}
	tokenizer := TokenizerFile{
		Version:      TokenizerFileVersion,
		Tokens:       []string{"[PAD]", "[UNK]", "[CLS]", "[SEP]", "alpha", "beta"},
		PadToken:     "[PAD]",
		UnknownToken: "[UNK]",
		BOSToken:     "[CLS]",
		EOSToken:     "[SEP]",
	}
	if err := tokenizer.WriteFile(DefaultTokenizerPath(artifactPath)); err != nil {
		t.Fatalf("write tokenizer: %v", err)
	}
	manifest, manifestPath, err := RebuildSiblingPackageManifest(artifactPath)
	if err != nil {
		t.Fatalf("rebuild package manifest: %v", err)
	}
	if manifestPath != DefaultPackageManifestPath(artifactPath) {
		t.Fatalf("manifest path = %q, want %q", manifestPath, DefaultPackageManifestPath(artifactPath))
	}
	var hasTokenizer bool
	for _, item := range manifest.Files {
		if item.Role == "tokenizer" {
			hasTokenizer = true
			break
		}
	}
	if !hasTokenizer {
		t.Fatal("expected rebuilt package manifest to include tokenizer")
	}
	if err := manifest.VerifyFiles(map[string]string{
		"artifact":           paths.ArtifactPath,
		"embedding_manifest": paths.EmbeddingManifestPath,
		"tokenizer":          DefaultTokenizerPath(artifactPath),
		"weights":            paths.WeightFilePath,
		"memory_plan":        paths.MemoryPlanPath,
		"train_manifest":     paths.TrainManifestPath,
		"checkpoint":         paths.CheckpointPath,
		"train_profile":      paths.TrainProfilePath,
	}); err != nil {
		t.Fatalf("verify rebuilt package manifest: %v", err)
	}
}

func TestSyncEmbeddingTokenizerVocabUpdatesTrainingPackageState(t *testing.T) {
	trainer := newTinyTrainableFFNEmbeddingTrainer(t, 0.05)
	artifactPath := filepath.Join(t.TempDir(), "tiny_train_embed_q8.mll")
	if _, err := trainer.WriteTrainingPackage(artifactPath); err != nil {
		t.Fatalf("write training package: %v", err)
	}
	if err := (TokenizerFile{
		Version:      TokenizerFileVersion,
		Tokens:       []string{"[PAD]", "[UNK]", "[CLS]", "[SEP]", "alpha", "beta", "gamma"},
		PadToken:     "[PAD]",
		UnknownToken: "[UNK]",
		BOSToken:     "[CLS]",
		EOSToken:     "[SEP]",
	}).WriteFile(DefaultTokenizerPath(artifactPath)); err != nil {
		t.Fatalf("write tokenizer: %v", err)
	}
	if err := SyncEmbeddingTokenizerVocab(artifactPath, 7); err != nil {
		t.Fatalf("sync tokenizer vocab: %v", err)
	}
	embeddingManifest, err := ReadEmbeddingManifestFile(DefaultEmbeddingManifestPath(artifactPath))
	if err != nil {
		t.Fatalf("read embedding manifest: %v", err)
	}
	if embeddingManifest.Tokenizer.VocabSize != 7 {
		t.Fatalf("embedding manifest vocab size = %d, want 7", embeddingManifest.Tokenizer.VocabSize)
	}
	trainManifest, err := ReadEmbeddingTrainManifestFile(DefaultEmbeddingTrainManifestPath(artifactPath))
	if err != nil {
		t.Fatalf("read train manifest: %v", err)
	}
	if trainManifest.Embedding.Tokenizer.VocabSize != 7 {
		t.Fatalf("train manifest vocab size = %d, want 7", trainManifest.Embedding.Tokenizer.VocabSize)
	}
	checkpoint, err := ReadEmbeddingTrainCheckpointFile(DefaultEmbeddingCheckpointPath(artifactPath))
	if err != nil {
		t.Fatalf("read checkpoint: %v", err)
	}
	if checkpoint.Manifest.Tokenizer.VocabSize != 7 {
		t.Fatalf("checkpoint vocab size = %d, want 7", checkpoint.Manifest.Tokenizer.VocabSize)
	}
	if checkpoint.TokenEmbedding.Shape[0] != 7 {
		t.Fatalf("checkpoint token rows = %d, want 7", checkpoint.TokenEmbedding.Shape[0])
	}
	weights, err := ReadWeightFile(DefaultWeightFilePath(artifactPath))
	if err != nil {
		t.Fatalf("read weights: %v", err)
	}
	if weights.Weights[trainer.manifest.TokenEmbeddingParam].Shape[0] != 7 {
		t.Fatalf("weight token rows = %d, want 7", weights.Weights[trainer.manifest.TokenEmbeddingParam].Shape[0])
	}
	packageManifest, err := ReadPackageManifestFile(DefaultPackageManifestPath(artifactPath))
	if err != nil {
		t.Fatalf("read package manifest: %v", err)
	}
	if err := packageManifest.VerifyFiles(map[string]string{
		"artifact":           artifactPath,
		"embedding_manifest": DefaultEmbeddingManifestPath(artifactPath),
		"tokenizer":          DefaultTokenizerPath(artifactPath),
		"weights":            DefaultWeightFilePath(artifactPath),
		"memory_plan":        DefaultMemoryPlanPath(artifactPath),
		"train_manifest":     DefaultEmbeddingTrainManifestPath(artifactPath),
		"checkpoint":         DefaultEmbeddingCheckpointPath(artifactPath),
		"train_profile":      DefaultEmbeddingTrainProfilePath(artifactPath),
	}); err != nil {
		t.Fatalf("verify synced package manifest: %v", err)
	}
	if _, err := LoadEmbeddingTrainerPackage(artifactPath); err != nil {
		t.Fatalf("reload synced training package: %v", err)
	}
}

func TestLoadEmbeddingPackageRejectsTamperedWeightFile(t *testing.T) {
	trainer := newTinyTrainableFFNEmbeddingTrainer(t, 0.05)
	packagePath := filepath.Join(t.TempDir(), "tiny_train_embed_q8.barr")
	paths, err := trainer.WriteEmbeddingPackage(packagePath)
	if err != nil {
		t.Fatalf("write embedding package: %v", err)
	}
	if err := os.WriteFile(paths.WeightFilePath, []byte("tampered\n"), 0o644); err != nil {
		t.Fatalf("tamper weights: %v", err)
	}
	rt := New(cuda.New(), metal.New())
	_, err = rt.LoadEmbeddingPackage(context.Background(), packagePath)
	if err == nil {
		t.Fatal("expected tamper failure")
	}
	if !strings.Contains(err.Error(), "sha256 mismatch") {
		t.Fatalf("expected sha256 mismatch, got %v", err)
	}
}
