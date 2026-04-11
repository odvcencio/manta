package models

import (
	"os"
	"path/filepath"
	"testing"

	barruntime "github.com/odvcencio/barracuda/runtime"
)

func TestInitDefaultEmbeddingPackageCreatesTrainablePackage(t *testing.T) {
	path := filepath.Join(t.TempDir(), "manta-embed-v0.mll")
	paths, err := InitDefaultEmbeddingPackage(path, DefaultEmbeddingPackageConfig{
		VocabSize:    16,
		MaxSequence:  8,
		EmbeddingDim: 4,
		HiddenDim:    8,
		Seed:         7,
	})
	if err != nil {
		t.Fatalf("init default embedding package: %v", err)
	}
	for _, candidate := range []string{
		paths.ArtifactPath,
		paths.EmbeddingManifestPath,
		paths.WeightFilePath,
		paths.MemoryPlanPath,
		paths.TrainManifestPath,
		paths.CheckpointPath,
		paths.TrainProfilePath,
		paths.PackageManifestPath,
	} {
		if _, err := os.Stat(candidate); err != nil {
			t.Fatalf("expected package file %q: %v", candidate, err)
		}
	}
	manifest, err := barruntime.ReadEmbeddingManifestFile(paths.EmbeddingManifestPath)
	if err != nil {
		t.Fatalf("read embedding manifest: %v", err)
	}
	if manifest.Name != DefaultEmbeddingModelName {
		t.Fatalf("manifest name = %q, want %q", manifest.Name, DefaultEmbeddingModelName)
	}
	if manifest.EncoderRepeats != 2 {
		t.Fatalf("encoder repeats = %d, want 2", manifest.EncoderRepeats)
	}
	if manifest.Tokenizer.VocabSize != 16 || manifest.Tokenizer.MaxSequence != 8 {
		t.Fatalf("unexpected tokenizer contract: %+v", manifest.Tokenizer)
	}
	if manifest.Tokenizer.PadID != 0 || manifest.Tokenizer.BOSID != 1 || manifest.Tokenizer.EOSID != 2 || manifest.Tokenizer.UnknownID != 3 {
		t.Fatalf("unexpected tokenizer ids: %+v", manifest.Tokenizer)
	}
	checkpoint, err := barruntime.ReadEmbeddingTrainCheckpointFile(paths.CheckpointPath)
	if err != nil {
		t.Fatalf("read checkpoint: %v", err)
	}
	if checkpoint.Config.ContrastiveLoss != "infonce" {
		t.Fatalf("contrastive loss = %q, want infonce", checkpoint.Config.ContrastiveLoss)
	}
	if checkpoint.Config.Temperature != 0.05 {
		t.Fatalf("temperature = %f, want 0.05", checkpoint.Config.Temperature)
	}
	if _, err := barruntime.LoadEmbeddingTrainerPackage(path); err != nil {
		t.Fatalf("load training package: %v", err)
	}
}
