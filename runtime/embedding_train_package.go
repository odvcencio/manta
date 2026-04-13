package mantaruntime

import (
	"os"

	mantaartifact "github.com/odvcencio/manta/artifact/manta"
)

// EmbeddingTrainPackagePaths names the files that make up a packaged native-training embedder.
type EmbeddingTrainPackagePaths struct {
	ArtifactPath          string
	EmbeddingManifestPath string
	TokenizerPath         string
	WeightFilePath        string
	MemoryPlanPath        string
	TrainManifestPath     string
	CheckpointPath        string
	TrainProfilePath      string
	PackageManifestPath   string
}

// LoadEmbeddingTrainerPackage restores a trainer from a persisted training package.
func LoadEmbeddingTrainerPackage(artifactPath string) (*EmbeddingTrainer, error) {
	return LoadEmbeddingTrainerPackageWithPaths(EmbeddingTrainPackagePaths{
		ArtifactPath:          artifactPath,
		EmbeddingManifestPath: ResolveEmbeddingManifestPath(artifactPath),
		TokenizerPath:         DefaultTokenizerPath(artifactPath),
		WeightFilePath:        DefaultWeightFilePath(artifactPath),
		MemoryPlanPath:        DefaultMemoryPlanPath(artifactPath),
		TrainManifestPath:     ResolveEmbeddingTrainManifestPath(artifactPath),
		CheckpointPath:        DefaultEmbeddingCheckpointPath(artifactPath),
		TrainProfilePath:      DefaultEmbeddingTrainProfilePath(artifactPath),
		PackageManifestPath:   ResolvePackageManifestPath(artifactPath),
	})
}

// LoadEmbeddingTrainerPackageWithPaths restores a trainer from explicit package file paths.
func LoadEmbeddingTrainerPackageWithPaths(paths EmbeddingTrainPackagePaths) (*EmbeddingTrainer, error) {
	if paths.PackageManifestPath != "" {
		if _, err := os.Stat(paths.PackageManifestPath); err == nil {
			manifest, err := ReadPackageManifestFile(paths.PackageManifestPath)
			if err != nil {
				return nil, err
			}
			if err := manifest.VerifyFiles(map[string]string{
				"artifact":           paths.ArtifactPath,
				"embedding_manifest": paths.EmbeddingManifestPath,
				"tokenizer":          paths.TokenizerPath,
				"weights":            paths.WeightFilePath,
				"memory_plan":        paths.MemoryPlanPath,
				"train_manifest":     paths.TrainManifestPath,
				"checkpoint":         paths.CheckpointPath,
				"train_profile":      paths.TrainProfilePath,
			}); err != nil {
				return nil, err
			}
		}
	}
	mod, err := mantaartifact.ReadFile(paths.ArtifactPath)
	if err != nil {
		return nil, err
	}
	trainManifest, err := ReadEmbeddingTrainManifestFile(paths.TrainManifestPath)
	if err != nil {
		return nil, err
	}
	if err := trainManifest.ValidateModule(mod); err != nil {
		return nil, err
	}
	if paths.CheckpointPath != "" {
		if _, err := os.Stat(paths.CheckpointPath); err == nil {
			checkpoint, err := ReadEmbeddingTrainCheckpointFile(paths.CheckpointPath)
			if err != nil {
				return nil, err
			}
			trainer, err := NewEmbeddingTrainerFromCheckpoint(mod, checkpoint)
			if err != nil {
				return nil, err
			}
			if paths.MemoryPlanPath != "" {
				if _, err := os.Stat(paths.MemoryPlanPath); err == nil {
					plan, err := ReadMemoryPlanFile(paths.MemoryPlanPath)
					if err != nil {
						return nil, err
					}
					trainer.memoryPlan = cloneMemoryPlan(&plan)
				}
			}
			return trainer, nil
		}
	}
	weights, err := ReadWeightFile(paths.WeightFilePath)
	if err != nil {
		return nil, err
	}
	trainer, err := NewEmbeddingTrainer(mod, trainManifest.Embedding, weights.Weights, trainManifest.Config)
	if err != nil {
		return nil, err
	}
	if paths.MemoryPlanPath != "" {
		if _, err := os.Stat(paths.MemoryPlanPath); err == nil {
			plan, err := ReadMemoryPlanFile(paths.MemoryPlanPath)
			if err != nil {
				return nil, err
			}
			trainer.memoryPlan = cloneMemoryPlan(&plan)
		}
	}
	return trainer, nil
}
