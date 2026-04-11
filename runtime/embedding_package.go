package barruntime

import (
	"context"
	"os"

	"github.com/odvcencio/barracuda/artifact/barr"
)

// EmbeddingPackagePaths names the files that make up a packaged embedding model.
type EmbeddingPackagePaths struct {
	ArtifactPath        string
	ManifestPath        string
	TokenizerPath       string
	WeightFilePath      string
	MemoryPlanPath      string
	PackageManifestPath string
}

// LoadEmbeddingPackage loads a packaged embedding model from sibling artifact, manifest, and weight files.
func (rt *Runtime) LoadEmbeddingPackage(ctx context.Context, barrPath string) (*EmbeddingModel, error) {
	if model, ok, err := rt.tryLoadSealedEmbeddingPackage(ctx, barrPath); err != nil {
		return nil, err
	} else if ok {
		return model, nil
	}
	return rt.LoadEmbeddingPackageWithPaths(
		ctx,
		EmbeddingPackagePaths{
			ArtifactPath:        barrPath,
			ManifestPath:        ResolveEmbeddingManifestPath(barrPath),
			TokenizerPath:       DefaultTokenizerPath(barrPath),
			WeightFilePath:      DefaultWeightFilePath(barrPath),
			MemoryPlanPath:      DefaultMemoryPlanPath(barrPath),
			PackageManifestPath: ResolvePackageManifestPath(barrPath),
		},
	)
}

func (rt *Runtime) tryLoadSealedEmbeddingPackage(ctx context.Context, path string) (*EmbeddingModel, bool, error) {
	reader, meta, err := readSealedBarracudaMLL(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, false, nil
		}
		return nil, false, nil
	}
	if _, ok := meta.JSONFiles["embedding_manifest"]; !ok {
		return nil, false, nil
	}
	pkg, ok, err := sealedEmbeddingPackageFromReader(reader, meta)
	if err != nil || !ok {
		return nil, ok, err
	}
	opts := pkg.Weights.LoadOptions()
	if pkg.MemoryPlan != nil {
		opts = append(opts, WithMemoryPlan(*pkg.MemoryPlan))
	}
	model, err := rt.LoadEmbedding(ctx, pkg.Module, pkg.Manifest, opts...)
	if err != nil {
		return nil, true, err
	}
	return model, true, nil
}

// LoadEmbeddingPackageWithPaths loads a packaged embedding model from explicit artifact, manifest, and weight files.
func (rt *Runtime) LoadEmbeddingPackageWithPaths(ctx context.Context, paths EmbeddingPackagePaths) (*EmbeddingModel, error) {
	opts := make([]LoadOption, 0, 4)
	if paths.PackageManifestPath != "" {
		if _, err := os.Stat(paths.PackageManifestPath); err == nil {
			packageManifest, err := ReadPackageManifestFile(paths.PackageManifestPath)
			if err != nil {
				return nil, err
			}
			verifyPaths := map[string]string{
				"artifact":           paths.ArtifactPath,
				"embedding_manifest": paths.ManifestPath,
				"tokenizer":          paths.TokenizerPath,
				"weights":            paths.WeightFilePath,
				"memory_plan":        paths.MemoryPlanPath,
			}
			if packageManifest.Kind == PackageTraining {
				verifyPaths["train_manifest"] = DefaultEmbeddingTrainManifestPath(paths.ArtifactPath)
				verifyPaths["checkpoint"] = DefaultEmbeddingCheckpointPath(paths.ArtifactPath)
				verifyPaths["train_profile"] = DefaultEmbeddingTrainProfilePath(paths.ArtifactPath)
			}
			if err := packageManifest.VerifyFiles(verifyPaths); err != nil {
				return nil, err
			}
			opts = append(opts, WithPackageManifest(packageManifest))
		}
	}
	manifest, err := ReadEmbeddingManifestFile(paths.ManifestPath)
	if err != nil {
		return nil, err
	}
	weightFile, err := ReadWeightFile(paths.WeightFilePath)
	if err != nil {
		return nil, err
	}
	mod, err := barr.ReadFile(paths.ArtifactPath)
	if err != nil {
		return nil, err
	}
	opts = append(opts, weightFile.LoadOptions()...)
	if paths.MemoryPlanPath != "" {
		if _, err := os.Stat(paths.MemoryPlanPath); err == nil {
			plan, err := ReadMemoryPlanFile(paths.MemoryPlanPath)
			if err != nil {
				return nil, err
			}
			opts = append(opts, WithMemoryPlan(plan))
		}
	}
	return rt.LoadEmbedding(ctx, mod, manifest, opts...)
}
