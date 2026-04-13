package mantaruntime

import (
	"fmt"
	"math"
	"math/rand"
	"strconv"

	mantaartifact "github.com/odvcencio/manta/artifact/manta"
	"github.com/odvcencio/manta/runtime/backend"
)

// EmbeddingTrainInitOptions controls training-package initialization from an artifact.
type EmbeddingTrainInitOptions struct {
	Seed       int64
	ShapeSizes map[string]int
}

// InitializeEmbeddingTrainerPackage reads an artifact plus its sibling embedding manifest, initializes trainable weights, and writes a training package.
func InitializeEmbeddingTrainerPackage(artifactPath string, opts EmbeddingTrainInitOptions) (EmbeddingTrainPackagePaths, error) {
	manifest, err := ReadEmbeddingManifestFile(ResolveEmbeddingManifestPath(artifactPath))
	if err != nil {
		return EmbeddingTrainPackagePaths{}, err
	}
	return InitializeEmbeddingTrainerPackageWithManifest(artifactPath, manifest, EmbeddingTrainConfig{}, opts)
}

// InitializeEmbeddingTrainerPackageWithManifest initializes a training package from an explicit embedding manifest and trainer config.
func InitializeEmbeddingTrainerPackageWithManifest(artifactPath string, manifest EmbeddingManifest, cfg EmbeddingTrainConfig, opts EmbeddingTrainInitOptions) (EmbeddingTrainPackagePaths, error) {
	mod, err := mantaartifact.ReadFile(artifactPath)
	if err != nil {
		return EmbeddingTrainPackagePaths{}, err
	}
	trainManifest := EmbeddingTrainManifest{
		Name:      manifest.Name,
		Embedding: manifest,
		Config:    cfg,
	}
	if err := trainManifest.ValidateModule(mod); err != nil {
		return EmbeddingTrainPackagePaths{}, err
	}
	weights, err := initializedTrainingWeights(mod, manifest.Tokenizer, opts)
	if err != nil {
		return EmbeddingTrainPackagePaths{}, err
	}
	trainer, err := NewEmbeddingTrainer(mod, manifest, weights, cfg)
	if err != nil {
		return EmbeddingTrainPackagePaths{}, err
	}
	return trainer.WriteTrainingPackage(artifactPath)
}

func initializedTrainingWeights(mod *mantaartifact.Module, tokenizer TokenizerManifest, opts EmbeddingTrainInitOptions) (map[string]*backend.Tensor, error) {
	if mod == nil {
		return nil, fmt.Errorf("nil module")
	}
	seed := opts.Seed
	if seed == 0 {
		seed = 1
	}
	rng := rand.New(rand.NewSource(seed))
	weights := make(map[string]*backend.Tensor, len(mod.Params))
	for _, param := range mod.Params {
		if param.Type.Kind != mantaartifact.ValueTensor || param.Type.Tensor == nil {
			return nil, fmt.Errorf("param %q is not a tensor weight", param.Name)
		}
		shape, err := resolveTrainingInitShape(param.Type.Tensor.Shape, tokenizer, opts.ShapeSizes)
		if err != nil {
			return nil, fmt.Errorf("param %q: %w", param.Name, err)
		}
		tensor, err := initializedWeightTensor(param.Type.Tensor.DType, shape, rng)
		if err != nil {
			return nil, fmt.Errorf("param %q: %w", param.Name, err)
		}
		weights[param.Name] = tensor
	}
	return weights, nil
}

func resolveTrainingInitShape(shape []string, tokenizer TokenizerManifest, sizes map[string]int) ([]int, error) {
	out := make([]int, len(shape))
	for i, dim := range shape {
		if n, err := strconv.Atoi(dim); err == nil {
			if n <= 0 {
				return nil, fmt.Errorf("shape dim %q must be positive", dim)
			}
			out[i] = n
			continue
		}
		if sizes != nil && sizes[dim] > 0 {
			out[i] = sizes[dim]
			continue
		}
		switch dim {
		case "V":
			if tokenizer.VocabSize > 0 {
				out[i] = tokenizer.VocabSize
				continue
			}
		case "T":
			if tokenizer.MaxSequence > 0 {
				out[i] = tokenizer.MaxSequence
				continue
			}
		}
		return nil, fmt.Errorf("unresolved symbolic dim %q", dim)
	}
	return out, nil
}

func initializedWeightTensor(dtype string, shape []int, rng *rand.Rand) (*backend.Tensor, error) {
	if rng == nil {
		rng = rand.New(rand.NewSource(1))
	}
	n := 1
	for _, dim := range shape {
		if dim <= 0 {
			return nil, fmt.Errorf("shape %v is invalid", shape)
		}
		n *= dim
	}
	scale := initializerScale(shape)
	data := make([]float32, n)
	for i := range data {
		data[i] = (rng.Float32()*2 - 1) * scale
	}
	switch dtype {
	case "f16":
		return backend.NewTensorF16(shape, data), nil
	case "f32":
		return backend.NewTensorF32(shape, data), nil
	case "q4":
		return backend.NewTensorQ4(shape, data), nil
	case "q8":
		return backend.NewTensorQ8(shape, data), nil
	default:
		return nil, fmt.Errorf("unsupported weight dtype %q", dtype)
	}
}

func initializerScale(shape []int) float32 {
	if len(shape) >= 2 {
		fanIn := shape[len(shape)-2]
		fanOut := shape[len(shape)-1]
		if fanIn > 0 && fanOut > 0 {
			return float32(math.Sqrt(2.0 / float64(fanIn+fanOut)))
		}
	}
	if len(shape) == 1 && shape[0] > 0 {
		return float32(1.0 / math.Sqrt(float64(shape[0])))
	}
	return 0.02
}
