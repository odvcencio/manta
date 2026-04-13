package mantaruntime

import (
	"fmt"
	"math/rand"
	"os"

	mantaartifact "github.com/odvcencio/manta/artifact/manta"
	"github.com/odvcencio/manta/runtime/backend"
)

func SyncEmbeddingTokenizerVocab(artifactPath string, vocabSize int) error {
	if vocabSize <= 0 {
		return fmt.Errorf("tokenizer vocab size must be positive")
	}
	var tokenParam string
	if manifestPath := ResolveEmbeddingManifestPath(artifactPath); fileExists(manifestPath) {
		manifest, err := ReadEmbeddingManifestFile(manifestPath)
		if err != nil {
			return err
		}
		manifest.Tokenizer.VocabSize = vocabSize
		tokenParam = manifest.TokenEmbeddingParam
		if err := manifest.WriteFile(manifestPath); err != nil {
			return err
		}
	}
	if trainManifestPath := ResolveEmbeddingTrainManifestPath(artifactPath); fileExists(trainManifestPath) {
		manifest, err := ReadEmbeddingTrainManifestFile(trainManifestPath)
		if err != nil {
			return err
		}
		manifest.Embedding.Tokenizer.VocabSize = vocabSize
		if tokenParam == "" {
			tokenParam = manifest.Embedding.TokenEmbeddingParam
		}
		if err := manifest.WriteFile(trainManifestPath); err != nil {
			return err
		}
	}
	if tokenParam == "" {
		return fmt.Errorf("token embedding param is not available for tokenizer sync")
	}
	if weightPath := DefaultWeightFilePath(artifactPath); fileExists(weightPath) {
		weights, err := ReadWeightFile(weightPath)
		if err != nil {
			return err
		}
		tensor, ok := weights.Weights[tokenParam]
		if !ok {
			return fmt.Errorf("weight file missing token embedding %q", tokenParam)
		}
		weights.Weights[tokenParam] = expandTensorLeadingRows(tensor, vocabSize, false)
		if err := weights.WriteFile(weightPath); err != nil {
			return err
		}
		if memoryPlanPath := DefaultMemoryPlanPath(artifactPath); fileExists(memoryPlanPath) {
			mod, err := mantaartifact.ReadFile(artifactPath)
			if err != nil {
				return err
			}
			plan := NewMemoryPlan(mod, weights.Weights, MemoryPlanOptions{})
			if err := plan.WriteFile(memoryPlanPath); err != nil {
				return err
			}
		}
	}
	if checkpointPath := DefaultEmbeddingCheckpointPath(artifactPath); fileExists(checkpointPath) {
		checkpoint, err := ReadEmbeddingTrainCheckpointFile(checkpointPath)
		if err != nil {
			return err
		}
		checkpoint.Manifest.Tokenizer.VocabSize = vocabSize
		checkpoint.TokenEmbedding = expandTensorLeadingRows(checkpoint.TokenEmbedding, vocabSize, false)
		checkpoint.TokenMoment1 = expandTensorLeadingRows(checkpoint.TokenMoment1, vocabSize, true)
		checkpoint.TokenMoment2 = expandTensorLeadingRows(checkpoint.TokenMoment2, vocabSize, true)
		if err := checkpoint.WriteFile(checkpointPath); err != nil {
			return err
		}
	}
	if packageManifestPath := ResolvePackageManifestPath(artifactPath); fileExists(packageManifestPath) {
		if _, _, err := RebuildSiblingPackageManifest(artifactPath); err != nil {
			return err
		}
	}
	return nil
}

func fileExists(path string) bool {
	if path == "" {
		return false
	}
	_, err := os.Stat(path)
	return err == nil
}

func expandTensorLeadingRows(t *backend.Tensor, rows int, zeroFill bool) *backend.Tensor {
	if t == nil || len(t.Shape) == 0 || t.Shape[0] >= rows {
		return t
	}
	rowWidth := 1
	for _, dim := range t.Shape[1:] {
		if dim <= 0 {
			return t
		}
		rowWidth *= dim
	}
	newShape := append([]int(nil), t.Shape...)
	oldRows := newShape[0]
	newShape[0] = rows
	switch t.DType {
	case "f16", "f32", "q4", "q8":
		data := make([]float32, rows*rowWidth)
		copy(data, t.F32)
		if !zeroFill {
			rng := rand.New(rand.NewSource(int64(rows*rowWidth + oldRows)))
			scale := initializerScale(newShape)
			for i := oldRows * rowWidth; i < len(data); i++ {
				data[i] = (rng.Float32()*2 - 1) * scale
			}
		}
		switch t.DType {
		case "f16":
			return backend.NewTensorF16(newShape, data)
		case "f32":
			return backend.NewTensorF32(newShape, data)
		case "q4":
			return backend.NewTensorQ4(newShape, data)
		case "q8":
			return backend.NewTensorQ8(newShape, data)
		}
	case "i32":
		data := make([]int32, rows*rowWidth)
		copy(data, t.I32)
		return backend.NewTensorI32(newShape, data)
	case "i64":
		data := make([]int64, rows*rowWidth)
		copy(data, t.I64)
		return backend.NewTensorI64(newShape, data)
	}
	return t
}
