package mantaruntime

import (
	"context"
	"encoding/json"
	"fmt"
	"os"

	mantaartifact "github.com/odvcencio/manta/artifact/manta"
	mll "github.com/odvcencio/mll"
)

// SealedEmbeddingPackage is a monolithic exported embedding package.
type SealedEmbeddingPackage struct {
	Module     *mantaartifact.Module
	Manifest   EmbeddingManifest
	Tokenizer  *TokenizerFile
	Weights    WeightFile
	MemoryPlan *MemoryPlan
}

// LoadSealedEmbeddingPackage loads an embedding model from a single sealed MLL export.
func (rt *Runtime) LoadSealedEmbeddingPackage(ctx context.Context, path string, opts ...LoadOption) (*EmbeddingModel, error) {
	pkg, err := ReadSealedEmbeddingPackage(path)
	if err != nil {
		return nil, err
	}
	loadOpts := make([]LoadOption, 0, len(opts)+len(pkg.Weights.Weights)+1)
	loadOpts = append(loadOpts, opts...)
	loadOpts = append(loadOpts, pkg.Weights.LoadOptions()...)
	if pkg.MemoryPlan != nil {
		loadOpts = append(loadOpts, WithMemoryPlan(*pkg.MemoryPlan))
	}
	model, err := rt.LoadEmbedding(ctx, pkg.Module, pkg.Manifest, loadOpts...)
	if err != nil {
		return nil, err
	}
	if pkg.Tokenizer != nil {
		if err := model.attachTokenizer(*pkg.Tokenizer); err != nil {
			return nil, err
		}
	}
	return model, nil
}

// ReadSealedEmbeddingPackage decodes a single-file sealed MLL embedding package.
func ReadSealedEmbeddingPackage(path string) (SealedEmbeddingPackage, error) {
	reader, meta, err := readSealedMantaMLL(path)
	if err != nil {
		return SealedEmbeddingPackage{}, err
	}
	pkg, ok, err := sealedEmbeddingPackageFromReader(reader, meta)
	if err != nil {
		return SealedEmbeddingPackage{}, err
	}
	if !ok {
		return SealedEmbeddingPackage{}, fmt.Errorf("sealed MLL does not contain an embedding package")
	}
	return pkg, nil
}

func readSealedMantaMLL(path string) (*mll.Reader, mantaartifact.MLLMetadata, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, mantaartifact.MLLMetadata{}, err
	}
	if !mantaartifact.IsMLLBytes(data) {
		return nil, mantaartifact.MLLMetadata{}, fmt.Errorf("%q is not an MLL file", path)
	}
	reader, err := mll.ReadBytes(data, mll.WithDigestVerification())
	if err != nil {
		return nil, mantaartifact.MLLMetadata{}, err
	}
	if reader.Profile() != mll.ProfileSealed {
		return nil, mantaartifact.MLLMetadata{}, fmt.Errorf("sealed package profile = %d, want %d", reader.Profile(), mll.ProfileSealed)
	}
	body, ok := reader.Section(mantaartifact.MLLTagXMTA)
	if !ok {
		return nil, mantaartifact.MLLMetadata{}, fmt.Errorf("sealed package missing XMTA metadata")
	}
	meta, err := mantaartifact.DecodeMLLMetadata(body)
	if err != nil {
		return nil, mantaartifact.MLLMetadata{}, err
	}
	return reader, meta, nil
}

func sealedEmbeddingPackageFromReader(reader *mll.Reader, meta mantaartifact.MLLMetadata) (SealedEmbeddingPackage, bool, error) {
	body, ok := meta.JSONFiles["embedding_manifest"]
	if !ok {
		return SealedEmbeddingPackage{}, false, nil
	}
	var manifest EmbeddingManifest
	if err := json.Unmarshal(body, &manifest); err != nil {
		return SealedEmbeddingPackage{}, false, fmt.Errorf("decode embedded embedding_manifest: %w", err)
	}
	mod, err := mantaartifact.DecodeJSON(meta.Artifact)
	if err != nil {
		return SealedEmbeddingPackage{}, false, err
	}
	weights, err := decodeWeightFileFromMLLReader(reader, WeightFileVersion, meta.LogicalTensorDType)
	if err != nil {
		return SealedEmbeddingPackage{}, false, err
	}
	var plan *MemoryPlan
	if body, ok := meta.JSONFiles["memory_plan"]; ok {
		var decoded MemoryPlan
		if err := json.Unmarshal(body, &decoded); err != nil {
			return SealedEmbeddingPackage{}, false, fmt.Errorf("decode embedded memory_plan: %w", err)
		}
		if err := decoded.Validate(); err != nil {
			return SealedEmbeddingPackage{}, false, err
		}
		plan = &decoded
	}
	var tokenizer *TokenizerFile
	if body, ok := meta.JSONFiles["tokenizer"]; ok {
		var decoded TokenizerFile
		if err := json.Unmarshal(body, &decoded); err != nil {
			return SealedEmbeddingPackage{}, false, fmt.Errorf("decode embedded tokenizer: %w", err)
		}
		if err := decoded.Validate(); err != nil {
			return SealedEmbeddingPackage{}, false, err
		}
		tokenizer = &decoded
	}
	return SealedEmbeddingPackage{
		Module:     mod,
		Manifest:   manifest,
		Tokenizer:  tokenizer,
		Weights:    weights,
		MemoryPlan: plan,
	}, true, nil
}
