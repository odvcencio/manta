package models

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/odvcencio/manta/artifact/barr"
	"github.com/odvcencio/manta/compiler"
	barruntime "github.com/odvcencio/manta/runtime"
)

const DefaultEmbeddingModelName = "manta-embed-v1"

// DefaultEmbeddingPackageConfig controls Manta's built-in default
// embedding-model package initialization.
type DefaultEmbeddingPackageConfig struct {
	Name            string
	VocabSize       int
	MaxSequence     int
	EmbeddingDim    int
	HiddenDim       int
	Seed            int64
	LearningRate    float32
	WeightDecay     float32
	WeightBits      int
	Optimizer       string
	ContrastiveLoss string
	Temperature     float32
}

// InitDefaultEmbeddingPackage compiles Manta's default trainable embedding
// shape and writes a native training package ready for train-corpus/train-embed.
func InitDefaultEmbeddingPackage(path string, cfg DefaultEmbeddingPackageConfig) (barruntime.EmbeddingTrainPackagePaths, error) {
	if path == "" {
		return barruntime.EmbeddingTrainPackagePaths{}, fmt.Errorf("output artifact path is required")
	}
	cfg = cfg.normalized()
	if err := cfg.validate(); err != nil {
		return barruntime.EmbeddingTrainPackagePaths{}, err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return barruntime.EmbeddingTrainPackagePaths{}, err
	}

	moduleName := moduleNameForModel(cfg.Name)
	bundle, err := compiler.Build(nil, compiler.Options{
		ModuleName: moduleName,
		Preset:     compiler.PresetEncoderTrainableQ8x2,
	})
	if err != nil {
		return barruntime.EmbeddingTrainPackagePaths{}, err
	}
	if err := barr.WriteFile(path, bundle.Artifact); err != nil {
		return barruntime.EmbeddingTrainPackagePaths{}, err
	}

	manifest := DefaultEmbeddingManifest(cfg)
	if err := manifest.WriteFile(barruntime.DefaultEmbeddingManifestPath(path)); err != nil {
		return barruntime.EmbeddingTrainPackagePaths{}, err
	}
	return barruntime.InitializeEmbeddingTrainerPackageWithManifest(path, manifest, cfg.trainConfig(), barruntime.EmbeddingTrainInitOptions{
		Seed: cfg.Seed,
		ShapeSizes: map[string]int{
			"D": cfg.EmbeddingDim,
			"H": cfg.HiddenDim,
		},
	})
}

// DefaultEmbeddingManifest returns the serving/training contract for the
// built-in default embedding model shape.
func DefaultEmbeddingManifest(cfg DefaultEmbeddingPackageConfig) barruntime.EmbeddingManifest {
	cfg = cfg.normalized()
	return barruntime.EmbeddingManifest{
		Name:                  cfg.Name,
		PooledEntry:           "embed_pooled",
		BatchEntry:            "embed_pooled_batch",
		EncoderRepeats:        2,
		TokenInput:            "tokens",
		MaskInput:             "attention_mask",
		OutputName:            "result",
		OutputDType:           "f16",
		TokenEmbeddingParam:   "token_embedding",
		AttentionQueryParam:   "attn_q",
		AttentionKeyParam:     "attn_k",
		AttentionValueParam:   "attn_v",
		AttentionOutputParam:  "attn_o",
		AttentionResidual:     true,
		AttentionLayerNorm:    true,
		HiddenProjectionParam: "ffn_up",
		FFNResidual:           true,
		FFNLayerNorm:          true,
		ProjectionParam:       "projection",
		Tokenizer: barruntime.TokenizerManifest{
			VocabSize:   cfg.VocabSize,
			MaxSequence: cfg.MaxSequence,
			PadID:       0,
			BOSID:       1,
			EOSID:       2,
			UnknownID:   3,
		},
	}
}

func (cfg DefaultEmbeddingPackageConfig) normalized() DefaultEmbeddingPackageConfig {
	if cfg.Name == "" {
		cfg.Name = DefaultEmbeddingModelName
	}
	if cfg.VocabSize == 0 {
		cfg.VocabSize = 32768
	}
	if cfg.MaxSequence == 0 {
		cfg.MaxSequence = 256
	}
	if cfg.EmbeddingDim == 0 {
		cfg.EmbeddingDim = 128
	}
	if cfg.HiddenDim == 0 {
		cfg.HiddenDim = cfg.EmbeddingDim * 2
	}
	if cfg.Seed == 0 {
		cfg.Seed = 1
	}
	if cfg.LearningRate == 0 {
		cfg.LearningRate = 0.005
	}
	if cfg.WeightBits == 0 {
		cfg.WeightBits = 8
	}
	if cfg.Optimizer == "" {
		cfg.Optimizer = "adamw"
	}
	if cfg.ContrastiveLoss == "" {
		cfg.ContrastiveLoss = "infonce"
	}
	if cfg.Temperature == 0 {
		cfg.Temperature = 0.05
	}
	return cfg
}

func (cfg DefaultEmbeddingPackageConfig) validate() error {
	if cfg.VocabSize <= 0 {
		return fmt.Errorf("vocab size must be positive")
	}
	if cfg.MaxSequence <= 0 {
		return fmt.Errorf("max sequence must be positive")
	}
	if cfg.EmbeddingDim <= 0 {
		return fmt.Errorf("embedding dim must be positive")
	}
	if cfg.HiddenDim <= 0 {
		return fmt.Errorf("hidden dim must be positive")
	}
	if cfg.LearningRate <= 0 {
		return fmt.Errorf("learning rate must be positive")
	}
	if cfg.WeightBits <= 0 {
		return fmt.Errorf("weight bits must be positive")
	}
	if cfg.Temperature <= 0 {
		return fmt.Errorf("temperature must be positive")
	}
	return nil
}

func (cfg DefaultEmbeddingPackageConfig) trainConfig() barruntime.EmbeddingTrainConfig {
	return barruntime.EmbeddingTrainConfig{
		LearningRate:    cfg.LearningRate,
		WeightDecay:     cfg.WeightDecay,
		WeightBits:      cfg.WeightBits,
		Optimizer:       cfg.Optimizer,
		ContrastiveLoss: cfg.ContrastiveLoss,
		Temperature:     cfg.Temperature,
	}
}

func moduleNameForModel(name string) string {
	name = strings.ToLower(strings.TrimSpace(name))
	var b strings.Builder
	lastUnderscore := false
	for _, r := range name {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') {
			b.WriteRune(r)
			lastUnderscore = false
			continue
		}
		if !lastUnderscore {
			b.WriteByte('_')
			lastUnderscore = true
		}
	}
	out := strings.Trim(b.String(), "_")
	if out == "" {
		return "manta_model"
	}
	return out
}
