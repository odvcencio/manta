package barruntime

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/odvcencio/barracuda/artifact/barr"
)

const EmbeddingTrainManifestVersion = "barr/train-manifest/v0alpha1"

// EmbeddingTrainManifest describes the native training contract for an embedding module.
type EmbeddingTrainManifest struct {
	Name      string               `json:"name,omitempty"`
	Embedding EmbeddingManifest    `json:"embedding"`
	Config    EmbeddingTrainConfig `json:"config"`
}

// DefaultEmbeddingTrainManifestPath returns the conventional sibling train-manifest path for a .barr artifact.
func DefaultEmbeddingTrainManifestPath(barrPath string) string {
	return defaultManifestPath(barrPath, ".train.mll")
}

func ResolveEmbeddingTrainManifestPath(barrPath string) string {
	return resolveSiblingPath(barrPath, ".train.mll", ".train.json")
}

// ReadEmbeddingTrainManifestFile decodes a JSON training manifest.
func ReadEmbeddingTrainManifestFile(path string) (EmbeddingTrainManifest, error) {
	if doc, err := readAuthoredManifestMLL(path, "train_manifest", EmbeddingTrainManifestVersion); err == nil {
		return embeddingTrainManifestFromDoc(doc)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return EmbeddingTrainManifest{}, err
	}
	var manifest EmbeddingTrainManifest
	if err := json.Unmarshal(data, &manifest); err != nil {
		return EmbeddingTrainManifest{}, err
	}
	return manifest, nil
}

// WriteFile writes the training manifest as an authored MLL container.
func (m EmbeddingTrainManifest) WriteFile(path string) error {
	return writeAuthoredManifestMLL(path, "train_manifest", EmbeddingTrainManifestVersion, m.nameOrDefault(), "Barracuda training manifest", m.mllValues())
}

func (m EmbeddingTrainManifest) normalized() EmbeddingTrainManifest {
	m.Embedding = m.Embedding.normalized()
	return m
}

// ValidateModule checks that a module satisfies the training contract.
func (m EmbeddingTrainManifest) ValidateModule(mod *barr.Module) error {
	m = m.normalized()
	if mod == nil {
		return fmt.Errorf("nil module")
	}
	if err := m.Embedding.ValidateModule(mod); err != nil {
		return err
	}
	if _, err := requireTrainableEmbeddingParam(mod, m.Embedding.TokenEmbeddingParam); err != nil {
		return err
	}
	if err := validateTrainableAttentionParams(mod, m.Embedding); err != nil {
		return err
	}
	if m.Embedding.HiddenProjectionParam != "" {
		if _, err := requireTrainableEmbeddingParam(mod, m.Embedding.HiddenProjectionParam); err != nil {
			return err
		}
	}
	if _, err := requireTrainableEmbeddingParam(mod, m.Embedding.ProjectionParam); err != nil {
		return err
	}
	return nil
}

func validateTrainableAttentionParams(mod *barr.Module, manifest EmbeddingManifest) error {
	names := []string{
		manifest.AttentionQueryParam,
		manifest.AttentionKeyParam,
		manifest.AttentionValueParam,
		manifest.AttentionOutputParam,
	}
	set := 0
	for _, name := range names {
		if name != "" {
			set++
		}
	}
	if set == 0 {
		return nil
	}
	if set != len(names) {
		return fmt.Errorf("attention params must declare query, key, value, and output together")
	}
	for _, name := range names {
		if _, err := requireTrainableEmbeddingParam(mod, name); err != nil {
			return err
		}
	}
	return nil
}

func (m EmbeddingTrainManifest) nameOrDefault() string {
	if m.Name != "" {
		return m.Name
	}
	if m.Embedding.Name != "" {
		return m.Embedding.Name
	}
	return "train_manifest"
}

func (m EmbeddingTrainManifest) mllValues() map[string]authoredManifestValue {
	values := map[string]authoredManifestValue{
		"name":                    authoredString(m.Name),
		"config.optimizer":        authoredString(m.Config.Optimizer),
		"config.weight_bits":      authoredInt(int64(m.Config.WeightBits)),
		"config.learning_rate":    authoredFloat(float64(m.Config.LearningRate)),
		"config.weight_decay":     authoredFloat(float64(m.Config.WeightDecay)),
		"config.beta1":            authoredFloat(float64(m.Config.Beta1)),
		"config.beta2":            authoredFloat(float64(m.Config.Beta2)),
		"config.epsilon":          authoredFloat(float64(m.Config.Epsilon)),
		"config.contrastive_loss": authoredString(m.Config.ContrastiveLoss),
		"config.temperature":      authoredFloat(float64(m.Config.Temperature)),
	}
	for key, value := range m.Embedding.mllValues() {
		values["embedding."+key] = value
	}
	return values
}

func embeddingTrainManifestFromDoc(doc authoredManifestDoc) (EmbeddingTrainManifest, error) {
	var manifest EmbeddingTrainManifest
	var err error
	if manifest.Name, _, err = doc.string("name"); err != nil {
		return EmbeddingTrainManifest{}, err
	}
	embeddingDoc := authoredManifestDoc{values: map[string]authoredManifestValue{}}
	for key, value := range doc.values {
		if len(key) > len("embedding.") && key[:len("embedding.")] == "embedding." {
			embeddingDoc.values[key[len("embedding."):]] = value
		}
	}
	manifest.Embedding, err = embeddingManifestFromDoc(embeddingDoc)
	if err != nil {
		return EmbeddingTrainManifest{}, err
	}
	if manifest.Config.Optimizer, _, err = doc.string("config.optimizer"); err != nil {
		return EmbeddingTrainManifest{}, err
	}
	if value, _, err := doc.int("config.weight_bits"); err != nil {
		return EmbeddingTrainManifest{}, err
	} else {
		manifest.Config.WeightBits = int(value)
	}
	if value, _, err := doc.float("config.learning_rate"); err != nil {
		return EmbeddingTrainManifest{}, err
	} else {
		manifest.Config.LearningRate = float32(value)
	}
	if value, _, err := doc.float("config.weight_decay"); err != nil {
		return EmbeddingTrainManifest{}, err
	} else {
		manifest.Config.WeightDecay = float32(value)
	}
	if value, _, err := doc.float("config.beta1"); err != nil {
		return EmbeddingTrainManifest{}, err
	} else {
		manifest.Config.Beta1 = float32(value)
	}
	if value, _, err := doc.float("config.beta2"); err != nil {
		return EmbeddingTrainManifest{}, err
	} else {
		manifest.Config.Beta2 = float32(value)
	}
	if value, _, err := doc.float("config.epsilon"); err != nil {
		return EmbeddingTrainManifest{}, err
	} else {
		manifest.Config.Epsilon = float32(value)
	}
	if manifest.Config.ContrastiveLoss, _, err = doc.string("config.contrastive_loss"); err != nil {
		return EmbeddingTrainManifest{}, err
	}
	if value, _, err := doc.float("config.temperature"); err != nil {
		return EmbeddingTrainManifest{}, err
	} else {
		manifest.Config.Temperature = float32(value)
	}
	return manifest, nil
}
