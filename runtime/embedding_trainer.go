package barruntime

import (
	"fmt"
	"math"
	"os"
	"sort"
	"strings"

	"github.com/odvcencio/manta/artifact/barr"
	"github.com/odvcencio/manta/runtime/backend"
)

// EmbeddingPairExample is one supervised pairwise training example.
type EmbeddingPairExample struct {
	LeftTokens  []int32
	RightTokens []int32
	LeftMask    []int32
	RightMask   []int32
	Target      float32
}

// EmbeddingTrainConfig controls the narrow pooled-embedder trainer.
type EmbeddingTrainConfig struct {
	LearningRate    float32
	WeightDecay     float32
	WeightBits      int
	Optimizer       string
	Beta1           float32
	Beta2           float32
	Epsilon         float32
	ContrastiveLoss string
	Temperature     float32
}

// EmbeddingTrainMetrics summarizes one training/eval batch.
type EmbeddingTrainMetrics struct {
	Loss         float32
	AverageScore float32
	BatchSize    int
}

// EmbeddingEvalMetrics summarizes retrieval-oriented quality on pairwise eval data.
type EmbeddingEvalMetrics struct {
	Loss               float32
	AverageScore       float32
	PositiveMeanScore  float32
	NegativeMeanScore  float32
	PairAccuracy       float32
	ThresholdAccuracy  float32
	ScoreThreshold     float32
	ROCAUC             float32
	ScoreMargin        float32
	Top1Accuracy       float32
	Top5Accuracy       float32
	Top10Accuracy      float32
	MeanReciprocalRank float32
	MeanPositiveRank   float32
	PairCount          int
	PositiveCount      int
	NegativeCount      int
}

// EmbeddingForwardResidencyStats summarizes trainer-level bind suppression plus backend prep activity.
type EmbeddingForwardResidencyStats struct {
	BindSkips int64
	MatMul    backend.MatMulAcceleratorStats
}

type embeddingEvalScore struct {
	Score    float32
	Positive bool
}

// EmbeddingTrainer trains a pooled embedding model with quantization-aware forward passes.
type EmbeddingTrainer struct {
	module               *barr.Module
	manifest             EmbeddingManifest
	config               EmbeddingTrainConfig
	memoryPlan           *MemoryPlan
	step                 int
	tokenParam           barr.Param
	attnQParam           barr.Param
	attnKParam           barr.Param
	attnVParam           barr.Param
	attnOParam           barr.Param
	hiddenParam          barr.Param
	projParam            barr.Param
	tokenEmbed           *backend.Tensor
	attentionQuery       *backend.Tensor
	attentionKey         *backend.Tensor
	attentionValue       *backend.Tensor
	attentionOutput      *backend.Tensor
	hiddenProjection     *backend.Tensor
	projection           *backend.Tensor
	tokenMom1            *backend.Tensor
	tokenMom2            *backend.Tensor
	attnQMom1            *backend.Tensor
	attnQMom2            *backend.Tensor
	attnKMom1            *backend.Tensor
	attnKMom2            *backend.Tensor
	attnVMom1            *backend.Tensor
	attnVMom2            *backend.Tensor
	attnOMom1            *backend.Tensor
	attnOMom2            *backend.Tensor
	hiddenMom1           *backend.Tensor
	hiddenMom2           *backend.Tensor
	projMom1             *backend.Tensor
	projMom2             *backend.Tensor
	forwardMatMul        backend.MatMulAccelerator
	forwardBackend       barr.BackendKind
	optimizerAccel       backend.OptimizerAccelerator
	optimizerBackend     barr.BackendKind
	activationAccel      backend.ActivationAccelerator
	activationBackend    barr.BackendKind
	activationAccelFull  bool
	softmaxBackwardAccel bool
	contrastiveAccel     backend.ContrastiveAccelerator
	contrastiveBackend   barr.BackendKind
	sequenceBindingID    int
	momentsDirty         bool
	forwardCache         *embeddingForwardWeights
	boundForward         embeddingForwardWeights
	forwardDirty         bool
	forwardNeedsBind     bool
	forwardBindSkips     int64
	scratchF32           [][]float32
}

type embeddingSequenceState struct {
	tokens              []int32
	mask                []int32
	activeCount         int
	input               []float32
	inputBinding        string
	hidden              []float32
	hiddenBinding       string
	attnQ               []float32
	attnK               []float32
	attnV               []float32
	bindingPrefix       string
	attnQBinding        string
	attnKBinding        string
	attnVBinding        string
	attnScores          []float32
	attnScoresBinding   string
	attnMixed           []float32
	attnMixedBinding    string
	attnOutput          []float32
	attnResidual        []float32
	attnResidualBinding string
	ffnHidden           []float32
	ffnHiddenBinding    string
	activated           []float32
	activatedBinding    string
	ffnOutput           []float32
	ffnResidual         []float32
	ffnResidualBinding  string
	projected           []float32
	projectedBinding    string
	normalized          []float32
	pooled              []float32
	skipUnboundActAccel bool
}

type embeddingEncodedSequence struct {
	layers []*embeddingSequenceState
	pooled []float32
	tokens []int32
}

func (s *embeddingEncodedSequence) finalLayer() *embeddingSequenceState {
	if s == nil || len(s.layers) == 0 {
		return nil
	}
	return s.layers[len(s.layers)-1]
}

type embeddingForwardWeights struct {
	token  *backend.Tensor
	attnQ  *backend.Tensor
	attnK  *backend.Tensor
	attnV  *backend.Tensor
	attnO  *backend.Tensor
	hidden *backend.Tensor
	proj   *backend.Tensor
}

// NewEmbeddingTrainer constructs the first native pooled-embedder trainer.
func NewEmbeddingTrainer(mod *barr.Module, manifest EmbeddingManifest, weights map[string]*backend.Tensor, cfg EmbeddingTrainConfig) (*EmbeddingTrainer, error) {
	manifest = manifest.normalized()
	if err := manifest.ValidateModule(mod); err != nil {
		return nil, err
	}
	tokenParam, err := requireTrainableEmbeddingParam(mod, manifest.TokenEmbeddingParam)
	if err != nil {
		return nil, err
	}
	attnQParam, attnEnabled, err := optionalTrainableEmbeddingParam(mod, manifest.AttentionQueryParam)
	if err != nil {
		return nil, err
	}
	attnKParam, _, err := optionalTrainableEmbeddingParam(mod, manifest.AttentionKeyParam)
	if err != nil {
		return nil, err
	}
	attnVParam, _, err := optionalTrainableEmbeddingParam(mod, manifest.AttentionValueParam)
	if err != nil {
		return nil, err
	}
	attnOParam, _, err := optionalTrainableEmbeddingParam(mod, manifest.AttentionOutputParam)
	if err != nil {
		return nil, err
	}
	hiddenParam, hiddenEnabled, err := optionalTrainableEmbeddingParam(mod, manifest.HiddenProjectionParam)
	if err != nil {
		return nil, err
	}
	if (manifest.AttentionResidual || manifest.AttentionLayerNorm) && !attnEnabled {
		return nil, fmt.Errorf("attention residual/layernorm requires attention weights")
	}
	if (manifest.FFNResidual || manifest.FFNLayerNorm) && !hiddenEnabled {
		return nil, fmt.Errorf("ffn residual/layernorm requires hidden projection weights")
	}
	projParam, err := requireTrainableEmbeddingParam(mod, manifest.ProjectionParam)
	if err != nil {
		return nil, err
	}
	tokenEmbed, ok := weights[tokenParam.Name]
	if !ok || tokenEmbed == nil {
		return nil, fmt.Errorf("missing training weight %q", tokenParam.Name)
	}
	projection, ok := weights[projParam.Name]
	if !ok || projection == nil {
		return nil, fmt.Errorf("missing training weight %q", projParam.Name)
	}
	var attentionQuery *backend.Tensor
	var attentionKey *backend.Tensor
	var attentionValue *backend.Tensor
	var attentionOutput *backend.Tensor
	if attnEnabled {
		attentionQuery, ok = weights[attnQParam.Name]
		if !ok || attentionQuery == nil {
			return nil, fmt.Errorf("missing training weight %q", attnQParam.Name)
		}
		attentionKey, ok = weights[attnKParam.Name]
		if !ok || attentionKey == nil {
			return nil, fmt.Errorf("missing training weight %q", attnKParam.Name)
		}
		attentionValue, ok = weights[attnVParam.Name]
		if !ok || attentionValue == nil {
			return nil, fmt.Errorf("missing training weight %q", attnVParam.Name)
		}
		attentionOutput, ok = weights[attnOParam.Name]
		if !ok || attentionOutput == nil {
			return nil, fmt.Errorf("missing training weight %q", attnOParam.Name)
		}
	}
	var hiddenProjection *backend.Tensor
	if hiddenEnabled {
		hiddenProjection, ok = weights[hiddenParam.Name]
		if !ok || hiddenProjection == nil {
			return nil, fmt.Errorf("missing training weight %q", hiddenParam.Name)
		}
	}
	if len(tokenEmbed.Shape) != 2 {
		return nil, fmt.Errorf("training weight %q rank = %d, want 2", tokenParam.Name, len(tokenEmbed.Shape))
	}
	if attnEnabled {
		for name, tensor := range map[string]*backend.Tensor{
			attnQParam.Name: attentionQuery,
			attnKParam.Name: attentionKey,
			attnVParam.Name: attentionValue,
			attnOParam.Name: attentionOutput,
		} {
			if len(tensor.Shape) != 2 {
				return nil, fmt.Errorf("training weight %q rank = %d, want 2", name, len(tensor.Shape))
			}
			if tensor.Shape[0] != tokenEmbed.Shape[1] || tensor.Shape[1] != tokenEmbed.Shape[1] {
				return nil, fmt.Errorf("attention weight %q shape %v does not match embedding width %d", name, tensor.Shape, tokenEmbed.Shape[1])
			}
		}
	}
	if hiddenEnabled && len(hiddenProjection.Shape) != 2 {
		return nil, fmt.Errorf("training weight %q rank = %d, want 2", hiddenParam.Name, len(hiddenProjection.Shape))
	}
	if len(projection.Shape) != 2 {
		return nil, fmt.Errorf("training weight %q rank = %d, want 2", projParam.Name, len(projection.Shape))
	}
	if hiddenEnabled {
		if tokenEmbed.Shape[1] != hiddenProjection.Shape[0] {
			return nil, fmt.Errorf("embedding/hidden projection shape mismatch %v x %v", tokenEmbed.Shape, hiddenProjection.Shape)
		}
		if hiddenProjection.Shape[1] != projection.Shape[0] {
			return nil, fmt.Errorf("hidden/output projection shape mismatch %v x %v", hiddenProjection.Shape, projection.Shape)
		}
		if manifest.FFNResidual && projection.Shape[1] != tokenEmbed.Shape[1] {
			return nil, fmt.Errorf("ffn residual requires projection output width %d to match hidden width %d", projection.Shape[1], tokenEmbed.Shape[1])
		}
	} else if tokenEmbed.Shape[1] != projection.Shape[0] {
		return nil, fmt.Errorf("embedding/projection shape mismatch %v x %v", tokenEmbed.Shape, projection.Shape)
	}
	if vocab := manifest.Tokenizer.VocabSize; vocab > 0 && tokenEmbed.Shape[0] < vocab {
		return nil, fmt.Errorf("training token embedding rows %d are smaller than vocab_size %d", tokenEmbed.Shape[0], vocab)
	}
	cfg = normalizedTrainConfig(cfg, tokenParam, attnQParam, attnKParam, attnVParam, attnOParam, hiddenParam, projParam)
	if err := validateTrainConfig(cfg); err != nil {
		return nil, err
	}
	accel, accelBackend, err := newTrainerMatMulAccelerator()
	if err != nil {
		return nil, err
	}
	optimizerAccel, optimizerBackend, err := newTrainerOptimizerAccelerator()
	if err != nil {
		if accel != nil {
			accel.Close()
		}
		return nil, err
	}
	activationAccel, activationBackend, activationMode, err := newTrainerActivationAccelerator()
	if err != nil {
		if accel != nil {
			accel.Close()
		}
		if optimizerAccel != nil {
			optimizerAccel.Close()
		}
		return nil, err
	}
	contrastiveAccel, contrastiveBackend, err := newTrainerContrastiveAccelerator()
	if err != nil {
		if accel != nil {
			accel.Close()
		}
		if optimizerAccel != nil {
			optimizerAccel.Close()
		}
		if activationAccel != nil {
			activationAccel.Close()
		}
		return nil, err
	}
	return &EmbeddingTrainer{
		module:               mod,
		manifest:             manifest,
		config:               cfg,
		memoryPlan:           nil,
		tokenParam:           tokenParam,
		attnQParam:           attnQParam,
		attnKParam:           attnKParam,
		attnVParam:           attnVParam,
		attnOParam:           attnOParam,
		hiddenParam:          hiddenParam,
		projParam:            projParam,
		tokenEmbed:           tensorAsMasterF32(tokenEmbed),
		attentionQuery:       tensorAsMasterF32(attentionQuery),
		attentionKey:         tensorAsMasterF32(attentionKey),
		attentionValue:       tensorAsMasterF32(attentionValue),
		attentionOutput:      tensorAsMasterF32(attentionOutput),
		hiddenProjection:     tensorAsMasterF32(hiddenProjection),
		projection:           tensorAsMasterF32(projection),
		tokenMom1:            zeroLikeMaster(tokenEmbed),
		tokenMom2:            zeroLikeMaster(tokenEmbed),
		attnQMom1:            zeroLikeMaster(attentionQuery),
		attnQMom2:            zeroLikeMaster(attentionQuery),
		attnKMom1:            zeroLikeMaster(attentionKey),
		attnKMom2:            zeroLikeMaster(attentionKey),
		attnVMom1:            zeroLikeMaster(attentionValue),
		attnVMom2:            zeroLikeMaster(attentionValue),
		attnOMom1:            zeroLikeMaster(attentionOutput),
		attnOMom2:            zeroLikeMaster(attentionOutput),
		hiddenMom1:           zeroLikeMaster(hiddenProjection),
		hiddenMom2:           zeroLikeMaster(hiddenProjection),
		projMom1:             zeroLikeMaster(projection),
		projMom2:             zeroLikeMaster(projection),
		forwardMatMul:        accel,
		forwardBackend:       accelBackend,
		optimizerAccel:       optimizerAccel,
		optimizerBackend:     optimizerBackend,
		activationAccel:      activationAccel,
		activationBackend:    activationBackend,
		activationAccelFull:  activationMode.fullBackward,
		softmaxBackwardAccel: activationMode.softmaxBackward,
		contrastiveAccel:     contrastiveAccel,
		contrastiveBackend:   contrastiveBackend,
	}, nil
}

// Close releases any backend-owned trainer acceleration state.
func (t *EmbeddingTrainer) Close() {
	if t == nil {
		return
	}
	if t.forwardMatMul != nil {
		t.forwardMatMul.Close()
		t.forwardMatMul = nil
		t.forwardBackend = ""
	}
	if t.optimizerAccel != nil {
		t.optimizerAccel.Close()
		t.optimizerAccel = nil
		t.optimizerBackend = ""
	}
	if t.activationAccel != nil {
		t.activationAccel.Close()
		t.activationAccel = nil
		t.activationBackend = ""
	}
	if t.contrastiveAccel != nil {
		t.contrastiveAccel.Close()
		t.contrastiveAccel = nil
		t.contrastiveBackend = ""
	}
}

func (t *EmbeddingTrainer) MemoryPlan() *MemoryPlan {
	if t == nil {
		return nil
	}
	return cloneMemoryPlan(t.memoryPlan)
}

func (t *EmbeddingTrainer) syncOptimizerState(includeMoments bool) error {
	if t == nil || t.optimizerAccel == nil {
		return nil
	}
	if !includeMoments || !t.momentsDirty {
		return nil
	}
	bindings := []struct {
		name string
		t    *backend.Tensor
		m1   *backend.Tensor
		m2   *backend.Tensor
	}{
		{name: t.tokenParam.Name, t: t.tokenEmbed, m1: t.tokenMom1, m2: t.tokenMom2},
		{name: t.attnQParam.Name, t: t.attentionQuery, m1: t.attnQMom1, m2: t.attnQMom2},
		{name: t.attnKParam.Name, t: t.attentionKey, m1: t.attnKMom1, m2: t.attnKMom2},
		{name: t.attnVParam.Name, t: t.attentionValue, m1: t.attnVMom1, m2: t.attnVMom2},
		{name: t.attnOParam.Name, t: t.attentionOutput, m1: t.attnOMom1, m2: t.attnOMom2},
		{name: t.hiddenParam.Name, t: t.hiddenProjection, m1: t.hiddenMom1, m2: t.hiddenMom2},
		{name: t.projParam.Name, t: t.projection, m1: t.projMom1, m2: t.projMom2},
	}
	for _, binding := range bindings {
		if binding.name == "" || binding.t == nil {
			continue
		}
		if err := t.optimizerAccel.SyncState(binding.name, binding.t, binding.m1, binding.m2, includeMoments); err != nil {
			return err
		}
	}
	t.momentsDirty = false
	return nil
}

func (t *EmbeddingTrainer) primeForwardWeightResidency(attnQForward, attnKForward, attnVForward, attnOForward, hiddenForward, projForward *backend.Tensor) {
	if t == nil || t.forwardMatMul == nil {
		return
	}
	if !t.forwardNeedsBind &&
		t.boundForward.attnQ == attnQForward &&
		t.boundForward.attnK == attnKForward &&
		t.boundForward.attnV == attnVForward &&
		t.boundForward.attnO == attnOForward &&
		t.boundForward.hidden == hiddenForward &&
		t.boundForward.proj == projForward {
		t.forwardBindSkips++
		return
	}
	bindings := []struct {
		name   string
		tensor *backend.Tensor
	}{
		{name: t.attnQParam.Name, tensor: attnQForward},
		{name: t.attnKParam.Name, tensor: attnKForward},
		{name: t.attnVParam.Name, tensor: attnVForward},
		{name: t.attnOParam.Name, tensor: attnOForward},
		{name: t.hiddenParam.Name, tensor: hiddenForward},
		{name: t.projParam.Name, tensor: projForward},
	}
	for _, binding := range bindings {
		if binding.name == "" || binding.tensor == nil {
			continue
		}
		if err := t.forwardMatMul.BindMatrix(binding.name, binding.tensor); err != nil {
			t.Close()
			return
		}
	}
	t.boundForward = embeddingForwardWeights{
		attnQ:  attnQForward,
		attnK:  attnKForward,
		attnV:  attnVForward,
		attnO:  attnOForward,
		hidden: hiddenForward,
		proj:   projForward,
	}
	t.forwardNeedsBind = false
}

// ForwardResidencyStats reports trainer-level residency reuse plus backend matmul prep activity.
func (t *EmbeddingTrainer) ForwardResidencyStats() EmbeddingForwardResidencyStats {
	if t == nil {
		return EmbeddingForwardResidencyStats{}
	}
	stats := EmbeddingForwardResidencyStats{
		BindSkips: t.forwardBindSkips,
	}
	if t.forwardMatMul != nil {
		stats.MatMul = t.forwardMatMul.Stats()
	}
	return stats
}

// TrainProfile snapshots backend and residency activity for the current trainer state.
func (t *EmbeddingTrainer) TrainProfile() EmbeddingTrainProfile {
	if t == nil {
		return EmbeddingTrainProfile{Version: EmbeddingTrainProfileVersion}
	}
	profile := EmbeddingTrainProfile{
		Version:            EmbeddingTrainProfileVersion,
		Step:               t.step,
		ForwardBackend:     t.forwardBackend,
		OptimizerBackend:   t.optimizerBackend,
		ActivationBackend:  t.activationBackend,
		ContrastiveBackend: t.contrastiveBackend,
		ForwardResidency:   t.ForwardResidencyStats(),
	}
	if t.optimizerAccel != nil {
		profile.Optimizer = t.optimizerAccel.Stats()
	}
	if t.activationAccel != nil {
		profile.Activation = t.activationAccel.Stats()
	}
	if t.contrastiveAccel != nil {
		profile.Contrastive = t.contrastiveAccel.Stats()
	}
	return profile
}

func (t *EmbeddingTrainer) nextSequenceBindingPrefix() string {
	if t == nil {
		return ""
	}
	t.sequenceBindingID++
	return fmt.Sprintf("seq_%d", t.sequenceBindingID)
}

func (t *EmbeddingTrainer) bindSequenceTensor(state *embeddingSequenceState, slot string, tensor *backend.Tensor, bindMatMul, bindActivation bool) string {
	if t == nil || state == nil || tensor == nil {
		return ""
	}
	bindMatMul = bindMatMul && t.forwardMatMul != nil && sequenceMatMulBindingsEnabled()
	bindActivation = bindActivation && t.activationAccel != nil
	if !bindMatMul && !bindActivation {
		return ""
	}
	prefix := state.bindingPrefix
	if prefix == "" {
		prefix = t.nextSequenceBindingPrefix()
		state.bindingPrefix = prefix
	}
	name := prefix + "_" + slot
	if bindMatMul {
		if err := t.forwardMatMul.BindMatrix(name, tensor); err != nil {
			return ""
		}
	}
	if bindActivation {
		if err := t.activationAccel.BindTensor(name, tensor); err != nil {
			if bindMatMul {
				_ = t.forwardMatMul.UnbindMatrix(name)
			}
			return ""
		}
	}
	return name
}

func (t *EmbeddingTrainer) bindSoftmaxActivationForBatchedForward() bool {
	return !batchedBackwardEnabled() || !t.softmaxBackwardAccelEnabled()
}

func (t *EmbeddingTrainer) bindFullActivationForBatchedForward() bool {
	return !batchedBackwardEnabled() || !t.fullActivationBackwardAccelEnabled()
}

func (state *embeddingSequenceState) skipUnboundActivationBackward(bindingNames ...string) bool {
	if state == nil || !state.skipUnboundActAccel {
		return false
	}
	for _, name := range bindingNames {
		if name == "" {
			return true
		}
	}
	return false
}

func (t *EmbeddingTrainer) unbindSequenceTensor(name string) {
	if t == nil || name == "" {
		return
	}
	if t.forwardMatMul != nil {
		_ = t.forwardMatMul.UnbindMatrix(name)
	}
	if t.activationAccel != nil {
		_ = t.activationAccel.UnbindTensor(name)
	}
}

func (t *EmbeddingTrainer) releaseSequenceBindings(state *embeddingSequenceState) {
	if t == nil || state == nil {
		return
	}
	for _, name := range []string{
		state.inputBinding,
		state.hiddenBinding,
		state.attnQBinding,
		state.attnKBinding,
		state.attnVBinding,
		state.attnScoresBinding,
		state.attnMixedBinding,
		state.attnResidualBinding,
		state.ffnHiddenBinding,
		state.activatedBinding,
		state.ffnResidualBinding,
		state.projectedBinding,
	} {
		if name == "" {
			continue
		}
		t.unbindSequenceTensor(name)
	}
	state.inputBinding = ""
	state.hiddenBinding = ""
	state.attnQBinding = ""
	state.attnKBinding = ""
	state.attnVBinding = ""
	state.attnScoresBinding = ""
	state.attnMixedBinding = ""
	state.attnResidualBinding = ""
	state.ffnHiddenBinding = ""
	state.activatedBinding = ""
	state.ffnResidualBinding = ""
	state.projectedBinding = ""
	state.bindingPrefix = ""
}

func (t *EmbeddingTrainer) releaseEncodedSequenceBindings(seq *embeddingEncodedSequence) {
	if t == nil || seq == nil {
		return
	}
	for _, layer := range seq.layers {
		t.releaseSequenceBindings(layer)
	}
}

func (t *EmbeddingTrainer) releaseEncodedSequences(seqs []*embeddingEncodedSequence) {
	if t == nil {
		return
	}
	for _, seq := range seqs {
		t.releaseEncodedSequenceBindings(seq)
	}
}

// EvalBatch runs the quantization-aware forward path without mutating weights.
func (t *EmbeddingTrainer) EvalBatch(batch []EmbeddingPairExample) (EmbeddingTrainMetrics, error) {
	return t.runBatch(batch, false)
}

// EvaluatePairs runs the forward path and returns ship-facing pairwise quality metrics.
func (t *EmbeddingTrainer) EvaluatePairs(batch []EmbeddingPairExample) (EmbeddingEvalMetrics, error) {
	if t == nil {
		return EmbeddingEvalMetrics{}, fmt.Errorf("embedding trainer is not initialized")
	}
	if len(batch) == 0 {
		return EmbeddingEvalMetrics{}, fmt.Errorf("evaluation batch is empty")
	}
	forward := t.prepareForwardWeights()
	t.primeForwardWeightResidency(
		forward.attnQ,
		forward.attnK,
		forward.attnV,
		forward.attnO,
		forward.hidden,
		forward.proj,
	)

	metrics := EmbeddingEvalMetrics{PairCount: len(batch)}
	scores := make([]embeddingEvalScore, 0, len(batch))
	if ok, err := t.evaluatePairsBatchedForward(batch, forward, &metrics, &scores); err != nil {
		return EmbeddingEvalMetrics{}, err
	} else if !ok {
		for i, example := range batch {
			score, loss, err := t.scoreExamplePair(example, forward)
			if err != nil {
				return EmbeddingEvalMetrics{}, fmt.Errorf("batch %d: %w", i, err)
			}
			accumulateEvalPairScore(&metrics, &scores, example.Target, score, loss)
		}
	}
	invPairs := float32(1) / float32(len(batch))
	metrics.Loss *= invPairs
	metrics.AverageScore *= invPairs
	metrics.PairAccuracy *= invPairs
	if metrics.PositiveCount > 0 {
		metrics.PositiveMeanScore /= float32(metrics.PositiveCount)
	}
	if metrics.NegativeCount > 0 {
		metrics.NegativeMeanScore /= float32(metrics.NegativeCount)
	}
	metrics.ScoreMargin = metrics.PositiveMeanScore - metrics.NegativeMeanScore
	finalizeEvalScoreMetrics(&metrics, scores)
	return metrics, nil
}

func (t *EmbeddingTrainer) evaluatePairsBatchedForward(batch []EmbeddingPairExample, forward *embeddingForwardWeights, metrics *EmbeddingEvalMetrics, scores *[]embeddingEvalScore) (bool, error) {
	if t == nil || t.forwardMatMul == nil || forward == nil || metrics == nil || scores == nil || len(batch) == 0 || !batchedPairwiseEvalEnabled() {
		return false, nil
	}
	chunkSize := pairwiseEvalBatchSize(len(batch))
	for start := 0; start < len(batch); start += chunkSize {
		end := start + chunkSize
		if end > len(batch) {
			end = len(batch)
		}
		chunk := batch[start:end]
		lefts, rights, ok, err := t.tryEncodePairBatchBatchedForward(chunk, forward, false)
		if err != nil {
			return true, err
		}
		if !ok {
			return false, nil
		}
		var chunkErr error
		for i, example := range chunk {
			if i >= len(lefts) || i >= len(rights) || lefts[i] == nil || rights[i] == nil {
				chunkErr = fmt.Errorf("batch %d: encoder produced nil pair", start+i)
				break
			}
			score, _, _ := cosineGrad(lefts[i].pooled, rights[i].pooled)
			scale := score - example.Target
			accumulateEvalPairScore(metrics, scores, example.Target, score, 0.5*scale*scale)
		}
		t.releaseEncodedSequences(lefts)
		t.releaseEncodedSequences(rights)
		if chunkErr != nil {
			return true, chunkErr
		}
	}
	return true, nil
}

func accumulateEvalPairScore(metrics *EmbeddingEvalMetrics, scores *[]embeddingEvalScore, target, score, loss float32) {
	if metrics == nil || scores == nil {
		return
	}
	metrics.Loss += loss
	metrics.AverageScore += score
	positive := target > 0
	*scores = append(*scores, embeddingEvalScore{Score: score, Positive: positive})
	if positive {
		metrics.PositiveCount++
		metrics.PositiveMeanScore += score
		if score > 0 {
			metrics.PairAccuracy++
		}
	} else {
		metrics.NegativeCount++
		metrics.NegativeMeanScore += score
		if score < 0 {
			metrics.PairAccuracy++
		}
	}
}

func finalizeEvalScoreMetrics(metrics *EmbeddingEvalMetrics, scores []embeddingEvalScore) {
	if metrics == nil || len(scores) == 0 || metrics.PositiveCount == 0 || metrics.NegativeCount == 0 {
		return
	}
	metrics.ThresholdAccuracy, metrics.ScoreThreshold = bestEvalScoreThresholdAccuracy(scores, metrics.PositiveCount, metrics.NegativeCount)
	metrics.ROCAUC = evalScoreROCAUC(scores, metrics.PositiveCount, metrics.NegativeCount)
}

func bestEvalScoreThresholdAccuracy(scores []embeddingEvalScore, positiveCount, negativeCount int) (float32, float32) {
	if len(scores) == 0 {
		return 0, 0
	}
	ordered := append([]embeddingEvalScore(nil), scores...)
	sort.Slice(ordered, func(i, j int) bool {
		return ordered[i].Score > ordered[j].Score
	})
	bestCorrect := negativeCount
	bestThreshold := ordered[0].Score
	seenPositives := 0
	seenNegatives := 0
	for i := 0; i < len(ordered); {
		j := i + 1
		groupPositives := 0
		groupNegatives := 0
		for j <= len(ordered) {
			current := ordered[j-1]
			if current.Positive {
				groupPositives++
			} else {
				groupNegatives++
			}
			if j == len(ordered) || ordered[j].Score != ordered[i].Score {
				break
			}
			j++
		}
		seenPositives += groupPositives
		seenNegatives += groupNegatives
		correct := seenPositives + (negativeCount - seenNegatives)
		if correct > bestCorrect {
			bestCorrect = correct
			bestThreshold = ordered[i].Score
		}
		i = j
	}
	return float32(bestCorrect) / float32(len(ordered)), bestThreshold
}

func evalScoreROCAUC(scores []embeddingEvalScore, positiveCount, negativeCount int) float32 {
	if len(scores) == 0 || positiveCount == 0 || negativeCount == 0 {
		return 0
	}
	ordered := append([]embeddingEvalScore(nil), scores...)
	sort.Slice(ordered, func(i, j int) bool {
		return ordered[i].Score < ordered[j].Score
	})
	rankSumPositives := float64(0)
	for i := 0; i < len(ordered); {
		j := i + 1
		positivesInGroup := 0
		if ordered[i].Positive {
			positivesInGroup++
		}
		for j < len(ordered) && ordered[j].Score == ordered[i].Score {
			if ordered[j].Positive {
				positivesInGroup++
			}
			j++
		}
		averageRank := (float64(i+1) + float64(j)) * 0.5
		rankSumPositives += float64(positivesInGroup) * averageRank
		i = j
	}
	positives := float64(positiveCount)
	negatives := float64(negativeCount)
	auc := (rankSumPositives - positives*(positives+1)*0.5) / (positives * negatives)
	if auc < 0 {
		return 0
	}
	if auc > 1 {
		return 1
	}
	return float32(auc)
}

// EvaluateContrastive runs pairwise contrastive evaluation without re-encoding each expanded pair.
func (t *EmbeddingTrainer) EvaluateContrastive(batch []EmbeddingContrastiveExample) (EmbeddingEvalMetrics, error) {
	if t == nil {
		return EmbeddingEvalMetrics{}, fmt.Errorf("embedding trainer is not initialized")
	}
	if len(batch) == 0 {
		return EmbeddingEvalMetrics{}, fmt.Errorf("evaluation batch is empty")
	}
	forward := t.prepareForwardWeights()
	t.primeForwardWeightResidency(
		forward.attnQ,
		forward.attnK,
		forward.attnV,
		forward.attnO,
		forward.hidden,
		forward.proj,
	)

	queries, positives, err := t.encodeContrastiveBatch(batch, forward, false)
	if err != nil {
		return EmbeddingEvalMetrics{}, err
	}
	defer t.releaseEncodedSequences(queries)
	defer t.releaseEncodedSequences(positives)
	return evaluateContrastiveEncodings(queries, positives, t.config), nil
}

// TrainStep runs one SGD-style update over a batch of pairwise examples.
func (t *EmbeddingTrainer) TrainStep(batch []EmbeddingPairExample) (EmbeddingTrainMetrics, error) {
	return t.runBatch(batch, true)
}

// TrainContrastiveStep runs one update over a contrastive batch without expanding it into repeated pair encodes.
func (t *EmbeddingTrainer) TrainContrastiveStep(batch []EmbeddingContrastiveExample) (EmbeddingTrainMetrics, error) {
	if t == nil {
		return EmbeddingTrainMetrics{}, fmt.Errorf("embedding trainer is not initialized")
	}
	if len(batch) < 2 {
		return EmbeddingTrainMetrics{}, fmt.Errorf("contrastive training batch needs at least 2 examples")
	}
	forward := t.prepareForwardWeights()
	t.primeForwardWeightResidency(forward.attnQ, forward.attnK, forward.attnV, forward.attnO, forward.hidden, forward.proj)
	queries, positives, err := t.encodeContrastiveBatch(batch, forward, true)
	if err != nil {
		return EmbeddingTrainMetrics{}, err
	}
	defer t.releaseEncodedSequences(queries)
	defer t.releaseEncodedSequences(positives)

	gradToken := make([]float32, len(t.tokenEmbed.F32))
	gradAttnQ := make([]float32, tensorDataLen(t.attentionQuery))
	gradAttnK := make([]float32, tensorDataLen(t.attentionKey))
	gradAttnV := make([]float32, tensorDataLen(t.attentionValue))
	gradAttnO := make([]float32, tensorDataLen(t.attentionOutput))
	gradHidden := make([]float32, len(t.hiddenProjectionData()))
	gradProj := make([]float32, len(t.projection.F32))
	queryGrads := make([][]float32, len(queries))
	positiveGrads := make([][]float32, len(positives))
	for i := range queries {
		queryGrads[i] = make([]float32, len(queries[i].pooled))
		positiveGrads[i] = make([]float32, len(positives[i].pooled))
	}

	totalLoss := float32(0)
	totalScore := float32(0)
	pairCount := len(batch) * len(batch)
	batchScale := float32(1) / float32(pairCount)
	lossScale := batchScale
	if t.config.ContrastiveLoss == "infonce" {
		var ok bool
		totalLoss, totalScore, ok = t.tryInfoNCEContrastiveAccelerator(queries, positives, queryGrads, positiveGrads)
		if !ok {
			totalLoss, totalScore = accumulateInfoNCEContrastiveGrads(queries, positives, t.config.Temperature, queryGrads, positiveGrads)
		}
		batchScale = float32(1) / float32(len(batch))
		lossScale = batchScale
	} else {
		totalLoss, totalScore = accumulatePairMSEContrastiveGrads(queries, positives, queryGrads, positiveGrads)
	}

	if !t.tryBackpropContrastiveBatch(
		queries,
		positives,
		queryGrads,
		positiveGrads,
		forward.attnQ,
		forward.attnK,
		forward.attnV,
		forward.attnO,
		forward.hidden,
		forward.proj,
		gradToken,
		gradAttnQ,
		gradAttnK,
		gradAttnV,
		gradAttnO,
		gradHidden,
		gradProj,
	) {
		for i, query := range queries {
			inputGrad := t.backpropEncodedSequence(query, queryGrads[i], forward.attnQ, forward.attnK, forward.attnV, forward.attnO, forward.hidden, forward.proj, gradToken, gradAttnQ, gradAttnK, gradAttnV, gradAttnO, gradHidden, gradProj)
			accumulateTokenGrad(query.tokens, inputGrad, gradToken, t.tokenEmbed.Shape[1], t.tokenEmbed.Shape[0])
		}
		for i, positive := range positives {
			inputGrad := t.backpropEncodedSequence(positive, positiveGrads[i], forward.attnQ, forward.attnK, forward.attnV, forward.attnO, forward.hidden, forward.proj, gradToken, gradAttnQ, gradAttnK, gradAttnV, gradAttnO, gradHidden, gradProj)
			accumulateTokenGrad(positive.tokens, inputGrad, gradToken, t.tokenEmbed.Shape[1], t.tokenEmbed.Shape[0])
		}
	}

	t.step++
	t.applyOptimizerUpdate(t.tokenParam.Name, t.tokenEmbed, t.tokenMom1, t.tokenMom2, gradToken, batchScale)
	t.applyOptimizerUpdate(t.attnQParam.Name, t.attentionQuery, t.attnQMom1, t.attnQMom2, gradAttnQ, batchScale)
	t.applyOptimizerUpdate(t.attnKParam.Name, t.attentionKey, t.attnKMom1, t.attnKMom2, gradAttnK, batchScale)
	t.applyOptimizerUpdate(t.attnVParam.Name, t.attentionValue, t.attnVMom1, t.attnVMom2, gradAttnV, batchScale)
	t.applyOptimizerUpdate(t.attnOParam.Name, t.attentionOutput, t.attnOMom1, t.attnOMom2, gradAttnO, batchScale)
	t.applyOptimizerUpdate(t.hiddenParam.Name, t.hiddenProjection, t.hiddenMom1, t.hiddenMom2, gradHidden, batchScale)
	t.applyOptimizerUpdate(t.projParam.Name, t.projection, t.projMom1, t.projMom2, gradProj, batchScale)
	return EmbeddingTrainMetrics{
		Loss:         totalLoss * lossScale,
		AverageScore: totalScore / float32(pairCount),
		BatchSize:    pairCount,
	}, nil
}

// ExportInferenceWeights returns runtime-loadable weights in the module's declared dtypes.
func (t *EmbeddingTrainer) ExportInferenceWeights() (map[string]*backend.Tensor, error) {
	if t == nil {
		return nil, fmt.Errorf("embedding trainer is not initialized")
	}
	out := map[string]*backend.Tensor{}
	token, err := exportTensorForParam(t.tokenParam, t.tokenEmbed)
	if err != nil {
		return nil, err
	}
	if t.attentionQuery != nil {
		query, err := exportTensorForParam(t.attnQParam, t.attentionQuery)
		if err != nil {
			return nil, err
		}
		key, err := exportTensorForParam(t.attnKParam, t.attentionKey)
		if err != nil {
			return nil, err
		}
		value, err := exportTensorForParam(t.attnVParam, t.attentionValue)
		if err != nil {
			return nil, err
		}
		output, err := exportTensorForParam(t.attnOParam, t.attentionOutput)
		if err != nil {
			return nil, err
		}
		out[t.attnQParam.Name] = query
		out[t.attnKParam.Name] = key
		out[t.attnVParam.Name] = value
		out[t.attnOParam.Name] = output
	}
	if t.hiddenProjection != nil {
		hidden, err := exportTensorForParam(t.hiddenParam, t.hiddenProjection)
		if err != nil {
			return nil, err
		}
		out[t.hiddenParam.Name] = hidden
	}
	proj, err := exportTensorForParam(t.projParam, t.projection)
	if err != nil {
		return nil, err
	}
	out[t.tokenParam.Name] = token
	out[t.projParam.Name] = proj
	return out, nil
}

// ExportLoadOptions adapts the current trained weights to runtime load options.
func (t *EmbeddingTrainer) ExportLoadOptions() ([]LoadOption, error) {
	weights, err := t.ExportInferenceWeights()
	if err != nil {
		return nil, err
	}
	return NewWeightFile(weights).LoadOptions(), nil
}

// ExportWeightFile builds a serialized weight file for package loading.
func (t *EmbeddingTrainer) ExportWeightFile() (WeightFile, error) {
	weights, err := t.ExportInferenceWeights()
	if err != nil {
		return WeightFile{}, err
	}
	return NewWeightFile(weights), nil
}

// RenameEmbeddingModel updates the module and embedding manifest identity before rewriting a package.
func (t *EmbeddingTrainer) RenameEmbeddingModel(name string) error {
	if t == nil {
		return fmt.Errorf("embedding trainer is not initialized")
	}
	name = strings.TrimSpace(name)
	if name == "" {
		return fmt.Errorf("embedding model name is required")
	}
	if t.module != nil {
		t.module.Name = name
	}
	t.manifest.Name = name
	return nil
}

// WriteEmbeddingPackage writes a packaged embedding model to sibling artifact, manifest, and weight files.
func (t *EmbeddingTrainer) WriteEmbeddingPackage(barrPath string) (EmbeddingPackagePaths, error) {
	if t == nil {
		return EmbeddingPackagePaths{}, fmt.Errorf("embedding trainer is not initialized")
	}
	if err := barr.WriteFile(barrPath, t.module); err != nil {
		return EmbeddingPackagePaths{}, err
	}
	manifestPath := DefaultEmbeddingManifestPath(barrPath)
	if err := t.manifest.WriteFile(manifestPath); err != nil {
		return EmbeddingPackagePaths{}, err
	}
	weightFile, err := t.ExportWeightFile()
	if err != nil {
		return EmbeddingPackagePaths{}, err
	}
	weightPath := DefaultWeightFilePath(barrPath)
	if err := weightFile.WriteFile(weightPath); err != nil {
		return EmbeddingPackagePaths{}, err
	}
	memoryPlan := NewMemoryPlan(t.module, weightFile.Weights, MemoryPlanOptions{})
	memoryPlanPath := DefaultMemoryPlanPath(barrPath)
	if err := memoryPlan.WriteFile(memoryPlanPath); err != nil {
		return EmbeddingPackagePaths{}, err
	}
	t.memoryPlan = cloneMemoryPlan(&memoryPlan)
	packageFiles := map[string]string{
		"artifact":           barrPath,
		"embedding_manifest": manifestPath,
		"weights":            weightPath,
		"memory_plan":        memoryPlanPath,
	}
	tokenizerPath := DefaultTokenizerPath(barrPath)
	if _, err := os.Stat(tokenizerPath); err == nil {
		packageFiles["tokenizer"] = tokenizerPath
	}
	packageManifest, err := BuildPackageManifest(PackageEmbedding, t.module, packageFiles)
	if err != nil {
		return EmbeddingPackagePaths{}, err
	}
	packageManifestPath := DefaultPackageManifestPath(barrPath)
	if err := packageManifest.WriteFile(packageManifestPath); err != nil {
		return EmbeddingPackagePaths{}, err
	}
	return EmbeddingPackagePaths{
		ArtifactPath:        barrPath,
		ManifestPath:        manifestPath,
		TokenizerPath:       tokenizerPath,
		WeightFilePath:      weightPath,
		MemoryPlanPath:      memoryPlanPath,
		PackageManifestPath: packageManifestPath,
	}, nil
}

// WriteTrainingPackage writes a packaged native-training embedder including trainer config and checkpoint state.
func (t *EmbeddingTrainer) WriteTrainingPackage(barrPath string) (EmbeddingTrainPackagePaths, error) {
	if t == nil {
		return EmbeddingTrainPackagePaths{}, fmt.Errorf("embedding trainer is not initialized")
	}
	if err := barr.WriteFile(barrPath, t.module); err != nil {
		return EmbeddingTrainPackagePaths{}, err
	}
	embeddingManifestPath := DefaultEmbeddingManifestPath(barrPath)
	if err := t.manifest.WriteFile(embeddingManifestPath); err != nil {
		return EmbeddingTrainPackagePaths{}, err
	}
	weightFile, err := t.ExportWeightFile()
	if err != nil {
		return EmbeddingTrainPackagePaths{}, err
	}
	weightPath := DefaultWeightFilePath(barrPath)
	if err := weightFile.WriteFile(weightPath); err != nil {
		return EmbeddingTrainPackagePaths{}, err
	}
	memoryPlan := NewMemoryPlan(t.module, weightFile.Weights, MemoryPlanOptions{})
	memoryPlanPath := DefaultMemoryPlanPath(barrPath)
	if err := memoryPlan.WriteFile(memoryPlanPath); err != nil {
		return EmbeddingTrainPackagePaths{}, err
	}
	t.memoryPlan = cloneMemoryPlan(&memoryPlan)
	trainManifest := EmbeddingTrainManifest{
		Name:      t.manifest.Name,
		Embedding: t.manifest,
		Config:    t.config,
	}
	trainManifestPath := DefaultEmbeddingTrainManifestPath(barrPath)
	if err := trainManifest.WriteFile(trainManifestPath); err != nil {
		return EmbeddingTrainPackagePaths{}, err
	}
	checkpoint, err := t.Checkpoint()
	if err != nil {
		return EmbeddingTrainPackagePaths{}, err
	}
	checkpointPath := DefaultEmbeddingCheckpointPath(barrPath)
	if err := checkpoint.WriteFile(checkpointPath); err != nil {
		return EmbeddingTrainPackagePaths{}, err
	}
	trainProfile := t.TrainProfile()
	trainProfilePath := DefaultEmbeddingTrainProfilePath(barrPath)
	if err := trainProfile.WriteFile(trainProfilePath); err != nil {
		return EmbeddingTrainPackagePaths{}, err
	}
	packageFiles := map[string]string{
		"artifact":           barrPath,
		"embedding_manifest": embeddingManifestPath,
		"weights":            weightPath,
		"memory_plan":        memoryPlanPath,
		"train_manifest":     trainManifestPath,
		"checkpoint":         checkpointPath,
		"train_profile":      trainProfilePath,
	}
	tokenizerPath := DefaultTokenizerPath(barrPath)
	if _, err := os.Stat(tokenizerPath); err == nil {
		packageFiles["tokenizer"] = tokenizerPath
	}
	packageManifest, err := BuildPackageManifest(PackageTraining, t.module, packageFiles)
	if err != nil {
		return EmbeddingTrainPackagePaths{}, err
	}
	packageManifestPath := DefaultPackageManifestPath(barrPath)
	if err := packageManifest.WriteFile(packageManifestPath); err != nil {
		return EmbeddingTrainPackagePaths{}, err
	}
	return EmbeddingTrainPackagePaths{
		ArtifactPath:          barrPath,
		EmbeddingManifestPath: embeddingManifestPath,
		TokenizerPath:         tokenizerPath,
		WeightFilePath:        weightPath,
		MemoryPlanPath:        memoryPlanPath,
		TrainManifestPath:     trainManifestPath,
		CheckpointPath:        checkpointPath,
		TrainProfilePath:      trainProfilePath,
		PackageManifestPath:   packageManifestPath,
	}, nil
}

func (t *EmbeddingTrainer) runBatch(batch []EmbeddingPairExample, update bool) (EmbeddingTrainMetrics, error) {
	if t == nil {
		return EmbeddingTrainMetrics{}, fmt.Errorf("embedding trainer is not initialized")
	}
	if len(batch) == 0 {
		return EmbeddingTrainMetrics{}, fmt.Errorf("training batch is empty")
	}
	forward := t.prepareForwardWeights()
	t.primeForwardWeightResidency(forward.attnQ, forward.attnK, forward.attnV, forward.attnO, forward.hidden, forward.proj)
	gradToken := make([]float32, len(t.tokenEmbed.F32))
	gradAttnQ := make([]float32, tensorDataLen(t.attentionQuery))
	gradAttnK := make([]float32, tensorDataLen(t.attentionKey))
	gradAttnV := make([]float32, tensorDataLen(t.attentionValue))
	gradAttnO := make([]float32, tensorDataLen(t.attentionOutput))
	gradHidden := make([]float32, len(t.hiddenProjectionData()))
	gradProj := make([]float32, len(t.projection.F32))

	totalLoss := float32(0)
	totalScore := float32(0)
	for i, example := range batch {
		left, right, err := t.encodeExamplePair(example, forward.token, forward.attnQ, forward.attnK, forward.attnV, forward.attnO, forward.hidden, forward.proj, update)
		if err != nil {
			return EmbeddingTrainMetrics{}, fmt.Errorf("batch %d: %w", i, err)
		}
		score, gradLeft, gradRight := cosineGrad(left.pooled, right.pooled)
		scale := score - example.Target
		totalLoss += 0.5 * scale * scale
		totalScore += score
		for j := range gradLeft {
			gradLeft[j] *= scale
			gradRight[j] *= scale
		}
		leftInputGrad := t.backpropEncodedSequence(left, gradLeft, forward.attnQ, forward.attnK, forward.attnV, forward.attnO, forward.hidden, forward.proj, gradToken, gradAttnQ, gradAttnK, gradAttnV, gradAttnO, gradHidden, gradProj)
		rightInputGrad := t.backpropEncodedSequence(right, gradRight, forward.attnQ, forward.attnK, forward.attnV, forward.attnO, forward.hidden, forward.proj, gradToken, gradAttnQ, gradAttnK, gradAttnV, gradAttnO, gradHidden, gradProj)
		accumulateTokenGrad(left.tokens, leftInputGrad, gradToken, t.tokenEmbed.Shape[1], t.tokenEmbed.Shape[0])
		accumulateTokenGrad(right.tokens, rightInputGrad, gradToken, t.tokenEmbed.Shape[1], t.tokenEmbed.Shape[0])
		t.releaseEncodedSequenceBindings(left)
		t.releaseEncodedSequenceBindings(right)
	}

	batchScale := float32(1) / float32(len(batch))
	if update {
		t.step++
		t.applyOptimizerUpdate(t.tokenParam.Name, t.tokenEmbed, t.tokenMom1, t.tokenMom2, gradToken, batchScale)
		t.applyOptimizerUpdate(t.attnQParam.Name, t.attentionQuery, t.attnQMom1, t.attnQMom2, gradAttnQ, batchScale)
		t.applyOptimizerUpdate(t.attnKParam.Name, t.attentionKey, t.attnKMom1, t.attnKMom2, gradAttnK, batchScale)
		t.applyOptimizerUpdate(t.attnVParam.Name, t.attentionValue, t.attnVMom1, t.attnVMom2, gradAttnV, batchScale)
		t.applyOptimizerUpdate(t.attnOParam.Name, t.attentionOutput, t.attnOMom1, t.attnOMom2, gradAttnO, batchScale)
		t.applyOptimizerUpdate(t.hiddenParam.Name, t.hiddenProjection, t.hiddenMom1, t.hiddenMom2, gradHidden, batchScale)
		t.applyOptimizerUpdate(t.projParam.Name, t.projection, t.projMom1, t.projMom2, gradProj, batchScale)
	}
	return EmbeddingTrainMetrics{
		Loss:         totalLoss * batchScale,
		AverageScore: totalScore * batchScale,
		BatchSize:    len(batch),
	}, nil
}

func (t *EmbeddingTrainer) scoreExamplePair(example EmbeddingPairExample, forward *embeddingForwardWeights) (float32, float32, error) {
	if forward == nil {
		return 0, 0, fmt.Errorf("missing forward weights")
	}
	left, right, err := t.encodeExamplePair(example, forward.token, forward.attnQ, forward.attnK, forward.attnV, forward.attnO, forward.hidden, forward.proj, false)
	if err != nil {
		return 0, 0, err
	}
	defer t.releaseEncodedSequenceBindings(left)
	defer t.releaseEncodedSequenceBindings(right)
	score, _, _ := cosineGrad(left.pooled, right.pooled)
	scale := score - example.Target
	return score, 0.5 * scale * scale, nil
}

func (t *EmbeddingTrainer) encodeContrastiveBatch(batch []EmbeddingContrastiveExample, forward *embeddingForwardWeights, captureBindings bool) ([]*embeddingEncodedSequence, []*embeddingEncodedSequence, error) {
	if forward == nil {
		return nil, nil, fmt.Errorf("missing forward weights")
	}
	if queries, positives, ok, err := t.tryEncodeContrastiveBatchBatchedForward(batch, forward, captureBindings); ok || err != nil {
		return queries, positives, err
	}
	queries := make([]*embeddingEncodedSequence, 0, len(batch))
	positives := make([]*embeddingEncodedSequence, 0, len(batch))
	for i, example := range batch {
		queryMask, err := t.prepareMask(example.QueryTokens, example.QueryMask)
		if err != nil {
			t.releaseEncodedSequences(queries)
			t.releaseEncodedSequences(positives)
			return nil, nil, fmt.Errorf("batch %d query: %w", i, err)
		}
		positiveMask, err := t.prepareMask(example.PositiveTokens, example.PositiveMask)
		if err != nil {
			t.releaseEncodedSequences(queries)
			t.releaseEncodedSequences(positives)
			return nil, nil, fmt.Errorf("batch %d positive: %w", i, err)
		}
		query, err := t.encodeSequence(example.QueryTokens, queryMask, forward.token, forward.attnQ, forward.attnK, forward.attnV, forward.attnO, forward.hidden, forward.proj, captureBindings)
		if err != nil {
			t.releaseEncodedSequences(queries)
			t.releaseEncodedSequences(positives)
			return nil, nil, fmt.Errorf("batch %d query: %w", i, err)
		}
		positive, err := t.encodeSequence(example.PositiveTokens, positiveMask, forward.token, forward.attnQ, forward.attnK, forward.attnV, forward.attnO, forward.hidden, forward.proj, captureBindings)
		if err != nil {
			t.releaseEncodedSequenceBindings(query)
			t.releaseEncodedSequences(queries)
			t.releaseEncodedSequences(positives)
			return nil, nil, fmt.Errorf("batch %d positive: %w", i, err)
		}
		queries = append(queries, query)
		positives = append(positives, positive)
	}
	return queries, positives, nil
}

type embeddingBatchSequence struct {
	tokens  []int32
	mask    []int32
	current []float32
	encoded *embeddingEncodedSequence
}

type embeddingBatchSequenceSlot struct {
	sequence *embeddingBatchSequence
}

func (t *EmbeddingTrainer) tryEncodeContrastiveBatchBatchedForward(batch []EmbeddingContrastiveExample, forward *embeddingForwardWeights, captureBindings bool) ([]*embeddingEncodedSequence, []*embeddingEncodedSequence, bool, error) {
	if t == nil || t.forwardMatMul == nil || forward == nil || len(batch) == 0 || !batchedContrastiveForwardEnabled() {
		return nil, nil, false, nil
	}
	sequences := make([]*embeddingBatchSequence, 0, len(batch)*2)
	queries := make([]*embeddingEncodedSequence, len(batch))
	positives := make([]*embeddingEncodedSequence, len(batch))
	groups := map[int][]embeddingBatchSequenceSlot{}
	lengths := make([]int, 0)
	for i, example := range batch {
		queryMask, err := t.prepareMask(example.QueryTokens, example.QueryMask)
		if err != nil {
			return nil, nil, true, fmt.Errorf("batch %d query: %w", i, err)
		}
		positiveMask, err := t.prepareMask(example.PositiveTokens, example.PositiveMask)
		if err != nil {
			return nil, nil, true, fmt.Errorf("batch %d positive: %w", i, err)
		}
		query, err := t.newEmbeddingBatchSequence(example.QueryTokens, queryMask, forward.token)
		if err != nil {
			t.releaseBatchEncodedSequences(sequences)
			return nil, nil, true, fmt.Errorf("batch %d query: %w", i, err)
		}
		sequences = append(sequences, query)
		positive, err := t.newEmbeddingBatchSequence(example.PositiveTokens, positiveMask, forward.token)
		if err != nil {
			t.releaseBatchEncodedSequences(sequences)
			return nil, nil, true, fmt.Errorf("batch %d positive: %w", i, err)
		}
		queries[i] = query.encoded
		positives[i] = positive.encoded
		sequences = append(sequences, positive)
		for _, sequence := range []*embeddingBatchSequence{query, positive} {
			seqLen := len(sequence.tokens)
			if _, ok := groups[seqLen]; !ok {
				lengths = append(lengths, seqLen)
			}
			groups[seqLen] = append(groups[seqLen], embeddingBatchSequenceSlot{sequence: sequence})
		}
	}
	if len(sequences) == 0 {
		return nil, nil, false, nil
	}
	if err := t.encodeBatchSequencesByLength(sequences, groups, lengths, forward, captureBindings); err != nil {
		t.releaseBatchEncodedSequences(sequences)
		return nil, nil, true, err
	}
	return queries, positives, true, nil
}

func (t *EmbeddingTrainer) tryEncodePairBatchBatchedForward(batch []EmbeddingPairExample, forward *embeddingForwardWeights, captureBindings bool) ([]*embeddingEncodedSequence, []*embeddingEncodedSequence, bool, error) {
	if t == nil || t.forwardMatMul == nil || forward == nil || len(batch) == 0 || !batchedPairwiseEvalEnabled() {
		return nil, nil, false, nil
	}
	sequences := make([]*embeddingBatchSequence, 0, len(batch)*2)
	lefts := make([]*embeddingEncodedSequence, len(batch))
	rights := make([]*embeddingEncodedSequence, len(batch))
	groups := map[int][]embeddingBatchSequenceSlot{}
	lengths := make([]int, 0)
	for i, example := range batch {
		leftMask, err := t.prepareMask(example.LeftTokens, example.LeftMask)
		if err != nil {
			return nil, nil, true, fmt.Errorf("batch %d left: %w", i, err)
		}
		rightMask, err := t.prepareMask(example.RightTokens, example.RightMask)
		if err != nil {
			return nil, nil, true, fmt.Errorf("batch %d right: %w", i, err)
		}
		left, err := t.newEmbeddingBatchSequence(example.LeftTokens, leftMask, forward.token)
		if err != nil {
			t.releaseBatchEncodedSequences(sequences)
			return nil, nil, true, fmt.Errorf("batch %d left: %w", i, err)
		}
		sequences = append(sequences, left)
		right, err := t.newEmbeddingBatchSequence(example.RightTokens, rightMask, forward.token)
		if err != nil {
			t.releaseBatchEncodedSequences(sequences)
			return nil, nil, true, fmt.Errorf("batch %d right: %w", i, err)
		}
		lefts[i] = left.encoded
		rights[i] = right.encoded
		sequences = append(sequences, right)
		for _, sequence := range []*embeddingBatchSequence{left, right} {
			seqLen := len(sequence.tokens)
			if _, ok := groups[seqLen]; !ok {
				lengths = append(lengths, seqLen)
			}
			groups[seqLen] = append(groups[seqLen], embeddingBatchSequenceSlot{sequence: sequence})
		}
	}
	if len(sequences) == 0 {
		return nil, nil, false, nil
	}
	if err := t.encodeBatchSequencesByLength(sequences, groups, lengths, forward, captureBindings); err != nil {
		t.releaseBatchEncodedSequences(sequences)
		return nil, nil, true, err
	}
	return lefts, rights, true, nil
}

func (t *EmbeddingTrainer) encodeBatchSequencesByLength(sequences []*embeddingBatchSequence, groups map[int][]embeddingBatchSequenceSlot, lengths []int, forward *embeddingForwardWeights, captureBindings bool) error {
	if len(sequences) == 0 {
		return nil
	}
	if forward == nil {
		return fmt.Errorf("missing forward weights")
	}
	for layer := 0; layer < t.encoderRepeats(); layer++ {
		for _, seqLen := range lengths {
			slots := groups[seqLen]
			states := make([]*embeddingSequenceState, len(slots))
			for i, slot := range slots {
				state, err := newEmbeddingSequenceState(slot.sequence.tokens, slot.sequence.mask, slot.sequence.current, forward.hidden, forward.proj)
				if err != nil {
					for _, state := range states {
						t.releaseSequenceBindings(state)
					}
					return err
				}
				if captureBindings {
					d := stateWidth(forward.hidden, forward.proj)
					state.inputBinding = t.bindSequenceTensor(state, "input", tensorF32View([]int{len(state.tokens), d}, state.input), true, false)
				}
				states[i] = state
			}
			if err := t.encodeBatchedLayerStates(states, forward.attnQ, forward.attnK, forward.attnV, forward.attnO, forward.hidden, forward.proj, captureBindings); err != nil {
				for _, state := range states {
					t.releaseSequenceBindings(state)
				}
				return err
			}
			for i, state := range states {
				sequence := slots[i].sequence
				sequence.encoded.layers = append(sequence.encoded.layers, state)
				sequence.current = state.projected
			}
		}
	}
	for _, sequence := range sequences {
		if len(sequence.encoded.layers) == 0 {
			return fmt.Errorf("encoder produced zero layers")
		}
		sequence.encoded.pooled = append([]float32(nil), sequence.encoded.layers[len(sequence.encoded.layers)-1].pooled...)
	}
	return nil
}

func batchedContrastiveForwardEnabled() bool {
	if trainEnvFlagEnabled("MANTA_TRAIN_DISABLE_BATCHED_FORWARD") {
		return false
	}
	switch trainEnv("MANTA_TRAIN_BATCHED_FORWARD") {
	case "0", "false", "FALSE", "no", "NO":
		return false
	default:
		return true
	}
}

func batchedPairwiseEvalEnabled() bool {
	if !batchedContrastiveForwardEnabled() || trainEnvFlagEnabled("MANTA_TRAIN_DISABLE_BATCHED_PAIR_EVAL") {
		return false
	}
	switch trainEnv("MANTA_TRAIN_BATCHED_PAIR_EVAL") {
	case "0", "false", "FALSE", "no", "NO":
		return false
	default:
		return true
	}
}

func pairwiseEvalBatchSize(total int) int {
	if total <= 0 {
		return 0
	}
	size := 256
	if raw := trainEnv("MANTA_TRAIN_PAIR_EVAL_BATCH_SIZE"); raw != "" {
		var parsed int
		if _, err := fmt.Sscanf(raw, "%d", &parsed); err == nil && parsed > 0 {
			size = parsed
		}
	}
	if size > total {
		size = total
	}
	return size
}

func sequenceMatMulBindingsEnabled() bool {
	if trainEnvFlagEnabled("MANTA_TRAIN_DISABLE_SEQUENCE_MATMUL_BINDINGS") {
		return false
	}
	return trainEnvFlagEnabled("MANTA_TRAIN_ENABLE_SEQUENCE_MATMUL_BINDINGS")
}

func qkvMultiBoundRightEnabled() bool {
	if trainEnvFlagEnabled("MANTA_TRAIN_DISABLE_QKV_MULTI_BOUND") {
		return false
	}
	switch trainEnv("MANTA_TRAIN_QKV_MULTI_BOUND") {
	case "0", "false", "FALSE", "no", "NO":
		return false
	default:
		return true
	}
}

func sharedLeftMatMulEnabled() bool {
	if trainEnvFlagEnabled("MANTA_TRAIN_DISABLE_SHARED_LEFT_MATMUL") {
		return false
	}
	switch trainEnv("MANTA_TRAIN_SHARED_LEFT_MATMUL") {
	case "0", "false", "FALSE", "no", "NO":
		return false
	default:
		return true
	}
}

func concatenatedSharedLeftMatMulEnabled() bool {
	if !sharedLeftMatMulEnabled() || trainEnvFlagEnabled("MANTA_TRAIN_DISABLE_CONCAT_SHARED_LEFT_MATMUL") {
		return false
	}
	switch trainEnv("MANTA_TRAIN_CONCAT_SHARED_LEFT_MATMUL") {
	case "0", "false", "FALSE", "no", "NO":
		return false
	default:
		return true
	}
}

func combinedAttentionVKGradMatMulEnabled() bool {
	if trainEnvFlagEnabled("MANTA_TRAIN_DISABLE_COMBINED_ATTENTION_VK_GRAD") {
		return false
	}
	switch trainEnv("MANTA_TRAIN_COMBINED_ATTENTION_VK_GRAD") {
	case "0", "false", "FALSE", "no", "NO":
		return false
	default:
		return true
	}
}

func accumulatedAttentionInputGradMatMulEnabled() bool {
	if trainEnvFlagEnabled("MANTA_TRAIN_DISABLE_ACCUMULATED_ATTENTION_INPUT_GRAD") {
		return false
	}
	switch trainEnv("MANTA_TRAIN_ACCUMULATED_ATTENTION_INPUT_GRAD") {
	case "0", "false", "FALSE", "no", "NO":
		return false
	default:
		return true
	}
}

func batchedBackwardEnabled() bool {
	if trainEnvFlagEnabled("MANTA_TRAIN_DISABLE_BATCHED_BACKWARD") {
		return false
	}
	switch trainEnv("MANTA_TRAIN_BATCHED_BACKWARD") {
	case "0", "false", "FALSE", "no", "NO":
		return false
	default:
		return true
	}
}

func activationAccelMaxElements() int {
	limit := 1 << 20
	if raw := trainEnv("MANTA_TRAIN_ACTIVATION_ACCEL_MAX_ELEMENTS"); raw != "" {
		var parsed int
		if _, err := fmt.Sscanf(raw, "%d", &parsed); err == nil {
			if parsed <= 0 {
				return 0
			}
			limit = parsed
		}
	}
	return limit
}

func activationAccelElementsAllowed(rows, cols int) bool {
	if rows <= 0 || cols <= 0 {
		return false
	}
	limit := activationAccelMaxElements()
	return limit <= 0 || rows <= limit/cols
}

func trainEnvFlagEnabled(name string) bool {
	switch trainEnv(name) {
	case "1", "true", "TRUE", "yes", "YES":
		return true
	default:
		return false
	}
}

func trainEnv(name string) string {
	if value, ok := os.LookupEnv(name); ok {
		return value
	}
	return ""
}

func (t *EmbeddingTrainer) newEmbeddingBatchSequence(tokens, mask []int32, tokenEmbed *backend.Tensor) (*embeddingBatchSequence, error) {
	input, err := embeddingInputForTokens(tokenEmbed, tokens)
	if err != nil {
		return nil, err
	}
	encoded := &embeddingEncodedSequence{
		layers: make([]*embeddingSequenceState, 0, t.encoderRepeats()),
		tokens: append([]int32(nil), tokens...),
	}
	return &embeddingBatchSequence{
		tokens:  append([]int32(nil), tokens...),
		mask:    append([]int32(nil), mask...),
		current: input,
		encoded: encoded,
	}, nil
}

func (t *EmbeddingTrainer) releaseBatchEncodedSequences(sequences []*embeddingBatchSequence) {
	for _, sequence := range sequences {
		if sequence != nil {
			t.releaseEncodedSequenceBindings(sequence.encoded)
		}
	}
}

func (t *EmbeddingTrainer) encodeBatchedLayerStates(states []*embeddingSequenceState, attentionQuery, attentionKey, attentionValue, attentionOutput, hiddenProjection, projection *backend.Tensor, captureBindings bool) error {
	if len(states) == 0 {
		return nil
	}
	if projection == nil || len(projection.Shape) != 2 {
		return fmt.Errorf("projection must be rank-2")
	}
	d := stateWidth(hiddenProjection, projection)
	h := 0
	if hiddenProjection != nil {
		if len(hiddenProjection.Shape) != 2 {
			return fmt.Errorf("hidden projection must be rank-2")
		}
		h = hiddenProjection.Shape[1]
	}
	e := projection.Shape[1]
	seqLen := len(states[0].tokens)
	bindSoftmaxActivation := t.bindSoftmaxActivationForBatchedForward()
	bindFullActivation := t.bindFullActivationForBatchedForward()
	skipUnboundActAccel := captureBindings && (!bindSoftmaxActivation || !bindFullActivation)
	if seqLen == 0 {
		return fmt.Errorf("tokens are empty")
	}
	for i, state := range states {
		if state == nil {
			return fmt.Errorf("batch state %d is nil", i)
		}
		if len(state.tokens) != seqLen || len(state.mask) != seqLen {
			return fmt.Errorf("batch state %d has inconsistent sequence length", i)
		}
		if len(state.input) != seqLen*d {
			return fmt.Errorf("batch state %d input size %d does not match tokens=%d width=%d", i, len(state.input), seqLen, d)
		}
		state.skipUnboundActAccel = skipUnboundActAccel
	}

	if attentionQuery != nil && attentionKey != nil && attentionValue != nil && attentionOutput != nil {
		if !t.fillBatchedForwardQKVMatMul(states, seqLen, d, attentionQuery, attentionKey, attentionValue) {
			t.fillBatchedForwardWeightMatMul(states, seqLen, d, func(state *embeddingSequenceState) []float32 {
				return state.input
			}, t.attnQParam.Name, attentionQuery, d, func(state *embeddingSequenceState) []float32 {
				return state.attnQ
			}, func(state *embeddingSequenceState, data []float32) {
				state.attnQ = data
			})
			t.fillBatchedForwardWeightMatMul(states, seqLen, d, func(state *embeddingSequenceState) []float32 {
				return state.input
			}, t.attnKParam.Name, attentionKey, d, func(state *embeddingSequenceState) []float32 {
				return state.attnK
			}, func(state *embeddingSequenceState, data []float32) {
				state.attnK = data
			})
			t.fillBatchedForwardWeightMatMul(states, seqLen, d, func(state *embeddingSequenceState) []float32 {
				return state.input
			}, t.attnVParam.Name, attentionValue, d, func(state *embeddingSequenceState) []float32 {
				return state.attnV
			}, func(state *embeddingSequenceState, data []float32) {
				state.attnV = data
			})
		}
		for _, state := range states {
			if captureBindings {
				state.attnQBinding = t.bindSequenceTensor(state, "q", tensorF32View([]int{seqLen, d}, state.attnQ), true, false)
				state.attnKBinding = t.bindSequenceTensor(state, "k", tensorF32View([]int{seqLen, d}, state.attnK), true, false)
				state.attnVBinding = t.bindSequenceTensor(state, "v", tensorF32View([]int{seqLen, d}, state.attnV), true, false)
			}
		}
		batchedScores, batchedScoresOK := t.tryBatchedAttentionScores(states, seqLen, d)
		for i, state := range states {
			if batchedScoresOK {
				state.attnScores = batchedScores[i]
			} else {
				kt := transpose2DData(state.attnK, seqLen, d)
				var (
					scores   []float32
					matmulOK bool
				)
				if captureBindings {
					scores, matmulOK = t.tryTrainerMatMulBoundRight(state.attnQ, seqLen, d, state.attnKBinding, tensorF32View([]int{seqLen, d}, state.attnK), false, true)
				} else {
					scores, matmulOK = t.tryTrainerMatMul(state.attnQ, seqLen, d, state.attnK, seqLen, d, false, true)
				}
				if matmulOK {
					state.attnScores = scores
				} else {
					fillHostMatMul(state.attnQ, seqLen, d, kt, seqLen, state.attnScores)
				}
			}
			softmaxRowsInPlace(state.attnScores, seqLen, seqLen)
			if captureBindings {
				state.attnScoresBinding = t.bindSequenceTensor(state, "scores", tensorF32View([]int{seqLen, seqLen}, state.attnScores), true, t.softmaxBackwardAccelEnabled() && bindSoftmaxActivation)
			}
		}
		batchedMixed, batchedMixedOK := t.tryBatchedAttentionMixed(states, seqLen, d)
		for i, state := range states {
			if batchedMixedOK {
				state.attnMixed = batchedMixed[i]
			} else {
				var (
					mixed    []float32
					matmulOK bool
				)
				if captureBindings {
					mixed, matmulOK = t.tryTrainerMatMulBoundRight(state.attnScores, seqLen, seqLen, state.attnVBinding, tensorF32View([]int{seqLen, d}, state.attnV), false, false)
				} else {
					mixed, matmulOK = t.tryTrainerMatMul(state.attnScores, seqLen, seqLen, state.attnV, seqLen, d, false, false)
				}
				if matmulOK {
					state.attnMixed = mixed
				} else {
					fillHostMatMul(state.attnScores, seqLen, seqLen, state.attnV, d, state.attnMixed)
				}
			}
			if captureBindings {
				state.attnMixedBinding = t.bindSequenceTensor(state, "mixed", tensorF32View([]int{seqLen, d}, state.attnMixed), true, false)
			}
		}

		t.fillBatchedForwardWeightMatMul(states, seqLen, d, func(state *embeddingSequenceState) []float32 {
			return state.attnMixed
		}, t.attnOParam.Name, attentionOutput, d, func(state *embeddingSequenceState) []float32 {
			return state.attnOutput
		}, func(state *embeddingSequenceState, data []float32) {
			state.attnOutput = data
		})
		for _, state := range states {
			if t.attentionResidualEnabled() || t.attentionLayerNormEnabled() {
				for i := range state.attnOutput {
					value := state.attnOutput[i]
					if t.attentionResidualEnabled() {
						value += state.input[i]
					}
					state.attnResidual[i] = value
				}
				if t.attentionLayerNormEnabled() {
					for row := range state.tokens {
						base := row * d
						layerNormRow(state.hidden[base:base+d], state.attnResidual[base:base+d])
					}
					if captureBindings {
						state.attnResidualBinding = t.bindSequenceTensor(state, "attn_residual", tensorF32View([]int{seqLen, d}, state.attnResidual), false, t.fullActivationBackwardAccelEnabled() && bindFullActivation)
					}
				} else {
					copy(state.hidden, state.attnResidual)
				}
			} else {
				copy(state.hidden, state.attnOutput)
			}
		}
	} else {
		for _, state := range states {
			copy(state.hidden, state.input)
		}
	}

	for _, state := range states {
		if captureBindings {
			state.hiddenBinding = t.bindSequenceTensor(state, "hidden", tensorF32View([]int{seqLen, d}, state.hidden), true, t.fullActivationBackwardAccelEnabled() && t.attentionLayerNormEnabled() && bindFullActivation)
		}
	}

	if hiddenProjection != nil {
		t.fillBatchedForwardWeightMatMul(states, seqLen, d, func(state *embeddingSequenceState) []float32 {
			return state.hidden
		}, t.hiddenParam.Name, hiddenProjection, h, func(state *embeddingSequenceState) []float32 {
			return state.ffnHidden
		}, func(state *embeddingSequenceState, data []float32) {
			state.ffnHidden = data
		})
		for _, state := range states {
			if captureBindings {
				state.ffnHiddenBinding = t.bindSequenceTensor(state, "ffn_hidden", tensorF32View([]int{seqLen, h}, state.ffnHidden), false, t.fullActivationBackwardAccelEnabled() && bindFullActivation)
			}
			for i, value := range state.ffnHidden {
				state.activated[i] = geluForward(value)
			}
			if captureBindings {
				state.activatedBinding = t.bindSequenceTensor(state, "activated", tensorF32View([]int{seqLen, h}, state.activated), true, false)
			}
		}
		t.fillBatchedForwardWeightMatMul(states, seqLen, h, func(state *embeddingSequenceState) []float32 {
			return state.activated
		}, t.projParam.Name, projection, e, func(state *embeddingSequenceState) []float32 {
			return state.ffnOutput
		}, func(state *embeddingSequenceState, data []float32) {
			state.ffnOutput = data
		})
		for _, state := range states {
			if t.ffnResidualEnabled() || t.ffnLayerNormEnabled() {
				for i := range state.ffnOutput {
					value := state.ffnOutput[i]
					if t.ffnResidualEnabled() {
						value += state.hidden[i]
					}
					state.ffnResidual[i] = value
				}
				if t.ffnLayerNormEnabled() {
					for row := range state.tokens {
						base := row * e
						layerNormRow(state.projected[base:base+e], state.ffnResidual[base:base+e])
					}
					if captureBindings {
						state.ffnResidualBinding = t.bindSequenceTensor(state, "ffn_residual", tensorF32View([]int{seqLen, e}, state.ffnResidual), false, t.fullActivationBackwardAccelEnabled() && bindFullActivation)
						state.projectedBinding = t.bindSequenceTensor(state, "projected", tensorF32View([]int{seqLen, e}, state.projected), false, t.fullActivationBackwardAccelEnabled() && bindFullActivation)
					}
				} else {
					copy(state.projected, state.ffnResidual)
				}
			} else {
				copy(state.projected, state.ffnOutput)
			}
		}
	} else {
		t.fillBatchedForwardWeightMatMul(states, seqLen, d, func(state *embeddingSequenceState) []float32 {
			return state.hidden
		}, t.projParam.Name, projection, e, func(state *embeddingSequenceState) []float32 {
			return state.projected
		}, func(state *embeddingSequenceState, data []float32) {
			state.projected = data
		})
	}

	for _, state := range states {
		if err := finalizeEncodedStatePooling(state, e); err != nil {
			return err
		}
	}
	return nil
}

func trainerF32TensorValueType() barr.ValueType {
	return barr.ValueType{
		Kind: barr.ValueTensor,
		Tensor: &barr.TensorType{
			DType: "f32",
		},
	}
}

func (t *EmbeddingTrainer) fillBatchedForwardQKVMatMul(states []*embeddingSequenceState, rows, width int, attentionQuery, attentionKey, attentionValue *backend.Tensor) bool {
	if t == nil || t.forwardMatMul == nil || len(states) == 0 || rows == 0 || width == 0 {
		return false
	}
	if !qkvMultiBoundRightEnabled() {
		return false
	}
	multi, ok := t.forwardMatMul.(backend.MultiBoundRightMatMulAccelerator)
	if !ok {
		return false
	}
	if attentionQuery == nil || attentionKey == nil || attentionValue == nil {
		return false
	}
	if len(attentionQuery.Shape) != 2 || len(attentionKey.Shape) != 2 || len(attentionValue.Shape) != 2 {
		return false
	}
	for _, tensor := range []*backend.Tensor{attentionQuery, attentionKey, attentionValue} {
		if tensor.Shape[0] != width || tensor.Shape[1] != width {
			return false
		}
	}
	perInput := rows * width
	batchedLHS := t.scratchFloat32(3, len(states)*perInput)
	for i, state := range states {
		if state == nil || len(state.input) != perInput || len(state.attnQ) != perInput || len(state.attnK) != perInput || len(state.attnV) != perInput {
			return false
		}
		copy(batchedLHS[i*perInput:(i+1)*perInput], state.input)
	}
	results, err := multi.RunMatMulWithBoundRights(
		tensorF32View([]int{len(states) * rows, width}, batchedLHS),
		[]string{t.attnQParam.Name, t.attnKParam.Name, t.attnVParam.Name},
		trainerF32TensorValueType(),
		false,
		false,
	)
	if err != nil || len(results) != 3 {
		return false
	}
	outs := make([][]float32, len(results))
	for i, result := range results {
		if len(result.Outputs) != 1 || result.Outputs[0] == nil {
			return false
		}
		out := result.Outputs[0].F32
		if len(out) != len(states)*perInput {
			return false
		}
		outs[i] = out
	}
	for i, state := range states {
		state.attnQ = outs[0][i*perInput : (i+1)*perInput]
		state.attnK = outs[1][i*perInput : (i+1)*perInput]
		state.attnV = outs[2][i*perInput : (i+1)*perInput]
	}
	return true
}

func (t *EmbeddingTrainer) fillBatchedForwardWeightMatMul(states []*embeddingSequenceState, rows, inner int, lhs func(*embeddingSequenceState) []float32, rhsName string, rhs *backend.Tensor, cols int, dst func(*embeddingSequenceState) []float32, assign func(*embeddingSequenceState, []float32)) {
	if len(states) == 0 || rhs == nil || rows == 0 || inner == 0 || cols == 0 {
		return
	}
	if t != nil && t.forwardMatMul != nil {
		perInput := rows * inner
		batchedLHS := t.scratchFloat32(3, len(states)*perInput)
		valid := true
		for i, state := range states {
			data := lhs(state)
			if len(data) != perInput {
				valid = false
				break
			}
			copy(batchedLHS[i*perInput:(i+1)*perInput], data)
		}
		if valid {
			if out, ok := t.tryForwardWeightMatMul(batchedLHS, len(states)*rows, inner, rhsName, rhs, cols); ok && len(out) == len(states)*rows*cols {
				for i, state := range states {
					view := out[i*rows*cols : (i+1)*rows*cols]
					if assign != nil {
						assign(state, view)
					} else {
						copy(dst(state), view)
					}
				}
				return
			}
		}
	}
	rhsData := forwardMatMulHostData(rhs)
	for _, state := range states {
		left := lhs(state)
		out := dst(state)
		if len(left) != rows*inner || len(out) != rows*cols {
			continue
		}
		fillHostMatMul(left, rows, inner, rhsData, cols, out)
	}
}

func (t *EmbeddingTrainer) tryBatchedAttentionScores(states []*embeddingSequenceState, seqLen, d int) ([][]float32, bool) {
	if len(states) < 2 || seqLen == 0 || d == 0 {
		return nil, false
	}
	queries := make([][]float32, len(states))
	keys := make([][]float32, len(states))
	for i, state := range states {
		if state == nil || len(state.attnQ) != seqLen*d || len(state.attnK) != seqLen*d {
			return nil, false
		}
		queries[i] = state.attnQ
		keys[i] = state.attnK
	}
	return t.tryTrainerBatchedMatMulTranspose(queries, seqLen, d, keys, seqLen, d, false, true)
}

func (t *EmbeddingTrainer) tryBatchedAttentionMixed(states []*embeddingSequenceState, seqLen, d int) ([][]float32, bool) {
	if len(states) < 2 || seqLen == 0 || d == 0 {
		return nil, false
	}
	scores := make([][]float32, len(states))
	values := make([][]float32, len(states))
	for i, state := range states {
		if state == nil || len(state.attnScores) != seqLen*seqLen || len(state.attnV) != seqLen*d {
			return nil, false
		}
		scores[i] = state.attnScores
		values[i] = state.attnV
	}
	return t.tryTrainerBatchedMatMul(scores, seqLen, seqLen, values, seqLen, d)
}

func finalizeEncodedStatePooling(state *embeddingSequenceState, e int) error {
	if state == nil {
		return fmt.Errorf("encoder state is nil")
	}
	for i := range state.normalized {
		state.normalized[i] = 0
	}
	for i := range state.pooled {
		state.pooled[i] = 0
	}
	state.activeCount = 0
	for row := range state.tokens {
		projectedBase := row * e
		norm := vectorNorm(state.projected[projectedBase : projectedBase+e])
		if norm == 0 {
			copy(state.normalized[projectedBase:projectedBase+e], state.projected[projectedBase:projectedBase+e])
		} else {
			for col := 0; col < e; col++ {
				state.normalized[projectedBase+col] = state.projected[projectedBase+col] / norm
			}
		}
		if state.mask[row] == 0 {
			continue
		}
		state.activeCount++
		for col := 0; col < e; col++ {
			state.pooled[col] += state.normalized[projectedBase+col]
		}
	}
	if state.activeCount == 0 {
		return fmt.Errorf("sequence mask selects zero tokens")
	}
	inv := 1 / float32(state.activeCount)
	for i := range state.pooled {
		state.pooled[i] *= inv
	}
	return nil
}

type contrastivePooledMatrix struct {
	rows  int
	width int
	data  []float32
	norms []float32
}

func newContrastivePooledMatrix(seqs []*embeddingEncodedSequence) contrastivePooledMatrix {
	matrix := contrastivePooledMatrix{
		rows:  len(seqs),
		norms: make([]float32, len(seqs)),
	}
	if len(seqs) == 0 || seqs[0] == nil || len(seqs[0].pooled) == 0 {
		return matrix
	}
	matrix.width = len(seqs[0].pooled)
	matrix.data = make([]float32, matrix.rows*matrix.width)
	for row, seq := range seqs {
		if seq == nil || len(seq.pooled) != matrix.width {
			continue
		}
		values := matrix.row(row)
		copy(values, seq.pooled)
		matrix.norms[row] = vectorNorm(values)
	}
	return matrix
}

func (m contrastivePooledMatrix) row(index int) []float32 {
	if index < 0 || index >= m.rows || m.width == 0 {
		return nil
	}
	base := index * m.width
	return m.data[base : base+m.width]
}

func cosineScoreWithNorms(left, right []float32, leftNorm, rightNorm float32) float32 {
	if len(left) != len(right) || len(left) == 0 || leftNorm == 0 || rightNorm == 0 {
		return 0
	}
	dot := float32(0)
	for i := range left {
		dot += left[i] * right[i]
	}
	return dot / (leftNorm * rightNorm)
}

func accumulateCosineGradFromScore(left, right []float32, leftNorm, rightNorm, score, scale float32, gradLeft, gradRight []float32) {
	if len(left) != len(right) || len(left) == 0 || leftNorm == 0 || rightNorm == 0 || len(gradLeft) < len(left) || len(gradRight) < len(right) {
		return
	}
	denom := leftNorm * rightNorm
	leftScale := score / (leftNorm * leftNorm)
	rightScale := score / (rightNorm * rightNorm)
	for i := range left {
		gradLeft[i] += scale * (right[i]/denom - left[i]*leftScale)
		gradRight[i] += scale * (left[i]/denom - right[i]*rightScale)
	}
}

func evaluateContrastiveEncodings(queries, positives []*embeddingEncodedSequence, cfg EmbeddingTrainConfig) EmbeddingEvalMetrics {
	metrics := EmbeddingEvalMetrics{
		PairCount: len(queries) * len(positives),
	}
	infoNCELoss := float32(0)
	queryMatrix := newContrastivePooledMatrix(queries)
	positiveMatrix := newContrastivePooledMatrix(positives)
	rowScores := make([]float32, len(positives))
	rowProbs := make([]float32, len(positives))
	scores := make([]embeddingEvalScore, 0, metrics.PairCount)
	for i := range queries {
		query := queryMatrix.row(i)
		queryNorm := queryMatrix.norms[i]
		for j := range positives {
			target := float32(-1)
			if i == j {
				target = 1
			}
			score := cosineScoreWithNorms(query, positiveMatrix.row(j), queryNorm, positiveMatrix.norms[j])
			scores = append(scores, embeddingEvalScore{Score: score, Positive: target > 0})
			rowScores[j] = score
			scale := score - target
			if cfg.ContrastiveLoss != "infonce" {
				metrics.Loss += 0.5 * scale * scale
			}
			metrics.AverageScore += score
			if target > 0 {
				metrics.PositiveCount++
				metrics.PositiveMeanScore += score
				if score > 0 {
					metrics.PairAccuracy++
				}
			} else {
				metrics.NegativeCount++
				metrics.NegativeMeanScore += score
				if score < 0 {
					metrics.PairAccuracy++
				}
			}
		}
		if len(rowScores) > 0 && i < len(rowScores) {
			best := 0
			for j := 1; j < len(rowScores); j++ {
				if rowScores[j] > rowScores[best] {
					best = j
				}
			}
			if best == i {
				metrics.Top1Accuracy++
			}
			rank := 1
			targetScore := rowScores[i]
			for j, score := range rowScores {
				if j != i && score > targetScore {
					rank++
				}
			}
			if rank <= 5 {
				metrics.Top5Accuracy++
			}
			if rank <= 10 {
				metrics.Top10Accuracy++
			}
			metrics.MeanReciprocalRank += 1 / float32(rank)
			metrics.MeanPositiveRank += float32(rank)
		}
		if cfg.ContrastiveLoss == "infonce" {
			infoNCELoss += infoNCERowProbsAndLossInto(rowScores, i, cfg.Temperature, rowProbs)
		}
	}
	if metrics.PairCount == 0 {
		return metrics
	}
	invPairs := float32(1) / float32(metrics.PairCount)
	if cfg.ContrastiveLoss == "infonce" && len(queries) > 0 {
		metrics.Loss = infoNCELoss / float32(len(queries))
	} else {
		metrics.Loss *= invPairs
	}
	metrics.AverageScore *= invPairs
	metrics.PairAccuracy *= invPairs
	if metrics.PositiveCount > 0 {
		metrics.PositiveMeanScore /= float32(metrics.PositiveCount)
	}
	if metrics.NegativeCount > 0 {
		metrics.NegativeMeanScore /= float32(metrics.NegativeCount)
	}
	if len(queries) > 0 {
		invRows := float32(1) / float32(len(queries))
		metrics.Top1Accuracy *= invRows
		metrics.Top5Accuracy *= invRows
		metrics.Top10Accuracy *= invRows
		metrics.MeanReciprocalRank *= invRows
		metrics.MeanPositiveRank *= invRows
	}
	metrics.ScoreMargin = metrics.PositiveMeanScore - metrics.NegativeMeanScore
	finalizeEvalScoreMetrics(&metrics, scores)
	return metrics
}

func (t *EmbeddingTrainer) tryInfoNCEContrastiveAccelerator(queries, positives []*embeddingEncodedSequence, queryGrads, positiveGrads [][]float32) (float32, float32, bool) {
	if t == nil || t.contrastiveAccel == nil || len(queries) == 0 || len(queries) != len(positives) {
		return 0, 0, false
	}
	queryMatrix := newContrastivePooledMatrix(queries)
	positiveMatrix := newContrastivePooledMatrix(positives)
	if queryMatrix.rows == 0 || queryMatrix.width == 0 || positiveMatrix.rows != queryMatrix.rows || positiveMatrix.width != queryMatrix.width {
		return 0, 0, false
	}
	result, err := t.contrastiveAccel.RunInfoNCE(
		tensorF32View([]int{queryMatrix.rows, queryMatrix.width}, queryMatrix.data),
		tensorF32View([]int{positiveMatrix.rows, positiveMatrix.width}, positiveMatrix.data),
		backend.ContrastiveLossConfig{Temperature: t.config.Temperature},
	)
	if err != nil || result.QueryGrads == nil || result.PositiveGrads == nil {
		return 0, 0, false
	}
	if result.QueryGrads.Rank() != 2 || result.PositiveGrads.Rank() != 2 ||
		result.QueryGrads.Shape[0] != queryMatrix.rows || result.QueryGrads.Shape[1] != queryMatrix.width ||
		result.PositiveGrads.Shape[0] != positiveMatrix.rows || result.PositiveGrads.Shape[1] != positiveMatrix.width {
		return 0, 0, false
	}
	for row := range queryGrads {
		copy(queryGrads[row], result.QueryGrads.F32[row*queryMatrix.width:(row+1)*queryMatrix.width])
		copy(positiveGrads[row], result.PositiveGrads.F32[row*positiveMatrix.width:(row+1)*positiveMatrix.width])
	}
	return result.LossSum, result.ScoreSum, true
}

func accumulatePairMSEContrastiveGrads(queries, positives []*embeddingEncodedSequence, queryGrads, positiveGrads [][]float32) (float32, float32) {
	totalLoss := float32(0)
	totalScore := float32(0)
	queryMatrix := newContrastivePooledMatrix(queries)
	positiveMatrix := newContrastivePooledMatrix(positives)
	for i := range queries {
		query := queryMatrix.row(i)
		queryNorm := queryMatrix.norms[i]
		for j := range positives {
			target := float32(-1)
			if i == j {
				target = 1
			}
			score := cosineScoreWithNorms(query, positiveMatrix.row(j), queryNorm, positiveMatrix.norms[j])
			scale := score - target
			totalLoss += 0.5 * scale * scale
			totalScore += score
			accumulateCosineGradFromScore(query, positiveMatrix.row(j), queryNorm, positiveMatrix.norms[j], score, scale, queryGrads[i], positiveGrads[j])
		}
	}
	return totalLoss, totalScore
}

func accumulateInfoNCEContrastiveGrads(queries, positives []*embeddingEncodedSequence, temperature float32, queryGrads, positiveGrads [][]float32) (float32, float32) {
	totalLoss := float32(0)
	totalScore := float32(0)
	if temperature <= 0 {
		temperature = 0.05
	}
	queryMatrix := newContrastivePooledMatrix(queries)
	positiveMatrix := newContrastivePooledMatrix(positives)
	rowScores := make([]float32, len(positives))
	rowProbs := make([]float32, len(positives))
	for i := range queries {
		query := queryMatrix.row(i)
		queryNorm := queryMatrix.norms[i]
		for j := range positives {
			score := cosineScoreWithNorms(query, positiveMatrix.row(j), queryNorm, positiveMatrix.norms[j])
			rowScores[j] = score
			totalScore += score
		}
		rowLoss := infoNCERowProbsAndLossInto(rowScores, i, temperature, rowProbs)
		totalLoss += rowLoss
		for j, prob := range rowProbs {
			target := float32(0)
			if i == j {
				target = 1
			}
			scale := (prob - target) / temperature
			accumulateCosineGradFromScore(query, positiveMatrix.row(j), queryNorm, positiveMatrix.norms[j], rowScores[j], scale, queryGrads[i], positiveGrads[j])
		}
	}
	return totalLoss, totalScore
}

func infoNCERowLoss(scores []float32, targetIndex int, temperature float32) float32 {
	_, loss := infoNCERowProbsAndLoss(scores, targetIndex, temperature)
	return loss
}

func infoNCERowProbsAndLoss(scores []float32, targetIndex int, temperature float32) ([]float32, float32) {
	probs := make([]float32, len(scores))
	loss := infoNCERowProbsAndLossInto(scores, targetIndex, temperature, probs)
	return probs, loss
}

func infoNCERowProbsAndLossInto(scores []float32, targetIndex int, temperature float32, probs []float32) float32 {
	if len(scores) == 0 || targetIndex < 0 || targetIndex >= len(scores) {
		for i := range probs {
			probs[i] = 0
		}
		return 0
	}
	if temperature <= 0 {
		temperature = 0.05
	}
	if len(probs) < len(scores) {
		return 0
	}
	probs = probs[:len(scores)]
	maxLogit := scores[0] / temperature
	for _, score := range scores[1:] {
		logit := score / temperature
		if logit > maxLogit {
			maxLogit = logit
		}
	}
	sum := float32(0)
	for i, score := range scores {
		value := float32(math.Exp(float64(score/temperature - maxLogit)))
		probs[i] = value
		sum += value
	}
	if sum == 0 {
		return 0
	}
	invSum := 1 / sum
	for i := range probs {
		probs[i] *= invSum
	}
	prob := probs[targetIndex]
	if prob < 1e-12 {
		prob = 1e-12
	}
	return -float32(math.Log(float64(prob)))
}

func (t *EmbeddingTrainer) prepareForwardWeights() *embeddingForwardWeights {
	if t == nil {
		return nil
	}
	if t.forwardCache != nil && !t.forwardDirty {
		return t.forwardCache
	}
	if t.forwardCache == nil {
		t.forwardCache = &embeddingForwardWeights{}
	}
	t.forwardCache.token = refreshForwardTensorForParam(t.tokenParam, t.tokenEmbed, t.config.WeightBits, t.forwardCache.token)
	t.forwardCache.attnQ = refreshForwardMatMulTensorForParam(t.attnQParam, t.attentionQuery, t.forwardCache.attnQ)
	t.forwardCache.attnK = refreshForwardMatMulTensorForParam(t.attnKParam, t.attentionKey, t.forwardCache.attnK)
	t.forwardCache.attnV = refreshForwardMatMulTensorForParam(t.attnVParam, t.attentionValue, t.forwardCache.attnV)
	t.forwardCache.attnO = refreshForwardMatMulTensorForParam(t.attnOParam, t.attentionOutput, t.forwardCache.attnO)
	t.forwardCache.hidden = refreshForwardMatMulTensorForParam(t.hiddenParam, t.hiddenProjection, t.forwardCache.hidden)
	t.forwardCache.proj = refreshForwardMatMulTensorForParam(t.projParam, t.projection, t.forwardCache.proj)
	t.forwardDirty = false
	return t.forwardCache
}

func (t *EmbeddingTrainer) invalidateForwardWeights() {
	if t == nil {
		return
	}
	t.forwardDirty = true
	t.forwardNeedsBind = true
}

func (t *EmbeddingTrainer) encodeExamplePair(example EmbeddingPairExample, tokenForward, attnQForward, attnKForward, attnVForward, attnOForward, hiddenForward, projForward *backend.Tensor, captureBindings bool) (*embeddingEncodedSequence, *embeddingEncodedSequence, error) {
	leftMask, err := t.prepareMask(example.LeftTokens, example.LeftMask)
	if err != nil {
		return nil, nil, fmt.Errorf("left: %w", err)
	}
	rightMask, err := t.prepareMask(example.RightTokens, example.RightMask)
	if err != nil {
		return nil, nil, fmt.Errorf("right: %w", err)
	}
	left, err := t.encodeSequence(example.LeftTokens, leftMask, tokenForward, attnQForward, attnKForward, attnVForward, attnOForward, hiddenForward, projForward, captureBindings)
	if err != nil {
		return nil, nil, fmt.Errorf("left: %w", err)
	}
	right, err := t.encodeSequence(example.RightTokens, rightMask, tokenForward, attnQForward, attnKForward, attnVForward, attnOForward, hiddenForward, projForward, captureBindings)
	if err != nil {
		t.releaseEncodedSequenceBindings(left)
		return nil, nil, fmt.Errorf("right: %w", err)
	}
	return left, right, nil
}

func (t *EmbeddingTrainer) prepareMask(tokens, mask []int32) ([]int32, error) {
	if err := t.validateTokenSequence(tokens); err != nil {
		return nil, err
	}
	if len(mask) == 0 {
		mask = make([]int32, len(tokens))
		for i := range mask {
			mask[i] = 1
		}
		return mask, nil
	}
	if len(mask) != len(tokens) {
		return nil, fmt.Errorf("mask length %d does not match token length %d", len(mask), len(tokens))
	}
	active := 0
	for _, v := range mask {
		if v != 0 {
			active++
		}
	}
	if active == 0 {
		return nil, fmt.Errorf("mask selects zero tokens")
	}
	return append([]int32(nil), mask...), nil
}

func (t *EmbeddingTrainer) validateTokenSequence(tokens []int32) error {
	if len(tokens) == 0 {
		return fmt.Errorf("tokens are empty")
	}
	if limit := t.manifest.Tokenizer.MaxSequence; limit > 0 && len(tokens) > limit {
		return fmt.Errorf("token sequence length %d exceeds max_sequence %d", len(tokens), limit)
	}
	if vocab := t.manifest.Tokenizer.VocabSize; vocab > 0 {
		for i, tok := range tokens {
			if tok < 0 || int(tok) >= vocab {
				return fmt.Errorf("token %d value %d is outside vocab_size %d", i, tok, vocab)
			}
		}
	}
	return nil
}

func (t *EmbeddingTrainer) encodeSequence(tokens, mask []int32, tokenEmbed, attentionQuery, attentionKey, attentionValue, attentionOutput, hiddenProjection, projection *backend.Tensor, captureBindings bool) (*embeddingEncodedSequence, error) {
	input, err := embeddingInputForTokens(tokenEmbed, tokens)
	if err != nil {
		return nil, err
	}
	encoded := &embeddingEncodedSequence{
		layers: make([]*embeddingSequenceState, 0, t.encoderRepeats()),
		pooled: nil,
		tokens: append([]int32(nil), tokens...),
	}
	current := input
	for layer := 0; layer < t.encoderRepeats(); layer++ {
		state, err := t.encodeLayer(tokens, mask, current, attentionQuery, attentionKey, attentionValue, attentionOutput, hiddenProjection, projection, captureBindings)
		if err != nil {
			t.releaseEncodedSequenceBindings(encoded)
			return nil, err
		}
		encoded.layers = append(encoded.layers, state)
		current = state.projected
	}
	if len(encoded.layers) == 0 {
		return nil, fmt.Errorf("encoder produced zero layers")
	}
	encoded.pooled = append([]float32(nil), encoded.layers[len(encoded.layers)-1].pooled...)
	return encoded, nil
}

func (t *EmbeddingTrainer) encodeLayer(tokens, mask []int32, input []float32, attentionQuery, attentionKey, attentionValue, attentionOutput, hiddenProjection, projection *backend.Tensor, captureBindings bool) (*embeddingSequenceState, error) {
	d := projection.Shape[0]
	h := 0
	if hiddenProjection != nil {
		d = hiddenProjection.Shape[0]
		h = hiddenProjection.Shape[1]
	}
	e := projection.Shape[1]
	state, err := newEmbeddingSequenceState(tokens, mask, input, hiddenProjection, projection)
	if err != nil {
		return nil, err
	}
	if captureBindings {
		state.inputBinding = t.bindSequenceTensor(state, "input", tensorF32View([]int{len(tokens), d}, state.input), true, false)
	}
	if attentionQuery != nil && attentionKey != nil && attentionValue != nil && attentionOutput != nil {
		q, ok := t.tryForwardWeightMatMul(state.input, len(tokens), d, t.attnQParam.Name, attentionQuery, d)
		if ok {
			copy(state.attnQ, q)
		} else {
			attnQData := forwardMatMulHostData(attentionQuery)
			fillHostMatMul(state.input, len(tokens), d, attnQData, d, state.attnQ)
		}
		if captureBindings {
			state.attnQBinding = t.bindSequenceTensor(state, "q", tensorF32View([]int{len(tokens), d}, state.attnQ), true, false)
		}
		k, ok := t.tryForwardWeightMatMul(state.input, len(tokens), d, t.attnKParam.Name, attentionKey, d)
		if ok {
			copy(state.attnK, k)
		} else {
			attnKData := forwardMatMulHostData(attentionKey)
			fillHostMatMul(state.input, len(tokens), d, attnKData, d, state.attnK)
		}
		if captureBindings {
			state.attnKBinding = t.bindSequenceTensor(state, "k", tensorF32View([]int{len(tokens), d}, state.attnK), true, false)
		}
		v, ok := t.tryForwardWeightMatMul(state.input, len(tokens), d, t.attnVParam.Name, attentionValue, d)
		if ok {
			copy(state.attnV, v)
		} else {
			attnVData := forwardMatMulHostData(attentionValue)
			fillHostMatMul(state.input, len(tokens), d, attnVData, d, state.attnV)
		}
		if captureBindings {
			state.attnVBinding = t.bindSequenceTensor(state, "v", tensorF32View([]int{len(tokens), d}, state.attnV), true, false)
		}
		var (
			scores   []float32
			mixed    []float32
			matmulOK bool
		)
		if captureBindings {
			scores, matmulOK = t.tryTrainerMatMulBoundRight(state.attnQ, len(tokens), d, state.attnKBinding, tensorF32View([]int{len(tokens), d}, state.attnK), false, true)
		} else {
			scores, matmulOK = t.tryTrainerMatMul(state.attnQ, len(tokens), d, state.attnK, len(tokens), d, false, true)
		}
		if matmulOK {
			copy(state.attnScores, scores)
		} else {
			kt := transpose2DData(state.attnK, len(tokens), d)
			fillHostMatMul(state.attnQ, len(tokens), d, kt, len(tokens), state.attnScores)
		}
		softmaxRowsInPlace(state.attnScores, len(tokens), len(tokens))
		if captureBindings {
			state.attnScoresBinding = t.bindSequenceTensor(state, "scores", tensorF32View([]int{len(tokens), len(tokens)}, state.attnScores), true, t.softmaxBackwardAccelEnabled())
		}
		if captureBindings {
			mixed, matmulOK = t.tryTrainerMatMulBoundRight(state.attnScores, len(tokens), len(tokens), state.attnVBinding, tensorF32View([]int{len(tokens), d}, state.attnV), false, false)
		} else {
			mixed, matmulOK = t.tryTrainerMatMul(state.attnScores, len(tokens), len(tokens), state.attnV, len(tokens), d, false, false)
		}
		if matmulOK {
			copy(state.attnMixed, mixed)
		} else {
			fillHostMatMul(state.attnScores, len(tokens), len(tokens), state.attnV, d, state.attnMixed)
		}
		if captureBindings {
			state.attnMixedBinding = t.bindSequenceTensor(state, "mixed", tensorF32View([]int{len(tokens), d}, state.attnMixed), true, false)
		}
		output, ok := t.tryForwardWeightMatMul(state.attnMixed, len(tokens), d, t.attnOParam.Name, attentionOutput, d)
		if ok {
			copy(state.attnOutput, output)
		} else {
			attnOData := forwardMatMulHostData(attentionOutput)
			fillHostMatMul(state.attnMixed, len(tokens), d, attnOData, d, state.attnOutput)
		}
		if t.attentionResidualEnabled() || t.attentionLayerNormEnabled() {
			for i := range state.attnOutput {
				value := state.attnOutput[i]
				if t.attentionResidualEnabled() {
					value += state.input[i]
				}
				state.attnResidual[i] = value
			}
			if t.attentionLayerNormEnabled() {
				for row := range tokens {
					base := row * d
					layerNormRow(state.hidden[base:base+d], state.attnResidual[base:base+d])
				}
				if captureBindings {
					state.attnResidualBinding = t.bindSequenceTensor(state, "attn_residual", tensorF32View([]int{len(tokens), d}, state.attnResidual), false, t.fullActivationBackwardAccelEnabled())
				}
			} else {
				copy(state.hidden, state.attnResidual)
			}
		} else {
			copy(state.hidden, state.attnOutput)
		}
	} else {
		copy(state.hidden, state.input)
	}
	if captureBindings {
		state.hiddenBinding = t.bindSequenceTensor(state, "hidden", tensorF32View([]int{len(tokens), d}, state.hidden), true, t.fullActivationBackwardAccelEnabled() && t.attentionLayerNormEnabled())
	}
	if hiddenProjection != nil {
		ffnHidden, ok := t.tryForwardWeightMatMul(state.hidden, len(tokens), d, t.hiddenParam.Name, hiddenProjection, h)
		if ok {
			copy(state.ffnHidden, ffnHidden)
		} else {
			hiddenData := forwardMatMulHostData(hiddenProjection)
			fillHostMatMul(state.hidden, len(tokens), d, hiddenData, h, state.ffnHidden)
		}
		if captureBindings {
			state.ffnHiddenBinding = t.bindSequenceTensor(state, "ffn_hidden", tensorF32View([]int{len(tokens), h}, state.ffnHidden), false, t.fullActivationBackwardAccelEnabled())
		}
		for i, value := range state.ffnHidden {
			state.activated[i] = geluForward(value)
		}
		if captureBindings {
			state.activatedBinding = t.bindSequenceTensor(state, "activated", tensorF32View([]int{len(tokens), h}, state.activated), true, false)
		}
		projected, ok := t.tryForwardWeightMatMul(state.activated, len(tokens), h, t.projParam.Name, projection, e)
		if ok {
			copy(state.ffnOutput, projected)
		} else {
			projData := forwardMatMulHostData(projection)
			fillHostMatMul(state.activated, len(tokens), h, projData, e, state.ffnOutput)
		}
		if t.ffnResidualEnabled() || t.ffnLayerNormEnabled() {
			for i := range state.ffnOutput {
				value := state.ffnOutput[i]
				if t.ffnResidualEnabled() {
					value += state.hidden[i]
				}
				state.ffnResidual[i] = value
			}
			if t.ffnLayerNormEnabled() {
				for row := range tokens {
					base := row * e
					layerNormRow(state.projected[base:base+e], state.ffnResidual[base:base+e])
				}
				if captureBindings {
					state.ffnResidualBinding = t.bindSequenceTensor(state, "ffn_residual", tensorF32View([]int{len(tokens), e}, state.ffnResidual), false, t.fullActivationBackwardAccelEnabled())
					state.projectedBinding = t.bindSequenceTensor(state, "projected", tensorF32View([]int{len(tokens), e}, state.projected), false, t.fullActivationBackwardAccelEnabled())
				}
			} else {
				copy(state.projected, state.ffnResidual)
			}
		} else {
			copy(state.projected, state.ffnOutput)
		}
	} else {
		projected, ok := t.tryForwardWeightMatMul(state.hidden, len(tokens), d, t.projParam.Name, projection, e)
		if ok {
			copy(state.projected, projected)
		} else {
			projData := forwardMatMulHostData(projection)
			fillHostMatMul(state.hidden, len(tokens), d, projData, e, state.projected)
		}
	}
	for row := range tokens {
		projectedBase := row * e
		norm := vectorNorm(state.projected[projectedBase : projectedBase+e])
		if norm == 0 {
			copy(state.normalized[projectedBase:projectedBase+e], state.projected[projectedBase:projectedBase+e])
		} else {
			for col := 0; col < e; col++ {
				state.normalized[projectedBase+col] = state.projected[projectedBase+col] / norm
			}
		}
		if mask[row] == 0 {
			continue
		}
		state.activeCount++
		for col := 0; col < e; col++ {
			state.pooled[col] += state.normalized[projectedBase+col]
		}
	}
	if state.activeCount == 0 {
		return nil, fmt.Errorf("sequence mask selects zero tokens")
	}
	inv := 1 / float32(state.activeCount)
	for i := range state.pooled {
		state.pooled[i] *= inv
	}
	return state, nil
}

func embeddingInputForTokens(tokenEmbed *backend.Tensor, tokens []int32) ([]float32, error) {
	if tokenEmbed == nil || len(tokenEmbed.Shape) != 2 {
		return nil, fmt.Errorf("token embedding must be rank-2")
	}
	d := tokenEmbed.Shape[1]
	input := make([]float32, len(tokens)*d)
	for row, tok := range tokens {
		if tok < 0 || int(tok) >= tokenEmbed.Shape[0] {
			return nil, fmt.Errorf("token %d is out of range", tok)
		}
		copy(input[row*d:(row+1)*d], tokenEmbed.F32[int(tok)*d:(int(tok)+1)*d])
	}
	return input, nil
}

func newEmbeddingSequenceState(tokens, mask []int32, input []float32, hiddenProjection, projection *backend.Tensor) (*embeddingSequenceState, error) {
	if projection == nil || len(projection.Shape) != 2 {
		return nil, fmt.Errorf("projection must be rank-2")
	}
	d := stateWidth(hiddenProjection, projection)
	if len(input) != len(tokens)*d {
		return nil, fmt.Errorf("encoder layer input size %d does not match tokens=%d width=%d", len(input), len(tokens), d)
	}
	h := 0
	if hiddenProjection != nil {
		h = hiddenProjection.Shape[1]
	}
	e := projection.Shape[1]
	state := &embeddingSequenceState{
		tokens:       append([]int32(nil), tokens...),
		mask:         append([]int32(nil), mask...),
		input:        input,
		hidden:       make([]float32, len(tokens)*d),
		attnQ:        make([]float32, len(tokens)*d),
		attnK:        make([]float32, len(tokens)*d),
		attnV:        make([]float32, len(tokens)*d),
		attnScores:   make([]float32, len(tokens)*len(tokens)),
		attnMixed:    make([]float32, len(tokens)*d),
		attnOutput:   make([]float32, len(tokens)*d),
		attnResidual: make([]float32, len(tokens)*d),
		ffnHidden:    make([]float32, len(tokens)*h),
		activated:    make([]float32, len(tokens)*h),
		ffnOutput:    make([]float32, len(tokens)*e),
		ffnResidual:  make([]float32, len(tokens)*e),
		projected:    make([]float32, len(tokens)*e),
		normalized:   make([]float32, len(tokens)*e),
		pooled:       make([]float32, e),
		activeCount:  0,
	}
	return state, nil
}

func stateWidth(hiddenProjection, projection *backend.Tensor) int {
	if hiddenProjection != nil {
		return hiddenProjection.Shape[0]
	}
	return projection.Shape[0]
}

func tensorF32View(shape []int, data []float32) *backend.Tensor {
	return &backend.Tensor{
		DType: "f32",
		Shape: append([]int(nil), shape...),
		F32:   data,
	}
}

func (t *EmbeddingTrainer) scratchFloat32(slot, elements int) []float32 {
	if elements <= 0 {
		return nil
	}
	if t == nil {
		return make([]float32, elements)
	}
	for len(t.scratchF32) <= slot {
		t.scratchF32 = append(t.scratchF32, nil)
	}
	buf := t.scratchF32[slot]
	if cap(buf) < elements {
		buf = make([]float32, elements)
	} else {
		buf = buf[:elements]
	}
	t.scratchF32[slot] = buf
	return buf
}

func (t *EmbeddingTrainer) flattenFixedFloat32MatricesScratch(slot int, matrices [][]float32, perMatrix int) ([]float32, bool) {
	if len(matrices) == 0 || perMatrix <= 0 {
		return nil, false
	}
	if out, ok := contiguousFixedFloat32Matrices(matrices, perMatrix); ok {
		return out, true
	}
	out := t.scratchFloat32(slot, len(matrices)*perMatrix)
	if !flattenFixedFloat32MatricesInto(out, matrices, perMatrix) {
		return nil, false
	}
	return out, true
}

func flattenFixedFloat32Matrices(matrices [][]float32, perMatrix int) ([]float32, bool) {
	if len(matrices) == 0 || perMatrix <= 0 {
		return nil, false
	}
	if out, ok := contiguousFixedFloat32Matrices(matrices, perMatrix); ok {
		return out, true
	}
	out := make([]float32, len(matrices)*perMatrix)
	if !flattenFixedFloat32MatricesInto(out, matrices, perMatrix) {
		return nil, false
	}
	return out, true
}

func contiguousFixedFloat32Matrices(matrices [][]float32, perMatrix int) ([]float32, bool) {
	if len(matrices) == 0 || perMatrix <= 0 {
		return nil, false
	}
	total := len(matrices) * perMatrix
	first := matrices[0]
	if len(first) != perMatrix || cap(first) < total {
		return nil, false
	}
	out := first[:total]
	for i, matrix := range matrices {
		if len(matrix) != perMatrix || &out[i*perMatrix] != &matrix[0] {
			return nil, false
		}
	}
	return out, true
}

func flattenFixedFloat32MatricesInto(out []float32, matrices [][]float32, perMatrix int) bool {
	if len(out) != len(matrices)*perMatrix {
		return false
	}
	for i, matrix := range matrices {
		if len(matrix) != perMatrix {
			return false
		}
		copy(out[i*perMatrix:(i+1)*perMatrix], matrix)
	}
	return true
}

func splitFloat32Views(out []float32, parts int) ([][]float32, bool) {
	if parts <= 0 || len(out)%parts != 0 {
		return nil, false
	}
	perMatrix := len(out) / parts
	result := make([][]float32, parts)
	for i := range result {
		result[i] = out[i*perMatrix : (i+1)*perMatrix]
	}
	return result, true
}

func (t *EmbeddingTrainer) tryForwardMatMul(lhsData []float32, rows, inner int, rhs *backend.Tensor, cols int) ([]float32, bool) {
	if rhs == nil {
		return nil, false
	}
	return t.tryTrainerMatMul(lhsData, rows, inner, rhs.F32, rhs.Shape[0], rhs.Shape[1], false, false)
}

func (t *EmbeddingTrainer) tryForwardWeightMatMul(lhsData []float32, rows, inner int, rhsName string, rhs *backend.Tensor, cols int) ([]float32, bool) {
	if rhs == nil {
		return nil, false
	}
	return t.tryTrainerMatMulBoundRight(lhsData, rows, inner, rhsName, rhs, false, false)
}

func (t *EmbeddingTrainer) tryTrainerMatMul(lhsData []float32, lhsRows, lhsCols int, rhsData []float32, rhsRows, rhsCols int, transposeLeft, transposeRight bool) ([]float32, bool) {
	if t == nil || t.forwardMatMul == nil || lhsRows == 0 || lhsCols == 0 || rhsRows == 0 || rhsCols == 0 {
		return nil, false
	}
	outRows, outCols, ok := trainerMatMulShape(lhsRows, lhsCols, rhsRows, rhsCols, transposeLeft, transposeRight)
	if !ok {
		return nil, false
	}
	result, err := t.forwardMatMul.RunMatMulWithTranspose(
		[]*backend.Tensor{
			tensorF32View([]int{lhsRows, lhsCols}, lhsData),
			tensorF32View([]int{rhsRows, rhsCols}, rhsData),
		},
		barr.ValueType{
			Kind: barr.ValueTensor,
			Tensor: &barr.TensorType{
				DType: "f32",
			},
		},
		transposeLeft,
		transposeRight,
	)
	if err != nil || len(result.Outputs) != 1 || result.Outputs[0] == nil {
		return nil, false
	}
	out := result.Outputs[0].F32
	if len(out) != outRows*outCols {
		return nil, false
	}
	return out, true
}

func (t *EmbeddingTrainer) tryTrainerBatchedMatMul(lhsMatrices [][]float32, lhsRows, lhsCols int, rhsMatrices [][]float32, rhsRows, rhsCols int) ([][]float32, bool) {
	return t.tryTrainerBatchedMatMulTranspose(lhsMatrices, lhsRows, lhsCols, rhsMatrices, rhsRows, rhsCols, false, false)
}

func (t *EmbeddingTrainer) tryTrainerBatchedMatMulTranspose(lhsMatrices [][]float32, lhsRows, lhsCols int, rhsMatrices [][]float32, rhsRows, rhsCols int, transposeLeft, transposeRight bool) ([][]float32, bool) {
	if t == nil || t.forwardMatMul == nil || len(lhsMatrices) == 0 || len(lhsMatrices) != len(rhsMatrices) || lhsRows == 0 || lhsCols == 0 || rhsRows == 0 || rhsCols == 0 {
		return nil, false
	}
	outRows, outCols, ok := trainerMatMulShape(lhsRows, lhsCols, rhsRows, rhsCols, transposeLeft, transposeRight)
	if !ok {
		return nil, false
	}
	batches := len(lhsMatrices)
	lhsBatch, ok := t.flattenFixedFloat32MatricesScratch(0, lhsMatrices, lhsRows*lhsCols)
	if !ok {
		return nil, false
	}
	rhsBatch, ok := t.flattenFixedFloat32MatricesScratch(1, rhsMatrices, rhsRows*rhsCols)
	if !ok {
		return nil, false
	}
	result, err := t.forwardMatMul.RunMatMulWithTranspose(
		[]*backend.Tensor{
			tensorF32View([]int{batches, lhsRows, lhsCols}, lhsBatch),
			tensorF32View([]int{batches, rhsRows, rhsCols}, rhsBatch),
		},
		barr.ValueType{
			Kind: barr.ValueTensor,
			Tensor: &barr.TensorType{
				DType: "f32",
			},
		},
		transposeLeft,
		transposeRight,
	)
	if err != nil || len(result.Outputs) != 1 || result.Outputs[0] == nil {
		return nil, false
	}
	perMatrix := outRows * outCols
	out := result.Outputs[0].F32
	if len(out) != batches*perMatrix {
		return nil, false
	}
	return splitFloat32Views(out, batches)
}

func (t *EmbeddingTrainer) tryTrainerMatMulBoundLeft(lhsName string, lhs, rhs *backend.Tensor, transposeLeft, transposeRight bool) ([]float32, bool) {
	if lhs == nil || rhs == nil {
		return nil, false
	}
	if t != nil && t.forwardMatMul != nil && lhsName != "" {
		outRows, outCols, ok := trainerMatMulShape(lhs.Shape[0], lhs.Shape[1], rhs.Shape[0], rhs.Shape[1], transposeLeft, transposeRight)
		if ok {
			result, err := t.forwardMatMul.RunMatMulWithBoundLeft(
				lhsName,
				rhs,
				barr.ValueType{
					Kind: barr.ValueTensor,
					Tensor: &barr.TensorType{
						DType: "f32",
					},
				},
				transposeLeft,
				transposeRight,
			)
			if err == nil && len(result.Outputs) == 1 && result.Outputs[0] != nil {
				out := result.Outputs[0].F32
				if len(out) == outRows*outCols {
					return out, true
				}
			}
		}
	}
	return t.tryTrainerMatMul(lhs.F32, lhs.Shape[0], lhs.Shape[1], rhs.F32, rhs.Shape[0], rhs.Shape[1], transposeLeft, transposeRight)
}

func (t *EmbeddingTrainer) tryTrainerMatMulBoundRight(lhsData []float32, lhsRows, lhsCols int, rhsName string, rhs *backend.Tensor, transposeLeft, transposeRight bool) ([]float32, bool) {
	if rhs == nil {
		return nil, false
	}
	if t != nil && t.forwardMatMul != nil && rhsName != "" {
		outRows, outCols, ok := trainerMatMulShape(lhsRows, lhsCols, rhs.Shape[0], rhs.Shape[1], transposeLeft, transposeRight)
		if ok {
			result, err := t.forwardMatMul.RunMatMulWithBoundRight(
				tensorF32View([]int{lhsRows, lhsCols}, lhsData),
				rhsName,
				barr.ValueType{
					Kind: barr.ValueTensor,
					Tensor: &barr.TensorType{
						DType: "f32",
					},
				},
				transposeLeft,
				transposeRight,
			)
			if err == nil && len(result.Outputs) == 1 && result.Outputs[0] != nil {
				out := result.Outputs[0].F32
				if len(out) == outRows*outCols {
					return out, true
				}
			}
		}
	}
	return t.tryTrainerMatMul(lhsData, lhsRows, lhsCols, rhs.F32, rhs.Shape[0], rhs.Shape[1], transposeLeft, transposeRight)
}

func fillHostMatMul(lhs []float32, rows, inner int, rhs []float32, cols int, out []float32) {
	fillHostMatMulTranspose(lhs, rows, inner, rhs, inner, cols, false, false, out)
}

func fillHostMatMulTranspose(lhs []float32, lhsRows, lhsCols int, rhs []float32, rhsRows, rhsCols int, transposeLeft, transposeRight bool, out []float32) {
	rows, inner, cols, ok := trainerMatMulDims(lhsRows, lhsCols, rhsRows, rhsCols, transposeLeft, transposeRight)
	if !ok {
		for i := range out {
			out[i] = 0
		}
		return
	}
	for row := 0; row < rows; row++ {
		outBase := row * cols
		for col := 0; col < cols; col++ {
			sum := float32(0)
			for kk := 0; kk < inner; kk++ {
				sum += trainerMatMulAt(lhs, lhsRows, lhsCols, row, kk, transposeLeft) * trainerMatMulAt(rhs, rhsRows, rhsCols, kk, col, transposeRight)
			}
			out[outBase+col] = sum
		}
	}
}

func addFloat32Slice(dst, src []float32) {
	limit := len(dst)
	if len(src) < limit {
		limit = len(src)
	}
	for i := 0; i < limit; i++ {
		dst[i] += src[i]
	}
}

func trainerMatMulShape(lhsRows, lhsCols, rhsRows, rhsCols int, transposeLeft, transposeRight bool) (rows, cols int, ok bool) {
	rows, _, cols, ok = trainerMatMulDims(lhsRows, lhsCols, rhsRows, rhsCols, transposeLeft, transposeRight)
	return rows, cols, ok
}

func trainerMatMulDims(lhsRows, lhsCols, rhsRows, rhsCols int, transposeLeft, transposeRight bool) (rows, inner, cols int, ok bool) {
	rows = lhsRows
	inner = lhsCols
	if transposeLeft {
		rows, inner = lhsCols, lhsRows
	}
	rhsInner := rhsRows
	cols = rhsCols
	if transposeRight {
		rhsInner, cols = rhsCols, rhsRows
	}
	if rows <= 0 || inner <= 0 || rhsInner <= 0 || cols <= 0 || inner != rhsInner {
		return 0, 0, 0, false
	}
	return rows, inner, cols, true
}

func trainerMatMulAt(data []float32, rows, cols, row, col int, transpose bool) float32 {
	if transpose {
		return data[col*cols+row]
	}
	return data[row*cols+col]
}

func requireTrainableEmbeddingParam(mod *barr.Module, name string) (barr.Param, error) {
	for _, param := range mod.Params {
		if param.Name != name {
			continue
		}
		if param.Type.Kind != barr.ValueTensor || param.Type.Tensor == nil {
			return barr.Param{}, fmt.Errorf("param %q is not a tensor", name)
		}
		if len(param.Type.Tensor.Shape) != 2 {
			return barr.Param{}, fmt.Errorf("param %q rank = %d, want 2", name, len(param.Type.Tensor.Shape))
		}
		if !param.Trainable {
			return barr.Param{}, fmt.Errorf("param %q is not marked @trainable", name)
		}
		return param, nil
	}
	return barr.Param{}, fmt.Errorf("missing param %q", name)
}

func optionalTrainableEmbeddingParam(mod *barr.Module, name string) (barr.Param, bool, error) {
	if name == "" {
		return barr.Param{}, false, nil
	}
	param, err := requireTrainableEmbeddingParam(mod, name)
	if err != nil {
		return barr.Param{}, false, err
	}
	return param, true, nil
}

func normalizedTrainConfig(cfg EmbeddingTrainConfig, params ...barr.Param) EmbeddingTrainConfig {
	if cfg.LearningRate == 0 {
		cfg.LearningRate = 0.05
	}
	if cfg.WeightBits == 0 {
		for _, param := range params {
			cfg.WeightBits = maxInt(cfg.WeightBits, paramQuantBits(param))
		}
	}
	if cfg.Optimizer == "" {
		cfg.Optimizer = "adamw"
	}
	if cfg.Beta1 == 0 {
		cfg.Beta1 = 0.9
	}
	if cfg.Beta2 == 0 {
		cfg.Beta2 = 0.999
	}
	if cfg.Epsilon == 0 {
		cfg.Epsilon = 1e-8
	}
	if cfg.ContrastiveLoss == "" {
		cfg.ContrastiveLoss = "pair_mse"
	}
	if cfg.Temperature == 0 {
		cfg.Temperature = 0.05
	}
	return cfg
}

func validateTrainConfig(cfg EmbeddingTrainConfig) error {
	switch cfg.ContrastiveLoss {
	case "pair_mse", "infonce":
	default:
		return fmt.Errorf("unsupported contrastive_loss %q", cfg.ContrastiveLoss)
	}
	if cfg.Temperature <= 0 {
		return fmt.Errorf("temperature must be positive")
	}
	return nil
}

func paramQuantBits(param barr.Param) int {
	if param.Type.Tensor == nil {
		return 0
	}
	switch param.Type.Tensor.DType {
	case "q4":
		return 4
	case "q8":
		return 8
	default:
		return 0
	}
}

func tensorAsMasterF32(t *backend.Tensor) *backend.Tensor {
	if t == nil {
		return nil
	}
	return backend.NewTensorF32(t.Shape, append([]float32(nil), t.F32...))
}

func zeroLikeMaster(t *backend.Tensor) *backend.Tensor {
	if t == nil {
		return nil
	}
	return backend.NewTensorF32(t.Shape, make([]float32, len(t.F32)))
}

func forwardTensorForParam(param barr.Param, master *backend.Tensor, bits int) *backend.Tensor {
	return refreshForwardTensorForParam(param, master, bits, nil)
}

func refreshForwardMatMulTensorForParam(param barr.Param, master *backend.Tensor, dst *backend.Tensor) *backend.Tensor {
	if master == nil {
		return nil
	}
	dtype := "f32"
	if param.Type.Tensor != nil && param.Type.Tensor.DType != "" {
		dtype = param.Type.Tensor.DType
	}
	if dst == nil {
		dst = &backend.Tensor{}
	}
	elements := len(master.F32)
	if len(dst.F32) != elements {
		dst.F32 = make([]float32, elements)
	}
	copy(dst.F32, master.F32)
	dst.DType = dtype
	dst.Shape = append(dst.Shape[:0], master.Shape...)
	dst.I32 = nil
	dst.I64 = nil
	return dst
}

func refreshForwardTensorForParam(param barr.Param, master *backend.Tensor, bits int, dst *backend.Tensor) *backend.Tensor {
	if master == nil {
		return nil
	}
	bits = clampQuantBits(bits, paramQuantBits(param))
	dtype := param.Type.Tensor.DType
	if dtype == "" {
		dtype = "f32"
	}
	if dst == nil {
		dst = &backend.Tensor{}
	}
	elements := len(master.F32)
	if len(dst.F32) != elements {
		dst.F32 = make([]float32, elements)
	}
	copy(dst.F32, master.F32)
	if bits > 0 {
		fakeQuantizeDataInPlace(dst.F32, bits)
	}
	dst.DType = dtype
	dst.Shape = append(dst.Shape[:0], master.Shape...)
	dst.I32 = nil
	dst.I64 = nil
	return dst
}

func forwardMatMulHostData(rhs *backend.Tensor) []float32 {
	if rhs == nil {
		return nil
	}
	data := append([]float32(nil), rhs.F32...)
	switch rhs.DType {
	case "q4":
		return fakeQuantizeData(data, 4)
	case "q8":
		return fakeQuantizeData(data, 8)
	default:
		return data
	}
}

func exportTensorForParam(param barr.Param, master *backend.Tensor) (*backend.Tensor, error) {
	if master == nil {
		return nil, fmt.Errorf("missing master tensor for %q", param.Name)
	}
	data := append([]float32(nil), master.F32...)
	switch param.Type.Tensor.DType {
	case "q4":
		return backend.NewTensorQ4(master.Shape, fakeQuantizeData(data, 4)), nil
	case "q8":
		return backend.NewTensorQ8(master.Shape, fakeQuantizeData(data, 8)), nil
	case "f16":
		return backend.NewTensorF16(master.Shape, data), nil
	case "f32":
		return backend.NewTensorF32(master.Shape, data), nil
	default:
		return nil, fmt.Errorf("unsupported export dtype %q for %q", param.Type.Tensor.DType, param.Name)
	}
}

func clampQuantBits(primary, fallback int) int {
	if primary > 0 {
		return primary
	}
	return fallback
}

func fakeQuantizeData(data []float32, bits int) []float32 {
	if bits <= 0 || len(data) == 0 {
		return data
	}
	fakeQuantizeDataInPlace(data, bits)
	return data
}

func fakeQuantizeDataInPlace(data []float32, bits int) {
	if bits <= 0 || len(data) == 0 {
		return
	}
	maxAbs := float32(0)
	for _, v := range data {
		abs := float32(math.Abs(float64(v)))
		if abs > maxAbs {
			maxAbs = abs
		}
	}
	if maxAbs == 0 {
		return
	}
	levelsInt := (1 << uint(bits-1)) - 1
	levels := float32(levelsInt)
	if levels <= 0 {
		return
	}
	scale := maxAbs / levels
	if scale == 0 {
		return
	}
	for i, v := range data {
		q := float32(math.Round(float64(v / scale)))
		if q > levels {
			q = levels
		}
		if q < -levels {
			q = -levels
		}
		data[i] = q * scale
	}
}

func pooledGradToProjected(state *embeddingSequenceState, gradPooled []float32, e int) []float32 {
	gradProjected := make([]float32, len(state.projected))
	if state == nil || state.activeCount == 0 {
		return gradProjected
	}
	invCount := 1 / float32(state.activeCount)
	for row := range state.tokens {
		if state.mask[row] == 0 {
			continue
		}
		projectedBase := row * e
		norm := vectorNorm(state.projected[projectedBase : projectedBase+e])
		if norm == 0 {
			continue
		}
		dotNG := float32(0)
		for col := 0; col < e; col++ {
			dotNG += state.normalized[projectedBase+col] * gradPooled[col] * invCount
		}
		for col := 0; col < e; col++ {
			gradProjected[projectedBase+col] = (gradPooled[col]*invCount - state.normalized[projectedBase+col]*dotNG) / norm
		}
	}
	return gradProjected
}

func (t *EmbeddingTrainer) backpropProjectedSequence(state *embeddingSequenceState, gradProjected []float32, projection *backend.Tensor, gradProj []float32, d, e int) []float32 {
	gradInput := make([]float32, len(state.hidden))
	if state == nil {
		return gradInput
	}
	seqLen := len(state.tokens)
	gradProjStep := make([]float32, d*e)
	if out, ok := t.tryTrainerMatMulBoundLeft(
		state.hiddenBinding,
		tensorF32View([]int{seqLen, d}, state.hidden),
		tensorF32View([]int{seqLen, e}, gradProjected),
		true,
		false,
	); ok {
		copy(gradProjStep, out)
	} else {
		fillHostMatMulTranspose(state.hidden, seqLen, d, gradProjected, seqLen, e, true, false, gradProjStep)
	}
	addFloat32Slice(gradProj, gradProjStep)
	if out, ok := t.tryTrainerMatMulBoundRight(gradProjected, seqLen, e, t.projParam.Name, projection, false, true); ok {
		copy(gradInput, out)
	} else {
		projData := forwardMatMulHostData(projection)
		fillHostMatMulTranspose(gradProjected, seqLen, e, projData, d, e, false, true, gradInput)
	}
	return gradInput
}

func (t *EmbeddingTrainer) backpropProjectedFFNSequence(state *embeddingSequenceState, gradProjected []float32, hiddenProjection, projection *backend.Tensor, gradHidden, gradProj []float32, d, h, e int) []float32 {
	gradInput := make([]float32, len(state.hidden))
	if state == nil {
		return gradInput
	}
	seqLen := len(state.tokens)
	gradOutputMatrix := make([]float32, seqLen*e)
	for row := range state.tokens {
		if state.mask[row] == 0 {
			continue
		}
		projectedBase := row * e
		gradOutput := gradOutputMatrix[projectedBase : projectedBase+e]
		copy(gradOutput, gradProjected[projectedBase:projectedBase+e])
	}
	if t.ffnLayerNormEnabled() {
		accelerated := false
		if !state.skipUnboundActivationBackward(state.projectedBinding, state.ffnResidualBinding) {
			if out, ok := t.tryLayerNormBackwardRows(gradOutputMatrix, state.projected, state.ffnResidual, seqLen, e, state.projectedBinding, state.ffnResidualBinding); ok {
				copy(gradOutputMatrix, out)
				accelerated = true
			}
		}
		if !accelerated {
			for row := range state.tokens {
				if state.mask[row] == 0 {
					continue
				}
				projectedBase := row * e
				backwardLayerNormRow(
					gradOutputMatrix[projectedBase:projectedBase+e],
					gradOutputMatrix[projectedBase:projectedBase+e],
					state.projected[projectedBase:projectedBase+e],
					state.ffnResidual[projectedBase:projectedBase+e],
				)
			}
		}
	}
	gradProjStep := make([]float32, h*e)
	if out, ok := t.tryTrainerMatMulBoundLeft(
		state.activatedBinding,
		tensorF32View([]int{seqLen, h}, state.activated),
		tensorF32View([]int{seqLen, e}, gradOutputMatrix),
		true,
		false,
	); ok {
		copy(gradProjStep, out)
	} else {
		fillHostMatMulTranspose(state.activated, seqLen, h, gradOutputMatrix, seqLen, e, true, false, gradProjStep)
	}
	addFloat32Slice(gradProj, gradProjStep)
	gradActivatedPre := make([]float32, seqLen*h)
	if out, ok := t.tryTrainerMatMulBoundRight(gradOutputMatrix, seqLen, e, t.projParam.Name, projection, false, true); ok {
		copy(gradActivatedPre, out)
	} else {
		projData := forwardMatMulHostData(projection)
		fillHostMatMulTranspose(gradOutputMatrix, seqLen, e, projData, h, e, false, true, gradActivatedPre)
	}
	gradActivated := make([]float32, seqLen*h)
	acceleratedGELU := false
	if !state.skipUnboundActivationBackward(state.ffnHiddenBinding) {
		if out, ok := t.tryGELUBackwardMul(gradActivatedPre, state.ffnHidden, seqLen, h, state.ffnHiddenBinding); ok {
			copy(gradActivated, out)
			acceleratedGELU = true
		}
	}
	if !acceleratedGELU {
		for row := range state.tokens {
			if state.mask[row] == 0 {
				continue
			}
			ffnBase := row * h
			for col := 0; col < h; col++ {
				gradActivated[ffnBase+col] = gradActivatedPre[ffnBase+col] * geluBackward(state.ffnHidden[ffnBase+col])
			}
		}
	}
	gradHiddenStep := make([]float32, d*h)
	if out, ok := t.tryTrainerMatMulBoundLeft(
		state.hiddenBinding,
		tensorF32View([]int{seqLen, d}, state.hidden),
		tensorF32View([]int{seqLen, h}, gradActivated),
		true,
		false,
	); ok {
		copy(gradHiddenStep, out)
	} else {
		fillHostMatMulTranspose(state.hidden, seqLen, d, gradActivated, seqLen, h, true, false, gradHiddenStep)
	}
	addFloat32Slice(gradHidden, gradHiddenStep)
	if out, ok := t.tryTrainerMatMulBoundRight(gradActivated, seqLen, h, t.hiddenParam.Name, hiddenProjection, false, true); ok {
		copy(gradInput, out)
	} else {
		hiddenData := forwardMatMulHostData(hiddenProjection)
		fillHostMatMulTranspose(gradActivated, seqLen, h, hiddenData, d, h, false, true, gradInput)
	}
	if t.ffnResidualEnabled() {
		addFloat32Slice(gradInput, gradOutputMatrix)
	}
	return gradInput
}

func (t *EmbeddingTrainer) backpropEncodedSequence(seq *embeddingEncodedSequence, gradPooled []float32, attentionQuery, attentionKey, attentionValue, attentionOutput, hiddenProjection, projection *backend.Tensor, gradToken, gradAttnQ, gradAttnK, gradAttnV, gradAttnO, gradHidden, gradProj []float32) []float32 {
	if seq == nil || len(seq.layers) == 0 {
		return nil
	}
	gradProjected := pooledGradToProjected(seq.layers[len(seq.layers)-1], gradPooled, t.projection.Shape[1])
	for layer := len(seq.layers) - 1; layer >= 0; layer-- {
		gradProjected = t.backpropEncodedLayer(seq.layers[layer], gradProjected, attentionQuery, attentionKey, attentionValue, attentionOutput, hiddenProjection, projection, gradAttnQ, gradAttnK, gradAttnV, gradAttnO, gradHidden, gradProj)
	}
	return gradProjected
}

type embeddingBackpropItem struct {
	seq           *embeddingEncodedSequence
	gradPooled    []float32
	gradProjected []float32
}

func (t *EmbeddingTrainer) tryBackpropContrastiveBatch(queries, positives []*embeddingEncodedSequence, queryGrads, positiveGrads [][]float32, attentionQuery, attentionKey, attentionValue, attentionOutput, hiddenProjection, projection *backend.Tensor, gradToken, gradAttnQ, gradAttnK, gradAttnV, gradAttnO, gradHidden, gradProj []float32) bool {
	if t == nil || !batchedBackwardEnabled() || hiddenProjection == nil || projection == nil {
		return false
	}
	items := make([]embeddingBackpropItem, 0, len(queries)+len(positives))
	for i, seq := range queries {
		if i >= len(queryGrads) || seq == nil || len(seq.layers) == 0 {
			return false
		}
		items = append(items, embeddingBackpropItem{seq: seq, gradPooled: queryGrads[i]})
	}
	for i, seq := range positives {
		if i >= len(positiveGrads) || seq == nil || len(seq.layers) == 0 {
			return false
		}
		items = append(items, embeddingBackpropItem{seq: seq, gradPooled: positiveGrads[i]})
	}
	if len(items) == 0 {
		return false
	}
	layerCount := len(items[0].seq.layers)
	for i := range items {
		if len(items[i].seq.layers) != layerCount {
			return false
		}
		items[i].gradProjected = pooledGradToProjected(items[i].seq.layers[layerCount-1], items[i].gradPooled, t.projection.Shape[1])
	}
	d := t.tokenEmbed.Shape[1]
	h := hiddenProjection.Shape[1]
	e := t.projection.Shape[1]
	for layer := layerCount - 1; layer >= 0; layer-- {
		groups := map[int][]int{}
		lengths := make([]int, 0, len(items))
		for i := range items {
			state := items[i].seq.layers[layer]
			if state == nil || len(state.tokens) == 0 {
				return false
			}
			seqLen := len(state.tokens)
			if _, ok := groups[seqLen]; !ok {
				lengths = append(lengths, seqLen)
			}
			groups[seqLen] = append(groups[seqLen], i)
		}

		for _, seqLen := range lengths {
			indexes := groups[seqLen]
			if len(indexes) == 1 {
				idx := indexes[0]
				items[idx].gradProjected = t.backpropEncodedLayer(items[idx].seq.layers[layer], items[idx].gradProjected, attentionQuery, attentionKey, attentionValue, attentionOutput, hiddenProjection, projection, gradAttnQ, gradAttnK, gradAttnV, gradAttnO, gradHidden, gradProj)
				continue
			}

			states := make([]*embeddingSequenceState, len(indexes))
			gradProjected := make([][]float32, len(indexes))
			for groupIndex, itemIndex := range indexes {
				states[groupIndex] = items[itemIndex].seq.layers[layer]
				gradProjected[groupIndex] = items[itemIndex].gradProjected
			}

			gradInput, ok := t.backpropProjectedFFNSequences(states, gradProjected, hiddenProjection, projection, gradHidden, gradProj, d, h, e)
			if !ok {
				for _, idx := range indexes {
					items[idx].gradProjected = t.backpropEncodedLayer(items[idx].seq.layers[layer], items[idx].gradProjected, attentionQuery, attentionKey, attentionValue, attentionOutput, hiddenProjection, projection, gradAttnQ, gradAttnK, gradAttnV, gradAttnO, gradHidden, gradProj)
				}
				continue
			}
			if t.attentionEnabled() {
				var ok bool
				gradInput, ok = t.backpropAttentionSequences(states, gradInput, attentionQuery, attentionKey, attentionValue, attentionOutput, gradAttnQ, gradAttnK, gradAttnV, gradAttnO, d)
				if !ok {
					for _, idx := range indexes {
						items[idx].gradProjected = t.backpropEncodedLayer(items[idx].seq.layers[layer], items[idx].gradProjected, attentionQuery, attentionKey, attentionValue, attentionOutput, hiddenProjection, projection, gradAttnQ, gradAttnK, gradAttnV, gradAttnO, gradHidden, gradProj)
					}
					continue
				}
			}
			for groupIndex, itemIndex := range indexes {
				items[itemIndex].gradProjected = gradInput[groupIndex]
			}
		}
	}
	for i := range items {
		accumulateTokenGrad(items[i].seq.tokens, items[i].gradProjected, gradToken, d, t.tokenEmbed.Shape[0])
	}
	return true
}

func (t *EmbeddingTrainer) backpropProjectedFFNSequences(states []*embeddingSequenceState, gradProjected [][]float32, hiddenProjection, projection *backend.Tensor, gradHidden, gradProj []float32, d, h, e int) ([][]float32, bool) {
	if len(states) == 0 || len(states) != len(gradProjected) || hiddenProjection == nil || projection == nil {
		return nil, false
	}
	seqLen := len(states[0].tokens)
	if seqLen == 0 {
		return nil, false
	}
	for i, state := range states {
		if state == nil || len(state.tokens) != seqLen || len(state.hidden) != seqLen*d || len(state.activated) != seqLen*h || len(gradProjected[i]) != seqLen*e {
			return nil, false
		}
	}

	gradOutputMatrices := make([][]float32, len(states))
	gradActivatedPreMatrices := make([][]float32, len(states))
	activatedMatrices := make([][]float32, len(states))
	batchActivations := t.fullActivationBackwardAccelEnabled()
	var (
		projectedMatrices   [][]float32
		ffnResidualMatrices [][]float32
		ffnHiddenMatrices   [][]float32
	)
	if batchActivations {
		projectedMatrices = make([][]float32, len(states))
		ffnResidualMatrices = make([][]float32, len(states))
		ffnHiddenMatrices = make([][]float32, len(states))
	}
	for i, state := range states {
		gradOutputMatrix := make([]float32, seqLen*e)
		for row := range state.tokens {
			if state.mask[row] == 0 {
				continue
			}
			projectedBase := row * e
			copy(gradOutputMatrix[projectedBase:projectedBase+e], gradProjected[i][projectedBase:projectedBase+e])
		}
		gradOutputMatrices[i] = gradOutputMatrix
		activatedMatrices[i] = state.activated
		if batchActivations {
			projectedMatrices[i] = state.projected
			ffnResidualMatrices[i] = state.ffnResidual
			ffnHiddenMatrices[i] = state.ffnHidden
		}
	}
	if t.ffnLayerNormEnabled() {
		batchedLayerNorm := false
		if batchActivations {
			if out, ok := t.tryBatchedLayerNormBackwardRows(gradOutputMatrices, projectedMatrices, ffnResidualMatrices, seqLen, e); ok {
				gradOutputMatrices = out
				batchedLayerNorm = true
			}
		}
		if !batchedLayerNorm {
			for i, state := range states {
				gradOutputMatrix := gradOutputMatrices[i]
				for row := range state.tokens {
					if state.mask[row] == 0 {
						continue
					}
					projectedBase := row * e
					backwardLayerNormRow(
						gradOutputMatrix[projectedBase:projectedBase+e],
						gradOutputMatrix[projectedBase:projectedBase+e],
						state.projected[projectedBase:projectedBase+e],
						state.ffnResidual[projectedBase:projectedBase+e],
					)
				}
			}
		}
	}

	if out, ok := t.tryAccumulatedTransposeMatMul(activatedMatrices, gradOutputMatrices, seqLen, h, e); ok {
		addFloat32Slice(gradProj, out)
	} else {
		for i, state := range states {
			gradProjStep := make([]float32, h*e)
			if out, ok := t.tryTrainerMatMulBoundLeft(
				state.activatedBinding,
				tensorF32View([]int{seqLen, h}, state.activated),
				tensorF32View([]int{seqLen, e}, gradOutputMatrices[i]),
				true,
				false,
			); ok {
				copy(gradProjStep, out)
			} else {
				fillHostMatMulTranspose(state.activated, seqLen, h, gradOutputMatrices[i], seqLen, e, true, false, gradProjStep)
			}
			addFloat32Slice(gradProj, gradProjStep)
		}
	}

	if out, ok := t.tryBatchedBoundRightMatMul(gradOutputMatrices, seqLen, e, t.projParam.Name, projection, false, true); ok {
		gradActivatedPreMatrices = out
	} else {
		projData := forwardMatMulHostData(projection)
		for i := range states {
			gradActivatedPre := make([]float32, seqLen*h)
			fillHostMatMulTranspose(gradOutputMatrices[i], seqLen, e, projData, h, e, false, true, gradActivatedPre)
			gradActivatedPreMatrices[i] = gradActivatedPre
		}
	}

	gradActivatedMatrices := make([][]float32, len(states))
	hiddenMatrices := make([][]float32, len(states))
	batchedGELU := false
	if batchActivations {
		if out, ok := t.tryBatchedGELUBackwardMul(gradActivatedPreMatrices, ffnHiddenMatrices, seqLen, h); ok {
			gradActivatedMatrices = out
			batchedGELU = true
		}
	}
	if !batchedGELU {
		for i, state := range states {
			gradActivated := make([]float32, seqLen*h)
			for row := range state.tokens {
				if state.mask[row] == 0 {
					continue
				}
				ffnBase := row * h
				for col := 0; col < h; col++ {
					gradActivated[ffnBase+col] = gradActivatedPreMatrices[i][ffnBase+col] * geluBackward(state.ffnHidden[ffnBase+col])
				}
			}
			gradActivatedMatrices[i] = gradActivated
		}
	}
	for i, state := range states {
		hiddenMatrices[i] = state.hidden
	}

	if out, ok := t.tryAccumulatedTransposeMatMul(hiddenMatrices, gradActivatedMatrices, seqLen, d, h); ok {
		addFloat32Slice(gradHidden, out)
	} else {
		for i, state := range states {
			gradHiddenStep := make([]float32, d*h)
			if out, ok := t.tryTrainerMatMulBoundLeft(
				state.hiddenBinding,
				tensorF32View([]int{seqLen, d}, state.hidden),
				tensorF32View([]int{seqLen, h}, gradActivatedMatrices[i]),
				true,
				false,
			); ok {
				copy(gradHiddenStep, out)
			} else {
				fillHostMatMulTranspose(state.hidden, seqLen, d, gradActivatedMatrices[i], seqLen, h, true, false, gradHiddenStep)
			}
			addFloat32Slice(gradHidden, gradHiddenStep)
		}
	}

	gradInputs := make([][]float32, len(states))
	if out, ok := t.tryBatchedBoundRightMatMul(gradActivatedMatrices, seqLen, h, t.hiddenParam.Name, hiddenProjection, false, true); ok {
		gradInputs = out
	} else {
		hiddenData := forwardMatMulHostData(hiddenProjection)
		for i := range states {
			gradInput := make([]float32, seqLen*d)
			fillHostMatMulTranspose(gradActivatedMatrices[i], seqLen, h, hiddenData, d, h, false, true, gradInput)
			gradInputs[i] = gradInput
		}
	}
	if t.ffnResidualEnabled() {
		for i := range states {
			addFloat32Slice(gradInputs[i], gradOutputMatrices[i])
		}
	}
	return gradInputs, true
}

func (t *EmbeddingTrainer) tryBatchedBoundRightMatMul(lhsMatrices [][]float32, rows, cols int, rhsName string, rhs *backend.Tensor, transposeLeft, transposeRight bool) ([][]float32, bool) {
	if len(lhsMatrices) == 0 || rows == 0 || cols == 0 || rhs == nil {
		return nil, false
	}
	totalRows := len(lhsMatrices) * rows
	batched, ok := t.flattenFixedFloat32MatricesScratch(0, lhsMatrices, rows*cols)
	if !ok {
		return nil, false
	}
	out, ok := t.tryTrainerMatMulBoundRight(batched, totalRows, cols, rhsName, rhs, transposeLeft, transposeRight)
	if !ok || len(out)%len(lhsMatrices) != 0 {
		return nil, false
	}
	return splitFloat32Views(out, len(lhsMatrices))
}

func (t *EmbeddingTrainer) tryAccumulatedTransposeMatMul(lhsMatrices, rhsMatrices [][]float32, rows, lhsCols, rhsCols int) ([]float32, bool) {
	if len(lhsMatrices) == 0 || len(lhsMatrices) != len(rhsMatrices) || rows == 0 || lhsCols == 0 || rhsCols == 0 {
		return nil, false
	}
	totalRows := len(lhsMatrices) * rows
	lhsBatch, ok := t.flattenFixedFloat32MatricesScratch(0, lhsMatrices, rows*lhsCols)
	if !ok {
		return nil, false
	}
	rhsBatch, ok := t.flattenFixedFloat32MatricesScratch(1, rhsMatrices, rows*rhsCols)
	if !ok {
		return nil, false
	}
	return t.tryTrainerMatMul(lhsBatch, totalRows, lhsCols, rhsBatch, totalRows, rhsCols, true, false)
}

func (t *EmbeddingTrainer) trySharedLeftAccumulatedTransposeMatMuls(lhsMatrices [][]float32, rhsMatrixSets [][][]float32, rows, lhsCols, rhsCols int) ([][]float32, bool) {
	if t == nil || t.forwardMatMul == nil || !sharedLeftMatMulEnabled() || len(lhsMatrices) == 0 || len(rhsMatrixSets) < 2 || rows == 0 || lhsCols == 0 || rhsCols == 0 {
		return nil, false
	}
	if out, ok := t.tryConcatenatedSharedLeftAccumulatedTransposeMatMuls(lhsMatrices, rhsMatrixSets, rows, lhsCols, rhsCols); ok {
		return out, true
	}
	shared, ok := t.forwardMatMul.(backend.SharedLeftMatMulAccelerator)
	if !ok {
		return nil, false
	}
	totalRows := len(lhsMatrices) * rows
	lhsBatch, ok := t.flattenFixedFloat32MatricesScratch(0, lhsMatrices, rows*lhsCols)
	if !ok {
		return nil, false
	}
	rhsTensors := make([]*backend.Tensor, len(rhsMatrixSets))
	for i, rhsMatrices := range rhsMatrixSets {
		if len(rhsMatrices) != len(lhsMatrices) {
			return nil, false
		}
		rhsBatch, ok := flattenFixedFloat32Matrices(rhsMatrices, rows*rhsCols)
		if !ok {
			return nil, false
		}
		rhsTensors[i] = tensorF32View([]int{totalRows, rhsCols}, rhsBatch)
	}
	results, err := shared.RunMatMulsWithSharedLeft(
		tensorF32View([]int{totalRows, lhsCols}, lhsBatch),
		rhsTensors,
		trainerF32TensorValueType(),
		true,
		false,
	)
	if err != nil || len(results) != len(rhsMatrixSets) {
		return nil, false
	}
	out := make([][]float32, len(results))
	for i, result := range results {
		if len(result.Outputs) != 1 || result.Outputs[0] == nil {
			return nil, false
		}
		data := result.Outputs[0].F32
		if len(data) != lhsCols*rhsCols {
			return nil, false
		}
		out[i] = data
	}
	return out, true
}

func (t *EmbeddingTrainer) tryConcatenatedSharedLeftAccumulatedTransposeMatMuls(lhsMatrices [][]float32, rhsMatrixSets [][][]float32, rows, lhsCols, rhsCols int) ([][]float32, bool) {
	if t == nil || !concatenatedSharedLeftMatMulEnabled() || len(lhsMatrices) == 0 || len(rhsMatrixSets) < 2 || rows == 0 || lhsCols == 0 || rhsCols == 0 {
		return nil, false
	}
	totalRows := len(lhsMatrices) * rows
	lhsBatch, ok := t.flattenFixedFloat32MatricesScratch(0, lhsMatrices, rows*lhsCols)
	if !ok {
		return nil, false
	}
	combinedCols := len(rhsMatrixSets) * rhsCols
	rhsBatch := t.scratchFloat32(1, totalRows*combinedCols)
	for setIndex, rhsMatrices := range rhsMatrixSets {
		if len(rhsMatrices) != len(lhsMatrices) {
			return nil, false
		}
		for matrixIndex, rhsMatrix := range rhsMatrices {
			if len(rhsMatrix) != rows*rhsCols {
				return nil, false
			}
			for row := 0; row < rows; row++ {
				srcBase := row * rhsCols
				dstBase := (matrixIndex*rows+row)*combinedCols + setIndex*rhsCols
				copy(rhsBatch[dstBase:dstBase+rhsCols], rhsMatrix[srcBase:srcBase+rhsCols])
			}
		}
	}
	out, ok := t.tryTrainerMatMul(lhsBatch, totalRows, lhsCols, rhsBatch, totalRows, combinedCols, true, false)
	if !ok || len(out) != lhsCols*combinedCols {
		return nil, false
	}
	results := make([][]float32, len(rhsMatrixSets))
	for setIndex := range results {
		results[setIndex] = make([]float32, lhsCols*rhsCols)
	}
	for row := 0; row < lhsCols; row++ {
		srcBase := row * combinedCols
		for setIndex := range results {
			dstBase := row * rhsCols
			chunk := out[srcBase+setIndex*rhsCols : srcBase+(setIndex+1)*rhsCols]
			copy(results[setIndex][dstBase:dstBase+rhsCols], chunk)
		}
	}
	return results, true
}

func (t *EmbeddingTrainer) tryCombinedAttentionValueKeyGradMatMul(attnScoreMatrices, gradMixedMatrices, gradPreSoftmaxMatrices, attnQMatrices [][]float32, seqLen, d int) ([][]float32, [][]float32, bool) {
	if t == nil || !combinedAttentionVKGradMatMulEnabled() || len(attnScoreMatrices) == 0 || len(attnScoreMatrices) != len(gradMixedMatrices) || len(attnScoreMatrices) != len(gradPreSoftmaxMatrices) || len(attnScoreMatrices) != len(attnQMatrices) || seqLen == 0 || d == 0 {
		return nil, nil, false
	}
	batches := len(attnScoreMatrices)
	lhsMatrices := make([][]float32, 0, batches*2)
	rhsMatrices := make([][]float32, 0, batches*2)
	lhsMatrices = append(lhsMatrices, attnScoreMatrices...)
	lhsMatrices = append(lhsMatrices, gradPreSoftmaxMatrices...)
	rhsMatrices = append(rhsMatrices, gradMixedMatrices...)
	rhsMatrices = append(rhsMatrices, attnQMatrices...)
	out, ok := t.tryTrainerBatchedMatMulTranspose(lhsMatrices, seqLen, seqLen, rhsMatrices, seqLen, d, true, false)
	if !ok || len(out) != batches*2 {
		return nil, nil, false
	}
	return out[:batches], out[batches:], true
}

func (t *EmbeddingTrainer) tryAccumulatedAttentionInputGradMatMul(gradQMatrices, gradKMatrices, gradVMatrices [][]float32, seqLen, d int, attentionQuery, attentionKey, attentionValue *backend.Tensor) ([][]float32, bool) {
	if t == nil || t.forwardMatMul == nil || !accumulatedAttentionInputGradMatMulEnabled() || len(gradQMatrices) == 0 || len(gradQMatrices) != len(gradKMatrices) || len(gradQMatrices) != len(gradVMatrices) || seqLen == 0 || d == 0 {
		return nil, false
	}
	accumulated, ok := t.forwardMatMul.(backend.AccumulatedBoundRightMatMulAccelerator)
	if !ok {
		return nil, false
	}
	for _, tensor := range []*backend.Tensor{attentionQuery, attentionKey, attentionValue} {
		if tensor == nil || len(tensor.Shape) != 2 || tensor.Shape[0] != d || tensor.Shape[1] != d {
			return nil, false
		}
	}
	perMatrix := seqLen * d
	gradQBatch, ok := t.flattenFixedFloat32MatricesScratch(0, gradQMatrices, perMatrix)
	if !ok {
		return nil, false
	}
	gradKBatch, ok := t.flattenFixedFloat32MatricesScratch(1, gradKMatrices, perMatrix)
	if !ok {
		return nil, false
	}
	gradVBatch, ok := t.flattenFixedFloat32MatricesScratch(2, gradVMatrices, perMatrix)
	if !ok {
		return nil, false
	}
	totalRows := len(gradQMatrices) * seqLen
	result, err := accumulated.RunAccumulatedMatMulsWithBoundRights(
		[]*backend.Tensor{
			tensorF32View([]int{totalRows, d}, gradQBatch),
			tensorF32View([]int{totalRows, d}, gradKBatch),
			tensorF32View([]int{totalRows, d}, gradVBatch),
		},
		[]string{t.attnQParam.Name, t.attnKParam.Name, t.attnVParam.Name},
		trainerF32TensorValueType(),
		false,
		true,
	)
	if err != nil || len(result.Outputs) != 1 || result.Outputs[0] == nil {
		return nil, false
	}
	out := result.Outputs[0].F32
	if len(out) != len(gradQMatrices)*perMatrix {
		return nil, false
	}
	return splitFloat32Views(out, len(gradQMatrices))
}

func (t *EmbeddingTrainer) backpropAttentionSequences(states []*embeddingSequenceState, gradHiddenMatrices [][]float32, attentionQuery, attentionKey, attentionValue, attentionOutput *backend.Tensor, gradAttnQ, gradAttnK, gradAttnV, gradAttnO []float32, d int) ([][]float32, bool) {
	if len(states) == 0 || len(states) != len(gradHiddenMatrices) || attentionQuery == nil || attentionKey == nil || attentionValue == nil || attentionOutput == nil {
		return nil, false
	}
	seqLen := len(states[0].tokens)
	if seqLen == 0 {
		return nil, false
	}
	for i, state := range states {
		if state == nil || len(state.tokens) != seqLen || len(state.input) != seqLen*d || len(gradHiddenMatrices[i]) != seqLen*d {
			return nil, false
		}
	}

	gradAttnOutputs := make([][]float32, len(states))
	gradResidualInputs := make([][]float32, len(states))
	attnMixedMatrices := make([][]float32, len(states))
	batchActivations := t.fullActivationBackwardAccelEnabled()
	var (
		attnHiddenMatrices   [][]float32
		attnResidualMatrices [][]float32
	)
	if batchActivations {
		attnHiddenMatrices = make([][]float32, len(states))
		attnResidualMatrices = make([][]float32, len(states))
	}
	for i, state := range states {
		gradAttnOutput := gradHiddenMatrices[i]
		gradAttnOutputs[i] = gradAttnOutput
		attnMixedMatrices[i] = state.attnMixed
		if batchActivations {
			attnHiddenMatrices[i] = state.hidden
			attnResidualMatrices[i] = state.attnResidual
		}
	}
	if t.attentionLayerNormEnabled() {
		batchedLayerNorm := false
		if batchActivations {
			if out, ok := t.tryBatchedLayerNormBackwardRows(gradAttnOutputs, attnHiddenMatrices, attnResidualMatrices, seqLen, d); ok {
				gradAttnOutputs = out
				batchedLayerNorm = true
			}
		}
		if !batchedLayerNorm {
			for i, state := range states {
				gradAttnOutput := gradAttnOutputs[i]
				for row := 0; row < seqLen; row++ {
					base := row * d
					backwardLayerNormRow(
						gradAttnOutput[base:base+d],
						gradHiddenMatrices[i][base:base+d],
						state.hidden[base:base+d],
						state.attnResidual[base:base+d],
					)
				}
			}
		}
	}
	if t.attentionResidualEnabled() {
		for i := range gradResidualInputs {
			gradResidualInputs[i] = gradAttnOutputs[i]
		}
	}

	if out, ok := t.tryAccumulatedTransposeMatMul(attnMixedMatrices, gradAttnOutputs, seqLen, d, d); ok {
		addFloat32Slice(gradAttnO, out)
	} else {
		for i, state := range states {
			gradAttnOStep := make([]float32, d*d)
			if out, ok := t.tryTrainerMatMulBoundLeft(
				state.attnMixedBinding,
				tensorF32View([]int{seqLen, d}, state.attnMixed),
				tensorF32View([]int{seqLen, d}, gradAttnOutputs[i]),
				true,
				false,
			); ok {
				copy(gradAttnOStep, out)
			} else {
				fillHostMatMulTranspose(state.attnMixed, seqLen, d, gradAttnOutputs[i], seqLen, d, true, false, gradAttnOStep)
			}
			addFloat32Slice(gradAttnO, gradAttnOStep)
		}
	}

	gradMixedMatrices, ok := t.tryBatchedBoundRightMatMul(gradAttnOutputs, seqLen, d, t.attnOParam.Name, attentionOutput, false, true)
	if !ok {
		attnOData := forwardMatMulHostData(attentionOutput)
		gradMixedMatrices = make([][]float32, len(states))
		for i := range states {
			gradMixed := make([]float32, seqLen*d)
			fillHostMatMulTranspose(gradAttnOutputs[i], seqLen, d, attnOData, d, d, false, true, gradMixed)
			gradMixedMatrices[i] = gradMixed
		}
	}

	gradQMatrices := make([][]float32, len(states))
	gradKMatrices := make([][]float32, len(states))
	gradVMatrices := make([][]float32, len(states))
	inputMatrices := make([][]float32, len(states))
	attnScoreMatrices := make([][]float32, len(states))
	attnQMatrices := make([][]float32, len(states))
	attnKMatrices := make([][]float32, len(states))
	attnVMatrices := make([][]float32, len(states))
	for i, state := range states {
		attnScoreMatrices[i] = state.attnScores
		attnQMatrices[i] = state.attnQ
		attnKMatrices[i] = state.attnK
		attnVMatrices[i] = state.attnV
		inputMatrices[i] = state.input
	}

	gradScoresMatrices, gradScoresOK := t.tryTrainerBatchedMatMulTranspose(gradMixedMatrices, seqLen, d, attnVMatrices, seqLen, d, false, true)
	if !gradScoresOK {
		gradScoresMatrices = make([][]float32, len(states))
		for i, state := range states {
			gradScoresFlat := make([]float32, seqLen*seqLen)
			if out, ok := t.tryTrainerMatMul(gradMixedMatrices[i], seqLen, d, state.attnV, seqLen, d, false, true); ok {
				copy(gradScoresFlat, out)
			} else {
				fillHostMatMulTranspose(gradMixedMatrices[i], seqLen, d, state.attnV, seqLen, d, false, true, gradScoresFlat)
			}
			gradScoresMatrices[i] = gradScoresFlat
		}
	}

	gradPreSoftmaxMatrices, gradPreSoftmaxOK := t.tryBatchedSoftmaxBackwardRows(gradScoresMatrices, attnScoreMatrices, seqLen, seqLen)
	if !gradPreSoftmaxOK {
		gradPreSoftmaxMatrices = make([][]float32, len(states))
		for i, state := range states {
			gradPreSoftmaxFlat := make([]float32, seqLen*seqLen)
			if !state.skipUnboundActivationBackward(state.attnScoresBinding) {
				if out, ok := t.trySoftmaxBackwardRows(gradScoresMatrices[i], state.attnScores, seqLen, seqLen, state.attnScoresBinding); ok {
					copy(gradPreSoftmaxFlat, out)
					gradPreSoftmaxMatrices[i] = gradPreSoftmaxFlat
					continue
				}
			}
			for row := 0; row < seqLen; row++ {
				rowScores := state.attnScores[row*seqLen : (row+1)*seqLen]
				gradScores := gradScoresMatrices[i][row*seqLen : (row+1)*seqLen]
				backwardSoftmaxRow(gradPreSoftmaxFlat[row*seqLen:(row+1)*seqLen], gradScores, rowScores)
			}
			gradPreSoftmaxMatrices[i] = gradPreSoftmaxFlat
		}
	}

	batchedGradV, batchedGradK, combinedVKOK := t.tryCombinedAttentionValueKeyGradMatMul(attnScoreMatrices, gradMixedMatrices, gradPreSoftmaxMatrices, attnQMatrices, seqLen, d)
	gradVOK := combinedVKOK
	gradKOK := combinedVKOK
	if !combinedVKOK {
		batchedGradV, gradVOK = t.tryTrainerBatchedMatMulTranspose(attnScoreMatrices, seqLen, seqLen, gradMixedMatrices, seqLen, d, true, false)
		batchedGradK, gradKOK = t.tryTrainerBatchedMatMulTranspose(gradPreSoftmaxMatrices, seqLen, seqLen, attnQMatrices, seqLen, d, true, false)
	}
	batchedGradQ, gradQOK := t.tryTrainerBatchedMatMulTranspose(gradPreSoftmaxMatrices, seqLen, seqLen, attnKMatrices, seqLen, d, false, false)
	for i, state := range states {
		gradMixed := gradMixedMatrices[i]
		gradPreSoftmaxFlat := gradPreSoftmaxMatrices[i]
		var gradQ, gradK, gradV []float32
		if gradVOK {
			gradV = batchedGradV[i]
		} else if out, ok := t.tryTrainerMatMulBoundLeft(
			state.attnScoresBinding,
			tensorF32View([]int{seqLen, seqLen}, state.attnScores),
			tensorF32View([]int{seqLen, d}, gradMixed),
			true,
			false,
		); ok {
			gradV = out
		} else {
			gradV = make([]float32, seqLen*d)
			fillHostMatMulTranspose(state.attnScores, seqLen, seqLen, gradMixed, seqLen, d, true, false, gradV)
		}
		if gradQOK {
			gradQ = batchedGradQ[i]
		} else if out, ok := t.tryTrainerMatMul(gradPreSoftmaxFlat, seqLen, seqLen, state.attnK, seqLen, d, false, false); ok {
			gradQ = out
		} else {
			gradQ = make([]float32, seqLen*d)
			fillHostMatMulTranspose(gradPreSoftmaxFlat, seqLen, seqLen, state.attnK, seqLen, d, false, false, gradQ)
		}
		if gradKOK {
			gradK = batchedGradK[i]
		} else if out, ok := t.tryTrainerMatMul(gradPreSoftmaxFlat, seqLen, seqLen, state.attnQ, seqLen, d, true, false); ok {
			gradK = out
		} else {
			gradK = make([]float32, seqLen*d)
			fillHostMatMulTranspose(gradPreSoftmaxFlat, seqLen, seqLen, state.attnQ, seqLen, d, true, false, gradK)
		}
		gradQMatrices[i] = gradQ
		gradKMatrices[i] = gradK
		gradVMatrices[i] = gradV
	}

	if sharedGradWeights, ok := t.trySharedLeftAccumulatedTransposeMatMuls(inputMatrices, [][][]float32{gradQMatrices, gradKMatrices, gradVMatrices}, seqLen, d, d); ok {
		addFloat32Slice(gradAttnQ, sharedGradWeights[0])
		addFloat32Slice(gradAttnK, sharedGradWeights[1])
		addFloat32Slice(gradAttnV, sharedGradWeights[2])
	} else {
		if out, ok := t.tryAccumulatedTransposeMatMul(inputMatrices, gradQMatrices, seqLen, d, d); ok {
			addFloat32Slice(gradAttnQ, out)
		} else {
			for i, state := range states {
				gradAttnQStep := make([]float32, d*d)
				if out, ok := t.tryTrainerMatMulBoundLeft(
					state.inputBinding,
					tensorF32View([]int{seqLen, d}, state.input),
					tensorF32View([]int{seqLen, d}, gradQMatrices[i]),
					true,
					false,
				); ok {
					copy(gradAttnQStep, out)
				} else {
					fillHostMatMulTranspose(state.input, seqLen, d, gradQMatrices[i], seqLen, d, true, false, gradAttnQStep)
				}
				addFloat32Slice(gradAttnQ, gradAttnQStep)
			}
		}
		if out, ok := t.tryAccumulatedTransposeMatMul(inputMatrices, gradKMatrices, seqLen, d, d); ok {
			addFloat32Slice(gradAttnK, out)
		} else {
			for i, state := range states {
				gradAttnKStep := make([]float32, d*d)
				if out, ok := t.tryTrainerMatMulBoundLeft(
					state.inputBinding,
					tensorF32View([]int{seqLen, d}, state.input),
					tensorF32View([]int{seqLen, d}, gradKMatrices[i]),
					true,
					false,
				); ok {
					copy(gradAttnKStep, out)
				} else {
					fillHostMatMulTranspose(state.input, seqLen, d, gradKMatrices[i], seqLen, d, true, false, gradAttnKStep)
				}
				addFloat32Slice(gradAttnK, gradAttnKStep)
			}
		}
		if out, ok := t.tryAccumulatedTransposeMatMul(inputMatrices, gradVMatrices, seqLen, d, d); ok {
			addFloat32Slice(gradAttnV, out)
		} else {
			for i, state := range states {
				gradAttnVStep := make([]float32, d*d)
				if out, ok := t.tryTrainerMatMulBoundLeft(
					state.inputBinding,
					tensorF32View([]int{seqLen, d}, state.input),
					tensorF32View([]int{seqLen, d}, gradVMatrices[i]),
					true,
					false,
				); ok {
					copy(gradAttnVStep, out)
				} else {
					fillHostMatMulTranspose(state.input, seqLen, d, gradVMatrices[i], seqLen, d, true, false, gradAttnVStep)
				}
				addFloat32Slice(gradAttnV, gradAttnVStep)
			}
		}
	}

	if accumulatedGradInputs, ok := t.tryAccumulatedAttentionInputGradMatMul(gradQMatrices, gradKMatrices, gradVMatrices, seqLen, d, attentionQuery, attentionKey, attentionValue); ok {
		gradInputs := make([][]float32, len(states))
		for i := range states {
			gradInput := accumulatedGradInputs[i]
			addFloat32Slice(gradInput, gradResidualInputs[i])
			gradInputs[i] = gradInput
		}
		return gradInputs, true
	}

	gradQInputs, okQ := t.tryBatchedBoundRightMatMul(gradQMatrices, seqLen, d, t.attnQParam.Name, attentionQuery, false, true)
	gradKInputs, okK := t.tryBatchedBoundRightMatMul(gradKMatrices, seqLen, d, t.attnKParam.Name, attentionKey, false, true)
	gradVInputs, okV := t.tryBatchedBoundRightMatMul(gradVMatrices, seqLen, d, t.attnVParam.Name, attentionValue, false, true)
	gradInputs := make([][]float32, len(states))
	if !okQ || !okK || !okV {
		attnQData := forwardMatMulHostData(attentionQuery)
		attnKData := forwardMatMulHostData(attentionKey)
		attnVData := forwardMatMulHostData(attentionValue)
		for i := range states {
			gradInput := make([]float32, seqLen*d)
			gradInputStep := make([]float32, seqLen*d)
			if okQ {
				addFloat32Slice(gradInput, gradQInputs[i])
			} else {
				fillHostMatMulTranspose(gradQMatrices[i], seqLen, d, attnQData, d, d, false, true, gradInputStep)
				addFloat32Slice(gradInput, gradInputStep)
			}
			if okK {
				addFloat32Slice(gradInput, gradKInputs[i])
			} else {
				for j := range gradInputStep {
					gradInputStep[j] = 0
				}
				fillHostMatMulTranspose(gradKMatrices[i], seqLen, d, attnKData, d, d, false, true, gradInputStep)
				addFloat32Slice(gradInput, gradInputStep)
			}
			if okV {
				addFloat32Slice(gradInput, gradVInputs[i])
			} else {
				for j := range gradInputStep {
					gradInputStep[j] = 0
				}
				fillHostMatMulTranspose(gradVMatrices[i], seqLen, d, attnVData, d, d, false, true, gradInputStep)
				addFloat32Slice(gradInput, gradInputStep)
			}
			addFloat32Slice(gradInput, gradResidualInputs[i])
			gradInputs[i] = gradInput
		}
		return gradInputs, true
	}
	for i := range states {
		gradInput := make([]float32, seqLen*d)
		addFloat32Slice(gradInput, gradQInputs[i])
		addFloat32Slice(gradInput, gradKInputs[i])
		addFloat32Slice(gradInput, gradVInputs[i])
		addFloat32Slice(gradInput, gradResidualInputs[i])
		gradInputs[i] = gradInput
	}
	return gradInputs, true
}

func (t *EmbeddingTrainer) backpropEncodedLayer(state *embeddingSequenceState, gradProjected []float32, attentionQuery, attentionKey, attentionValue, attentionOutput, hiddenProjection, projection *backend.Tensor, gradAttnQ, gradAttnK, gradAttnV, gradAttnO, gradHidden, gradProj []float32) []float32 {
	if hiddenProjection != nil {
		gradInput := t.backpropProjectedFFNSequence(state, gradProjected, hiddenProjection, projection, gradHidden, gradProj, t.tokenEmbed.Shape[1], hiddenProjection.Shape[1], t.projection.Shape[1])
		if t.attentionEnabled() {
			return t.backpropAttentionSequence(state, gradInput, attentionQuery, attentionKey, attentionValue, attentionOutput, gradAttnQ, gradAttnK, gradAttnV, gradAttnO, t.tokenEmbed.Shape[1])
		}
		return gradInput
	}
	gradInput := t.backpropProjectedSequence(state, gradProjected, projection, gradProj, t.tokenEmbed.Shape[1], t.projection.Shape[1])
	if t.attentionEnabled() {
		return t.backpropAttentionSequence(state, gradInput, attentionQuery, attentionKey, attentionValue, attentionOutput, gradAttnQ, gradAttnK, gradAttnV, gradAttnO, t.tokenEmbed.Shape[1])
	}
	return gradInput
}

func (t *EmbeddingTrainer) backpropAttentionSequence(state *embeddingSequenceState, gradHidden []float32, attentionQuery, attentionKey, attentionValue, attentionOutput *backend.Tensor, gradAttnQ, gradAttnK, gradAttnV, gradAttnO []float32, d int) []float32 {
	gradInput := make([]float32, len(state.input))
	if state == nil || len(state.tokens) == 0 || attentionQuery == nil || attentionKey == nil || attentionValue == nil || attentionOutput == nil {
		return gradInput
	}
	seqLen := len(state.tokens)
	gradAttnOutput := make([]float32, seqLen*d)
	gradResidualInput := make([]float32, seqLen*d)
	for row := 0; row < seqLen; row++ {
		base := row * d
		copy(gradAttnOutput[base:base+d], gradHidden[base:base+d])
		if t.attentionResidualEnabled() {
			copy(gradResidualInput[base:base+d], gradAttnOutput[base:base+d])
		}
	}
	if t.attentionLayerNormEnabled() {
		accelerated := false
		if !state.skipUnboundActivationBackward(state.hiddenBinding, state.attnResidualBinding) {
			if out, ok := t.tryLayerNormBackwardRows(gradAttnOutput, state.hidden, state.attnResidual, seqLen, d, state.hiddenBinding, state.attnResidualBinding); ok {
				copy(gradAttnOutput, out)
				if t.attentionResidualEnabled() {
					copy(gradResidualInput, out)
				}
				accelerated = true
			}
		}
		if !accelerated {
			for row := 0; row < seqLen; row++ {
				base := row * d
				backwardLayerNormRow(
					gradAttnOutput[base:base+d],
					gradHidden[base:base+d],
					state.hidden[base:base+d],
					state.attnResidual[base:base+d],
				)
				if t.attentionResidualEnabled() {
					copy(gradResidualInput[base:base+d], gradAttnOutput[base:base+d])
				}
			}
		}
	}
	gradMixed := make([]float32, seqLen*d)
	gradAttnOStep := make([]float32, d*d)
	if out, ok := t.tryTrainerMatMulBoundLeft(
		state.attnMixedBinding,
		tensorF32View([]int{seqLen, d}, state.attnMixed),
		tensorF32View([]int{seqLen, d}, gradAttnOutput),
		true,
		false,
	); ok {
		copy(gradAttnOStep, out)
	} else {
		fillHostMatMulTranspose(state.attnMixed, seqLen, d, gradAttnOutput, seqLen, d, true, false, gradAttnOStep)
	}
	addFloat32Slice(gradAttnO, gradAttnOStep)
	if out, ok := t.tryTrainerMatMulBoundRight(gradAttnOutput, seqLen, d, t.attnOParam.Name, attentionOutput, false, true); ok {
		copy(gradMixed, out)
	} else {
		attnOData := forwardMatMulHostData(attentionOutput)
		fillHostMatMulTranspose(gradAttnOutput, seqLen, d, attnOData, d, d, false, true, gradMixed)
	}

	gradQ := make([]float32, seqLen*d)
	gradK := make([]float32, seqLen*d)
	gradV := make([]float32, seqLen*d)
	gradScoresFlat := make([]float32, seqLen*seqLen)
	if out, ok := t.tryTrainerMatMul(gradMixed, seqLen, d, state.attnV, seqLen, d, false, true); ok {
		copy(gradScoresFlat, out)
	} else {
		fillHostMatMulTranspose(gradMixed, seqLen, d, state.attnV, seqLen, d, false, true, gradScoresFlat)
	}
	gradPreSoftmaxFlat := make([]float32, seqLen*seqLen)
	acceleratedSoftmax := false
	if !state.skipUnboundActivationBackward(state.attnScoresBinding) {
		if out, ok := t.trySoftmaxBackwardRows(gradScoresFlat, state.attnScores, seqLen, seqLen, state.attnScoresBinding); ok {
			copy(gradPreSoftmaxFlat, out)
			acceleratedSoftmax = true
		}
	}
	if !acceleratedSoftmax {
		for i := 0; i < seqLen; i++ {
			rowScores := state.attnScores[i*seqLen : (i+1)*seqLen]
			gradScores := gradScoresFlat[i*seqLen : (i+1)*seqLen]
			backwardSoftmaxRow(gradPreSoftmaxFlat[i*seqLen:(i+1)*seqLen], gradScores, rowScores)
		}
	}
	if out, ok := t.tryTrainerMatMulBoundLeft(
		state.attnScoresBinding,
		tensorF32View([]int{seqLen, seqLen}, state.attnScores),
		tensorF32View([]int{seqLen, d}, gradMixed),
		true,
		false,
	); ok {
		addFloat32Slice(gradV, out)
	} else {
		gradVStep := make([]float32, seqLen*d)
		fillHostMatMulTranspose(state.attnScores, seqLen, seqLen, gradMixed, seqLen, d, true, false, gradVStep)
		addFloat32Slice(gradV, gradVStep)
	}
	if out, ok := t.tryTrainerMatMul(gradPreSoftmaxFlat, seqLen, seqLen, state.attnK, seqLen, d, false, false); ok {
		copy(gradQ, out)
	} else {
		fillHostMatMulTranspose(gradPreSoftmaxFlat, seqLen, seqLen, state.attnK, seqLen, d, false, false, gradQ)
	}
	if out, ok := t.tryTrainerMatMul(gradPreSoftmaxFlat, seqLen, seqLen, state.attnQ, seqLen, d, true, false); ok {
		copy(gradK, out)
	} else {
		fillHostMatMulTranspose(gradPreSoftmaxFlat, seqLen, seqLen, state.attnQ, seqLen, d, true, false, gradK)
	}

	gradAttnQStep := make([]float32, d*d)
	if out, ok := t.tryTrainerMatMulBoundLeft(
		state.inputBinding,
		tensorF32View([]int{seqLen, d}, state.input),
		tensorF32View([]int{seqLen, d}, gradQ),
		true,
		false,
	); ok {
		copy(gradAttnQStep, out)
	} else {
		fillHostMatMulTranspose(state.input, seqLen, d, gradQ, seqLen, d, true, false, gradAttnQStep)
	}
	addFloat32Slice(gradAttnQ, gradAttnQStep)
	gradAttnKStep := make([]float32, d*d)
	if out, ok := t.tryTrainerMatMulBoundLeft(
		state.inputBinding,
		tensorF32View([]int{seqLen, d}, state.input),
		tensorF32View([]int{seqLen, d}, gradK),
		true,
		false,
	); ok {
		copy(gradAttnKStep, out)
	} else {
		fillHostMatMulTranspose(state.input, seqLen, d, gradK, seqLen, d, true, false, gradAttnKStep)
	}
	addFloat32Slice(gradAttnK, gradAttnKStep)
	gradAttnVStep := make([]float32, d*d)
	if out, ok := t.tryTrainerMatMulBoundLeft(
		state.inputBinding,
		tensorF32View([]int{seqLen, d}, state.input),
		tensorF32View([]int{seqLen, d}, gradV),
		true,
		false,
	); ok {
		copy(gradAttnVStep, out)
	} else {
		fillHostMatMulTranspose(state.input, seqLen, d, gradV, seqLen, d, true, false, gradAttnVStep)
	}
	addFloat32Slice(gradAttnV, gradAttnVStep)
	if out, ok := t.tryTrainerMatMulBoundRight(gradQ, seqLen, d, t.attnQParam.Name, attentionQuery, false, true); ok {
		addFloat32Slice(gradInput, out)
	} else {
		attnQData := forwardMatMulHostData(attentionQuery)
		gradInputStep := make([]float32, seqLen*d)
		fillHostMatMulTranspose(gradQ, seqLen, d, attnQData, d, d, false, true, gradInputStep)
		addFloat32Slice(gradInput, gradInputStep)
	}
	if out, ok := t.tryTrainerMatMulBoundRight(gradK, seqLen, d, t.attnKParam.Name, attentionKey, false, true); ok {
		addFloat32Slice(gradInput, out)
	} else {
		attnKData := forwardMatMulHostData(attentionKey)
		gradInputStep := make([]float32, seqLen*d)
		fillHostMatMulTranspose(gradK, seqLen, d, attnKData, d, d, false, true, gradInputStep)
		addFloat32Slice(gradInput, gradInputStep)
	}
	if out, ok := t.tryTrainerMatMulBoundRight(gradV, seqLen, d, t.attnVParam.Name, attentionValue, false, true); ok {
		addFloat32Slice(gradInput, out)
	} else {
		attnVData := forwardMatMulHostData(attentionValue)
		gradInputStep := make([]float32, seqLen*d)
		fillHostMatMulTranspose(gradV, seqLen, d, attnVData, d, d, false, true, gradInputStep)
		addFloat32Slice(gradInput, gradInputStep)
	}
	for i := range gradInput {
		gradInput[i] += gradResidualInput[i]
	}
	return gradInput
}

func accumulateTokenGrad(tokens []int32, gradInput, gradToken []float32, d, vocab int) {
	if d == 0 || vocab == 0 {
		return
	}
	for row, tok := range tokens {
		tokenBase := int(tok) * d
		if tokenBase < 0 || tokenBase+d > vocab*d {
			continue
		}
		rowBase := row * d
		for i := 0; i < d; i++ {
			gradToken[tokenBase+i] += gradInput[rowBase+i]
		}
	}
}

func backwardSoftmaxRow(dX, dOut, probs []float32) {
	dot := float32(0)
	for i := range probs {
		dot += dOut[i] * probs[i]
	}
	for i := range probs {
		dX[i] = probs[i] * (dOut[i] - dot)
	}
}

func transpose2DData(data []float32, rows, cols int) []float32 {
	out := make([]float32, rows*cols)
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			out[c*rows+r] = data[r*cols+c]
		}
	}
	return out
}

func softmaxRowsInPlace(data []float32, rows, cols int) {
	for row := 0; row < rows; row++ {
		base := row * cols
		maxVal := data[base]
		for col := 1; col < cols; col++ {
			if data[base+col] > maxVal {
				maxVal = data[base+col]
			}
		}
		sum := float32(0)
		for col := 0; col < cols; col++ {
			value := float32(math.Exp(float64(data[base+col] - maxVal)))
			data[base+col] = value
			sum += value
		}
		if sum == 0 {
			continue
		}
		inv := 1 / sum
		for col := 0; col < cols; col++ {
			data[base+col] *= inv
		}
	}
}

func layerNormRow(dst, src []float32) {
	if len(dst) != len(src) || len(src) == 0 {
		return
	}
	mean := float32(0)
	for _, value := range src {
		mean += value
	}
	mean /= float32(len(src))
	variance := float32(0)
	for _, value := range src {
		centered := value - mean
		variance += centered * centered
	}
	variance /= float32(len(src))
	invStd := float32(1.0 / math.Sqrt(float64(variance)+1e-5))
	for i, value := range src {
		dst[i] = (value - mean) * invStd
	}
}

func backwardLayerNormRow(dst, gradOut, normalized, pre []float32) {
	if len(dst) != len(gradOut) || len(gradOut) != len(normalized) || len(normalized) != len(pre) || len(pre) == 0 {
		return
	}
	mean := float32(0)
	for _, value := range pre {
		mean += value
	}
	mean /= float32(len(pre))
	variance := float32(0)
	for _, value := range pre {
		centered := value - mean
		variance += centered * centered
	}
	variance /= float32(len(pre))
	invStd := float32(1.0 / math.Sqrt(float64(variance)+1e-5))
	sumGrad := float32(0)
	sumGradNorm := float32(0)
	for i := range gradOut {
		sumGrad += gradOut[i]
		sumGradNorm += gradOut[i] * normalized[i]
	}
	n := float32(len(pre))
	for i := range gradOut {
		dst[i] = (invStd / n) * (n*gradOut[i] - sumGrad - normalized[i]*sumGradNorm)
	}
}

func cosineGrad(left, right []float32) (float32, []float32, []float32) {
	gradLeft := make([]float32, len(left))
	gradRight := make([]float32, len(right))
	if len(left) != len(right) || len(left) == 0 {
		return 0, gradLeft, gradRight
	}
	dot := float32(0)
	leftNormSq := float32(0)
	rightNormSq := float32(0)
	for i := range left {
		dot += left[i] * right[i]
		leftNormSq += left[i] * left[i]
		rightNormSq += right[i] * right[i]
	}
	leftNorm := float32(math.Sqrt(float64(leftNormSq)))
	rightNorm := float32(math.Sqrt(float64(rightNormSq)))
	if leftNorm == 0 || rightNorm == 0 {
		return 0, gradLeft, gradRight
	}
	denom := leftNorm * rightNorm
	score := dot / denom
	leftScale := score / (leftNorm * leftNorm)
	rightScale := score / (rightNorm * rightNorm)
	for i := range left {
		gradLeft[i] = right[i]/denom - left[i]*leftScale
		gradRight[i] = left[i]/denom - right[i]*rightScale
	}
	return score, gradLeft, gradRight
}

func (t *EmbeddingTrainer) optimizerUpdateConfig(scale float32) backend.OptimizerUpdateConfig {
	return backend.OptimizerUpdateConfig{
		Optimizer:    t.config.Optimizer,
		Step:         t.step,
		LearningRate: t.config.LearningRate,
		WeightDecay:  t.config.WeightDecay,
		Beta1:        t.config.Beta1,
		Beta2:        t.config.Beta2,
		Epsilon:      t.config.Epsilon,
		Scale:        scale,
	}
}

func (t *EmbeddingTrainer) applyOptimizerUpdate(name string, tensor, mom1, mom2 *backend.Tensor, grad []float32, scale float32) {
	if tensor == nil {
		return
	}
	if t != nil && t.optimizerAccel != nil && len(grad) == len(tensor.F32) {
		err := t.optimizerAccel.ApplyUpdate(
			name,
			t.optimizerUpdateConfig(scale),
			tensor,
			mom1,
			mom2,
			tensorF32View(tensor.Shape, grad),
		)
		if err == nil {
			t.momentsDirty = true
			t.invalidateForwardWeights()
			return
		}
	}
	applyOptimizerUpdate(t.config, t.step, tensor, mom1, mom2, grad, scale)
	if t != nil {
		t.invalidateForwardWeights()
	}
}

func (t *EmbeddingTrainer) tryGELUBackwardMul(gradOut, preAct []float32, rows, cols int, preActBinding string) ([]float32, bool) {
	if !t.fullActivationBackwardAccelEnabled() || rows == 0 || cols == 0 {
		return nil, false
	}
	if preActBinding == "" && !activationAccelElementsAllowed(rows, cols) {
		return nil, false
	}
	gradTensor := tensorF32View([]int{rows, cols}, gradOut)
	var (
		result *backend.Tensor
		err    error
	)
	if preActBinding != "" {
		result, err = t.activationAccel.RunGELUBackwardMulWithBoundPreAct(gradTensor, preActBinding)
	} else {
		result, err = t.activationAccel.RunGELUBackwardMul(
			gradTensor,
			tensorF32View([]int{rows, cols}, preAct),
		)
	}
	if err != nil || result == nil {
		return nil, false
	}
	out := result.F32
	if len(out) != len(gradOut) {
		return nil, false
	}
	return out, true
}

func (t *EmbeddingTrainer) trySoftmaxBackwardRows(gradOut, probs []float32, rows, cols int, probsBinding string) ([]float32, bool) {
	if !t.softmaxBackwardAccelEnabled() || rows == 0 || cols == 0 {
		return nil, false
	}
	if probsBinding == "" && !activationAccelElementsAllowed(rows, cols) {
		return nil, false
	}
	gradTensor := tensorF32View([]int{rows, cols}, gradOut)
	var (
		result *backend.Tensor
		err    error
	)
	if probsBinding != "" {
		result, err = t.activationAccel.RunSoftmaxBackwardRowsWithBoundProbs(gradTensor, probsBinding)
	} else {
		result, err = t.activationAccel.RunSoftmaxBackwardRows(
			gradTensor,
			tensorF32View([]int{rows, cols}, probs),
		)
	}
	if err != nil || result == nil {
		return nil, false
	}
	out := result.F32
	if len(out) != len(gradOut) {
		return nil, false
	}
	return out, true
}

func (t *EmbeddingTrainer) tryBatchedSoftmaxBackwardRows(gradOutMatrices, probsMatrices [][]float32, rows, cols int) ([][]float32, bool) {
	if !t.softmaxBackwardAccelEnabled() || len(gradOutMatrices) == 0 || len(gradOutMatrices) != len(probsMatrices) || rows == 0 || cols == 0 {
		return nil, false
	}
	if !activationAccelElementsAllowed(len(gradOutMatrices)*rows, cols) {
		return nil, false
	}
	perMatrix := rows * cols
	gradOut, ok := t.flattenFixedFloat32MatricesScratch(0, gradOutMatrices, perMatrix)
	if !ok {
		return nil, false
	}
	probs, ok := t.flattenFixedFloat32MatricesScratch(1, probsMatrices, perMatrix)
	if !ok {
		return nil, false
	}
	out, ok := t.trySoftmaxBackwardRows(gradOut, probs, len(gradOutMatrices)*rows, cols, "")
	if !ok || len(out) != len(gradOut) {
		return nil, false
	}
	return splitFloat32Views(out, len(gradOutMatrices))
}

func (t *EmbeddingTrainer) tryBatchedGELUBackwardMul(gradOutMatrices, preActMatrices [][]float32, rows, cols int) ([][]float32, bool) {
	if !t.fullActivationBackwardAccelEnabled() || len(gradOutMatrices) == 0 || len(gradOutMatrices) != len(preActMatrices) || rows == 0 || cols == 0 {
		return nil, false
	}
	if !activationAccelElementsAllowed(len(gradOutMatrices)*rows, cols) {
		return nil, false
	}
	perMatrix := rows * cols
	gradOut, ok := t.flattenFixedFloat32MatricesScratch(0, gradOutMatrices, perMatrix)
	if !ok {
		return nil, false
	}
	preAct, ok := t.flattenFixedFloat32MatricesScratch(1, preActMatrices, perMatrix)
	if !ok {
		return nil, false
	}
	out, ok := t.tryGELUBackwardMul(gradOut, preAct, len(gradOutMatrices)*rows, cols, "")
	if !ok || len(out) != len(gradOut) {
		return nil, false
	}
	return splitFloat32Views(out, len(gradOutMatrices))
}

func (t *EmbeddingTrainer) tryBatchedLayerNormBackwardRows(gradOutMatrices, normalizedMatrices, preMatrices [][]float32, rows, cols int) ([][]float32, bool) {
	if !t.fullActivationBackwardAccelEnabled() || len(gradOutMatrices) == 0 || len(gradOutMatrices) != len(normalizedMatrices) || len(gradOutMatrices) != len(preMatrices) || rows == 0 || cols == 0 {
		return nil, false
	}
	if !activationAccelElementsAllowed(len(gradOutMatrices)*rows, cols) {
		return nil, false
	}
	perMatrix := rows * cols
	gradOut, ok := t.flattenFixedFloat32MatricesScratch(0, gradOutMatrices, perMatrix)
	if !ok {
		return nil, false
	}
	normalized, ok := t.flattenFixedFloat32MatricesScratch(1, normalizedMatrices, perMatrix)
	if !ok {
		return nil, false
	}
	pre, ok := t.flattenFixedFloat32MatricesScratch(2, preMatrices, perMatrix)
	if !ok {
		return nil, false
	}
	out, ok := t.tryLayerNormBackwardRows(gradOut, normalized, pre, len(gradOutMatrices)*rows, cols, "", "")
	if !ok || len(out) != len(gradOut) {
		return nil, false
	}
	return splitFloat32Views(out, len(gradOutMatrices))
}

func (t *EmbeddingTrainer) tryLayerNormBackwardRows(gradOut, normalized, pre []float32, rows, cols int, normalizedBinding, preBinding string) ([]float32, bool) {
	if !t.fullActivationBackwardAccelEnabled() || rows == 0 || cols == 0 {
		return nil, false
	}
	if (normalizedBinding == "" || preBinding == "") && !activationAccelElementsAllowed(rows, cols) {
		return nil, false
	}
	gradTensor := tensorF32View([]int{rows, cols}, gradOut)
	var (
		result *backend.Tensor
		err    error
	)
	if normalizedBinding != "" && preBinding != "" {
		result, err = t.activationAccel.RunLayerNormBackwardRowsWithBoundInputs(gradTensor, normalizedBinding, preBinding)
	} else {
		result, err = t.activationAccel.RunLayerNormBackwardRows(
			gradTensor,
			tensorF32View([]int{rows, cols}, normalized),
			tensorF32View([]int{rows, cols}, pre),
		)
	}
	if err != nil || result == nil {
		return nil, false
	}
	out := result.F32
	if len(out) != len(gradOut) {
		return nil, false
	}
	return out, true
}

func applyOptimizerUpdate(cfg EmbeddingTrainConfig, step int, tensor, mom1, mom2 *backend.Tensor, grad []float32, scale float32) {
	if tensor == nil {
		return
	}
	switch cfg.Optimizer {
	case "", "sgd":
		for i := range tensor.F32 {
			g := grad[i] * scale
			if cfg.WeightDecay != 0 {
				g += cfg.WeightDecay * tensor.F32[i]
			}
			tensor.F32[i] -= cfg.LearningRate * g
		}
	default:
		applyAdamWUpdate(cfg, step, tensor, mom1, mom2, grad, scale)
	}
}

func applyAdamWUpdate(cfg EmbeddingTrainConfig, step int, tensor, mom1, mom2 *backend.Tensor, grad []float32, scale float32) {
	if tensor == nil || mom1 == nil || mom2 == nil {
		return
	}
	beta1Pow := float32(math.Pow(float64(cfg.Beta1), float64(step)))
	beta2Pow := float32(math.Pow(float64(cfg.Beta2), float64(step)))
	corr1 := float32(1) - beta1Pow
	corr2 := float32(1) - beta2Pow
	for i := range tensor.F32 {
		g := grad[i] * scale
		if cfg.WeightDecay != 0 {
			g += cfg.WeightDecay * tensor.F32[i]
		}
		mom1.F32[i] = cfg.Beta1*mom1.F32[i] + (1-cfg.Beta1)*g
		mom2.F32[i] = cfg.Beta2*mom2.F32[i] + (1-cfg.Beta2)*g*g
		mHat := mom1.F32[i]
		vHat := mom2.F32[i]
		if corr1 != 0 {
			mHat /= corr1
		}
		if corr2 != 0 {
			vHat /= corr2
		}
		tensor.F32[i] -= cfg.LearningRate * (mHat / (float32(math.Sqrt(float64(vHat))) + cfg.Epsilon))
	}
}

func vectorNorm(v []float32) float32 {
	sum := float64(0)
	for _, x := range v {
		sum += float64(x * x)
	}
	return float32(math.Sqrt(sum))
}

func geluForward(x float32) float32 {
	cubic := x * x * x
	inner := float32(0.7978845608) * (x + float32(0.044715)*cubic)
	return 0.5 * x * (1 + float32(math.Tanh(float64(inner))))
}

func geluBackward(x float32) float32 {
	cubic := x * x * x
	inner := float32(0.7978845608) * (x + float32(0.044715)*cubic)
	tanhInner := float32(math.Tanh(float64(inner)))
	sech2 := 1 - tanhInner*tanhInner
	innerGrad := float32(0.7978845608) * (1 + float32(3*0.044715)*x*x)
	return 0.5*(1+tanhInner) + 0.5*x*sech2*innerGrad
}

func (t *EmbeddingTrainer) hiddenProjectionData() []float32 {
	if t == nil || t.hiddenProjection == nil {
		return nil
	}
	return t.hiddenProjection.F32
}

func (t *EmbeddingTrainer) attentionEnabled() bool {
	return t != nil && t.attentionQuery != nil && t.attentionKey != nil && t.attentionValue != nil && t.attentionOutput != nil
}

func (t *EmbeddingTrainer) attentionResidualEnabled() bool {
	return t != nil && t.attentionEnabled() && t.manifest.AttentionResidual
}

func (t *EmbeddingTrainer) attentionLayerNormEnabled() bool {
	return t != nil && t.attentionEnabled() && t.manifest.AttentionLayerNorm
}

func (t *EmbeddingTrainer) ffnResidualEnabled() bool {
	return t != nil && t.hiddenProjection != nil && t.manifest.FFNResidual
}

func (t *EmbeddingTrainer) ffnLayerNormEnabled() bool {
	return t != nil && t.hiddenProjection != nil && t.manifest.FFNLayerNorm
}

func (t *EmbeddingTrainer) encoderRepeats() int {
	if t == nil || t.manifest.EncoderRepeats <= 0 {
		return 1
	}
	return t.manifest.EncoderRepeats
}

func tensorDataLen(t *backend.Tensor) int {
	if t == nil {
		return 0
	}
	return len(t.F32)
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
