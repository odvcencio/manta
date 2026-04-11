package barruntime

import (
	"testing"
	"time"

	"github.com/odvcencio/barracuda/compiler"
	"github.com/odvcencio/barracuda/runtime/backend"
)

func TestEstimateContrastiveTrainWorkload(t *testing.T) {
	workload := EstimateContrastiveTrainWorkload(512, 128, EmbeddingTrainRunConfig{
		Epochs:         1,
		BatchSize:      64,
		EvalEveryEpoch: 4,
	})
	if workload.TrainMode != "contrastive" {
		t.Fatalf("train mode = %q, want contrastive", workload.TrainMode)
	}
	if workload.TrainBatchesPerEpoch != 8 {
		t.Fatalf("train batches/epoch = %d, want 8", workload.TrainBatchesPerEpoch)
	}
	if workload.TrainPairsPerEpoch != 32768 {
		t.Fatalf("train pairs/epoch = %d, want 32768", workload.TrainPairsPerEpoch)
	}
	if workload.EvalPairsPerPass != 16384 {
		t.Fatalf("eval pairs/pass = %d, want 16384", workload.EvalPairsPerPass)
	}
	if workload.PlannedEvalPasses != 1 {
		t.Fatalf("planned eval passes = %d, want 1", workload.PlannedEvalPasses)
	}
	if workload.PlannedTotalPairs != 49152 {
		t.Fatalf("planned total pairs = %d, want 49152", workload.PlannedTotalPairs)
	}
}

func TestEmbeddingTrainerFitImprovesEvalAndTracksBest(t *testing.T) {
	trainer := newTinyTrainableEmbeddingTrainer(t, 0.05)
	trainSet := tinyEmbeddingPairDataset()

	before, err := trainer.EvaluatePairs(trainSet)
	if err != nil {
		t.Fatalf("eval before: %v", err)
	}
	summary, err := trainer.Fit(trainSet, trainSet, EmbeddingTrainRunConfig{
		Epochs:      6,
		BatchSize:   2,
		Shuffle:     true,
		Seed:        7,
		RestoreBest: true,
	})
	if err != nil {
		t.Fatalf("fit: %v", err)
	}
	if summary.EpochsCompleted != 6 {
		t.Fatalf("epochs completed = %d, want 6", summary.EpochsCompleted)
	}
	if summary.StepsCompleted != 12 {
		t.Fatalf("steps completed = %d, want 12", summary.StepsCompleted)
	}
	if len(summary.History) != 6 {
		t.Fatalf("history len = %d, want 6", len(summary.History))
	}
	if summary.BestEval == nil {
		t.Fatal("expected best eval metrics")
	}
	if summary.LastEval == nil {
		t.Fatal("expected last eval metrics")
	}
	if summary.FinalEval == nil {
		t.Fatal("expected final eval metrics")
	}
	if summary.BestEpoch <= 0 || summary.BestStep <= 0 {
		t.Fatalf("best epoch/step = %d/%d, want both positive", summary.BestEpoch, summary.BestStep)
	}
	if !summary.RestoredBest {
		t.Fatal("expected best checkpoint restore")
	}
	if summary.FinalEval.ScoreMargin <= before.ScoreMargin {
		t.Fatalf("score margin did not improve: before=%f after=%f", before.ScoreMargin, summary.FinalEval.ScoreMargin)
	}
	if summary.FinalEval.PairAccuracy < before.PairAccuracy {
		t.Fatalf("pair accuracy regressed: before=%f after=%f", before.PairAccuracy, summary.FinalEval.PairAccuracy)
	}

	finalEval, err := trainer.EvaluatePairs(trainSet)
	if err != nil {
		t.Fatalf("eval final: %v", err)
	}
	assertClose(t, finalEval.ScoreMargin, summary.FinalEval.ScoreMargin, 0.000001)
	assertClose(t, finalEval.PairAccuracy, summary.FinalEval.PairAccuracy, 0.000001)
}

func TestEmbeddingTrainerFitStopsEarlyOnPlateau(t *testing.T) {
	trainer := newTinyTrainableEmbeddingTrainer(t, 0)
	trainSet := tinyEmbeddingPairDataset()

	summary, err := trainer.Fit(trainSet, trainSet, EmbeddingTrainRunConfig{
		Epochs:                8,
		BatchSize:             2,
		Shuffle:               false,
		Seed:                  1,
		EarlyStoppingPatience: 1,
		MinDelta:              2,
		RestoreBest:           true,
	})
	if err != nil {
		t.Fatalf("fit: %v", err)
	}
	if !summary.StoppedEarly {
		t.Fatal("expected early stopping")
	}
	if summary.EpochsCompleted != 2 {
		t.Fatalf("epochs completed = %d, want 2", summary.EpochsCompleted)
	}
	if summary.StepsCompleted != 4 {
		t.Fatalf("steps completed = %d, want 4", summary.StepsCompleted)
	}
	if summary.BestEval == nil || summary.FinalEval == nil {
		t.Fatal("expected best and final eval metrics")
	}
	assertClose(t, summary.FinalEval.ScoreMargin, summary.BestEval.ScoreMargin, 0.000001)
	assertClose(t, summary.FinalEval.Loss, summary.BestEval.Loss, 0.000001)
}

func TestEmbeddingTrainerFitRejectsInvalidRunConfig(t *testing.T) {
	trainer := newTinyTrainableEmbeddingTrainer(t, 0.05)
	trainSet := tinyEmbeddingPairDataset()

	if _, err := trainer.Fit(nil, nil, EmbeddingTrainRunConfig{}); err == nil {
		t.Fatal("expected empty training dataset error")
	}
	if _, err := trainer.Fit(trainSet, nil, EmbeddingTrainRunConfig{Epochs: 1, BatchSize: 1, EvalEveryEpoch: 1, SelectMetric: "unknown"}); err == nil {
		t.Fatal("expected select metric error")
	}
	if _, err := trainer.Fit(trainSet, nil, EmbeddingTrainRunConfig{Epochs: 1, BatchSize: -1}); err == nil {
		t.Fatal("expected batch size error")
	}
}

func TestEmbeddingTrainerFitContrastiveImprovesEval(t *testing.T) {
	trainer := newTinyTrainableEmbeddingTrainer(t, 0.05)
	trainSet := tinyEmbeddingContrastiveDataset()

	before, err := trainer.EvaluateContrastive(trainSet)
	if err != nil {
		t.Fatalf("eval before: %v", err)
	}
	summary, err := trainer.FitContrastive(trainSet, trainSet, EmbeddingTrainRunConfig{
		Epochs:      6,
		BatchSize:   2,
		Shuffle:     true,
		Seed:        7,
		RestoreBest: true,
	})
	if err != nil {
		t.Fatalf("fit contrastive: %v", err)
	}
	if summary.FinalEval == nil {
		t.Fatal("expected final eval metrics")
	}
	if summary.FinalEval.ScoreMargin <= before.ScoreMargin {
		t.Fatalf("score margin did not improve: before=%f after=%f", before.ScoreMargin, summary.FinalEval.ScoreMargin)
	}
	if summary.FinalEval.PairAccuracy < before.PairAccuracy {
		t.Fatalf("pair accuracy regressed: before=%f after=%f", before.PairAccuracy, summary.FinalEval.PairAccuracy)
	}
}

func TestEmbeddingTrainerFitContrastiveTracksWorkloadAndTiming(t *testing.T) {
	trainer := newTinyTrainableEmbeddingTrainer(t, 0.05)
	trainSet := tinyEmbeddingContrastiveDataset()
	cfg := EmbeddingTrainRunConfig{
		Epochs:      1,
		BatchSize:   2,
		Shuffle:     false,
		Seed:        1,
		RestoreBest: true,
	}

	summary, err := trainer.FitContrastive(trainSet, trainSet, cfg)
	if err != nil {
		t.Fatalf("fit contrastive: %v", err)
	}

	expected := EstimateContrastiveTrainWorkload(len(trainSet), len(trainSet), cfg)
	if summary.Workload.TrainPairsPerEpoch != expected.TrainPairsPerEpoch {
		t.Fatalf("train pairs/epoch = %d, want %d", summary.Workload.TrainPairsPerEpoch, expected.TrainPairsPerEpoch)
	}
	if summary.Workload.PlannedTotalPairs != expected.PlannedTotalPairs {
		t.Fatalf("planned total pairs = %d, want %d", summary.Workload.PlannedTotalPairs, expected.PlannedTotalPairs)
	}
	if summary.Workload.ActualTrainPairs != expected.PlannedTrainPairs {
		t.Fatalf("actual train pairs = %d, want %d", summary.Workload.ActualTrainPairs, expected.PlannedTrainPairs)
	}
	if summary.Workload.ActualTrainExamples != int64(len(trainSet)) {
		t.Fatalf("actual train examples = %d, want %d", summary.Workload.ActualTrainExamples, len(trainSet))
	}
	if summary.Workload.ActualEvalPairs != expected.PlannedEvalPairs {
		t.Fatalf("actual eval pairs = %d, want %d", summary.Workload.ActualEvalPairs, expected.PlannedEvalPairs)
	}
	expectedEvalExamples := int64(len(trainSet) * expected.PlannedEvalPasses)
	if summary.Workload.ActualEvalExamples != expectedEvalExamples {
		t.Fatalf("actual eval examples = %d, want %d", summary.Workload.ActualEvalExamples, expectedEvalExamples)
	}
	if summary.Workload.ActualTotalPairs != summary.Workload.ActualTrainPairs+summary.Workload.ActualEvalPairs {
		t.Fatalf("actual total pairs = %d, want %d", summary.Workload.ActualTotalPairs, summary.Workload.ActualTrainPairs+summary.Workload.ActualEvalPairs)
	}
	if summary.Workload.ActualTotalExamples != summary.Workload.ActualTrainExamples+summary.Workload.ActualEvalExamples {
		t.Fatalf("actual total examples = %d, want %d", summary.Workload.ActualTotalExamples, summary.Workload.ActualTrainExamples+summary.Workload.ActualEvalExamples)
	}
	if summary.Workload.ActualEvalPasses != expected.PlannedEvalPasses {
		t.Fatalf("actual eval passes = %d, want %d", summary.Workload.ActualEvalPasses, expected.PlannedEvalPasses)
	}
	if summary.StepsRun != expected.TrainBatchesPerEpoch {
		t.Fatalf("run steps = %d, want %d", summary.StepsRun, expected.TrainBatchesPerEpoch)
	}
	if summary.Elapsed <= 0 {
		t.Fatalf("elapsed = %s, want > 0", summary.Elapsed)
	}
	if summary.TrainDuration <= 0 {
		t.Fatalf("train duration = %s, want > 0", summary.TrainDuration)
	}
	if summary.EvalDuration <= 0 {
		t.Fatalf("eval duration = %s, want > 0", summary.EvalDuration)
	}
	if summary.Elapsed+time.Millisecond < summary.TrainDuration+summary.EvalDuration {
		t.Fatalf("elapsed = %s, train+eval = %s, want elapsed >= train+eval", summary.Elapsed, summary.TrainDuration+summary.EvalDuration)
	}
}

func TestEmbeddingTrainerFitContrastiveUsesFinalEvalAsBestWhenNoEpochEvalRuns(t *testing.T) {
	trainer := newTinyTrainableEmbeddingTrainer(t, 0.05)
	trainSet := tinyEmbeddingContrastiveDataset()
	cfg := EmbeddingTrainRunConfig{
		Epochs:         1,
		BatchSize:      2,
		Shuffle:        false,
		Seed:           1,
		EvalEveryEpoch: 4,
		RestoreBest:    true,
	}

	summary, err := trainer.FitContrastive(trainSet, trainSet, cfg)
	if err != nil {
		t.Fatalf("fit contrastive: %v", err)
	}
	if summary.FinalEval == nil {
		t.Fatal("expected final eval metrics")
	}
	if summary.BestEval == nil {
		t.Fatal("expected best eval metrics")
	}
	if summary.BestEpoch != 1 {
		t.Fatalf("best epoch = %d, want 1", summary.BestEpoch)
	}
	if summary.BestStep != summary.StepsCompleted {
		t.Fatalf("best step = %d, want %d", summary.BestStep, summary.StepsCompleted)
	}
	assertClose(t, summary.BestEval.ScoreMargin, summary.FinalEval.ScoreMargin, 0.000001)
	assertClose(t, summary.BestEval.PairAccuracy, summary.FinalEval.PairAccuracy, 0.000001)
}

func TestEmbeddingTrainerFitContrastiveFFNImprovesEval(t *testing.T) {
	trainer := newTinyTrainableFFNEmbeddingTrainer(t, 0.05)
	trainSet := tinyEmbeddingContrastiveDataset()

	before, err := trainer.EvaluateContrastive(trainSet)
	if err != nil {
		t.Fatalf("eval before: %v", err)
	}
	summary, err := trainer.FitContrastive(trainSet, trainSet, EmbeddingTrainRunConfig{
		Epochs:      6,
		BatchSize:   2,
		Shuffle:     true,
		Seed:        7,
		RestoreBest: true,
	})
	if err != nil {
		t.Fatalf("fit contrastive ffn: %v", err)
	}
	if summary.FinalEval == nil {
		t.Fatal("expected final eval metrics")
	}
	if summary.FinalEval.ScoreMargin <= before.ScoreMargin {
		t.Fatalf("ffn score margin did not improve: before=%f after=%f", before.ScoreMargin, summary.FinalEval.ScoreMargin)
	}
	if summary.FinalEval.PairAccuracy < before.PairAccuracy {
		t.Fatalf("ffn pair accuracy regressed: before=%f after=%f", before.PairAccuracy, summary.FinalEval.PairAccuracy)
	}
}

func TestEmbeddingTrainerFitContrastiveAttentionImprovesEval(t *testing.T) {
	trainer := newTinyTrainableAttentionEmbeddingTrainer(t, 0.05)
	trainSet := tinyEmbeddingContrastiveDataset()

	before, err := trainer.EvaluateContrastive(trainSet)
	if err != nil {
		t.Fatalf("eval before: %v", err)
	}
	summary, err := trainer.FitContrastive(trainSet, trainSet, EmbeddingTrainRunConfig{
		Epochs:      6,
		BatchSize:   2,
		Shuffle:     true,
		Seed:        7,
		RestoreBest: true,
	})
	if err != nil {
		t.Fatalf("fit contrastive attention: %v", err)
	}
	if summary.FinalEval == nil {
		t.Fatal("expected final eval metrics")
	}
	if summary.FinalEval.ScoreMargin <= before.ScoreMargin {
		t.Fatalf("attention score margin did not improve: before=%f after=%f", before.ScoreMargin, summary.FinalEval.ScoreMargin)
	}
	if summary.FinalEval.PairAccuracy < before.PairAccuracy {
		t.Fatalf("attention pair accuracy regressed: before=%f after=%f", before.PairAccuracy, summary.FinalEval.PairAccuracy)
	}
}

func TestEmbeddingTrainerFitContrastiveEncoderProducesStableEval(t *testing.T) {
	trainer := newTinyTrainableEncoderEmbeddingTrainer(t, 0.02)
	trainSet := tinyEncoderContrastiveDataset()

	before, err := trainer.EvaluateContrastive(trainSet)
	if err != nil {
		t.Fatalf("eval before: %v", err)
	}
	summary, err := trainer.FitContrastive(trainSet, trainSet, EmbeddingTrainRunConfig{
		Epochs:      6,
		BatchSize:   2,
		Shuffle:     true,
		Seed:        7,
		RestoreBest: true,
	})
	if err != nil {
		t.Fatalf("fit contrastive encoder: %v", err)
	}
	if summary.FinalEval == nil || summary.BestEval == nil {
		t.Fatal("expected final and best eval metrics")
	}
	if summary.StepsCompleted == 0 || len(summary.History) == 0 {
		t.Fatalf("expected encoder fit to execute training steps, got summary %+v", summary)
	}
	if summary.FinalEval.ScoreMargin+0.000001 < before.ScoreMargin {
		t.Fatalf("encoder score margin regressed: before=%f after=%f", before.ScoreMargin, summary.FinalEval.ScoreMargin)
	}
	if summary.FinalEval.PairAccuracy+0.000001 < before.PairAccuracy {
		t.Fatalf("encoder pair accuracy regressed: before=%f after=%f", before.PairAccuracy, summary.FinalEval.PairAccuracy)
	}
}

func TestEmbeddingTrainerFitContrastiveRepeatedEncoderProducesStableEval(t *testing.T) {
	trainer := newTinyTrainableRepeatedEncoderEmbeddingTrainer(t, 0.02)
	trainSet := tinyEncoderContrastiveDataset()

	before, err := trainer.EvaluateContrastive(trainSet)
	if err != nil {
		t.Fatalf("eval before: %v", err)
	}
	summary, err := trainer.FitContrastive(trainSet, trainSet, EmbeddingTrainRunConfig{
		Epochs:      6,
		BatchSize:   2,
		Shuffle:     true,
		Seed:        7,
		RestoreBest: true,
	})
	if err != nil {
		t.Fatalf("fit contrastive repeated encoder: %v", err)
	}
	if summary.FinalEval == nil || summary.BestEval == nil {
		t.Fatal("expected final and best eval metrics")
	}
	if summary.StepsCompleted == 0 || len(summary.History) == 0 {
		t.Fatalf("expected repeated encoder fit to execute training steps, got summary %+v", summary)
	}
	if summary.FinalEval.ScoreMargin+0.000001 < before.ScoreMargin {
		t.Fatalf("repeated encoder score margin regressed: before=%f after=%f", before.ScoreMargin, summary.FinalEval.ScoreMargin)
	}
	if summary.FinalEval.PairAccuracy+0.000001 < before.PairAccuracy {
		t.Fatalf("repeated encoder pair accuracy regressed: before=%f after=%f", before.PairAccuracy, summary.FinalEval.PairAccuracy)
	}
}

func TestEmbeddingTrainerEvaluateContrastiveMatchesExpandedPairs(t *testing.T) {
	trainer := newTinyTrainableAttentionEmbeddingTrainer(t, 0.05)
	contrastive := tinyEmbeddingContrastiveDataset()

	got, err := trainer.EvaluateContrastive(contrastive)
	if err != nil {
		t.Fatalf("evaluate contrastive: %v", err)
	}
	want, err := trainer.EvaluatePairs(expandContrastiveExamples(contrastive))
	if err != nil {
		t.Fatalf("evaluate expanded pairs: %v", err)
	}

	assertClose(t, got.Loss, want.Loss, 0.000001)
	assertClose(t, got.AverageScore, want.AverageScore, 0.000001)
	assertClose(t, got.PositiveMeanScore, want.PositiveMeanScore, 0.000001)
	assertClose(t, got.NegativeMeanScore, want.NegativeMeanScore, 0.000001)
	assertClose(t, got.PairAccuracy, want.PairAccuracy, 0.000001)
	assertClose(t, got.ScoreMargin, want.ScoreMargin, 0.000001)
	if got.PairCount != want.PairCount {
		t.Fatalf("pair count = %d, want %d", got.PairCount, want.PairCount)
	}
	if got.PositiveCount != want.PositiveCount {
		t.Fatalf("positive count = %d, want %d", got.PositiveCount, want.PositiveCount)
	}
	if got.NegativeCount != want.NegativeCount {
		t.Fatalf("negative count = %d, want %d", got.NegativeCount, want.NegativeCount)
	}
}

func TestEmbeddingTrainerTrainContrastiveStepMatchesExpandedPairStep(t *testing.T) {
	left := newTinyTrainableAttentionEmbeddingTrainer(t, 0.05)
	checkpoint, err := left.Checkpoint()
	if err != nil {
		t.Fatalf("checkpoint: %v", err)
	}
	right, err := NewEmbeddingTrainerFromCheckpoint(left.module, checkpoint)
	if err != nil {
		t.Fatalf("restore: %v", err)
	}
	left.Close()
	right.Close()

	batch := tinyEmbeddingContrastiveDataset()
	got, err := left.TrainContrastiveStep(batch)
	if err != nil {
		t.Fatalf("train contrastive step: %v", err)
	}
	want, err := right.TrainStep(expandContrastiveExamples(batch))
	if err != nil {
		t.Fatalf("train expanded pairs: %v", err)
	}

	assertClose(t, got.Loss, want.Loss, 0.000001)
	assertClose(t, got.AverageScore, want.AverageScore, 0.000001)
	if got.BatchSize != want.BatchSize {
		t.Fatalf("batch size = %d, want %d", got.BatchSize, want.BatchSize)
	}

	leftEval, err := left.EvaluateContrastive(batch)
	if err != nil {
		t.Fatalf("eval left: %v", err)
	}
	rightEval, err := right.EvaluateContrastive(batch)
	if err != nil {
		t.Fatalf("eval right: %v", err)
	}
	assertClose(t, leftEval.Loss, rightEval.Loss, 0.000001)
	assertClose(t, leftEval.ScoreMargin, rightEval.ScoreMargin, 0.000001)
	assertClose(t, leftEval.PairAccuracy, rightEval.PairAccuracy, 0.000001)
}

func TestEmbeddingTrainerTrainContrastiveStepSupportsInfoNCE(t *testing.T) {
	trainer := newTinyTrainableAttentionEmbeddingTrainer(t, 0.005)
	trainer.config.ContrastiveLoss = "infonce"
	trainer.config.Temperature = 0.05
	batch := tinyEmbeddingContrastiveDataset()

	before, err := trainer.EvaluateContrastive(batch)
	if err != nil {
		t.Fatalf("eval before: %v", err)
	}
	metrics, err := trainer.TrainContrastiveStep(batch)
	if err != nil {
		t.Fatalf("train infonce: %v", err)
	}
	if metrics.BatchSize != len(batch)*len(batch) {
		t.Fatalf("batch size = %d, want %d", metrics.BatchSize, len(batch)*len(batch))
	}
	after, err := trainer.EvaluateContrastive(batch)
	if err != nil {
		t.Fatalf("eval after: %v", err)
	}
	if after.PairCount != len(batch)*len(batch) {
		t.Fatalf("pair count = %d, want %d", after.PairCount, len(batch)*len(batch))
	}
	if loss := infoNCERowLoss([]float32{0.2, 0.1}, 0, 0.05); loss <= 0 {
		t.Fatalf("infonce row loss = %f, want positive", loss)
	}
	if after.Loss > before.Loss+0.000001 {
		t.Fatalf("infonce eval loss regressed: before=%f after=%f", before.Loss, after.Loss)
	}
}

func TestContrastiveCosineFastPathMatchesCosineGrad(t *testing.T) {
	left := []float32{0.25, -0.5, 0.75, 1.25}
	right := []float32{-0.75, 0.5, 0.125, 1.5}
	score, wantLeft, wantRight := cosineGrad(left, right)
	leftNorm := vectorNorm(left)
	rightNorm := vectorNorm(right)

	gotScore := cosineScoreWithNorms(left, right, leftNorm, rightNorm)
	assertClose(t, gotScore, score, 0.000001)

	gotLeft := make([]float32, len(left))
	gotRight := make([]float32, len(right))
	accumulateCosineGradFromScore(left, right, leftNorm, rightNorm, gotScore, 1, gotLeft, gotRight)
	for i := range wantLeft {
		assertClose(t, gotLeft[i], wantLeft[i], 0.000001)
		assertClose(t, gotRight[i], wantRight[i], 0.000001)
	}
}

func BenchmarkAccumulateInfoNCEContrastiveGrads128x64(b *testing.B) {
	queries, positives := syntheticContrastiveEncodings(128, 64)
	queryGrads := make([][]float32, len(queries))
	positiveGrads := make([][]float32, len(positives))
	for i := range queries {
		queryGrads[i] = make([]float32, len(queries[i].pooled))
		positiveGrads[i] = make([]float32, len(positives[i].pooled))
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for row := range queryGrads {
			clear(queryGrads[row])
			clear(positiveGrads[row])
		}
		accumulateInfoNCEContrastiveGrads(queries, positives, 0.05, queryGrads, positiveGrads)
	}
}

func syntheticContrastiveEncodings(rows, width int) ([]*embeddingEncodedSequence, []*embeddingEncodedSequence) {
	queries := make([]*embeddingEncodedSequence, rows)
	positives := make([]*embeddingEncodedSequence, rows)
	for row := 0; row < rows; row++ {
		query := make([]float32, width)
		positive := make([]float32, width)
		for col := 0; col < width; col++ {
			value := float32(((row+1)*(col+3))%23) / 23
			query[col] = value - 0.5
			positive[col] = value*0.75 + float32((row+col)%7)/17 - 0.35
		}
		queries[row] = &embeddingEncodedSequence{pooled: query}
		positives[row] = &embeddingEncodedSequence{pooled: positive}
	}
	return queries, positives
}

func newTinyTrainableEmbeddingTrainer(t *testing.T, learningRate float32) *EmbeddingTrainer {
	t.Helper()
	src := []byte(`
param token_embedding: q8[V, D] @weight("weights/token_embedding") @trainable
param projection: q8[D, E] @weight("weights/projection") @trainable

pipeline embed_pooled(tokens: i32[T], attention_mask: i32[T]) -> f16[E] {
    let hidden_q = gather(token_embedding, tokens)
    let hidden = dequant(hidden_q)
    let projection_f = dequant(projection)
    let projected = @matmul(hidden, projection_f)
    let normalized = normalize(projected)
    return mean_pool(normalized, attention_mask)
}

pipeline embed_pooled_batch(tokens: i32[B, T], attention_mask: i32[B, T]) -> f16[B, E] {
    let hidden_q = gather(token_embedding, tokens)
    let hidden = dequant(hidden_q)
    let projection_f = dequant(projection)
    let projected = @matmul(hidden, projection_f)
    let normalized = normalize(projected)
    return mean_pool(normalized, attention_mask)
}
`)

	bundle, err := compiler.Build(src, compiler.Options{ModuleName: "tiny_train_embed_q8"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	manifest := tinyMaskedEmbeddingManifest()
	manifest.Name = "tiny_train_embed_q8"
	trainer, err := NewEmbeddingTrainer(bundle.Artifact, manifest, map[string]*backend.Tensor{
		"token_embedding": backend.NewTensorF32([]int{3, 2}, []float32{
			1, 0,
			0, 1,
			1, 1,
		}),
		"projection": backend.NewTensorF32([]int{2, 2}, []float32{
			1, 0,
			0, 1,
		}),
	}, EmbeddingTrainConfig{LearningRate: learningRate})
	if err != nil {
		t.Fatalf("new trainer: %v", err)
	}
	return trainer
}

func tinyEmbeddingPairDataset() []EmbeddingPairExample {
	return []EmbeddingPairExample{
		{LeftTokens: []int32{0}, RightTokens: []int32{0}, LeftMask: []int32{1}, RightMask: []int32{1}, Target: 1},
		{LeftTokens: []int32{1}, RightTokens: []int32{1}, LeftMask: []int32{1}, RightMask: []int32{1}, Target: 1},
		{LeftTokens: []int32{0}, RightTokens: []int32{1}, LeftMask: []int32{1}, RightMask: []int32{1}, Target: -1},
		{LeftTokens: []int32{1}, RightTokens: []int32{0}, LeftMask: []int32{1}, RightMask: []int32{1}, Target: -1},
	}
}

func tinyEmbeddingContrastiveDataset() []EmbeddingContrastiveExample {
	return []EmbeddingContrastiveExample{
		{QueryTokens: []int32{0}, PositiveTokens: []int32{0}, QueryMask: []int32{1}, PositiveMask: []int32{1}},
		{QueryTokens: []int32{1}, PositiveTokens: []int32{1}, QueryMask: []int32{1}, PositiveMask: []int32{1}},
	}
}

func tinyEncoderPairDataset() []EmbeddingPairExample {
	return []EmbeddingPairExample{
		{LeftTokens: []int32{0, 2}, RightTokens: []int32{0, 0}, LeftMask: []int32{1, 1}, RightMask: []int32{1, 1}, Target: 0.5},
		{LeftTokens: []int32{1, 2}, RightTokens: []int32{1, 1}, LeftMask: []int32{1, 1}, RightMask: []int32{1, 1}, Target: 0.5},
		{LeftTokens: []int32{0, 0}, RightTokens: []int32{1, 1}, LeftMask: []int32{1, 1}, RightMask: []int32{1, 1}, Target: -0.5},
		{LeftTokens: []int32{1, 1}, RightTokens: []int32{0, 0}, LeftMask: []int32{1, 1}, RightMask: []int32{1, 1}, Target: -0.5},
	}
}

func tinyEncoderContrastiveDataset() []EmbeddingContrastiveExample {
	return []EmbeddingContrastiveExample{
		{QueryTokens: []int32{0, 2}, PositiveTokens: []int32{0, 0}, QueryMask: []int32{1, 1}, PositiveMask: []int32{1, 1}},
		{QueryTokens: []int32{1, 2}, PositiveTokens: []int32{1, 1}, QueryMask: []int32{1, 1}, PositiveMask: []int32{1, 1}},
	}
}
