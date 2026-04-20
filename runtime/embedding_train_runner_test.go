package mantaruntime

import (
	"math/rand"
	"testing"
	"time"

	mantaartifact "github.com/odvcencio/manta/artifact/manta"
	"github.com/odvcencio/manta/compiler"
	"github.com/odvcencio/manta/runtime/backend"
)

func TestGroupHardNegativeOrderBySource(t *testing.T) {
	// Build a trainSet where Source tags are interleaved in the input
	// order and each source has enough examples for full batches.
	//
	//   sources: A=5, B=3, C=7 examples; batch=2
	//   expected counts after regrouping: A=4 (1 dropped), B=2 (1
	//   dropped), C=6 (1 dropped).
	sources := []string{
		"A", "B", "C", "A", "C", "B", "A", "C", "A", "B",
		"C", "A", "C", "C", "C",
	}
	trainSet := make([]EmbeddingHardNegativeExample, len(sources))
	order := make([]int, len(sources))
	for i, src := range sources {
		trainSet[i] = EmbeddingHardNegativeExample{Source: src}
		order[i] = i
	}
	rng := rand.New(rand.NewSource(17))
	got := groupHardNegativeOrderBySource(trainSet, order, 2, rng, "")

	// Invariant 1: length is a multiple of batch size.
	if len(got)%2 != 0 {
		t.Fatalf("regrouped order length %d is not a multiple of batch size 2", len(got))
	}
	// Invariant 2: each batch-size window is source-homogeneous.
	for start := 0; start < len(got); start += 2 {
		window := got[start : start+2]
		first := trainSet[window[0]].Source
		for _, idx := range window[1:] {
			if trainSet[idx].Source != first {
				t.Fatalf("batch starting at %d mixes sources: %v (indices %v)",
					start,
					[]string{trainSet[window[0]].Source, trainSet[window[1]].Source},
					window)
			}
		}
	}
	// Invariant 3: every emitted index is in range and unique.
	seen := map[int]bool{}
	for _, idx := range got {
		if idx < 0 || idx >= len(trainSet) {
			t.Fatalf("regrouped index %d out of range 0..%d", idx, len(trainSet)-1)
		}
		if seen[idx] {
			t.Fatalf("regrouped index %d appears twice", idx)
		}
		seen[idx] = true
	}
	// Invariant 4: counts-per-source match the batch-aligned totals.
	counts := map[string]int{}
	for _, idx := range got {
		counts[trainSet[idx].Source]++
	}
	want := map[string]int{"A": 4, "B": 2, "C": 6}
	for src, n := range want {
		if counts[src] != n {
			t.Fatalf("source %q contributed %d examples, want %d", src, counts[src], n)
		}
	}
}

func TestGroupHardNegativeOrderBySourceFallsThroughWhenSingleOrUnsourced(t *testing.T) {
	// Single source: no regrouping needed, return original order.
	trainSet := []EmbeddingHardNegativeExample{
		{Source: "solo"}, {Source: "solo"}, {Source: "solo"},
	}
	order := []int{0, 1, 2}
	got := groupHardNegativeOrderBySource(trainSet, order, 2, rand.New(rand.NewSource(1)), "")
	if len(got) != len(order) {
		t.Fatalf("single-source regroup changed length: got %d want %d", len(got), len(order))
	}

	// No sources at all: fall through unchanged.
	unsourced := []EmbeddingHardNegativeExample{{}, {}, {}}
	got = groupHardNegativeOrderBySource(unsourced, order, 2, rand.New(rand.NewSource(1)), "")
	if len(got) != len(order) {
		t.Fatalf("unsourced regroup changed length: got %d want %d", len(got), len(order))
	}
}

func TestGroupHardNegativeOrderBySourceCollapsesPrefix(t *testing.T) {
	// With prefixSeparator=":", "scifact" and "scifact:model" must be
	// grouped together as one "scifact" bucket; ditto for "fiqa" and
	// "fiqa:model". Without the separator they are separate.
	sources := []string{
		"scifact", "scifact:model", "scifact", "scifact:model",
		"fiqa", "fiqa:model", "fiqa", "fiqa:model",
	}
	trainSet := make([]EmbeddingHardNegativeExample, len(sources))
	order := make([]int, len(sources))
	for i, src := range sources {
		trainSet[i] = EmbeddingHardNegativeExample{Source: src}
		order[i] = i
	}
	rng := rand.New(rand.NewSource(1))
	got := groupHardNegativeOrderBySource(trainSet, order, 4, rng, ":")
	// With batchSize=4 and two collapsed groups of 4 each we should
	// keep everyone and emit exactly 8 indices.
	if len(got) != 8 {
		t.Fatalf("collapsed regroup dropped too many: got %d want 8", len(got))
	}
	// Each 4-element window must share a collapsed prefix.
	for start := 0; start < len(got); start += 4 {
		window := got[start : start+4]
		prefix := trainSet[window[0]].Source
		if i := len(prefix); i > 0 {
			if idx := indexOfColon(prefix); idx >= 0 {
				prefix = prefix[:idx]
			}
		}
		for _, idx := range window[1:] {
			src := trainSet[idx].Source
			if colon := indexOfColon(src); colon >= 0 {
				src = src[:colon]
			}
			if src != prefix {
				t.Fatalf("collapsed batch at %d mixes prefixes %q vs %q", start, prefix, src)
			}
		}
	}
}

func indexOfColon(s string) int {
	for i := 0; i < len(s); i++ {
		if s[i] == ':' {
			return i
		}
	}
	return -1
}

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
	if _, err := trainer.Fit(trainSet, nil, EmbeddingTrainRunConfig{Epochs: 1, BatchSize: 2, ProgressEverySteps: -1}); err == nil {
		t.Fatal("expected progress interval error")
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

func TestEmbeddingTrainerFitContrastiveReportsProgress(t *testing.T) {
	trainer := newTinyTrainableEmbeddingTrainer(t, 0.05)
	trainSet := tinyEmbeddingContrastiveDataset()
	var reports []EmbeddingTrainProgress

	summary, err := trainer.FitContrastive(trainSet, nil, EmbeddingTrainRunConfig{
		Epochs:             1,
		BatchSize:          2,
		Shuffle:            false,
		ProgressEverySteps: 1,
		Progress: func(progress EmbeddingTrainProgress) {
			reports = append(reports, progress)
		},
	})
	if err != nil {
		t.Fatalf("fit contrastive: %v", err)
	}
	if len(reports) != summary.StepsRun {
		t.Fatalf("progress reports = %d, want %d", len(reports), summary.StepsRun)
	}
	got := reports[0]
	if got.Epoch != 1 || got.Batch != 1 || got.Batches != 1 {
		t.Fatalf("progress position = epoch %d batch %d/%d, want 1 1/1", got.Epoch, got.Batch, got.Batches)
	}
	if got.BatchExamples != len(trainSet) {
		t.Fatalf("batch examples = %d, want %d", got.BatchExamples, len(trainSet))
	}
	if got.BatchPairs != int64(len(trainSet)*len(trainSet)) {
		t.Fatalf("batch pairs = %d, want %d", got.BatchPairs, len(trainSet)*len(trainSet))
	}
	if got.Step != trainer.step {
		t.Fatalf("progress step = %d, want trainer step %d", got.Step, trainer.step)
	}
	if got.Elapsed <= 0 {
		t.Fatalf("progress elapsed = %s, want > 0", got.Elapsed)
	}
}

func TestEmbeddingTrainerFitContrastiveEvalOnlyDoesNotTrain(t *testing.T) {
	trainer := newTinyTrainableEmbeddingTrainer(t, 0.05)
	evalSet := tinyEmbeddingContrastiveDataset()

	summary, err := trainer.FitContrastive(nil, evalSet, EmbeddingTrainRunConfig{EvalOnly: true})
	if err != nil {
		t.Fatalf("fit contrastive eval-only: %v", err)
	}
	if summary.EpochsCompleted != 0 {
		t.Fatalf("epochs completed = %d, want 0", summary.EpochsCompleted)
	}
	if summary.StepsRun != 0 {
		t.Fatalf("steps run = %d, want 0", summary.StepsRun)
	}
	if summary.FinalEval == nil {
		t.Fatal("expected final eval metrics")
	}
	if summary.Workload.PlannedTrainPairs != 0 || summary.Workload.ActualTrainPairs != 0 {
		t.Fatalf("train pairs planned/actual = %d/%d, want 0/0", summary.Workload.PlannedTrainPairs, summary.Workload.ActualTrainPairs)
	}
	if summary.Workload.PlannedEvalPasses != 1 || summary.Workload.ActualEvalPasses != 1 {
		t.Fatalf("eval passes planned/actual = %d/%d, want 1/1", summary.Workload.PlannedEvalPasses, summary.Workload.ActualEvalPasses)
	}
}

func TestEmbeddingTrainerFitContrastiveEvaluatesWithinEpoch(t *testing.T) {
	trainer := newTinyTrainableEmbeddingTrainer(t, 0.05)
	trainSet := tinyEmbeddingContrastiveDataset()

	summary, err := trainer.FitContrastive(trainSet, trainSet, EmbeddingTrainRunConfig{
		Epochs:         1,
		BatchSize:      2,
		Shuffle:        false,
		EvalEveryEpoch: 99,
		EvalEverySteps: 1,
		SelectMetric:   "mrr",
		RestoreBest:    true,
	})
	if err != nil {
		t.Fatalf("fit contrastive: %v", err)
	}
	if summary.Workload.PlannedEvalPasses != 3 || summary.Workload.ActualEvalPasses != 3 {
		t.Fatalf("eval passes planned/actual = %d/%d, want 3/3", summary.Workload.PlannedEvalPasses, summary.Workload.ActualEvalPasses)
	}
	if summary.BestEval == nil || summary.FinalEval == nil {
		t.Fatal("expected best and final eval metrics")
	}
	if !summary.RestoredBest {
		t.Fatal("expected restore-best to use the best checkpoint")
	}
}

func TestBucketContrastiveOrderByLengthSortsWithinDefaultWindows(t *testing.T) {
	t.Setenv("MANTA_TRAIN_LENGTH_BUCKET_WINDOW", "")
	trainSet := []EmbeddingContrastiveExample{
		{QueryTokens: make([]int32, 8), PositiveTokens: make([]int32, 1)},
		{QueryTokens: make([]int32, 2), PositiveTokens: make([]int32, 1)},
		{QueryTokens: make([]int32, 7), PositiveTokens: make([]int32, 1)},
		{QueryTokens: make([]int32, 3), PositiveTokens: make([]int32, 1)},
		{QueryTokens: make([]int32, 6), PositiveTokens: make([]int32, 1)},
		{QueryTokens: make([]int32, 4), PositiveTokens: make([]int32, 1)},
		{QueryTokens: make([]int32, 5), PositiveTokens: make([]int32, 1)},
		{QueryTokens: make([]int32, 1), PositiveTokens: make([]int32, 1)},
		{QueryTokens: make([]int32, 1), PositiveTokens: make([]int32, 1)},
		{QueryTokens: make([]int32, 9), PositiveTokens: make([]int32, 1)},
	}
	order := []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

	bucketContrastiveOrderByLength(trainSet, order, 2)

	want := []int{7, 1, 3, 5, 6, 4, 2, 0, 8, 9}
	for i := range want {
		if order[i] != want[i] {
			t.Fatalf("order[%d] = %d, want %d (full order %v)", i, order[i], want[i], order)
		}
	}
}

func TestBucketContrastiveOrderByLengthPreservesEqualLengthShuffleOrder(t *testing.T) {
	t.Setenv("MANTA_TRAIN_LENGTH_BUCKET_WINDOW", "")
	trainSet := []EmbeddingContrastiveExample{
		{QueryTokens: make([]int32, 8), PositiveTokens: make([]int32, 10)},
		{QueryTokens: make([]int32, 2), PositiveTokens: make([]int32, 10)},
		{QueryTokens: make([]int32, 5), PositiveTokens: make([]int32, 10)},
		{QueryTokens: make([]int32, 10), PositiveTokens: make([]int32, 3)},
	}
	order := []int{0, 1, 2, 3}

	bucketContrastiveOrderByLength(trainSet, order, 2)

	want := []int{0, 1, 2, 3}
	for i := range want {
		if order[i] != want[i] {
			t.Fatalf("order[%d] = %d, want %d (full order %v)", i, order[i], want[i], order)
		}
	}
}

func TestBucketContrastiveOrderByLengthHonorsWindowOverride(t *testing.T) {
	t.Setenv("MANTA_TRAIN_LENGTH_BUCKET_WINDOW", "4")
	trainSet := []EmbeddingContrastiveExample{
		{QueryTokens: make([]int32, 8), PositiveTokens: make([]int32, 1)},
		{QueryTokens: make([]int32, 1), PositiveTokens: make([]int32, 1)},
		{QueryTokens: make([]int32, 7), PositiveTokens: make([]int32, 1)},
		{QueryTokens: make([]int32, 2), PositiveTokens: make([]int32, 1)},
		{QueryTokens: make([]int32, 6), PositiveTokens: make([]int32, 1)},
		{QueryTokens: make([]int32, 3), PositiveTokens: make([]int32, 1)},
	}
	order := []int{0, 1, 2, 3, 4, 5}

	bucketContrastiveOrderByLength(trainSet, order, 2)

	want := []int{1, 3, 2, 0, 5, 4}
	for i := range want {
		if order[i] != want[i] {
			t.Fatalf("order[%d] = %d, want %d (full order %v)", i, order[i], want[i], order)
		}
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
		RestoreBest:    false,
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

func TestEmbeddingTrainerFitHardNegativesTracksPairwiseEval(t *testing.T) {
	trainer := newTinyTrainableAttentionEmbeddingTrainer(t, 0.005)
	trainSet := tinyEmbeddingHardNegativeDataset()
	evalSet := tinyEmbeddingPairDataset()
	summary, err := trainer.FitHardNegatives(trainSet, evalSet, EmbeddingTrainRunConfig{
		Epochs:                2,
		BatchSize:             2,
		EvalEveryEpoch:        1,
		SelectMetric:          "mrr",
		RestoreBest:           true,
		HardNegativeTrain:     true,
		HardNegativesPerQuery: 1,
	})
	if err != nil {
		t.Fatalf("fit hard negatives: %v", err)
	}
	if summary.Workload.TrainMode != "hard_negative_contrastive" {
		t.Fatalf("train mode = %q", summary.Workload.TrainMode)
	}
	if summary.FinalEval == nil || summary.FinalEval.PairCount != len(evalSet) {
		t.Fatalf("final eval = %+v, want %d pairs", summary.FinalEval, len(evalSet))
	}
	if summary.StepsRun == 0 {
		t.Fatal("expected optimizer steps")
	}
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
	assertClose(t, got.ThresholdAccuracy, want.ThresholdAccuracy, 0.000001)
	assertClose(t, got.ScoreThreshold, want.ScoreThreshold, 0.000001)
	assertClose(t, got.ROCAUC, want.ROCAUC, 0.000001)
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

func TestEvaluateContrastiveEncodingsTracksRankingMetrics(t *testing.T) {
	queries := []*embeddingEncodedSequence{
		{pooled: []float32{1, 0, 0}},
		{pooled: []float32{0, 1, 0}},
		{pooled: []float32{0, 0, 1}},
	}
	positives := []*embeddingEncodedSequence{
		{pooled: []float32{0, 1, 0}},
		{pooled: []float32{1, 0, 0}},
		{pooled: []float32{0, 0, 1}},
	}

	got := evaluateContrastiveEncodings(queries, positives, EmbeddingTrainConfig{})

	assertClose(t, got.Top1Accuracy, 1.0/3.0, 0.000001)
	assertClose(t, got.Top5Accuracy, 1, 0.000001)
	assertClose(t, got.Top10Accuracy, 1, 0.000001)
	assertClose(t, got.MeanReciprocalRank, 2.0/3.0, 0.000001)
	assertClose(t, got.MeanPositiveRank, 5.0/3.0, 0.000001)
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

func TestEmbeddingTrainerTrainHardNegativeContrastiveStep(t *testing.T) {
	trainer := newTinyTrainableAttentionEmbeddingTrainer(t, 0.005)
	trainer.config.ContrastiveLoss = "infonce"
	trainer.config.Temperature = 0.05
	batch := tinyEmbeddingHardNegativeDataset()

	metrics, err := trainer.TrainHardNegativeContrastiveStep(batch)
	if err != nil {
		t.Fatalf("train hard-negative step: %v", err)
	}
	if metrics.BatchSize != 8 {
		t.Fatalf("batch size = %d, want 8 rectangular query-candidate scores", metrics.BatchSize)
	}
	if metrics.Loss <= 0 {
		t.Fatalf("loss = %f, want positive", metrics.Loss)
	}
	if trainer.step != 1 {
		t.Fatalf("step = %d, want 1", trainer.step)
	}
}

func TestEmbeddingTrainerTrainHardNegativeContrastiveStepUsesRectangularAccelerator(t *testing.T) {
	trainer := newTinyTrainableAttentionEmbeddingTrainer(t, 0.005)
	trainer.config.ContrastiveLoss = "infonce"
	trainer.config.Temperature = 0.05
	accel := &countingContrastiveAccelerator{}
	trainer.contrastiveAccel = accel

	_, err := trainer.TrainHardNegativeContrastiveStep(tinyEmbeddingHardNegativeDataset())
	if err != nil {
		t.Fatalf("train hard-negative step: %v", err)
	}
	if accel.rectCalls != 1 {
		t.Fatalf("rectangular calls = %d, want 1", accel.rectCalls)
	}
	if accel.squareCalls != 0 {
		t.Fatalf("square calls = %d, want 0", accel.squareCalls)
	}
	if accel.queryRows != 2 || accel.candidateRows != 4 || accel.width == 0 {
		t.Fatalf("accelerator shape query=%d candidate=%d width=%d, want 2x4xwidth", accel.queryRows, accel.candidateRows, accel.width)
	}
	if len(accel.targetIndexes) != 2 || accel.targetIndexes[0] != 0 || accel.targetIndexes[1] != 2 {
		t.Fatalf("target indexes = %v, want [0 2]", accel.targetIndexes)
	}
}

func TestCollectContrastiveBackpropItemsMergesDuplicateSequences(t *testing.T) {
	shared := &embeddingEncodedSequence{layers: []*embeddingSequenceState{{}}, pooled: []float32{0, 0}}
	other := &embeddingEncodedSequence{layers: []*embeddingSequenceState{{}}, pooled: []float32{0, 0}}
	queryGrad := []float32{1, 2}
	firstCandidateGrad := []float32{3, 4}
	secondCandidateGrad := []float32{5, 6}

	items, ok := collectContrastiveBackpropItems(
		[]*embeddingEncodedSequence{shared},
		[]*embeddingEncodedSequence{other, shared},
		[][]float32{queryGrad},
		[][]float32{firstCandidateGrad, secondCandidateGrad},
	)
	if !ok {
		t.Fatal("collect backprop items failed")
	}
	if len(items) != 2 {
		t.Fatalf("items = %d, want 2 unique sequences", len(items))
	}
	if items[0].seq != shared {
		t.Fatalf("first item seq = %p, want shared %p", items[0].seq, shared)
	}
	if got, want := items[0].gradPooled, []float32{6, 8}; !equalFloat32Slices(got, want) {
		t.Fatalf("merged shared grad = %v, want %v", got, want)
	}
	if !items[0].gradPooledOwned {
		t.Fatal("merged item should own copied gradient storage")
	}
	if got, want := queryGrad, []float32{1, 2}; !equalFloat32Slices(got, want) {
		t.Fatalf("original query grad mutated to %v, want %v", got, want)
	}
	if items[1].seq != other {
		t.Fatalf("second item seq = %p, want other %p", items[1].seq, other)
	}
	if got, want := items[1].gradPooled, firstCandidateGrad; !equalFloat32Slices(got, want) {
		t.Fatalf("other grad = %v, want %v", got, want)
	}
}

func equalFloat32Slices(left, right []float32) bool {
	if len(left) != len(right) {
		return false
	}
	for i := range left {
		if left[i] != right[i] {
			return false
		}
	}
	return true
}

type countingContrastiveAccelerator struct {
	squareCalls   int
	rectCalls     int
	queryRows     int
	candidateRows int
	width         int
	targetIndexes []int
}

func (a *countingContrastiveAccelerator) Backend() mantaartifact.BackendKind {
	return mantaartifact.BackendCUDA
}

func (a *countingContrastiveAccelerator) RunInfoNCE(query, positive *backend.Tensor, cfg backend.ContrastiveLossConfig) (backend.ContrastiveGradResult, error) {
	a.squareCalls++
	if query == nil || positive == nil || query.Rank() != 2 || positive.Rank() != 2 {
		return backend.ContrastiveGradResult{}, nil
	}
	return backend.ContrastiveGradResult{
		QueryGrads:    backend.NewTensorF32(query.Shape, make([]float32, len(query.F32))),
		PositiveGrads: backend.NewTensorF32(positive.Shape, make([]float32, len(positive.F32))),
		LossSum:       1,
		ScoreSum:      1,
	}, nil
}

func (a *countingContrastiveAccelerator) RunInfoNCEWithTargets(query, candidates *backend.Tensor, targetIndexes []int, cfg backend.ContrastiveLossConfig) (backend.ContrastiveGradResult, error) {
	a.rectCalls++
	a.targetIndexes = append([]int(nil), targetIndexes...)
	if query == nil || candidates == nil || query.Rank() != 2 || candidates.Rank() != 2 {
		return backend.ContrastiveGradResult{}, nil
	}
	a.queryRows = query.Shape[0]
	a.candidateRows = candidates.Shape[0]
	a.width = query.Shape[1]
	return backend.ContrastiveGradResult{
		QueryGrads:    backend.NewTensorF32(query.Shape, make([]float32, len(query.F32))),
		PositiveGrads: backend.NewTensorF32(candidates.Shape, make([]float32, len(candidates.F32))),
		LossSum:       1,
		ScoreSum:      float32(query.Shape[0] * candidates.Shape[0]),
	}, nil
}

func (a *countingContrastiveAccelerator) Stats() backend.ContrastiveAcceleratorStats {
	return backend.ContrastiveAcceleratorStats{RunCalls: int64(a.squareCalls + a.rectCalls)}
}

func (a *countingContrastiveAccelerator) Close() {}

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

func tinyEmbeddingHardNegativeDataset() []EmbeddingHardNegativeExample {
	return []EmbeddingHardNegativeExample{
		{QueryTokens: []int32{0}, PositiveTokens: []int32{0}, NegativeTokens: [][]int32{{1}}, QueryMask: []int32{1}, PositiveMask: []int32{1}, NegativeMasks: [][]int32{{1}}},
		{QueryTokens: []int32{1}, PositiveTokens: []int32{1}, NegativeTokens: [][]int32{{0}}, QueryMask: []int32{1}, PositiveMask: []int32{1}, NegativeMasks: [][]int32{{1}}},
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
