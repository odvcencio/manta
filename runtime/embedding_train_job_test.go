package mantaruntime

import (
	"os"
	"path/filepath"
	"testing"
)

func TestTrainEmbeddingPackageFromContrastiveFiles(t *testing.T) {
	trainer := newTinyTrainableEmbeddingTrainer(t, 0.05)
	path := writeTinyTrainingPackage(t, trainer)

	trainPath := path[:len(path)-len(".mll")] + ".pairs.jsonl"
	evalPath := path[:len(path)-len(".mll")] + ".eval.jsonl"
	trainSet := []EmbeddingContrastiveExample{
		{QueryTokens: []int32{0}, PositiveTokens: []int32{0}, QueryMask: []int32{1}, PositiveMask: []int32{1}},
		{QueryTokens: []int32{1}, PositiveTokens: []int32{1}, QueryMask: []int32{1}, PositiveMask: []int32{1}},
	}
	if err := WriteEmbeddingContrastiveExamplesFile(trainPath, trainSet); err != nil {
		t.Fatalf("write train dataset: %v", err)
	}
	if err := WriteEmbeddingContrastiveExamplesFile(evalPath, trainSet); err != nil {
		t.Fatalf("write eval dataset: %v", err)
	}

	summary, paths, err := TrainEmbeddingPackageFromContrastiveFiles(path, trainPath, evalPath, EmbeddingTrainRunConfig{
		Epochs:      4,
		BatchSize:   2,
		Shuffle:     true,
		Seed:        7,
		RestoreBest: true,
	})
	if err != nil {
		t.Fatalf("train package: %v", err)
	}
	if summary.FinalEval == nil || summary.BestEval == nil {
		t.Fatal("expected eval metrics in summary")
	}
	if paths.ArtifactPath != path {
		t.Fatalf("artifact path = %q, want %q", paths.ArtifactPath, path)
	}
	reloaded, err := LoadEmbeddingTrainerPackage(path)
	if err != nil {
		t.Fatalf("reload trainer package: %v", err)
	}
	finalEval, err := reloaded.EvaluatePairs(expandContrastiveExamples(trainSet))
	if err != nil {
		t.Fatalf("eval reloaded: %v", err)
	}
	assertClose(t, finalEval.ScoreMargin, summary.FinalEval.ScoreMargin, 0.000001)
	assertClose(t, finalEval.PairAccuracy, summary.FinalEval.PairAccuracy, 0.000001)
}

func TestTrainFFNEmbeddingPackageFromContrastiveFiles(t *testing.T) {
	trainer := newTinyTrainableFFNEmbeddingTrainer(t, 0.05)
	path := writeTinyTrainingPackage(t, trainer)

	trainPath := path[:len(path)-len(".mll")] + ".pairs.jsonl"
	evalPath := path[:len(path)-len(".mll")] + ".eval.jsonl"
	trainSet := []EmbeddingContrastiveExample{
		{QueryTokens: []int32{0}, PositiveTokens: []int32{0}, QueryMask: []int32{1}, PositiveMask: []int32{1}},
		{QueryTokens: []int32{1}, PositiveTokens: []int32{1}, QueryMask: []int32{1}, PositiveMask: []int32{1}},
	}
	if err := WriteEmbeddingContrastiveExamplesFile(trainPath, trainSet); err != nil {
		t.Fatalf("write train dataset: %v", err)
	}
	if err := WriteEmbeddingContrastiveExamplesFile(evalPath, trainSet); err != nil {
		t.Fatalf("write eval dataset: %v", err)
	}

	summary, _, err := TrainEmbeddingPackageFromContrastiveFiles(path, trainPath, evalPath, EmbeddingTrainRunConfig{
		Epochs:      4,
		BatchSize:   2,
		Shuffle:     true,
		Seed:        7,
		RestoreBest: true,
	})
	if err != nil {
		t.Fatalf("train ffn package: %v", err)
	}
	if summary.FinalEval == nil || summary.BestEval == nil {
		t.Fatal("expected eval metrics in summary")
	}
	reloaded, err := LoadEmbeddingTrainerPackage(path)
	if err != nil {
		t.Fatalf("reload ffn trainer package: %v", err)
	}
	if reloaded.hiddenProjection == nil {
		t.Fatal("expected hidden projection after ffn package reload")
	}
	exported, err := reloaded.ExportInferenceWeights()
	if err != nil {
		t.Fatalf("export reloaded ffn weights: %v", err)
	}
	if _, ok := exported["ffn_up"]; !ok {
		t.Fatal("expected ffn_up in exported weights")
	}
	finalEval, err := reloaded.EvaluatePairs(expandContrastiveExamples(trainSet))
	if err != nil {
		t.Fatalf("eval reloaded ffn: %v", err)
	}
	assertClose(t, finalEval.ScoreMargin, summary.FinalEval.ScoreMargin, 0.000001)
	assertClose(t, finalEval.PairAccuracy, summary.FinalEval.PairAccuracy, 0.000001)
}

func TestTrainAttentionEmbeddingPackageFromContrastiveFiles(t *testing.T) {
	trainer := newTinyTrainableAttentionEmbeddingTrainer(t, 0.05)
	path := writeTinyTrainingPackage(t, trainer)

	trainPath := path[:len(path)-len(".mll")] + ".pairs.jsonl"
	evalPath := path[:len(path)-len(".mll")] + ".eval.jsonl"
	trainSet := []EmbeddingContrastiveExample{
		{QueryTokens: []int32{0}, PositiveTokens: []int32{0}, QueryMask: []int32{1}, PositiveMask: []int32{1}},
		{QueryTokens: []int32{1}, PositiveTokens: []int32{1}, QueryMask: []int32{1}, PositiveMask: []int32{1}},
	}
	if err := WriteEmbeddingContrastiveExamplesFile(trainPath, trainSet); err != nil {
		t.Fatalf("write train dataset: %v", err)
	}
	if err := WriteEmbeddingContrastiveExamplesFile(evalPath, trainSet); err != nil {
		t.Fatalf("write eval dataset: %v", err)
	}

	summary, _, err := TrainEmbeddingPackageFromContrastiveFiles(path, trainPath, evalPath, EmbeddingTrainRunConfig{
		Epochs:      4,
		BatchSize:   2,
		Shuffle:     true,
		Seed:        7,
		RestoreBest: true,
	})
	if err != nil {
		t.Fatalf("train attention package: %v", err)
	}
	if summary.FinalEval == nil || summary.BestEval == nil {
		t.Fatal("expected eval metrics in summary")
	}
	reloaded, err := LoadEmbeddingTrainerPackage(path)
	if err != nil {
		t.Fatalf("reload attention trainer package: %v", err)
	}
	if !reloaded.attentionEnabled() {
		t.Fatal("expected attention weights after package reload")
	}
	exported, err := reloaded.ExportInferenceWeights()
	if err != nil {
		t.Fatalf("export reloaded attention weights: %v", err)
	}
	for _, name := range []string{"attn_q", "attn_k", "attn_v", "attn_o"} {
		if _, ok := exported[name]; !ok {
			t.Fatalf("expected %s in exported weights", name)
		}
	}
	finalEval, err := reloaded.EvaluatePairs(expandContrastiveExamples(trainSet))
	if err != nil {
		t.Fatalf("eval reloaded attention: %v", err)
	}
	assertClose(t, finalEval.ScoreMargin, summary.FinalEval.ScoreMargin, 0.000001)
	assertClose(t, finalEval.PairAccuracy, summary.FinalEval.PairAccuracy, 0.000001)
}

func TestTrainEncoderEmbeddingPackageFromContrastiveFiles(t *testing.T) {
	trainer := newTinyTrainableEncoderEmbeddingTrainer(t, 0.02)
	path := writeTinyTrainingPackage(t, trainer)

	trainPath := path[:len(path)-len(".mll")] + ".pairs.jsonl"
	evalPath := path[:len(path)-len(".mll")] + ".eval.jsonl"
	trainSet := tinyEncoderContrastiveDataset()
	if err := WriteEmbeddingContrastiveExamplesFile(trainPath, trainSet); err != nil {
		t.Fatalf("write train dataset: %v", err)
	}
	if err := WriteEmbeddingContrastiveExamplesFile(evalPath, trainSet); err != nil {
		t.Fatalf("write eval dataset: %v", err)
	}

	summary, _, err := TrainEmbeddingPackageFromContrastiveFiles(path, trainPath, evalPath, EmbeddingTrainRunConfig{
		Epochs:      4,
		BatchSize:   2,
		Shuffle:     true,
		Seed:        7,
		RestoreBest: true,
	})
	if err != nil {
		t.Fatalf("train encoder package: %v", err)
	}
	if summary.FinalEval == nil || summary.BestEval == nil {
		t.Fatal("expected eval metrics in summary")
	}
	reloaded, err := LoadEmbeddingTrainerPackage(path)
	if err != nil {
		t.Fatalf("reload encoder trainer package: %v", err)
	}
	if !reloaded.attentionEnabled() {
		t.Fatal("expected attention weights after encoder package reload")
	}
	if reloaded.hiddenProjection == nil {
		t.Fatal("expected hidden projection after encoder package reload")
	}
	exported, err := reloaded.ExportInferenceWeights()
	if err != nil {
		t.Fatalf("export reloaded encoder weights: %v", err)
	}
	for _, name := range []string{"attn_q", "attn_k", "attn_v", "attn_o", "ffn_up"} {
		if _, ok := exported[name]; !ok {
			t.Fatalf("expected %s in exported weights", name)
		}
	}
	finalEval, err := reloaded.EvaluatePairs(expandContrastiveExamples(trainSet))
	if err != nil {
		t.Fatalf("eval reloaded encoder: %v", err)
	}
	assertClose(t, finalEval.ScoreMargin, summary.FinalEval.ScoreMargin, 0.000001)
	assertClose(t, finalEval.PairAccuracy, summary.FinalEval.PairAccuracy, 0.000001)
}

func TestTrainEmbeddingPackageFromTextContrastiveFiles(t *testing.T) {
	trainer := newTinyTrainableEmbeddingTrainer(t, 0.05)
	path := writeTinyTrainingPackage(t, trainer)

	tokenizerPath := DefaultTokenizerPath(path)
	tokenizerFile := TokenizerFile{
		Version: TokenizerFileVersion,
		Tokens:  []string{"a", "b", "c"},
	}
	if err := tokenizerFile.WriteFile(tokenizerPath); err != nil {
		t.Fatalf("write tokenizer: %v", err)
	}

	trainPath := path[:len(path)-len(".mll")] + ".text-pairs.jsonl"
	evalPath := path[:len(path)-len(".mll")] + ".text-eval.jsonl"
	trainSet := []EmbeddingTextContrastiveExample{
		{Query: "a", Positive: "a"},
		{Query: "b", Positive: "b"},
	}
	if err := WriteEmbeddingTextContrastiveExamplesFile(trainPath, trainSet); err != nil {
		t.Fatalf("write train text dataset: %v", err)
	}
	if err := WriteEmbeddingTextContrastiveExamplesFile(evalPath, trainSet); err != nil {
		t.Fatalf("write eval text dataset: %v", err)
	}

	summary, _, err := TrainEmbeddingPackageFromTextContrastiveFiles(path, tokenizerPath, trainPath, evalPath, EmbeddingTrainRunConfig{
		Epochs:      4,
		BatchSize:   2,
		Shuffle:     true,
		Seed:        7,
		RestoreBest: true,
	})
	if err != nil {
		t.Fatalf("train text package: %v", err)
	}
	if summary.FinalEval == nil || summary.BestEval == nil {
		t.Fatal("expected eval metrics in summary")
	}
}

func TestTrainEmbeddingPackageFromCorpusFile(t *testing.T) {
	trainer := newTinyTrainableEncoderEmbeddingTrainer(t, 0.02)
	path := writeTinyTrainingPackage(t, trainer)
	corpusPath := filepath.Join(t.TempDir(), "corpus.txt")
	corpus := "" +
		"alpha beta gamma. gamma delta epsilon.\n" +
		"delta epsilon zeta. eta theta iota.\n" +
		"kappa lambda mu. nu xi omicron.\n" +
		"pi rho sigma. tau upsilon phi.\n"
	if err := os.WriteFile(corpusPath, []byte(corpus), 0o644); err != nil {
		t.Fatalf("write corpus: %v", err)
	}
	summary, paths, err := TrainEmbeddingPackageFromCorpusFile(path, corpusPath, EmbeddingCorpusTrainConfig{
		TokenizerVocabSize: 24,
		TokenizerMinFreq:   1,
		Mining: EmbeddingTextMiningConfig{
			MinChars:  5,
			EvalPairs: 2,
			Seed:      7,
		},
		Run: EmbeddingTrainRunConfig{
			Epochs:      2,
			BatchSize:   2,
			Shuffle:     true,
			Seed:        7,
			RestoreBest: true,
		},
	})
	if err != nil {
		t.Fatalf("train from corpus: %v", err)
	}
	if summary.StepsCompleted == 0 {
		t.Fatal("expected training steps")
	}
	if _, err := os.Stat(paths.TokenizerPath); err != nil {
		t.Fatalf("expected tokenizer output: %v", err)
	}
	if _, err := os.Stat(paths.TrainPairsPath); err != nil {
		t.Fatalf("expected mined train pairs: %v", err)
	}
	if paths.EvalPairsPath == "" {
		t.Fatal("expected eval pairs path")
	}
	if _, err := os.Stat(paths.EvalPairsPath); err != nil {
		t.Fatalf("expected mined eval pairs: %v", err)
	}
	reloaded, err := LoadEmbeddingTrainerPackage(path)
	if err != nil {
		t.Fatalf("reload trained package: %v", err)
	}
	if reloaded.manifest.Tokenizer.VocabSize <= trainer.manifest.Tokenizer.VocabSize {
		t.Fatalf("expected vocab size to grow, got %d", reloaded.manifest.Tokenizer.VocabSize)
	}
}

func TestTrainEmbeddingPackageFromContrastiveFilesAcceptsTokenPairEval(t *testing.T) {
	trainer := newTinyTrainableEmbeddingTrainer(t, 0.05)
	path := writeTinyTrainingPackage(t, trainer)

	trainPath := path[:len(path)-len(".mll")] + ".pairs.jsonl"
	evalPath := path[:len(path)-len(".mll")] + ".eval-pairs.jsonl"
	trainSet := []EmbeddingContrastiveExample{
		{QueryTokens: []int32{0}, PositiveTokens: []int32{0}, QueryMask: []int32{1}, PositiveMask: []int32{1}},
		{QueryTokens: []int32{1}, PositiveTokens: []int32{1}, QueryMask: []int32{1}, PositiveMask: []int32{1}},
	}
	evalSet := []EmbeddingPairExample{
		{LeftTokens: []int32{0}, RightTokens: []int32{0}, LeftMask: []int32{1}, RightMask: []int32{1}, Target: 1},
		{LeftTokens: []int32{0}, RightTokens: []int32{1}, LeftMask: []int32{1}, RightMask: []int32{1}, Target: 0},
	}
	if err := WriteEmbeddingContrastiveExamplesFile(trainPath, trainSet); err != nil {
		t.Fatalf("write train dataset: %v", err)
	}
	if err := WriteEmbeddingPairExamplesFile(evalPath, evalSet); err != nil {
		t.Fatalf("write eval pair dataset: %v", err)
	}

	summary, _, err := TrainEmbeddingPackageFromContrastiveFiles(path, trainPath, evalPath, EmbeddingTrainRunConfig{
		Epochs:      2,
		BatchSize:   2,
		Shuffle:     true,
		Seed:        7,
		RestoreBest: true,
	})
	if err != nil {
		t.Fatalf("train package with token pair eval: %v", err)
	}
	if summary.FinalEval == nil {
		t.Fatal("expected final eval metrics in summary")
	}
	if summary.FinalEval.PairCount != len(evalSet) {
		t.Fatalf("final eval pair count = %d, want %d", summary.FinalEval.PairCount, len(evalSet))
	}
	if summary.Workload.EvalMode != "pairwise" {
		t.Fatalf("eval mode = %q, want pairwise", summary.Workload.EvalMode)
	}
}

func TestTrainEmbeddingPackageFromTextContrastiveFilesAcceptsLabeledEvalPairs(t *testing.T) {
	trainer := newTinyTrainableEmbeddingTrainer(t, 0.05)
	path := writeTinyTrainingPackage(t, trainer)

	tokenizerPath := DefaultTokenizerPath(path)
	tokenizerFile := TokenizerFile{
		Version:      TokenizerFileVersion,
		Tokens:       []string{"a", "b", "[UNK]"},
		UnknownToken: "[UNK]",
	}
	if err := tokenizerFile.WriteFile(tokenizerPath); err != nil {
		t.Fatalf("write tokenizer: %v", err)
	}

	trainPath := path[:len(path)-len(".mll")] + ".text-pairs.jsonl"
	evalPath := path[:len(path)-len(".mll")] + ".text-eval.jsonl"
	trainData := "" +
		"{\"query\":\"a\",\"positive\":\"a\"}\n" +
		"{\"query\":\"b\",\"positive\":\"b\"}\n"
	evalData := "" +
		"{\"query\":\"a\",\"document\":\"a\",\"label\":1}\n" +
		"{\"left\":\"a\",\"right\":\"b\",\"label\":0}\n"
	if err := os.WriteFile(trainPath, []byte(trainData), 0o644); err != nil {
		t.Fatalf("write train text dataset: %v", err)
	}
	if err := os.WriteFile(evalPath, []byte(evalData), 0o644); err != nil {
		t.Fatalf("write eval text dataset: %v", err)
	}

	summary, _, err := TrainEmbeddingPackageFromTextContrastiveFiles(path, tokenizerPath, trainPath, evalPath, EmbeddingTrainRunConfig{
		Epochs:      2,
		BatchSize:   2,
		Shuffle:     true,
		Seed:        7,
		RestoreBest: true,
	})
	if err != nil {
		t.Fatalf("train text package with labeled eval: %v", err)
	}
	if summary.FinalEval == nil {
		t.Fatal("expected final eval metrics in summary")
	}
	if summary.FinalEval.PairCount != 2 {
		t.Fatalf("final eval pair count = %d, want 2", summary.FinalEval.PairCount)
	}
	if summary.BestEpoch <= 0 || summary.BestStep <= 0 {
		t.Fatalf("best epoch/step = %d/%d, want both positive", summary.BestEpoch, summary.BestStep)
	}
}

func writeTinyTrainingPackage(t *testing.T, trainer *EmbeddingTrainer) string {
	t.Helper()
	path := t.TempDir() + "/tiny_train_embed_q8.mll"
	if _, err := trainer.WriteTrainingPackage(path); err != nil {
		t.Fatalf("write training package: %v", err)
	}
	return path
}
