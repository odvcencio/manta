package main

import (
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/odvcencio/manta/artifact/barr"
	"github.com/odvcencio/manta/compiler"
	barruntime "github.com/odvcencio/manta/runtime"
	mll "github.com/odvcencio/mll"
)

func TestFormatTrainThroughputIncludesExamplePairAndStepRates(t *testing.T) {
	summary := barruntime.EmbeddingTrainRunSummary{
		Workload: barruntime.EmbeddingTrainWorkload{
			ActualTotalExamples: 100,
			ActualTotalPairs:    10000,
			ActualTrainExamples: 80,
			ActualTrainPairs:    8000,
			ActualEvalExamples:  20,
			ActualEvalPairs:     2000,
		},
		Elapsed:       10 * time.Second,
		TrainDuration: 4 * time.Second,
		EvalDuration:  2 * time.Second,
		StepsRun:      8,
	}

	output := formatTrainThroughput(summary)
	for _, want := range []string{
		"elapsed=10s",
		"examples/s=10.00",
		"pairs/s=1000.00",
		"train_examples/s=20.00",
		"train_pairs/s=2000.00",
		"eval_examples/s=10.00",
		"eval_pairs/s=1000.00",
		"optimizer_steps/s=2.00",
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("throughput output missing %q\noutput:\n%s", want, output)
		}
	}
}

func TestRunInitTrainCreatesTrainingPackage(t *testing.T) {
	path := writeTrainableArtifact(t)
	if err := run([]string{"init-train", "--dim", "D=4", "--dim", "E=3", path}); err != nil {
		t.Fatalf("run init-train: %v", err)
	}
	for _, candidate := range []string{
		barruntime.DefaultWeightFilePath(path),
		barruntime.DefaultMemoryPlanPath(path),
		barruntime.DefaultEmbeddingTrainManifestPath(path),
		barruntime.DefaultEmbeddingCheckpointPath(path),
		barruntime.DefaultEmbeddingTrainProfilePath(path),
	} {
		if _, err := os.Stat(candidate); err != nil {
			t.Fatalf("expected package file %q: %v", candidate, err)
		}
	}
}

func TestRunInitTrainAppliesTrainingConfigWithDefaultManifest(t *testing.T) {
	path := writeTrainableArtifact(t)
	if err := run([]string{"init-train", "--dim", "D=4", "--dim", "E=3", "--lr", "0.0125", "--weight-decay", "0.001", "--contrastive-loss", "infonce", "--temperature", "0.05", path}); err != nil {
		t.Fatalf("run init-train: %v", err)
	}
	checkpoint, err := barruntime.ReadEmbeddingTrainCheckpointFile(barruntime.DefaultEmbeddingCheckpointPath(path))
	if err != nil {
		t.Fatalf("read checkpoint: %v", err)
	}
	if checkpoint.Config.LearningRate != 0.0125 {
		t.Fatalf("learning rate = %f, want 0.0125", checkpoint.Config.LearningRate)
	}
	if checkpoint.Config.WeightDecay != 0.001 {
		t.Fatalf("weight decay = %f, want 0.001", checkpoint.Config.WeightDecay)
	}
	if checkpoint.Config.ContrastiveLoss != "infonce" {
		t.Fatalf("contrastive loss = %q, want infonce", checkpoint.Config.ContrastiveLoss)
	}
	if checkpoint.Config.Temperature != 0.05 {
		t.Fatalf("temperature = %f, want 0.05", checkpoint.Config.Temperature)
	}
}

func TestRunInitModelCreatesDefaultEmbeddingTrainingPackage(t *testing.T) {
	path := filepath.Join(t.TempDir(), "manta-embed-v0.mll")
	if err := run([]string{
		"init-model",
		"--vocab-size", "16",
		"--max-seq", "8",
		"--embedding-dim", "4",
		"--hidden-dim", "8",
		"--seed", "7",
		path,
	}); err != nil {
		t.Fatalf("run init-model: %v", err)
	}
	manifest, err := barruntime.ReadEmbeddingManifestFile(barruntime.DefaultEmbeddingManifestPath(path))
	if err != nil {
		t.Fatalf("read manifest: %v", err)
	}
	if manifest.Name != "manta-embed-v0" {
		t.Fatalf("model name = %q, want manta-embed-v0", manifest.Name)
	}
	if manifest.EncoderRepeats != 2 {
		t.Fatalf("encoder repeats = %d, want 2", manifest.EncoderRepeats)
	}
	if manifest.Tokenizer.VocabSize != 16 || manifest.Tokenizer.MaxSequence != 8 {
		t.Fatalf("unexpected tokenizer contract: %+v", manifest.Tokenizer)
	}
	checkpoint, err := barruntime.ReadEmbeddingTrainCheckpointFile(barruntime.DefaultEmbeddingCheckpointPath(path))
	if err != nil {
		t.Fatalf("read checkpoint: %v", err)
	}
	if checkpoint.Config.ContrastiveLoss != "infonce" {
		t.Fatalf("contrastive loss = %q, want infonce", checkpoint.Config.ContrastiveLoss)
	}
	if _, err := barruntime.LoadEmbeddingTrainerPackage(path); err != nil {
		t.Fatalf("reload initialized model package: %v", err)
	}
}

func TestRunInitModelTrainCorpusExportFlow(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "manta-embed-v0.mll")
	if err := run([]string{
		"init-model",
		"--vocab-size", "16",
		"--max-seq", "8",
		"--embedding-dim", "4",
		"--hidden-dim", "8",
		"--seed", "7",
		path,
	}); err != nil {
		t.Fatalf("run init-model: %v", err)
	}
	corpusPath := filepath.Join(dir, "corpus.txt")
	corpus := "" +
		"ab ab cd. cd ab cd.\n" +
		"cd cd ab. ab cd ab.\n" +
		"ab cd ef. ef cd ab.\n" +
		"ef ef ab. ab ef ef.\n"
	if err := os.WriteFile(corpusPath, []byte(corpus), 0o644); err != nil {
		t.Fatalf("write corpus: %v", err)
	}
	if err := run([]string{"train-corpus", "--vocab-size", "16", "--min-freq", "1", "--epochs", "2", "--batch-size", "2", "--min-chars", "2", "--eval-pairs", "2", path, corpusPath}); err != nil {
		t.Fatalf("run train-corpus: %v", err)
	}
	if _, err := barruntime.LoadEmbeddingTrainerPackage(path); err != nil {
		t.Fatalf("reload trained default package: %v", err)
	}
	if err := run([]string{"export-mll", path}); err != nil {
		t.Fatalf("run export-mll: %v", err)
	}
	sealedPath := barruntime.DefaultMLLPath(path)
	if sealedPath == path {
		t.Fatalf("sealed export path reused artifact path %q", path)
	}
	if _, err := mll.ReadFile(sealedPath, mll.WithDigestVerification()); err != nil {
		t.Fatalf("read sealed default model MLL: %v", err)
	}
	sealedInspect := captureRunOutput(t, []string{"inspect", sealedPath})
	for _, want := range []string{
		"embedding manifest: embedded",
		"package: embedded sealed MLL",
		"package verify: OK",
		"embedding model: manta-embed-v0",
	} {
		if !strings.Contains(sealedInspect, want) {
			t.Fatalf("sealed inspect output missing %q\noutput:\n%s", want, sealedInspect)
		}
	}
	if err := run([]string{"inspect", path}); err != nil {
		t.Fatalf("inspect trained default package after export: %v", err)
	}
}

func TestRunTrainEmbedFitsContrastivePackage(t *testing.T) {
	path := writeTrainableArtifact(t)
	if err := run([]string{"init-train", "--dim", "D=4", "--dim", "E=3", path}); err != nil {
		t.Fatalf("run init-train: %v", err)
	}
	trainPath := filepath.Join(t.TempDir(), "train.jsonl")
	evalPath := filepath.Join(t.TempDir(), "eval.jsonl")
	examples := []barruntime.EmbeddingContrastiveExample{
		{QueryTokens: []int32{1, 2}, PositiveTokens: []int32{1, 2}},
		{QueryTokens: []int32{2, 3}, PositiveTokens: []int32{2, 3}},
		{QueryTokens: []int32{3, 4}, PositiveTokens: []int32{3, 4}},
		{QueryTokens: []int32{4, 5}, PositiveTokens: []int32{4, 5}},
	}
	if err := barruntime.WriteEmbeddingContrastiveExamplesFile(trainPath, examples); err != nil {
		t.Fatalf("write train dataset: %v", err)
	}
	if err := barruntime.WriteEmbeddingContrastiveExamplesFile(evalPath, examples); err != nil {
		t.Fatalf("write eval dataset: %v", err)
	}
	if err := run([]string{"train-embed", "--epochs", "2", "--batch-size", "2", "--lr", "0.003", "--contrastive-loss", "infonce", "--temperature", "0.07", path, trainPath, evalPath}); err != nil {
		t.Fatalf("run train-embed: %v", err)
	}
	if _, err := barruntime.LoadEmbeddingTrainerPackage(path); err != nil {
		t.Fatalf("reload trained package: %v", err)
	}
	checkpoint, err := barruntime.ReadEmbeddingTrainCheckpointFile(barruntime.DefaultEmbeddingCheckpointPath(path))
	if err != nil {
		t.Fatalf("read checkpoint: %v", err)
	}
	if checkpoint.Config.LearningRate < 0.00299 || checkpoint.Config.LearningRate > 0.00301 {
		t.Fatalf("learning rate = %f, want 0.003", checkpoint.Config.LearningRate)
	}
	if checkpoint.Config.ContrastiveLoss != "infonce" {
		t.Fatalf("contrastive loss = %q, want infonce", checkpoint.Config.ContrastiveLoss)
	}
	if checkpoint.Config.Temperature < 0.06999 || checkpoint.Config.Temperature > 0.07001 {
		t.Fatalf("temperature = %f, want 0.07", checkpoint.Config.Temperature)
	}
}

func TestRunRenameEmbedRewritesPackageIdentity(t *testing.T) {
	path := writeTrainableArtifact(t)
	if err := run([]string{"init-train", "--dim", "D=4", "--dim", "E=3", path}); err != nil {
		t.Fatalf("run init-train: %v", err)
	}
	tokenizer := barruntime.TokenizerFile{
		Version:      barruntime.TokenizerFileVersion,
		Tokens:       []string{"<pad>", "<unk>", "alpha", "beta"},
		PadToken:     "<pad>",
		UnknownToken: "<unk>",
	}
	if err := tokenizer.WriteFile(barruntime.DefaultTokenizerPath(path)); err != nil {
		t.Fatalf("write tokenizer: %v", err)
	}
	renamedPath := filepath.Join(t.TempDir(), "manta-embed-v0.mll")

	if err := run([]string{"rename-embed", "--name", "manta-embed-v0", path, renamedPath}); err != nil {
		t.Fatalf("run rename-embed: %v", err)
	}

	mod, err := barr.ReadFile(renamedPath)
	if err != nil {
		t.Fatalf("read renamed artifact: %v", err)
	}
	if mod.Name != "manta-embed-v0" {
		t.Fatalf("module name = %q, want manta-embed-v0", mod.Name)
	}
	manifest, err := barruntime.ReadEmbeddingManifestFile(barruntime.DefaultEmbeddingManifestPath(renamedPath))
	if err != nil {
		t.Fatalf("read renamed manifest: %v", err)
	}
	if manifest.Name != "manta-embed-v0" {
		t.Fatalf("manifest name = %q, want manta-embed-v0", manifest.Name)
	}
	checkpoint, err := barruntime.ReadEmbeddingTrainCheckpointFile(barruntime.DefaultEmbeddingCheckpointPath(renamedPath))
	if err != nil {
		t.Fatalf("read renamed checkpoint: %v", err)
	}
	if checkpoint.Manifest.Name != "manta-embed-v0" {
		t.Fatalf("checkpoint manifest name = %q, want manta-embed-v0", checkpoint.Manifest.Name)
	}
	if _, err := os.Stat(barruntime.DefaultTokenizerPath(renamedPath)); err != nil {
		t.Fatalf("renamed tokenizer sidecar missing: %v", err)
	}
	if _, err := barruntime.LoadEmbeddingTrainerPackage(renamedPath); err != nil {
		t.Fatalf("reload renamed package: %v", err)
	}
}

func TestRunTrainEmbedPlanOnlyShowsWorkload(t *testing.T) {
	path := writeTrainableArtifact(t)
	if err := run([]string{"init-train", "--dim", "D=4", "--dim", "E=3", path}); err != nil {
		t.Fatalf("run init-train: %v", err)
	}
	trainPath := filepath.Join(t.TempDir(), "train.jsonl")
	evalPath := filepath.Join(t.TempDir(), "eval.jsonl")
	examples := []barruntime.EmbeddingContrastiveExample{
		{QueryTokens: []int32{1, 2}, PositiveTokens: []int32{1, 2}},
		{QueryTokens: []int32{2, 3}, PositiveTokens: []int32{2, 3}},
		{QueryTokens: []int32{3, 4}, PositiveTokens: []int32{3, 4}},
		{QueryTokens: []int32{4, 5}, PositiveTokens: []int32{4, 5}},
	}
	if err := barruntime.WriteEmbeddingContrastiveExamplesFile(trainPath, examples); err != nil {
		t.Fatalf("write train dataset: %v", err)
	}
	if err := barruntime.WriteEmbeddingContrastiveExamplesFile(evalPath, examples); err != nil {
		t.Fatalf("write eval dataset: %v", err)
	}
	output := captureRunOutput(t, []string{"train-embed", "--plan-only", "--epochs", "2", "--batch-size", "2", path, trainPath, evalPath})
	for _, want := range []string{
		"planned workload:",
		"train=4 contrastive examples",
		"steps/epoch=2",
		"train_pairs/epoch=8",
		"eval=4 contrastive examples",
		"eval_pairs/pass=16",
		"pairs(planned=80 actual=0)",
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("plan-only output missing %q\noutput:\n%s", want, output)
		}
	}
}

func TestRunTrainEmbedFitsTextContrastivePackage(t *testing.T) {
	path := writeTrainableArtifact(t)
	if err := run([]string{"init-train", "--dim", "D=4", "--dim", "E=3", path}); err != nil {
		t.Fatalf("run init-train: %v", err)
	}
	tokenizer := barruntime.TokenizerFile{
		Version: barruntime.TokenizerFileVersion,
		Tokens:  []string{"[PAD]", "[UNK]", "[CLS]", "[SEP]", "a", "b", "c", "d"},
	}
	if err := tokenizer.WriteFile(barruntime.DefaultTokenizerPath(path)); err != nil {
		t.Fatalf("write tokenizer: %v", err)
	}
	trainPath := filepath.Join(t.TempDir(), "train-text.jsonl")
	evalPath := filepath.Join(t.TempDir(), "eval-text.jsonl")
	examples := []barruntime.EmbeddingTextContrastiveExample{
		{Query: "ab", Positive: "ab"},
		{Query: "cd", Positive: "cd"},
		{Query: "ab", Positive: "ab"},
		{Query: "cd", Positive: "cd"},
	}
	if err := barruntime.WriteEmbeddingTextContrastiveExamplesFile(trainPath, examples); err != nil {
		t.Fatalf("write train text dataset: %v", err)
	}
	if err := barruntime.WriteEmbeddingTextContrastiveExamplesFile(evalPath, examples); err != nil {
		t.Fatalf("write eval text dataset: %v", err)
	}
	if err := run([]string{"train-embed", "--epochs", "2", "--batch-size", "2", path, trainPath, evalPath}); err != nil {
		t.Fatalf("run train-embed text: %v", err)
	}
	if _, err := barruntime.LoadEmbeddingTrainerPackage(path); err != nil {
		t.Fatalf("reload trained package: %v", err)
	}
}

func TestRunTrainEmbedFitsTextContrastivePackageWithLabeledEvalPairs(t *testing.T) {
	path := writeTrainableArtifact(t)
	if err := run([]string{"init-train", "--dim", "D=4", "--dim", "E=3", path}); err != nil {
		t.Fatalf("run init-train: %v", err)
	}
	tokenizer := barruntime.TokenizerFile{
		Version: barruntime.TokenizerFileVersion,
		Tokens:  []string{"[PAD]", "[UNK]", "[CLS]", "[SEP]", "a", "b", "c", "d"},
	}
	if err := tokenizer.WriteFile(barruntime.DefaultTokenizerPath(path)); err != nil {
		t.Fatalf("write tokenizer: %v", err)
	}
	trainPath := filepath.Join(t.TempDir(), "train-text.jsonl")
	evalPath := filepath.Join(t.TempDir(), "eval-text.jsonl")
	trainData := "" +
		"{\"query\":\"ab\",\"positive\":\"ab\"}\n" +
		"{\"query\":\"cd\",\"positive\":\"cd\"}\n"
	evalData := "" +
		"{\"query\":\"ab\",\"document\":\"ab\",\"label\":1}\n" +
		"{\"left\":\"ab\",\"right\":\"cd\",\"label\":0}\n"
	if err := os.WriteFile(trainPath, []byte(trainData), 0o644); err != nil {
		t.Fatalf("write train text dataset: %v", err)
	}
	if err := os.WriteFile(evalPath, []byte(evalData), 0o644); err != nil {
		t.Fatalf("write eval text dataset: %v", err)
	}
	if err := run([]string{"train-embed", "--epochs", "2", "--batch-size", "2", path, trainPath, evalPath}); err != nil {
		t.Fatalf("run train-embed text with labeled eval: %v", err)
	}
	if _, err := barruntime.LoadEmbeddingTrainerPackage(path); err != nil {
		t.Fatalf("reload trained package: %v", err)
	}
}

func TestRunInspectReadsPackageManifest(t *testing.T) {
	path := writeTrainableArtifact(t)
	if err := run([]string{"init-train", "--dim", "D=4", "--dim", "E=3", path}); err != nil {
		t.Fatalf("run init-train: %v", err)
	}
	if err := run([]string{"inspect", path}); err != nil {
		t.Fatalf("run inspect: %v", err)
	}
}

func TestRunInspectShowsRepeatedEncoderEmbeddingDetails(t *testing.T) {
	dir := t.TempDir()
	srcPath := copyExampleFile(t, dir, "encoder_trainable_q8x2.bar")
	artifactPath := filepath.Join(dir, "encoder_trainable_q8x2.mll")
	copyExampleFile(t, dir, "encoder_trainable_q8x2.embedding.mll")
	if err := run([]string{"compile", srcPath, artifactPath}); err != nil {
		t.Fatalf("run compile: %v", err)
	}
	output := captureRunOutput(t, []string{"inspect", artifactPath})
	for _, want := range []string{
		"embedding manifest:",
		"embedding model: encoder-trainable-q8x2 pooled=embed_pooled batch=embed_pooled_batch output=result/f16",
		"encoder repeats: 2",
		"tokenizer: vocab=32768 max_sequence=256",
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("inspect output missing %q\noutput:\n%s", want, output)
		}
	}
}

func TestRunExportMLLWritesContainer(t *testing.T) {
	path := writeTrainableArtifact(t)
	if err := run([]string{"init-train", "--dim", "D=4", "--dim", "E=3", path}); err != nil {
		t.Fatalf("run init-train: %v", err)
	}
	if err := run([]string{"export-mll", path}); err != nil {
		t.Fatalf("run export-mll: %v", err)
	}
	outPath := barruntime.DefaultMLLPath(path)
	if _, err := os.Stat(outPath); err != nil {
		t.Fatalf("expected MLL file %q: %v", outPath, err)
	}
	if _, err := mll.ReadFile(outPath, mll.WithDigestVerification()); err != nil {
		t.Fatalf("read exported MLL: %v", err)
	}
}

func TestRunTrainTokenizerWritesSiblingTokenizer(t *testing.T) {
	path := writeTrainableArtifact(t)
	corpusPath := filepath.Join(t.TempDir(), "corpus.txt")
	if err := os.WriteFile(corpusPath, []byte("ab ab cd ab cd\n"), 0o644); err != nil {
		t.Fatalf("write corpus: %v", err)
	}
	if err := run([]string{"train-tokenizer", "--vocab-size", "12", path, corpusPath}); err != nil {
		t.Fatalf("run train-tokenizer: %v", err)
	}
	tokenizerPath := barruntime.DefaultTokenizerPath(path)
	if _, err := os.Stat(tokenizerPath); err != nil {
		t.Fatalf("expected tokenizer file %q: %v", tokenizerPath, err)
	}
	if _, err := barruntime.ReadTokenizerFile(tokenizerPath); err != nil {
		t.Fatalf("read tokenizer file: %v", err)
	}
	manifest, err := barruntime.ReadEmbeddingManifestFile(barruntime.DefaultEmbeddingManifestPath(path))
	if err != nil {
		t.Fatalf("read manifest: %v", err)
	}
	if manifest.Tokenizer.VocabSize != 11 {
		t.Fatalf("expected manifest vocab size to track tokenizer, got %d", manifest.Tokenizer.VocabSize)
	}
}

func TestRunMineTextPairsWritesTrainAndEvalFiles(t *testing.T) {
	corpusPath := filepath.Join(t.TempDir(), "corpus.txt")
	corpus := "" +
		"alpha beta gamma. gamma delta epsilon.\n" +
		"delta epsilon zeta. eta theta iota.\n" +
		"kappa lambda mu. nu xi omicron.\n" +
		"pi rho sigma. tau upsilon phi.\n"
	if err := os.WriteFile(corpusPath, []byte(corpus), 0o644); err != nil {
		t.Fatalf("write corpus: %v", err)
	}
	trainPath := filepath.Join(t.TempDir(), "train.jsonl")
	evalPath := filepath.Join(t.TempDir(), "eval.jsonl")
	if err := run([]string{"mine-text-pairs", "--min-chars", "5", "--eval-pairs", "2", corpusPath, trainPath, evalPath}); err != nil {
		t.Fatalf("run mine-text-pairs: %v", err)
	}
	if _, err := barruntime.ReadEmbeddingTextContrastiveExamplesFile(trainPath); err != nil {
		t.Fatalf("read mined train pairs: %v", err)
	}
	evalSet, err := barruntime.ReadEmbeddingTextPairExamplesFile(evalPath)
	if err != nil {
		t.Fatalf("read mined eval pairs: %v", err)
	}
	var positives, negatives int
	for _, example := range evalSet {
		if example.Target > 0 {
			positives++
		} else if example.Target == 0 {
			negatives++
		}
	}
	if positives == 0 || negatives == 0 {
		t.Fatalf("expected mined eval set to include both classes, got positives=%d negatives=%d", positives, negatives)
	}
}

func TestRunMineTextPairsThenTrainEmbedFlow(t *testing.T) {
	path := writeTrainableArtifact(t)
	if err := run([]string{"init-train", "--dim", "D=4", "--dim", "E=3", path}); err != nil {
		t.Fatalf("run init-train: %v", err)
	}
	corpusPath := filepath.Join(t.TempDir(), "corpus.txt")
	corpus := "" +
		"ab ab cd. cd ab cd.\n" +
		"cd cd ab. ab cd ab.\n" +
		"ab cd ef. ef cd ab.\n" +
		"ef ef ab. ab ef ef.\n"
	if err := os.WriteFile(corpusPath, []byte(corpus), 0o644); err != nil {
		t.Fatalf("write corpus: %v", err)
	}
	if err := run([]string{"train-tokenizer", "--vocab-size", "16", path, corpusPath}); err != nil {
		t.Fatalf("run train-tokenizer: %v", err)
	}
	trainPath := filepath.Join(t.TempDir(), "train.jsonl")
	evalPath := filepath.Join(t.TempDir(), "eval.jsonl")
	if err := run([]string{"mine-text-pairs", "--min-chars", "2", "--eval-pairs", "2", corpusPath, trainPath, evalPath}); err != nil {
		t.Fatalf("run mine-text-pairs: %v", err)
	}
	if err := run([]string{"train-embed", "--epochs", "2", "--batch-size", "2", path, trainPath, evalPath}); err != nil {
		t.Fatalf("run train-embed from mined text: %v", err)
	}
	if _, err := barruntime.LoadEmbeddingTrainerPackage(path); err != nil {
		t.Fatalf("reload trained package: %v", err)
	}
}

func TestRunTrainCorpusFlow(t *testing.T) {
	path := writeTrainableArtifact(t)
	if err := run([]string{"init-train", "--dim", "D=4", "--dim", "E=3", path}); err != nil {
		t.Fatalf("run init-train: %v", err)
	}
	corpusPath := filepath.Join(t.TempDir(), "corpus.txt")
	corpus := "" +
		"alpha beta gamma. gamma delta epsilon.\n" +
		"delta epsilon zeta. eta theta iota.\n" +
		"kappa lambda mu. nu xi omicron.\n" +
		"pi rho sigma. tau upsilon phi.\n"
	if err := os.WriteFile(corpusPath, []byte(corpus), 0o644); err != nil {
		t.Fatalf("write corpus: %v", err)
	}
	if err := run([]string{"train-corpus", "--vocab-size", "20", "--min-freq", "1", "--epochs", "2", "--batch-size", "2", "--min-chars", "5", "--eval-pairs", "2", path, corpusPath}); err != nil {
		t.Fatalf("run train-corpus: %v", err)
	}
	if _, err := os.Stat(barruntime.DefaultTokenizerPath(path)); err != nil {
		t.Fatalf("expected tokenizer file: %v", err)
	}
	if _, err := os.Stat(barruntime.DefaultMinedTrainPairsPath(path)); err != nil {
		t.Fatalf("expected mined train pairs: %v", err)
	}
	if _, err := os.Stat(barruntime.DefaultMinedEvalPairsPath(path)); err != nil {
		t.Fatalf("expected mined eval pairs: %v", err)
	}
	if _, err := barruntime.LoadEmbeddingTrainerPackage(path); err != nil {
		t.Fatalf("reload trained package: %v", err)
	}
}

func TestRunTrainCorpusRepeatedEncoderExampleFlow(t *testing.T) {
	dir := t.TempDir()
	srcPath := copyExampleFile(t, dir, "encoder_trainable_q8x2.bar")
	artifactPath := filepath.Join(dir, "encoder_trainable_q8x2.mll")
	copyExampleFile(t, dir, "encoder_trainable_q8x2.embedding.mll")

	if err := run([]string{"compile", srcPath, artifactPath}); err != nil {
		t.Fatalf("run compile: %v", err)
	}
	if err := run([]string{"init-train", "--dim", "D=16", "--dim", "H=32", artifactPath}); err != nil {
		t.Fatalf("run init-train: %v", err)
	}
	corpusPath := filepath.Join(dir, "corpus.txt")
	corpus := "" +
		"Manta trains and serves compact transformer encoders.\n" +
		"CorkScrewDB needs a small default model with strong retrieval quality.\n" +
		"Quantized embeddings should be fast, portable, and cheap to ship.\n" +
		"Native CUDA training should reuse weights, activations, and optimizer state.\n" +
		"Metal parity matters later, but the package path must already be clean.\n" +
		"Attention, residuals, and layernorm make the encoder more realistic.\n"
	if err := os.WriteFile(corpusPath, []byte(corpus), 0o644); err != nil {
		t.Fatalf("write corpus: %v", err)
	}
	if err := run([]string{"train-corpus", "--vocab-size", "48", "--min-freq", "1", "--epochs", "2", "--batch-size", "2", "--min-chars", "12", "--eval-pairs", "3", artifactPath, corpusPath}); err != nil {
		t.Fatalf("run train-corpus: %v", err)
	}
	if err := run([]string{"inspect", artifactPath}); err != nil {
		t.Fatalf("run inspect: %v", err)
	}
	manifest, err := barruntime.ReadEmbeddingManifestFile(barruntime.DefaultEmbeddingManifestPath(artifactPath))
	if err != nil {
		t.Fatalf("read manifest: %v", err)
	}
	if manifest.EncoderRepeats != 2 {
		t.Fatalf("encoder repeats = %d, want 2", manifest.EncoderRepeats)
	}
	profile, err := barruntime.ReadEmbeddingTrainProfileFile(barruntime.DefaultEmbeddingTrainProfilePath(artifactPath))
	if err != nil {
		t.Fatalf("read training profile: %v", err)
	}
	if profile.Step == 0 {
		t.Fatal("expected non-zero training profile step")
	}
	if profile.ForwardBackend != "" && profile.ForwardResidency.MatMul.BindCalls == 0 {
		t.Fatal("expected matmul bind activity in repeated encoder train profile")
	}
	for _, candidate := range []string{
		barruntime.DefaultTokenizerPath(artifactPath),
		barruntime.DefaultMinedTrainPairsPath(artifactPath),
		barruntime.DefaultMinedEvalPairsPath(artifactPath),
	} {
		if _, err := os.Stat(candidate); err != nil {
			t.Fatalf("expected generated training artifact %q: %v", candidate, err)
		}
	}
	if _, err := barruntime.LoadEmbeddingTrainerPackage(artifactPath); err != nil {
		t.Fatalf("reload trained repeated-encoder package: %v", err)
	}
}

func TestRunCompileDefaultMLLThenTrainFlow(t *testing.T) {
	dir := t.TempDir()
	srcPath := filepath.Join(dir, "tiny_train_embed.bar")
	source := []byte(`
param token_embedding: q8[V, D] @weight("weights/token_embedding") @trainable
param projection: q8[D, E] @weight("weights/projection") @trainable

pipeline embed_pooled(tokens: i32[T]) -> q8[E] {
    let embeddings = gather(token_embedding, tokens)
    let projected = @matmul(embeddings, projection)
    return mean_pool(projected)
}

pipeline embed_pooled_batch(tokens: i32[B, T]) -> q8[B, E] {
    let embeddings = gather(token_embedding, tokens)
    let projected = @matmul(embeddings, projection)
    return mean_pool(projected)
}
`)
	if err := os.WriteFile(srcPath, source, 0o644); err != nil {
		t.Fatalf("write source: %v", err)
	}
	if err := run([]string{"compile", srcPath}); err != nil {
		t.Fatalf("run compile: %v", err)
	}
	artifactPath := filepath.Join(dir, "tiny_train_embed.mll")
	if _, err := os.Stat(artifactPath); err != nil {
		t.Fatalf("expected compiled .mll artifact %q: %v", artifactPath, err)
	}
	manifest := barruntime.EmbeddingManifest{
		Name:                "tiny-train-embed",
		PooledEntry:         "embed_pooled",
		BatchEntry:          "embed_pooled_batch",
		TokenInput:          "tokens",
		OutputName:          "result",
		OutputDType:         "q8",
		TokenEmbeddingParam: "token_embedding",
		ProjectionParam:     "projection",
		Tokenizer: barruntime.TokenizerManifest{
			VocabSize:   8,
			MaxSequence: 8,
			PadID:       0,
		},
	}
	if err := manifest.WriteFile(barruntime.DefaultEmbeddingManifestPath(artifactPath)); err != nil {
		t.Fatalf("write manifest: %v", err)
	}
	if err := run([]string{"init-train", "--dim", "D=4", "--dim", "E=3", artifactPath}); err != nil {
		t.Fatalf("run init-train: %v", err)
	}
	trainPath := filepath.Join(dir, "train.jsonl")
	evalPath := filepath.Join(dir, "eval.jsonl")
	examples := []barruntime.EmbeddingContrastiveExample{
		{QueryTokens: []int32{1, 2}, PositiveTokens: []int32{1, 2}},
		{QueryTokens: []int32{2, 3}, PositiveTokens: []int32{2, 3}},
		{QueryTokens: []int32{3, 4}, PositiveTokens: []int32{3, 4}},
		{QueryTokens: []int32{4, 5}, PositiveTokens: []int32{4, 5}},
	}
	if err := barruntime.WriteEmbeddingContrastiveExamplesFile(trainPath, examples); err != nil {
		t.Fatalf("write train dataset: %v", err)
	}
	if err := barruntime.WriteEmbeddingContrastiveExamplesFile(evalPath, examples); err != nil {
		t.Fatalf("write eval dataset: %v", err)
	}
	if err := run([]string{"train-embed", "--epochs", "2", "--batch-size", "2", artifactPath, trainPath, evalPath}); err != nil {
		t.Fatalf("run train-embed: %v", err)
	}
	if _, err := barruntime.LoadEmbeddingTrainerPackage(artifactPath); err != nil {
		t.Fatalf("reload trained package: %v", err)
	}
}

func writeTrainableArtifact(t *testing.T) string {
	t.Helper()
	source := []byte(`
param token_embedding: q8[V, D] @weight("weights/token_embedding") @trainable
param projection: q8[D, E] @weight("weights/projection") @trainable

pipeline embed_pooled(tokens: i32[T]) -> q8[E] {
    let embeddings = gather(token_embedding, tokens)
    let projected = @matmul(embeddings, projection)
    return mean_pool(projected)
}

pipeline embed_pooled_batch(tokens: i32[B, T]) -> q8[B, E] {
    let embeddings = gather(token_embedding, tokens)
    let projected = @matmul(embeddings, projection)
    return mean_pool(projected)
}
`)
	bundle, err := compiler.Build(source, compiler.Options{ModuleName: "tiny_train_embed"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	path := filepath.Join(t.TempDir(), "tiny_train_embed.barr")
	if err := barr.WriteFile(path, bundle.Artifact); err != nil {
		t.Fatalf("write artifact: %v", err)
	}
	manifest := barruntime.EmbeddingManifest{
		Name:                "tiny-train-embed",
		PooledEntry:         "embed_pooled",
		BatchEntry:          "embed_pooled_batch",
		TokenInput:          "tokens",
		OutputName:          "result",
		OutputDType:         "q8",
		TokenEmbeddingParam: "token_embedding",
		ProjectionParam:     "projection",
		Tokenizer: barruntime.TokenizerManifest{
			VocabSize:   8,
			MaxSequence: 8,
			PadID:       0,
		},
	}
	if err := manifest.WriteFile(barruntime.DefaultEmbeddingManifestPath(path)); err != nil {
		t.Fatalf("write manifest: %v", err)
	}
	return path
}

func copyExampleFile(t *testing.T, dir, name string) string {
	t.Helper()
	srcPath := filepath.Join("..", "..", "examples", name)
	data, err := os.ReadFile(srcPath)
	if err != nil {
		t.Fatalf("read example file %q: %v", srcPath, err)
	}
	dstPath := filepath.Join(dir, name)
	if err := os.WriteFile(dstPath, data, 0o644); err != nil {
		t.Fatalf("write example file %q: %v", dstPath, err)
	}
	return dstPath
}

func captureRunOutput(t *testing.T, args []string) string {
	t.Helper()
	origStdout := os.Stdout
	reader, writer, err := os.Pipe()
	if err != nil {
		t.Fatalf("create stdout pipe: %v", err)
	}
	os.Stdout = writer
	defer func() {
		os.Stdout = origStdout
	}()
	runErr := run(args)
	if err := writer.Close(); err != nil {
		t.Fatalf("close stdout writer: %v", err)
	}
	data, err := io.ReadAll(reader)
	if err != nil {
		t.Fatalf("read stdout capture: %v", err)
	}
	if err := reader.Close(); err != nil {
		t.Fatalf("close stdout reader: %v", err)
	}
	if runErr != nil {
		t.Fatalf("run %v: %v\noutput:\n%s", args, runErr, string(data))
	}
	return string(data)
}
