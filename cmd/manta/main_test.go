package main

import (
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	mantaartifact "github.com/odvcencio/manta/artifact/manta"
	"github.com/odvcencio/manta/compiler"
	mantaruntime "github.com/odvcencio/manta/runtime"
	mll "github.com/odvcencio/mll"
)

func TestFormatTrainThroughputIncludesExamplePairAndStepRates(t *testing.T) {
	summary := mantaruntime.EmbeddingTrainRunSummary{
		Workload: mantaruntime.EmbeddingTrainWorkload{
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
		mantaruntime.DefaultWeightFilePath(path),
		mantaruntime.DefaultMemoryPlanPath(path),
		mantaruntime.DefaultEmbeddingTrainManifestPath(path),
		mantaruntime.DefaultEmbeddingCheckpointPath(path),
		mantaruntime.DefaultEmbeddingTrainProfilePath(path),
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
	checkpoint, err := mantaruntime.ReadEmbeddingTrainCheckpointFile(mantaruntime.DefaultEmbeddingCheckpointPath(path))
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
	path := filepath.Join(t.TempDir(), "manta-embed-v1.mll")
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
	manifest, err := mantaruntime.ReadEmbeddingManifestFile(mantaruntime.DefaultEmbeddingManifestPath(path))
	if err != nil {
		t.Fatalf("read manifest: %v", err)
	}
	if manifest.Name != "manta-embed-v1" {
		t.Fatalf("model name = %q, want manta-embed-v1", manifest.Name)
	}
	if manifest.EncoderRepeats != 2 {
		t.Fatalf("encoder repeats = %d, want 2", manifest.EncoderRepeats)
	}
	if manifest.Tokenizer.VocabSize != 16 || manifest.Tokenizer.MaxSequence != 8 {
		t.Fatalf("unexpected tokenizer contract: %+v", manifest.Tokenizer)
	}
	checkpoint, err := mantaruntime.ReadEmbeddingTrainCheckpointFile(mantaruntime.DefaultEmbeddingCheckpointPath(path))
	if err != nil {
		t.Fatalf("read checkpoint: %v", err)
	}
	if checkpoint.Config.ContrastiveLoss != "infonce" {
		t.Fatalf("contrastive loss = %q, want infonce", checkpoint.Config.ContrastiveLoss)
	}
	if _, err := mantaruntime.LoadEmbeddingTrainerPackage(path); err != nil {
		t.Fatalf("reload initialized model package: %v", err)
	}
}

func TestRunInitMirageCreatesArtifact(t *testing.T) {
	path := filepath.Join(t.TempDir(), "nested", "mirage-v1.mll")
	output := captureRunOutput(t, []string{
		"init-mirage",
		"--height", "16",
		"--width", "16",
		"--latent-channels", "8",
		"--bits", "2",
		path,
	})
	for _, want := range []string{
		"initialized Mirage Image v1 module",
		"capabilities: image_ops, turboquant, training_losses, host_fallback",
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("init-mirage output missing %q\noutput:\n%s", want, output)
		}
	}
	mod, err := mantaartifact.ReadFile(path)
	if err != nil {
		t.Fatalf("read artifact: %v", err)
	}
	if mod.Name != "mirage_image_v1" || len(mod.EntryPoints) != 4 {
		t.Fatalf("unexpected Mirage artifact: %+v", mod)
	}
}

func TestRunInitModelTrainCorpusExportFlow(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "manta-embed-v1.mll")
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
	if _, err := mantaruntime.LoadEmbeddingTrainerPackage(path); err != nil {
		t.Fatalf("reload trained default package: %v", err)
	}
	if err := run([]string{"export-mll", path}); err != nil {
		t.Fatalf("run export-mll: %v", err)
	}
	sealedPath := mantaruntime.DefaultMLLPath(path)
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
		"embedding model: manta-embed-v1",
	} {
		if !strings.Contains(sealedInspect, want) {
			t.Fatalf("sealed inspect output missing %q\noutput:\n%s", want, sealedInspect)
		}
	}
	if err := run([]string{"inspect", path}); err != nil {
		t.Fatalf("inspect trained default package after export: %v", err)
	}
}

func TestRunEmbedTextLoadsSealedMLLTokenizer(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "manta-embed-v1.mll")
	if err := run([]string{
		"init-model",
		"--vocab-size", "8",
		"--max-seq", "8",
		"--embedding-dim", "4",
		"--hidden-dim", "8",
		path,
	}); err != nil {
		t.Fatalf("run init-model: %v", err)
	}
	tokenizer := mantaruntime.TokenizerFile{
		Version:      mantaruntime.TokenizerFileVersion,
		Tokens:       []string{"[PAD]", "[UNK]", "a"},
		UnknownToken: "[UNK]",
	}
	if err := tokenizer.WriteFile(mantaruntime.DefaultTokenizerPath(path)); err != nil {
		t.Fatalf("write tokenizer: %v", err)
	}
	if _, _, err := mantaruntime.RebuildSiblingPackageManifest(path); err != nil {
		t.Fatalf("rebuild package manifest: %v", err)
	}
	sealedPath := filepath.Join(dir, "manta-embed-v1.sealed.mll")
	if err := run([]string{"export-mll", path, sealedPath}); err != nil {
		t.Fatalf("run export-mll: %v", err)
	}

	output := captureRunOutput(t, []string{"embed-text", sealedPath, "a"})
	for _, want := range []string{
		"loaded embedding \"manta-embed-v1\"",
		"tokens: 1",
		"output: result",
		"embedding: f16[4]",
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("embed-text output missing %q\noutput:\n%s", want, output)
		}
	}
}

func TestRunEvalRetrievalWritesMetricsJSON(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "manta-embed-v1.mll")
	if err := run([]string{
		"init-model",
		"--vocab-size", "8",
		"--max-seq", "8",
		"--embedding-dim", "4",
		"--hidden-dim", "8",
		path,
	}); err != nil {
		t.Fatalf("run init-model: %v", err)
	}
	tokenizer := mantaruntime.TokenizerFile{
		Version:      mantaruntime.TokenizerFileVersion,
		Tokens:       []string{"[PAD]", "[UNK]", "a", "b"},
		UnknownToken: "[UNK]",
	}
	if err := tokenizer.WriteFile(mantaruntime.DefaultTokenizerPath(path)); err != nil {
		t.Fatalf("write tokenizer: %v", err)
	}
	if _, _, err := mantaruntime.RebuildSiblingPackageManifest(path); err != nil {
		t.Fatalf("rebuild package manifest: %v", err)
	}
	sealedPath := filepath.Join(dir, "manta-embed-v1.sealed.mll")
	if err := run([]string{"export-mll", path, sealedPath}); err != nil {
		t.Fatalf("run export-mll: %v", err)
	}
	datasetDir := filepath.Join(dir, "dataset")
	if err := os.MkdirAll(filepath.Join(datasetDir, "qrels"), 0o755); err != nil {
		t.Fatalf("mkdir dataset: %v", err)
	}
	if err := os.WriteFile(filepath.Join(datasetDir, "corpus.jsonl"), []byte(
		`{"_id":"d1","text":"a"}`+"\n"+
			`{"_id":"d2","text":"b"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write corpus: %v", err)
	}
	if err := os.WriteFile(filepath.Join(datasetDir, "queries.jsonl"), []byte(`{"_id":"q1","text":"a"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write queries: %v", err)
	}
	if err := os.WriteFile(filepath.Join(datasetDir, "qrels", "test.tsv"), []byte("query-id\tcorpus-id\tscore\nq1\td1\t1\n"), 0o644); err != nil {
		t.Fatalf("write qrels: %v", err)
	}
	metricsPath := filepath.Join(dir, "retrieval.metrics.json")

	output := captureRunOutput(t, []string{"eval-retrieval", "--dataset", "tiny", "--batch-size", "2", "--metrics-json", metricsPath, sealedPath, datasetDir})
	for _, want := range []string{
		"retrieval eval: dataset=tiny",
		"quality: ndcg@10=",
		"recall@100=",
		"metrics: " + metricsPath,
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("eval-retrieval output missing %q\noutput:\n%s", want, output)
		}
	}
	var metrics struct {
		Schema  string `json:"schema"`
		Dataset string `json:"dataset"`
		Inputs  struct {
			Documents int `json:"documents"`
			Queries   int `json:"queries"`
		} `json:"inputs"`
	}
	data, err := os.ReadFile(metricsPath)
	if err != nil {
		t.Fatalf("read metrics: %v", err)
	}
	if err := json.Unmarshal(data, &metrics); err != nil {
		t.Fatalf("decode metrics: %v", err)
	}
	if metrics.Schema != mantaruntime.RetrievalEvalMetricsSchema || metrics.Dataset != "tiny" || metrics.Inputs.Documents != 2 || metrics.Inputs.Queries != 1 {
		t.Fatalf("metrics = %+v", metrics)
	}
}

func TestRunTrainEmbedFitsContrastivePackage(t *testing.T) {
	path := writeTrainableArtifact(t)
	if err := run([]string{"init-train", "--dim", "D=4", "--dim", "E=3", path}); err != nil {
		t.Fatalf("run init-train: %v", err)
	}
	trainPath := filepath.Join(t.TempDir(), "train.jsonl")
	evalPath := filepath.Join(t.TempDir(), "eval.jsonl")
	examples := []mantaruntime.EmbeddingContrastiveExample{
		{QueryTokens: []int32{1, 2}, PositiveTokens: []int32{1, 2}},
		{QueryTokens: []int32{2, 3}, PositiveTokens: []int32{2, 3}},
		{QueryTokens: []int32{3, 4}, PositiveTokens: []int32{3, 4}},
		{QueryTokens: []int32{4, 5}, PositiveTokens: []int32{4, 5}},
	}
	if err := mantaruntime.WriteEmbeddingContrastiveExamplesFile(trainPath, examples); err != nil {
		t.Fatalf("write train dataset: %v", err)
	}
	if err := mantaruntime.WriteEmbeddingContrastiveExamplesFile(evalPath, examples); err != nil {
		t.Fatalf("write eval dataset: %v", err)
	}
	if err := run([]string{"train-embed", "--epochs", "2", "--batch-size", "2", "--lr", "0.003", "--contrastive-loss", "infonce", "--temperature", "0.07", path, trainPath, evalPath}); err != nil {
		t.Fatalf("run train-embed: %v", err)
	}
	if _, err := mantaruntime.LoadEmbeddingTrainerPackage(path); err != nil {
		t.Fatalf("reload trained package: %v", err)
	}
	checkpoint, err := mantaruntime.ReadEmbeddingTrainCheckpointFile(mantaruntime.DefaultEmbeddingCheckpointPath(path))
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
	tokenizer := mantaruntime.TokenizerFile{
		Version:      mantaruntime.TokenizerFileVersion,
		Tokens:       []string{"<pad>", "<unk>", "alpha", "beta"},
		PadToken:     "<pad>",
		UnknownToken: "<unk>",
	}
	if err := tokenizer.WriteFile(mantaruntime.DefaultTokenizerPath(path)); err != nil {
		t.Fatalf("write tokenizer: %v", err)
	}
	renamedPath := filepath.Join(t.TempDir(), "manta-embed-v1.mll")

	if err := run([]string{"rename-embed", "--name", "manta-embed-v1", path, renamedPath}); err != nil {
		t.Fatalf("run rename-embed: %v", err)
	}

	mod, err := mantaartifact.ReadFile(renamedPath)
	if err != nil {
		t.Fatalf("read renamed artifact: %v", err)
	}
	if mod.Name != "manta-embed-v1" {
		t.Fatalf("module name = %q, want manta-embed-v1", mod.Name)
	}
	manifest, err := mantaruntime.ReadEmbeddingManifestFile(mantaruntime.DefaultEmbeddingManifestPath(renamedPath))
	if err != nil {
		t.Fatalf("read renamed manifest: %v", err)
	}
	if manifest.Name != "manta-embed-v1" {
		t.Fatalf("manifest name = %q, want manta-embed-v1", manifest.Name)
	}
	checkpoint, err := mantaruntime.ReadEmbeddingTrainCheckpointFile(mantaruntime.DefaultEmbeddingCheckpointPath(renamedPath))
	if err != nil {
		t.Fatalf("read renamed checkpoint: %v", err)
	}
	if checkpoint.Manifest.Name != "manta-embed-v1" {
		t.Fatalf("checkpoint manifest name = %q, want manta-embed-v1", checkpoint.Manifest.Name)
	}
	if _, err := os.Stat(mantaruntime.DefaultTokenizerPath(renamedPath)); err != nil {
		t.Fatalf("renamed tokenizer sidecar missing: %v", err)
	}
	if _, err := mantaruntime.LoadEmbeddingTrainerPackage(renamedPath); err != nil {
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
	examples := []mantaruntime.EmbeddingContrastiveExample{
		{QueryTokens: []int32{1, 2}, PositiveTokens: []int32{1, 2}},
		{QueryTokens: []int32{2, 3}, PositiveTokens: []int32{2, 3}},
		{QueryTokens: []int32{3, 4}, PositiveTokens: []int32{3, 4}},
		{QueryTokens: []int32{4, 5}, PositiveTokens: []int32{4, 5}},
	}
	if err := mantaruntime.WriteEmbeddingContrastiveExamplesFile(trainPath, examples); err != nil {
		t.Fatalf("write train dataset: %v", err)
	}
	if err := mantaruntime.WriteEmbeddingContrastiveExamplesFile(evalPath, examples); err != nil {
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

func TestRunTrainEmbedEvalOnlyUsesSingleContrastiveDataset(t *testing.T) {
	path := writeTrainableArtifact(t)
	if err := run([]string{"init-train", "--dim", "D=4", "--dim", "E=3", path}); err != nil {
		t.Fatalf("run init-train: %v", err)
	}
	evalPath := filepath.Join(t.TempDir(), "eval.jsonl")
	examples := []mantaruntime.EmbeddingContrastiveExample{
		{QueryTokens: []int32{1, 2}, PositiveTokens: []int32{1, 2}},
		{QueryTokens: []int32{2, 3}, PositiveTokens: []int32{2, 3}},
	}
	if err := mantaruntime.WriteEmbeddingContrastiveExamplesFile(evalPath, examples); err != nil {
		t.Fatalf("write eval dataset: %v", err)
	}

	output := captureRunOutput(t, []string{"train-embed", "--eval-only", path, evalPath})
	for _, want := range []string{
		"evaluated package",
		"epochs: 0",
		"run_steps: 0",
		"final eval:",
		"train=0 contrastive examples",
		"eval=2 contrastive examples",
		"pairs(planned=4 actual=4)",
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("eval-only output missing %q\noutput:\n%s", want, output)
		}
	}
}

func TestRunTrainEmbedEvalOnlyUsesSingleTextPairDataset(t *testing.T) {
	path := writeTrainableArtifact(t)
	if err := run([]string{"init-train", "--dim", "D=4", "--dim", "E=3", path}); err != nil {
		t.Fatalf("run init-train: %v", err)
	}
	tokenizer := mantaruntime.TokenizerFile{
		Version: mantaruntime.TokenizerFileVersion,
		Tokens:  []string{"[PAD]", "[UNK]", "[CLS]", "[SEP]", "a", "b", "c", "d"},
	}
	if err := tokenizer.WriteFile(mantaruntime.DefaultTokenizerPath(path)); err != nil {
		t.Fatalf("write tokenizer: %v", err)
	}
	evalPath := filepath.Join(t.TempDir(), "eval-text.jsonl")
	evalData := "" +
		"{\"query\":\"ab\",\"document\":\"ab\",\"label\":1}\n" +
		"{\"left\":\"ab\",\"right\":\"cd\",\"label\":0}\n"
	if err := os.WriteFile(evalPath, []byte(evalData), 0o644); err != nil {
		t.Fatalf("write eval text dataset: %v", err)
	}

	output := captureRunOutput(t, []string{"train-embed", "--eval-only", path, evalPath})
	for _, want := range []string{
		"evaluated package",
		"tokenizer:",
		"epochs: 0",
		"run_steps: 0",
		"final eval:",
		"pairs=2",
		"train=0 pairwise examples",
		"eval=2 pairwise examples",
		"pairs(planned=2 actual=2)",
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("eval-only text output missing %q\noutput:\n%s", want, output)
		}
	}
}

func TestRunTrainEmbedEvalOnlyWritesMetricsJSON(t *testing.T) {
	path := writeTrainableArtifact(t)
	if err := run([]string{"init-train", "--dim", "D=4", "--dim", "E=3", path}); err != nil {
		t.Fatalf("run init-train: %v", err)
	}
	evalPath := filepath.Join(t.TempDir(), "eval.jsonl")
	examples := []mantaruntime.EmbeddingContrastiveExample{
		{QueryTokens: []int32{1, 2}, PositiveTokens: []int32{1, 2}},
		{QueryTokens: []int32{2, 3}, PositiveTokens: []int32{2, 3}},
	}
	if err := mantaruntime.WriteEmbeddingContrastiveExamplesFile(evalPath, examples); err != nil {
		t.Fatalf("write eval dataset: %v", err)
	}
	metricsPath := filepath.Join(t.TempDir(), "metrics.json")

	output := captureRunOutput(t, []string{"train-embed", "--eval-only", "--metrics-json", metricsPath, path, evalPath})
	if !strings.Contains(output, "metrics: "+metricsPath) {
		t.Fatalf("eval-only output missing metrics path %q\noutput:\n%s", metricsPath, output)
	}
	data, err := os.ReadFile(metricsPath)
	if err != nil {
		t.Fatalf("read metrics json: %v", err)
	}
	var got struct {
		Schema  string `json:"schema"`
		Command string `json:"command"`
		Mode    string `json:"mode"`
		Summary struct {
			StepsRun int `json:"steps_run"`
		} `json:"summary"`
		FinalEval *struct {
			PairCount int     `json:"pair_count"`
			Top1      float32 `json:"top1_accuracy"`
			MRR       float32 `json:"mean_reciprocal_rank"`
		} `json:"final_eval"`
		Workload struct {
			ActualEvalPairs int64 `json:"actual_eval_pairs"`
		} `json:"workload"`
		Throughput struct {
			EvalPairsPerSecond float64 `json:"eval_pairs_per_second"`
		} `json:"throughput"`
		ProfileDelta struct {
			OptimizerUpdates int64 `json:"optimizer_updates"`
		} `json:"profile_delta"`
	}
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("decode metrics json: %v\n%s", err, string(data))
	}
	if got.Schema != "manta.embedding_train_metrics.v1" || got.Command != "train-embed" || got.Mode != "eval" {
		t.Fatalf("unexpected metrics identity: %+v", got)
	}
	if got.Summary.StepsRun != 0 {
		t.Fatalf("steps_run = %d, want 0", got.Summary.StepsRun)
	}
	if got.FinalEval == nil || got.FinalEval.PairCount != 4 {
		t.Fatalf("final_eval = %+v, want pair_count=4", got.FinalEval)
	}
	if got.FinalEval.Top1 <= 0 || got.FinalEval.MRR <= 0 {
		t.Fatalf("expected ranking metrics in JSON, got final_eval %+v", *got.FinalEval)
	}
	if got.Workload.ActualEvalPairs != 4 {
		t.Fatalf("actual_eval_pairs = %d, want 4", got.Workload.ActualEvalPairs)
	}
	if got.Throughput.EvalPairsPerSecond <= 0 {
		t.Fatalf("eval_pairs_per_second = %f, want positive", got.Throughput.EvalPairsPerSecond)
	}
	if got.ProfileDelta.OptimizerUpdates != 0 {
		t.Fatalf("optimizer_updates = %d, want 0", got.ProfileDelta.OptimizerUpdates)
	}
}

func TestRunCompareTrainMetricsReportsCurrentAndBaselineDeltas(t *testing.T) {
	dir := t.TempDir()
	currentPath := filepath.Join(dir, "current.metrics.json")
	baselinePath := filepath.Join(dir, "baseline.metrics.json")
	current := trainMetricsJSON{
		Schema:   "manta.embedding_train_metrics.v1",
		Command:  "train-embed",
		Mode:     "eval",
		Artifact: "current.mll",
		FinalEval: &evalMetricsJSON{
			Top1Accuracy:       0.9,
			Top5Accuracy:       0.98,
			Top10Accuracy:      1,
			MeanReciprocalRank: 0.95,
			ROCAUC:             0.73,
			ScoreMargin:        0.12,
			Loss:               0.11,
			MeanPositiveRank:   1.1,
			PairCount:          128,
		},
		Throughput: trainThroughputJSON{
			TrainPairsPerSecond:     120000,
			EvalPairsPerSecond:      300,
			OptimizerStepsPerSecond: 0.15,
			PairsPerSecond:          150000,
			ElapsedSeconds:          10,
		},
		Accelerators: trainAcceleratorsJSON{Forward: "cuda", Optimizer: "cuda", Activation: "cuda", Contrastive: "cuda"},
		ProfileDelta: trainProfileDeltaJSON{
			MatMulRuns:          1000,
			MatMulRunUploadMB:   100,
			MatMulRunDownloadMB: 80,
			OptimizerUpdates:    4,
			ActivationCalls:     3,
			ContrastiveCalls:    2,
		},
	}
	baseline := current
	baseline.Artifact = "baseline.mll"
	baseline.FinalEval = &evalMetricsJSON{
		Top1Accuracy:       0.8,
		Top5Accuracy:       0.95,
		Top10Accuracy:      0.99,
		MeanReciprocalRank: 0.9,
		ROCAUC:             0.7,
		ScoreMargin:        0.10,
		Loss:               0.13,
		MeanPositiveRank:   1.3,
		PairCount:          128,
	}
	baseline.Throughput.TrainPairsPerSecond = 100000
	baseline.ProfileDelta.MatMulRuns = 1500
	writeMetricsJSONForTest(t, currentPath, current)
	writeMetricsJSONForTest(t, baselinePath, baseline)

	output := captureRunOutput(t, []string{"compare-train-metrics", currentPath, baselinePath})
	for _, want := range []string{
		"identity: schema=manta.embedding_train_metrics.v1 command=train-embed mode=eval artifact=current.mll",
		"quality: top1=0.900000",
		"throughput: train_pairs/s=120000.00",
		"accelerators: forward=cuda optimizer=cuda activation=cuda contrastive=cuda",
		"profile_delta: matmul_runs=1000",
		"baseline: " + baselinePath,
		"quality_delta: top1=+0.100000",
		"throughput_delta: train_pairs/s=+20000.00",
		"profile_delta_delta: matmul_runs=-500",
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("compare output missing %q\noutput:\n%s", want, output)
		}
	}
}

func TestRunDiagnoseTrainMetricsReportsDeviceBackedEfficiency(t *testing.T) {
	dir := t.TempDir()
	metricsPath := filepath.Join(dir, "current.metrics.json")
	metrics := trainMetricsJSON{
		Schema:   "manta.embedding_train_metrics.v1",
		Command:  "train-embed",
		Mode:     "train",
		Artifact: "current.mll",
		Workload: trainWorkloadJSON{
			ActualTrainPairs: 100000,
		},
		Throughput: trainThroughputJSON{
			ElapsedSeconds:          90,
			TrainSeconds:            80,
			EvalSeconds:             10,
			TrainPairsPerSecond:     120000,
			EvalPairsPerSecond:      50000,
			OptimizerStepsPerSecond: 0.5,
		},
		Accelerators: trainAcceleratorsJSON{
			Forward:     "cuda",
			Optimizer:   "cuda",
			Activation:  "host",
			Contrastive: "cuda",
		},
		ProfileDelta: trainProfileDeltaJSON{
			MatMulRuns:          1000,
			MatMulRunUploadMB:   500,
			MatMulRunDownloadMB: 250,
			OptimizerUpdates:    10,
			OptimizerSyncs:      20,
		},
	}
	writeMetricsJSONForTest(t, metricsPath, metrics)

	output := captureRunOutput(t, []string{"diagnose-train-metrics", metricsPath})
	for _, want := range []string{
		"metrics: " + metricsPath,
		"backend: forward=cuda optimizer=cuda activation=host contrastive=cuda",
		"efficiency: matmul_runs/update=100.00 pairs/matmul_run=100.00 optimizer_syncs/update=2.00",
		"transfer: total_mb=750.00 mb/matmul_run=0.7500 kb/pair=7.6800",
		"finding: ok production-critical accelerators are device-backed",
		"finding: note activation accelerator is host",
		"diagnosis: OK warnings=0 notes=1",
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("diagnosis output missing %q\noutput:\n%s", want, output)
		}
	}
}

func TestRunDiagnoseTrainMetricsWarnsOnHostFallbacks(t *testing.T) {
	dir := t.TempDir()
	metricsPath := filepath.Join(dir, "current.metrics.json")
	metrics := trainMetricsJSON{
		Schema:   "manta.embedding_train_metrics.v1",
		Command:  "train-embed",
		Mode:     "train",
		Artifact: "current.mll",
		Workload: trainWorkloadJSON{
			ActualTrainPairs: 100,
		},
		Throughput: trainThroughputJSON{
			ElapsedSeconds:      10,
			TrainSeconds:        10,
			TrainPairsPerSecond: 0,
		},
		Accelerators: trainAcceleratorsJSON{
			Forward:     "host",
			Optimizer:   "host",
			Activation:  "host",
			Contrastive: "host",
		},
	}
	writeMetricsJSONForTest(t, metricsPath, metrics)

	output := captureRunOutput(t, []string{"diagnose-train-metrics", metricsPath})
	for _, want := range []string{
		"finding: warn production-critical accelerators include host fallback: forward=host optimizer=host contrastive=host",
		"finding: warn training run recorded zero optimizer updates",
		"finding: warn training pairs were processed but train_pairs/s is zero",
		"diagnosis: WARN warnings=3 notes=1",
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("warning diagnosis output missing %q\noutput:\n%s", want, output)
		}
	}
}

func TestRunDiagnoseTrainMetricsWarnsOnMissingOptimizerStepRate(t *testing.T) {
	dir := t.TempDir()
	metricsPath := filepath.Join(dir, "current.metrics.json")
	metrics := trainMetricsJSON{
		Schema:   "manta.embedding_train_metrics.v1",
		Command:  "train-embed",
		Mode:     "train",
		Artifact: "current.mll",
		Throughput: trainThroughputJSON{
			TrainSeconds:            2,
			TrainPairsPerSecond:     500,
			OptimizerStepsPerSecond: 0,
		},
		Accelerators: trainAcceleratorsJSON{
			Forward:     "cuda",
			Optimizer:   "cuda",
			Activation:  "cuda",
			Contrastive: "cuda",
		},
		ProfileDelta: trainProfileDeltaJSON{
			OptimizerUpdates: 2,
		},
	}
	writeMetricsJSONForTest(t, metricsPath, metrics)

	output := captureRunOutput(t, []string{"diagnose-train-metrics", metricsPath})
	for _, want := range []string{
		"finding: warn optimizer updates were recorded but optimizer_steps/s is zero",
		"diagnosis: WARN warnings=1 notes=0",
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("optimizer-rate diagnosis output missing %q\noutput:\n%s", want, output)
		}
	}
}

func TestRunGateTrainMetricsChecksThresholdFile(t *testing.T) {
	clearTrainMetricGateEnv(t)
	dir := t.TempDir()
	metricsPath := filepath.Join(dir, "current.metrics.json")
	thresholdsPath := filepath.Join(dir, "thresholds.env")
	metrics := trainMetricsJSON{
		Schema:   "manta.embedding_train_metrics.v1",
		Command:  "train-embed",
		Mode:     "eval",
		Artifact: "current.mll",
		FinalEval: &evalMetricsJSON{
			Top1Accuracy:       0.9,
			Top5Accuracy:       0.98,
			Top10Accuracy:      1,
			MeanReciprocalRank: 0.95,
			ROCAUC:             0.73,
			ScoreMargin:        0.12,
			Loss:               0.11,
			MeanPositiveRank:   1.1,
		},
		Throughput: trainThroughputJSON{
			TrainPairsPerSecond:     120000,
			OptimizerStepsPerSecond: 0.15,
		},
		ProfileDelta: trainProfileDeltaJSON{
			MatMulRuns:          1000,
			MatMulRunUploadMB:   100,
			MatMulRunDownloadMB: 80,
			OptimizerUpdates:    0,
		},
	}
	writeMetricsJSONForTest(t, metricsPath, metrics)
	thresholds := "" +
		"MANTA_MIN_MRR=0.90\n" +
		"MANTA_MIN_TOP1=0.80\n" +
		"MANTA_MAX_MEAN_RANK=1.20\n" +
		"MANTA_MIN_TRAIN_PAIRS_PER_SEC=100000\n" +
		"MANTA_MAX_MATMUL_RUNS=2000\n"
	if err := os.WriteFile(thresholdsPath, []byte(thresholds), 0o644); err != nil {
		t.Fatalf("write thresholds: %v", err)
	}

	output := captureRunOutput(t, []string{"gate-train-metrics", "--thresholds", thresholdsPath, metricsPath})
	for _, want := range []string{
		"metrics: " + metricsPath,
		"thresholds: " + thresholdsPath,
		"scope: all",
		"pass: mrr=0.95 >= 0.9",
		"pass: train_pairs/s=120000 >= 100000",
		"pass: matmul_runs=1000 <= 2000",
		"gate: PASS checks=5",
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("gate output missing %q\noutput:\n%s", want, output)
		}
	}
}

func TestRunGateTrainMetricsChecksEvalOnlyOptimizerUpdates(t *testing.T) {
	clearTrainMetricGateEnv(t)
	dir := t.TempDir()
	metricsPath := filepath.Join(dir, "eval.metrics.json")
	metrics := trainMetricsJSON{
		Schema:       "manta.embedding_train_metrics.v1",
		Command:      "train-embed",
		Mode:         "eval",
		Artifact:     "current.mll",
		ProfileDelta: trainProfileDeltaJSON{OptimizerUpdates: 0},
	}
	writeMetricsJSONForTest(t, metricsPath, metrics)

	output := captureRunOutput(t, []string{"gate-train-metrics", "--scope", "eval-only", metricsPath})
	for _, want := range []string{
		"scope: eval-only",
		"pass: optimizer_updates=0 == 0 (eval-only)",
		"gate: PASS checks=1",
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("eval-only gate output missing %q\noutput:\n%s", want, output)
		}
	}
}

func TestRunGateTrainMetricsFailsMissedThreshold(t *testing.T) {
	clearTrainMetricGateEnv(t)
	dir := t.TempDir()
	metricsPath := filepath.Join(dir, "current.metrics.json")
	thresholdsPath := filepath.Join(dir, "thresholds.env")
	metrics := trainMetricsJSON{
		Schema:   "manta.embedding_train_metrics.v1",
		Command:  "train-embed",
		Mode:     "eval",
		Artifact: "current.mll",
		FinalEval: &evalMetricsJSON{
			MeanReciprocalRank: 0.5,
		},
	}
	writeMetricsJSONForTest(t, metricsPath, metrics)
	if err := os.WriteFile(thresholdsPath, []byte("MANTA_MIN_MRR=0.90\n"), 0o644); err != nil {
		t.Fatalf("write thresholds: %v", err)
	}

	output, err := captureRunOutputAndError(t, []string{"gate-train-metrics", "--thresholds", thresholdsPath, metricsPath})
	if err == nil {
		t.Fatalf("gate unexpectedly passed\noutput:\n%s", output)
	}
	for _, want := range []string{
		"fail: mrr=0.5 >= 0.9",
		"gate: FAIL checks=1 failed=1",
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("failed gate output missing %q\noutput:\n%s", want, output)
		}
	}
}

func clearTrainMetricGateEnv(t *testing.T) {
	t.Helper()
	for _, threshold := range trainMetricThresholds {
		t.Setenv(threshold.Env, "")
	}
}

func writeMetricsJSONForTest(t *testing.T, path string, metrics trainMetricsJSON) {
	t.Helper()
	data, err := json.Marshal(metrics)
	if err != nil {
		t.Fatalf("marshal metrics json: %v", err)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatalf("write metrics json: %v", err)
	}
}

func TestRunTrainEmbedFitsTextContrastivePackage(t *testing.T) {
	path := writeTrainableArtifact(t)
	if err := run([]string{"init-train", "--dim", "D=4", "--dim", "E=3", path}); err != nil {
		t.Fatalf("run init-train: %v", err)
	}
	tokenizer := mantaruntime.TokenizerFile{
		Version: mantaruntime.TokenizerFileVersion,
		Tokens:  []string{"[PAD]", "[UNK]", "[CLS]", "[SEP]", "a", "b", "c", "d"},
	}
	if err := tokenizer.WriteFile(mantaruntime.DefaultTokenizerPath(path)); err != nil {
		t.Fatalf("write tokenizer: %v", err)
	}
	trainPath := filepath.Join(t.TempDir(), "train-text.jsonl")
	evalPath := filepath.Join(t.TempDir(), "eval-text.jsonl")
	examples := []mantaruntime.EmbeddingTextContrastiveExample{
		{Query: "ab", Positive: "ab"},
		{Query: "cd", Positive: "cd"},
		{Query: "ab", Positive: "ab"},
		{Query: "cd", Positive: "cd"},
	}
	if err := mantaruntime.WriteEmbeddingTextContrastiveExamplesFile(trainPath, examples); err != nil {
		t.Fatalf("write train text dataset: %v", err)
	}
	if err := mantaruntime.WriteEmbeddingTextContrastiveExamplesFile(evalPath, examples); err != nil {
		t.Fatalf("write eval text dataset: %v", err)
	}
	if err := run([]string{"train-embed", "--epochs", "2", "--batch-size", "2", path, trainPath, evalPath}); err != nil {
		t.Fatalf("run train-embed text: %v", err)
	}
	if _, err := mantaruntime.LoadEmbeddingTrainerPackage(path); err != nil {
		t.Fatalf("reload trained package: %v", err)
	}
}

func TestRunTrainEmbedFitsTextContrastivePackageWithLabeledEvalPairs(t *testing.T) {
	path := writeTrainableArtifact(t)
	if err := run([]string{"init-train", "--dim", "D=4", "--dim", "E=3", path}); err != nil {
		t.Fatalf("run init-train: %v", err)
	}
	tokenizer := mantaruntime.TokenizerFile{
		Version: mantaruntime.TokenizerFileVersion,
		Tokens:  []string{"[PAD]", "[UNK]", "[CLS]", "[SEP]", "a", "b", "c", "d"},
	}
	if err := tokenizer.WriteFile(mantaruntime.DefaultTokenizerPath(path)); err != nil {
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
	if _, err := mantaruntime.LoadEmbeddingTrainerPackage(path); err != nil {
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
	srcPath := copyExampleFile(t, dir, "encoder_trainable_q8x2.manta")
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
	outPath := mantaruntime.DefaultMLLPath(path)
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
	tokenizerPath := mantaruntime.DefaultTokenizerPath(path)
	if _, err := os.Stat(tokenizerPath); err != nil {
		t.Fatalf("expected tokenizer file %q: %v", tokenizerPath, err)
	}
	if _, err := mantaruntime.ReadTokenizerFile(tokenizerPath); err != nil {
		t.Fatalf("read tokenizer file: %v", err)
	}
	manifest, err := mantaruntime.ReadEmbeddingManifestFile(mantaruntime.DefaultEmbeddingManifestPath(path))
	if err != nil {
		t.Fatalf("read manifest: %v", err)
	}
	if manifest.Tokenizer.VocabSize != 11 {
		t.Fatalf("expected manifest vocab size to track tokenizer, got %d", manifest.Tokenizer.VocabSize)
	}
}

func TestRunTokenizeEmbedWritesTokenDatasets(t *testing.T) {
	path := writeTrainableArtifact(t)
	if err := run([]string{"init-train", "--dim", "D=4", "--dim", "E=3", path}); err != nil {
		t.Fatalf("run init-train: %v", err)
	}
	tokenizer := mantaruntime.TokenizerFile{
		Version: mantaruntime.TokenizerFileVersion,
		Tokens:  []string{"[PAD]", "[UNK]", "[CLS]", "[SEP]", "a", "b", "c", "d"},
	}
	if err := tokenizer.WriteFile(mantaruntime.DefaultTokenizerPath(path)); err != nil {
		t.Fatalf("write tokenizer: %v", err)
	}

	dir := t.TempDir()
	trainTextPath := filepath.Join(dir, "train-text.jsonl")
	evalTextPath := filepath.Join(dir, "eval-text.jsonl")
	trainTokenPath := filepath.Join(dir, "train-token.jsonl")
	evalTokenPath := filepath.Join(dir, "eval-token.jsonl")
	if err := os.WriteFile(trainTextPath, []byte("{\"query\":\"ab\",\"positive\":\"ab\"}\n"), 0o644); err != nil {
		t.Fatalf("write train text: %v", err)
	}
	evalText := "" +
		"{\"query\":\"ab\",\"document\":\"ab\",\"label\":1}\n" +
		"{\"left\":\"ab\",\"right\":\"cd\",\"label\":0}\n"
	if err := os.WriteFile(evalTextPath, []byte(evalText), 0o644); err != nil {
		t.Fatalf("write eval text: %v", err)
	}

	if err := run([]string{"tokenize-embed", "--mode", "contrastive", path, trainTextPath, trainTokenPath}); err != nil {
		t.Fatalf("run tokenize contrastive: %v", err)
	}
	if err := run([]string{"tokenize-embed", "--mode", "pair", path, evalTextPath, evalTokenPath}); err != nil {
		t.Fatalf("run tokenize pair: %v", err)
	}
	if _, err := mantaruntime.ReadEmbeddingContrastiveExamplesFile(trainTokenPath); err != nil {
		t.Fatalf("read tokenized train: %v", err)
	}
	pairs, err := mantaruntime.ReadEmbeddingPairExamplesFile(evalTokenPath)
	if err != nil {
		t.Fatalf("read tokenized eval: %v", err)
	}
	if len(pairs) != 2 || pairs[0].Target != 1 || pairs[1].Target != 0 {
		t.Fatalf("tokenized eval targets = %+v", pairs)
	}
	output := captureRunOutput(t, []string{"train-embed", "--eval-only", "--no-tokenizer", path, evalTokenPath})
	if !strings.Contains(output, "evaluated package") || !strings.Contains(output, "eval=2 pairwise examples") {
		t.Fatalf("eval-only token output missing expected summary:\n%s", output)
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
	if _, err := mantaruntime.ReadEmbeddingTextContrastiveExamplesFile(trainPath); err != nil {
		t.Fatalf("read mined train pairs: %v", err)
	}
	evalSet, err := mantaruntime.ReadEmbeddingTextPairExamplesFile(evalPath)
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
	if _, err := mantaruntime.LoadEmbeddingTrainerPackage(path); err != nil {
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
	if _, err := os.Stat(mantaruntime.DefaultTokenizerPath(path)); err != nil {
		t.Fatalf("expected tokenizer file: %v", err)
	}
	if _, err := os.Stat(mantaruntime.DefaultMinedTrainPairsPath(path)); err != nil {
		t.Fatalf("expected mined train pairs: %v", err)
	}
	if _, err := os.Stat(mantaruntime.DefaultMinedEvalPairsPath(path)); err != nil {
		t.Fatalf("expected mined eval pairs: %v", err)
	}
	if _, err := mantaruntime.LoadEmbeddingTrainerPackage(path); err != nil {
		t.Fatalf("reload trained package: %v", err)
	}
}

func TestRunTrainCorpusRepeatedEncoderExampleFlow(t *testing.T) {
	dir := t.TempDir()
	srcPath := copyExampleFile(t, dir, "encoder_trainable_q8x2.manta")
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
	manifest, err := mantaruntime.ReadEmbeddingManifestFile(mantaruntime.DefaultEmbeddingManifestPath(artifactPath))
	if err != nil {
		t.Fatalf("read manifest: %v", err)
	}
	if manifest.EncoderRepeats != 2 {
		t.Fatalf("encoder repeats = %d, want 2", manifest.EncoderRepeats)
	}
	profile, err := mantaruntime.ReadEmbeddingTrainProfileFile(mantaruntime.DefaultEmbeddingTrainProfilePath(artifactPath))
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
		mantaruntime.DefaultTokenizerPath(artifactPath),
		mantaruntime.DefaultMinedTrainPairsPath(artifactPath),
		mantaruntime.DefaultMinedEvalPairsPath(artifactPath),
	} {
		if _, err := os.Stat(candidate); err != nil {
			t.Fatalf("expected generated training artifact %q: %v", candidate, err)
		}
	}
	if _, err := mantaruntime.LoadEmbeddingTrainerPackage(artifactPath); err != nil {
		t.Fatalf("reload trained repeated-encoder package: %v", err)
	}
}

func TestRunCompileDefaultMLLThenTrainFlow(t *testing.T) {
	dir := t.TempDir()
	srcPath := filepath.Join(dir, "tiny_train_embed.manta")
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
	manifest := mantaruntime.EmbeddingManifest{
		Name:                "tiny-train-embed",
		PooledEntry:         "embed_pooled",
		BatchEntry:          "embed_pooled_batch",
		TokenInput:          "tokens",
		OutputName:          "result",
		OutputDType:         "q8",
		TokenEmbeddingParam: "token_embedding",
		ProjectionParam:     "projection",
		Tokenizer: mantaruntime.TokenizerManifest{
			VocabSize:   8,
			MaxSequence: 8,
			PadID:       0,
		},
	}
	if err := manifest.WriteFile(mantaruntime.DefaultEmbeddingManifestPath(artifactPath)); err != nil {
		t.Fatalf("write manifest: %v", err)
	}
	if err := run([]string{"init-train", "--dim", "D=4", "--dim", "E=3", artifactPath}); err != nil {
		t.Fatalf("run init-train: %v", err)
	}
	trainPath := filepath.Join(dir, "train.jsonl")
	evalPath := filepath.Join(dir, "eval.jsonl")
	examples := []mantaruntime.EmbeddingContrastiveExample{
		{QueryTokens: []int32{1, 2}, PositiveTokens: []int32{1, 2}},
		{QueryTokens: []int32{2, 3}, PositiveTokens: []int32{2, 3}},
		{QueryTokens: []int32{3, 4}, PositiveTokens: []int32{3, 4}},
		{QueryTokens: []int32{4, 5}, PositiveTokens: []int32{4, 5}},
	}
	if err := mantaruntime.WriteEmbeddingContrastiveExamplesFile(trainPath, examples); err != nil {
		t.Fatalf("write train dataset: %v", err)
	}
	if err := mantaruntime.WriteEmbeddingContrastiveExamplesFile(evalPath, examples); err != nil {
		t.Fatalf("write eval dataset: %v", err)
	}
	if err := run([]string{"train-embed", "--epochs", "2", "--batch-size", "2", artifactPath, trainPath, evalPath}); err != nil {
		t.Fatalf("run train-embed: %v", err)
	}
	if _, err := mantaruntime.LoadEmbeddingTrainerPackage(artifactPath); err != nil {
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
	path := filepath.Join(t.TempDir(), "tiny_train_embed.mll")
	if err := mantaartifact.WriteFile(path, bundle.Artifact); err != nil {
		t.Fatalf("write artifact: %v", err)
	}
	manifest := mantaruntime.EmbeddingManifest{
		Name:                "tiny-train-embed",
		PooledEntry:         "embed_pooled",
		BatchEntry:          "embed_pooled_batch",
		TokenInput:          "tokens",
		OutputName:          "result",
		OutputDType:         "q8",
		TokenEmbeddingParam: "token_embedding",
		ProjectionParam:     "projection",
		Tokenizer: mantaruntime.TokenizerManifest{
			VocabSize:   8,
			MaxSequence: 8,
			PadID:       0,
		},
	}
	if err := manifest.WriteFile(mantaruntime.DefaultEmbeddingManifestPath(path)); err != nil {
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
	output, runErr := captureRunOutputAndError(t, args)
	if runErr != nil {
		t.Fatalf("run %v: %v\noutput:\n%s", args, runErr, output)
	}
	return output
}

func captureRunOutputAndError(t *testing.T, args []string) (string, error) {
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
	return string(data), runErr
}
