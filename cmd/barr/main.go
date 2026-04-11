package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"runtime/pprof"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/odvcencio/barracuda/artifact/barr"
	"github.com/odvcencio/barracuda/compiler"
	"github.com/odvcencio/barracuda/models"
	barruntime "github.com/odvcencio/barracuda/runtime"
	"github.com/odvcencio/barracuda/runtime/backend"
	"github.com/odvcencio/barracuda/runtime/backends/cuda"
	"github.com/odvcencio/barracuda/runtime/backends/metal"
)

func main() {
	stopProfile, err := startOptionalProfiles()
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	defer stopProfile()
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func startOptionalProfiles() (func(), error) {
	cpuPath := os.Getenv("BARR_CPU_PROFILE")
	memPath := os.Getenv("BARR_MEM_PROFILE")
	var cpuFile *os.File
	if cpuPath != "" {
		file, err := os.Create(cpuPath)
		if err != nil {
			return nil, fmt.Errorf("create CPU profile %q: %w", cpuPath, err)
		}
		if err := pprof.StartCPUProfile(file); err != nil {
			_ = file.Close()
			return nil, fmt.Errorf("start CPU profile %q: %w", cpuPath, err)
		}
		cpuFile = file
	}
	return func() {
		if cpuFile != nil {
			pprof.StopCPUProfile()
			_ = cpuFile.Close()
			fmt.Fprintf(os.Stderr, "cpu profile: %s\n", cpuPath)
		}
		if memPath != "" {
			runtime.GC()
			file, err := os.Create(memPath)
			if err != nil {
				fmt.Fprintf(os.Stderr, "write memory profile %q: %v\n", memPath, err)
				return
			}
			if err := pprof.WriteHeapProfile(file); err != nil {
				fmt.Fprintf(os.Stderr, "write memory profile %q: %v\n", memPath, err)
			}
			_ = file.Close()
			fmt.Fprintf(os.Stderr, "memory profile: %s\n", memPath)
		}
	}, nil
}

func run(args []string) error {
	if len(args) == 0 {
		printUsage()
		return nil
	}

	switch args[0] {
	case "version":
		fmt.Println("barr dev")
		return nil
	case "compile":
		return runCompile(args[1:])
	case "run":
		return runArtifact(args[1:])
	case "demo":
		return runDemo(args[1:])
	case "inspect":
		return runInspect(args[1:])
	case "export-mll":
		return runExportMLL(args[1:])
	case "init-model":
		return runInitModel(args[1:])
	case "init-train":
		return runInitTrain(args[1:])
	case "train-tokenizer":
		return runTrainTokenizer(args[1:])
	case "mine-text-pairs":
		return runMineTextPairs(args[1:])
	case "train-embed":
		return runTrainEmbed(args[1:])
	case "train-corpus":
		return runTrainCorpus(args[1:])
	default:
		return fmt.Errorf("unknown subcommand %q", args[0])
	}
}

func runCompile(args []string) error {
	if len(args) == 0 || args[0] == "" {
		return fmt.Errorf("usage: barr compile <source.bar> [output.mll]")
	}
	srcPath := args[0]
	outPath := defaultArtifactPath(srcPath)
	if len(args) > 1 && args[1] != "" {
		outPath = args[1]
	}

	src, err := os.ReadFile(srcPath)
	if err != nil {
		return err
	}
	moduleName := strings.TrimSuffix(filepath.Base(srcPath), filepath.Ext(srcPath))
	bundle, err := compiler.Build(src, compiler.Options{ModuleName: moduleName})
	if err != nil {
		return err
	}
	if err := barr.WriteFile(outPath, bundle.Artifact); err != nil {
		return err
	}

	fmt.Printf("compiled %q -> %q\n", srcPath, outPath)
	fmt.Printf("entrypoints: %d, steps: %d, kernels: %d\n",
		len(bundle.Artifact.EntryPoints), len(bundle.Artifact.Steps), len(bundle.Artifact.Kernels))
	fmt.Printf("kernel ops: %d\n", totalKernelOps(bundle.Artifact.Kernels))
	return nil
}

func runDemo(args []string) error {
	name := "tiny_embed"
	if len(args) > 0 && args[0] != "" {
		name = args[0]
	}
	preset := compiler.PresetTinyEmbed
	switch name {
	case "tiny_embed_pooled":
		preset = compiler.PresetTinyEmbedPooled
	case "tiny_embed_masked_pooled":
		preset = compiler.PresetTinyEmbedMaskedPooled
	case "tiny_decode":
		preset = compiler.PresetTinyDecode
	case "tiny_score":
		preset = compiler.PresetTinyScore
	case "tiny_rerank":
		preset = compiler.PresetTinyRerank
	case "tiny_select":
		preset = compiler.PresetTinySelect
	case "tiny_retrieve":
		preset = compiler.PresetTinyRetrieve
	case "tiny_candidates":
		preset = compiler.PresetTinyCandidates
	case "tiny_batch_candidates":
		preset = compiler.PresetTinyBatchCandidates
	case "tiny_packed_candidates":
		preset = compiler.PresetTinyPackedCandidates
	}

	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: name, Preset: preset})
	if err != nil {
		return err
	}

	rt := barruntime.New(cuda.New(), metal.New())
	prog, err := rt.Load(context.Background(), bundle.Artifact, stubLoadOptions(bundle.Artifact)...)
	if err != nil {
		return err
	}

	entryName := defaultEntryName(bundle.Artifact)
	entry, err := entryPointByName(bundle.Artifact, entryName)
	if err != nil {
		return err
	}
	result, err := prog.Run(context.Background(), backend.Request{
		Entry:  entryName,
		Inputs: stubInputs(entry),
	})
	if err != nil {
		return err
	}

	fmt.Printf("loaded module %q for backend %q\n", bundle.Artifact.Name, prog.Backend())
	fmt.Printf("entrypoints: %d, steps: %d, kernels: %d\n",
		len(bundle.Artifact.EntryPoints), len(bundle.Artifact.Steps), len(bundle.Artifact.Kernels))
	fmt.Printf("kernel ops: %d\n", totalKernelOps(bundle.Artifact.Kernels))
	fmt.Printf("params: %d\n", len(bundle.Artifact.Params))
	fmt.Printf("artifact version: %s\n", bundle.Artifact.Version)
	fmt.Printf("ran entrypoint: %s\n", entryName)
	fmt.Printf("outputs: %s\n", strings.Join(sortedValueKeys(result.Outputs), ", "))
	fmt.Printf("output summary: %s\n", strings.Join(outputSummaries(result.Outputs), "; "))
	fmt.Printf("trace steps: %d\n", len(result.Trace))
	return nil
}

func runArtifact(args []string) error {
	if len(args) == 0 || args[0] == "" {
		return fmt.Errorf("usage: barr run <artifact.mll> [entry]")
	}
	path := args[0]
	mod, err := barr.ReadFile(path)
	if err != nil {
		return err
	}
	entryName := defaultEntryName(mod)
	if len(args) > 1 && args[1] != "" {
		entryName = args[1]
	}
	entry, err := entryPointByName(mod, entryName)
	if err != nil {
		return err
	}
	rt := barruntime.New(cuda.New(), metal.New())
	prog, err := rt.Load(context.Background(), mod, stubLoadOptions(mod)...)
	if err != nil {
		return err
	}
	result, err := prog.Run(context.Background(), backend.Request{
		Entry:  entryName,
		Inputs: stubInputs(entry),
	})
	if err != nil {
		return err
	}

	fmt.Printf("loaded artifact %q for backend %q\n", mod.Name, prog.Backend())
	fmt.Printf("ran entrypoint: %s\n", entryName)
	fmt.Printf("outputs: %s\n", strings.Join(sortedValueKeys(result.Outputs), ", "))
	fmt.Printf("output summary: %s\n", strings.Join(outputSummaries(result.Outputs), "; "))
	fmt.Printf("trace steps: %d\n", len(result.Trace))
	return nil
}

func runInspect(args []string) error {
	if len(args) == 0 || args[0] == "" {
		return fmt.Errorf("usage: barr inspect <artifact.mll>")
	}
	path := args[0]
	mod, err := barr.ReadFile(path)
	if err != nil {
		return err
	}
	fmt.Printf("module: %s\n", mod.Name)
	fmt.Printf("artifact version: %s\n", mod.Version)
	fmt.Printf("entrypoints: %d, steps: %d, kernels: %d\n", len(mod.EntryPoints), len(mod.Steps), len(mod.Kernels))
	fmt.Printf("backends: %s\n", joinBackendKinds(mod.Requirements.SupportedBackends))
	if len(mod.Requirements.Capabilities) > 0 {
		fmt.Printf("capabilities: %s\n", strings.Join(mod.Requirements.Capabilities, ", "))
	}
	embeddedPackage := false
	embeddingManifestPath := barruntime.ResolveEmbeddingManifestPath(path)
	if _, err := os.Stat(embeddingManifestPath); err == nil {
		manifest, err := barruntime.ReadEmbeddingManifestFile(embeddingManifestPath)
		if err != nil {
			return err
		}
		fmt.Printf("embedding manifest: %s\n", embeddingManifestPath)
		printEmbeddingManifestSummary(manifest)
	} else if sealed, err := barruntime.ReadSealedEmbeddingPackage(path); err == nil {
		fmt.Println("embedding manifest: embedded")
		printEmbeddingManifestSummary(sealed.Manifest)
		fmt.Println("package: embedded sealed MLL")
		fmt.Println("package verify: OK")
		embeddedPackage = true
	}
	packagePath := barruntime.ResolvePackageManifestPath(path)
	if !embeddedPackage {
		if _, err := os.Stat(packagePath); err == nil {
			pkg, err := barruntime.ReadPackageManifestFile(packagePath)
			if err != nil {
				return err
			}
			verifyPaths := map[string]string{
				"artifact":           path,
				"embedding_manifest": barruntime.DefaultEmbeddingManifestPath(path),
				"tokenizer":          barruntime.DefaultTokenizerPath(path),
				"weights":            barruntime.DefaultWeightFilePath(path),
				"memory_plan":        barruntime.DefaultMemoryPlanPath(path),
				"train_manifest":     barruntime.DefaultEmbeddingTrainManifestPath(path),
				"checkpoint":         barruntime.DefaultEmbeddingCheckpointPath(path),
				"train_profile":      barruntime.DefaultEmbeddingTrainProfilePath(path),
			}
			if pkg.Kind == barruntime.PackageEmbedding {
				delete(verifyPaths, "train_manifest")
				delete(verifyPaths, "checkpoint")
				delete(verifyPaths, "train_profile")
			}
			verifyErr := pkg.VerifyFiles(verifyPaths)
			fmt.Printf("package: %s (%s)\n", packagePath, pkg.Kind)
			if verifyErr != nil {
				fmt.Printf("package verify: FAIL (%v)\n", verifyErr)
			} else {
				fmt.Println("package verify: OK")
			}
		}
	}
	profilePath := barruntime.DefaultEmbeddingTrainProfilePath(path)
	if _, err := os.Stat(profilePath); err == nil {
		profile, err := barruntime.ReadEmbeddingTrainProfileFile(profilePath)
		if err != nil {
			return err
		}
		fmt.Printf("train profile: step=%d forward=%s optimizer=%s activation=%s contrastive=%s\n",
			profile.Step,
			displayTrainBackend(profile.ForwardBackend),
			displayTrainBackend(profile.OptimizerBackend),
			displayTrainBackend(profile.ActivationBackend),
			displayTrainBackend(profile.ContrastiveBackend),
		)
	}
	return nil
}

func printEmbeddingManifestSummary(manifest barruntime.EmbeddingManifest) {
	fmt.Printf("embedding model: %s pooled=%s batch=%s output=%s/%s\n",
		displayManifestName(manifest.Name),
		manifest.PooledEntry,
		displayManifestName(manifest.BatchEntry),
		manifest.OutputName,
		manifest.OutputDType,
	)
	fmt.Printf("encoder repeats: %d\n", manifest.EncoderRepeats)
	if manifest.Tokenizer.VocabSize > 0 || manifest.Tokenizer.MaxSequence > 0 {
		fmt.Printf("tokenizer: vocab=%d max_sequence=%d\n", manifest.Tokenizer.VocabSize, manifest.Tokenizer.MaxSequence)
	}
}

func runExportMLL(args []string) error {
	if len(args) == 0 || args[0] == "" {
		return fmt.Errorf("usage: barr export-mll <artifact.mll> [output.mll]")
	}
	artifactPath := args[0]
	outputPath := ""
	if len(args) > 1 {
		outputPath = args[1]
	}
	writtenPath, err := barruntime.ExportPackageToMLL(artifactPath, outputPath)
	if err != nil {
		return err
	}
	fmt.Printf("exported %q -> %q\n", artifactPath, writtenPath)
	return nil
}

func runInitModel(args []string) error {
	fs := flag.NewFlagSet("init-model", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	var name string
	var vocabSize int
	var maxSequence int
	var embeddingDim int
	var hiddenDim int
	var seed int64
	var learningRate float64
	var weightDecay float64
	var weightBits int
	var optimizer string
	var contrastiveLoss string
	var temperature float64
	fs.StringVar(&name, "name", "", "model name")
	fs.IntVar(&vocabSize, "vocab-size", 0, "tokenizer vocab size")
	fs.IntVar(&maxSequence, "max-seq", 0, "maximum token sequence length")
	fs.IntVar(&embeddingDim, "embedding-dim", 0, "embedding/model dimension")
	fs.IntVar(&hiddenDim, "hidden-dim", 0, "FFN hidden dimension")
	fs.Int64Var(&seed, "seed", 0, "initialization seed")
	fs.Float64Var(&learningRate, "lr", 0, "trainer learning rate")
	fs.Float64Var(&weightDecay, "weight-decay", 0, "trainer weight decay")
	fs.IntVar(&weightBits, "weight-bits", 0, "forward fake-quant bits")
	fs.StringVar(&optimizer, "optimizer", "", "optimizer name")
	fs.StringVar(&contrastiveLoss, "contrastive-loss", "", "contrastive loss: pair_mse or infonce")
	fs.Float64Var(&temperature, "temperature", 0, "contrastive softmax temperature")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 1 || fs.Arg(0) == "" {
		return fmt.Errorf("usage: barr init-model [flags] <artifact.mll>")
	}
	if learningRate < 0 {
		return fmt.Errorf("lr must be non-negative")
	}
	if weightDecay < 0 {
		return fmt.Errorf("weight-decay must be non-negative")
	}
	if temperature < 0 {
		return fmt.Errorf("temperature must be non-negative")
	}
	path := fs.Arg(0)
	paths, err := models.InitDefaultEmbeddingPackage(path, models.DefaultEmbeddingPackageConfig{
		Name:            name,
		VocabSize:       vocabSize,
		MaxSequence:     maxSequence,
		EmbeddingDim:    embeddingDim,
		HiddenDim:       hiddenDim,
		Seed:            seed,
		LearningRate:    float32(learningRate),
		WeightDecay:     float32(weightDecay),
		WeightBits:      weightBits,
		Optimizer:       optimizer,
		ContrastiveLoss: contrastiveLoss,
		Temperature:     float32(temperature),
	})
	if err != nil {
		return err
	}
	manifest, err := barruntime.ReadEmbeddingManifestFile(paths.EmbeddingManifestPath)
	if err != nil {
		return err
	}
	checkpoint, err := barruntime.ReadEmbeddingTrainCheckpointFile(paths.CheckpointPath)
	if err != nil {
		return err
	}
	fmt.Printf("initialized default embedding model %q\n", manifest.Name)
	fmt.Printf("artifact: %s\n", paths.ArtifactPath)
	fmt.Printf("embedding manifest: %s\n", paths.EmbeddingManifestPath)
	fmt.Printf("tokenizer contract: vocab=%d max_sequence=%d\n", manifest.Tokenizer.VocabSize, manifest.Tokenizer.MaxSequence)
	fmt.Printf("encoder repeats: %d\n", manifest.EncoderRepeats)
	fmt.Printf("training: optimizer=%s loss=%s lr=%.6f temperature=%.6f weight_bits=%d\n",
		checkpoint.Config.Optimizer,
		checkpoint.Config.ContrastiveLoss,
		checkpoint.Config.LearningRate,
		checkpoint.Config.Temperature,
		checkpoint.Config.WeightBits,
	)
	fmt.Printf("weights: %s\n", paths.WeightFilePath)
	fmt.Printf("checkpoint: %s\n", paths.CheckpointPath)
	fmt.Printf("profile: %s\n", paths.TrainProfilePath)
	return nil
}

func runInitTrain(args []string) error {
	fs := flag.NewFlagSet("init-train", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	var manifestPath string
	var seed int64
	var learningRate float64
	var weightDecay float64
	var weightBits int
	var optimizer string
	var contrastiveLoss string
	var beta1 float64
	var beta2 float64
	var epsilon float64
	var temperature float64
	var dims dimFlag
	fs.StringVar(&manifestPath, "manifest", "", "path to embedding manifest (defaults to sibling .embedding.mll)")
	fs.Int64Var(&seed, "seed", 1, "initialization seed")
	fs.Float64Var(&learningRate, "lr", 0, "trainer learning rate")
	fs.Float64Var(&weightDecay, "weight-decay", 0, "trainer weight decay")
	fs.IntVar(&weightBits, "weight-bits", 0, "forward fake-quant bits")
	fs.StringVar(&optimizer, "optimizer", "", "optimizer name")
	fs.StringVar(&contrastiveLoss, "contrastive-loss", "", "contrastive loss: pair_mse or infonce")
	fs.Float64Var(&beta1, "beta1", 0, "optimizer beta1")
	fs.Float64Var(&beta2, "beta2", 0, "optimizer beta2")
	fs.Float64Var(&epsilon, "epsilon", 0, "optimizer epsilon")
	fs.Float64Var(&temperature, "temperature", 0, "contrastive softmax temperature")
	fs.Var(&dims, "dim", "symbolic dimension binding in NAME=VALUE form; repeatable")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 1 || fs.Arg(0) == "" {
		return fmt.Errorf("usage: barr init-train [flags] <artifact.mll>")
	}
	path := fs.Arg(0)
	cfg := barruntime.EmbeddingTrainConfig{
		LearningRate:    float32(learningRate),
		WeightDecay:     float32(weightDecay),
		WeightBits:      weightBits,
		Optimizer:       optimizer,
		Beta1:           float32(beta1),
		Beta2:           float32(beta2),
		Epsilon:         float32(epsilon),
		ContrastiveLoss: contrastiveLoss,
		Temperature:     float32(temperature),
	}
	opts := barruntime.EmbeddingTrainInitOptions{
		Seed:       seed,
		ShapeSizes: dims.values(),
	}
	var (
		paths barruntime.EmbeddingTrainPackagePaths
		err   error
	)
	if manifestPath == "" {
		manifestPath = barruntime.ResolveEmbeddingManifestPath(path)
	}
	manifest, readErr := barruntime.ReadEmbeddingManifestFile(manifestPath)
	if readErr != nil {
		return readErr
	}
	paths, err = barruntime.InitializeEmbeddingTrainerPackageWithManifest(path, manifest, cfg, opts)
	if err != nil {
		return err
	}
	fmt.Printf("initialized training package %q\n", path)
	fmt.Printf("embedding manifest: %s\n", paths.EmbeddingManifestPath)
	fmt.Printf("weights: %s\n", paths.WeightFilePath)
	fmt.Printf("train manifest: %s\n", paths.TrainManifestPath)
	fmt.Printf("checkpoint: %s\n", paths.CheckpointPath)
	fmt.Printf("profile: %s\n", paths.TrainProfilePath)
	return nil
}

func runTrainEmbed(args []string) error {
	fs := flag.NewFlagSet("train-embed", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	var epochs int
	var batchSize int
	var shuffle bool
	var seed int64
	var evalEvery int
	var patience int
	var selectMetric string
	var minDelta float64
	var restoreBest bool
	var lengthBucketBatches bool
	var progressEvery int
	var tokenizerPath string
	var planOnly bool
	var learningRate float64
	var contrastiveLoss string
	var temperature float64
	fs.IntVar(&epochs, "epochs", 10, "number of epochs")
	fs.IntVar(&batchSize, "batch-size", 8, "batch size")
	fs.BoolVar(&shuffle, "shuffle", true, "shuffle training set each epoch")
	fs.Int64Var(&seed, "seed", 1, "shuffle seed")
	fs.IntVar(&evalEvery, "eval-every", 1, "evaluate every N epochs")
	fs.IntVar(&patience, "patience", 3, "early stopping patience in evals")
	fs.StringVar(&selectMetric, "select-metric", "top1_accuracy", "selection metric: top1_accuracy, score_margin, pair_accuracy, or loss")
	fs.Float64Var(&minDelta, "min-delta", 0, "minimum eval improvement to count as better")
	fs.BoolVar(&restoreBest, "restore-best", true, "restore best checkpoint at end")
	fs.BoolVar(&lengthBucketBatches, "length-bucket-batches", false, "cluster contrastive batches by token length to improve batched GPU training")
	fs.IntVar(&progressEvery, "progress-every", 0, "print training progress every N optimizer steps (0 disables)")
	fs.StringVar(&tokenizerPath, "tokenizer", "", "path to tokenizer JSON for text-pair datasets")
	fs.BoolVar(&planOnly, "plan-only", false, "print planned workload and exit without training")
	fs.Float64Var(&learningRate, "lr", 0, "override package learning rate for this run")
	fs.StringVar(&contrastiveLoss, "contrastive-loss", "", "override package contrastive loss: pair_mse or infonce")
	fs.Float64Var(&temperature, "temperature", 0, "override package contrastive softmax temperature")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 2 || fs.Arg(0) == "" || fs.Arg(1) == "" {
		return fmt.Errorf("usage: barr train-embed [flags] <artifact.mll> <train.jsonl> [eval.jsonl]")
	}
	if learningRate < 0 {
		return fmt.Errorf("lr must be non-negative")
	}
	if temperature < 0 {
		return fmt.Errorf("temperature must be non-negative")
	}
	if progressEvery < 0 {
		return fmt.Errorf("progress-every must be non-negative")
	}
	path := fs.Arg(0)
	trainPath := fs.Arg(1)
	evalPath := ""
	if fs.NArg() > 2 {
		evalPath = fs.Arg(2)
	}
	if tokenizerPath == "" {
		defaultTokenizerPath := barruntime.DefaultTokenizerPath(path)
		if _, err := os.Stat(defaultTokenizerPath); err == nil {
			tokenizerPath = defaultTokenizerPath
		}
	}
	runConfig := barruntime.EmbeddingTrainRunConfig{
		Epochs:                epochs,
		BatchSize:             batchSize,
		Shuffle:               shuffle,
		Seed:                  seed,
		EvalEveryEpoch:        evalEvery,
		EarlyStoppingPatience: patience,
		SelectMetric:          selectMetric,
		MinDelta:              float32(minDelta),
		RestoreBest:           restoreBest,
		LengthBucketBatches:   lengthBucketBatches,
		LearningRate:          float32(learningRate),
		ContrastiveLoss:       contrastiveLoss,
		Temperature:           float32(temperature),
		ProgressEverySteps:    progressEvery,
	}
	if progressEvery > 0 {
		runConfig.Progress = printTrainProgress
	}
	workload, workloadErr := estimateTrainEmbedWorkload(tokenizerPath, trainPath, evalPath, runConfig)
	if workloadErr == nil {
		fmt.Printf("planned workload: %s\n", formatTrainWorkload(workload))
	}
	if planOnly {
		if workloadErr != nil {
			return workloadErr
		}
		return nil
	}
	var (
		summary barruntime.EmbeddingTrainRunSummary
		paths   barruntime.EmbeddingTrainPackagePaths
		err     error
	)
	if tokenizerPath != "" {
		summary, paths, err = barruntime.TrainEmbeddingPackageFromTextContrastiveFiles(path, tokenizerPath, trainPath, evalPath, runConfig)
	} else {
		summary, paths, err = barruntime.TrainEmbeddingPackageFromContrastiveFiles(path, trainPath, evalPath, runConfig)
	}
	if err != nil {
		return err
	}
	fmt.Printf("trained package %q\n", path)
	if tokenizerPath != "" {
		fmt.Printf("tokenizer: %s\n", tokenizerPath)
	}
	fmt.Printf("epochs: %d, steps: %d, run_steps: %d, best_epoch: %d, best_step: %d\n", summary.EpochsCompleted, summary.StepsCompleted, summary.StepsRun, summary.BestEpoch, summary.BestStep)
	fmt.Printf("final train: loss=%.6f avg_score=%.6f batch=%d\n", summary.FinalTrain.Loss, summary.FinalTrain.AverageScore, summary.FinalTrain.BatchSize)
	if summary.FinalEval != nil {
		fmt.Printf("final eval: loss=%.6f margin=%.6f accuracy=%.6f top1=%.6f mean_rank=%.3f pairs=%d\n", summary.FinalEval.Loss, summary.FinalEval.ScoreMargin, summary.FinalEval.PairAccuracy, summary.FinalEval.Top1Accuracy, summary.FinalEval.MeanPositiveRank, summary.FinalEval.PairCount)
	}
	fmt.Printf("workload: %s\n", formatTrainWorkload(summary.Workload))
	fmt.Printf("throughput: %s\n", formatTrainThroughput(summary))
	fmt.Printf("accelerators: forward=%s optimizer=%s activation=%s contrastive=%s\n",
		displayTrainBackend(summary.EndProfile.ForwardBackend),
		displayTrainBackend(summary.EndProfile.OptimizerBackend),
		displayTrainBackend(summary.EndProfile.ActivationBackend),
		displayTrainBackend(summary.EndProfile.ContrastiveBackend),
	)
	fmt.Printf("profile delta: matmul_bind_calls=%d matmul_runs=%d matmul_run_upload_mb=%.2f matmul_run_download_mb=%.2f optimizer_updates=%d activation_calls=%d contrastive_calls=%d\n",
		summary.DeltaProfile.ForwardResidency.MatMul.BindCalls,
		summary.DeltaProfile.ForwardResidency.MatMul.RunCalls,
		float64(summary.DeltaProfile.ForwardResidency.MatMul.RunUploadedBytes)/(1024*1024),
		float64(summary.DeltaProfile.ForwardResidency.MatMul.RunDownloadedBytes)/(1024*1024),
		summary.DeltaProfile.Optimizer.UpdateCalls,
		summary.DeltaProfile.Activation.GELUBackwardCalls+summary.DeltaProfile.Activation.SoftmaxBackwardCalls+summary.DeltaProfile.Activation.LayerNormBackwardCalls,
		summary.DeltaProfile.Contrastive.RunCalls,
	)
	fmt.Printf("checkpoint: %s\n", paths.CheckpointPath)
	fmt.Printf("profile: %s\n", paths.TrainProfilePath)
	return nil
}

func estimateTrainEmbedWorkload(tokenizerPath, trainPath, evalPath string, cfg barruntime.EmbeddingTrainRunConfig) (barruntime.EmbeddingTrainWorkload, error) {
	if tokenizerPath != "" {
		trainSet, err := barruntime.ReadEmbeddingTextContrastiveExamplesFile(trainPath)
		if err != nil {
			return barruntime.EmbeddingTrainWorkload{}, err
		}
		if evalPath == "" {
			return barruntime.EstimateContrastiveTrainWorkload(len(trainSet), 0, cfg), nil
		}
		evalPairs, err := barruntime.ReadEmbeddingTextPairExamplesFile(evalPath)
		if err != nil {
			return barruntime.EmbeddingTrainWorkload{}, err
		}
		allPositive := true
		positiveCount := 0
		for _, example := range evalPairs {
			if example.Target > 0 {
				positiveCount++
			} else {
				allPositive = false
			}
		}
		if allPositive {
			return barruntime.EstimateContrastiveTrainWorkload(len(trainSet), positiveCount, cfg), nil
		}
		workload := barruntime.EstimateContrastiveTrainWorkload(len(trainSet), 0, cfg)
		workload.EvalMode = "pairwise"
		workload.EvalExamples = len(evalPairs)
		workload.EvalPairsPerPass = int64(len(evalPairs))
		workload.PlannedEvalPasses = 1
		workload.PlannedEvalPairs = int64(len(evalPairs))
		workload.PlannedTotalPairs = workload.PlannedTrainPairs + workload.PlannedEvalPairs
		return workload, nil
	}

	trainSet, err := barruntime.ReadEmbeddingContrastiveExamplesFile(trainPath)
	if err != nil {
		return barruntime.EmbeddingTrainWorkload{}, err
	}
	evalCount := 0
	if evalPath != "" {
		evalSet, err := barruntime.ReadEmbeddingContrastiveExamplesFile(evalPath)
		if err != nil {
			return barruntime.EmbeddingTrainWorkload{}, err
		}
		evalCount = len(evalSet)
	}
	return barruntime.EstimateContrastiveTrainWorkload(len(trainSet), evalCount, cfg), nil
}

func formatTrainWorkload(workload barruntime.EmbeddingTrainWorkload) string {
	parts := []string{
		fmt.Sprintf("train=%d %s examples", workload.TrainExamples, workload.TrainMode),
		fmt.Sprintf("batch=%d", workload.BatchSize),
		fmt.Sprintf("steps/epoch=%d", workload.TrainBatchesPerEpoch),
		fmt.Sprintf("train_pairs/epoch=%d", workload.TrainPairsPerEpoch),
	}
	if workload.EvalMode != "" {
		parts = append(parts,
			fmt.Sprintf("eval=%d %s examples", workload.EvalExamples, workload.EvalMode),
			fmt.Sprintf("eval_pairs/pass=%d", workload.EvalPairsPerPass),
			fmt.Sprintf("eval_passes(planned=%d actual=%d)", workload.PlannedEvalPasses, workload.ActualEvalPasses),
		)
	}
	parts = append(parts,
		fmt.Sprintf("pairs(planned=%d actual=%d)", workload.PlannedTotalPairs, workload.ActualTotalPairs),
	)
	return strings.Join(parts, " ")
}

func formatTrainThroughput(summary barruntime.EmbeddingTrainRunSummary) string {
	parts := []string{fmt.Sprintf("elapsed=%s", summary.Elapsed.Round(time.Millisecond))}
	if rate := itemsPerSecond(summary.Workload.ActualTotalExamples, summary.Elapsed); rate > 0 {
		parts = append(parts, fmt.Sprintf("examples/s=%.2f", rate))
	}
	if rate := pairsPerSecond(summary.Workload.ActualTotalPairs, summary.Elapsed); rate > 0 {
		parts = append(parts, fmt.Sprintf("pairs/s=%.2f", rate))
	}
	if rate := itemsPerSecond(summary.Workload.ActualTrainExamples, summary.TrainDuration); rate > 0 {
		parts = append(parts, fmt.Sprintf("train_examples/s=%.2f", rate))
	}
	if rate := pairsPerSecond(summary.Workload.ActualTrainPairs, summary.TrainDuration); rate > 0 {
		parts = append(parts, fmt.Sprintf("train_pairs/s=%.2f", rate))
	}
	if rate := itemsPerSecond(summary.Workload.ActualEvalExamples, summary.EvalDuration); rate > 0 {
		parts = append(parts, fmt.Sprintf("eval_examples/s=%.2f", rate))
	}
	if rate := pairsPerSecond(summary.Workload.ActualEvalPairs, summary.EvalDuration); rate > 0 {
		parts = append(parts, fmt.Sprintf("eval_pairs/s=%.2f", rate))
	}
	if rate := itemsPerSecond(int64(summary.StepsRun), summary.TrainDuration); rate > 0 {
		parts = append(parts, fmt.Sprintf("optimizer_steps/s=%.2f", rate))
	}
	return strings.Join(parts, " ")
}

func printTrainProgress(progress barruntime.EmbeddingTrainProgress) {
	epochPairs := fmt.Sprintf("%d", progress.EpochTrainPairs)
	if progress.PlannedEpochPairs > 0 {
		epochPairs = fmt.Sprintf("%d/%d", progress.EpochTrainPairs, progress.PlannedEpochPairs)
	}
	fmt.Printf(
		"progress: epoch=%d batch=%d/%d step=%d loss=%.6f avg_score=%.6f batch_examples=%d batch_pairs=%d epoch_examples=%d epoch_pairs=%s elapsed=%s\n",
		progress.Epoch,
		progress.Batch,
		progress.Batches,
		progress.Step,
		progress.Loss,
		progress.AverageScore,
		progress.BatchExamples,
		progress.BatchPairs,
		progress.EpochTrainExamples,
		epochPairs,
		progress.Elapsed.Round(time.Millisecond),
	)
}

func itemsPerSecond(items int64, elapsed time.Duration) float64 {
	if items <= 0 || elapsed <= 0 {
		return 0
	}
	return float64(items) / elapsed.Seconds()
}

func pairsPerSecond(pairs int64, elapsed time.Duration) float64 {
	if pairs <= 0 || elapsed <= 0 {
		return 0
	}
	return float64(pairs) / elapsed.Seconds()
}

func runTrainCorpus(args []string) error {
	fs := flag.NewFlagSet("train-corpus", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	var epochs int
	var batchSize int
	var shuffle bool
	var seed int64
	var evalEvery int
	var patience int
	var selectMetric string
	var minDelta float64
	var restoreBest bool
	var lengthBucketBatches bool
	var progressEvery int
	var tokenizerPath string
	var vocabSize int
	var minFreq int
	var trainPairsPath string
	var evalPairsPath string
	var minChars int
	var maxPairs int
	var evalPairs int
	var learningRate float64
	var contrastiveLoss string
	var temperature float64
	fs.IntVar(&epochs, "epochs", 10, "number of epochs")
	fs.IntVar(&batchSize, "batch-size", 8, "batch size")
	fs.BoolVar(&shuffle, "shuffle", true, "shuffle training set each epoch")
	fs.Int64Var(&seed, "seed", 1, "shuffle/mining seed")
	fs.IntVar(&evalEvery, "eval-every", 1, "evaluate every N epochs")
	fs.IntVar(&patience, "patience", 3, "early stopping patience in evals")
	fs.StringVar(&selectMetric, "select-metric", "top1_accuracy", "selection metric: top1_accuracy, score_margin, pair_accuracy, or loss")
	fs.Float64Var(&minDelta, "min-delta", 0, "minimum eval improvement to count as better")
	fs.BoolVar(&restoreBest, "restore-best", true, "restore best checkpoint at end")
	fs.BoolVar(&lengthBucketBatches, "length-bucket-batches", false, "cluster contrastive batches by token length to improve batched GPU training")
	fs.IntVar(&progressEvery, "progress-every", 0, "print training progress every N optimizer steps (0 disables)")
	fs.StringVar(&tokenizerPath, "tokenizer", "", "output tokenizer path")
	fs.IntVar(&vocabSize, "vocab-size", 0, "tokenizer vocab size override")
	fs.IntVar(&minFreq, "min-freq", 2, "minimum pair frequency for tokenizer merges")
	fs.StringVar(&trainPairsPath, "train-pairs", "", "output mined train pair dataset path")
	fs.StringVar(&evalPairsPath, "eval-pairs-path", "", "output mined eval pair dataset path")
	fs.IntVar(&minChars, "min-chars", 8, "minimum normalized text length for mined segments")
	fs.IntVar(&maxPairs, "max-pairs", 0, "maximum number of positive training pairs to keep (0 = all)")
	fs.IntVar(&evalPairs, "eval-pairs", 32, "number of positive pairs to hold out for eval")
	fs.Float64Var(&learningRate, "lr", 0, "override package learning rate for this run")
	fs.StringVar(&contrastiveLoss, "contrastive-loss", "", "override package contrastive loss: pair_mse or infonce")
	fs.Float64Var(&temperature, "temperature", 0, "override package contrastive softmax temperature")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 2 || fs.Arg(0) == "" || fs.Arg(1) == "" {
		return fmt.Errorf("usage: barr train-corpus [flags] <artifact.mll> <corpus.txt>")
	}
	if learningRate < 0 {
		return fmt.Errorf("lr must be non-negative")
	}
	if temperature < 0 {
		return fmt.Errorf("temperature must be non-negative")
	}
	if progressEvery < 0 {
		return fmt.Errorf("progress-every must be non-negative")
	}
	path := fs.Arg(0)
	corpusPath := fs.Arg(1)
	runConfig := barruntime.EmbeddingTrainRunConfig{
		Epochs:                epochs,
		BatchSize:             batchSize,
		Shuffle:               shuffle,
		Seed:                  seed,
		EvalEveryEpoch:        evalEvery,
		EarlyStoppingPatience: patience,
		SelectMetric:          selectMetric,
		MinDelta:              float32(minDelta),
		RestoreBest:           restoreBest,
		LengthBucketBatches:   lengthBucketBatches,
		LearningRate:          float32(learningRate),
		ContrastiveLoss:       contrastiveLoss,
		Temperature:           float32(temperature),
		ProgressEverySteps:    progressEvery,
	}
	if progressEvery > 0 {
		runConfig.Progress = printTrainProgress
	}
	summary, paths, err := barruntime.TrainEmbeddingPackageFromCorpusFile(path, corpusPath, barruntime.EmbeddingCorpusTrainConfig{
		TokenizerPath:      tokenizerPath,
		TokenizerVocabSize: vocabSize,
		TokenizerMinFreq:   minFreq,
		TrainPairsPath:     trainPairsPath,
		EvalPairsPath:      evalPairsPath,
		Mining: barruntime.EmbeddingTextMiningConfig{
			MinChars:  minChars,
			MaxPairs:  maxPairs,
			EvalPairs: evalPairs,
			Seed:      seed,
		},
		Run: runConfig,
	})
	if err != nil {
		return err
	}
	fmt.Printf("trained package %q from corpus\n", path)
	fmt.Printf("tokenizer: %s\n", paths.TokenizerPath)
	fmt.Printf("train pairs: %s\n", paths.TrainPairsPath)
	if paths.EvalPairsPath != "" {
		fmt.Printf("eval pairs: %s\n", paths.EvalPairsPath)
	}
	fmt.Printf("epochs: %d, steps: %d, run_steps: %d, best_epoch: %d, best_step: %d\n", summary.EpochsCompleted, summary.StepsCompleted, summary.StepsRun, summary.BestEpoch, summary.BestStep)
	fmt.Printf("final train: loss=%.6f avg_score=%.6f batch=%d\n", summary.FinalTrain.Loss, summary.FinalTrain.AverageScore, summary.FinalTrain.BatchSize)
	if summary.FinalEval != nil {
		fmt.Printf("final eval: loss=%.6f margin=%.6f accuracy=%.6f top1=%.6f mean_rank=%.3f pairs=%d\n", summary.FinalEval.Loss, summary.FinalEval.ScoreMargin, summary.FinalEval.PairAccuracy, summary.FinalEval.Top1Accuracy, summary.FinalEval.MeanPositiveRank, summary.FinalEval.PairCount)
	}
	fmt.Printf("workload: %s\n", formatTrainWorkload(summary.Workload))
	fmt.Printf("throughput: %s\n", formatTrainThroughput(summary))
	fmt.Printf("accelerators: forward=%s optimizer=%s activation=%s contrastive=%s\n",
		displayTrainBackend(summary.EndProfile.ForwardBackend),
		displayTrainBackend(summary.EndProfile.OptimizerBackend),
		displayTrainBackend(summary.EndProfile.ActivationBackend),
		displayTrainBackend(summary.EndProfile.ContrastiveBackend),
	)
	fmt.Printf("profile delta: matmul_bind_calls=%d matmul_runs=%d matmul_run_upload_mb=%.2f matmul_run_download_mb=%.2f optimizer_updates=%d activation_calls=%d contrastive_calls=%d\n",
		summary.DeltaProfile.ForwardResidency.MatMul.BindCalls,
		summary.DeltaProfile.ForwardResidency.MatMul.RunCalls,
		float64(summary.DeltaProfile.ForwardResidency.MatMul.RunUploadedBytes)/(1024*1024),
		float64(summary.DeltaProfile.ForwardResidency.MatMul.RunDownloadedBytes)/(1024*1024),
		summary.DeltaProfile.Optimizer.UpdateCalls,
		summary.DeltaProfile.Activation.GELUBackwardCalls+summary.DeltaProfile.Activation.SoftmaxBackwardCalls+summary.DeltaProfile.Activation.LayerNormBackwardCalls,
		summary.DeltaProfile.Contrastive.RunCalls,
	)
	fmt.Printf("checkpoint: %s\n", paths.Package.CheckpointPath)
	fmt.Printf("profile: %s\n", paths.Package.TrainProfilePath)
	return nil
}

func runTrainTokenizer(args []string) error {
	fs := flag.NewFlagSet("train-tokenizer", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	var outputPath string
	var manifestPath string
	var vocabSize int
	var minFreq int
	fs.StringVar(&outputPath, "output", "", "output tokenizer path")
	fs.StringVar(&manifestPath, "manifest", "", "embedding manifest path")
	fs.IntVar(&vocabSize, "vocab-size", 0, "tokenizer vocab size override")
	fs.IntVar(&minFreq, "min-freq", 2, "minimum pair frequency for merges")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 2 || fs.Arg(0) == "" || fs.Arg(1) == "" {
		return fmt.Errorf("usage: barr train-tokenizer [flags] <artifact.mll> <corpus.txt>")
	}
	artifactPath := fs.Arg(0)
	corpusPath := fs.Arg(1)
	if outputPath == "" {
		outputPath = barruntime.DefaultTokenizerPath(artifactPath)
	}
	if manifestPath == "" {
		manifestPath = barruntime.DefaultEmbeddingManifestPath(artifactPath)
	}
	if vocabSize == 0 {
		manifest, err := barruntime.ReadEmbeddingManifestFile(manifestPath)
		if err != nil {
			return fmt.Errorf("read embedding manifest for vocab size: %w", err)
		}
		vocabSize = manifest.Tokenizer.VocabSize
	}
	if vocabSize <= 0 {
		return fmt.Errorf("tokenizer vocab size must be set via --vocab-size or embedding manifest")
	}
	tokenizer, err := barruntime.TrainTokenizerFromCorpus(barruntime.TokenizerTrainConfig{
		CorpusPath: corpusPath,
		VocabSize:  vocabSize,
		MinFreq:    minFreq,
	})
	if err != nil {
		return err
	}
	if err := tokenizer.WriteFile(outputPath); err != nil {
		return err
	}
	if err := barruntime.SyncEmbeddingTokenizerVocab(artifactPath, len(tokenizer.Tokens)); err != nil {
		return fmt.Errorf("sync tokenizer vocab through Barracuda package: %w", err)
	}
	fmt.Printf("trained tokenizer %q\n", outputPath)
	fmt.Printf("vocab: %d tokens, merges: %d\n", len(tokenizer.Tokens), len(tokenizer.Merges))
	return nil
}

func runMineTextPairs(args []string) error {
	fs := flag.NewFlagSet("mine-text-pairs", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	var minChars int
	var maxPairs int
	var evalPairs int
	var seed int64
	fs.IntVar(&minChars, "min-chars", 8, "minimum normalized text length for mined segments")
	fs.IntVar(&maxPairs, "max-pairs", 0, "maximum number of positive training pairs to keep (0 = all)")
	fs.IntVar(&evalPairs, "eval-pairs", 32, "number of positive pairs to hold out for eval")
	fs.Int64Var(&seed, "seed", 1, "shuffle seed used when truncating mined pairs")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 2 || fs.Arg(0) == "" || fs.Arg(1) == "" {
		return fmt.Errorf("usage: barr mine-text-pairs [flags] <corpus.txt> <train.jsonl> [eval.jsonl]")
	}
	corpusPath := fs.Arg(0)
	trainPath := fs.Arg(1)
	evalPath := ""
	if fs.NArg() > 2 {
		evalPath = fs.Arg(2)
	}
	trainSet, evalSet, err := barruntime.MineEmbeddingTextDatasetsFromCorpusFile(corpusPath, barruntime.EmbeddingTextMiningConfig{
		MinChars:  minChars,
		MaxPairs:  maxPairs,
		EvalPairs: evalPairs,
		Seed:      seed,
	})
	if err != nil {
		return err
	}
	if err := barruntime.WriteEmbeddingTextContrastiveExamplesFile(trainPath, trainSet); err != nil {
		return err
	}
	if evalPath != "" && len(evalSet) > 0 {
		if err := barruntime.WriteEmbeddingTextPairExamplesFile(evalPath, evalSet); err != nil {
			return err
		}
	}
	fmt.Printf("mined train pairs: %d\n", len(trainSet))
	if evalPath != "" {
		fmt.Printf("mined eval pairs: %d\n", len(evalSet))
	}
	fmt.Printf("train: %s\n", trainPath)
	if evalPath != "" {
		fmt.Printf("eval: %s\n", evalPath)
	}
	return nil
}

func printUsage() {
	fmt.Println("usage:")
	fmt.Println("  barr version")
	fmt.Println("  barr compile <source.bar> [output.mll]")
	fmt.Println("  barr inspect <artifact.mll>")
	fmt.Println("  barr export-mll <artifact.mll> [output.mll]")
	fmt.Println("  barr init-model [flags] <artifact.mll>")
	fmt.Println("  barr init-train [flags] <artifact.mll>")
	fmt.Println("  barr train-tokenizer [flags] <artifact.mll> <corpus.txt>")
	fmt.Println("  barr train-corpus [flags] <artifact.mll> <corpus.txt>")
	fmt.Println("  barr train-embed [flags] <artifact.mll> <train.jsonl> [eval.jsonl]")
	fmt.Println("  barr run <artifact.mll> [entry]")
	fmt.Println("  barr demo [module-name]")
	fmt.Println()
	fmt.Println("compile lowers a .bar source file into an .mll Barracuda artifact.")
	fmt.Println("inspect summarizes an artifact and verifies its sibling package manifest when present.")
	fmt.Println("export-mll seals an artifact package into a weight-carrying .mll container while preserving Barracuda metadata in XBAR.")
	fmt.Println("init-model creates the Barracuda-owned default quantized embedding training package.")
	fmt.Println("init-train creates a native training package next to an artifact.")
	fmt.Println("train-tokenizer builds a sibling .tokenizer.mll from a raw text corpus, using embedding-manifest vocab_size by default.")
	fmt.Println("train-corpus trains tokenizer + mined text pairs + embedder in one Barracuda job from a raw text corpus.")
	fmt.Println("train-embed reloads a training package, fits it on token JSONL or text JSONL (with --tokenizer or a sibling .tokenizer.mll), and writes it back.")
	fmt.Println("run loads an artifact, binds stub weights and inputs, and executes one entrypoint.")
	fmt.Println("demo creates a tiny inference-style module and loads it through the runtime.")
}

func totalKernelOps(kernels []barr.Kernel) int {
	total := 0
	for _, kernel := range kernels {
		total += len(kernel.Body)
	}
	return total
}

func defaultArtifactPath(srcPath string) string {
	ext := filepath.Ext(srcPath)
	if ext == "" {
		return srcPath + ".mll"
	}
	return strings.TrimSuffix(srcPath, ext) + ".mll"
}

func stubLoadOptions(mod *barr.Module) []barruntime.LoadOption {
	sizes := defaultSymbolSizes(mod)
	opts := make([]barruntime.LoadOption, 0, len(mod.Params))
	for _, param := range mod.Params {
		opts = append(opts, barruntime.WithWeight(param.Name, stubTensorForParam(param.Name, param.Type, sizes)))
	}
	return opts
}

func defaultEntryName(mod *barr.Module) string {
	if mod != nil && len(mod.EntryPoints) > 0 {
		return mod.EntryPoints[0].Name
	}
	return ""
}

func entryPointByName(mod *barr.Module, name string) (barr.EntryPoint, error) {
	for _, entry := range mod.EntryPoints {
		if entry.Name == name {
			return entry, nil
		}
	}
	return barr.EntryPoint{}, fmt.Errorf("unknown entrypoint %q", name)
}

func stubInputs(entry barr.EntryPoint) map[string]any {
	sizes := defaultShapeSizes(entry)
	out := make(map[string]any, len(entry.Inputs))
	for _, input := range entry.Inputs {
		if input.Type.Kind == barr.ValueKVCache {
			out[input.Name] = backend.NewKVCache(backend.NewTensorF16([]int{sizes["T"], sizes["D"]}, make([]float32, sizes["T"]*sizes["D"])))
			continue
		}
		out[input.Name] = stubTensorForInput(input.Name, input.Type, sizes)
	}
	return out
}

func displayTrainBackend(kind barr.BackendKind) string {
	if kind == "" {
		return "host"
	}
	return string(kind)
}

func displayManifestName(value string) string {
	if value == "" {
		return "-"
	}
	return value
}

func joinBackendKinds(kinds []barr.BackendKind) string {
	if len(kinds) == 0 {
		return ""
	}
	out := make([]string, 0, len(kinds))
	for _, kind := range kinds {
		out = append(out, string(kind))
	}
	return strings.Join(out, ", ")
}

func sortedValueKeys[V any](m map[string]V) []string {
	keys := make([]string, 0, len(m))
	for key := range m {
		keys = append(keys, key)
	}
	slices.Sort(keys)
	return keys
}

func outputSummaries(outputs map[string]backend.Value) []string {
	keys := sortedValueKeys(outputs)
	out := make([]string, 0, len(keys))
	for _, key := range keys {
		value := outputs[key]
		tensor, ok := value.Data.(*backend.Tensor)
		if ok && tensor != nil {
			out = append(out, fmt.Sprintf("%s=%s%v", key, tensor.DType, tensor.Shape))
			continue
		}
		if pack, ok := value.Data.(*backend.CandidatePack); ok && pack != nil {
			out = append(out, fmt.Sprintf("%s=candidate_pack(ids=i64%v,scores=f32%v,docs=%s%v)", key, pack.IDs.Shape, pack.Scores.Shape, pack.Docs.DType, pack.Docs.Shape))
			continue
		}
		out = append(out, key+"=<non-tensor>")
	}
	return out
}

type dimFlag struct {
	pairs []string
}

func (f *dimFlag) String() string {
	return strings.Join(f.pairs, ",")
}

func (f *dimFlag) Set(value string) error {
	if value == "" {
		return fmt.Errorf("empty dimension binding")
	}
	parts := strings.SplitN(value, "=", 2)
	if len(parts) != 2 || parts[0] == "" || parts[1] == "" {
		return fmt.Errorf("invalid dimension binding %q, want NAME=VALUE", value)
	}
	if _, err := strconv.Atoi(parts[1]); err != nil {
		return fmt.Errorf("invalid dimension binding %q: %w", value, err)
	}
	f.pairs = append(f.pairs, value)
	return nil
}

func (f *dimFlag) values() map[string]int {
	if len(f.pairs) == 0 {
		return nil
	}
	out := make(map[string]int, len(f.pairs))
	for _, pair := range f.pairs {
		parts := strings.SplitN(pair, "=", 2)
		if len(parts) != 2 {
			continue
		}
		n, err := strconv.Atoi(parts[1])
		if err != nil {
			continue
		}
		out[parts[0]] = n
	}
	return out
}

func defaultSymbolSizes(mod *barr.Module) map[string]int {
	sizes := map[string]int{
		"V":  3,
		"D":  2,
		"E":  2,
		"T":  2,
		"N":  2,
		"K":  2,
		"H":  2,
		"HD": 2,
	}
	if mod == nil {
		return sizes
	}
	for _, param := range mod.Params {
		if param.Type.Tensor == nil {
			continue
		}
		for _, dim := range param.Type.Tensor.Shape {
			if _, ok := sizes[dim]; !ok {
				sizes[dim] = 2
			}
		}
	}
	for _, entry := range mod.EntryPoints {
		for _, input := range entry.Inputs {
			if input.Type.Tensor == nil {
				continue
			}
			for _, dim := range input.Type.Tensor.Shape {
				if _, ok := sizes[dim]; !ok {
					sizes[dim] = 2
				}
			}
		}
	}
	return sizes
}

func defaultShapeSizes(entry barr.EntryPoint) map[string]int {
	sizes := map[string]int{
		"V":  3,
		"D":  2,
		"E":  2,
		"T":  2,
		"N":  2,
		"K":  2,
		"H":  2,
		"HD": 2,
	}
	for _, input := range entry.Inputs {
		if input.Type.Tensor == nil {
			continue
		}
		for _, dim := range input.Type.Tensor.Shape {
			if _, ok := sizes[dim]; !ok {
				sizes[dim] = 2
			}
		}
	}
	return sizes
}

func stubTensorForParam(name string, typ barr.ValueType, sizes map[string]int) *backend.Tensor {
	shape := concreteShape(typ, sizes)
	switch name {
	case "token_embedding":
		return backend.NewTensorF16(shape, []float32{
			1, 0,
			0, 1,
			1, 1,
		})
	case "projection", "wq":
		return backend.NewTensorF16(shape, []float32{
			1, 0,
			0, 1,
		})
	case "docs":
		return backend.NewTensorQ4(shape, []float32{
			1, 0,
			0, 1,
		})
	default:
		return fillTensor(typ, shape, 1)
	}
}

func stubTensorForInput(name string, typ barr.ValueType, sizes map[string]int) *backend.Tensor {
	shape := concreteShape(typ, sizes)
	if typ.Tensor != nil && typ.Tensor.DType == "i32" {
		values := make([]int32, product(shape))
		if name == "attention_mask" {
			for i := range values {
				values[i] = 1
			}
			return backend.NewTensorI32(shape, values)
		}
		limit := sizes["V"]
		if limit <= 0 {
			limit = len(values)
		}
		for i := range values {
			values[i] = int32(i % limit)
		}
		return backend.NewTensorI32(shape, values)
	}
	if name == "x" {
		return backend.NewTensorF16(shape, []float32{
			1, 0,
			0, 1,
		})
	}
	if name == "query" {
		values := make([]float32, product(shape))
		if len(values) > 0 {
			values[0] = 1
		}
		return backend.NewTensorF16(shape, values)
	}
	if name == "queries" && typ.Tensor != nil && typ.Tensor.DType == "f16" && len(shape) == 2 && shape[0] == 2 && shape[1] == 2 {
		return backend.NewTensorF16(shape, []float32{
			1, 0,
			0, 1,
		})
	}
	if name == "docs" && typ.Tensor != nil && typ.Tensor.DType == "q4" && len(shape) == 2 && shape[1] == 2 {
		values := []float32{1, 0, 0, 1}
		if shape[0] >= 3 {
			values = []float32{1, 0, 0, 1, 1, 1}
		}
		if product(shape) > len(values) {
			return fillTensor(typ, shape, 0)
		}
		return backend.NewTensorQ4(shape, values[:product(shape)])
	}
	if name == "docs" && typ.Tensor != nil && typ.Tensor.DType == "q4" && len(shape) == 3 && shape[0] == 2 && shape[2] == 2 {
		values := []float32{
			1, 0,
			0, 1,
			1, 1,
			0, 1,
			1, 0,
			1, 1,
		}
		if product(shape) > len(values) {
			return fillTensor(typ, shape, 0)
		}
		return backend.NewTensorQ4(shape, values[:product(shape)])
	}
	if name == "candidate_ids" && typ.Tensor != nil && typ.Tensor.DType == "i64" {
		values := make([]int64, product(shape))
		for i := range values {
			values[i] = int64(1001 + i*1001)
		}
		return backend.NewTensorI64(shape, values)
	}
	if typ.Tensor != nil && typ.Tensor.DType == "i64" {
		values := make([]int64, product(shape))
		for i := range values {
			values[i] = int64(i)
		}
		return backend.NewTensorI64(shape, values)
	}
	return fillTensor(typ, shape, 0)
}

func fillTensor(typ barr.ValueType, shape []int, offset float32) *backend.Tensor {
	n := product(shape)
	switch typ.Tensor.DType {
	case "i32":
		values := make([]int32, n)
		for i := range values {
			values[i] = int32(i)
		}
		return backend.NewTensorI32(shape, values)
	case "i64":
		values := make([]int64, n)
		for i := range values {
			values[i] = int64(i)
		}
		return backend.NewTensorI64(shape, values)
	case "f16":
		values := make([]float32, n)
		for i := range values {
			values[i] = offset + float32(i+1)/10
		}
		return backend.NewTensorF16(shape, values)
	case "q4":
		values := make([]float32, n)
		for i := range values {
			values[i] = offset + float32(i+1)/10
		}
		return backend.NewTensorQ4(shape, values)
	case "q8":
		values := make([]float32, n)
		for i := range values {
			values[i] = offset + float32(i+1)/10
		}
		return backend.NewTensorQ8(shape, values)
	default:
		values := make([]float32, n)
		for i := range values {
			values[i] = offset + float32(i+1)/10
		}
		return backend.NewTensorF32(shape, values)
	}
}

func concreteShape(typ barr.ValueType, sizes map[string]int) []int {
	if typ.Tensor == nil {
		return []int{1}
	}
	shape := make([]int, len(typ.Tensor.Shape))
	for i, dim := range typ.Tensor.Shape {
		n, ok := sizes[dim]
		if !ok {
			n = 2
		}
		shape[i] = n
	}
	return shape
}

func product(shape []int) int {
	if len(shape) == 0 {
		return 1
	}
	n := 1
	for _, dim := range shape {
		n *= dim
	}
	return n
}
