package main

import (
	"context"
	"encoding/json"
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

	mantaartifact "github.com/odvcencio/manta/artifact/manta"
	"github.com/odvcencio/manta/compiler"
	"github.com/odvcencio/manta/models"
	mantaruntime "github.com/odvcencio/manta/runtime"
	"github.com/odvcencio/manta/runtime/backend"
	"github.com/odvcencio/manta/runtime/backends/cuda"
	"github.com/odvcencio/manta/runtime/backends/directml"
	"github.com/odvcencio/manta/runtime/backends/metal"
	"github.com/odvcencio/manta/runtime/backends/vulkan"
	"github.com/odvcencio/manta/runtime/backends/webgpu"
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
	cpuPath := mantaEnv("MANTA_CPU_PROFILE")
	memPath := mantaEnv("MANTA_MEM_PROFILE")
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

func mantaEnv(name string) string {
	if value, ok := os.LookupEnv(name); ok {
		return value
	}
	return ""
}

func run(args []string) error {
	if len(args) == 0 {
		printUsage()
		return nil
	}

	switch args[0] {
	case "version":
		fmt.Println("manta dev")
		return nil
	case "compile":
		return runCompile(args[1:])
	case "run":
		return runArtifact(args[1:])
	case "embed-text":
		return runEmbedText(args[1:])
	case "eval-retrieval":
		return runEvalRetrieval(args[1:])
	case "demo":
		return runDemo(args[1:])
	case "inspect":
		return runInspect(args[1:])
	case "export-mll":
		return runExportMLL(args[1:])
	case "init-model":
		return runInitModel(args[1:])
	case "init-mirage":
		return runInitMirage(args[1:])
	case "init-train":
		return runInitTrain(args[1:])
	case "rename-embed":
		return runRenameEmbed(args[1:])
	case "train-tokenizer":
		return runTrainTokenizer(args[1:])
	case "tokenize-embed":
		return runTokenizeEmbed(args[1:])
	case "mine-text-pairs":
		return runMineTextPairs(args[1:])
	case "train-embed":
		return runTrainEmbed(args[1:])
	case "train-corpus":
		return runTrainCorpus(args[1:])
	case "compare-train-metrics":
		return runCompareTrainMetrics(args[1:])
	case "diagnose-train-metrics":
		return runDiagnoseTrainMetrics(args[1:])
	case "gate-train-metrics":
		return runGateTrainMetrics(args[1:])
	default:
		return fmt.Errorf("unknown subcommand %q", args[0])
	}
}

func runCompile(args []string) error {
	if len(args) == 0 || args[0] == "" {
		return fmt.Errorf("usage: manta compile <source.manta> [output.mll]")
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
	if err := mantaartifact.WriteFile(outPath, bundle.Artifact); err != nil {
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
	case "mirage_v1":
		mod, err := models.DefaultMirageV1Module(models.MirageV1Config{})
		if err != nil {
			return err
		}
		return runDemoModule(mod)
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

	rt := mantaruntime.New(cuda.New(), metal.New(), vulkan.New(), directml.New(), webgpu.New())
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
	fmt.Printf("artifact version: %s\n", displayArtifactVersion(bundle.Artifact.Version))
	fmt.Printf("ran entrypoint: %s\n", entryName)
	fmt.Printf("outputs: %s\n", strings.Join(sortedValueKeys(result.Outputs), ", "))
	fmt.Printf("output summary: %s\n", strings.Join(outputSummaries(result.Outputs), "; "))
	fmt.Printf("trace steps: %d\n", len(result.Trace))
	return nil
}

func runDemoModule(mod *mantaartifact.Module) error {
	rt := mantaruntime.New(cuda.New(), metal.New(), vulkan.New(), directml.New(), webgpu.New())
	prog, err := rt.Load(context.Background(), mod, stubLoadOptions(mod)...)
	if err != nil {
		return err
	}
	entryName := defaultEntryName(mod)
	entry, err := entryPointByName(mod, entryName)
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
	fmt.Printf("loaded module %q for backend %q\n", mod.Name, prog.Backend())
	fmt.Printf("entrypoints: %d, steps: %d, kernels: %d\n", len(mod.EntryPoints), len(mod.Steps), len(mod.Kernels))
	fmt.Printf("kernel ops: %d\n", totalKernelOps(mod.Kernels))
	fmt.Printf("params: %d\n", len(mod.Params))
	fmt.Printf("artifact version: %s\n", displayArtifactVersion(mod.Version))
	fmt.Printf("ran entrypoint: %s\n", entryName)
	fmt.Printf("outputs: %s\n", strings.Join(sortedValueKeys(result.Outputs), ", "))
	fmt.Printf("output summary: %s\n", strings.Join(outputSummaries(result.Outputs), "; "))
	fmt.Printf("trace steps: %d\n", len(result.Trace))
	return nil
}

func runArtifact(args []string) error {
	if len(args) == 0 || args[0] == "" {
		return fmt.Errorf("usage: manta run <artifact.mll> [entry]")
	}
	path := args[0]
	mod, err := mantaartifact.ReadFile(path)
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
	rt := mantaruntime.New(cuda.New(), metal.New(), vulkan.New(), directml.New(), webgpu.New())
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

func runEmbedText(args []string) error {
	if len(args) < 2 || args[0] == "" {
		return fmt.Errorf("usage: manta embed-text <artifact.mll> <text...>")
	}
	path := args[0]
	text := strings.Join(args[1:], " ")
	rt := mantaruntime.New(cuda.New(), metal.New(), vulkan.New(), directml.New(), webgpu.New())
	model, err := rt.LoadEmbeddingPackage(context.Background(), path)
	if err != nil {
		return err
	}
	tokens, _, err := model.TokenizeText(text)
	if err != nil {
		return err
	}
	result, err := model.EmbedText(context.Background(), text)
	if err != nil {
		return err
	}
	if result.Embeddings == nil {
		return fmt.Errorf("embedding output tensor is nil")
	}
	manifest := model.Manifest()
	fmt.Printf("loaded embedding %q for backend %q\n", displayManifestName(manifest.Name), model.Backend())
	fmt.Printf("tokens: %d\n", len(tokens))
	fmt.Printf("output: %s\n", displayManifestName(result.OutputName))
	fmt.Printf("embedding: %s%v\n", result.Embeddings.DType, result.Embeddings.Shape)
	return nil
}

func runEvalRetrieval(args []string) error {
	fs := flag.NewFlagSet("eval-retrieval", flag.ContinueOnError)
	datasetName := fs.String("dataset", "", "dataset name for metrics output")
	split := fs.String("split", "test", "qrels split under <dataset-dir>/qrels")
	qrelsPath := fs.String("qrels", "", "explicit qrels TSV path")
	batchSize := fs.Int("batch-size", 64, "embedding batch size")
	topK := fs.Int("top-k", 100, "retrieval depth for scoring")
	maxDocs := fs.Int("max-docs", 0, "limit corpus documents for smoke checks")
	maxQueries := fs.Int("max-queries", 0, "limit qrels queries for smoke checks")
	metricsPath := fs.String("metrics-json", "", "write retrieval metrics JSON")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 2 || fs.Arg(0) == "" || fs.Arg(1) == "" {
		return fmt.Errorf("usage: manta eval-retrieval [flags] <artifact.mll> <beir-dataset-dir>")
	}
	artifactPath := fs.Arg(0)
	datasetDir := fs.Arg(1)
	corpusPath, queriesPath, defaultQrelsPath := mantaruntime.BEIRRetrievalPaths(datasetDir, *split)
	if *qrelsPath == "" {
		*qrelsPath = defaultQrelsPath
	}
	if *datasetName == "" {
		*datasetName = filepath.Base(datasetDir)
	}

	rt := mantaruntime.New(cuda.New(), metal.New(), vulkan.New(), directml.New(), webgpu.New())
	model, err := rt.LoadEmbeddingPackage(context.Background(), artifactPath)
	if err != nil {
		return err
	}
	metrics, err := mantaruntime.EvaluateEmbeddingRetrieval(context.Background(), model, mantaruntime.RetrievalEvalConfig{
		DatasetName:  *datasetName,
		ArtifactPath: artifactPath,
		CorpusPath:   corpusPath,
		QueriesPath:  queriesPath,
		QrelsPath:    *qrelsPath,
		BatchSize:    *batchSize,
		TopK:         *topK,
		MaxDocs:      *maxDocs,
		MaxQueries:   *maxQueries,
	})
	if err != nil {
		return err
	}
	if *metricsPath != "" {
		data, err := json.MarshalIndent(metrics, "", "  ")
		if err != nil {
			return err
		}
		data = append(data, '\n')
		if err := os.WriteFile(*metricsPath, data, 0o644); err != nil {
			return err
		}
	}
	fmt.Printf("retrieval eval: dataset=%s backend=%s docs=%d queries=%d relevant_pairs=%d scored_pairs=%d\n",
		metrics.Dataset, metrics.Backend, metrics.Inputs.Documents, metrics.Inputs.Queries, metrics.Inputs.RelevantPairs, metrics.Inputs.ScoredPairs)
	fmt.Printf("quality: ndcg@10=%.6f mrr@10=%.6f recall@10=%.6f recall@100=%.6f\n",
		metrics.Quality.NDCGAt10, metrics.Quality.MRRAt10, metrics.Quality.RecallAt10, metrics.Quality.RecallAt100)
	fmt.Printf("throughput: elapsed=%.3fs docs/s=%.2f queries/s=%.2f scores/s=%.2f\n",
		metrics.Throughput.ElapsedSeconds, metrics.Throughput.DocumentsPerSecond, metrics.Throughput.QueriesPerSecond, metrics.Throughput.ScoresPerSecond)
	if *metricsPath != "" {
		fmt.Printf("metrics: %s\n", *metricsPath)
	}
	return nil
}

func runInspect(args []string) error {
	if len(args) == 0 || args[0] == "" {
		return fmt.Errorf("usage: manta inspect <artifact.mll>")
	}
	path := args[0]
	mod, err := mantaartifact.ReadFile(path)
	if err != nil {
		return err
	}
	fmt.Printf("module: %s\n", mod.Name)
	fmt.Printf("artifact version: %s\n", displayArtifactVersion(mod.Version))
	fmt.Printf("entrypoints: %d, steps: %d, kernels: %d\n", len(mod.EntryPoints), len(mod.Steps), len(mod.Kernels))
	fmt.Printf("backends: %s\n", joinBackendKinds(mod.Requirements.SupportedBackends))
	if len(mod.Requirements.Capabilities) > 0 {
		fmt.Printf("capabilities: %s\n", strings.Join(mod.Requirements.Capabilities, ", "))
	}
	embeddedPackage := false
	embeddingManifestPath := mantaruntime.ResolveEmbeddingManifestPath(path)
	if _, err := os.Stat(embeddingManifestPath); err == nil {
		manifest, err := mantaruntime.ReadEmbeddingManifestFile(embeddingManifestPath)
		if err != nil {
			return err
		}
		fmt.Printf("embedding manifest: %s\n", embeddingManifestPath)
		printEmbeddingManifestSummary(manifest)
	} else if sealed, err := mantaruntime.ReadSealedEmbeddingPackage(path); err == nil {
		fmt.Println("embedding manifest: embedded")
		printEmbeddingManifestSummary(sealed.Manifest)
		fmt.Println("package: embedded sealed MLL")
		fmt.Println("package verify: OK")
		embeddedPackage = true
	}
	packagePath := mantaruntime.ResolvePackageManifestPath(path)
	if !embeddedPackage {
		if _, err := os.Stat(packagePath); err == nil {
			pkg, err := mantaruntime.ReadPackageManifestFile(packagePath)
			if err != nil {
				return err
			}
			verifyPaths := map[string]string{
				"artifact":           path,
				"embedding_manifest": mantaruntime.DefaultEmbeddingManifestPath(path),
				"tokenizer":          mantaruntime.DefaultTokenizerPath(path),
				"weights":            mantaruntime.DefaultWeightFilePath(path),
				"memory_plan":        mantaruntime.DefaultMemoryPlanPath(path),
				"train_manifest":     mantaruntime.DefaultEmbeddingTrainManifestPath(path),
				"checkpoint":         mantaruntime.DefaultEmbeddingCheckpointPath(path),
				"train_profile":      mantaruntime.DefaultEmbeddingTrainProfilePath(path),
			}
			if pkg.Kind == mantaruntime.PackageEmbedding {
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
	profilePath := mantaruntime.DefaultEmbeddingTrainProfilePath(path)
	if _, err := os.Stat(profilePath); err == nil {
		profile, err := mantaruntime.ReadEmbeddingTrainProfileFile(profilePath)
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

func printEmbeddingManifestSummary(manifest mantaruntime.EmbeddingManifest) {
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
		return fmt.Errorf("usage: manta export-mll <artifact.mll> [output.mll]")
	}
	artifactPath := args[0]
	outputPath := ""
	if len(args) > 1 {
		outputPath = args[1]
	}
	writtenPath, err := mantaruntime.ExportPackageToMLL(artifactPath, outputPath)
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
		return fmt.Errorf("usage: manta init-model [flags] <artifact.mll>")
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
	manifest, err := mantaruntime.ReadEmbeddingManifestFile(paths.EmbeddingManifestPath)
	if err != nil {
		return err
	}
	checkpoint, err := mantaruntime.ReadEmbeddingTrainCheckpointFile(paths.CheckpointPath)
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

func runInitMirage(args []string) error {
	fs := flag.NewFlagSet("init-mirage", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	var name string
	var imageHeight int
	var imageWidth int
	var latentChannels int
	var hyperChannels int
	var bits int
	var factorization string
	var lambda float64
	fs.StringVar(&name, "name", "", "model name")
	fs.IntVar(&imageHeight, "height", 0, "image height")
	fs.IntVar(&imageWidth, "width", 0, "image width")
	fs.IntVar(&latentChannels, "latent-channels", 0, "latent channels")
	fs.IntVar(&hyperChannels, "hyper-channels", 0, "hyperprior channels")
	fs.IntVar(&bits, "bits", 0, "TurboQuant bits: 2, 4, or 8")
	fs.StringVar(&factorization, "factorization", "", "coordinate entropy factorization: categorical or bit-plane")
	fs.Float64Var(&lambda, "lambda", 0, "rate-distortion lambda")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 1 || fs.Arg(0) == "" {
		return fmt.Errorf("usage: manta init-mirage [flags] <artifact.mll>")
	}
	cfg := models.MirageV1Config{
		Name:           name,
		ImageHeight:    imageHeight,
		ImageWidth:     imageWidth,
		LatentChannels: latentChannels,
		HyperChannels:  hyperChannels,
		BitWidth:       bits,
		Factorization:  factorization,
		Lambda:         lambda,
	}
	if err := models.InitMirageV1Artifact(fs.Arg(0), cfg); err != nil {
		return err
	}
	mod, err := mantaartifact.ReadFile(fs.Arg(0))
	if err != nil {
		return err
	}
	fmt.Printf("initialized Mirage Image v1 module %q\n", mod.Name)
	fmt.Printf("artifact: %s\n", fs.Arg(0))
	fmt.Printf("entrypoints: %d, steps: %d, kernels: %d\n", len(mod.EntryPoints), len(mod.Steps), len(mod.Kernels))
	if len(mod.Requirements.Capabilities) > 0 {
		fmt.Printf("capabilities: %s\n", strings.Join(mod.Requirements.Capabilities, ", "))
	}
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
		return fmt.Errorf("usage: manta init-train [flags] <artifact.mll>")
	}
	path := fs.Arg(0)
	cfg := mantaruntime.EmbeddingTrainConfig{
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
	opts := mantaruntime.EmbeddingTrainInitOptions{
		Seed:       seed,
		ShapeSizes: dims.values(),
	}
	var (
		paths mantaruntime.EmbeddingTrainPackagePaths
		err   error
	)
	if manifestPath == "" {
		manifestPath = mantaruntime.ResolveEmbeddingManifestPath(path)
	}
	manifest, readErr := mantaruntime.ReadEmbeddingManifestFile(manifestPath)
	if readErr != nil {
		return readErr
	}
	paths, err = mantaruntime.InitializeEmbeddingTrainerPackageWithManifest(path, manifest, cfg, opts)
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

func runRenameEmbed(args []string) error {
	fs := flag.NewFlagSet("rename-embed", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	var name string
	fs.StringVar(&name, "name", "", "new embedding model name")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 2 || fs.Arg(0) == "" || fs.Arg(1) == "" {
		return fmt.Errorf("usage: manta rename-embed --name <model-name> <input.mll> <output.mll>")
	}
	inputPath := fs.Arg(0)
	outputPath := fs.Arg(1)
	trainer, err := mantaruntime.LoadEmbeddingTrainerPackage(inputPath)
	if err != nil {
		return err
	}
	if err := trainer.RenameEmbeddingModel(name); err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(outputPath), 0o755); err != nil {
		return err
	}
	if err := copyTokenizerIfPresent(inputPath, outputPath); err != nil {
		return err
	}
	paths, err := trainer.WriteTrainingPackage(outputPath)
	if err != nil {
		return err
	}
	fmt.Printf("renamed embedding package %q -> %q\n", inputPath, outputPath)
	fmt.Printf("model: %s\n", name)
	fmt.Printf("embedding manifest: %s\n", paths.EmbeddingManifestPath)
	fmt.Printf("weights: %s\n", paths.WeightFilePath)
	fmt.Printf("checkpoint: %s\n", paths.CheckpointPath)
	fmt.Printf("profile: %s\n", paths.TrainProfilePath)
	return nil
}

func copyTokenizerIfPresent(inputPath, outputPath string) error {
	sourcePath := mantaruntime.DefaultTokenizerPath(inputPath)
	if _, err := os.Stat(sourcePath); err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}
	data, err := os.ReadFile(sourcePath)
	if err != nil {
		return err
	}
	return os.WriteFile(mantaruntime.DefaultTokenizerPath(outputPath), data, 0o644)
}

func runTrainEmbed(args []string) error {
	fs := flag.NewFlagSet("train-embed", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	var epochs int
	var batchSize int
	var shuffle bool
	var seed int64
	var evalEvery int
	var evalEverySteps int
	var patience int
	var selectMetric string
	var minDelta float64
	var restoreBest bool
	var lengthBucketBatches bool
	var progressEvery int
	var tokenizerPath string
	var noTokenizer bool
	var planOnly bool
	var evalOnly bool
	var pairwiseTrain bool
	var metricsJSONPath string
	var learningRate float64
	var contrastiveLoss string
	var temperature float64
	fs.IntVar(&epochs, "epochs", 10, "number of epochs")
	fs.IntVar(&batchSize, "batch-size", 8, "batch size")
	fs.BoolVar(&shuffle, "shuffle", true, "shuffle training set each epoch")
	fs.Int64Var(&seed, "seed", 1, "shuffle seed")
	fs.IntVar(&evalEvery, "eval-every", 1, "evaluate every N epochs")
	fs.IntVar(&evalEverySteps, "eval-every-steps", 0, "evaluate every N optimizer steps within an epoch (0 disables)")
	fs.IntVar(&patience, "patience", 3, "early stopping patience in evals")
	fs.StringVar(&selectMetric, "select-metric", "top1_accuracy", "selection metric: top1_accuracy, top5_accuracy, top10_accuracy, mrr, mean_rank, score_margin, pair_accuracy, threshold_accuracy, auc, or loss")
	fs.Float64Var(&minDelta, "min-delta", 0, "minimum eval improvement to count as better")
	fs.BoolVar(&restoreBest, "restore-best", true, "restore best checkpoint at end")
	fs.BoolVar(&lengthBucketBatches, "length-bucket-batches", true, "cluster contrastive batches by token length to improve batched GPU training")
	fs.IntVar(&progressEvery, "progress-every", 0, "print training progress every N optimizer steps (0 disables)")
	fs.StringVar(&tokenizerPath, "tokenizer", "", "path to tokenizer JSON for text-pair datasets")
	fs.BoolVar(&noTokenizer, "no-tokenizer", false, "disable sibling tokenizer discovery and treat JSONL as tokenized")
	fs.BoolVar(&planOnly, "plan-only", false, "print planned workload and exit without training")
	fs.BoolVar(&evalOnly, "eval-only", false, "evaluate the package without running optimizer steps")
	fs.BoolVar(&pairwiseTrain, "pairwise-train", false, "treat the training JSONL as labeled pair examples instead of contrastive positives")
	fs.StringVar(&metricsJSONPath, "metrics-json", "", "write machine-readable run metrics JSON to this path")
	fs.Float64Var(&learningRate, "lr", 0, "override package learning rate for this run")
	fs.StringVar(&contrastiveLoss, "contrastive-loss", "", "override package contrastive loss: pair_mse or infonce")
	fs.Float64Var(&temperature, "temperature", 0, "override package contrastive softmax temperature")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 2 || fs.Arg(0) == "" || fs.Arg(1) == "" {
		return fmt.Errorf("usage: manta train-embed [flags] <artifact.mll> <train.jsonl> [eval.jsonl]\n       manta train-embed --eval-only [flags] <artifact.mll> <eval.jsonl>")
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
	if evalEverySteps < 0 {
		return fmt.Errorf("eval-every-steps must be non-negative")
	}
	path := fs.Arg(0)
	trainPath := fs.Arg(1)
	evalPath := ""
	if fs.NArg() > 2 {
		evalPath = fs.Arg(2)
	}
	if evalOnly && fs.NArg() == 2 {
		evalPath = trainPath
		trainPath = ""
	}
	if noTokenizer && tokenizerPath != "" {
		return fmt.Errorf("set either --tokenizer or --no-tokenizer, not both")
	}
	if tokenizerPath == "" && !noTokenizer {
		defaultTokenizerPath := mantaruntime.DefaultTokenizerPath(path)
		if _, err := os.Stat(defaultTokenizerPath); err == nil {
			tokenizerPath = defaultTokenizerPath
		}
	}
	runConfig := mantaruntime.EmbeddingTrainRunConfig{
		Epochs:                epochs,
		BatchSize:             batchSize,
		Shuffle:               shuffle,
		Seed:                  seed,
		EvalEveryEpoch:        evalEvery,
		EvalEverySteps:        evalEverySteps,
		EarlyStoppingPatience: patience,
		SelectMetric:          selectMetric,
		MinDelta:              float32(minDelta),
		RestoreBest:           restoreBest,
		LengthBucketBatches:   lengthBucketBatches,
		LearningRate:          float32(learningRate),
		ContrastiveLoss:       contrastiveLoss,
		Temperature:           float32(temperature),
		ProgressEverySteps:    progressEvery,
		EvalOnly:              evalOnly,
		PairwiseTrain:         pairwiseTrain,
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
		summary mantaruntime.EmbeddingTrainRunSummary
		paths   mantaruntime.EmbeddingTrainPackagePaths
		err     error
	)
	if tokenizerPath != "" {
		summary, paths, err = mantaruntime.TrainEmbeddingPackageFromTextContrastiveFiles(path, tokenizerPath, trainPath, evalPath, runConfig)
	} else {
		summary, paths, err = mantaruntime.TrainEmbeddingPackageFromContrastiveFiles(path, trainPath, evalPath, runConfig)
	}
	if err != nil {
		return err
	}
	if evalOnly {
		fmt.Printf("evaluated package %q\n", path)
	} else {
		fmt.Printf("trained package %q\n", path)
	}
	if tokenizerPath != "" {
		fmt.Printf("tokenizer: %s\n", tokenizerPath)
	}
	fmt.Printf("epochs: %d, steps: %d, run_steps: %d, best_epoch: %d, best_step: %d\n", summary.EpochsCompleted, summary.StepsCompleted, summary.StepsRun, summary.BestEpoch, summary.BestStep)
	fmt.Printf("final train: loss=%.6f avg_score=%.6f batch=%d\n", summary.FinalTrain.Loss, summary.FinalTrain.AverageScore, summary.FinalTrain.BatchSize)
	if summary.FinalEval != nil {
		fmt.Printf("final eval: loss=%.6f margin=%.6f accuracy=%.6f threshold_accuracy=%.6f threshold=%.6f auc=%.6f top1=%.6f top5=%.6f top10=%.6f mrr=%.6f mean_rank=%.3f pairs=%d\n", summary.FinalEval.Loss, summary.FinalEval.ScoreMargin, summary.FinalEval.PairAccuracy, summary.FinalEval.ThresholdAccuracy, summary.FinalEval.ScoreThreshold, summary.FinalEval.ROCAUC, summary.FinalEval.Top1Accuracy, summary.FinalEval.Top5Accuracy, summary.FinalEval.Top10Accuracy, summary.FinalEval.MeanReciprocalRank, summary.FinalEval.MeanPositiveRank, summary.FinalEval.PairCount)
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
	if metricsJSONPath != "" {
		mode := "train"
		if evalOnly {
			mode = "eval"
		}
		if err := writeTrainMetricsJSON(metricsJSONPath, "train-embed", mode, path, tokenizerPath, summary, paths, nil); err != nil {
			return err
		}
		fmt.Printf("metrics: %s\n", metricsJSONPath)
	}
	return nil
}

func estimateTrainEmbedWorkload(tokenizerPath, trainPath, evalPath string, cfg mantaruntime.EmbeddingTrainRunConfig) (mantaruntime.EmbeddingTrainWorkload, error) {
	if cfg.EvalOnly && evalPath == "" {
		evalPath = trainPath
		trainPath = ""
	}
	if tokenizerPath != "" {
		if cfg.EvalOnly {
			evalPairs, err := mantaruntime.ReadEmbeddingTextPairExamplesFile(evalPath)
			if err != nil {
				return mantaruntime.EmbeddingTrainWorkload{}, err
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
				return mantaruntime.EstimateContrastiveTrainWorkload(0, positiveCount, cfg), nil
			}
			return mantaruntime.EstimatePairwiseTrainWorkload(0, len(evalPairs), cfg), nil
		}
		if cfg.PairwiseTrain {
			trainPairs, err := mantaruntime.ReadEmbeddingTextPairExamplesFile(trainPath)
			if err != nil {
				return mantaruntime.EmbeddingTrainWorkload{}, err
			}
			evalCount := 0
			if evalPath != "" {
				evalPairs, err := mantaruntime.ReadEmbeddingTextPairExamplesFile(evalPath)
				if err != nil {
					return mantaruntime.EmbeddingTrainWorkload{}, err
				}
				evalCount = len(evalPairs)
			}
			return mantaruntime.EstimatePairwiseTrainWorkload(len(trainPairs), evalCount, cfg), nil
		}
		trainSet, err := mantaruntime.ReadEmbeddingTextContrastiveExamplesFile(trainPath)
		if err != nil {
			return mantaruntime.EmbeddingTrainWorkload{}, err
		}
		if evalPath == "" {
			return mantaruntime.EstimateContrastiveTrainWorkload(len(trainSet), 0, cfg), nil
		}
		evalPairs, err := mantaruntime.ReadEmbeddingTextPairExamplesFile(evalPath)
		if err != nil {
			return mantaruntime.EmbeddingTrainWorkload{}, err
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
			return mantaruntime.EstimateContrastiveTrainWorkload(len(trainSet), positiveCount, cfg), nil
		}
		workload := mantaruntime.EstimateContrastiveTrainWorkload(len(trainSet), 0, cfg)
		workload.EvalMode = "pairwise"
		workload.EvalExamples = len(evalPairs)
		workload.EvalPairsPerPass = int64(len(evalPairs))
		workload.PlannedEvalPasses = 1
		workload.PlannedEvalPairs = int64(len(evalPairs))
		workload.PlannedTotalPairs = workload.PlannedTrainPairs + workload.PlannedEvalPairs
		return workload, nil
	}

	if cfg.EvalOnly {
		evalSet, err := mantaruntime.ReadEmbeddingContrastiveExamplesFile(evalPath)
		if err != nil {
			evalPairs, pairErr := mantaruntime.ReadEmbeddingPairExamplesFile(evalPath)
			if pairErr != nil {
				return mantaruntime.EmbeddingTrainWorkload{}, err
			}
			return mantaruntime.EstimatePairwiseTrainWorkload(0, len(evalPairs), cfg), nil
		}
		return mantaruntime.EstimateContrastiveTrainWorkload(0, len(evalSet), cfg), nil
	}
	if cfg.PairwiseTrain {
		trainPairs, err := mantaruntime.ReadEmbeddingPairExamplesFile(trainPath)
		if err != nil {
			return mantaruntime.EmbeddingTrainWorkload{}, err
		}
		evalCount := 0
		if evalPath != "" {
			evalPairs, err := mantaruntime.ReadEmbeddingPairExamplesFile(evalPath)
			if err != nil {
				return mantaruntime.EmbeddingTrainWorkload{}, err
			}
			evalCount = len(evalPairs)
		}
		return mantaruntime.EstimatePairwiseTrainWorkload(len(trainPairs), evalCount, cfg), nil
	}
	trainSet, err := mantaruntime.ReadEmbeddingContrastiveExamplesFile(trainPath)
	if err != nil {
		return mantaruntime.EmbeddingTrainWorkload{}, err
	}
	evalCount := 0
	if evalPath != "" {
		evalSet, err := mantaruntime.ReadEmbeddingContrastiveExamplesFile(evalPath)
		if err != nil {
			evalPairs, pairErr := mantaruntime.ReadEmbeddingPairExamplesFile(evalPath)
			if pairErr != nil {
				return mantaruntime.EmbeddingTrainWorkload{}, err
			}
			workload := mantaruntime.EstimateContrastiveTrainWorkload(len(trainSet), 0, cfg)
			workload.EvalMode = "pairwise"
			workload.EvalExamples = len(evalPairs)
			workload.EvalPairsPerPass = int64(len(evalPairs))
			workload.PlannedEvalPasses = 1
			workload.PlannedEvalPairs = int64(len(evalPairs))
			workload.PlannedTotalPairs = workload.PlannedTrainPairs + workload.PlannedEvalPairs
			return workload, nil
		}
		evalCount = len(evalSet)
	}
	return mantaruntime.EstimateContrastiveTrainWorkload(len(trainSet), evalCount, cfg), nil
}

func formatTrainWorkload(workload mantaruntime.EmbeddingTrainWorkload) string {
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

func formatTrainThroughput(summary mantaruntime.EmbeddingTrainRunSummary) string {
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

func printTrainProgress(progress mantaruntime.EmbeddingTrainProgress) {
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
	var evalEverySteps int
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
	var metricsJSONPath string
	var learningRate float64
	var contrastiveLoss string
	var temperature float64
	fs.IntVar(&epochs, "epochs", 10, "number of epochs")
	fs.IntVar(&batchSize, "batch-size", 8, "batch size")
	fs.BoolVar(&shuffle, "shuffle", true, "shuffle training set each epoch")
	fs.Int64Var(&seed, "seed", 1, "shuffle/mining seed")
	fs.IntVar(&evalEvery, "eval-every", 1, "evaluate every N epochs")
	fs.IntVar(&evalEverySteps, "eval-every-steps", 0, "evaluate every N optimizer steps within an epoch (0 disables)")
	fs.IntVar(&patience, "patience", 3, "early stopping patience in evals")
	fs.StringVar(&selectMetric, "select-metric", "top1_accuracy", "selection metric: top1_accuracy, top5_accuracy, top10_accuracy, mrr, mean_rank, score_margin, pair_accuracy, threshold_accuracy, auc, or loss")
	fs.Float64Var(&minDelta, "min-delta", 0, "minimum eval improvement to count as better")
	fs.BoolVar(&restoreBest, "restore-best", true, "restore best checkpoint at end")
	fs.BoolVar(&lengthBucketBatches, "length-bucket-batches", true, "cluster contrastive batches by token length to improve batched GPU training")
	fs.IntVar(&progressEvery, "progress-every", 0, "print training progress every N optimizer steps (0 disables)")
	fs.StringVar(&tokenizerPath, "tokenizer", "", "output tokenizer path")
	fs.IntVar(&vocabSize, "vocab-size", 0, "tokenizer vocab size override")
	fs.IntVar(&minFreq, "min-freq", 2, "minimum pair frequency for tokenizer merges")
	fs.StringVar(&trainPairsPath, "train-pairs", "", "output mined train pair dataset path")
	fs.StringVar(&evalPairsPath, "eval-pairs-path", "", "output mined eval pair dataset path")
	fs.IntVar(&minChars, "min-chars", 8, "minimum normalized text length for mined segments")
	fs.IntVar(&maxPairs, "max-pairs", 0, "maximum number of positive training pairs to keep (0 = all)")
	fs.IntVar(&evalPairs, "eval-pairs", 32, "number of positive pairs to hold out for eval")
	fs.StringVar(&metricsJSONPath, "metrics-json", "", "write machine-readable run metrics JSON to this path")
	fs.Float64Var(&learningRate, "lr", 0, "override package learning rate for this run")
	fs.StringVar(&contrastiveLoss, "contrastive-loss", "", "override package contrastive loss: pair_mse or infonce")
	fs.Float64Var(&temperature, "temperature", 0, "override package contrastive softmax temperature")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 2 || fs.Arg(0) == "" || fs.Arg(1) == "" {
		return fmt.Errorf("usage: manta train-corpus [flags] <artifact.mll> <corpus.txt>")
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
	if evalEverySteps < 0 {
		return fmt.Errorf("eval-every-steps must be non-negative")
	}
	path := fs.Arg(0)
	corpusPath := fs.Arg(1)
	runConfig := mantaruntime.EmbeddingTrainRunConfig{
		Epochs:                epochs,
		BatchSize:             batchSize,
		Shuffle:               shuffle,
		Seed:                  seed,
		EvalEveryEpoch:        evalEvery,
		EvalEverySteps:        evalEverySteps,
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
	summary, paths, err := mantaruntime.TrainEmbeddingPackageFromCorpusFile(path, corpusPath, mantaruntime.EmbeddingCorpusTrainConfig{
		TokenizerPath:      tokenizerPath,
		TokenizerVocabSize: vocabSize,
		TokenizerMinFreq:   minFreq,
		TrainPairsPath:     trainPairsPath,
		EvalPairsPath:      evalPairsPath,
		Mining: mantaruntime.EmbeddingTextMiningConfig{
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
		fmt.Printf("final eval: loss=%.6f margin=%.6f accuracy=%.6f threshold_accuracy=%.6f threshold=%.6f auc=%.6f top1=%.6f top5=%.6f top10=%.6f mrr=%.6f mean_rank=%.3f pairs=%d\n", summary.FinalEval.Loss, summary.FinalEval.ScoreMargin, summary.FinalEval.PairAccuracy, summary.FinalEval.ThresholdAccuracy, summary.FinalEval.ScoreThreshold, summary.FinalEval.ROCAUC, summary.FinalEval.Top1Accuracy, summary.FinalEval.Top5Accuracy, summary.FinalEval.Top10Accuracy, summary.FinalEval.MeanReciprocalRank, summary.FinalEval.MeanPositiveRank, summary.FinalEval.PairCount)
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
	if metricsJSONPath != "" {
		extra := map[string]string{
			"tokenizer":   paths.TokenizerPath,
			"train_pairs": paths.TrainPairsPath,
		}
		if paths.EvalPairsPath != "" {
			extra["eval_pairs"] = paths.EvalPairsPath
		}
		if err := writeTrainMetricsJSON(metricsJSONPath, "train-corpus", "train", path, paths.TokenizerPath, summary, paths.Package, extra); err != nil {
			return err
		}
		fmt.Printf("metrics: %s\n", metricsJSONPath)
	}
	return nil
}

type trainMetricsJSON struct {
	Schema       string                `json:"schema"`
	Command      string                `json:"command"`
	Mode         string                `json:"mode"`
	Artifact     string                `json:"artifact"`
	Tokenizer    string                `json:"tokenizer,omitempty"`
	Summary      trainRunSummaryJSON   `json:"summary"`
	Config       trainRunConfigJSON    `json:"config"`
	FinalTrain   trainBatchMetricsJSON `json:"final_train"`
	LastEval     *evalMetricsJSON      `json:"last_eval,omitempty"`
	BestEval     *evalMetricsJSON      `json:"best_eval,omitempty"`
	FinalEval    *evalMetricsJSON      `json:"final_eval,omitempty"`
	Workload     trainWorkloadJSON     `json:"workload"`
	Throughput   trainThroughputJSON   `json:"throughput"`
	Accelerators trainAcceleratorsJSON `json:"accelerators"`
	ProfileDelta trainProfileDeltaJSON `json:"profile_delta"`
	Package      trainPackagePathsJSON `json:"package"`
	Artifacts    map[string]string     `json:"artifacts,omitempty"`
}

type trainRunSummaryJSON struct {
	EpochsCompleted int  `json:"epochs_completed"`
	StepsCompleted  int  `json:"steps_completed"`
	StepsRun        int  `json:"steps_run"`
	BestEpoch       int  `json:"best_epoch"`
	BestStep        int  `json:"best_step"`
	RestoredBest    bool `json:"restored_best"`
	StoppedEarly    bool `json:"stopped_early"`
}

type trainRunConfigJSON struct {
	Epochs              int     `json:"epochs"`
	BatchSize           int     `json:"batch_size"`
	Shuffle             bool    `json:"shuffle"`
	Seed                int64   `json:"seed"`
	EvalEveryEpoch      int     `json:"eval_every_epoch"`
	EvalEverySteps      int     `json:"eval_every_steps"`
	Patience            int     `json:"patience"`
	SelectMetric        string  `json:"select_metric"`
	MinDelta            float32 `json:"min_delta"`
	RestoreBest         bool    `json:"restore_best"`
	LengthBucketBatches bool    `json:"length_bucket_batches"`
	LearningRate        float32 `json:"learning_rate"`
	ContrastiveLoss     string  `json:"contrastive_loss,omitempty"`
	Temperature         float32 `json:"temperature"`
	ProgressEverySteps  int     `json:"progress_every_steps"`
	EvalOnly            bool    `json:"eval_only"`
	PairwiseTrain       bool    `json:"pairwise_train"`
}

type trainBatchMetricsJSON struct {
	Loss         float32 `json:"loss"`
	AverageScore float32 `json:"average_score"`
	BatchSize    int     `json:"batch_size"`
}

type evalMetricsJSON struct {
	Loss               float32 `json:"loss"`
	AverageScore       float32 `json:"average_score"`
	PositiveMeanScore  float32 `json:"positive_mean_score"`
	NegativeMeanScore  float32 `json:"negative_mean_score"`
	PairAccuracy       float32 `json:"pair_accuracy"`
	ThresholdAccuracy  float32 `json:"threshold_accuracy"`
	ScoreThreshold     float32 `json:"score_threshold"`
	ROCAUC             float32 `json:"roc_auc"`
	ScoreMargin        float32 `json:"score_margin"`
	Top1Accuracy       float32 `json:"top1_accuracy"`
	Top5Accuracy       float32 `json:"top5_accuracy"`
	Top10Accuracy      float32 `json:"top10_accuracy"`
	MeanReciprocalRank float32 `json:"mean_reciprocal_rank"`
	MeanPositiveRank   float32 `json:"mean_positive_rank"`
	PairCount          int     `json:"pair_count"`
	PositiveCount      int     `json:"positive_count"`
	NegativeCount      int     `json:"negative_count"`
}

type trainWorkloadJSON struct {
	TrainMode            string `json:"train_mode"`
	EvalMode             string `json:"eval_mode,omitempty"`
	TrainExamples        int    `json:"train_examples"`
	EvalExamples         int    `json:"eval_examples"`
	BatchSize            int    `json:"batch_size"`
	PlannedEpochs        int    `json:"planned_epochs"`
	CompletedEpochs      int    `json:"completed_epochs"`
	TrainBatchesPerEpoch int    `json:"train_batches_per_epoch"`
	TrainPairsPerEpoch   int64  `json:"train_pairs_per_epoch"`
	EvalPairsPerPass     int64  `json:"eval_pairs_per_pass"`
	PlannedEvalPasses    int    `json:"planned_eval_passes"`
	ActualEvalPasses     int    `json:"actual_eval_passes"`
	PlannedTrainPairs    int64  `json:"planned_train_pairs"`
	ActualTrainPairs     int64  `json:"actual_train_pairs"`
	ActualTrainExamples  int64  `json:"actual_train_examples"`
	PlannedEvalPairs     int64  `json:"planned_eval_pairs"`
	ActualEvalPairs      int64  `json:"actual_eval_pairs"`
	ActualEvalExamples   int64  `json:"actual_eval_examples"`
	PlannedTotalPairs    int64  `json:"planned_total_pairs"`
	ActualTotalPairs     int64  `json:"actual_total_pairs"`
	ActualTotalExamples  int64  `json:"actual_total_examples"`
}

type trainThroughputJSON struct {
	ElapsedSeconds          float64 `json:"elapsed_seconds"`
	TrainSeconds            float64 `json:"train_seconds"`
	EvalSeconds             float64 `json:"eval_seconds"`
	ExamplesPerSecond       float64 `json:"examples_per_second"`
	PairsPerSecond          float64 `json:"pairs_per_second"`
	TrainExamplesPerSecond  float64 `json:"train_examples_per_second"`
	TrainPairsPerSecond     float64 `json:"train_pairs_per_second"`
	EvalExamplesPerSecond   float64 `json:"eval_examples_per_second"`
	EvalPairsPerSecond      float64 `json:"eval_pairs_per_second"`
	OptimizerStepsPerSecond float64 `json:"optimizer_steps_per_second"`
}

type trainAcceleratorsJSON struct {
	Forward     string `json:"forward"`
	Optimizer   string `json:"optimizer"`
	Activation  string `json:"activation"`
	Contrastive string `json:"contrastive"`
}

type trainProfileDeltaJSON struct {
	MatMulBindCalls     int64   `json:"matmul_bind_calls"`
	MatMulRuns          int64   `json:"matmul_runs"`
	MatMulRunUploadMB   float64 `json:"matmul_run_upload_mb"`
	MatMulRunDownloadMB float64 `json:"matmul_run_download_mb"`
	OptimizerUpdates    int64   `json:"optimizer_updates"`
	OptimizerSyncs      int64   `json:"optimizer_syncs"`
	ActivationCalls     int64   `json:"activation_calls"`
	ContrastiveCalls    int64   `json:"contrastive_calls"`
}

type trainPackagePathsJSON struct {
	Artifact     string `json:"artifact"`
	Checkpoint   string `json:"checkpoint"`
	TrainProfile string `json:"train_profile"`
	Manifest     string `json:"manifest,omitempty"`
	Weights      string `json:"weights,omitempty"`
	MemoryPlan   string `json:"memory_plan,omitempty"`
	Package      string `json:"package,omitempty"`
}

func writeTrainMetricsJSON(outputPath, command, mode, artifactPath, tokenizerPath string, summary mantaruntime.EmbeddingTrainRunSummary, paths mantaruntime.EmbeddingTrainPackagePaths, extraArtifacts map[string]string) error {
	payload := trainMetricsPayload(command, mode, artifactPath, tokenizerPath, summary, paths, extraArtifacts)
	data, err := json.MarshalIndent(payload, "", "  ")
	if err != nil {
		return fmt.Errorf("encode metrics JSON: %w", err)
	}
	data = append(data, '\n')
	if err := os.WriteFile(outputPath, data, 0o644); err != nil {
		return fmt.Errorf("write metrics JSON %q: %w", outputPath, err)
	}
	return nil
}

func trainMetricsPayload(command, mode, artifactPath, tokenizerPath string, summary mantaruntime.EmbeddingTrainRunSummary, paths mantaruntime.EmbeddingTrainPackagePaths, extraArtifacts map[string]string) trainMetricsJSON {
	return trainMetricsJSON{
		Schema:       "manta.embedding_train_metrics.v1",
		Command:      command,
		Mode:         mode,
		Artifact:     artifactPath,
		Tokenizer:    tokenizerPath,
		Summary:      trainRunSummaryPayload(summary),
		Config:       trainRunConfigPayload(summary.Config),
		FinalTrain:   trainBatchMetricsPayload(summary.FinalTrain),
		LastEval:     evalMetricsPayload(summary.LastEval),
		BestEval:     evalMetricsPayload(summary.BestEval),
		FinalEval:    evalMetricsPayload(summary.FinalEval),
		Workload:     trainWorkloadPayload(summary.Workload),
		Throughput:   trainThroughputPayload(summary),
		Accelerators: trainAcceleratorsPayload(summary.EndProfile),
		ProfileDelta: trainProfileDeltaPayload(summary.DeltaProfile),
		Package:      trainPackagePathsPayload(paths),
		Artifacts:    extraArtifacts,
	}
}

func trainRunSummaryPayload(summary mantaruntime.EmbeddingTrainRunSummary) trainRunSummaryJSON {
	return trainRunSummaryJSON{
		EpochsCompleted: summary.EpochsCompleted,
		StepsCompleted:  summary.StepsCompleted,
		StepsRun:        summary.StepsRun,
		BestEpoch:       summary.BestEpoch,
		BestStep:        summary.BestStep,
		RestoredBest:    summary.RestoredBest,
		StoppedEarly:    summary.StoppedEarly,
	}
}

func trainRunConfigPayload(cfg mantaruntime.EmbeddingTrainRunConfig) trainRunConfigJSON {
	return trainRunConfigJSON{
		Epochs:              cfg.Epochs,
		BatchSize:           cfg.BatchSize,
		Shuffle:             cfg.Shuffle,
		Seed:                cfg.Seed,
		EvalEveryEpoch:      cfg.EvalEveryEpoch,
		EvalEverySteps:      cfg.EvalEverySteps,
		Patience:            cfg.EarlyStoppingPatience,
		SelectMetric:        cfg.SelectMetric,
		MinDelta:            cfg.MinDelta,
		RestoreBest:         cfg.RestoreBest,
		LengthBucketBatches: cfg.LengthBucketBatches,
		LearningRate:        cfg.LearningRate,
		ContrastiveLoss:     cfg.ContrastiveLoss,
		Temperature:         cfg.Temperature,
		ProgressEverySteps:  cfg.ProgressEverySteps,
		EvalOnly:            cfg.EvalOnly,
		PairwiseTrain:       cfg.PairwiseTrain,
	}
}

func trainBatchMetricsPayload(metrics mantaruntime.EmbeddingTrainMetrics) trainBatchMetricsJSON {
	return trainBatchMetricsJSON{
		Loss:         metrics.Loss,
		AverageScore: metrics.AverageScore,
		BatchSize:    metrics.BatchSize,
	}
}

func evalMetricsPayload(metrics *mantaruntime.EmbeddingEvalMetrics) *evalMetricsJSON {
	if metrics == nil {
		return nil
	}
	return &evalMetricsJSON{
		Loss:               metrics.Loss,
		AverageScore:       metrics.AverageScore,
		PositiveMeanScore:  metrics.PositiveMeanScore,
		NegativeMeanScore:  metrics.NegativeMeanScore,
		PairAccuracy:       metrics.PairAccuracy,
		ThresholdAccuracy:  metrics.ThresholdAccuracy,
		ScoreThreshold:     metrics.ScoreThreshold,
		ROCAUC:             metrics.ROCAUC,
		ScoreMargin:        metrics.ScoreMargin,
		Top1Accuracy:       metrics.Top1Accuracy,
		Top5Accuracy:       metrics.Top5Accuracy,
		Top10Accuracy:      metrics.Top10Accuracy,
		MeanReciprocalRank: metrics.MeanReciprocalRank,
		MeanPositiveRank:   metrics.MeanPositiveRank,
		PairCount:          metrics.PairCount,
		PositiveCount:      metrics.PositiveCount,
		NegativeCount:      metrics.NegativeCount,
	}
}

func trainWorkloadPayload(workload mantaruntime.EmbeddingTrainWorkload) trainWorkloadJSON {
	return trainWorkloadJSON{
		TrainMode:            workload.TrainMode,
		EvalMode:             workload.EvalMode,
		TrainExamples:        workload.TrainExamples,
		EvalExamples:         workload.EvalExamples,
		BatchSize:            workload.BatchSize,
		PlannedEpochs:        workload.PlannedEpochs,
		CompletedEpochs:      workload.CompletedEpochs,
		TrainBatchesPerEpoch: workload.TrainBatchesPerEpoch,
		TrainPairsPerEpoch:   workload.TrainPairsPerEpoch,
		EvalPairsPerPass:     workload.EvalPairsPerPass,
		PlannedEvalPasses:    workload.PlannedEvalPasses,
		ActualEvalPasses:     workload.ActualEvalPasses,
		PlannedTrainPairs:    workload.PlannedTrainPairs,
		ActualTrainPairs:     workload.ActualTrainPairs,
		ActualTrainExamples:  workload.ActualTrainExamples,
		PlannedEvalPairs:     workload.PlannedEvalPairs,
		ActualEvalPairs:      workload.ActualEvalPairs,
		ActualEvalExamples:   workload.ActualEvalExamples,
		PlannedTotalPairs:    workload.PlannedTotalPairs,
		ActualTotalPairs:     workload.ActualTotalPairs,
		ActualTotalExamples:  workload.ActualTotalExamples,
	}
}

func trainThroughputPayload(summary mantaruntime.EmbeddingTrainRunSummary) trainThroughputJSON {
	return trainThroughputJSON{
		ElapsedSeconds:          summary.Elapsed.Seconds(),
		TrainSeconds:            summary.TrainDuration.Seconds(),
		EvalSeconds:             summary.EvalDuration.Seconds(),
		ExamplesPerSecond:       itemsPerSecond(summary.Workload.ActualTotalExamples, summary.Elapsed),
		PairsPerSecond:          pairsPerSecond(summary.Workload.ActualTotalPairs, summary.Elapsed),
		TrainExamplesPerSecond:  itemsPerSecond(summary.Workload.ActualTrainExamples, summary.TrainDuration),
		TrainPairsPerSecond:     pairsPerSecond(summary.Workload.ActualTrainPairs, summary.TrainDuration),
		EvalExamplesPerSecond:   itemsPerSecond(summary.Workload.ActualEvalExamples, summary.EvalDuration),
		EvalPairsPerSecond:      pairsPerSecond(summary.Workload.ActualEvalPairs, summary.EvalDuration),
		OptimizerStepsPerSecond: itemsPerSecond(int64(summary.StepsRun), summary.TrainDuration),
	}
}

func trainAcceleratorsPayload(profile mantaruntime.EmbeddingTrainProfile) trainAcceleratorsJSON {
	return trainAcceleratorsJSON{
		Forward:     displayTrainBackend(profile.ForwardBackend),
		Optimizer:   displayTrainBackend(profile.OptimizerBackend),
		Activation:  displayTrainBackend(profile.ActivationBackend),
		Contrastive: displayTrainBackend(profile.ContrastiveBackend),
	}
}

func trainProfileDeltaPayload(profile mantaruntime.EmbeddingTrainProfile) trainProfileDeltaJSON {
	return trainProfileDeltaJSON{
		MatMulBindCalls:     profile.ForwardResidency.MatMul.BindCalls,
		MatMulRuns:          profile.ForwardResidency.MatMul.RunCalls,
		MatMulRunUploadMB:   bytesToMiB(profile.ForwardResidency.MatMul.RunUploadedBytes),
		MatMulRunDownloadMB: bytesToMiB(profile.ForwardResidency.MatMul.RunDownloadedBytes),
		OptimizerUpdates:    profile.Optimizer.UpdateCalls,
		OptimizerSyncs:      profile.Optimizer.SyncCalls,
		ActivationCalls:     profile.Activation.GELUBackwardCalls + profile.Activation.SoftmaxBackwardCalls + profile.Activation.LayerNormBackwardCalls,
		ContrastiveCalls:    profile.Contrastive.RunCalls,
	}
}

func trainPackagePathsPayload(paths mantaruntime.EmbeddingTrainPackagePaths) trainPackagePathsJSON {
	return trainPackagePathsJSON{
		Artifact:     paths.ArtifactPath,
		Checkpoint:   paths.CheckpointPath,
		TrainProfile: paths.TrainProfilePath,
		Manifest:     paths.EmbeddingManifestPath,
		Weights:      paths.WeightFilePath,
		MemoryPlan:   paths.MemoryPlanPath,
		Package:      paths.PackageManifestPath,
	}
}

func bytesToMiB(bytes int64) float64 {
	return float64(bytes) / (1024 * 1024)
}

func runCompareTrainMetrics(args []string) error {
	fs := flag.NewFlagSet("compare-train-metrics", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 1 || fs.Arg(0) == "" {
		return fmt.Errorf("usage: manta compare-train-metrics <current.metrics.json> [baseline.metrics.json]")
	}
	currentPath := fs.Arg(0)
	current, err := readTrainMetricsJSON(currentPath)
	if err != nil {
		return err
	}
	var baseline *trainMetricsJSON
	baselinePath := ""
	if fs.NArg() > 1 && fs.Arg(1) != "" {
		baselinePath = fs.Arg(1)
		loaded, err := readTrainMetricsJSON(baselinePath)
		if err != nil {
			return err
		}
		baseline = &loaded
	}
	fmt.Print(formatTrainMetricsReport(currentPath, current, baselinePath, baseline))
	return nil
}

func readTrainMetricsJSON(path string) (trainMetricsJSON, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return trainMetricsJSON{}, err
	}
	var metrics trainMetricsJSON
	if err := json.Unmarshal(data, &metrics); err != nil {
		return trainMetricsJSON{}, fmt.Errorf("parse metrics JSON %q: %w", path, err)
	}
	if metrics.Schema == "" {
		return trainMetricsJSON{}, fmt.Errorf("metrics JSON %q is missing schema", path)
	}
	return metrics, nil
}

func formatTrainMetricsReport(currentPath string, current trainMetricsJSON, baselinePath string, baseline *trainMetricsJSON) string {
	var b strings.Builder
	fmt.Fprintf(&b, "metrics: %s\n", currentPath)
	fmt.Fprintf(&b, "identity: schema=%s command=%s mode=%s artifact=%s\n", current.Schema, current.Command, current.Mode, current.Artifact)
	if current.FinalEval != nil {
		fmt.Fprintf(&b, "quality: top1=%.6f top5=%.6f top10=%.6f mrr=%.6f auc=%.6f margin=%.6f loss=%.6f mean_rank=%.3f pairs=%d\n",
			current.FinalEval.Top1Accuracy,
			current.FinalEval.Top5Accuracy,
			current.FinalEval.Top10Accuracy,
			current.FinalEval.MeanReciprocalRank,
			current.FinalEval.ROCAUC,
			current.FinalEval.ScoreMargin,
			current.FinalEval.Loss,
			current.FinalEval.MeanPositiveRank,
			current.FinalEval.PairCount,
		)
	}
	fmt.Fprintf(&b, "throughput: train_pairs/s=%.2f eval_pairs/s=%.2f optimizer_steps/s=%.2f pairs/s=%.2f elapsed_s=%.3f\n",
		current.Throughput.TrainPairsPerSecond,
		current.Throughput.EvalPairsPerSecond,
		current.Throughput.OptimizerStepsPerSecond,
		current.Throughput.PairsPerSecond,
		current.Throughput.ElapsedSeconds,
	)
	fmt.Fprintf(&b, "accelerators: forward=%s optimizer=%s activation=%s contrastive=%s\n",
		current.Accelerators.Forward,
		current.Accelerators.Optimizer,
		current.Accelerators.Activation,
		current.Accelerators.Contrastive,
	)
	fmt.Fprintf(&b, "profile_delta: matmul_runs=%d upload_mb=%.2f download_mb=%.2f optimizer_updates=%d activation_calls=%d contrastive_calls=%d\n",
		current.ProfileDelta.MatMulRuns,
		current.ProfileDelta.MatMulRunUploadMB,
		current.ProfileDelta.MatMulRunDownloadMB,
		current.ProfileDelta.OptimizerUpdates,
		current.ProfileDelta.ActivationCalls,
		current.ProfileDelta.ContrastiveCalls,
	)
	if baseline == nil {
		return b.String()
	}
	fmt.Fprintf(&b, "baseline: %s\n", baselinePath)
	if current.FinalEval != nil && baseline.FinalEval != nil {
		fmt.Fprintf(&b, "quality_delta: top1=%+.6f top5=%+.6f top10=%+.6f mrr=%+.6f auc=%+.6f margin=%+.6f loss=%+.6f mean_rank=%+.3f\n",
			current.FinalEval.Top1Accuracy-baseline.FinalEval.Top1Accuracy,
			current.FinalEval.Top5Accuracy-baseline.FinalEval.Top5Accuracy,
			current.FinalEval.Top10Accuracy-baseline.FinalEval.Top10Accuracy,
			current.FinalEval.MeanReciprocalRank-baseline.FinalEval.MeanReciprocalRank,
			current.FinalEval.ROCAUC-baseline.FinalEval.ROCAUC,
			current.FinalEval.ScoreMargin-baseline.FinalEval.ScoreMargin,
			current.FinalEval.Loss-baseline.FinalEval.Loss,
			current.FinalEval.MeanPositiveRank-baseline.FinalEval.MeanPositiveRank,
		)
	}
	fmt.Fprintf(&b, "throughput_delta: train_pairs/s=%+.2f eval_pairs/s=%+.2f optimizer_steps/s=%+.2f pairs/s=%+.2f elapsed_s=%+.3f\n",
		current.Throughput.TrainPairsPerSecond-baseline.Throughput.TrainPairsPerSecond,
		current.Throughput.EvalPairsPerSecond-baseline.Throughput.EvalPairsPerSecond,
		current.Throughput.OptimizerStepsPerSecond-baseline.Throughput.OptimizerStepsPerSecond,
		current.Throughput.PairsPerSecond-baseline.Throughput.PairsPerSecond,
		current.Throughput.ElapsedSeconds-baseline.Throughput.ElapsedSeconds,
	)
	fmt.Fprintf(&b, "profile_delta_delta: matmul_runs=%+d upload_mb=%+.2f download_mb=%+.2f optimizer_updates=%+d activation_calls=%+d contrastive_calls=%+d\n",
		current.ProfileDelta.MatMulRuns-baseline.ProfileDelta.MatMulRuns,
		current.ProfileDelta.MatMulRunUploadMB-baseline.ProfileDelta.MatMulRunUploadMB,
		current.ProfileDelta.MatMulRunDownloadMB-baseline.ProfileDelta.MatMulRunDownloadMB,
		current.ProfileDelta.OptimizerUpdates-baseline.ProfileDelta.OptimizerUpdates,
		current.ProfileDelta.ActivationCalls-baseline.ProfileDelta.ActivationCalls,
		current.ProfileDelta.ContrastiveCalls-baseline.ProfileDelta.ContrastiveCalls,
	)
	return b.String()
}

func runDiagnoseTrainMetrics(args []string) error {
	fs := flag.NewFlagSet("diagnose-train-metrics", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 1 || fs.Arg(0) == "" {
		return fmt.Errorf("usage: manta diagnose-train-metrics <metrics.json>")
	}
	metricsPath := fs.Arg(0)
	metrics, err := readTrainMetricsJSON(metricsPath)
	if err != nil {
		return err
	}
	fmt.Print(formatTrainMetricsDiagnosis(metricsPath, metrics))
	return nil
}

func formatTrainMetricsDiagnosis(metricsPath string, metrics trainMetricsJSON) string {
	var b strings.Builder
	warnings := 0
	notes := 0
	fmt.Fprintf(&b, "metrics: %s\n", metricsPath)
	fmt.Fprintf(&b, "identity: schema=%s command=%s mode=%s artifact=%s\n", metrics.Schema, metrics.Command, metrics.Mode, metrics.Artifact)
	fmt.Fprintf(&b, "backend: forward=%s optimizer=%s activation=%s contrastive=%s\n",
		metrics.Accelerators.Forward,
		metrics.Accelerators.Optimizer,
		metrics.Accelerators.Activation,
		metrics.Accelerators.Contrastive,
	)
	fmt.Fprintf(&b, "throughput: train_pairs/s=%.2f eval_pairs/s=%.2f optimizer_steps/s=%.2f elapsed_s=%.3f\n",
		metrics.Throughput.TrainPairsPerSecond,
		metrics.Throughput.EvalPairsPerSecond,
		metrics.Throughput.OptimizerStepsPerSecond,
		metrics.Throughput.ElapsedSeconds,
	)
	pairCount := trainMetricsDiagnosisPairCount(metrics)
	if metrics.ProfileDelta.MatMulRuns > 0 {
		parts := []string{}
		if metrics.ProfileDelta.OptimizerUpdates > 0 {
			parts = append(parts, fmt.Sprintf("matmul_runs/update=%.2f", float64(metrics.ProfileDelta.MatMulRuns)/float64(metrics.ProfileDelta.OptimizerUpdates)))
		} else {
			parts = append(parts, fmt.Sprintf("matmul_runs=%d", metrics.ProfileDelta.MatMulRuns))
		}
		if pairCount > 0 {
			parts = append(parts, fmt.Sprintf("pairs/matmul_run=%.2f", float64(pairCount)/float64(metrics.ProfileDelta.MatMulRuns)))
		}
		if metrics.ProfileDelta.OptimizerUpdates > 0 {
			parts = append(parts, fmt.Sprintf("optimizer_syncs/update=%.2f", float64(metrics.ProfileDelta.OptimizerSyncs)/float64(metrics.ProfileDelta.OptimizerUpdates)))
		}
		fmt.Fprintf(&b, "efficiency: %s\n", strings.Join(parts, " "))
	}
	totalTransferMB := metrics.ProfileDelta.MatMulRunUploadMB + metrics.ProfileDelta.MatMulRunDownloadMB
	if totalTransferMB > 0 {
		parts := []string{fmt.Sprintf("total_mb=%.2f", totalTransferMB)}
		if metrics.ProfileDelta.MatMulRuns > 0 {
			parts = append(parts, fmt.Sprintf("mb/matmul_run=%.4f", totalTransferMB/float64(metrics.ProfileDelta.MatMulRuns)))
		}
		if pairCount > 0 {
			parts = append(parts, fmt.Sprintf("kb/pair=%.4f", totalTransferMB*1024/float64(pairCount)))
		}
		fmt.Fprintf(&b, "transfer: %s\n", strings.Join(parts, " "))
	}
	hostBackends := trainMetricsHostCriticalBackends(metrics)
	if len(hostBackends) == 0 {
		fmt.Fprintf(&b, "finding: ok production-critical accelerators are device-backed\n")
	} else {
		warnings++
		fmt.Fprintf(&b, "finding: warn production-critical accelerators include host fallback: %s\n", strings.Join(hostBackends, " "))
	}
	if trainMetricsEvalOnly(metrics) {
		if metrics.ProfileDelta.OptimizerUpdates == 0 {
			fmt.Fprintf(&b, "finding: ok eval-only run recorded zero optimizer updates\n")
		} else {
			warnings++
			fmt.Fprintf(&b, "finding: warn eval-only run recorded optimizer_updates=%d\n", metrics.ProfileDelta.OptimizerUpdates)
		}
	} else {
		if metrics.ProfileDelta.OptimizerUpdates == 0 {
			warnings++
			fmt.Fprintf(&b, "finding: warn training run recorded zero optimizer updates\n")
		} else if metrics.Throughput.OptimizerStepsPerSecond == 0 && metrics.Throughput.TrainSeconds > 0 {
			warnings++
			fmt.Fprintf(&b, "finding: warn optimizer updates were recorded but optimizer_steps/s is zero\n")
		}
		if metrics.Throughput.TrainPairsPerSecond == 0 && metrics.Workload.ActualTrainPairs > 0 {
			warnings++
			fmt.Fprintf(&b, "finding: warn training pairs were processed but train_pairs/s is zero\n")
		}
		if metrics.Throughput.EvalSeconds > metrics.Throughput.TrainSeconds && metrics.Throughput.TrainSeconds > 0 {
			notes++
			fmt.Fprintf(&b, "finding: note eval time exceeds train time; keep production gates in final eval-only passes unless debugging convergence\n")
		}
	}
	if metrics.Accelerators.Activation == "" || metrics.Accelerators.Activation == "host" {
		notes++
		fmt.Fprintf(&b, "finding: note activation accelerator is host; this can be intentional until activation residency avoids extra transfers\n")
	}
	status := "OK"
	if warnings > 0 {
		status = "WARN"
	}
	fmt.Fprintf(&b, "diagnosis: %s warnings=%d notes=%d\n", status, warnings, notes)
	return b.String()
}

func trainMetricsDiagnosisPairCount(metrics trainMetricsJSON) int64 {
	if metrics.Workload.ActualTrainPairs > 0 {
		return metrics.Workload.ActualTrainPairs
	}
	if metrics.Workload.ActualEvalPairs > 0 {
		return metrics.Workload.ActualEvalPairs
	}
	return metrics.Workload.ActualTotalPairs
}

func trainMetricsEvalOnly(metrics trainMetricsJSON) bool {
	return strings.EqualFold(metrics.Mode, "eval") || metrics.Config.EvalOnly
}

func trainMetricsHostCriticalBackends(metrics trainMetricsJSON) []string {
	backends := []string{}
	if trainMetricsHostBackend(metrics.Accelerators.Forward) {
		backends = append(backends, "forward="+displayBackendLabel(metrics.Accelerators.Forward))
	}
	if !trainMetricsEvalOnly(metrics) && trainMetricsHostBackend(metrics.Accelerators.Optimizer) {
		backends = append(backends, "optimizer="+displayBackendLabel(metrics.Accelerators.Optimizer))
	}
	if trainMetricsHostBackend(metrics.Accelerators.Contrastive) {
		backends = append(backends, "contrastive="+displayBackendLabel(metrics.Accelerators.Contrastive))
	}
	return backends
}

func trainMetricsHostBackend(backend string) bool {
	return backend == "" || backend == "host"
}

func displayBackendLabel(backend string) string {
	if backend == "" {
		return "unknown"
	}
	return backend
}

type trainMetricThreshold struct {
	Env    string
	Metric string
	Op     string
	Scope  string
}

var trainMetricThresholds = []trainMetricThreshold{
	{Env: "MANTA_MIN_MRR", Metric: "mrr", Op: ">=", Scope: "quality"},
	{Env: "MANTA_MIN_TOP1", Metric: "top1", Op: ">=", Scope: "quality"},
	{Env: "MANTA_MIN_TOP5", Metric: "top5", Op: ">=", Scope: "quality"},
	{Env: "MANTA_MIN_TOP10", Metric: "top10", Op: ">=", Scope: "quality"},
	{Env: "MANTA_MAX_MEAN_RANK", Metric: "mean_rank", Op: "<=", Scope: "quality"},
	{Env: "MANTA_MIN_AUC", Metric: "auc", Op: ">=", Scope: "quality"},
	{Env: "MANTA_MIN_THRESHOLD_ACCURACY", Metric: "threshold_accuracy", Op: ">=", Scope: "quality"},
	{Env: "MANTA_MIN_SCORE_MARGIN", Metric: "margin", Op: ">=", Scope: "quality"},
	{Env: "MANTA_MIN_PAIR_ACCURACY", Metric: "accuracy", Op: ">=", Scope: "quality"},
	{Env: "MANTA_MAX_LOSS", Metric: "loss", Op: "<=", Scope: "quality"},
	{Env: "MANTA_MIN_TRAIN_PAIRS_PER_SEC", Metric: "train_pairs/s", Op: ">=", Scope: "efficiency"},
	{Env: "MANTA_MIN_OPTIMIZER_STEPS_PER_SEC", Metric: "optimizer_steps/s", Op: ">=", Scope: "efficiency"},
	{Env: "MANTA_MAX_MATMUL_RUNS", Metric: "matmul_runs", Op: "<=", Scope: "efficiency"},
	{Env: "MANTA_MAX_MATMUL_RUN_UPLOAD_MB", Metric: "matmul_run_upload_mb", Op: "<=", Scope: "efficiency"},
	{Env: "MANTA_MAX_MATMUL_RUN_DOWNLOAD_MB", Metric: "matmul_run_download_mb", Op: "<=", Scope: "efficiency"},
	{Env: "MANTA_MAX_OPTIMIZER_UPDATES", Metric: "optimizer_updates", Op: "<=", Scope: "eval-only"},
}

func runGateTrainMetrics(args []string) error {
	fs := flag.NewFlagSet("gate-train-metrics", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	var thresholdsPath string
	var scope string
	fs.StringVar(&thresholdsPath, "thresholds", "", "optional KEY=VALUE threshold env file")
	fs.StringVar(&scope, "scope", "all", "gate scope: all, quality, efficiency, or eval-only")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 1 || fs.Arg(0) == "" {
		return fmt.Errorf("usage: manta gate-train-metrics [--thresholds thresholds.env] [--scope all|quality|efficiency|eval-only] <metrics.json>")
	}
	scope = strings.ToLower(scope)
	if !validTrainMetricsGateScope(scope) {
		return fmt.Errorf("unsupported gate scope %q", scope)
	}
	metricsPath := fs.Arg(0)
	metrics, err := readTrainMetricsJSON(metricsPath)
	if err != nil {
		return err
	}
	thresholds, err := trainMetricThresholdValues(thresholdsPath)
	if err != nil {
		return err
	}
	fmt.Printf("metrics: %s\n", metricsPath)
	if thresholdsPath != "" {
		fmt.Printf("thresholds: %s\n", thresholdsPath)
	}
	fmt.Printf("scope: %s\n", scope)
	checked := 0
	failed := 0
	for _, threshold := range trainMetricThresholds {
		if !trainMetricThresholdInScope(threshold.Scope, scope) {
			continue
		}
		limitText := thresholds[threshold.Env]
		if limitText == "" {
			continue
		}
		limit, err := strconv.ParseFloat(limitText, 64)
		if err != nil {
			return fmt.Errorf("%s=%q is not numeric: %w", threshold.Env, limitText, err)
		}
		got, ok := trainMetricValue(metrics, threshold.Metric)
		if !ok {
			return fmt.Errorf("metric %s is unavailable in %s", threshold.Metric, metricsPath)
		}
		checked++
		passed := trainMetricThresholdPassed(got, limit, threshold.Op)
		status := "pass"
		if !passed {
			status = "fail"
			failed++
		}
		fmt.Printf("%s: %s=%.6g %s %.6g (%s)\n", status, threshold.Metric, got, threshold.Op, limit, threshold.Env)
	}
	if scope == "eval-only" {
		got, ok := trainMetricValue(metrics, "optimizer_updates")
		if !ok {
			return fmt.Errorf("metric optimizer_updates is unavailable in %s", metricsPath)
		}
		checked++
		if got == 0 {
			fmt.Printf("pass: optimizer_updates=%.6g == 0 (eval-only)\n", got)
		} else {
			fmt.Printf("fail: optimizer_updates=%.6g == 0 (eval-only)\n", got)
			failed++
		}
	}
	if checked == 0 {
		return fmt.Errorf("no thresholds selected for scope %q", scope)
	}
	if failed > 0 {
		fmt.Printf("gate: FAIL checks=%d failed=%d\n", checked, failed)
		return fmt.Errorf("metrics gate failed")
	}
	fmt.Printf("gate: PASS checks=%d\n", checked)
	return nil
}

func validTrainMetricsGateScope(scope string) bool {
	switch scope {
	case "all", "quality", "efficiency", "eval-only":
		return true
	default:
		return false
	}
}

func trainMetricThresholdInScope(thresholdScope, requestedScope string) bool {
	switch requestedScope {
	case "all":
		return thresholdScope == "quality" || thresholdScope == "efficiency"
	default:
		return thresholdScope == requestedScope
	}
}

func trainMetricThresholdValues(path string) (map[string]string, error) {
	values := map[string]string{}
	for _, env := range os.Environ() {
		key, value, ok := strings.Cut(env, "=")
		if ok && strings.HasPrefix(key, "MANTA_") {
			values[key] = value
		}
	}
	if path == "" {
		return values, nil
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	for i, raw := range strings.Split(string(data), "\n") {
		line := strings.TrimSpace(raw)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		key, value, ok := strings.Cut(line, "=")
		if !ok {
			return nil, fmt.Errorf("%s:%d: expected KEY=VALUE", path, i+1)
		}
		key = strings.TrimSpace(key)
		value = strings.TrimSpace(value)
		if !strings.HasPrefix(key, "MANTA_") {
			return nil, fmt.Errorf("%s:%d: threshold key must start with MANTA_: %s", path, i+1, key)
		}
		if values[key] == "" {
			values[key] = value
		}
	}
	return values, nil
}

func trainMetricValue(metrics trainMetricsJSON, metric string) (float64, bool) {
	switch metric {
	case "train_pairs/s":
		return metrics.Throughput.TrainPairsPerSecond, true
	case "optimizer_steps/s":
		return metrics.Throughput.OptimizerStepsPerSecond, true
	case "matmul_runs":
		return float64(metrics.ProfileDelta.MatMulRuns), true
	case "matmul_run_upload_mb":
		return metrics.ProfileDelta.MatMulRunUploadMB, true
	case "matmul_run_download_mb":
		return metrics.ProfileDelta.MatMulRunDownloadMB, true
	case "optimizer_updates":
		return float64(metrics.ProfileDelta.OptimizerUpdates), true
	}
	if metrics.FinalEval == nil {
		return 0, false
	}
	switch metric {
	case "mrr":
		return float64(metrics.FinalEval.MeanReciprocalRank), true
	case "top1":
		return float64(metrics.FinalEval.Top1Accuracy), true
	case "top5":
		return float64(metrics.FinalEval.Top5Accuracy), true
	case "top10":
		return float64(metrics.FinalEval.Top10Accuracy), true
	case "mean_rank":
		return float64(metrics.FinalEval.MeanPositiveRank), true
	case "auc":
		return float64(metrics.FinalEval.ROCAUC), true
	case "threshold_accuracy":
		return float64(metrics.FinalEval.ThresholdAccuracy), true
	case "margin":
		return float64(metrics.FinalEval.ScoreMargin), true
	case "accuracy":
		return float64(metrics.FinalEval.PairAccuracy), true
	case "loss":
		return float64(metrics.FinalEval.Loss), true
	default:
		return 0, false
	}
}

func trainMetricThresholdPassed(got, limit float64, op string) bool {
	switch op {
	case ">=":
		return got >= limit
	case "<=":
		return got <= limit
	default:
		return false
	}
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
		return fmt.Errorf("usage: manta train-tokenizer [flags] <artifact.mll> <corpus.txt>")
	}
	artifactPath := fs.Arg(0)
	corpusPath := fs.Arg(1)
	if outputPath == "" {
		outputPath = mantaruntime.DefaultTokenizerPath(artifactPath)
	}
	if manifestPath == "" {
		manifestPath = mantaruntime.DefaultEmbeddingManifestPath(artifactPath)
	}
	if vocabSize == 0 {
		manifest, err := mantaruntime.ReadEmbeddingManifestFile(manifestPath)
		if err != nil {
			return fmt.Errorf("read embedding manifest for vocab size: %w", err)
		}
		vocabSize = manifest.Tokenizer.VocabSize
	}
	if vocabSize <= 0 {
		return fmt.Errorf("tokenizer vocab size must be set via --vocab-size or embedding manifest")
	}
	tokenizer, err := mantaruntime.TrainTokenizerFromCorpus(mantaruntime.TokenizerTrainConfig{
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
	if err := mantaruntime.SyncEmbeddingTokenizerVocab(artifactPath, len(tokenizer.Tokens)); err != nil {
		return fmt.Errorf("sync tokenizer vocab through Manta package: %w", err)
	}
	fmt.Printf("trained tokenizer %q\n", outputPath)
	fmt.Printf("vocab: %d tokens, merges: %d\n", len(tokenizer.Tokens), len(tokenizer.Merges))
	return nil
}

func runTokenizeEmbed(args []string) error {
	fs := flag.NewFlagSet("tokenize-embed", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	var tokenizerPath string
	var mode string
	fs.StringVar(&tokenizerPath, "tokenizer", "", "path to tokenizer .mll (default: sibling tokenizer)")
	fs.StringVar(&mode, "mode", "contrastive", "output mode: contrastive or pair")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 3 || fs.Arg(0) == "" || fs.Arg(1) == "" || fs.Arg(2) == "" {
		return fmt.Errorf("usage: manta tokenize-embed [--mode contrastive|pair] [--tokenizer tokenizer.mll] <artifact.mll> <input-text.jsonl> <output-token.jsonl>")
	}
	artifactPath := fs.Arg(0)
	inputPath := fs.Arg(1)
	outputPath := fs.Arg(2)
	if tokenizerPath == "" {
		tokenizerPath = mantaruntime.DefaultTokenizerPath(artifactPath)
	}
	tokenizerFile, err := mantaruntime.ReadTokenizerFile(tokenizerPath)
	if err != nil {
		return fmt.Errorf("read tokenizer: %w", err)
	}
	manifest, err := mantaruntime.ReadEmbeddingManifestFile(mantaruntime.ResolveEmbeddingManifestPath(artifactPath))
	if err != nil {
		return fmt.Errorf("read embedding manifest: %w", err)
	}
	tokenizer, err := mantaruntime.NewBPETokenizer(tokenizerFile, manifest.Tokenizer)
	if err != nil {
		return fmt.Errorf("build tokenizer: %w", err)
	}
	switch strings.ToLower(mode) {
	case "contrastive":
		examples, err := mantaruntime.ReadEmbeddingTextContrastiveExamplesFile(inputPath)
		if err != nil {
			return fmt.Errorf("read text contrastive dataset: %w", err)
		}
		tokenized, err := mantaruntime.TokenizeEmbeddingTextContrastiveExamples(examples, tokenizer)
		if err != nil {
			return fmt.Errorf("tokenize contrastive dataset: %w", err)
		}
		if err := mantaruntime.WriteEmbeddingContrastiveExamplesFile(outputPath, tokenized); err != nil {
			return err
		}
		fmt.Printf("tokenized contrastive examples: %d\n", len(tokenized))
	case "pair":
		examples, err := mantaruntime.ReadEmbeddingTextPairExamplesFile(inputPath)
		if err != nil {
			return fmt.Errorf("read text pair dataset: %w", err)
		}
		tokenized, err := mantaruntime.TokenizeEmbeddingTextPairExamples(examples, tokenizer)
		if err != nil {
			return fmt.Errorf("tokenize pair dataset: %w", err)
		}
		if err := mantaruntime.WriteEmbeddingPairExamplesFile(outputPath, tokenized); err != nil {
			return err
		}
		fmt.Printf("tokenized pair examples: %d\n", len(tokenized))
	default:
		return fmt.Errorf("unsupported tokenize mode %q: want contrastive or pair", mode)
	}
	fmt.Printf("tokenizer: %s\n", tokenizerPath)
	fmt.Printf("input: %s\n", inputPath)
	fmt.Printf("output: %s\n", outputPath)
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
		return fmt.Errorf("usage: manta mine-text-pairs [flags] <corpus.txt> <train.jsonl> [eval.jsonl]")
	}
	corpusPath := fs.Arg(0)
	trainPath := fs.Arg(1)
	evalPath := ""
	if fs.NArg() > 2 {
		evalPath = fs.Arg(2)
	}
	trainSet, evalSet, err := mantaruntime.MineEmbeddingTextDatasetsFromCorpusFile(corpusPath, mantaruntime.EmbeddingTextMiningConfig{
		MinChars:  minChars,
		MaxPairs:  maxPairs,
		EvalPairs: evalPairs,
		Seed:      seed,
	})
	if err != nil {
		return err
	}
	if err := mantaruntime.WriteEmbeddingTextContrastiveExamplesFile(trainPath, trainSet); err != nil {
		return err
	}
	if evalPath != "" && len(evalSet) > 0 {
		if err := mantaruntime.WriteEmbeddingTextPairExamplesFile(evalPath, evalSet); err != nil {
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
	fmt.Println("  manta version")
	fmt.Println("  manta compile <source.manta> [output.mll]")
	fmt.Println("  manta inspect <artifact.mll>")
	fmt.Println("  manta export-mll <artifact.mll> [output.mll]")
	fmt.Println("  manta embed-text <artifact.mll> <text...>")
	fmt.Println("  manta eval-retrieval [flags] <artifact.mll> <beir-dataset-dir>")
	fmt.Println("  manta init-model [flags] <artifact.mll>")
	fmt.Println("  manta init-mirage [flags] <artifact.mll>")
	fmt.Println("  manta init-train [flags] <artifact.mll>")
	fmt.Println("  manta rename-embed --name <model-name> <input.mll> <output.mll>")
	fmt.Println("  manta train-tokenizer [flags] <artifact.mll> <corpus.txt>")
	fmt.Println("  manta tokenize-embed [flags] <artifact.mll> <input-text.jsonl> <output-token.jsonl>")
	fmt.Println("  manta train-corpus [flags] <artifact.mll> <corpus.txt>")
	fmt.Println("  manta train-embed [flags] <artifact.mll> <train.jsonl> [eval.jsonl]")
	fmt.Println("  manta compare-train-metrics <current.metrics.json> [baseline.metrics.json]")
	fmt.Println("  manta diagnose-train-metrics <metrics.json>")
	fmt.Println("  manta gate-train-metrics [flags] <metrics.json>")
	fmt.Println("  manta run <artifact.mll> [entry]")
	fmt.Println("  manta demo [module-name]")
	fmt.Println()
	fmt.Println("compile lowers a Manta source file into an .mll artifact.")
	fmt.Println("inspect summarizes an artifact and verifies its sibling package manifest when present.")
	fmt.Println("export-mll seals an artifact package into a weight-carrying .mll container while preserving Manta metadata in XMTA.")
	fmt.Println("embed-text loads a packaged or sealed embedding .mll and embeds text with its tokenizer.")
	fmt.Println("eval-retrieval scores a sealed embedding .mll on BEIR-style corpus/query/qrels files with nDCG/MRR/Recall metrics.")
	fmt.Println("init-model creates the Manta-owned default quantized embedding training package.")
	fmt.Println("init-mirage creates the Manta-owned Mirage Image v1 host-reference artifact.")
	fmt.Println("init-train creates a native training package next to an artifact.")
	fmt.Println("rename-embed rewrites a training package under a new embedding model identity.")
	fmt.Println("train-tokenizer builds a sibling .tokenizer.mll from a raw text corpus, using embedding-manifest vocab_size by default.")
	fmt.Println("tokenize-embed converts text JSONL into reusable token JSONL for training or eval.")
	fmt.Println("train-corpus trains tokenizer + mined text pairs + embedder in one Manta job from a raw text corpus.")
	fmt.Println("train-embed reloads a training package, fits or --eval-only evaluates token JSONL or text JSONL (with --tokenizer or a sibling .tokenizer.mll; use --no-tokenizer for token JSONL beside a tokenizer), and writes it back.")
	fmt.Println("compare-train-metrics summarizes metrics JSON and prints deltas against a baseline metrics JSON when provided.")
	fmt.Println("diagnose-train-metrics explains backend use, transfer pressure, and suspicious training/eval counters from metrics JSON.")
	fmt.Println("gate-train-metrics checks metrics JSON against MANTA_* thresholds from the environment or a thresholds env file.")
	fmt.Println("run loads an artifact, binds stub weights and inputs, and executes one entrypoint.")
	fmt.Println("demo creates a tiny inference-style module and loads it through the runtime.")
}

func totalKernelOps(kernels []mantaartifact.Kernel) int {
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

func stubLoadOptions(mod *mantaartifact.Module) []mantaruntime.LoadOption {
	sizes := defaultSymbolSizes(mod)
	opts := make([]mantaruntime.LoadOption, 0, len(mod.Params))
	for _, param := range mod.Params {
		opts = append(opts, mantaruntime.WithWeight(param.Name, stubTensorForParam(param.Name, param.Type, sizes)))
	}
	return opts
}

func defaultEntryName(mod *mantaartifact.Module) string {
	if mod != nil && len(mod.EntryPoints) > 0 {
		return mod.EntryPoints[0].Name
	}
	return ""
}

func entryPointByName(mod *mantaartifact.Module, name string) (mantaartifact.EntryPoint, error) {
	for _, entry := range mod.EntryPoints {
		if entry.Name == name {
			return entry, nil
		}
	}
	return mantaartifact.EntryPoint{}, fmt.Errorf("unknown entrypoint %q", name)
}

func stubInputs(entry mantaartifact.EntryPoint) map[string]any {
	sizes := defaultShapeSizes(entry)
	out := make(map[string]any, len(entry.Inputs))
	for _, input := range entry.Inputs {
		if input.Type.Kind == mantaartifact.ValueKVCache {
			out[input.Name] = backend.NewKVCache(backend.NewTensorF16([]int{sizes["T"], sizes["D"]}, make([]float32, sizes["T"]*sizes["D"])))
			continue
		}
		out[input.Name] = stubTensorForInput(input.Name, input.Type, sizes)
	}
	return out
}

func displayTrainBackend(kind mantaartifact.BackendKind) string {
	if kind == "" {
		return "host"
	}
	return string(kind)
}

func displayArtifactVersion(version string) string {
	return version
}

func displayManifestName(value string) string {
	if value == "" {
		return "-"
	}
	return value
}

func joinBackendKinds(kinds []mantaartifact.BackendKind) string {
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

func defaultSymbolSizes(mod *mantaartifact.Module) map[string]int {
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

func defaultShapeSizes(entry mantaartifact.EntryPoint) map[string]int {
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

func stubTensorForParam(name string, typ mantaartifact.ValueType, sizes map[string]int) *backend.Tensor {
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

func stubTensorForInput(name string, typ mantaartifact.ValueType, sizes map[string]int) *backend.Tensor {
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
		if product(shape) != 4 {
			return fillTensor(typ, shape, 0)
		}
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

func fillTensor(typ mantaartifact.ValueType, shape []int, offset float32) *backend.Tensor {
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
	case "q2", "q4", "q_norm":
		values := make([]float32, n)
		for i := range values {
			values[i] = offset + float32(i+1)/10
		}
		if typ.Tensor.DType == "q2" {
			return backend.NewTensorQ2(shape, values)
		}
		if typ.Tensor.DType == "q_norm" {
			return backend.NewTensorQNorm(shape, values)
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

func concreteShape(typ mantaartifact.ValueType, sizes map[string]int) []int {
	if typ.Tensor == nil {
		return []int{1}
	}
	shape := make([]int, len(typ.Tensor.Shape))
	for i, dim := range typ.Tensor.Shape {
		if literal, err := strconv.Atoi(dim); err == nil {
			shape[i] = literal
			continue
		}
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
