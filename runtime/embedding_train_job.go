package mantaruntime

import (
	"fmt"
	"time"
)

type EmbeddingCorpusTrainConfig struct {
	TokenizerPath      string
	TokenizerVocabSize int
	TokenizerMinFreq   int
	TrainPairsPath     string
	EvalPairsPath      string
	Mining             EmbeddingTextMiningConfig
	Run                EmbeddingTrainRunConfig
}

type EmbeddingCorpusTrainPaths struct {
	TokenizerPath  string
	TrainPairsPath string
	EvalPairsPath  string
	Package        EmbeddingTrainPackagePaths
}

// TrainEmbeddingPackageFromContrastiveFiles reloads a packaged trainer, fits it on a JSONL contrastive dataset, and writes the updated package back.
func TrainEmbeddingPackageFromContrastiveFiles(artifactPath, trainPath, evalPath string, cfg EmbeddingTrainRunConfig) (EmbeddingTrainRunSummary, EmbeddingTrainPackagePaths, error) {
	trainer, err := LoadEmbeddingTrainerPackage(artifactPath)
	if err != nil {
		return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, err
	}
	if cfg.EvalOnly && evalPath == "" {
		evalPath = trainPath
		trainPath = ""
	}
	if cfg.PairwiseTrain {
		var trainPairs []EmbeddingPairExample
		if !cfg.EvalOnly {
			trainPairs, err = ReadEmbeddingPairExamplesFile(trainPath)
			if err != nil {
				return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, fmt.Errorf("read train pair dataset: %w", err)
			}
		}
		var evalPairs []EmbeddingPairExample
		if evalPath != "" {
			evalPairs, err = ReadEmbeddingPairExamplesFile(evalPath)
			if err != nil {
				return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, fmt.Errorf("read eval pair dataset: %w", err)
			}
		}
		summary, err := trainer.Fit(trainPairs, evalPairs, cfg)
		if err != nil {
			return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, err
		}
		paths, err := trainer.WriteTrainingPackage(artifactPath)
		if err != nil {
			return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, err
		}
		return summary, paths, nil
	}
	var trainSet []EmbeddingContrastiveExample
	if !cfg.EvalOnly {
		trainSet, err = ReadEmbeddingContrastiveExamplesFile(trainPath)
		if err != nil {
			return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, fmt.Errorf("read train dataset: %w", err)
		}
	}
	var evalSet []EmbeddingContrastiveExample
	var evalPairs []EmbeddingPairExample
	if evalPath != "" {
		evalSet, err = ReadEmbeddingContrastiveExamplesFile(evalPath)
		if err != nil {
			evalPairs, err = ReadEmbeddingPairExamplesFile(evalPath)
			if err != nil {
				return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, fmt.Errorf("read eval dataset: %w", err)
			}
		}
	}
	var summary EmbeddingTrainRunSummary
	if len(evalPairs) > 0 && len(evalSet) == 0 {
		if cfg.EvalOnly {
			summary, err = trainer.Fit(nil, evalPairs, cfg)
			if err != nil {
				return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, err
			}
		} else {
			summary, err = trainer.FitContrastive(trainSet, evalSet, cfg)
			if err != nil {
				return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, err
			}
			evalStart := time.Now()
			finalEval, err := trainer.EvaluatePairs(evalPairs)
			if err != nil {
				return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, err
			}
			applyPairwiseEvalSummary(&summary, finalEval, time.Since(evalStart), cfg)
		}
	} else {
		summary, err = trainer.FitContrastive(trainSet, evalSet, cfg)
		if err != nil {
			return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, err
		}
	}
	paths, err := trainer.WriteTrainingPackage(artifactPath)
	if err != nil {
		return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, err
	}
	return summary, paths, nil
}

// TrainEmbeddingPackageFromTextContrastiveFiles reloads a packaged trainer, tokenizes text-pair JSONL with a Manta tokenizer file,
// fits it, and writes the updated package back.
func TrainEmbeddingPackageFromTextContrastiveFiles(artifactPath, tokenizerPath, trainPath, evalPath string, cfg EmbeddingTrainRunConfig) (EmbeddingTrainRunSummary, EmbeddingTrainPackagePaths, error) {
	trainer, err := LoadEmbeddingTrainerPackage(artifactPath)
	if err != nil {
		return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, err
	}
	if cfg.EvalOnly && evalPath == "" {
		evalPath = trainPath
		trainPath = ""
	}
	tokenizerFile, err := ReadTokenizerFile(tokenizerPath)
	if err != nil {
		return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, fmt.Errorf("read tokenizer: %w", err)
	}
	tokenizer, err := NewBPETokenizer(tokenizerFile, trainer.manifest.Tokenizer)
	if err != nil {
		return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, fmt.Errorf("build tokenizer: %w", err)
	}
	tokenCache := embeddingTextTokenCache{}
	if cfg.PairwiseTrain {
		var trainPairs []EmbeddingPairExample
		if !cfg.EvalOnly {
			trainText, err := ReadEmbeddingTextPairExamplesFile(trainPath)
			if err != nil {
				return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, fmt.Errorf("read train text pair dataset: %w", err)
			}
			trainPairs, err = tokenizeEmbeddingTextPairExamples(trainText, tokenizer, tokenCache, false)
			if err != nil {
				return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, fmt.Errorf("tokenize train pair dataset: %w", err)
			}
		}
		var evalPairs []EmbeddingPairExample
		if evalPath != "" {
			evalText, err := ReadEmbeddingTextPairExamplesFile(evalPath)
			if err != nil {
				return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, fmt.Errorf("read eval text pair dataset: %w", err)
			}
			evalPairs, err = tokenizeEmbeddingTextPairExamples(evalText, tokenizer, tokenCache, false)
			if err != nil {
				return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, fmt.Errorf("tokenize eval pair dataset: %w", err)
			}
		}
		summary, err := trainer.Fit(trainPairs, evalPairs, cfg)
		if err != nil {
			return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, err
		}
		paths, err := trainer.WriteTrainingPackage(artifactPath)
		if err != nil {
			return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, err
		}
		return summary, paths, nil
	}
	var trainSet []EmbeddingContrastiveExample
	if !cfg.EvalOnly {
		trainText, err := ReadEmbeddingTextContrastiveExamplesFile(trainPath)
		if err != nil {
			return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, fmt.Errorf("read train text dataset: %w", err)
		}
		trainSet, err = tokenizeEmbeddingTextContrastiveExamples(trainText, tokenizer, tokenCache, false)
		if err != nil {
			return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, fmt.Errorf("tokenize train dataset: %w", err)
		}
	}
	var (
		evalSet   []EmbeddingContrastiveExample
		evalPairs []EmbeddingPairExample
	)
	if evalPath != "" {
		evalText, err := ReadEmbeddingTextPairExamplesFile(evalPath)
		if err != nil {
			return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, fmt.Errorf("read eval text dataset: %w", err)
		}
		evalPairs, err = tokenizeEmbeddingTextPairExamples(evalText, tokenizer, tokenCache, false)
		if err != nil {
			return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, fmt.Errorf("tokenize eval dataset: %w", err)
		}
		allPositive := true
		for _, example := range evalText {
			if example.Target <= 0 {
				allPositive = false
				break
			}
		}
		if allPositive {
			evalSet, err = tokenizeEmbeddingTextContrastiveExamples(toTextContrastiveExamples(evalText), tokenizer, tokenCache, false)
			if err != nil {
				return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, fmt.Errorf("tokenize eval contrastive dataset: %w", err)
			}
		}
	}
	var summary EmbeddingTrainRunSummary
	if len(evalPairs) > 0 && len(evalSet) == 0 {
		if cfg.EvalOnly {
			summary, err = trainer.Fit(nil, evalPairs, cfg)
			if err != nil {
				return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, err
			}
		} else {
			summary, err = trainer.FitContrastive(trainSet, evalSet, cfg)
			if err != nil {
				return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, err
			}
			evalStart := time.Now()
			finalEval, err := trainer.EvaluatePairs(evalPairs)
			if err != nil {
				return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, err
			}
			applyPairwiseEvalSummary(&summary, finalEval, time.Since(evalStart), cfg)
		}
	} else {
		summary, err = trainer.FitContrastive(trainSet, evalSet, cfg)
		if err != nil {
			return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, err
		}
	}
	paths, err := trainer.WriteTrainingPackage(artifactPath)
	if err != nil {
		return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, err
	}
	return summary, paths, nil
}

func applyPairwiseEvalSummary(summary *EmbeddingTrainRunSummary, finalEval EmbeddingEvalMetrics, elapsed time.Duration, cfg EmbeddingTrainRunConfig) {
	if summary == nil {
		return
	}
	summary.LastEval = cloneEvalMetrics(finalEval)
	summary.FinalEval = cloneEvalMetrics(finalEval)
	summary.EvalDuration += elapsed
	summary.Elapsed += elapsed
	summary.Workload.EvalMode = workloadEvalMode(finalEval.PairCount, "pairwise")
	summary.Workload.EvalExamples = finalEval.PairCount
	summary.Workload.EvalPairsPerPass = int64(finalEval.PairCount)
	summary.Workload.PlannedEvalPasses = 1
	summary.Workload.ActualEvalPasses++
	summary.Workload.PlannedEvalPairs = int64(finalEval.PairCount)
	summary.Workload.ActualEvalPairs += int64(finalEval.PairCount)
	summary.Workload.PlannedTotalPairs = summary.Workload.PlannedTrainPairs + summary.Workload.PlannedEvalPairs
	summary.Workload.ActualTotalPairs = summary.Workload.ActualTrainPairs + summary.Workload.ActualEvalPairs
	if summary.BestEval == nil || betterEvalMetrics(finalEval, *summary.BestEval, cfg.SelectMetric, cfg.MinDelta) {
		summary.BestEval = cloneEvalMetrics(finalEval)
		if summary.BestEpoch == 0 {
			summary.BestEpoch = summary.EpochsCompleted
		}
		if summary.BestStep == 0 {
			summary.BestStep = summary.StepsCompleted
		}
	}
}

func TrainEmbeddingPackageFromCorpusFile(artifactPath, corpusPath string, cfg EmbeddingCorpusTrainConfig) (EmbeddingTrainRunSummary, EmbeddingCorpusTrainPaths, error) {
	trainer, err := LoadEmbeddingTrainerPackage(artifactPath)
	if err != nil {
		return EmbeddingTrainRunSummary{}, EmbeddingCorpusTrainPaths{}, err
	}
	defer trainer.Close()

	tokenizerPath := cfg.TokenizerPath
	if tokenizerPath == "" {
		tokenizerPath = DefaultTokenizerPath(artifactPath)
	}
	vocabSize := cfg.TokenizerVocabSize
	if vocabSize == 0 {
		vocabSize = trainer.manifest.Tokenizer.VocabSize
	}
	minVocabSize, err := MinimumTokenizerVocabSizeForCorpus(corpusPath)
	if err != nil {
		return EmbeddingTrainRunSummary{}, EmbeddingCorpusTrainPaths{}, err
	}
	if vocabSize < minVocabSize {
		vocabSize = minVocabSize
	}
	if vocabSize <= 0 {
		return EmbeddingTrainRunSummary{}, EmbeddingCorpusTrainPaths{}, fmt.Errorf("tokenizer vocab size must be set via config or embedding manifest")
	}
	minFreq := cfg.TokenizerMinFreq
	if minFreq <= 0 {
		minFreq = 2
	}
	tokenizer, err := TrainTokenizerFromCorpus(TokenizerTrainConfig{
		CorpusPath: corpusPath,
		VocabSize:  vocabSize,
		MinFreq:    minFreq,
	})
	if err != nil {
		return EmbeddingTrainRunSummary{}, EmbeddingCorpusTrainPaths{}, err
	}
	if err := tokenizer.WriteFile(tokenizerPath); err != nil {
		return EmbeddingTrainRunSummary{}, EmbeddingCorpusTrainPaths{}, err
	}
	if err := SyncEmbeddingTokenizerVocab(artifactPath, len(tokenizer.Tokens)); err != nil {
		return EmbeddingTrainRunSummary{}, EmbeddingCorpusTrainPaths{}, err
	}
	trainPairs, evalPairs, err := MineEmbeddingTextDatasetsFromCorpusFile(corpusPath, cfg.Mining)
	if err != nil {
		return EmbeddingTrainRunSummary{}, EmbeddingCorpusTrainPaths{}, err
	}
	trainPairsPath := cfg.TrainPairsPath
	if trainPairsPath == "" {
		trainPairsPath = DefaultMinedTrainPairsPath(artifactPath)
	}
	evalPairsPath := cfg.EvalPairsPath
	if evalPairsPath == "" {
		evalPairsPath = DefaultMinedEvalPairsPath(artifactPath)
	}
	if err := WriteEmbeddingTextContrastiveExamplesFile(trainPairsPath, trainPairs); err != nil {
		return EmbeddingTrainRunSummary{}, EmbeddingCorpusTrainPaths{}, err
	}
	effectiveEvalPath := ""
	if len(evalPairs) > 0 {
		if err := WriteEmbeddingTextPairExamplesFile(evalPairsPath, evalPairs); err != nil {
			return EmbeddingTrainRunSummary{}, EmbeddingCorpusTrainPaths{}, err
		}
		effectiveEvalPath = evalPairsPath
	}
	summary, paths, err := TrainEmbeddingPackageFromTextContrastiveFiles(artifactPath, tokenizerPath, trainPairsPath, effectiveEvalPath, cfg.Run)
	if err != nil {
		return EmbeddingTrainRunSummary{}, EmbeddingCorpusTrainPaths{}, err
	}
	return summary, EmbeddingCorpusTrainPaths{
		TokenizerPath:  tokenizerPath,
		TrainPairsPath: trainPairsPath,
		EvalPairsPath:  effectiveEvalPath,
		Package:        paths,
	}, nil
}

func toTextContrastiveExamples(examples []EmbeddingTextPairExample) []EmbeddingTextContrastiveExample {
	out := make([]EmbeddingTextContrastiveExample, 0, len(examples))
	for _, example := range examples {
		if example.Target <= 0 {
			continue
		}
		out = append(out, EmbeddingTextContrastiveExample{
			Query:    example.Query,
			Positive: example.Right,
		})
	}
	return out
}
