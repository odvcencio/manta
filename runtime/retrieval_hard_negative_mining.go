package mantaruntime

import (
	"context"
	"fmt"
	"slices"
	"strings"
)

// RetrievalHardNegativeMiningConfig describes BEIR hard-negative mining.
type RetrievalHardNegativeMiningConfig struct {
	DatasetName          string
	CorpusPath           string
	QueriesPath          string
	QrelsPath            string
	NegativesPerPositive int
	CandidateTopK        int
	BatchSize            int
	MaxExamples          int
	MaxDocs              int
	MaxQueries           int
}

type RetrievalHardNegativeMiningSummary struct {
	DatasetName              string
	Queries                  int
	PositivePairs            int
	Examples                 int
	Negatives                int
	SkippedQueriesNoText     int
	SkippedPositiveDocs      int
	SkippedQueriesNoNegative int
}

type retrievalPositiveDoc struct {
	ID    string
	Score float64
	Text  string
}

// MineBM25TextHardNegatives mines text hard negatives from BEIR data using the same BM25 scorer as the lexical baseline.
func MineBM25TextHardNegatives(ctx context.Context, cfg RetrievalHardNegativeMiningConfig) ([]EmbeddingTextHardNegativeExample, RetrievalHardNegativeMiningSummary, error) {
	cfg = normalizeRetrievalHardNegativeMiningConfig(cfg)
	if cfg.CorpusPath == "" || cfg.QueriesPath == "" || cfg.QrelsPath == "" {
		return nil, RetrievalHardNegativeMiningSummary{}, fmt.Errorf("corpus, queries, and qrels paths are required")
	}
	qrels, err := readBEIRQrels(cfg.QrelsPath)
	if err != nil {
		return nil, RetrievalHardNegativeMiningSummary{}, err
	}
	corpus, err := readBEIRCorpus(cfg.CorpusPath, cfg.MaxDocs)
	if err != nil {
		return nil, RetrievalHardNegativeMiningSummary{}, err
	}
	queries, skippedQueries, err := readBEIRQueries(cfg.QueriesPath, qrels, cfg.MaxQueries)
	if err != nil {
		return nil, RetrievalHardNegativeMiningSummary{}, err
	}
	if len(corpus) == 0 {
		return nil, RetrievalHardNegativeMiningSummary{}, fmt.Errorf("corpus is empty")
	}
	if len(queries) == 0 {
		return nil, RetrievalHardNegativeMiningSummary{}, fmt.Errorf("no qrels queries found in queries file")
	}
	index, err := buildBM25Index(ctx, corpus)
	if err != nil {
		return nil, RetrievalHardNegativeMiningSummary{}, err
	}
	docText := make(map[string]string, len(corpus))
	for _, doc := range corpus {
		docText[doc.ID] = doc.Text
	}
	summary := RetrievalHardNegativeMiningSummary{
		DatasetName:          cfg.DatasetName,
		Queries:              len(queries),
		SkippedQueriesNoText: skippedQueries,
	}
	out := []EmbeddingTextHardNegativeExample{}
	for _, query := range queries {
		if err := ctx.Err(); err != nil {
			return nil, RetrievalHardNegativeMiningSummary{}, err
		}
		positives, skippedPositiveDocs := bm25MiningPositiveDocs(qrels[query.ID], docText)
		summary.SkippedPositiveDocs += skippedPositiveDocs
		if len(positives) == 0 {
			continue
		}
		positiveIDs := make(map[string]bool, len(positives))
		for _, positive := range positives {
			positiveIDs[positive.ID] = true
		}
		negativeTexts := bm25MiningNegativeTexts(query.Text, positiveIDs, index, docText, cfg)
		if len(negativeTexts) == 0 {
			summary.SkippedQueriesNoNegative++
			continue
		}
		for _, positive := range positives {
			if cfg.MaxExamples > 0 && len(out) >= cfg.MaxExamples {
				break
			}
			exampleNegatives := negativeTexts
			if len(exampleNegatives) > cfg.NegativesPerPositive {
				exampleNegatives = exampleNegatives[:cfg.NegativesPerPositive]
			}
			out = append(out, EmbeddingTextHardNegativeExample{
				Query:     query.Text,
				Positive:  positive.Text,
				Negatives: append([]string(nil), exampleNegatives...),
			})
			summary.PositivePairs++
			summary.Negatives += len(exampleNegatives)
		}
		if cfg.MaxExamples > 0 && len(out) >= cfg.MaxExamples {
			break
		}
	}
	summary.Examples = len(out)
	if len(out) == 0 {
		return nil, summary, fmt.Errorf("BM25 hard-negative mining produced no examples")
	}
	return out, summary, nil
}

// MineModelTextHardNegatives mines text hard negatives from BEIR data using the embedding model's own retrieval ranking.
func MineModelTextHardNegatives(ctx context.Context, model *EmbeddingModel, cfg RetrievalHardNegativeMiningConfig) ([]EmbeddingTextHardNegativeExample, RetrievalHardNegativeMiningSummary, error) {
	if model == nil {
		return nil, RetrievalHardNegativeMiningSummary{}, fmt.Errorf("embedding model is not loaded")
	}
	cfg = normalizeRetrievalHardNegativeMiningConfig(cfg)
	if cfg.CorpusPath == "" || cfg.QueriesPath == "" || cfg.QrelsPath == "" {
		return nil, RetrievalHardNegativeMiningSummary{}, fmt.Errorf("corpus, queries, and qrels paths are required")
	}
	qrels, err := readBEIRQrels(cfg.QrelsPath)
	if err != nil {
		return nil, RetrievalHardNegativeMiningSummary{}, err
	}
	corpus, err := readBEIRCorpus(cfg.CorpusPath, cfg.MaxDocs)
	if err != nil {
		return nil, RetrievalHardNegativeMiningSummary{}, err
	}
	queries, skippedQueries, err := readBEIRQueries(cfg.QueriesPath, qrels, cfg.MaxQueries)
	if err != nil {
		return nil, RetrievalHardNegativeMiningSummary{}, err
	}
	if len(corpus) == 0 {
		return nil, RetrievalHardNegativeMiningSummary{}, fmt.Errorf("corpus is empty")
	}
	if len(queries) == 0 {
		return nil, RetrievalHardNegativeMiningSummary{}, fmt.Errorf("no qrels queries found in queries file")
	}
	docVectors, err := embedRetrievalTexts(ctx, model, corpus, cfg.BatchSize)
	if err != nil {
		return nil, RetrievalHardNegativeMiningSummary{}, fmt.Errorf("embed corpus: %w", err)
	}
	queryVectors, err := embedRetrievalTexts(ctx, model, queries, cfg.BatchSize)
	if err != nil {
		return nil, RetrievalHardNegativeMiningSummary{}, fmt.Errorf("embed queries: %w", err)
	}
	queryText := make(map[string]string, len(queries))
	for _, query := range queries {
		queryText[query.ID] = query.Text
	}
	docText := make(map[string]string, len(corpus))
	for _, doc := range corpus {
		docText[doc.ID] = doc.Text
	}
	summary := RetrievalHardNegativeMiningSummary{
		DatasetName:          cfg.DatasetName,
		Queries:              len(queries),
		SkippedQueriesNoText: skippedQueries,
	}
	out := []EmbeddingTextHardNegativeExample{}
	for _, query := range queryVectors {
		if err := ctx.Err(); err != nil {
			return nil, RetrievalHardNegativeMiningSummary{}, err
		}
		positives, skippedPositiveDocs := bm25MiningPositiveDocs(qrels[query.ID], docText)
		summary.SkippedPositiveDocs += skippedPositiveDocs
		if len(positives) == 0 {
			continue
		}
		positiveIDs := make(map[string]bool, len(positives))
		for _, positive := range positives {
			positiveIDs[positive.ID] = true
		}
		candidateDepth := cfg.CandidateTopK + len(positiveIDs)
		scores := topRetrievalScores(query.Vector, docVectors, candidateDepth)
		negativeTexts := modelMiningNegativeTexts(scores, positiveIDs, docText, cfg)
		if len(negativeTexts) == 0 {
			summary.SkippedQueriesNoNegative++
			continue
		}
		for _, positive := range positives {
			if cfg.MaxExamples > 0 && len(out) >= cfg.MaxExamples {
				break
			}
			exampleNegatives := negativeTexts
			if len(exampleNegatives) > cfg.NegativesPerPositive {
				exampleNegatives = exampleNegatives[:cfg.NegativesPerPositive]
			}
			out = append(out, EmbeddingTextHardNegativeExample{
				Query:     queryText[query.ID],
				Positive:  positive.Text,
				Negatives: append([]string(nil), exampleNegatives...),
			})
			summary.PositivePairs++
			summary.Negatives += len(exampleNegatives)
		}
		if cfg.MaxExamples > 0 && len(out) >= cfg.MaxExamples {
			break
		}
	}
	summary.Examples = len(out)
	if len(out) == 0 {
		return nil, summary, fmt.Errorf("model hard-negative mining produced no examples")
	}
	return out, summary, nil
}

func normalizeRetrievalHardNegativeMiningConfig(cfg RetrievalHardNegativeMiningConfig) RetrievalHardNegativeMiningConfig {
	if cfg.DatasetName == "" {
		cfg.DatasetName = "retrieval"
	}
	if cfg.NegativesPerPositive <= 0 {
		cfg.NegativesPerPositive = 1
	}
	if cfg.CandidateTopK <= 0 {
		cfg.CandidateTopK = 100
	}
	if cfg.CandidateTopK < cfg.NegativesPerPositive {
		cfg.CandidateTopK = cfg.NegativesPerPositive
	}
	if cfg.BatchSize <= 0 {
		cfg.BatchSize = 64
	}
	return cfg
}

func bm25MiningPositiveDocs(rels map[string]float64, docText map[string]string) ([]retrievalPositiveDoc, int) {
	positives := make([]retrievalPositiveDoc, 0, len(rels))
	skipped := 0
	for docID, rel := range rels {
		text := strings.TrimSpace(docText[docID])
		if text == "" {
			skipped++
			continue
		}
		positives = append(positives, retrievalPositiveDoc{ID: docID, Score: rel, Text: text})
	}
	slices.SortFunc(positives, func(a, b retrievalPositiveDoc) int {
		if a.Score > b.Score {
			return -1
		}
		if a.Score < b.Score {
			return 1
		}
		if a.ID < b.ID {
			return -1
		}
		if a.ID > b.ID {
			return 1
		}
		return 0
	})
	return positives, skipped
}

func bm25MiningNegativeTexts(queryText string, positiveIDs map[string]bool, index bm25Index, docText map[string]string, cfg RetrievalHardNegativeMiningConfig) []string {
	queryTokens := tokenizeBM25Text(queryText)
	negatives := topBM25NonPositiveTexts(queryTokens, positiveIDs, index, docText, cfg.CandidateTopK)
	if len(negatives) > cfg.NegativesPerPositive {
		negatives = negatives[:cfg.NegativesPerPositive]
	}
	return negatives
}

func modelMiningNegativeTexts(scores []retrievalScoredDoc, positiveIDs map[string]bool, docText map[string]string, cfg RetrievalHardNegativeMiningConfig) []string {
	limit := cfg.NegativesPerPositive
	if cfg.CandidateTopK > 0 && cfg.CandidateTopK < limit {
		limit = cfg.CandidateTopK
	}
	negatives := make([]string, 0, limit)
	seen := map[string]bool{}
	candidates := 0
	for _, score := range scores {
		if positiveIDs[score.ID] {
			continue
		}
		text := strings.TrimSpace(docText[score.ID])
		if text == "" || seen[text] {
			continue
		}
		candidates++
		if cfg.CandidateTopK > 0 && candidates > cfg.CandidateTopK {
			break
		}
		seen[text] = true
		negatives = append(negatives, text)
		if len(negatives) >= limit {
			break
		}
	}
	return negatives
}
