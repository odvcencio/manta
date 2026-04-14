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
		summary.PositivePairs += len(positives)
		for _, positive := range positives {
			exampleNegatives := negativeTexts
			if len(exampleNegatives) > cfg.NegativesPerPositive {
				exampleNegatives = exampleNegatives[:cfg.NegativesPerPositive]
			}
			out = append(out, EmbeddingTextHardNegativeExample{
				Query:     query.Text,
				Positive:  positive.Text,
				Negatives: append([]string(nil), exampleNegatives...),
			})
			summary.Negatives += len(exampleNegatives)
		}
	}
	summary.Examples = len(out)
	if len(out) == 0 {
		return nil, summary, fmt.Errorf("BM25 hard-negative mining produced no examples")
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
